"""Split a set of curve labels into a common symbol, the per-curve values and
a common unit.

Datasheet curve families are labelled "T_A = 25 °C", "T_A = 100 °C", ... --
the symbol and unit repeat, only the number changes.  So instead of five text
names the CSV gets one column header ``T_A in °C`` and plain numbers ``25``,
``100``, ...
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# "[symbol] [=|:] number [unit]"; symbol may carry a "_subscript".
_LABEL_RE = re.compile(
    r"^\s*(?:(?P<sym>[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?)\s*[=:]\s*)?"
    r"(?P<num>[-+−]?\d+(?:[.,]\d+)?)\s*(?P<unit>[^\d\s=].*?)?\s*$"
)


@dataclass
class LabelFamily:
    symbol: str | None          # "T_A"
    unit: str | None            # "°C"
    values: list[str]           # per input label: "25", or "" if not a member
    members: list[int]          # indices of the labels that belong to the family

    @property
    def header(self) -> str:
        if self.symbol and self.unit:
            return f"{self.symbol} in {self.unit}"
        return self.symbol or (f"in {self.unit}" if self.unit else "curve")


def normalize_symbol(sym: str) -> str:
    """"TA" -> "T_A", "VR" -> "V_R", "Tj" -> "T_j"; "T_A", "D", "f" unchanged.

    Electrical symbols are one main letter plus a subscript; OCR (and PDFs
    without size information) lose the subscript formatting.
    """
    if "_" in sym or len(sym) < 2 or not sym[0].isupper():
        return sym
    return f"{sym[0]}_{sym[1:]}"


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.strip().replace("º", "°").replace("˚", "°")
    u = re.sub(r"^°\s*([CF])$", r"°\1", u)
    return u or None


def parse_label(text: str) -> tuple[str | None, str, str | None] | None:
    """Return (symbol, value, unit) for one label, or None if it has no number."""
    m = _LABEL_RE.match(text.replace(" ", " ").replace(" ", " "))
    if not m:
        return None
    sym = normalize_symbol(m["sym"]) if m["sym"] else None
    num = m["num"].replace(",", ".").replace("−", "-")
    return sym, num, _normalize_unit(m["unit"])


def parse_labels(texts: list[str]) -> LabelFamily | None:
    """Find the largest group of labels sharing symbol and unit.

    Returns None if fewer than two labels share a (symbol, unit) pair, or
    if the pair is empty (bare numbers): a single "T_J = 25 °C" next to
    "f = 1.0 MHz" is a test-condition note, not a curve family.
    """
    parsed = [parse_label(t) for t in texts]
    keys = Counter((p[0], p[2]) for p in parsed if p is not None)
    if not keys:
        return None
    (sym, unit), n = keys.most_common(1)[0]
    if n < 2 or (sym is None and unit is None):
        return None
    members = [i for i, p in enumerate(parsed) if p is not None and (p[0], p[2]) == (sym, unit)]
    values = [parsed[i][1] if i in members else "" for i in range(len(texts))]
    return LabelFamily(symbol=sym, unit=unit, values=values, members=members)


def _densify(pts, step: float = 2.0):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.vstack(out)


def _arc_length(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _y_at(pts, x: float) -> float:
    pts = np.asarray(pts, dtype=float)
    o = np.argsort(pts[:, 0])
    return float(np.interp(x, pts[o, 0], pts[o, 1]))


def _ray_hit(dense: np.ndarray, tip: tuple[float, float], u: np.ndarray,
             tol: float) -> float | None:
    """|Distance| along ray (tip, u) to the nearest point of *dense* within *tol* of it."""
    rel = dense - np.asarray(tip)
    t = rel @ u
    perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    ok = (t > -tol) & (perp < tol)
    return float(np.abs(t[ok]).min()) if ok.any() else None


def _box_dist(b: np.ndarray, pts: np.ndarray) -> np.ndarray:
    dx = np.maximum(0, np.maximum(b[0] - pts[:, 0], pts[:, 0] - b[2]))
    dy = np.maximum(0, np.maximum(b[1] - pts[:, 1], pts[:, 1] - b[3]))
    return np.hypot(dx, dy)


def assign_labels(curves: dict, labels: list[dict], arrows: list[dict],
                  plot_area: tuple[int, int, int, int]
                  ) -> tuple[dict[str, str], str, list[str]]:
    """Map curve names to label values.

    curves:    {name: Nx2 (x, y) image pixels}
    labels:    [{text, box}] in canvas pixels
    arrows:    detect_arrows() output (``label``, ``tip_x``, ``tip_y``, canvas px)
    plot_area: canvas (x0, y0, x1, y1) in the image

    Returns ({curve_name: value}, column header, arrow-shaft curve names).

    * Short traced fragments are never named.  A fragment with one end at a
      label box is that label's arrow shaft (the tracer picks shafts up): its
      direction is followed from the far end to the first curve it hits.  The
      shafts are returned so the caller can drop them from the data.
    * Labels without a shaft use a detected arrowhead tip (nearest curve);
      tips on the canvas border are frame false positives and ignored.
    * Without any arrows and with as many labels as curves the labels sit
      next to their curves in the same vertical order: matched by rank (robust
      whether labels are placed above or below their curve).
    * Otherwise: Hungarian assignment on the distance label box -> curve.
    """
    family = parse_labels([lab["text"] for lab in labels])
    if family is None or not curves:
        return {}, "curve", []
    x0, y0, x1, y1 = plot_area
    names = list(curves)
    lengths = {n: _arc_length(curves[n]) for n in names}
    longest = max(lengths.values())
    cands = [n for n in names if lengths[n] >= 0.25 * longest]
    frags = [n for n in names if n not in cands]
    dense = [_densify(curves[n]) for n in cands]

    members = family.members
    boxes = [np.array(labels[i]["box"], dtype=float) + (x0, y0, x0, y0) for i in members]
    # Per label: ("ray", tip, direction) | ("tip", tip) | None
    pointers: list[tuple | None] = [None] * len(members)

    shafts: list[str] = []
    for k, b in enumerate(boxes):
        reach = 1.5 * (b[3] - b[1])
        best, best_d = None, reach
        for n in frags:
            if n in shafts:
                continue
            pts = np.asarray(curves[n], dtype=float)
            d = _box_dist(b, pts[[0, -1]])
            j = int(np.argmin(d))
            if d[j] < best_d:
                best, best_d = n, d[j]
                far = pts[-1] if j == 0 else pts[0]
                mid = pts[len(pts) // 2]
        if best is not None:
            u = far - mid
            u = u / max(np.hypot(*u), 1e-9)
            shafts.append(best)
            pointers[k] = ("ray", (float(far[0]), float(far[1])), u)

    free_arrows = [a for a in arrows
                   if 8 < a["tip_x"] + x0 - x0 < (x1 - x0) - 8 and 8 < a["tip_y"] < (y1 - y0) - 8]
    for k, i in enumerate(members):
        if pointers[k] is not None:
            continue
        hit = next((a for a in free_arrows if a["label"] == labels[i]["text"]), None)
        if hit is not None:
            free_arrows.remove(hit)
            pointers[k] = ("tip", (hit["tip_x"] + x0, hit["tip_y"] + y0))

    if all(p is None for p in pointers) and len(members) == len(cands):
        xs = [(b[0] + b[2]) / 2 for b in boxes]
        lab_order = np.argsort([(b[1] + b[3]) / 2 for b in boxes])
        cur_order = np.argsort([np.mean([_y_at(curves[n], x) for x in xs]) for n in cands])
        pairs = list(zip(lab_order, cur_order))
    else:
        tol = 0.25 * float(np.median([b[3] - b[1] for b in boxes]))
        cost = np.zeros((len(members), len(cands)))
        for r, (ptr, b) in enumerate(zip(pointers, boxes)):
            for c, d in enumerate(dense):
                if ptr is not None:
                    tip = np.asarray(ptr[1])
                    near = np.hypot(d[:, 0] - tip[0], d[:, 1] - tip[1]).min()
                    t = _ray_hit(d, ptr[1], ptr[2], tol) if ptr[0] == "ray" else None
                    # A ray hit is decisive; otherwise fall back to (penalised) proximity.
                    cost[r, c] = t if t is not None else 1e3 + near
                else:
                    cost[r, c] = 1e4 + _box_dist(b, d).min()
        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows, cols))

    mapping = {cands[c]: family.values[members[r]] for r, c in pairs}
    return mapping, family.header, shafts
