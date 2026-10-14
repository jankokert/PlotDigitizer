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

    kind: str = "value"         # "value" ("T_A = 25 °C") or "part" ("CMZ5334B")

    @property
    def header(self) -> str:
        if self.kind == "part":
            return "part"
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


_PART_RE = re.compile(r"^([A-Za-z]{2,})(\d{2,}[A-Za-z0-9-]*)$")


def _part_family(texts: list[str]) -> LabelFamily | None:
    """Labels that are part numbers of one series ("CMZ5334B", "CMZ5335B"):
    same letter prefix, differing numbers."""
    keys = [(_PART_RE.match(t.strip()) or None) for t in texts]
    prefixes = Counter(m.group(1).upper() for m in keys if m)
    if not prefixes:
        return None
    prefix, n = prefixes.most_common(1)[0]
    if n < 2:
        return None
    values = [""] * len(texts)
    members = []
    for i, t in enumerate(texts):
        m = keys[i]
        if m and m.group(1).upper() == prefix:
            # OCR may mix the case of the letters ("cmz5346B"): normalise.
            values[i] = prefix + m.group(2)
            members.append(i)
            continue
        fixed = _ocr_fix_part(t, prefix)
        if fixed:
            values[i] = fixed
            members.append(i)
    return LabelFamily(symbol=None, unit=None, values=values, members=members, kind="part")


# Letters OCR typically reads in place of digits.
_DIGIT_LOOKALIKE = str.maketrans({"S": "5", "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                                  "|": "1", "Z": "2", "E": "6", "G": "6", "B": "8", "T": "7"})


def _ocr_fix_part(text: str, prefix: str) -> str | None:
    """'cMzs34eB' with series prefix 'CMZ' -> 'CMZ5346B': after the prefix,
    letters that look like digits are read as digits (the last character may
    be a genuine suffix letter)."""
    t = text.strip().upper()
    if not t.startswith(prefix) or len(t) < len(prefix) + 3:
        return None
    rest = t[len(prefix):]
    body, last = rest[:-1].translate(_DIGIT_LOOKALIKE), rest[-1]
    if not body.isdigit():
        return None
    if not (last.isdigit() or last.isalpha()):
        return None
    return prefix + body + last


def parse_labels(texts: list[str]) -> LabelFamily | None:
    """Find the largest group of labels sharing symbol and unit.

    Returns None if fewer than two labels share a (symbol, unit) pair, or
    if the pair is empty (bare numbers): a single "T_J = 25 °C" next to
    "f = 1.0 MHz" is a test-condition note, not a curve family.  Labels that
    are part numbers of one series form a "part" family instead (the larger
    family wins).
    """
    parsed = [parse_label(t) for t in texts]
    keys = Counter((p[0], p[2]) for p in parsed if p is not None)
    part = _part_family(texts)
    if not keys:
        return part
    (sym, unit), n = keys.most_common(1)[0]
    if n < 2 or (sym is None and unit is None):
        return part
    if part is not None and len(part.members) > n:
        return part
    members = [i for i, p in enumerate(parsed) if p is not None and (p[0], p[2]) == (sym, unit)]
    values = [parsed[i][1] if i in members else "" for i in range(len(texts))]
    return LabelFamily(symbol=sym, unit=unit, values=values, members=members)


def _densify(pts, step: float = 2.0) -> np.ndarray:
    from .curve_refine import densify
    return densify(pts, step)


def _arc_length(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def _y_at(pts, x: float) -> float:
    pts = np.asarray(pts, dtype=float)
    o = np.argsort(pts[:, 0])
    return float(np.interp(x, pts[o, 0], pts[o, 1]))


def _box_dist(b: np.ndarray, pts: np.ndarray) -> np.ndarray:
    dx = np.maximum(0, np.maximum(b[0] - pts[:, 0], pts[:, 0] - b[2]))
    dy = np.maximum(0, np.maximum(b[1] - pts[:, 1], pts[:, 1] - b[3]))
    return np.hypot(dx, dy)


def assign_labels(curves: dict, labels: list[dict],
                  plot_area: tuple[int, int, int, int],
                  extra_fragments: dict | None = None,
                  ) -> tuple[dict[str, str], str, list[str]]:
    """Map curve names to label values.

    curves:    {name: Nx2 (x, y) image pixels}
    labels:    [{text, box}] in canvas pixels
    plot_area: canvas (x0, y0, x1, y1) in the image
    extra_fragments: arrow candidates {name: [end, ..., end]} in image pixels
               (vector shafts, or raster shafts followed to their tip)

    Returns ({curve_name: value}, column header, names of the fragments used
    as arrows — the caller drops those that are traced curves).

    * Arrows: a fragment (short traced curve or extra fragment) with one end
      at a label box and pointing away from it is that label's arrow; its
      other end is the tip.  (Arrowheads paired greedily with the nearest
      label proved unreliable and are not used: a label without a shaft takes
      what the global assignment leaves it.)
    * Labels with an arrow, or a mix: Hungarian assignment on tip -> curve
      distance (labels without an arrow pay a large offset plus their box
      distance), so a sloppy arrow touching a neighbour is still resolved
      globally.
    * No arrows at all and as many labels as curves: labels sit next to their
      curves in the same vertical order, matched by rank (robust whether they
      are placed above or below).
    """
    family = parse_labels([lab["text"] for lab in labels])
    if family is None or not curves:
        return {}, "curve", []
    x0, y0, x1, y1 = plot_area
    names = list(curves)
    lengths = {n: _arc_length(curves[n]) for n in names}
    longest = max(lengths.values())
    cands = [n for n in names if lengths[n] >= 0.25 * longest]
    pool = {n: curves[n] for n in names if n not in cands}
    pool.update(extra_fragments or {})
    dense = [_densify(curves[n]) for n in cands]

    members = family.members
    boxes = [np.array(labels[i]["box"], dtype=float) + (x0, y0, x0, y0) for i in members]
    tips: list[np.ndarray | None] = [None] * len(members)

    # Arrow = fragment starting at the label box and pointing AWAY from it
    # (far end clearly farther from the box); nearest pairs first.
    used: list[str] = []
    cand = []
    for k, b in enumerate(boxes):
        reach = 1.5 * (b[3] - b[1])
        for n, frag in pool.items():
            pts = np.asarray(frag, dtype=float)
            d = _box_dist(b, pts[[0, -1]])
            j = int(np.argmin(d))
            length = float(np.hypot(*(pts[-1] - pts[0])))
            if d[j] < reach and d[1 - j] - d[j] >= 0.5 * length:
                cand.append((float(d[j]), k, n, j))
    for _, k, n, j in sorted(cand):
        if tips[k] is not None or n in used:
            continue
        pts = np.asarray(pool[n], dtype=float)
        tips[k] = pts[-1] if j == 0 else pts[0]
        used.append(n)

    if all(t is None for t in tips) and len(members) == len(cands):
        xs = [(b[0] + b[2]) / 2 for b in boxes]
        lab_order = np.argsort([(b[1] + b[3]) / 2 for b in boxes])
        cur_order = np.argsort([np.mean([_y_at(curves[n], x) for x in xs]) for n in cands])
        pairs = list(zip(lab_order, cur_order))
    else:
        cost = np.zeros((len(members), len(cands)))
        for r, (tip, b) in enumerate(zip(tips, boxes)):
            for c, d in enumerate(dense):
                if tip is not None:
                    cost[r, c] = np.hypot(d[:, 0] - tip[0], d[:, 1] - tip[1]).min()
                else:
                    cost[r, c] = 1e4 + _box_dist(b, d).min()
        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows, cols))

    mapping = {cands[c]: family.values[members[r]] for r, c in pairs}
    return mapping, family.header, used
