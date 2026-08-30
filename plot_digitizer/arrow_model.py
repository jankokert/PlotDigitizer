"""Parametric annotation-arrow reconstruction and anti-aliased subtraction.

Same philosophy as :mod:`grid_model`: an arrow is not a loose bag of dark
pixels but an *exact* geometric object — a solid triangular head plus a straight
shaft — so we reconstruct that object and subtract its rendered (anti-aliased)
footprint, rather than deciding pixel-by-pixel.

Detection is **anchored** on the confirmed callout heads returned by
:func:`annotation_detector.detect_arrows` (an OCR text label paired with a solid
arrowhead).  Anchoring keeps precision perfect — we never invent a head on a
curve — while this module adds the geometric *model* on top of each anchor:

1. **Fit the triangle** of each head from its thick pixels' convex hull: apex +
   two base corners → orientation ``theta``, head length ``L`` and half-angle
   ``alpha`` (the tip angle).  All rotation-invariant.
2. **Share the style chart-wide.**  Within one chart every arrow has the same
   head size and angle (only position, rotation and shaft length differ), so we
   take the median ``L``/``alpha``/shaft-width and re-impose it on every arrow —
   regularising noisy single fits the way the equidistant grid fit does.
3. **Rasterise a "sector" coverage map** (triangle ∪ shaft) with sub-pixel
   anti-aliasing, and subtract it: the whole head is removed, the shaft only
   where it is thin (a curve crossing the shaft is thicker and is spared, so no
   gap is punched in the data curve).

The reconstructed model is returned as an :class:`ArrowModel` (serialisable) and
its coverage map is drawn into the debug PNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

_T_BODY = 1.2         # thick-pixel floor for the hull (drops 1 px shaft/curve)
_HALF_ANGLE_RANGE = (6.0, 55.0)   # plausible arrowhead half-angle (deg)
_LEN_RANGE = (4.0, 34.0)          # plausible head length (px)


@dataclass
class Arrow:
    """One reconstructed arrow (canvas pixel coords)."""
    apex: tuple[float, float]          # tip point (rests on the curve)
    base_mid: tuple[float, float]      # centre of the head base
    tail: tuple[float, float]          # far end of the shaft (at the label)
    theta_deg: float                   # heading, base→apex
    head_len: float                    # apex→base distance (shared style)
    half_angle_deg: float              # shared tip half-angle
    shaft_len: float
    label: Optional[str] = None


@dataclass
class ArrowModel:
    """All arrows plus the single chart-wide head style."""
    arrows: list[Arrow]
    head_len: float
    half_angle_deg: float
    shaft_width: float

    def to_dict(self) -> dict:
        return {"head_len": self.head_len,
                "half_angle_deg": self.half_angle_deg,
                "shaft_width": self.shaft_width,
                "arrows": [asdict(a) for a in self.arrows]}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _approx_triangle(pts: np.ndarray):
    """Approximate a point cloud by 3 corners.

    ``pts`` is ``(N, 2)`` of ``(x, y)``.  The farthest-apart hull pair spans the
    two longest edges of the arrowhead; the hull point farthest from that line is
    the third corner.
    """
    try:
        from scipy.spatial import ConvexHull
        hull = pts[ConvexHull(pts).vertices]
    except Exception:
        hull = pts
    if len(hull) < 3:
        return None
    d = hull[:, None, :] - hull[None, :, :]
    dd = (d * d).sum(-1)
    i, j = np.unravel_index(int(np.argmax(dd)), dd.shape)
    c1, c2 = hull[i].astype(float), hull[j].astype(float)
    e = c2 - c1
    el = math.hypot(e[0], e[1])
    if el < 1e-6:
        return None
    nrm = np.array([-e[1], e[0]]) / el
    perp = np.abs((hull - c1) @ nrm)
    k = int(np.argmax(perp))
    if perp[k] < 1.0:
        return None
    return c1, c2, hull[k].astype(float)


def _sample(mask: np.ndarray, p) -> bool:
    xi, yi = int(round(p[0])), int(round(p[1]))
    return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])


def _fit_from_anchor(fg: np.ndarray, dist: np.ndarray, anchor: dict) -> Optional[dict]:
    """Fit the head triangle for one confirmed callout (its head_box + tip)."""
    h, w = fg.shape
    hx0, hy0, hx1, hy1 = anchor["head_box"]
    tip = np.array([float(anchor["tip_x"]), float(anchor["tip_y"])])
    tail = np.array([float(anchor["tail_x"]), float(anchor["tail_y"])])
    pad = 3
    y0, y1 = max(0, hy0 - pad), min(h, hy1 + pad + 1)
    x0, x1 = max(0, hx0 - pad), min(w, hx1 + pad + 1)
    ys, xs = np.where((fg[y0:y1, x0:x1]) & (dist[y0:y1, x0:x1] >= _T_BODY))
    if len(xs) < 5:
        return None
    P = np.stack([xs + x0, ys + y0], axis=1).astype(float)
    tri = _approx_triangle(P)
    if tri is None:
        return None
    c1, c2, c3 = tri
    # apex = the corner nearest the OCR-confirmed tip; base = the other two
    corners = [c1, c2, c3]
    ai = int(np.argmin([np.hypot(*(c - tip)) for c in corners]))
    apex = corners[ai]
    base = [c for idx, c in enumerate(corners) if idx != ai]
    base_mid = 0.5 * (base[0] + base[1])
    axis = apex - base_mid
    L = math.hypot(axis[0], axis[1])
    base_len = math.hypot(*(base[0] - base[1]))
    if not (_LEN_RANGE[0] <= L <= _LEN_RANGE[1]) or base_len < 2.0:
        return None
    half_angle = math.degrees(math.atan2(base_len / 2.0, L))
    if not (_HALF_ANGLE_RANGE[0] <= half_angle <= _HALF_ANGLE_RANGE[1]):
        return None
    u = axis / L                                   # base→apex unit
    # make the shaft point from base to the tail (away from the apex)
    v_tail = tail - base_mid
    if v_tail @ (-u) < 0:                           # tail should be on the -u side
        pass
    theta = math.degrees(math.atan2(axis[1], axis[0]))
    # shaft width: perpendicular ink thickness just behind the base
    n = np.array([-u[1], u[0]])
    back = base_mid - u * 2.0
    wpos = 0
    while wpos < 8 and _sample(fg, back + n * (wpos + 1)):
        wpos += 1
    wneg = 0
    while wneg < 8 and _sample(fg, back - n * (wneg + 1)):
        wneg += 1
    shaft_w = float(min(max(wpos + wneg + 1, 2), 8))
    shaft_len = float(math.hypot(*(tail - base_mid)))
    return {"apex": apex, "base_mid": base_mid, "u": u, "theta": theta,
            "L": L, "half_angle": half_angle, "shaft_w": shaft_w,
            "shaft_len": shaft_len, "tail": tail, "label": anchor.get("label")}


# ---------------------------------------------------------------------------
# anti-aliased rasterisation ("sector graphic" on the pixels)
# ---------------------------------------------------------------------------

def _tri_cover(X, Y, a, b, c):
    d1 = (X - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (Y - b[1])
    d2 = (X - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (Y - c[1])
    d3 = (X - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (Y - a[1])
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _raster_into(cover: np.ndarray, apex, cA, cB, base_mid, tail,
                 shaft_w: float, ss: int = 3):
    """Add one arrow's AA coverage (triangle ∪ shaft); return (bbox, tri_cov)."""
    h, w = cover.shape
    pts = np.array([apex, cA, cB, base_mid, tail])
    x0 = max(0, int(np.floor(pts[:, 0].min())) - 1)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + 2)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - 1)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return None
    off = (np.arange(ss) + 0.5) / ss
    gx = (np.arange(x0, x1)[:, None] + off[None, :]).ravel()
    gy = (np.arange(y0, y1)[:, None] + off[None, :]).ravel()
    X, Y = np.meshgrid(gx, gy)
    tri = _tri_cover(X, Y, apex, cA, cB)
    seg = tail - base_mid
    sl = math.hypot(seg[0], seg[1])
    if sl > 1e-6:
        u = seg / sl
        nrm = np.array([-u[1], u[0]])
        px, py = X - base_mid[0], Y - base_mid[1]
        along = px * u[0] + py * u[1]
        perp = px * nrm[0] + py * nrm[1]
        shaft = (along >= 0) & (along <= sl) & (np.abs(perp) <= shaft_w / 2.0)
    else:
        shaft = np.zeros_like(tri)

    def _ds(grid):
        return grid.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))

    tri_cov = _ds(tri)
    cover[y0:y1, x0:x1] = np.maximum(cover[y0:y1, x0:x1], _ds(tri | shaft))
    return (x0, y0, x1, y1, tri_cov)


def _mask_shaft(mask: np.ndarray, ink: np.ndarray, base_mid: np.ndarray,
                u: np.ndarray, shaft_len: float, shaft_w: float,
                cross_factor: float = 3.0):
    """Remove the shaft core, sparing columns where a thicker curve crosses."""
    if shaft_len < 2:
        return
    h, w = ink.shape
    d = -u                                   # along the shaft, away from apex
    n = np.array([-u[1], u[0]])
    core = shaft_w / 2.0
    cross_thick = max(shaft_w * cross_factor, shaft_w + 4)
    t = 0.0
    while t <= shaft_len:
        c = base_mid + d * t
        pos = 0
        while pos < 20 and _sample(ink, c + n * (pos + 1)):
            pos += 1
        neg = 0
        while neg < 20 and _sample(ink, c - n * (neg + 1)):
            neg += 1
        if pos + neg + 1 <= cross_thick:
            for o in np.arange(-core, core + 1e-6, 0.5):
                xi = int(round(c[0] + n[0] * o))
                yi = int(round(c[1] + n[1] * o))
                if 0 <= yi < h and 0 <= xi < w and ink[yi, xi]:
                    mask[yi, xi] = True
        t += 0.5


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def model_arrows(
    s: np.ndarray,
    v: np.ndarray,
    grid_mask: np.ndarray,
    anchors: list[dict],
    resid_tol: float = 0.28,
    cover_on: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, ArrowModel]:
    """Reconstruct the anchored arrows; return ``(arrow_gray, arrow_mask, model)``.

    * ``arrow_gray`` – anti-aliased model coverage in [0, 1] (the sector graphic).
    * ``arrow_mask`` – ink pixels to suppress: the whole head, plus the shaft
      where it is thin (a thicker curve crossing the shaft is spared).
    """
    from scipy import ndimage

    ink = (s < 0.30) & (v < 0.90)
    fg = ink & ~grid_mask
    h, w = fg.shape
    arrow_gray = np.zeros((h, w), dtype=np.float32)
    arrow_mask = np.zeros((h, w), dtype=bool)

    if not anchors:
        return arrow_gray, arrow_mask, ArrowModel([], 0.0, 0.0, 0.0)

    dist = ndimage.distance_transform_edt(fg)
    fits = [f for f in (_fit_from_anchor(fg, dist, a) for a in anchors) if f]
    if not fits:
        return arrow_gray, arrow_mask, ArrowModel([], 0.0, 0.0, 0.0)

    # One head style for the whole chart (median = robust to a noisy single fit).
    L_star = float(np.median([f["L"] for f in fits]))
    a_star = float(np.median([f["half_angle"] for f in fits]))
    w_star = float(np.median([f["shaft_w"] for f in fits]))
    base_half = L_star * math.tan(math.radians(a_star))

    darkness = 1.0 - v
    arrows: list[Arrow] = []
    for f in fits:
        base_mid = f["base_mid"]
        u = f["u"]
        nrm = np.array([-u[1], u[0]])
        apex = base_mid + u * L_star
        cA = base_mid + nrm * base_half
        cB = base_mid - nrm * base_half
        tail = f["tail"]
        shaft_len = f["shaft_len"]

        info = _raster_into(arrow_gray, apex, cA, cB, base_mid, tail, w_star)
        if info is not None:
            bx0, by0, bx1, by1, tri_cov = info
            arrow_mask[by0:by1, bx0:bx1] |= ink[by0:by1, bx0:bx1] & (tri_cov >= 0.5)
        _mask_shaft(arrow_mask, ink, base_mid, u, shaft_len, w_star)

        arrows.append(Arrow(
            apex=(float(apex[0]), float(apex[1])),
            base_mid=(float(base_mid[0]), float(base_mid[1])),
            tail=(float(tail[0]), float(tail[1])),
            theta_deg=f["theta"], head_len=L_star, half_angle_deg=a_star,
            shaft_len=float(shaft_len), label=f["label"]))

    # Protect genuine curve crossings: where the model is thin yet the ink is
    # much darker than predicted, a curve crosses — keep it.
    keep = arrow_mask & (arrow_gray < cover_on) & (darkness - arrow_gray > resid_tol)
    arrow_mask &= ~keep

    return arrow_gray, arrow_mask, ArrowModel(
        arrows=arrows, head_len=L_star, half_angle_deg=a_star, shaft_width=w_star)
