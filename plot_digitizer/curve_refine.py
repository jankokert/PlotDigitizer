"""Joint refinement of traced curves so that together they explain all ink.

The greedy tracer follows one curve at a time with a forward-cone centroid.
Where several curves merge into one fat band (a diode family converging at
high current), every centroid is pulled to the middle of the band: all
traces collapse onto one line and the outer ink stays unused (a "dent").

Physically each curve is a pen stroke of width ``w``.  Across the band (along
the curve normal) the ink is one run ``[lo, hi]``; if ``m`` curves pass
through it, their ``m`` strokes must cover it, so

* one curve:    centre = (lo + hi) / 2,
* ``m`` curves: the outermost sit ``w/2`` inside the band edges, the others
  evenly between, in their current order (order may change only where the
  band is a single stroke wide, i.e. at a crossing).

Iterating "measure band -> place curves -> smooth along the arc" lets the
curves spread to the ink they actually came from.  Ink is measured on a mask
cleaned of slivers thinner than ~40 % of the pen (grid-subtraction residue),
so a crossing grid remnant does not widen a band.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def densify(pts, step: float) -> np.ndarray:
    """Insert points so no segment is longer than `step`; keeps every vertex
    (corners and both ends stay exact)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts.reshape(-1, 2)
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    return np.vstack(out)


def _resample(p: np.ndarray, step: float) -> np.ndarray:
    """Evenly spaced points along the arc (both ends kept)."""
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < step:
        return p
    q = np.arange(0.0, s[-1], step)
    if s[-1] - q[-1] > 0.3 * step:
        q = np.append(q, s[-1])
    else:
        q[-1] = s[-1]
    return np.column_stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])])


def _normals(p: np.ndarray) -> np.ndarray:
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    return np.column_stack([-t[:, 1], t[:, 0]]), t


def stroke_width(mask: np.ndarray, paths: list[np.ndarray]) -> float | None:
    """Pen width measured on the traces: 2 × median distance-to-edge."""
    if not paths:
        return None
    dist = ndimage.distance_transform_edt(mask)
    h, w = mask.shape
    vals = []
    for p in paths:
        q = _resample(p, 2.0)
        xi = np.clip(np.rint(q[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(q[:, 1]).astype(int), 0, h - 1)
        vals.append(dist[yi, xi])
    v = np.concatenate(vals)
    v = v[v > 0]
    return 2.0 * float(np.median(v)) if len(v) else None


def _band(row: np.ndarray, walk: np.ndarray, centre: int, slack: int,
          max_gap: int) -> tuple[int, int] | None:
    """Index range of the ink run at (or within `slack` of) `centre` in a 1-D
    profile.  The run continues across gaps of at most `max_gap` px that were
    ink before grid/arrow subtraction (`walk`): a grid line crossing a band of
    merged curves punches such a gap, but the band is one."""
    if not row[centre]:
        idx = np.flatnonzero(row)
        if len(idx) == 0:
            return None
        nearest = int(idx[np.argmin(np.abs(idx - centre))])
        if abs(nearest - centre) > slack:
            return None
        centre = nearest
    n = len(row)
    lo = centre
    while True:
        while lo > 0 and row[lo - 1]:
            lo -= 1
        g = lo - 1
        while g >= 0 and not row[g] and walk[g] and lo - g <= max_gap:
            g -= 1
        if g >= 0 and row[g] and lo - g - 1 <= max_gap:
            lo = g
            continue
        break
    hi = centre
    while True:
        while hi < n - 1 and row[hi + 1]:
            hi += 1
        g = hi + 1
        while g < n and not row[g] and walk[g] and g - hi <= max_gap:
            g += 1
        if g < n and row[g] and g - hi - 1 <= max_gap:
            hi = g
            continue
        break
    return lo, hi


def _orient(p: np.ndarray) -> np.ndarray:
    """Left-to-right (then top-to-bottom), so all normals point the same way."""
    d = p[-1] - p[0]
    if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
        return p[::-1].copy()
    return p


def _drop_duplicates(curves: list[np.ndarray], w: float) -> list[np.ndarray]:
    """Drop traces that mostly run inside a longer trace's stroke (re-traces
    of an already followed tube, wandering through a merged band)."""
    order = sorted(range(len(curves)), key=lambda i: -len(curves[i]))
    kept: list[int] = []
    for i in order:
        c = curves[i]
        dup = False
        for j in kept:
            d = cKDTree(curves[j]).query(c)[0]
            if (d <= w / 3).mean() >= 0.6:
                dup = True
                break
        if not dup:
            kept.append(i)
    return [curves[i] for i in sorted(kept)]


def _stacking_order(curves, profs, L, slack, max_gap, w, step) -> dict:
    """{(i, j): True} if curve j lies on the +normal side of curve i.

    Voted only at points where i is alone in its band (resolvable), from j's
    nearest point across; a relation needs >= 8 votes and 90 % agreement.
    Curves that genuinely cross get no relation (votes split) and fall back to
    their current positions.
    """
    votes: dict = {}
    trees = [cKDTree(c) for c in curves]
    for i, c in enumerate(curves):
        n, tan = _normals(c)
        prof, wprof = profs[i]
        alone = np.zeros(len(c), bool)
        for q in range(len(c)):
            band = _band(prof[q], wprof[q], L, slack, max_gap)
            alone[q] = band is not None and band[1] - band[0] + 1 <= 1.3 * w
        for j, t in enumerate(trees):
            if j == i:
                continue
            d, idx = t.query(c[alone], distance_upper_bound=6 * w)
            hit = np.isfinite(d)
            if not hit.any():
                continue
            rel = curves[j][idx[hit]] - c[alone][hit]
            along = np.abs(np.einsum("ij,ij->i", rel, tan[alone][hit]))
            across = np.einsum("ij,ij->i", rel, n[alone][hit])[along <= step]
            pos, neg = int((across > 0).sum()), int((across < 0).sum())
            votes[(i, j)] = (pos, neg)
    above = {}
    for (i, j), (pos, neg) in votes.items():
        tot = pos + neg
        if tot >= 8 and max(pos, neg) >= 0.9 * tot:
            above[(i, j)] = pos > neg          # j on +normal side of i
    return above


def _rank(members: list[tuple[float, int]], above: dict) -> list[tuple[float, int]]:
    """Order band members along the normal: by current offset, then fixed up
    with the known stacking relations (bubble passes, so unknown pairs keep
    their measured order)."""
    out = sorted(members)
    for _ in range(len(out)):
        swapped = False
        for a in range(len(out) - 1):
            i, j = out[a][1], out[a + 1][1]
            if above.get((i, j)) is False or above.get((j, i)) is True:
                out[a], out[a + 1] = out[a + 1], out[a]
                swapped = True
        if not swapped:
            break
    return out


def refine_bundle(paths: list[np.ndarray], mask: np.ndarray,
                  walkable: np.ndarray | None = None,
                  width: float | None = None, iters: int = 8,
                  damping: float = 0.7) -> list[np.ndarray]:
    """Refine curves (canvas pixel coords) jointly against their ink `mask`.

    walkable: pixels that were ink before grid/arrow subtraction (gaps in
    `mask` that may be bridged inside a band).

    Returns densely resampled curves (spacing ~w/2); the caller re-thins them.
    """
    if not paths:
        return paths
    if width is None:
        width = stroke_width(mask, paths)
    if not width or width < 3.0:
        return paths                     # thin pens: centroid tracing is already exact
    w = float(width)
    r = max(1, int(round(0.2 * w)))
    clean = ndimage.binary_opening(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    H, W = clean.shape
    step = max(2.0, w / 2)
    L = int(np.ceil(3.0 * w))                 # look this far across the curve
    ts = np.arange(-L, L + 1, dtype=float)
    slack = int(np.ceil(w / 2))
    max_gap = int(np.ceil(0.8 * w))           # grid lines are thinner than curves
    walk = clean if walkable is None else (walkable | clean)
    curves = [_orient(_resample(np.asarray(p, float), step)) for p in paths]
    curves = [c for c in curves if len(c) >= 3]       # np.gradient needs 3 points
    if not curves:
        return paths
    curves = _drop_duplicates(curves, w)

    def profiles(c, n):
        """Ink / walkable profiles along each point's normal (rows: points)."""
        xs = np.rint(c[:, 0, None] + ts[None, :] * n[:, 0, None]).astype(int)
        ys = np.rint(c[:, 1, None] + ts[None, :] * n[:, 1, None]).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        prof = np.zeros(xs.shape, bool)
        prof[ok] = clean[ys[ok], xs[ok]]
        wprof = np.zeros(xs.shape, bool)
        wprof[ok] = walk[ys[ok], xs[ok]]
        return prof, wprof

    # Stacking order from where the curves are resolvable (alone in their
    # band): used inside merged bands, where the current positions say nothing.
    profs0 = [profiles(c, _normals(c)[0]) for c in curves]
    above = _stacking_order(curves, profs0, L, slack, max_gap, w, step)

    # The pen measured on the traces is biased low where grid subtraction ate
    # into strokes running along grid lines; the intact single-stroke bands
    # (upper part of the band-width distribution) show the true pen.
    Ts = []
    for prof, wprof in profs0:
        for i in range(len(prof)):
            band = _band(prof[i], wprof[i], L, slack, max_gap)
            if band is not None:
                Ts.append(band[1] - band[0] + 1)
    Ts = np.array([t for t in Ts if t <= 1.5 * w])
    if len(Ts) >= 20:
        w = max(w, float(np.percentile(Ts, 80)))

    for _ in range(iters):
        frames = [_normals(c) for c in curves]
        ids = np.concatenate([np.full(len(c), k) for k, c in enumerate(curves)])
        allp = np.vstack(curves)
        tree = cKDTree(allp)
        moved = 0.0
        new_curves = []
        for k, c in enumerate(curves):
            n, tan = frames[k]
            prof, wprof = profiles(c, n)
            off = np.full(len(c), np.nan)
            near_lists = tree.query_ball_point(c, L)
            for i in range(len(c)):
                band = _band(prof[i], wprof[i], L, slack, max_gap)
                if band is None:
                    continue
                lo, hi = band[0] - L, band[1] - L          # offsets along the normal
                # Other curves crossing this band near here (one point each).
                members = [(0.0, k)]
                seen = {k}
                rel = allp[near_lists[i]] - c[i]
                along = np.abs(rel @ tan[i])
                across = rel @ n[i]
                order = np.argsort(along)
                for j in order:
                    other = ids[near_lists[i][j]]
                    if other in seen or along[j] > step:
                        continue
                    seen.add(other)
                    if lo - w / 4 <= across[j] <= hi + w / 4:
                        members.append((float(across[j]), int(other)))
                m = len(members)
                T = hi - lo + 1
                if m == 1 and T < 0.85 * w:
                    # Stroke thinner than the pen: if one side was eaten by
                    # grid/arrow subtraction, measure from the intact edge.
                    b0, b1 = band
                    eaten_lo = b0 > 0 and wprof[i][b0 - 1] and not prof[i][b0 - 1]
                    eaten_hi = b1 < len(prof[i]) - 1 and wprof[i][b1 + 1] and not prof[i][b1 + 1]
                    if eaten_lo and not eaten_hi:
                        target = hi - (w - 1) / 2
                    elif eaten_hi and not eaten_lo:
                        target = lo + (w - 1) / 2
                    else:
                        target = (lo + hi) / 2
                elif m == 1 or T <= w:
                    target = (lo + hi) / 2
                else:
                    rank = [mid for _, mid in _rank(members, above)].index(k)
                    first, last = lo + (w - 1) / 2, hi - (w - 1) / 2
                    target = first + rank * (last - first) / (m - 1)
                off[i] = target
            # Smooth the requested shifts along the arc; never move by more than w.
            good = np.isfinite(off)
            if good.sum() < 3:
                new_curves.append(c)
                continue
            off = np.interp(np.arange(len(c)), np.flatnonzero(good), off[good])
            off = ndimage.median_filter(off, size=5, mode="nearest")
            off = ndimage.uniform_filter1d(off, size=5, mode="nearest")
            off = np.clip(off, -w, w) * damping
            moved = max(moved, float(np.abs(off).max()))
            new_curves.append(c + off[:, None] * n)
        curves = new_curves
        if moved < 0.25:
            break
    return curves
