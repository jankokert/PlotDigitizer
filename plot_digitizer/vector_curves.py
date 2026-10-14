"""Vector mode: read the curves of a figure straight from its PDF paths.

Datasheet plots exported from CAD/plotting tools are usually vector art:
every curve is a stroked polyline, so it can be taken over exactly instead of
being traced from pixels.  The sidecar JSON written by ``--pdf`` carries the
figure's vector paths (``vectors``, PNG pixel coords); this module sorts them
into

* curves       - stroked open paths spanning a good part of the canvas,
* arrow shafts - short straight strokes with a filled arrowhead at one end
                 (returned as [tail, tip] fragments for label assignment),
* the rest     - grid, ticks, frame, label boxes: ignored.

The raster tracer remains the fallback for scanned / bitmap figures.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MIN_CURVE_EXTENT = 0.10   # of the smaller canvas side
_MAX_HEAD_EXTENT = 0.08    # arrowheads are small filled polygons
_JOIN_TOL = 1.0            # px: sub-paths meeting this close are one curve


def _extent(p: np.ndarray) -> float:
    span = p.max(axis=0) - p.min(axis=0)
    return float(max(span))


def _inside(p: np.ndarray, plot_area, margin: float) -> bool:
    x0, y0, x1, y1 = plot_area
    c = (p.min(axis=0) + p.max(axis=0)) / 2
    return x0 - margin <= c[0] <= x1 + margin and y0 - margin <= c[1] <= y1 + margin


def _axis_aligned(p: np.ndarray, tol: float = 0.5) -> bool:
    span = p.max(axis=0) - p.min(axis=0)
    return bool(span[0] < tol or span[1] < tol)


def _join(paths: list[dict]) -> list[dict]:
    """Chain sub-paths of the same paint whose ends meet (one curve, many pieces)."""
    paths = [dict(p, points=np.asarray(p["points"], float)) for p in paths]
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(paths):
            for j, b in enumerate(paths):
                if i == j or a["width"] != b["width"] or a["color"] != b["color"]:
                    continue
                if np.hypot(*(a["points"][-1] - b["points"][0])) <= _JOIN_TOL:
                    a["points"] = np.vstack([a["points"], b["points"][1:]])
                    del paths[j]
                    merged = True
                    break
            if merged:
                break
    return paths


def _split_out_and_back(p: np.ndarray, min_share: float = 0.3) -> list[np.ndarray]:
    """Split where the x direction reverses, if both sides are long.

    Some plot exports draw two neighbouring curves as ONE stroke that runs up
    one curve and back down the other (MUH1PB Fig. 3, 175 °C).  A genuine
    wiggle in a single curve is short, so only long legs are split off.
    """
    dx = np.sign(np.diff(p[:, 0]))
    nz = np.flatnonzero(dx)
    if len(nz) < 2:
        return [p]
    total = _extent(p)
    for a, b in zip(nz[:-1], nz[1:]):
        if dx[a] != dx[b]:
            left, right = p[:b + 1], p[b:]
            if _extent(left) >= min_share * total and _extent(right) >= min_share * total:
                return _split_out_and_back(left, min_share) + _split_out_and_back(right, min_share)
    return [p]


def vector_curves(vectors: list[dict], plot_area) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Classify sidecar vector paths -> ({curve name: Nx2}, {shaft name: [tail, tip]}).

    Coordinates are image pixels, resampled to ~100 points per canvas width.
    Curve names are ``curve_vNN`` in left-to-right order (like the raster
    tracer's ``_segNN``).
    """
    from .curve_extractor import _reading_order

    x0, y0, x1, y1 = plot_area
    side = min(x1 - x0, y1 - y0)
    margin = 0.02 * side
    open_strokes = [v for v in vectors if v["stroke"] and not v["closed"] and len(v["points"]) >= 2]
    heads = [np.asarray(v["points"], float) for v in vectors
             if v["fill"] and v["closed"] and 3 <= len(v["points"]) <= 6
             and _extent(np.asarray(v["points"], float)) < _MAX_HEAD_EXTENT * side]

    shafts: dict[str, np.ndarray] = {}
    rest = []
    for v in _join(open_strokes):
        p = v["points"]
        if not _inside(p, plot_area, margin):
            continue
        if len(p) == 2 and heads:
            # Straight stroke with an arrowhead at one end -> callout arrow.
            best = None
            for h in heads:
                c = h.mean(axis=0)
                d = np.hypot(*(p - c).T)
                j = int(np.argmin(d))
                reach = _extent(h) + 2 * max(v["width"], 1.0)
                if d[j] <= reach and (best is None or d[j] < best[0]):
                    best = (d[j], j, h)
            if best is not None:
                _, j, h = best
                tail = p[1 - j]
                tip = h[int(np.argmax(np.hypot(*(h - tail).T)))]
                shafts[f"arrow_v{len(shafts):02d}"] = np.vstack([tail, tip])
                continue
        rest.append(v)

    # Straight axis-parallel strokes are grid lines / ticks — unless clearly
    # thicker than the grid (a flat stretch of a curve drawn as its own path).
    grid_w = [v["width"] for v in rest if len(v["points"]) == 2 and _axis_aligned(v["points"])]
    grid_w = float(np.median(grid_w)) if grid_w else 0.0
    pieces = []
    for v in rest:
        p = v["points"]
        if _axis_aligned(p) and len(p) == 2 and v["width"] <= 1.5 * grid_w:
            continue
        for q in _split_out_and_back(p):
            if _extent(q) >= _MIN_CURVE_EXTENT * side:
                pieces.append(q)

    # Resample densely ON the exact paths: a straight PDF segment is straight
    # in plot space (log space on log axes), so a consumer interpolating the
    # CSV linearly in data units needs points along it, not just its corners.
    from .curve_refine import densify
    step = max(1.0, (x1 - x0) / 100.0)
    pieces = [densify(q, step) for q in pieces]

    curves: dict[str, np.ndarray] = {}
    if pieces:
        order = _reading_order(np.array([[np.median(q[:, 0]), np.median(q[:, 1])] for q in pieces]),
                               force_x=True)
        for k, i in enumerate(order):
            curves[f"curve_v{k:02d}"] = pieces[i]
    logger.info(f"Vector mode: {len(curves)} curve(s), {len(shafts)} arrow(s) "
                f"from {len(vectors)} vector paths")
    return curves, shafts
