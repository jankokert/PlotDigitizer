"""Mathematical grid reconstruction and grayscale subtraction.

A plot grid is an exact object: straight full-length lines at fixed positions.
Rather than deciding pixel-by-pixel whether each dark pixel is "grid" (which
flatters — scattered dots, gaps, points off the lines), we:

1. locate every grid line's position by projecting the ink onto each axis (a
   grid line concentrates in one column/row with near-full coverage; a data
   curve — even a steep one — drifts across columns and never fills one),
2. model each line's grayscale darkness profile from the image itself (core
   dark, aliased grey edges) — nothing hardcoded, measured per line, and
3. subtract that modeled grid darkness from the original.  A pixel only as dark
   as the grid predicts is pure grid and is removed; a pixel darker than the
   grid (a curve crossing the line) stays.  So curves survive crossings and no
   gap is punched — exactly how a person reads it: where the grey of a line
   suddenly gets darker, another object is there.

Doing this in ONE pass, before OCR / text / arrow detection, gives every later
stage a clean, grid-free image.  Subtracting the ideal grid where the original
grid is interrupted (a white legend box) does no harm — and the mismatch there
is itself a useful "negative" for locating such boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

_LOG_MANT = np.log10(np.arange(1, 10))     # fractional log10 of 1..9 (minor lines)


@dataclass
class GridAxis:
    """One axis of the reconstructed grid (matplotlib-style)."""
    scale: str                      # "linear" | "log"
    lines_px: list[float]           # gridline pixel positions (outliers removed)
    px_lo: float                    # pixel position of the low end of the span
    px_hi: float                    # pixel position of the high end
    spacing_px: Optional[float] = None      # linear: constant gap; log: decade width
    ticks_per_decade: Optional[int] = None  # log only
    vmin: Optional[float] = None    # data value at px_lo (filled after OCR)
    vmax: Optional[float] = None    # data value at px_hi


@dataclass
class GridModel:
    """The plot grid as an exact object; serialisable for a matplotlib rebuild."""
    x: GridAxis
    y: GridAxis
    pixel_origin: tuple[int, int] = (0, 0)   # (x,y) of the canvas top-left in image
    canvas_size: tuple[int, int] = (0, 0)    # (w,h)

    def to_dict(self) -> dict:
        return {"x": asdict(self.x), "y": asdict(self.y),
                "pixel_origin": list(self.pixel_origin),
                "canvas_size": list(self.canvas_size)}


def _fit_linear(pos: np.ndarray) -> Optional[dict]:
    if len(pos) < 3:
        return None
    d = np.median(np.diff(pos))
    if d <= 0:
        return None
    k = np.round((pos - pos[0]) / d).astype(int)
    b, a = np.polyfit(k, pos, 1)                       # pos ≈ a + b*k
    resid = np.abs(pos - (a + b * k))
    inl = resid < 0.3 * abs(b)
    seen: dict[int, int] = {}
    for i in np.argsort(resid):                        # de-duplicate slots
        kk = int(k[i])
        if kk in seen:
            inl[i] = False
        elif inl[i]:
            seen[kk] = i
    return dict(scale="linear", inl=inl, spacing=float(abs(b)), n_in=int(inl.sum()))


def _fit_log(pos: np.ndarray, tol: float = 0.025) -> Optional[dict]:
    if len(pos) < 6:
        return None
    span = pos[-1] - pos[0]
    best = None
    for W in np.arange(25.0, span * 1.2 + 1.0, 2.0):
        base = (pos - pos[0]) / W
        for phi in np.linspace(0, 1, 41, endpoint=False):
            frac = (base + phi) % 1.0
            dd = np.min(np.abs(frac[:, None] - _LOG_MANT[None, :]), axis=1)
            dd = np.minimum(dd, np.minimum(frac, 1 - frac))
            inl = dd < tol
            sc = int(inl.sum())
            if best is None or sc > best["n_in"]:
                best = dict(scale="log", inl=inl, spacing=float(W), n_in=sc)
    return best


def fit_axis(centers: list[float]) -> dict:
    """
    Classify a set of gridline positions as linear or log and flag outliers.

    Linear grids are equidistant; a spurious line (e.g. a reference line drawn at
    a non-grid value) breaks that and is dropped.  Log grids have gaps that
    shrink across each decade — detected by the low fraction of near-median gaps;
    there the fit is kept conservative (all lines retained) since the exact log
    phase is harder to pin down than a spurious-line removal is worth.
    """
    pos = np.asarray(sorted(centers), float)
    if len(pos) < 4:
        lin = _fit_linear(pos)
        return {**(lin or {"scale": "linear", "spacing": None}),
                "pos": pos, "keep": np.ones(len(pos), bool)}
    diffs = np.diff(pos)
    d = np.median(diffs)
    regular = float(np.mean((diffs > 0.6 * d) & (diffs < 1.4 * d)))
    if regular >= 0.65:
        lin = _fit_linear(pos)
        return {**lin, "pos": pos, "keep": lin["inl"]}
    log = _fit_log(pos)
    if log is None:
        lin = _fit_linear(pos)
        return {**lin, "pos": pos, "keep": lin["inl"]}
    # log: keep all detected lines (conservative)
    return {**log, "pos": pos, "keep": np.ones(len(pos), bool)}


def _group_runs(cov: np.ndarray, min_cov: float, merge_gap: int) -> list[tuple[int, int]]:
    """Group indices whose coverage exceeds `min_cov` into (lo, hi) line spans."""
    idx = np.where(cov > min_cov)[0]
    if len(idx) == 0:
        return []
    runs = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= merge_gap:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    return runs


def model_grid(
    s: np.ndarray,
    v: np.ndarray,
    min_cov: float = 0.45,
    merge_gap: int = 2,
    resid_tol: float = 0.25,
    on_grid: float = 0.15,
    edge_pad: int = 1,
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Reconstruct the grid and return ``(grid_mask, grid_gray, v_lines, h_lines)``.

    * ``grid_gray`` – the modeled grid darkness in [0, 1] (the mathematical grid,
      full-length lines with their measured soft profile).
    * ``grid_mask`` – boolean pixels to suppress: ink that is grid and not a
      darker curve crossing it (``darkness - grid_gray < resid_tol``).
    * ``v_lines`` / ``h_lines`` – detected line spans ``(lo, hi)`` in px.

    Line positions come from ink projected onto each axis; the per-column /
    per-row MEDIAN darkness gives each line's profile robustly (crossing curves
    and text are outliers the median ignores).
    """
    h, w = v.shape
    ink = (s < 0.30) & (v < 0.90)
    darkness = 1.0 - v

    col_cov = ink.sum(axis=0) / float(h)
    row_cov = ink.sum(axis=1) / float(w)
    v_lines = _group_runs(col_cov, min_cov, merge_gap)
    h_lines = _group_runs(row_cov, min_cov, merge_gap)

    # Fit each axis (linear/log) and drop lines that break the pattern (a
    # spurious reference line, or a curve grazing a column).
    fx = fit_axis([(lo + hi) / 2 for lo, hi in v_lines])
    fy = fit_axis([(lo + hi) / 2 for lo, hi in h_lines])
    v_order = np.argsort([(lo + hi) / 2 for lo, hi in v_lines])
    h_order = np.argsort([(lo + hi) / 2 for lo, hi in h_lines])
    v_keep = [v_lines[v_order[i]] for i in range(len(v_lines)) if fx["keep"][i]]
    h_keep = [h_lines[h_order[i]] for i in range(len(h_lines)) if fy["keep"][i]]

    grid_gray = np.zeros((h, w), dtype=np.float32)
    for lo, hi in v_keep:
        a, b = max(0, lo - edge_pad), min(w, hi + edge_pad + 1)
        med = np.median(darkness[:, a:b], axis=0)          # per-column profile
        grid_gray[:, a:b] = np.maximum(grid_gray[:, a:b], med[None, :])
    for lo, hi in h_keep:
        a, b = max(0, lo - edge_pad), min(h, hi + edge_pad + 1)
        med = np.median(darkness[a:b, :], axis=1)          # per-row profile
        grid_gray[a:b, :] = np.maximum(grid_gray[a:b, :], med[:, None])

    residual = darkness - grid_gray
    grid_mask = ink & (grid_gray > on_grid) & (residual < resid_tol)

    vc = [(lo + hi) / 2 for lo, hi in v_keep]
    hc = [(lo + hi) / 2 for lo, hi in h_keep]
    model = GridModel(
        x=GridAxis(scale=fx["scale"], lines_px=vc,
                   px_lo=min(vc) if vc else 0.0, px_hi=max(vc) if vc else float(w),
                   spacing_px=fx.get("spacing")),
        y=GridAxis(scale=fy["scale"], lines_px=hc,
                   px_lo=min(hc) if hc else 0.0, px_hi=max(hc) if hc else float(h),
                   spacing_px=fy.get("spacing")),
        canvas_size=(w, h),
    )
    return grid_mask, grid_gray, model
