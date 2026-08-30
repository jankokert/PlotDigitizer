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

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

_LOG_MANT = np.log10(np.arange(1, 10))     # fractional log10 of 1..9 (minor lines)
_LOG_MANT_MINOR = np.log10(np.arange(2, 10))   # 2..9 only (a decade line is at 0/1)
# A log axis can run either way in pixel space (value up = ascending, or value
# down = descending, the common orientation for a Y current axis).  The mantissa
# lines then sit at log10(m) OR at 1 − log10(m); both keep the decade at 0.  The
# fit tries both and remembers which set matched, so predicted minor lines land
# on the real ones instead of their mirror image.
_MANT_ASC = _LOG_MANT
_MANT_DESC = (1.0 - _LOG_MANT) % 1.0


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
    minor_width: Optional[float] = None      # measured minor grid-line width (px)
    major_width: Optional[float] = None      # measured major/decade grid-line width (px)

    def to_dict(self) -> dict:
        return {"x": asdict(self.x), "y": asdict(self.y),
                "pixel_origin": list(self.pixel_origin),
                "canvas_size": list(self.canvas_size),
                "minor_width": self.minor_width,
                "major_width": self.major_width}


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
    return dict(scale="linear", inl=inl, spacing=float(abs(b)), n_in=int(inl.sum()),
                a=float(a), b=float(b), major=np.zeros(len(pos), bool))


def _fit_log(pos: np.ndarray, tol: float = 0.025) -> Optional[dict]:
    if len(pos) < 6:
        return None
    span = pos[-1] - pos[0]
    best = None
    for W in np.arange(25.0, span * 1.2 + 1.0, 2.0):
        base = (pos - pos[0]) / W
        for phi in np.linspace(0, 1, 41, endpoint=False):
            frac = (base + phi) % 1.0
            # DESC (value increases upward = toward negative pixels) is the normal
            # orientation of a plot's Y axis, so it is tried first and wins ties
            # (pref bonus); ASC only overrides it with strictly more inliers.
            for mant, pref in ((_MANT_DESC, 0.5), (_MANT_ASC, 0.0)):
                dd = np.min(np.abs(frac[:, None] - mant[None, :]), axis=1)
                dd = np.minimum(dd, np.minimum(frac, 1 - frac))
                inl = dd < tol
                sc = int(inl.sum())
                score = sc + pref
                if best is None or score > best["score"]:
                    best = dict(scale="log", inl=inl, spacing=float(W), n_in=sc,
                                score=score, W=float(W), phi=float(phi), mant=mant)
    # Refine W and the phase precisely.  The coarse search steps φ in 1/41 (≈ W·
    # 0.025 ≈ 3 px) and W in 2 px, so the best lattice can sit ~1–2 px off the real
    # ink — enough to paint a 1 px fringe of "modeled but white" (green) beside
    # every line.  Assign each inlier its lattice coordinate t = offset + decade
    # and least-squares fit ``pos ≈ anchor + W·t``, so predicted lines land on ink.
    W = best["W"]; phi = best["phi"]; mant = best["mant"]; p0 = float(pos[0])
    u = (pos - p0) / W + phi                            # continuous lattice coordinate
    kf = np.floor(u); frac = u - kf
    cand = np.concatenate([mant - 1.0, mant, mant + 1.0])   # allow decade wrap
    snap = cand[np.argmin(np.abs(frac[:, None] - cand[None, :]), axis=1)]
    t = kf + snap                                       # lattice coord of each line
    inl = best["inl"].copy()
    if int(inl.sum()) >= 3 and np.ptp(t[inl]) > 0:
        slope, intercept = np.polyfit(t[inl], pos[inl], 1)     # pos ≈ slope·t + intercept
        resid = np.abs(pos - (slope * t + intercept))
        inl &= resid < 0.35 * abs(slope) * 0.1 + 2.0   # reject lattice-misfit outliers
        if int(inl.sum()) >= 3 and np.ptp(t[inl]) > 0:
            slope, intercept = np.polyfit(t[inl], pos[inl], 1)
        best["W"] = float(abs(slope))
        best["anchor"] = float(intercept)
    else:
        best["anchor"] = float(p0 - W * phi)           # center = anchor + W·t
    # Tag each line major/minor from the SAME refined lattice the enumeration uses
    # (anchor + W·t): a decade (×10^k) sits at an integer lattice coord t, minor
    # lines 2..9 sit ≥0.046 away.  Using the coarse phase here instead would
    # mis-tag the endpoints (e.g. the top frame line 10000) as minor → drawn 1 px.
    tt = (pos - best["anchor"]) / best["W"]
    best["major"] = best["inl"] & (np.abs(tt - np.round(tt)) < 0.025)
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
        base = lin or {"scale": "linear", "spacing": None,
                       "major": np.zeros(len(pos), bool)}
        return {**base, "pos": pos, "keep": np.ones(len(pos), bool)}
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


def _measure_line(darkness: np.ndarray, lo: int, hi: int, pad: int, axis: str):
    """Robust (center, peak, effective-width) of one line from its soft profile.

    The profile is the per-column (vertical) / per-row (horizontal) MEDIAN
    darkness over the line's span; for a vertical line it ignores crossing curves
    (a minority of rows).  ``width_eff = Σ profile / peak`` is the line's true
    soft width in px (core 1 + two ~0.5 aliased edges ≈ 2).
    """
    if axis == "v":
        a, b = max(0, lo - pad), min(darkness.shape[1], hi + pad + 1)
        p = np.median(darkness[:, a:b], axis=0)
    else:
        a, b = max(0, lo - pad), min(darkness.shape[0], hi + pad + 1)
        p = np.median(darkness[a:b, :], axis=1)
    tot = float(p.sum())
    peak = float(p.max()) if p.size else 0.0
    if tot <= 1e-6 or peak <= 1e-6:
        return None
    xs = np.arange(a, b)
    center = float((xs * p).sum() / tot)
    return center, peak, tot / peak


def _stamp_line(grid_gray: np.ndarray, center: float, width: float,
                amp: float, axis: str) -> None:
    """Draw one line at a CONSISTENT width, sub-pixel centered.

    Coverage at integer pixel ``x`` is ``clip((width/2 + 0.5) - |x - center|, 0, 1)``
    — for width 2 on an integer center that is a full centre pixel and a 50 %
    pixel each side (= 2 px), exactly the intended soft profile, and it shifts
    smoothly for a sub-pixel center.  The same width is used for every line of a
    class, so an arrow lying on a line can no longer inflate it.
    """
    half = width / 2.0 + 0.5
    limit = grid_gray.shape[1] if axis == "v" else grid_gray.shape[0]
    lo = max(0, int(math.floor(center - half)))
    hi = min(limit - 1, int(math.ceil(center + half)))
    if hi < lo:
        return
    idx = np.arange(lo, hi + 1)
    cov = np.clip(half - np.abs(idx - center), 0.0, 1.0) * amp
    if axis == "v":
        grid_gray[:, idx] = np.maximum(grid_gray[:, idx], cov[None, :])
    else:
        grid_gray[idx, :] = np.maximum(grid_gray[idx, :], cov[:, None])


def _split_major(widths) -> np.ndarray:
    """Split measured line widths into a thin (minor) and thick (major) cluster.

    Used for a *linear* axis, where the fit gives no decade/minor tag: a plot's
    heavier lines (e.g. CMZ's even-value verticals at 2 px vs 1 px minors) show
    up as a distinctly wider group.  Unimodal widths → no majors.
    """
    w = np.asarray(widths, float)
    if len(w) < 3:
        return np.zeros(len(w), bool)
    lo, hi = float(w.min()), float(w.max())
    if hi < 1.4 * max(lo, 1e-6):
        return np.zeros(len(w), bool)
    return w >= 0.5 * (lo + hi)


def _model_positions(fit: dict, c_lo: float, c_hi: float) -> list[tuple[float, bool]]:
    """Every line the fitted model predicts across ``[c_lo, c_hi]``, tagged major.

    This is the point of the whole model: once the spacing/phase is fixed we KNOW
    where each line sits, so lines a projection missed because text (labels,
    legends) sat on them are reconstructed and subtracted just the same.  Log:
    center = p0 + W·(log10(m) − φ + k) for decade k and mantissa m∈1..9 (decade =
    major).  Linear: center = a + b·k (major from a learned period, e.g. even
    values every 2nd line).
    """
    out: list[tuple[float, bool]] = []
    scale = fit.get("scale")
    if scale == "log" and fit.get("W") and fit.get("anchor") is not None:
        W = float(fit["W"]); anchor = float(fit["anchor"])
        mant = fit.get("mant", _LOG_MANT)
        k0 = int(math.floor((c_lo - anchor) / W)) - 1   # center = anchor + W·(off+k)
        k1 = int(math.ceil((c_hi - anchor) / W)) + 1
        for k in range(k0, k1 + 1):
            for mi in range(len(mant)):
                c = anchor + W * (float(mant[mi]) + k)
                if c_lo <= c <= c_hi:
                    out.append((c, mi == 0))          # mant[0] == 0 → decade = major
    elif scale == "linear" and fit.get("b"):
        a = float(fit["a"]); b = float(fit["b"])
        k0 = int(math.floor((c_lo - a) / b)) - 1
        k1 = int(math.ceil((c_hi - a) / b)) + 1
        per = fit.get("maj_period"); res = fit.get("maj_res", 0)
        for k in range(min(k0, k1), max(k0, k1) + 1):
            c = a + b * k
            if c_lo <= c <= c_hi:
                out.append((c, bool(per) and ((k - res) % per == 0)))
    out.sort()
    ded: list[tuple[float, bool]] = []                 # merge coincident predictions
    for c, mj in out:
        if ded and abs(c - ded[-1][0]) < 0.75:
            if mj and not ded[-1][1]:
                ded[-1] = (ded[-1][0], True)
            continue
        ded.append((c, mj))
    return ded


def _line_support(darkness: np.ndarray, c: float, axis: str) -> float:
    """Fraction of ink along a line, taken as the WEAKER of its two halves.

    A true full-length grid line has ink across its whole span, so both halves
    score high; a localized dark blob that merely projected onto this row/column
    (a block of label text, a legend) fills only one region and scores ~0 in the
    other half — so it is not mistaken for a grid line.
    """
    if axis == "v":
        lo = max(0, int(math.floor(c)) - 1); hi = min(darkness.shape[1], int(math.ceil(c)) + 2)
        strip = darkness[:, lo:hi].max(axis=1)
    else:
        lo = max(0, int(math.floor(c)) - 1); hi = min(darkness.shape[0], int(math.ceil(c)) + 2)
        strip = darkness[lo:hi, :].max(axis=0)
    n = len(strip)
    if n < 4:
        return 0.0
    ink = strip > 0.25
    return float(min(ink[: n // 2].mean(), ink[n // 2:].mean()))


def _render_axis(grid_gray: np.ndarray, darkness: np.ndarray, pad: int,
                 kept: list[tuple[int, int]], majors: list[bool],
                 fit: dict, axis: str) -> tuple[Optional[float], Optional[float]]:
    """Render one axis at a consistent per-class width, filling model-predicted gaps.

    Two passes: (1) every DETECTED line at its own measured peak (so real profiles
    and curve-crossing residuals stay exact), then (2) every line the fitted MODEL
    predicts — including ones hidden under text that the projection never saw — at
    the class-median peak.  ``np.maximum`` merges the two, so nothing is weakened.
    """
    limit_dim = grid_gray.shape[1] if axis == "v" else grid_gray.shape[0]
    meas = []
    for (lo, hi), mj in zip(kept, majors):
        m = _measure_line(darkness, lo, hi, pad, axis)
        if m is not None and _line_support(darkness, m[0], axis) >= 0.15:
            meas.append([m[0], m[1], m[2], bool(mj)])   # a real full-length line, not a text blob
    if not meas:
        return None, None
    # linear axis: the fit tags no majors → classify the thick lines by width
    if fit.get("scale") == "linear" and not any(r[3] for r in meas):
        for row, f in zip(meas, _split_major([r[2] for r in meas])):
            row[3] = bool(f)
    minw = [wd for _, _, wd, mj in meas if not mj]
    majw = [wd for _, _, wd, mj in meas if mj]
    w_minor = float(np.median(minw)) if minw else (float(np.median(majw)) if majw else 2.0)
    w_major = float(np.median(majw)) if majw else w_minor
    w_minor = float(np.clip(w_minor, 1.0, 8.0))
    w_major = float(np.clip(w_major, 1.0, 8.0))
    allp = [pk for _, pk, _, _ in meas]
    minp = [pk for _, pk, _, mj in meas if not mj]
    majp = [pk for _, pk, _, mj in meas if mj]
    p_minor = float(np.median(minp)) if minp else float(np.median(allp))
    p_major = float(np.median(majp)) if majp else p_minor

    for center, peak, _wd, mj in meas:                 # pass 1: detected lines
        _stamp_line(grid_gray, center, w_major if mj else w_minor, peak, axis)

    centers = np.array(sorted(r[0] for r in meas))     # pass 2: fill ONLY gaps
    c_lo = max(0.0, float(centers.min()))
    c_hi = min(limit_dim - 1.0, float(centers.max()))
    efit = dict(fit)
    if fit.get("scale") == "linear" and fit.get("b"):
        a, b = float(fit["a"]), float(fit["b"])        # learn the major period (even-value spacing)
        mk = sorted(int(round((c - a) / b)) for c, _, _, mj in meas if mj)
        if len(mk) >= 2:
            diffs = np.diff(mk)
            per = int(diffs.min())
            if per >= 1 and np.all(diffs % per == 0):
                efit["maj_period"] = per
                efit["maj_res"] = mk[0] % per
    for c, mj in _model_positions(efit, c_lo, c_hi):
        j = int(np.searchsorted(centers, c))           # nearest already-detected line
        near = min(abs(centers[max(0, j - 1)] - c),
                   abs(centers[min(len(centers) - 1, j)] - c))
        if near <= 2.0:                                # already drawn accurately in pass 1
            continue
        # Only fill where the grid line REALLY is: a line the projection missed
        # because text sat on it still has dark ink in both halves; a position
        # that is simply white (a cut-off top decade, a blank margin) has none and
        # must not be invented (it would paint a phantom green line).
        if _line_support(darkness, c, axis) < 0.20:
            continue
        _stamp_line(grid_gray, c, w_major if mj else w_minor,
                    p_major if mj else p_minor, axis)
    return w_minor, (w_major if majw else None)


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
    # decade/minor class per kept line (aligned to the sorted keep order)
    vmaj_all = fx.get("major", np.zeros(len(v_lines), bool))
    hmaj_all = fy.get("major", np.zeros(len(h_lines), bool))
    v_major = [bool(vmaj_all[i]) for i in range(len(v_lines)) if fx["keep"][i]]
    h_major = [bool(hmaj_all[i]) for i in range(len(h_lines)) if fy["keep"][i]]

    # Render every line at a CONSISTENT width per class (decade vs minor): the
    # width is the class median, so a line an arrow happens to lie on is drawn at
    # the same 2 px as its peers instead of the inflated local ink width.
    grid_gray = np.zeros((h, w), dtype=np.float32)
    vw = _render_axis(grid_gray, darkness, edge_pad, v_keep, v_major, fx, "v")
    hw = _render_axis(grid_gray, darkness, edge_pad, h_keep, h_major, fy, "h")
    minors = [x for x in (vw[0], hw[0]) if x is not None]
    majors_w = [x for x in (vw[1], hw[1]) if x is not None]
    minor_width = float(np.median(minors)) if minors else None
    major_width = float(np.median(majors_w)) if majors_w else None

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
        minor_width=minor_width, major_width=major_width,
    )
    return grid_mask, grid_gray, model
