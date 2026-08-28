"""Extract curves from the plot canvas by color."""

import logging
from typing import Optional
import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


def extract_curves(
    img_array: np.ndarray,
    plot_area: tuple[int, int, int, int],
    method: str = "naive",
    n_samples: int = 500,
    smoothing: float = 1.0,
    n_curves: Optional[int] = None,
    has_grid: bool = False,
    sat_threshold: float = 0.20,
    min_col_coverage: float = 0.20,
    hue_bins: int = 18,
    span_frac: float = 0.55,
    max_thickness: Optional[int] = None,
    notch_factor: float = 3.0,
    target_colors: Optional[list[tuple[int, int, int]]] = None,
    color_tolerance: float = 40.0,
    debug: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """
    Extract digitised curves from the plot area.

    Args:
        n_curves:       If given, keep only the top-N color segments by pixel count.
        has_grid:       If True, suppress grid lines before colour segmentation.
        sat_threshold:  HSV saturation cutoff; lower values detect paler curves.
        min_col_coverage: Fraction of canvas width a colour must span to be kept.
        hue_bins:       Number of hue bins for colour segmentation (finer = more bins).
        span_frac:      Row/col must span this fraction of canvas to count as a grid line.
        max_thickness:  Max pixel cluster height for per-column extraction (auto if None).
        notch_factor:   Spread > max_thickness * notch_factor triggers deep-notch mode.
        target_colors:  If given, extract only pixels near these RGB colours
                        instead of automatic hue-bin segmentation.
        color_tolerance: Euclidean RGB distance for target-colour matching.

    Returns dict mapping colour label → Nx2 float array of
    (pixel_x, pixel_y) values in image coordinates.
    """
    x_min, y_min, x_max, y_max = plot_area
    canvas = img_array[y_min:y_max, x_min:x_max].astype(np.float32)

    color_masks = _segment_by_color(
        canvas,
        has_grid=has_grid,
        sat_threshold=sat_threshold,
        min_col_coverage=min_col_coverage,
        hue_bins=hue_bins,
        span_frac=span_frac,
        target_colors=target_colors,
        color_tolerance=color_tolerance,
        debug=debug,
    )

    # Keep only the top-N masks by pixel count (skip when the user picked colours)
    if target_colors is None and n_curves is not None and len(color_masks) > n_curves:
        ranked = sorted(color_masks.items(), key=lambda kv: kv[1].sum(), reverse=True)
        color_masks = dict(ranked[:n_curves])
        logger.info(f"Keeping top {n_curves} segments by pixel count")

    logger.info(f"Color segments found: {len(color_masks)}")

    curves: dict[str, np.ndarray] = {}
    for label, mask in color_masks.items():
        if method == "cv":
            pts = _extract_cv(mask, x_min, y_min)
        else:
            pts = _extract_naive(
                mask, x_min, y_min, n_samples, smoothing,
                max_thickness=max_thickness, notch_factor=notch_factor,
            )

        if pts is not None and len(pts) > 1:
            curves[label] = pts
            logger.info(f"  {label}: {len(pts)} points")

    return curves


# ---------------------------------------------------------------------------
# Color segmentation
# ---------------------------------------------------------------------------

def _segment_by_color(
    canvas: np.ndarray,
    sat_threshold: float = 0.20,
    min_pixel_fraction: float = 0.002,
    hue_bins: int = 18,
    has_grid: bool = False,
    min_col_coverage: float = 0.20,
    span_frac: float = 0.55,
    target_colors: Optional[list[tuple[int, int, int]]] = None,
    color_tolerance: float = 40.0,
    debug: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """
    Return {label: bool_mask} for each distinct colour group.

    Handles coloured curves (via hue binning) and achromatic gray curves.
    Optionally removes grid lines before analysis.
    If target_colors is given, match pixels by RGB proximity instead of hue bins.
    """
    r = canvas[:, :, 0] / 255.0
    g = canvas[:, :, 1] / 255.0
    b = canvas[:, :, 2] / 255.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    v = max_c
    safe_max = np.where(max_c > 1e-6, max_c, 1.0)
    s = np.where(max_c > 1e-6, delta / safe_max, 0.0)

    # Detect and suppress grid pixels before any further analysis
    grid_mask = _detect_grid(s, v, canvas.shape[:2], span_frac=span_frac) if has_grid else np.zeros(canvas.shape[:2], dtype=bool)
    if has_grid:
        grid_px = int(grid_mask.sum())
        logger.info(f"Grid pixels suppressed: {grid_px}")
    if debug is not None:
        debug["grid_mask"] = grid_mask

    if target_colors:
        return _segment_by_target_colors(
            canvas, target_colors, color_tolerance, grid_mask,
            min_col_coverage, min_pixel_fraction,
        )

    h_px, w_px = canvas.shape[:2]
    total_pixels = h_px * w_px
    # Scale pixel-fraction threshold proportionally when min_col_coverage is
    # lower than the default (0.20), so partial curves aren't doubly rejected.
    effective_min_px_frac = min_pixel_fraction * (min_col_coverage / 0.20)

    # --- Auto-detect achromatic (black/white) plots ----------------------
    # When the plot carries no real colour (datasheet line on a grid), hue and
    # gray segmentation only produce garbage from grid / anti-aliasing bands.
    # Switch to a single dark-line extraction instead — this is the automatic
    # fallback the user asked for ("no colours → new method").
    colored_fg = (s > sat_threshold) & (v > 0.15) & ~grid_mask
    colored_cols = int(colored_fg.any(axis=0).sum())
    if colored_cols < w_px * min_col_coverage:
        logger.info(
            f"Achromatic plot detected ({colored_cols}/{w_px} coloured columns) "
            f"— using dark-line extraction only"
        )
        return _segment_achromatic(
            s, v, grid_mask, w_px, total_pixels,
            min_col_coverage, effective_min_px_frac,
        )

    # Hue in [0, 360)
    hue = np.zeros_like(r)
    eps = 1e-8
    mask_r = (max_c == r) & (delta > eps)
    mask_g = (max_c == g) & (delta > eps)
    mask_b = (max_c == b) & (delta > eps)
    hue[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    hue[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120
    hue[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240

    # Foreground: saturated, not too dark, not a grid line
    fg = colored_fg

    masks: dict[str, np.ndarray] = {}

    if fg.sum() >= 10:
        bin_size = 360.0 / hue_bins
        hue_bin = (hue / bin_size).astype(int) % hue_bins

        for b_idx in range(hue_bins):
            bin_mask = fg & (hue_bin == b_idx)
            if bin_mask.sum() < total_pixels * effective_min_px_frac:
                continue
            col_hits = bin_mask.any(axis=0).sum()
            if col_hits < w_px * min_col_coverage:
                continue
            r_mean = int(canvas[:, :, 0][bin_mask].mean())
            g_mean = int(canvas[:, :, 1][bin_mask].mean())
            b_mean = int(canvas[:, :, 2][bin_mask].mean())
            label = f"curve_h{b_idx:02d}_rgb{r_mean:03d}{g_mean:03d}{b_mean:03d}"
            masks[label] = bin_mask
    else:
        logger.warning("No saturated foreground pixels found in plot area")

    # --- Gray / achromatic curves ----------------------------------------
    gray_fg = (s < 0.15) & (v > 0.25) & (v < 0.90) & ~grid_mask
    n_gray_bins = 8
    gray_bin_size = (0.90 - 0.25) / n_gray_bins
    gray_candidates: list[tuple[str, np.ndarray, np.ndarray]] = []

    for i in range(n_gray_bins):
        v_lo = 0.25 + i * gray_bin_size
        v_hi = v_lo + gray_bin_size
        bin_mask = gray_fg & (v >= v_lo) & (v < v_hi)
        if bin_mask.sum() < total_pixels * effective_min_px_frac:
            continue
        col_profile = bin_mask.any(axis=0)
        if col_profile.sum() < w_px * min_col_coverage:
            continue
        v_mean = int((v_lo + v_hi) / 2 * 255)
        gray_candidates.append((f"curve_gray_v{v_mean:03d}", bin_mask, col_profile))

    # Merge gray bins with >60 % column overlap (same curve, different shade)
    used = [False] * len(gray_candidates)
    for i, (label_i, mask_i, col_i) in enumerate(gray_candidates):
        if used[i]:
            continue
        combined = mask_i.copy()
        for j in range(i + 1, len(gray_candidates)):
            if used[j]:
                continue
            col_j = gray_candidates[j][2]
            overlap = (col_i & col_j).sum() / max(col_i.sum(), col_j.sum(), 1)
            if overlap > 0.60:
                combined |= gray_candidates[j][1]
                used[j] = True
        masks[label_i] = combined
        used[i] = True

    # --- Black / near-black curves ---------------------------------------
    # A thick black data line (v ≈ 0) is excluded from the gray range above.
    # After grid suppression, remaining dark ink in the plot area is the
    # curve itself (isolated text blobs fail the column-coverage test below).
    black_fg = (s < 0.30) & (v < 0.25) & ~grid_mask
    if black_fg.sum() >= total_pixels * effective_min_px_frac:
        col_profile = black_fg.any(axis=0)
        if col_profile.sum() >= w_px * min_col_coverage:
            masks["curve_black"] = black_fg

    return masks


def _segment_by_target_colors(
    canvas: np.ndarray,
    target_colors: list[tuple[int, int, int]],
    color_tolerance: float,
    grid_mask: np.ndarray,
    min_col_coverage: float,
    min_pixel_fraction: float,
) -> dict[str, np.ndarray]:
    """
    Build one mask per user-picked RGB colour using Euclidean proximity.

    Exact channel equality is not required: anti-aliased / JPEG pixels around
    the curve still match if they fall within color_tolerance.
    """
    h_px, w_px = canvas.shape[:2]
    total_pixels = h_px * w_px
    coverage = min(min_col_coverage, 0.05)
    effective_min_px_frac = min_pixel_fraction * (coverage / 0.20)

    masks: dict[str, np.ndarray] = {}
    for i, rgb in enumerate(target_colors):
        tr, tg, tb = (int(c) for c in rgb[:3])
        tr = max(0, min(255, tr))
        tg = max(0, min(255, tg))
        tb = max(0, min(255, tb))
        diff = canvas - np.array([tr, tg, tb], dtype=np.float32)
        dist = np.sqrt((diff * diff).sum(axis=2))
        bin_mask = (dist <= color_tolerance) & ~grid_mask
        n_px = int(bin_mask.sum())
        if n_px < total_pixels * effective_min_px_frac:
            logger.warning(
                f"Target colour rgb({tr},{tg},{tb}): only {n_px} pixels "
                f"(tolerance={color_tolerance:.0f})"
            )
            continue
        col_hits = int(bin_mask.any(axis=0).sum())
        if col_hits < w_px * coverage:
            logger.warning(
                f"Target colour rgb({tr},{tg},{tb}): spans {col_hits}/{w_px} columns"
            )
            continue
        label = f"curve_hint{i:02d}_rgb{tr:03d}{tg:03d}{tb:03d}"
        masks[label] = bin_mask
        logger.info(
            f"Target colour {label}: {n_px} px, {col_hits} columns "
            f"(tolerance={color_tolerance:.0f})"
        )
    return masks


def _segment_achromatic(
    s: np.ndarray,
    v: np.ndarray,
    grid_mask: np.ndarray,
    w_px: int,
    total_pixels: int,
    min_col_coverage: float,
    effective_min_px_frac: float,
) -> dict[str, np.ndarray]:
    """
    Extract a single dark data line from a black/white (achromatic) plot.

    Tries progressively lighter value cut-offs and keeps the darkest one that
    still spans enough of the canvas, so a pure-black line is preferred but a
    mid-grey line is caught as a fallback.  Only one mask ("curve_black") is
    returned — colour/gray binning is deliberately skipped here.
    """
    masks: dict[str, np.ndarray] = {}
    for v_thresh in (0.25, 0.40, 0.55):
        bin_mask = (s < 0.35) & (v < v_thresh) & ~grid_mask
        if bin_mask.sum() < total_pixels * effective_min_px_frac:
            continue
        if bin_mask.any(axis=0).sum() < w_px * min_col_coverage:
            continue
        masks["curve_black"] = bin_mask
        logger.info(
            f"Dark line captured at v<{v_thresh} "
            f"({int(bin_mask.sum())} px, "
            f"{int(bin_mask.any(axis=0).sum())}/{w_px} columns)"
        )
        break
    else:
        logger.warning("Achromatic plot: no dark line met the coverage threshold")
    return masks


def _detect_grid(
    s: np.ndarray,
    v: np.ndarray,
    shape: tuple[int, int],
    span_frac: float = 0.55,
) -> np.ndarray:
    """
    Identify grid / spine pixels to suppress before curve extraction.

    Grid lines are separated from the data curve **geometrically**, not by
    colour — this is essential for datasheet plots where the grid and the
    curve are both black (colour segmentation alone cannot tell them apart).

    Pass 1 — light periodic grid lines (pale grey mesh):
        Achromatic *light* rows/columns spanning ≥ span_frac of the canvas.

    Pass 2 — dark / black grid lines (and the plot frame):
        A true grid line is thin yet spans (almost) the entire width (a
        horizontal line) or height (a vertical line).  A data curve is a
        function: it is thick but never fills a whole row/column.  So we
        flag any achromatic *ink* row/column whose coverage is ≥ full_span
        (a deliberately high threshold, ~0.9) as a grid line.  The curve
        survives because even its flat tail covers well under 90 % of the
        width in any single row.  Curve pixels sitting on a grid crossing
        are lost as small gaps, which the spline fit bridges cleanly.

    Pass 3 — canvas border / tick-mark area:
        * Top & sides : small fixed margin (spine line width).
        * Bottom      : larger margin (h // 10) to cover x-axis tick marks
          and their anti-aliasing bleed, which can look like gray curves.
    """
    h, w = shape

    grid_mask = np.zeros((h, w), dtype=bool)

    # --- Pass 1: light grey grid mesh -----------------------------------
    light = (s < 0.12) & (v > 0.60) & (v < 0.97)
    grid_mask[light.sum(axis=1) > w * span_frac, :] = True
    grid_mask[:, light.sum(axis=0) > h * span_frac] = True

    # --- Pass 2: dark / black grid lines + frame ------------------------
    # Any achromatic ink pixel (grid, curve, frame all qualify here).
    ink = (s < 0.30) & (v < 0.45)
    # Full-span threshold: a grid line spans nearly the whole axis; a curve
    # never does.  Kept high so the curve's flat tail is not mistaken for a
    # horizontal grid line.
    full_span = 0.90
    grid_mask[ink.sum(axis=1) > w * full_span, :] = True
    grid_mask[:, ink.sum(axis=0) > h * full_span] = True

    # --- Pass 3: fixed border ------------------------------------------
    # Just the spine line width on every side.  The x-axis line and its tick
    # bleed are already removed by the full-span passes above, so a big bottom
    # margin is no longer needed (it used to blank the whole lower tenth).
    top = 5
    side = 5
    bottom = 5

    grid_mask[:top, :] = True
    grid_mask[-bottom:, :] = True
    grid_mask[:, :side] = True
    grid_mask[:, -side:] = True

    return grid_mask


# ---------------------------------------------------------------------------
# Naive extraction: per-column tight-cluster median + outlier filter + spline
# ---------------------------------------------------------------------------

def _tight_cluster_median(
    active: np.ndarray,
    max_thickness: int = 30,
    min_pixels: int = 2,
    notch_factor: float = 3.0,
) -> Optional[float]:
    """
    Return the median y of the densest compact group of pixels in `active`.

    Uses a sliding-window approach: find the window of height ≤ max_thickness
    that contains the most pixels, then return its median.  This makes the
    extractor robust to scattered artefacts (axis lines, grid ticks) that
    appear alongside the actual curve pixels in a column.

    Returns None if no window meets the min_pixels requirement.
    """
    if active.size == 0:
        return None

    a = np.sort(active)

    if a[-1] - a[0] <= max_thickness:
        # All pixels already form a tight cluster
        return float(np.median(a))

    # Very wide spread *with high pixel density* indicates a near-vertical segment
    # (e.g. deep resonance notch) — use the deepest point (max y).
    # A sparse wide spread is noise alongside the real curve; fall through to the
    # sliding-window logic which finds the denser cluster instead.
    if a[-1] - a[0] > max_thickness * notch_factor:
        density = len(a) / (a[-1] - a[0] + 1)
        if density > 0.4:
            return float(a[-1])

    # Sliding window O(n)
    best_count = 0
    best_lo = 0
    lo = 0
    for hi in range(len(a)):
        while a[hi] - a[lo] > max_thickness:
            lo += 1
        count = hi - lo + 1
        if count > best_count:
            best_count = count
            best_lo = lo

    if best_count < min_pixels:
        return None

    cluster = a[best_lo : best_lo + best_count]
    return float(np.median(cluster))

def _extract_naive(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
    n_samples: int,
    smoothing: float = 1.0,
    max_thickness: Optional[int] = None,
    notch_factor: float = 3.0,
) -> Optional[np.ndarray]:
    """
    For each x-column, find the median y of active pixels.
    Apply a median-filter pass to reject outlier columns, then fit a spline.
    """
    h, w = mask.shape
    thickness = max_thickness if max_thickness is not None else max(8, h // 25)
    xs_raw: list[int] = []
    ys_raw: list[float] = []

    for x in range(w):
        active = np.where(mask[:, x])[0]
        if active.size == 0:
            continue
        y_val = _tight_cluster_median(active, max_thickness=thickness, min_pixels=2,
                                      notch_factor=notch_factor)
        if y_val is None:
            continue
        xs_raw.append(x)
        ys_raw.append(y_val)

    if len(xs_raw) < 4:
        return None

    xs = np.array(xs_raw)
    ys = np.array(ys_raw)

    # Outlier rejection: remove points deviating > 3 MAD from local median
    ys = _reject_outliers(xs, ys)

    if len(xs) < 4:
        return None

    # Smoothing spline
    try:
        s_param = max(len(xs) * 0.5 * smoothing, 1.0)
        spl = interpolate.UnivariateSpline(xs, ys, k=3, s=s_param)
        xs_dense = np.linspace(xs[0], xs[-1], n_samples)
        ys_dense = spl(xs_dense)
    except Exception as exc:
        logger.debug(f"Spline failed ({exc}), using raw points")
        xs_dense = xs.astype(float)
        ys_dense = ys

    return np.column_stack([xs_dense + x_offset, ys_dense + y_offset])


def _reject_outliers(
    xs: np.ndarray,
    ys: np.ndarray,
    window: int = 21,
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Remove outlier y-values using a sliding-window MAD filter.
    Returns filtered ys (xs are modified in-place via boolean indexing caller-side).
    """
    n = len(ys)
    half = window // 2
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = ys[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        if mad < 1e-6:
            continue
        if abs(ys[i] - med) > threshold * mad * 1.4826:
            keep[i] = False

    # Re-assign xs to share the filter result (caller uses returned ys length)
    xs[:] = xs  # no-op — caller already has xs; we just shrink ys
    # Return filtered versions by rebuilding — caller must re-slice xs too.
    # We overwrite xs in-place via a trick: store result length in global? No.
    # Simpler: just zero outliers to local median instead of dropping them.
    result = ys.copy()
    for i in range(n):
        if not keep[i]:
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            result[i] = np.median(ys[lo:hi])

    return result


# ---------------------------------------------------------------------------
# CV extraction: OpenCV skeleton
# ---------------------------------------------------------------------------

def _extract_cv(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> Optional[np.ndarray]:
    """
    Morphological skeleton → per-column median y.
    Falls back to naive if OpenCV is unavailable.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not available, falling back to naive method")
        return _extract_naive(mask, x_offset, y_offset, n_samples=500)

    img_u8 = mask.astype(np.uint8) * 255
    skeleton = np.zeros_like(img_u8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = img_u8.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded
        if cv2.countNonZero(temp) == 0:
            break

    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return None

    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    xs_u, inverse = np.unique(xs, return_inverse=True)
    ys_u = np.array([np.median(ys[inverse == i]) for i in range(len(xs_u))])

    return np.column_stack([xs_u + x_offset, ys_u + y_offset])
