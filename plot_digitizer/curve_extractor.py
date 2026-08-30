"""Extract curves from the plot canvas by color."""

import logging
import math
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
    if debug is not None and debug.get("grid_model") is not None:
        debug["grid_model"]["pixel_origin"] = [x_min, y_min]

    # Keep only the top-N masks by pixel count (skip when the user picked colours)
    if target_colors is None and n_curves is not None and len(color_masks) > n_curves:
        ranked = sorted(color_masks.items(), key=lambda kv: kv[1].sum(), reverse=True)
        color_masks = dict(ranked[:n_curves])
        logger.info(f"Keeping top {n_curves} segments by pixel count")

    logger.info(f"Color segments found: {len(color_masks)}")

    curves: dict[str, np.ndarray] = {}
    for label, mask in color_masks.items():
        if method == "trace":
            # Arc-length tracing may split one mask into several curves
            # (e.g. same-colour curves that cross) — emit each as its own label.
            paths = _extract_trace(mask, x_min, y_min)
            for k, pts in enumerate(paths):
                lbl = label if len(paths) == 1 else f"{label}_seg{k:02d}"
                curves[lbl] = pts
                logger.info(f"  {lbl}: {len(pts)} points (trace)")
            continue

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
    arrows: list[dict] = []
    bridge = np.zeros(canvas.shape[:2], dtype=bool)
    label_arrows: list[dict] = []
    _ocr_labels: list[dict] = []
    arrow_mask = np.zeros(canvas.shape[:2], dtype=bool)
    grid_gray = np.zeros(canvas.shape[:2], dtype=np.float32)
    if has_grid:
        # 1-pass: reconstruct the mathematical grid and subtract it in grayscale
        # (curves darker than the modeled grid survive the crossings).
        from .grid_model import model_grid
        grid_mask, grid_gray, grid_obj = model_grid(s, v)
        logger.info(
            f"Grid: x={grid_obj.x.scale}({len(grid_obj.x.lines_px)} lines) "
            f"y={grid_obj.y.scale}({len(grid_obj.y.lines_px)} lines); "
            f"{int(grid_mask.sum())} px suppressed"
        )
        if debug is not None:
            debug["grid_model"] = grid_obj.to_dict()
        # Then subtract annotation-arrow shafts (they survive as they are darker
        # than the grid and derail the tracer where they cross a curve); the
        # removal is crossing-protected so no gap is punched.  (OCR ~4 s.)
        from .annotation_detector import detect_text_labels_ocr, detect_arrows
        _ink0 = (s < 0.30) & (v < 0.90)
        _ocr_labels = detect_text_labels_ocr(canvas)
        label_arrows = detect_arrows(canvas, grid_mask, labels=_ocr_labels)
        if label_arrows:
            arrow_mask = _arrow_removal_mask(label_arrows, _ink0)
            grid_mask |= arrow_mask
    else:
        grid_mask = np.zeros(canvas.shape[:2], dtype=bool)
    if debug is not None:
        from .annotation_detector import detect_legend_boxes, expand_to_white, detect_grid_boxes
        _ink = (s < 0.30) & (v < 0.90)
        debug["grid_mask"] = grid_mask
        debug["grid_gray"] = grid_gray
        debug["ink"] = _ink
        debug["arrows"] = arrows
        debug["arrow_mask"] = arrow_mask
        debug["label_arrows"] = label_arrows
        # Text-label boxes = clean OCR word boxes (glyph clustering merged whole
        # label rows into bogus mega-boxes; word OCR gives one tight box each).
        tboxes = [lab["box"] for lab in _ocr_labels]
        label_texts = [lab["text"] for lab in _ocr_labels]
        # A legend is a genuine vertical stack of aligned labels (USFF 0.05–0.25
        # A); scattered callouts yield none. Grow it to the white-box edges.
        legends = [expand_to_white(_ink, lg) for lg in detect_legend_boxes(tboxes)]
        # Arrow-less notes on a white patch (e.g. "f=1.0MHz") are found by the
        # hole they punch in the grid, and come with their OCR reading already.
        for gbox, gtext in detect_grid_boxes(canvas, span_frac=span_frac):
            tboxes.append(gbox)
            label_texts.append(gtext)
        debug["text_boxes"] = tboxes
        debug["legend_boxes"] = legends
        debug["label_texts"] = label_texts

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
            min_col_coverage, effective_min_px_frac, bridge,
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
    black_fg = ((s < 0.30) & (v < 0.25) & ~grid_mask) | bridge
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
    bridge: Optional[np.ndarray] = None,
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
        if bridge is not None:
            bin_mask = bin_mask | bridge   # synthetic arrowhead connectors
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


def _run_length(mask: np.ndarray, axis: int) -> np.ndarray:
    """
    For every True pixel, return the length of the contiguous True run it
    belongs to along `axis` (0 = vertical, 1 = horizontal); 0 elsewhere.

    This measures local line *thickness*: a 1–2 px grid line has a small run
    perpendicular to itself, while a thick data curve has a large one.
    """
    if axis == 1:
        return _run_length(mask.T, axis=0).T

    m = mask.astype(np.int32)
    n = m.shape[0]
    # end[i] = length of the run ending at row i (0 if not ink)
    end = np.zeros_like(m)
    end[0] = m[0]
    for i in range(1, n):
        end[i] = (end[i - 1] + 1) * m[i]
    # Propagate the full run length back up to every pixel in the run.
    length = np.zeros_like(m)
    length[n - 1] = end[n - 1]
    for i in range(n - 2, -1, -1):
        cont = mask[i] & mask[i + 1]
        length[i] = np.where(mask[i], np.where(cont, length[i + 1], end[i]), 0)
    return length


def _arrow_removal_mask(arrows: list[dict], ink: np.ndarray,
                        cross_thick: int = 8) -> np.ndarray:
    """
    Mask the horizontal callout-arrow shafts for subtraction, **sparing curve
    crossings** so no gap is punched in the data curve.

    Walk each arrow tail→tip.  At every column the local vertical ink run is the
    line thickness there: a bare arrow shaft is only ~4–5 px tall, a (near-)
    vertical curve crossing the shaft is much taller.  Columns thicker than
    ``cross_thick`` are a curve and kept; the rest are the arrow and removed —
    only the measured core, leaving the outermost aliased row on each side.
    Nothing is hardcoded per file; the band comes from the pixels.
    """
    h, w = ink.shape
    mask = np.zeros((h, w), dtype=bool)
    for a in arrows:
        ry = int(a["tip_y"])
        lo = int(min(a["tip_x"], a["tail_x"]))
        hi = int(max(a["tip_x"], a["tail_x"]))
        for xx in range(max(0, lo), min(w, hi + 1)):
            # snap to the shaft row (it may sit ±2 px off tip_y)
            ryy = None
            for dy in (0, -1, 1, -2, 2):
                if 0 <= ry + dy < h and ink[ry + dy, xx]:
                    ryy = ry + dy
                    break
            if ryy is None:
                continue
            up = 0
            while ryy - up - 1 >= 0 and ink[ryy - up - 1, xx]:
                up += 1
            dn = 0
            while ryy + dn + 1 < h and ink[ryy + dn + 1, xx]:
                dn += 1
            if up + dn + 1 > cross_thick:      # a curve crosses here → keep
                continue
            mask[ryy - up + 1:ryy + dn, xx] = True   # core, spare aliased edges
    return mask


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
# Trace extraction: arc-length curve following on the thick trace
# ---------------------------------------------------------------------------

def _extract_trace(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> list[np.ndarray]:
    """
    Follow each curve along its arc length (not column-by-column).

    The plotted trace is a thick "tube" (many overlapping pen dots).  We ridge-
    follow it: each step advances a fixed distance along the local heading, then
    re-centres *perpendicular* to the heading using the median of the local
    cross-section.  Decoupling forward motion from lateral centring keeps the
    centre-line smooth (no side-to-side oscillation) and monotone along the
    curve — data points are measurements, locally monotone, never doubling back.

    Each connected component is traced from its **middle** outward to both ends
    (cleaner end handling than starting at a tip).  A component containing a
    crossing yields several centre-lines; each is returned separately, in image
    pixel coordinates.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    if mask.sum() < 8:
        return []

    # Tube width from the distance transform → step size and search radius.
    dist = ndimage.distance_transform_edt(mask)
    width = max(2.0, 2.0 * float(np.median(dist[mask])))
    step = max(2.0, 0.6 * width)
    search_r = 1.6 * width + step        # large enough to bridge grid-crossing gaps
    cover_r = max(2.0, 0.9 * width)

    # One global point set: the mask is punched with small holes at every grid
    # crossing, so we must let the march bridge gaps rather than rely on
    # connected components (which would shatter one curve into many fragments).
    coords = np.argwhere(mask)[:, ::-1].astype(float)  # (x, y)
    tree = cKDTree(coords)
    covered = np.zeros(len(coords), dtype=bool)

    paths: list[np.ndarray] = []

    while True:
        remaining = np.where(~covered)[0]
        if len(remaining) == 0:
            break
        # Seed at an extreme (left-most, then top-most) uncovered pixel — a real
        # curve end.  We then march to the far end; ends fall out naturally.
        seed_i = remaining[np.lexsort((coords[remaining, 1], coords[remaining, 0]))[0]]
        seed = coords[seed_i].copy()

        if len(tree.query_ball_point(seed, cover_r)) < 3:
            covered[seed_i] = True
            continue

        # Start on the centre-line of the tip blob, not its raw edge pixel.
        seed_center = coords[tree.query_ball_point(seed, cover_r)].mean(axis=0)

        pre = covered.copy()
        d0 = _local_direction(tree, coords, seed, search_r)
        fwd = _march(tree, coords, seed_center, d0, step, search_r,
                     cover_r, width)
        bwd = _march(tree, coords, seed_center, -d0, step, search_r,
                     cover_r, width)
        covered[tree.query_ball_point(seed, cover_r)] = True

        path = list(reversed(bwd)) + [seed_center] + fwd
        arr = np.array(path, dtype=float)
        # Cover this marched footprint in one batched KD-tree query — replaces
        # the per-step query that used to run inside _march (half the tracer's
        # tree lookups).  Done for every path, kept or dropped, so a dropped
        # streak or short stub is not re-seeded.
        for grp in tree.query_ball_point(arr, cover_r):
            covered[grp] = True
        if len(path) < 5:
            continue

        _, nn = tree.query(arr)
        if pre[nn].mean() > 0.5:        # duplicate re-trace
            continue

        extent = np.hypot(*(arr.max(axis=0) - arr.min(axis=0)))
        if extent < max(6.0 * width, 0.06 * max(mask.shape)):
            continue

        # Drop flat, near-horizontal streaks (leftover grid lines / label
        # arrows): a data curve has real vertical extent.
        span = arr.max(axis=0) - arr.min(axis=0)   # (dx, dy)
        flat_thresh = max(8.0 * width, 0.035 * mask.shape[0])
        if span[1] < flat_thresh and span[1] < 0.2 * span[0]:
            # Mark the whole streak covered so its off-centre rows aren't
            # re-seeded and marched a second time (the doubled log lines) — a
            # confirmed non-curve, so covering its pixels is safe.
            for grp in tree.query_ball_point(arr, 1.5 * width):
                covered[grp] = True
            logger.info(
                f"  (dropped horizontal annotation/grid streak: "
                f"dx={span[0]:.0f} dy={span[1]:.0f})"
            )
            continue

        arr[:, 0] += x_offset
        arr[:, 1] += y_offset
        arr = _clean_path(arr, width)
        # Curvature-adaptive density: ~TARGET_DENSITY points across the canvas
        # width where the curve bends, up to 3× sparser on straight runs.
        base_step = mask.shape[1] / _TARGET_DENSITY
        paths.append(_adaptive_resample(arr, base_step))

    paths.sort(key=len, reverse=True)
    return paths


_TARGET_DENSITY = 100  # base points across the canvas width (curvy regions)


def _adaptive_resample(path: np.ndarray, base_step: float,
                       max_factor: float = 3.0) -> np.ndarray:
    """
    Resample a curve so point spacing follows its curvature: ≈ base_step where
    the curve bends, growing up to max_factor·base_step on straight runs (so a
    ruler-straight, steep segment carries few points and no lateral pendulum).
    """
    if len(path) < 3 or base_step <= 0:
        return path
    seg = np.diff(path, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    L = float(s[-1])
    if L <= base_step:
        return np.vstack([path[0], path[-1]])

    def at(q: float) -> np.ndarray:
        i = int(np.searchsorted(s, q) - 1)
        i = min(max(i, 0), len(seglen) - 1)
        t = (q - s[i]) / (seglen[i] if seglen[i] > 1e-9 else 1.0)
        return path[i] + t * (path[i + 1] - path[i])

    def straightness(q: float) -> float:
        a, b = max(0.0, q - base_step), min(L, q + base_step)
        pa, pb = at(a), at(b)
        arc = b - a
        return float(np.hypot(*(pb - pa)) / arc) if arc > 1e-9 else 1.0

    out = [path[0]]
    pos = 0.0
    while pos < L:
        sigma = straightness(pos)                 # 1 = straight, <1 = curvy
        grow = np.clip((sigma - 0.9) / 0.1, 0.0, 1.0)
        pos += base_step * (1.0 + (max_factor - 1.0) * grow)
        if pos < L:
            out.append(at(pos))
    out.append(path[-1])
    return np.array(out)


def _clean_path(path: np.ndarray, width: float) -> np.ndarray:
    """
    Stabilise a traced curve so it reads as the *function* it is:

    (1) A rolling-median filter (window 5) along the arc removes the lateral
        oscillation the beaded tube produces on steep, near-vertical segments —
        there the x-position must not pendulum back and forth.  Median (not
        mean) keeps genuine corners (e.g. the BAT46GW knee) sharp.
    (2) A short sharp hook at either end (crossing/arrow artefact) is trimmed.

    This corrects the trace along its own arc; it does not resample per x-column
    (which was the multivalued failure the tracer was built to avoid).
    """
    if len(path) < 7:
        return path

    from scipy.signal import savgol_filter

    n = len(path)
    med = path.copy()
    k = 2  # half-window → length-5 median (kills spikes, keeps corners)
    for i in range(k, n - k):
        med[i, 0] = np.median(path[i - k:i + k + 1, 0])
        med[i, 1] = np.median(path[i - k:i + k + 1, 1])

    # Savitzky-Golay (quadratic, window 7) removes residual lateral jitter while
    # a local quadratic keeps genuine corners (e.g. the BAT46GW knee) sharp.
    win = 7 if n >= 7 else (n if n % 2 else n - 1)
    out = med.copy()
    if win >= 5:
        out[:, 0] = savgol_filter(med[:, 0], win, 2)
        out[:, 1] = savgol_filter(med[:, 1], win, 2)

    def _turn(a, b, c):
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(u.dot(v) / (nu * nv), -1, 1))))

    lo_i, hi_i = 0, len(out)
    for _ in range(3):
        if hi_i - lo_i >= 5 and _turn(out[hi_i - 3], out[hi_i - 2], out[hi_i - 1]) > 70:
            hi_i -= 1
        else:
            break
    for _ in range(3):
        if hi_i - lo_i >= 5 and _turn(out[lo_i], out[lo_i + 1], out[lo_i + 2]) > 70:
            lo_i += 1
        else:
            break
    return out[lo_i:hi_i]


def _local_direction(tree, coords, seed, radius) -> np.ndarray:
    """Principal (PCA) direction of the mask pixels around `seed`."""
    idxs = tree.query_ball_point(seed, radius)
    if len(idxs) < 2:
        return np.array([1.0, 0.0])
    pts = coords[idxs] - coords[idxs].mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    d = vt[0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def _march(tree, coords, start, direction, step, search_r, cover_r,
           width, max_steps: int = 6000) -> list[np.ndarray]:
    """
    Walk from `start` along `direction`, one `step` at a time, returning the
    ordered centre-line points (excluding the start point).

    Each step: look in a forward cone (bridges the small holes the grid removal
    punches), take the centroid of the forward pixels — this reliably finds the
    way ahead and turns corners.  Then re-centre that point *perpendicular* to
    the heading using the median of the local cross-section: this removes the
    side-to-side oscillation the plain centroid showed on the beaded trace,
    while forward progress keeps the centre-line monotone along the curve.
    Continues over already-covered pixels (to pass crossings); stops when no
    mask remains ahead.
    """
    STRAIGHT = np.cos(np.radians(45.0))   # window that defines "goes straight"
    AHEAD = np.cos(np.radians(107.0))     # laxer cone so sharp bends still count

    pts: list[np.ndarray] = []
    p = start.astype(float).copy()
    d = direction / (math.hypot(direction[0], direction[1]) + 1e-12)

    # Coil guard: a curve's travelled length stays close to its bounding-box
    # extent (ratio ≈ 1); a compact text/annotation blob makes the march spiral,
    # so its length balloons far past its extent.  Bail once that happens instead
    # of grinding to max_steps — this is what made the flat "streak" drops slow.
    x0 = x1 = float(p[0])
    y0 = y1 = float(p[1])

    for _ in range(max_steps):
        q = p + step * d
        idxs = np.asarray(tree.query_ball_point(q, search_r), dtype=int)
        if idxs.size == 0:
            break

        rel = coords[idxs] - p
        # ‖rel‖ per row — sqrt(sum of squares) is the same value as
        # np.linalg.norm(axis=1) but skips its generic dispatch, which is the
        # tracer's single hottest line.
        norms = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        good = norms > 1e-6
        idxs, rel, norms = idxs[good], rel[good], norms[good]
        if idxs.size == 0:
            break

        cosang = rel.dot(d) / norms
        ahead = cosang > AHEAD
        if not ahead.any():
            break
        idxs_a = idxs[ahead]
        cos_a = cosang[ahead]

        # Prefer the straightest continuation (splits crossings); fall back to
        # all forward pixels at a genuine sharp bend.
        straight = cos_a > STRAIGHT
        chosen = idxs_a[straight] if straight.any() else idxs_a

        pts_chosen = coords[chosen]
        c = pts_chosen.sum(axis=0) / len(pts_chosen)   # == mean(axis=0), cheaper

        # (Perpendicular median re-centring removed.)  With the grid subtracted
        # as an exact object the tube is clean, so the forward-cone centroid
        # already sits on the centre-line; the residual wobble that the median
        # used to damp is smoothed by _clean_path (rolling median + Savitzky-
        # Golay).  Dropping it removes the tracer's per-step median plus two
        # dot products — the remaining hot spot after the earlier pass.

        move = c - p
        dist_moved = math.hypot(move[0], move[1])
        if dist_moved < 0.5:
            break

        newd = move / dist_moved
        d = 0.6 * d + 0.4 * newd
        d /= math.hypot(d[0], d[1]) + 1e-12
        p = c
        pts.append(p.copy())

        px, py = float(p[0]), float(p[1])
        if px < x0: x0 = px
        elif px > x1: x1 = px
        if py < y0: y0 = py
        elif py > y1: y1 = py
        n = len(pts)
        if (n & 63) == 0:
            extent = math.hypot(x1 - x0, y1 - y0)
            if n * step > 3.0 * extent + 4.0 * width:   # spiralling in a blob
                break

    return pts


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
