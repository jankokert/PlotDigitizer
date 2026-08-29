"""Read numeric axis labels from the plot image using OCR (pytesseract)."""

import logging
import math
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Calibration: (pixel_low, m_low, pixel_high, m_high, scale) where m is the data
# value on a "lin" axis or log10(value) on a "log" axis.  A legacy 4-tuple
# (no scale) is treated as linear.
Calibration = tuple


def read_axes(
    img_array: np.ndarray,
    plot_area: tuple[int, int, int, int],
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    debug: Optional[dict] = None,
) -> tuple[Calibration, Calibration]:
    """
    Return (x_calib, y_calib) as linear calibration tuples.

    Each calibration = (px_low, val_low, px_high, val_high) mapping
    a pixel coordinate to a data-space value.
    """
    x_min, y_min, x_max, y_max = plot_area

    if x_range is not None:
        x_calib: Calibration = (x_min, x_range[0], x_max, x_range[1], "lin")
        logger.info(f"X axis: manual range {x_range}")
    else:
        x_calib = _read_x_axis(img_array, plot_area, debug) or _default_x(plot_area)

    if y_range is not None:
        # pixel y increases downward; data y increases upward
        # y_range = (y_min_value, y_max_value) — min at bottom, max at top
        y_calib: Calibration = (y_min, y_range[1], y_max, y_range[0], "lin")
        logger.info(f"Y axis: manual range {y_range}")
    else:
        y_calib = _read_y_axis(img_array, plot_area, debug) or _default_y(plot_area)

    # Clear, unmissable summary of what the axes were calibrated to.
    x_lo = pixel_to_data(x_min, x_calib)
    x_hi = pixel_to_data(x_max, x_calib)
    y_bot = pixel_to_data(y_max, y_calib)   # bottom pixel row
    y_top = pixel_to_data(y_min, y_calib)   # top pixel row
    x_scale = x_calib[4] if len(x_calib) > 4 else "lin"
    y_scale = y_calib[4] if len(y_calib) > 4 else "lin"
    logger.info("--- Axis calibration ---")
    logger.info(f"  X ({x_scale}): data {x_lo:g} … {x_hi:g}")
    logger.info(f"  Y ({y_scale}): data {y_bot:g} (bottom) … {y_top:g} (top)")
    return x_calib, y_calib


def pixel_to_data(px: float, calib: Calibration) -> float:
    """
    Map a pixel coordinate to a data value.  Linear by default; if the
    calibration carries scale == "log", the stored endpoints are log10(value)
    and the result is exponentiated back.
    """
    p_low, m_low, p_high, m_high = calib[0], calib[1], calib[2], calib[3]
    scale = calib[4] if len(calib) > 4 else "lin"
    if p_high == p_low:
        m = m_low
    else:
        m = m_low + (px - p_low) * (m_high - m_low) / (p_high - p_low)
    return 10.0 ** m if scale == "log" else m


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tesseract():
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        logger.warning("pytesseract not installed — axis OCR unavailable")
        return None


def _read_x_axis(img_array: np.ndarray, plot_area, debug=None) -> Optional[Calibration]:
    pt = _tesseract()
    if pt is None:
        return None

    x_min, y_min, x_max, y_max = plot_area
    h, w = img_array.shape[:2]
    crop_top = y_max + 3
    crop_bot = min(h, y_max + 70)
    if crop_top >= crop_bot:
        return None

    # Widen the strip past the box so edge labels (e.g. 0 and 100) are not clipped.
    pad = 20
    cx0 = max(0, x_min - pad)
    crop = img_array[crop_top:crop_bot, cx0:min(w, x_max + pad)]
    readings = _ocr_numbers(crop, pt)  # (crop_col, value)

    if len(readings) < 2:
        return None

    # Convert crop_col → image_x
    calibration_pts = sorted((col + cx0, val) for col, val in readings)
    logger.info(
        f"X OCR: {len(readings)} labels → "
        + ", ".join(f"{val:g}@px{int(px)}" for px, val in calibration_pts)
    )
    calib = _fit_axis(calibration_pts, x_min, x_max, "X")
    if debug is not None:
        debug["x_ticks"] = _ticks_for_debug(calibration_pts, calib)
    return calib


def _read_y_axis(img_array: np.ndarray, plot_area, debug=None) -> Optional[Calibration]:
    pt = _tesseract()
    if pt is None:
        return None

    x_min, y_min, x_max, y_max = plot_area
    h = img_array.shape[0]
    crop_left = max(0, x_min - 90)
    crop_right = max(0, x_min - 3)
    if crop_left >= crop_right:
        return None

    # Widen the strip past the box so edge labels (e.g. 0 and 20) are not clipped.
    pad = 20
    cy0 = max(0, y_min - pad)
    crop = img_array[cy0:min(h, y_max + pad), crop_left:crop_right]
    readings = _ocr_numbers(crop, pt, horizontal=False)  # (crop_row, value)

    if len(readings) < 2:
        return None

    calibration_pts = sorted((row + cy0, val) for row, val in readings)
    logger.info(
        f"Y OCR: {len(readings)} labels → "
        + ", ".join(f"{val:g}@px{int(py)}" for py, val in calibration_pts)
    )
    calib = _fit_axis(calibration_pts, y_min, y_max, "Y")
    if debug is not None:
        debug["y_ticks"] = _ticks_for_debug(calibration_pts, calib)
    return calib


def _ticks_for_debug(calibration_pts, calib):
    """Return [(pixel, value, is_inlier)] — inlier = the label fits the fit line."""
    if calib is None:
        return [(int(p), v, False) for p, v in calibration_pts]
    scale = calib[4] if len(calib) > 4 else "lin"
    out = []
    for p, v in calibration_pts:
        model = pixel_to_data(p, calib)
        if scale == "log" and v > 0 and model > 0:
            ok = abs(math.log10(v) - math.log10(model)) <= 0.05
        else:
            span = abs(calib[3] - calib[1]) or 1.0
            ok = abs(v - model) <= max(0.03 * span, 0.5)
        out.append((int(round(p)), v, bool(ok)))
    return out


def _fit_axis(
    pts: list[tuple[float, float]], p_lo: float, p_hi: float, name: str
) -> Optional[Calibration]:
    """
    Fit an axis calibration from OCR ticks, auto-detecting linear vs log10.

    Ticks are collinear against the pixel position in *value* space on a linear
    axis and in *log10(value)* space on a log axis.  We robustly fit both and
    keep whichever the ticks support better (more inliers), then evaluate the
    fitted line at the plot-box edges.  Returns a 5-tuple calibration or None.
    """
    lin = _robust_line(pts)
    log = None
    if len(pts) >= 3 and all(v > 0 for _, v in pts):
        # 0.05 in log10 ≈ 12 % value error — tight enough that a linear axis
        # does *not* masquerade as log.
        log = _robust_line([(p, math.log10(v)) for p, v in pts], tol=0.05)

    n_lin = len(lin[2]) if lin else 0
    n_log = len(log[2]) if log else 0
    # A positive axis whose ticks span more than ~1.7 decades is log — and with
    # a huge value span the linear tolerance balloons so a log ladder can look
    # "linear", so when ≥3 ticks are collinear in log space we trust log.
    vals = [v for _, v in pts]
    wide_range = all(v > 0 for _, v in pts) and (max(vals) / min(vals) > 50)
    if log is not None and n_log >= 3 and (wide_range or n_log > n_lin):
        use_log = True
    else:
        use_log = False
    fit = log if use_log else lin
    if fit is None:
        return None
    slope, intercept, kept = fit
    scale = "log" if use_log else "lin"

    if len(kept) < len(pts):
        kept_px = {p for p, _ in kept}
        dropped = [f"{v:g}@px{int(p)}" for p, v in pts if p not in kept_px]
        logger.info(f"{name} fit: rejected outlier label(s) {', '.join(dropped)}")
    logger.info(f"{name} axis scale detected: {scale} ({len(kept)}/{len(pts)} ticks)")

    return (p_lo, slope * p_lo + intercept, p_hi, slope * p_hi + intercept, scale)


def _robust_line(
    points: list[tuple[float, float]],
    tol: Optional[float] = None,
) -> Optional[tuple[float, float, list[tuple[float, float]]]]:
    """
    Fit value = slope * pixel + intercept through OCR tick labels, iteratively
    discarding outliers (e.g. "100" misread as "1", or a stray "5").

    Axis ticks are collinear in (pixel, value) space, so a single misread label
    stands out as a large residual and is dropped.  Using *all* inlier labels
    (not just the two endpoints) makes calibration robust to both misreads and
    missing end labels — the fitted line extrapolates the true 0 / 100 edges.

    Returns (slope, intercept, kept_points) or None if < 2 usable points.
    """
    pts = list(dict.fromkeys(points))  # dedupe, preserve order
    if len(pts) < 2:
        return None
    if len(pts) == 2:
        (p1, v1), (p2, v2) = pts
        if p1 == p2:
            return None
        slope = (v2 - v1) / (p2 - p1)
        return slope, v1 - slope * p1, pts

    # RANSAC: a single misread label tilts a least-squares fit through *all*
    # points, so instead find the line supported by the largest collinear
    # subset. Axis ticks are exactly collinear, so the true line wins.
    vals = [v for _, v in pts]
    span = max(vals) - min(vals)
    if tol is None:
        tol = max(0.02 * span, 0.5)  # data-units a label may deviate and still count

    best_inliers: list[tuple[float, float]] = []
    best_err = float("inf")
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            (pa, va), (pb, vb) = pts[a], pts[b]
            if pa == pb:
                continue
            slope = (vb - va) / (pb - pa)
            intercept = va - slope * pa
            inliers = [(p, v) for p, v in pts if abs(v - (slope * p + intercept)) <= tol]
            err = sum((v - (slope * p + intercept)) ** 2 for p, v in inliers)
            if len(inliers) > len(best_inliers) or (
                len(inliers) == len(best_inliers) and err < best_err
            ):
                best_inliers, best_err = inliers, err

    if len(best_inliers) < 2:
        return None

    px = np.array([p for p, _ in best_inliers], dtype=float)
    val = np.array([v for _, v in best_inliers], dtype=float)
    slope, intercept = np.polyfit(px, val, 1)
    return float(slope), float(intercept), best_inliers


def _ocr_numbers(
    crop: np.ndarray,
    pt,
    horizontal: bool = True,
) -> list[tuple[float, float]]:
    """
    Run OCR on `crop` and return list of (pixel_pos, numeric_value).
    pixel_pos is the column-centre (horizontal=True) or row-centre (False).
    """
    from PIL import Image as PILImage

    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return []

    import re

    # Binarise (dark ink → black on white) so small axis digits read reliably,
    # then upscale.  Run two page-segmentation modes and merge: psm 6
    # (uniform block) catches edge labels, psm 11 (sparse) separates crowded
    # ones — RANSAC in the fit discards whatever disagrees.
    gray = np.asarray(PILImage.fromarray(crop).convert("L"))
    binary = np.where(gray < 128, 0, 255).astype(np.uint8)
    scale = 4
    pil_img = PILImage.fromarray(binary).resize(
        (binary.shape[1] * scale, binary.shape[0] * scale), PILImage.LANCZOS
    )

    num_re = re.compile(r"\d*\.?\d+")
    results: list[tuple[float, float]] = []
    for psm in (6, 11):
        try:
            config = f"--psm {psm} -c tessedit_char_whitelist=0123456789.-eE+"
            data = pt.image_to_data(pil_img, output_type=pt.Output.DICT, config=config)
        except Exception as exc:
            logger.error(f"OCR error: {exc}")
            continue
        for i, text in enumerate(data["text"]):
            m = num_re.search(text.strip())
            if not m:
                continue
            try:
                value = float(m.group())
            except ValueError:
                continue
            col_c = (data["left"][i] + data["width"][i] / 2) / scale
            row_c = (data["top"][i] + data["height"][i] / 2) / scale
            pos = col_c if horizontal else row_c
            results.append((pos, value))

    # Deduplicate near-identical (pos, value) from the two passes.
    deduped: list[tuple[float, float]] = []
    for pos, value in results:
        if not any(abs(pos - p) < 6 and abs(value - v) < 1e-6 for p, v in deduped):
            deduped.append((pos, value))
    return deduped


def _default_x(plot_area) -> Calibration:
    x_min, _, x_max, _ = plot_area
    logger.warning("X axis: OCR failed — defaulting to [0, 1]")
    return (x_min, 0.0, x_max, 1.0, "lin")


def _default_y(plot_area) -> Calibration:
    _, y_min, _, y_max = plot_area
    logger.warning("Y axis: OCR failed — defaulting to [0, 1] (inverted pixel)")
    return (y_min, 1.0, y_max, 0.0, "lin")
