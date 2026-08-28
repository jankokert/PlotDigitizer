"""Read numeric axis labels from the plot image using OCR (pytesseract)."""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Calibration: (pixel_low, value_low, pixel_high, value_high)
Calibration = tuple[float, float, float, float]


def read_axes(
    img_array: np.ndarray,
    plot_area: tuple[int, int, int, int],
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
) -> tuple[Calibration, Calibration]:
    """
    Return (x_calib, y_calib) as linear calibration tuples.

    Each calibration = (px_low, val_low, px_high, val_high) mapping
    a pixel coordinate to a data-space value.
    """
    x_min, y_min, x_max, y_max = plot_area

    if x_range is not None:
        x_calib: Calibration = (x_min, x_range[0], x_max, x_range[1])
        logger.info(f"X axis: manual range {x_range}")
    else:
        x_calib = _read_x_axis(img_array, plot_area) or _default_x(plot_area)

    if y_range is not None:
        # pixel y increases downward; data y increases upward
        # y_range = (y_min_value, y_max_value) — min at bottom, max at top
        y_calib: Calibration = (y_min, y_range[1], y_max, y_range[0])
        logger.info(f"Y axis: manual range {y_range}")
    else:
        y_calib = _read_y_axis(img_array, plot_area) or _default_y(plot_area)

    # Clear, unmissable summary of what the axes were calibrated to.
    x_lo = pixel_to_data(x_min, x_calib)
    x_hi = pixel_to_data(x_max, x_calib)
    y_bot = pixel_to_data(y_max, y_calib)   # bottom pixel row
    y_top = pixel_to_data(y_min, y_calib)   # top pixel row
    logger.info("--- Axis calibration (LINEAR only; log/semi-log not yet handled) ---")
    logger.info(f"  X: data {x_lo:g} … {x_hi:g}   (calib {x_calib})")
    logger.info(f"  Y: data {y_bot:g} (bottom) … {y_top:g} (top)   (calib {y_calib})")
    return x_calib, y_calib


def pixel_to_data(px: float, calib: Calibration) -> float:
    """Linearly map a pixel coordinate to a data value."""
    p_low, v_low, p_high, v_high = calib
    if p_high == p_low:
        return v_low
    return v_low + (px - p_low) * (v_high - v_low) / (p_high - p_low)


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


def _read_x_axis(img_array: np.ndarray, plot_area) -> Optional[Calibration]:
    pt = _tesseract()
    if pt is None:
        return None

    x_min, y_min, x_max, y_max = plot_area
    h = img_array.shape[0]
    crop_top = y_max + 3
    crop_bot = min(h, y_max + 70)
    if crop_top >= crop_bot:
        return None

    crop = img_array[crop_top:crop_bot, x_min:x_max]
    readings = _ocr_numbers(crop, pt)  # (crop_col, value)

    if len(readings) < 2:
        return None

    # Convert crop_col → image_x
    calibration_pts = sorted((col + x_min, val) for col, val in readings)
    logger.info(
        f"X OCR: {len(readings)} labels → "
        + ", ".join(f"{val:g}@px{int(px)}" for px, val in calibration_pts)
    )
    fit = _robust_line(calibration_pts)
    if fit is None:
        return None
    slope, intercept, kept = fit
    if len(kept) < len(calibration_pts):
        dropped = [f"{v:g}@px{int(p)}" for p, v in calibration_pts if (p, v) not in kept]
        logger.info(f"X fit: rejected outlier label(s) {', '.join(dropped)}")
    # Evaluate the fitted line at the plot-box edges for a full-span calibration.
    return (x_min, slope * x_min + intercept, x_max, slope * x_max + intercept)


def _read_y_axis(img_array: np.ndarray, plot_area) -> Optional[Calibration]:
    pt = _tesseract()
    if pt is None:
        return None

    x_min, y_min, x_max, y_max = plot_area
    crop_left = max(0, x_min - 90)
    crop_right = max(0, x_min - 3)
    if crop_left >= crop_right:
        return None

    crop = img_array[y_min:y_max, crop_left:crop_right]
    readings = _ocr_numbers(crop, pt, horizontal=False)  # (crop_row, value)

    if len(readings) < 2:
        return None

    calibration_pts = sorted((row + y_min, val) for row, val in readings)
    logger.info(
        f"Y OCR: {len(readings)} labels → "
        + ", ".join(f"{val:g}@px{int(py)}" for py, val in calibration_pts)
    )
    fit = _robust_line(calibration_pts)
    if fit is None:
        return None
    slope, intercept, kept = fit
    if len(kept) < len(calibration_pts):
        dropped = [f"{v:g}@px{int(p)}" for p, v in calibration_pts if (p, v) not in kept]
        logger.info(f"Y fit: rejected outlier label(s) {', '.join(dropped)}")
    return (y_min, slope * y_min + intercept, y_max, slope * y_max + intercept)


def _robust_line(
    points: list[tuple[float, float]],
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

    pil_img = PILImage.fromarray(crop)
    scale = 3
    pil_img = pil_img.resize(
        (pil_img.width * scale, pil_img.height * scale), PILImage.LANCZOS
    )

    try:
        config = "--psm 11 -c tessedit_char_whitelist=0123456789.-eE+"
        data = pt.image_to_data(pil_img, output_type=pt.Output.DICT, config=config)
    except Exception as exc:
        logger.error(f"OCR error: {exc}")
        return []

    results: list[tuple[float, float]] = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        col_c = (data["left"][i] + data["width"][i] / 2) / scale
        row_c = (data["top"][i] + data["height"][i] / 2) / scale
        pos = col_c if horizontal else row_c
        results.append((pos, value))

    return results


def _default_x(plot_area) -> Calibration:
    x_min, _, x_max, _ = plot_area
    logger.warning("X axis: OCR failed — defaulting to [0, 1]")
    return (x_min, 0.0, x_max, 1.0)


def _default_y(plot_area) -> Calibration:
    _, y_min, _, y_max = plot_area
    logger.warning("Y axis: OCR failed — defaulting to [0, 1] (inverted pixel)")
    return (y_min, 1.0, y_max, 0.0)
