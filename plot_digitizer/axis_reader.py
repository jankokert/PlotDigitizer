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
    text_labels: Optional[list[dict]] = None,
) -> tuple[Calibration, Calibration]:
    """
    Return (x_calib, y_calib) as linear calibration tuples.

    Each calibration = (px_low, val_low, px_high, val_high) mapping
    a pixel coordinate to a data-space value.

    text_labels: known text lines ``{text, box}`` in image pixels (PDF text
    layer).  Their numeric tick labels are used instead of OCR when an axis
    has at least two of them.
    """
    x_min, y_min, x_max, y_max = plot_area
    tick_x, tick_y = _ticks_from_text(text_labels, plot_area) if text_labels else ([], [])

    if x_range is not None:
        x_calib: Calibration = (x_min, x_range[0], x_max, x_range[1], "lin")
        logger.info(f"X axis: manual range {x_range}")
    elif len(tick_x) >= 2:
        tick_x = snap_ticks(tick_x, img_array, plot_area, "x")
        x_calib = _calib_from_ticks(tick_x, x_min, x_max, "X", debug, "x_ticks") or _default_x(plot_area)
    else:
        x_calib = _read_x_axis(img_array, plot_area, debug) or _default_x(plot_area)

    if y_range is not None:
        # pixel y increases downward; data y increases upward
        # y_range = (y_min_value, y_max_value) — min at bottom, max at top
        y_calib: Calibration = (y_min, y_range[1], y_max, y_range[0], "lin")
        logger.info(f"Y axis: manual range {y_range}")
    elif len(tick_y) >= 2:
        tick_y = snap_ticks(tick_y, img_array, plot_area, "y")
        y_calib = _calib_from_ticks(tick_y, y_min, y_max, "Y", debug, "y_ticks") or _default_y(plot_area)
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
    # Strip sizes relative to the plot (tuned at ~150 dpi: 70 px / 20 px for a
    # ~450 px tall canvas), so high-resolution PDF renders work the same.
    strip, pad = _strip_sizes(plot_area)
    crop_top = y_max + max(3, int(0.005 * (y_max - y_min)))
    crop_bot = min(h, y_max + strip)
    if crop_top >= crop_bot:
        return None

    # Widen the strip past the box so edge labels (e.g. 0 and 100) are not clipped.
    cx0 = max(0, x_min - pad)
    crop = img_array[crop_top:crop_bot, cx0:min(w, x_max + pad)]
    readings = _ocr_numbers(crop, pt, scale=_ocr_scale(strip))  # (crop_col, value)

    if len(readings) < 2:
        return None

    # Convert crop_col → image_x
    calibration_pts = snap_ticks(sorted((col + cx0, val) for col, val in readings),
                                 img_array, plot_area, "x")
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
    strip, pad = _strip_sizes(plot_area)
    ystrip = int(round(strip * 90 / 70))         # y labels are wider ("10,000")
    crop_left = max(0, x_min - ystrip)
    crop_right = max(0, x_min - max(3, int(0.005 * (x_max - x_min))))
    if crop_left >= crop_right:
        return None

    # Widen the strip past the box so edge labels (e.g. 0 and 20) are not clipped.
    cy0 = max(0, y_min - pad)
    crop = img_array[cy0:min(h, y_max + pad), crop_left:crop_right]
    readings = _ocr_numbers(crop, pt, horizontal=False, scale=_ocr_scale(strip))

    if len(readings) < 2:
        return None

    calibration_pts = snap_ticks(sorted((row + cy0, val) for row, val in readings),
                                 img_array, plot_area, "y")
    logger.info(
        f"Y OCR: {len(readings)} labels → "
        + ", ".join(f"{val:g}@px{int(py)}" for py, val in calibration_pts)
    )
    calib = _fit_axis(calibration_pts, y_min, y_max, "Y")
    if debug is not None:
        debug["y_ticks"] = _ticks_for_debug(calibration_pts, calib)
    return calib


def _parse_number(text: str) -> Optional[float]:
    """A text line that is exactly one number (PDF tick label)."""
    import re
    t = text.strip().replace("−", "-").replace(" ", "")
    if not re.fullmatch(r"[-+]?[\d.,]+", t):
        return None
    return parse_tick_number(t)


def _ticks_from_text(text_labels: list[dict], plot_area):
    """Split numeric text lines into x ticks (below the canvas) and y ticks
    (left of it): lists of (pixel position of the label centre, value)."""
    x_min, y_min, x_max, y_max = plot_area
    w, h = x_max - x_min, y_max - y_min
    xs, ys = [], []
    for lab in text_labels:
        v = _parse_number(lab["text"])
        if v is None or not lab.get("horizontal", True):
            continue
        x0, y0, x1, y1 = lab["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if y_max - 0.02 * h <= y0 and cy <= y_max + 0.15 * h                 and x_min - 0.05 * w <= cx <= x_max + 0.05 * w:
            xs.append((cx, v))
        elif x1 <= x_min + 0.02 * w and cx >= x_min - 0.25 * w                 and y_min - 0.05 * h <= cy <= y_max + 0.05 * h:
            ys.append((cy, v))
    return sorted(xs), sorted(ys)


def _axis_lines(img_array: np.ndarray, plot_area, axis: str) -> list[tuple[float, int]]:
    """Grid lines and tick marks crossing the axis: [(centre px, thickness px)].

    Two narrow bands are scanned, clear of the frame line itself:
    * inside the frame - a line there counts only if it runs straight through
      the whole plot (a grid line), so curves crossing the band are ignored;
    * outside the frame - tick marks; curves never reach out there.
    """
    x_min, y_min, x_max, y_max = plot_area
    gray = img_array.mean(axis=2) if img_array.ndim == 3 else img_array
    dark = gray < 128
    if axis == "x":                       # work on the transposed image: rows = x
        dark = dark.T
        x_min, y_min, x_max, y_max = y_min, x_min, y_max, x_max
    h, w = dark.shape
    span = x_max - x_min
    off = max(4, int(0.01 * span))
    bw = max(5, int(0.012 * span))
    if axis == "y":
        inside = dark[y_min:y_max, x_min + off:min(w, x_min + off + bw)]
        outside = dark[y_min:y_max, max(0, x_min - off - bw):max(0, x_min - off)]
        full = dark[y_min:y_max, x_min + off:max(x_min + off + 1, x_max - off)]
    else:                                  # x axis: bands just above/below y_max (= x_max here)
        inside = dark[y_min:y_max, max(0, x_max - off - bw):max(0, x_max - off)]
        outside = dark[y_min:y_max, x_max + off:min(w, x_max + off + bw)]
        full = dark[y_min:y_max, x_min + off:max(x_min + off + 1, x_max - off)]

    def groups(band):
        if band.size == 0:
            return []
        rows = np.flatnonzero(band.mean(axis=1) >= 0.6)
        if len(rows) == 0:
            return []
        return np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)

    lines = []
    for g in groups(inside):
        if full[g].any(axis=0).mean() >= 0.5:          # straight through the plot
            lines.append((y_min + float(g.mean()), len(g)))
    for g in groups(outside):
        c = y_min + float(g.mean())
        if not any(abs(c - l[0]) <= 1.5 for l in lines):
            lines.append((c, len(g)))
    return sorted(lines)


def snap_ticks(pts: list[tuple[float, float]], img_array: np.ndarray, plot_area,
               axis: str) -> list[tuple[float, float]]:
    """Move each tick label onto the line it labels.

    A text box centre is not the tick position: the box is asymmetric ("1,000"
    has a descending comma) and OCR passes disagree by several pixels.  The
    labelled line is the THICKEST line near the label (a major line beats the
    minor lines around it; among equals the nearest wins), searched within a
    third of the label spacing.  Duplicate readings then coincide and merge.
    """
    if len(pts) < 2:
        return pts
    lines = _axis_lines(img_array, plot_area, axis)
    if not lines:
        return pts
    pos = sorted({round(p) for p, _ in pts})
    gaps = [b - a for a, b in zip(pos[:-1], pos[1:]) if b - a > 2]
    if not gaps:
        return pts
    tol = 0.35 * float(np.median(gaps))
    out = []
    for p, v in pts:
        near = [(c, t) for c, t in lines if abs(c - p) <= tol]
        if near:
            tmax = max(t for _, t in near)
            c = min((c for c, t in near if t >= 0.8 * tmax), key=lambda c: abs(c - p))
            p = c
        out.append((p, v))
    return sorted(dict.fromkeys(out))


def _calib_from_ticks(pts, p_lo, p_hi, name, debug, key) -> Optional[Calibration]:
    logger.info(f"{name} ticks (PDF text): "
                + ", ".join(f"{v:g}@px{int(p)}" for p, v in pts))
    calib = _fit_axis(pts, p_lo, p_hi, name)
    if debug is not None:
        debug[key] = _ticks_for_debug(pts, calib)
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
    # Non-positive readings can't lie on a log axis: on a log hypothesis they
    # are outliers (an OCR "-4" from "10^-4"), not a veto against log.
    pos = [(p, v) for p, v in pts if v > 0]
    if len(pos) >= 3:
        # 0.05 in log10 ≈ 12 % value error — tight enough that a linear axis
        # does *not* masquerade as log.
        log = _robust_line([(p, math.log10(v)) for p, v in pos], tol=0.05)

    n_lin = len(lin[2]) if lin else 0
    n_log = len(log[2]) if log else 0
    # A positive axis whose ticks span more than ~1.7 decades is log — and with
    # a huge value span the linear tolerance balloons so a log ladder can look
    # "linear", so when ≥3 ticks are collinear in log space we trust log.
    # Judge the span on the ticks that actually lie on the log line: OCR
    # garbage ("1.0012e+21") must not make a linear axis look wide.
    vals = [10.0 ** lv for _, lv in log[2]] if log is not None else []
    wide_range = len(vals) >= 3 and max(vals) / min(vals) > 50
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
    # Inlier test in PIXELS: a tick counts if its value puts it where it is
    # on the axis (within 1.5 % of the tick extent).  A data-unit tolerance
    # derived from the value span is inflated by a single misread ("210" for
    # "10") and then admits garbage readings.  `tol` (data units) is an
    # additional upper bound when given.
    pix = [p for p, _ in pts]
    tol_px = max(3.0, 0.015 * (max(pix) - min(pix)))

    best_inliers: list[tuple[float, float]] = []
    best_err = float("inf")
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            (pa, va), (pb, vb) = pts[a], pts[b]
            if pa == pb or va == vb:
                continue
            slope = (vb - va) / (pb - pa)
            intercept = va - slope * pa
            inliers = [(p, v) for p, v in pts
                       if abs(p - (v - intercept) / slope) <= tol_px
                       and (tol is None or abs(v - (slope * p + intercept)) <= tol)]
            err = sum((p - (v - intercept) / slope) ** 2 for p, v in inliers)
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


def _strip_sizes(plot_area) -> tuple[int, int]:
    """(label strip depth, edge padding) in px, scaled with the plot size.

    The original constants (70 px strip, 20 px padding) were tuned on ~150 dpi
    screenshots whose canvases are ~450 px tall; never smaller than those."""
    x_min, y_min, x_max, y_max = plot_area
    ref = max(1.0, min(x_max - x_min, y_max - y_min) / 450.0)
    return int(round(70 * ref)), int(round(20 * ref))


def _ocr_scale(strip: int) -> int:
    """Upscale factor so the digits reach the size Tesseract was tuned on
    (4x for a 70 px strip; none for a high-resolution render)."""
    return int(np.clip(round(4 * 70 / max(strip, 1)), 1, 4))


def parse_tick_number(text: str) -> Optional[float]:
    """'10,000' -> 10000, '0,5' -> 0.5, '-40' -> -40, '1.2' -> 1.2."""
    import re
    t = text.strip().replace("−", "-").replace(" ", "")
    m = re.search(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*[.,]?\d+", t)
    if not m:
        return None
    g = m.group()
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", g) and not g.lstrip("+-").startswith("0,"):
        g = g.replace(",", "")                       # thousands separators
    else:
        g = g.replace(",", ".")                      # decimal comma
    try:
        v = float(g)
    except ValueError:
        return None
    return v if abs(v) < 1e12 else None          # OCR garbage, not a tick


def _ocr_numbers(
    crop: np.ndarray,
    pt,
    horizontal: bool = True,
    scale: int = 4,
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
    pil_img = PILImage.fromarray(binary).resize(
        (binary.shape[1] * scale, binary.shape[0] * scale), PILImage.LANCZOS
    )

    results: list[tuple[float, float]] = []
    for psm in (6, 11):
        try:
            config = f"--psm {psm} -c tessedit_char_whitelist=0123456789.,-eE+"
            data = pt.image_to_data(pil_img, output_type=pt.Output.DICT, config=config)
        except Exception as exc:
            logger.error(f"OCR error: {exc}")
            continue
        for i, text in enumerate(data["text"]):
            value = parse_tick_number(text)
            if value is None:
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
