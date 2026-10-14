"""Main digitisation pipeline."""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .axis_reader import pixel_to_data, read_axes
from .curve_extractor import extract_curves
from .label_parser import assign_labels
from .plot_detector import detect_plot_area

logger = logging.getLogger(__name__)


def _arc_length(pts) -> float:
    p = np.asarray(pts, dtype=float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _drop_fragments(curves: dict, keep: set, rel: float = 0.2) -> dict:
    """Drop curves shorter than `rel` × the median curve length (with >= 3
    curves; labelled curves are always kept)."""
    if len(curves) < 3:
        return curves
    lengths = {k: _arc_length(v) for k, v in curves.items()}
    med = float(np.median(list(lengths.values())))
    drop = [k for k, L in lengths.items() if L < rel * med and k not in keep]
    if drop:
        logger.info(f"Dropped short fragments (< {rel:.0%} of median length): {', '.join(drop)}")
    return {k: v for k, v in curves.items() if k not in drop}


def _drop_unlabelled_sketches(curves: dict, labelled: set, plot_area,
                              rel: float = 0.3) -> dict:
    """In a labelled curve family, an unlabelled path that stays small in both
    directions is an inset sketch (e.g. the duty-cycle pulse "D = tp/T"),
    not a data curve."""
    x0, y0, x1, y1 = plot_area
    drop = []
    for k, v in curves.items():
        if k in labelled:
            continue
        span = np.ptp(np.asarray(v, dtype=float), axis=0)
        if span[0] < rel * (x1 - x0) and span[1] < rel * (y1 - y0):
            drop.append(k)
    if drop:
        logger.info(f"Dropped unlabelled sketch(es) in a labelled family: {', '.join(drop)}")
    return {k: v for k, v in curves.items() if k not in drop}


def _rename(curves: dict, names: dict) -> dict:
    """Apply label names; a value used twice (two "T_A = 25 °C" labels) gets
    a suffix instead of silently overwriting the other curve."""
    out: dict = {}
    for k, v in curves.items():
        new = names.get(k, k)
        base, i = new, 2
        while new in out:
            new = f"{base}_{i}"
            i += 1
        out[new] = v
    return out


def digitize_plot(
    image_path: Path,
    output_path: Path,
    method: str = "naive",
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    smoothing: float = 1.0,
    n_curves: Optional[int] = None,
    has_grid: bool = False,
    n_samples: int = 500,
    sat_threshold: float = 0.20,
    min_col_coverage: float = 0.20,
    hue_bins: int = 18,
    span_frac: float = 0.55,
    max_thickness: Optional[int] = None,
    notch_factor: float = 3.0,
    target_colors: Optional[list[tuple[int, int, int]]] = None,
    color_tolerance: float = 40.0,
    debug_svg: Optional[Path] = None,
    mode: str = "auto",
) -> dict:
    """
    Full pipeline: load image → detect plot area → read axes → extract curves
    → convert to data coords → write CSV.

    mode: "vector" takes the curves exactly from the PDF vector paths stored
    in the image's sidecar JSON (written by --pdf); "raster" traces pixels;
    "auto" uses vector when the sidecar has vector curves, raster otherwise.

    Returns a summary (mode, plot area, calibrations, curve header, curves in
    image pixels, residual ink, run time) for regression checks.
    """
    if mode not in ("auto", "vector", "raster"):
        raise ValueError(f"mode must be auto, vector or raster, not {mode!r}")
    t0 = time.perf_counter()

    # 1. Load image
    with Image.open(image_path) as img:
        img_array = np.array(img.convert("RGB"))
    h, w = img_array.shape[:2]
    logger.info(f"Loaded image: {image_path.name}  size={w}x{h}")

    # 2. Detect plot canvas
    plot_area = detect_plot_area(img_array)
    x_min, y_min, x_max, y_max = plot_area
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    logger.info(f"Canvas: {canvas_w}x{canvas_h} px  ({x_min},{y_min})→({x_max},{y_max})")

    # Text lines from a PDF sidecar (<image>.json written by --pdf) replace OCR
    # of the in-plot labels: exact text and position, subscripts included.
    meta: dict = {}
    sidecar = image_path.with_suffix(".json")
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning(f"Ignoring unreadable sidecar {sidecar.name}")
        if not isinstance(meta, dict):
            meta = {}
    text_labels = meta.get("text")
    if text_labels is not None:
        logger.info(f"Using {len(text_labels)} text lines from {sidecar.name}")

    # 3. Read axis calibration
    debug: Optional[dict] = {} if debug_svg else None
    x_calib, y_calib = read_axes(
        img_array, plot_area, x_range=x_range, y_range=y_range, debug=debug,
        text_labels=text_labels,
    )

    # 4. Extract curves: exactly from the PDF vectors, or by tracing pixels
    annot: dict = {}
    vector_shafts: dict = {}
    curves: dict = {}
    if mode != "raster" and meta.get("vectors"):
        from .curve_extractor import labels_in_canvas
        from .vector_curves import vector_curves
        curves, vector_shafts = vector_curves(meta["vectors"], plot_area)
        if not curves:
            logger.info("Sidecar has vectors but no curve paths — falling back to raster")
    elif mode == "vector":
        logger.warning("Vector mode requested, but the image has no sidecar vectors "
                       "— falling back to raster")
    used_mode = "vector" if curves else "raster"
    if used_mode == "vector":
        annot = {"labels": labels_in_canvas(text_labels or [], plot_area), "label_arrows": []}
        if debug is not None:
            debug["text_boxes"] = [lab["box"] for lab in annot["labels"]]
            debug["label_texts"] = [lab["text"] for lab in annot["labels"]]
    else:
        curves = extract_curves(
            img_array, plot_area,
            method=method, smoothing=smoothing,
            n_curves=n_curves, has_grid=has_grid,
            n_samples=n_samples, sat_threshold=sat_threshold,
            min_col_coverage=min_col_coverage, hue_bins=hue_bins,
            span_frac=span_frac, max_thickness=max_thickness,
            notch_factor=notch_factor,
            target_colors=target_colors, color_tolerance=color_tolerance,
            debug=debug,
            text_labels=text_labels,
            annot=annot,
        )

    # 4a. Name the curves by their labels ("T_A = 25 °C" -> column "T_A in °C", value "25")
    names, curve_header, shafts = assign_labels(
        curves, annot.get("labels") or [], annot.get("label_arrows") or [],
        plot_area, extra_fragments={**annot.get("fragments", {}), **vector_shafts})
    if shafts:
        logger.info(f"Arrow shafts used for labels: {', '.join(shafts)}")
        curves = {k: v for k, v in curves.items() if k not in shafts}
    # Short leftovers (arrow pieces, glyph strokes) are annotation, not data.
    curves = _drop_fragments(curves, keep=set(names))
    if names:
        curves = _drop_unlabelled_sketches(curves, set(names), plot_area)
    if names:
        logger.info(f"Curve labels ({curve_header}): "
                    + ", ".join(f"{k}->{v}" for k, v in names.items()))
        curves = _rename(curves, names)
    n_found = len(curves)
    logger.info(f"Curves found: {n_found}  "
                f"({'vector' if used_mode == 'vector' else f'raster, method={method}'})")

    if n_found == 0:
        logger.warning("No curves detected — check image or try --debug-files")

    # 4b. Optional debug SVG overlay + pixel-exact grid PNG
    if debug_svg:
        from .debug_svg import write_debug_svg, write_grid_debug_png
        write_debug_svg(
            image_path=image_path,
            out_path=debug_svg,
            plot_area=plot_area,
            curves=curves,
            grid_mask=(debug or {}).get("grid_mask"),
            arrows=(debug or {}).get("arrows"),
            text_boxes=(debug or {}).get("text_boxes"),
            legend_boxes=(debug or {}).get("legend_boxes"),
            label_texts=(debug or {}).get("label_texts"),
            label_arrows=(debug or {}).get("label_arrows"),
            x_ticks=(debug or {}).get("x_ticks"),
            y_ticks=(debug or {}).get("y_ticks"),
        )
        write_grid_debug_png(
            image_path=image_path,
            out_path=debug_svg.with_name(debug_svg.stem + "_grid.png"),
            plot_area=plot_area,
            grid_gray=(debug or {}).get("grid_gray"),
            ink=(debug or {}).get("ink"),
            arrow_mask=(debug or {}).get("arrow_mask"),
            arrow_gray=(debug or {}).get("arrow_gray"),
        )
        if (debug or {}).get("residual_mask") is not None:
            from .debug_svg import write_residual_png
            write_residual_png(
                image_path=image_path,
                out_path=debug_svg.with_name(debug_svg.stem + "_residual.png"),
                plot_area=plot_area,
                residual=debug["residual_mask"],
                curves=curves,
            )
        grid_model = (debug or {}).get("grid_model")
        if grid_model is not None:
            json_path = debug_svg.with_name(debug_svg.stem + "_grid.json")
            json_path.write_text(json.dumps(grid_model, indent=2), encoding="utf-8")
            logger.info(f"Grid model JSON written to: {json_path}")
        arrow_model = (debug or {}).get("arrow_model")
        if arrow_model is not None:
            widths = (debug or {}).get("widths")
            if widths is not None:
                arrow_model = {**arrow_model, "widths_px": widths}
            apath = debug_svg.with_name(debug_svg.stem + "_arrows.json")
            apath.write_text(json.dumps(arrow_model, indent=2), encoding="utf-8")
            logger.info(f"Arrow model JSON written to: {apath}")

    # 5. Convert pixel → data coordinates and collect rows
    rows: list[tuple[str, float, float]] = []
    for curve_name, pts in curves.items():
        for px, py in pts:
            dx = pixel_to_data(px, x_calib)
            dy = pixel_to_data(py, y_calib)
            rows.append((curve_name, dx, dy))

    # Sort for cleaner output (numeric label values numerically)
    def _name_key(name: str):
        try:
            return (0, float(name), "")
        except ValueError:
            return (1, 0.0, name)
    rows.sort(key=lambda r: (_name_key(r[0]), r[1]))

    # 6. Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([curve_header, "x", "y"])
        for curve_name, dx, dy in rows:
            writer.writerow([curve_name, f"{dx:.6f}", f"{dy:.6f}"])

    elapsed = time.perf_counter() - t0

    # Metrics log
    logger.info("--- Metrics ---")
    logger.info(f"  Total data points : {len(rows)}")
    logger.info(f"  Curves            : {n_found}")
    for cname, pts in curves.items():
        logger.info(f"    {cname}: {len(pts)} pts")
    logger.info(f"  Processing time   : {elapsed:.2f} s")
    logger.info(f"  Output            : {output_path}")

    return {
        "mode": used_mode,
        "plot_area": [int(v) for v in plot_area],
        "x_calib": list(x_calib),
        "y_calib": list(y_calib),
        "curve_header": curve_header,
        "curves_px": {k: np.asarray(v, dtype=float).round(2).tolist() for k, v in curves.items()},
        "residual_fraction": round(float(annot.get("residual_fraction", 0.0)), 5),
        "time_s": round(elapsed, 2),
    }
