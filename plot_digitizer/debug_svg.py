"""Render a debug SVG overlaying detection results on the source image.

The SVG embeds the original plot image as a base64 background and draws:
  * the detected plot area (green rectangle),
  * the suppressed grid pixels (red translucent tint), if a grid mask is given,
  * the extracted curve points (one colour per curve).

It is meant purely as a development aid so you can see, at a glance, whether
grid lines are leaking into the data or the plot box is misaligned.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

import numpy as np

logger = logging.getLogger(__name__)

# Distinct fallback colours for curves whose label carries no RGB hint.
_FALLBACK_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
]


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _image_data_uri(image_path: Path) -> tuple[str, int, int]:
    """Return (data-URI, width, height) for the source image."""
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    raw = image_path.read_bytes()
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix or "png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{mime};base64,{b64}", w, h


def _grid_overlay_uri(
    grid_mask: np.ndarray,
    plot_area: tuple[int, int, int, int],
    img_size: tuple[int, int],
) -> Optional[str]:
    """Encode the grid mask as a full-image translucent red PNG data URI."""
    from PIL import Image

    x_min, y_min, x_max, y_max = plot_area
    w, h = img_size
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    ys, xs = np.where(grid_mask)
    if len(xs) == 0:
        return None
    overlay[ys + y_min, xs + x_min] = (255, 0, 0, 110)
    buf = io.BytesIO()
    Image.fromarray(overlay, mode="RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _curve_color(label: str, index: int) -> str:
    """Pick a display colour: use the RGB embedded in the label if present."""
    marker = "rgb"
    pos = label.find(marker)
    if pos != -1:
        digits = label[pos + len(marker):pos + len(marker) + 9]
        if len(digits) == 9 and digits.isdigit():
            r, g, b = int(digits[0:3]), int(digits[3:6]), int(digits[6:9])
            # Avoid near-white / near-black which vanish against the plot.
            if 30 <= (r + g + b) / 3 <= 225:
                return f"#{r:02x}{g:02x}{b:02x}"
    return _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)]


def write_debug_svg(
    image_path: Path,
    out_path: Path,
    plot_area: tuple[int, int, int, int],
    curves: dict[str, np.ndarray],
    grid_mask: Optional[np.ndarray] = None,
    arrows: Optional[list[dict]] = None,
    text_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    legend_boxes: Optional[list[tuple[int, int, int, int]]] = None,
    label_texts: Optional[list[str]] = None,
    label_arrows: Optional[list[dict]] = None,
    x_ticks: Optional[list[tuple[int, float, bool]]] = None,
    y_ticks: Optional[list[tuple[int, float, bool]]] = None,
) -> None:
    """Write the debug SVG to out_path.

    Args:
        image_path: source plot image.
        out_path:   destination .svg path.
        plot_area:  (x_min, y_min, x_max, y_max) in image pixel coords.
        curves:     {label: Nx2 array of (px, py)} in image pixel coords.
        grid_mask:  optional canvas-relative bool mask of suppressed pixels.
        arrows:     optional list of detected annotation-arrow dicts
                    (canvas-relative row / x_from / x_to / tip_x / tip_dir).
    """
    data_uri, w, h = _image_data_uri(image_path)
    x_min, y_min, x_max, y_max = plot_area

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="sans-serif">',
        f'<image href="{data_uri}" x="0" y="0" width="{w}" height="{h}"/>',
    ]

    if grid_mask is not None:
        overlay_uri = _grid_overlay_uri(grid_mask, plot_area, (w, h))
        if overlay_uri:
            parts.append(
                f'<image href="{overlay_uri}" x="0" y="0" width="{w}" height="{h}"/>'
            )

    # Plot-area rectangle
    parts.append(
        f'<rect x="{x_min}" y="{y_min}" width="{x_max - x_min}" '
        f'height="{y_max - y_min}" fill="none" stroke="#00c000" '
        f'stroke-width="2"/>'
    )

    # OCR'd axis ticks: inliers (used in the fit) green, rejected outliers red.
    for (px, val, ok) in (x_ticks or []):
        col = "#00a000" if ok else "#d02020"
        parts.append(
            f'<line x1="{px}" y1="{y_max}" x2="{px}" y2="{y_max + 9}" '
            f'stroke="{col}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{px}" y="{y_max + 22}" font-size="11" fill="{col}" '
            f'text-anchor="middle">{val:g}</text>'
        )
    for (py, val, ok) in (y_ticks or []):
        col = "#00a000" if ok else "#d02020"
        parts.append(
            f'<line x1="{x_min - 9}" y1="{py}" x2="{x_min}" y2="{py}" '
            f'stroke="{col}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x_min - 12}" y="{py + 4}" font-size="11" fill="{col}" '
            f'text-anchor="end">{val:g}</text>'
        )

    # Detected annotation arrows (shaft line + arrowhead), in cyan.
    for a in (arrows or []):
        ay = a["row"] + y_min
        x_tail = a["tail_x"] + x_min
        x_tip = a["tip_x"] + x_min
        d = a["tip_dir"]
        parts.append(
            f'<line x1="{x_tail}" y1="{ay}" x2="{x_tip}" y2="{ay}" '
            f'stroke="#00b8d4" stroke-width="1.6" opacity="0.95"/>'
        )
        # Arrowhead triangle pointing along tip_dir.
        hx = x_tip + d * 9
        parts.append(
            f'<polygon points="{x_tip},{ay} {hx},{ay - 5} {hx},{ay + 5}" '
            f'fill="#00b8d4" opacity="0.95"/>'
        )

    # Annotation arrows detected via OCR label + solid arrowhead (orange):
    # shaft from the label (tail) to the tip resting on the curve, plus the
    # OCR'd series label at the tail.
    for a in (label_arrows or []):
        tlx, tly = a["tail_x"] + x_min, a["tail_y"] + y_min
        tpx, tpy = a["tip_x"] + x_min, a["tip_y"] + y_min
        parts.append(
            f'<line x1="{tlx}" y1="{tly}" x2="{tpx}" y2="{tpy}" '
            f'stroke="#ff6f00" stroke-width="2" opacity="0.95"/>'
        )
        parts.append(
            f'<circle cx="{tpx}" cy="{tpy}" r="4" fill="#ff6f00" '
            f'stroke="#7a3400" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{tlx + 3}" y="{tly - 3}" font-size="11" fill="#ff6f00" '
            f'font-weight="bold">{_xml_escape(a.get("label", ""))}</text>'
        )

    # Detected legend region (encloses stacked labels), in solid purple.
    for (lx0, ly0, lx1, ly1) in (legend_boxes or []):
        parts.append(
            f'<rect x="{lx0 + x_min}" y="{ly0 + y_min}" '
            f'width="{lx1 - lx0}" height="{ly1 - ly0}" fill="#7b1fa2" '
            f'fill-opacity="0.10" stroke="#7b1fa2" stroke-width="2.4"/>'
        )
        parts.append(
            f'<text x="{lx0 + x_min}" y="{ly0 + y_min - 4}" font-size="12" '
            f'fill="#7b1fa2" font-weight="bold">legend</text>'
        )

    # Detected text-label boxes (canvas-relative), in magenta, annotated with
    # the OCR reading when available.
    for i, (bx0, by0, bx1, by1) in enumerate(text_boxes or []):
        reading = ""
        if label_texts and i < len(label_texts) and label_texts[i]:
            reading = label_texts[i]
        caption = reading if reading else "text"
        parts.append(
            f'<rect x="{bx0 + x_min}" y="{by0 + y_min}" '
            f'width="{bx1 - bx0}" height="{by1 - by0}" fill="none" '
            f'stroke="#d400aa" stroke-width="1.8" stroke-dasharray="4,2"/>'
        )
        parts.append(
            f'<text x="{bx0 + x_min}" y="{by0 + y_min - 2}" font-size="12" '
            f'fill="#d400aa" font-weight="bold">{_xml_escape(caption)}</text>'
        )

    # Curve points
    legend: list[str] = []
    for i, (label, pts) in enumerate(curves.items()):
        color = _curve_color(label, i)
        legend.append((label, color))
        if len(pts) >= 2:
            poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            parts.append(
                f'<polyline points="{poly}" fill="none" stroke="{color}" '
                f'stroke-width="1.5" opacity="0.9"/>'
            )
        # Sparse dots so individual detected samples stay visible.
        step = max(1, len(pts) // 120)
        for px, py in pts[::step]:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.6" fill="{color}"/>'
            )

    # Legend (top-left, inside a translucent box)
    if legend:
        box_h = 14 * len(legend) + 8
        parts.append(
            f'<rect x="4" y="4" width="260" height="{box_h}" fill="#ffffff" '
            f'opacity="0.75" stroke="#888"/>'
        )
        for i, (label, color) in enumerate(legend):
            ty = 18 + i * 14
            parts.append(
                f'<rect x="8" y="{ty - 8}" width="10" height="10" fill="{color}"/>'
            )
            parts.append(
                f'<text x="22" y="{ty}" font-size="10" fill="#000">'
                f'{escape(label)}</text>'
            )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info(f"Debug SVG written to: {out_path}")
