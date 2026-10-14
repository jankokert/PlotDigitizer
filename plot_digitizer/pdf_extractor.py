"""Locate a numbered figure ("Fig. 3", "Figure 3", ...) in a datasheet PDF and
export it as a vector PDF crop, a high-resolution PNG and a JSON sidecar.

Datasheet plots are almost always vector graphics placed directly above (or
below) their caption.  We therefore:

1. find the caption line that *starts* with the figure label (so body-text
   references like "see fig. 3" are ignored),
2. cluster all vector drawings / embedded images on that page into connected
   groups and pick the group adjacent to the caption,
3. grow that group by nearby non-caption text (tick numbers, axis titles),
4. crop.  The caption itself is excluded from the crop; its text goes into
   the file name and the JSON.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# "Fig. 3", "Fig.3", "FIG 3", "Figure 3", "Figure 3a" -- at the start of a line.
_CAPTION_RE = re.compile(r"^\s*fig(?:ure|\.)?\s*(\d+)([a-z]?)\b\.?\s*[-–—:.]?\s*(.*)$", re.I)

_CLUSTER_GAP = 3.0       # pt: drawings closer than this belong to the same figure
_TEXT_MARGIN = 14.0      # pt: text this close to the figure is pulled in (ticks, axis titles)
_FAR_SIDE_MARGIN = 4.0   # pt: same, on the side facing away from the caption (only tick
                         #     overhang lives there; beyond it are section headings etc.)
_MAX_TEXT_GROWTH = 45.0  # pt: never grow further than this beyond the graphics
_PADDING = 4.0           # pt: white border around the final crop
_MIN_FIGURE_SIZE = 40.0  # pt: smaller clusters are logos / bullets, not plots


@dataclass
class FigureInfo:
    source_pdf: str
    figure: str
    page: int              # 1-based
    caption: str
    title: str
    clip: tuple[float, float, float, float]  # PDF points, top-left origin
    dpi: int
    png: str | None = None
    pdf: str | None = None
    # Text lines inside the crop, box in PNG pixel coords (x0, y0, x1, y1);
    # subscripts are marked with "_" ("T_A = 25 °C").  Lets the digitizer use
    # exact labels instead of OCR.
    text: list[dict] | None = None


@dataclass
class _Caption:
    page: int
    rect: pymupdf.Rect
    number: str
    text: str
    title: str


def _find_captions(doc: pymupdf.Document) -> list[_Caption]:
    caps = []
    for pno, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                m = _CAPTION_RE.match(text)
                if m:
                    caps.append(_Caption(pno, pymupdf.Rect(line["bbox"]),
                                         m.group(1) + m.group(2).lower(), text,
                                         m.group(3).strip()))
    return caps


def _graphic_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Bounding boxes of vector drawings and images, minus page-wide rules."""
    pw = page.rect.width
    rects = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
    for img in page.get_images(full=True):
        rects.extend(page.get_image_rects(img[0]))
    # Header/footer separator lines span most of the page width.
    return [r for r in rects if r.width < 0.6 * pw and not r.is_infinite]


def _cluster(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Merge rects that touch (within _CLUSTER_GAP) into connected groups."""
    clusters = [pymupdf.Rect(r) for r in rects]
    merged = True
    while merged:
        merged = False
        out: list[pymupdf.Rect] = []
        for r in clusters:
            for o in out:
                if (o + (-_CLUSTER_GAP, -_CLUSTER_GAP, _CLUSTER_GAP, _CLUSTER_GAP)).intersects(r):
                    o.include_rect(r)
                    merged = True
                    break
            else:
                out.append(pymupdf.Rect(r))
        clusters = out
    return clusters


def _x_overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _pick_cluster(clusters: list[pymupdf.Rect], cap: pymupdf.Rect) -> pymupdf.Rect | None:
    best, best_cost = None, float("inf")
    for c in clusters:
        if c.width < _MIN_FIGURE_SIZE or c.height < _MIN_FIGURE_SIZE:
            continue
        if _x_overlap(c, cap) < 0.5 * min(c.width, cap.width):
            continue
        if c.y1 <= cap.y0 + 1:           # figure above caption (usual case)
            cost = cap.y0 - c.y1
        elif c.y0 >= cap.y1 - 1:         # figure below caption
            cost = (c.y0 - cap.y1) * 1.5  # mild preference for "above"
        else:
            continue
        if cost < best_cost:
            best, best_cost = c, cost
    return best


def _grow_with_text(page: pymupdf.Page, fig: pymupdf.Rect, cap: pymupdf.Rect,
                    caption_rects: list[pymupdf.Rect]) -> pymupdf.Rect:
    """Add tick labels / axis titles lying just outside the graphics.

    The x-axis title sits on the caption side; the opposite side gets only a
    small margin so headings or notes above the plot are not pulled in.
    """
    caption_below = cap.y0 >= fig.y1 - 1
    m_top = _FAR_SIDE_MARGIN if caption_below else _TEXT_MARGIN
    m_bot = _TEXT_MARGIN if caption_below else _FAR_SIDE_MARGIN
    limit = fig + (-_MAX_TEXT_GROWTH, -_MAX_TEXT_GROWTH, _MAX_TEXT_GROWTH, _MAX_TEXT_GROWTH)
    words = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]
    words = [w for w in words
             if w in limit and not any(w.intersects(c) for c in caption_rects)]
    out = pymupdf.Rect(fig)
    grown = True
    while grown:
        grown = False
        zone = out + (-_TEXT_MARGIN, -m_top, _TEXT_MARGIN, m_bot)
        for w in words:
            if zone.intersects(w) and not w in out:
                out |= w
                grown = True
    # Never extend across a caption (e.g. the one of the figure above).
    for c in caption_rects:
        if _x_overlap(c, out) > 0 and c.intersects(out):
            if c.y0 >= fig.y1:
                out.y1 = min(out.y1, c.y0 - 1)
            elif c.y1 <= fig.y0:
                out.y0 = max(out.y0, c.y1 + 1)
    return out


def _line_text(line: dict) -> str:
    """Join the spans of a line, marking smaller lowered spans as "_subscript"."""
    spans = line["spans"]
    big = max(sp["size"] for sp in spans)
    out = ""
    for sp in spans:
        sub = sp["size"] < 0.85 * big and sp["origin"][1] > spans[0]["origin"][1] - 0.1
        t = sp["text"]
        if sub and t.strip() and out and not out.endswith(" "):
            t = "_" + t.strip()
        out += t
    return re.sub(r"\s+", " ", out).strip()


def _text_lines(page: pymupdf.Page, clip: pymupdf.Rect, dpi: int) -> list[dict]:
    k = dpi / 72.0
    out = []
    for block in page.get_text("dict", clip=clip)["blocks"]:
        for line in block.get("lines", []):
            text = _line_text(line)
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            out.append({"text": text,
                        "box": [round((x0 - clip.x0) * k), round((y0 - clip.y0) * k),
                                round((x1 - clip.x0) * k), round((y1 - clip.y0) * k)],
                        "horizontal": abs(line["dir"][1]) < 0.1})
    return out


def _safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:max_len].rstrip(" .")


def locate_figure(doc: pymupdf.Document, figure: str) -> tuple[_Caption, pymupdf.Rect]:
    figure = str(figure).strip().lower()
    caps = _find_captions(doc)
    hits = [c for c in caps if c.number == figure]
    if not hits:
        found = sorted({c.number for c in caps}, key=lambda n: (len(n), n))
        raise LookupError(f"No caption for figure {figure!r} found "
                          f"(captions present: {', '.join(found) or 'none'})")
    for cap in hits:   # several hits (e.g. table of figures): first with a plot wins
        page = doc[cap.page]
        same_page_caps = [c.rect for c in caps if c.page == cap.page]
        fig = _pick_cluster(_cluster(_graphic_rects(page)), cap.rect)
        if fig is None:
            continue
        clip = _grow_with_text(page, fig, cap.rect, same_page_caps)
        clip = (clip + (-_PADDING, -_PADDING, _PADDING, _PADDING)) & page.rect
        return cap, clip
    raise LookupError(f"Caption for figure {figure!r} found on page(s) "
                      f"{sorted({c.page + 1 for c in hits})}, but no graphics next to it "
                      "(scanned PDF without vector plots?)")


def extract_figure(pdf_path: Path, figure: str, out_dir: Path | None = None,
                   dpi: int = 600, part_name: str | None = None) -> FigureInfo:
    """Export figure *figure* of *pdf_path* as PNG + cropped PDF + JSON.

    Files are named ``<part> - Figure <n> - <caption title>.{png,pdf,json}``.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir) if out_dir else pdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    part = part_name or pdf_path.stem

    with pymupdf.open(pdf_path) as doc:
        cap, clip = locate_figure(doc, figure)
        page = doc[cap.page]
        stem = _safe_filename(f"{part} - Figure {cap.number}"
                              + (f" - {cap.title}" if cap.title else ""), 150)
        png_path = out_dir / f"{stem}.png"
        pdf_out = out_dir / f"{stem}.pdf"

        page.get_pixmap(clip=clip, dpi=dpi, alpha=False).save(png_path)
        text = _text_lines(page, clip, dpi)

        with pymupdf.open() as crop:
            p = crop.new_page(width=clip.width, height=clip.height)
            p.show_pdf_page(p.rect, doc, cap.page, clip=clip)
            try:
                crop.save(pdf_out, garbage=3, deflate=True)
            except Exception as exc:   # e.g. the old crop is open in a PDF viewer
                logger.warning(f"Could not write {pdf_out.name}: {exc}")

    info = FigureInfo(source_pdf=pdf_path.name, figure=cap.number, page=cap.page + 1,
                      caption=cap.text, title=cap.title,
                      clip=tuple(round(v, 2) for v in clip), dpi=dpi,
                      png=png_path.name, pdf=pdf_out.name, text=text)
    (out_dir / f"{stem}.json").write_text(json.dumps(asdict(info), indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    logger.info(f"Figure {cap.number} (page {cap.page + 1}) -> {png_path}")
    return info
