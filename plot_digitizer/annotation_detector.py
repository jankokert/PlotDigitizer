"""Detect textual annotations (labels, notes, legends) and callout arrows.

* ``detect_text_labels_ocr`` - word-level OCR of in-plot labels (used when the
  PDF has no text layer, or for bitmap input).
* ``detect_arrows`` - pairs labels with solid arrowheads (and, for legends,
  horizontal on-grid arrows); feeds the parametric arrow model.
* ``detect_legend_boxes`` - a legend is a regular stack holding all labels.
* ``detect_grid_boxes`` / ``expand_to_white`` - white text patches punched
  into the grid (debug display, arrow-less notes).

Pixel constants are tuned at ~150 dpi; the arrow code scales them with the
resolution factor passed in by the caller.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .curve_extractor import _run_length


def detect_legend_boxes(
    text_boxes: list[tuple[int, int, int, int]],
    texts: Optional[list[str]] = None,
    max_pitch_spread: float = 0.25,
    min_members: int = 3,
) -> list[tuple[int, int, int, int]]:
    """
    Merge vertically-stacked, horizontally-aligned text boxes into one enclosing
    "legend" region (e.g. the USFF stack "0.050 A" … "0.250 A").  Knowing the
    whole legend rectangle lets later stages exclude it from grid/curve analysis.

    Two boxes join a legend when their x-ranges overlap by most of the narrower
    box and they sit at different y.  A legend is a REGULAR stack: at least
    `min_members` rows at an even pitch (spread of the row-to-row distances
    ≤ `max_pitch_spread` × their median).  Callout labels that merely happen
    to be left-aligned (MUH1PB Fig. 4: each label next to its own curve) sit
    at irregular distances.

    A plot labels its curves either with callouts or with ONE legend holding
    all of them.  So if `texts` form a label family ("T_A = 25 °C", …), a
    legend must contain every member of that family.
    """
    from .label_parser import parse_labels
    family = parse_labels(texts) if texts is not None else None
    n = len(text_boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def x_overlap(a, b):
        lo = max(a[0], b[0])
        hi = min(a[2], b[2])
        inter = max(0, hi - lo)
        narrow = min(a[2] - a[0], b[2] - b[0])
        return inter / narrow if narrow > 0 else 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if x_overlap(text_boxes[i], text_boxes[j]) >= 0.5:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    legends = []
    for members in groups.values():
        if len(members) < min_members:
            continue
        cy = sorted((text_boxes[m][1] + text_boxes[m][3]) / 2 for m in members)
        pitch = np.diff(cy)
        if (pitch.max() - pitch.min()) > max_pitch_spread * float(np.median(pitch)):
            continue
        if family is not None and not set(family.members) <= set(members):
            continue
        mb = [text_boxes[m] for m in members]
        # A real legend is a vertical STACK: the members' left edges are aligned
        # (small x-spread) and they sit at distinct y. Scattered callout labels
        # (e.g. CMZ5334B…5350B spread across the top) chain via x-overlap but are
        # NOT aligned, so reject them to avoid huge bogus "legend" rectangles.
        lefts = [b[0] for b in mb]
        med_w = sorted(b[2] - b[0] for b in mb)[len(mb) // 2]
        if max(lefts) - min(lefts) > 0.4 * med_w:
            continue
        xs0 = min(b[0] for b in mb); ys0 = min(b[1] for b in mb)
        xs1 = max(b[2] for b in mb); ys1 = max(b[3] for b in mb)
        if ys1 - ys0 <= xs1 - xs0:      # a stack is taller than it is wide
            continue
        legends.append((xs0, ys0, xs1, ys1))
    return legends


def expand_to_white(
    ink: np.ndarray,
    box: tuple[int, int, int, int],
    thresh: int = 4,
    margin: int = 300,
) -> tuple[int, int, int, int]:
    """
    Grow a box outward to the full extent of the white region around it — the
    edges land exactly where the grid/curve ink resumes, so the box marks where
    the grid stops (and could be extrapolated across).

    Each edge advances while the next border row/column, measured over the box's
    current cross-span, carries at most `thresh` ink pixels (blank, or only a
    thin curve grazing through); it halts at a grid line or a solid stroke.
    """
    h, w = ink.shape
    x0, y0, x1, y1 = box
    lo_x, hi_x = max(0, x0 - margin), min(w, x1 + margin)
    lo_y, hi_y = max(0, y0 - margin), min(h, y1 + margin)

    grew = True
    while grew:
        grew = False
        if y0 - 1 >= lo_y and int(ink[y0 - 1, x0:x1].sum()) <= thresh:
            y0 -= 1; grew = True
        if y1 + 1 <= hi_y and int(ink[y1, x0:x1].sum()) <= thresh:
            y1 += 1; grew = True
        if x0 - 1 >= lo_x and int(ink[y0:y1, x0 - 1].sum()) <= thresh:
            x0 -= 1; grew = True
        if x1 + 1 <= hi_x and int(ink[y0:y1, x1].sum()) <= thresh:
            x1 += 1; grew = True
    return (x0, y0, x1, y1)


def detect_grid_boxes(
    canvas_rgb: np.ndarray,
    span_frac: float = 0.55,
    min_lines: int = 2,
    pad: int = 2,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """
    Detect a white text box (e.g. the arrow-less note ``f=1.0MHz``) by the hole
    it punches in the grid, then read it.

    A label sits on a small patch of white paper that **interrupts the grid**:
    where the box is, the horizontal *and* vertical grid lines simply stop.  We
    already know where the grid lines are (rows/cols that span the axis), so we
    look for lattice pixels that are genuine white paper (not dark ink, and
    unsaturated so coloured curves crossing the grid don't count).  Those
    interrupted-lattice pixels cluster into the box; OCR then confirms it carries
    text (plot whitespace reads empty and is dropped).

    Returns a list of ``((x0, y0, x1, y1), text)`` in canvas pixel coords.
    """
    from scipy import ndimage

    r = canvas_rgb[:, :, 0] / 255.0
    g = canvas_rgb[:, :, 1] / 255.0
    b = canvas_rgb[:, :, 2] / 255.0
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    v = max_c
    s = np.where(max_c > 1e-6, (max_c - min_c) / np.where(max_c > 1e-6, max_c, 1.0), 0.0)
    ink = (s < 0.30) & (v < 0.90)
    h, w = ink.shape

    H = np.where(ink.sum(axis=1) > span_frac * w)[0]
    V = np.where(ink.sum(axis=0) > span_frac * h)[0]
    if len(H) < min_lines + 1 or len(V) < min_lines + 1:
        return []

    def _spacing(idx):
        if len(idx) < 2:
            return 20
        d = np.diff(np.sort(idx)); d = d[d > 3]
        return int(np.median(d)) if len(d) else 20
    sp = max(_spacing(H), _spacing(V))

    lat = np.zeros((h, w), bool)
    lat[H, :] = True
    lat[:, V] = True
    # Interrupted grid pixel: on the lattice, but white paper instead of a line
    # (not dark ink, and unsaturated — a coloured curve crossing is ~ink too but
    # highly saturated, so it is excluded).
    interrupted = lat & ~ink & (s < 0.20)

    it = max(3, sp // 2 + 2)
    dil = ndimage.binary_dilation(interrupted, iterations=it)
    lbl, n = ndimage.label(dil)

    results: list[tuple[tuple[int, int, int, int], str]] = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        inside = interrupted[by0:by1 + 1, bx0:bx1 + 1]
        rows_hit = sum(1 for rr in H if by0 <= rr <= by1 and inside[rr - by0].sum() > 3)
        cols_hit = sum(1 for cc in V if bx0 <= cc <= bx1 and inside[:, cc - bx0].sum() > 3)
        if rows_hit < min_lines or cols_hit < min_lines:
            continue
        # tighten to the actual interrupted pixels, then read the crop
        iy, ix = np.where(inside)
        tx0, tx1 = bx0 + int(ix.min()), bx0 + int(ix.max())
        ty0, ty1 = by0 + int(iy.min()), by0 + int(iy.max())
        box = (max(0, tx0 - pad), max(0, ty0 - pad),
               min(w, tx1 + pad + 1), min(h, ty1 + pad + 1))
        text = ocr_boxes(canvas_rgb, [box], whitelist="")[0]
        # Keep only boxes OCR confirms carry text (>=2 non-space chars); plot
        # whitespace reads empty and is discarded.
        if len(text.replace(" ", "")) >= 2:
            results.append((box, text))
    return results


def detect_text_labels_ocr(
    canvas_rgb: np.ndarray,
    min_len: int = 4,
    min_conf: int = 30,
) -> list[dict]:
    """
    Read word-level text labels with OCR (``image_to_data``), returning one dict
    per word: ``{text, box=(x0,y0,x1,y1), conf}`` in canvas pixel coords.

    Word-level OCR gives clean per-label boxes for the crisp series labels on
    datasheet plots ("CMZ5334B", "0.050 A") — far more reliable than clustering
    glyph blobs, and it is the anchor the arrow detector builds on.
    """
    try:
        import pytesseract as pt
        from PIL import Image as PILImage
    except Exception:
        return []
    data = pt.image_to_data(PILImage.fromarray(canvas_rgb.astype(np.uint8)),
                            config="--psm 11", output_type=pt.Output.DICT)
    out = []
    for i, t in enumerate(data["text"]):
        t = t.strip()
        try:
            conf = int(float(data["conf"][i]))
        except (ValueError, TypeError):
            conf = -1
        if len(t) >= min_len and conf >= min_conf:
            L, T = int(data["left"][i]), int(data["top"][i])
            R, B = L + int(data["width"][i]), T + int(data["height"][i])
            out.append({"text": t, "box": (L, T, R, B), "conf": conf})
    return out


def detect_arrows(
    canvas_rgb: np.ndarray,
    grid_mask: np.ndarray,
    labels: Optional[list[dict]] = None,
    match_radius: int = 200,
    scale: float = 1.0,
) -> list[dict]:
    """
    Detect annotation arrows by pairing OCR text labels with solid arrowheads.

    A datasheet callout is a text label plus an arrow whose **solid triangular
    head** touches the curve it names.  The head is the one robust, general cue:
    unlike the thin shaft (which may lie on a grid line and vanish with the grid)
    it is a compact *fat* ink blob, detectable in any orientation — so this works
    for diagonal arrows and varied heads.

    Method: remove the grid, find compact thick-ink cores (arrowhead candidates,
    telling them from elongated curve bodies by shape), then greedily match each
    OCR label to its nearest head.  The **tip** is the head pixel farthest from
    the label (the apex resting on the curve = the arrow's end / the point that
    identifies the curve); the **tail** is the label-box point nearest the tip.

    Returns dicts ``{label, tail_x, tail_y, tip_x, tip_y, head_box}`` in canvas
    pixel coords.  Two detectors run in order: solid-arrowhead matching (general,
    diagonal), then — only for labels it left unmatched — a horizontal on-grid
    shaft tracer (the USFF case: shafts lie on grid lines and vanish with the
    grid, but the label anchors a leftward trace to the solid head).  A label
    matched by neither yields no arrow (a missing arrow beats a wrong one).
    """
    from scipy import ndimage
    from .curve_extractor import _run_length

    r = canvas_rgb[:, :, 0] / 255.0
    g = canvas_rgb[:, :, 1] / 255.0
    b = canvas_rgb[:, :, 2] / 255.0
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    v = max_c
    s = np.where(max_c > 1e-6, (max_c - min_c) / np.where(max_c > 1e-6, max_c, 1.0), 0.0)
    ink = (s < 0.30) & (v < 0.90)
    fg = ink & ~grid_mask

    if labels is None:
        labels = detect_text_labels_ocr(canvas_rgb)
    if not labels:
        return []

    # Arrowhead candidates: compact fat cores (survive erosion by ~3 px), and
    # compact (curve bodies erode to long thin cores, so shape rejects them).
    dist = ndimage.distance_transform_edt(fg)
    # Pixel constants tuned at ~150 dpi, multiplied by the resolution `scale`.
    lbl, n = ndimage.label(dist >= 3.25 * scale)
    heads = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        bw, bh = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        if (len(xs) < 8 * scale * scale or max(bw, bh) > 22 * scale or min(bw, bh) < 3 * scale
                or max(bw, bh) > 3.5 * min(bw, bh)):
            continue
        heads.append({"cx": int(xs.mean()), "cy": int(ys.mean()),
                      "xs": xs, "ys": ys,
                      "box": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))})

    arrows: list[dict] = []
    used: set[int] = set()
    matched_labels: set[int] = set()
    for li, lab in enumerate(labels):
        lx0, ly0, lx1, ly1 = lab["box"]
        acx, acy = (lx0 + lx1) // 2, (ly0 + ly1) // 2
        best, best_d = None, (match_radius * scale) ** 2
        for j, hd in enumerate(heads):
            if j in used:
                continue
            d2 = (hd["cx"] - acx) ** 2 + (hd["cy"] - acy) ** 2
            if d2 < best_d:
                best_d, best = d2, j
        if best is None:
            continue
        used.add(best)
        matched_labels.add(li)
        hd = heads[best]
        k = int(np.argmax((hd["xs"] - acx) ** 2 + (hd["ys"] - acy) ** 2))
        tip_x, tip_y = int(hd["xs"][k]), int(hd["ys"][k])
        tail_x = min(max(tip_x, lx0), lx1)
        tail_y = min(max(tip_y, ly0), ly1)
        arrows.append({"label": lab["text"],
                       "tail_x": tail_x, "tail_y": tail_y,
                       "tip_x": tip_x, "tip_y": tip_y,
                       "head_box": hd["box"]})

    # Horizontal on-grid fallback for labels without a solid diagonal head — but
    # only for labels that form a LEGEND (a vertical stack, e.g. USFF 0.05–0.25
    # A). Horizontal callout arrows are a legend pattern; restricting to it keeps
    # a lone scattered label from latching onto a bold grid line as a "shaft".
    legends = detect_legend_boxes([lab["box"] for lab in labels],
                                  [lab["text"] for lab in labels])

    def _in_legend(box):
        for lx0, ly0, lx1, ly1 in legends:
            if (box[0] >= lx0 - 4 and box[2] <= lx1 + 4
                    and box[1] >= ly0 - 4 and box[3] <= ly1 + 4):
                return True
        return False

    unmatched = [labels[i] for i in range(len(labels))
                 if i not in matched_labels and _in_legend(labels[i]["box"])]
    if unmatched:
        h_img, w_img = ink.shape
        vrun = _run_length(ink, axis=0)
        bold = ink & (vrun >= 3 * scale)            # bolder than a thin grid line
        label_zone = np.zeros_like(ink)
        for lab in labels:                          # a neighbour's glyphs are not a shaft
            a, bb, c, dd = lab["box"]
            label_zone[max(0, bb - 2):min(h_img, dd + 2),
                       max(0, a - 2):min(w_img, c + 2)] = True
        bold_m = bold & ~label_zone
        ink_m = ink & ~label_zone
        for lab in unmatched:
            arr = _trace_horizontal_arrow(lab, ink_m, bold_m, scale=scale)
            if arr is not None:
                arrows.append(arr)
    return arrows


def _trace_horizontal_arrow(lab: dict, ink_m: np.ndarray, bold_m: np.ndarray,
                            half_band: int = 7, scale: float = 1.0) -> Optional[dict]:
    """
    Trace a horizontal callout arrow that points LEFT from a right-side label to
    its curve (the USFF case).  The shaft lies on a grid line, so instead of the
    (grid-removed) shaft we locate the solid arrowhead: the arrow row is the band
    row carrying the most bold ink left of the label; the tip is the leftmost
    *wide* thickness run (the triangular head ~8 px, vs 1–2 px grid crossings).

    Returns an arrow dict or None.  Gated on span>=40 and a head thickness peak
    >=10 so grid/curve crossings don't masquerade as arrows.  All pixel
    constants are for ~150 dpi and multiplied by `scale`.
    """
    def k(v: float) -> int:
        return int(round(v * scale))

    h_img, w_img = ink_m.shape
    lx0, ly0, lx1, ly1 = lab["box"]
    yc = (ly0 + ly1) // 2
    half_band = k(half_band)
    y_lo, y_hi = max(1, yc - k(9)), min(h_img - 1, yc + k(10))
    counts = [(int(bold_m[yy, 0:lx0].sum()), yy) for yy in range(y_lo, y_hi)]
    if not counts:
        return None
    cnt, row = max(counts)
    if cnt < k(25):
        return None
    thick = ink_m[max(0, row - half_band):row + half_band + 1, 0:lx0].sum(axis=0)
    wide = np.where(thick >= k(8))[0]
    if len(wide) == 0:
        return None
    # leftmost run of >=4 consecutive wide columns = the arrowhead
    runs = []
    start = prev = wide[0]
    for xi in wide[1:]:
        if xi == prev + 1:
            prev = xi
        else:
            if prev - start + 1 >= k(4):
                runs.append((start, prev))
            start = prev = xi
    if prev - start + 1 >= k(4):
        runs.append((start, prev))
    if not runs:
        return None
    hs, he = min(runs)
    span = lx0 - hs
    peak = int(thick[hs:he + 1].max())
    if span < k(40) or peak < k(10):
        return None
    return {"label": lab["text"],
            "tail_x": min(max(hs, lx0), lx1), "tail_y": yc,
            "tip_x": int(hs), "tip_y": int(row),
            "head_box": (int(hs), int(row - half_band), int(he), int(row + half_band))}


def ocr_boxes(
    canvas_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    whitelist: str = "0123456789.A ",
) -> list[str]:
    """
    Read each box with OCR on its isolated crop (upscaled) — far more reliable
    than OCR on the whole cluttered plot.  Returns one string per box ("" if OCR
    is unavailable or reads nothing).
    """
    try:
        import pytesseract as pt
        from PIL import Image as PILImage
    except Exception:
        return ["" for _ in boxes]

    out = []
    cfg = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    for (x0, y0, x1, y1) in boxes:
        crop = canvas_rgb[y0:y1, x0:x1].astype(np.uint8)
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            out.append("")
            continue
        pim = PILImage.fromarray(crop)
        pim = pim.resize((pim.width * 4, pim.height * 4), PILImage.LANCZOS)
        try:
            out.append(pt.image_to_string(pim, config=cfg).strip())
        except Exception:
            out.append("")
    return out
