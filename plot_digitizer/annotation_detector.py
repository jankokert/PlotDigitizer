"""Detect textual annotations (labels) in a plot canvas.

Datasheet plots carry text labels — axis-style callouts like "0.160 A" or a
note like "f=1.0MHz" — that almost always sit on a small **white background box**
punched into the grid so the text stays readable.  That gives two robust,
model-free cues we combine here:

1. the glyphs are *compact* dark ink (short runs in both directions), unlike the
   long thin grid lines or the thick data curves; and
2. the box interrupts the grid — inside it there is no long line, only the
   compact glyphs on white.

Detecting and masking these regions *before* the grid/curve pixel arithmetic
keeps annotations from corrupting the geometry (the "annotations first" order).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .curve_extractor import _run_length


def detect_text_boxes(
    ink: np.ndarray,
    glyph_run: int = 12,
    pad: int = 6,
) -> list[tuple[int, int, int, int]]:
    """
    Return bounding boxes (x0, y0, x1, y1) of text-label regions in canvas
    pixel coords.

    `ink` is the boolean achromatic-ink mask of the canvas.  A glyph pixel has a
    short contiguous run in *both* axes (`<= glyph_run`); grid lines and curves
    do not.  Compact glyph components are clustered into text lines, and a
    cluster is kept only if the ink inside its box is mostly glyph-ink (i.e. no
    long grid/curve line runs through it — the white-box cue).
    """
    from scipy import ndimage

    h, w = ink.shape
    vr = _run_length(ink, axis=0)
    hr = _run_length(ink, axis=1)
    glyph = ink & (vr <= glyph_run) & (hr <= glyph_run)

    lbl, n = ndimage.label(glyph, structure=np.ones((3, 3), dtype=int))
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        bw = int(xs.max() - xs.min()) + 1
        bh = int(ys.max() - ys.min()) + 1
        if 3 <= bw <= 45 and 6 <= bh <= 45 and len(xs) >= 12 and max(bw, bh) <= 6 * min(bw, bh):
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    # Cluster glyph boxes that sit on the same text line (small gaps).
    groups = _cluster_boxes(boxes, x_gap=30, y_gap=12)

    # Keep clusters that look like real multi-glyph labels.  We do NOT insist
    # the grid be absent inside — datasheet labels often sit *on* grid lines —
    # only that the cluster is several compact glyphs forming a short line of
    # text (wider than tall, a handful of characters).
    confirmed: list[tuple[int, int, int, int]] = []
    for (gx0, gy0, gx1, gy1, count) in groups:
        bw, bh = gx1 - gx0 + 1, gy1 - gy0 + 1
        if count < 3:
            continue
        if bw < 18 or bh < 8:
            continue
        if bw < 1.3 * bh:          # a text line is clearly wider than tall
            continue
        confirmed.append((max(0, gx0 - pad), max(0, gy0 - pad),
                          min(w, gx1 + pad + 1), min(h, gy1 + pad + 1)))
    return confirmed


def detect_legend_boxes(
    text_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Merge vertically-stacked, horizontally-aligned text boxes into one enclosing
    "legend" region (e.g. the USFF stack "0.050 A" … "0.250 A").  Knowing the
    whole legend rectangle lets later stages exclude it from grid/curve analysis.

    Two boxes join a legend when their x-ranges overlap by most of the narrower
    box and they sit at different y.  A legend needs at least two members.
    """
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
        if len(members) < 2:
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
    lbl, n = ndimage.label(dist >= 3.25)
    heads = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        bw, bh = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        if (len(xs) < 8 or max(bw, bh) > 22 or min(bw, bh) < 3
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
        best, best_d = None, match_radius ** 2
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
    legends = detect_legend_boxes([lab["box"] for lab in labels])

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
        bold = ink & (vrun >= 3)                    # bolder than a thin grid line
        label_zone = np.zeros_like(ink)
        for lab in labels:                          # a neighbour's glyphs are not a shaft
            a, bb, c, dd = lab["box"]
            label_zone[max(0, bb - 2):min(h_img, dd + 2),
                       max(0, a - 2):min(w_img, c + 2)] = True
        bold_m = bold & ~label_zone
        ink_m = ink & ~label_zone
        for lab in unmatched:
            arr = _trace_horizontal_arrow(lab, ink_m, bold_m)
            if arr is not None:
                arrows.append(arr)
    return arrows


def _trace_horizontal_arrow(lab: dict, ink_m: np.ndarray, bold_m: np.ndarray,
                            half_band: int = 7) -> Optional[dict]:
    """
    Trace a horizontal callout arrow that points LEFT from a right-side label to
    its curve (the USFF case).  The shaft lies on a grid line, so instead of the
    (grid-removed) shaft we locate the solid arrowhead: the arrow row is the band
    row carrying the most bold ink left of the label; the tip is the leftmost
    *wide* thickness run (the triangular head ~8 px, vs 1–2 px grid crossings).

    Returns an arrow dict or None.  Gated on span>=40 and a head thickness peak
    >=10 so grid/curve crossings don't masquerade as arrows.
    """
    h_img, w_img = ink_m.shape
    lx0, ly0, lx1, ly1 = lab["box"]
    yc = (ly0 + ly1) // 2
    y_lo, y_hi = max(1, yc - 9), min(h_img - 1, yc + 10)
    counts = [(int(bold_m[yy, 0:lx0].sum()), yy) for yy in range(y_lo, y_hi)]
    cnt, row = max(counts)
    if cnt < 25:
        return None
    thick = ink_m[row - half_band:row + half_band + 1, 0:lx0].sum(axis=0)
    wide = np.where(thick >= 8)[0]
    if len(wide) == 0:
        return None
    # leftmost run of >=4 consecutive wide columns = the arrowhead
    runs = []
    start = prev = wide[0]
    for xi in wide[1:]:
        if xi == prev + 1:
            prev = xi
        else:
            if prev - start + 1 >= 4:
                runs.append((start, prev))
            start = prev = xi
    if prev - start + 1 >= 4:
        runs.append((start, prev))
    if not runs:
        return None
    hs, he = min(runs)
    span = lx0 - hs
    peak = int(thick[hs:he + 1].max())
    if span < 40 or peak < 10:
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


def trace_label_arrow(
    ink: np.ndarray,
    label_box: tuple[int, int, int, int],
    x_stop: int = 0,
    search_gap: int = 30,
) -> Optional[dict]:
    """
    Follow the arrow that belongs to a label: start at the label's left edge and
    walk the dark line LEFT (allowing vertical drift, so diagonal arrows and
    varied heads work) until it ends on the curve it points at.

    Guided detection — because the start point and direction come from the label
    it is robust even where the shaft lies on a grid line.  Returns an arrow
    dict {row, tip_x, tip_y, tail_x, tip_dir, path} or None if no shaft is found.
    """
    x0, y0, x1, y1 = label_box
    h, w = ink.shape
    best = None
    for yy in range(y0, y1 + 1):
        # locate the shaft start a little left of the label (skip the text gap)
        start = None
        for xx in range(x0 - 1, max(x_stop, x0 - search_gap) - 1, -1):
            if any(0 <= yy + dy < h and ink[yy + dy, xx] for dy in (-1, 0, 1)):
                start = xx
                break
        if start is None:
            continue
        # march left, following ink with small vertical drift
        yc, x, run, gaps = yy, start, 0, 0
        path = [(start, yy)]
        while x - 1 > x_stop and gaps < 6:
            moved = False
            for dy in (0, -1, 1, -2, 2, -3, 3):
                if 0 <= yc + dy < h and ink[yc + dy, x - 1]:
                    yc += dy
                    moved = True
                    break
            x -= 1
            if moved:
                run += 1
                gaps = 0
                path.append((x, yc))
            else:
                gaps += 1
        if best is None or run > best[0]:
            best = (run, yy, path)
    if best is None or best[0] < 15:
        return None
    run, yy, path = best
    tip = path[-1]
    return {
        "row": yy, "tip_x": tip[0], "tip_y": tip[1],
        "tail_x": path[0][0], "tip_dir": -1, "path": path,
    }


def detect_label_arrows(
    ink: np.ndarray,
    labels: list[tuple[int, int, int, int]],
    x_stop: int = 0,
) -> list[dict]:
    """Trace one arrow per label (the plausibility link: N labels ⇒ N arrows)."""
    arrows = []
    for b in labels:
        a = trace_label_arrow(ink, b, x_stop=x_stop)
        if a is not None:
            arrows.append(a)
    return arrows


def snap_labels_to_legend(
    text_boxes: list[tuple[int, int, int, int]],
    legends: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Extend each label that belongs to a legend to the legend's full x-range, so
    a label the glyph clustering under-segmented (e.g. USFF "0.160 A" caught as
    just "60 A" where the arrow crossed it) gets its complete box back.
    """
    out = []
    for b in text_boxes:
        for lg in legends:
            inside_y = b[1] >= lg[1] - 4 and b[3] <= lg[3] + 4
            overlap_x = min(b[2], lg[2]) - max(b[0], lg[0]) > 0
            if inside_y and overlap_x:
                b = (lg[0], b[1], lg[2], b[3])
                break
        out.append(b)
    return out


def _cluster_boxes(boxes, x_gap, y_gap):
    """Union glyph boxes whose (padded) rectangles touch; return group boxes."""
    used = [False] * len(boxes)
    groups = []
    for i, bi in enumerate(boxes):
        if used[i]:
            continue
        gx0, gy0, gx1, gy1 = bi
        used[i] = True
        count = 1
        changed = True
        while changed:
            changed = False
            for j, bj in enumerate(boxes):
                if used[j]:
                    continue
                if (bj[0] <= gx1 + x_gap and bj[2] >= gx0 - x_gap
                        and bj[1] <= gy1 + y_gap and bj[3] >= gy0 - y_gap):
                    gx0, gy0 = min(gx0, bj[0]), min(gy0, bj[1])
                    gx1, gy1 = max(gx1, bj[2]), max(gy1, bj[3])
                    used[j] = True
                    count += 1
                    changed = True
        groups.append((gx0, gy0, gx1, gy1, count))
    return groups
