"""
Regression check over the real datasheet plots in tests/.

Runs the full pipeline (CLI defaults: trace + grid suppression, with debug
files) on every case in parallel and compares the result with a frozen
baseline:

* plot canvas, axis calibration (scale + data range at the canvas edges),
* CSV column header, curve set (names),
* per curve: geometry in image pixels — symmetric mean distance and
  Hausdorff (max) distance between baseline and current polyline,
* run time,
* residual ink: curve pixels no extracted curve explains (lower = better),
* for PDF-sourced cases: error against the exact VECTOR curves of the PDF
  (ground truth) — mean / 95th-percentile distance in px and coverage (share
  of the true curve length within one stroke width of an extracted curve).

Usage (from the project root):
    .venv\\Scripts\\python.exe tests\\regression.py                  # compare
    .venv\\Scripts\\python.exe tests\\regression.py --update-baseline
    .venv\\Scripts\\python.exe tests\\regression.py --cases CMZ USFF   # substring filter

Outputs of the current run (CSV, debug SVG/PNG/JSON, log, summary JSON) go
to tests/regression/current/<case>/ for inspection; the baseline summaries
live in tests/regression/baseline/<case>.json (commit these).
Exit code 1 if anything differs from the baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
BASE_DIR = TESTS / "regression" / "baseline"
CUR_DIR = TESTS / "regression" / "current"
sys.path.insert(0, str(ROOT))

# Case name -> image in tests/.  A sibling <image>.json (PDF sidecar) is picked
# up by the pipeline automatically, exactly as in normal use.  PDF-sourced
# figures run twice: "auto" (= vector mode, exact) and forced "raster", whose
# tracer is then scored against the vector ground truth.
_MUH3 = "MUH1PB,PC,PD - Figure 3 - Typical Instantaneous Forward Characteristics.png"
_MUH4 = "MUH1PB,PC,PD - Figure 4 - Typical Reverse Characteristics.png"
CASES = {
    "BAT46GW_fig3": "BAT46GW - Figure 3 - Typical Junction Capacitance.png",
    "CMZ53xxB": "CMZ53xxB Zener diode - V_Z-I_R.png",
    "USFF_1206": "USFF_1206 - with axis labels.png",
    "MUH1PB_fig3": _MUH3,
    "MUH1PB_fig4": _MUH4,
    "MUH1PB_fig3_raster": _MUH3,
    "MUH1PB_fig4_raster": _MUH4,
}
MODES = {"MUH1PB_fig3_raster": "raster", "MUH1PB_fig4_raster": "raster"}
VECTOR_TWIN = {"MUH1PB_fig3_raster": "MUH1PB_fig3", "MUH1PB_fig4_raster": "MUH1PB_fig4"}

SAME_PX = 0.5      # Hausdorff below this: identical
CLOSE_PX = 3.0     # below this: minor change


def _run_case(name: str) -> dict:
    """Worker: run one case, write artefacts to current/<case>/, return summary."""
    from plot_digitizer.digitizer import digitize_plot

    out = CUR_DIR / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    src = TESTS / CASES[name]
    img = out / src.name                       # copy so outputs land next to it
    shutil.copy(src, img)
    if src.with_suffix(".json").exists():
        shutil.copy(src.with_suffix(".json"), img.with_suffix(".json"))

    log = logging.getLogger()
    log.handlers.clear()
    fh = logging.FileHandler(out / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(fh)
    log.setLevel(logging.INFO)

    try:
        summary = digitize_plot(
            image_path=img, output_path=img.with_suffix(".csv"),
            method="trace", has_grid=True, mode=MODES.get(name, "auto"),
            debug_svg=img.with_name(img.stem + "_debug.svg"),
        )
    except Exception as exc:                  # report, don't kill the whole run
        logging.exception("pipeline failed")
        return {"error": f"{type(exc).__name__}: {exc}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _densify(pts, step: float = 1.0) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts.reshape(-1, 2)
    seg = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.hypot(*(b - a)) / step))
        seg.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    return np.vstack(seg)


def _curve_dist(a, b) -> tuple[float, float]:
    """(symmetric mean distance, Hausdorff distance) in pixels."""
    da, db = _densify(a), _densify(b)
    if len(da) == 0 or len(db) == 0:
        return float("inf"), float("inf")
    ab = cKDTree(db).query(da)[0]
    ba = cKDTree(da).query(db)[0]
    return float((ab.mean() + ba.mean()) / 2), float(max(ab.max(), ba.max()))


def _edges(calib, p_lo, p_hi) -> tuple[str, float, float]:
    from plot_digitizer.axis_reader import pixel_to_data
    scale = calib[4] if len(calib) > 4 else "lin"
    return scale, pixel_to_data(p_lo, calib), pixel_to_data(p_hi, calib)


def _calib_diff(base: dict, cur: dict, axis: str) -> str | None:
    pa_b, pa_c = base["plot_area"], cur["plot_area"]
    lo, hi = (0, 2) if axis == "x" else (1, 3)
    sb, b0, b1 = _edges(base[f"{axis}_calib"], pa_b[lo], pa_b[hi])
    sc, c0, c1 = _edges(cur[f"{axis}_calib"], pa_c[lo], pa_c[hi])
    span = max(abs(b1 - b0), 1e-12)
    if sb == sc and abs(c0 - b0) / span < 1e-3 and abs(c1 - b1) / span < 1e-3:
        return None
    return f"{axis}: {sb} {b0:.4g}..{b1:.4g} -> {sc} {c0:.4g}..{c1:.4g}"


def compare(base: dict, cur: dict) -> dict:
    """Return {status, notes: [...], curves: [(name, status, mean, haus)]}."""
    notes: list[str] = []
    if "error" in cur:
        return {"status": "ERROR", "notes": [cur["error"]], "curves": []}
    if base["plot_area"] != cur["plot_area"]:
        notes.append(f"canvas {base['plot_area']} -> {cur['plot_area']}")
    for ax in "xy":
        d = _calib_diff(base, cur, ax)
        if d:
            notes.append(d)
    if base["curve_header"] != cur["curve_header"]:
        notes.append(f"header '{base['curve_header']}' -> '{cur['curve_header']}'")

    bc, cc = base["curves_px"], cur["curves_px"]
    # Match curves by GEOMETRY (names like segNN shift when a curve is added
    # or dropped); a differing name is reported separately as RENAMED.
    pairs = []
    rest_b, rest_c = list(bc), list(cc)
    if bc and cc:
        cost = np.array([[_curve_dist(bc[b], cc[c])[0] for c in rest_c] for b in rest_b])
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            if cost[i, j] < 20:
                pairs.append((rest_b[i], rest_c[j]))
        matched_b = {p[0] for p in pairs}
        matched_c = {p[1] for p in pairs}
        rest_b = [n for n in rest_b if n not in matched_b]
        rest_c = [n for n in rest_c if n not in matched_c]

    rows = []
    for b, c in pairs:
        mean, haus = _curve_dist(bc[b], cc[c])
        st = "same" if haus < SAME_PX else ("close" if haus < CLOSE_PX else "CHANGED")
        label = b if b == c else f"{b} -> {c}"
        if b != c and st != "CHANGED":
            st = "RENAMED"
        rows.append((label, st, mean, haus))
    rows += [(n, "MISSING", None, None) for n in rest_b]
    rows += [(n, "NEW", None, None) for n in rest_c]

    changed = bool(notes) or any(r[1] not in ("same", "close") for r in rows)
    minor = any(r[1] == "close" for r in rows)
    status = "CHANGED" if changed else ("close" if minor else "same")
    return {"status": status, "notes": notes, "curves": rows}


def ground_truth(name: str) -> tuple[list[np.ndarray], float] | None:
    """Exact curves of a PDF-sourced case: the widest stroked vector paths,
    and their stroke width in PNG pixels."""
    side = (TESTS / CASES[name]).with_suffix(".json")
    if not side.exists():
        return None
    meta = json.loads(side.read_text(encoding="utf-8"))
    if "source_pdf" not in meta:
        return None
    from plot_digitizer.pdf_extractor import vector_polylines
    polys = vector_polylines(TESTS / meta["source_pdf"], meta["page"], meta["clip"], meta["dpi"])
    if not polys:
        return None
    w = max(p["width_pt"] for p in polys)
    curves = [np.array(p["points"]) for p in polys if abs(p["width_pt"] - w) < 1e-3]
    return curves, w * meta["dpi"] / 72.0


def gt_metrics(curves_px: dict, gt: list[np.ndarray], stroke_px: float) -> dict:
    """Match each true curve to an extracted one; pooled error statistics."""
    if not curves_px:
        return {"mean": float("inf"), "p95": float("inf"), "coverage": 0.0}
    names = list(curves_px)
    trees = [cKDTree(_densify(curves_px[n])) for n in names]
    dense_gt = [_densify(g) for g in gt]
    cost = np.array([[t.query(g)[0].mean() for t in trees] for g in dense_gt])
    r, c = linear_sum_assignment(cost)
    errs = []
    for i in range(len(gt)):
        if i in r:
            errs.append(trees[c[list(r).index(i)]].query(dense_gt[i])[0])
        else:                                    # unmatched true curve: all missed
            errs.append(np.full(len(dense_gt[i]), np.inf))
    e = np.concatenate(errs)
    # Resolvable = the nearest OTHER true curve is more than one stroke away;
    # where curves overlap (a family converging) they cannot be told apart.
    sep = []
    for i, g in enumerate(dense_gt):
        others = [cKDTree(h) for j, h in enumerate(dense_gt) if j != i]
        sep.append(np.min([t.query(g)[0] for t in others], axis=0) if others
                   else np.full(len(g), np.inf))
    res = np.concatenate(sep) > stroke_px
    fin = e[np.isfinite(e)]
    er = e[res]
    return {"mean": float(fin.mean()) if len(fin) else float("inf"),
            "p95": float(np.percentile(fin, 95)) if len(fin) else float("inf"),
            "coverage": float((e <= stroke_px).mean()),
            "res_p95": float(np.percentile(er, 95)) if len(er) and np.isfinite(er).all() else float("inf"),
            "res_cov": float((er <= stroke_px / 2).mean()) if len(er) else 1.0}


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def _print_quality(name: str, base: dict | None, cur: dict, gt) -> None:
    """Residual ink and ground-truth error, baseline -> current."""
    if "error" in cur:
        return
    def q(summary):
        if summary is None:
            return None
        out = {"res": summary.get("residual_fraction")}
        if gt is not None:
            # stroke width of the true curves in px = tolerance for "covered"
            out.update(gt_metrics(summary["curves_px"], gt[0], stroke_px=gt[1]))
        return out
    b, c = q(base), q(cur)
    def pair(key, fmt):
        cv = c.get(key)
        bv = b.get(key) if b else None
        f = lambda v: "-" if v is None else fmt(v)
        return f(cv) if b is None else f"{f(bv)}->{f(cv)}"
    parts = [f"residual {pair('res', lambda v: f'{100 * v:.2f}%')}"]
    if gt is not None:
        parts.append(f"GT err mean {pair('mean', lambda v: f'{v:.2f}')}px "
                     f"p95 {pair('p95', lambda v: f'{v:.1f}')}px "
                     f"coverage {pair('coverage', lambda v: f'{100 * v:.1f}%')}; "
                     f"resolvable p95 {pair('res_p95', lambda v: f'{v:.1f}')}px "
                     f"hit {pair('res_cov', lambda v: f'{100 * v:.1f}%')}")
    twin = VECTOR_TWIN.get(name)
    if twin and (CUR_DIR / twin / "summary.json").exists():
        ref = json.loads((CUR_DIR / twin / "summary.json").read_text(encoding="utf-8"))
        ok, tot, wrong = label_check(cur["curves_px"], ref["curves_px"])
        parts.append(f"labels vs vector {ok}/{tot}" + (f" WRONG: {', '.join(wrong)}" if wrong else ""))
    print(f"{'':<20}   quality  {'; '.join(parts)}")


def label_check(curves: dict, ref: dict) -> tuple[int, int, list[str]]:
    """Each reference (vector, exact) curve -> nearest extracted curve; count
    matching names."""
    names = list(curves)
    trees = [cKDTree(_densify(curves[n])) for n in names]
    ok, wrong = 0, []
    for rn, rc in ref.items():
        d = [t.query(_densify(rc))[0].mean() for t in trees]
        got = names[int(np.argmin(d))]
        if got == rn:
            ok += 1
        else:
            wrong.append(f"{rn}->{got}")
    return ok, len(ref), wrong


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true",
                    help="Store this run as the new baseline")
    ap.add_argument("--cases", nargs="*", help="Substring filter on case names")
    ap.add_argument("--jobs", type=int, default=len(CASES))
    ap.add_argument("--compare-only", action="store_true",
                    help="Do not run the pipeline; compare the last run in current/")
    args = ap.parse_args()

    names = [n for n in CASES if not args.cases or any(f.lower() in n.lower() for f in args.cases)]
    gts = {n: ground_truth(n) for n in names}
    CUR_DIR.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        results = {}
        for n in names:
            f = CUR_DIR / n / "summary.json"
            results[n] = (json.loads(f.read_text(encoding="utf-8")) if f.exists()
                          else {"error": "no current run"})
    else:
        with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(names)))) as ex:
            results = dict(zip(names, ex.map(_run_case, names)))

    failed = False
    print()
    print(f"{'case':<20} {'status':<8} {'time b->c [s]':<15} {'curves b->c':<12} details")
    print("-" * 90)
    for name in names:
        cur = results[name]
        bpath = BASE_DIR / f"{name}.json"
        if args.update_baseline:
            if "error" not in cur:
                BASE_DIR.mkdir(parents=True, exist_ok=True)
                bpath.write_text(json.dumps(cur, indent=1), encoding="utf-8")
            st = "ERROR" if "error" in cur else "stored"
            print(f"{name:<20} {st:<8} {cur.get('time_s', '-')!s:<15} "
                  f"{len(cur.get('curves_px', {})):<12} {cur.get('error', '')}")
            _print_quality(name, None, cur, gts.get(name))
            continue
        if not bpath.exists():
            print(f"{name:<20} NO-BASE  (run with --update-baseline)")
            failed = True
            continue
        base = json.loads(bpath.read_text(encoding="utf-8"))
        res = compare(base, cur)
        failed |= res["status"] in ("CHANGED", "ERROR")
        t = f"{base['time_s']:.0f}->{cur.get('time_s', 0):.0f}"
        n = f"{len(base['curves_px'])}->{len(cur.get('curves_px', {}))}"
        print(f"{name:<20} {res['status']:<8} {t:<15} {n:<12} {'; '.join(res['notes'])}")
        for label, st, mean, haus in res["curves"]:
            if st != "same":
                print(f"{'':<20}   {st:<8} {label:<40} mean={_fmt(mean)}px max={_fmt(haus)}px")
        _print_quality(name, base, cur, gts.get(name))
    print()
    print(f"Artefacts: {CUR_DIR}")
    return 1 if failed and not args.update_baseline else 0


if __name__ == "__main__":
    sys.exit(main())
