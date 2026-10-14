"""
Regression check over the real datasheet plots in tests/.

Runs the full pipeline (CLI defaults: trace + grid suppression, with debug
files) on every case in parallel and compares the result with a frozen
baseline:

* plot canvas, axis calibration (scale + data range at the canvas edges),
* CSV column header, curve set (names),
* per curve: geometry in image pixels — symmetric mean distance and
  Hausdorff (max) distance between baseline and current polyline,
* run time.

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
# up by the pipeline automatically, exactly as in normal use.
CASES = {
    "BAT46GW_fig3": "BAT46GW - Figure 3 - Typical Junction Capacitance.png",
    "CMZ54xxB": "CMZ54xxB Zender diode - V_Z-I_R.png",
    "USFF_1206": "USFF_1206 - with axis labels.png",
    "MUH1PB_fig3": "MUH1PB,PC,PD - Figure 3 - Typical Instantaneous Forward Characteristics.png",
    "MUH1PB_fig4": "MUH1PB,PC,PD - Figure 4 - Typical Reverse Characteristics.png",
}

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
            method="trace", has_grid=True,
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
    pairs = [(n, n) for n in bc if n in cc]
    rest_b = [n for n in bc if n not in cc]
    rest_c = [n for n in cc if n not in bc]
    if rest_b and rest_c:                     # renamed: match by geometry
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
        if b != c:
            st = "RENAMED" if st != "CHANGED" else "CHANGED"
        rows.append((label, st, mean, haus))
    rows += [(n, "MISSING", None, None) for n in rest_b]
    rows += [(n, "NEW", None, None) for n in rest_c]

    changed = bool(notes) or any(r[1] not in ("same", "close") for r in rows)
    minor = any(r[1] == "close" for r in rows)
    status = "CHANGED" if changed else ("close" if minor else "same")
    return {"status": status, "notes": notes, "curves": rows}


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true",
                    help="Store this run as the new baseline")
    ap.add_argument("--cases", nargs="*", help="Substring filter on case names")
    ap.add_argument("--jobs", type=int, default=len(CASES))
    args = ap.parse_args()

    names = [n for n in CASES if not args.cases or any(f.lower() in n.lower() for f in args.cases)]
    CUR_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, len(names)))) as ex:
        results = dict(zip(names, ex.map(_run_case, names)))

    failed = False
    print()
    print(f"{'case':<14} {'status':<8} {'time b->c [s]':<15} {'curves b->c':<12} details")
    print("-" * 90)
    for name in names:
        cur = results[name]
        bpath = BASE_DIR / f"{name}.json"
        if args.update_baseline:
            if "error" not in cur:
                BASE_DIR.mkdir(parents=True, exist_ok=True)
                bpath.write_text(json.dumps(cur, indent=1), encoding="utf-8")
            st = "ERROR" if "error" in cur else "stored"
            print(f"{name:<14} {st:<8} {cur.get('time_s', '-')!s:<15} "
                  f"{len(cur.get('curves_px', {})):<12} {cur.get('error', '')}")
            continue
        if not bpath.exists():
            print(f"{name:<14} NO-BASE  (run with --update-baseline)")
            failed = True
            continue
        base = json.loads(bpath.read_text(encoding="utf-8"))
        res = compare(base, cur)
        failed |= res["status"] in ("CHANGED", "ERROR")
        t = f"{base['time_s']:.0f}->{cur.get('time_s', 0):.0f}"
        n = f"{len(base['curves_px'])}->{len(cur.get('curves_px', {}))}"
        print(f"{name:<14} {res['status']:<8} {t:<15} {n:<12} {'; '.join(res['notes'])}")
        for label, st, mean, haus in res["curves"]:
            if st != "same":
                print(f"{'':<14}   {st:<8} {label:<40} mean={_fmt(mean)}px max={_fmt(haus)}px")
    print()
    print(f"Artefacts: {CUR_DIR}")
    return 1 if failed and not args.update_baseline else 0


if __name__ == "__main__":
    sys.exit(main())
