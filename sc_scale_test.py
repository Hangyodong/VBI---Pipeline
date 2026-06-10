#!/usr/bin/env python
"""SC scaling vs FC-fit ceiling — does log1p over-compress the connectome?

The joint ceiling (joint_opt.py) landed at mean real-FC corr ~0.22 (<0.30) even
with the strongest knobs (c_ee, c_ei) searched jointly under delays. VERDICT:
simulator misspecification, not parameters (docs/03). Leading suspect: the SC
weight scaling. ``data_loader._scale_weights`` applies ``log1p(w + 0.5)`` before
max-norm, which crushes the raw count dynamic range (e.g. 1000x -> ~4x) and may
homogenize coupling -> weak FC spatial structure -> sim FC std stuck at 0.088.

This script runs, for each scaling MODE (env ``VBI_SC_SCALE``):
  1. SC dynamic-range stats (CPU) — how much the mode compresses the connectome.
  2. joint_opt ceiling + per-param influence (the joint-space sensitivity) —
     reuses joint_opt.run_subject / _run_group verbatim.
Then prints a side-by-side comparison so you can see whether dropping log1p
raises the ceiling past 0.30.

Usage
-----
    # single subject, compare log1p vs maxnorm (fast, ~2-3 min/mode @ samples 1000)
    python sc_scale_test.py --subject sub-421529 --params g_e,c_ei,c_ee --samples 1000

    # whole ctr+MPTP group, all three modes (slow — one joint batch per subject)
    python sc_scale_test.py --all --modes log1p,maxnorm,sqrt \
        --params g_e,c_ei,c_ee --samples 1000 --workers 8 --gpus 0,1

Cost: identical to joint_opt per mode (delays ON ~125s/subject @ samples 1000;
multiply by #modes). Use --samples 1000 + --gpus 0,1 to keep it cheap.
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np


# --------------------------------------------------------------------------
# SC dynamic-range stats (CPU, cheap) — quantify how a mode compresses weights
# --------------------------------------------------------------------------

def sc_stats(data):
    """Aggregate nonzero-weight distribution stats across the group's SC."""
    vals = []
    for d in data.values():
        w = d["sc"]
        nz = w[w > 0]
        if nz.size:
            vals.append(nz)
    if not vals:
        return {}
    nz = np.concatenate(vals)
    p = np.percentile(nz, [50, 90, 99, 100])
    med = p[0] if p[0] > 0 else np.nan
    return {
        "n_edges": int(nz.size),
        "median": float(p[0]),
        "p90": float(p[1]),
        "p99": float(p[2]),
        "max": float(p[3]),
        "max_over_median": float(p[3] / med) if med == med else np.nan,
        "cv": float(nz.std() / nz.mean()) if nz.mean() > 0 else np.nan,
    }


# --------------------------------------------------------------------------
# One mode: load (with scaling), SC stats, joint ceiling
# --------------------------------------------------------------------------

def run_mode(mode, args):
    """Set scaling mode, reload SC, run joint_opt ceiling. Returns summary."""
    os.environ["VBI_SC_SCALE"] = mode  # read by data_loader._scale_weights
    import importlib
    import data_loader
    importlib.reload(data_loader)      # ensure fresh load under new env
    import joint_opt

    subjects, data = joint_opt.load_group(args.subject if not args.all else None)
    stats = sc_stats(data)

    # joint_opt expects an args-ish namespace; per-mode output prefix.
    jargs = SimpleNamespace(
        params=args.params, samples=args.samples, seed=args.seed,
        topk=args.topk, workers=args.workers, gpus=args.gpus,
        heartbeat=args.heartbeat, all=args.all,
        out=f"{args.out}_{mode}",
    )

    print("\n" + "#" * 70)
    print(f"#  MODE = {mode}   subjects={len(subjects)}   samples={args.samples}")
    print("#" * 70)
    if stats:
        print(f"  SC nonzero weights: n={stats['n_edges']}  "
              f"median={stats['median']:.3g}  p99={stats['p99']:.3g}  "
              f"max={stats['max']:.3g}")
        print(f"  dynamic range max/median={stats['max_over_median']:.1f}x  "
              f"CV={stats['cv']:.3f}   "
              f"(higher = less compressed = more heterogeneous coupling)")

    t0 = time.time()
    if args.all:
        results = joint_opt._run_group(subjects, data, jargs)
    else:
        results = [joint_opt.run_subject(subjects[0], data[subjects[0]],
                                         jargs, quiet=False)]
    el = time.time() - t0

    ok = [r for r in results if "error" not in r]
    corrs = np.array([r["best_corr"] for r in ok]) if ok else np.array([])
    keys = ok[0]["keys"] if ok else []
    infl = {k: float(np.nanmean([r["influence"][k] for r in ok
                                 if np.isfinite(r["influence"][k])]))
            for k in keys} if ok else {}
    return {
        "mode": mode, "stats": stats, "corrs": corrs, "keys": keys,
        "influence": infl, "elapsed": el, "n_ok": len(ok), "n": len(results),
    }


# --------------------------------------------------------------------------
# Comparison table across modes
# --------------------------------------------------------------------------

def compare(summaries):
    print("\n" + "=" * 78)
    print("  SC SCALING COMPARISON — ceiling vs compression")
    print("=" * 78)
    keys = next((s["keys"] for s in summaries if s["keys"]), [])
    head = (f"  {'mode':9s} {'ceiling':>9s} {'median':>8s} {'max':>7s}  "
            f"{'range(x)':>9s} {'CV':>6s}  "
            + "  ".join(f"infl:{k:<5s}" for k in keys))
    print(head)
    print("  " + "-" * (len(head) - 2))
    base = None
    for s in summaries:
        c = s["corrs"]
        if c.size == 0:
            print(f"  {s['mode']:9s}   (no successful subjects)")
            continue
        mean, med, mx = c.mean(), np.median(c), c.max()
        if base is None:
            base = mean
        st = s["stats"]
        rng = st.get("max_over_median", float("nan"))
        cv = st.get("cv", float("nan"))
        infl_s = "  ".join(f"{s['influence'].get(k, float('nan')):10.3f}"
                           for k in keys)
        print(f"  {s['mode']:9s} {mean:+9.4f} {med:+8.4f} {mx:+7.4f}  "
              f"{rng:9.1f} {cv:6.3f}  {infl_s}")

    print("\n  VERDICT")
    print("  " + "-" * 60)
    best = max((s for s in summaries if s["corrs"].size),
               key=lambda s: s["corrs"].mean(), default=None)
    if best is None:
        print("  no results.")
        return
    bm = best["corrs"].mean()
    print(f"  best mode = {best['mode']}  (mean ceiling {bm:+.4f})")
    if bm >= 0.5:
        print("  => Ceiling OK (>=0.50): SC scaling was the main lever. Switch")
        print("     data_loader default to this mode, re-select params, re-infer.")
    elif bm >= 0.3:
        print("  => Ceiling MODERATE (>=0.30): scaling helps, structural gap")
        print("     remains. Combine with delays/WC operating-point checks.")
    else:
        print("  => Still LOW (<0.30): SC compression is NOT the bottleneck.")
        print("     Move to WC operating point (damped/fixed-point regime?) and")
        print("     BOLD/noise. log1p was not the culprit.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="log1p,maxnorm",
                    help="comma list of SC scaling modes (log1p|maxnorm|sqrt)")
    ap.add_argument("--subject", default=None,
                    help="single subject (ignored with --all)")
    ap.add_argument("--all", action="store_true",
                    help="run every subject in config.GROUP_FILTER")
    ap.add_argument("--params", default="g_e,c_ei,c_ee",
                    help="joint search params (passed to joint_opt)")
    ap.add_argument("--samples", type=int, default=1000,
                    help="random joint draws per subject (one GPU batch)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heartbeat", type=float, default=10.0)
    ap.add_argument("--out", default="output_mouse_mptp/sc_scale")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid = {"log1p", "maxnorm", "sqrt", "raw", "none"}
    for m in modes:
        if m not in valid:
            raise SystemExit(f"unknown mode {m!r}; use {sorted(valid)}")

    print(f"  SC scaling test — modes={modes}  params={args.params}  "
          f"samples={args.samples}  "
          f"{'GROUP' if args.all else (args.subject or 'first subject')}")

    summaries = []
    t0 = time.time()
    for m in modes:
        summaries.append(run_mode(m, args))
        print(f"  [mode {m} done, {time.time() - t0:.0f}s total]")
    compare(summaries)


if __name__ == "__main__":
    main()
