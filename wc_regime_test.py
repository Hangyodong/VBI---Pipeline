#!/usr/bin/env python
"""WC operating-point / regime diagnostic — why is sim-FC std stuck at 0.088?

SC scaling is exonerated (sc_scale_test.py: 100x more dynamic range moved the
ceiling +0.014). The joint ceiling stays <0.30 with every strong knob searched.
Remaining suspect: the WCVBI operating point is a deeply-damped low fixed point
(rest sigmoid arg ~ P_E - theta_E = -1.5 -> E* ~ 0.12) driven by weak additive
noise (sigma 0.02). Near-independent regional fluctuations -> FC ~ chance ->
std ~ 0.088. Rich resting-state FC needs the model near a bifurcation so noise
correlates ACROSS regions through the coupling.

This script does NOT chase real-FC fit. It asks the prior question:

  Does the coupling shape FC AT ALL, and can ANY regime raise sim-FC std off
  the chance floor?

For a grid over (G=g_e, c_ee) x sigma it reports, per sim:
  - simFC_std      : std of upper-tri sim FC          (real ~0.17-0.23; floor ~0.088)
  - FC~SC (edges)  : corr(sim FC, SC) over SC>0 edges (structure-function coupling)
  - FC~SC (all)    : corr(sim FC, SC) over all pairs
  - FC~real        : masked corr to this subject's real FC
  - bold_std       : mean across regions of each region's temporal BOLD std (alive?)

Read it as:
  * FC~SC ~ 0 everywhere  -> coupling does not shape FC: regime dead or adapter
                            bug. No parameter fixes this; the model is the issue.
  * simFC_std rises with G/sigma toward ~0.2 -> regime IS the lever: push the
                            operating point (sigma, G range, theta/alpha) there.
  * bold_std ~ 0          -> neural signal is flat: noise too weak / fixed point
                            too stable; raise sigma or move the operating point.

Usage
-----
    python wc_regime_test.py --subject sub-421529
    python wc_regime_test.py --subject sub-421529 \
        --g_e 0.5,1,2 --c_ee 8,16,24 --sigma 0.02,0.05,0.1,0.2

Cost: one GPU call per sigma value (the g_e x c_ee grid rides in one batch).
delays ON ~ per-call cost like a small joint_opt batch. 4 sigmas ~ a few min.
"""
import argparse
import time

import numpy as np


def _corr(a, b, mask=None):
    """Pearson corr of two flat vectors over an optional boolean mask."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None, help="subject id (default: first)")
    ap.add_argument("--g_e", default="0.5,1.0,2.0",
                    help="comma G(=g_e) grid")
    ap.add_argument("--c_ee", default="8.0,16.0,24.0",
                    help="comma c_ee grid")
    ap.add_argument("--sigma", default="0.02,0.05,0.1,0.2",
                    help="comma noise sigma (noise_amp) grid; one GPU call each")
    ap.add_argument("--c_ei", type=float, default=12.0,
                    help="fixed c_ei for the sweep (baseline)")
    ap.add_argument("--out", default="output_mouse_mptp/wc_regime")
    args = ap.parse_args()

    import config
    import joint_opt
    from cuBNM.simulate import simulate_gpu_batch
    from simulator import compute_fc

    g_es = [float(x) for x in args.g_e.split(",") if x]
    c_ees = [float(x) for x in args.c_ee.split(",") if x]
    sigmas = [float(x) for x in args.sigma.split(",") if x]

    subjects, data = joint_opt.load_group(args.subject)
    sid = subjects[0]
    d = data[sid]
    weights, delays, fc_real = d["sc"], d["delays"], d["fc"]
    N = fc_real.shape[0]
    iu = np.triu_indices(N, k=1)
    real_vec = fc_real[iu]
    nan_vec = (d["fc_nan"][iu] if "fc_nan" in d
               else np.zeros(iu[0].shape, dtype=bool))
    real_mask = ~nan_vec
    sc_vec = weights[iu]
    sc_edge = sc_vec > 0
    real_std = float(real_vec[real_mask].std())

    # theta batch: c_ei fixed, (g_e, c_ee) grid. param order must name columns.
    pn = ["g_e", "c_ei", "c_ee"]
    combos = [(g, c) for g in g_es for c in c_ees]
    theta = np.array([[g, args.c_ei, c] for (g, c) in combos], dtype=np.float64)

    print("=" * 92)
    print(f"  WC regime diagnostic — subject {sid}   N={N}")
    print(f"  real-FC std (valid edges) = {real_std:.4f}   "
          f"(chance floor ~0.088, target ~0.17-0.23)")
    print(f"  grid: g_e={g_es}  c_ee={c_ees}  c_ei={args.c_ei}  sigma={sigmas}")
    print(f"  SC>0 edges: {int(sc_edge.sum())}/{len(sc_vec)}")
    print("=" * 92)
    hdr = (f"  {'sigma':>6s} {'g_e':>5s} {'c_ee':>6s} | "
           f"{'simFC_std':>9s} {'FC~SC(edge)':>11s} {'FC~SC(all)':>10s} "
           f"{'FC~real':>8s} {'bold_std':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for sg in sigmas:
        t0 = time.time()
        bolds = simulate_gpu_batch(
            weights, theta, param_names=pn, delays=delays, apply_bw=True,
            fixed_overrides={"noise_amp": sg},
        )
        dt = time.time() - t0
        for (g, c), b in zip(combos, bolds):
            b = np.asarray(b, float)              # (T, N)
            bold_std = float(np.mean(b.std(axis=0)))
            fc = compute_fc(b)
            v = fc[iu]
            simfc_std = float(v[np.isfinite(v)].std()) if np.isfinite(v).any() \
                else np.nan
            fcsc_edge = _corr(v, sc_vec, sc_edge)
            fcsc_all = _corr(v, sc_vec)
            fcreal = _corr(v, real_vec, real_mask)
            rows.append(dict(sigma=sg, g_e=g, c_ee=c, simfc_std=simfc_std,
                             fcsc_edge=fcsc_edge, fcsc_all=fcsc_all,
                             fcreal=fcreal, bold_std=bold_std))
            print(f"  {sg:6.3f} {g:5.2f} {c:6.1f} | {simfc_std:9.4f} "
                  f"{fcsc_edge:11.4f} {fcsc_all:10.4f} {fcreal:8.4f} "
                  f"{bold_std:9.5f}")
        print(f"  -- sigma={sg} batch done ({dt:.0f}s) --")

    # ---- save + verdict --------------------------------------------------
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(f"{args.out}_{sid}.npz", subject=sid, real_std=real_std, **arr)
    print(f"\n  saved: {args.out}_{sid}.npz")

    simstd = arr["simfc_std"]
    fcsc = arr["fcsc_edge"]
    print("\n" + "=" * 92)
    print("  VERDICT")
    print("=" * 92)
    print(f"  sim-FC std range : {np.nanmin(simstd):.4f} .. {np.nanmax(simstd):.4f}"
          f"   (real {real_std:.3f})")
    print(f"  FC~SC(edge) range: {np.nanmin(fcsc):+.4f} .. {np.nanmax(fcsc):+.4f}")
    best_sc = np.nanmax(np.abs(fcsc))
    near_real = np.nanmax(simstd) >= 0.5 * real_std
    if best_sc < 0.1:
        print("  => Coupling does NOT shape FC (|FC~SC| < 0.1 everywhere).")
        print("     Structure-function link is broken: operating point is a dead")
        print("     fixed point (or an adapter/coupling bug). No params fix this —")
        print("     move the operating point (sigma, theta_E/alpha_E, G range) or")
        print("     audit how G*globalinput enters the kernel.")
    elif not near_real:
        print("  => Coupling shapes FC but sim-FC std stays near the chance floor")
        print("     for all regimes tested. Push harder: higher sigma / G, lower")
        print("     theta_E (less damped fixed point), or widen the grid.")
    else:
        print("  => Some regime reaches real-like FC std AND structure-function")
        print("     coupling. The operating point IS the lever — re-center the")
        print("     priors there and re-test the ceiling / re-infer.")


if __name__ == "__main__":
    main()
