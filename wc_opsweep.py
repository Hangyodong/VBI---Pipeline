#!/usr/bin/env python
"""Generic WC operating-point sweep — 2 grid axes x sigma, structure-function metrics.

Follow-up to wc_regime_test.py, which found: the coupling DOES shape FC
(FC~SC up to +0.35) and c_ee must be LOW (8; 16/24 saturate the sigmoid and
kill structure), but sim-FC std is stuck ~0.05 (real ~0.18) in every regime
tested. Diagnosis: the operating point is too damped — at rest the sigmoid
argument is P_E - theta_E = -1.5, deep in the low-gain tail, so noise produces
weak uncorrelated fluctuations. Pushing the fixed point toward the steep
(high-gain, near-critical) part of the sigmoid should amplify correlated
fluctuations and raise sim-FC std toward the real ~0.18.

This script sweeps ANY two WC params on an in-batch grid (one GPU call per
sigma) and reports the same metrics, so you can target the operating point:

    # push the fixed point up: lower theta_E / raise P_E, c_ee held LOW
    python wc_opsweep.py --subject sub-421529 \
        --px theta_e --pxv 0.5,1.0,1.5,2.0 \
        --py P        --pyv 0.5,1.5,3.0 \
        --fixed c_ee=8,g_e=1.0,c_ei=12 --sigma 0.1,0.2

    # alpha_E gain vs theta_E
    python wc_opsweep.py --subject sub-421529 --px alpha_e --pxv 1,2,4 \
        --py theta_e --pyv 0.5,1,1.5,2 --fixed c_ee=8,g_e=1 --sigma 0.1

Any WC param name routes per-sim: g_e->G, c_ei/g_i inferred, the rest via
cuBNM.runner_vbi._FIXED_TO_VBI (theta_e->theta_E, P->P_E, c_ee, alpha_e, ...).

Metrics (per sim): simFC_std, FC~SC over SC>0 edges (structure-function),
FC~SC all pairs, FC~real (masked), bold_std (mean regional temporal std).
Goal regime: simFC_std climbing toward real_std WHILE FC~SC stays high.
"""
import argparse
import time

import numpy as np


def _corr(a, b, mask=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _parse_fixed(s):
    """'c_ee=8,g_e=1.0' -> {'c_ee':8.0,'g_e':1.0}."""
    out = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None)
    ap.add_argument("--px", required=True, help="grid axis 1 param name")
    ap.add_argument("--pxv", required=True, help="comma values for px")
    ap.add_argument("--py", default=None, help="grid axis 2 param name (optional)")
    ap.add_argument("--pyv", default=None, help="comma values for py")
    ap.add_argument("--fixed", default="",
                    help="held params 'name=val,...' (any WC param; per-sim)")
    ap.add_argument("--sigma", default="0.1,0.2",
                    help="noise sigma grid; one GPU call each")
    ap.add_argument("--out", default="output_mouse_mptp/wc_opsweep")
    args = ap.parse_args()

    import joint_opt
    from cuBNM.simulate import simulate_gpu_batch
    from simulator import compute_fc

    pxv = [float(x) for x in args.pxv.split(",") if x]
    pyv = ([float(x) for x in args.pyv.split(",") if x]
           if args.py and args.pyv else [None])
    sigmas = [float(x) for x in args.sigma.split(",") if x]
    fixed = _parse_fixed(args.fixed)

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

    # theta columns: [px, (py), fixed keys]. One row per grid cell.
    fixed_keys = [k for k in fixed if k not in (args.px, args.py)]
    pn = [args.px] + ([args.py] if args.py else []) + fixed_keys
    combos = [(x, y) for x in pxv for y in pyv]
    rows_theta = []
    for x, y in combos:
        r = [x] + ([y] if args.py else []) + [fixed[k] for k in fixed_keys]
        rows_theta.append(r)
    theta = np.array(rows_theta, dtype=np.float64)

    print("=" * 96)
    print(f"  WC operating-point sweep — subject {sid}   N={N}")
    print(f"  real-FC std = {real_std:.4f}   (floor ~0.088, target ~0.17-0.23)")
    print(f"  px={args.px} {pxv}   py={args.py} {pyv if args.py else '-'}")
    print(f"  fixed={fixed}   sigma={sigmas}   theta cols={pn}")
    print("=" * 96)
    hdr = (f"  {'sigma':>6s} {args.px:>8s} "
           f"{(args.py if args.py else ''):>8s} | "
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
        for (x, y), b in zip(combos, bolds):
            b = np.asarray(b, float)
            bold_std = float(np.mean(b.std(axis=0)))
            v = compute_fc(b)[iu]
            simfc_std = (float(v[np.isfinite(v)].std())
                         if np.isfinite(v).any() else np.nan)
            rec = dict(sigma=sg, px=x, py=(y if args.py else np.nan),
                       simfc_std=simfc_std,
                       fcsc_edge=_corr(v, sc_vec, sc_edge),
                       fcsc_all=_corr(v, sc_vec),
                       fcreal=_corr(v, real_vec, real_mask),
                       bold_std=bold_std)
            rows.append(rec)
            ys = f"{y:8.3f}" if args.py else f"{'':8s}"
            print(f"  {sg:6.3f} {x:8.3f} {ys} | {rec['simfc_std']:9.4f} "
                  f"{rec['fcsc_edge']:11.4f} {rec['fcsc_all']:10.4f} "
                  f"{rec['fcreal']:8.4f} {rec['bold_std']:9.5f}")
        print(f"  -- sigma={sg} ({dt:.0f}s) --")

    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(f"{args.out}_{sid}.npz", subject=sid, real_std=real_std,
             px_name=args.px, py_name=(args.py or ""), **arr)
    print(f"\n  saved: {args.out}_{sid}.npz")

    simstd, fcsc, fcreal = arr["simfc_std"], arr["fcsc_edge"], arr["fcreal"]
    i_std = int(np.nanargmax(simstd))
    i_real = int(np.nanargmax(fcreal))
    print("\n" + "=" * 96)
    print("  VERDICT")
    print("=" * 96)
    print(f"  sim-FC std : {np.nanmin(simstd):.4f} .. {np.nanmax(simstd):.4f}"
          f"   (real {real_std:.3f}, {100*np.nanmax(simstd)/real_std:.0f}% of real)")
    print(f"  max sim-FC std at: sigma={rows[i_std]['sigma']} "
          f"{args.px}={rows[i_std]['px']:.3f}"
          + (f" {args.py}={rows[i_std]['py']:.3f}" if args.py else "")
          + f"  (FC~SC={rows[i_std]['fcsc_edge']:+.3f}, "
            f"FC~real={rows[i_std]['fcreal']:+.3f})")
    print(f"  max FC~real at   : sigma={rows[i_real]['sigma']} "
          f"{args.px}={rows[i_real]['px']:.3f}"
          + (f" {args.py}={rows[i_real]['py']:.3f}" if args.py else "")
          + f"  (FC~real={rows[i_real]['fcreal']:+.3f}, "
            f"std={rows[i_real]['simfc_std']:.4f})")
    top = np.nanmax(simstd)
    if top >= 0.8 * real_std:
        print("  => Operating point FOUND a real-like FC-std regime. Re-center the")
        print("     priors / WC_FIXED here, re-test joint ceiling, re-infer.")
    elif top >= 0.5 * real_std:
        print("  => sim-FC std climbing toward real — right direction. Push the")
        print("     best axis further (extend the grid past its current edge).")
    else:
        print("  => sim-FC std still <50% of real. This axis is not the lever yet —")
        print("     try theta_E/P_E/alpha_E (gain), or suspect the BOLD/obs model")
        print("     (Balloon-Windkessel washing out variance, TR, duration).")


if __name__ == "__main__":
    main()
