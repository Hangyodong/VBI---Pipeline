#!/usr/bin/env python
"""Neural-FC vs BOLD-FC — is the Balloon-Windkessel observation washing out FC?

sim-FC std is capped ~0.05-0.08 across the ENTIRE neural parameter space
(g_e, c_ee, theta_E, alpha_E, P, sigma, tau — wc_regime_test / wc_opsweep),
while real-FC std is ~0.18 and bold_std swings 100x with no effect on FC std.
That pattern says the cap is downstream of the dynamics. This script isolates
it: for the same parameters and seed it computes FC TWO ways —

  1. on the raw neural E series   (grp.sim_states['E'], states_ts=True)
  2. on the cuBNM BW BOLD          (run_cubnm_vbi_batch hrf='bw')

and compares upper-tri FC std + structure-function (FC~SC).

Read it as:
  * neuralFC_std >> boldFC_std  -> Balloon-Windkessel is washing out the
        cross-region correlation. Fix the observation model (BW params / HRF /
        downsampling), not the dynamics.
  * neuralFC_std ~ boldFC_std ~ 0.08  -> the dynamics themselves are weakly
        correlated even before BOLD. The noise-driven fixed-point WC never
        builds a strong shared mode; need a near-critical / oscillatory regime
        or a different neural model.

Usage
-----
    python wc_neural_fc.py --subject sub-421529
    python wc_neural_fc.py --subject sub-421529 \
        --fixed c_ee=8,g_e=1,c_ei=12,alpha_e=4,theta_e=0.5 --sigma 0.1
"""
import argparse
import time

import numpy as np


def _corr(a, b, mask=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _fc_std_sc(ts, iu, sc_vec, sc_edge, real_vec, real_mask):
    """FC from a (T, N) series: upper-tri std + FC~SC(edge/all) + FC~real."""
    ts = np.asarray(ts, float)
    if ts.ndim != 2 or ts.shape[0] < 3:
        return dict(std=np.nan, sc_edge=np.nan, sc_all=np.nan, real=np.nan)
    fc = np.corrcoef(ts.T)                      # (N, N)
    v = fc[iu]
    std = float(v[np.isfinite(v)].std()) if np.isfinite(v).any() else np.nan
    return dict(std=std,
                sc_edge=_corr(v, sc_vec, sc_edge),
                sc_all=_corr(v, sc_vec),
                real=_corr(v, real_vec, real_mask))


def _build_neural_grp(weights, theta, pn, fixed, delays, sim_seed):
    """Mirror runner_vbi.run_cubnm_vbi_batch but keep the neural E series."""
    import config
    from cuBNM.runner_vbi import _import_wcvbi, build_param_lists
    WCVBISimGroup = _import_wcvbi()

    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    n_nodes = weights.shape[0]
    theta = np.asarray(theta, dtype=np.float64)
    n_sims = theta.shape[0]
    neural_dt_ms = float(config.DT) * float(config.DECIMATE)

    use_delay = delays is not None and np.any(np.asarray(delays) > 0) \
        and bool(getattr(config, "USE_DELAYS", False))
    sc_dist = (np.ascontiguousarray(np.asarray(delays, dtype=np.float64))
               if use_delay else None)

    grp = WCVBISimGroup(
        duration=float(config.T_END) / 1000.0,
        TR=float(config.TR_SEC),
        sc=weights,
        sc_dist=sc_dist,
        dt=str(float(config.DT)),
        bold_remove_s=0.0,                 # keep full series; we trim below
        do_fc=False, do_fcd=False, gof_terms=[],
        force_gpu=True, sim_seed=int(sim_seed),
        states_ts=True,
        states_sampling=neural_dt_ms / 1000.0,
    )
    grp.N = n_sims
    for k, v in build_param_lists(theta, list(pn), n_nodes, fixed).items():
        grp.param_lists[k] = v
    if use_delay:
        grp.param_lists["v"] = np.full(n_sims, 1.0, dtype=np.float64)
    grp.run()
    E = np.asarray(grp.sim_states["E"], dtype=np.float32)   # (S, T, N)
    # trim the transient (burn_in) at the neural sampling rate
    t_cut_steps = int(float(config.T_CUT) / neural_dt_ms)
    if E.shape[1] > t_cut_steps:
        E = E[:, t_cut_steps:, :]
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None)
    ap.add_argument("--fixed", default="c_ee=8,g_e=1,c_ei=12,alpha_e=4,theta_e=0.5",
                    help="param set 'name=val,...' (best operating point)")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", default=None,
                    help="neural-only criticality probe: 'g_e=1,2,5,10,20'. "
                         "Loops one param to large values, reports neural FC "
                         "std + structure-function per value (no BOLD). Tests "
                         "whether stronger coupling reaches the critical regime "
                         "(std climbing toward real) or never does (switch model).")
    ap.add_argument("--out", default="output_mouse_mptp/wc_neural_fc")
    args = ap.parse_args()

    import joint_opt
    from cuBNM.simulate import simulate_gpu_batch

    fixed = {}
    for kv in args.fixed.split(","):
        if kv.strip():
            k, v = kv.split("="); fixed[k.strip()] = float(v)

    subjects, data = joint_opt.load_group(args.subject)
    sid = subjects[0]; d = data[sid]
    weights, delays, fc_real = d["sc"], d["delays"], d["fc"]
    N = fc_real.shape[0]
    iu = np.triu_indices(N, k=1)
    real_vec = fc_real[iu]
    nan_vec = (d["fc_nan"][iu] if "fc_nan" in d
               else np.zeros(iu[0].shape, dtype=bool))
    real_mask = ~nan_vec
    sc_vec = weights[iu]; sc_edge = sc_vec > 0
    real_std = float(real_vec[real_mask].std())

    sim_fixed = {"noise_amp": args.sigma, "seed": args.seed}

    # --- neural-only criticality sweep -----------------------------------
    if args.sweep:
        sname, svals = args.sweep.split("=")
        sname = sname.strip()
        svals = [float(x) for x in svals.split(",") if x]
        print("=" * 84)
        print(f"  Neural criticality probe — subject {sid}   sweep {sname}")
        print(f"  base={fixed}  sigma={args.sigma}  real-FC std={real_std:.4f}")
        print("=" * 84)
        print(f"  {sname:>8s} | {'neuralFC_std':>12s} {'FC~SC(edge)':>11s} "
              f"{'FC~real':>8s}  {'%of real std':>12s}")
        print("  " + "-" * 60)
        srows = []
        for val in svals:
            fx = dict(fixed); fx[sname] = val
            pn = list(fx.keys())
            theta = np.array([[fx[k] for k in pn]], dtype=np.float64)
            t0 = time.time()
            try:
                E = _build_neural_grp(weights, theta, pn, sim_fixed,
                                      delays, args.seed)[0]
                m = _fc_std_sc(E, iu, sc_vec, sc_edge, real_vec, real_mask)
            except Exception as e:  # noqa: BLE001
                print(f"  {val:8.2f} | ERROR {type(e).__name__}: {e}")
                continue
            srows.append((val, m))
            print(f"  {val:8.2f} | {m['std']:12.4f} {m['sc_edge']:11.4f} "
                  f"{m['real']:8.4f}  {100*m['std']/real_std:11.0f}%  "
                  f"({time.time()-t0:.0f}s)")
        if srows:
            stds = np.array([m["std"] for _, m in srows])
            print("\n  VERDICT")
            print("  " + "-" * 60)
            top = float(np.nanmax(stds))
            print(f"  max neural FC std = {top:.4f} = {100*top/real_std:.0f}% "
                  f"of real ({real_std:.4f})")
            if top >= 0.7 * real_std:
                print("  => Stronger coupling REACHES a real-like std. The WC can")
                print("     be tuned critical — widen g_e prior, re-center, re-infer.")
            elif stds[-1] > 1.3 * stds[0]:
                print("  => std climbing with coupling but not there yet — push the")
                print("     sweep higher; the critical point is beyond the range.")
            else:
                print("  => std flat / falling even at extreme coupling. This WC")
                print("     parameterization does NOT reach criticality. Switch to a")
                print("     Hopf (Stuart-Landau) node model (Deco 2017 resting-FC).")
        return

    # single param set -> one theta row. route everything via theta columns.
    pn = list(fixed.keys())
    theta = np.array([[fixed[k] for k in pn]], dtype=np.float64)

    print("=" * 84)
    print(f"  Neural-FC vs BOLD-FC — subject {sid}   N={N}")
    print(f"  params={fixed}  sigma={args.sigma}  seed={args.seed}")
    print(f"  real-FC std = {real_std:.4f}")
    print("=" * 84)

    # --- BW BOLD path (the adapter the pipeline actually uses) ---
    t0 = time.time()
    bold = simulate_gpu_batch(weights, theta, param_names=pn, delays=delays,
                              apply_bw=True,
                              fixed_overrides=sim_fixed)[0]
    bold_m = _fc_std_sc(bold, iu, sc_vec, sc_edge, real_vec, real_mask)
    print(f"  [BW BOLD]  T={np.asarray(bold).shape[0]}  ({time.time()-t0:.0f}s)")

    # --- raw neural E path ---
    t0 = time.time()
    E = _build_neural_grp(weights, theta, pn, sim_fixed, delays, args.seed)[0]
    neural_m = _fc_std_sc(E, iu, sc_vec, sc_edge, real_vec, real_mask)
    print(f"  [neural E] T={E.shape[0]}  ({time.time()-t0:.0f}s)")

    print("\n" + "-" * 84)
    print(f"  {'signal':10s} {'FC_std':>8s} {'FC~SC(edge)':>11s} "
          f"{'FC~SC(all)':>10s} {'FC~real':>8s}")
    print("  " + "-" * 50)
    for name, m in [("neural E", neural_m), ("BW BOLD", bold_m),
                    ("real", dict(std=real_std, sc_edge=_corr(real_vec, sc_vec, sc_edge & real_mask),
                                  sc_all=np.nan, real=1.0))]:
        print(f"  {name:10s} {m['std']:8.4f} {m['sc_edge']:11.4f} "
              f"{m['sc_all']:10.4f} {m['real']:8.4f}")

    np.savez(f"{args.out}_{sid}.npz", subject=sid, real_std=real_std,
             neural=neural_m, bold=bold_m, params=fixed, sigma=args.sigma)
    print(f"\n  saved: {args.out}_{sid}.npz")

    print("\n" + "=" * 84)
    print("  VERDICT")
    print("=" * 84)
    ns, bs = neural_m["std"], bold_m["std"]
    if np.isfinite(ns) and np.isfinite(bs) and ns >= 1.8 * bs and ns >= 0.5 * real_std:
        print(f"  neural FC std ({ns:.4f}) >> BOLD FC std ({bs:.4f}).")
        print("  => Balloon-Windkessel washes out the cross-region correlation.")
        print("     Fix the OBSERVATION model (BW params / HRF / TR downsample),")
        print("     not the neural dynamics.")
    elif np.isfinite(ns) and ns < 0.5 * real_std:
        print(f"  neural FC std ({ns:.4f}) is ALSO low (real {real_std:.4f}).")
        print("  => The dynamics are weakly correlated before BOLD. Noise-driven")
        print("     fixed-point WC builds no strong shared mode. Need a near-")
        print("     critical / oscillatory regime, or a different neural model.")
    else:
        print(f"  neural {ns:.4f} vs BOLD {bs:.4f} vs real {real_std:.4f} —")
        print("  inconclusive; inspect the table and try other operating points.")


if __name__ == "__main__":
    main()
