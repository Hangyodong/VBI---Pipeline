#!/usr/bin/env python
"""FC parameter sweep — which WC params actually move the simulated FC?

For each parameter we hold every other parameter at a baseline value and
sweep it across a grid, simulating via cuBNM (GPU, batched). Two questions
per parameter:

  1. SENSITIVITY  — does the simulated FC change at all as the param moves?
                    (dead knob  -> FC identical across the grid)
  2. REAL-FIT     — does any grid value bring the simulated FC close to a
                    real subject's FC?  (max corr over the grid)

Important implementation note
-----------------------------
The cuBNM adapter (``cuBNM/runner_vbi.build_param_lists``) only reads
``g_e, g_i, c_ei`` from the theta columns. ``P`` (P_E) and ``Q`` (P_I),
plus the coupling constants ``c_ee/c_ie/c_ii``, are applied through the
``fixed_overrides`` path instead (``_FIXED_TO_VBI`` maps P->P_E, Q->P_I).
So this script routes each parameter the correct way:

  theta-route  : g_e, g_i, c_ei   (one big batched GPU call)
  fixed-route  : P, Q, c_ee, c_ie, c_ii  (one call per grid value)

This is also why the main pipeline cannot identify P/Q: those theta
columns never reach the simulator. The sweep confirms whether P/Q have
any *physical* effect when actually applied.

Usage
-----
    python sweep_fc.py                       # default subject, all params
    python sweep_fc.py --subject sub-419090 --grid 15 --repeats 3
    python sweep_fc.py --params P,Q,c_ee     # subset
"""
import argparse
import time

import numpy as np

import config


# --------------------------------------------------------------------------
# Baseline operating point + sweep ranges
# --------------------------------------------------------------------------
# theta-route params (read from theta columns by the cuBNM adapter)
THETA_PARAMS = ["g_e", "g_i", "c_ei"]
# fixed-route params (applied via fixed_overrides -> _FIXED_TO_VBI)
FIXED_PARAMS = ["P", "Q", "c_ee", "c_ie", "c_ii"]

BASELINE = {
    "g_e": 0.5, "g_i": 0.0, "c_ei": 12.0,        # theta center
    "P": 0.5, "Q": 0.0,                          # WC_FIXED defaults
    "c_ee": 16.0, "c_ie": 15.0, "c_ii": 3.0,     # WC_FIXED couplings
}

SWEEP_RANGE = {
    "P":    (0.0, 3.0),
    "Q":    (0.0, 3.0),
    "g_e":  (0.0, 2.0),
    "g_i":  (0.0, 2.0),
    "c_ei": (6.0, 18.0),
    "c_ee": (8.0, 24.0),
    "c_ie": (8.0, 24.0),
    "c_ii": (0.0, 6.0),
}


# --------------------------------------------------------------------------
# FC helpers
# --------------------------------------------------------------------------

def _fc_vec(fc_mat, iu):
    return fc_mat[iu]


def _masked_corr(a, b, nan_vec):
    """Pearson corr on finite, non-original-NaN entries."""
    m = np.isfinite(a) & np.isfinite(b) & ~nan_vec
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# --------------------------------------------------------------------------
# Simulation wrappers
# --------------------------------------------------------------------------

def _simulate(weights, theta_batch, fixed, delays):
    """Run one cuBNM batch -> list of FC matrices (one per row)."""
    from cuBNM.simulate import simulate_gpu_batch
    from simulator import compute_fc

    bolds = simulate_gpu_batch(
        weights, theta_batch, param_names=THETA_PARAMS,
        delays=delays, apply_bw=True, fixed_overrides=fixed,
    )
    return [compute_fc(np.asarray(b)) for b in bolds]


def _avg_fc(fc_list):
    """Mean FC over repeats (NaN-safe)."""
    stk = np.stack(fc_list, axis=0)
    return np.nanmean(stk, axis=0)


def _theta_row(overrides):
    """Build a THETA_PARAMS row from baseline + overrides."""
    vals = dict(BASELINE)
    vals.update(overrides)
    return np.array([vals[p] for p in THETA_PARAMS], dtype=np.float64)


def _fixed_base(overrides=None):
    """Baseline fixed_overrides dict (couplings + P/Q) + optional overrides."""
    f = {k: BASELINE[k] for k in FIXED_PARAMS}
    if overrides:
        f.update(overrides)
    return f


def sweep_one(param, grid, weights, delays, n_repeat):
    """Return (sim_fc_per_grid, elapsed) for one parameter sweep.

    sim_fc_per_grid : list of (N,N) mean FC, one per grid value.
    """
    t0 = time.time()
    if param in THETA_PARAMS:
        # One batched GPU call: all grid values x repeats stacked.
        rows = []
        for v in grid:
            row = _theta_row({param: v})
            rows.extend([row] * n_repeat)
        theta_batch = np.stack(rows, axis=0)
        fc_all = _simulate(weights, theta_batch, _fixed_base(), delays)
        sim_fc = [
            _avg_fc(fc_all[i * n_repeat:(i + 1) * n_repeat])
            for i in range(len(grid))
        ]
    else:
        # fixed-route: scalar per call -> one call per grid value.
        theta_batch = np.tile(_theta_row({})[None, :], (n_repeat, 1))
        sim_fc = []
        for v in grid:
            fc_list = _simulate(
                weights, theta_batch, _fixed_base({param: v}), delays,
            )
            sim_fc.append(_avg_fc(fc_list))
    return sim_fc, time.time() - t0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_subject(sid):
    """Load one subject's SC weights, real FC, NaN mask, delays."""
    import data_loader
    df, fc_mat, sc_mat, fc_ids, sc_ids, _, _ = data_loader.load_raw_data()
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    if sid is None:
        sid = subjects[0]
    if sid not in subjects:
        raise SystemExit(f"subject {sid} not in {subjects}")
    d = data_loader.get_subject_data(sid, fc_mat, sc_mat, fc_ids, sc_ids)
    return sid, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None, help="subject id (default: first)")
    ap.add_argument("--grid", type=int, default=13, help="grid points per param")
    ap.add_argument("--repeats", type=int, default=3, help="noise repeats per point")
    ap.add_argument("--params", default=None,
                    help="comma list (default: all 8)")
    ap.add_argument("--out", default="output_mouse_mptp/sweep_fc")
    args = ap.parse_args()

    params = (args.params.split(",") if args.params
              else THETA_PARAMS + FIXED_PARAMS)

    sid, d = load_subject(args.subject)
    weights = d["sc"]
    delays = d["delays"]
    fc_real = d["fc"]
    N = fc_real.shape[0]
    iu = np.triu_indices(N, k=1)
    real_vec = fc_real[iu]
    nan_vec = (d["fc_nan"][iu] if "fc_nan" in d
               else np.zeros(iu[0].shape, dtype=bool))

    print("=" * 70)
    print(f"  FC parameter sweep — subject {sid}")
    print(f"  grid={args.grid}  repeats={args.repeats}  N_regions={N}")
    print(f"  valid FC entries (mask applied): "
          f"{int((~nan_vec).sum())}/{len(nan_vec)}")
    print("=" * 70)

    # Baseline FC (all params at BASELINE) for the sensitivity reference.
    base_fc = _avg_fc(_simulate(
        weights, np.tile(_theta_row({})[None, :], (args.repeats, 1)),
        _fixed_base(), delays,
    ))
    base_vec = base_fc[iu]
    base_real = _masked_corr(base_vec, real_vec, nan_vec)
    print(f"\n  baseline real-FC corr = {base_real:+.4f}\n")

    results = {}
    header = (f"  {'param':6s} {'route':6s} {'sensitivity':>12s} "
              f"{'realfit_max':>12s} {'@value':>9s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for p in params:
        if p not in SWEEP_RANGE:
            print(f"  {p:6s}  (unknown param, skipped)")
            continue
        lo, hi = SWEEP_RANGE[p]
        grid = np.linspace(lo, hi, args.grid)
        route = "theta" if p in THETA_PARAMS else "fixed"
        sim_fc, dt = sweep_one(p, grid, weights, delays, args.repeats)

        sim_vecs = [fc[iu] for fc in sim_fc]
        # sensitivity: 1 - corr(simFC(v), simFC(baseline_point)).
        # baseline_point = grid value closest to BASELINE[p].
        b_idx = int(np.argmin(np.abs(grid - BASELINE[p])))
        ref = sim_vecs[b_idx]
        valid = np.isfinite(ref) & ~nan_vec
        sens = []
        for v in sim_vecs:
            mm = valid & np.isfinite(v)
            if mm.sum() > 10 and ref[mm].std() > 0 and v[mm].std() > 0:
                sens.append(1.0 - np.corrcoef(ref[mm], v[mm])[0, 1])
        sensitivity = float(np.nanmax(sens)) if sens else np.nan
        # real-fit: best corr to real FC across grid.
        real_corrs = [_masked_corr(v, real_vec, nan_vec) for v in sim_vecs]
        rc = np.array(real_corrs, dtype=float)
        if np.all(np.isnan(rc)):
            best_i, best_corr = 0, np.nan
        else:
            best_i = int(np.nanargmax(rc))
            best_corr = float(rc[best_i])

        results[p] = {
            "grid": grid, "real_corrs": rc,
            "sensitivity": sensitivity,
            "realfit_max": best_corr, "realfit_at": grid[best_i],
            "route": route, "elapsed": dt,
        }
        print(f"  {p:6s} {route:6s} {sensitivity:12.4f} "
              f"{best_corr:12.4f} {grid[best_i]:9.3f}  ({dt:.1f}s)")

    # ---- save ----------------------------------------------------------
    np.savez(
        args.out + ".npz",
        subject=sid, baseline_real_corr=base_real,
        **{f"{p}_grid": results[p]["grid"] for p in results},
        **{f"{p}_realcorr": results[p]["real_corrs"] for p in results},
    )
    print(f"\n  saved: {args.out}.npz")

    try:
        _plot(results, base_real, sid, args.out + ".png")
        print(f"  saved: {args.out}.png")
    except Exception as e:  # noqa: BLE001
        print(f"  [plot skipped] {type(e).__name__}: {e}")

    # ---- verdict -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    dead = [p for p in results if results[p]["sensitivity"] < 0.02]
    live = [p for p in results if results[p]["sensitivity"] >= 0.02]
    print(f"  FC-sensitive (infer these) : {live}")
    print(f"  dead knobs   (drop these)  : {dead}")
    best_overall = max(
        (results[p]["realfit_max"] for p in results
         if np.isfinite(results[p]["realfit_max"])), default=np.nan,
    )
    print(f"  best single-param real-FC corr reachable: {best_overall:+.4f}")
    print("  (if this stays low for every param, the simulator itself "
          "cannot\n   reproduce this subject's FC = misspecification ceiling)")


def _plot(results, base_real, sid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    for ax, (p, r) in zip(axes.ravel(), results.items()):
        ax.plot(r["grid"], r["real_corrs"], "o-")
        ax.axhline(base_real, ls="--", c="gray", lw=1, label="baseline")
        ax.axvline(BASELINE[p], ls=":", c="red", lw=1)
        ax.set_title(f"{p} ({r['route']})  sens={r['sensitivity']:.3f}")
        ax.set_xlabel(p)
        ax.set_ylabel("real-FC corr")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"FC sweep — {sid}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
