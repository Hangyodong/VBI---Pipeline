#!/usr/bin/env python
"""Three FC diagnostics on the active HCP model (real cuBNM engine), from ONE
batch of random-parameter simulations:

  1. per-parameter sensitivity  — how much each inferred param moves the FC
  2. most-sensitive edges        — which FC edges respond most to parameters
  3. empirical-in-support test   — is the empirical FC inside the simulated FC
                                   distribution? (reachability / OOD, per edge)

Task 3 is the decisive one: if most empirical edges fall OUTSIDE the simulated
range, the empirical FC is genuinely unreachable (OOD) for this model at any of
the sampled params; if inside, it is reachable and the bottleneck is inference
/ operating point, not the model's expressive range.

Mirrors main_HCP setup (rwweib2, cortical-only 360, group-avg FC, maxnorm SC).
GPU required. Run on the GPU node:

    python fc_support_diag.py                 # 1000 random sims
    python fc_support_diag.py --n 2000 --subject 113922
"""
import argparse
import os
import numpy as np

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")

from pipeline_setup import PipelineConfig, setup_pipeline

_cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat", SC_FILE="HCP_SC.mat",
    N_REGIONS=360, N_SUBJECTS=100, N_TRAIN=70, N_VAL=10, N_TEST=20, SEED=42,
    N_SIM=2_000, GPU_BATCH=2_000,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
)
setup_pipeline(_cfg)
import config

config.INFERENCE_MODEL = "rwweib2"
# 8 params (matches main_HCP): g_LRE,g_FFI,sigma,I_o,w_E,w_I,J_i,w_p.
# NOTE: do NOT pass --fic when analyzing J_i (FIC would override the swept J_i).
config.STAGE1_PARAMS      = ["g_LRE", "g_FFI", "sigma", "I_o", "w_E", "w_I", "J_i", "w_p"]
config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
config.STAGE1_PRIOR_LOW   = [0.0, 0.0, 0.0,  0.30, 0.0, 0.0, 0.0, 0.0]
config.STAGE1_PRIOR_HIGH  = [3.0, 3.0, 0.03, 0.45, 5.0, 5.0, 5.0, 5.0]
config.RWWEIB2_FIXED      = {"J_N": 0.15, "lambda_IE": 1.0}
config.GROUP_AVG_FC       = True
config.USE_DELAYS         = False
config.USE_FCD            = False

import data_loader_hcp as data_loader
from engine_select import get_simulate_gpu_batch
from cuBNM.fc import compute_fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="random-param simulations")
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--prior_scale", type=float, default=1.0,
                    help="widen prior ranges by this factor (0-based params: hi*=s; else: center +/- halfwidth*s)")
    ap.add_argument("--fic", action="store_true",
                    help="apply numerical FIC (per-sim/node J_i to operating point) before sim")
    ap.add_argument("--io", type=str, default=None,
                    help="override I_o prior range as 'lo,hi' (e.g. 0,5)")
    ap.add_argument("--gsr", choices=["none", "demean", "eig1"], default="none",
                    help="FC-level global-signal removal (applied to emp AND every sim): "
                         "demean=subtract each FC's mean (kills +offset); "
                         "eig1=remove top eigenmode (global mode)")
    ap.add_argument("--hetero", action="store_true",
                    help="HETEROGENEOUS: each NODE gets its own random param ~U(lo,hi) per sim "
                         "(regional params; g_LRE stays global). Capability test only — "
                         "per-node random is non-identifiable for inference.")
    args = ap.parse_args()

    o = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = o
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    train, val, test = data_loader.three_way_split(subjects)
    sid = args.subject if args.subject is not None else test[0]
    d = data_loader.load_all_subjects([sid], fc_mat, sc_mat, fc_ids, sc_ids)[sid]

    sc = d["sc"]; delays = d["delays"]; fc_real = d["fc"]
    N = sc.shape[0]; iu = np.triu_indices(N, 1)
    sc_vec = sc[iu]
    fin = np.isfinite(fc_real[iu])

    def gsr_vec(mat):
        """Upper-tri of an FC matrix after FC-level global-signal removal."""
        m = np.asarray(mat, float)
        if args.gsr == "eig1":
            m = np.nan_to_num(m, nan=0.0)
            w, V = np.linalg.eigh(m)
            m = m - w[-1] * np.outer(V[:, -1], V[:, -1])   # drop global mode
        v = m[iu]
        if args.gsr == "demean":
            v = v.copy(); v[fin] = v[fin] - v[fin].mean()  # kill +offset
        return v

    emp = gsr_vec(fc_real)
    nz = (sc_vec > 0) & fin              # existing-SC edges
    zero = (sc_vec == 0) & fin           # absent-SC edges

    pn = list(config.STAGE1_PARAMS)
    lo = np.asarray(config.STAGE1_PRIOR_LOW, float).copy()
    hi = np.asarray(config.STAGE1_PRIOR_HIGH, float).copy()
    s = args.prior_scale
    if s != 1.0:
        for j, p in enumerate(pn):
            if lo[j] == 0.0:                       # 0-based (g_LRE/g_FFI/sigma): extend upper
                hi[j] = hi[j] * s
            else:                                  # I_o etc.: widen both ways about center
                c = (lo[j] + hi[j]) / 2.0; hw = (hi[j] - lo[j]) / 2.0 * s
                lo[j] = max(c - hw, 0.05); hi[j] = c + hw
    if args.io is not None and "I_o" in pn:                # explicit I_o range override
        _ilo, _ihi = (float(x) for x in args.io.split(","))
        j = pn.index("I_o"); lo[j] = _ilo; hi[j] = _ihi
    rng = np.random.RandomState(args.seed)
    theta = rng.uniform(lo, hi, size=(args.n, len(pn)))

    print("=" * 74)
    print(f"  FC support diagnostics — {config.INFERENCE_MODEL}  subj {sid}  "
          f"N={N} cortical  {args.n} random sims")
    print(f"  target FC = {'group-avg' if config.GROUP_AVG_FC else 'per-subject'}"
          f"   nz edges={int(nz.sum())}/{int(fin.sum())} ({nz.mean()*100:.0f}%)")
    print(f"  prior x{s:g}: " + "  ".join(f"{p}[{lo[j]:.3g},{hi[j]:.3g}]"
                                          for j, p in enumerate(pn)))
    print(f"  GSR mode = {args.gsr}  (FC-level global-signal removal, emp + all sims)")
    print("=" * 74, flush=True)

    fixed = {"seed": args.seed}
    if args.hetero:
        # each node gets its own random value ~U(lo,hi) per sim (regional only)
        nreg = 0
        for jp, p in enumerate(pn):
            if p == "g_LRE":
                continue                       # global scalar, not per-node
            fixed[f"{p}_matrix"] = rng.uniform(lo[jp], hi[jp], size=(args.n, N))
            nreg += 1
        print(f"  [HETERO] {nreg} regional params randomized per-node "
              f"({args.n} sims x {N} nodes). Task1 param-sensitivity is N/A.", flush=True)
    if args.fic:
        from fic_tune import compute_fic_ji
        print(f"  [FIC] tuning J_i for {args.n} sims ...", flush=True)
        ji = compute_fic_ji(sc, theta, pn, J_N=config.RWWEIB2_FIXED["J_N"])
        fixed["J_i_matrix"] = ji

    sim = get_simulate_gpu_batch()
    bolds = sim(sc, theta, param_names=pn, delays=delays, apply_bw=True,
                fixed_overrides=fixed)
    FC = np.stack([gsr_vec(compute_fc(np.asarray(b))) for b in bolds])   # (n_sims, edges)
    FC = FC[:, fin]; empf = emp[fin]; scf = sc_vec[fin]
    nzf = (scf > 0)
    rvf = empf
    # node indices (i,j) of each finite edge, for FC-matrix edge reporting
    edge_i = iu[0][fin]; edge_j = iu[1][fin]

    # ---- Task 1: per-parameter sensitivity ----
    print("\n--- 1. per-parameter sensitivity (over random sims) ---")
    fit = np.array([np.corrcoef(FC[k], rvf)[0, 1] for k in range(args.n)])  # simFC~real per sim
    fit = np.nan_to_num(fit)
    print(f"  {'param':8s} | mean|corr(param,edge)| | corr(param, simFC~real) | var-share")
    # variance of FC explained by each param (R^2 of linear fit of each edge on params, avg)
    edge_var = FC.var(0)
    for j, p in enumerate(pn):
        col = theta[:, j]
        if col.std() == 0:
            continue
        # mean abs corr of this param with each edge
        cc = np.array([np.corrcoef(col, FC[:, e])[0, 1] for e in range(0, FC.shape[1], max(1, FC.shape[1] // 3000))])
        mabs = np.nanmean(np.abs(cc))
        cfit = np.corrcoef(col, fit)[0, 1]
        # variance share: corr(param, edge)^2 weighted by edge variance
        share = np.nansum(np.nan_to_num(cc) ** 2) / len(cc)
        print(f"  {p:8s} |        {mabs:.3f}          |        {cfit:+.3f}          |  {share:.3f}")

    # ---- Task 2: most-sensitive edges ----
    print("\n--- 2. most-sensitive edges (sim-FC std across params) ---")
    estd = FC.std(0)
    order = np.argsort(-estd)[:args.topk]
    print(f"  edge-FC std: nz-SC mean={estd[nzf].mean():.3f}  zero-SC mean={estd[~nzf].mean():.3f}")
    print(f"  top {args.topk} most-variable FC-matrix edges (node i,j):")
    print(f"    {'edge(i,j)':>12s} {'SC':>6s} {'emp':>7s} {'sim[min':>8s} {'max':>7s} {'std]':>6s}  inSupport?")
    for e in order:
        lo_e, hi_e = FC[:, e].min(), FC[:, e].max()
        inside = lo_e <= empf[e] <= hi_e
        print(f"    ({int(edge_i[e]):>3d},{int(edge_j[e]):>3d})   {scf[e]:6.3f} {empf[e]:+7.3f} "
              f"{lo_e:+8.3f} {hi_e:+7.3f} {estd[e]:6.3f}  {'YES' if inside else 'NO'}")

    # ---- Task 3: empirical FC in simulated support ----
    print("\n--- 3. empirical FC in simulated support (reachability) ---")
    fmin, fmax = FC.min(0), FC.max(0)
    p5, p95 = np.percentile(FC, 5, axis=0), np.percentile(FC, 95, axis=0)
    fmean, fstd = FC.mean(0), FC.std(0) + 1e-9
    in_range = (empf >= fmin) & (empf <= fmax)
    in_90 = (empf >= p5) & (empf <= p95)
    z = np.abs((empf - fmean) / fstd)
    def pct(mask, sel):
        return 100.0 * (mask & sel).sum() / max(sel.sum(), 1)
    print(f"  %% empirical edges inside sim [min,max]:  all={in_range.mean()*100:.1f}%  "
          f"nz={pct(in_range,nzf):.1f}%  zero={pct(in_range,~nzf):.1f}%")
    print(f"  %% inside sim [p5,p95]:                   all={in_90.mean()*100:.1f}%  "
          f"nz={pct(in_90,nzf):.1f}%  zero={pct(in_90,~nzf):.1f}%")
    print(f"  mean |z(emp vs sim)|:                     all={z.mean():.2f}  "
          f"nz={z[nzf].mean():.2f}  zero={z[~nzf].mean():.2f}")
    print(f"  emp mean={empf.mean():+.3f}  sim-grand-mean={fmean.mean():+.3f}  "
          f"(offset={empf.mean()-fmean.mean():+.3f})")
    print("\n" + "=" * 74)
    if in_range.mean() > 0.8:
        print("  VERDICT: empirical mostly INSIDE sim support -> reachable.")
        print("  Bottleneck = inference / operating point, NOT model range.")
    elif in_range.mean() > 0.5:
        print("  VERDICT: empirical PARTIALLY in support. Some edges (esp. zero-SC")
        print("  / homotopic) unreachable; model range covers ~half.")
    else:
        print("  VERDICT: empirical mostly OUTSIDE sim support -> OOD/unreachable.")
        print("  Model cannot produce this FC at any sampled params (expand model:")
        print("  FIC/operating point, delays, or per-edge effective connectivity).")
    print("=" * 74)


if __name__ == "__main__":
    main()
