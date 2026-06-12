#!/usr/bin/env python
"""Inferred-parameter sensitivity for the active HCP model (real cuBNM engine).

Sweeps each STAGE1 parameter across its prior range (others held at the prior
midpoint), runs ONE cuBNM batch per parameter, and reports how much the FC
responds — distinguishing "moves the operating point" from "moves toward the
SC / empirical FC structure".

Per parameter it reports:
  simFC~SC      : corr(sim FC, SC)        at each sweep value  (SC-structure)
  simFC~real    : corr(sim FC, target FC) at each sweep value  (what we want)
  meanS_E       : operating point         at each sweep value  (saturation 0..1)
  FCstd         : sim FC dispersion
  reorg         : 1 - corr(FC@min, FC@max)  -> how much the FC PATTERN moves
  influence     : |corr(param value, simFC~real)|  -> monotone push on the fit

Mirrors main_HCP setup (rwweib2, cortical-only 360, group-avg FC, maxnorm SC).
GPU required (cuBNM). Run on the GPU node:

    python sensitivity.py                       # all 4 params, 7 values each
    python sensitivity.py --samples 9 --subject 113922
    python sensitivity.py --params g_LRE,g_FFI
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

# active model = same as main_HCP (edit here if main_HCP changes)
config.INFERENCE_MODEL = "rwweib2"
config.STAGE1_PARAMS      = ["g_LRE", "g_FFI", "sigma", "I_o"]
config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
config.STAGE1_PRIOR_LOW   = [0.0, 0.0, 0.0,  0.30]
config.STAGE1_PRIOR_HIGH  = [3.0, 3.0, 0.03, 0.45]
config.RWWEIB2_FIXED      = {"w_p": 1.4, "J_N": 0.15, "J_i": 1.0, "lambda_IE": 1.0}
config.GROUP_AVG_FC       = True
config.USE_DELAYS         = False
config.USE_FCD            = False

import data_loader_hcp as data_loader
from engine_select import get_simulate_gpu_batch
from cuBNM.fc import compute_fc


def _masked_corr(a, b, mask):
    m = np.isfinite(a) & np.isfinite(b) & mask
    if m.sum() < 2 or a[m].std() <= 0 or b[m].std() <= 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    ap = argparse.ArgumentParser(description="Inferred-param FC sensitivity (cuBNM).")
    ap.add_argument("--subject", type=int, default=None, help="subject id (default: first test)")
    ap.add_argument("--samples", type=int, default=7, help="sweep values per param")
    ap.add_argument("--params", type=str, default=None, help="comma list (default: all STAGE1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    train, val, test = data_loader.three_way_split(subjects)
    sid = args.subject if args.subject is not None else test[0]
    d = data_loader.load_all_subjects([sid], fc_mat, sc_mat, fc_ids, sc_ids)[sid]

    sc = d["sc"]; delays = d["delays"]; fc_real = d["fc"]   # fc_real = group FC (GROUP_AVG_FC)
    N = sc.shape[0]; iu = np.triu_indices(N, 1)
    sc_vec = sc[iu]; real_vec = fc_real[iu]
    fin = np.isfinite(real_vec)
    nz = (sc_vec > 0) & fin

    # ALL runtime model parameters (name, default/center, lo, hi). The yaml
    # constants (a_E/b_E/d/tau/gamma/w_E/w_I) are compile-time in cuBNM and
    # cannot be swept without a rebuild, so they are not listed here.
    MODEL_PARAMS = [
        ("g_LRE",     1.0,   0.0,   3.0),    # inferred
        ("g_FFI",     1.0,   0.0,   3.0),    # inferred
        ("sigma",     0.01,  0.001, 0.03),   # inferred
        ("I_o",       0.382, 0.30,  0.45),   # inferred
        ("w_p",       1.4,   0.7,   2.1),    # fixed (local E recurrence)
        ("J_N",       0.15,  0.075, 0.30),   # fixed (NMDA coupling)
        ("J_i",       1.0,   0.3,   2.0),    # fixed (I->E weight)
        ("lambda_IE", 1.0,   0.0,   2.0),    # fixed (I long-range scaling)
    ]
    pnames = [m[0] for m in MODEL_PARAMS]
    center = np.array([m[1] for m in MODEL_PARAMS], float)
    lo = np.array([m[2] for m in MODEL_PARAMS], float)
    hi = np.array([m[3] for m in MODEL_PARAMS], float)
    sweep_params = (args.params.split(",") if args.params else pnames)

    print("=" * 78)
    print(f"  Sensitivity — {config.INFERENCE_MODEL}  subject {sid}  N={N} (cortical-only)")
    print(f"  target FC = {'group-avg' if config.GROUP_AVG_FC else 'per-subject'}; "
          f"real-FC~SC: all={_masked_corr(sc_vec, real_vec, fin):+.3f} "
          f"nz={_masked_corr(sc_vec, real_vec, nz):+.3f}")
    print(f"  center (prior mid): "
          + "  ".join(f"{n}={center[i]:.3f}" for i, n in enumerate(pnames)))
    print("=" * 78, flush=True)

    sim = get_simulate_gpu_batch()
    K = args.samples
    print(f"\n  {'param':6s} | sweep value -> [simFC~SC  simFC~real  meanS_E]"
          f"   | reorg  influence")
    print("  " + "-" * 74)

    for p in sweep_params:
        j = pnames.index(p)
        vals = np.linspace(lo[j], hi[j], K)
        theta = np.tile(center[None, :], (K, 1))
        theta[:, j] = vals
        bolds = sim(sc, theta, param_names=pnames, delays=delays, apply_bw=True,
                    fixed_overrides={"seed": args.seed})
        fcs = [compute_fc(np.asarray(b)) for b in bolds]
        rows = []
        for k, b in enumerate(bolds):
            fcv = fcs[k][iu]
            csc = _masked_corr(fcv, sc_vec, nz)
            cr = _masked_corr(fcv, real_vec, fin)
            mse = float(np.asarray(b).mean())  # proxy for operating point (BOLD mean ~ S_E level)
            rows.append((vals[k], csc, cr, mse, fcv))
        # reorg = how much FC pattern moves from min->max value
        reorg = 1.0 - _masked_corr(rows[0][4], rows[-1][4], fin)
        # influence = |corr(param value, simFC~real)|
        crs = np.array([r[2] for r in rows])
        infl = abs(np.corrcoef(vals, np.nan_to_num(crs))[0, 1]) if crs.std() > 0 else 0.0
        cell = "  ".join(f"{v:.2f}:[{csc:+.3f} {cr:+.3f} {ms:+.3f}]"
                         for v, csc, cr, ms, _ in rows)
        print(f"  {p:6s} | {cell}  | reorg={reorg:.2f}  infl={infl:.2f}", flush=True)

    print("\n  reorg high + simFC~SC rises  -> useful (moves toward SC structure)")
    print("  reorg high + meanS_E swings   -> wasted on operating point (needs FIC)")
    print("  reorg ~0 / flat               -> dead parameter (drop / fix)")


if __name__ == "__main__":
    main()
