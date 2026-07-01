#!/usr/bin/env python
"""Measure the BASIS forward-model ceiling — the decisive fwd-vs-inference experiment.

Question: what is the best corr(simFC, realFC) the rwweib2 + basis_regionwise model
can reach by searching the 12 basis coefficients DIRECTLY (no SNPE)? This resolves
the paradox: homogeneous 4-param search tops at ~0.03 (output_hcp/joint_opt_*.npz),
yet the (buggy) SNPE run reported ~0.13.

  - ceiling >= ~0.10  -> CASE A: regional heterogeneity genuinely lifts the forward
                          ceiling; the basis is the real lever -> richer basis.
  - ceiling ~  0.03   -> CASE B: the 0.13 was an embedding/wide-bounds artifact; the
                          model cannot fit per-subject FC -> architecture/target change.

Method (identical decode path to training, no inference):
  random betas (S, 12) in coeff range
    -> engine_select.latent_wrap -> decode -> per-region {param}_matrix (S, 360)
    -> cuBNM rwweib2 (hrf="bw", SAME as production) -> BOLD -> simFC
    -> corr(simFC_upper, realFC_upper)  -> best over the batch.
One GPU batch of S draws per subject (sim cost dominates). --rounds accumulates the
best across several batches (random search is sparse in 12-D). CMA could refine later.

Usage
-----
  python basis_ceiling.py                              # first 2 subjects, 3000 draws
  python basis_ceiling.py --subjects 100307,106016 --n_search 4000 --rounds 3
  python basis_ceiling.py --coeff-range 2.5            # widen search beyond prior [-2,2]
  GROUP_AVG_FC=1 python basis_ceiling.py               # vs group-avg target
  USE_DELAYS=1 python basis_ceiling.py                 # with tract delays (slower)
"""
import argparse
import os

import numpy as np

from pipeline_setup import PipelineConfig, setup_pipeline

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")

# ── production config (mirror main_HCP.py basis_regionwise + cabnp) ───────────
_cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat",
    SC_FILE=os.environ.get("SC_FILE", "HCP_CABNP381_SC_first100.mat"),
    N_REGIONS=360, N_SUBJECTS=int(os.environ.get("N_SUBJECTS", "10")),
    N_TRAIN=7, N_VAL=1, N_TEST=2, SEED=42, N_SIM=2_000, GPU_BATCH=2_000,
    T_END_MS=630_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,  # 864 TR total, 780 analyzed (match main_HCP)
)
setup_pipeline(_cfg, print_summary=False)

import config
config.INFERENCE_MODEL = "rwweib2"
config.PARAMETER_MODE = "basis_regionwise"
config.N_REGIONS = 360
config.HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
# Bounds mirror main_HCP.py production (env-tunable): I_o operating-point fix
# defaults to (0.30,0.45). Reproduce the OLD (0,1) ceiling with IO_BOUND_LO=0
# IO_BOUND_HI=1; widen coupling with G_BOUND_HIGH. Was hardcoded (0,1) -> the
# ceiling re-measure would have silently used the old saturated bound.
_G_HI = float(os.environ.get("G_BOUND_HIGH", "3.0"))
_IO_LO = float(os.environ.get("IO_BOUND_LO", "0.30"))
_IO_HI = float(os.environ.get("IO_BOUND_HI", "0.45"))
config.BASIS_BOUNDS = {"g_LRE": (0.0, _G_HI), "g_FFI": (0.0, _G_HI),
                       "I_o": (_IO_LO, _IO_HI), "sigma": (0.0, 0.05)}
config.HETERO_BOUNDS = dict(config.BASIS_BOUNDS)
config.BASIS_COEFF_PRIOR = (-2.0, 2.0)
config.BASIS_PATH = os.environ.get("BASIS_PATH", "basis.npy")
config.BASIS_REZSCORE = True
config.RWWEIB2_FIXED = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
                        "J_N": 0.15, "lambda_IE": 1.0}
config.SC_DATASET = os.environ.get("SC_DATASET", "cabnp381")
config.GROUP_AVG_FC = (os.environ.get("GROUP_AVG_FC", "0") == "1")
config.VELOCITY_M_PER_S = 3.0
config.USE_FCD = False
config.USE_DELAYS = (os.environ.get("USE_DELAYS", "0") == "1")

import data_loader_hcp as data_loader
import engine_select
from basis_decoder import get_decoder
from features.fc import compute_fc, fc_to_upper_tri

_dec = get_decoder(config)
config.STAGE1_PARAMS = _dec.coeff_names()           # 12 coeff names (latent_wrap routing)
config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS


def _masked_corr(a, b, nan_mask=None):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if nan_mask is not None:
        m &= ~nan_mask
    if m.sum() < 3:
        return -1.0
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def run_subject(sid, d, simulate_batch, args, rng):
    sc = d["sc"].astype(np.float64)
    delays = d["delays"].astype(np.float64) if config.USE_DELAYS else None
    real_fc = d["fc"]
    iu = np.triu_indices(config.N_REGIONS, k=1)
    real_vec = np.asarray(real_fc)[iu]
    nan_vec = (np.asarray(d.get("fc_nan", np.zeros_like(real_fc)))[iu] > 0)

    best_corr = -1.0
    best_beta = None
    cr = float(args.coeff_range)
    total = 0
    for r in range(args.rounds):
        theta = rng.uniform(-cr, cr, (args.n_search, _dec.theta_dim)).astype(np.float64)
        bolds = simulate_batch(sc, theta, config.STAGE1_PARAMS,
                               delays=delays, apply_bw=True)
        corrs = np.array([
            _masked_corr(fc_to_upper_tri(compute_fc(np.asarray(b))), real_vec, nan_vec)
            for b in bolds
        ])
        corrs = np.where(np.isfinite(corrs), corrs, -1.0)
        k = int(np.argmax(corrs))
        if corrs[k] > best_corr:
            best_corr = float(corrs[k]); best_beta = theta[k].copy()
        total += args.n_search
        print(f"    [{sid}] round {r+1}/{args.rounds}  batch_best={corrs.max():+.4f}  "
              f"running_best={best_corr:+.4f}  (n={total})", flush=True)
    return best_corr, best_beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma ids; default = first N from split")
    ap.add_argument("--n-subjects", type=int, default=2, help="how many (if --subjects empty)")
    ap.add_argument("--n_search", type=int, default=3000, help="random draws per round (one GPU batch)")
    ap.add_argument("--rounds", type=int, default=2, help="rounds accumulated for best")
    ap.add_argument("--coeff-range", type=float, default=2.0, help="search betas in [-r, r] (prior=2)")
    args = ap.parse_args()

    df, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi = data_loader.load_raw_data()
    if args.subjects:
        sids = [int(s) for s in args.subjects.split(",") if s]
    else:
        sids = data_loader.get_target_subjects(df, fc_ids, sc_ids)[:args.n_subjects]
    subj_data = data_loader.load_all_subjects(sids, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi)

    simulate_batch = engine_select.get_simulate_gpu_batch()   # latent_wrap(rwweib2)

    print("=" * 64)
    print("  BASIS forward-model ceiling (rwweib2, no SNPE)")
    print("=" * 64)
    print(f"  subjects={sids}  n_search={args.n_search} x rounds={args.rounds}="
          f"{args.n_search*args.rounds}/subj")
    print(f"  coeff range=[-{args.coeff_range},{args.coeff_range}]  delays={config.USE_DELAYS}  "
          f"target={'group-avg' if config.GROUP_AVG_FC else 'per-subject'}")
    print(f"  compare: homogeneous 4-param ceiling ~0.016-0.034 (joint_opt_*.npz)")
    print("-" * 64)

    rng = np.random.RandomState(0)
    results = {}
    for sid in sids:
        print(f"  subject {sid}:")
        bc, bb = run_subject(sid, subj_data[sid], simulate_batch, args, rng)
        results[sid] = bc
        # decode best beta -> param map stats (for case-A interpretability)
        if bb is not None:
            maps = _dec.decode(bb[None, :])
            stat = "  ".join(
                f"{p}:[{np.asarray(maps[p]).min():.3f},{np.asarray(maps[p]).max():.3f}]"
                for p in config.HETERO_PARAMS)
            print(f"  >>> {sid} CEILING = {bc:+.4f}   best-map ranges: {stat}")

    print("=" * 64)
    cs = np.array(list(results.values()))
    print(f"  BASIS CEILING per subject: " +
          "  ".join(f"{s}={c:+.4f}" for s, c in results.items()))
    print(f"  mean={cs.mean():+.4f}  max={cs.max():+.4f}")
    print(f"  homogeneous ceiling ~0.03  |  (buggy) SNPE reported ~0.13")
    if cs.max() >= 0.10:
        print("  => CASE A: basis heterogeneity lifts the ceiling. Push richer basis.")
    elif cs.max() <= 0.045:
        print("  => CASE B: ceiling ~ homogeneous. The 0.13 was an artifact. "
              "Fix forward (HRF/SC/FIC) or target (group-avg).")
    else:
        print("  => INTERMEDIATE: re-run with more rounds / CMA / wider coeff-range.")
    print("=" * 64)


if __name__ == "__main__":
    main()
