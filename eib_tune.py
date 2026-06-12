#!/usr/bin/env python
"""EIB: per-edge effective-connectivity tuning to match empirical FC.

Part 2 of the EI_Tuning method (notebook), wrapped around the cuBNM RWWEIB_2CPL
engine. Instead of inferring a few global params, this directly tunes a per-edge
effective connectome W (init = structural SC) with a heuristic Hebbian rule so
the simulated FC reproduces the empirical (group) FC:

    W <- clip( W + eta * (FC_target - FC_pred) , 0, Wmax )   # symmetric, diag 0

Interleaved with FIC (Part 1): J_i is re-tuned to the operating point as W
changes. This is gradient-free (cuBNM is not differentiable) — each iteration is
one cuBNM simulation, FC, then a local update.

NOTE on interpretation: this is effective-connectivity ESTIMATION (descriptive),
not generative parameter inference. With ~N^2 free weights it CAN reach high FC
corr; per-subject it would overfit, so the default target is the GROUP FC.

GPU required. Run on the GPU node:

    python eib_tune.py                              # 200 iters, group FC
    python eib_tune.py --n_iter 400 --eta 0.03 --sim_s 60
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
    N_SIM=1, GPU_BATCH=1,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
)
setup_pipeline(_cfg)
import config

config.INFERENCE_MODEL = "rwweib2"
config.STAGE1_PARAMS      = ["g_LRE", "g_FFI", "sigma", "I_o", "w_p"]
config.RWWEIB2_FIXED      = {"w_p": 1.4, "J_N": 0.15, "J_i": 1.0, "lambda_IE": 1.0}
config.GROUP_AVG_FC       = True
config.USE_DELAYS         = False
config.USE_FCD            = False

import data_loader_hcp as data_loader
from cuBNM.fc import compute_fc
from cuBNM.runner_rwweib_2cpl import run_cubnm_rwweib2_batch
from fic_tune import compute_fic_ji


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.03, help="EIB step size (annealed)")
    ap.add_argument("--sim_s", type=float, default=60.0, help="sim seconds per tuning iter")
    ap.add_argument("--refic_every", type=int, default=20)
    ap.add_argument("--wmax", type=float, default=2.5)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # fixed operating-point params (EIB tunes W, not these)
    ap.add_argument("--g_lre", type=float, default=1.0)
    ap.add_argument("--g_ffi", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--io", type=float, default=0.35)
    ap.add_argument("--w_p", type=float, default=1.4)
    args = ap.parse_args()

    o = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = o
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    train, val, test = data_loader.three_way_split(subjects)
    sid = args.subject if args.subject is not None else test[0]
    d = data_loader.load_all_subjects([sid], fc_mat, sc_mat, fc_ids, sc_ids)[sid]

    SC = d["sc"]; N = SC.shape[0]; iu = np.triu_indices(N, 1)
    FC_t = d["fc"]                                  # group FC matrix (target)
    tgt = FC_t[iu]; fin = np.isfinite(tgt)

    pn = list(config.STAGE1_PARAMS)
    theta = np.array([[args.g_lre, args.g_ffi, args.sigma, args.io, args.w_p]], float)

    def fc_corr(fcm):
        v = fcm[iu]
        m = fin & np.isfinite(v)
        return float(np.corrcoef(v[m], tgt[m])[0, 1])

    def run(W, Ji, sim_s, burn_s):
        bolds = run_cubnm_rwweib2_batch(
            W, theta, pn, duration_s=sim_s, burn_in_s=burn_s,
            force_gpu=True, hrf="bw",
            fixed={"J_i_matrix": Ji, "seed": args.seed},
        )
        return compute_fc(np.asarray(bolds[0]))

    print("=" * 70)
    print(f"  EIB effective-connectivity tuning — subj {sid}  N={N}  "
          f"target=group FC")
    print(f"  fixed op-point: g_LRE={args.g_lre} g_FFI={args.g_ffi} "
          f"sigma={args.sigma} I_o={args.io} w_p={args.w_p}")
    print(f"  n_iter={args.n_iter} eta={args.eta} sim_s={args.sim_s} "
          f"refic_every={args.refic_every} wmax={args.wmax}")
    print("=" * 70, flush=True)

    burn_tune = min(15.0, max(5.0, args.sim_s * 0.25))   # tuning burn-in < sim_s
    W = SC.copy().astype(np.float64)
    Ji = compute_fic_ji(W, theta, pn, J_N=config.RWWEIB2_FIXED["J_N"], verbose=True)
    traj = []
    for it in range(args.n_iter):
        fcm = run(W, Ji, args.sim_s, burn_tune)
        c = fc_corr(fcm)
        traj.append(c)
        # EIB Hebbian update (annealed): strengthen where FC too low
        eta = args.eta * (it + 1) / args.n_iter
        diff = np.nan_to_num(FC_t - fcm)
        W = W + eta * diff
        W = np.clip(W, 0.0, args.wmax)
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 0.0)
        if (it + 1) % args.refic_every == 0:
            Ji = compute_fic_ji(W, theta, pn, J_N=config.RWWEIB2_FIXED["J_N"], verbose=False)
        if it < 5 or (it + 1) % 10 == 0:
            print(f"  iter {it+1:4d}/{args.n_iter}  FC corr={c:+.4f}  "
                  f"(best {max(traj):+.4f})  W=[{W.min():.2f},{W.max():.2f}]", flush=True)

    # final eval at full sim length
    Ji = compute_fic_ji(W, theta, pn, J_N=config.RWWEIB2_FIXED["J_N"], verbose=False)
    fc_final = run(W, Ji, float(config.T_END) / 1000.0, float(config.T_CUT) / 1000.0)
    c_final = fc_corr(fc_final)
    print("=" * 70)
    print(f"  start FC corr (SC coupling): {traj[0]:+.4f}")
    print(f"  best tuning FC corr        : {max(traj):+.4f}")
    print(f"  FINAL (full sim) FC corr   : {c_final:+.4f}")
    wpath = os.path.join(config.OUTPUT_DIR, "eib_W.npy")
    np.save(wpath, W)
    print(f"  saved effective connectome -> {wpath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
