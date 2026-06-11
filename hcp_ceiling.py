#!/usr/bin/env python
"""RWW-EIB-FFI ceiling for HCP — max real-FC corr reachable by parameter search.

The decisive diagnostic after the engine-routing fix. The final test gives
FC corr ~0 from the *posterior-mean* params. That cannot tell apart:
  (A) structure-function misspec — RWW on HCP SC simply cannot reproduce the
      subject's FC spatial pattern at ANY params  (ceiling stays < ~0.1)
  (B) posterior is OOD/bad — model CAN fit but the amortized net returns wrong
      params on real (out-of-distribution) FC          (ceiling >> posterior)

This script measures the ceiling directly: per subject, draw N random params
from the STAGE1 prior, simulate via the RWW-EIB-FFI engine, and keep the best
corr(sim FC, real FC). High ceiling -> fix the posterior. Low ceiling -> fix
the model/SC/parcellation.

GPU required (cuBNM) — run on the GPU node, not in Claude's sandbox.

Usage
-----
    python hcp_ceiling.py                      # 5 test subjects, 1000 samples
    python hcp_ceiling.py --subjects 5 --samples 2000
    python hcp_ceiling.py --ids 113922,118225  # specific subject ids

Writes output_hcp/joint_opt_group.npz (same schema as the mouse joint_opt, so
evaluate_project.py's scientific-validity check reads it automatically) plus
per-subject npz.
"""
import argparse
import time

import numpy as np

# ── replicate main_HCP setup so config matches the trained pipeline ──────────
from pipeline_setup import PipelineConfig, setup_pipeline

_cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi",
    OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat",
    SC_FILE="HCP_SC.mat",
    N_REGIONS=381,
    N_SUBJECTS=100, N_TRAIN=80, N_VAL=0, N_TEST=20, SEED=42,
    N_SIM=2_000, GPU_BATCH=2_000,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
    HRF_A1=3.0, HRF_A2=7.0, HRF_L=1.0, HRF_C=0.3,
    HRF_LENGTH_SEC=32.0, HRF_LENGTH_MS=20_000.0,
    EMBED_DIM=128, REGION_TRANSFORMER_HEADS=4, REGION_TRANSFORMER_LAYERS=2,
    REGION_TRANSFORMER_D_MODEL=128, SELECTION_K=300,
    SBI_DEVICE="cuda", N_POSTERIOR=2000, N_SBC=200,
    NDE_HIDDEN=128, NDE_TRANSFORMS=8, N_TEST_RESIM=10,
)
setup_pipeline(_cfg)

import config

config.INFERENCE_MODEL = "rwweib"
config.STAGE1_PARAMS = ["g_LRE", "g_FFI", "sigma", "I_o"]
config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
config.STAGE1_PRIOR_LOW = [0.0, 0.0, 0.0, 0.30]
config.STAGE1_PRIOR_HIGH = [3.0, 3.0, 0.03, 0.45]
config.VELOCITY_M_PER_S = 3.0
config.USE_FCD = False
config.USE_DELAYS = False

import data_loader_hcp as data_loader  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _masked_corr(sim_vec, real_vec, nan_vec):
    """Pearson corr on finite, non-NaN-FC upper-tri entries."""
    m = np.isfinite(sim_vec) & np.isfinite(real_vec) & (~nan_vec)
    if m.sum() < 2:
        return np.nan
    a, b = sim_vec[m], real_vec[m]
    if a.std() <= 0 or b.std() <= 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _sim_fc_batch(weights, theta_batch, delays):
    """One RWW-EIB-FFI batch -> list of FC matrices (engine = INFERENCE_MODEL)."""
    from engine_select import get_simulate_gpu_batch
    from simulator import compute_fc
    simulate_gpu_batch = get_simulate_gpu_batch()
    bolds = simulate_gpu_batch(
        weights, theta_batch, param_names=config.STAGE1_PARAMS,
        delays=delays, apply_bw=True,
    )
    return [compute_fc(np.asarray(b)) for b in bolds]


def _sample_prior(n, rng):
    lo = np.asarray(config.STAGE1_PRIOR_LOW, dtype=np.float64)
    hi = np.asarray(config.STAGE1_PRIOR_HIGH, dtype=np.float64)
    return rng.uniform(lo, hi, size=(n, len(lo)))


# --------------------------------------------------------------------------
# per-subject ceiling
# --------------------------------------------------------------------------

def run_subject(sid, d, samples, seed):
    weights, delays = d["sc"], d["delays"]
    fc_real = d["fc"]
    N = fc_real.shape[0]
    iu = np.triu_indices(N, k=1)
    real_vec = fc_real[iu]
    nan_vec = (d["fc_nan"][iu] if d.get("fc_nan") is not None
               else np.zeros(iu[0].shape, dtype=bool))

    rng = np.random.RandomState(seed)
    theta = _sample_prior(samples, rng)

    print("=" * 70)
    print(f"  RWW-EIB-FFI ceiling — subject {sid}")
    print(f"  params: {config.STAGE1_PARAMS}   samples: {samples}   "
          f"valid FC: {int((~nan_vec).sum())}/{len(nan_vec)}")
    print("=" * 70, flush=True)

    t0 = time.time()
    fc_list = _sim_fc_batch(weights, theta, delays)
    corrs = np.array([_masked_corr(fc[iu], real_vec, nan_vec)
                      for fc in fc_list], dtype=float)
    corrs = np.where(np.isfinite(corrs), corrs, -1.0)
    dt = time.time() - t0

    order = np.argsort(-corrs)
    best_i = int(order[0])
    best_corr = float(corrs[best_i])
    best_p = theta[best_i]

    print(f"  best corr = {best_corr:+.4f}   ({dt:.0f}s)")
    print("  top 5:")
    for r in order[:5]:
        ps = "  ".join(f"{k}={theta[r, j]:.3f}"
                       for j, k in enumerate(config.STAGE1_PARAMS))
        print(f"    corr={corrs[r]:+.4f}   {ps}")

    # per-param influence on fit (|corr(param, fit)|)
    infl = {}
    for j, k in enumerate(config.STAGE1_PARAMS):
        col = theta[:, j]
        infl[k] = (abs(np.corrcoef(col, corrs)[0, 1]) if col.std() > 0
                   else float("nan"))
    print("  param influence: "
          + "  ".join(f"{k}={infl[k]:.3f}" for k in config.STAGE1_PARAMS),
          flush=True)

    out = f"{config.OUTPUT_DIR}/joint_opt_{sid}"
    np.savez(out + ".npz", subject=str(sid),
             param_names=np.array(config.STAGE1_PARAMS),
             params=theta, corrs=corrs,
             best_params=best_p, best_corr=best_corr)
    return {"subject": str(sid), "best_corr": best_corr, "best_params": best_p}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="RWW-EIB-FFI HCP FC ceiling.")
    ap.add_argument("--subjects", type=int, default=5,
                    help="number of test subjects to measure (default 5)")
    ap.add_argument("--ids", type=str, default=None,
                    help="comma-separated subject ids (overrides --subjects)")
    ap.add_argument("--samples", type=int, default=1000,
                    help="random param draws per subject (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    train, val, test = data_loader.three_way_split(subjects)

    if args.ids:
        chosen = [int(x) for x in args.ids.split(",") if x]
    else:
        chosen = test[:args.subjects]

    subject_data = data_loader.load_all_subjects(
        chosen, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids,
    )

    print(f"\nMeasuring ceiling for {len(chosen)} subjects: {chosen}\n", flush=True)
    results = []
    for sid in chosen:
        results.append(run_subject(sid, subject_data[sid], args.samples, args.seed))

    corrs = np.array([r["best_corr"] for r in results])
    subs = np.array([r["subject"] for r in results])
    best_params = np.array([r["best_params"] for r in results])
    np.savez(f"{config.OUTPUT_DIR}/joint_opt_group.npz",
             subjects=subs, best_corrs=corrs,
             param_names=np.array(config.STAGE1_PARAMS),
             best_params=best_params)

    print("\n" + "=" * 70)
    print("  RWW-EIB-FFI CEILING — group summary")
    print("=" * 70)
    for r in results:
        print(f"  {r['subject']:>10}  best_corr = {r['best_corr']:+.4f}")
    print("-" * 70)
    print(f"  mean={corrs.mean():+.4f}  median={np.median(corrs):+.4f}  "
          f"max={corrs.max():+.4f}  min={corrs.min():+.4f}")
    print("=" * 70)
    if corrs.mean() < 0.10:
        print("  VERDICT: ceiling ~0 -> STRUCTURAL misspec. RWW on HCP SC")
        print("  cannot reproduce FC pattern at any params. Fix model/SC/atlas,")
        print("  NOT the posterior.")
    elif corrs.mean() < 0.30:
        print("  VERDICT: low ceiling. Model weakly fits; posterior not the")
        print("  bottleneck. Tune operating point (g_LRE range, I_o) / SC.")
    else:
        print("  VERDICT: ceiling OK but posterior gives ~0 -> POSTERIOR/OOD")
        print("  problem. Model can fit; amortized net fails on real FC.")
        print("  Fix: prior range, embedding, or sim-vs-real feature gap.")
    print("=" * 70)


if __name__ == "__main__":
    main()
