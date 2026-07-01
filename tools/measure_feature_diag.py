#!/usr/bin/env python
"""measure_feature_diag.py — is the FC FEATURE PIPELINE the culprit, or is empirical
FC fundamentally outside the simulator's reach (model capacity)?

Decisive question this answers (cheap, CPU-OK if the sim-FC cache exists):

  Fit PCA on SIMULATED FC (exactly as inference/feature_pipeline.py does), then
  push each subject's EMPIRICAL FC through it.

  recon_corr(dim) = corr( empFC , PCA_inv(PCA(empFC)) )      # info PCA keeps
     - rises to ~0.95+ as dim grows           -> TRUNCATION loss = FIXABLE
                                                  (raise FC_PCA_DIM / drop PCA)
     - saturates LOW even at large dim         -> empFC lives OUTSIDE the simulated
                                                  FC span = MODEL CAPACITY limit
                                                  (per-node DOF too small / wrong
                                                   model -> need richer theta / EC)

  ood_z_rms = rms z-score of empFC projected into the sim-PCA space
     - large (>>1)  -> empirical FC is far OOD vs simulated FCs -> the flow
                       extrapolates -> rejection accept collapses (the 0.4-1% seen)

  explained_var(dim) = sim-FC variance captured (sanity; the live run logs 0.742@256)

This is the "PCA" half. Pair with the "raw" half:
    python basis_ceiling.py --n-subjects 1            # model ceiling on RAW empFC (optim, GPU)
Compare:
    raw-FC optim ceiling HIGH + recon_corr saturates LOW  -> PCA/feature pipeline NOT
        the (only) problem; empFC is reachable in raw space but PCA can't represent it
        -> the issue is the feature representation fed to SNPE.
    raw-FC optim ceiling LOW (~0.2) AND recon_corr LOW at full dim -> MODEL capacity
        -> feature fixes won't reach 0.7; need richer theta (1440 / EC).
    raw-FC optim ceiling HIGH + recon_corr HIGH -> feature info is fine; the loss
        objective / OOD / inference is the limiter.

Usage
-----
  # CPU-OK if output_hcp/features_stage1.npz exists (uses cached simulated FC):
  python tools/measure_feature_diag.py
  python tools/measure_feature_diag.py --dims 128,256,512,1024,2048 --n-subjects 5
  python tools/measure_feature_diag.py --n-fit 5000     # PCA fit subsample (RAM)

  # No cache -> must simulate sim FC first (GPU node):
  python tools/measure_feature_diag.py --sim-source simulate --n-sim 2000
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
# Bounds mirror main_HCP.py production (env-tunable); I_o operating-point fix
# defaults to (0.30,0.45). Only matters when --sim-source simulate (cache reuse
# is bound-agnostic). Reproduce OLD with IO_BOUND_LO=0 IO_BOUND_HI=1.
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

N = int(config.N_REGIONS)
IU = np.triu_indices(N, k=1)               # upper-tri (FC_DIM = 64,620 for 360)


def _emp_upper(fc_mat):
    return np.asarray(fc_mat, dtype=np.float64)[IU]


def _corr(a, b, nan_mask=None):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if nan_mask is not None:
        m &= ~nan_mask
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_sim_fc(args):
    """Simulated FC upper-tri matrix (n_samples, FC_DIM) used to FIT the PCA.

    Default: reuse the cached features_stage1.npz (CPU, no GPU). Fall back to a
    fresh GPU simulation only if asked / cache absent.
    """
    cache = os.path.join(config.OUTPUT_DIR, "features_stage1.npz")
    if args.sim_source == "cache" and os.path.exists(cache):
        z = np.load(cache)
        key = "fc_raw" if "fc_raw" in z.files else ("fc" if "fc" in z.files else None)
        if key is None:
            raise KeyError(f"{cache} has no fc_raw/fc (keys={z.files})")
        fc = np.asarray(z[key], dtype=np.float32)
        print(f"  sim FC: cache {cache}  shape={fc.shape}  (key='{key}')")
        return fc
    # simulate (GPU)
    print("  sim FC: simulating fresh batch (GPU required) ...")
    import engine_select
    from basis_decoder import get_decoder
    from features.fc import compute_fc, fc_to_upper_tri
    _dec = get_decoder(config)
    config.STAGE1_PARAMS = _dec.coeff_names()
    config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
    sim_fn = engine_select.get_simulate_gpu_batch()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi = data_loader.load_raw_data()
    sid = data_loader.get_target_subjects(df, fc_ids, sc_ids)[0]
    d = data_loader.load_all_subjects([sid], fc_mat, sc_mat, fc_ids, sc_ids, bm, bi)[sid]
    rng = np.random.RandomState(0)
    theta = rng.uniform(-2, 2, (args.n_sim, _dec.theta_dim)).astype(np.float64)
    delays = d["delays"].astype(np.float64) if config.USE_DELAYS else None
    bolds = sim_fn(d["sc"].astype(np.float64), theta, config.STAGE1_PARAMS,
                   delays=delays, apply_bw=True)
    fc = np.array([fc_to_upper_tri(compute_fc(np.asarray(b))) for b in bolds],
                  dtype=np.float32)
    print(f"  sim FC: simulated  shape={fc.shape}")
    return fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="128,256,512,1024,2048",
                    help="PCA component counts to test")
    ap.add_argument("--n-subjects", type=int, default=5,
                    help="how many empirical subjects (recon_corr averaged)")
    ap.add_argument("--n-fit", type=int, default=5000,
                    help="sim-FC subsample to FIT PCA (RAM; randomized SVD)")
    ap.add_argument("--sim-source", choices=["cache", "simulate"], default="cache")
    ap.add_argument("--n-sim", type=int, default=2000, help="sims if --sim-source simulate")
    args = ap.parse_args()

    from sklearn.decomposition import PCA

    dims = [int(x) for x in args.dims.split(",") if x]

    # empirical FCs (targets)
    df, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi = data_loader.load_raw_data()
    sids = data_loader.get_target_subjects(df, fc_ids, sc_ids)[:args.n_subjects]
    subj = data_loader.load_all_subjects(sids, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi)
    emp = {s: _emp_upper(subj[s]["fc"]) for s in sids}
    nanm = {s: (_emp_upper(subj[s].get("fc_nan", np.zeros((N, N)))) > 0) for s in sids}

    # simulated FCs (fit PCA on these — same as the live pipeline)
    sim_fc = _load_sim_fc(args)
    sim_fc = sim_fc[np.isfinite(sim_fc).all(axis=1)]          # drop non-finite rows
    if sim_fc.shape[0] > args.n_fit:
        ridx = np.random.RandomState(0).choice(sim_fc.shape[0], args.n_fit, replace=False)
        sim_fit = sim_fc[ridx]
    else:
        sim_fit = sim_fc
    max_dim = min(max(dims), sim_fit.shape[0] - 1, sim_fc.shape[1])
    dims = sorted({min(d, max_dim) for d in dims})

    print("=" * 78)
    print("  FEATURE-PIPELINE DIAGNOSTIC  (PCA fit on simulated FC)")
    print("=" * 78)
    print(f"  subjects={sids}")
    print(f"  PCA fit on sim FC: {sim_fit.shape}  | empirical FC_dim={emp[sids[0]].shape[0]}")
    print(f"  target={'group-avg' if config.GROUP_AVG_FC else 'per-subject'}  "
          f"T_END={config.T_END_MS:.0f}ms")
    print("  note: recon_corr/ood_z are whiten-INVARIANT (whitening is invertible).")
    print("        whiten's harm = it feeds the flow the z-coords below as INPUT;")
    print("        low-variance comps -> huge input -> ood_z_rms IS that input scale.")
    print("-" * 78)
    hdr = (f"  {'dim':>5} | {'sim_explvar':>11} | "
           f"{'emp_recon_corr':>14} | {'ood_z_rms':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    rows = []
    for dim in dims:
        pca = PCA(n_components=int(dim), whiten=False,
                  svd_solver="randomized", random_state=0).fit(sim_fit)
        ev = float(pca.explained_variance_ratio_.sum())
        std = np.sqrt(pca.explained_variance_) + 1e-12        # per-comp sim std
        rcs, zs = [], []
        for s in sids:
            e = emp[s][None, :].astype(np.float32)
            x = pca.transform(e)                              # raw projection (1, dim)
            recon = pca.inverse_transform(x)[0]
            rcs.append(_corr(recon, emp[s], nanm[s]))
            z = x[0] / std                                    # intrinsic z = whitened coord
            zs.append(float(np.sqrt(np.mean(z ** 2))))
        rc = float(np.nanmean(rcs)); zr = float(np.mean(zs))
        rows.append((dim, ev, rc, zr))
        print(f"  {dim:>5} | {ev:>11.3f} | {rc:>14.4f} | {zr:>9.2f}")
    print("=" * 78)

    # interpretation
    rc_lo = rows[0][2]; rc_hi = rows[-1][2]
    z_any = max(r[3] for r in rows)
    print("  READOUT:")
    print(f"  - emp_recon_corr  {rows[0][0]}d={rc_lo:.3f}  ->  {rows[-1][0]}d={rc_hi:.3f}")
    if rc_hi >= 0.95:
        print(f"    => empFC IS representable; recon saturates high. PCA TRUNCATION "
              f"was the loss -> raise FC_PCA_DIM (or drop PCA). FEATURE-fixable.")
    elif rc_hi < 0.6:
        print(f"    => empFC stays POORLY reconstructed even at {rows[-1][0]} comps "
              f"-> empirical FC is OUTSIDE the simulated-FC span = MODEL CAPACITY. "
              f"Feature fixes won't reach high corr; need richer theta (1440 / EC).")
    else:
        print(f"    => INTERMEDIATE: partial truncation + partial span mismatch. "
              f"Raise dim AND consider richer model.")
    _ood_note = ("HIGH: empFC far OOD -> accept collapse; whiten worsens input scale"
                 if z_any > 2 else "moderate")
    print(f"  - ood_z_rms max={z_any:.2f}  ({_ood_note})")
    print(f"  - pair with: python basis_ceiling.py --n-subjects 1   "
          f"(model ceiling on RAW empFC)")
    print("=" * 78)


if __name__ == "__main__":
    main()
