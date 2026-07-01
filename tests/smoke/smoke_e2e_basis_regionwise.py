#!/usr/bin/env python
"""Smallest end-to-end smoke of the basis_regionwise INFERENCE path (CPU).

This shell has no CUDA driver, and the production collector
(inference.training_data.collect_training_data) hard-calls cupy
(free_all_blocks) -> unusable here. So we drive the SAME real components
collect_training_data would, minus the cupy wrapper:

  prior/scaler -> sample basis coeffs (theta) -> WRAPPED 2cpl sim
  (engine_select.get_simulate_gpu_batch -> latent_wrap DECODES coeffs into
  per-region g_LRE/g_FFI/I_o/sigma maps -> cuBNM RWWEIB_2CPL, CPU) -> FC
  -> FeaturePipeline -> SNPE-C train -> posterior -> decode -> resim score.

cuBNM is forced onto its CPU fallback. Tiny everything. Mechanics only.
"""
import os
import numpy as np

# Allow running from repo root: `python tests/smoke/smoke_e2e_basis_regionwise.py`
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

# ── 0. CPU fallback for cuBNM (no GPU on this shell) ─────────────────────────
import cuBNM.runner_rwweib_2cpl as _R
_orig_run = _R.run_cubnm_rwweib2_batch
def _run_cpu(*a, **k):
    k["force_gpu"] = False
    return _orig_run(*a, **k)
_R.run_cubnm_rwweib2_batch = _run_cpu

# ── 1. main_HCP-style basis_regionwise config (tiny) ─────────────────────────
N = 12
config.PARAMETER_MODE = "basis_regionwise"
config.INFERENCE_MODEL = "rwweib2"
config.N_REGIONS = N
config.FC_DIM = N * (N - 1) // 2
config.HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
config.BASIS_BOUNDS = {"g_LRE": (0.0, 3.0), "g_FFI": (0.0, 3.0),
                       "I_o": (0.0, 1.0), "sigma": (0.0, 0.05)}
config.HETERO_BOUNDS = dict(config.BASIS_BOUNDS)
config.BASIS_COEFF_PRIOR = (-2.0, 2.0)
config.BASIS_PATH = "basis.npy"
config.BASIS_REZSCORE = True
config.RWWEIB2_FIXED = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
                        "J_N": 0.15, "lambda_IE": 1.0}
config.USE_DELAYS = False
config.USE_FCD = False
config.SBI_DEVICE = "cpu"
config.T_END = 4000.0; config.T_CUT = 0.0; config.DT = 1.0; config.DECIMATE = 1
config.TR_SEC = 0.72
config.ANALYSIS_BOLD_T = int((config.T_END - config.T_CUT) / (config.DT * config.DECIMATE)
                             / (config.TR_SEC * 1000.0 / (config.DT * config.DECIMATE)))
config.N_SIM = 16
config.NDE_HIDDEN = 16; config.NDE_TRANSFORMS = 2; config.NDE_MODEL = "maf"
config.N_POSTERIOR = 32; config.N_TEST_RESIM = 4
config.OUTPUT_DIR = f"/tmp/smoke_hcp_{os.getpid()}"; os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ── 2. basis dispatch (verbatim from main_HCP basis_regionwise branch) ───────
from basis_decoder import get_decoder
_dec = get_decoder(config)
config.STAGE1_PARAMS = _dec.coeff_names()
config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
config.STAGE1_PRIOR_LOW = [config.BASIS_COEFF_PRIOR[0]] * _dec.theta_dim
config.STAGE1_PRIOR_HIGH = [config.BASIS_COEFF_PRIOR[1]] * _dec.theta_dim
print(f"[smoke] PARAMETER_MODE={config.PARAMETER_MODE} INFERENCE_MODEL={config.INFERENCE_MODEL} "
      f"N_REGIONS={N} theta_dim={_dec.theta_dim} basis={_dec.basis.shape}")

# ── 3. synthetic tiny subject (no 1.1GB HCP load) ────────────────────────────
rng = np.random.RandomState(0)
def _subject():
    W = np.abs(rng.randn(N, N)); W = (W + W.T) / 2.0; np.fill_diagonal(W, 0.0); W /= W.max()
    F = rng.uniform(-0.4, 0.8, (N, N)); F = (F + F.T) / 2.0; np.fill_diagonal(F, 0.0)
    return {"sc": W, "fc": F, "fcd": np.zeros((N, N)), "fc_nan": np.zeros((N, N), bool),
            "lengths_mm": np.zeros((N, N)), "delays": np.zeros((N, N))}
subject_data = {0: _subject(), 1: _subject()}
train, test = 0, 1

# ── 4. prior + scaler + sample basis coeffs ──────────────────────────────────
from inference.scaling import make_stage1_param_scaler
from inference.priors import make_scaled_prior
param_scaler = make_stage1_param_scaler()
prior_scaled = make_scaled_prior(_dec.theta_dim, device="cpu")
theta_scaled = prior_scaled.sample((config.N_SIM,)).cpu().numpy().astype(np.float32)
theta_raw = param_scaler.inverse_transform(theta_scaled)   # basis coeffs in [-2,2]

# ── 5. SIMULATE training set via WRAPPED 2cpl sim (decodes coeffs->maps) ──────
import engine_select
assert engine_select.is_regionwise()
simulate_batch = engine_select.get_simulate_gpu_batch()    # latent_wrap(2cpl adapter)
print(f"\n[smoke] === simulate {config.N_SIM} train sims (CPU 2cpl, basis-decoded) ===")
from simulator import compute_fc, fc_to_upper_tri
bolds = simulate_batch(subject_data[train]["sc"], theta_raw, config.STAGE1_PARAMS,
                       delays=subject_data[train]["delays"], apply_bw=True)
fc_raw = np.stack([fc_to_upper_tri(compute_fc(np.asarray(b))) for b in bolds]).astype(np.float32)
fcd_raw = np.zeros((config.N_SIM, 0), np.float32)
print(f"[smoke] fc_raw={fc_raw.shape}  (n_sim, FC_DIM={config.FC_DIM})  finite={np.isfinite(fc_raw).all()}")

# ── 6. feature pipeline + SNPE-C train ───────────────────────────────────────
from inference.feature_pipeline import FeaturePipeline
from inference.snpe import train_snpe
feature_pipeline = FeaturePipeline()
x_input = feature_pipeline.fit_transform(fc_raw, fcd_raw, verbose=False)
print(f"\n[smoke] === SNPE-C train  theta={theta_scaled.shape} x={np.asarray(x_input).shape} (CPU) ===")
posterior, _ = train_snpe(theta_scaled, x_input, prior_scaled,
                          embedding_net=None, use_embedding=False,
                          sc_matrix=None, verbose=False)

# ── 7. posterior @ test -> basis coeffs (12-dim) ─────────────────────────────
from inference import infer_subject_raw
d = subject_data[test]
x_obs = feature_pipeline.transform(fc_to_upper_tri(d["fc"])[None, :], np.zeros((1, 0)))
samples_raw, means_raw, stds_raw, _ = infer_subject_raw(
    posterior, x_obs, param_scaler, n_samples=config.N_POSTERIOR, verbose=False)
print(f"\n[smoke] posterior samples_raw={samples_raw.shape} (expect (32,12) coeffs)")

# ── 8. decode posterior-mean coeffs -> per-region maps ───────────────────────
from param_decoder import decode_to_param_maps
maps = decode_to_param_maps(means_raw[None, :], None, config, n_regions=N)
print("[smoke] decoded posterior-mean maps:")
for p in config.HETERO_PARAMS:
    m = np.asarray(maps[p])[0]; lo, hi = config.HETERO_BOUNDS[p]
    print(f"        {p:7s} shape={m.shape} mean={m.mean():.4f} "
          f"in[{lo},{hi}]={lo-1e-9<=m.min() and m.max()<=hi+1e-9}")

# ── 9. resim from posterior (wrapped batch -> decodes) -> FC corr ─────────────
from evaluation.metrics import _resimulate_and_score
fc_corrs, _, _, _ = _resimulate_and_score(
    config.N_TEST_RESIM, samples_raw, config.STAGE1_PARAMS, None,
    d["sc"], d["delays"], d["fc"], np.zeros(0), True,
    sid=f"smoke:{test}", verbose=False, fc_nan=d["fc_nan"])
print(f"\n[smoke] resim: {len(fc_corrs)}/{config.N_TEST_RESIM} OK  "
      f"FC corr mean={(np.mean(fc_corrs) if fc_corrs else float('nan')):.4f}")
print("\n[smoke] END-TO-END BASIS PIPELINE (CPU): OK")
