#!/usr/bin/env python
"""GPU-feasibility verification for basis_regionwise RWW-EIB 2-coupling.

Reproduces main_HCP.py's basis-mode config + startup banner, checks decoder
shapes, and drives the SMALLEST possible cuBNM 2cpl run with decoded per-region
<param>_matrix overrides. No SNPE training, no full-length sim, tiny batch.
"""
import os
import numpy as np

# Allow running from repo root: `python tests/smoke/verify_basis_regionwise_rwweib2.py`
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config

# ── Mirror main_HCP.py basis_regionwise config (minus GPU/data deps) ──────────
config.PARAMETER_MODE = os.environ.get("PARAMETER_MODE", "basis_regionwise")
config.INFERENCE_MODEL = os.environ.get("INFERENCE_MODEL", "rwweib2")
config.N_REGIONS = 360
config.HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
config.BASIS_BOUNDS = {"g_LRE": (0.0, 3.0), "g_FFI": (0.0, 3.0),
                       "I_o": (0.0, 1.0), "sigma": (0.0, 0.05)}
config.HETERO_BOUNDS = dict(config.BASIS_BOUNDS)
config.BASIS_COEFF_PRIOR = (-2.0, 2.0)
# requested path; fall back to repo-local if /mnt/d not mounted on this shell
_req = os.environ.get("BASIS_PATH", "/mnt/d/hcp_basis/basis.npy")
config.BASIS_PATH = _req if os.path.exists(_req) else "basis.npy"
config.BASIS_REZSCORE = True
config.RWWEIB2_FIXED = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
                        "J_N": 0.15, "lambda_IE": 1.0}
config.USE_DELAYS = False

from basis_decoder import get_decoder
from param_decoder import decode_to_param_maps, make_fixed_overrides_from_param_maps

print("\n########## (1) IMPORT CHECK ##########")
try:
    from cubnm.sim import RWWEIB_2CPLSimGroup
    print("  from cubnm.sim import RWWEIB_2CPLSimGroup -> OK")
    print(f"  class = {RWWEIB_2CPLSimGroup}")
except Exception as e:
    RWWEIB_2CPLSimGroup = None
    print(f"  IMPORT FAILED: {type(e).__name__}: {e}")

# ── (2) Startup banner (verbatim logic from main_HCP.py) ──────────────────────
print("\n########## (2) STARTUP BANNER ##########")
if config.BASIS_PATH != _req:
    print(f"  NOTE: requested BASIS_PATH={_req!r} not found on this shell; "
          f"using repo-local {config.BASIS_PATH!r} (same (381,3) artifact).")
_pm = str(getattr(config, "PARAMETER_MODE", "homogeneous"))
try:
    if _pm == "basis_regionwise":
        _theta_dim = get_decoder(config).theta_dim
    elif _pm == "direct_regionwise":
        _theta_dim = len(config.HETERO_PARAMS) * int(config.N_REGIONS)
    elif _pm == "homogeneous":
        _theta_dim = len(config.STAGE1_PARAMS)
    else:
        _theta_dim = "set after Step 1 (SC-dependent)"
except Exception as _e:
    _theta_dim = f"?? ({type(_e).__name__}: {_e})"
print("=" * 70)
print("  [HCP] ACTIVE INFERENCE CONFIG")
print("=" * 70)
print(f"    PARAMETER_MODE  : {_pm}")
print(f"    INFERENCE_MODEL : {getattr(config, 'INFERENCE_MODEL', '?')}")
print(f"    N_REGIONS       : {getattr(config, 'N_REGIONS', '?')}")
print(f"    BASIS_PATH      : {getattr(config, 'BASIS_PATH', '?')}")
print(f"    theta_dim       : {_theta_dim}")
print("=" * 70)
_REQUIRE_BASIS = (os.environ.get("REQUIRE_BASIS", "1") == "1")
print(f"  guard: REQUIRE_BASIS={int(_REQUIRE_BASIS)}  mode_ok="
      f"{_pm == 'basis_regionwise'}  -> "
      f"{'PASS (would proceed)' if (not _REQUIRE_BASIS or _pm=='basis_regionwise') else 'WOULD RAISE'}")

# ── (3) Decoder shape check ───────────────────────────────────────────────────
print("\n########## (3) DECODER SHAPE CHECK ##########")
dec = get_decoder(config)
print(f"  active basis shape : {dec.basis.shape}  (expect (360, 3))")
print(f"  theta_dim          : {dec.theta_dim}   (expect 12)")
single = dec.decode(np.zeros(dec.theta_dim))
print(f"  decode(zeros[12]) single -> g_LRE map shape {single['g_LRE'].shape} "
      f"(expect (360,))")
for p in config.HETERO_PARAMS:
    v = np.asarray(single[p])
    print(f"    {p:7s} z=0 midpoint -> {float(v[0]):.4f} (uniform={v.max()-v.min() < 1e-9})")
S = 4
batch = dec.decode(np.random.RandomState(0).uniform(-2, 2, (S, dec.theta_dim)))
print(f"  decode(rand[{S},12]) batch -> g_LRE map shape {batch['g_LRE'].shape} "
      f"(expect ({S}, 360))")
for p in config.HETERO_PARAMS:
    m = np.asarray(batch[p]); lo, hi = config.HETERO_BOUNDS[p]
    print(f"    {p:7s} in-bounds={lo-1e-9 <= m.min() and m.max() <= hi+1e-9} "
          f"[{m.min():.4f},{m.max():.4f}] bound=[{lo},{hi}]")

# ── (4) Simulator matrix-override check (decode -> overrides) ─────────────────
print("\n########## (4) SIMULATOR MATRIX OVERRIDE CHECK ##########")
N = 2
theta = np.random.RandomState(1).uniform(-2, 2, (N, dec.theta_dim))
maps = decode_to_param_maps(theta, None, config, n_regions=config.N_REGIONS)
overrides = make_fixed_overrides_from_param_maps(maps)
print(f"  override keys fed to simulator: {sorted(overrides)}")
for k in ("g_LRE_matrix", "g_FFI_matrix", "I_o_matrix", "sigma_matrix"):
    print(f"    {k:14s} present={k in overrides} shape={overrides[k].shape}")
# confirm build_param_lists assigns them to the cuBNM param_lists (no GPU)
from cuBNM.runner_rwweib_2cpl import build_param_lists
pl = build_param_lists(np.zeros((N, 0)), [], config.N_REGIONS, fixed=overrides)
for p in config.HETERO_PARAMS:
    ok = np.allclose(pl[p], maps[p])
    print(f"    param_lists[{p:7s}] == decoded map : {ok}  shape={pl[p].shape}")

# ── (5) Smallest real cuBNM 2cpl run (CPU if no GPU) ─────────────────────────
print("\n########## (5) SMALL SMOKE RUN ##########")
if RWWEIB_2CPLSimGroup is None:
    print("  SKIP: RWWEIB_2CPLSimGroup not importable.")
else:
    import torch
    has_gpu = torch.cuda.is_available()
    print(f"  torch.cuda.is_available() = {has_gpu}")
    from cuBNM.runner_rwweib_2cpl import run_cubnm_rwweib2_batch
    nodes = config.N_REGIONS
    rng = np.random.RandomState(7)
    W = np.abs(rng.randn(nodes, nodes)); W = (W + W.T) / 2.0; np.fill_diagonal(W, 0.0)
    W /= W.max()
    try:
        bolds = run_cubnm_rwweib2_batch(
            W, np.zeros((N, 0)), [],
            duration_s=2.0, tr_s=0.72, dt_ms=1.0, burn_in_s=0.0,
            fixed=overrides, force_gpu=bool(has_gpu), hrf="bw",
        )
        b0 = np.asarray(bolds[0])
        print(f"  RUN OK  ({'GPU' if has_gpu else 'CPU'}): {len(bolds)} sims, "
              f"BOLD[0] shape={b0.shape}, finite={np.isfinite(b0).all()}")
    except Exception as e:
        print(f"  RUN FAILED ({'GPU' if has_gpu else 'CPU'}): {type(e).__name__}: {e}")
        print("  (matrix injection already verified in step 4; run needs GPU.)")

print("\n########## DONE ##########")
