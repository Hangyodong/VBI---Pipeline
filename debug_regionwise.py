#!/usr/bin/env python
"""One-shot debug for 4-param region-wise g_LRE (run AFTER cuBNM rebuild, on GPU).

Checks, in order:
  1. rebuilt RWWEIB_2CPL has g_LRE as a regional_param (0 global_params).
  2. decoder: random z -> 4 per-region maps, all within bounds incl g_LRE[0,9].
  3. build_param_lists: g_LRE injected as (n_sims, n_nodes) per-node matrix.
  4. GPU sim (latent path): 4 matrices incl g_LRE_matrix -> FC, no crash.
  5. homogeneous path still works (g_LRE from theta, broadcast uniformly).

    python debug_regionwise.py
"""
import os
import numpy as np

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")
from pipeline_setup import PipelineConfig, setup_pipeline

setup_pipeline(PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat", SC_FILE="HCP_SC.mat",
    N_REGIONS=360, N_SUBJECTS=2, N_TRAIN=1, N_VAL=1, N_TEST=0, SEED=42,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
))
import config
config.INFERENCE_MODEL = "rwweib2"
config.HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
config.HETERO_BOUNDS = {"g_LRE": (0.0, 9.0), "g_FFI": (0.0, 9.0),
                        "I_o": (0.15, 0.60), "sigma": (0.0, 0.09)}
config.N_LAPLACIAN_BASIS = 4
config.USE_NETWORK_BASIS = True
config.NETWORK_LABELS_CSV = None

import data_loader_hcp as dl
from region_basis import build_region_basis
from param_decoder import (decode_latent_to_param_maps, latent_dim,
                           make_fixed_overrides_from_param_maps)
from cuBNM.runner_rwweib_2cpl import (build_param_lists, run_cubnm_rwweib2_batch,
                                      _import_rwweib2)
from cuBNM.fc import compute_fc

PASS, FAIL = "PASS", "FAIL"
def chk(name, ok, extra=""):
    print(f"  [{PASS if ok else FAIL}] {name}{('  '+extra) if extra else ''}")
    return ok

print("=" * 66)
print("  debug_regionwise — 4-param region-wise g_LRE")
print("=" * 66)
all_ok = True

# ---- 1. model has g_LRE regional / 0 globals ----
Grp = _import_rwweib2()
ng = getattr(Grp, "n_global_params", None)
gpn = list(getattr(Grp, "global_param_names", []) or [])
rpn = list(getattr(Grp, "regional_param_names", []) or [])
print(f"  model: n_global_params={ng}  global={gpn}")
print(f"         regional={rpn}")
ok = ("g_LRE" in rpn) and ("g_LRE" not in gpn)
if ng is not None:
    ok = ok and (ng == 0)
all_ok &= chk("g_LRE is regional (0 globals)", ok,
              "<- if FAIL: cuBNM not rebuilt from updated yaml")

# ---- data: one real subject SC ----
o = dl.load_raw_data()
sid = dl.get_target_subjects(o[0], o[3], o[4])[0]
d = dl.load_all_subjects([sid], o[1], o[2], o[3], o[4])[sid]
sc = d["sc"]; N = sc.shape[0]; iu = np.triu_indices(N, 1)

# ---- 2. decoder bounds (incl g_LRE) ----
basis = build_region_basis(sc, labels=None, config=config)
ld = latent_dim(basis, config)
nS = 4
rng = np.random.RandomState(0)
z = rng.uniform(-1, 1, size=(nS, ld))
maps = decode_latent_to_param_maps(z, basis, config)
print(f"  latent_dim={ld} (expect 4x(1+0+4)=20)")
ok = (ld == 20)
for p in config.HETERO_PARAMS:
    lo, hi = config.HETERO_BOUNDS[p]
    m = maps[p]
    inb = m.shape == (nS, N) and m.min() >= lo - 1e-9 and m.max() <= hi + 1e-9
    ok &= inb
    print(f"      {p:6s} shape={m.shape} range=[{m.min():.3f},{m.max():.3f}] bound=[{lo},{hi}]")
all_ok &= chk("decode -> 4 maps within bounds (incl g_LRE[0,9])", ok)

# ---- 3. build_param_lists g_LRE per-node ----
ov = make_fixed_overrides_from_param_maps(maps)
ov["seed"] = 42
pl = build_param_lists(np.zeros((nS, 0)), [], N, fixed=ov)
glre = pl["g_LRE"]
ok = glre.shape == (nS, N) and np.allclose(glre, maps["g_LRE"])
all_ok &= chk("build_param_lists g_LRE is per-node (n_sims,n_nodes)", ok,
              f"shape={glre.shape}")

# ---- 4. GPU sim latent path (4 matrices) ----
try:
    bolds = run_cubnm_rwweib2_batch(sc, np.zeros((nS, 0)), [], duration_s=90.0,
                                    burn_in_s=15.0, fixed=ov, force_gpu=True, hrf="bw")
    fc = compute_fc(np.asarray(bolds[0]))
    ok = len(bolds) == nS and np.isfinite(fc[iu]).all()
    all_ok &= chk("GPU sim w/ 4 region-wise matrices -> finite FC", ok,
                  f"n_bold={len(bolds)} fc_dim={fc[iu].size}")
except Exception as e:
    all_ok &= chk("GPU sim w/ 4 region-wise matrices", False, f"{type(e).__name__}: {e}")

# ---- 5. homogeneous path (g_LRE from theta, uniform) ----
try:
    pn = ["g_LRE", "g_FFI", "I_o", "sigma"]
    th = np.array([[1.5, 1.0, 0.35, 0.01]] * nS)
    plh = build_param_lists(th, pn, N)
    homo_ok = plh["g_LRE"].shape == (nS, N) and np.allclose(plh["g_LRE"], 1.5)
    bh = run_cubnm_rwweib2_batch(sc, th, pn, duration_s=90.0, burn_in_s=15.0,
                                 force_gpu=True, hrf="bw")
    homo_ok &= len(bh) == nS
    all_ok &= chk("homogeneous path (g_LRE uniform 1.5) -> sim OK", homo_ok)
except Exception as e:
    all_ok &= chk("homogeneous path", False, f"{type(e).__name__}: {e}")

print("=" * 66)
print(f"  RESULT: {'ALL PASS — 4-param g_LRE region-wise ready' if all_ok else 'FAIL (see above)'}")
print("=" * 66)
