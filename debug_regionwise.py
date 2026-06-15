#!/usr/bin/env python
"""One-shot debug for 4-param region-wise g_LRE (run AFTER cuBNM rebuild, on GPU).

PART A — unit checks (fast, ~8 sims):
  1. rebuilt RWWEIB_2CPL has g_LRE regional (0 global_params).
  2. decoder: random z -> 4 per-region maps within bounds incl g_LRE[0,9].
  3. build_param_lists: g_LRE injected per-node (n_sims, n_nodes).
  4. GPU sim (latent path): 4 region-wise matrices -> finite FC.
  5. homogeneous path still works (g_LRE from theta, uniform).

PART B — end-to-end smoke (runs main_HCP at tiny scale, then verifies output):
  6. latent_regionwise + subject FC pipeline completes.
  7. param_maps.npz shape (n_subjects, N_REGIONS, 4).
  8. param order == [g_LRE, g_FFI, I_o, sigma].
  9. g_LRE map values within [0, 9].

    python debug_regionwise.py            # A + B
    python debug_regionwise.py --quick    # A only (no main_HCP run)
"""
import argparse
import os
import subprocess
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="unit checks only (skip main_HCP smoke)")
args = ap.parse_args()

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
_results = []
def chk(name, ok, extra=""):
    _results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}{('  '+extra) if extra else ''}")
    return ok

print("=" * 66)
print("  debug_regionwise — 4-param region-wise g_LRE (A: units, B: smoke)")
print("=" * 66)

# ============ PART A — unit checks ============
print("\n--- PART A: unit checks ---")
Grp = _import_rwweib2()
ng = getattr(Grp, "n_global_params", None)
gpn = list(getattr(Grp, "global_param_names", []) or [])
rpn = list(getattr(Grp, "regional_param_names", []) or [])
print(f"  model: n_global_params={ng}  global={gpn}")
print(f"         regional={rpn}")
ok = ("g_LRE" in rpn) and ("g_LRE" not in gpn)
if ng is not None:
    ok = ok and (ng == 0)
chk("1. g_LRE is regional (0 globals)", ok, "<- FAIL: cuBNM not rebuilt")

o = dl.load_raw_data()
sid = dl.get_target_subjects(o[0], o[3], o[4])[0]
d = dl.load_all_subjects([sid], o[1], o[2], o[3], o[4])[sid]
sc = d["sc"]; N = sc.shape[0]; iu = np.triu_indices(N, 1)

basis = build_region_basis(sc, labels=None, config=config)
ld = latent_dim(basis, config)
nS = 4
z = np.random.RandomState(0).uniform(-1, 1, size=(nS, ld))
maps = decode_latent_to_param_maps(z, basis, config)
ok = (ld == 20)
for p in config.HETERO_PARAMS:
    lo, hi = config.HETERO_BOUNDS[p]
    m = maps[p]
    ok &= m.shape == (nS, N) and m.min() >= lo - 1e-9 and m.max() <= hi + 1e-9
    print(f"      {p:6s} range=[{m.min():.3f},{m.max():.3f}] bound=[{lo},{hi}]")
chk(f"2. decode -> 4 maps within bounds (latent_dim={ld})", ok)

ov = make_fixed_overrides_from_param_maps(maps); ov["seed"] = 42
pl = build_param_lists(np.zeros((nS, 0)), [], N, fixed=ov)
chk("3. build_param_lists g_LRE per-node",
    pl["g_LRE"].shape == (nS, N) and np.allclose(pl["g_LRE"], maps["g_LRE"]))

try:
    b = run_cubnm_rwweib2_batch(sc, np.zeros((nS, 0)), [], duration_s=90.0,
                                burn_in_s=15.0, fixed=ov, force_gpu=True, hrf="bw")
    fc = compute_fc(np.asarray(b[0]))
    chk("4. GPU sim w/ 4 region-wise matrices -> finite FC",
        len(b) == nS and np.isfinite(fc[iu]).all())
except Exception as e:
    chk("4. GPU sim w/ 4 region-wise matrices", False, f"{type(e).__name__}: {e}")

try:
    pn = ["g_LRE", "g_FFI", "I_o", "sigma"]; th = np.array([[1.5, 1.0, 0.35, 0.01]] * nS)
    bh = run_cubnm_rwweib2_batch(sc, th, pn, duration_s=90.0, burn_in_s=15.0,
                                 force_gpu=True, hrf="bw")
    chk("5. homogeneous path (g_LRE uniform) -> sim OK", len(bh) == nS)
except Exception as e:
    chk("5. homogeneous path", False, f"{type(e).__name__}: {e}")

# ============ PART B — end-to-end smoke ============
if not args.quick:
    print("\n--- PART B: end-to-end smoke (main_HCP, tiny) ---")
    out_dir = config.OUTPUT_DIR
    for f in ("features_stage1.npz", "features_stage1_meta.json", "param_maps.npz"):
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass
    env = dict(os.environ)
    env.update(PARAMETER_MODE="latent_regionwise", GROUP_AVG_FC="0",
               N_SUBJECTS="4", N_TRAIN="2", N_VAL="1", N_TEST="1",
               N_SIM="5", GPU_BATCH="2", VBI_SC_SCALE="maxnorm")
    print("  running: main_HCP.py (latent_regionwise, subject FC, tiny) ...", flush=True)
    r = subprocess.run([sys.executable, "main_HCP.py"], env=env,
                       capture_output=True, text=True)
    ok_run = (r.returncode == 0)
    if not ok_run:
        print("  --- main_HCP stderr tail ---")
        print("\n".join(r.stderr.strip().splitlines()[-15:]))
    chk("6. latent_regionwise main_HCP completed", ok_run)

    pm_path = os.path.join(out_dir, "param_maps.npz")
    if ok_run and os.path.exists(pm_path):
        npz = np.load(pm_path, allow_pickle=True)
        pm = npz["param_maps"]; order = list(npz["param_order"])
        chk("7. param_maps shape (n_subj, N_REGIONS, 4)",
            pm.ndim == 3 and pm.shape[1] == config.N_REGIONS and pm.shape[2] == 4,
            f"shape={pm.shape}")
        chk("8. param order [g_LRE,g_FFI,I_o,sigma]",
            order == ["g_LRE", "g_FFI", "I_o", "sigma"], f"{order}")
        gi = order.index("g_LRE") if "g_LRE" in order else 0
        gmap = pm[:, :, gi]
        chk("9. g_LRE map within [0,9]",
            gmap.min() >= -1e-9 and gmap.max() <= 9 + 1e-9,
            f"range=[{gmap.min():.3f},{gmap.max():.3f}]")
    else:
        chk("7-9. param_maps.npz present", False, "missing (run failed?)")

print("=" * 66)
n_ok = sum(_results); n = len(_results)
print(f"  RESULT: {n_ok}/{n}  "
      f"{'ALL PASS' if n_ok == n else 'FAIL (see above)'}")
print("=" * 66)
sys.exit(0 if n_ok == n else 1)
