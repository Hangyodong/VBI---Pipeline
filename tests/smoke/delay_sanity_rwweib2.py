#!/usr/bin/env python
"""Stage 0 — delay GPU sanity for the active rwweib2 / cuBNM engine.

Question this answers: does turning conduction delays ON actually change the
simulated FC, i.e. does ``RWWEIB_2CPLSimGroup`` truly consume the per-subject
delay matrix (fed as ``sc_dist`` + ``v``)? If delay-ON and delay-OFF produce
IDENTICAL FC (same theta, same sim_seed), the cuBNM build is silently ignoring
``sc_dist`` and decision-8 (delay in final pipeline) is INVALID until a
delay-capable engine is built/routed.

Run on a GPU node (this engine forces force_gpu=True):

    PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
        SC_DATASET=cabnp381 SC_FILE=HCP_CABNP381_SC_first100.mat \
        python tests/smoke/delay_sanity_rwweib2.py

Method (isolates delay, nothing else):
  * one subject's SC_weight + tract-length-derived delay matrix
  * a small batch of IDENTICAL homogeneous params (g_LRE/g_FFI/I_o/sigma broadcast)
  * sim_seed fixed (42) so the noise realization is identical across the two runs
  * run A: config.USE_DELAYS=False  -> simulate
  * run B: config.USE_DELAYS=True   -> simulate (same theta, same seed)
  * compare FC_A vs FC_B: if delay is applied they DIFFER; if identical -> FAIL

PASS criterion: mean|FC_on - FC_off| over the upper triangle is clearly above
floating-point noise (>1e-4) AND corr(FC_on, FC_off) < ~0.999 for at least one
sim. Also reports that RWWEIB_2CPLSimGroup accepted sc_dist without error.
"""
import os
import sys

import numpy as np

# repo root on sys.path (this file lives in tests/smoke/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")
os.environ.setdefault("SC_DATASET", "cabnp381")
os.environ.setdefault("SC_FILE", "HCP_CABNP381_SC_first100.mat")

from pipeline_setup import PipelineConfig, setup_pipeline

cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi",
    OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat",
    SC_FILE=os.environ["SC_FILE"],
    N_REGIONS=360,
    N_SUBJECTS=4, N_TRAIN=2, N_VAL=1, N_TEST=1,
    N_SIM=8, GPU_BATCH=8,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
    SBI_DEVICE="cuda",
)
setup_pipeline(cfg)

import config
config.INFERENCE_MODEL = "rwweib2"
config.SC_DATASET = "cabnp381"
config.STAGE1_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
config.RWWEIB2_FIXED = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
                        "J_N": 0.15, "lambda_IE": 1.0}
config.VELOCITY_M_PER_S = 3.0

import data_loader_hcp as data_loader
from features.fc import compute_fc, fc_to_upper_tri
from cuBNM import simulate_rwweib_2cpl as eng


def _load_one_subject():
    df, fc_mat, sc_mat, fc_ids, sc_ids, *_ = data_loader.load_raw_data()
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    sid = subjects[0]
    d = data_loader.load_all_subjects([sid], fc_mat, sc_mat, fc_ids, sc_ids)[sid]
    return sid, d


def _simulate(sc, delays, theta, use_delays):
    """Run the production wrapper with config.USE_DELAYS toggled."""
    config.USE_DELAYS = bool(use_delays)
    # reset the module-level one-shot delay warning so both runs print status
    eng._DELAY_WARNED = False
    bolds = eng.simulate_gpu_batch(
        sc, theta, param_names=list(config.STAGE1_PARAMS),
        delays=delays, apply_bw=True, label=f"delay={use_delays}", n_total=len(theta),
    )
    return [np.asarray(b, dtype=np.float64) for b in bolds]


def main():
    sid, d = _load_one_subject()
    sc = np.ascontiguousarray(d["sc"], dtype=np.float64)
    delays = np.ascontiguousarray(d["delays"], dtype=np.float64)
    n = sc.shape[0]
    dpos = delays[delays > 0]
    dt = float(config.DT)
    steps = np.round(dpos / dt)
    print("=" * 70)
    print(f"  Stage 0 delay sanity — subject {sid}, N={n}")
    print("=" * 70)
    print(f"  delay matrix: nonzero={dpos.size}  "
          f"range=[{dpos.min():.2f}, {dpos.max():.2f}] ms  "
          f"(= length_mm / {config.VELOCITY_M_PER_S} m/s)")
    print(f"  delay_steps = round(delay_ms / dt={dt}ms): "
          f"[{int(steps.min())}, {int(steps.max())}]")

    n_sims = 8
    # homogeneous params (broadcast to all nodes by build_param_lists); identical
    # for both runs so any FC difference is delay-only.
    theta = np.tile(np.array([1.5, 1.5, 0.5, 0.025], dtype=np.float64),
                    (n_sims, 1))

    try:
        bolds_off = _simulate(sc, delays, theta, use_delays=False)
        bolds_on = _simulate(sc, delays, theta, use_delays=True)
    except Exception as e:
        print(f"\n  [FAIL] engine raised: {type(e).__name__}: {e}")
        print("  -> RWWEIB_2CPLSimGroup likely does NOT support sc_dist in this "
              "cubnm build. Decision-8 (delay) needs a delay-capable engine.")
        raise SystemExit(2)

    iu = np.triu_indices(n, k=1)
    deltas, corrs = [], []
    for i in range(n_sims):
        fc_off = compute_fc(bolds_off[i])[iu]
        fc_on = compute_fc(bolds_on[i])[iu]
        m = np.isfinite(fc_off) & np.isfinite(fc_on)
        if m.sum() < 10:
            continue
        deltas.append(float(np.mean(np.abs(fc_off[m] - fc_on[m]))))
        if fc_off[m].std() > 0 and fc_on[m].std() > 0:
            corrs.append(float(np.corrcoef(fc_off[m], fc_on[m])[0, 1]))

    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    min_corr = float(np.min(corrs)) if corrs else 1.0
    print(f"\n  FC(delay=OFF) vs FC(delay=ON), {len(deltas)} sims, same seed:")
    print(f"    mean |ΔFC| (upper-tri) = {mean_delta:.6f}")
    print(f"    min  corr(FC_off,FC_on) = {min_corr:.6f}")

    applied = (mean_delta > 1e-4) and (min_corr < 0.999)
    print("\n" + "=" * 70)
    if applied:
        print("  [PASS] delay IS applied by rwweib2/cuBNM "
              "(delay-ON changes FC). Decision-8 valid.")
    else:
        print("  [FAIL] delay-ON FC ~ identical to delay-OFF "
              "-> sc_dist is being IGNORED by the engine. Decision-8 INVALID; "
              "route to a delay-capable engine (e.g. rwweibdelay) or fix the build.")
    print("=" * 70)
    raise SystemExit(0 if applied else 1)


if __name__ == "__main__":
    main()
