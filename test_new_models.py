#!/usr/bin/env python
"""GPU validation for the new cuBNM models after the multi-coupling + delay
kernel surgery. Run on a GPU node AFTER rebuilding cubnm:

    python test_new_models.py

Covers what the CPU codegen checks could NOT:
  T1 contract   : the 3 generated SimGroup classes import with correct params.
  T2 routing    : build_param_lists maps theta -> param_lists (CPU, no GPU).
  T3 regression : single-coupling model (n_conn=1) still deterministic + finite
                  after the kernel change. (If your repo has a pre-surgery FC
                  reference, compare against it; else just assert finite+stable.)
  T4 two-cpl    : RWWEIB_2CPL drives I from SC@S_I (NOT SC@S_E). Proven by:
                  (a) g_FFI=0  -> 2CPL == FFI single-coupling   (both drop I-coupling)
                  (b) g_FFI>0  -> 2CPL != FFI                   (S_I vs S_E coupling)
  T5 delay      : RWWEIB_DELAY with USE_DELAYS=True differs from False, and
                  collapses back to no-delay when the delay matrix -> 0.

All sims use sigma=0 (noise off) + fixed seed => deterministic, exact compares.
"""
import numpy as np

# --- pipeline config (mirror main_HCP enough to drive the engines) ----------
from pipeline_setup import PipelineConfig, setup_pipeline
setup_pipeline(PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat", SC_FILE="HCP_SC.mat",
    N_REGIONS=4, N_SUBJECTS=2, N_TRAIN=1, N_VAL=0, N_TEST=1,
    T_END_MS=10_000.0, T_CUT_MS=2_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
))
import config

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []
def check(name, ok, extra=""):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}  {extra}")


# small deterministic test network (4 nodes, ring) + delay matrix
N = 4
SC = np.array([[0, 1, 0, 1],
               [1, 0, 1, 0],
               [0, 1, 0, 1],
               [1, 0, 1, 0]], dtype=np.float64) * 0.5
lengths = (SC > 0) * 60.0                       # 60 mm where connected
delays = lengths / 3.0                          # tau = L / v(=3 mm/ms) -> 20 ms


def run(engine, theta, params, use_delays, dl=delays, seed=42):
    """Drive one engine via engine_select. Returns (n,T,N) BOLD stack."""
    config.INFERENCE_MODEL = engine
    config.STAGE1_PARAMS = params
    config.USE_DELAYS = use_delays
    from engine_select import get_simulate_gpu_batch
    sim = get_simulate_gpu_batch()
    bolds = sim(SC, np.asarray(theta, float), param_names=params,
                delays=dl, apply_bw=True, fixed_overrides={"seed": seed})
    return np.stack([np.asarray(b) for b in bolds])


# ---------------------------------------------------------------------------
# T1 — contract: classes import with expected static properties
# ---------------------------------------------------------------------------
print("T1 contract (generated SimGroup classes)")
try:
    from cubnm.sim import (RWWEIBSimGroup, RWWEIB_2CPLSimGroup,
                           RWWEIB_DELAYSimGroup)
    check("RWWEIB_2CPL n_noise==2", RWWEIB_2CPLSimGroup.n_noise == 2,
          f"(got {RWWEIB_2CPLSimGroup.n_noise})")
    check("RWWEIB_2CPL global=g_LRE",
          RWWEIB_2CPLSimGroup.global_param_names == ['g_LRE'])
    check("RWWEIB_2CPL has lambda_IE",
          'lambda_IE' in RWWEIB_2CPLSimGroup.regional_param_names)
    check("RWWEIB_DELAY states E/I",
          RWWEIB_DELAYSimGroup.state_names[-2:] == ['S_E', 'S_I'])
except Exception as e:
    check("import generated classes", False, f"-> {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# T2 — routing: build_param_lists (CPU, no GPU)
# ---------------------------------------------------------------------------
print("T2 routing (build_param_lists)")
from cuBNM.runner_rwweib_2cpl import build_param_lists as bpl2
pl = bpl2(np.array([[1.5, 0.8, 0.01, 0.40]]), ["g_LRE","g_FFI","sigma","I_o"], N)
check("2CPL g_FFI from theta", np.allclose(pl["g_FFI"][:, 0], 0.8))
check("2CPL lambda_IE fixed=1", np.allclose(pl["lambda_IE"], 1.0))
from cuBNM.runner_rwweib_delay import build_param_lists as bpld
pld = bpld(np.array([[1.0, 2.0, 0.01, 0.38]]), ["g_LRE","g_FFI","sigma","I_o"], N)
check("DELAY I_o from theta", np.allclose(pld["I_o"][:, 0], 0.38))


# ---------------------------------------------------------------------------
# T3 — regression: single-coupling model still deterministic + finite
# ---------------------------------------------------------------------------
print("T3 regression (single-coupling n_conn=1 path)")
P4 = ["g_LRE", "g_FFI", "sigma", "I_o"]
b_a = run("rwweib", [[1.0, 1.0, 0.0, 0.382]], P4, use_delays=False)
b_b = run("rwweib", [[1.0, 1.0, 0.0, 0.382]], P4, use_delays=False)
check("rwweib finite", np.isfinite(b_a).all())
check("rwweib deterministic (sigma=0, same seed)", np.allclose(b_a, b_b))
# NOTE: if you saved a pre-surgery FC for rwweib, load + assert np.allclose here.


# ---------------------------------------------------------------------------
# T4 — two-coupling: I driven by SC@S_I, not SC@S_E
# ---------------------------------------------------------------------------
print("T4 two-coupling separation (RWWEIB_2CPL vs FFI)")
# (a) g_FFI=0 -> both models drop the I long-range term -> identical dynamics
ffi0  = run("rwweib",  [[1.2, 0.0, 0.0, 0.382]], P4, use_delays=False)
two0  = run("rwweib2", [[1.2, 0.0, 0.0, 0.382]], P4, use_delays=False)
check("g_FFI=0: 2CPL == FFI (consistency)", np.allclose(ffi0, two0, atol=1e-6))
# (b) g_FFI>0 -> FFI uses SC@S_E, 2CPL uses SC@S_I -> MUST differ
ffi1  = run("rwweib",  [[1.2, 1.5, 0.0, 0.382]], P4, use_delays=False)
two1  = run("rwweib2", [[1.2, 1.5, 0.0, 0.382]], P4, use_delays=False)
diff = np.abs(ffi1 - two1).max()
check("g_FFI>0: 2CPL != FFI (S_I coupling active)", diff > 1e-6,
      f"(max|diff|={diff:.2e})")


# ---------------------------------------------------------------------------
# T5 — delay: USE_DELAYS changes dynamics; ~0 delay collapses to undelayed
# ---------------------------------------------------------------------------
print("T5 delay (RWWEIB_DELAY)")
nod = run("rwweibdelay", [[1.0, 1.0, 0.0, 0.382]], P4, use_delays=False)
dly = run("rwweibdelay", [[1.0, 1.0, 0.0, 0.382]], P4, use_delays=True)
check("delay ON != OFF", np.abs(nod - dly).max() > 1e-6,
      f"(max|diff|={np.abs(nod-dly).max():.2e})")
# near-zero delay (tiny lengths) under USE_DELAYS=True ~= undelayed
tiny = run("rwweibdelay", [[1.0, 1.0, 0.0, 0.382]], P4, use_delays=True,
           dl=delays * 0.0 + 1e-6)
check("delay ~0 collapses to undelayed", np.allclose(nod, tiny, atol=1e-4),
      f"(max|diff|={np.abs(nod-tiny).max():.2e})")


# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print(f"  {sum(results)}/{len(results)} checks passed")
print("=" * 50)
raise SystemExit(0 if all(results) else 1)
