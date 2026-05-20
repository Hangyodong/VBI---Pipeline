# 29 — Claude Code Harness: Verification

Part of the Claude Code Harness for VBI-SBI pipeline.
Generated: 2026-05-18  Branch: refactor/02-simulation
Repo: /scratch/home/wog3597/vbi

## HOW TO USE THIS FILE

This is the **proof-of-work checklist**. After every task, run the
appropriate tier(s) before claiming completion. Every command is
copy-pasteable from the repo root.

| When | Tier | Time | GPU required |
|---|---|---|---|
| Every task (mandatory) | 0 | ~30 s | No |
| After `inference/snpe.py` change | 1 | ~30 s | No |
| After `simulation/wc_runner.py` or `simulation/*` change | 2 | ~10 min | **Yes** |
| After `main.ipynb` change | 3 | ~10 s | No |
| After any significant code change | 4 | ~60 s | No |

If any verification step fails, **STOP** and follow the protocol-3.12
rollback procedure (file 28).

---

## 4.1 TIER 0 — ALWAYS RUN (mandatory, ~30 s, no GPU)

### T-0a — Compile-check the whole tree

```bash
python -m compileall -q .
echo "exit: $?"
```

**Expected**: `exit: 0`. Zero stderr output. If anything else, you
have a syntax error somewhere — fix immediately.

### T-0b — Package import smoke

```bash
python -c "
import config, data_loader
import simulation, features, inference, evaluation, pipelines
print('All 5 packages OK')
"
```

**Expected**: `All 5 packages OK`. Any `ImportError` means a
dependency rule (R3) or attribute is broken.

### T-0c — P-1 invariant

```bash
python -c "
import pipelines.stage1_stage2 as pl
assert pl.evaluate.__name__ == 'evaluation', (
    f'P-1 FAIL: pl.evaluate.__name__ = {pl.evaluate.__name__!r}'
)
assert 'evaluation/__init__' in pl.evaluate.__file__, (
    f'P-1 FAIL: pl.evaluate.__file__ = {pl.evaluate.__file__}'
)
print('P-1 OK')
"
```

**Expected**: `P-1 OK`. Otherwise: doc 19 has been reverted —
re-apply P-1 immediately.

### T-0d — P-2 invariant

```bash
python -c "
import inspect, data_loader
src = inspect.getsource(data_loader.get_subject_data)
assert 'from simulation.delays import compute_delay_matrix' in src, 'P-2 FAIL: new form missing'
assert 'from simulator import compute_delay_matrix' not in src, 'P-2 FAIL: old form present'
print('P-2 OK')
"
```

**Expected**: `P-2 OK`. Otherwise: doc 21 has been reverted —
re-apply P-2.

### T-0e — GPU-1 invariants (lazy-cupy preserved, helpers present)

```bash
python -c "
import inspect
from simulation import wc_runner
src = inspect.getsource(wc_runner)
assert '_alloc_stride_buffers' in src, 'GPU-1 FAIL: _alloc_stride_buffers missing'
assert '_trim_memory_pool' in src, 'GPU-1 FAIL: _trim_memory_pool missing'
# Verify cupy is NOT imported at module level (must remain deferred).
module_top = src.split('\\ndef ')[0]
assert 'import cupy' not in module_top, 'GPU-1 FAIL: cupy imported at module top'
print('GPU-1 OK')
"
```

**Expected**: `GPU-1 OK`. Otherwise: the GPU-1 refactor was undone —
restore the helpers + deferred import pattern.

---

## 4.2 TIER 1 — AFTER `inference/snpe.py` CHANGES (~30 s, no GPU)

### T-1a — Module import

```bash
python -c "import inference.snpe; print('snpe OK')"
```

### T-1b — Public-API smoke

```bash
python -c "
from inference import (
    ParameterScaler, FeaturePipeline, FeatureEmbedding,
    run_stage1_snpe, run_stage2_snpe,
    save_artifacts, load_artifacts,
)
print('inference API OK')
"
```

### T-1c — Pipeline + evaluation API smoke

```bash
python -c "
from pipelines import run_pipeline
from evaluation import (
    fc_metrics, fcd_vec_rmse,
    evaluate_validation_stage1, evaluate_validation_stage2,
    select_best_model, final_test, print_final_summary,
    plot_posteriors, plot_fc_comparison, plot_sbc_rank_histogram,
    plot_pca_diagnostic,
)
print('pipeline+eval API OK')
"
```

### T-1d — Confirm public signatures unchanged

```bash
python -c "
import inspect
from inference.snpe import (
    train_snpe, step4_fit_feature_scalers,
    step5_fit_feature_pipeline, step6_pca_diagnostic,
    step7_fit_param_scaler, step8_train_snpe,
)
for fn in (train_snpe, step4_fit_feature_scalers,
           step5_fit_feature_pipeline, step6_pca_diagnostic,
           step7_fit_param_scaler, step8_train_snpe):
    print(f'{fn.__name__}{inspect.signature(fn)}')
"
```

Expected signatures (must match exactly — any new required arg is
an HC-1 violation):

```
train_snpe(theta_scaled, x_input, prior_scaled, embedding_net=None, proposal=None, verbose=True)
step4_fit_feature_scalers(fc_raw, fcd_raw, verbose=True)
step5_fit_feature_pipeline(fc_raw, fcd_raw, verbose=True)
step6_pca_diagnostic(pipeline, fc_raw, fcd_raw, verbose=True)
step7_fit_param_scaler(verbose=True)
step8_train_snpe(theta_scaled, x_input, prior_scaled, verbose=True)
```

---

## 4.3 TIER 2 — AFTER SIMULATION CHANGES (~10 min, GPU required)

### T-2a — Module import (no GPU needed — lazy import)

```bash
python -c "import simulation.wc_runner; print('wc_runner OK')"
```

Must succeed even on machines without a working CUDA driver.

### T-2b — QC check (bit-identical / atol output)

```bash
python -c "
import numpy as np, config
from simulation.qc import run_theta_specific_check

N = config.N_REGIONS
rng = np.random.RandomState(42)
sc = rng.rand(N, N)
sc = sc / sc.max()
np.fill_diagonal(sc, 0)

result = run_theta_specific_check(
    weights=sc, delays=None,
    param_names=config.STAGE1_PARAMS,
    theta_a=[2.0, 1.5, 0.3, 0.3],
    theta_b=[0.6, 0.2, 1.3, 1.3],
    atol=1e-3, verbose=True,
)
print('QC diff:', result['diff'], '  pass:', result['pass'])
"
```

**Expected for bit-identical optimizations (e.g. GPU-1)**:
`pass: True`, `diff: 0.0`.
**Expected for within-atol optimizations**: `pass: True`,
`diff < 1e-3`.

### T-2c — 50-sim timing benchmark

```bash
python -c "
import time, numpy as np, config
from simulation.wc_runner import simulate_gpu_batch

N = config.N_REGIONS
rng = np.random.RandomState(0)
sc = rng.rand(N, N).astype(np.float64)
sc /= sc.max()
np.fill_diagonal(sc, 0)

theta = np.column_stack([
    np.random.uniform(0.5, 2.5, 50),
    np.random.uniform(0.0, 2.0, 50),
    np.random.uniform(0.0, 1.5, 50),
    np.random.uniform(0.0, 1.5, 50),
]).astype(np.float32)

t0 = time.perf_counter()
bolds = simulate_gpu_batch(sc, theta, config.STAGE1_PARAMS,
                           delays=None, apply_bw=True)
elapsed = time.perf_counter() - t0
print(f'50 sims: {elapsed:.2f}s  '
      f'({elapsed/50*1000:.1f}ms/sim)  '
      f'BOLD shape: {bolds[0].shape}')
"
```

**Expected post-GPU-1**: ~3–6 s for 50 sims (60–120 ms/sim) on H100
NVL. BOLD shape: `(240, 115)`. Anything substantially slower is a
regression.

### T-2d — Confirm wc_runner public signatures unchanged

```bash
python -c "
import inspect
from simulation.wc_runner import (
    simulate_gpu_batch, simulate_single,
    _run_streaming_hrf, _alloc_stride_buffers, _trim_memory_pool,
)
for fn in (simulate_gpu_batch, simulate_single,
           _run_streaming_hrf, _alloc_stride_buffers, _trim_memory_pool):
    print(f'{fn.__name__}{inspect.signature(fn)}')
"
```

Expected (any deviation is an HC-1 violation):

```
simulate_gpu_batch(weights, theta_batch, param_names, fixed_overrides=None, delays=None, apply_bw=True, _allow_fallback=True)
simulate_single(weights, params_dict, n_repeat=1, delays=None, apply_bw=True)
_run_streaming_hrf(model, n_nodes, num_sim, dt_ms, apply_bw)
_alloc_stride_buffers(cp, stride, nn, ns)
_trim_memory_pool(cp)
```

---

## 4.4 TIER 3 — AFTER `main.ipynb` CHANGES (~10 s, no GPU)

### T-3a — Cell count + Integrated Debug Cell sanity

```bash
python -c "
import json
nb = json.load(open('main.ipynb'))
cells = nb['cells']
print(f'Cell count: {len(cells)}')
assert cells[2]['cell_type'] == 'code', 'cell[2] must be code (Setup)'
assert cells[3]['cell_type'] == 'markdown', 'cell[3] must be markdown'
assert cells[4]['cell_type'] == 'code', 'cell[4] must be code'
debug_heading = ''.join(cells[3]['source'])
assert '## Integrated VBI Pipeline Debug Cell' in debug_heading, 'cell[3] heading missing'
debug_src = ''.join(cells[4]['source'])
for flag in ('RUN_SMOKE_CHECKS', 'RUN_DEBUG_BASIC', 'RUN_SMALL_SIMULATION',
             'RUN_FEATURE_EXTRACTION', 'RUN_FEATURE_EMBEDDING',
             'RUN_STAGE1_DRY_RUN', 'RUN_STAGE2_DRY_RUN',
             'RUN_FULL_PIPELINE', 'CONFIRM_FULL_RUN'):
    assert flag in debug_src, f'cell[4] flag missing: {flag}'
print('Integrated Debug Cell OK')
"
```

### T-3b — Compile every code cell

```bash
python -c "
import json
nb = json.load(open('main.ipynb'))
fails = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        try:
            compile(src, f'<cell[{i}]>', 'exec')
        except SyntaxError as e:
            fails.append((i, str(e)))
if fails:
    for i, msg in fails:
        print(f'FAIL cell[{i}]: {msg}')
else:
    print('All code cells compile OK')
"
```

**Expected**: `All code cells compile OK`.

### T-3c — Setup cell unchanged

```bash
python -c "
import json
nb = json.load(open('main.ipynb'))
src = ''.join(nb['cells'][2]['source'])
assert 'setup_pipeline' in src
assert 'PipelineConfig' in src
assert 'import config' in src
assert 'import data_loader' in src
assert 'import inference' in src
print('Setup cell unchanged')
"
```

---

## 4.5 TIER 4 — FULL DEBUG SUITE (~60 s, no GPU)

### T-4a — `debug.py --basic`

```bash
python debug.py --basic
```

Look at the final summary line:

**Expected**: `PASS: 9  |  FAIL: 3  |  SKIP: 0`

The 3 expected FAILs are pre-existing (doc 25 §9.1) and unrelated:
- `config consistency` — `FCD_DIM=5 != FC_DIM=6555`
- `FeaturePipeline` — output dim mismatch (test-expectation drift)
- `FCD upper triangle (simulated)` — vec dim mismatch

### T-4b — PASS-count regression check

```bash
python debug.py --basic 2>&1 | grep -E "^  PASS:" | tail -1
```

**Decision logic**:

| Result | Diagnosis |
|---|---|
| `PASS: 9 \| FAIL: 3 \| SKIP: 0` | Baseline OK |
| `PASS: <9` | A previously-passing test regressed. Identify which and roll back. |
| `PASS: >9` | An unexpected new test exists. Confirm it was added intentionally; otherwise investigate. |
| `FAIL: <3` | Improvement: a Tier-X bug was fixed. Was it authorized? |
| `FAIL: >3` | A new failure was introduced — roll back. |

### T-4c — `patch invariants (P-1, P-2)` test individually

The new `test_patch_invariants` added in doc 25:

```bash
python debug.py --basic 2>&1 | grep "patch invariants"
```

**Expected**: `PASS  patch invariants (P-1, P-2)`. If this fails:
P-1 or P-2 has been reverted — see doc 25 §6 for the test source.

---

## 4.6 PATCH-INVARIANT SUMMARY TABLE

Print this table at the end of every task, filled in with the
actual result of each command.

| Invariant | Command | Expected | Status |
|---|---|---|---|
| P-1 | `python -c "import pipelines.stage1_stage2 as pl; assert pl.evaluate.__name__=='evaluation'"` | `P-1 OK` | ✅ / ❌ |
| P-2 | `python -c "import inspect, data_loader; src=inspect.getsource(data_loader.get_subject_data); assert 'from simulation.delays' in src"` | `P-2 OK` | ✅ / ❌ |
| GPU-1 | `python -c "import inspect; from simulation import wc_runner; src=inspect.getsource(wc_runner); assert '_alloc_stride_buffers' in src and '_trim_memory_pool' in src"` | `GPU-1 OK` | ✅ / ❌ |
| Compile | `python -m compileall -q .` | `exit 0` | ✅ / ❌ |
| Imports | `python -c "import config,data_loader,simulation,features,inference,evaluation,pipelines"` | clean | ✅ / ❌ |
| debug.py --basic | `python debug.py --basic` | `PASS=9 FAIL=3 SKIP=0` | ✅ / ❌ |
| Notebook cells compile | T-3b above | `All code cells compile OK` | ✅ / ❌ |
| Integrated Debug Cell intact | T-3a above | `Integrated Debug Cell OK` | ✅ / ❌ |

If any row is ❌, follow the rollback protocol (file 28 §3.12).

---

## 4.7 COMBINED ONE-LINER (for quick sanity)

```bash
python -m compileall -q . && \
python -c "
import config, data_loader, simulation, features, inference, evaluation, pipelines
import pipelines.stage1_stage2 as pl
assert pl.evaluate.__name__ == 'evaluation', 'P-1 FAIL'
import inspect
src = inspect.getsource(data_loader.get_subject_data)
assert 'from simulation.delays import compute_delay_matrix' in src, 'P-2 FAIL'
assert 'from simulator import compute_delay_matrix' not in src, 'P-2 FAIL'
from simulation import wc_runner
wsrc = inspect.getsource(wc_runner)
assert '_alloc_stride_buffers' in wsrc and '_trim_memory_pool' in wsrc, 'GPU-1 FAIL'
print('ALL INVARIANTS OK')
"
```

**Expected**: `ALL INVARIANTS OK`. This is the minimum bar before
declaring a task complete.

---

## 4.8 WHAT TO DO ON VERIFICATION FAILURE

1. **Do not retry the failing edit blindly.** Read the actual
   exception / diff first.
2. **Read the relevant file in full** — your assumption about the
   surrounding code is likely wrong.
3. **Rollback** with `git checkout -- <file>` or the appropriate
   `.bak_*` restore (notebook).
4. **Re-run Tier 0** to confirm rollback restored the baseline.
5. **Surface the root cause** to the user via the file-30 STOP
   template. Do not propose a "fix-the-fix" without authorization.

---

**End of verification suite. Read 30 (stop conditions) next.**
