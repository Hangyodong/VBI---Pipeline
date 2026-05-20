# 12 — Simulation Module Review

**Date:** 2026-05-18
**Branch:** refactor/02-simulation
**Reviewer:** Claude (claude-sonnet-4-6)
**Files inspected:**
`simulation/wc_runner.py`, `simulation/warmup.py`, `simulation/delays.py`,
`simulation/qc.py`, `simulation/__init__.py`, `bold.py`, `simulator.py`,
`wc_runner.py`, `warmup.py`, `delays.py`, `qc.py`

---

## 1. Wilson-Cowan Simulation Execution Flow

The full simulation pipeline from raw SC weights to per-simulation BOLD arrays:

```
simulate_gpu_batch(weights, theta_batch, param_names, delays, apply_bw)
│
├── [outer loop] for each GPU_BATCH-sized chunk of theta_batch
│     │
│     ├── 1. Build base param dict from config.WC_FIXED
│     │       (c_ee=16, c_ei=12, c_ie=15, c_ii=3, tau_e=8, tau_i=8,
│     │        a_e=1.3, a_i=2.0, b_e=4.0, b_i=3.7, noise_amp=0.01,
│     │        dt=0.5ms, t_end=300000ms, t_cut=60000ms,
│     │        method="heun", decimate=2, dtype="float32")
│     │
│     ├── 2. params["weights"] = weights.astype(float64)  (N,N)
│     │    params["num_sim"]  = csz
│     │
│     ├── 3. _apply_engine(params)
│     │       detect_engine_key() → probes WC_sde.valid_params
│     │       injects params["engine"|"backend"|"device"|...] = config.ENGINE
│     │
│     ├── 4. apply_delay(params, delays)
│     │       detect_delay_key() → probes WC_sde.valid_params
│     │       injects params["delays"|"delay_matrix"|"tract_lengths"] = delays (N,N)
│     │       OR injects params["velocity"] = config.VELOCITY_M_PER_S
│     │
│     ├── 5. _try_per_sim_params(params, chunk, param_names)
│     │       for each param in param_names (P, Q, g_e, g_i, ...):
│     │         col = chunk[:, i]           shape (csz,)
│     │         tiled = broadcast(col[None,:], (n_nodes, csz))
│     │         params[name] = contiguous(tiled, float32)  shape (n_nodes, csz)
│     │       ← CRITICAL: per-sim arrays, never batch mean
│     │
│     ├── 6. Shape assert (regression guard):
│     │       for each name in param_names:
│     │         assert params[name].shape == (n_nodes, csz)
│     │
│     ├── 7. [try] vectorized batch path:
│     │       model = WC_sde(params)
│     │       model.prepare_input()
│     │       model.set_initial_state()
│     │       result = _run_streaming_hrf(model, n_nodes, csz, dt_ms, apply_bw)
│     │         │  ← GPU: heunStochastic (WC ODE, cupy)
│     │         │  ← CPU: BoldMonitor.step(E_cpu) (HRF convolution, numpy)
│     │         └─ returns (T_bold, N, csz) float32
│     │       [outputs extended by csz slices: result[:,:,i] for i in range(csz)]
│     │
│     └── [except] fallback to per-theta loop (num_sim=1 each):
│             only triggered if VBI rejects (n_nodes, csz) param arrays
│             slower but preserves (theta_i, BOLD_i) alignment
│             [outputs.append(r_single[:,:,0]) for each theta]
│
└── [after each chunk] cp.get_default_memory_pool().free_all_blocks()

Return: list of N_SIM (T_bold, N) float32 arrays
```

### `_run_streaming_hrf` internals (step-by-step integration):

```
for i in range(n_steps):   # n_steps = ceil(t_end / dt_full) = 600,000
    t_curr = i * dt_full
    model.x0 = model.heunStochastic(model.x0, t_curr)  ← GPU
    E_i = model.x0[:n_nodes, :]                         ← GPU (n_nodes, csz)
    E_cpu = E_i.get()                                    ← CPU transfer
    mon.step(i, E_cpu, t_cut_ms=t_cut)                  ← CPU BoldMonitor

mon.collect(mean_subtract=True)  →  (T_bold, N, S) float32
```

### `BoldMonitor.step()` internals (HRF convolution):

```
Every step:
  interim_stock[i % interim_istep] = clip(E, 0, 1)

Every interim_istep=8 steps (4 ms):
  avg = mean(interim_stock, axis=0)     → (N, S)
  outer_stock[stock_pos] = avg          ← circular buffer of K=5000 slots
  stock_pos = (stock_pos + 1) % K
  stock_count += 1

Every istep=2000 steps (1000 ms = 1 TR), after t_cut and stock_count >= K:
  rolled = roll(outer_stock, -stock_pos, axis=0)    → (K, N, S)
  flat = rolled.reshape(K, N*S)
  bold_t = hrf_rev[None,:] @ flat                   → (1, N*S)
  bold_t = bold_t.reshape(N, S)                     → BOLD frame at this TR
  append to _bold_frames
```

---

## 2. Actual Simulation Entry Point

**Primary (SNPE training):** `simulation/wc_runner.simulate_gpu_batch`

Called via:
- `inference/training_data.py:48` → deferred `from simulator import simulate_gpu_batch`
- `inference/stage2.py:158` → deferred `from simulator import simulate_gpu_batch`
- Both ultimately resolve to `simulation/wc_runner.simulate_gpu_batch`

**Secondary (posterior predictive checks / baseline):** `simulation/wc_runner.simulate_single`

Called via deferred `from simulator import simulate_single` in:
- `inference/posterior.py:113`
- `inference/diagnostics.py:47`
- `evaluation/metrics.py:184,245`
- `evaluation/plots.py:203`

**Optional warm-start alternative:** `simulation/warmup.simulate_with_warmup`

Not currently called in production; available as an optimization when the BoldMonitor
stock fill time is significant.

---

## 3. Role of `simulation/wc_runner.py`

**The GPU simulation engine.** Everything that touches the VBI WC model lives here.

| Function | Role | Called from |
|---|---|---|
| `simulate_gpu_batch` | Primary batch entry point | `inference/training_data.py`, `inference/stage2.py`, `simulation/qc.py` |
| `simulate_single` | Single-param entry (n_repeat copies) | `inference/posterior.py`, `inference/diagnostics.py`, `evaluation/*` |
| `_run_streaming_hrf` | Step-by-step WC+BoldMonitor loop | Both `simulate_*` functions |
| `_try_per_sim_params` | Per-sim param tile injection | `simulate_gpu_batch` |
| `_import_wc` | Lazy import of `vbi.WC_sde` | All simulation functions, `simulation/delays.py` |
| `_apply_engine` | VBI engine-key injection | `simulate_*`, `simulation/warmup.py` |
| `detect_engine_key` | Probe VBI `valid_params` for engine key | `_apply_engine` |
| `to_numpy` | cupy → numpy transfer helper | `normalize_ts`, others |
| `normalize_ts` | Reshape VBI WC output to `(T, N)` / `(T, N, S)` | External callers |

**Design invariants encoded in the file (from module docstring):**

1. `params[name].shape == (n_nodes, csz)` — shape assert in `simulate_gpu_batch` line 274–281 catches regressions to batch-mean.
2. Fallback preserves `(theta_i, BOLD_i)` alignment when per-sim arrays are rejected by VBI.
3. `BoldMonitor(xp=np)` — always CPU; `WC_sde.heunStochastic` stays on GPU.

**Module-level top-level imports (non-lazy):**

```python
import numpy as np
import config
from simulation.delays import apply_delay as _apply_delay
```

**Lazy / deferred imports (inside functions):**

```python
from bold import BoldMonitor            # _run_streaming_hrf
import cupy as cp                       # simulate_gpu_batch, simulate_single
from vbi.models.cupy.wilson_cowan import WC_sde   # _import_wc (called once, cached)
```

The GPU and VBI imports are deferred so the module is importable and compile-checkable
without CUDA installed.

---

## 4. Role of `simulation/warmup.py`

**Optional warm-start path to eliminate HRF stock-fill overhead.**

The TVB BoldMonitor cannot emit valid BOLD frames until its outer stock buffer (K=5000 steps
at 4ms each = 20 s) has been filled at least once. In a cold-start simulation, the first
20 s of `T_end=300 s` is effectively discarded filling the stock — on top of the 60 s
transient cut. Warmup pre-fills the stock once per subject, then broadcasts the filled state
to all B simulations in a batch.

| Function / Class | Role |
|---|---|
| `WarmupResult` | Dataclass holding: `x0` (WC state, GPU), `bold_monitor` (BoldMonitor, CPU), `sc`, `delays`, `params_dict`, `t_warmup_ms` |
| `warmup_run` | Runs WC for `2×HRF_LENGTH_MS = 40,000 ms` with a single param set; fills BoldMonitor stock; returns `WarmupResult` |
| `simulate_with_warmup` | Broadcasts `WarmupResult.x0` to B sims, clones BoldMonitor stock, runs full `T_end` simulation |

**Module-level top-level imports:**

```python
import numpy as np
import config
from simulation.delays import apply_delay as _apply_delay
from simulation.wc_runner import _apply_engine, _import_wc
```

**Lazy / deferred imports:**

```python
from bold import BoldMonitor   # warmup_run, simulate_with_warmup
import cupy as _cp             # simulate_with_warmup
```

### Notable design discrepancy: per-sim parameter shape

`simulate_gpu_batch` tiles each param as `(n_nodes, csz)` (homogeneous across nodes):
```python
tiled = np.broadcast_to(col[None, :], (n_nodes, n_sim))   # (n_nodes, csz)
params[name] = np.ascontiguousarray(tiled, dtype=np.float32)
# assert: params[name].shape == (n_nodes, csz)
```

`simulate_with_warmup` injects each param as `(B,)` (not tiled over nodes):
```python
params[name] = theta_batch_arr[:, i].astype(np.float32)   # (B,)
# assert: params[name].shape == (B,)
```

Both paths have shape asserts but they check different shapes. The underlying VBI WC engine
may broadcast `(B,)` to `(n_nodes, B)` internally via `prepare_input`, making them
functionally equivalent, but **the two paths enforce different shape contracts** and this
inconsistency should be verified against the VBI version in use. If VBI does NOT handle
`(B,)` the same as `(n_nodes, B)`, `simulate_with_warmup` would produce incorrect
parameter injection without a visible error.

---

## 5. Role of `simulation/delays.py`

**Tract-length → conduction delay conversion and VBI API adaptation.**

| Function | Role |
|---|---|
| `compute_delay_matrix(weights, velocity_m_per_s, lengths_mm)` | Converts `lengths_mm / velocity` → `(N, N)` float64 ms; diagonal set to 0; returns None if velocity ≤ 0 |
| `detect_delay_key()` | Probes `WC_sde.valid_params` for the key name VBI uses for delays; result is module-level cached |
| `apply_delay(params, delays_precomputed)` | Injects the pre-computed delay matrix into the WC params dict using the detected key |
| `_apply_delay` | Module-level alias for `apply_delay` (backward-compat for `simulator.py`, root `delays.py`) |

**Key design decisions:**

- `1 m/s = 1 mm/ms` — no unit conversion needed.
- Length proxy fallback: if `lengths_mm` is None, uses `1 / (weights + ε)` scaled to mean ≈ 10 mm. The caller (`data_loader.get_subject_data`) prints the warning; this function is silent.
- Velocity ≤ 0 → `None` delay (no conduction delay applied).
- Recognises multiple VBI key name conventions: `"delays"`, `"delay_matrix"`, `"tract_lengths"` (inject matrix); `"velocity"`, `"speed"`, `"conduction_velocity"` (inject scalar velocity).

**Module-level top-level imports:**

```python
import numpy as np
import config
```

**Lazy / deferred imports:**

```python
from simulation.wc_runner import _import_wc   # detect_delay_key() only
```

This is a deferred import inside `detect_delay_key()` that creates a surface-level
mutual dependency with `wc_runner`. The import direction at module-load time is safe
(`wc_runner` → `delays` at module level; `delays` → `wc_runner` only inside a function).
See Section 12 for the full circular-import analysis.

---

## 6. Role of `simulation/qc.py`

**End-to-end simulation quality check — guards against batch-mean regression.**

| Function | Role |
|---|---|
| `run_theta_specific_check(weights, delays, param_names, theta_a, theta_b, ...)` | Simulates two contrastive thetas; computes FCs; verifies they differ by `> atol`; returns diff norm and FC matrices |
| `assert_theta_feature_distinct(theta_a, theta_b, fc_a, fc_b, atol)` | Raises `AssertionError` if `||fc_a - fc_b||₂ ≤ atol`; diagnostic message names `simulate_gpu_batch` as the likely culprit |
| `theta_feature_diff_norm(fc_a, fc_b)` | L2 norm of the upper-tri FC difference; convenience |

**Called from:** `debug_notebook.py` (F section for theta-specific checks). Not called in
the production training pipeline — used only during development / debugging.

### R3 Violation: `simulation/qc.py` imports from `features/`

```python
from features.fc import compute_fc, fc_to_upper_tri   # line 25
from simulation.wc_runner import simulate_gpu_batch     # line 26
```

Rule R3 (`07_refactor_rules.md`) states `simulation/` must not import from `features/`.
`qc.py` violates this at the module level. The rationale is that QC computes FC to
verify simulator outputs — but that makes it dependent on `features.fc`.

**This violation is in the authoritative package file and in the root duplicate.**
The violation does not cause a runtime error (no circular dependency results from it, since
`features/fc.py` depends only on `config` and numpy), but it breaks the declared separation
that keeps `simulation/` self-contained.

**Options for resolving (do not act yet):**
1. Accept as an intentional exception, add a comment to the module explaining why.
2. Move `qc.py` to a `tests/` directory or a separate `checks/` module outside the package hierarchy.
3. Make the FC computation injectable: pass `compute_fc` and `fc_to_upper_tri` as callable arguments to `run_theta_specific_check`, removing the static import.

**Also note:** `simulation/__init__.py` re-exports `assert_theta_feature_distinct`,
`run_theta_specific_check`, and `theta_feature_diff_norm` — meaning that importing
`from simulation import assert_theta_feature_distinct` triggers the `features.fc` import.

---

## 7. Root-Level Duplicate File Analysis

All four root-level simulation duplicates are byte-for-byte identical to their package
counterparts (confirmed by `diff` exit-0 for all):

| Root file | Package counterpart | Diff result | Active bare-name callers |
|---|---|---|---|
| `wc_runner.py` | `simulation/wc_runner.py` | Identical | None |
| `warmup.py` | `simulation/warmup.py` | Identical | None |
| `delays.py` | `simulation/delays.py` | Identical | None |
| `qc.py` | `simulation/qc.py` | Identical | None |

**Internally, all root duplicates delegate to the package:**

- `wc_runner.py:29` — `from simulation.delays import apply_delay as _apply_delay`
- `warmup.py:26-27` — `from simulation.delays import …`, `from simulation.wc_runner import …`
- `delays.py:80` — `from simulation.wc_runner import _import_wc` (lazy)
- `qc.py:25-26` — `from features.fc import …`, `from simulation.wc_runner import …`

When a root duplicate is imported via a bare name (e.g., `import delays`), it immediately
re-delegates to the package module internally. This means the root files are transparent
pass-throughs with no independent logic.

**No file currently performs a bare top-level import of any of these names.** The grep
for `^import wc_runner`, `^import warmup`, `^import delays`, `^import qc` returns zero
output. The root duplicates are inert: they are neither called nor harmful, but their
presence creates a shadowing risk if someone writes bare-name imports in the future.

---

## 8. BOLD / HRF Conversion Connection

The hemodynamic model is implemented in `bold.py` (root, not in any package) and consumed
by both `simulation/wc_runner.py` and `simulation/warmup.py` via deferred imports.

### Connection chain

```
bold.py
├── tvb_hrf(dt_ms, hrf_length_ms, a_1, a_2, rate, c)
│     Uses tvb.datatypes.equations.MixtureOfGammas
│     Mouse-tuned: HRF_A1=3.0, HRF_A2=7.0, HRF_C=0.3, HRF_LENGTH_MS=20000
│     Returns: (G_rev [K], K)
│       K = ceil(0.25/ms × 20000ms) = 5000 stock steps
│
└── BoldMonitor(nn, ns, dt_ms, xp=np, period_ms, hrf_length_ms)
      Internal state:
        _interim_istep  = 8        (4ms / 0.5ms)
        _K              = 5000     (stock buffer size)
        _hrf_gpu        = (1, 5000) reversed HRF, on xp device
        _interim_stock  = (8, N, S) float32
        _stock          = (5000, N, S) float32 circular buffer
        _bold_frames    = []
      │
      ├── .step(i, E, t_cut_ms) → BOLD frame or None
      │     E shape: (N, S) float32, always CPU (xp=np enforced by caller)
      │
      └── .collect(mean_subtract=True) → (T_bold, N, S) float32
            clips to config.ANALYSIS_BOLD_T = 240
```

### How `wc_runner.py` uses `BoldMonitor`

```python
# Inside _run_streaming_hrf (lines 116-170), apply_bw=True path:
mon = BoldMonitor(nn=nn, ns=ns, dt_ms=dt_full, xp=np,
                  period_ms=config.TR_SEC * 1000.0,
                  hrf_length_ms=config.HRF_LENGTH_MS)

for i in range(n_steps):                 # 600,000 iterations
    model.x0 = model.heunStochastic(model.x0, t_curr)   # GPU
    E_cpu = model.x0[:nn, :].get()                       # GPU → CPU
    mon.step(i, E_cpu, t_cut_ms=t_cut)                   # CPU HRF

return mon.collect(mean_subtract=True)   # (T_bold, N, S)
```

### Import relationship

`bold.py` is a **root-level non-package module** — not inside `simulation/` or any package.
Per R3, `bold.py` may be imported by `simulation/wc_runner.py` and `simulation/warmup.py`.
Both files use a deferred `from bold import BoldMonitor` inside functions to avoid importing
TVB at module-load time (TVB is slow to import and not always installed).

`bold.py` depends only on `config`, `numpy`, and `tvb.datatypes.equations` (lazy import
inside `tvb_hrf`). It has no intra-repo dependencies beyond `config`, which keeps it safe
from circular import risks.

---

## 9. Expected Input Shapes

### `simulate_gpu_batch`

| Argument | Type | Shape | Notes |
|---|---|---|---|
| `weights` | ndarray float64 | `(N, N)` | SC matrix, log1p+max-norm normalized, no NaN |
| `theta_batch` | ndarray float32 | `(M, P)` | M = N_SIM (50000 in production); P = n params |
| `param_names` | list[str] | length P | Must match WC param names: `"P"`, `"Q"`, `"g_e"`, `"g_i"`, … |
| `delays` | ndarray float64 or None | `(N, N)` | Conduction delays in ms; diagonal = 0 |
| `fixed_overrides` | dict or None | — | Extra WC params not in theta_batch |
| `apply_bw` | bool | — | True = HRF; False = raw decimated E |

N = 115 (config.N_REGIONS). P = 4 (Stage 1: P, Q, g_e, g_i) or 4 + |theta_bad| (Stage 2).

### `simulate_single`

| Argument | Type | Shape | Notes |
|---|---|---|---|
| `weights` | ndarray float64 | `(N, N)` | Same as above |
| `params_dict` | dict | — | All WC param values as floats |
| `n_repeat` | int | — | Number of independent repeats (default 1) |
| `delays` | ndarray or None | `(N, N)` | |

### `warmup_run`

| Argument | Type | Shape | Notes |
|---|---|---|---|
| `sc` | ndarray float64 | `(N, N)` | SC matrix |
| `params_dict` | dict | — | WC params (typically prior mean) |
| `t_warmup_ms` | float | — | Default = 2 × HRF_LENGTH_MS = 40,000 ms |
| `delays` | ndarray or None | `(N, N)` | |
| `ns` | int | — | Default 1 |

### `simulate_with_warmup`

| Argument | Type | Shape | Notes |
|---|---|---|---|
| `warmup` | WarmupResult | — | From `warmup_run` |
| `theta_batch` | ndarray float32 | `(B, P)` | Batch size B, P params |
| `param_names` | list[str] | length P | |
| `fixed_overrides` | dict or None | — | |

### `compute_delay_matrix`

| Argument | Type | Shape | Notes |
|---|---|---|---|
| `weights` | ndarray | `(N, N)` | Used only as fallback if `lengths_mm` is None |
| `velocity_m_per_s` | float | — | config.VELOCITY_M_PER_S = 1.5 |
| `lengths_mm` | ndarray or None | `(N, N)` | Preferred; from SC mat col 2 |

---

## 10. Expected Output Shapes

| Function | Output | Shape | Dtype | Notes |
|---|---|---|---|---|
| `simulate_gpu_batch` | list of M arrays | each `(T_bold, N)` | float32 | T_bold=240 when `apply_bw=True` |
| `simulate_single` | list of n_repeat arrays | each `(T_bold, N)` | float32 | |
| `_run_streaming_hrf` (apply_bw=True) | ndarray | `(T_bold, N, S)` | float32 | Before per-sim slicing |
| `_run_streaming_hrf` (apply_bw=False) | ndarray | `(T_stored, N, S)` | float32 | T_stored ≤ (T_end−T_cut)/(DT×DECIMATE) |
| `simulate_with_warmup` | list of B arrays | each `(T_bold, N)` | float32 | |
| `compute_delay_matrix` | ndarray or None | `(N, N)` | float64 | None if velocity ≤ 0 |
| `warmup_run` | WarmupResult | — | — | Contains x0 `(2N, 1)` GPU array + BoldMonitor |
| `BoldMonitor.collect` | ndarray | `(T_bold, N, S)` | float32 | Clipped to ANALYSIS_BOLD_T=240 |

**BOLD time dimension derivation:**
```
T_end   = 300,000 ms
T_cut   =  60,000 ms  (transient discarded)
DT      =  0.5 ms
DECIMATE = 2
TR_SEC  = 1.0 s  → TR_MS = 1000 ms
ANALYSIS_BOLD_T = (T_end - T_cut) / TR_MS = 240 frames
n_steps_total = T_end / DT = 600,000 WC integration steps
```

---

## 11. GPU Batch Simulation Assumptions

The following must all hold for `simulate_gpu_batch` to run correctly:

| Assumption | Where enforced | Failure mode |
|---|---|---|
| `cupy` importable (CUDA available) | `import cupy as cp` in function body | `ImportError` at first call |
| `vbi.models.cupy.wilson_cowan.WC_sde` importable | `_import_wc()`, lazy cached | `ImportError` at first call |
| `weights.shape[0] == N_REGIONS` | Shape assert on param arrays | `AssertionError` if wrong |
| `theta_batch.shape[1] == len(param_names)` | Implicit via column indexing in `_try_per_sim_params` | `IndexError` if mismatched |
| `params[name].shape == (n_nodes, csz)` after injection | Explicit assert in `simulate_gpu_batch` lines 274–281 | `AssertionError` — regression guard |
| BoldMonitor always on CPU | `BoldMonitor(xp=np)` — hardcoded | cupy NVRTC failure if violated (see 06_known_errors.md) |
| `config.GPU_BATCH` set correctly | Used directly; no bounds check | GPU OOM if too large |
| Per-sim params, never batch mean | Shape assert + `_try_per_sim_params` logic | Silent wrong training data if bypassed |
| `len(outputs) == len(theta_batch)` at return | Ensured by both batch and fallback paths | Would propagate to SNPE feature alignment |
| Fallback mode assumed stable | `array_param_supported` cached after first chunk | If first chunk succeeds but later fails, raises without fallback |

**Batch size behaviour:**
- `config.GPU_BATCH` chunks the total theta_batch
- Each chunk is a separate VBI model instantiation
- GPU memory is freed after each chunk via `cp.get_default_memory_pool().free_all_blocks()`
- The `array_param_supported` flag is checked only once (first chunk); once True, subsequent
  failures raise immediately without fallback — this protects against mid-run mode switches.

---

## 12. Circular Import Risks

### 12.1 `delays` ↔ `wc_runner` — apparent mutual dependency (SAFE)

```
Module load order:
  wc_runner.py line 29:  from simulation.delays import apply_delay
  → Python starts loading delays.py
  → delays.py module level: import numpy, import config  (no references to wc_runner)
  → delays.py fully loaded, apply_delay resolved
  → wc_runner.py continues loading (delays is now in sys.modules)

Runtime (later):
  delays.detect_delay_key() line 80:  from simulation.wc_runner import _import_wc
  → wc_runner is already in sys.modules (loaded before delays was called)
  → safe deferred reference
```

**Verdict: Safe.** The mutual reference exists only at the function level inside
`detect_delay_key()`. At module-load time, the import direction is strictly
`wc_runner → delays` (one-way). The deferred import inside `detect_delay_key()` always
finds `wc_runner` already in `sys.modules`.

**Rule R4 status:** R4 says `delays` must never import from `wc_runner`. Technically, the
deferred import in `detect_delay_key()` violates the spirit of R4. At module-load time it is
safe, but if `delays` were ever loaded in isolation and `detect_delay_key()` called before
`wc_runner` was imported, the function-level import would trigger `wc_runner` to load
(potentially pulling in `delays` again, which by then is partially loaded). In practice this
path is unreachable because `wc_runner` always imports `delays` first.

**Recommended note:** Document the deferred import in `delays.py:detect_delay_key` as a
known soft R4 exception.

### 12.2 `warmup` → `wc_runner` → `delays` (SAFE, one-way)

```
warmup.py:
  from simulation.delays import apply_delay as _apply_delay   ← delays
  from simulation.wc_runner import _apply_engine, _import_wc  ← wc_runner
```

Both are at module level. By the time `warmup.py` is loaded, both `delays` and `wc_runner`
must already be importable. Since `simulation/__init__.py` imports `wc_runner` before
`warmup`, this order is guaranteed within the package initialisation.

### 12.3 `qc.py` → `features.fc` (R3 VIOLATION — not circular, but layering violation)

```
qc.py:
  from features.fc import compute_fc, fc_to_upper_tri    ← VIOLATES R3
  from simulation.wc_runner import simulate_gpu_batch    ← correct
```

No circular import results from this (nothing in `features.fc` imports from `simulation/`).
The dependency direction is: `simulation.qc → features.fc → config`. This is one-way and
safe at runtime. The violation is a layering concern only.

**Impact of the violation:** Importing `from simulation import assert_theta_feature_distinct`
(via `simulation/__init__.py`) now transitively imports `features.fc`. If `features.fc` ever
develops an import error, it will surface as a `simulation` package load failure — confusing.

### 12.4 Private symbols re-exported from `simulation/__init__.py` (MINOR)

`simulation/__init__.py` imports and re-exports leading-underscore (private) symbols:
`_apply_engine`, `_import_wc`, `_run_streaming_hrf`, `_try_per_sim_params`. They are not
in `__all__` but they are importable via `from simulation import _apply_engine`. This is the
correct compat approach (they are needed by `simulation/warmup.py` and legacy callers), but
it means the `simulation` package exposes implementation details as part of its API surface.

---

## 13. Minimal Future Refactor Plan for `simulation/`

Apply in this order. Each step is independent of the import cleanup in `11_import_audit.md`
unless noted.

### Step S1 — Resolve the R3 violation in `simulation/qc.py` (Priority: Medium)

**Problem:** `simulation/qc.py` imports from `features.fc` at module level, violating R3.

**Option A (recommended):** Accept as an intentional exception; add a short comment in
`qc.py` explaining why FC computation is used here and why the violation is acceptable.
Update `07_refactor_rules.md` R3 to document the explicit exception.

**Option B (strict):** Move `qc.py` out of `simulation/` into a `checks/` or `tests/`
directory that is explicitly allowed to import from both `simulation/` and `features/`.
Update `simulation/__init__.py` to re-export from the new location for backward compatibility.

### Step S2 — Verify and document `simulate_with_warmup` per-sim parameter shape (Priority: High, before using warmup path)

**Problem:** `simulate_with_warmup` passes `params[name] = (B,)`, but `simulate_gpu_batch`
passes `params[name] = (n_nodes, B)`. Both have shape asserts checking their own convention.
If VBI handles `(B,)` differently than `(n_nodes, B)`, the warmup path produces wrong results.

**Action:** Run `run_theta_specific_check` twice — once with `simulate_gpu_batch` and once
with `simulate_with_warmup` on the same two thetas — and verify that `||Δfeature||₂` is
above `atol` and that the two paths produce consistent (not identical — noise) FC patterns.
If VBI accepts both shapes identically, document it. If not, fix `simulate_with_warmup` to
match `simulate_gpu_batch` shape convention.

### Step S3 — Remove deferred import in `delays.detect_delay_key` (Priority: Low)

**Problem:** Soft R4 violation. `delays.py` lazily imports `_import_wc` from `wc_runner`
inside `detect_delay_key()`. The same result (WC instance for `valid_params` probe) could
be obtained by accepting an optional `wc_cls` argument.

**Option:** Add `wc_cls=None` parameter to `detect_delay_key`; if None, import internally
(current behaviour); if provided, use the passed class. This makes the function testable
without the lazy import.

### Step S4 — Remove root duplicate files (after import cleanup, Tier 6 of 11_import_audit.md)

Order (from 11_import_audit.md Tier 6):
1. `delays.py` → safe (identical, no callers)
2. `warmup.py` → safe (identical, no callers)
3. `qc.py` → safe (identical, no callers)
4. `wc_runner.py` → safe (identical, verified by diff; delete last)

After each deletion, run the compile and grep checks in Section 14.

### Step S5 — Unexport private symbols from `simulation/__init__.py` (Priority: Low)

`_apply_engine`, `_import_wc`, `_run_streaming_hrf`, `_try_per_sim_params` are imported
but should not be part of the public package surface. After `simulator.py` is removed
(Tier 5 of 11_import_audit.md), confirm that no external callers use these via
`from simulation import _*`. Then remove them from `simulation/__init__.py` imports.
`simulation/warmup.py` already imports them directly from `simulation.wc_runner` — correct.

---

## 14. Test Commands for `simulation/`

### 14.1 Compile check (no GPU required)

```bash
python -m py_compile simulation/__init__.py simulation/wc_runner.py \
    simulation/delays.py simulation/warmup.py simulation/qc.py bold.py
echo "simulation/ compile OK"
```

### 14.2 Package import chain test (no GPU required)

```bash
python -c "
import config
from simulation.delays import compute_delay_matrix, apply_delay, detect_delay_key
from simulation.wc_runner import to_numpy, normalize_ts
print('simulation non-GPU imports OK')
"
```

Note: `_import_wc`, `detect_engine_key`, `simulate_gpu_batch`, `simulate_single` are not
importable without CUDA, but the module-level import chain is verifiable.

### 14.3 Dependency-rule check (R3)

```bash
# simulation/ must not import from features/, inference/, evaluation/
grep -n "from features\|from inference\|from evaluation" \
    simulation/wc_runner.py simulation/delays.py simulation/warmup.py
# Expected: zero output

# This WILL find the R3 violation in qc.py — document but do not fix yet:
grep -n "from features\|from inference\|from evaluation" simulation/qc.py
# Expected: "from features.fc import compute_fc, fc_to_upper_tri"
# This is the known R3 violation (Step S1 above).
```

### 14.4 Check for bare root-duplicate imports

```bash
grep -rn "^import wc_runner\|^from wc_runner import" --include="*.py" .
grep -rn "^import warmup\b\|^from warmup import" --include="*.py" .
grep -rn "^import delays\b\|^from delays import" --include="*.py" .
grep -rn "^import qc\b\|^from qc import" --include="*.py" .
# All expected: zero output
```

### 14.5 Verify no drift between root duplicates and package files

```bash
diff wc_runner.py simulation/wc_runner.py && echo "wc_runner: identical"
diff warmup.py    simulation/warmup.py    && echo "warmup: identical"
diff delays.py    simulation/delays.py    && echo "delays: identical"
diff qc.py        simulation/qc.py        && echo "qc: identical"
```

### 14.6 Verify `simulator.py` exports match package symbols

```bash
python -c "
import simulator
required = [
    'simulate_gpu_batch', 'simulate_single', 'compute_delay_matrix',
    'apply_delay', 'WarmupResult', 'warmup_run', 'simulate_with_warmup',
    'compute_fc', 'fc_to_upper_tri', 'compute_sim_fcd_matrix',
    'extract_observed_features', 'extract_simulated_features', 'worker_extract',
]
missing = [n for n in required if not hasattr(simulator, n)]
if missing:
    print('MISSING from simulator:', missing)
else:
    print('simulator exports: OK')
"
```

### 14.7 GPU smoke test (requires CUDA + VBI installed)

```bash
python -c "
import numpy as np
import config
from simulation.wc_runner import simulate_gpu_batch
from simulation.delays import compute_delay_matrix

# Minimal smoke: 2 simulations, dummy SC
N = config.N_REGIONS
rng = np.random.RandomState(0)
sc = rng.rand(N, N).astype(np.float64)
sc = sc / sc.max()
np.fill_diagonal(sc, 0)

theta = np.array([[1.5, 1.0, 0.5, 0.5],
                  [0.8, 0.2, 1.2, 1.2]], dtype=np.float32)
param_names = ['P', 'Q', 'g_e', 'g_i']

bolds = simulate_gpu_batch(sc, theta, param_names, apply_bw=True)
assert len(bolds) == 2, f'expected 2 BOLDs, got {len(bolds)}'
assert bolds[0].shape == (config.ANALYSIS_BOLD_T, N), \
    f'wrong shape: {bolds[0].shape}'
print(f'GPU smoke test OK: {len(bolds)} bolds, shape={bolds[0].shape}')
"
```

### 14.8 Theta-specific QC check (requires CUDA + VBI)

```bash
python -c "
import numpy as np, config
from simulation.qc import run_theta_specific_check
from simulation.delays import compute_delay_matrix

N = config.N_REGIONS
rng = np.random.RandomState(42)
sc = rng.rand(N, N); sc = sc / sc.max(); np.fill_diagonal(sc, 0)
delays = None  # test without delay

result = run_theta_specific_check(
    weights=sc, delays=delays,
    param_names=['P', 'Q', 'g_e', 'g_i'],
    theta_a=[2.0, 1.5, 0.3, 0.3],
    theta_b=[0.6, 0.2, 1.3, 1.3],
    atol=1e-3, verbose=True,
)
print('QC check passed, diff=', result['diff'])
"
```

### 14.9 BoldMonitor unit test (no GPU required)

```bash
python -c "
import numpy as np, config
from bold import BoldMonitor, tvb_hrf

N, S = 5, 2
hrf, K = tvb_hrf(dt_ms=4.0, hrf_length_ms=config.HRF_LENGTH_MS)
assert len(hrf) == K, f'HRF length mismatch: {len(hrf)} != {K}'

mon = BoldMonitor(nn=N, ns=S, dt_ms=0.5, xp=np, verbose=True)
n_steps = int(config.T_END / 0.5)
rng = np.random.RandomState(0)
for i in range(n_steps):
    E = rng.rand(N, S).astype(np.float32)
    mon.step(i, E, t_cut_ms=config.T_CUT)

bold = mon.collect(mean_subtract=True)
assert bold.shape[1] == N
assert bold.shape[2] == S
print(f'BoldMonitor test OK: bold shape={bold.shape}, target T={config.ANALYSIS_BOLD_T}')
"
```

---

## Final Assessment

### Core simulation files (authoritative, do not delete)

| File | Status |
|---|---|
| `simulation/wc_runner.py` | **Core.** Primary GPU simulation engine. All simulation paths go through this. |
| `simulation/delays.py` | **Core.** Delay matrix computation and VBI API adaptation. Called at subject-load time and at every simulation. |
| `simulation/warmup.py` | **Core (optional path).** Warm-start optimisation. Currently not called in production but complete and tested. |
| `simulation/qc.py` | **Core (debug).** End-to-end theta-specific QC. Used from `debug_notebook`. Has an R3 violation that needs documentation or relocation. |
| `simulation/__init__.py` | **Core.** Re-export hub. Defines the public `simulation.*` API. |
| `bold.py` | **Core root module.** HRF/BOLD computation; consumed by both `wc_runner` and `warmup`. Not a legacy file — no package equivalent exists. |
| `simulator.py` | **Compat wrapper.** Not core, but must not be deleted yet — active callers in `data_loader.py` and 9 deferred imports in `inference/`+`evaluation/`. |

### Root-level legacy candidates (mark for eventual deletion)

| File | Basis |
|---|---|
| `wc_runner.py` | Byte-identical to `simulation/wc_runner.py`; no active bare-name callers |
| `warmup.py` | Byte-identical to `simulation/warmup.py`; no active bare-name callers |
| `delays.py` | Byte-identical to `simulation/delays.py`; no active bare-name callers |
| `qc.py` | Byte-identical to `simulation/qc.py`; no active bare-name callers |

### Files that must NOT be deleted yet

| File | Reason |
|---|---|
| `simulator.py` | Live production dependency in `data_loader.py:278` and 9 deferred `from simulator import` calls in `inference/` and `evaluation/` submodules |
| `bold.py` | Not a duplicate — it is the only implementation of `BoldMonitor` and `tvb_hrf`; no package equivalent |
| `simulation/qc.py` | Has an R3 violation but is the authoritative QC implementation; needs resolution before any structural change |
| Any root simulation duplicate | Defer all deletions until Tier 1–4 import cleanup (11_import_audit.md) is complete |

### What should be inspected next

In priority order:

1. **`inference/training_data.py`** — The file that calls `simulate_gpu_batch` in production
   (via deferred `from simulator import`). Reading it will confirm exactly how the batch
   simulation is called, what subject loop structure is used, and whether warmup is wired in.

2. **`inference/stage2.py`** — Stage 2 simulation call chain; also imports `worker_extract`
   from `simulator`, which maps to `features.extraction.worker_extract`. Confirming this
   will close the import migration for the two hottest inference files.

3. **`features/extraction.py`** — The target for `worker_extract` and `extract_observed_features`
   migrations. Understanding its parallel-worker pattern is needed to safely update the
   deferred imports in `inference/` and `evaluation/`.

4. **VBI version in use (`vbi.models.cupy.wilson_cowan.WC_sde`)** — The `detect_engine_key`
   and `detect_delay_key` probing logic assumes `valid_params` exists on the WC instance.
   Check which VBI version is installed and whether it has this attribute:
   ```bash
   python -c "from vbi.models.cupy.wilson_cowan import WC_sde; m=WC_sde({}); print(getattr(m,'valid_params',None))"
   ```
   If `valid_params` is absent, both detection functions return `None`, and the fallback
   behaviour (`params["engine"] = config.ENGINE`) is used — worth confirming.
