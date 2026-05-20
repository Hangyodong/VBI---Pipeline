# 10 — Entry-Point and Orchestration Flow Analysis

**Date:** 2026-05-18
**Branch:** refactor/02-simulation
**Files inspected:** `main.py`, `pipeline_setup.py`, `pipelines/stage1_stage2.py`,
`config.py`, `data_loader.py`, `README.md`

---

## 1. Execution Flow Diagram

```
python main.py
│
├── [env vars parsed by main._env_int / _env_float / _env_bool]
│   N_SIM, N_SIM_S2, RUN_STAGE2, SENS_THRESHOLD, SHR_THRESHOLD
│
└── pipelines.run_pipeline(n_sim, n_sim_s2, run_stage2, sens_threshold,
                           shr_threshold, verbose)
    │
    │  [module-level imports in pipelines/stage1_stage2.py]
    │  import config              ← root module (legitimate)
    │  import data_loader         ← root module (legitimate)
    │  import evaluate            ← ROOT COMPAT WRAPPER  ⚠
    │  import inference           ← inference/ package (correct, Python 3 wins)
    │
    ├── Step 1-3: step_data_split()
    │     data_loader.load_raw_data()
    │       → _load_mat(config.FC_PATH)     scipy.io.loadmat
    │       → _load_mat(config.SC_PATH)     scipy.io.loadmat
    │       → pd.read_csv(config.TSV_PATH)
    │       → _load_optional_bold()         [skipped if BOLD file absent]
    │     data_loader.get_target_subjects(df, fc_ids, sc_ids)
    │     data_loader.three_way_split(subjects)   → train(4), val(2), test(2)
    │     data_loader.load_all_subjects(all_subjects, ...)
    │       └── get_subject_data(sid, ...)  [called per-subject]
    │             from simulator import compute_delay_matrix  ← COMPAT WRAPPER ⚠
    │             _load_fc_fcd(), _scale_weights()
    │             simulation.delays.compute_delay_matrix()   [via simulator.py]
    │
    ├── Stage 1: stage1_pipeline()
    │     inference.run_stage1_snpe(train_subjects, subject_data, n_sim)
    │       [inference/ package — simulation + features + SNPE all inside here]
    │       └── simulation.wc_runner.simulate_gpu_batch()     [GPU, cupy]
    │       └── features.extraction.extract_simulated_features()
    │       └── inference.snpe.step4..8 → SNPE-C posterior
    │
    ├── Stage 1 validation: stage1_validation()
    │     evaluate.evaluate_validation_stage1(val_subjects, ...)
    │       [via evaluate.py compat wrapper → evaluation/ package]
    │
    ├── θ_bad selection: select_theta_bad_from_val()
    │     inference.select_theta_bad(sensitivity, shrinkage, ...)
    │
    ├── Stage 2 (optional): stage2_pipeline()
    │     inference.run_stage2_snpe(...)
    │       [same simulation + features + SNPE flow as Stage 1]
    │
    ├── Stage 2 validation: stage2_validation()
    │     evaluate.evaluate_validation_stage2(...)
    │       [via evaluate.py compat wrapper → evaluation/ package]
    │
    ├── Model selection: select_model()
    │     evaluate.select_best_model(stage1_agg, stage2_agg, ...)
    │       [via evaluate.py compat wrapper → evaluation/ package]
    │
    ├── Final test: run_final_test()
    │     evaluate.final_test(test_subjects, ...)
    │       [via evaluate.py compat wrapper → evaluation/ package]
    │
    └── Save & summary: save_and_summarize()
          inference.save_artifacts(save_path, ...)
          evaluate.print_final_summary(...)
            [via evaluate.py compat wrapper → evaluation/ package]
```

---

## 2. Config Loading Flow

```
config.py is imported as a module — no class, no factory function.
All values are module-level constants set at import time.

Import order:
  1. config.py runs on first import:
       - Sets all constants (DATA_DIR, N_SIM, WC_FIXED, etc.)
       - Computes ANALYSIS_BOLD_T dynamically
       - Runs 6 module-level assert statements
       - Optional: imports torch to detect GPU

  2. data_loader, pipelines/stage1_stage2, and all subpackages
     each call `import config` — they all share the same module object.

  3. pipeline_setup.setup_pipeline() can override config attributes:
       _apply_to_config(cfg) mutates config.N_SIM, config.T_END, etc.
       This propagates to all already-imported modules sharing the reference.

Key config consumers:
  config.N_REGIONS      → data_loader (shape checks), features.fc, evaluation.metrics
  config.FC_PATH etc.   → data_loader.load_raw_data()
  config.N_SIM          → pipelines (stage1_pipeline default), inference.training_data
  config.WC_FIXED       → simulation.wc_runner (GPU engine params)
  config.STAGE1_PARAMS  → inference.scaling, inference.priors, pipelines
  config.SBI_DEVICE     → inference.snpe (cuda vs cpu)
  config.SEED           → pipelines.run_pipeline (np.random.seed, torch.manual_seed)
  config.OUTPUT_DIR     → inference.io, pipelines.save_and_summarize
  config.N_TEST_RESIM   → pipelines.run_final_test → evaluate.final_test
```

**Mutation side-effect risk:** `data_loader.py` writes back into `config` at runtime:
- `config.NAN_MASK = None` (line 161)
- `config.NAN_REGIONS = nan_regions.tolist()` (line 162)
- `config.HAS_BOLD = False / True` (lines 183, 188)

These mutations happen inside `load_raw_data()`, which is called every pipeline run.

---

## 3. Data Loading Flow

```
data_loader.load_raw_data()
  → scipy.io.loadmat(FC_PATH)["data"]   → fc_mat  shape (n_subjects, 3)
  → scipy.io.loadmat(SC_PATH)["data"]   → sc_mat  shape (n_subjects, 3)
  → pd.read_csv(TSV_PATH, sep="\t")     → df      participants
  → _check_shapes(fc_mat, sc_mat)       [assert FC/SC shape == (115, 115)]
  → _record_nan_mask(fc_mat)            [writes config.NAN_REGIONS]
  → _check_fcd_nan(fc_mat)
  → _load_optional_bold()               [reads BOLD_PATH; sets config.HAS_BOLD]

data_loader.get_target_subjects(df, fc_ids, sc_ids)
  → filters by config.GROUP_FILTER = ("ctr", "MPTP")
  → matches participants.tsv "group" and "treatment" columns

data_loader.three_way_split(subjects)
  → uses config.N_TRAIN=4, N_VAL=2, N_TEST=2, SEED=42
  → deterministic: np.random.RandomState(seed).shuffle(sorted(subjects))

data_loader.load_all_subjects(subjects, ...)
  └── get_subject_data(sid, ...)  per subject:
        from simulator import compute_delay_matrix      ← COMPAT WRAPPER CALL ⚠
        _load_fc_fcd()    → FC (NaN→0, symmetrized), FCD
        _scale_weights()  → SC: log1p + max-norm + sparsity mask
        compute_delay_matrix(sc, velocity, lengths_mm)
                          → delays (N, N) float64 (ms)
```

---

## 4. Where Simulation Is Called

Simulation (`simulate_gpu_batch`) is **not** called directly from the orchestration
layer. It is encapsulated entirely inside the `inference/` package:

```
pipelines/stage1_stage2.py
  → inference.run_stage1_snpe()         [inference/stage1.py]
      → inference.training_data.step2_simulate_train()
          → simulation.wc_runner.simulate_gpu_batch()   ← GPU simulation here

  → inference.run_stage2_snpe()         [inference/stage2.py]
      → (same path as Stage 1)
          → simulation.wc_runner.simulate_gpu_batch()
```

The delay matrix is computed in `data_loader.get_subject_data()` (before simulation)
via the `simulator` compat wrapper. The result (`subject_data[sid]["delays"]`) is
passed through to `simulate_gpu_batch` at simulation time.

---

## 5. Where Feature Extraction Is Called

Feature extraction is also fully encapsulated inside `inference/`:

```
inference.training_data.step2_simulate_train()
  → returns bold_list (one BOLD array per simulation)

inference.training_data.step3_summary_features()   [called inside run_stage1_snpe]
  → features.extraction.extract_simulated_features(bold_list)
      → features.fc.compute_fc()
      → features.fc.fc_to_upper_tri()
      → [features.fcd.*  — skipped if USE_FCD=False]
  → returns fc_raw (N_SIM, 6555), fcd_raw (N_SIM, 0)
```

Observed feature extraction happens at validation/test time:

```
evaluate.evaluate_validation_stage1()    [via evaluate.py wrapper]
  → evaluation.validation.evaluate_validation_stage1()
      → inference.posterior.transform_observed()
          → features.fc.fc_to_upper_tri(observed_fc)
          → pipeline.transform(x_fc)   [FamilyScaler + PCA + MLP embedding]
```

---

## 6. Where Inference Is Called

```
Stage 1:
  pipelines.stage1_pipeline()
    → inference.run_stage1_snpe()          [inference/__init__.py re-export]
        → inference.stage1.run_stage1_snpe()
            → step2 (simulate) + step3 (features) + step4-8 (SNPE-C)

Stage 2:
  pipelines.stage2_pipeline()
    → inference.run_stage2_snpe()          [inference/__init__.py re-export]
        → inference.stage2.run_stage2_snpe()
            → same simulation + feature + SNPE-C chain

θ_bad:
  pipelines.select_theta_bad_from_val()
    → inference.select_theta_bad()         [inference/__init__.py re-export]
        → inference.stage2.select_theta_bad()

Artifact I/O:
  pipelines.save_and_summarize()
    → inference.save_artifacts()           [inference/__init__.py re-export]
        → inference.io.save_artifacts()
```

---

## 7. Package-Level Modules Actually Used

| Package | Called from | Via |
|---|---|---|
| `inference/` (whole package) | `pipelines/stage1_stage2.py` | `import inference` (package wins) |
| `simulation/wc_runner` | `inference/training_data.py` | `from simulation.wc_runner import simulate_gpu_batch` |
| `simulation/delays` | `simulator.py` compat wrapper → `data_loader.get_subject_data()` | `from simulator import compute_delay_matrix` |
| `simulation/warmup` | `inference/training_data.py` | (via simulation package internals) |
| `features/fc` | `inference/training_data.py`, `inference/posterior.py` | `from features.fc import ...` |
| `features/fcd` | `inference/training_data.py` | `from features.fcd import ...` (skipped if USE_FCD=False) |
| `features/extraction` | `inference/training_data.py` | `from features.extraction import ...` |
| `evaluation/` (whole package) | `pipelines/stage1_stage2.py` | via `evaluate.py` compat wrapper |
| `evaluation/validation` | `evaluate.py` compat wrapper | re-exported as `evaluate_validation_stage1/2` |
| `evaluation/model_selection` | `evaluate.py` compat wrapper | re-exported as `select_best_model` |
| `evaluation/final_test` | `evaluate.py` compat wrapper | re-exported as `final_test` |
| `evaluation/reports` | `evaluate.py` compat wrapper | re-exported as `print_final_summary` |

---

## 8. Root-Level Legacy Imports Detected

The following root-level legacy file imports were found in the inspected files:

### `pipelines/stage1_stage2.py` — line 38

```python
import evaluate   # line 38
```

**Status:** Active call — `evaluate` compat wrapper is used for **every** pipeline step
that calls evaluation functions: `evaluate_validation_stage1`, `evaluate_validation_stage2`,
`select_best_model`, `final_test`, `print_final_summary`.

**Risk:** `evaluate.py` is a compat wrapper with no logic. If it were deleted, the pipeline
would crash at `import evaluate` on module load. The fix is to change this import to
`import evaluation as evaluate` or update all call sites to use `evaluation.*`.

### `data_loader.py` — line 279 (inside `get_subject_data()`)

```python
from simulator import compute_delay_matrix   # line 279
```

**Status:** Active call — executed for every subject loaded in every pipeline run.
`simulator.py` re-exports `compute_delay_matrix` from `simulation.delays`.

**Risk:** If `simulator.py` were deleted, `data_loader.get_subject_data()` would raise
`ModuleNotFoundError` on the first subject load. The fix is to change this to
`from simulation.delays import compute_delay_matrix`.

### `pipeline_setup.py` — line 116

```python
_PIPELINE_MODULES = (
    "config", "data_loader", "bold", "simulator", "inference", "evaluate",
)
```

**Status:** Not a direct import, but lists `"simulator"` and `"evaluate"` as module names
for `sys.modules` cache invalidation in `reload_pipeline_modules()`. The reload logic
`del sys.modules[mod]` will drop cached entries for the compat wrappers but not for the
underlying package submodules. This makes `setup_pipeline()` only partially effective:
after reload, `pipelines.stage1_stage2` will re-import `evaluate.py` and `simulator.py`
as before, but `evaluation/` and `simulation/` submodule objects remain cached.

**Risk:** Incomplete module reload chain — a stale `evaluation.validation` or
`simulation.wc_runner` object will not be refreshed by `setup_pipeline()`.

---

## 9. Import Dependency Summary

```
main.py
  └── pipelines (package)
        └── pipelines/stage1_stage2.py
              ├── config                         [root, legitimate]
              ├── data_loader                    [root, legitimate]
              │     └── config                  [root]
              │     └── simulator.compute_delay_matrix  ← COMPAT WRAPPER ⚠
              │           └── simulation.delays.compute_delay_matrix
              ├── evaluate                       ← COMPAT WRAPPER ⚠
              │     └── evaluation.*             [package, all functions]
              └── inference                      [package, Python 3 wins]
                    ├── simulation.*             [simulate_gpu_batch, delays, warmup]
                    ├── features.*               [fc, fcd, extraction]
                    └── inference.*              [scaling, priors, pipeline, snpe, etc.]

pipeline_setup.py  [optional, called from notebook or before main.py]
  └── config  [mutates module-level attributes]
  └── sys.modules manipulation for reload (partial coverage)
```

---

## 10. Potential Import Conflict Points

### Conflict 1 — `import inference` in `pipelines/stage1_stage2.py` (LOW risk at runtime)

`import inference` on line 39 resolves to `inference/` package at runtime (Python 3
package-wins rule). **Runtime is safe.** However:

- Any editor or linter that resolves by filename may point to `inference.py` (55 KB
  monolith), causing false "attribute not found" warnings for names that exist in
  `inference/__init__.py` but not in the monolith's exports.
- If `inference.py` were ever compiled into `__pycache__/inference.cpython-310.pyc`
  before `inference/` existed, a stale `.pyc` could shadow the package.

### Conflict 2 — `import evaluate` in `pipelines/stage1_stage2.py` (MEDIUM risk)

`evaluate.py` is a compat wrapper that re-exports from `evaluation/`. If `evaluate.py`
content ever diverges from `evaluation/__init__.py` exports, callers will silently get
stale re-exports. Currently there is no enforcement that these stay in sync.

Additionally, the name `evaluate` collides with the common stdlib-adjacent pattern
(`evaluate` as a verb is a popular module name). Any future `pip install` that creates
a top-level `evaluate` would be shadowed by the root file, or vice versa.

### Conflict 3 — `from simulator import compute_delay_matrix` in `data_loader.py` (MEDIUM risk)

`simulator.py` re-exports from `simulation/`. `data_loader.py` is one of the most
foundational modules (called first in every pipeline run). If `simulator.py` is removed
before this import is updated, the failure will appear as a data-loading error rather
than a simulation error, making it harder to diagnose.

### Conflict 4 — `pipeline_setup._PIPELINE_MODULES` lists legacy names (LOW risk)

The reload list `("config", "data_loader", "bold", "simulator", "inference", "evaluate")`
will not correctly flush `evaluation.*` submodules from `sys.modules` — only `evaluate`
(the wrapper) is dropped. After `setup_pipeline()`, `evaluation.metrics`, `evaluation.plots`,
etc. retain their old cached objects. This can cause stale-config bugs in notebook sessions
where `setup_pipeline()` is called to change parameters mid-session.

### Conflict 5 — README.md documents pre-refactor architecture (INFO)

`README.md` lists `simulator.py`, `inference.py`, `evaluate.py` as primary pipeline
files in its Files table. New contributors following the README will reference dead or
compat files rather than the authoritative packages. This is a documentation-only issue
but directly increases the risk of future bare imports into root duplicates.

---

## 11. Package-Level Imports Detected (clean paths)

These import patterns were observed and are correct:

```python
# pipelines/stage1_stage2.py — inference package (correct)
import inference                         # → inference/__init__.py

# data_loader.py — config (correct)
import config

# config.py — stdlib only (correct)
import os
import torch  # optional, guarded

# main.py — pipelines package (correct)
from pipelines import run_pipeline

# pipeline_setup.py — config mutated at runtime (correct)
import config  # inside _apply_to_config()
```

---

## 12. Recommended Next File Group to Inspect

The two active compat-wrapper call sites (`evaluate.py` and `simulator.py`) are the
highest-priority blockers for root-level cleanup. Inspect their contents to understand
exactly which names they re-export and whether the re-exports are complete.

### Next group — Compat wrappers and their counterparts

| File | Why inspect | What to verify |
|---|---|---|
| `evaluate.py` | Called by `pipelines/stage1_stage2.py` for all evaluation | Which names it re-exports; whether `evaluation/__init__.py` exports the same superset |
| `evaluation/__init__.py` | Authoritative evaluation exports | Whether every name used via `evaluate.*` in the pipeline exists here |
| `simulator.py` | Called by `data_loader.get_subject_data()` for delay matrix | Which names it re-exports; specifically `compute_delay_matrix` |
| `simulation/__init__.py` | Authoritative simulation exports | Whether `compute_delay_matrix` and all other names used via `simulator.*` exist here |

### After that — Inference internals (simulation call chain)

| File | Why inspect |
|---|---|
| `inference/stage1.py` | Entry point for Stage 1; calls training_data + snpe |
| `inference/training_data.py` | Where `simulate_gpu_batch` is actually called |
| `inference/snpe.py` | Steps 4-8; most complex inference logic |
| `inference/__init__.py` | Verify all names used in `pipelines/stage1_stage2.py` are exported |

### Grep commands to run first

```bash
# Confirm exactly which names pipelines uses from the evaluate wrapper
grep -n "evaluate\." pipelines/stage1_stage2.py

# Confirm exactly which names data_loader uses from simulator
grep -n "simulator\." data_loader.py
grep -n "from simulator" data_loader.py

# Check for any other files that import from root compat wrappers
grep -rn "from simulator import\|import simulator\b" --include="*.py" .
grep -rn "from evaluate import\|import evaluate\b" --include="*.py" .
grep -rn "from inference import\|import inference\b" --include="*.py" .

# Check notebook for legacy imports
grep -n "simulator\|evaluate\b\|from fc\|from wc_runner" main.ipynb | head -40
```

---

## Summary of Findings

| Finding | Severity | File | Line |
|---|---|---|---|
| `import evaluate` (compat wrapper) actively used | High | `pipelines/stage1_stage2.py` | 38 |
| `from simulator import compute_delay_matrix` actively used | High | `data_loader.py` | 279 |
| `_PIPELINE_MODULES` lists legacy names; reload is incomplete | Medium | `pipeline_setup.py` | 116 |
| `import inference` resolves correctly but monolith still on disk | Low | `pipelines/stage1_stage2.py` | 39 |
| `data_loader` mutates `config` at runtime | Low | `data_loader.py` | 161–188 |
| README documents pre-refactor file list | Info | `README.md` | 37–45 |

**The two compat wrappers (`evaluate.py` and `simulator.py`) are live dependencies,
not dead code. They cannot be deleted until `pipelines/stage1_stage2.py` and
`data_loader.py` are updated to use the authoritative package imports.**
