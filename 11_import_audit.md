# 11 — Import Audit

**Date:** 2026-05-18
**Branch:** refactor/02-simulation
**Auditor:** Claude (claude-sonnet-4-6)
**Sources:** `import_legacy_candidates.txt`, `import_package_candidates.txt`,
`09_architecture_review.md`, `10_entrypoint_flow.md`, `03_module_index.md`,
`06_known_errors.md`, `07_refactor_rules.md`, live grep runs.

---

## 1. Summary of Import Resolution Results

| Concern | Result |
|---|---|
| `import inference` runtime resolution | **Package wins.** `inference/__init__.py` confirmed via `python -c "import inference; print(inference.__file__)"` |
| Root duplicate drift vs. package | **Zero drift.** All 8 root duplicates are byte-for-byte identical to their package counterparts (diff exit 0). |
| Root `__init__.py` vs. `simulation/__init__.py` | **Identical** (diff exit 0). |
| Top-level legacy imports in production code | **One.** `pipelines/stage1_stage2.py:38` — `import evaluate` |
| Top-level legacy imports of bare root-duplicate names | **None.** No file uses `import fc`, `import wc_runner`, `import delays`, etc. at the top level. |
| Deferred (inside-function) legacy simulator imports | **Active in 9 package submodules** — see Section 3. |
| Notebook (`main.ipynb`) legacy imports | **Two cells.** `import evaluate` (cell ~163) and `import simulator` (cell ~165). |

**Bottom line:** the modular package layer is structurally complete and resolves correctly at
runtime. The remaining legacy surface is one top-level production import (`import evaluate`)
and a widespread pattern of deferred `from simulator import …` calls inside function bodies
across `inference/` and `evaluation/` submodules. No root duplicate file has drifted from
its package counterpart.

---

## 2. Legacy Bare Imports Detected

### 2.1 Top-level module-scope legacy imports (highest priority)

These execute at module load time, before any function is called.

| File | Line | Import | Risk |
|---|---|---|---|
| `pipelines/stage1_stage2.py` | 38 | `import evaluate` | **Critical.** This is the orchestration driver. Every pipeline run goes through this file. If `evaluate.py` is deleted before this line is updated, the pipeline crashes at import. |
| `main.ipynb` | cell ~163 | `import evaluate` | High. Notebook cell; will fail if compat wrapper removed. |
| `main.ipynb` | cell ~165 | `import simulator` | High. Notebook cell. |

### 2.2 Deferred (inside-function) legacy imports

All of the following are indented inside function bodies. They execute only when the enclosing
function is called, not at module load time. However, several are called in every pipeline run.

#### `data_loader.py`

| Line | Import | Called by | Frequency |
|---|---|---|---|
| 278 | `from simulator import compute_delay_matrix` | `get_subject_data()` | **Every subject load in every pipeline run** |

**Note:** `get_subject_data()` is called by `load_all_subjects()` which is called once per
pipeline run for all subjects. This is effectively always executed.

#### `inference/` package submodules

| File | Line | Symbols imported from `simulator` | When called |
|---|---|---|---|
| `inference/training_data.py` | 48 | `simulate_gpu_batch, worker_extract` | Inside simulate-train step — called every Stage 1 and Stage 2 run |
| `inference/stage2.py` | 158 | `simulate_gpu_batch, worker_extract` | Inside Stage 2 simulation step |
| `inference/stage2.py` | 319 | `extract_observed_features` | Inside Stage 2 observed-features step |
| `inference/posterior.py` | 113 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, simulate_single` | Inside `posterior_predictive_check()` |
| `inference/diagnostics.py` | 47 | `compute_fc, compute_sim_fcd_matrix, fc_to_upper_tri, fcd_to_upper_tri, simulate_single` | Inside SBC / embedding probe |

#### `evaluation/` package submodules

| File | Line | Symbols imported from `simulator` | When called |
|---|---|---|---|
| `evaluation/metrics.py` | 110 | `extract_observed_features` | Inside `evaluate_subject()` |
| `evaluation/metrics.py` | 184 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, simulate_single` | Inside `baseline_eval()` |
| `evaluation/metrics.py` | 245 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, extract_observed_features, simulate_single` | Inside `baseline_eval_subjects()` |
| `evaluation/validation.py` | 123 | `extract_observed_features` | Inside `evaluate_validation_stage1/2()` — called every val step |
| `evaluation/final_test.py` | 112 | `extract_observed_features` | Inside `final_test()` — called once per run |
| `evaluation/plots.py` | 203 | `simulate_single, compute_fc` | Inside `plot_one_simulation()` |

#### Debug / notebook tools (lower urgency)

| File | Lines | Import |
|---|---|---|
| `debug.py` | 443 | `from simulator import fc_to_upper_tri` |
| `debug.py` | 466 | `from simulator import compute_sim_fcd_matrix, fcd_to_upper_tri` |
| `debug.py` | 565 | `import simulator` |
| `debug_notebook.py` | 80 | `import simulator` |
| `debug_notebook.py` | 206 | `from simulator import compute_fc, fc_to_upper_tri` |
| `debug_notebook.py` | 230 | `import simulator` |
| `debug_notebook.py` | 82 | `import evaluate` |
| `debug_notebook.py` | 341 | `import evaluate` |

#### `inference.py` dead monolith (not a concern at runtime)

`inference.py` contains five `from simulator import` occurrences (lines 387, 985, 1070,
1317, 1473). These are in dead code — the file is never loaded at runtime because
`inference/` package wins. They are recorded here for completeness; do not edit `inference.py`
(R8).

---

## 3. Package Imports Detected (Correct Paths)

The following import patterns are confirmed correct and should be used as the target form
for all migrations.

### 3.1 Top-level production imports — all correct

| File | Import | Status |
|---|---|---|
| `main.py:42` | `from pipelines import run_pipeline` | ✓ |
| `pipelines/stage1_stage2.py:39` | `import inference` | ✓ (resolves to `inference/` package) |
| `pipelines/__init__.py:15` | `from pipelines.stage1_stage2 import run_pipeline` | ✓ |

### 3.2 Submodule-internal package imports — all correct

| Package | Pattern confirmed |
|---|---|
| `simulation/wc_runner.py` | `from simulation.delays import apply_delay` |
| `simulation/warmup.py` | `from simulation.delays import …`, `from simulation.wc_runner import …` |
| `simulation/delays.py` | `from simulation.wc_runner import _import_wc` (lazy) |
| `features/extraction.py` | `from features.fc import …`, `from features.fcd import …` |
| `inference/stage1.py` | `from inference.snpe import …`, `from inference.training_data import …` |
| `inference/stage2.py` | `from inference._utils import …`, `from inference.feature_pipeline import …`, etc. |
| `inference/snpe.py` | `from inference._utils import …`, `from inference.embedding import …`, etc. |
| `inference/__init__.py` | all `from inference.*` |
| `evaluation/__init__.py` | all `from evaluation.*` |
| `evaluation/validation.py` | `from evaluation.metrics import …`, `from inference import …` |

### 3.3 Root duplicates — internally correct despite being legacy files

Every root duplicate file already uses package-level imports internally. They are not
self-contained — they delegate to the package on first import:

| Root file | Internal import |
|---|---|
| `wc_runner.py` | `from simulation.delays import apply_delay` |
| `warmup.py` | `from simulation.delays import …`, `from simulation.wc_runner import …` |
| `delays.py` | `from simulation.wc_runner import _import_wc` |
| `extraction.py` | `from features.fc import …`, `from features.fcd import …` |
| `__init__.py` (root) | all `from simulation.*` |

---

## 4. High-Risk Import Conflicts

### CONFLICT-1 — `inference.py` monolith on disk alongside `inference/` package

**Risk level:** HIGH for tooling; LOW at runtime.

`python -c "import inference; print(inference.__file__)"` confirms:
```
/scratch/home/wog3597/vbi/inference/__init__.py
```
The package wins at runtime. However:
- Editors and static analysers (pylint, pyright, Pylance) may resolve `import inference` to
  `inference.py` (55 KB), causing false "attribute not found" errors for names that only
  exist in `inference/__init__.py`.
- Any developer opening `inference.py` and editing it will see no runtime effect — silent
  divergence risk.
- Stale `__pycache__/inference.cpython-*.pyc` from before `inference/` existed could shadow
  the package on some Python versions.

**Do not edit `inference.py` (R8).** Safe to mark as legacy candidate now; requires diff
audit before deletion.

### CONFLICT-2 — `import evaluate` in `pipelines/stage1_stage2.py:38`

**Risk level:** CRITICAL — only one line to fix, but it is on the hot path.

`evaluate.py` is a compat wrapper. Its only role is re-exporting from `evaluation/`.
If it is deleted before this line is updated, the pipeline crashes at the `import evaluate`
statement — before any data loads or simulation starts.

**The fix is a single-line change.** See Section 6.

### CONFLICT-3 — `from simulator import compute_delay_matrix` in `data_loader.py:278`

**Risk level:** HIGH — hot path, and the function name is critical.

Called inside `get_subject_data()` which runs for every subject in every pipeline run.
`simulator.py` re-routes this to `simulation.delays.compute_delay_matrix`. If `simulator.py`
is removed before this is updated, the error surfaces as a data-loading failure during
subject processing — potentially misleading (looks like a data error, not an import error).

**The fix is a single-line change.** See Section 6.

### CONFLICT-4 — Root `__init__.py` makes repo root importable as `vbi` package

**Risk level:** MEDIUM (latent).

`/scratch/home/wog3597/vbi/__init__.py` exists and is identical to `simulation/__init__.py`.
If `/scratch/home/wog3597` is ever added to `sys.path` (e.g., by a Jupyter notebook kernel
that adds parent dirs), `import vbi` succeeds and exposes simulation-package symbols at
the root level. This is not the intended public interface.

**Currently latent.** Becomes active only if parent dir enters `sys.path`.

### CONFLICT-5 — Root duplicate `wc_runner.py` name shadows `simulation/wc_runner`

**Risk level:** MEDIUM (conditional).

`import wc_runner` resolves to root `wc_runner.py`, not `simulation/wc_runner.py`. Since
no file currently uses a bare `import wc_runner` (confirmed by grep), this shadowing is
dormant. Content is also byte-for-byte identical. Risk activates if someone writes
`import wc_runner` in new code without the package prefix — a very likely mistake given
the file's name.

Same pattern applies to all other root duplicates (`fc.py`, `delays.py`, etc.) but none
currently have active bare-import callers.

### CONFLICT-6 — `pipeline_setup._PIPELINE_MODULES` lists legacy names (incomplete reload)

**Risk level:** LOW — affects notebook sessions only.

`pipeline_setup.py:116` defines:
```python
_PIPELINE_MODULES = ("config", "data_loader", "bold", "simulator", "inference", "evaluate")
```
`setup_pipeline()` drops these from `sys.modules` to force re-import. However:
- Dropping `"evaluate"` does not flush `evaluation.metrics`, `evaluation.validation`, etc. —
  those submodule objects remain cached.
- Dropping `"simulator"` does not flush `simulation.wc_runner`, `simulation.delays`, etc.
- Result: after `setup_pipeline()` changes a config value, already-loaded evaluation and
  simulation submodules still hold references to pre-change config values.

---

## 5. Files That Need Manual Inspection Before Any Deletion

Do not delete or rename any of the following without running the checks noted:

| File | Reason | Check required |
|---|---|---|
| `inference.py` | 55 KB monolith; must confirm 100% of logic is replicated in `inference/` submodules | Full diff against `inference/` package content; grep for any name exported in `inference.py` but absent in `inference/__init__.py` |
| `simulator.py` | Compat wrapper; must confirm every symbol it re-exports has a working equivalent in the package | Read `simulator.py` exports vs. every `from simulator import` call site in Sections 2.2/2.3 |
| `evaluate.py` | Compat wrapper; same concern as `simulator.py` | Read `evaluate.py` exports vs. every `evaluate.*` call in `pipelines/stage1_stage2.py` and `debug_notebook.py` |
| `main.ipynb` | Notebook cells use `import evaluate`, `import simulator`, and `simulator.simulate_single()` | Grep / open notebook; update affected cells before removing compat wrappers |
| `debug.py` | Imports `simulator` names in three functions | Update lines 443, 466, 565 before removing `simulator.py` |
| `debug_notebook.py` | Imports `simulator` and `evaluate` in multiple cells | Update lines 80, 82, 206, 230, 341 before removing compat wrappers |
| `pipeline_setup.py:116` | Lists `"simulator"` and `"evaluate"` as module names for reload | Update `_PIPELINE_MODULES` after compat wrappers are removed (R7 compliance) |
| `wc_runner.py` (root) | Listed as modified in `refactor/02-simulation` branch (git status `M`) | Already confirmed identical to `simulation/wc_runner.py` via `diff` (exit 0); safe to mark legacy, but verify final git diff before deleting |

---

## 6. Minimal Future Patch Plan

Apply patches in strict order. Each tier depends on the previous tier being complete and
tested. **Do not edit Python code until the diff/grep checks in 09_architecture_review.md
Section 8 are complete.**

### Tier 1 — Remove the two live top-level compat wrapper dependencies (2 files, 2 lines)

These are the only blockers for eventually removing `evaluate.py` and `simulator.py`.

**Patch 1-A: `pipelines/stage1_stage2.py:38`**

```python
# Before
import evaluate

# After — option A (drop-in replacement preserving call syntax)
import evaluation as evaluate

# After — option B (explicit, per R7)
from evaluation import (
    evaluate_validation_stage1, evaluate_validation_stage2,
    select_best_model, final_test, print_final_summary,
    evaluate_all_two_stage, print_summary_two_stage,
)
```

Option A is a one-character change and preserves all existing `evaluate.X` call sites.
Option B is more explicit but requires grepping all `evaluate.` usages first.

**Patch 1-B: `data_loader.py:278`**

```python
# Before
from simulator import compute_delay_matrix

# After
from simulation.delays import compute_delay_matrix
```

### Tier 2 — Update deferred `from simulator import` in `inference/` submodules (5 files)

Apply in this order (innermost → outermost):

| File | Line | Replace with |
|---|---|---|
| `inference/training_data.py` | 48 | `from simulation.wc_runner import simulate_gpu_batch` + `from features.extraction import worker_extract` |
| `inference/stage2.py` | 158 | Same as `training_data.py` |
| `inference/stage2.py` | 319 | `from features.extraction import extract_observed_features` |
| `inference/posterior.py` | 113 | `from simulation.wc_runner import simulate_single` + `from features.fc import compute_fc` + `from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri` |
| `inference/diagnostics.py` | 47 | Same as `posterior.py` + `from features.fc import fc_to_upper_tri` |

### Tier 3 — Update deferred `from simulator import` in `evaluation/` submodules (4 files)

| File | Lines | Replace with |
|---|---|---|
| `evaluation/validation.py` | 123 | `from features.extraction import extract_observed_features` |
| `evaluation/final_test.py` | 112 | `from features.extraction import extract_observed_features` |
| `evaluation/metrics.py` | 110, 184, 245 | `from features.extraction import extract_observed_features` + `from simulation.wc_runner import simulate_single` + `from features.fc import compute_fc` + `from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri` |
| `evaluation/plots.py` | 203 | `from simulation.wc_runner import simulate_single` + `from features.fc import compute_fc` |

### Tier 4 — Update debug tools and notebook (lower urgency)

| File | Lines | Action |
|---|---|---|
| `debug.py` | 443, 466, 565 | Replace with `features.fc`, `features.fcd` package imports |
| `debug_notebook.py` | 80, 82, 206, 230, 341 | Replace with `simulation.*`, `features.*`, `evaluation.*` |
| `main.ipynb` | cells ~163, ~165, ~454 | Replace with package imports |
| `pipeline_setup.py` | 116 | Update `_PIPELINE_MODULES` to `"simulation", "features", "inference", "evaluation"` |

### Tier 5 — Remove compat wrappers (after Tiers 1–4 are complete and tested)

After all callers have been migrated:

1. Confirm zero remaining `from simulator import` or `import simulator` in `*.py` and
   `*.ipynb` (see test commands, Section 7).
2. Confirm zero remaining `import evaluate` or `from evaluate import` (except inside
   `evaluate.py` itself).
3. Delete `simulator.py`.
4. Delete `evaluate.py`.

### Tier 6 — Remove root duplicate files (after Tier 5)

All 8 root duplicates are currently byte-for-byte identical to their package counterparts.
Deletion order (least risk first):

1. `screening.py` — future stub, no logic
2. `fcd.py` — FCD disabled in production
3. `delays.py` — stable
4. `warmup.py` — stable
5. `qc.py` — stable
6. `fc.py` — stable
7. `extraction.py` — stable
8. `wc_runner.py` — modified in current branch (but diff confirmed identical); delete last
9. `__init__.py` (root) — makes repo importable as a package; delete after confirming parent
   dir is never in `sys.path`

### Tier 7 — Remove `inference.py` monolith (last, after full audit)

Only after verifying that `inference/__init__.py` exports a strict superset of everything
in `inference.py` and no tool is pointing at `inference.py` directly.

---

## 7. Test Commands to Run After Import Cleanup

Run these in order. All should pass before any compat wrapper is deleted.

### 7.1 Verify Python resolves all packages correctly

```bash
python -c "import inference; print(inference.__file__)"
# Expected: .../vbi/inference/__init__.py

python -c "from simulation import simulate_gpu_batch; print('simulation OK')"
python -c "from features import compute_fc; print('features OK')"
python -c "from evaluation import fc_metrics; print('evaluation OK')"
python -c "from pipelines import run_pipeline; print('pipelines OK')"
```

### 7.2 Verify no remaining simulator imports in package submodules

```bash
grep -rn "from simulator import\|import simulator\b" \
  inference/ evaluation/ pipelines/ --include="*.py"
# Expected: zero output
```

### 7.3 Verify no remaining evaluate compat imports

```bash
grep -rn "import evaluate\b\|from evaluate import" \
  pipelines/ inference/ evaluation/ --include="*.py"
# Expected: zero output
```

### 7.4 Verify no top-level bare imports of root-duplicate names

```bash
grep -rn "^import fc\b\|^from fc import\|^import fcd\b\|^from fcd import" \
  --include="*.py" .
grep -rn "^import wc_runner\|^from wc_runner import\|^import delays\b\|^from delays import" \
  --include="*.py" .
grep -rn "^import warmup\b\|^from warmup import\|^import qc\b\|^from qc import" \
  --include="*.py" .
grep -rn "^import extraction\b\|^import screening\b" --include="*.py" .
# All expected: zero output
```

### 7.5 Compile-check all package modules (no GPU required)

```bash
python -m py_compile config.py data_loader.py bold.py main.py pipeline_setup.py
python -m py_compile simulation/wc_runner.py simulation/delays.py \
  simulation/warmup.py simulation/qc.py
python -m py_compile features/fc.py features/fcd.py features/extraction.py \
  features/screening.py
python -m py_compile inference/scaling.py inference/priors.py \
  inference/feature_pipeline.py inference/embedding.py
python -m py_compile inference/training_data.py inference/snpe.py \
  inference/stage1.py inference/stage2.py
python -m py_compile inference/posterior.py inference/diagnostics.py \
  inference/io.py
python -m py_compile evaluation/metrics.py evaluation/validation.py \
  evaluation/model_selection.py evaluation/final_test.py \
  evaluation/plots.py evaluation/reports.py
python -m py_compile pipelines/stage1_stage2.py
echo "All modules compile-clean."
```

### 7.6 Check dependency-rule violations (R3)

```bash
# simulation/ must not import from features/, inference/, evaluation/
grep -n "from features\|from inference\|from evaluation" simulation/*.py
# Expected: zero output

# features/ must not import from simulation/, inference/, evaluation/
grep -n "from simulation\|from inference\|from evaluation" features/*.py
# Expected: zero output

# inference/ must not import from evaluation/
grep -n "from evaluation" inference/*.py
# Expected: zero output
```

### 7.7 Check for circular import risk in inference/

```bash
grep -n "from inference.stage1\|import stage1" inference/snpe.py
# Expected: zero output (snpe must not import stage1)

grep -n "from inference.snpe\|import snpe" inference/stage1.py inference/stage2.py
# Expected: present (stage1/stage2 imports from snpe — correct direction)
```

### 7.8 Purge stale pycache before final test

```bash
find /scratch/home/wog3597/vbi -name "*.pyc" -delete
find /scratch/home/wog3597/vbi -name "__pycache__" -type d -exec rm -rf {} +
echo "pycache cleared"
```

---

## Appendix A — Symbol-to-Package Mapping for Simulator Migration

Quick reference for replacing `from simulator import X`:

| Symbol | Correct package source |
|---|---|
| `simulate_gpu_batch` | `from simulation.wc_runner import simulate_gpu_batch` |
| `simulate_single` | `from simulation.wc_runner import simulate_single` |
| `worker_extract` | `from features.extraction import worker_extract` |
| `extract_observed_features` | `from features.extraction import extract_observed_features` |
| `extract_simulated_features` | `from features.extraction import extract_simulated_features` |
| `compute_delay_matrix` | `from simulation.delays import compute_delay_matrix` |
| `compute_fc` | `from features.fc import compute_fc` |
| `fc_to_upper_tri` | `from features.fc import fc_to_upper_tri` |
| `compute_sim_fcd_matrix` | `from features.fcd import compute_sim_fcd_matrix` |
| `fcd_to_upper_tri` | `from features.fcd import fcd_to_upper_tri` |
| `fcd_to_summary_stats` | `from features.fcd import fcd_to_summary_stats` |
| `warmup_run` | `from simulation.warmup import warmup_run` |
| `simulate_with_warmup` | `from simulation.warmup import simulate_with_warmup` |

---

## Appendix B — Root File Status Matrix

| File | Category | Byte-identical to package? | Active callers? | Safe to mark legacy? |
|---|---|---|---|---|
| `inference.py` | Dead monolith | N/A (not a duplicate) | None at runtime | Yes — but requires full content audit before deletion |
| `simulator.py` | Compat wrapper | N/A (re-export logic) | `data_loader.py:278` + 9 deferred in `inference/`+`evaluation/` | No — still has active callers |
| `evaluate.py` | Compat wrapper | N/A (re-export logic) | `pipelines/stage1_stage2.py:38` + debug tools | No — still has active callers |
| `wc_runner.py` | Root duplicate | **Yes** (diff exit 0) | None (bare `import wc_runner` grep: zero) | Yes |
| `fc.py` | Root duplicate | **Yes** | None | Yes |
| `fcd.py` | Root duplicate | **Yes** | None | Yes |
| `extraction.py` | Root duplicate | **Yes** | None | Yes |
| `screening.py` | Root duplicate | **Yes** | None | Yes |
| `delays.py` | Root duplicate | **Yes** | None | Yes |
| `warmup.py` | Root duplicate | **Yes** | None | Yes |
| `qc.py` | Root duplicate | **Yes** | None | Yes |
| `__init__.py` (root) | Package leak | **Yes** (= `simulation/__init__.py`) | None (`import vbi` not found) | Yes — after confirming parent dir never in `sys.path` |
