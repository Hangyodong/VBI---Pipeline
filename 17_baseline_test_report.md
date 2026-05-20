# 17 — Baseline Test Report

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7
**Repository:** `/scratch/home/wog3597/vbi`
**Branch:** `refactor/02-simulation`
**Predecessor docs read:** 06, 07, 11, 16
**Purpose:** Capture the **pre-patch** state of the repository. Every check
in this report was read-only. No Python source, no documentation, and no
files on disk were modified.

---

## 1. Commands Run

All commands executed from the repository root. None modified disk state
except `python -m compileall` (which writes `__pycache__/` byte-cache —
harmless and easily purgeable).

| # | Command | Purpose |
|---|---|---|
| 1 | `pwd` | Confirm working directory |
| 2 | `python --version` | Confirm interpreter version |
| 3 | `ls -lh *.md` | Inventory documentation |
| 4 | `python -c "<importlib.util.find_spec loop>"` | Resolve 13 module names |
| 5 | `python -c "<import each package>"` | Smoke import the 5 packages |
| 6 | `python -m compileall -q .` | Compile every `.py` in the repo |
| 7 | `python -m py_compile <39 files>` | Targeted compile re-check of every authoritative source file |
| 8 | `grep -rn "^import X\b\|^from X import" --include="*.py" .` | Detect bare-name imports for each of 8 root duplicate names + `simulator` + `evaluate` |
| 9 | `grep -rn "^from <pkg>\|^import <pkg>" --include="*.py" .` | Detect package-prefixed imports for each of 5 packages |
| 10 | `grep -n "import simulator\|import evaluate\b\|..." main.ipynb` | Detect notebook legacy imports |
| 11 | R3-rule grep across `simulation/`, `features/`, `inference/`, `evaluation/` | Confirm dependency-rule status |

Excluded from this baseline (would change state or require GPU):
`debug.py`, `debug.py --basic`, any pipeline run, any `git` write operations.

---

## 2. Results

### 2.1 Environment

| Item | Value |
|---|---|
| Working directory | `/scratch/home/wog3597/vbi` |
| Python version | **`3.13.9`** |
| `compileall` exit code | `0` (clean) |
| Documentation count | 16 markdown files (01–16 + `PATCH_REPORT.md`, `README.md`) |
| Latest doc size | `16_opus_refactor_decision_plan.md`: 51 KB |

> **Note:** doc 05 (`05_runbook.md`) lists "Python 3.10, CUDA 12.x, H100 NVL"
> as the production target. The installed interpreter is **3.13.9** — three
> minor versions ahead. This is the largest single environmental drift
> between docs and reality. None of the imports tested below failed because
> of it, but any future fix that depends on stdlib behaviour (e.g.,
> `importlib.metadata`, `typing` syntax) may need a version check.

### 2.2 Markdown inventory

```
01_repo_overview.md                5.7K  May 18 12:18
02_repo_tree.md                    5.7K  May 18 12:18
03_module_index.md                  11K  May 18 12:19
04_data_flow.md                    7.3K  May 18 12:20
05_runbook.md                      5.0K  May 18 12:21
06_known_errors.md                 7.6K  May 18 12:21
07_refactor_rules.md               6.5K  May 18 12:22
08_claude_project_upload_guide.md  5.8K  May 18 12:23
09_architecture_review.md           19K  May 18 12:28
10_entrypoint_flow.md               20K  May 18 12:32
11_import_audit.md                  24K  May 18 12:41
12_simulation_module_review.md      37K  May 18 12:46
13_feature_module_review.md         37K  May 18 12:53
14_inference_module_review.md       40K  May 18 14:32
15_evaluation_module_review.md      42K  May 18 14:49
16_opus_refactor_decision_plan.md   51K  May 18 14:56
PATCH_REPORT.md                    1.8K  May 15 16:23
README.md                          2.9K  May 15 19:23
```

All 16 numbered docs present. `15_evaluation_module_review.md` is now on
disk (was missing during the doc 16 first pass — see doc 15 / doc 16
update notice).

### 2.3 Compile check

```
python -m compileall -q .
exit code: 0
```

**Every `.py` file in the repository compiles clean.** The monolith
`inference.py` (with its known `n_subj` NameError in `collect_stage2_data`)
compiles fine because the bug is a runtime NameError, not a syntax error.
This is consistent with doc 14 / doc 16 findings.

Targeted re-compile of the 39 authoritative source files plus the 12
legacy/duplicate files: all pass.

### 2.4 Smoke import of the 5 packages

```
simulation    OK   /scratch/home/wog3597/vbi/simulation/__init__.py
features      OK   /scratch/home/wog3597/vbi/features/__init__.py
inference     OK   /scratch/home/wog3597/vbi/inference/__init__.py
evaluation    OK   /scratch/home/wog3597/vbi/evaluation/__init__.py
pipelines     OK   /scratch/home/wog3597/vbi/pipelines/__init__.py
```

All 5 packages import without error. Each resolves to its `__init__.py`
(package-wins precedence confirmed; the `inference` resolution is the
critical case — it correctly beats the 55 KB `inference.py` monolith).

---

## 3. Failures

**Zero failures during this baseline pass.**

- 0 import errors
- 0 syntax errors
- 0 R3 violations beyond the one already documented (`simulation/qc.py:25`)
- 0 stale `__pycache__` shadows detected
- 0 inconsistencies between actual disk state and the doc 16 working-tree
  snapshot

The repository is in a runtime-clean state today.

---

## 4. Import Resolution Table

Output of `importlib.util.find_spec(name)` for every name relevant to
the refactor. Column "Shadow risk" = whether a bare-name `import X`
would resolve to a root file instead of the intended package submodule.

| Name | Kind | Resolved file | Authoritative? | Shadow risk |
|---|---|---|---|---|
| `inference` | package | `inference/__init__.py` | ✅ Yes — package wins over `inference.py` | LOW (package precedence verified) |
| `simulation` | package | `simulation/__init__.py` | ✅ Yes | None |
| `features` | package | `features/__init__.py` | ✅ Yes | None |
| `evaluation` | package | `evaluation/__init__.py` | ✅ Yes | None |
| `pipelines` | package | `pipelines/__init__.py` | ✅ Yes | None |
| `fc` | module | `./fc.py` | ❌ Root duplicate of `features/fc.py` | HIGH — any bare `import fc` hits root |
| `fcd` | module | `./fcd.py` | ❌ Root duplicate of `features/fcd.py` | HIGH |
| `wc_runner` | module | `./wc_runner.py` | ❌ Root duplicate of `simulation/wc_runner.py` | HIGH |
| `delays` | module | `./delays.py` | ❌ Root duplicate of `simulation/delays.py` | HIGH |
| `warmup` | module | `./warmup.py` | ❌ Root duplicate of `simulation/warmup.py` | HIGH |
| `qc` | module | `./qc.py` | ❌ Root duplicate of `simulation/qc.py` | HIGH |
| `simulator` | module | `./simulator.py` | ⚠ Compat wrapper (re-exports from packages) | n/a — intended use |
| `evaluate` | module | `./evaluate.py` | ⚠ Compat wrapper (re-exports from `evaluation/`) | n/a — intended use |

**Shadow status: dormant.** Section 5 confirms that none of the
HIGH-risk shadows is currently triggered by any caller. The risk is
latent; deletion of the duplicates (Tier 6 of doc 16) eliminates it.

`importlib.util.find_spec` did **not** find a bare `extraction` or
`screening` because those names were not in the queried set — but they
do exist as root duplicates. For completeness:

```
./extraction.py   shadows features/extraction.py     HIGH (dormant)
./screening.py    shadows features/screening.py      HIGH (dormant)
./__init__.py     makes /scratch/home/wog3597/vbi/   loadable as `vbi` (only if parent dir in sys.path)
```

---

## 5. Risky Imports Detected

### 5.1 Bare-name imports of root duplicates — **NONE FOUND**

```
^import fc\b              0 matches
^from fc import            0
^import fcd\b              0
^from fcd import           0
^import wc_runner\b        0
^from wc_runner import     0
^import delays\b           0
^from delays import        0
^import warmup\b           0
^from warmup import        0
^import qc\b               0
^from qc import            0
^import extraction\b       0
^from extraction import    0
^import screening\b        0
^from screening import     0
```

**Confirmed: the 8 root duplicates have zero bare-name callers.** The
shadow risk recorded in Section 4 is purely theoretical — no production,
notebook, or debug file invokes any of these by bare name. Deletion in
Tier 6 is safe **once** the deferred-import migrations in Tiers 1–4 land
and the compat wrappers are gone.

### 5.2 `import evaluate` — **1 TOP-LEVEL OCCURRENCE**

```
pipelines/stage1_stage2.py:38:import evaluate
```

This is the **single load-bearing legacy import in production code**. It
fires at module load every time the pipeline runs. Every `evaluate.X(...)`
call in that file routes through this name binding.

### 5.3 `from simulator import` / `import simulator` — 23 occurrences in 12 files

Categorised by where they appear and what they cost to migrate:

**A. Live package callers (12 sites, 7 files) — Tier 2–4 migration targets:**

| File | Line | Symbols | Migration tier |
|---|---|---|---|
| `data_loader.py` | 278 | `compute_delay_matrix` | Tier 2 |
| `inference/training_data.py` | 48 | `simulate_gpu_batch, worker_extract` | Tier 3 |
| `inference/stage2.py` | 158 | `simulate_gpu_batch, worker_extract` | Tier 3 |
| `inference/stage2.py` | 319 | `extract_observed_features` | Tier 3 |
| `inference/posterior.py` | 113 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, simulate_single` | Tier 3 |
| `inference/diagnostics.py` | 47 | `compute_fc, compute_sim_fcd_matrix, fc_to_upper_tri, fcd_to_upper_tri, simulate_single` | Tier 3 |
| `evaluation/metrics.py` | 110 | `extract_observed_features` | Tier 4 |
| `evaluation/metrics.py` | 184 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, simulate_single` | Tier 4 |
| `evaluation/metrics.py` | 245 | `compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, extract_observed_features, simulate_single` | Tier 4 |
| `evaluation/validation.py` | 123 | `extract_observed_features` | Tier 4 |
| `evaluation/final_test.py` | 112 | `extract_observed_features` | Tier 4 |
| `evaluation/plots.py` | 203 | `simulate_single, compute_fc` | Tier 4 |

Count matches doc 11's audit exactly (12 sites).

**B. Debug / notebook-helper callers (6 sites, 2 files) — Tier 4b:**

| File | Line | Symbols |
|---|---|---|
| `debug.py` | 443 | `fc_to_upper_tri` |
| `debug.py` | 466 | `compute_sim_fcd_matrix, fcd_to_upper_tri` |
| `debug.py` | 565 | `import simulator` |
| `debug_notebook.py` | 80 | `import simulator` |
| `debug_notebook.py` | 206 | `compute_fc, fc_to_upper_tri` |
| `debug_notebook.py` | 230 | `import simulator` |

**C. Dead-monolith callers (5 sites, 1 file) — Tier 7 (delete entire file):**

| File | Line | Note |
|---|---|---|
| `inference.py` | 387 | Inside dead `collect_training_data` |
| `inference.py` | 985 | Inside dead `posterior_predictive_check` |
| `inference.py` | 1070 | Inside dead `simulation_based_calibration` |
| `inference.py` | 1317 | Inside dead `collect_stage2_data` (the same function with the `n_subj` NameError) |
| `inference.py` | 1473 | Inside dead `run_stage2_snpe` |

Not migrated — `inference.py` is unreachable at runtime (package wins);
the file is deleted wholesale in Tier 7 after diff-audit.

**D. Cosmetic appearances in `simulator.py` docstring (lines 6–8) — NOT real imports.**

The grep also flagged `simulator.py:6/7/8`, but these are inside the
module's triple-quoted docstring as `from simulator import ...` example
text. The file is the compat wrapper itself; it does not import its own
public surface. No action.

### 5.4 Notebook legacy imports — 2 cells

```
main.ipynb:163:  "import evaluate\n"
main.ipynb:165:  "import simulator\n"
```

Both top-level (not deferred). Tier 4b addresses these together with the
debug-tool callers.

### 5.5 R3 dependency-rule violation — 1 site, persists

```
simulation/qc.py:25: from features.fc import compute_fc, fc_to_upper_tri
```

Identical line also visible in the root duplicate (`./qc.py:25`).
Otherwise:

- `simulation/*.py` → 0 other cross-package imports ✅
- `features/*.py` → 0 cross-package imports ✅
- `inference/*.py` → 0 evaluation imports ✅
- `evaluation/*.py` → 0 pipelines imports ✅

The single violation in `simulation/qc.py` is documented in doc 12 and
doc 16 Risk A5. Not a runtime bug; resolution is a Tier-X-style scoped
change requiring user authorization.

---

## 6. Package Imports Detected

All package-prefixed imports are correct and were observed in the
locations documented by doc 11 Section 3. Highlights:

### 6.1 Top-level production imports — all correct

| File | Line | Import |
|---|---|---|
| `main.py` | 42 | `from pipelines import run_pipeline` |
| `pipelines/stage1_stage2.py` | 39 | `import inference` |
| `pipelines/__init__.py` | 15 | `from pipelines.stage1_stage2 import run_pipeline` |

### 6.2 Cross-submodule package imports — counts by package

| Package | Number of `from <pkg>.X import ...` lines in `*.py` |
|---|---|
| `simulation.*` | 14 (split across `simulation/`, `simulator.py`, and root duplicates) |
| `features.*` | 12 |
| `inference.*` | 26+ (most concentrated in `inference/__init__.py` and `inference/stage2.py`) |
| `evaluation.*` | 11 |
| `pipelines.*` | 2 |

### 6.3 Root duplicate internal imports

Confirmed (consistent with doc 11 Section 3.3): root duplicates
internally import from packages, not from themselves. Example:

```
wc_runner.py:29:        from simulation.delays import apply_delay as _apply_delay
warmup.py:26:           from simulation.delays import apply_delay as _apply_delay
warmup.py:27:           from simulation.wc_runner import _apply_engine, _import_wc
extraction.py:27:       from features.fc import compute_fc, fc_to_upper_tri
extraction.py:28:       from features.fcd import compute_sim_fcd_matrix, fcd_to_summary_stats
qc.py:25:               from features.fc import compute_fc, fc_to_upper_tri
qc.py:26:               from simulation.wc_runner import simulate_gpu_batch
__init__.py (root):     same 4 `from simulation.*` lines as simulation/__init__.py
```

This means **bare-importing any root duplicate would still load the
package internals**. There is no scenario where a root duplicate executes
in isolation — the package is always pulled in transitively. This is a
relevant safety property for Tier 6 deletions.

### 6.4 Compat wrappers — verified shape

```
simulator.py  — 65 lines, re-exports only (no logic). Pulls from simulation.* and features.*
evaluate.py   — 35 lines, re-exports only (no logic). Pulls from evaluation
```

`evaluate.py` uses `from evaluation import *` plus an explicit re-import
of 8 private helpers (`_aggregate_validation`, `_print_selection_table`,
`_print_test_summary`, `_print_validation_summary`, `_progress`,
`_resimulate_and_score`, `_test_stage1`, `_test_stage2`) — matches doc 15
findings exactly.

---

## 7. Is the Current Repo Safe to Patch?

**Yes.** All 9 safety preconditions are satisfied:

| Precondition | Status | Evidence |
|---|---|---|
| All packages resolve to `__init__.py` (no stale `.pyc` shadow) | ✅ | Section 2.4 |
| `compileall` clean | ✅ | Section 2.3 |
| `inference/` package wins over `inference.py` monolith | ✅ | Section 4 |
| No bare `import fc / fcd / wc_runner / ...` callers anywhere | ✅ | Section 5.1 |
| Only one top-level legacy `import evaluate` | ✅ | Section 5.2 |
| All `from simulator import` calls are deferred (none top-level) | ✅ | Section 5.3 |
| R3 violations match the documented set (1 site, `simulation/qc.py`) | ✅ | Section 5.5 |
| Compat wrappers contain zero logic | ✅ | Section 6.4 |
| Doc 16's "files-must-not-touch-yet" list is consistent with current state | ✅ | Section 6 confirms `inference.py`, `simulator.py`, `evaluate.py` still on disk |

### Items the user must confirm before any patch lands

These are not failures of this baseline, but matters of user judgment.
Decisions belong to the user, not to this report:

1. **Working-tree status `M` on 6 files** (per doc 16 Appendix A):
   `config.py`, `debug_notebook.py`, `evaluate.py`, `main.ipynb`,
   `main.py`, `simulation/wc_runner.py`. Recommended: run
   `git diff <file>` for each before any patch lands, and confirm
   none of the M-status files contains scientific changes that need
   to be committed first.
2. **Python 3.13.9 vs documented 3.10**. Patches in this plan do not
   touch Python-version-sensitive syntax, but confirm the user's
   intended deployment target before assuming 3.13 is acceptable.
3. **Untracked `.md` files**. Docs 01–17 are not yet in git history.
   The user should decide whether to stage them with the first
   structural-cleanup commit, or as a separate "docs" commit.

---

## 8. Recommended First Patch Based on Actual Baseline

**No change to the recommendation from doc 16 Section 12.**

### Patch P-1 — `pipelines/stage1_stage2.py:38`

```diff
-import evaluate
+import evaluation as evaluate
```

### Why this is reinforced by today's baseline

| Baseline evidence | Implication for Patch P-1 |
|---|---|
| `import evaluate` is the **only** top-level legacy import in production (Section 5.2) | Removing it is the single biggest cleanup win for the smallest diff |
| `evaluation` resolves to `evaluation/__init__.py` cleanly (Section 4) | The right-hand side of the diff is already verified to work |
| `evaluation/__init__.py` re-exports every public name + 8 private helpers (Section 6.4 + doc 15) | All `evaluate.X` call sites in `pipelines/stage1_stage2.py` continue to resolve |
| Zero bare-imports of root duplicates anywhere (Section 5.1) | This patch has no interaction with the dormant shadow risk |
| `compileall` is clean before the patch (Section 2.3) | A regression after P-1 would be unambiguously caused by P-1 |
| `evaluate.py` itself is a 35-line zero-logic wrapper (Section 6.4 + doc 15) | Patch P-1 begins the process of making `evaluate.py` deletable in Tier 5 |

### Why not start with P-2 (`data_loader.py:278`) instead?

P-2 is a 1-line replacement inside `get_subject_data()`. It is also safe.
However:

- P-2 is **deferred** (inside a function body) — failure surfaces only at
  the first data load.
- P-1 is **top-level** — failure surfaces at module load, immediately
  visible.
- P-1 unblocks `evaluate.py` removal completely; P-2 only chips away at
  the 12-caller `simulator.py` migration.

The user is free to start with P-2 if they prefer to test the deferred-
import contract first; the baseline supports either choice. The default
recommendation remains P-1.

---

## 9. Commands to Rerun After Patch 1

Run these in order. Stop and revert P-1 at the first failure.

### 9.1 Purge stale pycache before re-test

```bash
find /scratch/home/wog3597/vbi -name "*.pyc" -delete 2>/dev/null
find /scratch/home/wog3597/vbi -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "pycache cleared"
```

### 9.2 Re-run compile check

```bash
python -m compileall -q .
# Expected exit code: 0
```

### 9.3 Re-run package resolution

```bash
python -c "
import inference, simulation, features, evaluation, pipelines
for m in (inference, simulation, features, evaluation, pipelines):
    assert m.__name__ + '/__init__' in m.__file__, m.__file__
print('all packages resolve correctly')
"
```

### 9.4 Re-run smoke import of public API

```bash
python -c "
from pipelines import run_pipeline
from inference import (
    ParameterScaler, FeaturePipeline, FeatureEmbedding,
    run_stage1_snpe, run_stage2_snpe, save_artifacts, load_artifacts,
)
from evaluation import (
    fc_metrics, evaluate_validation_stage1, select_best_model,
    final_test, print_final_summary,
)
print('public-API smoke OK')
"
```

### 9.5 Patch-specific check — `evaluate` is now bound to `evaluation` package

```bash
python -c "
import pipelines.stage1_stage2 as pl
assert hasattr(pl, 'evaluate'), 'pl.evaluate missing'
assert pl.evaluate.__name__ == 'evaluation', f'pl.evaluate.__name__ = {pl.evaluate.__name__}'
assert 'evaluation/__init__' in pl.evaluate.__file__, pl.evaluate.__file__
# spot-check the names actually used by pipelines/stage1_stage2.py
for name in ('evaluate_validation_stage1', 'evaluate_validation_stage2',
             'select_best_model', 'final_test', 'print_final_summary',
             'fc_metrics', 'plot_posteriors', 'plot_fc_comparison',
             'plot_sbc_rank_histogram', 'plot_pca_diagnostic',
             'print_summary_two_stage', 'report_step1', 'report_step9',
             'report_step12', 'report_step14'):
    assert hasattr(pl.evaluate, name), f'pl.evaluate.{name} missing'
print('P-1 surface OK')
"
```

### 9.6 Confirm legacy top-level `import evaluate` is gone

```bash
grep -n "^import evaluate\b\|^from evaluate import" pipelines/*.py inference/*.py evaluation/*.py *.py
# Expected: zero output (the one line in pipelines/stage1_stage2.py is replaced)
```

### 9.7 Confirm everything else is unchanged

```bash
# Sanity: no other `import evaluate` regressed in
grep -rn "^import evaluate\b\|^from evaluate import" --include="*.py" --include="*.ipynb" .
# Expected: only main.ipynb cell 163, plus debug_notebook.py if applicable
#           (the notebook + debug callers are Tier 4b, not P-1's scope)

# Sanity: the simulator deferred imports are untouched (P-1 doesn't address them)
grep -rn "from simulator import\|import simulator\b" --include="*.py" . | wc -l
# Expected: count unchanged from baseline (23 total occurrences in *.py)
```

### 9.8 Optional — full debug smoke (no GPU)

```bash
python debug.py --basic
```

If this passes after P-1, the patch is verified end-to-end (modulo
anything that requires a GPU or real data files).

---

## Appendix A — Raw Resolution Listing

For audit traceability, the verbatim output of step 4 (resolution loop):

```
inference       package   /scratch/home/wog3597/vbi/inference/__init__.py
simulation      package   /scratch/home/wog3597/vbi/simulation/__init__.py
features        package   /scratch/home/wog3597/vbi/features/__init__.py
evaluation      package   /scratch/home/wog3597/vbi/evaluation/__init__.py
pipelines       package   /scratch/home/wog3597/vbi/pipelines/__init__.py
fc              module    /scratch/home/wog3597/vbi/fc.py
fcd             module    /scratch/home/wog3597/vbi/fcd.py
wc_runner       module    /scratch/home/wog3597/vbi/wc_runner.py
delays          module    /scratch/home/wog3597/vbi/delays.py
warmup          module    /scratch/home/wog3597/vbi/warmup.py
qc              module    /scratch/home/wog3597/vbi/qc.py
simulator       module    /scratch/home/wog3597/vbi/simulator.py
evaluate        module    /scratch/home/wog3597/vbi/evaluate.py
```

## Appendix B — Doc-vs-Reality Consistency Check

Every item in this report cross-references the predecessor docs. No
discrepancies found between this baseline and docs 11 / 16.

| Predecessor claim | Baseline result | Match? |
|---|---|---|
| 5 packages resolve correctly | ✅ Section 2.4 | YES |
| All 8 root duplicates exist | ✅ Section 4 | YES |
| `inference.py` is 55 KB monolith, package wins | ✅ Section 4 | YES |
| 12 deferred `from simulator import` calls in package code | ✅ Section 5.3 list A | YES (exact match) |
| 1 top-level `import evaluate` | ✅ Section 5.2 | YES |
| 2 notebook legacy import cells | ✅ Section 5.4 | YES |
| R3 violation in `simulation/qc.py:25` only | ✅ Section 5.5 | YES |
| Compat wrappers contain no logic | ✅ Section 6.4 | YES |
| Zero bare-name callers of root duplicates | ✅ Section 5.1 | YES |
| `compileall` is clean | ✅ Section 2.3 | YES |

The baseline is consistent with the doc 16 plan. The recommended first
patch is unchanged: `pipelines/stage1_stage2.py:38`,
`import evaluate` → `import evaluation as evaluate`.
