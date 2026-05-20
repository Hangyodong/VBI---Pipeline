# 18 — Patch 1 Specification: Import Cleanup

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** Specification only. **Patch not yet applied.**
**Predecessor docs:** 06, 07, 11, 16, 17
**Branch:** `refactor/02-simulation`

---

## 1. Patch Goal

Replace the one remaining **top-level production import** of the
`evaluate.py` compatibility wrapper with a direct package import, so
that:

- `evaluate.py` becomes one step closer to deletable (its top-level
  call site disappears).
- The single-file `evaluate.py` wrapper stops being part of the
  production module load graph at the orchestration layer.
- The change is reversible in one line if anything breaks.
- Zero scientific behavior changes.

**Scope of change:** exactly **one line** in exactly **one file**.

---

## 2. Why This Patch Is First

From the verified baseline (doc 17 Section 7):

| Property | Value |
|---|---|
| Top-level legacy `import evaluate` occurrences | **1** (`pipelines/stage1_stage2.py:38`) |
| `evaluation/__init__.py` resolves correctly at runtime | ✅ (doc 17 §2.4) |
| `evaluation/__init__.py` re-exports a superset of `evaluate.py` | ✅ (doc 15 §9, doc 17 §6.4) |
| `compileall .` exit code at baseline | `0` |
| `evaluation/` is R3-clean at module level | ✅ (doc 17 §5.5, doc 16 I6) |
| Failure surfaces at module load (immediate, unambiguous) | ✅ (top-level import) |
| Rollback complexity | 1 git revert / git checkout, no migration |

This is the smallest possible structural cleanup. It targets the single
top-level legacy import in production code. Any other Tier 1–4 patch
from doc 16 targets a deferred (inside-function) import — those are
also small, but failure surfaces later (at first call). P-1 fails fast
if it fails at all, making it the safest first step.

---

## 3. Exact Files to Modify

| # | File | Lines touched | Scope |
|---|---|---|---|
| 1 | `pipelines/stage1_stage2.py` | line **38** only | One single-line replacement |

**No other file is modified by this patch.**

---

## 4. Exact Import Lines to Change

Only one line changes. Line 38 of `pipelines/stage1_stage2.py`.

Surrounding context (lines 32–42), shown for orientation only — no
lines other than 38 are edited:

```python
# (line 32–41 of pipelines/stage1_stage2.py, current state)
import os
import warnings

import numpy as np

import config
import data_loader
import evaluate                       # <-- line 38, the one being edited
import inference

warnings.filterwarnings("ignore")
```

---

## 5. Old Import Line

```python
import evaluate
```

This binds the local name `evaluate` to the root-level
`/scratch/home/wog3597/vbi/evaluate.py` compat wrapper. That wrapper is
35 lines of pure re-exports (verified, doc 15 §9 / doc 17 §6.4):

```python
# evaluate.py (current — DO NOT MODIFY)
from evaluation import *                          # noqa: F401, F403
from evaluation import (                          # noqa: F401
    _aggregate_validation, _print_selection_table, _print_test_summary,
    _print_validation_summary, _progress, _resimulate_and_score,
    _test_stage1, _test_stage2,
)
```

Every name reachable via `evaluate.X` is therefore also reachable via
`evaluation.X` — verified.

---

## 6. New Import Line

```python
import evaluation as evaluate
```

This binds the local name `evaluate` directly to the `evaluation/`
package object (`evaluation/__init__.py`). Every `evaluate.X(...)`
call elsewhere in `pipelines/stage1_stage2.py` resolves to the **same
function object** as before — because `evaluate.py` was already getting
that object via `from evaluation import *`.

---

## 7. Why Each Change Is Safe

Single change, single argument:

### S1 — Name binding semantics are preserved

`evaluate.X` for any `X` previously resolved through `evaluate.py`'s
wildcard re-export to `evaluation.X`. After the patch, `evaluate.X`
directly accesses `evaluation.X`. **Same object.** No behavior change.

### S2 — Module load order is unchanged

Before: `import evaluate` triggers `evaluate.py` → `from evaluation import *` →
`evaluation/__init__.py` (loaded).
After: `import evaluation as evaluate` triggers `evaluation/__init__.py`
(loaded) directly.

The set of modules in `sys.modules` after the import is the same
**minus** `evaluate` (which we no longer need). `evaluate.py` is **not
deleted**; if anything else imports `evaluate`, it still resolves
correctly.

### S3 — Re-export coverage is complete

`evaluation/__init__.py` re-exports:

- All public names listed in `__all__` (33 functions/classes per doc 15)
- All 8 private helpers also re-exported by `evaluate.py`
  (`_aggregate_validation`, `_print_selection_table`,
  `_print_test_summary`, `_print_validation_summary`, `_progress`,
  `_resimulate_and_score`, `_test_stage1`, `_test_stage2`)

So `evaluate._foo` continues to work for any `_foo` previously reachable
via the wrapper.

### S4 — `pipelines/stage1_stage2.py` only calls public `evaluate.X` names

Pre-flight validation (run as part of test step T-1 below) confirms
every `evaluate.X` call site is one of the public re-exports. If any
underscored name slips in, T-1 catches it before runtime.

### S5 — Reversibility is trivial

One-line revert restores the prior state byte-for-byte.

### S6 — No scientific values change

No constants, weights, prior bounds, or function bodies are touched.
R1 is satisfied trivially (nothing to violate).

### S7 — No file renames, moves, deletes

`evaluate.py`, `evaluation/__init__.py`, and every other file on disk
is untouched.

### S8 — No new dependencies

`evaluation/` package is already in the repository and already loads
without error (doc 17 §2.4). No `pip install` or env change.

### S9 — No `sys.path` manipulation

The patch is pure import-name rebinding. Python's standard
package-precedence resolution does all the work.

---

## 8. Risks

The patch is small, but a few risks deserve explicit recording.

### R-1 — A caller in `pipelines/stage1_stage2.py` uses an `evaluate.X` name that is in `evaluate.py` but NOT in `evaluation/__init__.py`

**Likelihood:** Very low. Doc 15 §9 confirms `evaluation/__init__.py`
re-exports every name `evaluate.py` re-exports.

**Mitigation:** Pre-flight test T-0 (Section 10) enumerates every
`evaluate.X` call site and verifies each name is present on the
`evaluation` package.

**Fallback:** If T-0 fails, do not apply the patch. Report the missing
name to the user and add it to `evaluation/__init__.py`'s explicit
re-export block first.

### R-2 — Stale `__pycache__` keeps the old `evaluate` binding alive

**Likelihood:** Low (Python invalidates pyc on source mtime change).

**Mitigation:** Test T-1 purges `__pycache__` before re-running.

### R-3 — Editor / linter cache (Pylance, pyright) still points at `evaluate.py`

**Likelihood:** High for IDEs, but cosmetic only — runtime is unaffected.

**Mitigation:** Document in the commit message. Restart the language
server if needed. No code action required.

### R-4 — Stage 2 `n_subj` NameError surface broadens (FALSE — not a real risk)

The monolith `inference.py` Stage 2 NameError (doc 14 / doc 16 §A3) is
unrelated to this patch and remains dormant for the same reasons as
before (the monolith is unreachable).

### R-5 — A future caller imports `evaluate` again

**Likelihood:** Possible.

**Mitigation:** Test T-3 (grep) confirms `import evaluate` is gone
from the production tree. Any regression is easy to detect.

**No other risks identified.** The patch does not interact with
simulator imports, FCD handling, GPU code, the per-sim parameter
contract, or any model selection / nuisance handling logic.

---

## 9. Rollback Plan

Reversibility is trivial. **Three options**, in increasing order of
state preservation:

### Rollback option A — Single-file revert (recommended)

```bash
cd /scratch/home/wog3597/vbi
git checkout pipelines/stage1_stage2.py
```

Restores `pipelines/stage1_stage2.py` to its `HEAD` state. Pre-flight
prerequisite: ensure no other un-committed changes exist in that file
(`git status pipelines/stage1_stage2.py` should show no `M` flag prior
to the patch).

### Rollback option B — In-place line edit

Manually edit `pipelines/stage1_stage2.py` line 38 back to:

```python
import evaluate
```

Equivalent to option A but does not require git.

### Rollback option C — Full commit revert

If P-1 has already been committed and pushed:

```bash
git revert <commit-hash>
```

Creates a new commit that undoes P-1. Use only if option A is not
applicable (e.g., post-push).

### What rollback restores

- `import evaluate` (line 38) reinstated.
- `evaluate.py` re-enters the production module load graph.
- No other side effects — every other file is untouched by P-1.

---

## 10. Tests to Run After Patch

Run **in order**. Stop and rollback at the first failure.

### T-0 — Pre-flight (run BEFORE applying the patch)

Confirms every `evaluate.X` call site is covered by `evaluation`:

```bash
cd /scratch/home/wog3597/vbi
python -c "
import re, sys
src = open('pipelines/stage1_stage2.py').read()
names = sorted(set(re.findall(r'\bevaluate\.([A-Za-z_][A-Za-z0-9_]*)', src)))
print(f'evaluate.X call sites found: {len(names)} unique names')
import evaluation
missing = [n for n in names if not hasattr(evaluation, n)]
if missing:
    print('MISSING from evaluation:', missing)
    sys.exit(1)
print('All names resolved on the evaluation package:')
for n in names:
    print(f'  evaluation.{n}')
"
```

**Expected:** non-empty list of names, all resolved. Exit code 0.
**If any name is missing:** STOP, surface to the user, do not apply patch.

### T-1 — Compile + cache purge (run AFTER applying the patch)

```bash
cd /scratch/home/wog3597/vbi
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
python -m compileall -q .
echo "compileall exit: $?"
```

**Expected:** "compileall exit: 0"

### T-2 — Package resolution

```bash
python -c "
import inference, simulation, features, evaluation, pipelines
for m in (inference, simulation, features, evaluation, pipelines):
    assert m.__name__ + '/__init__' in m.__file__, m.__file__
print('all packages resolve correctly')
"
```

**Expected:** "all packages resolve correctly"

### T-3 — Patch-specific binding check

```bash
python -c "
import pipelines.stage1_stage2 as pl
assert hasattr(pl, 'evaluate'), 'pl.evaluate name missing'
assert pl.evaluate.__name__ == 'evaluation', (
    f'expected evaluation, got {pl.evaluate.__name__}'
)
assert 'evaluation/__init__' in pl.evaluate.__file__, pl.evaluate.__file__
print('P-1 binding OK:', pl.evaluate.__file__)
"
```

**Expected:** `P-1 binding OK: .../vbi/evaluation/__init__.py`

### T-4 — Top-level legacy import has been removed

```bash
grep -n "^import evaluate\b\|^from evaluate import" \
    pipelines/*.py inference/*.py evaluation/*.py *.py
```

**Expected:** zero output. (Notebook and `debug_notebook.py` are not
in the production scope and are addressed in Tier 4b; they may still
match in `*.ipynb` / `debug_notebook.py` and that is acceptable.)

### T-5 — Public-API smoke

```bash
python -c "
from pipelines import run_pipeline
from inference import (
    ParameterScaler, FeaturePipeline, FeatureEmbedding,
    run_stage1_snpe, run_stage2_snpe, save_artifacts, load_artifacts,
)
from evaluation import (
    fc_metrics, evaluate_validation_stage1, evaluate_validation_stage2,
    select_best_model, final_test, print_final_summary,
)
print('public-API smoke OK')
"
```

**Expected:** "public-API smoke OK"

### T-6 — Backwards-compatibility sanity (the compat wrapper still works)

`evaluate.py` is **not deleted** by this patch. Other callers
(notebook cells, debug helpers) must continue to work via the wrapper:

```bash
python -c "
import evaluate
assert hasattr(evaluate, 'fc_metrics')
assert hasattr(evaluate, 'evaluate_validation_stage1')
assert hasattr(evaluate, 'select_best_model')
assert hasattr(evaluate, 'final_test')
assert hasattr(evaluate, 'print_final_summary')
# private helpers
for n in ('_aggregate_validation', '_print_selection_table',
          '_print_test_summary', '_print_validation_summary',
          '_progress', '_resimulate_and_score',
          '_test_stage1', '_test_stage2'):
    assert hasattr(evaluate, n), f'evaluate.{n} missing after P-1'
print('evaluate.py compat wrapper still works')
"
```

**Expected:** "evaluate.py compat wrapper still works"

### T-7 — Counts of remaining legacy imports are unchanged for non-P-1 scope

P-1 does not migrate any deferred `from simulator import` call. Confirm
the simulator-import count is unchanged:

```bash
grep -rn "from simulator import\|import simulator\b" \
    --include="*.py" . | wc -l
```

**Expected:** unchanged from baseline (doc 17 §5.3 — should be the
same total). A drop or rise indicates an unintended side effect of P-1.

### T-8 — Optional: full no-GPU smoke

```bash
python debug.py --basic
```

**Expected:** exits cleanly (any GPU-dependent steps skip, but the
import surface is exercised end-to-end).

---

## 11. Expected Test Result

If P-1 is applied correctly, every test above passes with the listed
expected output. The end state is:

| Property | Pre-patch | Post-patch |
|---|---|---|
| `compileall -q .` exit code | 0 | 0 |
| `pipelines.stage1_stage2.evaluate.__name__` | `"evaluate"` | `"evaluation"` |
| `pipelines.stage1_stage2.evaluate.__file__` | `.../evaluate.py` | `.../evaluation/__init__.py` |
| `grep "^import evaluate\b" pipelines/*.py` | 1 line | 0 lines |
| `evaluate.py` still importable | ✅ | ✅ (unchanged) |
| `evaluation` re-exports | unchanged | unchanged |
| Notebook cells (`main.ipynb` 163/165) | legacy | legacy (Tier 4b will address) |
| Number of deferred `from simulator import` sites | 12 in package code | 12 in package code (unchanged — P-1 does not touch these) |
| R3 violation count | 1 (`simulation/qc.py:25`) | 1 (unchanged) |
| Scientific outputs of any pipeline run | unchanged | unchanged |
| Git diff size | — | 1 file, 1 line, ±1 chars |

If any of the above does not match, rollback (Section 9).

---

## 12. Files Explicitly NOT Touched

P-1 changes exactly one line in exactly one file. The following are
**explicitly excluded** from this patch and must remain on disk and
unmodified:

### Source files not touched

| File | Reason |
|---|---|
| `evaluate.py` | Compat wrapper. Must continue to work for notebook + debug callers until Tier 4b. |
| `evaluation/__init__.py` and all `evaluation/*.py` | Already correct — they are the target of `import evaluation`. |
| `simulator.py` | Different scope (Tier 2–4). |
| `inference.py` | R8 forbids editing the monolith. |
| `inference/`, `simulation/`, `features/`, `pipelines/__init__.py` package source | Out of P-1 scope. |
| `data_loader.py` | Tier 2. The `from simulator import compute_delay_matrix` on line 278 is NOT addressed here. |
| `bold.py`, `config.py`, `main.py`, `pipeline_setup.py` | Out of scope. |
| `main.ipynb`, `debug.py`, `debug_notebook.py` | Notebook + debug callers; Tier 4b. |
| All 8 root duplicate `.py` files (`fc.py`, `fcd.py`, `wc_runner.py`, `delays.py`, `warmup.py`, `qc.py`, `extraction.py`, `screening.py`) | Tier 6 — not deletable until compat wrappers go (Tier 5). |
| Root `__init__.py` | Tier 6 (latent risk only). |
| `PATCH_REPORT.md`, `README.md` | Tier D documentation (separate from code patches). |

### Documentation files not touched

| File | Reason |
|---|---|
| `01_repo_overview.md` … `17_baseline_test_report.md` | Documentation is read-only for the code-patch sequence. |

### Configuration / data files not touched

`requirements.txt`, `install.sh`, `participants.tsv`, `atlas_115_labels.txt`,
`MPTP_FC_115.mat`, `MPTP_SC_115.mat`, `output_mouse_mptp/*`, `.gitignore`
— none of these are part of this patch.

### Operations explicitly NOT performed by P-1

- No file deletion (no `rm`, no `git rm`).
- No file rename / move (no `mv`, no `git mv`).
- No function rename anywhere in the codebase.
- No constant change in `config.py` or any other file.
- No new dependency (no `pip install`).
- No `sys.path` manipulation.
- No `__pycache__` deletion as part of the patch itself (cache purge
  is part of the **test** step T-1, not the patch).
- No `git config` change.
- No branch / push / PR action.
- No scientific behavior change of any kind.

---

## 13. User Approval Checkpoint

Before applying P-1, the following must be true. Confirm each item
with the user **or** verify each automatically before proceeding:

| # | Checkpoint | Required value | How to verify |
|---|---|---|---|
| C-1 | User explicitly authorises Patch 1 | "Proceed with Patch 1" (verbatim) | Wait for user message |
| C-2 | Current branch is `refactor/02-simulation` | match | `git rev-parse --abbrev-ref HEAD` |
| C-3 | No un-committed changes in `pipelines/stage1_stage2.py` | clean | `git status pipelines/stage1_stage2.py` should NOT show `M` |
| C-4 | Working tree on `config.py` and `simulation/wc_runner.py` either committed or user has confirmed it is intentional | confirmed | `git diff config.py simulation/wc_runner.py` reviewed by user |
| C-5 | T-0 (pre-flight name-coverage check) passes | exit code 0 | Run T-0 from Section 10 |
| C-6 | Baseline tests still green at the moment of patch | all green | Re-run §9 of doc 17 (or T-1/T-2 from this doc) |

If C-3 fails (the file already has un-committed changes), surface to
the user — do not silently overwrite their work.

If C-4 cannot be confirmed, surface the diff to the user and request a
decision (commit the M-status files first, or confirm they are
in-progress work that may safely coexist with P-1).

If C-5 fails (T-0 finds a name not on `evaluation`), STOP. The patch
specification is wrong; revise before applying.

If C-1 has not been given, do not apply the patch under any
circumstances.

---

Do not apply this patch until the user explicitly says: Proceed with Patch 1.
