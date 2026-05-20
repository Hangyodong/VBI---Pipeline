# 19 — Patch 1 Result: Import Cleanup Applied

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** **Patch P-1 applied and verified.**
**Predecessor docs:** 16, 17, 18
**Branch:** `refactor/02-simulation`

---

## 1. Changed Files

Exactly **one file** modified:

| # | File | Change kind | Lines touched |
|---|---|---|---|
| 1 | `pipelines/stage1_stage2.py` | In-place edit | line 38 only |

No other source files were touched. No files were created, deleted,
renamed, or moved (other than this result document, which the user
requested).

`git status --porcelain pipelines/stage1_stage2.py` → `?? pipelines/stage1_stage2.py`
(the file is untracked in the current branch state; the edit is captured
on disk).

---

## 2. Diff Summary

```diff
--- pipelines/stage1_stage2.py
+++ pipelines/stage1_stage2.py
@@ -35,7 +35,7 @@
 import numpy as np

 import config
 import data_loader
-import evaluate
+import evaluation as evaluate
 import inference
```

- 1 line removed, 1 line added.
- Net diff size: ±1 line, ±17 characters (`import evaluate` → `import evaluation as evaluate`).
- No comments, docstrings, function bodies, or any other surrounding
  code touched. No whitespace changes.

---

## 3. Exact Import Lines Changed

**Before (verbatim, line 38):**

```python
import evaluate
```

**After (verbatim, line 38):**

```python
import evaluation as evaluate
```

Local name `evaluate` is now bound to the `evaluation/` package object
(`/scratch/home/wog3597/vbi/evaluation/__init__.py`) instead of the
35-line `evaluate.py` compat wrapper. All five `evaluate.X` call sites
inside `pipelines/stage1_stage2.py` resolve to the same function
objects as before (verified pre-flight in T-0):

```
evaluation.evaluate_validation_stage1
evaluation.evaluate_validation_stage2
evaluation.final_test
evaluation.print_final_summary
evaluation.select_best_model
```

---

## 4. Tests Run

In the order specified by doc 18 Section 10. T-0 was the pre-flight
check; T-1 through T-7 were run after the edit. T-8 (`python debug.py
--basic`) was optional and was **not** executed (it requires real data
files and the spec marked it optional).

| Test | Command (summary) | Run? |
|---|---|---|
| T-0 | Pre-flight: every `evaluate.X` in `pipelines/stage1_stage2.py` is on the `evaluation` package | YES |
| T-1 | Purge `__pycache__` then `python -m compileall -q .` | YES |
| T-2 | `python -c "import inference, simulation, features, evaluation, pipelines; assert m.__name__ + '/__init__' in m.__file__"` | YES |
| T-3 | `python -c "import pipelines.stage1_stage2 as pl; assert pl.evaluate.__name__ == 'evaluation'"` | YES |
| T-4 | `grep -n "^import evaluate\b\|^from evaluate import" pipelines/*.py inference/*.py evaluation/*.py *.py` | YES |
| T-5 | Public-API smoke: import the six public evaluation names + the six inference names + `run_pipeline` | YES |
| T-6 | Backwards-compat: `import evaluate` (the wrapper) still provides every public + 8 private helper | YES |
| T-7 | `grep -rn "from simulator import\|import simulator\b" --include="*.py" .` | YES |
| T-8 | `python debug.py --basic` | **NOT RUN** — marked optional in spec, not required to verify P-1 |

---

## 5. Test Results

### T-0 (pre-flight)
```
evaluate.X call sites found: 5 unique names
All names resolved on the evaluation package:
  evaluation.evaluate_validation_stage1
  evaluation.evaluate_validation_stage2
  evaluation.final_test
  evaluation.print_final_summary
  evaluation.select_best_model
```
**Expected:** exit 0, all names resolved. **Result: PASS.**

### T-1 (compile + cache purge)
```
compileall exit: 0
```
**Expected:** `0`. **Result: PASS.**

### T-2 (package resolution)
```
all packages resolve correctly
```
**Expected:** that exact string. **Result: PASS.**

### T-3 (patch-specific binding)
```
P-1 binding OK: /scratch/home/wog3597/vbi/evaluation/__init__.py
```
**Expected:** `P-1 binding OK: .../vbi/evaluation/__init__.py`. **Result: PASS.**

### T-4 (legacy top-level import removed)
```
(no output, grep exit code 1 — no matches)
```
**Expected:** zero output. **Result: PASS.** The single load-bearing
top-level `import evaluate` in production code is gone.

### T-5 (public-API smoke)
```
public-API smoke OK
```
**Expected:** that exact string. **Result: PASS.**

### T-6 (compat wrapper sanity)
```
evaluate.py compat wrapper still works
```
**Expected:** that exact string. **Result: PASS.** `evaluate.py` is
**not deleted**; remaining notebook/debug callers continue to work.
All five public names (`fc_metrics`, `evaluate_validation_stage1`,
`select_best_model`, `final_test`, `print_final_summary`) and all
eight private helpers (`_aggregate_validation`,
`_print_selection_table`, `_print_test_summary`,
`_print_validation_summary`, `_progress`, `_resimulate_and_score`,
`_test_stage1`, `_test_stage2`) are still resolvable via
`import evaluate`.

### T-7 (simulator import count unchanged)
```
26
```
**Expected:** unchanged from baseline. **Result: PASS.** The 26 count
matches doc 17 §5.3's tally: 12 deferred live-package imports + 6
debug/notebook-helper imports + 5 dead-monolith imports + 3
cosmetic docstring matches inside `simulator.py` itself = 26. P-1 did
not touch any `simulator` import, so this count is invariant by
construction.

### T-8 (optional debug smoke)
**Not run.** Marked optional in doc 18 Section 10. Spec did not require
it for P-1 verification. T-2, T-3, T-5, T-6 already exercise the
relevant import surface.

---

## 6. Remaining Failures

**None.**

- 0 test failures (T-0 through T-7 all PASS; T-8 not executed by design)
- 0 compile errors
- 0 import errors
- 0 attribute-resolution errors
- 0 regressions in `evaluate.py` compat wrapper
- 0 unintended changes to any other file

The end-state table in doc 18 Section 11 matches reality:

| Property | Spec'd post-patch | Actual post-patch |
|---|---|---|
| `compileall -q .` exit code | 0 | 0 |
| `pipelines.stage1_stage2.evaluate.__name__` | `"evaluation"` | `"evaluation"` |
| `pipelines.stage1_stage2.evaluate.__file__` | `.../evaluation/__init__.py` | `/scratch/home/wog3597/vbi/evaluation/__init__.py` |
| `grep "^import evaluate\b" pipelines/*.py` | 0 lines | 0 lines |
| `evaluate.py` still importable | ✅ | ✅ |
| `evaluation` re-exports unchanged | ✅ | ✅ |
| Number of deferred `from simulator import` sites unchanged | ✅ | ✅ (26 grep hits, identical to baseline) |
| Git diff size | 1 file, 1 line | 1 file, 1 line (±17 chars) |

---

## 7. Whether Rollback Is Needed

**No.** All tests pass. The patch achieved its intended effect with
zero side effects. No reason to roll back.

If rollback ever becomes desirable, doc 18 Section 9 lists three
options. Because `pipelines/stage1_stage2.py` is currently untracked
(`?? pipelines/stage1_stage2.py`), option A (`git checkout <file>`) is
not applicable until the file is first staged or committed. The
working fallback is option B — in-place edit line 38 of
`pipelines/stage1_stage2.py` back to `import evaluate`. That single
character-level revert restores the baseline byte-for-byte.

---

## 8. Recommended Next Patch

**Patch P-2 — `data_loader.py:278`** (Tier 2 of doc 16 Section 7).

```diff
-    from simulator import compute_delay_matrix
+    from simulation.delays import compute_delay_matrix
```

### Why P-2 is the next logical step

1. P-1 closed the **only** top-level legacy import in production code.
   The remaining `simulator` imports are all deferred (inside function
   bodies), which is a different and slightly higher-risk class
   because the failure surfaces only at first call. Walking the chain
   from least-risky to most-risky next means starting with the single
   deferred site that fires on **every pipeline run** (the
   `data_loader` hot path).

2. `data_loader.get_subject_data` is exercised by every pipeline
   invocation, so a regression would be caught by the first smoke run
   rather than waiting for Stage 2 to engage.

3. The migration is mechanical: `simulator.compute_delay_matrix` is a
   re-export of `simulation.delays.compute_delay_matrix` (verified in
   doc 11 / doc 17 §6.4). Same function object, no signature change.

4. The change is **deferred** (inside a function), so module load
   order is unaffected and module-loading interactions with already-
   imported `simulation/` package state are unchanged.

5. Doc 18-style isolation: P-2 still touches exactly one line in
   exactly one file. Reversibility is identical (one-line revert).

### What P-2 does NOT do

- Does not touch any of the 11 remaining deferred sites (5 in
  `inference/` + 6 in `evaluation/`). Those are Tier 3 and Tier 4.
- Does not approach the FCD dual-role bug (Tier X / doc 16 §3.1 / Risk A6).
- Does not delete `simulator.py` (Tier 5, only after all 12 deferred
  sites migrate).
- Does not touch `inference.py`, `config.py`, or
  `simulation/wc_runner.py` (R1, R8, doc 16 §13 stop conditions).

### Pre-conditions before P-2 lands

- `pipelines/stage1_stage2.py` is in its new P-1 state (verified
  above).
- `data_loader.py` working-tree status: must be clean of unrelated
  changes (run `git status data_loader.py` before applying — the
  baseline doc 17 lists `data_loader.py` as unmodified).
- `from simulation.delays import compute_delay_matrix` resolves
  correctly today (doc 17 §2.4 confirmed `simulation/` resolves to its
  `__init__.py`; `simulation/delays.py` is on disk).

### Test sequence for P-2

Re-run T-1, T-2, T-5 (unchanged from P-1). Add the P-2-specific check:

```bash
python -c "
import inspect, data_loader
src = inspect.getsource(data_loader.get_subject_data)
assert 'from simulation.delays import compute_delay_matrix' in src, src
assert 'from simulator import' not in src, src
print('P-2 source OK')
"
```

Plus an updated T-4-style audit:

```bash
grep -rn "from simulator import\|import simulator\b" data_loader.py
# Expected: zero output after P-2 lands.
```

### Do not proceed past P-2 without user authorization

Per the user's standing instruction ("Do not proceed to Patch 2"),
P-2 itself is **not applied here**. This section is a recommendation
only. The user must explicitly authorize Patch 2 before any edit to
`data_loader.py`.

---

## Appendix A — Verbatim Test Output Capture

For audit traceability, the raw stdout of every test executed after
the patch:

```
$ find . -name "*.pyc" -delete 2>/dev/null
$ find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
$ python -m compileall -q .
$ echo "compileall exit: $?"
compileall exit: 0
```

```
$ python -c "import inference, simulation, features, evaluation, pipelines; ..."
all packages resolve correctly
```

```
$ python -c "import pipelines.stage1_stage2 as pl; ..."
P-1 binding OK: /scratch/home/wog3597/vbi/evaluation/__init__.py
```

```
$ grep -n "^import evaluate\b\|^from evaluate import" pipelines/*.py inference/*.py evaluation/*.py *.py
(no output; grep exit 1)
```

```
$ python -c "from pipelines import run_pipeline; ..."
public-API smoke OK
```

```
$ python -c "import evaluate; ..."
evaluate.py compat wrapper still works
```

```
$ grep -rn "from simulator import\|import simulator\b" --include="*.py" . | wc -l
26
```

---

## Appendix B — Files Confirmed Untouched

Doc 18 Section 12 lists every file P-1 is forbidden from touching.
Verified:

- `evaluate.py` — present, untouched (35 lines, zero-logic re-export)
- `evaluation/__init__.py` and all `evaluation/*.py` — present, untouched
- `simulator.py` — present, untouched
- `inference.py` — present, untouched (R8)
- `inference/`, `simulation/`, `features/`, `pipelines/__init__.py` — present, untouched
- `data_loader.py` — present, untouched (Tier 2 target, deferred)
- `bold.py`, `config.py`, `main.py`, `pipeline_setup.py` — present, untouched
- `main.ipynb`, `debug.py`, `debug_notebook.py` — untouched (Tier 4b)
- All 8 root duplicates (`fc.py`, `fcd.py`, `wc_runner.py`, `delays.py`,
  `warmup.py`, `qc.py`, `extraction.py`, `screening.py`) — untouched
- Root `__init__.py` — untouched
- `PATCH_REPORT.md`, `README.md` — untouched
- `01_repo_overview.md` … `18_patch1_import_cleanup_spec.md` — untouched

P-1 created exactly one new file beyond the source edit: this report
(`19_patch1_result.md`), in compliance with the user's explicit
request.

---

**End of Patch 1 result. Awaiting user instruction before proceeding
to Patch 2.**
