# 21 — Patch 2 Result: `data_loader` Deferred Import Cleanup Applied

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** **Patch P-2 applied and verified.**
**Predecessor docs:** 11, 16, 17, 18, 19, 20
**Branch:** `refactor/02-simulation`

---

## 1. Patch Summary

Replaced the single deferred legacy `simulator` import inside
`data_loader.get_subject_data()` with a direct package-level import
from `simulation.delays`.

- One file changed: `data_loader.py`.
- One line changed: line 278 only.
- Zero scientific behavior changes.
- All 8 mandatory tests (T-1…T-8) PASS.
- T-9 (optional `debug.py --basic`) reported 3 pre-existing failures
  unrelated to this patch. See §7.
- Rollback NOT needed.

After P-2 the hot path of `get_subject_data()` (one call per subject,
every pipeline run) bypasses the `simulator.py` re-export shim and
loads `compute_delay_matrix` directly from
`simulation/delays.py`. Whole-tree `simulator` grep count dropped
from 26 to 25.

---

## 2. Changed Files

Exactly **one file** modified:

| # | File | Change kind | Lines touched |
|---|---|---|---|
| 1 | `data_loader.py` | In-place edit | line 278 only |

No other source files were touched. No files created, deleted,
renamed, or moved (other than this report and
`patch2_remaining_simulator_imports.txt`, which the test sequence
explicitly produced via `tee`).

---

## 3. Exact Line Changed

**Before (verbatim, line 278):**

```python
    from simulator import compute_delay_matrix
```

**After (verbatim, line 278):**

```python
    from simulation.delays import compute_delay_matrix
```

Indentation preserved (4 spaces, deferred-import position inside
`get_subject_data`). Same imported name. Same call site on line 313
(unchanged). Same function object (verified via T-5 identity check).

---

## 4. Diff Summary

```diff
--- data_loader.py
+++ data_loader.py
@@ -275,7 +275,7 @@
     delays      : (N, N) delay (ms) = lengths/velocity
     bold        : (T, N) optional ROI BOLD time series
     """
-    from simulator import compute_delay_matrix
+    from simulation.delays import compute_delay_matrix

     fc, fcd, fc_nan = _load_fc_fcd(fc_mat, fc_ids, sid)
```

- 1 line removed, 1 line added.
- Net change: 1 file × 1 line × ~17 characters.
- No comments, docstrings, function bodies, whitespace, or any other
  surrounding code touched.

---

## 5. Pre-flight Results

Per spec §10 / user instructions before editing:

| Check | Command | Result |
|---|---|---|
| Branch | `git rev-parse --abbrev-ref HEAD` | `refactor/02-simulation` ✅ |
| Working-tree status of `data_loader.py` | `git status --short data_loader.py` | (clean, empty output) ✅ |
| Old import present | `grep -n "from simulator import compute_delay_matrix" data_loader.py` | `278:    from simulator import compute_delay_matrix` ✅ |
| T-0 identity check | `simulator.compute_delay_matrix is simulation.delays.compute_delay_matrix` | `True` ✅ |

All 4 pre-flight checks PASSED. Proceeding with edit was authorized
by the patch spec and the user's explicit `Proceed with Patch 2`.

---

## 6. Test Commands Run

In the exact order specified by the user:

| # | Test | Tool |
|---|---|---|
| T-1 | Compile clean after cache purge | `compileall -q .` |
| T-2 | Package import smoke for 5 packages | `python -c "..."` |
| T-3 | `data_loader.get_subject_data` source check | `inspect.getsource` |
| T-4 | No `simulator` references in `data_loader.py` | `grep -n "simulator"` |
| T-5 | Runtime identity of `simulator.compute_delay_matrix is cdm` | `python -c "..."` |
| T-6 | Whole-tree `simulator` import count | `grep -rn ... \| tee patch2_remaining_simulator_imports.txt` |
| T-7 | Patch 1 invariant still holds | `pl.evaluate.__name__ == "evaluation"` |
| T-8 | Public-API smoke (`pipelines`, `inference`, `evaluation`) | `python -c "..."` |
| T-9 | Optional debug smoke | `python debug.py --basic` |

---

## 7. Test Results for T-1 through T-9

### T-1 — Compile check
```
compileall exit code: 0
```
**Expected:** `0`. **Result: PASS.**

### T-2 — Package import smoke
```
inference -> /scratch/home/wog3597/vbi/inference/__init__.py
simulation -> /scratch/home/wog3597/vbi/simulation/__init__.py
features -> /scratch/home/wog3597/vbi/features/__init__.py
evaluation -> /scratch/home/wog3597/vbi/evaluation/__init__.py
pipelines -> /scratch/home/wog3597/vbi/pipelines/__init__.py
package imports OK
```
**Expected:** all 5 packages resolve to `__init__.py`. **Result: PASS.**

### T-3 — `data_loader` source check
```
T-3 data_loader deferred import updated correctly
```
**Expected:** that exact string. **Result: PASS.**

### T-4 — No simulator references in `data_loader.py`
```
(no output, grep produced nothing)
```
**Expected:** zero output. **Result: PASS.** The `simulator` token
appears zero times anywhere in `data_loader.py` — not in code,
comments, or docstrings.

### T-5 — Runtime binding identity
```
T-5 identity OK: simulator.compute_delay_matrix is simulation.delays.compute_delay_matrix
```
**Expected:** that exact string. **Result: PASS.** Confirms the
function object is identical before and after the patch — the call
on line 313 (`delays = compute_delay_matrix(...)`) calls the same
callable as before.

### T-6 — Remaining simulator imports
```
25 patch2_remaining_simulator_imports.txt
```
**Expected:** 25 (baseline 26 − 1 line removed by P-2). **Result: PASS.**

See §8 for the full breakdown.

### T-7 — Patch 1 invariant
```
Patch 1 invariant still OK: /scratch/home/wog3597/vbi/evaluation/__init__.py
```
**Expected:** `pl.evaluate.__name__ == "evaluation"` and file
contains `evaluation/__init__`. **Result: PASS.** The P-1 binding
state is intact — P-2 did not disturb it.

### T-8 — Public-API smoke
```
public API smoke OK
```
**Expected:** that exact string. **Result: PASS.** All requested
public symbols import successfully (`run_pipeline`,
`ParameterScaler`, `FeaturePipeline`, `run_stage1_snpe`,
`run_stage2_snpe`, `fc_metrics`, `evaluate_validation_stage1`,
`select_best_model`, `final_test`).

### T-9 — Optional debug smoke (`python debug.py --basic`)

Ran to completion. Exit summary:

```
======================================================================
  Test summary
======================================================================
  FAIL  config consistency
  PASS  imports
  PASS  ParameterScaler
  PASS  Stage 2 ParameterScaler
  PASS  FamilyScaler
  PASS  FCPCAScaler
  FAIL  FeaturePipeline
  PASS  FeatureEmbedding (torch)
  PASS  FC upper triangle
  FAIL  FCD upper triangle (simulated)
  PASS  observed FCD = direct file load
----------------------------------------------------------------------
  PASS: 8  |  FAIL: 3  |  SKIP: 0
```

**Status: optional failure (NOT a rollback trigger).** The 3 failures
are **pre-existing** and **unrelated** to the deferred-import change.
Detail:

| Subtest | Reported failure | Relation to P-2 |
|---|---|---|
| `config consistency` | `FCD_DIM=5 != FC_DIM=6555` | Pre-existing config-dim drift (the FCD pipeline path expects 5-dim summary stats while FC is 6555-dim). Doesn't touch `compute_delay_matrix` or `get_subject_data`. Documented as the FCD dual-role surface in doc 16 §3.1 Risk A6 (Tier X). |
| `FeaturePipeline` | output `(200, 50) != (200, 80)` | Pre-existing feature-dim expectation drift in the debug test itself. PCA output is 50; the debug check expects 80 (likely FC 50 + FCD 30). Independent of `data_loader` imports. |
| `FCD upper triangle (simulated)` | vec `(6555,) != (5,)` | Pre-existing FCD-shape mismatch in the same dual-role bug surface (Risk A6 / Tier X). Debug feeds a 6555-dim upper-tri vector into code that, under `USE_FCD=True`, expects 5-dim summary stats. Production runs with `USE_FCD=False`, so this is dormant in normal pipelines. |

**Crucially the `imports` subtest of T-9 PASSED**, which directly
exercises `import data_loader, simulator, evaluate, inference,
config, bold`. That confirms the P-2 import change itself loads
cleanly. None of the 3 failing subtests touches
`compute_delay_matrix`, `get_subject_data`, `simulation.delays`,
`simulator.py`, or any code path P-2 modified.

Per the user's rule: "If T-9 fails … record it as optional failure,
not as automatic rollback." These are pre-existing test/code
drift, recorded here as **optional failures unrelated to P-2**.

---

## 8. Remaining Simulator Imports After Patch 2

`patch2_remaining_simulator_imports.txt` contents (25 lines):

```
evaluation/validation.py:123:    from simulator import extract_observed_features
debug_notebook.py:80:    import simulator    # noqa: F401
debug_notebook.py:206:    from simulator import compute_fc, fc_to_upper_tri
debug_notebook.py:230:    import simulator
simulator.py:6:    from simulator import simulate_gpu_batch, compute_fc
simulator.py:7:    from simulator import worker_extract, extract_observed_features
simulator.py:8:    import simulator
evaluation/final_test.py:112:    from simulator import extract_observed_features
evaluation/metrics.py:110:    from simulator import extract_observed_features
evaluation/metrics.py:184:    from simulator import (
evaluation/metrics.py:245:    from simulator import (
inference.py:387:    from simulator import simulate_gpu_batch, worker_extract
inference.py:985:    from simulator import (
inference.py:1070:    from simulator import (
inference.py:1317:    from simulator import simulate_gpu_batch, worker_extract
inference.py:1473:    from simulator import extract_observed_features
inference/training_data.py:48:    from simulator import simulate_gpu_batch, worker_extract
debug.py:443:    from simulator import fc_to_upper_tri
debug.py:466:    from simulator import compute_sim_fcd_matrix, fcd_to_upper_tri
debug.py:565:    import simulator
inference/posterior.py:113:    from simulator import (
inference/stage2.py:158:    from simulator import simulate_gpu_batch, worker_extract
inference/stage2.py:319:    from simulator import extract_observed_features
evaluation/plots.py:203:    from simulator import simulate_single, compute_fc
inference/diagnostics.py:47:    from simulator import (
```

### Categorisation (matches doc 17 §5.3 minus P-2's one removal)

| Category | Count | Files | Tier |
|---|---|---|---|
| `data_loader.py` deferred site | **0 (was 1)** | — | **Tier 2 — closed by P-2** |
| `inference/` package deferred sites | 5 | `inference/training_data.py:48`, `inference/stage2.py:158`, `inference/stage2.py:319`, `inference/posterior.py:113`, `inference/diagnostics.py:47` | Tier 3 |
| `evaluation/` package deferred sites | 6 | `evaluation/metrics.py:110/184/245`, `evaluation/validation.py:123`, `evaluation/final_test.py:112`, `evaluation/plots.py:203` | Tier 4 |
| Debug / notebook helpers | 6 | `debug.py:443/466/565`, `debug_notebook.py:80/206/230` | Tier 4b |
| Dead-monolith callers | 5 | `inference.py:387/985/1070/1317/1473` (R8 forbids edit; Tier 7 = delete) | Tier 7 |
| Cosmetic docstring matches inside `simulator.py` itself | 3 | `simulator.py:6/7/8` (docstring examples) | n/a |
| **Total** | **25** | | |

This matches the spec §9 expected post-P-2 state exactly:
**26 − 1 = 25**. The single closed entry is the `data_loader.py:278`
line P-2 replaced.

---

## 9. Whether Rollback Is Needed

**No.**

- All 8 mandatory tests (T-1 … T-8) PASS.
- T-9 (optional) reported 3 pre-existing failures with **zero
  connection to the changed import**. T-9 `imports` subtest, which
  directly exercises the P-2 change, PASSED.
- `simulator.py` is untouched and still re-exports
  `compute_delay_matrix` correctly (verified in T-5 identity check).
- Function object identity holds: any unchanged caller continues to
  receive the same callable.
- Diff size matches the spec exactly: 1 file, 1 line.

If rollback ever becomes desirable, doc 20 Section 7 lists three
options. The simplest is `git checkout -- data_loader.py` once the
file enters git's tracked state; until then,
in-place edit line 278 back to
`from simulator import compute_delay_matrix` restores the baseline
byte-for-byte.

---

## 10. Recommended Patch 3 Candidate

**Patch P-3a — `inference/training_data.py:48`** (Tier 3 of doc 16
§7 / doc 11 §6).

```diff
-    from simulator import simulate_gpu_batch, worker_extract
+    from simulation.wc_runner import simulate_gpu_batch
+    from features.extraction import worker_extract
```

### Why P-3a is the next logical step

1. **Smallest unit of Tier 3.** Tier 2 closed with P-2. The next
   live-package deferred site is the lowest-line-number `inference/`
   entry. Doc 16 §7 explicitly lists `inference/training_data.py:48`
   as the first Tier 3 patch.

2. **Symbols are split, but the destinations are unambiguous.** Doc
   11 Appendix A confirms:
   - `simulate_gpu_batch` → `simulation.wc_runner.simulate_gpu_batch`
   - `worker_extract` → `features.extraction.worker_extract`
   Both are direct re-exports through `simulator.py`. The function
   objects are identical (T-5-style identity check can verify this
   pre-flight, exactly mirroring P-2's pattern).

3. **Deferred (inside function body).** Module load order is
   unaffected. Failure mode is identical to P-2 — surfaces at first
   call, no silent corruption.

4. **One file, one logical block** (the spec calls this two-line
   replacement of a single import statement). Reversibility remains
   trivial.

5. **Hot path** — `inference.training_data` is invoked on every
   Stage 1 and Stage 2 run. Any regression will surface quickly
   under the same test fixtures used so far.

### What P-3a does NOT do

- Does not touch the other 4 `inference/` deferred sites (lines
  158, 319 in `stage2.py`; 113 in `posterior.py`; 47 in
  `diagnostics.py`). Those are P-3b, P-3c, P-3d, P-3e.
- Does not touch any `evaluation/` site (Tier 4).
- Does not address the FCD dual-role bug (Tier X / Risk A6) — even
  if the same files contain Tier X surface.
- Does not delete `simulator.py`, `evaluate.py`, `inference.py`,
  or any root duplicate (Tiers 5–7).

### Alternative — single bundled Tier 3 patch

Doc 16 §7 enumerates all 5 `inference/` sites and treats them as a
tier rather than as independent patches. If the user prefers, all
5 could land as a single commit ("P-3"). Today's recommendation is
the **finer-grained P-3a only** approach because:

- It mirrors P-1 / P-2's "exactly 1 file, exactly 1 import" rhythm.
- It keeps rollback even simpler (one file, one line block).
- The remaining 4 sites can each follow as P-3b…P-3e, each verified
  individually.

Either approach is structurally safe. The user authorizes which.

### Pre-conditions before P-3a lands

- `data_loader.py` is in its new P-2 state (verified by §5 and §7).
- `inference/training_data.py` is clean of unstaged changes (verify
  before applying with `git status --short inference/training_data.py`).
- T-0-style identity check before edit:
  ```python
  import simulator
  from simulation.wc_runner import simulate_gpu_batch
  from features.extraction import worker_extract
  assert simulator.simulate_gpu_batch is simulate_gpu_batch
  assert simulator.worker_extract is worker_extract
  ```

### Do not proceed past P-2 without user authorization

Per the user's standing instruction ("Do not proceed to Patch 3"),
P-3a itself is **not applied here**. This section is a
recommendation only.

---

## 11. Files Explicitly NOT Touched

Per doc 20 §10 / the user's rules, the following files were verified
to be **untouched by P-2**:

### Source files

| File | Reason |
|---|---|
| `simulator.py` | Kept on disk; required by 17 remaining `simulator`-referencing lines (11 live deferred + 6 debug + 3 docstring). |
| `evaluate.py` | Out of scope (Tier 5). |
| `evaluation/__init__.py` and all `evaluation/*.py` | Out of scope (Tier 4 sites within still match grep but are unchanged). |
| `inference.py` | R8 forbids edits to the monolith. Five dead-monolith `simulator` lines remain in the file (unchanged). |
| `inference/`, `simulation/`, `features/`, `pipelines/` package source | Out of scope. |
| `pipelines/stage1_stage2.py` | Patch 1 target; verified unchanged here (T-7 invariant). |
| `config.py` | R1 (no scientific changes). |
| `simulation/wc_runner.py` | Working-tree status `M` from prior work; not touched by P-2. |
| `main.ipynb`, `debug.py`, `debug_notebook.py` | Tier 4b (notebook + debug callers). |
| All 8 root duplicate `.py` files (`fc.py`, `fcd.py`, `wc_runner.py`, `delays.py`, `warmup.py`, `qc.py`, `extraction.py`, `screening.py`) | Tier 6 (deletion candidates, not edited). |
| Root `__init__.py` | Tier 6. |
| `bold.py`, `main.py`, `pipeline_setup.py` | Out of scope. |

### Documentation files

| File | Reason |
|---|---|
| `01_repo_overview.md` … `20_patch2_spec.md` | Documentation is read-only for the code-patch sequence. This report (`21_patch2_result.md`) is the only new doc created by P-2. |
| `PATCH_REPORT.md`, `README.md` | Tier D (separate from code patches). |

### Configuration / data files

`requirements.txt`, `install.sh`, `participants.tsv`,
`atlas_115_labels.txt`, `MPTP_FC_115.mat`, `MPTP_SC_115.mat`,
`output_mouse_mptp/*`, `.gitignore` — none touched.

### Operations explicitly NOT performed by P-2

- No file deletion (no `rm`, no `git rm`).
- No file rename / move (no `mv`, no `git mv`).
- No function rename anywhere in the codebase.
- No constant change in `config.py` or any other file.
- No new dependency (no `pip install`).
- No `sys.path` manipulation.
- No `__pycache__` deletion as part of the patch itself (cache purge
  is part of the test step T-1, not the patch).
- No `git config` change.
- No branch / push / PR action.
- No scientific behavior change of any kind.
- No bundling of any other Tier 3 / Tier 4 / Tier 4b patch.

### Two new files created by P-2's test run

These are intentional by-products of the test script the user
provided, not source-code edits:

- `patch2_remaining_simulator_imports.txt` (produced by T-6 via `tee`)
- `21_patch2_result.md` (this report)

Neither modifies any pre-existing source file.

---

**End of Patch 2 result. Awaiting user instruction before proceeding
to Patch 3.**
