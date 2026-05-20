# 20 — Patch 2 Specification: `data_loader` Deferred Import Cleanup

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** Specification only. **Patch not yet applied.**
**Predecessor docs:** 07, 11, 16, 17, 18, 19
**Branch:** `refactor/02-simulation`

---

## 1. Patch 2 Goal

Replace the single **deferred** legacy import of the `simulator.py`
compat wrapper inside `data_loader.get_subject_data()` with a direct
package import from `simulation.delays`.

After this patch:

- `data_loader.py` no longer carries any legacy `simulator` reference
  (it is the lone `data_loader` site flagged in doc 11 §2.2 and doc 17
  §5.3 row A).
- The hot path of every pipeline run (one call per subject) skips the
  `simulator.py` re-export shim and resolves directly to
  `simulation.delays.compute_delay_matrix`.
- The 12-site `simulator` deferred-import surface (doc 17 §5.3 list A)
  drops to **11 sites in 6 files** (the 5 `inference/` calls + 6
  `evaluation/` calls remain, untouched).
- `simulator.py` is one step closer to deletable, but stays on disk
  until every Tier 3 / Tier 4 deferred site has been migrated.
- Zero scientific behavior changes; one mechanical replacement on a
  single line.

**Scope of change:** exactly **one line** in exactly **one file**.

---

## 2. Evidence from Patch 1 Result (doc 19)

Patch 1 (`pipelines/stage1_stage2.py:38`, `import evaluate` →
`import evaluation as evaluate`) landed cleanly. Doc 19 records:

| Property | Status |
|---|---|
| T-0 pre-flight (name coverage) | PASS (5 names, all on `evaluation`) |
| T-1 `compileall -q .` | exit 0 |
| T-2 package resolution for all 5 packages | PASS |
| T-3 `pl.evaluate.__file__` is `evaluation/__init__.py` | PASS |
| T-4 `^import evaluate\b` in production tree | zero matches |
| T-5 public-API smoke | PASS |
| T-6 `evaluate.py` compat wrapper still importable | PASS |
| T-7 `simulator` grep count | 26 (unchanged from baseline) |
| Rollback needed? | No |
| Remaining failures | None |

**Implications for Patch 2:**

- The "one top-level legacy import" problem (doc 17 §5.2) is solved.
- The `simulator` deferred-import problem (doc 17 §5.3, 12 live sites)
  is **untouched** by P-1 — confirmed by T-7's invariant count.
- The next-smallest, lowest-risk patch identified by doc 16 §7 (Tier
  2) and doc 19 §8 is the **single `data_loader.py:278` migration**.
- `evaluate.py` was not deleted, and is still importable by notebook
  + debug callers — this remains true after P-2 (P-2 does not touch
  `evaluate.py` or the evaluation graph at all).

---

## 3. Exact Files to Modify

| # | File | Lines touched | Scope |
|---|---|---|---|
| 1 | `data_loader.py` | line **278** only | One single-line replacement |

**No other file is modified by this patch.**

The change is **deferred** (inside `get_subject_data()`); module load
order is unaffected.

---

## 4. Exact Changes

Only one line changes. Line 278 of `data_loader.py`, inside the body
of `get_subject_data(sid, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat,
bold_ids)`.

### 4.1 Surrounding context (lines 264–284, current state — for orientation only)

```python
def get_subject_data(sid, fc_mat, sc_mat, fc_ids, sc_ids,
                     bold_mat=None, bold_ids=None):
    """Bundle FC, FCD, SC, tract lengths, delays, and optional BOLD.

    Returned dict keys
    ------------------
    fc          : (N, N) Pearson FC, NaN→0, symmetrized, zero diagonal
    fcd         : (N, N) FCD matrix (used only if USE_FCD)
    fc_nan      : (N, N) bool, original NaN positions (diagnostic)
    sc          : (N, N) coupling weight, log1p + max-norm
    lengths_mm  : (N, N) tract length (mm)        — for delay calc
    delays      : (N, N) delay (ms) = lengths/velocity
    bold        : (T, N) optional ROI BOLD time series
    """
    from simulator import compute_delay_matrix          # <-- line 278

    fc, fcd, fc_nan = _load_fc_fcd(fc_mat, fc_ids, sid)
```

### 4.2 Old line

```python
    from simulator import compute_delay_matrix
```

This resolves through `simulator.py:17`:

```python
# simulator.py — current
from simulation.delays    import compute_delay_matrix
...
compute_delay_matrix,
```

i.e. the wrapper simply re-exports the symbol unchanged from
`simulation.delays`.

### 4.3 New line

```python
    from simulation.delays import compute_delay_matrix
```

Same indentation (4 spaces, inside the function body). Same imported
name. Direct package source instead of the compat wrapper.

### 4.4 Call site is unaffected

The only consumer of the imported name inside `get_subject_data()` is
the call on line 313:

```python
delays = compute_delay_matrix(
    ...
)
```

Doc 11 Appendix A confirms the symbol mapping:

| Symbol | Correct package source |
|---|---|
| `compute_delay_matrix` | `from simulation.delays import compute_delay_matrix` |

Function object is identical (verified via `simulator.py:17` direct
re-export).

---

## 5. Why Each Change Is Safe

### S1 — Function object identity is preserved

`simulator.compute_delay_matrix is simulation.delays.compute_delay_matrix`.
The compat wrapper does not redefine the function; it imports it. So
the patched line binds the local name `compute_delay_matrix` to the
same callable object as before. Call semantics are unchanged.

### S2 — The signature is unchanged

`simulation/delays.py:28` defines
`compute_delay_matrix(weights, velocity_m_per_s, lengths_mm=None)`.
This is the same signature visible through the compat wrapper. P-2
does not touch any call site.

### S3 — Module load order is unchanged

The import is deferred (inside a function body). It fires at first
invocation of `get_subject_data()`, not at module load. Before P-2,
the deferred import loads `simulator.py` → `simulation.delays`. After
P-2, it loads `simulation.delays` directly. `simulator.py` is no
longer pulled in by this particular call path, but **may still be
imported elsewhere** (11 other deferred sites in `inference/` + `evaluation/`
still touch it). P-2 neither requires nor causes the removal of
`simulator.py`.

### S4 — Doc 17 confirms `simulation/delays.py` resolves correctly

Doc 17 §2.4: `simulation` imports cleanly to its `__init__.py`.
`simulation/delays.py` is on disk (doc 17 §2.4 / Appendix A listing of
`simulation/` contents). No new dependency.

### S5 — Reversibility is trivial

One-line revert restores the prior state byte-for-byte. See Section 7.

### S6 — No scientific values change

No constants, weights, prior bounds, function bodies, or numerical
defaults are touched. R1 (07_refactor_rules.md) is satisfied
trivially.

### S7 — R3 dependency rule is satisfied

`data_loader.py` is a non-package module. R3 from doc 07 specifies:

> `data_loader.py` may be imported by `pipelines/` and
> `inference/training_data.py`.

…and the data flow contract permits `data_loader` to depend on the
`simulation/` package. P-2 strengthens this — replacing the legacy
wrapper import with the explicit package path makes the dependency
direction unambiguous in the source itself.

### S8 — No file renames, moves, deletes

`simulator.py`, `simulation/delays.py`, and every other file on disk
is untouched. The patch is purely an in-place text replacement.

### S9 — No new dependencies, no `sys.path` hacks, no function rename

The patch is a pure import-name replacement. Python's standard
package resolution covers everything.

### S10 — The hot-path failure mode is loud, not silent

`get_subject_data()` is called for every subject in every pipeline
run (doc 11 §2.2). If `simulation.delays` ever fails to resolve (it
does not today), the failure surfaces immediately at subject-load
time with a clean `ImportError`, not silently corrupted data. The
old `from simulator import compute_delay_matrix` had the same
failure-loudness; P-2 does not change it.

### S11 — Backwards compatibility for other `simulator` callers is preserved

`simulator.py` continues to re-export `compute_delay_matrix` after
P-2. Any future or unknown caller using `from simulator import
compute_delay_matrix` still works. The patch makes the production hot
path more direct without removing the compat surface.

---

## 6. Risks

The patch is small. Risks are explicitly enumerated below.

### R-1 — A name unintentionally captured elsewhere in `data_loader.py`

**Likelihood:** Zero.
**Mitigation:** Pre-flight test T-0 (Section 8) greps the entire file
for any other `compute_delay_matrix` or `simulator.` reference. Only
the deferred import on line 278 and the call on line 313 should
match, and both are in `get_subject_data()`.

### R-2 — Stale `__pycache__` retains the old binding

**Likelihood:** Low (Python invalidates pyc on source mtime change).
**Mitigation:** Test T-1 purges `__pycache__` before re-running.

### R-3 — `simulation/delays.py` does not export `compute_delay_matrix`

**Likelihood:** Zero — verified at three layers:
1. `grep -n "def compute_delay_matrix" simulation/delays.py` → line 28
2. `simulator.py:17` already imports from `simulation.delays`
3. `simulator.py:29` re-lists the symbol in `__all__`-style exports

**Mitigation:** T-0's `importlib` probe confirms the attribute exists.

### R-4 — Editor / linter cache still points at `simulator.py`

**Likelihood:** High for IDEs, but cosmetic — runtime is unaffected.
**Mitigation:** Same as P-1 (doc 18 R-3) — note in commit message;
restart the language server if needed. No code action required.

### R-5 — Side-effect of the old import chain that the new chain doesn't trigger

**Likelihood:** Zero. `simulator.py` is verified zero-logic
(doc 17 §6.4; 65-line re-export-only file). Importing it has no side
effects beyond loading `simulation/*` and `features/*` modules, both
of which load cleanly today.

**Mitigation:** No action required — fact is verified.

### R-6 — A future caller imports `simulator` again

**Likelihood:** Possible (developer habit).
**Mitigation:** Test T-3 (grep) confirms no `simulator` reference
remains in `data_loader.py`. The other 11 deferred sites in
`inference/` and `evaluation/` are out of P-2's scope and continue to
match grep until Tier 3 / Tier 4 land.

### R-7 — P-2 is mistakenly bundled with another change

**Likelihood:** Avoidable.
**Mitigation:** The spec restricts the patch to one line. The diff
must be `±1 line, ≤~30 characters`. Any deviation triggers stop.

**No other risks identified.** The patch does not interact with FCD
handling, GPU code, the per-sim parameter contract, model selection,
nuisance handling, the FCD dual-role bug surface (doc 16 Risk A6), or
the `n_subj` NameError dormant in `inference.py`.

---

## 7. Rollback Plan

Reversibility is trivial. Three options, in increasing order of
state preservation.

### Rollback option A — Single-file revert (recommended)

```bash
cd /scratch/home/wog3597/vbi
git checkout data_loader.py
```

Restores `data_loader.py` to its `HEAD` state. Pre-flight prerequisite:
`git status data_loader.py` showed `clean` at the moment this spec
was authored (confirmed). After applying P-2 the only delta is
line 278, so `git checkout` is precise.

### Rollback option B — In-place line edit

Manually edit `data_loader.py` line 278 back to:

```python
    from simulator import compute_delay_matrix
```

Equivalent to option A but does not require git.

### Rollback option C — Full commit revert

If P-2 has already been committed:

```bash
git revert <commit-hash>
```

Creates a new commit that undoes P-2. Use only if option A is not
applicable (e.g., post-push).

### What rollback restores

- `from simulator import compute_delay_matrix` (line 278) reinstated.
- `simulator.py` re-enters this particular call path.
- No other side effects — every other file is untouched by P-2.

---

## 8. Tests to Run

Run **in order**. Stop and rollback at the first failure.

### T-0 — Pre-flight (run BEFORE applying the patch)

Confirms `simulation.delays.compute_delay_matrix` exists and matches
the wrapper's re-export. Also confirms no other `simulator.` or
unexpected `compute_delay_matrix` reference exists in `data_loader.py`.

```bash
cd /scratch/home/wog3597/vbi
python -c "
import simulation.delays as d
import simulator as s
assert hasattr(d, 'compute_delay_matrix'), 'simulation.delays missing'
assert s.compute_delay_matrix is d.compute_delay_matrix, (
    'simulator does not re-export the same object'
)
print('pre-flight: simulation.delays.compute_delay_matrix is simulator.compute_delay_matrix:', True)
"

grep -n "simulator\|compute_delay_matrix" data_loader.py
```

**Expected stdout (from the python step):**
```
pre-flight: simulation.delays.compute_delay_matrix is simulator.compute_delay_matrix: True
```
**Expected grep output:** exactly two matches —
- `278:    from simulator import compute_delay_matrix`
- `313:    delays = compute_delay_matrix(`

If any other match appears, STOP and report. If the `is`-check fails,
STOP — the wrapper redefined the symbol, which contradicts doc 17
§6.4, and the spec must be revised before applying.

### T-1 — Compile + cache purge (run AFTER applying the patch)

```bash
cd /scratch/home/wog3597/vbi
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
python -m compileall -q .
echo "compileall exit: $?"
```

**Expected:** `compileall exit: 0`

### T-2 — Package resolution (sanity, unchanged from P-1)

```bash
python -c "
import inference, simulation, features, evaluation, pipelines
for m in (inference, simulation, features, evaluation, pipelines):
    assert m.__name__ + '/__init__' in m.__file__, m.__file__
print('all packages resolve correctly')
"
```

**Expected:** `all packages resolve correctly`

### T-3 — Patch-specific source check

```bash
python -c "
import inspect, data_loader
src = inspect.getsource(data_loader.get_subject_data)
assert 'from simulation.delays import compute_delay_matrix' in src, src
assert 'from simulator import' not in src, src
print('P-2 source OK')
"
```

**Expected:** `P-2 source OK`

### T-4 — `data_loader.py` is free of `simulator` references

```bash
grep -n "simulator\|^import simulator\|^from simulator" data_loader.py
```

**Expected:** zero output (grep exits 1).

Reason for searching the broader pattern `simulator` (not just
`import simulator`): we want to confirm no stray comment, docstring,
or name leak refers to the wrapper any longer.

### T-5 — Patch-specific binding check at runtime

```bash
python -c "
import data_loader, simulation.delays
# Locate the deferred import in the function body via getclosurevars or by string match.
import inspect
src = inspect.getsource(data_loader.get_subject_data)
assert 'simulation.delays' in src, 'package import not present'
# Verify the resolved object is the package's compute_delay_matrix.
from simulation.delays import compute_delay_matrix as cdm
assert callable(cdm), 'compute_delay_matrix is not callable'
import simulator
assert simulator.compute_delay_matrix is cdm, (
    'simulator wrapper diverged from simulation.delays — STOP'
)
print('P-2 binding OK:', cdm.__module__)
"
```

**Expected:** `P-2 binding OK: simulation.delays`

### T-6 — Whole-tree audit of `simulator` import counts

```bash
grep -rn "from simulator import\|import simulator\b" --include="*.py" . | wc -l
```

**Expected:** `25` (i.e. baseline 26 minus the one line P-2 removes).

Detailed expectation: doc 17 §5.3 enumerated 26 grep hits:
- 12 live package callers (data_loader: 1; inference: 5; evaluation: 6) — **the data_loader entry drops to 0**
- 6 debug/notebook-helper callers (unchanged)
- 5 dead-monolith callers in `inference.py` (unchanged)
- 3 cosmetic docstring matches in `simulator.py` (unchanged)

Post-P-2: 11 live + 6 debug + 5 dead + 3 docstring = **25**.

### T-7 — Evaluate compat is still functional (unchanged invariant from P-1)

```bash
python -c "
import evaluate  # compat wrapper (still on disk; P-2 does not touch it)
assert hasattr(evaluate, 'fc_metrics')
assert hasattr(evaluate, 'final_test')
print('evaluate.py compat wrapper still works')
"
```

**Expected:** `evaluate.py compat wrapper still works`. (P-2 is not
supposed to affect the `evaluation`/`evaluate` graph; this guards
against accidental coupling.)

### T-8 — Public-API smoke (carry-over from P-1 T-5)

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

**Expected:** `public-API smoke OK`

### T-9 — Optional: full no-GPU smoke

```bash
python debug.py --basic
```

**Expected:** exits cleanly. Optional — the spec accepts the patch as
verified after T-0 through T-8 even without this step (data file
availability gate).

---

## 9. Expected Test Result

If P-2 is applied correctly, every test above passes with the listed
expected output. The end state is:

| Property | Pre-P-2 | Post-P-2 |
|---|---|---|
| `compileall -q .` exit code | 0 | 0 |
| `data_loader.py` contains `from simulator import` | 1 line (278) | 0 lines |
| `data_loader.py` contains `from simulation.delays import compute_delay_matrix` | 0 lines | 1 line (278) |
| `simulator.compute_delay_matrix is simulation.delays.compute_delay_matrix` | True | True |
| `simulator.py` on disk | yes (untouched) | yes (untouched) |
| `evaluate.py` on disk | yes (untouched) | yes (untouched) |
| `inference.py` on disk | yes (untouched, R8) | yes (untouched, R8) |
| Whole-tree `grep simulator` count (*.py) | 26 | 25 |
| Live deferred `from simulator import` package-call sites | 12 | 11 |
| `data_loader.py` `simulator` references | 1 deferred | 0 |
| `inference/` `simulator` references | 5 deferred (unchanged) | 5 deferred (unchanged) |
| `evaluation/` `simulator` references | 6 deferred (unchanged) | 6 deferred (unchanged) |
| R3 violation count | 1 (`simulation/qc.py:25`) | 1 (unchanged) |
| Scientific outputs of any pipeline run | unchanged | unchanged |
| Git diff size | — | 1 file, 1 line, ~30 chars |

If any of the above does not match, rollback (Section 7).

---

## 10. Approval Checkpoint

Before applying P-2, every item below must hold. Confirm each with
the user **or** verify automatically before proceeding.

| # | Checkpoint | Required value | How to verify |
|---|---|---|---|
| C-1 | User explicitly authorises Patch 2 | `Proceed with Patch 2` (verbatim) | Wait for user message |
| C-2 | Patch 1 result is recorded and tests pass | doc 19 exists; T-1..T-7 PASS | Read 19_patch1_result.md |
| C-3 | Current branch is `refactor/02-simulation` | match | `git rev-parse --abbrev-ref HEAD` |
| C-4 | `data_loader.py` is clean of unstaged changes | clean | `git status data_loader.py` — must report no `M` |
| C-5 | T-0 pre-flight passes | `True` + exactly 2 grep matches | Run T-0 from Section 8 |
| C-6 | The wider repo is still in the doc 17 baseline state for everything outside `pipelines/stage1_stage2.py` | unchanged | Spot-check via T-2 and T-8 |

If C-4 fails (the file has unstaged changes), STOP and surface to the
user — do not silently overwrite.

If C-5 fails (T-0 finds an unexpected name or a re-export drift),
STOP. The spec is wrong; revise before applying.

If C-1 has not been given, do not apply the patch under any
circumstances.

### Files explicitly NOT touched by Patch 2

- `simulator.py` — kept on disk; required by 11 remaining deferred
  callers (5 in `inference/`, 6 in `evaluation/`).
- `evaluate.py` — out of scope.
- `evaluation/__init__.py` and all `evaluation/*.py` — out of scope.
- `inference.py` — R8 forbids edits.
- `inference/`, `simulation/`, `features/`, `pipelines/` — out of scope.
- `config.py` — R1 (no scientific changes).
- `simulation/wc_runner.py` — status `M` already (per doc 16 Appendix A).
- `main.ipynb`, `debug.py`, `debug_notebook.py` — Tier 4b.
- Any root duplicate (`fc.py`, `fcd.py`, `wc_runner.py`, `delays.py`,
  `warmup.py`, `qc.py`, `extraction.py`, `screening.py`) — Tier 6.
- Root `__init__.py` — Tier 6.
- `bold.py`, `main.py`, `pipeline_setup.py`, `README.md`,
  `PATCH_REPORT.md` — out of scope.
- All documentation files (`01_*.md` … `19_*.md`) — read-only for
  this patch sequence.

### Operations explicitly NOT performed by Patch 2

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
- No bundling of any other Tier 3 / Tier 4 / Tier 4b patch.

---

Do not apply this patch until the user explicitly says: Proceed with Patch 2.
