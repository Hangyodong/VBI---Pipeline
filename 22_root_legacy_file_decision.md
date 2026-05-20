# 22 — Root-Level Legacy File Decision Report

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** Decision report only. **No deletion, no move, no Python code change.**
**Predecessor docs:** 07, 11, 16, 17, 19, 20, 21
**Branch:** `refactor/02-simulation`

---

## 0. Read-Only Inventory (verified live before authoring)

Every claim in §1 is derived from a live read-only check performed
today. Findings:

### 0.1 Byte-identity vs package counterpart

```
fc.py            vs features/fc.py            : diff exit 0
fcd.py           vs features/fcd.py           : diff exit 0
extraction.py    vs features/extraction.py    : diff exit 0
screening.py     vs features/screening.py     : diff exit 0
wc_runner.py     vs simulation/wc_runner.py   : diff exit 0
warmup.py        vs simulation/warmup.py      : diff exit 0
delays.py        vs simulation/delays.py      : diff exit 0
qc.py            vs simulation/qc.py          : diff exit 0
__init__.py      vs simulation/__init__.py    : diff exit 0
```

All 9 root duplicates are still **byte-for-byte identical** to their
package counterparts (consistent with doc 17 §5.1 baseline).

### 0.2 Bare-name top-level import grep across `.py` (and notebook cell count)

```
fc           py-bare=  0  ipynb-cells=0
fcd          py-bare=  0  ipynb-cells=0
extraction   py-bare=  0  ipynb-cells=0
screening    py-bare=  0  ipynb-cells=0
wc_runner    py-bare=  0  ipynb-cells=0
warmup       py-bare=  0  ipynb-cells=0
delays       py-bare=  0  ipynb-cells=0
qc           py-bare=  0  ipynb-cells=0
inference    py-bare=  1  ipynb-cells=1   (`pipelines/stage1_stage2.py:39: import inference` resolves to inference/__init__.py)
simulator    py-bare=  0  ipynb-cells=1
evaluate     py-bare=  0  ipynb-cells=1
bold         py-bare=  0  ipynb-cells=0
vbi          py-bare=  0  ipynb-cells=0
```

Plus a deferred / any-position grep across all `.py`:

| Name | Any-position imports |
|---|---|
| `fc`, `fcd`, `extraction`, `screening`, `wc_runner`, `warmup`, `delays`, `qc` | **0 each** |
| `bold` | **6 hits** in package + root duplicate copies of `simulation/wc_runner.py` and `simulation/warmup.py` — `from bold import BoldMonitor` |

### 0.3 Entrypoint markers (`if __name__ == "__main__":`)

```
fc.py, fcd.py, extraction.py, screening.py, wc_runner.py, warmup.py,
delays.py, qc.py, inference.py, simulator.py, evaluate.py, bold.py,
__init__.py:  NONE have a __main__ block.
```

None of these 13 files is a script entrypoint.

### 0.4 Production `simulator` / `evaluate` reference counts (post-P-2)

From `21_patch2_result.md` §8, the 25 remaining `simulator` import
hits split as:

| Category | Count |
|---|---|
| `inference/` deferred package callers | 5 |
| `evaluation/` deferred package callers | 6 |
| `debug.py` / `debug_notebook.py` callers | 6 |
| `inference.py` dead-monolith callers | 5 |
| `simulator.py` docstring lines | 3 |

`evaluate` import hits remaining (post-P-1):

| Location | Count |
|---|---|
| `pipelines/`, `inference/`, `evaluation/`, `*.py` (top-level production) | **0** |
| `main.ipynb` notebook cell ~163 | 1 |
| `debug_notebook.py` (lines 82, 341) | 2 |

### 0.5 Working-tree status

`git status --short` (root duplicates):

- Mostly untracked (`??`) — they were never staged into git history.
- `wc_runner.py` mtime is May 18 11:16 (recent touch); content
  remains byte-identical to `simulation/wc_runner.py`. Per doc 16
  Appendix A the *package* copy had `M` status earlier; that line is
  unrelated to the root duplicate.

---

## 1. Per-File Analysis

The required schema for each file:

1. **Package equivalent** — where the canonical source lives.
2. **Imported anywhere?** — current caller surface.
3. **Script entrypoint?** — has `__main__` or runs standalone.
4. **Unique logic?** — anything not in the package version.
5. **Risk if deleted today.**
6. **Recommended action** — keep / archive later / merge unique
   logic first / safe to remove later.
7. **Required tests before action.**

---

### 1.1 `fc.py`

| Field | Value |
|---|---|
| Package equivalent | `features/fc.py` |
| Byte-identity | **Identical** (diff exit 0; 1,747 bytes / 61 lines) |
| Imported anywhere? | **No.** Zero `import fc` / `from fc import` matches in `*.py` and `*.ipynb`. All callers use `from features.fc import …`. |
| Script entrypoint? | No (no `__main__` block) |
| Unique logic? | No — verified by diff |
| Risk if deleted | **LOW.** No callers. The package copy is the canonical source and is used everywhere. |
| Recommended action | **archive later** (Tier 6 of doc 16 §7) — defer until P-3/P-4 finish so the simulator/evaluate wrappers can also drop. |
| Required tests | Re-run T-1 (`compileall`) and `grep -rn "^import fc\b\|^from fc import"` (must stay zero) right before archiving. Same for the matching package smoke test. |

---

### 1.2 `fcd.py`

| Field | Value |
|---|---|
| Package equivalent | `features/fcd.py` |
| Byte-identity | **Identical** (3,195 bytes / 92 lines) |
| Imported anywhere? | **No.** Zero matches. |
| Script entrypoint? | No |
| Unique logic? | No |
| Risk if deleted | **LOW.** FCD is disabled in production (`USE_FCD=False`); even if it were on, only `features/fcd.py` would be touched. |
| Recommended action | **archive later** (Tier 6) |
| Required tests | Same as `fc.py`. Plus: the FCD dual-role surface (doc 16 Risk A6) is dormant; archiving the root duplicate does not change that surface. |

---

### 1.3 `extraction.py`

| Field | Value |
|---|---|
| Package equivalent | `features/extraction.py` |
| Byte-identity | **Identical** (5,252 bytes / 139 lines) |
| Imported anywhere? | **No.** Zero matches. |
| Script entrypoint? | No |
| Unique logic? | No |
| Risk if deleted | **LOW**, but doc 17 §6.3 flags that `extraction.py` internally imports `from features.fc import …` and `from features.fcd import …`. So loading the *root* copy would also pull in `features/`. This is dormant (no bare-name callers) and does not change the deletion calculus. |
| Recommended action | **archive later** (Tier 6) |
| Required tests | Same as `fc.py`. Also: `python -c "from features import extract_observed_features, worker_extract"` to confirm the canonical surface still resolves through the package. |

---

### 1.4 `screening.py`

| Field | Value |
|---|---|
| Package equivalent | `features/screening.py` |
| Byte-identity | **Identical** (2,558 bytes / 78 lines) |
| Imported anywhere? | **No.** Zero matches. |
| Script entrypoint? | No |
| Unique logic? | No. Doc 16 §6 explicitly calls this file a "future stub" / "no logic to lose" duplicate. |
| Risk if deleted | **LOWEST of any duplicate.** |
| Recommended action | **archive later** (Tier 6 — first to go in the ordered deletion sequence per doc 16 §7). |
| Required tests | Compile + grep, same as above. |

---

### 1.5 `wc_runner.py`

| Field | Value |
|---|---|
| Package equivalent | `simulation/wc_runner.py` |
| Byte-identity | **Identical** today (diff exit 0; 13,900 bytes / 380 lines). Root file mtime is `May 18 11:16` — touched in this branch, but content still matches. |
| Imported anywhere? | **No bare `import wc_runner`** anywhere in `*.py` / `*.ipynb`. All callers use `simulation.wc_runner.*`. |
| Script entrypoint? | No |
| Unique logic? | No — but this is the highest-stakes duplicate (it contains the per-sim parameter injection contract; doc 16 §8 / R1). Treat with extra care **only if** divergence is suspected; current diff confirms there is none. |
| Risk if deleted | **MEDIUM today**, **LOW after a re-verified diff**. Risk is procedural: the package copy was modified in this branch, and deleting the root duplicate is irreversible. |
| Recommended action | **archive later** (Tier 6 — second-to-last in the ordered deletion sequence; doc 16 §7 says delete *after* the smaller-risk duplicates). |
| Required tests | (a) Re-run `diff wc_runner.py simulation/wc_runner.py` (must exit 0) immediately before archiving. (b) `python -c "from simulation.wc_runner import simulate_gpu_batch, simulate_single, _import_wc, _apply_engine"` to confirm the canonical surface still resolves. (c) Optional: spot-run `python debug.py --basic` to verify the imports subtest still passes. |

---

### 1.6 `warmup.py`

| Field | Value |
|---|---|
| Package equivalent | `simulation/warmup.py` |
| Byte-identity | **Identical** (8,894 bytes / 267 lines) |
| Imported anywhere? | **No.** Zero bare-name matches. |
| Script entrypoint? | No |
| Unique logic? | No |
| Risk if deleted | **LOW.** |
| Recommended action | **archive later** (Tier 6) |
| Required tests | Compile + grep + `from simulation.warmup import WarmupResult, warmup_run, simulate_with_warmup`. |

Note: the root `warmup.py` (like the package copy) contains
`from bold import BoldMonitor` at lines 98 and 187. That import is
not affected by archiving `warmup.py`; the **package** copy of
`warmup.py` is what production loads, and it makes the same call.
See §1.12 for the `bold` analysis.

---

### 1.7 `delays.py`

| Field | Value |
|---|---|
| Package equivalent | `simulation/delays.py` |
| Byte-identity | **Identical** (3,989 bytes / 117 lines) |
| Imported anywhere? | **No.** After Patch 2 (doc 21), the only remaining usage of `compute_delay_matrix` in production is via `from simulation.delays import compute_delay_matrix` (in `data_loader.py:278`). No bare `import delays` anywhere. |
| Script entrypoint? | No |
| Unique logic? | No |
| Risk if deleted | **LOW.** |
| Recommended action | **archive later** (Tier 6) |
| Required tests | Compile + grep + `python -c "from simulation.delays import compute_delay_matrix, detect_delay_key, apply_delay"`. Plus the data-loader exercise from doc 20 T-3 (`inspect.getsource(data_loader.get_subject_data)` contains the package import) — already passing post-P-2. |

---

### 1.8 `qc.py`

| Field | Value |
|---|---|
| Package equivalent | `simulation/qc.py` |
| Byte-identity | **Identical** (4,281 bytes / 126 lines) |
| Imported anywhere? | **No.** |
| Script entrypoint? | No |
| Unique logic? | No |
| Risk if deleted | **LOW.** Note: the **package** copy still contains the documented R3 violation (`simulation/qc.py:25: from features.fc import …`; doc 17 §5.5). Archiving the *root* duplicate does not change that; it only removes the dormant root copy. |
| Recommended action | **archive later** (Tier 6). The R3 fix is a separate, scoped concern. |
| Required tests | Compile + grep + `python -c "from simulation.qc import assert_theta_feature_distinct, run_theta_specific_check, theta_feature_diff_norm"`. |

---

### 1.9 Root `__init__.py`

| Field | Value |
|---|---|
| Package equivalent | `simulation/__init__.py` (byte-identical; 1,603 bytes / 62 lines) |
| Imported anywhere? | **No `import vbi` callers** found anywhere in `*.py` or `*.ipynb`. The file activates only if the **parent** of `/scratch/home/wog3597/vbi` is on `sys.path`. Latent risk only (doc 11 CONFLICT-4 / doc 17 §4). |
| Script entrypoint? | No |
| Unique logic? | No |
| Risk if deleted | **LOW today; LATENT MEDIUM if any environment puts `/scratch/home/wog3597` on `sys.path`.** Removing makes the repo no longer importable as a package named `vbi`. No file currently relies on that. |
| Recommended action | **archive later** (Tier 6 — *last*, after all other duplicates have been moved and one final grep audit confirms no `import vbi` exists anywhere in the org's other repos or scratch dirs that share this sys.path). |
| Required tests | (a) `grep -rn "^import vbi\b\|^from vbi import\|^import vbi\." --include="*.py" --include="*.ipynb" .` returns zero. (b) After archiving, `python -m compileall -q .` and the public-API smoke (P-1/P-2 T-2/T-5) still pass. |

---

### 1.10 `inference.py`

| Field | Value |
|---|---|
| Package equivalent | `inference/` package (55,157 bytes / 1,535 lines vs the package's many submodules) |
| Byte-identity | N/A — this is a **monolith**, not a duplicate. The package is a restructured/superset version with additional submodules and a corrected Stage 2 path. |
| Imported anywhere? | `pipelines/stage1_stage2.py:39: import inference` and `main.ipynb` cell — both resolve to the **package** at runtime (doc 17 §4, §2.4). The monolith is **unreachable** in normal Python execution. |
| Script entrypoint? | No |
| Unique logic? | **Uncertain.** Doc 16 §2 verified that `inference/__init__.py` exports every public name plus 8 private helpers from the monolith, but a full per-line audit has not been performed. Doc 17 / 16 also confirms the monolith has a hard `NameError` on `n_subj` in `collect_stage2_data` — *proving* it has never been executed, but not proving the rest of the file has no unique code. **Diff audit required before deletion.** |
| Risk if deleted | **HIGH (procedural) until diff audit is complete; LOW (runtime) because the file is unreachable.** Editor / linter UX immediately improves once it is gone. |
| Recommended action | **merge unique logic first → safe to remove later.** Specifically: (i) audit `inference.py` vs `inference/__init__.py` `__all__`; (ii) for every public/private name in `inference.py`, confirm it exists in `inference/` and is functionally equivalent; (iii) if any name exists *only* in `inference.py`, decide whether to port it or leave it dead. Then archive. |
| Required tests | (a) `python -c "import inference; print(inference.__file__)"` must return the package path (already true). (b) `compileall -q .` (c) Public-API smoke (`from inference import ParameterScaler, FeaturePipeline, run_stage1_snpe, run_stage2_snpe, save_artifacts, load_artifacts`). (d) **Diff audit:** enumerate every top-level name in `inference.py` and confirm presence in the package. (e) Optional `python debug.py --basic` (no-GPU). |

This is Tier 7 in doc 16 §7 (last to go, after all simulator and
evaluate wrappers).

---

### 1.11 `simulator.py`

| Field | Value |
|---|---|
| Package equivalent | `simulation/` + `features/` packages. `simulator.py` is a **compat wrapper** (64 lines, zero logic — pure re-exports). |
| Byte-identity | N/A (wrapper, not a duplicate). Doc 17 §6.4 confirms zero logic. |
| Imported anywhere? | **Yes, heavily.** Post-P-2: 11 live deferred callers (5 in `inference/`, 6 in `evaluation/`) + 6 debug/notebook helpers + 5 in the dead `inference.py` monolith + 3 cosmetic docstring matches in the wrapper itself = 25 grep hits (doc 21 §8). |
| Script entrypoint? | No |
| Unique logic? | No — pure re-exports of `simulation.*` and `features.*`. |
| Risk if deleted today | **CRITICAL.** Every Tier 3 / Tier 4 deferred import would fail at first call. |
| Recommended action | **keep** until all Tier 3 + Tier 4 + Tier 4b patches close. After that, **safe to remove later** (Tier 5 of doc 16 §7). |
| Required tests | Before any archival: (a) `grep -rn "from simulator import\|import simulator\b" --include="*.py" --include="*.ipynb" .` returns **zero** (today: 25 hits). (b) `compileall -q .` and full pipeline smoke. (c) Backwards-compat smoke: deletion is gated on the wrapper having no remaining callers anywhere on disk. |

---

### 1.12 `evaluate.py`

| Field | Value |
|---|---|
| Package equivalent | `evaluation/` package. `evaluate.py` is a **35-line compat wrapper** (doc 17 §6.4) that re-exports every public name plus 8 private helpers. |
| Byte-identity | N/A (wrapper). |
| Imported anywhere? | **Yes (low volume).** Post-P-1: zero top-level production callers; 1 notebook cell (`main.ipynb` ~163); 2 cells in `debug_notebook.py` (lines 82, 341). |
| Script entrypoint? | No |
| Unique logic? | No — pure re-exports. |
| Risk if deleted today | **HIGH.** Notebook + debug callers would break. |
| Recommended action | **keep** until Tier 4b migrates the notebook + debug callers; then **safe to remove later** (Tier 5). |
| Required tests | (a) `grep -rn "import evaluate\b\|from evaluate import" --include="*.py" --include="*.ipynb" .` returns zero. (b) Public-API smoke for the `evaluation` package (already exercised by P-1 T-5, P-2 T-8). (c) `python debug.py --basic` `imports` subtest still PASSes (already passing today). |

---

### 1.13 `bold.py` — **NOT a duplicate**

| Field | Value |
|---|---|
| Package equivalent | **NONE.** There is no `simulation/bold.py`. `bold.py` is the *sole* source of `BoldMonitor` and the TVB-style HRF helpers (`tvb_hrf`, etc.). |
| Imported anywhere? | **Yes — load-bearing.** Confirmed callers: `simulation/wc_runner.py:128`, `simulation/warmup.py:98`, `simulation/warmup.py:187` — all `from bold import BoldMonitor`. Plus the byte-identical lines in the root duplicates `wc_runner.py` / `warmup.py` (which point at the same `bold.py`). |
| Script entrypoint? | No |
| Unique logic? | **Yes — entirely unique.** 253 lines / 9,653 bytes including `tvb_hrf`, the `BoldMonitor` state machine, HRF kernel logic, and the TVB-comparison docstring. **Not present in `simulation/`** or anywhere else in the tree. |
| Risk if deleted | **CRITICAL.** Every WC simulation that drives BOLD output (i.e., every production pipeline run) collapses immediately. |
| Recommended action | **keep** — permanently, in its current location, until and unless someone moves it into the `simulation/` package as a deliberate refactor (out of scope here). Doc 07 R3 explicitly carves out the exception: "bold.py may be imported by simulation/wc_runner.py and simulation/warmup.py." Doc 16 §8 corroborates: "bold.py — Untouched in the cleanup scope. CPU-only enforced (R13). Out of scope." |
| Required tests | None — it is not being modified. (For paranoia: confirm that `import bold` still resolves to `./bold.py` and that `BoldMonitor` is importable; both are exercised in every pipeline run already.) |

`bold.py` should be excluded from any deletion or archive plan. The
user's question included it specifically; the answer is that
**it does not overlap with `simulation/`** — it is a sibling module
that `simulation/` depends on.

---

## 2. Files That Must Remain

| File | Why |
|---|---|
| `bold.py` | Sole source of `BoldMonitor` + TVB-style HRF. Load-bearing for every pipeline run. R3 allows this as an explicit cross-module exception. |
| `simulator.py` | 25 grep hits today (11 live deferred + 6 debug + 5 dead-monolith + 3 docstring). Required by every Tier 3 / Tier 4 / Tier 4b deferred call path until those are migrated. |
| `evaluate.py` | 3 notebook/debug-callsites still resolve through it. Required until Tier 4b lands. |
| `inference.py` | Cannot move yet — diff audit not complete. R8 forbids edits. Remove only after full per-name audit confirms `inference/__init__.py` is a strict superset. |

`config.py`, `data_loader.py`, `main.py`, `pipeline_setup.py`,
`main.ipynb`, `debug.py`, `debug_notebook.py`, `bold.py` are non-
duplicate root files and remain in place by default. They were
**not** the subject of this report's question but are noted here for
completeness.

---

## 3. Files Likely Safe To Archive Later

All 9 root duplicates are byte-identical to their package
counterparts and have zero bare-name callers anywhere in `*.py` or
`*.ipynb`. They are deletion-safe **after** the Tier 1–5 patches
finish (i.e., after `simulator.py` and `evaluate.py` are themselves
removed). Order of archival, lowest-risk first (matches doc 16 §7
Tier 6 ordering):

1. `screening.py` — pure stub, no logic to lose.
2. `fcd.py` — FCD disabled in production today.
3. `fc.py` — stable.
4. `delays.py` — stable (P-2 cemented the package path).
5. `warmup.py` — stable.
6. `qc.py` — stable (R3 violation lives in the *package* copy, not the root one).
7. `extraction.py` — stable; internal imports of `features.fc` / `features.fcd` are dormant since no caller hits the root copy.
8. `wc_runner.py` — high-stakes content (per-sim parameter injection); diff again right before archiving.
9. `__init__.py` (root) — archive *last*, after a final repo-wide `import vbi` grep is clean.

These should be moved into an archive directory rather than deleted
outright. See §5 for the proposed command.

---

## 4. Files Needing Manual Review Before Any Archive Decision

| File | What to verify |
|---|---|
| `inference.py` | Full diff audit against `inference/__init__.py` `__all__`. Enumerate every top-level name (def / class / module-level assignment) in `inference.py` and confirm presence + functional equivalence in the package. Spot-check the known `n_subj` NameError in `collect_stage2_data` is genuinely gone in `inference/stage2.py:163`. Confirm no test, notebook cell, or external script imports a name that lives only in the monolith. |
| `simulator.py` | Re-run the post-P-4f grep before archiving. Must show **zero** remaining `from simulator import` / `import simulator` matches in the entire tree (including `*.ipynb`). The 3 cosmetic docstring matches inside `simulator.py` itself disappear automatically when the file is archived. |
| `evaluate.py` | Same as above but for `import evaluate` / `from evaluate import`. Must be zero. The 8 private helpers (`_aggregate_validation`, `_print_selection_table`, `_print_test_summary`, `_print_validation_summary`, `_progress`, `_resimulate_and_score`, `_test_stage1`, `_test_stage2`) must be re-checked against `evaluation/` to ensure none has slipped out of the package re-export list. |
| `wc_runner.py` (root) | Final `diff wc_runner.py simulation/wc_runner.py` must exit 0 immediately before archive. If diff is non-zero, **stop** — investigate which copy is canonical before moving anything. The May 18 mtime on the root file is a yellow flag (but the diff currently confirms parity). |
| `__init__.py` (root) | One last `grep -rn "^import vbi\b\|^from vbi\b" --include="*.py" --include="*.ipynb" .` (anywhere on the filesystem the user can reach, including any sibling repos sharing `sys.path`). |

For the 7 other root duplicates (`fc.py`, `fcd.py`, `extraction.py`,
`screening.py`, `warmup.py`, `delays.py`, `qc.py`) the diff +
zero-caller status today is the same evidence base that has stood
since doc 11 / doc 17 — no further manual review is required beyond
re-confirming the same two checks at archive time.

---

## 5. Proposed Archive Command (DO NOT EXECUTE)

The user explicitly requested an exact command proposal **but no
execution**. Below is the safest reversible recipe. It

- Moves files into a tracked archive directory rather than deleting them.
- Preserves git history for already-tracked files via `git mv`.
- For untracked root duplicates (most of them), uses plain `mv` (no
  `git rm`, no irreversible action).
- Runs in two phases so the highest-stakes files (`wc_runner.py`,
  root `__init__.py`) move *last*.

```bash
# === PROPOSED — DO NOT EXECUTE WITHOUT EXPLICIT USER GO-AHEAD ===

cd /scratch/home/wog3597/vbi

# Pre-flight (must all return zero output before proceeding):
grep -rn "^import fc\b\|^from fc import"                     --include="*.py" --include="*.ipynb" .
grep -rn "^import fcd\b\|^from fcd import"                   --include="*.py" --include="*.ipynb" .
grep -rn "^import extraction\b\|^from extraction import"     --include="*.py" --include="*.ipynb" .
grep -rn "^import screening\b\|^from screening import"       --include="*.py" --include="*.ipynb" .
grep -rn "^import wc_runner\b\|^from wc_runner import"       --include="*.py" --include="*.ipynb" .
grep -rn "^import warmup\b\|^from warmup import"             --include="*.py" --include="*.ipynb" .
grep -rn "^import delays\b\|^from delays import"             --include="*.py" --include="*.ipynb" .
grep -rn "^import qc\b\|^from qc import"                     --include="*.py" --include="*.ipynb" .
grep -rn "^import vbi\b\|^from vbi\b"                        --include="*.py" --include="*.ipynb" .

# Pre-flight (must all return diff exit 0):
diff fc.py         features/fc.py
diff fcd.py        features/fcd.py
diff extraction.py features/extraction.py
diff screening.py  features/screening.py
diff wc_runner.py  simulation/wc_runner.py
diff warmup.py     simulation/warmup.py
diff delays.py     simulation/delays.py
diff qc.py         simulation/qc.py
diff __init__.py   simulation/__init__.py

# Create archive directory (tracked):
mkdir -p _legacy_root

# Phase A — low-risk root duplicates (move-only; preserves files):
mv screening.py  _legacy_root/
mv fcd.py        _legacy_root/
mv fc.py         _legacy_root/
mv delays.py     _legacy_root/
mv warmup.py     _legacy_root/
mv qc.py         _legacy_root/
mv extraction.py _legacy_root/

# Post-Phase-A verification:
python -m compileall -q .
python -c "
import inference, simulation, features, evaluation, pipelines
from features import fc, fcd, extraction, screening
from simulation import wc_runner, warmup, delays, qc
print('phase A imports OK')
"

# Phase B — higher-stakes root duplicates (only after Phase A passes):
mv wc_runner.py  _legacy_root/   # mtime hot; re-diff above must have been clean
mv __init__.py   _legacy_root/   # makes repo no longer importable as 'vbi'

# Post-Phase-B verification:
python -m compileall -q .
python -c "
import inference, simulation, features, evaluation, pipelines
print('phase B packages still OK')
"

# Phase C — compat wrappers and monolith (DO NOT INCLUDE in Tier-6
# archive; these are Tier 5 / Tier 7 and require their own gating
# patches first):
# DO NOT run yet:
#   mv simulator.py _legacy_root/
#   mv evaluate.py  _legacy_root/
#   mv inference.py _legacy_root/

# === END PROPOSED COMMAND ===
```

### Notes on the proposed command

- **Reversibility:** `mv … _legacy_root/` is reversible by
  `mv _legacy_root/<name> .`. No data is destroyed.
- **No `git rm`:** the files are mostly untracked today (`??` in
  `git status`). Plain `mv` is correct. If the user later commits the
  archive directory, history of the archived files is preserved as a
  rename in the same commit.
- **No `--no-verify`, no `git config` change, no push, no PR.**
- **No edits to any `.py` file.** This satisfies "Do NOT modify
  Python code."
- **Phase C is intentionally commented out.** Removing `simulator.py`,
  `evaluate.py`, or `inference.py` requires their gating patches
  (P-3*, P-4*, P-4b, Tier-7 audit) to land first. They are **not** in
  this Tier-6 archive command.

---

## 6. Summary Table

| # | File | Action | When |
|---|---|---|---|
| 1 | `bold.py` | **keep** (permanent) | now and forever (subject to a future deliberate refactor) |
| 2 | `simulator.py` | **keep** | until P-3* + P-4* + P-4b close all 11 live deferred call sites + 6 debug/notebook references |
| 3 | `evaluate.py` | **keep** | until P-4b updates `main.ipynb` cell ~163 and `debug_notebook.py` lines 82 + 341 |
| 4 | `inference.py` | **merge unique logic first → archive later** | after a full per-name audit against `inference/__init__.py` |
| 5 | `screening.py` | **archive later** | Tier 6, first |
| 6 | `fcd.py` | **archive later** | Tier 6, second |
| 7 | `fc.py` | **archive later** | Tier 6, third |
| 8 | `delays.py` | **archive later** | Tier 6, fourth |
| 9 | `warmup.py` | **archive later** | Tier 6, fifth |
| 10 | `qc.py` | **archive later** | Tier 6, sixth |
| 11 | `extraction.py` | **archive later** | Tier 6, seventh |
| 12 | `wc_runner.py` | **archive later** | Tier 6, eighth (high-stakes; re-diff at archive time) |
| 13 | `__init__.py` (root) | **archive later** | Tier 6, last (after final `import vbi` grep) |

No Python code is modified by this report. No file is moved or
deleted. The proposed command in §5 is for the user's review only and
must be approved before execution.

---

**End of decision report. Awaiting user instruction.**
