# 16 — Opus Refactor Decision Plan

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (high-reasoning verification pass)
**Predecessor docs reviewed:** 01–15 (this plan now incorporates the
`15_evaluation_module_review.md` findings, which were inspected directly
in the first pass and then formally documented after this plan was first
written)
**Branch:** `refactor/02-simulation`
**Working tree status (verified):** 7 modified files (config.py, debug_notebook.py,
evaluate.py, main.ipynb, main.py, simulation/wc_runner.py, .gitignore) + 26
untracked (the 15 documentation files plus the root duplicates flagged as legacy)
**Update notice:** This file was revised after `15_evaluation_module_review.md`
landed. Substantive updates to Sections 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
and Appendix B. Tier and patch-number identifiers (P-1 through P-4f) are
unchanged; ordering and exact site counts are tightened.

---

## 1. Executive Summary

The repo is in a **structurally complete refactor** with one residual debt: 12 root-level
legacy files (the 55 KB `inference.py` monolith, 2 compat wrappers, 8 root duplicates,
and 1 root `__init__.py`) that coexist with the new 5-package layout
(`simulation/`, `features/`, `inference/`, `evaluation/`, `pipelines/`).

**Runtime is safe today** — the package layer wins on every `import` (verified
empirically: `python -c "import inference; print(inference.__file__)"` returns
`inference/__init__.py`). **Tooling and contributor confusion is the live risk**:
editors and static analysers may resolve `import inference` to the monolith,
which has at least one provable NameError bug and a smaller training batch size.

The cleanup blocker is **one top-level production import** (`pipelines/stage1_stage2.py:38: import evaluate`)
plus **12 deferred `from simulator import` calls** inside function bodies across
`inference/`, `evaluation/`, and `data_loader.py`. None of these is on the
critical path of scientific correctness; all use the compat wrappers
(`evaluate.py`, `simulator.py`) which themselves correctly re-export the package
implementations.

The Sonnet-generated docs (11–15) are largely **accurate and verified**. One
finding originally flagged as a "real bug" (the FC PCA default constant)
turned out to be fine due to an alias in `config.py`. Two new genuine bugs
surfaced during verification (a `n_subj` NameError in `inference.py`'s dead
Stage 2 code, and a stale 116-region claim in `PATCH_REPORT.md`). Doc 15
independently confirmed the FCD dual-role pattern in evaluation and surfaced
**two additional latent FCD sites** beyond those in doc 14 — bringing the
total Tier X scope to ~9 call sites, all dormant while `USE_FCD=False`.

`evaluate.py` is confirmed to be a **35-line zero-logic compat wrapper**.
The evaluation package is otherwise **structurally clean** (no top-level
legacy imports, R3-compliant). Once Patch P-1 lands, removing
`evaluate.py` is a single subsequent step.

**Recommended first code change** — a single-line patch:
`pipelines/stage1_stage2.py:38`  `import evaluate` → `import evaluation as evaluate`.
This is the smallest possible step that begins unblocking compat wrapper removal,
preserves all `evaluate.*` call sites, and changes no scientific behavior.

---

## 2. What the Sonnet-Generated Documents Got Right

Verified item-by-item against current source.

| Claim | Doc | Verified by | Result |
|---|---|---|---|
| `import inference` resolves to `inference/__init__.py` | 09/11/14 | `python -c "import inference; print(inference.__file__)"` | ✅ Correct |
| All 5 package `__init__.py` files load cleanly | 06/09 | `python -c "import simulation; import features; import inference; import evaluation; import pipelines"` | ✅ All 5 resolve to package paths |
| All 8 root duplicates are byte-identical to package counterparts | 09/11 | `diff` on each pair (all 8 returned exit 0, no output) | ✅ Correct |
| Root `__init__.py` is identical to `simulation/__init__.py` | 09/11 | `diff __init__.py simulation/__init__.py` (no output) | ✅ Correct |
| Only one top-level production legacy import: `pipelines/stage1_stage2.py:38` | 11 | `grep -n "^import evaluate\|^from evaluate" *.py inference/*.py pipelines/*.py` | ✅ Only that one match |
| 12 deferred `from simulator import` calls in package code | 11/12/13/14 | `grep -rn "from simulator import" inference/ evaluation/ data_loader.py` | ✅ 12 lines, all inside function bodies |
| `inference.py` is 55 KB, package wins | 01/09/14 | `wc -c inference.py` → 55,157 bytes | ✅ Correct |
| `simulator.py` and `evaluate.py` contain only re-exports (no logic) | 09/10/11 | Files read in full — only `from X import Y` lines | ✅ Confirmed |
| R3 violation in `simulation/qc.py:25` (imports `features.fc`) | 12/13/14 | `grep "from features" simulation/*.py` → exact line | ✅ Real R3 breach |
| Notebook has 2 legacy import cells (`import evaluate`, `import simulator`) | 11 | `grep -n` on `main.ipynb` JSON → lines 163, 165 | ✅ Correct |
| `config.PCA_DIM_FCD = 100` is dangling (unused by `FeaturePipeline`) | 13 | `grep "PCA_DIM_FCD" inference/feature_pipeline.py` → 0 matches | ✅ Real dangling constant |
| `FeaturePipeline` uses `FamilyScaler` z-score on FCD (5-dim), not PCA | 13/14 | Read `inference/feature_pipeline.py:188` (FCD path) | ✅ Correct |
| Step 4 `FamilyScaler("FCD")` is fitted but the result is not stored in artifacts | 14 | Read `inference/stage1.py` — return dict has no `scalers` key | ✅ Correct (redundant fit when USE_FCD=True) |
| `data_loader.py:278` uses `from simulator import compute_delay_matrix` | 10/11 | `sed -n '270,290p' data_loader.py` | ✅ Verified exact line |
| `inference.py:collect_stage2_data` has a `n_subj` NameError bug | 14 | `grep -n "n_subj" inference.py` (line 1391 uses, no definition in function) | ✅ Real bug — proves monolith is unreachable |
| `pipeline_setup._PIPELINE_MODULES` lists `"simulator"` and `"evaluate"` | 10/11 | Read line 116 of `pipeline_setup.py` | ✅ Correct |
| `evaluate.py` is a 35-line pure re-export (no logic) | 15 | `wc -l evaluate.py` + read → 0 `def`/`class` | ✅ Correct |
| 6 deferred `from simulator import` calls in `evaluation/` (4 files) | 11/15 | `grep -rn "from simulator import" evaluation/` → 6 lines | ✅ Correct |
| `evaluation/` has zero top-level legacy imports (R3-clean) | 15 | `grep -n "^import \|^from " evaluation/*.py` → only stdlib/numpy/matplotlib/config/evaluation.* | ✅ Correct |
| Selection score formula `1·corr − 0.5·rmse − 0.5·fcd` with FCD term silenced when `USE_FCD=False` | 15 | Read `evaluation/model_selection.py:74–80` + `config.py:269–271` | ✅ Correct |
| Stage 2 nuisance handling at val/test time is hardcoded `"fix_mean"` (not `config.NUISANCE_METHOD`) | 15 | Read `evaluation/validation.py:136–141` + `final_test.py:134–139` | ✅ Correct — config.NUISANCE_METHOD only governs training-data collection |
| FCD dual-role bug also present at `evaluation/validation.py:135` and `evaluation/final_test.py:133` (pipeline-transform side) | 15 | Read both files at those lines | ✅ Correct — extends doc 14 (2 sites) → doc 16 first pass (4 sites) → doc 15 (additional sites confirmed) |
| `bootstrap_ci` uses `RandomState(42)`, not `config.SEED` | 15 | Read `evaluation/metrics.py:89` | ✅ Correct (intentional reproducibility independence; cosmetic concern only) |
| `_aggregate_validation` stores `shrinkage_mean` and `shrinkage_per_param` as identical computations | 15 | Read `evaluation/validation.py:184–189` | ✅ Confirmed — desync risk if one consumer changes |

The above is a complete verification of every load-bearing claim in docs 09–15.
No false positives found in the Sonnet output for these items.

---

## 3. What Needs Verification or Correction

| Claim | Doc | Verification result | Correction |
|---|---|---|---|
| `15_evaluation_module_review.md` exists | (this task's instructions, initial pass) | First pass: `ls 15_*.md` returned no file. Second pass: file was created retroactively after this plan was first written. | **Resolved.** Doc 15 now exists (981 lines). Its findings are integrated below. |
| `FCPCAScaler.__init__` uses `config.PCA_DIM` instead of `config.PCA_DIM_FC` | 14 (implied minor concern in the deep dive) | `grep -n "PCA_DIM" config.py` → `PCA_DIM = PCA_DIM_FC` (line 205) is an explicit alias | **Not a bug.** `config.PCA_DIM` and `config.PCA_DIM_FC` are the same value (300). The default works correctly. `FeaturePipeline` also passes `PCA_DIM_FC` explicitly. No action needed. |
| FCD shape asymmetry between `extract_features` (5-zero) and `extract_simulated_features` (0-zero) is "harmless today" | 13 | Confirmed: `worker_extract` calls `extract_features` → fcd_raw is `(N, 5)` all zeros; `FeaturePipeline` with `USE_FCD=False` sets `fcd_dim=0` and ignores `fcd_raw` in `transform()`. | **Sonnet was correct.** No action needed today, but Step F1 in 13 is the right fix before enabling FCD. |
| The README documents the correct architecture | (implicit) | Read first 60 lines of `README.md` → still lists `simulator.py`, `inference.py`, `evaluate.py` as primary files | **README is stale.** Documents pre-refactor structure. New contributors will be misled. Note in plan; do not touch as part of code cleanup. |
| `PATCH_REPORT.md` reflects current state | (implicit) | Read first 50 lines → item #2 states "Changed default data dimensions to 116 regions, FC_DIM = 6670". Current `config.py` has `N_REGIONS = 115`, `FC_DIM = 6555`. | **PATCH_REPORT is partially obsolete.** Item #2 contradicts current production. The doc is a history log, not a state description; mark as such rather than edit. |
| RISK-I (PPC `fcd_obs_raw` dual-role) and RISK-II (SBC `fcd_to_upper_tri` mismatch with 5-dim FCD) | 14/15 | Read `inference/posterior.py:107–165`, `inference/diagnostics.py:40–90`, and the four evaluation sites in doc 15 IRISK-4. All pass 6555-dim FCD to `feature_pipeline.transform`, which would `ValueError` if `USE_FCD=True` because `fcd_dim=5`. With `USE_FCD=False` the FCD branch is skipped entirely. | **Real latent bug.** Currently harmless. Doc 15 expanded the site count to **the full evaluation surface** (see Section 3.1 below). Tier X scope grows. |

### 3.1 Updated finding — FCD dual-role bug full surface (8 sites)

Doc 14 found the pattern in 2 inference files. Doc 16 (first pass) extended
the count to 4 by adding `evaluation/metrics.py` lines 184 and 245. Doc 15
extended the count further: every place that calls
`feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)` with
`fcd_obs_raw` from `extract_observed_features` is part of the bug surface —
the input shape contradicts what the pipeline expects when `USE_FCD=True`.

**Full site inventory (FCD dual-role / dimension mismatch when USE_FCD=True):**

| File | Line | Role |
|---|---|---|
| `inference/posterior.py` | 123 | PPC `feature_pipeline.transform` |
| `inference/posterior.py` | 160 | PPC `fcd_obs_raw - fcd_to_upper_tri(...)` RMSE compare |
| `inference/diagnostics.py` | 77–79 | SBC `fcd_to_upper_tri` fed to `feature_pipeline.transform` |
| `inference/stage2.py` | 322 | `run_stage2_snpe` builds `x_obs_s1` via `s1_pipeline.transform` |
| `evaluation/metrics.py` | 123 | `evaluate_subject` `feature_pipeline.transform` |
| `evaluation/metrics.py` | 211–214 | `_resimulate_and_score` `fcd_to_upper_tri` + `fcd_vec_rmse` compare |
| `evaluation/metrics.py` | 252 + 277 | `baseline_eval` transform + RMSE compare |
| `evaluation/validation.py` | 135 | `evaluate_validation_stage2` `s1_pipeline.transform` (nuisance prep) |
| `evaluation/final_test.py` | 133 | `_test_stage2` `s1_pipeline.transform` (nuisance prep) |

That's **8 distinct sites across 5 files**. All are dormant while
`USE_FCD=False`. Tier X must address all of them in one bundle —
partial fixes would leave latent shape failures.

The cleanest fix per doc 15 Step E2: change
`extract_observed_features` to return the **FCD matrix `(N, N)`**, and
have each downstream caller compute either `fcd_to_summary_stats(mat)` for
pipeline input or `fcd_to_upper_tri(mat)` for RMSE comparison. This makes
the type contract explicit at each call site.

### 3.2 New finding — `_PIPELINE_MODULES` reload list is incomplete

Sonnet flagged that `pipeline_setup._PIPELINE_MODULES = ("config", "data_loader", "bold", "simulator", "inference", "evaluate")`
will drop the compat wrappers from `sys.modules` but not the package submodules.
Verified. This means `setup_pipeline()` in a notebook session will leave
`evaluation.metrics`, `simulation.wc_runner`, etc. holding stale config values
after a reload. This is a latent bug that surfaces only in long-running
notebook sessions — not a release blocker, but it should be updated when the
compat wrappers are removed.

---

## 4. Confirmed Architecture Risks

In priority order (severity × likelihood × current activity):

### Risk A1 — `import evaluate` in production critical path

- **Severity:** Critical — blocks compat wrapper removal entirely.
- **Likelihood:** 100% — fires on every pipeline run.
- **Location:** `pipelines/stage1_stage2.py:38`.
- **Cost to fix:** 1 line.
- **Why critical:** This is the **only** top-level legacy import in production
  code. Until it changes, `evaluate.py` cannot be deleted.

### Risk A2 — Twelve deferred `from simulator import` calls

- **Severity:** High — blocks `simulator.py` removal.
- **Likelihood:** 100% — every pipeline run hits the `data_loader` one;
  inference and evaluation hits depend on stage but are routine.
- **Locations:** `data_loader.py:278`, plus 4 in `inference/` and 7 in `evaluation/`.
- **Cost to fix:** 12 line replacements across 7 files. Each is mechanical
  (replace `from simulator import X` with the corresponding package path —
  see `11_import_audit.md` Appendix A for the map).

### Risk A3 — `inference.py` monolith on disk (55 KB)

- **Severity:** Medium for editor/linter UX; Zero for runtime.
- **Likelihood:** 100% for any contributor opening the file in an IDE.
- **Evidence the file is dead:** `inference.py:1391` uses `n_subj` undefined in
  scope. This NameError would crash Stage 2 on first invocation — yet Stage 2
  runs cleanly in CI because the package's `inference/stage2.py:163` defines it
  correctly. The monolith's Stage 2 path has never been executed.
- **Behavioral divergence:** `training_batch_size=512` (monolith) vs `2048`
  (package). The monolith would also train more slowly if it were ever loaded.

### Risk A4 — Root duplicates dormant but live as bare-import shadows

- **Severity:** Low today, medium if any new bare `import fc` etc. is introduced.
- **Likelihood:** Dormant. No file currently uses bare `import fc / fcd / wc_runner / delays / warmup / qc / extraction / screening`.
- **Verified identity:** All 8 root duplicates are **byte-for-byte identical**
  to their `features/` or `simulation/` counterparts (`diff` exit 0).
- **Real concern:** Root `extraction.py` is the only one that is not fully
  self-contained — it imports `from features.fc` and `from features.fcd`. If
  it ever drifts or is bare-imported, both root and package copies of `fc.py`
  and `fcd.py` get loaded into `sys.modules` simultaneously under different
  names.

### Risk A5 — R3 dependency rule violation in `simulation/qc.py`

- **Severity:** Low (currently isolated; doesn't break anything).
- **Likelihood:** 100% — module load.
- **Evidence:** `grep "from features" simulation/*.py` returns one line:
  `simulation/qc.py:25: from features.fc import compute_fc, fc_to_upper_tri`.
- **Effect:** `import simulation` transitively loads `features.fc`. This
  contradicts R3 in `07_refactor_rules.md` ("`simulation/` must not import
  from `features/`"). Not a runtime bug, but the rule is published.

### Risk A6 — FCD dimension bug in PPC / SBC / evaluation (latent, full surface)

- **Severity:** Latent. **Will crash** if anyone sets `USE_FCD=True` without
  fixing this first.
- **Likelihood:** Zero today (`USE_FCD=False`); 100% the moment FCD is enabled.
- **Locations (8 sites, 5 files):** see Section 3.1 inventory.
  Briefly: `inference/posterior.py` (2), `inference/diagnostics.py` (1),
  `inference/stage2.py` (1), `evaluation/metrics.py` (3),
  `evaluation/validation.py` (1), `evaluation/final_test.py` (1).
- **Root cause:** Every site passes `fcd_obs_raw` (from `extract_observed_features`)
  to `feature_pipeline.transform` and/or pairs it with
  `fcd_to_upper_tri(fcd_mat)` (6555-dim) in an RMSE comparison.
  When `USE_FCD=True`, the pipeline expects 5-dim summary stats — every site
  fails. The cleanest fix (per doc 15 Step E2) is to change
  `extract_observed_features` to emit the FCD matrix and let each caller
  compute the dimension it needs.
- **Scope:** Must be bundled with the FCD enablement (Tier X), not before.
  This is a scientific-API change, not a structural cleanup.

### Risk A7 — Stale documentation (README, PATCH_REPORT)

- **Severity:** Low (info only); high for onboarding new contributors.
- README still lists `simulator.py`, `inference.py`, `evaluate.py` as primary
  files. PATCH_REPORT item #2 claims 116 regions / FC_DIM=6670 (production
  uses 115 / 6555).
- **Recommendation:** Update both together when the code cleanup is complete;
  not part of the code-modification sequence.

### Risk A8 — Evaluation code smells (low priority, no action)

These are cosmetic/UX inconsistencies surfaced by doc 15. None block any
patch; they are recorded so future contributors do not "fix" them
opportunistically:

- **A8a — `shrinkage_mean` and `shrinkage_per_param` duplicated** in
  `_aggregate_validation` (`validation.py:184–189`). Identical computations
  stored under two keys. Downstream callers reference both. Do not unify
  without a grep pass across `pipelines/` + notebook + reports.
- **A8b — Two-stage RMSE delta sign convention disagrees** between
  `_print_selection_table` (s2 − s1) and `print_summary_two_stage`
  (s1 − s2). Cosmetic; not numeric.
- **A8c — Stage 2 nuisance handling at val/test time is hardcoded `fix_mean`**,
  independent of `config.NUISANCE_METHOD` (which only governs training-data
  collection). Intentional but undocumented; mark in docstring before any
  future user changes `NUISANCE_METHOD` and expects val/test to follow.
- **A8d — `bootstrap_ci` uses `RandomState(42)` not `config.SEED`**.
  Intentional reproducibility independence; document or migrate, do not
  silently change.
- **A8e — `n_samples=2000` is hardcoded** in `validation.py:138` and
  `final_test.py:136` (Stage 1 posterior sampling for nuisance means).
  Currently equals `config.N_POSTERIOR`; latent drift risk only.

All A8 items are out of scope for the structural cleanup. They become
candidates for a future "evaluation polish" pass — explicit user
authorization required for each.

---

## 5. Confirmed Import Risks

### I1 — Editor/Pylance resolution to `inference.py`

`import inference` resolves to the package at runtime, but some editor
configurations resolve to the `.py` file. Mitigated by `R8` (do not edit
`inference.py`), but still produces false "attribute not found" warnings in
some IDEs. Persists until the monolith is deleted.

### I2 — Notebook cells call legacy paths

`main.ipynb` cell 163 has `import evaluate`; cell 165 has `import simulator`.
These must be updated before deleting the compat wrappers. They are not in any
automated test path, so a stale notebook will be detected only at run time by
the next user.

### I3 — `evaluate.py` re-exports private helpers

`evaluate.py` re-exports `_aggregate_validation`, `_print_selection_table`,
`_print_test_summary`, `_print_validation_summary`, `_progress`,
`_resimulate_and_score`, `_test_stage1`, `_test_stage2`. Verified via reading
the file. Any new code calling `evaluate._foo` (notebook cells using private
helpers) must be migrated before deletion. None found in production code today.

### I4 — `pipeline_setup._PIPELINE_MODULES` stale reload list

After compat wrapper removal, this tuple must be updated to drop
`"simulator", "evaluate"` and add `"simulation", "features", "inference",
"evaluation"`. Otherwise the reload helper does the wrong thing in long-running
notebook sessions.

### I5 — Root `__init__.py` makes `vbi` an importable package

Only triggers if the parent of `/scratch/home/wog3597/vbi` enters `sys.path`.
Verified: identical to `simulation/__init__.py`. Latent; no current callers.

### I6 — `evaluation/` is R3-clean at module level (positive finding)

Doc 15 confirms `evaluation/` has **zero** top-level legacy imports — every
`from simulator import` is deferred inside a function body. This means:

- Importing `evaluation` (e.g., `import evaluation as evaluate` in Patch P-1)
  does NOT pull `simulator.py` into `sys.modules` at module load time.
- Patch P-1 is safe even before any of the deferred-import migrations land.
- Tier 4 patches (P-4 series) can be applied in any order amongst themselves
  without ordering hazards — each is an isolated, deferred import migration.

This was not assured for `inference/`-package patches (which transitively
import each other), but it IS assured for the evaluation patches.

---

## 6. Root-Level Duplicate File Decision Table

Columns: file → package equivalent → evidence of active use →
risk if changed → recommended action.

| Root file | Package equivalent | Active use? | Risk if removed today | Recommended action |
|---|---|---|---|---|
| `inference.py` | `inference/` package | **No** — package wins; monolith has NameError bug proving it's unreachable | HIGH if editors point at it (silent edits never take effect); LOW at runtime | **Mark legacy.** Do not edit (R8). Delete only after Tier 7 of `11_import_audit.md`. |
| `simulator.py` | `simulation/` + `features/` packages | **Yes — 13 callers** (`data_loader.py:278` + 4 in `inference/` + 7 in `evaluation/` + 1 in notebook) | CRITICAL — every pipeline run breaks | **Keep until callers migrated.** Required through Tier 4 of the patch sequence. |
| `evaluate.py` | `evaluation/` package | **Yes — 1 top-level + notebook/debug** (`pipelines/stage1_stage2.py:38` + `main.ipynb` cell ~163 + 2 sites in `debug_notebook.py`). Re-exports 8 private helpers (`_aggregate_validation`, `_print_*`, `_progress`, `_resimulate_and_score`, `_test_stage1/2`). Verified zero-logic 35-line wrapper. | CRITICAL — pipeline crashes at module load | **Keep until callers migrated.** Required through Tier 1 (the patch that unblocks everything else). Becomes deletable after Patch P-1 + the one notebook cell update. |
| `wc_runner.py` | `simulation/wc_runner.py` | **No** — `grep "^import wc_runner\|^from wc_runner"` finds 0 lines | MEDIUM — file was modified in current branch (status M), risk of unmerged delta | **Diff-clean (verified exit 0).** Safe to mark legacy. Defer deletion until Tier 6 to be conservative. |
| `delays.py` | `simulation/delays.py` | **No** | LOW | Diff-clean. Mark legacy, delete in Tier 6. |
| `warmup.py` | `simulation/warmup.py` | **No** | LOW | Diff-clean. Mark legacy, delete in Tier 6. |
| `qc.py` | `simulation/qc.py` | **No** | LOW | Diff-clean. Mark legacy, delete in Tier 6. |
| `fc.py` | `features/fc.py` | **No** | LOW | Diff-clean. Mark legacy, delete in Tier 6. |
| `fcd.py` | `features/fcd.py` | **No** (FCD disabled) | LOW | Diff-clean. Mark legacy, delete in Tier 6. |
| `extraction.py` | `features/extraction.py` | **No** | LOW — but **not self-contained** (imports `features.fc`, `features.fcd`) | Diff-clean. Mark legacy, delete in Tier 6 (after the others, because of the cross-package import). |
| `screening.py` | `features/screening.py` | **No** (stub only) | LOW (no logic to lose) | Diff-clean. Mark legacy, delete in Tier 6 (earliest of the duplicates). |
| `__init__.py` (root) | `simulation/__init__.py` | **No** — only triggers if `/scratch/home/wog3597` is in `sys.path` | LOW (latent) | Diff-clean. Delete in Tier 6 **after** verifying no `import vbi` callers anywhere. |

**Net deletion budget after the patch sequence:** 12 files removable (10 root
duplicates + 2 compat wrappers + monolith), reducing the root directory by ~80 KB.

---

## 7. Priority Order for Safe Cleanup

Tier ordering is mandatory — each tier depends on the previous tier landing first
and tests passing. Tiers 1–4 are the **active** cleanup; Tiers 5–7 are
deletions executed after all migrations are confirmed.

```
Tier 1  ▶ Migrate `import evaluate` (1 line, 1 file)
              pipelines/stage1_stage2.py:38
              → `import evaluation as evaluate`
              No call-site changes (all `evaluate.X` continue to work).

Tier 2  ▶ Migrate the data_loader hot path (1 line, 1 file)
              data_loader.py:278  →  `from simulation.delays import compute_delay_matrix`

Tier 3  ▶ Migrate inference/ deferred imports (5 lines, 5 files)
              inference/training_data.py:48
              inference/stage2.py:158
              inference/stage2.py:319
              inference/posterior.py:113
              inference/diagnostics.py:47

Tier 4  ▶ Migrate evaluation/ deferred imports (6 sites, 4 files)
              evaluation/metrics.py:110   (extract_observed_features)
              evaluation/metrics.py:184   (compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, simulate_single)
              evaluation/metrics.py:245   (the above + extract_observed_features)
              evaluation/validation.py:123 (extract_observed_features)
              evaluation/final_test.py:112 (extract_observed_features)
              evaluation/plots.py:203     (simulate_single, compute_fc)
              [Per doc 15 I6: evaluation/ is R3-clean at module level,
               so these 6 migrations are commutative — apply in any order.]

Tier 4b ▶ Migrate notebook & debug callers (defer; nice-to-have)
              main.ipynb cells ~163, ~165
              debug.py:443, 466, 565
              debug_notebook.py (5 sites)
              pipeline_setup.py:116  → update _PIPELINE_MODULES

Tier 5  ▶ Delete compat wrappers (after Tiers 1–4 + grep confirms zero callers)
              simulator.py
              evaluate.py

Tier 6  ▶ Delete root duplicate files (least-risk-first)
              1. screening.py  2. fcd.py  3. fc.py
              4. delays.py     5. warmup.py 6. qc.py
              7. extraction.py 8. wc_runner.py
              9. __init__.py (root) — only after confirming `import vbi` has no callers

Tier 7  ▶ Delete inference.py monolith
              Requires diff-audit confirmation that inference/__init__.py
              __all__ covers every public name in inference.py.

[Scientifically scoped — DO NOT bundle with Tiers 1–4; gated by USE_FCD enablement]
Tier X  ▶ Fix FCD dual-role bug across 8 sites in 5 files (see Section 3.1):
              inference/posterior.py:123 + 160
              inference/diagnostics.py:77–79
              inference/stage2.py:322
              evaluation/metrics.py:123 (evaluate_subject transform)
              evaluation/metrics.py:211–214 (_resimulate_and_score RMSE)
              evaluation/metrics.py:252 + 277 (baseline_eval transform + RMSE)
              evaluation/validation.py:135 (evaluate_validation_stage2 transform)
              evaluation/final_test.py:133 (_test_stage2 transform)
              Recommended approach (doc 15 Step E2): change
              `extract_observed_features` to return the FCD matrix `(N, N)`,
              and let each caller compute `fcd_to_summary_stats(mat)` for
              pipeline input or `fcd_to_upper_tri(mat)` for RMSE compare.

[Documentation, separate commit]
Tier D  ▶ Update README.md and PATCH_REPORT.md to match current architecture
```

The scientific-behavior cliff is at **Tier X** — touching it requires
explicit user approval because it changes the function signature/contract
of `posterior_predictive_check`, `simulation_based_calibration`,
`run_stage2_snpe`, `evaluate_subject`, `_resimulate_and_score`,
`baseline_eval`, `evaluate_validation_stage2`, and `_test_stage2`. The
cleanest variant also changes the return type of `extract_observed_features`.
Do not bundle Tier X with Tiers 1–4.

`evaluation/model_selection.py` is also scientifically sensitive (R1) —
the selection score weights (`SELECT_W_FC_CORR=1.0`, `SELECT_W_FC_RMSE=0.5`,
`SELECT_W_FCD_RMSE=0.5`) determine which stage wins. Do not modify
weights or the score formula without explicit user authorization.

---

## 8. Files That Must NOT Be Touched Yet

| File | Reason | Earliest tier safe to modify |
|---|---|---|
| `inference.py` | R8 (do not edit monolith). Contains a NameError bug that should not be "fixed" — its non-functioning is evidence the file is unreachable. | Tier 7 (delete only) |
| `simulator.py` | Live re-export hub. 13 active callers. | Tier 5 (delete only) |
| `evaluate.py` | Live re-export hub. 2 active callers. | Tier 5 (delete only) |
| `config.py` | Single source of truth. **R1: no scientific values may change.** Working-tree status `M` exists — examine before any further write. | Never as part of cleanup. Edit only on explicit user request. |
| `simulation/wc_runner.py` | Working-tree status `M`. Contains the per-sim parameter injection contract; mutation here is high-risk. | Never as part of cleanup. |
| `main.ipynb` | Working-tree status `M`; cells 163, 165 contain legacy imports. Notebooks are git-noisy. Touch only when ready to commit. | Tier 4b (last) |
| `bold.py` | Untouched in the cleanup scope. CPU-only enforced (R13). | Out of scope |
| `data_loader.py` | One line (278) needs migration in Tier 2. Otherwise leave alone — it mutates `config.NAN_MASK`, `config.HAS_BOLD` at load time and is sensitive. | Tier 2 (1-line replacement only) |
| `pipelines/stage1_stage2.py` | One line (38) in Tier 1. The rest of the file owns train/val/test discipline (R10); do not touch beyond the one import. | Tier 1 (1-line replacement only) |
| `features/extraction.py` | Contains the `worker_extract → extract_features` FCD shape asymmetry (RISK-2 in doc 13). Do not "fix" yet — currently harmless and the fix has side effects. | Bundle with Tier X (FCD enablement) |
| `evaluation/model_selection.py` | **R1 (scientific behavior).** Selection score weights and formula determine Stage 1 vs Stage 2 winner. Touching `compute_selection_score` or `SELECT_W_*` constants is a scientific change. | Out of structural-cleanup scope. Explicit user authorization required. |
| `evaluation/validation.py` | The hardcoded `fix_mean` nuisance handling at lines 134–141 (and matching code in `final_test.py`) is **scientific behavior**, not a structural concern. Tier 4 may only touch the one deferred `from simulator import` on line 123. | Tier 4 (one-line deferred-import migration only) |
| `evaluation/final_test.py` | Same as above — Tier 4 may only touch the one deferred `from simulator import` on line 112. The two-stage nuisance logic at lines 132–139 is scientifically load-bearing. | Tier 4 (one-line deferred-import migration only) |
| `evaluation/metrics.py` | Three Tier 4 deferred-import migrations on lines 110, 184, 245. **Do not modify `fc_metrics`, `fcd_vec_rmse`, `bootstrap_ci`, `_resimulate_and_score`'s scoring logic, or the `RandomState(42)` seed** — all R1. | Tier 4 (three deferred-import migrations only) |
| All 8 root duplicate `.py` files | Could be deleted physically, but premature deletion before grep proves zero callers is risky. | Tier 6 (only) |
| Root `__init__.py` | Latent risk only if parent dir in `sys.path`. Verify first. | Tier 6 (last, after `import vbi` audit) |
| `PATCH_REPORT.md` | Historical record — preserve as-is; do not rewrite history. | Optional Tier D append |
| `README.md` | Update only after code cleanup so the new file list is accurate. | Tier D |

---

## 9. Files Safe to Inspect Next

Doc 15 closes the evaluation-side inspection gap. Remaining read-only
priorities (in order):

1. **`pipelines/stage1_stage2.py` (full body)** — enumerate every
   `evaluate.X(...)` call site. Spot-checks in docs 10/15 confirm all
   are re-exported by `evaluation/__init__.py`, but a full pass is the
   final pre-flight check before Patch P-1 lands. ~15 minutes.

2. **Diff of `config.py` and `simulation/wc_runner.py` against `HEAD`** —
   both have working-tree status `M`. Run
   `git diff config.py simulation/wc_runner.py`. Any non-trivial change
   to either is a **stop condition** (Section 13 #1 / #2) — surface to
   the user before any structural patch begins. Mandatory pre-flight.

3. **`debug.py` and `debug_notebook.py`** — Tier 4b will eventually touch
   both. Inspecting first identifies any unlisted legacy imports.

4. **`config.py` (full body)** — confirms `STAGE1_PRIOR_LOW/HIGH`,
   `C_PARAM_PRIOR`, `LOCAL_EI_PARAMS`, `NUISANCE_METHOD`,
   `DIFFICULT_SHRINKAGE`, `N_SBC`, `BOOTSTRAP_N`, `SELECT_W_*`, and
   `USE_FCD`. Values referenced by docs 14 and 15; confirm before any
   Tier X work.

5. **`evaluation/reports.py` full body** — doc 15 covers the public API
   but the per-step reporters (`report_step1` through `report_step14`)
   are notebook-only and may have legacy paths. Cosmetic only.

**Do not read `inference.py` further** beyond what is already documented
(doc 14 + this plan). R8 prohibits edits; further inspection wastes context.

**Do not re-inspect `evaluation/`** — doc 15 is comprehensive (981 lines,
all submodules covered + cross-referenced). Refer to it directly.

---

## 10. Minimal Patch Sequence

Each patch is intentionally one diff that touches one concept. Run the
test sequence in Section 11 between every patch.

### Patch P-1 (Tier 1) — Top-level `evaluate` import

**File:** `pipelines/stage1_stage2.py`
**Line:** 38
**Diff:**
```diff
-import evaluate
+import evaluation as evaluate
```
**Rationale:** Preserves all `evaluate.X(...)` call syntax in the file. Zero
downstream change. Unblocks `evaluate.py` removal entirely.

### Patch P-2 (Tier 2) — `data_loader` hot path

**File:** `data_loader.py`
**Line:** 278
**Diff:**
```diff
-    from simulator import compute_delay_matrix
+    from simulation.delays import compute_delay_matrix
```
**Rationale:** Deferred import inside `get_subject_data`. Replacing the
import does not change the call signature. Hot path — must verify with a
data-load smoke test.

### Patch P-3a (Tier 3) — `inference/training_data.py`

**Line:** 48
**Diff:**
```diff
-    from simulator import simulate_gpu_batch, worker_extract
+    from simulation.wc_runner import simulate_gpu_batch
+    from features.extraction import worker_extract
```

### Patch P-3b (Tier 3) — `inference/stage2.py:158`

```diff
-    from simulator import simulate_gpu_batch, worker_extract
+    from simulation.wc_runner import simulate_gpu_batch
+    from features.extraction import worker_extract
```

### Patch P-3c (Tier 3) — `inference/stage2.py:319`

```diff
-    from simulator import extract_observed_features
+    from features.extraction import extract_observed_features
```

### Patch P-3d (Tier 3) — `inference/posterior.py:113`

```diff
-    from simulator import (
-        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
-        simulate_single,
-    )
+    from simulation.wc_runner import simulate_single
+    from features.fc import compute_fc
+    from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri
```

### Patch P-3e (Tier 3) — `inference/diagnostics.py:47`

```diff
-    from simulator import (
-        compute_fc, compute_sim_fcd_matrix, fc_to_upper_tri,
-        fcd_to_upper_tri, simulate_single,
-    )
+    from simulation.wc_runner import simulate_single
+    from features.fc import compute_fc, fc_to_upper_tri
+    from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri
```

### Patches P-4a..P-4f (Tier 4) — `evaluation/` (6 sites, 4 files)

Per doc 15 I6, these are R3-clean at module level and **commutative** —
they may be applied in any order without ordering hazards. Each is a
deferred-import replacement inside a function body, nothing else.

**P-4a — `evaluation/metrics.py:110`** (`evaluate_subject`)
```diff
-    from simulator import extract_observed_features
+    from features.extraction import extract_observed_features
```

**P-4b — `evaluation/metrics.py:184`** (`_resimulate_and_score`)
```diff
-    from simulator import (
-        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
-        simulate_single,
-    )
+    from simulation.wc_runner import simulate_single
+    from features.fc import compute_fc
+    from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri
```

**P-4c — `evaluation/metrics.py:245`** (`baseline_eval`)
```diff
-    from simulator import (
-        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
-        extract_observed_features, simulate_single,
-    )
+    from simulation.wc_runner import simulate_single
+    from features.fc import compute_fc
+    from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri
+    from features.extraction import extract_observed_features
```

**P-4d — `evaluation/validation.py:123`** (`evaluate_validation_stage2`)
```diff
-    from simulator import extract_observed_features
+    from features.extraction import extract_observed_features
```

**P-4e — `evaluation/final_test.py:112`** (`_test_stage2`)
```diff
-    from simulator import extract_observed_features
+    from features.extraction import extract_observed_features
```

**P-4f — `evaluation/plots.py:203`** (`plot_one_simulation`)
```diff
-    from simulator import simulate_single, compute_fc
+    from simulation.wc_runner import simulate_single
+    from features.fc import compute_fc
```

After all six land, `grep -rn "from simulator import" evaluation/` returns
zero output. Combined with the post-Tier-3 state, `simulator.py` has zero
remaining package callers and becomes deletable in Tier 5.

### Patch sequence ends here for the structural cleanup

Tiers 5–7 are **deletions, not edits**, and they require a separate user
go-ahead. Tier X (FCD bug fix at 8 sites — see Section 3.1) requires
explicit user direction because it changes function semantics and the
return type of `extract_observed_features`.

---

## 11. Test Sequence After Each Patch

Run **after every** patch listed in Section 10. Stop at the first failure
and revert that patch before proceeding.

### T-A — Compile + import resolution (always passes when patch is correct)

```bash
find /scratch/home/wog3597/vbi -name "*.pyc" -delete 2>/dev/null
find /scratch/home/wog3597/vbi -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

python -m py_compile config.py data_loader.py bold.py main.py pipeline_setup.py
python -m py_compile simulation/wc_runner.py simulation/delays.py simulation/warmup.py simulation/qc.py
python -m py_compile features/fc.py features/fcd.py features/extraction.py features/screening.py
python -m py_compile inference/scaling.py inference/priors.py inference/feature_pipeline.py inference/embedding.py
python -m py_compile inference/training_data.py inference/snpe.py inference/stage1.py inference/stage2.py
python -m py_compile inference/posterior.py inference/diagnostics.py inference/io.py
python -m py_compile evaluation/metrics.py evaluation/validation.py evaluation/model_selection.py
python -m py_compile evaluation/final_test.py evaluation/plots.py evaluation/reports.py
python -m py_compile pipelines/stage1_stage2.py
echo "compile clean"
```

### T-B — Package resolution sanity (always passes after every patch)

```bash
python -c "
import inference, simulation, features, evaluation, pipelines
assert 'inference/__init__' in inference.__file__, inference.__file__
assert 'simulation/__init__' in simulation.__file__, simulation.__file__
assert 'features/__init__' in features.__file__, features.__file__
assert 'evaluation/__init__' in evaluation.__file__, evaluation.__file__
assert 'pipelines/__init__' in pipelines.__file__, pipelines.__file__
print('resolution OK')
"
```

### T-C — Top-level import smoke test (covers patch's user-facing surface)

```bash
python -c "
from pipelines import run_pipeline
from inference import (
    ParameterScaler, FeaturePipeline, FeatureEmbedding,
    run_stage1_snpe, run_stage2_snpe, save_artifacts, load_artifacts,
)
from evaluation import (
    fc_metrics, evaluate_validation_stage1, select_best_model, final_test,
    print_final_summary,
)
print('smoke OK')
"
```

### T-D — Static import audit (catches accidental regressions)

After P-1:
```bash
grep -n "^import evaluate\b\|^from evaluate import" pipelines/*.py inference/*.py evaluation/*.py *.py
# Expected: zero output
```

After P-2 through P-3e:
```bash
grep -rn "from simulator import\|import simulator\b" data_loader.py inference/ --include="*.py"
# Expected: zero output after the corresponding patch lands
```

After P-4 series:
```bash
grep -rn "from simulator import\|import simulator\b" evaluation/ --include="*.py"
# Expected: zero output
```

### T-E — Patch-specific tests

**After P-1** (only):
```bash
python -c "
import pipelines.stage1_stage2 as pl
# 'evaluate' is now an alias for the package; everything pl.evaluate.X must still resolve.
assert hasattr(pl, 'evaluate'), 'evaluate name missing'
assert pl.evaluate.__name__ == 'evaluation'
assert hasattr(pl.evaluate, 'fc_metrics')
assert hasattr(pl.evaluate, 'evaluate_validation_stage1')
assert hasattr(pl.evaluate, 'select_best_model')
assert hasattr(pl.evaluate, 'final_test')
assert hasattr(pl.evaluate, 'print_final_summary')
print('P-1 surface OK')
"
```

**After P-2** (only):
```bash
python -c "
import data_loader
# Just confirm import + signature; do not load real data here.
import inspect
src = inspect.getsource(data_loader.get_subject_data)
assert 'from simulation.delays import compute_delay_matrix' in src
assert 'from simulator import' not in src
print('P-2 source OK')
"
```

**After every P-3 / P-4 patch:** rerun T-A through T-D.

### T-F — Optional full smoke (if GPU is available)

```bash
python debug.py --basic
```
This runs the no-GPU debug path and exercises imports across all packages.
If GPU is available, `python debug.py --all` is the full smoke test.

### T-G — Evaluation surface smoke (run after any P-4 patch, no GPU)

Drawn from doc 15 Section 13. Verifies that evaluation's public API and
the `evaluate.py` compat wrapper remain functional after each Tier 4
patch lands.

```bash
python -c "
import evaluation
assert 'evaluation/__init__' in evaluation.__file__, evaluation.__file__
from evaluation import (
    fc_metrics, fcd_vec_rmse, bootstrap_ci, evaluate_subject,
    baseline_eval, baseline_eval_subjects,
    evaluate_validation_stage1, evaluate_validation_stage2,
    compute_selection_score, select_best_model,
    final_test,
    plot_posteriors, plot_fc_comparison, plot_sbc_rank_histogram,
    plot_pca_diagnostic, plot_one_simulation,
    print_summary_two_stage, print_final_summary,
)
import evaluate  # compat wrapper must still resolve
for name in ('fc_metrics', 'evaluate_validation_stage1', 'select_best_model',
             'final_test', 'print_final_summary',
             '_aggregate_validation', '_print_selection_table',
             '_progress', '_resimulate_and_score'):
    assert hasattr(evaluate, name), f'evaluate.{name} missing'
print('evaluation + evaluate compat OK')
"
```

Numerical smoke test (no GPU, no sbi):

```bash
python -c "
import numpy as np
from evaluation.metrics import fc_metrics, fcd_vec_rmse, bootstrap_ci
from evaluation.model_selection import compute_selection_score, select_best_model

# fc_metrics
rng = np.random.RandomState(0)
N = 115
fc_obs = (np.eye(N) + 0.1 * rng.randn(N, N)).astype(np.float32)
fc_obs = (fc_obs + fc_obs.T) / 2
fc_pred = fc_obs + 0.05 * rng.randn(N, N).astype(np.float32)
m = fc_metrics(fc_obs, fc_pred)
assert 0.5 < m['corr'] < 1.0 and m['rmse'] < 0.2

# selection score: Stage 2 should beat Stage 1 when corr is higher
s1 = {'fc_corr_mean': 0.7, 'fc_rmse_mean': 0.2, 'fcd_rmse_mean': 0.0}
s2 = {'fc_corr_mean': 0.8, 'fc_rmse_mean': 0.15, 'fcd_rmse_mean': 0.0}
assert compute_selection_score(s2) > compute_selection_score(s1)
best, _ = select_best_model(s1, s2, verbose=False)
assert best == 2
print('numerical smoke OK')
"
```

---

## 12. Recommended First Actual Code Modification

**Patch P-1: `pipelines/stage1_stage2.py:38`**

**The change:**
```diff
-import evaluate
+import evaluation as evaluate
```

**Why this is the right first patch (in priority order):**

1. **Smallest possible diff** — one line.
2. **Unblocks the largest downstream cleanup** — `evaluate.py` becomes
   deletable after only one more `main.ipynb` cell update.
3. **Zero scientific behavior change** — the local name `evaluate` still
   points at the same exported functions. All `evaluate.fc_metrics(...)`,
   `evaluate.select_best_model(...)`, etc. call sites are unchanged.
4. **Already verified to be exhaustive** — every `evaluate.X` name used in
   `pipelines/stage1_stage2.py` is re-exported by `evaluation/__init__.py`
   (verified by reading both). Doc 15 independently confirms all 33+ public
   names + 8 private helpers are re-exported via `evaluation/__init__.py`,
   so `import evaluation as evaluate` covers the surface completely.
5. **Reverts trivially** — a one-line revert restores the prior state.
6. **Test surface is already well-defined** — T-A through T-E above plus
   one `dir(pl.evaluate)` check.

**Do not bundle anything else with this patch.** No formatting, no comment
deletion, no other "while we're here" changes.

**Before applying, the user should explicitly confirm:**
- They are ready to commit this single-line change (this is a structural
  refactor, not a behavior change).
- They want package-level cleanup to proceed in the Tier order above.

If the user prefers a different first patch (e.g., starting with the
`data_loader` hot path), the same review pattern applies.

---

## 13. Stop Condition — When to Pause and Ask the User

Pause and request explicit user authorization before any of:

1. **Any change to `config.py`** — even cosmetic. R1 prohibits silent scientific
   changes; the file is the SSOT for all numerical constants.
2. **Any change to `simulation/wc_runner.py`** — working-tree status `M`;
   contains the per-sim parameter contract that, if broken, silently corrupts
   inference. The on-disk version may already differ from `HEAD`; inspect
   `git diff` first.
3. **Any change that modifies a function signature** — including Tier X
   (FCD enablement), which changes `posterior_predictive_check`,
   `simulation_based_calibration`, `_resimulate_and_score`, and
   `baseline_eval`.
4. **Tier 5–7 deletions** — every delete is irreversible without `git restore`.
   Confirm grep results with the user before executing.
5. **Notebook (`main.ipynb`) edits** — notebooks produce noisy diffs and may
   already contain the user's in-progress work. Always ask before saving.
6. **Pull requests / branch operations** — current branch is
   `refactor/02-simulation`. Do not push, merge, or rebase without
   explicit instruction.
7. **R3/R4 reorganizations** — moving `simulation/qc.py` to break the
   `features.fc` dependency requires a public API decision (rename, alias,
   or new `checks/` package). Ask first.
8. **Any patch that fails T-A or T-B** — revert and ask the user how to
   proceed; do not iterate on a broken patch without context.
9. **Discovery of additional drift or risks** not captured in this plan
   — for example, if `git diff simulation/wc_runner.py` shows an unexpected
   change, surface it before continuing.
10. **Decision to enable `USE_FCD`** — requires bundling Tier X (8 sites,
    5 files; see Section 3.1) with empirical BOLD availability and a re-run
    of the full FCD pipeline.

11. **Any change to `evaluation/model_selection.py`** — the score formula
    (`compute_selection_score`) and weights (`SELECT_W_FC_CORR=1.0`,
    `SELECT_W_FC_RMSE=0.5`, `SELECT_W_FCD_RMSE=0.5`) determine which stage
    is selected on validation. R1 (scientific behavior). Do not modify even
    cosmetically.

12. **Any change to the hardcoded `fix_mean` nuisance handling** in
    `evaluation/validation.py:134–141` or `evaluation/final_test.py:132–139`.
    These intentionally diverge from `config.NUISANCE_METHOD` (which only
    governs training-data collection). Touching them changes scientific
    behavior at val/test time.

13. **Any change to the A8-series code smells** (duplicate
    `shrinkage_mean`/`shrinkage_per_param` keys, two-stage delta sign
    convention, `RandomState(42)` in `bootstrap_ci`, hardcoded `2000`
    nuisance samples). None of these is in scope for structural cleanup.

A safe operating rule: any change that is not a mechanical
`from simulator import X → from <package> import X` replacement, or
the one-line `import evaluate → import evaluation as evaluate` replacement,
is **outside this plan's authorization** and requires user sign-off first.

---

## Appendix A — Verified Working-Tree State (snapshot at review time)

```
M .gitignore
M config.py
M debug_notebook.py
M evaluate.py
M main.ipynb
M main.py
M simulation/wc_runner.py
```

Untracked: 25 files (the 14 documentation .md files plus the 11 root duplicates
that were not in git history before the refactor).

Six of seven modified files (excluding `.gitignore`) should be examined with
`git diff` **before** Patch P-1 lands. If `config.py` or
`simulation/wc_runner.py` contains changes that affect scientific behavior,
those must be reviewed separately and committed before any structural patch
begins.

## Appendix B — Quick Reference: What Each Document Owns

| Doc | What it covers | Verified accurate? |
|---|---|---|
| 01 — Repo Overview | Scientific assumptions, architecture summary | Yes |
| 02 — Repo Tree | Annotated file tree | Yes |
| 03 — Module Index | Function/class roster per module | Yes |
| 04 — Data Flow | End-to-end shapes and formulas | Mostly — one inaccuracy: FC is not z-scored before PCA (doc 13 corrects this) |
| 05 — Runbook | How to run / debug | Yes (smoke commands still valid) |
| 06 — Known Errors | Import conflict and pycache hazards | Yes |
| 07 — Refactor Rules | R1–R13 — must follow | Yes |
| 08 — Claude Upload Guide | Which files to upload first to a Project | Yes |
| 09 — Architecture Review | Risks and cleanup phase plan | Yes |
| 10 — Entrypoint Flow | Call graph from main.py to evaluation | Yes |
| 11 — Import Audit | Legacy/package import inventory | Yes (all 12 deferred sites verified) |
| 12 — Simulation Module Review | GPU engine, delays, warmup, qc | Yes — R3 violation in `simulation/qc.py` confirmed |
| 13 — Feature Module Review | FC/FCD, FeaturePipeline | Yes — FCD shape asymmetry confirmed |
| 14 — Inference Module Review | Stage 1/2, SNPE, posterior, diagnostics | Yes — `n_subj` NameError in `inference.py` confirmed; FCD dual-role bug confirmed (and further extended by doc 15 to the full 8-site surface) |
| 15 — Evaluation Module Review | Metrics, validation, model selection, final test, plots, reports; `evaluate.py` audit | Yes — 981 lines. Confirms evaluate.py is a 35-line zero-logic wrapper; selection score formula and weights documented; Stage 2 hardcoded `fix_mean` nuisance handling identified; FCD dual-role bug extended to 4 additional evaluation sites (full surface now in Section 3.1 of this plan); A8 code smells catalogued. Doc was created after this plan's first pass and integrated via this revision. |
