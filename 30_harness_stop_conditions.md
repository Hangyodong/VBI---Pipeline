# 30 — Claude Code Harness: Stop Conditions

Part of the Claude Code Harness for VBI-SBI pipeline.
Generated: 2026-05-18  Branch: refactor/02-simulation
Repo: /scratch/home/wog3597/vbi

## HOW TO USE THIS FILE

This is the **brake system**. When any condition below is detected,
halt immediately and use the §5.6 STOP template. **Make no changes
to the working tree until the user authorizes resumption.**

The format of each condition:

```
🛑 STOP CONDITION SC-N
Trigger:         what triggers it
Risk:            what breaks if we continue
Required action: what to say to the user
Resume on:      exactly what the user must say
```

The conditions are organized into five families. Use the §5.7
decision tree as a quick pre-action gate.

---

## 5.1 CRITICAL STOPS (stop before ANY action)

### 🛑 SC-1 — `config.py` edit proposed

- **Trigger**: any plan or diff that touches `config.py`.
- **Risk**: R1 (scientific behavior is frozen). A silent change to
  `WC_FIXED`, prior bounds, `SEED`, `ANALYSIS_BOLD_T`, `PCA_DIM_FC`,
  or any other numerical constant invalidates all prior pipeline
  runs and breaks reproducibility.
- **Required action**: surface the proposed edit via §5.6 template.
  Quote the exact line(s) and constant(s) that would change.
- **Resume on**: `Proceed with config.py change of <KEY>:<old>→<new>`.

### 🛑 SC-2 — `inference.py` edit proposed

- **Trigger**: any read-write operation targeting `inference.py`.
- **Risk**: R8 (dead monolith, package wins at runtime). Edits there
  are silently ineffective and confuse future readers. The file
  contains a known `NameError` (`n_subj` in `collect_stage2_data`)
  that proves it is unreachable — DO NOT "fix" it.
- **Required action**: surface via §5.6. Recommend Tier 7 deletion
  via the proper protocol instead.
- **Resume on**: never — only deletion authorized.

### 🛑 SC-3 — `evaluation/model_selection.py` scoring change

- **Trigger**: a change to `SELECT_W_FC_CORR=1.0`, `SELECT_W_FC_RMSE=0.5`,
  `SELECT_W_FCD_RMSE=0.5`, or to `compute_selection_score`.
- **Risk**: R1 (scoring weights determine Stage 1 vs Stage 2 winner
  on validation). A weight tweak changes the production model.
- **Required action**: surface via §5.6.
- **Resume on**: `Proceed with selection-score change <details>`.

### 🛑 SC-4 — `evaluation/validation.py` / `final_test.py` fix_mean

- **Trigger**: a change to the hardcoded `nuisance_method="fix_mean"`
  block in `evaluation/validation.py:134-141` or
  `evaluation/final_test.py:132-139`.
- **Risk**: These deliberately diverge from `config.NUISANCE_METHOD`
  (which only governs training-data collection). Touching them is a
  scientific behavior change.
- **Required action**: surface via §5.6.
- **Resume on**: `Proceed with nuisance-handling change`.

### 🛑 SC-5 — root duplicate logic-change proposed

- **Trigger**: edit (not delete) of any of `fc.py`, `fcd.py`,
  `wc_runner.py`, `delays.py`, `warmup.py`, `qc.py`, `extraction.py`,
  `screening.py` at the repo root.
- **Risk**: All 8 are byte-identical to their package counterparts
  today. An edit creates drift between the root copy and the
  package — a latent bug-shadow.
- **Required action**: surface via §5.6.
- **Resume on**: never edit; `Proceed with Tier 6 archive of <names>`
  instead.

### 🛑 SC-6 — `bold.py` edit proposed

- **Trigger**: any edit to `bold.py`.
- **Risk**: Sole source of `BoldMonitor`, imported by
  `simulation/wc_runner.py` and `simulation/warmup.py`. Any
  behavioral change to `BoldMonitor.step()` invalidates all BOLD
  output (HC-2, INV-1).
- **Required action**: surface via §5.6.
- **Resume on**: `Proceed with bold.py change <details>`.

### 🛑 SC-7 — VBI / sbi / torch package source edit proposed

- **Trigger**: target path under
  `…/site-packages/vbi/…`, `…/site-packages/sbi/…`, `…/torch/…`.
- **Risk**: Breaks pip-reinstall reproducibility; user upgrade will
  silently discard the patch.
- **Required action**: surface via §5.6. Recommend monkey-patching
  at our call site instead.
- **Resume on**: never edit installed packages directly.

### 🛑 SC-8 — no explicit "Proceed" authorization

- **Trigger**: a code-modifying action is about to run but the user
  has not said exactly "Proceed" / "Apply" / "Go ahead" for this
  specific task.
- **Risk**: Auto-execution of a plan the user was still reviewing.
- **Required action**: present the plan and wait.
- **Resume on**: `Proceed with <task>` (verbatim).

### 🛑 SC-9 — `simulation/wc_runner.py` has un-discussed unstaged changes

- **Trigger**: `git status` shows `simulation/wc_runner.py` with
  `M` flag and the diff contents have not been reviewed in the
  current session.
- **Risk**: An in-progress edit by the user gets overwritten.
- **Required action**: `git diff simulation/wc_runner.py` and
  surface to user.
- **Resume on**: `These are mine, proceed` or `Stash these first`.

### 🛑 SC-10 — `(n_nodes, csz)` param shape contract violation

- **Trigger**: a change to `_try_per_sim_params` or the
  `for name in param_names: assert v.shape == (n_nodes_chk, csz)`
  block in `simulate_gpu_batch`.
- **Risk**: Silently feeds wrong-shape params to VBI WC — either
  raises mid-chunk or (worse) corrupts theta↔BOLD alignment
  (INV-2).
- **Required action**: surface via §5.6.
- **Resume on**: `Proceed with param-shape change <details>`.

---

## 5.2 COMPILE / IMPORT STOPS (stop immediately on failure)

### 🛑 SC-11 — `python -m compileall -q .` returns non-zero

- **Trigger**: file 29 §4.1 T-0a fails after a code edit.
- **Risk**: Syntax error left in the tree — every subsequent test
  will fail and the user's notebook cells will fail too.
- **Required action**: identify which file, roll back the edit,
  re-run T-0a until clean, surface via §5.6.
- **Resume on**: clean compile (`exit: 0`).

### 🛑 SC-12 — package import fails

- **Trigger**: T-0b reports `ImportError` for any of `config`,
  `data_loader`, `simulation`, `features`, `inference`,
  `evaluation`, `pipelines`.
- **Risk**: A package can no longer be loaded — the pipeline is
  broken.
- **Required action**: identify the broken import path, roll back.
- **Resume on**: all 5 packages import clean.

### 🛑 SC-13 — notebook cell fails `compile(src, ...)`

- **Trigger**: T-3b reports a `SyntaxError` on any code cell.
- **Risk**: That cell will not execute in Jupyter.
- **Required action**: roll back the affected cell from its
  `.bak_<timestamp>` backup. Report the cell index + error to the
  user.
- **Resume on**: all code cells compile clean.

### 🛑 SC-14 — P-1 invariant fails

- **Trigger**: T-0c reports `P-1 FAIL`.
- **Risk**: `pipelines/stage1_stage2.py:38` was reverted; the
  pipeline still calls `evaluate.X(...)` but the binding no longer
  resolves through the `evaluation` package.
- **Required action**: re-apply P-1 per file 28 §3.1 protocol or
  restore via `git checkout -- pipelines/stage1_stage2.py`.
- **Resume on**: `P-1 OK`.

### 🛑 SC-15 — P-2 invariant fails

- **Trigger**: T-0d reports `P-2 FAIL`.
- **Risk**: `data_loader.py:278` reverted; legacy `simulator` path
  re-introduced; `simulator.py` deletion would now break the hot
  path.
- **Required action**: re-apply P-2 or restore via git.
- **Resume on**: `P-2 OK`.

### 🛑 SC-16 — GPU-1 invariants fail

- **Trigger**: T-0e reports missing `_alloc_stride_buffers` /
  `_trim_memory_pool`, OR `import cupy` appears at module top.
- **Risk**: The GPU optimization was unwound, OR the lazy-import
  pattern was broken (module no longer importable without CUDA).
- **Required action**: restore from the doc-driven GPU-1 patch or
  `git checkout -- simulation/wc_runner.py`.
- **Resume on**: `GPU-1 OK`.

### 🛑 SC-17 — `run_theta_specific_check` returns `pass=False`

- **Trigger**: T-2b reports `pass: False` or `diff > atol`.
- **Risk**: Scientific output drift detected. For bit-identical
  optimizations (GPU-1), even `diff > 0.0` is a regression.
- **Required action**: roll back the simulation change.
  Re-run T-2b after rollback to confirm clean baseline restored.
- **Resume on**: `pass: True` with diff at the expected level.

---

## 5.3 CONTENT STOPS (stop when discovered during reading)

### 🛑 SC-18 — uncommitted changes in `config.py` to scientific values

- **Trigger**: `git diff config.py` shows changes touching
  `WC_FIXED`, `STAGE1_PRIOR_*`, `C_PARAM_PRIOR`, `LOCAL_EI_PARAMS`,
  `SEED`, `DT`, `T_END`, `T_CUT`, `ANALYSIS_BOLD_T`, `PCA_DIM_*`,
  `EMBED_*`, `NDE_*`, `N_SIM*`, `N_POSTERIOR`, `N_SBC`,
  `N_TEST_RESIM`, `SELECT_W_*`, `NUISANCE_METHOD`, or `BW`.
- **Risk**: A scientific drift is in flight; downstream work would
  bake in the drift unwittingly.
- **Required action**: surface the diff to the user before any
  edits. Ask: "Are these intentional? Commit first?"
- **Resume on**: `These are mine, proceed` / `Discard, proceed`.

### 🛑 SC-19 — FC z-score logic discovered in a new path

- **Trigger**: a planned edit introduces FC z-scoring before PCA.
- **Risk**: FCPCAScaler is designed for raw Pearson r ∈ [-1, 1].
  Z-scoring distorts the variance structure.
- **Required action**: surface via §5.6. The fix is usually to
  remove the z-score, not to add one.
- **Resume on**: confirmation that FC stays raw.

### 🛑 SC-20 — module-scope deferred-library import

- **Trigger**: a proposed file has `import cupy` / `import torch` /
  `import sbi` / `import vbi` at module top.
- **Risk**: Breaks no-GPU / no-torch import. The harness invariant
  is that every package module imports cleanly without those
  dependencies present.
- **Required action**: surface via §5.6. Recommend deferring the
  import inside the function that needs it.
- **Resume on**: imports moved inside function bodies.

### 🛑 SC-21 — `BoldMonitor(..., xp=cp)` discovered

- **Trigger**: a `BoldMonitor` instantiation with `xp != np`.
- **Risk**: NVRTC compilation failure on cupy's HRF kernel
  (INV-1).
- **Required action**: surface via §5.6.
- **Resume on**: `xp=np` restored.

### 🛑 SC-22 — `FeaturePipeline.fit()` on val or test

- **Trigger**: a `.fit(...)` call inside a function that operates
  on validation or test data.
- **Risk**: R10 — fit-on-train-only contract broken. Causes data
  leakage and invalidates SBI calibration.
- **Required action**: surface via §5.6.
- **Resume on**: fit restricted to training data.

### 🛑 SC-23 — new `config` key proposed

- **Trigger**: any patch that adds a new attribute to `config.py`.
- **Risk**: Breaks reproducibility against doc-recorded constants;
  all session HC-4 rules forbid new keys.
- **Required action**: surface via §5.6. Suggest passing a kwarg
  with a default that preserves prior behavior instead.
- **Resume on**: explicit `Proceed with new config key <NAME>`.

### 🛑 SC-24 — new third-party dependency

- **Trigger**: a new top-level `import <package>` not already used
  in the codebase (excluding the deferred set: cupy/torch/sbi/vbi).
- **Risk**: Breaks reproducibility; HC-4.
- **Required action**: surface via §5.6.
- **Resume on**: `Proceed with adding dependency <name>`.

### 🛑 SC-25 — circular-import risk

- **Trigger**: a planned `from X import Y` that, combined with an
  existing import chain in `Y`'s module, would form a cycle.
  Verify against R3 (file 27 §2.3) and R4 cycles
  (`delays↔wc_runner`, `snpe↔stage1`).
- **Risk**: Import-time crash at module load.
- **Required action**: surface via §5.6. Suggest extracting shared
  code into `inference/_utils.py` or similar.
- **Resume on**: revised plan that avoids the cycle.

---

## 5.4 SCIENTIFIC STOPS (stop when scientific behavior would change)

### 🛑 SC-26 — `STAGE1_PRIOR_LOW` / `STAGE1_PRIOR_HIGH` change

- **Trigger**: any edit to either list in `config.py`.
- **Risk**: Prior bounds define the SBI inference space; any
  change requires re-running all simulations.
- **Required action**: surface via §5.6 (overlaps with SC-1).
- **Resume on**: explicit user authorization.

### 🛑 SC-27 — `SEED` change

- **Trigger**: any edit to `config.SEED`.
- **Risk**: SBI training, theta sampling, and SBC ranks all depend
  on a stable seed. Changes invalidate prior results.
- **Required action**: surface via §5.6.
- **Resume on**: explicit user authorization.

### 🛑 SC-28 — `ANALYSIS_BOLD_T` hardcoded

- **Trigger**: a constant `240` introduced where the original code
  computes `int((T_END - T_CUT) / (DT*DECIMATE) / (TR_SEC*1000/(DT*DECIMATE)))`.
- **Risk**: Magic number; breaks if `T_END`/`T_CUT` ever change.
  R12 explicit.
- **Required action**: surface via §5.6.
- **Resume on**: reference `config.ANALYSIS_BOLD_T` instead.

### 🛑 SC-29 — `PCA_DIM_FC` or `EMBED_DIM` change

- **Trigger**: edit to `config.PCA_DIM_FC = 300` or
  `config.EMBED_DIM = 128`.
- **Risk**: Changes feature-space size; invalidates saved
  artifacts; potentially changes scientific result.
- **Required action**: surface via §5.6.
- **Resume on**: explicit authorization.

### 🛑 SC-30 — `SELECT_W_*` change

- **Trigger**: edit to any of `SELECT_W_FC_CORR=1.0`,
  `SELECT_W_FC_RMSE=0.5`, `SELECT_W_FCD_RMSE=0.5`.
- **Risk**: Changes which Stage wins on validation; R1 scientific
  behavior.
- **Required action**: surface via §5.6 (overlaps with SC-3).
- **Resume on**: explicit authorization.

---

## 5.5 AUTHORIZATION STOPS (stop until user confirms)

### 🛑 SC-31 — file deletion proposed

- **Trigger**: any `rm`, `git rm`, or `os.remove` against a
  versioned file. Includes the 8 root duplicates,
  `simulator.py`, `evaluate.py`, `inference.py`.
- **Risk**: Irreversible without `git restore`. If file is
  untracked, irreversible without backup.
- **Required action**: surface via §5.6. List exact files and
  byte sizes.
- **Resume on**: `Proceed with delete of <exact filename(s)>`.

### 🛑 SC-32 — function rename proposed

- **Trigger**: any change to a public function name (one in a
  `__init__.py`'s `__all__` or referenced by other modules).
- **Risk**: HC-2 / R6 — breaks every caller.
- **Required action**: surface via §5.6.
- **Resume on**: `Proceed with rename of <X> to <Y>`.

### 🛑 SC-33 — public API signature change

- **Trigger**: adding a positional argument without a default, or
  removing an argument. (Adding a kwarg with default that
  preserves prior behavior is allowed under HC-2.)
- **Risk**: breaks every caller.
- **Required action**: surface via §5.6.
- **Resume on**: `Proceed with signature change <details>`.

### 🛑 SC-34 — branch operation (push, merge, rebase)

- **Trigger**: any `git push`, `git merge`, `git rebase`,
  `git reset --hard`, `git branch -D`.
- **Risk**: Force-push to `main` rewrites history; rebase to
  `main` mid-task loses in-flight work.
- **Required action**: surface via §5.6 with the exact command.
- **Resume on**: `Proceed with <exact command>`.

### 🛑 SC-35 — Tier X FCD work touched

- **Trigger**: an edit proposed to any of the 8 FCD dual-role
  sites (`inference/posterior.py:123,160`,
  `inference/diagnostics.py:77`, `inference/stage2.py:322`,
  `evaluation/metrics.py:123,211,252`,
  `evaluation/validation.py:135`, `evaluation/final_test.py:133`),
  OR a change to `extract_observed_features`.
- **Risk**: Tier X must migrate all 8 sites + change
  `extract_observed_features` return type, **as one bundle**.
  Partial fixes leave latent shape failures.
- **Required action**: surface via §5.6. Recommend a
  user-driven planning round.
- **Resume on**: `Proceed with Tier X — full 8-site bundle`.

### 🛑 SC-36 — >5 files modified in one task

- **Trigger**: a single task's diff would touch >5 source files
  (excluding docs and backups).
- **Risk**: Beyond cognitive load; harder to verify; rollback
  becomes painful.
- **Required action**: surface via §5.6. Suggest splitting into
  smaller tasks, each with its own verify cycle.
- **Resume on**: `Proceed even though it's <N> files` (with
  acknowledgment) OR a revised smaller plan.

---

## 5.6 STOP MESSAGE TEMPLATE

When any stop condition triggers, emit **exactly** this message
structure (Markdown is acceptable; the box drawing is illustrative):

```
🛑 STOP — SC-<N> triggered

Condition: <name>
Detected:  <what exactly was found>
Risk:      <what would break if we continued>

I have made NO changes. Please confirm one of:
  A) Proceed anyway — <specific authorization phrasing for SC-N>
  B) Skip this change and continue with the rest
  C) Abort the task entirely
```

Worked example for SC-1:

```
🛑 STOP — SC-1 triggered

Condition: config.py edit proposed
Detected:  Plan step 3 changes config.STAGE1_PRIOR_HIGH from
           [2.5, 2.0, 1.5, 1.5] to [3.0, 2.0, 1.5, 1.5]
Risk:      R1 — changes the SBI prior for the `P` parameter. All
           saved Stage 1 posteriors become invalid.

I have made NO changes. Please confirm one of:
  A) Proceed with config.py change of STAGE1_PRIOR_HIGH[0]: 2.5 → 3.0
  B) Skip this change and continue with the rest
  C) Abort the task entirely
```

---

## 5.7 QUICK-REFERENCE STOP DECISION TREE

Run this gate **before** taking any action that modifies the
working tree.

```
START
  │
  ▼
Is config.py / inference.py / bold.py in the edit target?
  │                            └── YES ── ► 🛑 SC-1 / SC-2 / SC-6
  │ NO
  ▼
Is a root duplicate, evaluate.py, simulator.py, or root __init__.py
the target for a logic edit (not deletion)?
  │                            └── YES ── ► 🛑 SC-5
  │ NO
  ▼
Is the user's "Proceed" given for this exact task?
  │                            └── NO  ── ► 🛑 SC-8
  │ YES
  ▼
Does `python -m compileall -q .` exit 0 RIGHT NOW (baseline clean)?
  │                            └── NO  ── ► 🛑 SC-11 (fix baseline first)
  │ YES
  ▼
Does the change touch the FCD dual-role surface, USE_FCD, or
extract_observed_features?
  │                            └── YES ── ► 🛑 SC-35
  │ NO
  ▼
Would the change introduce a module-scope import of cupy/torch/sbi/vbi?
  │                            └── YES ── ► 🛑 SC-20
  │ NO
  ▼
Would the change add a new config key or a new dependency?
  │                            └── YES ── ► 🛑 SC-23 / SC-24
  │ NO
  ▼
Would the change break the (n_nodes, csz) param shape contract?
  │                            └── YES ── ► 🛑 SC-10
  │ NO
  ▼
Would the change rename a public function or change its signature?
  │                            └── YES ── ► 🛑 SC-32 / SC-33
  │ NO
  ▼
Would the change touch >5 files?
  │                            └── YES ── ► 🛑 SC-36
  │ NO
  ▼
APPLY change
  │
  ▼
Run TIER 0 verification (file 29 §4.1)
  │
  ▼
All five invariants (compile, imports, P-1, P-2, GPU-1) pass?
  │                            └── NO  ── ► 🛑 SC-11..SC-16 (rollback, report)
  │ YES
  ▼
Tier-specific verification (29 §4.2..§4.5) passes?
  │                            └── NO  ── ► 🛑 SC-17 (rollback if QC) or surface
  │ YES
  ▼
DONE ✅ — write summary table + new doc
```

---

## 5.8 RAPID-RECALL: STOP-CONDITION SHORT FORMS

For when scanning a plan quickly:

| Pattern in plan | Stop |
|---|---|
| "edit `config.py`" | SC-1 |
| "edit `inference.py`" | SC-2 |
| "change scoring weight" / "model_selection.py" | SC-3 |
| "change fix_mean" | SC-4 |
| "edit fc.py / fcd.py / wc_runner.py (root) / …" | SC-5 |
| "edit `bold.py`" | SC-6 |
| "edit vbi/* or sbi/*" | SC-7 |
| no explicit "Proceed" | SC-8 |
| compile / import error | SC-11 / SC-12 |
| invariant FAIL | SC-14 / SC-15 / SC-16 |
| QC pass=False | SC-17 |
| z-score before PCA | SC-19 |
| module-scope cupy/torch/sbi import | SC-20 |
| BoldMonitor xp=cp | SC-21 |
| fit() on val/test | SC-22 |
| new config key | SC-23 |
| new dependency | SC-24 |
| change prior bounds / SEED / PCA_DIM | SC-26 / SC-27 / SC-29 |
| hardcoded 240 | SC-28 |
| file delete | SC-31 |
| rename | SC-32 |
| signature change | SC-33 |
| git push/merge/rebase | SC-34 |
| FCD / USE_FCD work | SC-35 |
| >5 files | SC-36 |

---

**End of stop conditions. The harness is complete.**

After reading 26 → 27 → 28 → 29 → 30, you are oriented, fenced,
proceduralized, verifiable, and braked. You may now plan.
