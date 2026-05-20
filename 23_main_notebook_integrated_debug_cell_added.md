# 23 — `main.ipynb` Integrated Debug Cell Added

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** **Notebook edited.** Backup made. Notebook NOT executed.
**Predecessor docs:** 16, 17, 19, 20, 21, 22
**Branch:** `refactor/02-simulation`

---

## 1. What Was Added

Two new cells were spliced into `main.ipynb`, immediately after the
Setup cell:

| New cell index | Type | Title / Purpose |
|---|---|---|
| 3 | markdown | `## Integrated VBI Pipeline Debug Cell` + a short orientation paragraph |
| 4 | code | Staged validation runner with sections A–M (see §4) |

The previous cells at index ≥ 3 all shifted by +2:

| Cell | Before | After |
|---|---|---|
| `## DEBUG CHECKS` markdown (`run_all_checks` intro) | 3 | 5 |
| `from debug_notebook import run_all_checks` code | 4 | 6 |
| `## (Optional) 자원 진단 + 최적 GPU_BATCH ...` markdown | 5 | 7 |
| ... rest of notebook (Steps 1–14) | 6 … 40 | 8 … 42 |

Cell count: **41 → 43.**

No existing cell content was modified — only positions shifted.

---

## 2. Where It Was Inserted

- Inserted **after** the existing setup cell at original index 2
  (the cell that runs `setup_pipeline(cfg)` and `import config /
  data_loader / evaluate / inference / simulator`).
- Markdown heading is now at **index 3**.
- Code cell is now at **index 4**.

Confirmed by reading the notebook post-write:

```
cell[2] (setup, unchanged) tail:
  '...\nimport evaluate\nimport inference\nimport simulator\n\n%matplotlib inline\n'

cell[3] (new markdown):
  ## Integrated VBI Pipeline Debug Cell
  Staged validation runner. Inserted immediately after the Setup cell.
  All stages are individually gated by flags at the top of the next code cell.
  Defaults: smoke + debug.py --basic + small simulation probe + feature extraction.
  ...

cell[4] (new code), first line:
  # ===========================================================================
  # Integrated VBI Pipeline Debug Cell  (Patch 1 + Patch 2 aware)
  ...

cell[5] (was DEBUG CHECKS md, unchanged content):
  ## DEBUG CHECKS
  ...
```

Syntax of the new code cell was verified with Python's built-in
`compile(src, '<cell[4]>', 'exec')` — 20,553 chars, compiled cleanly.
The notebook itself was **not** executed.

---

## 3. Backup File Path

```
/scratch/home/wog3597/vbi/main.ipynb.bak_before_integrated_debug_cell
```

| Property | Value |
|---|---|
| Backup size | 1,279,250 bytes (== pre-edit `main.ipynb`) |
| New `main.ipynb` size | 1,305,521 bytes (+26,271 bytes; the two new cells) |
| Created via | `cp main.ipynb main.ipynb.bak_before_integrated_debug_cell` |

Rollback recipe: `cp main.ipynb.bak_before_integrated_debug_cell main.ipynb`.

---

## 4. Default Flags

The new code cell exposes seven stage gates and one belt-and-braces
safety variable at the very top:

```python
RUN_SMOKE_CHECKS       = True
RUN_DEBUG_BASIC        = True
RUN_SMALL_SIMULATION   = True
RUN_FEATURE_EXTRACTION = True
RUN_STAGE1_DRY_RUN     = False
RUN_STAGE2_DRY_RUN     = False
RUN_FULL_PIPELINE      = False
CONFIRM_FULL_RUN       = False   # belt-and-braces gate for the full run
```

These are the **defaults out of the box**. They were chosen so the
cell can be executed immediately without touching any heavy code
paths.

---

## 5. Which Stages Run By Default

When the cell is executed with the defaults above, the following
sections actually do work:

| Section | Default | What runs |
|---|---|---|
| A. Setup validation | always | `cwd` / `sys.path` / Python version / git branch + short status |
| B. File existence | always | 18 required source files |
| C. Compile + import smoke | `RUN_SMOKE_CHECKS=True` | `compileall.compile_dir`; imports of `config`, `data_loader`, 5 packages; `find_spec` table for 13 names (5 packages + 6 root duplicates + `simulator` + `evaluate`) |
| D. Patch invariant checks | always | P-1: `pipelines.stage1_stage2.evaluate.__name__ == "evaluation"`. P-2: `data_loader.get_subject_data` contains `from simulation.delays import …` and NOT the old `from simulator import …` |
| E. Public API smoke | always | `pipelines.run_pipeline`; `inference.ParameterScaler / FeaturePipeline / run_stage1_snpe / run_stage2_snpe`; `evaluation.fc_metrics / evaluate_validation_stage1 / select_best_model / final_test` |
| F. Data file checks | always | `scipy.io.loadmat` of `MPTP_FC_115.mat`, `MPTP_SC_115.mat`; prints keys + shapes (missing → WARN, not FAIL) |
| G. `debug.py --basic` | `RUN_DEBUG_BASIC=True` | `subprocess.run([sys.executable, "debug.py", "--basic"], timeout=180)`; tail of stdout / stderr printed; **import/syntax breakage → FAIL, env/data drift → WARN** |
| H. Small simulation probe | `RUN_SMALL_SIMULATION=True` | inspects `simulation.wc_runner.simulate_single`'s signature and attempts one call with N=5 zero-coupling weights + config-derived params; **failure → WARN (likely GPU/cupy absence)** |
| I. Feature extraction | `RUN_FEATURE_EXTRACTION=True` | `compute_fc((60,5))`, `fc_to_upper_tri`, then `compute_sim_fcd_matrix((240,5))` + `fcd_to_summary_stats` + `fcd_to_upper_tri` |
| J. Stage 1 dry run | **OFF** | SKIP |
| K. Stage 2 dry run | **OFF** | SKIP |
| L. Full pipeline | **OFF (double gate)** | SKIP |
| M. Summary | always | totals + lists of FAIL/WARN names + recommended next action |

The cell does **not** stop on warnings or non-critical failures. Only
true import / syntax breakage in Section G or compile errors in
Section C land as FAIL.

---

## 6. Which Stages Require Manual Enabling

Three stages are gated OFF by default to protect against accidental
long runs:

| Stage | Flag(s) to flip | What flipping them does |
|---|---|---|
| Stage 1 dry run (Section J) | `RUN_STAGE1_DRY_RUN = True` | Inspects `inference.run_stage1_snpe` and prints its signature + a recommended tiny-budget invocation. **Does not** actually run SNPE — the cell intentionally emits a WARN with the exact command the user should run from a separate cell (`run_pipeline(n_sim=1000, run_stage2=False)`). |
| Stage 2 dry run (Section K) | `RUN_STAGE2_DRY_RUN = True` | Same shape as J: inspects `inference.run_stage2_snpe`, prints signature, recommends `run_pipeline(n_sim=1000, n_sim_s2=1000, run_stage2=True)`. Does not execute. |
| Full pipeline (Section L) | `RUN_FULL_PIPELINE = True` **AND** `CONFIRM_FULL_RUN = True` | Both flags must be `True` simultaneously. Then runs `pipelines.run_pipeline()` with current `config` values. Any other combination → SKIP with an explanation. |

The reason J and K do not actually invoke `run_stage1_snpe` /
`run_stage2_snpe` directly is that those functions assume the full
pipeline state (data already loaded, scalers fitted, family splits
ready). The cleanest dry run is to call `run_pipeline` with a small
`n_sim`, which the cell surfaces as a recommended command.

---

## 7. How To Run The Cell Manually

The cell is a normal Jupyter code cell. Steps:

1. Open `main.ipynb` in Jupyter / VS Code / Cursor / etc.
2. Restart the kernel (recommended whenever flags change).
3. Run the **Setup cell** (`cell[2]`, the existing one with
   `setup_pipeline(cfg)` etc.) so that `config`, `data_loader`,
   `evaluate`, `inference`, `simulator` are imported and `config` is
   in sync.
4. Run the **Integrated VBI Pipeline Debug Cell** (`cell[4]`, the one
   inserted by this patch).
5. Inspect the section-by-section output. The final **Summary**
   block prints PASS/WARN/FAIL/SKIP counts plus a recommended next
   action.

To enable a heavier stage, edit the flag block at the top of cell 4
and re-execute the cell — no kernel restart required.

Example incremental enablement:

```python
# Smallest dry-run check that touches Stage 1 path:
RUN_STAGE1_DRY_RUN = True

# Smallest dry-run check that touches Stage 2 path (after Stage 1 confirmed):
RUN_STAGE2_DRY_RUN = True

# Actual production run — requires BOTH flags True:
RUN_FULL_PIPELINE = True
CONFIRM_FULL_RUN  = True
```

---

## 8. Safety Notes

The cell was designed under tight safety constraints. A non-exhaustive
list of what it does and does **not** do:

### What it does NOT do by default

- **No SNPE training.** Stage 1 / Stage 2 dry-run sections do not call
  `run_stage1_snpe` / `run_stage2_snpe` directly; they only inspect
  signatures and print the recommended `run_pipeline(...)` command.
- **No large simulation.** The small-simulation probe (Section H)
  uses `N=5` regions, zero-coupling weights, `n_repeat=1`. It
  catches **all** exceptions and downgrades them to WARN — so a
  missing cupy / GPU does not break the cell.
- **No full pipeline run.** Section L requires both
  `RUN_FULL_PIPELINE=True` **and** `CONFIRM_FULL_RUN=True`. Either
  flag alone results in SKIP.
- **No filesystem mutation.** The cell never writes, deletes, moves,
  renames, or chmods any file. It only reads.
- **No `sys.path` hack beyond `REPO_ROOT`.** The cell prepends
  `/scratch/home/wog3597/vbi` to `sys.path` only if missing.
- **No mocking, monkey-patching, or scientific-behavior change.**
  Synthetic-BOLD arrays are local to the cell and never written
  back to `config` or to any module's global state.

### What it does as defensive coding

- **Continue on non-critical failures.** Only import / syntax errors
  in Sections C and G are escalated to FAIL. Everything else degrades
  to WARN so the cell always reaches Section M (Summary).
- **`debug.py --basic` is timed out.** `subprocess.run(..., timeout=180)`
  prevents a stuck child from blocking the kernel.
- **Backups before editing.** The notebook itself was backed up to
  `main.ipynb.bak_before_integrated_debug_cell` before this patch
  modified it.
- **Patch-aware invariants.** Section D directly verifies the two
  patches that have landed so far (P-1 in doc 19, P-2 in doc 21). If
  either invariant breaks in the future, the cell flags it as FAIL.
- **No `evaluate.py` / `simulator.py` deletion assumed.** The cell
  uses `find_spec` to *report* whether the compat wrappers still
  resolve, but does not depend on them being present or absent.
- **No notebook execution.** This editing step writes JSON only;
  Jupyter has not run the new cell.

### Things to be aware of when running

- Section H may emit a WARN like `could not run tiny simulation
  (likely needs GPU/cupy): ...` — that is **expected** on a CPU-only
  host. It is not a regression.
- Section G may emit a WARN about `debug.py --basic` returncode != 0
  if the pre-existing FCD dual-role / FeaturePipeline mismatches
  (doc 16 §3.1 Risk A6) trip the basic tests. The cell distinguishes
  this from import/syntax breakage and downgrades it to WARN
  intentionally.
- Section F gracefully degrades to WARN if `MPTP_FC_115.mat` /
  `MPTP_SC_115.mat` are not on disk — but as of today both are
  present (verified during cell authoring).

### Confirmed checkpoints (after editing)

1. Backup exists: `main.ipynb.bak_before_integrated_debug_cell`
   (1,279,250 bytes — byte-identical to pre-edit `main.ipynb`).
2. `main.ipynb` was modified: 1,279,250 bytes → 1,305,521 bytes.
3. Markdown heading inserted at **index 3**; code cell inserted at
   **index 4**; downstream cells shifted +2; cell count 41 → 43.
4. The notebook was **not executed**; the new cell's source was
   syntax-checked with `compile(src, '<cell[4]>', 'exec')` only.
5. The helper insertion script (`_insert_debug_cell.py`) was removed
   after use; no temporary artifacts remain.

---

**End of integrated-debug-cell report. The notebook is ready for the
user to open and run interactively.**
