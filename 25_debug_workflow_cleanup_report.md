# 25 — Debug Workflow Cleanup Report

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** **Notebook cleaned up; `debug.py` extended with one new lightweight test.**
**Predecessor docs:** 19, 20, 21, 22, 23, 24
**Branch:** `refactor/02-simulation`

---

## 1. Which Old Notebook Debug Cells Were Removed

Two cells were removed from `main.ipynb` because they constituted an
older, narrower debug workflow that is now fully subsumed by the
Integrated VBI Pipeline Debug Cell:

| Old index | Type | First line | Reason for removal |
|---|---|---|---|
| 5 | markdown | `## DEBUG CHECKS` | Older debug heading — duplicate of the new `## Integrated VBI Pipeline Debug Cell` heading at index 3. |
| 6 | code | `from debug_notebook import run_all_checks` … `debug_results = run_all_checks(run_sbi_smoke=False, run_gpu_smoke=False, verbose=True)` | Older debug entry point. Its checks are now superseded by the new Integrated cell's sections A–M, which additionally cover P-1/P-2 invariants and feature-embedding subsections. |

Nothing else was removed. In particular, **kept**:

| Index now | Type | Purpose |
|---|---|---|
| 5 (was 7) | markdown | `## (Optional) 자원 진단 + 최적 GPU_BATCH 자동 탐색` — GPU/RAM resource diagnostic + batch-size tuner. Scientific utility, not a debug cell. |
| 6 (was 8) | markdown-typed code | The 192-line body for the batch-size tuner. Scientific utility, not a debug cell. |
| 7 → 40 (was 9 → 42) | mixed | All `## Step 1` through `## Step 14` scientific cells. |

Cell count: **43 → 41.** The setup cell (`cell[2]`) is untouched. The
Integrated VBI Pipeline Debug Cell heading is at index **3**; the
code cell itself is at index **4**.

---

## 2. Which Notebook Cell Was Kept

The single retained debug section:

| Index | Type | Content |
|---|---|---|
| 3 | markdown | `## Integrated VBI Pipeline Debug Cell` |
| 4 | code | 27,637-char staged debug runner with sections A–M (Setup validation, File existence, Compile/import smoke, P-1/P-2 invariants, Public-API smoke, Data file checks, `debug.py --basic`, Small simulation probe, Feature extraction + feature embedding, Stage 1 dry run, Stage 2 dry run, Full pipeline, Summary). |

This is the newest cell — it includes the feature extraction +
feature embedding checks added in doc 24, plus the new
`RUN_FEATURE_EMBEDDING` switch added in this cleanup pass.

---

## 3. Where the Final Debug Cell Is Located

| Aspect | Value |
|---|---|
| Notebook | `/scratch/home/wog3597/vbi/main.ipynb` |
| Markdown heading cell index | **3** |
| Code cell index | **4** |
| Backup of pre-cleanup state | `/scratch/home/wog3597/vbi/main.ipynb.bak_before_debug_cell_cleanup` (1,314,875 bytes) |
| Backup of pre-doc-24 state | `/scratch/home/wog3597/vbi/main.ipynb.bak_before_feature_embedding_debug_update` (1,305,521 bytes) |
| Backup of pre-doc-23 state | `/scratch/home/wog3597/vbi/main.ipynb.bak_before_integrated_debug_cell` (1,279,250 bytes) |
| New `main.ipynb` size | 1,317,253 bytes |

All three backups remain on disk and are not deleted by this report.

---

## 4. Feature Embedding Checks: Included

Yes. Section I of cell 4 was split into two sub-blocks gated by two
separate flags. Both default to `True`.

### I.A — Feature extraction (gated by `RUN_FEATURE_EXTRACTION`)
- I.1 — `compute_fc` / `fc_to_upper_tri` on synthetic `(60, 5)` BOLD.
- I.2 — `compute_sim_fcd_matrix` / `fcd_to_summary_stats` /
  `fcd_to_upper_tri` on synthetic `(240, 5)` BOLD.
- I.3 — `features.extraction.extract_features` on `(80, 5)` BOLD.

### I.B — Feature embedding (gated by `RUN_FEATURE_EMBEDDING`)
- I.4 — `inference.scaling.ParameterScaler` (data-free; uses
  `make_stage1_param_scaler()`, falls back to a tiny synthetic
  `ParameterScaler` if needed).
- I.5 — `inference.feature_pipeline.FamilyScaler` fit on tiny
  synthetic `(32, 10)` batch.
- I.6 — `inference.feature_pipeline.FCPCAScaler(n_components=4)`
  on tiny synthetic `(32, 30)` batch (no GPU, no large fit).
- I.7 — `inference.feature_pipeline.FeaturePipeline` — **always WARN
  by design**. Inspects fit/transform signatures and prints them, but
  does **not** fake a trained PCA / scaler. Per the task brief:
  > "If FeaturePipeline or embedding class requires trained
  > parameters, fitted PCA, or real config, do not fake it silently."
- I.8 — `inference.embedding.FeatureEmbedding` constructed as
  `(input_dim=10, hidden_dim=8, out_dim=4)`, forced to CPU even if
  CUDA is available, runs a single `torch.no_grad()` forward pass on
  `torch.randn(2, 10)`.

Either sub-block independently SKIPs if its flag is `False`:
- `RUN_FEATURE_EXTRACTION=False` → emits one `feat.extraction / RUN_FEATURE_EXTRACTION=False` SKIP and bypasses I.1–I.3.
- `RUN_FEATURE_EMBEDDING=False` → emits one `feat.embedding / RUN_FEATURE_EMBEDDING=False` SKIP and bypasses I.4–I.8.

---

## 5. Which Debugging Script / Module Was Updated

**`debug.py`** — exactly one new test function plus one new line of
orchestration.

Nothing else in `debug.py` changed. `debug_notebook.py` was **not**
modified (it remains on disk as legacy, since "do not delete files"
applies; the notebook no longer references it).

---

## 6. What Changed in `debug.py`

### 6.1 Added function — `test_patch_invariants()`

Placed in the natural slot just below `test_observed_fcd_direct` and
above the "Mock inference flow" section header (lines 490+):

```python
def test_patch_invariants():
    """Structural-cleanup patch invariants (P-1, P-2)."""
    import importlib
    import inspect

    pl = importlib.import_module("pipelines.stage1_stage2")
    assert hasattr(pl, "evaluate"), (
        "P-1 broken: pipelines.stage1_stage2 has no 'evaluate' name"
    )
    assert pl.evaluate.__name__ == "evaluation", (
        f"P-1 broken: pl.evaluate.__name__ = {pl.evaluate.__name__!r} "
        "(expected 'evaluation')"
    )
    print(f"  P-1 OK : pl.evaluate -> {pl.evaluate.__file__}")

    import data_loader
    src = inspect.getsource(data_loader.get_subject_data)
    assert "from simulation.delays import compute_delay_matrix" in src, (
        "P-2 broken: data_loader.get_subject_data does not import "
        "compute_delay_matrix from simulation.delays"
    )
    assert "from simulator import compute_delay_matrix" not in src, (
        "P-2 broken: legacy 'from simulator import compute_delay_matrix' "
        "still present in data_loader.get_subject_data"
    )
    print("  P-2 OK : data_loader.get_subject_data uses simulation.delays")
```

Pure structural assertion. No GPU, no SBI, no large allocation, no
side effects on `config` or any module's global state.

### 6.2 Wired into `run_basic_tests`

```diff
 def run_basic_tests(runner):
     """No GPU, no SBI required."""
     runner.run("config consistency", test_config_consistency)
     runner.run("imports", test_imports)
+    runner.run("patch invariants (P-1, P-2)", test_patch_invariants)
     runner.run("ParameterScaler", test_parameter_scaler)
     ...
```

No other change. `--basic`, `--all`, `--data`, `--pipeline`, `--sim`
flags are preserved. The full pipeline still requires an explicit
opt-in (the notebook's `RUN_FULL_PIPELINE` + `CONFIRM_FULL_RUN`
double gate; `debug.py` itself never runs the full pipeline).

### 6.3 What was NOT changed in `debug.py`

- Existing tests (`test_config_consistency`, `test_imports`,
  `test_parameter_scaler`, `test_stage2_param_scaler`,
  `test_family_scaler`, `test_fc_pca_scaler`, `test_feature_pipeline`,
  `test_embedding_net`, `test_fc_upper_tri`, `test_fcd_summary`,
  `test_observed_fcd_direct`, `test_data_files_exist`,
  `test_data_loading`, `test_atlas_labels`,
  `test_mock_inference_flow`, `test_real_simulation`).
- CLI flags (`--all`, `--basic`, `--data`, `--pipeline`, `--sim`).
- The `TestRunner` class.
- Any scientific value, prior bound, FC/FCD math, or SBI logic.
- `if __name__ == "__main__":` semantics.

---

## 7. Safety Flags Now Present in `main.ipynb`

Verbatim from the top of cell 4 (after the cleanup):

```python
RUN_SMOKE_CHECKS       = True
RUN_DEBUG_BASIC        = True
RUN_SMALL_SIMULATION   = True
RUN_FEATURE_EXTRACTION = True
RUN_FEATURE_EMBEDDING  = True
RUN_STAGE1_DRY_RUN     = False
RUN_STAGE2_DRY_RUN     = False
RUN_FULL_PIPELINE      = False
CONFIRM_FULL_RUN       = False   # belt-and-braces gate for the full run
```

- `RUN_FEATURE_EXTRACTION` and `RUN_FEATURE_EMBEDDING` are
  **independent** switches (added in this cleanup pass).
- Stage 1 / Stage 2 dry runs remain disabled by default.
- The full pipeline only executes when **both** `RUN_FULL_PIPELINE`
  **and** `CONFIRM_FULL_RUN` are simultaneously `True`. Any other
  combination → SKIP.

---

## 8. Tests Run After Editing

Per the task brief, only lightweight validation; **no notebook
execution**, no full SBI, no large simulation.

### 8.1 `python -m compileall -q .`

```
compileall: exit 0
```

**Result: PASS.** Every `.py` file in the repository compiles clean
with the new `debug.py`.

### 8.2 `python debug.py --basic`

Tail of the test summary:

```
======================================================================
  Test summary
======================================================================
  FAIL  config consistency
  PASS  imports
  PASS  patch invariants (P-1, P-2)        <-- NEW
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
  PASS: 9  |  FAIL: 3  |  SKIP: 0
```

**Result: the new `patch invariants (P-1, P-2)` test PASSes.**

The PASS count grew from 8 (doc 21 §7) to 9. The three FAIL entries
are unchanged from the pre-existing baseline and are documented in
doc 16 §3.1 Risk A6 and doc 21 §7. They were not introduced by this
cleanup. See §9 below.

### 8.3 Notebook execution

**Skipped — explicitly forbidden by the task brief.** Syntax of cell
4 was verified with Python's built-in
`compile(src, '<cell[4]>', 'exec')` → 27,637 characters, compiled
cleanly.

---

## 9. WARN / SKIP / Pre-existing FAIL Items That Remain

### 9.1 In `debug.py --basic`

These three FAILs are **pre-existing** and unrelated to the cleanup:

| Test | Failure | Source |
|---|---|---|
| `config consistency` | `FCD_DIM=5 != FC_DIM=6555` | FCD dual-role surface (doc 16 §3.1 Risk A6 / Tier X); dormant in production because `USE_FCD=False`. |
| `FeaturePipeline` | `output (200, 50) != (200, 80)` | Test-assertion drift in the debug script; runtime pipeline outputs FC-only 50-dim features as expected when `USE_FCD=False`. |
| `FCD upper triangle (simulated)` | `vec (6555,) != (5,)` | Same FCD dual-role surface. |

All three were already FAILing before this cleanup; the cleanup does
not improve or worsen them. They are flagged as Tier X scientific
work (doc 16 §7), out of structural-cleanup scope.

### 9.2 In the notebook's Integrated cell (expected on next manual run)

| Section | Recorded check | Expected on default flags |
|---|---|---|
| H | `sim.small` | likely **WARN** on CPU-only hosts (cupy / GPU absent) — by design. |
| I.B-7 | `feat.FeaturePipeline` | **WARN** by design (no fitted PCA). |
| I.B-7 / I.B-others | `feat.FCPCAScaler`, `feat.FamilyScaler`, `feat.ParameterScaler`, `feat.FeatureEmbedding` | PASS (synthetic tiny inputs). |
| J | `stage1.dryrun` | SKIP (flag off by default). |
| K | `stage2.dryrun` | SKIP. |
| L | `full.pipeline` | SKIP (double gate not satisfied). |
| Section M | summary | reports PASS / WARN / FAIL / SKIP counts + failed/warning names + recommended next action. |

None of the above is a regression. All WARN/SKIP behaviors are
intentional safety choices documented in doc 23 §8, doc 24 §6, and
the task brief.

---

## 10. Whether the Notebook Is Ready To Use

**Yes.** The notebook is in a clean, self-consistent state:

- Exactly **one** debug section after the setup cell, with the
  required markdown heading `## Integrated VBI Pipeline Debug Cell`
  at index 3 and the staged code cell at index 4.
- The newest feature-extraction + feature-embedding checks (doc 24)
  are preserved and now split between two independent flags.
- The full set of safety flags is present at the top of cell 4 in the
  exact form the task brief requires.
- All other cells (setup, scientific Steps 1–14, GPU batch tuner) are
  unchanged.
- `debug.py --basic` runs cleanly modulo the pre-existing Tier X
  FCD failures, with the new `patch invariants (P-1, P-2)` test now
  passing.

To run the workflow:

1. Open `main.ipynb` and restart the kernel.
2. Run cell[2] (Setup).
3. Run cell[4] (Integrated VBI Pipeline Debug Cell) — defaults
   exercise smoke + import checks + patch invariants +
   `debug.py --basic` + small simulation probe + feature extraction +
   feature embedding.
4. To enable heavier stages, flip the relevant flag(s) at the top of
   cell 4 and re-execute the cell (no kernel restart required).
5. The full pipeline requires both `RUN_FULL_PIPELINE = True` **and**
   `CONFIRM_FULL_RUN = True`. Either alone results in SKIP.

---

**End of debug-workflow cleanup report. Backups preserved; no source
file deleted; no scientific logic changed.**
