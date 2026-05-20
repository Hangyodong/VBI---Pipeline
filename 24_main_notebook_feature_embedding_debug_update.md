# 24 — `main.ipynb` Feature Extraction + Feature Embedding Debug Update

**Date:** 2026-05-18
**Author:** Claude Opus 4.7
**Status:** **Notebook updated.** Backup made. Notebook NOT executed.
**Predecessor docs:** 16, 19, 20, 21, 22, 23
**Branch:** `refactor/02-simulation`

---

## 1. What Was Changed

The existing Integrated VBI Pipeline Debug Cell (cell index 4) had a
single Section I called "I. Feature extraction (synthetic BOLD)" that
tested only `compute_fc` / `fc_to_upper_tri` / `compute_sim_fcd_matrix`
+ FCD helpers. That section was **replaced in place** with a longer,
explicitly-staged section:

| Aspect | Before | After |
|---|---|---|
| Section banner | `I. Feature extraction (synthetic BOLD)` | `I. Feature extraction + feature embedding (synthetic BOLD)` |
| Subsections (numbered with `# ---- I.N`) | (none) | I.1 — I.8 (see §4) |
| Section size | 2,701 chars / ~60 lines | 10,494 chars / ~190 lines |
| `_record(...)` call sites in Section I | 7 | 14 |
| New modules touched | `features.fc`, `features.fcd` | + `features.extraction`, `inference.scaling`, `inference.feature_pipeline`, `inference.embedding`, `torch` |

No other section of the cell was modified. No other cell of the
notebook was modified. No `.py` file was modified. The two pre-
existing stage-gate flags continue to govern this section:

```python
RUN_FEATURE_EXTRACTION = True   # default
```

If `RUN_FEATURE_EXTRACTION` is `False`, the whole expanded section
SKIPs (single `record_skip`).

---

## 2. Which Notebook Cell Was Updated

| Item | Value |
|---|---|
| Cell index | **4** (the Integrated VBI Pipeline Debug Cell inserted by doc 23) |
| Cell type | `code` |
| Notebook total cells | unchanged: 43 |
| Cell size | 20,553 chars → 28,346 chars (+ ~7.8 KB) |
| New `main.ipynb` size | 1,305,521 bytes → 1,314,875 bytes (+ ~9.1 KB) |
| Section H (Small simulation probe) | untouched, immediately above |
| Section J (Stage 1 dry run) | untouched, immediately below |
| Stage flags at top of cell | unchanged (still seven flags + `CONFIRM_FULL_RUN`) |

Syntax was verified after the edit via Python's built-in
`compile(src, '<cell[4]>', 'exec')` — clean. The notebook itself was
**not executed**.

---

## 3. Backup File Path

```
/scratch/home/wog3597/vbi/main.ipynb.bak_before_feature_embedding_debug_update
```

| Property | Value |
|---|---|
| Backup size | 1,305,521 bytes (== pre-edit `main.ipynb`) |
| New `main.ipynb` size | 1,314,875 bytes |
| Created via | `cp main.ipynb main.ipynb.bak_before_feature_embedding_debug_update` |

Rollback: `cp main.ipynb.bak_before_feature_embedding_debug_update main.ipynb`.

The earlier backup made in doc 23
(`main.ipynb.bak_before_integrated_debug_cell`) also still exists
and represents the notebook state from before the debug cell itself
was inserted.

---

## 4. What Feature Extraction Checks Are Now Included

The new Section I has 8 numbered subsections. The first three cover
the canonical extraction surface:

### I.1 — FC function test
- Generates a tiny synthetic BOLD array `(T=60, N=5)`.
- Calls `features.fc.compute_fc(ts)` → expects `(5, 5)` matrix.
- Calls `features.fc.fc_to_upper_tri(fc)` → expects length-`10` vector.
- Prints the synthetic BOLD shape, output type, and shape.

### I.2 — FCD function test
- Generates a longer synthetic BOLD `(T=240, N=5)` so the FCD sliding
  window has room to slide.
- Calls `features.fcd.compute_sim_fcd_matrix(bold)` → records type + shape.
- Calls `features.fcd.fcd_to_summary_stats(fcd_mat)` → records type + shape.
- Calls `features.fcd.fcd_to_upper_tri(fcd_mat)` → records type + shape.
- If `compute_sim_fcd_matrix` raises, inspects + prints its signature
  before downgrading to WARN.

### I.3 — Combined `extract_features` (legacy entry point)
- Calls `features.extraction.extract_features(bold)` on a synthetic
  `(T=80, N=5)` BOLD array.
- Records both returned tuple elements: `fc_vec` (length-10) and
  `fcd_stats` (length-5 zeros when `config.USE_FCD=False`, which is
  the default).

---

## 5. What Feature Embedding Checks Are Now Included

The remaining five subsections cover the full inference-stage feature
pipeline + embedding surface:

### I.4 — `ParameterScaler` (data-free)
- Calls `inference.scaling.make_stage1_param_scaler()` and runs
  `.transform(theta)` on a synthetic `(1, 4)` theta vector.
- Falls back to a direct `ParameterScaler(param_names=['a','b'], …)`
  construction if the convenience factory fails for any reason
  (factory failure → WARN; direct construction PASS).
- Records output type and shape.

### I.5 — `FamilyScaler` (fit on tiny synthetic batch)
- Fits `inference.feature_pipeline.FamilyScaler(name="synthetic")` on
  a tiny synthetic train batch `(32, 10)`.
- Transforms the same batch and checks the output preserves shape.
- This is a real fit on synthetic data — fast, CPU-only, no scientific
  side effects.

### I.6 — `FCPCAScaler` (tiny PCA, no GPU, no huge fit)
- Constructs `FCPCAScaler(n_components=4)` (overrides the
  `config.PCA_DIM` default = 300, which would not fit our tiny data).
- Fits on synthetic `(32, 30)` and transforms.
- Validates the output shape is `(32, 4)`.
- Verbose mode is silenced to keep notebook output readable.

### I.7 — `FeaturePipeline` (config-driven; SKIP/WARN by design)
- **Does NOT attempt fit.** `FeaturePipeline.__init__()` reads
  `config.PCA_DIM_FC` (= 300), `config.USE_FCD`, `config.FCD_DIM`, and
  expects training matrices with `FC_DIM`-sized features (= 6555). The
  task brief explicitly says: *"If FeaturePipeline or embedding class
  requires trained parameters, fitted PCA, or real config, do not fake
  it silently."*
- Inspects `FeaturePipeline.fit` and `FeaturePipeline.transform`
  signatures and prints them.
- Records a single WARN that names exactly which inputs are needed
  (config-sized fc/fcd training matrices, or a fitted pipeline from
  `inference.run_stage1_snpe` artifacts).

### I.8 — `FeatureEmbedding` (tiny torch MLP forward pass on CPU)
- Imports `inference.embedding.FeatureEmbedding`.
- Constructs a tiny model: `input_dim=10`, `hidden_dim=8`, `out_dim=4`.
- Forces `device = torch.device("cpu")` even if CUDA is available
  (per the task: *"Do not run GPU jobs"*).
- Runs one `with torch.no_grad():` forward pass on `torch.randn(2, 10)`.
- Records input shape, output shape, dtype, and device.
- Missing torch → WARN (`record_warn("feat.FeatureEmbedding.import", …)`),
  not FAIL.

The full check list contributed by Section I (assuming `USE_FCD=False`,
the production default, and a working environment):

| Subsection | Recorded checks | Expected status |
|---|---|---|
| I.1 | `feat.compute_fc`, `feat.fc_to_upper_tri` | PASS, PASS |
| I.2 | `feat.compute_sim_fcd_matrix`, `feat.fcd_to_summary_stats`, `feat.fcd_to_upper_tri` | PASS / WARN / PASS depending on FCD dimension drift (doc 16 §3.1 Risk A6) |
| I.3 | `feat.extract_features` | PASS |
| I.4 | `feat.ParameterScaler.stage1` (or fallback `feat.ParameterScaler.synthetic`) | PASS |
| I.5 | `feat.FamilyScaler` | PASS |
| I.6 | `feat.FCPCAScaler` | PASS |
| I.7 | `feat.FeaturePipeline` | **WARN by design** (no fitted pipeline) |
| I.8 | `feat.FeatureEmbedding` | PASS |

---

## 6. Which Checks Are PASS / WARN / SKIP By Design

The task brief explicitly classifies missing optional embedding APIs
as WARN/SKIP and core import failures as FAIL. The new section
implements that contract as follows:

### PASS by design
- `feat.compute_fc`, `feat.fc_to_upper_tri` — pure numpy; deterministic.
- `feat.compute_sim_fcd_matrix`, `feat.fcd_to_upper_tri` — when the
  synthetic time series is long enough (T=240 is sized for it).
- `feat.extract_features` — pure numpy; deterministic.
- `feat.ParameterScaler.stage1` — data-free transform; pure numpy.
- `feat.FamilyScaler` — fits on tiny synthetic data; preserves shape.
- `feat.FCPCAScaler` — tiny synthetic PCA with explicit `n_components=4`.
- `feat.FeatureEmbedding` — tiny CPU MLP forward pass with
  `torch.no_grad()`; deterministic seed not strictly needed.

### WARN by design
- `feat.FeaturePipeline` — **always WARN**. The pipeline requires
  `config.PCA_DIM_FC`-shaped FC training data and either
  `config.FCD_DIM` summary stats or `config.USE_FCD=False`. Faking a
  fit on tiny synthetic data would violate the task's "do not fake
  silently" rule. The WARN message names the exact ingredient that
  would unblock it.
- `feat.fcd_to_summary_stats` — WARN if the FCD dual-role / dim drift
  issue (doc 16 §3.1 Risk A6) tweaks the expected shape. On a clean
  install with `USE_FCD=False`, PASS.

### WARN if environment lacks dependency
- `feat.FeatureEmbedding.import` — WARN if torch is not importable.
  (On the current host torch 2.6.0 is installed → PASS expected.)
- `feat.ParameterScaler.import`, `feat.FCPCAScaler.import` — WARN if
  inference package fails to import (would normally also surface in
  Section C as FAIL).

### SKIP by design
- All Section I subsections SKIP collectively if
  `RUN_FEATURE_EXTRACTION = False`. Records a single
  `feat / RUN_FEATURE_EXTRACTION=False` SKIP.

### FAIL by design
- `feat.fc` (the whole I.1 block) — FAILs only on a true import error
  (`features.fc` not importable). Numerical issues degrade to WARN.
- `feat.compute_fc` / `feat.fc_to_upper_tri` — FAIL if numpy output
  shapes are wrong. (Indicates real regression, not env drift.)

All non-fatal subsections continue execution. The Section M summary
at the bottom of the cell aggregates everything from Section I
through Section L into the global PASS/WARN/FAIL/SKIP totals.

---

## 7. How To Run The Updated Cell Manually

The cell is unchanged outside Section I. Same usage as doc 23 §7:

1. Open `main.ipynb` in Jupyter / VS Code / Cursor / equivalent.
2. Restart the kernel (recommended if you changed any flag).
3. Run the **Setup cell** (`cell[2]`) so that `config`,
   `data_loader`, `evaluate`, `inference`, `simulator` are imported
   and `config` is in sync.
4. Run the **Integrated VBI Pipeline Debug Cell** (`cell[4]`).
5. Scroll to the **Section I** banner:
   `I. Feature extraction + feature embedding (synthetic BOLD)`.
   You will see eight subsections labelled `# ---- I.1` through
   `# ---- I.8`, each emitting one or more PASS/WARN/SKIP/FAIL lines.
6. The **Section M Summary** at the bottom reports global totals
   (now including the new Section I subsections).

### To temporarily disable the expanded Section I

Edit the flag at the top of cell 4:

```python
RUN_FEATURE_EXTRACTION = False
```

This causes the cell to emit one `feat / RUN_FEATURE_EXTRACTION=False`
SKIP and bypass the entire expanded section. All other sections
remain unaffected.

### To exercise the real `FeaturePipeline` (out of scope for this cell)

Section I.7 deliberately does not fit `FeaturePipeline`. To exercise
that path, run the official pipeline with a tiny budget from a
**separate** cell (as Section J already recommends):

```python
from pipelines import run_pipeline
run_pipeline(n_sim=1000, run_stage2=False, verbose=True)
```

That path fits the real `FCPCAScaler` + `FamilyScaler("FCD")` inside
the production `FeaturePipeline.fit` and writes the artifacts the
`FeaturePipeline.transform` step expects.

---

## 8. Confirmation Checkpoints

1. **Backup exists:** `main.ipynb.bak_before_feature_embedding_debug_update`,
   1,305,521 bytes (byte-identical to the pre-edit `main.ipynb`).
2. **`main.ipynb` was modified:** 1,305,521 → 1,314,875 bytes.
3. **Updated cell index:** `4` (the Integrated VBI Pipeline Debug Cell).
4. **No other cell or `.py` file changed.** No file was created,
   deleted, renamed, or moved (other than this report). The helper
   script `_update_section_I.py` was deleted immediately after use.
5. **Syntax verified** via `compile(src, '<cell[4]>', 'exec')` — no
   exceptions raised.
6. **Notebook NOT executed.** Jupyter has not run any cell as part of
   this update.

---

**End of feature-extraction + feature-embedding debug update. The
notebook is ready for the user to open and run interactively.**
