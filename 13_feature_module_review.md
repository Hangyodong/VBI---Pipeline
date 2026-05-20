# 13 — Feature Module Review

**Date:** 2026-05-18
**Branch:** refactor/02-simulation
**Reviewer:** Claude (claude-sonnet-4-6)
**Files inspected:**
`features/fc.py`, `features/fcd.py`, `features/extraction.py`,
`features/screening.py`, `features/__init__.py`,
`inference/feature_pipeline.py`, `inference/scaling.py`,
`inference/embedding.py`,
`fc.py`, `fcd.py`, `extraction.py`, `screening.py`
**Cross-referenced:** `inference/snpe.py` (steps 4–8), `config.py`
(feature constants), `11_import_audit.md`, `12_simulation_module_review.md`

---

## 1. BOLD to FC Computation Flow

Full path from simulation output to the feature vector fed into inference:

```
simulate_gpu_batch(...)
  → list of (T_bold, N) float32 BOLD arrays
      T_bold = 240  (ANALYSIS_BOLD_T after T_cut)
      N      = 115  (N_REGIONS)

  → extract_simulated_features(bold)   OR   worker_extract(bold) via ProcessPool
      │
      ├── compute_fc(bold)
      │     np.corrcoef(bold.T)           → (N, N) float64  Pearson r
      │     np.nan_to_num(fc, nan=0.0)   → zero any NaN entries
      │     np.fill_diagonal(fc, 0.0)    → zero the diagonal
      │     return fc                     (N, N) float64
      │
      └── fc_to_upper_tri(fc)
            np.triu_indices(N, k=1)      → upper triangle indices
            apply config.NAN_MASK if set → drop masked region-pairs
            vec.astype(float32)
            return vec                   (FC_DIM,) float32
                                         FC_DIM = 6555 when NAN_MASK=None

  → fc_raw  : (N_SIM, FC_DIM=6555) float32
  → fcd_raw : (N_SIM, 0)           float32  [fc_only mode, USE_FCD=False]

  → FeaturePipeline.fit(fc_raw, fcd_raw)        [train only — R10]
      FCPCAScaler.fit(fc_raw)                   PCA on raw FC, no z-score
      → FeaturePipeline.transform(fc_raw, fcd_raw)
          FCPCAScaler.transform(fc_raw)          (N_SIM, 6555) → (N_SIM, 300)
          → x_input  (N_SIM, 300) float32

  → FeatureEmbedding.forward(x_input)
      Linear(300, 512) → ReLU → Dropout(0.1)
      Linear(512, 256) → ReLU
      Linear(256, 128)
      → embedding  (N_SIM, 128) float32

  → SNPE-C: joint train (theta_scaled, embedding)
```

---

## 2. FC Matrix Generation

**Function:** `features/fc.py:compute_fc(ts)` — lines 17–33

```python
fc = np.corrcoef(ts.T)             # (N, N) float64 — rows = variables
fc = np.nan_to_num(fc, nan=0.0)    # replace NaN with 0 (not with mean)
np.fill_diagonal(fc, 0.0)          # remove self-correlation
return fc                           # (N, N) float64
```

**Input:**
- `ts`: `(T, N)` float32 BOLD time series — rows are time points, columns are regions
- Transposed to `(N, T)` by `ts.T` for `corrcoef` which expects rows-as-variables

**Why NaN replacement is 0, not mean:**
A constant-activity region produces NaN in `np.corrcoef` (zero std denominator). Replacing
with 0 means it contributes no correlation signal, matching the observed-FC preprocessing
in `data_loader.py` (NaN → 0 on empirical FC). R1 prohibits changing this convention.

**Output:**
- `(N, N)` float64, symmetric, zero diagonal, values in `[-1, 1]`
- `corrcoef` always returns float64 regardless of input dtype

**No z-scoring applied here.** Per the module docstring: "per-feature scaling is the job
of the inference-stage FamilyScaler / FeaturePipeline." This is confirmed by
`inference/snpe.py:step4_fit_feature_scalers` which explicitly sets `fc_z = None` with the
comment "FC: no z-score (already Pearson r in [-1, 1])".

---

## 3. FC Upper Triangle Vector Generation

**Function:** `features/fc.py:fc_to_upper_tri(fc, nan_mask=None)` — lines 36–61

```python
n = fc.shape[0]
iu = np.triu_indices(n, k=1)           # upper triangle, k=1 excludes diagonal
if nan_mask is None:
    nan_mask = getattr(config, "NAN_MASK", None)
if nan_mask is not None and nan_mask.shape == fc.shape:
    valid = ~nan_mask[iu]               # boolean mask on upper-tri entries
    vec = fc[iu[0][valid], iu[1][valid]]
else:
    vec = fc[iu]
return vec.astype(np.float32)
```

**Normal case (`NAN_MASK=None`):**
- Returns `(FC_DIM,)` float32 where `FC_DIM = N*(N-1)/2 = 115*114/2 = 6555`
- `config.py` line 51: `FC_DIM = N_REGIONS * (N_REGIONS - 1) // 2  # 6555`
- `config.py` line 318: assert `FC_DIM == N_REGIONS * (N_REGIONS - 1) // 2`

**NAN_MASK case:**
- `config.NAN_MASK` is set by `data_loader._record_nan_mask(fc_mat)` at load time
- A NaN-affected entry is masked out of the vector
- Result dimension: `< 6555` if any regions have NaN contamination
- **This is the only place where FC_DIM can differ from the config constant.** If NAN_MASK
  removes any entries, the FeaturePipeline will be fitted on a shorter vector, and
  `FeaturePipeline.fc_dim` will reflect the actual dimension — not the config constant.

**NAN_MASK symmetry requirement:**
Both observed FC (`extract_observed_features`) and simulated FC (`extract_simulated_features`)
call `fc_to_upper_tri` which reads the same `config.NAN_MASK`. This guarantees symmetric
dimension matching as long as the mask is set before any feature extraction.

---

## 4. FCD Computation Flow

**Function:** `features/fcd.py:compute_sim_fcd_matrix(bold, window_tr, stride_tr)` — lines 20–61

```
Input:  bold (T, N) float32   [T_bold=240, N=115]
Config: FCD_WINDOW_TR = 60    (window length in TRs)
        FCD_STRIDE_TR = 3     (stride in TRs)

Safety: if T < window_tr + stride_tr → return zeros (N, N)

Sliding windows:
  starts = np.arange(0, T - window_tr + 1, stride_tr)
         = np.arange(0, 181, 3)    →  [0, 3, 6, ..., 180]   (61 windows)

Per window [s, s+60]:
  if seg.std() < 1e-8 → zeros (N, N)   [flat signal guard]
  else:
    fc_seg = np.corrcoef(seg.T)          → (N, N) float64
    np.nan_to_num(fc_seg, nan=0.0)       → float32

Stack:  fcs    (61, N, N) float32
FCD:    fcs.std(axis=0)                  → element-wise σ across windows
Symmetrise: (fcd + fcd.T) / 2
Zero diagonal
Return: (N, N) float32
```

**Semantic meaning:** FCD is defined as the *standard deviation* of FC across time windows,
not as the correlation-of-FCs matrix (TVB convention). It measures how much each pairwise
FC value fluctuates over the session. This is a non-standard but internally consistent
definition.

---

## 5. Sliding-Window FC Logic

With the default configuration:
```
T_bold      = 240 TRs (ANALYSIS_BOLD_T)
FCD_WINDOW_TR = 60    (25% of total duration)
FCD_STRIDE_TR = 3     (non-overlapping windows spaced every 3 TRs)
```

Window count: `floor((240 - 60) / 3) + 1 = 61 windows`

Each window: `np.corrcoef(seg.T)` on a 60-TR BOLD slice → `(115, 115)` FC.
Flat-signal guard: `seg.std() < 1e-8` prevents corrcoef NaN propagation when a BOLD
segment has no variance (e.g., early transient or dead-channel artifact).

**Number of corrcoef calls per simulation:** 61 × `compute_sim_fcd_matrix`. With N_SIM=50,000
simulations, this is 3.05M corrcoef calls. This is why `USE_FCD=False` is the default —
enabling FCD would roughly 62× the feature extraction compute cost.

---

## 6. FCD Summary Statistic Logic

**Function:** `features/fcd.py:fcd_to_summary_stats(fcd_matrix, nan_mask=None)` — lines 78–92

```python
vec = fcd_to_upper_tri(fcd_matrix, nan_mask=nan_mask)   # (6555,) float32
return np.array([
    vec.mean(),
    vec.std(),
    float(np.percentile(vec, 25)),
    float(np.percentile(vec, 50)),
    float(np.percentile(vec, 75)),
], dtype=np.float32)                                     # (5,) float32
```

**Rationale for 5 summary stats instead of full 6555-dim FCD:**
From the function docstring: "avoids the poor PCA explained-variance caused by raw FCD
spread." The FCD values (σ of FC across windows) have high inter-dimensional correlation
and poor PCA structure on small training sets (4 subjects × N_SIM simulations). Using
5 summary statistics avoids fitting a second PCA on noisy 6555-dim data.

This is reflected in `config.py`: `FCD_DIM = 5`, and `PCA_DIM_FCD = 100` exists as a
constant (likely a legacy or future-use value for full FCD PCA) but is **not used** in
`FeaturePipeline` — the pipeline applies `FamilyScaler` z-score to the 5-dim FCD vector,
not PCA.

**Also note:** `fcd_to_upper_tri` is re-exported in `features/__init__.py` and by
`simulator.py`, so it is part of the public API. However, it is not used by any production
code path (`fc_only` mode). Its callers are in debug tools and evaluation code via deferred
`from simulator import fcd_to_upper_tri`.

---

## 7. Feature Scaling Logic

Three distinct scaling objects, each with a specific scope:

### 7.1 No scaling on raw FC

`step4_fit_feature_scalers` in `inference/snpe.py` explicitly sets `fc_z = None`:
```python
fc_z = None  # FC: no z-score (already Pearson r in [-1, 1])
```

**This resolves a discrepancy in `04_data_flow.md`**, which describes "z-score → PCA" for
FC in Steps 4→5. The code does NOT z-score FC before PCA. The data-flow doc is inaccurate
on this point. FC goes raw into PCA.

### 7.2 `FamilyScaler` — per-feature z-score (FCD only, currently disabled)

```python
class FamilyScaler:
    def fit(self, x_train):
        self.mean_ = x_train.mean(axis=0, keepdims=True)   # (1, D)
        self.std_  = x_train.std(axis=0, keepdims=True)    # (1, D)
        self.std_  = where(std < 1e-8, 1.0, std)           # guard zero-var dims
    def transform(self, x):
        return (x - self.mean_) / self.std_
```

- Fit only on training simulations (R10).
- Used for FCD summary stats (5-dim) when `config.USE_FCD=True`.
- Currently inactive: `USE_FCD=False` → `FeaturePipeline.fcd_z = None`.
- Clips constant-variance dimensions to std=1.0 (safe for the 5 FCD summary stats since
  they typically have nonzero variance across N_SIM=50,000 simulations).

**Note:** `step4_fit_feature_scalers` in `inference/snpe.py` creates its own standalone
`FamilyScaler("FCD")` for logging/diagnostic purposes, separate from
`FeaturePipeline.fcd_z`. When `USE_FCD=True`, both are fit on the same data. The step4
scaler is stored in the returned dict `{"fc_z": None, "fcd_z": fcd_z}` but is not used
by `FeaturePipeline.transform()` — the pipeline uses its own internal `fcd_z`. This is
a **redundant fit** when FCD is enabled.

### 7.3 `FCPCAScaler` — FC raw → PCA (active)

```python
class FCPCAScaler:
    def fit(self, fc_train_raw):
        n_comp = min(self.n_components, n_samples, n_features)   # cap silently
        self.pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
        self.pca.fit(fc_train_raw)
    def transform(self, fc_raw):
        return self.pca.transform(fc_raw).astype(float32)
```

- Input: `(N_SIM, 6555)` raw FC upper-tri vectors (Pearson r, not z-scored)
- Output: `(N_SIM, 300)` PCA projections
- `n_components = config.PCA_DIM_FC = 300`; capped to `min(300, n_samples, n_features)`
- Uses sklearn randomized SVD: ~30× faster than full SVD for 300 << 6555
- Has `inverse_transform` for reconstruction quality check
- Has `diagnostic()` method checking EVR, reconstruction corr, train/val distribution shift

### 7.4 `FeaturePipeline` — combines FC PCA + FCD z-score

```
fc_raw  (N_SIM, 6555)  →  FCPCAScaler   → (N_SIM, 300)  ──┐
fcd_raw (N_SIM, 5)    →  FamilyScaler  → (N_SIM, 5)    ──┤ concatenate
                                                            └→ (N_SIM, 305)  [fc_fcd]
                                                            or (N_SIM, 300)  [fc_only]
```

Dimension mismatch protection:
- `transform()` raises `ValueError` if `fc_raw.shape[1] != self.fc_dim` (fitted dim)
- `transform()` raises `ValueError` if `fcd_raw.shape[1] != self.fcd_dim`
- Post-concat assert: `out.shape[1] != self.input_dim`
- Never broadcasts, pads, or silently truncates (design intent)

---

## 8. Feature Embedding Connection

**File:** `inference/embedding.py`

```python
class FeatureEmbedding(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=None, out_dim=None):
        # hidden_dim = config.EMBED_HIDDEN = 512
        # out_dim    = config.EMBED_DIM    = 128
        self.net = Sequential(
            Linear(input_dim, 512),
            ReLU(),
            Dropout(0.1),
            Linear(512, 256),      # 512 // 2
            ReLU(),
            Linear(256, 128),
        )
```

**Input:** `x_input` from `FeaturePipeline.transform()` — shape `(N_SIM, 300)` float32
(or `(N_SIM, 305)` if FCD enabled)

**Output:** `(N_SIM, 128)` float32 — the final representation passed to SNPE-C's MAF

**Training:** `FeatureEmbedding` is jointly trained with the SNPE-C density estimator
inside `step8_train_snpe`. The `embedding_net` is passed to `posterior_nn(embedding_net=...)`.
The MLP weights are optimised by the SNPE-C objective. The pipeline (FC PCA + FCD z-score)
is fitted before SNPE training and frozen — only the embedding MLP is end-to-end trained.

**Torch guard:** The module-level `try: import torch` guard means `inference/embedding.py`
can be imported on torch-less systems without error. Instantiating `FeatureEmbedding` will
raise `ImportError` if torch is absent, but the module itself loads cleanly.

**Connection to `FeaturePipeline`:**

```
FeaturePipeline fitted → x_input (N_SIM, 300) float32
                                      ↓
                    FeatureEmbedding(input_dim=x_input.shape[1])
                                      ↓ jointly trained
                    embedding (N_SIM, 128)
                                      ↓
                    SNPE-C MAF posterior p(θ | x)
```

The `input_dim` of `FeatureEmbedding` is set dynamically from `x_input.shape[1]` in
`step8_train_snpe` — it is not hardcoded. This means if `PCA_DIM_FC` or FCD mode changes,
the embedding architecture changes automatically.

---

## 9. FC PCA — Implementation Status

**Status: Fully implemented and active.**

`FCPCAScaler` in `inference/feature_pipeline.py` is complete:

| Capability | Status |
|---|---|
| `fit(fc_train_raw)` | Implemented — sklearn PCA, randomized SVD, verbose output |
| `transform(fc_raw)` | Implemented — projects to PC space, returns float32 |
| `inverse_transform(fc_pca)` | Implemented — for reconstruction diagnostic |
| `diagnostic(fc_train_raw, fc_val_raw)` | Implemented — EVR, reconstruction corr, train/val shift |
| `n_components` auto-cap | Implemented — `min(300, n_samples, n_features)` |
| Threshold checks | `PCA_EVR_THRESHOLD = 0.90`, `PCA_RECON_CORR_THRESH = 0.95` |

`FeaturePipeline` wraps `FCPCAScaler` and is the object stored in inference artifacts:
```python
pipeline = FeaturePipeline()
pipeline.fit(fc_raw, fcd_raw)    # → fits FCPCAScaler internally
x_input = pipeline.transform(fc_raw, fcd_raw)
```

**Note:** `config.PCA_DIM_FCD = 100` is defined but is not connected to any code path.
`FeaturePipeline` uses `FamilyScaler` z-score (not PCA) for FCD. This constant is either
a legacy remnant or reserved for a future FCD-PCA path.

---

## 10. FCD Summary Z-Scoring — Implementation Status

**Status: Fully implemented; currently inactive (`USE_FCD=False`).**

`FeaturePipeline` creates `FamilyScaler("FCD")` conditioned on `config.USE_FCD`:
```python
self.fcd_z = FamilyScaler(name="FCD") if config.USE_FCD else None
```

When `USE_FCD=True`:
- `pipeline.fit(fc_raw, fcd_raw)` fits z-score on `fcd_raw (N_SIM, 5)` — training data only
- `pipeline.transform(fc_raw, fcd_raw)` applies z-score to FCD, concatenates with FC PCA
- Output dimension: `300 + 5 = 305`

When `USE_FCD=False` (current production):
- `fcd_raw = np.zeros((N_SIM, 0), dtype=float32)` (from `extract_simulated_features`)
- `fcd_z = None` — no z-score computation
- Output dimension: `300`

**To enable FCD** (not currently recommended — see `config.py` comment truncation at line 231):
```python
# In config.py:
USE_FCD = True
FEATURE_SET = "fc_fcd"
```
AND provide empirical BOLD or FCD data for observed subjects. Without empirical BOLD,
`extract_observed_features` will raise `ValueError`.

---

## 11. Root-Level `fc.py`, `fcd.py`, `extraction.py`, `screening.py` Usage

### 11.1 Identity confirmed

All four root files are **byte-for-byte identical** to their `features/` package counterparts
(confirmed by `diff` exit 0 in `11_import_audit.md`).

### 11.2 Internal import behaviour

| Root file | Internal imports | Behaviour when bare-imported |
|---|---|---|
| `fc.py` | `import numpy as np`, `import config` only | Fully self-contained; no package delegation |
| `fcd.py` | `import numpy as np`, `import config` only | Fully self-contained; no package delegation |
| `screening.py` | `import numpy as np`, `import config` only | Fully self-contained; no package delegation |
| `extraction.py` | `from features.fc import …`, `from features.fcd import …` | Delegates to `features.fc` and `features.fcd` — NOT self-contained |

**Note:** `fc.py`, `fcd.py`, and `screening.py` are true standalone copies — if someone
imports `import fc`, they get a fully functional module with no link to `features.fc`. Since
both share the same `config` module object, `NAN_MASK` and other runtime mutations
propagate equally to both. Content divergence between root copies and package files
remains impossible to detect at runtime — it would only appear as a `diff` mismatch.

`extraction.py` (root), by contrast, is NOT fully standalone: it imports from `features.fc`
and `features.fcd`. If someone bare-imports `import extraction`, Python loads root
`extraction.py` which in turn loads `features/fc.py` and `features/fcd.py`. Both root and
package modules would then be live simultaneously.

### 11.3 Active caller status

```bash
grep -rn "^import fc\b\|^from fc import\|^import fcd\b\|^from fcd import"  → zero
grep -rn "^import extraction\b\|^import screening\b"                        → zero
```

**No file currently imports any of these root files via bare name.** They are inert.

### 11.4 `simulator.py` re-exports

`simulator.py` (the compat wrapper) re-exports all four packages' symbols:

```python
from features.fc import (compute_fc, fc_to_upper_tri)
from features.fcd import (compute_sim_fcd_matrix, fcd_to_summary_stats, fcd_to_upper_tri)
from features.extraction import (
    extract_features, extract_observed_features,
    extract_simulated_features, worker_extract,
)
```

These go directly to the `features/` package, NOT to the root duplicates. The root
duplicates are completely bypassed by `simulator.py`.

---

## 12. Import Conflict Risks

### RISK-1 — `simulation/qc.py` imports from `features.fc` (R3 violation, active)

```python
# simulation/qc.py lines 25-26
from features.fc import compute_fc, fc_to_upper_tri
from simulation.wc_runner import simulate_gpu_batch
```

This violates R3: `simulation/` must not import from `features/`. Documented in
`12_simulation_module_review.md`. The consequence is that `import simulation` (or
`from simulation import anything`) transitively loads `features/fc.py`. This makes
`features.fc` a silent boot-time dependency of the `simulation` package, which could
complicate future packaging or layered testing.

### RISK-2 — `extract_features` vs `extract_simulated_features` FCD shape asymmetry

`worker_extract` (the parallel-worker wrapper) calls `extract_features`, which always
returns a **5-element** FCD vector:
```python
fcd_vec = np.zeros(5, dtype=np.float32)   # always 5 zeros if USE_FCD=False
```

`extract_simulated_features` (the FEATURE_SET-aware function) returns a **0-element** FCD:
```python
return fc_vec, np.zeros(0, dtype=np.float32)   # if fc_only or USE_FCD=False
```

`inference/training_data.py` uses `worker_extract` for the parallel path (ProcessPool),
which returns 5-element FCD zeros. If any code path then calls
`FeaturePipeline.fit(fc_raw, fcd_raw)` with a `fcd_raw` built from `extract_features`
output, `fcd_raw` will have shape `(N_SIM, 5)`. But `FeaturePipeline.use_fcd = False`
(because `USE_FCD=False`), so `self.fcd_dim = 0` and `fcd_raw` is ignored in
`fit()` and `transform()`. This makes the asymmetry harmless in production.

**However:** if someone enables `USE_FCD=True` without updating `worker_extract` to call
`extract_simulated_features` instead of `extract_features`, the parallel worker would
return 5-dim FCD zeros while the FEATURE_SET-aware path expects non-zero FCD. The
dimension would match (both 5), but the parallel path would inject all-zeros FCD into
training data. This would be a silent data-corruption bug.

### RISK-3 — `NAN_MASK` mutation can break fitted pipelines

`config.NAN_MASK` is set at data-load time by `data_loader._record_nan_mask(fc_mat)`. If:
1. Pipeline is fit on `fc_raw` with shape `(N_SIM, D)` where D < 6555 (some NaN entries masked)
2. `config.NAN_MASK` is later mutated to None (e.g., by reload / second dataset)
3. New `fc_to_upper_tri` calls return `(6555,)` vectors
4. `FeaturePipeline.transform()` raises `ValueError` (fc dim mismatch)

This is mitigated by the dimension-lock check in `FeaturePipeline.transform()` — it will
raise rather than silently produce wrong embeddings. But the error message references
`fc_dim` stored at fit time, not mentioning NAN_MASK, which could confuse diagnosis.

### RISK-4 — Root `extraction.py` loads `features.fc` and `features.fcd` simultaneously

If root `extraction.py` is ever bare-imported, Python will:
1. Execute `from features.fc import …` → load `features/fc.py`
2. Execute `from features.fcd import …` → load `features/fcd.py`
Both root-level `fc.py` / `fcd.py` and package `features/fc.py` / `features/fcd.py` would
then coexist in `sys.modules` as separate objects (since their module names differ:
`fc` vs `features.fc`). If code then mix-imports from both, computed FC arrays from one
path would not be "is" equal to the other, potentially confusing identity checks (rare but
possible in test code).

### RISK-5 — `config.PCA_DIM_FCD = 100` is dangling (documentation risk)

This constant exists in `config.py` and is printed by `config.print_config()`:
```python
print(f"  PCA: FC -> {PCA_DIM_FC}, FCD -> {PCA_DIM_FCD}")
```

But `FeaturePipeline` does NOT use `PCA_DIM_FCD = 100`. FCD uses 5-dim summary stats with
z-score, not PCA. A developer reading `config.print_config()` output would see "FCD -> 100"
and incorrectly assume FCD goes through PCA. This is a misleading configuration printout.

---

## 13. Feature Dimension Assumptions

| Quantity | Config value | Source | Fragility |
|---|---|---|---|
| N (brain regions) | 115 | `config.N_REGIONS` | Hardcoded; changing requires re-running everything |
| FC_DIM (full upper tri) | 6555 | `config.FC_DIM = N*(N-1)/2` | Assert in config; safe |
| FC_DIM (actual, post-NAN_MASK) | ≤ 6555 | `fc_to_upper_tri()` runtime | Changes silently if NAN_MASK set; pipeline stores actual dim |
| FCD summary stats dim | 5 | `config.FCD_DIM = 5`, hardcoded in `fcd_to_summary_stats` | Fragile: adding/removing a stat would silently break FCD pipeline |
| FCD upper-tri dim | 6555 | same as FC | Only used in `fcd_to_upper_tri`, not in pipeline |
| PCA output dim (FC) | 300 | `config.PCA_DIM_FC` | Capped to `min(300, n_samples, n_features)` |
| PCA output dim (FCD) | 100 | `config.PCA_DIM_FCD` | **NOT USED** by FeaturePipeline — dangling constant |
| Pipeline output (fc_only) | 300 | PCA_DIM_FC | Active |
| Pipeline output (fc_fcd) | 305 | PCA_DIM_FC + FCD_DIM | Inactive (USE_FCD=False) |
| Embedding input | 300 or 305 | `x_input.shape[1]` | Set dynamically — safe |
| Embedding hidden | 512 | `config.EMBED_HIDDEN` | |
| Embedding output | 128 | `config.EMBED_DIM` | |
| Sliding-window count | 61 | derived from T=240, W=60, S=3 | If T_bold changes, window count changes |
| N_SIM (training budget) | 50,000 | `config.N_SIM` | GPU_BATCH=50,000 — one chunk per subject |

### The 5-stat FCD vector — fragile hardcoding

`fcd_to_summary_stats` returns exactly 5 values: `[mean, std, q25, q50, q75]`. This is
hardcoded — there is no config constant controlling the number of FCD summary statistics.
`config.FCD_DIM = 5` is merely documentation. If a future developer adds a 6th statistic
(e.g., kurtosis) to `fcd_to_summary_stats`, the FCD feature vector becomes 6-dim, silently
breaking any fitted `FeaturePipeline.fcd_dim = 5` that was saved to disk. The dimension
mismatch guard in `FeaturePipeline.transform` would catch this at transform time, but only
after load — the error would appear as an artifact-incompatibility rather than an API change.

---

## 14. Minimal Future Refactor Plan for `features/`

### Step F1 — Resolve `extract_features` / `extract_simulated_features` FCD shape asymmetry (Priority: Medium, before enabling FCD)

**Problem (RISK-2):** `worker_extract` calls `extract_features` (returns 5-zero FCD), while
`extract_simulated_features` returns 0-zero FCD in `fc_only` mode. This is harmless today
but will silently inject all-zeros FCD into parallel-worker training data if FCD is enabled
without updating `worker_extract`.

**Fix:** Update `worker_extract` to call `extract_simulated_features` instead of
`extract_features`:
```python
def worker_extract(bold):
    try:
        return extract_simulated_features(bold)
    except Exception:
        return None
```

After this change, `extract_features` (legacy 5-element FCD path) can be deprecated. Do
not delete it until `inference/training_data.py:22` comment is updated.

### Step F2 — Remove dangling `config.PCA_DIM_FCD = 100` or wire it in (Priority: Low)

**Problem (RISK-5):** `config.PCA_DIM_FCD = 100` is printed but not used. Misleads readers.

**Option A (recommended):** Remove `PCA_DIM_FCD` from `config.py` and the print statement.
If full FCD PCA is ever needed, add the constant back at that time.

**Option B:** Wire `PCA_DIM_FCD` into `FeaturePipeline` as an alternative FCD handling mode.

### Step F3 — Clarify `04_data_flow.md` Step 4 description (Priority: Low, documentation)

`04_data_flow.md` states: `fc_raw → z-score → PCA`. The code does NOT z-score FC before
PCA. Update the data-flow doc to match the actual code:
- Step 4: FC → no scaling; FCD → z-score (if `USE_FCD=True`)
- Step 5: FC → PCA → 300 PCs; FCD (z-scored) appended as-is → 305 dims total (if enabled)

### Step F4 — Remove root duplicate feature files (Priority: Low, after import cleanup)

All four root duplicates are byte-for-byte identical and have zero bare-name callers.
Safe to remove after `simulator.py` compat wrapper is deleted (Tier 5 of
`11_import_audit.md`). Deletion order from `11_import_audit.md`:
1. `screening.py` — future stub, no logic beyond identity_screen
2. `fcd.py` — production is FC-only
3. `fc.py` — after confirming no bare `import fc` in notebooks
4. `extraction.py` — delegates to features.* anyway; delete last since it has package imports

### Step F5 — Resolve `simulation/qc.py` R3 violation (Priority: Medium)

Documented in `12_simulation_module_review.md` Step S1. From `features/` perspective:
if `simulation/qc.py` is moved outside `simulation/`, the `features` package no longer
becomes a transitive dependency of `import simulation`. The preferred resolution is either
a comment explaining the exception or moving `qc.py` to a `checks/` or `tests/` module.

### Step F6 — Add NAN_MASK handling documentation (Priority: Low)

`fc_to_upper_tri` can produce vectors shorter than `config.FC_DIM = 6555`. The pipeline
handles this correctly (stores actual dim at fit time), but no documentation explains when
FC_DIM < 6555 can occur and what the downstream effects are. Add a note to the pipeline
fit method or to `11_import_audit.md`'s section on `FeaturePipeline`.

---

## 15. Test Commands for `features/`

### 15.1 Compile check (no GPU required)

```bash
python -m py_compile features/__init__.py features/fc.py features/fcd.py \
    features/extraction.py features/screening.py
python -m py_compile inference/feature_pipeline.py inference/scaling.py \
    inference/embedding.py
echo "features/ and feature pipeline compile OK"
```

### 15.2 Import chain test (no GPU required)

```bash
python -c "
from features.fc import compute_fc, fc_to_upper_tri
from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri, fcd_to_summary_stats
from features.extraction import (
    extract_features, extract_observed_features,
    extract_simulated_features, worker_extract,
)
from features.screening import identity_screen, apply_mask
from inference.feature_pipeline import FamilyScaler, FCPCAScaler, FeaturePipeline
from inference.scaling import ParameterScaler, make_stage1_param_scaler
from inference.embedding import FeatureEmbedding
print('all feature imports OK')
"
```

### 15.3 Dependency rule check (R3)

```bash
# features/ must not import from simulation/, inference/, evaluation/
grep -rn "from simulation\|from inference\|from evaluation" features/*.py
# Expected: zero output

# This confirms the R3 violation is in simulation/qc.py, not in features/:
grep -rn "from features" simulation/qc.py
# Expected: "from features.fc import compute_fc, fc_to_upper_tri"
```

### 15.4 FC computation unit test (no GPU)

```bash
python -c "
import numpy as np, config
from features.fc import compute_fc, fc_to_upper_tri

N = config.N_REGIONS
T = config.ANALYSIS_BOLD_T
rng = np.random.RandomState(0)
bold = rng.randn(T, N).astype(np.float32)

fc = compute_fc(bold)
assert fc.shape == (N, N), f'fc shape: {fc.shape}'
assert np.all(np.isfinite(fc)), 'fc has inf/nan'
assert fc[0, 0] == 0.0, 'diagonal not zeroed'
assert abs(fc[1, 2] - fc[2, 1]) < 1e-6, 'fc not symmetric'

vec = fc_to_upper_tri(fc)
expected_dim = N * (N - 1) // 2
assert vec.shape == (expected_dim,), f'fc_vec shape: {vec.shape}'
assert vec.dtype == np.float32
assert vec.min() >= -1.0 and vec.max() <= 1.0, f'fc_vec out of [-1,1]: {vec.min()},{vec.max()}'
print(f'FC test OK: fc={fc.shape}, vec={vec.shape}, range=[{vec.min():.3f},{vec.max():.3f}]')
"
```

### 15.5 FCD computation unit test (no GPU)

```bash
python -c "
import numpy as np, config
from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri, fcd_to_summary_stats

N = config.N_REGIONS
T = config.ANALYSIS_BOLD_T
rng = np.random.RandomState(1)
bold = rng.randn(T, N).astype(np.float32)

fcd = compute_sim_fcd_matrix(bold)
assert fcd.shape == (N, N), f'fcd shape: {fcd.shape}'
assert fcd.dtype == np.float32
assert np.all(fcd >= 0), 'fcd std should be non-negative'
assert fcd[0, 0] == 0.0, 'fcd diagonal not zeroed'
assert abs(fcd[1, 2] - fcd[2, 1]) < 1e-5, 'fcd not symmetric'

stats = fcd_to_summary_stats(fcd)
assert stats.shape == (5,), f'fcd summary shape: {stats.shape}'
assert stats.dtype == np.float32
print(f'FCD test OK: fcd={fcd.shape}, stats={stats.tolist()}')
"
```

### 15.6 Feature extraction consistency test (no GPU)

```bash
python -c "
import numpy as np, config
from features.extraction import (
    extract_features, extract_observed_features,
    extract_simulated_features, worker_extract,
)

N = config.N_REGIONS
T = config.ANALYSIS_BOLD_T
rng = np.random.RandomState(2)
bold = rng.randn(T, N).astype(np.float32)
fc_obs = np.corrcoef(bold.T).astype(np.float32)

# Test extract_simulated_features (fc_only mode)
fc_vec, fcd_vec = extract_simulated_features(bold)
assert fc_vec.shape == (config.FC_DIM,), f'simulated fc_vec: {fc_vec.shape}'
assert fcd_vec.shape == (0,), f'simulated fcd_vec should be (0,) in fc_only: {fcd_vec.shape}'

# Test extract_observed_features (fc_only mode)
subject = {'fc': fc_obs}
fc_vec_obs, fcd_vec_obs = extract_observed_features(subject)
assert fc_vec_obs.shape == (config.FC_DIM,), f'observed fc_vec: {fc_vec_obs.shape}'
assert fcd_vec_obs.shape == (0,), f'observed fcd_vec: {fcd_vec_obs.shape}'

# Test worker_extract (legacy — returns 5-element FCD zeros)
result = worker_extract(bold)
assert result is not None, 'worker_extract returned None'
fc_w, fcd_w = result
assert fcd_w.shape == (5,), f'worker FCD should be (5,) [ASYMMETRY]: {fcd_w.shape}'
print(f'Extraction test OK: simulated/observed FCD=(0,), worker FCD=(5,) [asymmetry confirmed]')
"
```

### 15.7 FeaturePipeline round-trip test (no GPU)

```bash
python -c "
import numpy as np, config
from inference.feature_pipeline import FeaturePipeline

N_SIM = 200
FC_DIM = config.FC_DIM
rng = np.random.RandomState(3)
fc_train = rng.uniform(-1, 1, (N_SIM, FC_DIM)).astype(np.float32)
fcd_train = np.zeros((N_SIM, 0), dtype=np.float32)   # fc_only

pipeline = FeaturePipeline()
x_train = pipeline.fit_transform(fc_train, fcd_train)
assert x_train.shape == (N_SIM, config.PCA_DIM_FC), f'x_train shape: {x_train.shape}'

# Dimension-lock test: wrong FC dim should raise
import traceback
try:
    pipeline.transform(fc_train[:, :100], fcd_train)
    print('ERROR: should have raised ValueError')
except ValueError as e:
    print(f'Dimension guard OK: {str(e)[:60]}...')

diag = pipeline.diagnostic(fc_train)
print(f'Pipeline test OK: output={x_train.shape}, EVR={diag[\"fc_pca\"][\"explained_variance_sum\"]:.2%}')
"
```

### 15.8 ParameterScaler round-trip test

```bash
python -c "
import numpy as np, config
from inference.scaling import make_stage1_param_scaler

scaler = make_stage1_param_scaler()
raw = np.array([1.5, 1.0, 0.5, 0.5], dtype=np.float32)  # P, Q, g_e, g_i
scaled = scaler.transform(raw)
recovered = scaler.inverse_transform(scaled)
assert np.allclose(raw, recovered, atol=1e-5), f'round-trip failed: {raw} vs {recovered}'
assert scaled.min() >= -1.0 and scaled.max() <= 1.0, 'scaled out of [-1,1]'
print(f'ParameterScaler OK: {raw} -> {scaled.tolist()} -> {recovered.tolist()}')
"
```

### 15.9 Verify root duplicate integrity

```bash
diff fc.py features/fc.py       && echo "fc: identical"
diff fcd.py features/fcd.py     && echo "fcd: identical"
diff extraction.py features/extraction.py && echo "extraction: identical"
diff screening.py features/screening.py   && echo "screening: identical"
```

---

## Final Assessment

### Core feature files (authoritative, do not delete)

| File | Status | Notes |
|---|---|---|
| `features/fc.py` | **Core.** Active in every pipeline run. | Implements `compute_fc`, `fc_to_upper_tri`. No z-score — raw Pearson r. |
| `features/fcd.py` | **Core (dormant).** Ready for activation when BOLD available. | `USE_FCD=False` in production. All logic is complete. |
| `features/extraction.py` | **Core.** Used by `inference/training_data.py` via `worker_extract`. | Contains shape asymmetry (Step F1 to fix). |
| `features/screening.py` | **Core (placeholder).** Only `identity_screen` and `apply_mask` exist; no real screen implemented. | Deletion would break any future caller; keep as architectural home. |
| `features/__init__.py` | **Core.** Re-export hub for the `features.*` public API. | |
| `inference/feature_pipeline.py` | **Core.** `FeaturePipeline`, `FCPCAScaler`, `FamilyScaler`. | Lives in `inference/` by design (R10 — train-only fit). |
| `inference/scaling.py` | **Core.** `ParameterScaler` — bridge between raw params and SBI [-1,1] space. | Data-free; safe to use everywhere. |
| `inference/embedding.py` | **Core.** `FeatureEmbedding` MLP — jointly trained with SNPE-C. | Torch-guarded import. |

### Root-level legacy candidates (mark for eventual deletion)

| File | Basis |
|---|---|
| `fc.py` | Byte-identical to `features/fc.py`; zero bare-name callers; all production imports use `features.fc` |
| `fcd.py` | Byte-identical to `features/fcd.py`; zero bare-name callers; FCD disabled in production |
| `extraction.py` | Byte-identical to `features/extraction.py`; delegates to `features.*` internally; zero bare-name callers |
| `screening.py` | Byte-identical to `features/screening.py`; stub only; zero bare-name callers |

### Files that must NOT be deleted yet

| File | Reason |
|---|---|
| `simulator.py` | Re-exports `compute_fc`, `fc_to_upper_tri`, `extract_observed_features`, `worker_extract`, and others — still has active deferred callers in `inference/` and `evaluation/`. Cannot delete until Tier 1–4 of `11_import_audit.md` complete. |
| `features/extraction.py` | Contains `worker_extract` with the `extract_features` asymmetry (RISK-2). Must fix Step F1 before this path changes, since it is the parallel-worker entry for `inference/training_data.py`. |
| `features/fcd.py` | Not a legacy file — the full FCD path is disabled but complete; deleting would destroy the activation path. |
| `features/screening.py` | Placeholder but the API contract is part of the architecture. |

### What should be inspected next

In priority order:

1. **`inference/training_data.py`** — the file that calls `worker_extract` in
   `ProcessPoolExecutor`. Confirms the actual FCD dimension seen during training (5-zero
   vs 0-zero), and shows how `fc_raw` / `fcd_raw` arrays are assembled from parallel
   worker output. Reading this file closes the loop on RISK-2.

2. **`inference/stage1.py` and `inference/stage2.py`** — these chain steps 2–8 and call
   `step5_fit_feature_pipeline` / `step8_train_snpe`. Seeing how step4's returned
   `{"fc_z": None, "fcd_z": fcd_z}` is actually used (or not) will confirm whether the
   step4/step5 FamilyScaler redundancy is real.

3. **`evaluation/metrics.py`** — imports `extract_observed_features` (via deferred
   `from simulator import`). Understanding how observed features are used during evaluation
   will confirm whether `NAN_MASK` is consistently applied at eval time.

4. **`config.py` full read** — to verify the precise values of `FCD_WINDOW_TR`,
   `FCD_STRIDE_TR`, `PCA_DIM_FC`, `PCA_EVR_THRESHOLD`, `PCA_RECON_CORR_THRESH` and
   whether `HAS_BOLD` / `USE_FCD` interaction at runtime is handled correctly.
