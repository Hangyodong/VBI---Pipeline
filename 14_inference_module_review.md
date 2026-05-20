# 14 — Inference Module Review

**Date:** 2026-05-18
**Branch:** refactor/02-simulation
**Reviewer:** Claude (claude-sonnet-4-6)
**Files inspected:**
`inference/__init__.py`, `inference/priors.py`, `inference/snpe.py`,
`inference/stage1.py`, `inference/stage2.py`, `inference/posterior.py`,
`inference/training_data.py`, `inference/scaling.py`, `inference/embedding.py`,
`inference/diagnostics.py`, `inference/io.py`, `inference/feature_pipeline.py`,
`inference/_utils.py`, `inference.py` (root monolith)
**Cross-referenced:** `11_import_audit.md`, `13_feature_module_review.md`,
`03_module_index.md`, `04_data_flow.md`, `07_refactor_rules.md`

---

## 1. Overall Inference Flow

The full two-stage inference pipeline, from prior sample to posterior, runs in this order:

```
Step 7: step7_fit_param_scaler()
          ParameterScaler(STAGE1_PARAMS, low, high)   [data-free]
          prior_scaled = make_scaled_prior(4)          BoxUniform([-1,1]^4)

Step 2: step2_simulate_train(train_subjects, subject_data, prior_scaled, param_scaler)
          collect_training_data(...)
            theta_s ~ prior_scaled.sample(n_sim)       (N_SIM, 4)  scaled
            theta_r = scaler.inverse_transform(theta_s) (N_SIM, 4)  raw
            BOLD = simulate_gpu_batch(sc, theta_r, ...)  list of (T,N) arrays  [GPU]
            worker_extract(BOLD) via ProcessPoolExecutor → (fc_vec, fcd_vec)    [CPU]
            _drain_one_future() filters None + non-finite
          → theta_s (N_SUBJ*N_SIM, 4), theta_r, fc_raw (N, 6555), fcd_raw (N, 5 or 0)

Step 3: step3_summary_features(fc_raw, fcd_raw)        [diagnostic print only]

Step 4: step4_fit_feature_scalers(fc_raw, fcd_raw)
          fc_z  = None  [FC already Pearson r in [-1,1], no z-score needed]
          fcd_z = FamilyScaler("FCD").fit(fcd_raw)  if USE_FCD else None
          → {"fc_z": None, "fcd_z": ...}  [diagnostic only, NOT stored in artifacts]

Step 5: step5_fit_feature_pipeline(fc_raw, fcd_raw)
          pipeline = FeaturePipeline()
          pipeline.fit(fc_raw, fcd_raw)
            FCPCAScaler.fit(fc_raw)  PCA 6555 → 300 (no z-score on FC)
            FamilyScaler.fit(fcd_raw)  z-score on 5-dim FCD (if USE_FCD)
          x_input = pipeline.transform(fc_raw, fcd_raw)  (N, 300) or (N, 305)

Step 6: step6_pca_diagnostic(pipeline, fc_raw, fcd_raw)
          EVR, reconstruction corr, train/val shift  [diagnostic dict]

Step 8: step8_train_snpe(theta_s, x_input, prior_scaled)
          train_snpe(...)
            FeatureEmbedding(input_dim=300)  [MLP jointly trained with SNPE-C]
            SNPE_C(prior=prior_scaled)
            inferer.append_simulations(theta_scaled, x_input)
            inferer.train(batch=2048, stop_after_epochs=20, max_epochs=300)
            posterior = inferer.build_posterior(estimator)
          → (posterior, embedding_net)

─────────────────────────── inference time ───────────────────────────

Observed subject:
  fc_obs_raw, fcd_obs_raw = extract_observed_features(subject_data[sid])
  x_obs = pipeline.transform(fc_obs_raw, fcd_obs_raw)   (300,) float32

Posterior sampling:
  samples_scaled = posterior.sample(N_POSTERIOR, x=x_obs)  (2000, 4)
  samples_raw    = param_scaler.inverse_transform(samples_scaled)
  → posterior mean/std in raw parameter units

Stage 2 (if run_stage2_snpe called):
  → θ_bad = select_theta_bad(sensitivity, shrinkage, ...)
  → stage2_params = θ_bad + LOCAL_EI_PARAMS (c_ee, c_ei, c_ie, c_ii)
  → new prior, scaler, simulations with nuisance from Stage 1 posterior
  → new FeaturePipeline + new SNPE-C trained on Stage 2 data
```

**Step ordering note:** Step 7 runs *before* Step 2 in `run_stage1_snpe` (contra the step numbering). The parameter scaler must exist before simulation starts to inverse-transform prior samples into raw space. The step numbers in the code reflect a notebook ordering where steps 1–6 could be inspected independently; the driver function reorders for correctness.

---

## 2. Stage 1 Inference Role

**Entry point:** `inference/stage1.py:run_stage1_snpe`

Stage 1 infers 4 Wilson-Cowan biophysical parameters: **P, Q, g_e, g_i**.

```
Prior bounds (config.STAGE1_PRIOR_LOW / HIGH):
  P    ∈ [0.5, 2.5]   external excitatory drive
  Q    ∈ [0.0, 2.0]   external inhibitory drive
  g_e  ∈ [0.0, 1.5]   global excitatory gain
  g_i  ∈ [0.0, 1.5]   global inhibitory gain

n_sim = config.N_SIM = 50,000 per training subject
```

`run_stage1_snpe` is a pure wrapper that chains steps 7→2→3→4→5→6→8 and returns a dict:

```python
{
    "posterior":        sbi SNPE-C posterior  (amortized over all train subjects)
    "embedding_net":    FeatureEmbedding       MLP jointly trained
    "theta_scaled":     (N_SUBJ*N_SIM, 4)     training thetas in [-1,1]
    "theta_raw":        (N_SUBJ*N_SIM, 4)     training thetas in raw units
    "fc_raw":           (N_SUBJ*N_SIM, 6555)  raw FC upper-tri vectors
    "fcd_raw":          (N_SUBJ*N_SIM, 0)     empty (USE_FCD=False)
    "x_input":          (N_SUBJ*N_SIM, 300)   pipeline-transformed features
    "param_scaler":     ParameterScaler       for inverse-transform at inference
    "feature_pipeline": FeaturePipeline       for transforming observed FC
    "prior_scaled":     BoxUniform([-1,1]^4)
    "pca_diagnostic":   dict                  EVR, reconstruction corr
}
```

The artifacts dict is the handoff to `evaluation/` and Stage 2. Every downstream function
that samples the posterior needs `posterior`, `param_scaler`, and `feature_pipeline` from
this dict.

---

## 3. Stage 2 Inference Role

**Entry point:** `inference/stage2.py:run_stage2_snpe`

Stage 2 re-infers parameters that were **sensitive** (high output effect) but **poorly
identified** (low shrinkage) in Stage 1, plus fixed local E/I coupling constants.

```
θ_bad  = select_theta_bad(sensitivity, shrinkage,
                          sens_threshold=0.5,
                          shrinkage_threshold=0.2)
       = {name | sensitivity[name] >= 0.5 AND shrinkage[name] < 0.2}

stage2_params = θ_bad + LOCAL_EI_PARAMS
LOCAL_EI_PARAMS = ["c_ee", "c_ei", "c_ie", "c_ii"]
  Prior bounds (config.C_PARAM_PRIOR):
    c_ee ∈ [12, 20],  c_ei ∈ [8, 16]
    c_ie ∈ [10, 20],  c_ii ∈ [1, 6]
```

**Nuisance handling:** Stage 1 parameters NOT in θ_bad are treated as nuisance. Controlled
by `config.NUISANCE_METHOD`:
- `"posterior_sample"` (default): each Stage 2 simulation samples nuisance values from
  the Stage 1 posterior for that training subject. Requires running Stage 1 posterior
  forward `n_sim` times per subject.
- `"fix_mean"`: all Stage 2 simulations for a subject use the Stage 1 posterior mean as
  nuisance values. 1000 Stage 1 samples used to compute the mean.

Stage 2 artifacts dict mirrors Stage 1 (same keys) plus:
```python
{
    "stage2_params":   list[str]    e.g. ["Q", "c_ee", "c_ei", "c_ie", "c_ii"]
    "nuisance_params": list[str]    e.g. ["P", "g_e", "g_i"]
    "param_scaler":    ParameterScaler   Stage 2 only (theta_bad + c-params)
    "feature_pipeline": FeaturePipeline  NEW fit on Stage 2 data (separate from S1)
    ...
}
```

**Important: Stage 2 fits its own `FeaturePipeline` independently** on Stage 2 simulation
data. Stage 2 theta space is different (fewer params, different bounds), so an independent
pipeline is correct. The Stage 1 pipeline is reused only to compute `x_obs_s1` (the Stage 1
embedding of each training subject's observed FC) for nuisance sampling — it is not the
inference pipeline for Stage 2.

---

## 4. Prior Definition

**File:** `inference/priors.py:make_scaled_prior`

```python
def make_scaled_prior(n_dim):
    return BoxUniform(
        low=torch.full((n_dim,), -1.0, dtype=torch.float32),
        high=torch.full((n_dim,), +1.0, dtype=torch.float32),
    )
```

**Critically:** the prior is defined in **scaled space**, not raw parameter space.

- Stage 1: `make_scaled_prior(4)` → BoxUniform([-1,1]^4) for P, Q, g_e, g_i
- Stage 2: `make_scaled_prior(len(stage2_params))` → BoxUniform([-1,1]^k) for k Stage 2 params

The mapping between raw bounds and scaled space is entirely owned by `ParameterScaler`
(see Section 5). The prior itself has no knowledge of raw parameter semantics.

**SBI training stability:** defining the prior as BoxUniform([-1,1]) avoids numerical issues
in the MAF density estimator that arise when parameter scales differ by orders of magnitude
(e.g., P≈1.5 vs g_e≈0.3). All SNPE-C training and inference happens in scaled space.

`torch` is imported inside `make_scaled_prior` (deferred) — `inference/priors.py` loads
cleanly on CPU-only systems without torch until `make_scaled_prior()` is called.

---

## 5. Parameter Scaling and Inverse-Scaling

**File:** `inference/scaling.py:ParameterScaler`

```python
scaled = 2.0 * (raw - low) / range - 1.0        # raw → [-1, 1]
raw    = (scaled + 1.0) / 2.0 * range + low       # [-1, 1] → raw
range  = high - low
```

Key design properties:

| Property | Detail |
|---|---|
| Data-free | Depends only on prior bounds. Safe to call before any simulation. |
| Per-parameter | Each parameter has independent `low`, `high`, `range`. |
| Float32 | `low`, `high`, `range` stored as float32; both directions return float32. |
| Positive-range check | Constructor raises `ValueError` if any `range ≤ 0`. |
| `to_dict(theta)` | Returns `{name: value}` dict for raw `theta` — format expected by `simulate_gpu_batch`. |
| `subset(names)` | Returns a new `ParameterScaler` for a subset of params. Used by Stage 2 to subset Stage 1 scaler. |
| `from_dict(d)` | Reconstructs from serialized dict. Used by `load_artifacts`. |

**Critical invariant:** the SAME `ParameterScaler` instance (or a reconstructed copy with
identical `low`/`high`) must be used for both the forward transform during training-data
collection and the inverse transform during posterior sampling. If the scalers differ,
the posterior samples are silently mis-aligned with the simulation parameter space — the
posterior says one thing, but the simulated result uses different raw values.

`make_stage1_param_scaler()` reads `config.STAGE1_PARAMS`, `STAGE1_PRIOR_LOW`,
`STAGE1_PRIOR_HIGH`. `make_stage2_param_scaler(stage2_params)` looks up each param in
the Stage 1 config or in `config.C_PARAM_PRIOR` for c-params.

---

## 6. Feature Pipeline Connection

**Fit:** Step 5 fits `FeaturePipeline` on training simulations only (R10).

**Transform at inference time:**

```
observed subject FC (115×115)
  ↓ fc_to_upper_tri(fc)
  fc_obs_raw (6555,) float32
  ↓ pipeline.transform(fc_obs_raw, fcd_obs_raw)
      FCPCAScaler.transform: (6555,) → (300,) PCA
  x_obs (300,) float32
  ↓ posterior.sample(N_POSTERIOR, x=x_obs)
```

The `feature_pipeline` artifact is the only object that connects:
1. Training simulations (fit on these)
2. Observed FC at inference time (transform applied)
3. Simulated FC in validation/test (transform applied for posterior_predictive_check)

**Dimension lock:** `FeaturePipeline.transform()` raises `ValueError` on dimension mismatch.
This means if `NAN_MASK` changes between fitting and inference, the pipeline will raise
rather than silently produce wrong embeddings. The error message mentions the stored
`fc_dim` but not `NAN_MASK` — misleading for diagnosis.

**Stage 2 pipeline independence:** `run_stage2_snpe` creates a new `FeaturePipeline()` and
calls `.fit(fc_raw, fcd_raw)` on Stage 2 data. Stage 1 pipeline is used only to embed
observed FC for nuisance parameter sampling (`x_obs_s1`), not for Stage 2 posterior
inference.

---

## 7. SNPE/NPE Training Flow

**File:** `inference/snpe.py:train_snpe`

```
Inputs:
  theta_scaled  (N_SUBJ*N_SIM, 4) float32  — training params, scaled [-1,1]
  x_input       (N_SUBJ*N_SIM, 300) float32 — pipeline-transformed features
  prior_scaled  BoxUniform([-1,1]^4)

Architecture:
  FeatureEmbedding(input_dim=300):
    Linear(300,512) → ReLU → Dropout(0.1)
    Linear(512,256) → ReLU
    Linear(256,128)

  posterior_nn(model="maf", embedding_net=embedding, hidden=128, transforms=8)
  SNPE_C(prior=prior_scaled, device=SBI_DEVICE)

Training (package version):
  batch_size = 2048                  # H100-tuned (was 512 in monolith)
  stop_after_epochs = 20
  max_num_epochs = 300
  torch.set_float32_matmul_precision("high")   # H100 Tensor Core
  allow_bf16_reduced_precision_reduction = True
  cudnn.benchmark = True

Output:
  posterior     = inferer.build_posterior(estimator)
  embedding_net = FeatureEmbedding (with trained weights)
```

**sbi API compatibility:** `posterior_nn` moved between sbi versions. The code tries three
import paths in order: `sbi.neural_nets` → `sbi.utils` → `sbi.utils.get_nn_models`. This
gracefully handles sbi 0.22+ and older releases.

**Embedding network is jointly trained.** The MLP weights (Linear layers in FeatureEmbedding)
are optimized by the SNPE-C objective together with the MAF density estimator. The FC PCA
(inside FeaturePipeline) is fitted before training and frozen — it is not backprop-able.

**Step 4 result is discarded.** `step4_fit_feature_scalers()` returns `{"fc_z": None, "fcd_z": ...}`.
`run_stage1_snpe` calls it but does NOT store the result in the artifacts dict. The FCD
`FamilyScaler` from step4 is diagnostic-only. `FeaturePipeline` creates its own internal
`fcd_z` in step5. When `USE_FCD=True`, both are fitted on the same data — a redundant fit.

---

## 8. Posterior Sampling Flow

**File:** `inference/posterior.py:infer_subject_raw`

```python
def infer_subject_raw(posterior, x_obs_input, param_scaler,
                      n_samples=None):
    x_t = torch.tensor(x_obs_input, dtype=torch.float32)
    samples_scaled = posterior.sample(
        (n_samples,), x=x_t, show_progress_bars=False
    ).cpu().numpy().astype(np.float32)   # (N_POSTERIOR, 4)

    samples_raw = param_scaler.inverse_transform(samples_scaled)  # raw units
    means_raw   = samples_raw.mean(axis=0)                         # (4,)
    stds_raw    = samples_raw.std(axis=0)                          # (4,)
    return samples_raw, means_raw, stds_raw, samples_scaled
```

Default `n_samples = config.N_POSTERIOR = 2000`.

**Two-space discipline:** SBI samples in scaled `[-1, 1]` space. `inverse_transform` maps
back to raw space (e.g., P ∈ [0.5, 2.5]). The returned `samples_raw` is what gets passed
to `simulate_single` or `simulate_gpu_batch` for posterior predictive checks. Never mix
scaled and raw arrays.

**Shrinkage metrics:**

```python
compute_shrinkage_scaled(samples_scaled):
    prior_std = 2.0 / sqrt(12.0)   # BoxUniform[-1,1] std
    post_std  = samples_scaled.std(axis=0)
    return clip(1 - post_std / prior_std, 0, 1)
```

A shrinkage near 1.0 means the posterior is much tighter than the prior (parameter well-
identified). Near 0.0 means the posterior matches the prior (uninformative). Stage 2 uses
`compute_shrinkage_scaled` on Stage 1 samples to select `θ_bad = shrinkage < 0.2`.

`posterior_correlation(samples)` returns `np.corrcoef(samples.T)` — a diagnostic for
parameter degeneracy. `np.eye(n)` returned if fewer than 2 parameters.

---

## 9. Posterior Predictive Validation Connection

**File:** `inference/posterior.py:posterior_predictive_check`

```
For each of n_predictive posterior draws:
  theta_raw[i] → simulate_single(sc, params, ...) → BOLD (T, N)
  fc_pred = compute_fc(BOLD)
  FC corr = corrcoef(obs_fc_upper_tri, pred_fc_upper_tri)
  FC RMSE
  fcd_pred_vec = fcd_to_upper_tri(compute_sim_fcd_matrix(BOLD))  (6555,)
  FCD RMSE = sqrt(mean((fcd_obs_raw - fcd_pred_vec)^2))
```

Returns: `fc_corr_mean`, `fc_corr_std`, `fc_rmse_mean`, `fcd_rmse_mean`.

### RISK-I — Dual-role `fcd_obs_raw` parameter (latent bug when USE_FCD=True)

`posterior_predictive_check` receives `fcd_obs_raw` and uses it for **two purposes**:

1. **Pipeline input:** `feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)` at line 123
   — requires 5-dim FCD summary stats when `USE_FCD=True`.
2. **RMSE comparand:** `fcd_obs_raw - fcd_to_upper_tri(fcd_mat)` at line 162
   — requires 6555-dim FCD upper triangle.

These are **contradictory dimensional requirements**. When `USE_FCD=False` (current state),
the pipeline ignores `fcd_obs_raw` entirely, so any dimension works (the RMSE branch
runs against whatever is passed). But when `USE_FCD=True`:
- If caller passes 5-dim summary stats → pipeline works but RMSE fails (5 vs 6555)
- If caller passes 6555-dim upper-tri → RMSE works but pipeline raises `ValueError`

There is no input that would satisfy both simultaneously. This is a design bug in
`posterior_predictive_check` that must be resolved before enabling FCD.

The same dual-role issue exists in the monolith `inference.py:posterior_predictive_check`
(lines 984–1057), which is dead code and would have the same problem.

---

## 10. Training Data Generation / Loading

**File:** `inference/training_data.py:collect_training_data`

```
Per training subject (sid):
  1. Sample n_sim thetas from prior:
       theta_s = prior_scaled.sample((n_sim,))    (n_sim, 4)  scaled
       theta_r = scaler.inverse_transform(theta_s) (n_sim, 4)  raw
  2. GPU batch simulation (chunked by GPU_BATCH):
       bolds = simulate_gpu_batch(sc, chunk_r, param_names=..., delays=...)
       → list of (T_bold=240, N=115) BOLD arrays
  3. CPU feature extraction (ProcessPoolExecutor, N_CPU workers):
       future = executor.map(worker_extract, bolds, chunksize=16)
       → list of (fc_vec (6555,), fcd_vec (5,)) pairs   [worker_extract calls extract_features]
  4. _drain_one_future filters None and non-finite results
  5. GPU memory flush: cp.get_default_memory_pool().free_all_blocks()

Accumulate: all_theta_s, all_theta_r, all_fc, all_fcd (Python lists)
Final: np.array(all_fc, dtype=float32)  → fc_raw
       np.array(all_fcd, dtype=float32) → fcd_raw
```

**`fcd_raw` shape from worker:** `worker_extract` calls `extract_features` which returns
a 5-element zero FCD vector when `USE_FCD=False`. So `fcd_raw` is `(N, 5)` float32 all-zeros
in production. `FeaturePipeline.fit` ignores it because `use_fcd=False`. This is harmless
now but is the RISK-2 from `13_feature_module_review.md` — if FCD is enabled without
updating `worker_extract` to call `extract_simulated_features`, all-zero FCD would
silently enter training. See Section 13.2 for full analysis.

**Finite filtering:** `_drain_one_future` requires BOTH `fc_vec` AND `fcd_vec` to be all-
finite. Since `fcd_vec` is all zeros (finite), only `fc_vec` effectively filters results.
Simulations that produce NaN BOLD (e.g., GPU overflow) are silently dropped without
error — the only indication is `n_collected < n_requested`. The empty-check at the end of
`step2_simulate_train` catches the total-failure case (0 collected).

**Persistence:** `save_extracted_features` / `load_extracted_features` use compressed `.npz`.
Note: the verbose output inside `save_extracted_features` prints `"(Fisher-z upper tri)"` for
`fc_raw` — this is a **stale comment** from before the FC convention was fixed. FC is raw
Pearson r (not Fisher-z transformed). The same stale comment appears in the monolith at
line 685.

---

## 11. Root-Level `inference.py` Conflict Risk

**Risk level:** HIGH for tooling; LOW at runtime.

The package wins unconditionally at runtime:
```bash
python -c "import inference; print(inference.__file__)"
# → /scratch/home/wog3597/vbi/inference/__init__.py
```

### 11.1 Behavioral divergence from the package

`inference.py` is not a static copy of the package — it has diverged:

| Location | `inference.py` (monolith, dead) | `inference/snpe.py` (package, active) |
|---|---|---|
| `train_snpe`: `training_batch_size` | `512` | `2048` (H100-tuned) |
| `train_snpe`: CUDA precision | `"medium"` | `"high"` + bf16 flag |
| `FCPCAScaler.fit()` | no verbose output | verbose with EVR, top-5 PCs |
| `FeaturePipeline.fit()` | no verbose output | verbose with progress |
| `step5_fit_feature_pipeline` message | "FCD → PCA" | "FC PCA" (correct) |
| `collect_stage2_data` | `n_subj` NameError bug | correctly defined |

**The `n_subj` NameError** is the clearest proof that `inference.py` is dead code.
`collect_stage2_data` in `inference.py` uses `total = n_subj * n_sim` at line 1391, but
`n_subj` is never defined within that function (it is defined in `collect_training_data`
at line 396, a different function). If `inference.py` were ever loaded, calling
`collect_stage2_data` would raise `NameError: name 'n_subj' is not defined`. The package
version correctly defines `n_subj = len(train_subjects)` at the top of the function.

### 11.2 Static analysis / editor risk

Editors using Pylance, pyright, or pylint that do not recognise package-over-module
precedence will resolve `import inference` to `inference.py` and report false errors
(e.g., missing `run_stage2_snpe`, `select_theta_bad`, attributes that only exist in
`inference/__init__.py`). This can mask real problems if a developer relies on static
analysis.

### 11.3 Stale pycache risk

If a stale `__pycache__/inference.cpython-*.pyc` file exists from before `inference/`
was created, some Python implementations may load the cached bytecode. Clearing pycache
before tests (see Section 14, Test T-8) eliminates this risk.

---

## 12. Deferred `from simulator import` Calls in `inference/`

All five deferred legacy imports are documented in `11_import_audit.md` Section 2.2.
Reconfirmed here with exact locations:

| File | Line | Import | Called when |
|---|---|---|---|
| `inference/training_data.py` | 48 | `from simulator import simulate_gpu_batch, worker_extract` | Every Stage 1 and Stage 2 training run |
| `inference/stage2.py` | 158 | `from simulator import simulate_gpu_batch, worker_extract` | Every Stage 2 `collect_stage2_data` call |
| `inference/stage2.py` | 319 | `from simulator import extract_observed_features` | Every `run_stage2_snpe` call |
| `inference/posterior.py` | 113 | `from simulator import compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri, simulate_single` | Every PPC call |
| `inference/diagnostics.py` | 47 | `from simulator import compute_fc, compute_sim_fcd_matrix, fc_to_upper_tri, fcd_to_upper_tri, simulate_single` | Every SBC call |

All five are inside function bodies — they execute when the function is first called,
not at module load time. `simulator.py` remains a required file until these are migrated.

---

## 13. Circular Import Analysis

The `inference/` package import graph is acyclic:

```
_utils.py          (imports: time only)
  ↑
priors.py          (imports: torch [deferred])
scaling.py         (imports: numpy, config)
feature_pipeline.py (imports: numpy, config, sklearn [deferred])
embedding.py       (imports: config, torch [guarded])

snpe.py            (imports: _utils, embedding, feature_pipeline, priors, scaling, torch [guarded])
  ↑
training_data.py   (imports: _utils)
  ↑
stage1.py          (imports: snpe, training_data)
stage2.py          (imports: _utils, feature_pipeline, priors, scaling, snpe, training_data)

posterior.py       (imports: config, torch [guarded])
diagnostics.py     (imports: _utils, torch [guarded])
io.py              (imports: pickle only)
```

**No circular imports.** Key rules enforced:
- `snpe.py` does NOT import from `stage1.py` or `stage2.py` (R4)
- `stage1.py` imports from `snpe.py` and `training_data.py` — correct direction
- `stage2.py` imports `snpe._print_pca_diagnostic` and `training_data._drain_one_future`
  — exposing private helpers is slightly unusual but not circular
- `posterior.py` and `diagnostics.py` have no intra-package imports (pure function modules)

### 13.1 latent SBC + PPC FCD dimension bug (RISK-II)

`inference/diagnostics.py:simulation_based_calibration` builds the observation feature
using:
```python
fcd_vec = fcd_to_upper_tri(compute_sim_fcd_matrix(bold))   # (6555,)
x_obs = feature_pipeline.transform(fc_vec, fcd_vec)         # passes 6555-dim FCD
```

`FeaturePipeline.transform` with `USE_FCD=True` expects 5-dim FCD summary stats
(`fcd_dim=5`) and would raise:
```
ValueError: FCD input dim mismatch: got 6555, pipeline was fitted on 5.
```

With `USE_FCD=False` (current): `fcd_vec` is silently ignored — no bug.

**This is the same root cause as RISK-I in PPC (Section 9):** both functions use
`fcd_to_upper_tri` when they should use `fcd_to_summary_stats` if FCD is enabled. The
fix is the same: replace `fcd_to_upper_tri(fcd_mat)` with `fcd_to_summary_stats(fcd_mat)`
in both `diagnostics.py` and `posterior.py` (and add a `nan_mask` argument to pass the
config mask consistently).

---

## 14. `inference/__init__.py` — Package Public API

`inference/__init__.py` re-exports every public symbol from all 11 submodules. Full
`__all__` list contains 33 names. Notable exports:

**Also exported (semi-private helpers):**
- `_progress` (from `_utils`) — timestamp print
- `_print_pca_diagnostic` (from `snpe`) — internal formatter
- `_drain_one_future` (from `training_data`) — private accumulation helper
- `_build_nuisance_array` (from `stage2`) — private nuisance constructor

These leading-underscore names are exported at the package level for legacy compatibility
with code that calls `inference._drain_one_future(...)` directly. They should not be
used in new code.

**Note:** `inference/__init__.py` docstring says `"The old inference.py module is removed"` —
this is aspirational, not factual. `inference.py` is still present on disk.

---

## 15. Minimal Future Refactor Plan for `inference/`

### Step I1 — Migrate deferred `from simulator import` in `inference/` (Priority: HIGH)

Per `11_import_audit.md` Tier 2. Three files, five locations:

| File | Line | Replace `from simulator import X` with |
|---|---|---|
| `inference/training_data.py` | 48 | `from simulation.wc_runner import simulate_gpu_batch` + `from features.extraction import worker_extract` |
| `inference/stage2.py` | 158 | Same as above |
| `inference/stage2.py` | 319 | `from features.extraction import extract_observed_features` |
| `inference/posterior.py` | 113 | `from simulation.wc_runner import simulate_single` + `from features.fc import compute_fc` + `from features.fcd import compute_sim_fcd_matrix, fcd_to_upper_tri` |
| `inference/diagnostics.py` | 47 | Same as `posterior.py` + `from features.fc import fc_to_upper_tri` |

### Step I2 — Fix FCD dimension in SBC and PPC (Priority: MEDIUM, before enabling FCD)

**Problem (RISK-I, RISK-II):** both `diagnostics.py:simulation_based_calibration` (line 77)
and `posterior.py:posterior_predictive_check` (line 160) use `fcd_to_upper_tri(fcd_mat)`
(6555-dim) when passing to `feature_pipeline.transform` and for RMSE comparison.

When `USE_FCD=True`, `FeaturePipeline` expects 5-dim FCD summary stats, so:
- SBC will crash with `ValueError` (6555 ≠ 5)
- PPC has a dual-role conflict: 5-dim for pipeline, 6555-dim for RMSE

**Fix options:**

For SBC (purely a feature input issue — no RMSE):
```python
# Replace line 77 of diagnostics.py:
fcd_vec = fcd_to_summary_stats(compute_sim_fcd_matrix(bold))   # (5,) always
```

For PPC (dual-role conflict):
```python
# Split fcd_obs_raw into two separate variables:
fcd_obs_pipeline = fcd_to_summary_stats(fcd_mat_obs)     # (5,)  for pipeline
fcd_obs_rmse     = fcd_to_upper_tri(fcd_mat_obs)          # (6555,) for RMSE
x_obs = feature_pipeline.transform(fc_obs_raw, fcd_obs_pipeline)
...
fcd_pred_vec = fcd_to_upper_tri(fcd_mat)
fcd_rmses.append(sqrt(mean((fcd_obs_rmse - fcd_pred_vec)**2)))
```

This requires the caller to pass `fcd_obs_mat` (N×N matrix) or both vectors separately.
The PPC signature should be updated when FCD is enabled.

### Step I3 — Remove stale `"Fisher-z upper tri"` comments (Priority: LOW)

`inference/training_data.py:253` and its counterpart in `inference.py:685` both print
`"(Fisher-z upper tri)"` for `fc_raw`. FC is raw Pearson r, not Fisher-z. Remove from
both the package version and document the correction in `04_data_flow.md`.

### Step I4 — Archive/delete `inference.py` monolith (Priority: LOW, Tier 7)

Per `11_import_audit.md` Tier 7. Prerequisite steps:
1. Verify `inference/__init__.py` `__all__` is a strict superset of all names exported
   by `inference.py` (it is — confirmed by reading both).
2. Confirm no tool or script references `inference.py` by explicit path (e.g., no `exec`,
   `importlib.import_module("inference")` with explicit `__file__` override).
3. Delete `inference.py`.

**Do not rename to `inference.py.bak`** — this would still shadow the package for editors
that look for `.py` files.

### Step I5 — Remove `_drain_one_future` and `_build_nuisance_array` from `__all__` (Priority: LOW)

These are private implementation details re-exported from `inference/__init__.py` for
legacy compatibility. Once all callers use the package path, remove them from the top-
level exports. They are not in `__all__` by name but are importable as `inference._drain_one_future`.
When removing, verify via grep that no caller uses the `inference.` prefix to call them.

### Step I6 — Disambiguate `step4` FamilyScaler vs `FeaturePipeline.fcd_z` (Priority: LOW)

When `USE_FCD=True`, `step4_fit_feature_scalers` creates a `FamilyScaler("FCD")` for
diagnostics; `FeaturePipeline` creates a separate `fcd_z` FamilyScaler internally. Both
are fitted on the same `fcd_raw` data. Options:
- Remove the step4 `FamilyScaler` entirely (diagnostics already printed by `FeaturePipeline.fit`)
- Or reuse `FeaturePipeline.fcd_z` in step4's diagnostic output

---

## 16. Test Commands for `inference/`

### T-1 Compile check (no GPU, no sbi)

```bash
python -m py_compile inference/__init__.py
python -m py_compile inference/_utils.py inference/priors.py inference/scaling.py
python -m py_compile inference/feature_pipeline.py inference/embedding.py
python -m py_compile inference/training_data.py inference/snpe.py
python -m py_compile inference/stage1.py inference/stage2.py
python -m py_compile inference/posterior.py inference/diagnostics.py inference/io.py
echo "inference/ compile-clean"
```

### T-2 Import chain (no GPU, no sbi required)

```bash
python -c "
import inference
print('inference package file:', inference.__file__)
assert 'inference/__init__' in inference.__file__, 'wrong file!'

from inference import (
    ParameterScaler, make_stage1_param_scaler, make_stage2_param_scaler,
    FamilyScaler, FCPCAScaler, FeaturePipeline,
    collect_training_data, save_extracted_features, load_extracted_features,
    step2_simulate_train, step3_summary_features,
    run_stage1_snpe, run_stage2_snpe,
    build_stage2_param_set, select_theta_bad,
    transform_observed, infer_subject_raw,
    compute_shrinkage_scaled, compute_shrinkage_raw,
    save_artifacts, load_artifacts,
)
print('all inference imports OK')
"
```

### T-3 ParameterScaler round-trip

```bash
python -c "
import numpy as np
from inference.scaling import make_stage1_param_scaler

scaler = make_stage1_param_scaler()
raw = np.array([[0.5, 0.0, 0.0, 0.0],
                [2.5, 2.0, 1.5, 1.5],
                [1.5, 1.0, 0.75, 0.75]], dtype=np.float32)

scaled = scaler.transform(raw)
assert scaled.min() >= -1.0 - 1e-6 and scaled.max() <= 1.0 + 1e-6, 'out of [-1,1]'
assert np.allclose(raw, scaler.inverse_transform(scaled), atol=1e-5), 'round-trip fail'
assert scaled[0].tolist() == [-1.0, -1.0, -1.0, -1.0], 'lower bound must map to -1'
assert scaled[1].tolist() == [ 1.0,  1.0,  1.0,  1.0], 'upper bound must map to +1'
print(f'ParameterScaler OK: {raw[2]} -> {scaled[2].tolist()}')
"
```

### T-4 make_scaled_prior (requires torch)

```bash
python -c "
import torch
from inference.priors import make_scaled_prior

prior = make_scaled_prior(4)
samples = prior.sample((1000,))
assert samples.shape == (1000, 4), f'shape: {samples.shape}'
assert float(samples.min()) >= -1.0, 'below -1'
assert float(samples.max()) <=  1.0, 'above +1'
print(f'make_scaled_prior OK: {samples.shape}, range=[{float(samples.min()):.3f},{float(samples.max()):.3f}]')
"
```

### T-5 FeaturePipeline fit/transform (no GPU, no sbi)

```bash
python -c "
import numpy as np, config
from inference.feature_pipeline import FeaturePipeline

N_SIM = 500
fc_train = np.random.uniform(-1, 1, (N_SIM, config.FC_DIM)).astype(np.float32)
fcd_train = np.zeros((N_SIM, 0), dtype=np.float32)

p = FeaturePipeline()
x = p.fit(fc_train, fcd_train, verbose=False).transform(fc_train, fcd_train)
assert x.shape == (N_SIM, config.PCA_DIM_FC), f'{x.shape}'
assert x.dtype == np.float32, f'dtype: {x.dtype}'

# verify dimension lock
try:
    p.transform(fc_train[:, :100], fcd_train)
    assert False, 'should raise'
except ValueError as e:
    print(f'dim lock OK: {str(e)[:60]}')

print(f'FeaturePipeline OK: {x.shape}')
"
```

### T-6 FeatureEmbedding forward pass (requires torch)

```bash
python -c "
import torch
from inference.embedding import FeatureEmbedding
import config

emb = FeatureEmbedding(input_dim=config.PCA_DIM_FC)
x = torch.randn(32, config.PCA_DIM_FC)
out = emb(x)
assert out.shape == (32, config.EMBED_DIM), f'{out.shape}'
assert out.dtype == torch.float32
print(f'FeatureEmbedding OK: {x.shape} -> {out.shape}')
"
```

### T-7 Verify package wins over monolith

```bash
python -c "
import inference
assert 'inference/__init__' in inference.__file__, \
    f'WRONG: loaded {inference.__file__} not the package'
print('Package resolution OK:', inference.__file__)
"
```

### T-8 Purge pycache before resolution test

```bash
find /scratch/home/wog3597/vbi -name "*.pyc" -delete
find /scratch/home/wog3597/vbi -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "pycache cleared"
python -c "import inference; print(inference.__file__)"
```

### T-9 Dependency rule check (R3) — inference/ must not import evaluation/

```bash
grep -rn "from evaluation" inference/*.py
# Expected: zero output

# Verify inference correctly imports simulation and features (allowed by R3):
grep -rn "from simulation\|from features" inference/*.py
# Expected: lines in training_data.py, stage2.py, posterior.py, diagnostics.py
# (deferred imports — all currently via 'from simulator import', to be migrated in Step I1)
```

### T-10 No bare simulator import at module level in inference/

```bash
grep -n "^from simulator import\|^import simulator" inference/*.py
# Expected: zero output (all simulator imports are deferred inside functions)
```

### T-11 Monolith divergence check

```bash
# Confirm inference.py has the n_subj NameError bug (proves it's dead code):
grep -n "n_subj" inference.py | grep -v "collect_training_data\|def \|n_subj = "
# Expected: lines inside collect_stage2_data using n_subj without definition nearby

# Confirm training_batch_size divergence:
grep -n "training_batch_size" inference.py inference/snpe.py
# Expected: 512 in inference.py, 2048 in inference/snpe.py
```

### T-12 save_artifacts / load_artifacts round-trip (no GPU)

```bash
python -c "
import tempfile, os
from inference.io import save_artifacts, load_artifacts
import numpy as np

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'test.pkl')
    save_artifacts(path, x=np.array([1.0, 2.0]), y='hello')
    loaded = load_artifacts(path)
    assert np.allclose(loaded['x'], [1.0, 2.0])
    assert loaded['y'] == 'hello'
    print('save/load_artifacts OK')
"
```

---

## Final Assessment

### Core inference files (do not delete)

| File | Status | Notes |
|---|---|---|
| `inference/__init__.py` | **Core.** Package entry point. Re-exports all 33 public names. | If deleted, `import inference` falls back to the monolith — catastrophic. |
| `inference/scaling.py` | **Core.** `ParameterScaler` — only bridge between raw params and SNPE. | Data-free; safe for all environments. |
| `inference/priors.py` | **Core.** `make_scaled_prior` — defines the SBI prior in scaled space. | Tiny file; torch deferred. |
| `inference/feature_pipeline.py` | **Core.** `FeaturePipeline`, `FCPCAScaler`, `FamilyScaler`. | Train-only fit, frozen at inference. |
| `inference/embedding.py` | **Core.** `FeatureEmbedding` — MLP jointly trained with SNPE-C. | Torch-guarded; can import on CPU. |
| `inference/training_data.py` | **Core.** `collect_training_data`, step2/3, persistence. | The GPU+CPU simulation-extraction loop. |
| `inference/snpe.py` | **Core.** `train_snpe`, step4–8. | H100-optimized training. |
| `inference/stage1.py` | **Core.** `run_stage1_snpe` entry point. | Single-round SNPE-C driver. |
| `inference/stage2.py` | **Core.** `run_stage2_snpe`, θ_bad selection, nuisance handling. | Two-stage design. |
| `inference/posterior.py` | **Core.** `infer_subject_raw`, shrinkage, PPC. | Has RISK-I FCD dual-role bug (dormant). |
| `inference/diagnostics.py` | **Core.** SBC, embedding probing R². | Has RISK-II FCD dimension bug (dormant). |
| `inference/io.py` | **Core.** Pickle-based artifact persistence. | No versioning — convention-based. |
| `inference/_utils.py` | **Core (internal).** `_progress` timestamp printer. | Used by all submodules. |

### Whether root `inference.py` is dangerous

**YES — for static analysis and editor tooling.** `inference.py` causes editors to resolve
`import inference` to the 55 KB monolith instead of the package, producing false attribute
errors and suppressing real ones. It also contains the NameError bug (Stage 2
`collect_stage2_data` uses `n_subj` without defining it) and a behavioral divergence
(`training_batch_size=512` vs package's `2048`).

**NO — at runtime.** Python's package-over-module precedence ensures `inference/` package
always wins. The monolith is never loaded during any pipeline run.

### Whether root `inference.py` can be archived later

**YES — after a full diff audit confirming `inference/__init__.py` exports a strict superset.**
The audit is already partially done: `inference/__init__.py` exports all 33 public names
that appear in the monolith, plus private helpers. However, the monolith has docstrings
and inline comments that are no longer in the submodules — these have historical value
and should be reviewed before deletion.

**Safe to mark as legacy now (do not edit per R8).** Deletion is Tier 7 per
`11_import_audit.md` — last in the cleanup sequence.

### Files that must NOT be deleted yet

| File | Reason |
|---|---|
| `simulator.py` | Active deferred callers in all 5 `inference/` locations listed in Section 12. Must complete Step I1 first. |
| `inference.py` | Requires full diff audit before deletion. Do not edit (R8). |
| `inference/stage2.py` | Contains `from simulator import` at lines 158 and 319 — active in every Stage 2 run. |
| `inference/posterior.py` | Contains `from simulator import` at line 113 — active in every PPC call. Has RISK-I FCD bug to fix before enabling FCD. |
| `inference/diagnostics.py` | Contains `from simulator import` at line 47 — active in every SBC call. Has RISK-II FCD bug to fix before enabling FCD. |

### What should be inspected next

In priority order:

1. **`pipelines/stage1_stage2.py`** — the 14-step pipeline orchestrator that calls
   `run_stage1_snpe`, `run_stage2_snpe`, and the `evaluate.*` functions. Reading it would
   confirm exactly how Stage 1 and Stage 2 artifacts are handed to evaluation, and whether
   the shrinkage/sensitivity values passed to `build_stage2_param_set` come from validation
   subjects or training subjects. This is the one production file with the known top-level
   `import evaluate` (CONFLICT-2 in `11_import_audit.md`).

2. **`evaluation/metrics.py`** — the file with three deferred `from simulator import` calls
   (lines 110, 184, 245) that imports `extract_observed_features`, `compute_fc`,
   `compute_sim_fcd_matrix`, `fcd_to_upper_tri`, and `simulate_single`. Understanding how
   evaluation uses observed features and posterior predictive simulations closes the loop
   on the full data flow from empirical FC to model selection score.

3. **`evaluation/validation.py`** — calls `evaluate_validation_stage1/2` which is the
   function that drives model selection. Confirming whether it calls
   `infer_subject_raw` or `posterior_predictive_check` directly determines whether
   RISK-I is on the validation critical path.

4. **`config.py` (full read)** — to verify `N_SIM_S2`, `DIFFICULT_SHRINKAGE`,
   `LOCAL_EI_PARAMS`, `C_PARAM_PRIOR`, `NUISANCE_METHOD`, `USE_MIXED_PRECISION`,
   `EMB_PROBE_R2_THRESHOLD`, `N_SBC`, `N_PPC`, `N_CPU`, `GPU_BATCH`, and whether any
   Stage 2 defaults are overridden at the top level vs. pipeline level.
