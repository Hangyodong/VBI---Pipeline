# 04 — Data Flow

## Input Data

| File | Format | Contents |
|---|---|---|
| `MPTP_FC_115.mat` | MATLAB `.mat`, `data` field, shape `(n_subjects, 3)` object array | col 0 = subject ID; col 1 = FC (115×115, float64, NaN-affected); col 2 = FCD (115×115, float64, NaN-free) |
| `MPTP_SC_115.mat` | MATLAB `.mat`, `data` field, shape `(n_subjects, 3)` object array | col 0 = subject ID; col 1 = SC raw uint16 counts; col 2 = tract length (float64, mm) |
| `participants.tsv` | TSV | subject ID, group label (`ctr` or `MPTP`) |
| `atlas_115_labels.txt` | TSV (tab-separated) | region index, hemisphere, name |

## Step 1: Data Loading and Splitting

```
data_loader.load_raw_data()
  → df (participants), fc_mat, sc_mat, bold_mat (None if absent)

data_loader.get_target_subjects(df, fc_ids, sc_ids)
  → subjects filtered by GROUP_FILTER = ("ctr", "MPTP")

data_loader.three_way_split(subjects)
  → train (4), val (2), test (2)   [SEED=42]

data_loader.load_all_subjects(...)
  → subject_data dict {sid: {"fc": (115,115), "sc": (115,115),
                              "lengths": (115,115) or None,
                              "delays": (115,115) or None,
                              "bold": (T,115) or None}}
```

**SC preprocessing inside `load_all_subjects`:**
```
raw_sc (uint16, 115×115)
  → sc_weight = log1p(raw_sc) / max(log1p(raw_sc))    [0, 1] float64
  → delays (ms) = lengths_mm / VELOCITY_M_PER_S        [1.5 m/s]
```

**FC preprocessing:**
```
raw_fc (115×115, float64, NaN-affected)
  → NaN replaced with 0.0 (constant-zero rows contribute no variance)
  → stored as subject_data[sid]["fc"], shape (115, 115)
```

## Step 2: Wilson-Cowan Simulation

For each training subject and each prior sample θ = (P, Q, g_e, g_i):

```
Prior samples: sbi.BoxUniform(low, high) in scaled [-1,1] space
  → ParameterScaler.inverse_transform() → raw θ (P∈[0.5,2.5], Q∈[0,2], ...)

simulate_gpu_batch(
    weights   = subject_data[sid]["sc"],   # (115, 115) float64
    theta_batch = theta_raw,               # (N_SIM, 4)
    param_names = ["P", "Q", "g_e", "g_i"],
    delays    = subject_data[sid]["delays"],
    apply_bw  = True
)
```

**Inside `simulate_gpu_batch`:**
```
WC_FIXED params (c_ee=16, c_ei=12, c_ie=15, c_ii=3, tau_e=8, tau_i=8,
                 a_e=1.3, a_i=2.0, b_e=4.0, b_i=3.7, noise_amp=0.01,
                 dt=0.5ms, t_end=300000ms, t_cut=60000ms,
                 method="heun", decimate=2, dtype="float32")

Per-simulation parameter tiling:
  chunk[:, i] → (n_nodes=115, n_sim) broadcast array

VBI WC_sde.heunStochastic() (GPU / cupy):
  E(t), I(t) — neural activity, shape (2×115, n_sim) per step

_run_streaming_hrf() with BoldMonitor (CPU / numpy):
  step-by-step: E_i → BoldMonitor.step() → BOLD accumulation
  output: BOLD (T_bold, 115, n_sim) float32

Split per simulation:
  outputs = [BOLD[:, :, i] for i in range(n_sim)]
  each: (T_bold, 115) float32
```

**BOLD dimensions:**
```
T_end = 300,000 ms
T_cut =  60,000 ms
DT = 0.5 ms, DECIMATE = 2 → neural step = 1 ms
TR = 1,000 ms (TR_SEC = 1.0 s)
ANALYSIS_BOLD_T = (300000 - 60000) / 1000 = 240 TRs

BOLD output shape per simulation: (240, 115) float32
```

## Step 3: Feature Extraction

```
extract_simulated_features(bold_list)
  → fc_raw  (N_SIM, 6555) float32   ← FC upper triangle
  → fcd_raw (N_SIM, 0)   float32   ← empty (USE_FCD=False)
```

**FC extraction:**
```
BOLD (240, 115)
  → compute_fc(ts) = np.corrcoef(ts.T)       → FC (115, 115)
  → fc_to_upper_tri(fc)                       → vec (6555,) float32

FC_DIM = 115 × 114 / 2 = 6555
```

**FCD extraction (disabled by default):**
```
USE_FCD = False → fcd_raw shape is (N_SIM, 0) or empty

If enabled:
  BOLD (240, 115)
    → compute_sim_fcd_matrix(bold, window_tr=60, stride_tr=3)
    → FCD (115, 115): element-wise std of sliding-window FCs
    → fcd_to_upper_tri or fcd_to_summary_stats → vec (6555,) or (5,)
```

## Step 4–5: Feature Pipeline

```
Step 4: FamilyScaler fit on training fc_raw (6555 dims)
  → z-score per feature (mean/std from train set only)

Step 5: FCPCAScaler + FeaturePipeline
  PCA_DIM_FC = 300    (≥90% explained variance threshold)
  PCA_DIM_FCD = 100   (disabled in fc_only mode)

  fc_raw (M, 6555)
    → z-score → PCA → fc_pca (M, 300)

  FEATURE_SET = "fc_only":
    x_input = fc_pca    shape (M, 300)

  FEATURE_SET = "fc_fcd" (requires empirical BOLD):
    x_input = concat(fc_pca, fcd_pca)    shape (M, 400)
```

## Step 6: Embedding (MLP)

```
FeatureEmbedding: MLP
  input_dim  = 300 (or 400 if fc_fcd)
  hidden_dim = 512   (EMBED_HIDDEN)
  output_dim = 128   (EMBED_DIM)

x_input (M, 300) → MLP → embedding (M, 128)
```

## Step 7: Parameter Scaling

```
ParameterScaler:
  raw θ ∈ [prior_low, prior_high]  →  scaled θ ∈ [-1, 1]
  inverse: scaled θ → raw θ for VBI simulation

Stage 1 prior bounds:
  P    ∈ [0.5, 2.5]
  Q    ∈ [0.0, 2.0]
  g_e  ∈ [0.0, 1.5]
  g_i  ∈ [0.0, 1.5]
```

## Step 8: SNPE-C Training

```
SBI inputs:
  theta_scaled  (M_train, 4)      in [-1, 1]
  x_embedding   (M_train, 128)    from MLP

SNPE-C (sbi library):
  NDE_MODEL     = "maf"
  NDE_HIDDEN    = 128
  NDE_TRANSFORMS = 8
  SBI_DEVICE    = "cuda" or "cpu"

Output: posterior density estimator p(θ | x)
```

## Observed Feature Flow (at inference time)

```
empirical FC (115×115, from MPTP_FC_115.mat col 1, NaN→0)
  → fc_to_upper_tri()              → vec (6555,) float32
  → FamilyScaler.transform()       → z-scored vec
  → PCA.transform()                → (300,) PC scores
  → MLP embedding                  → (128,) embedding
  → posterior.sample(N_POSTERIOR=2000)
  → ParameterScaler.inverse_transform()
  → posterior samples in raw param space
```

## Stage 2 Flow

```
Stage 1 posterior → compute_shrinkage_* per parameter
  → theta_bad: params with high sensitivity AND low shrinkage

Stage 2 params = theta_bad ∪ {c_ee, c_ei, c_ie, c_ii}
  prior bounds from C_PARAM_PRIOR:
    c_ee ∈ [12, 20],  c_ei ∈ [8, 16]
    c_ie ∈ [10, 20],  c_ii ∈ [1, 6]

NUISANCE_METHOD = "posterior_sample":
  nuisance parameters (not in theta_bad) sampled from Stage 1 posterior

New simulations with Stage 2 param set → SNPE-C Stage 2 posterior
```

## Evaluation Flow

```
Validation subjects:
  each val subject → observed FC → embedding → posterior samples
  → resimulate N_PPC=50 draws → fc_metrics(sim_FC, obs_FC)
  → (corr, rmse) per subject

Model selection score (val only):
  score = SELECT_W_FC_CORR × mean_corr
        - SELECT_W_FC_RMSE × mean_rmse
        - SELECT_W_FCD_RMSE × fcd_rmse   (0 weight when USE_FCD=False)

Final test (test subjects, one-shot after model selection):
  N_TEST_RESIM = 50 posterior draws re-simulated per test subject
  → fc_metrics + bootstrap CI (N_BOOTSTRAP=1000)
```

## Dimension Summary

| Quantity | Value | Source |
|---|---|---|
| Brain regions N | 115 | config.N_REGIONS |
| FC upper-triangle dim | 6555 | 115×114/2 |
| FCD summary stats dim | 5 (disabled) | config.FCD_DIM |
| BOLD TRs (post-cut) | 240 | config.ANALYSIS_BOLD_T |
| PCA output dim (FC) | 300 | config.PCA_DIM_FC |
| MLP embedding dim | 128 | config.EMBED_DIM |
| Stage 1 param dim | 4 | P, Q, g_e, g_i |
| Stage 2 param dim | 4 + len(theta_bad) | c_ee/ei/ie/ii + theta_bad |
| N_SIM per subject | 50,000 | config.N_SIM (production) |
| Posterior samples | 2,000 | config.N_POSTERIOR |
