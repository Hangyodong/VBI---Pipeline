# VBI-SBI Brain Parameter Inference Pipeline

Whole-brain parameter inference with simulation-based inference (SNPE-C).
GPU forward simulation via **cuBNM**, amortized posterior via **sbi**.

Two entrypoints:
- **`main_HCP.py`** — HCP human, **RWWEIB_2CPL** model, 360 cortical regions. (canonical / active)
- **`main.py`** — mouse MPTP, Wilson-Cowan model, 115 regions. (legacy)

> Detailed, per-stage documentation: **[`PIPELINE.md`](PIPELINE.md)**.

---

## HCP pipeline (`main_HCP.py`)

### Model — RWWEIB_2CPL (cuBNM)
Two-population reduced Wong-Wang with **two independent connectome couplings**
(E driven by `SC@S_E`, I driven by `SC@S_I`):

```
I_E = w_E·I_o + w_p·J_N·S_E + g_LRE·J_N·(SC@S_E) − J_i·S_I
I_I = w_I·I_o +     J_N·S_E + g_FFI·J_N·λ_IE·(SC@S_I) − S_I
dS_E/dt = −S_E/τ_E + (1−S_E)·γ_E·H_E(I_E) + σ·ξ
dS_I/dt = −S_I/τ_I +          γ_I·H_I(I_I) + σ·ξ
```
BOLD = Balloon-Windkessel HRF on `S_E`.

### Inferred parameters (Stage 1)
| param | role | prior (3× original) |
|-------|------|------|
| `g_LRE` | excitatory long-range coupling | U(0, 9) |
| `g_FFI` | inhibitory long-range coupling | U(0, 9) |
| `I_o`   | background input current       | U(0.15, 0.60) |
| `sigma` | noise amplitude                | U(0, 0.09) |

Fixed: `w_E=1.0, w_I=0.7, J_i=1.0, w_p=1.4, J_N=0.15, λ_IE=1.0`.
Parameters are **homogeneous** (shared across nodes per simulation).

### Data
- `HCP_FC.mat` / `HCP_SC.mat` — 1039/1040 subjects, 381 regions.
- **cortical-only**: first 360 (Glasser) regions; 21 subcortical dropped.
- **group-averaged FC** target (`GROUP_AVG_FC=True`): mean FC over 1039 subjects.
- SC scaling via `VBI_SC_SCALE` (`maxnorm` default).

### Stages
| Step | Description |
|------|-------------|
| 1  | Data split + load (train/val/test) |
| 7  | Param scaler + BoxUniform prior (before Step 2) |
| 2  | Forward simulation → BOLD → FC (training data); streamed feature extraction |
| 3  | Feature summary + save (`features_stage1.npz`, with staleness guard) |
| 4  | Feature pipeline (raw FC passthrough — no PCA / z-score) |
| 8  | SNPE-C training (RegionTransformer embedding → posterior) |
| 3/4* | Phase 3/4 feature-selection — **skipped** (`RUN_PHASE24=False`) |
| 9  | Validation: resim, FC corr/RMSE, shrinkage, probing, ActiveSubspace, SBC |
| 13 | Model selection |
| 14 | Final test (bootstrap CI) |

### Run
```bash
# smoke
rm -f output_hcp/features_stage1.npz
N_SUBJECTS=8 N_TRAIN=4 N_VAL=2 N_TEST=2 N_SIM=50 python main_HCP.py
# full
python main_HCP.py
```
Env overrides: `N_SUBJECTS, N_TRAIN, N_VAL, N_TEST, N_SIM, GPU_BATCH, VBI_SC_SCALE`.

### cuBNM rebuild (after yaml / kernel changes)
```bash
cd /scratch/home/wog3597/cubnm_build
python codegen/generate_models.py
pip install -e . --no-build-isolation
```

---

## Diagnostic tools (standalone, GPU)

| Tool | Purpose |
|------|---------|
| `sensitivity.py`        | forward FC parameter sweep (per-param FC response) |
| `active_sensitivity.py` | sbi `ActiveSubspace` gradient sensitivity |
| `fc_support_diag.py`    | per-param sensitivity + sensitive edges + empirical-in-sim-support test (`--prior_scale --io --gsr --fic --hetero`) |
| `fic_tune.py`           | fast vectorized FIC → per-(sim,node) `J_i` |
| `eib_tune.py`           | EIB per-edge effective-connectivity tuning (descriptive) |
| `hcp_ceiling.py`        | param random-search FC-corr ceiling |

---

## Key files

| File | Purpose |
|------|---------|
| `main_HCP.py`            | HCP pipeline driver |
| `data_loader_hcp.py`     | HCP FC/SC load, cortical slice, group-avg FC, SC scaling |
| `engine_select.py`       | route active model (rwweib2 / rwweib / rww / vbi) |
| `cuBNM/rww_eib_2cpl.yaml` | RWWEIB_2CPL model definition (codegen source) |
| `cuBNM/runner_rwweib_2cpl.py` | param_lists builder + batch runner |
| `cuBNM/simulate_rwweib_2cpl.py` | `simulate_gpu_batch` adapter |
| `inference/`             | scalers, priors, SNPE-C, embedding, diagnostics |
| `evaluate.py` / `evaluation/` | validation/test metrics, plots |
| `PIPELINE.md`            | detailed per-stage documentation |

The cuBNM multi-coupling kernel surgery (conn_state_vars support) lives in a
separate cuBNM fork (`cubnm_build/`), required to build RWWEIB_2CPL.

---

## Legacy mouse pipeline (`main.py`)

Mouse MPTP, Wilson-Cowan, 115 regions, FC+FCD features, 14-step driver.
```bash
python main.py
python debug.py --basic   # quick checks, no GPU
```
Data: `MPTP_FC_115.mat` (FC col 1, FCD col 2), `MPTP_SC_115.mat` (SC col 1).

## Style
All modules conform to `pycodestyle --max-line-length=88`.
