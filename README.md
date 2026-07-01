# VBI-SBI Brain Parameter Inference Pipeline

Whole-brain parameter inference with simulation-based inference (SNPE-C).
GPU forward simulation via **cuBNM**, amortized posterior via **sbi**.

Infers region-wise RWW-EIB parameters — encoded as **myelin/gradient basis
coefficients** — from HCP functional connectivity (FC).

Entrypoint: **`main_HCP.py`** — HCP human, **RWWEIB_2CPL** model, 360 cortical
regions.

> **Source of truth = code + config.** This repo is trimmed to only the code
> and assets used by the full `main_HCP.py` run.

---

## Data

The active pipeline reads FC + SC + myelin/gradient maps. Large `.mat` files
exceed GitHub's 100MB limit and are **not** committed — obtain them separately
and place them in the repo root.

### In-repo (committed)
| File | Size | Contents |
|------|------|----------|
| `HCP_CABNP381_SC_first100.mat` | 43M | **active SC** — CAB-NP 381-region, first 100 subjects (`SC_DATASET=cabnp381`) |
| `myelin_subjects.npy` | 300K | per-subject myelin (T1w/T2w) maps — basis input |
| `gradient_subjects.npy` | 300K | per-subject principal functional-gradient maps — basis input |
| `basis.npy` | 9K | `(381,3) = [const, myelin, gradient]` basis matrix (sliced `[:360]`) |
| `Custom_Schaefer200_7net_PD25subcortex*.txt` | 6.5K | parcellation label tables |

### External (NOT committed — get separately)
| File | Size | Contents |
|------|------|----------|
| `HCP_FC.mat` (var `C`) | 1.1G | **active FC** target — per-subject 381-region FC |
| `HCP_SC.mat` | 224M | full HCP SC (alternate `SC_DATASET`) |
| `MPTP_FC_115.mat` / `MPTP_SC_115.mat` | | legacy mouse data |

Cortical-only: 381 → first **360** Glasser regions (21 subcortical dropped).
SC scaling via `VBI_SC_SCALE` (`main_HCP.py` forces `maxnorm`).

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

### Parameterization — `basis_regionwise` (default)
Four region-wise params (`g_LRE, g_FFI, I_o, sigma`) are **not** inferred
directly. Instead each is a linear combination of 3 basis maps
`[const, myelin, gradient]`, so the inferred vector `theta` has **12 = 4×3**
coefficients (`theta_dim=12`).

Decode (`basis_decoder.py`): per param,
```
z   = beta · basis.Tᵀ          # (S,360) region map from 3 coeffs
map = mid + half·tanh(z)        # mid=(lo+hi)/2, half=(hi-lo)/2
```
Basis bounds (`BASIS_BOUNDS`): `g_LRE(0,3) g_FFI(0,3) I_o(...) sigma(0,0.05)`.
Prior: scaled `BoxUniform[-1,1]^12`; raw coeff `(-2,2)`. `theta=0` → param
midpoints. Bounds/coeff-order are load-bearing — see `basis_decoder.py`.

### Dataflow (one line)
```
theta(12) → decode → {g_LRE,g_FFI,I_o,sigma}(S,360) → RWWEIB_2CPLSimGroup(GPU)
  → BOLD(T,360) → compute_fc (raw Pearson r) → upper-tri (64,620)
  → FeaturePipeline PCA-256 whitened → SNPE-C (MAF, nn.Identity embedding)
```
N_SIM is **per-subject** → real train tensor = N_TRAIN × N_SIM.

### ⚠️ Default gotchas (env overrides in `main_HCP.py`)
- `SMOKE=1` is the **default** → bare `python main_HCP.py` is a **tiny toy**
  (4/2/1/1 subjects, 64 sims). Real run = **`SMOKE=0`** (100/70/10/20, 2000 sims).
- `GROUP_AVG_FC=0` → **per-subject** FC (not group-averaged).
- `USE_DELAYS=0` → delays OFF (computed but not fed; ~0 BOLD-FC effect for 5.3× cost).
- `SC_CONDITION=0`, `GEOMETRY_COUPLING=0` → baseline (both OFF).

### Run
```bash
# REAL run (GPU node):
SMOKE=0 PARAMETER_MODE=basis_regionwise python main_HCP.py

# smoke / CPU-safe checks (no training):
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  python -m pytest test_basis_mode_smoke.py -q
```
Env overrides: `SMOKE, N_SUBJECTS, N_TRAIN, N_VAL, N_TEST, N_SIM, GPU_BATCH,
PARAMETER_MODE, SC_DATASET, SC_FILE, GROUP_AVG_FC, USE_DELAYS, VBI_SC_SCALE`.

### cuBNM rebuild (after yaml / kernel changes)
```bash
cd /scratch/home/wog3597/cubnm_build
python codegen/generate_models.py
pip install -e . --no-build-isolation
```
The multi-coupling kernel surgery (`conn_state_vars` support) lives in a
separate cuBNM fork (`cubnm_build/`), required to build RWWEIB_2CPL.

---

## Key files

| File | Purpose |
|------|---------|
| `main_HCP.py`                 | HCP pipeline driver; config @101-198, basis dispatch |
| `basis_decoder.py`            | `BasisParamDecoder`, `get_decoder`; myelin/gradient decode |
| `param_decoder.py`            | `decode_to_param_maps` dispatch |
| `engine_select.py`            | route active model (rwweib2 / rwweib / rww / vbi) |
| `data_loader_hcp.py`          | FC/SC load, 381→360 slice, SC scale, group-avg FC |
| `cuBNM/rww_eib_2cpl.yaml`     | RWWEIB_2CPL model (2 couplings, `conn_state_vars:[S_E,S_I]`) |
| `cuBNM/runner_rwweib_2cpl.py` | `build_param_lists`, `RWWEIB_2CPLSimGroup` |
| `inference/feature_pipeline.py` | FC PCA-256 whiten |
| `inference/snpe.py`           | SNPE-C; `nn.Identity` embedding; MAF |
| `evaluation/`                 | validation/test metrics, plots (engine-routed) |

---

## Style
All modules conform to `pycodestyle --max-line-length=88`.
