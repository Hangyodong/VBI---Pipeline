# 01 — Repository Overview

## Goal

Whole-brain parameter inference for the **mouse MPTP model** (115 cortical/subcortical
regions) using simulation-based inference (SBI / SNPE-C) applied to the VBI
Wilson-Cowan neural mass model.

## Main Purpose

Given empirical **functional connectivity (FC)** matrices per subject, infer the
Wilson-Cowan parameters (working-point P/Q and coupling gains g_e, g_i in Stage 1;
local E/I coupling c_ee, c_ei, c_ie, c_ii in Stage 2) that best explain each
subject's observed FC — without ever computing a likelihood.

## Scientific Assumptions

| Assumption | Detail |
|---|---|
| SC is **input**, not inferred | `MPTP_SC_115.mat` weight (col 1) and tract length (col 2) are loaded and fixed |
| SC weight preprocessing | uint16 raw counts → log1p + max-norm inside `data_loader` |
| Conduction delay | derived from tract length (mm) ÷ `VELOCITY_M_PER_S` (1.5 m/s) |
| Parameters inferred (Stage 1) | `P`, `Q`, `g_e`, `g_i` |
| Parameters inferred (Stage 2) | `theta_bad` from Stage 1 + `c_ee`, `c_ei`, `c_ie`, `c_ii` |
| FC computed from BOLD | Pearson correlation of `(T, N)` BOLD time series |
| FCD disabled by default | `USE_FCD = False`; no empirical BOLD time series available |
| SBI features | FC upper triangle only: dim = 115×114/2 = **6555** |
| PSD excluded | `USE_PSD = False`; only activated if empirical PSD exists |
| BOLD signal | Balloon-Windkessel HRF (TVB MixtureOfGammas, mouse-tuned: peak ~3 s) |

## Current Architecture

```
main.py / main.ipynb
      └─ pipelines.run_pipeline()          # 14-step orchestration
              ├─ data_loader               # data I/O and subject split
              ├─ simulation/               # WC simulation (GPU, cupy)
              ├─ features/                 # FC, FCD, extraction
              ├─ inference/                # scaling, PCA, MLP, SNPE-C
              └─ evaluation/              # metrics, plots, model selection
```

## Package Folders (authoritative, new modular structure)

| Package | Purpose |
|---|---|
| `simulation/` | Wilson-Cowan GPU engine, delays, warmup, QC |
| `features/` | FC/FCD computation, feature extraction |
| `inference/` | Parameter scaling, SBI priors, FeaturePipeline, SNPE-C training, posteriors |
| `evaluation/` | Metrics, validation, model selection, final test, plots, reports |
| `pipelines/` | End-to-end orchestration (`run_pipeline`) |

## WARNING: Duplicated Root-Level Files

Several root-level `.py` files are **duplicates or legacy versions** of submodule
files. They exist for backward compatibility or as historical artifacts:

| Root file | Status | Authoritative location |
|---|---|---|
| `inference.py` | **OLD MONOLITH** (55 KB, contains actual logic) | `inference/` package |
| `simulator.py` | Compat wrapper (re-exports simulation + features) | `simulation/` + `features/` |
| `evaluate.py` | Compat wrapper (re-exports evaluation) | `evaluation/` package |
| `wc_runner.py` | Duplicate of `simulation/wc_runner.py` | `simulation/wc_runner.py` |
| `fc.py` | Duplicate of `features/fc.py` | `features/fc.py` |
| `fcd.py` | Duplicate of `features/fcd.py` | `features/fcd.py` |
| `extraction.py` | Duplicate of `features/extraction.py` | `features/extraction.py` |
| `screening.py` | Duplicate of `features/screening.py` | `features/screening.py` |
| `delays.py` | Duplicate of `simulation/delays.py` | `simulation/delays.py` |
| `warmup.py` | Duplicate of `simulation/warmup.py` | `simulation/warmup.py` |
| `qc.py` | Duplicate of `simulation/qc.py` | `simulation/qc.py` |
| `__init__.py` | Root-level package init (mirrors `simulation/__init__.py`) | `simulation/__init__.py` |

**Risk:** Python prefers a package directory over a same-named `.py` file in Python 3,
so `import inference` resolves to the `inference/` package, not `inference.py`. However,
direct `from inference import SomeClass` still works via the package's `__init__.py`.
The root `inference.py` is effectively **shadowed** and dead code, but it remains on disk
and can confuse editors and static analysis tools.

## 14-Step Pipeline Overview

| Step | What | Key function |
|---|---|---|
| 1 | Data split 4:2:2 | `data_loader.three_way_split` |
| 2 | WC simulation (train set) | `inference.step2_simulate_train` |
| 3 | Feature summary stats | `inference.step3_summary_features` |
| 4 | Feature z-score scalers | `inference.step4_fit_feature_scalers` |
| 5 | FC PCA + FCD PCA pipeline | `inference.step5_fit_feature_pipeline` |
| 6 | PCA diagnostic | `inference.step6_pca_diagnostic` |
| 7 | Parameter scaler (→ [-1,1]) | `inference.step7_fit_param_scaler` |
| 8 | Stage 1 SNPE-C training | `inference.step8_train_snpe` |
| 9 | Stage 1 validation + SBC + MLP probe | `evaluate.evaluate_validation_stage1` |
| 10 | θ_bad selection | `inference.select_difficult_params` |
| 11 | Stage 2 SNPE-C training | `inference.run_stage2_snpe` |
| 12 | Stage 2 validation | `evaluate.evaluate_validation_stage2` |
| 13 | Model selection (val only) | `evaluate.select_best_model` |
| 14 | Final test (test set only) | `evaluate.final_test` |

## Key Configuration Constants (`config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `N_REGIONS` | 115 | brain regions |
| `FC_DIM` | 6555 | FC upper-triangle dimension |
| `FCD_DIM` | 5 | FCD summary stats (disabled) |
| `N_TRAIN / N_VAL / N_TEST` | 4 / 2 / 2 | subject split |
| `N_SIM` | 50,000 | Stage 1 simulations per subject |
| `T_END / T_CUT` | 300 s / 60 s | simulation length / transient cut |
| `DT` | 0.5 ms | integration step |
| `TR_SEC` | 1.0 s | BOLD repetition time |
| `ANALYSIS_BOLD_T` | 240 | BOLD TRs after transient cut |
| `STAGE1_PARAMS` | `[P, Q, g_e, g_i]` | Stage 1 inference targets |
| `FEATURE_SET` | `"fc_only"` | only FC upper-triangle used |
