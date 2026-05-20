# 03 — Module Index

All tables reference the **authoritative** source location (package submodules).
Root-level duplicates and compat wrappers are noted separately at the bottom.

---

## simulation/ package

| File | Role | Main functions / classes | Inputs | Outputs | Depends on | Notes |
|---|---|---|---|---|---|---|
| `simulation/__init__.py` | Re-export hub | (all names below) | — | — | simulation.* submodules | Same content as root `__init__.py` |
| `simulation/wc_runner.py` | GPU WC engine | `simulate_gpu_batch`, `simulate_single`, `_run_streaming_hrf`, `detect_engine_key`, `to_numpy`, `normalize_ts` | SC weights (N,N), theta_batch (M,P), param_names | list of BOLD arrays (T_bold, N) each | config, bold.BoldMonitor, vbi.WC_sde, simulation.delays | Per-sim param injection; falls back to per-theta loop if VBI rejects arrays |
| `simulation/delays.py` | Tract-length → delay | `compute_delay_matrix`, `detect_delay_key`, `apply_delay` | SC weights (N,N), lengths_mm (N,N), velocity | delays (N,N) float64 or None | config, simulation.wc_runner | 1 m/s = 1 mm/ms; no extra unit conversion |
| `simulation/warmup.py` | Warm-start batch | `WarmupResult`, `warmup_run`, `simulate_with_warmup` | SC, delays, WC params | WarmupResult (x0 state, BoldMonitor) | config, bold.BoldMonitor, simulation.wc_runner, simulation.delays | Pre-fills BoldMonitor stock buffers once per subject |
| `simulation/qc.py` | Simulation QC | `assert_theta_feature_distinct`, `theta_feature_diff_norm`, `run_theta_specific_check` | theta pairs, feature pairs | pass/fail asserts | — | Guards against batch-mean regression (theta_i must produce distinct x_i) |

---

## features/ package

| File | Role | Main functions / classes | Inputs | Outputs | Depends on | Notes |
|---|---|---|---|---|---|---|
| `features/__init__.py` | Re-export hub | (all names below) | — | — | features.* submodules | — |
| `features/fc.py` | FC computation | `compute_fc`, `fc_to_upper_tri` | BOLD (T, N) or FC (N, N) | FC (N, N) or vec (6555,) float32 | config (NAN_MASK) | NaN → 0; no z-scoring |
| `features/fcd.py` | FCD computation | `compute_sim_fcd_matrix`, `fcd_to_upper_tri`, `fcd_to_summary_stats` | BOLD (T, N) | FCD (N, N) or vec (6555,) or summary (5,) | config (FCD_WINDOW_TR, FCD_STRIDE_TR) | Disabled by default (USE_FCD=False); element-wise std of sliding-window FCs |
| `features/extraction.py` | Batch extraction | `extract_features`, `extract_observed_features`, `extract_simulated_features`, `worker_extract` | list of BOLD arrays or subject dict | (fc_raw, fcd_raw) arrays | features.fc, features.fcd, config | Parallel worker support via `worker_extract` |
| `features/screening.py` | Feature screening | (future) | — | — | — | Reserved for informative-dimension selection |

---

## inference/ package

| File | Role | Main functions / classes | Inputs | Outputs | Depends on | Notes |
|---|---|---|---|---|---|---|
| `inference/__init__.py` | Re-export hub | (all names below) | — | — | inference.* submodules | Replaces old `inference.py` monolith |
| `inference/scaling.py` | Parameter scaling | `ParameterScaler`, `make_stage1_param_scaler`, `make_stage2_param_scaler` | param_names, prior_low, prior_high | scaled theta in [-1,1] | config | Data-free; derived from prior bounds only |
| `inference/priors.py` | SBI prior | `make_scaled_prior` | param_names, low, high | sbi.utils.BoxUniform in [-1,1] | config, sbi | Prior is defined in scaled space |
| `inference/feature_pipeline.py` | Feature pre-processing | `FamilyScaler`, `FCPCAScaler`, `FeaturePipeline` | fc_raw (M, 6555), fcd_raw (M, 5 or 6555) | x_input (M, embed_dim) | config, sklearn.PCA | FIT on training data only; never on val/test |
| `inference/embedding.py` | MLP head | `FeatureEmbedding` | x_input (M, pca_out_dim) | embedding (M, EMBED_DIM=128) | config, torch | Jointly trained with SNPE-C |
| `inference/training_data.py` | Sim + extraction | `step2_simulate_train`, `step3_summary_features`, `collect_training_data`, `save/load_extracted_features` | train_subjects, subject_data, prior_scaled | theta_scaled, fc_raw, fcd_raw | simulation.*, features.*, inference.scaling | Parallel simulation across subjects |
| `inference/snpe.py` | SNPE-C training | `step4_fit_feature_scalers`, `step5_fit_feature_pipeline`, `step6_pca_diagnostic`, `step7_fit_param_scaler`, `step8_train_snpe`, `train_snpe` | fc_raw, fcd_raw, theta_scaled | posterior, embedding_net, pipeline | config, sbi, inference.feature_pipeline, inference.embedding | Uses MAF (8 transforms, hidden=128) |
| `inference/stage1.py` | Stage 1 driver | `run_stage1_snpe` | train_subjects, subject_data, n_sim | artifacts dict | inference.snpe, inference.training_data | Chains steps 2-8 in one call |
| `inference/stage2.py` | Stage 2 driver | `run_stage2_snpe`, `build_stage2_param_set`, `select_theta_bad`, `select_difficult_params`, `collect_stage2_data` | stage1_artifacts, val subjects | stage2_artifacts | inference.stage1, inference.snpe, inference.posterior | theta_bad = high-sensitivity + low-shrinkage params |
| `inference/posterior.py` | Posterior ops | `transform_observed`, `infer_subject_raw`, `compute_shrinkage_raw`, `compute_shrinkage_scaled`, `posterior_correlation`, `posterior_predictive_check` | posterior, observed_fc, param_scaler | raw posterior samples, shrinkage scalars | config, inference.scaling | N_POSTERIOR=2000 samples by default |
| `inference/diagnostics.py` | QC diagnostics | `simulation_based_calibration`, `evaluate_embedding_probing` | posterior, prior, subject_data | SBC rank stats, R² probe scores | config, sbi, inference.posterior | SBC: 200 rounds, 20 bins; MLP probe R²≥0.5 threshold |
| `inference/io.py` | Artifact I/O | `save_artifacts`, `load_artifacts` | artifacts dict, path | pickle files on disk | config | Writes to config.OUTPUT_DIR |
| `inference/_utils.py` | Internal helpers | `_progress` | message string | timestamped stdout | — | Used by all inference submodules |

---

## evaluation/ package

| File | Role | Main functions / classes | Inputs | Outputs | Depends on | Notes |
|---|---|---|---|---|---|---|
| `evaluation/__init__.py` | Re-export hub | (all names below) | — | — | evaluation.* submodules | — |
| `evaluation/metrics.py` | FC/FCD metrics | `fc_metrics`, `fcd_vec_rmse`, `fcd_summary_rmse`, `bootstrap_ci`, `evaluate_subject`, `baseline_eval`, `baseline_eval_subjects` | simulated FC, observed FC | corr, RMSE, bootstrap CI | config, features.* | `fc_metrics` returns (corr, rmse) tuple |
| `evaluation/validation.py` | Val set eval | `evaluate_validation_stage1`, `evaluate_validation_stage2` | val_subjects, stage_artifacts | (results dict, agg dict) | evaluation.metrics, inference.posterior | Never touches test subjects |
| `evaluation/model_selection.py` | Model selection | `compute_selection_score`, `select_best_model` | stage1/2 val results | best model label, score table | config (SELECT_W_*) | Score = w_fc_corr×FC_corr - w_fc_rmse×FC_RMSE - w_fcd_rmse×FCD_RMSE |
| `evaluation/final_test.py` | Test set eval | `final_test` | test_subjects, selected model artifacts | test results | evaluation.metrics, inference.posterior | One-shot; never used for tuning |
| `evaluation/plots.py` | Visualizations | `plot_posteriors`, `plot_fc_comparison`, `plot_sbc_rank_histogram`, `plot_pca_diagnostic`, `plot_one_simulation`, `plot_*_two_stage` | artifacts, results | matplotlib figures | matplotlib, config | Saves PNGs to OUTPUT_DIR |
| `evaluation/reports.py` | Console reports | `report_step1`..`report_step14`, `print_summary_two_stage`, `evaluate_all_two_stage`, `print_final_summary` | step results / artifacts | formatted stdout | evaluation.metrics | One report function per pipeline step |

---

## pipelines/ package

| File | Role | Main functions / classes | Inputs | Outputs | Depends on | Notes |
|---|---|---|---|---|---|---|
| `pipelines/__init__.py` | Re-export | `run_pipeline` | — | — | pipelines.stage1_stage2 | — |
| `pipelines/stage1_stage2.py` | 14-step driver | `run_pipeline`, `step_data_split`, `stage1_pipeline`, `stage2_pipeline`, … | env knobs (N_SIM, RUN_STAGE2, …) | artifacts dict + saved files | data_loader, inference, evaluate, config | Main orchestrator; enforces train/val/test discipline |

---

## Root-level standalone modules (not packages)

| File | Role | Main functions / classes | Depends on | Notes |
|---|---|---|---|---|
| `config.py` | Hyperparameter store | `print_config()` | os, torch (optional) | Single source of truth; edit only this to tune pipeline |
| `data_loader.py` | Data I/O | `load_raw_data`, `load_all_subjects`, `get_target_subjects`, `three_way_split`, `load_atlas_labels` | config, numpy, scipy.io, pandas | SC: log1p + max-norm; FC: NaN → 0 |
| `bold.py` | Hemodynamic transform | `BoldMonitor`, `tvb_hrf` | config, numpy, tvb.datatypes.equations | TVB MixtureOfGammas HRF; mouse-tuned (peak ~3 s) |
| `pipeline_setup.py` | Config override helper | `PipelineConfig`, `setup_pipeline` | config (mutates module attrs) | Used in notebook for quick param changes |
| `main.py` | CLI entry point | `main()` | pipelines | Reads env vars: N_SIM, RUN_STAGE2, SENS_THRESHOLD, SHR_THRESHOLD |
| `debug.py` | Smoke tests | various `check_*` functions | most modules | `--basic` for no-GPU, `--all` for GPU |
| `debug_notebook.py` | Notebook debug cells | debug helpers | most modules | Used inside `main.ipynb` cells |

---

## Root-level compat / legacy files (DO NOT import directly in new code)

| File | Status | Points to |
|---|---|---|
| `simulator.py` | Compat wrapper | `simulation/` + `features/` |
| `evaluate.py` | Compat wrapper | `evaluation/` |
| `inference.py` | OLD MONOLITH (55 KB, dead code) | Superseded by `inference/` package |
| `wc_runner.py` | Root duplicate | `simulation/wc_runner.py` |
| `fc.py` | Root duplicate | `features/fc.py` |
| `fcd.py` | Root duplicate | `features/fcd.py` |
| `extraction.py` | Root duplicate | `features/extraction.py` |
| `screening.py` | Root duplicate | `features/screening.py` |
| `delays.py` | Root duplicate | `simulation/delays.py` |
| `warmup.py` | Root duplicate | `simulation/warmup.py` |
| `qc.py` | Root duplicate | `simulation/qc.py` |
| `__init__.py` (root) | Duplicate of `simulation/__init__.py` | Makes repo root importable as a package |
