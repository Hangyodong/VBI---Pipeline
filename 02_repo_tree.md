# 02 — Repository Tree

Generated with:
```bash
tree -I "__pycache__|*.pyc|.git|output|output_mouse_mptp|*.mat|*.png|.ipynb_checkpoints"
```

```
.
├── 01_repo_overview.md
├── 02_repo_tree.md
├── 03_module_index.md
├── 04_data_flow.md
├── 05_runbook.md
├── 06_known_errors.md
├── 07_refactor_rules.md
├── 08_claude_project_upload_guide.md
├── atlas_115_labels.txt          # region label lookup (115 regions)
├── bold.py                       # Balloon-Windkessel BoldMonitor
├── config.py                     # ALL hyperparameters and paths
├── data_loader.py                # .mat / .tsv loading, SC scaling, split
├── debug_notebook.py             # debug cell helpers for main.ipynb
├── debug.py                      # standalone unit-style debug tests
├── delays.py                     # [ROOT DUPLICATE] → simulation/delays.py
├── evaluate.py                   # [COMPAT WRAPPER] → evaluation/ package
├── evaluation/
│   ├── __init__.py               # re-exports all public names
│   ├── final_test.py             # final_test() — test set only
│   ├── metrics.py                # fc_metrics, fcd_vec_rmse, bootstrap_ci
│   ├── model_selection.py        # compute_selection_score, select_best_model
│   ├── plots.py                  # plot_posteriors, plot_fc_comparison, etc.
│   ├── reports.py                # report_step1..14, print_*_summary
│   └── validation.py             # evaluate_validation_stage1/2
├── extraction.py                 # [ROOT DUPLICATE] → features/extraction.py
├── fc.py                         # [ROOT DUPLICATE] → features/fc.py
├── fcd.py                        # [ROOT DUPLICATE] → features/fcd.py
├── features/
│   ├── __init__.py               # re-exports all public names
│   ├── extraction.py             # extract_features, extract_observed/simulated
│   ├── fc.py                     # compute_fc, fc_to_upper_tri
│   ├── fcd.py                    # compute_sim_fcd_matrix, fcd_to_upper_tri
│   └── screening.py              # (future) informative-dimension screens
├── inference/
│   ├── __init__.py               # re-exports all public names
│   ├── _utils.py                 # _progress (internal)
│   ├── diagnostics.py            # simulation_based_calibration, evaluate_embedding_probing
│   ├── embedding.py              # FeatureEmbedding (MLP)
│   ├── feature_pipeline.py       # FamilyScaler, FCPCAScaler, FeaturePipeline
│   ├── io.py                     # save_artifacts, load_artifacts
│   ├── posterior.py              # infer_subject_raw, compute_shrinkage_*, posterior_*
│   ├── priors.py                 # make_scaled_prior
│   ├── scaling.py                # ParameterScaler, make_stage*_param_scaler
│   ├── snpe.py                   # step4-8, train_snpe
│   ├── stage1.py                 # run_stage1_snpe (chains steps 2-8)
│   ├── stage2.py                 # run_stage2_snpe, select_theta_bad, build_stage2_param_set
│   └── training_data.py          # step2_simulate_train, step3_summary_features
├── inference.py                  # [OLD MONOLITH, 55 KB] superseded by inference/ package
├── __init__.py                   # [ROOT] mirrors simulation/__init__.py
├── install.sh                    # environment setup script
├── main.ipynb                    # notebook version (one cell per step)
├── main.py                       # thin CLI entry point → pipelines.run_pipeline
├── participants.tsv              # subject metadata (ID, group: ctr / MPTP)
├── PATCH_REPORT.md               # history of key fixes and dimension changes
├── pipeline_setup.py             # PipelineConfig dataclass + setup_pipeline()
├── pipelines/
│   ├── __init__.py               # exports run_pipeline
│   └── stage1_stage2.py          # 14-step orchestration driver
├── qc.py                         # [ROOT DUPLICATE] → simulation/qc.py
├── README.md                     # original README (may be outdated)
├── requirements.txt              # Python dependencies
├── screening.py                  # [ROOT DUPLICATE] → features/screening.py
├── simulation/
│   ├── __init__.py               # re-exports all public simulation names
│   ├── delays.py                 # compute_delay_matrix, apply_delay
│   ├── qc.py                     # assert_theta_feature_distinct, run_theta_specific_check
│   ├── warmup.py                 # WarmupResult, warmup_run, simulate_with_warmup
│   └── wc_runner.py              # simulate_gpu_batch, simulate_single, BoldMonitor integration
├── simulator.py                  # [COMPAT WRAPPER] → simulation/ + features/ packages
├── warmup.py                     # [ROOT DUPLICATE] → simulation/warmup.py
└── wc_runner.py                  # [ROOT DUPLICATE] → simulation/wc_runner.py

Data files (excluded from tree above):
  MPTP_FC_115.mat                 # per-subject FC and FCD matrices (115×115)
  MPTP_SC_115.mat                 # per-subject SC weights and tract lengths (115×115)
```

## Legend

| Marker | Meaning |
|---|---|
| `[ROOT DUPLICATE]` | Same code as the authoritative package file; pre-refactor artifact |
| `[COMPAT WRAPPER]` | Thin re-export shim; contains no logic; exists for backward-compatible imports |
| `[OLD MONOLITH]` | Pre-refactor file with all logic; now shadowed by the package directory |
| `[ROOT]` | Root-level package init that duplicates `simulation/__init__.py` |
