# Mouse MPTP VBI-SBI Pipeline (115 regions)

Whole-brain parameter inference for mouse MPTP using the VBI Wilson-Cowan
model with SBI (SNPE-C).

## 14-step pipeline - one cell per step

| Step | Description | Function |
|------|-------------|----------|
| 1    | Data split (train / val / test)           | `data_loader.three_way_split` |
| 2    | VBI WC simulation (+ streaming features)  | `inference.step2_simulate_train` |
| 3    | Feature extraction summary                | `inference.step3_summary_features` |
| 4    | Feature preprocessing (z-score)           | `inference.step4_fit_feature_scalers` |
| 5    | Feature embedding (FC PCA + FCD PCA)      | `inference.step5_fit_feature_pipeline` |
| 6    | Embedding quality check (PCA diagnostic)  | `inference.step6_pca_diagnostic` |
| 7    | Parameter preprocessing ([-1, 1])         | `inference.step7_fit_param_scaler` |
| 8    | Stage 1 inference (single-round SNPE-C)   | `inference.step8_train_snpe` |
| 9    | Stage 1 analysis + MLP probing + SBC      | `evaluate.evaluate_validation_stage1` |
| 10   | Stage 2 parameter selection               | `inference.select_difficult_params` |
| 11   | Stage 2 inference                          | `inference.run_stage2_snpe` |
| 12   | Stage 2 analysis                          | `evaluate.evaluate_validation_stage2` |
| 13   | Model selection (validation)              | `evaluate.select_best_model` |
| 14   | Final test (one-shot)                     | `evaluate.final_test` |

The MLP linear probing (post-inference half of the embedding quality check)
requires a trained embedding network, so it runs in step 9 instead of 6.

`inference.run_stage1_snpe` remains as a convenience wrapper that chains
steps 2 through 8 in a single call.

## Files

| File | Purpose |
|------|---------|
| `config.py`       | Hyperparameters, paths, prior bounds |
| `data_loader.py`  | Mat / TSV loading, SC scaling, three-way split |
| `bold.py`         | Balloon-Windkessel hemodynamic transform |
| `simulator.py`    | VBI WC simulation + FC / FCD feature extraction |
| `inference.py`    | Step 2-8 functions, scalers, PCA, MLP, SNPE-C, diagnostics |
| `evaluate.py`     | Validation / test metrics, model selection, plots |
| `main.py`         | 14-step pipeline driver (one function per step) |
| `main.ipynb`      | Notebook version, one cell per step |
| `debug.py`        | PASS / FAIL unit-style tests |

## Run

```bash
python main.py               # full pipeline
python debug.py --basic      # quick checks, no GPU
python debug.py --all        # everything (GPU recommended)
```

## Data conventions

- FC source : `MPTP_FC_115.mat` col 1 (NaN replaced with 0)
- FCD source: `MPTP_FC_115.mat` col 2 (used directly, no computation)
- SC source : `MPTP_SC_115.mat` col 1 (uint16 raw -> log1p + max-norm)
- Feature dim: FC = FCD = 6555 (115 * 114 / 2 upper triangle)

## Style

All modules conform to `pycodestyle --max-line-length=88`. Verified.
