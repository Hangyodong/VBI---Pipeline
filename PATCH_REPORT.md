# Patch Report

## Main fixes

1. Removed old Stage-1 parameter set `['c_ee','g_e','noise_amp']`.
   - Stage 1 now uses `['P','Q','g_e','g_i']`.
   - Stage 2 uses `theta_bad + ['c_ee','c_ei','c_ie','c_ii']`.

2. Changed default data dimensions to 116 regions.
   - `FC_DIM = 116*115/2 = 6670`.
   - Paths now expect `MPTP_FC_data_116.mat`, `MPTP_SC_data_116.mat`, and `atlas_116_labels.txt`.

3. Split SC weight and tract length handling.
   - `SC_WEIGHT_COL=1` for coupling.
   - `SC_LENGTH_COL=2` for delay.
   - `delay_ms = lengths_mm / VELOCITY_M_PER_S`.

4. Removed batch-mean theta simulation bug.
   - `simulate_gpu_batch()` passes per-simulation parameter arrays when supported.
   - Falls back to per-theta `num_sim=1` loop if VBI rejects vector parameters.
   - Added theta-specific simulation debug check.

5. Removed observed feature fallback to `x_train.mean()`.
   - Observed feature dimension mismatch now raises `ValueError`.
   - Default `FEATURE_SET='fc_only'`.

6. Added parameter scaling.
   - `ParameterScaler` maps raw params to scaled `[-1,1]` for SBI training.
   - Raw params are restored before VBI simulation.

7. Replaced `main.py` with train/val/test + Stage-1 + optional Stage-2 flow.
   - Validation only is used for model selection.
   - Test is used only after model selection.

8. Replaced `main.ipynb` with setup, debug checks, and full run cells.
   - DEBUG CHECKS cell is inside the notebook immediately after setup.

## Checks run in this environment

```bash
python -m py_compile config.py data_loader.py simulator.py inference.py evaluate.py main.py debug_notebook.py debug.py pipeline_setup.py bold.py
VBI_DATA_DIR=/mnt/data python debug.py --basic
```

Both passed here. GPU/VBI execution was skipped because this sandbox has no VBI/cupy GPU runtime.
