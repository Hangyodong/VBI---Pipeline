# 05 — Runbook

## Environment Notes

- Python 3.10, CUDA 12.x, H100 NVL (production target)
- GPU required for production runs (cupy + VBI WC_sde)
- CPU-only mode works for imports, config checks, and debug tests without GPU
- Always run scripts from the **repository root** (`/scratch/home/wog3597/vbi`)
  to ensure `import config`, `import data_loader`, `from simulation import …`
  resolve correctly

## Package Installation

```bash
# 1. Core scientific stack + SBI
pip install -r requirements.txt

# 2. GPU array library (choose one based on CUDA version)
pip install cupy-cuda12x          # for CUDA 12.x
# pip install cupy-cuda11x        # for CUDA 11.x

# 3. VBI Wilson-Cowan backend
pip install vbi==0.4.3

# 4. TVB (for BoldMonitor HRF)
pip install tvb-library           # or as required by install.sh
```

See `install.sh` for the full environment setup sequence.

## Recommended Execution Location

Always run from the repository root:

```bash
cd /scratch/home/wog3597/vbi
python main.py
```

**Do NOT run from a subdirectory** (e.g., from inside `simulation/` or `inference/`).
Imports like `import config` or `from simulation import …` rely on the repo root
being in `sys.path`, which `python script.py` guarantees when run from root.

## Verify Installation (No GPU Required)

```bash
# Compile-check all core modules
python -m py_compile config.py data_loader.py bold.py main.py pipeline_setup.py
python -m py_compile simulation/wc_runner.py simulation/delays.py simulation/warmup.py
python -m py_compile features/fc.py features/fcd.py features/extraction.py
python -m py_compile inference/scaling.py inference/priors.py inference/feature_pipeline.py
python -m py_compile pipelines/stage1_stage2.py

# Test imports (no GPU, no VBI)
python -c "import config; config.print_config()"
python -c "from simulation import compute_delay_matrix, simulate_gpu_batch"
python -c "from features import compute_fc, extract_features"
python -c "from inference import ParameterScaler, run_stage1_snpe"
python -c "from evaluation import fc_metrics, final_test"
python -c "from pipelines import run_pipeline"
```

## Test Data Loading

```bash
python -c "
import data_loader
out = data_loader.load_raw_data()
df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
print('FC mat shape:', fc_mat.shape)
print('SC mat shape:', sc_mat.shape)
print('Subject IDs:', fc_ids[:3])
"
```

## Debug Smoke Tests (No GPU)

```bash
python debug.py --basic
```

## Full Debug Suite (GPU Required)

```bash
python debug.py --all
```

## Run Full Pipeline (Production)

```bash
# Full pipeline with config defaults (N_SIM=50000, T_END=300s)
python main.py

# Override n_sim via environment variable
N_SIM=5000 python main.py

# Skip Stage 2
RUN_STAGE2=0 python main.py

# Override theta_bad selection thresholds
SENS_THRESHOLD=0.6 SHR_THRESHOLD=0.15 python main.py

# Stage 2 n_sim override
N_SIM=5000 N_SIM_S2=5000 python main.py
```

## Run Stage 1 Only (Interactive / Notebook)

```python
from pipeline_setup import setup_pipeline
from pipelines.stage1_stage2 import step_data_split, stage1_pipeline

# Optional: reduce N_SIM for quick test
setup_pipeline(N_SIM=1000, T_END_MS=5000, T_CUT_MS=1000)

train, val, test, subject_data = step_data_split()
stage1_arts = stage1_pipeline(train, subject_data, n_sim=1000)
```

## Run Stage 2 Only (After Stage 1)

```python
from inference import select_difficult_params, run_stage2_snpe

theta_bad = select_difficult_params(stage1_arts)
stage2_arts = run_stage2_snpe(
    train_subjects=train,
    subject_data=subject_data,
    stage1_artifacts=stage1_arts,
    theta_bad=theta_bad,
    n_sim=1000,
)
```

## Debug a Single Subject Simulation

```bash
python -c "
import config, data_loader
from simulation import simulate_gpu_batch
import numpy as np

out = data_loader.load_raw_data()
df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
subjs = data_loader.get_target_subjects(df, fc_ids, sc_ids)
sd = data_loader.load_all_subjects(subjs[:1], fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids)
sid = subjs[0]
sc = sd[sid]['sc']
delays = sd[sid]['delays']
theta = np.array([[1.0, 0.5, 0.5, 0.5]], dtype=np.float32)
bolds = simulate_gpu_batch(sc, theta, config.STAGE1_PARAMS, delays=delays)
print('BOLD shape:', bolds[0].shape)
"
```

## Check Config

```bash
python -c "import config; config.print_config()"
```

## Debug Mode (Fast Smoke Test, < 5 s simulation)

```bash
python -c "
import config
# Temporarily patch for quick test
config.T_END = 5000.0
config.T_CUT = 1000.0
config.N_SIM = 10
config.SIM_MODE = 'debug'
import debug_notebook
"
```

Or use `pipeline_setup.PipelineConfig(N_SIM=10, T_END_MS=5000, T_CUT_MS=1000)`.

## Artifacts Location

Output files are written to `config.OUTPUT_DIR` (default: `./output_mouse_mptp`).
Use `inference.load_artifacts(path)` to reload saved posteriors.

## Notebook

Open `main.ipynb` in Jupyter or VS Code. Each cell corresponds to one pipeline step.
Run cells in order; each cell prints a step header and diagnostic output.
