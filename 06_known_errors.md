# 06 — Known Errors and Import Conflict Risks

## Error Template

Use this format when logging a new error:

```
## ERROR: <short title>
**Symptom:** <what the user sees>
**Root cause:** <why it happens>
**Fix:** <how to resolve it>
**Affected files:** <list of files>
**Status:** open / resolved / mitigated
```

---

## Import Conflict: `inference.py` vs `inference/` package

**Symptom:**
An editor or static analysis tool (pylint, pyright, mypy) reports that `inference.SomeClass`
does not exist, or imports from `inference` resolve to the wrong file.

**Root cause:**
Both `inference.py` (55 KB monolith) and `inference/` (package directory with `__init__.py`)
exist in the repo root. In Python 3, a package directory takes priority over a same-named
`.py` file, so `import inference` correctly resolves to `inference/`. However:
- Editors sometimes cache stale resolutions and point to `inference.py`
- `__pycache__/inference.cpython-310.pyc` (if present) may be stale
- `inference.py` is dead code but is misleading

**Fix:**
```bash
# Verify which inference is loaded
python -c "import inference; print(inference.__file__)"
# Should print: …/vbi/inference/__init__.py
# NOT: …/vbi/inference.py

# Clear pycache if stale
find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +
```

**Affected files:** `inference.py`, `inference/__init__.py`
**Status:** mitigated (Python 3 package-wins rule; root file is dead but on-disk)

---

## Import Conflict: Root Duplicate Files Shadowing Package Modules

**Symptom:**
`from fc import compute_fc` imports from root `fc.py` instead of `features/fc.py`.
`from wc_runner import simulate_gpu_batch` imports from root `wc_runner.py` instead of
`simulation/wc_runner.py`.

**Root cause:**
The following root-level files have the same name as package submodules:

| Root file | Conflicts with |
|---|---|
| `wc_runner.py` | `simulation/wc_runner.py` |
| `fc.py` | `features/fc.py` |
| `fcd.py` | `features/fcd.py` |
| `extraction.py` | `features/extraction.py` |
| `screening.py` | `features/screening.py` |
| `delays.py` | `simulation/delays.py` |
| `warmup.py` | `simulation/warmup.py` |
| `qc.py` | `simulation/qc.py` |

A bare `import fc` or `from fc import compute_fc` (without the `features.` prefix) hits
the root file, not the package submodule. The content is the same in most cases, but if
root files drift, divergence will cause subtle bugs.

**Fix:**
Always use fully qualified imports:
```python
# CORRECT
from features.fc import compute_fc
from simulation.wc_runner import simulate_gpu_batch
from simulation.delays import compute_delay_matrix

# RISKY (resolves to root duplicate, not package submodule)
from fc import compute_fc
import wc_runner
```

**Affected files:** All root-level duplicates listed above.
**Status:** open (root files not yet removed)

---

## Missing or Misplaced `__init__.py`

**Symptom:**
`ModuleNotFoundError: No module named 'simulation'` or
`ImportError: cannot import name 'simulate_gpu_batch' from 'simulation'`

**Root cause:**
If `simulation/__init__.py`, `features/__init__.py`, `inference/__init__.py`,
`evaluation/__init__.py`, or `pipelines/__init__.py` is absent or misnamed,
Python treats the directory as a namespace package and the re-exports inside
`__init__.py` are not available.

**Fix:**
```bash
# Verify all __init__.py files exist
ls simulation/__init__.py features/__init__.py inference/__init__.py evaluation/__init__.py pipelines/__init__.py
```

All five should be present. If any is missing, the package is broken.

---

## Running Scripts from a Subdirectory

**Symptom:**
`ModuleNotFoundError: No module named 'config'` when running a script from inside
`simulation/`, `inference/`, or `pipelines/`.

**Root cause:**
`config.py`, `data_loader.py`, `bold.py`, etc. live at the repo root. When Python adds
the script's directory to `sys.path[0]`, those root-level modules are not found.

**Fix:**
Always `cd` to the repo root before running any script:
```bash
cd /scratch/home/wog3597/vbi
python main.py             # correct
python simulation/wc_runner.py  # wrong — use import instead
```

---

## `__pycache__` Confusion After Refactor

**Symptom:**
After the Phase 1-4 refactor, old `.pyc` files in `__pycache__` may cause Python to
import stale compiled versions of moved or renamed modules.

**Fix:**
```bash
find /scratch/home/wog3597/vbi -name "*.pyc" -delete
find /scratch/home/wog3597/vbi -name "__pycache__" -type d -exec rm -rf {} +
```

---

## Root `__init__.py` Makes Repo a Package

**Symptom:**
`import vbi` works unexpectedly and exposes simulation names at the top level.
Or `python -c "import vbi; print(dir(vbi))"` shows WC simulation functions.

**Root cause:**
`/scratch/home/wog3597/vbi/__init__.py` exists (mirrors `simulation/__init__.py`).
If `/scratch/home/wog3597` is in `sys.path`, the `vbi` directory is importable as a
package named `vbi`.

**Impact:** Low unless something adds the parent directory to `sys.path`.
**Status:** open (root `__init__.py` exists and is a duplicate)

---

## `cupy` / NVRTC Compilation Failures

**Symptom:**
```
RuntimeError: Per-theta fallback also failed at theta 0:
  Fix: pip install nvidia-cuda-nvrtc-cu12 --force-reinstall
```
or:
```
cupy NVRTC compilation issue with mismatched CUDA headers
```

**Root cause:**
The `BoldMonitor` inside `_run_streaming_hrf` must run on CPU (`xp=np`). If it
accidentally runs on the cupy device, NVRTC compilation of the bold kernel fails.

**Fix:**
```bash
pip install nvidia-cuda-nvrtc-cu12 --force-reinstall
# AND ensure BoldMonitor is always called with xp=np (already enforced in code)
```

---

## FCD Disabled but Code Still References It

**Symptom:**
`USE_FCD = False` but code paths that build `fcd_raw` still execute (returning
empty arrays). No crash, but confusion about which features are active.

**Root cause:**
`config.USE_FCD = False` and `config.FEATURE_SET = "fc_only"`. FCD computation
functions exist and are importable but produce empty arrays or are skipped.

**Fix:**
No code change needed. To enable FCD, set `USE_FCD = True` and `FEATURE_SET = "fc_fcd"`
in `config.py` AND provide empirical BOLD time series.

---

## `ANALYSIS_BOLD_T` Mismatch

**Symptom:**
Dimension mismatch in `FeaturePipeline` when observed FC shape does not match
simulated FC shape.

**Root cause:**
`config.ANALYSIS_BOLD_T = 240` assumes `T_END=300000, T_CUT=60000, DT=0.5, DECIMATE=2, TR_SEC=1.0`.
Changing any of these without recomputing the derived constant causes mismatches.

**Fix:**
`ANALYSIS_BOLD_T` is computed dynamically in `config.py`:
```python
ANALYSIS_BOLD_T = int((T_END - T_CUT) / (DT * DECIMATE) /
                      (TR_SEC * 1000.0 / (DT * DECIMATE)))
```
It recalculates on import. Use `pipeline_setup.setup_pipeline(T_END_MS=..., T_CUT_MS=...)`
rather than patching constants manually.

---

## PATCH_REPORT: Dimension Was 116 (Now 115)

**Symptom:**
Code or data files that expect 116 regions fail with shape mismatches.

**Root cause:**
An earlier version of the pipeline used 116 regions (`FC_DIM = 6670`). The current
production code uses **115 regions** (`FC_DIM = 6555`). The PATCH_REPORT documents
this change.

**Fix:**
Ensure all `.mat` files are the 115-region variants (`MPTP_FC_115.mat`, `MPTP_SC_115.mat`).
Do not use 116-region data files with the current `config.py`.

---

## Legacy Files in `sys.modules` After Dynamic Reload

**Symptom:**
In a long-running notebook session, `importlib.reload(config)` does not propagate
to already-imported submodules (e.g., `simulation.wc_runner` still holds old
`config.GPU_BATCH`).

**Fix:**
Use `pipeline_setup.setup_pipeline(...)` which handles reload ordering, or restart
the kernel after changing `config.py`.
