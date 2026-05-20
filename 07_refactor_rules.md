# 07 — Refactor Rules

Rules to follow when modifying this codebase. Scientific behavior is the highest
priority; structural cleanliness is secondary.

---

## R1: Do Not Change Scientific Behavior Unless Explicitly Requested

- Never alter numerical values in `config.py` (WC_FIXED, prior bounds, BW params,
  HRF params) without an explicit instruction.
- Never change the FC or FCD computation logic (`compute_fc`, `compute_sim_fcd_matrix`).
- Never change the ParameterScaler mapping or the SBI prior bounds.
- Never swap `NAN_MASK` handling (NaN → 0 convention is intentional; see config comment).
- Never change the train/val/test split logic or SEED.
- `ANALYSIS_BOLD_T = 240` is derived; never hardcode it elsewhere.

---

## R2: Prefer Small, Surgical Patches

- Fix one thing per commit. A bug fix should not include cleanup of unrelated code.
- If you notice a smell while fixing something, log it in `06_known_errors.md` rather
  than fixing it on the spot unless it is directly related to the bug.
- Avoid reformatting entire files; it produces noisy diffs and makes review harder.

---

## R3: Keep simulation / features / inference / evaluation Separated

- `simulation/` must not import from `features/`, `inference/`, or `evaluation/`.
- `features/` must not import from `simulation/`, `inference/`, or `evaluation/`.
- `inference/` may import from `simulation/` and `features/` but not `evaluation/`.
- `evaluation/` may import from all packages.
- `pipelines/` is the only module allowed to import from all four packages.
- `config.py` is a special case: all packages may import from `config`.
- `data_loader.py` may be imported by `pipelines/` and `inference/training_data.py`.
- `bold.py` may be imported by `simulation/wc_runner.py` and `simulation/warmup.py`.

Dependency graph (arrows = "may import"):
```
config  ←──────────────────────── all packages
data_loader ←────────────────── inference/training_data, pipelines
bold ←─────────────────────── simulation/wc_runner, simulation/warmup
simulation ←─────────────── inference, pipelines
features ←───────────────── inference, evaluation, pipelines
inference ←──────────────── evaluation, pipelines
evaluation ←─────────────── pipelines
pipelines ←──────────────── main.py
```

---

## R4: Avoid Circular Imports

- `simulation.delays` → `simulation.wc_runner` would be circular. `delays` must
  never import from `wc_runner`; `wc_runner` imports from `delays`.
- `inference.snpe` → `inference.stage1` would be circular. `stage1` imports from
  `snpe`; `snpe` must not import from `stage1`.
- If you find yourself needing a circular import, introduce a new helper module or
  move the shared code to `inference/_utils.py`.

---

## R5: Avoid `sys.path` Hacks Unless Unavoidable

- Do not add `sys.path.insert(0, ...)` inside any module that ships in the repo.
- If a script must be run from a different directory, document it in `05_runbook.md`
  as an exception, not a code pattern.
- `pipeline_setup.py` is the only file that may manipulate `sys.modules` (for
  module reload ordering), and only at user request via `setup_pipeline()`.

---

## R6: Preserve Public Function Names

- When moving logic into a new submodule, keep the original function name and
  add a re-export in the package `__init__.py`.
- If a function must be renamed for clarity, keep the old name as an alias for
  at least one refactor cycle and mark it with a deprecation comment.
- Existing callers (`inference.step2_simulate_train`, `evaluate.fc_metrics`, etc.)
  must continue to work after any structural change.

---

## R7: Package-Level Modules Are Preferred Over Root-Level Duplicates

- All new code that calls simulation functions must import from `simulation.*`.
- All new code that calls feature functions must import from `features.*`.
- All new code that calls inference classes/functions must import from `inference.*`.
- All new code that calls evaluation functions must import from `evaluation.*`.
- The root-level duplicates (`fc.py`, `wc_runner.py`, `delays.py`, etc.) are
  legacy artifacts and should eventually be removed, but only after verifying
  no external caller depends on the bare name.
- `simulator.py` and `evaluate.py` are compat wrappers that should remain until
  all callers have been updated to use the package paths.

---

## R8: Do Not Edit `inference.py` (Root Monolith)

- `inference.py` is dead code. Python loads `inference/` package instead.
- Any edit to `inference.py` will not take effect and will cause confusion.
- Do not add new functions to `inference.py`; add them to the relevant submodule
  in `inference/` and re-export from `inference/__init__.py`.

---

## R9: Feature Toggles Are in `config.py` Only

- `USE_FC`, `USE_FCD`, `USE_PSD`, `FEATURE_SET` are in `config.py`.
- Do not add feature-toggle logic inside submodules; read from `config.*` instead.
- When adding a new feature type, add a boolean toggle and a dim constant to `config.py`.

---

## R10: Train-Only Fitting — Never Fit on Val or Test Data

- `FamilyScaler`, `FCPCAScaler`, `FeaturePipeline`: fit only on training simulations.
- `ParameterScaler`: derived from prior bounds (data-free), safe to use everywhere.
- `evaluate_validation_*` and `final_test`: must never call `.fit()` on the pipeline.
- If adding a new pre-processing step, mark clearly whether it is fit-on-train-only.

---

## R11: Notebook Cells Must Not Import From Root Duplicates

- Notebook cells in `main.ipynb` should use package imports:
  ```python
  from features.fc import compute_fc        # correct
  from fc import compute_fc                 # wrong (root duplicate)
  ```
- If a notebook cell currently uses a root duplicate, update it to the package path.

---

## R12: No Hardcoded Subject Counts or Array Dimensions

- Use `config.N_REGIONS`, `config.FC_DIM`, `config.ANALYSIS_BOLD_T`, etc.
- Do not write `115`, `6555`, or `240` as magic numbers in new code.

---

## R13: GPU / CPU Parity

- `BoldMonitor` must always be called with `xp=np` (CPU numpy). Do not pass `cp`.
- `WC_sde.heunStochastic` runs on GPU (cupy) and must not be called with numpy arrays.
- Helper functions (`to_numpy`, `normalize_ts`) handle array transfer; use them rather
  than calling `.get()` directly in high-level code.
