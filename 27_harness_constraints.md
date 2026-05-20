# 27 — Claude Code Harness: Constraints

Part of the Claude Code Harness for VBI-SBI pipeline.
Generated: 2026-05-18  Branch: refactor/02-simulation
Repo: /scratch/home/wog3597/vbi

## HOW TO USE THIS FILE

These are the **fences**. Read after 26 (context), before 28
(protocols). Every rule is a binary FORBIDDEN / REQUIRED. If a
proposed change appears to violate one, **stop and surface to the
user** via the SC stop conditions in file 30 — do not look for
workarounds.

Sources cited: `R1–R13` come from `07_refactor_rules.md`. `HC-*`
come from per-task hard constraints in docs 18, 20, 23, 24, 25,
and the GPU-optimization brief.

---

## 2.1 ABSOLUTE PROHIBITIONS

Each line is a binary rule. No exceptions without explicit user
authorization.

### Source-file edits

- ❌ **Edit `config.py` scientific constants** (R1). Includes
  `WC_FIXED`, all `STAGE1_PRIOR_*`, `C_PARAM_PRIOR`, `LOCAL_EI_PARAMS`,
  `SEED`, `DT`, `T_END`, `T_CUT`, `DECIMATE`, `ANALYSIS_BOLD_T`,
  `PCA_DIM_FC`, `PCA_DIM_FCD`, `EMBED_*`, `NDE_*`, `N_SIM*`,
  `N_POSTERIOR`, `N_SBC`, `N_TEST_RESIM`, `SELECT_W_*`,
  `NUISANCE_METHOD`, `VELOCITY_M_PER_S`, `BW`, `HRF_*`.
  Adding paths or non-numerical glue is also out of scope.
- ❌ **Edit `inference.py`** (R8). Dead monolith; package wins;
  delete only after Tier 7 full diff audit.
- ❌ **Edit any of the 8 root duplicates** at the repo root:
  `fc.py`, `fcd.py`, `wc_runner.py`, `delays.py`, `warmup.py`,
  `qc.py`, `extraction.py`, `screening.py`. All are byte-identical to
  their package counterparts (`diff` exit 0).
- ❌ **Edit `bold.py`**. Sole source of `BoldMonitor`; load-bearing.
- ❌ **Edit VBI package source files** under
  `…/site-packages/vbi/…`. Patch at call-site via monkey-patching in
  `simulation/wc_runner.py` only if absolutely necessary.
- ❌ **Edit `evaluate.py` or `simulator.py`** beyond their existing
  re-export-only shape. They are zero-logic compat wrappers and
  scheduled for Tier 5 deletion.
- ❌ **Edit `evaluation/model_selection.py` scoring logic** without
  explicit user authorization. The `SELECT_W_*` weights and
  `compute_selection_score` define which Stage wins on validation
  (R1).
- ❌ **Edit the `fix_mean` nuisance blocks** in
  `evaluation/validation.py:134-141` or
  `evaluation/final_test.py:132-139` without user authorization.
  Hardcoded `fix_mean` at val/test time is intentional, distinct from
  `config.NUISANCE_METHOD` (which governs training-data collection).
- ❌ **Edit root `__init__.py`**. Makes the repo loadable as `vbi`;
  delete only after a final `import vbi` audit in Tier 6.
- ❌ **Edit notebook cells `cell[2]` (Setup) or `cell[3]`+`cell[4]`
  (Integrated VBI Pipeline Debug Cell)**. They are the contract for
  the staged validation workflow (doc 23-25).

### Import / scope

- ❌ **`import evaluate`** anywhere (the old compat wrapper) —
  use `import evaluation as evaluate` instead (P-1).
- ❌ **`from simulator import …` at module level** anywhere —
  deferred imports inside function bodies only, and only as a
  transitional step. Migrate to `simulation.*` / `features.*`
  whenever possible.
- ❌ **`import cupy` / `import torch` / `import sbi` / `import vbi`
  at module scope** of any package file we own. These are deferred
  inside function bodies (HC-5 / HC-12). The pattern allows imports
  to succeed on machines without those packages installed.
- ❌ **`sys.path` manipulation** beyond the `REPO_ROOT` guard in the
  Integrated Debug Cell (R5).

### Scientific / numerical invariants

- ❌ **Change `BoldMonitor` to `xp=cp`** — NVRTC compilation fails;
  `BoldMonitor` must stay on CPU (xp=np). INV-1 in GPU spec.
- ❌ **Move `WC_sde.heunStochastic` to CPU** — must stay on GPU
  (cupy). INV-3.
- ❌ **Break the per-sim parameter shape contract**. WC params
  passed to `WC_sde` must be shape `(n_nodes, csz)`, never scalar
  or `(csz,)`. INV-2, enforced by the assert in
  `simulate_gpu_batch:271-281`.
- ❌ **Change BOLD output shape** `(T_bold, N, S)`. INV-4. T_bold =
  `config.ANALYSIS_BOLD_T = 240`.
- ❌ **Fit `FeaturePipeline` / `FCPCAScaler` / `FamilyScaler`
  on validation or test data** (R10). Fit on training simulations
  only. Re-use the fitted instance for val/test.
- ❌ **Apply z-score to FC before PCA**. FC is raw Pearson r ∈
  [−1, 1]; FCPCAScaler operates on raw upper-triangle vectors.
- ❌ **Mix scaled `[-1, 1]` theta and raw theta arrays silently**.
  SBI trains/samples in scaled space; simulation runs in raw space.
  Always pass through `ParameterScaler.inverse_transform` /
  `transform` at the boundary, and use the **same instance**.
- ❌ **Call `inferer.train()` more than once per SNPE run**. sbi's
  internal state is single-use per training call.
- ❌ **Patch / subclass sbi internals**. Use only the public sbi API;
  read `inferer._summary` only after `train()` returns.

### Process / authorization

- ❌ **Delete files** without an explicit `Proceed with delete X`
  authorization. Includes root duplicates (Tier 6) and the monolith
  (Tier 7).
- ❌ **`git commit`, `git push`, `git rebase`, `git merge`,
  `git reset --hard`, force-push** without an explicit instruction
  for that exact branch operation.
- ❌ **Execute the notebook programmatically** (e.g., `jupyter
  nbconvert --to notebook --execute`). Compile-check cell sources
  only; let the user execute interactively.
- ❌ **Revert P-1, P-2, or GPU-1 patches** without an explicit
  "rollback X" instruction.
- ❌ **Touch a file with un-discussed unstaged `M`-status changes**
  (e.g., `simulation/wc_runner.py` previously had `M` per doc 16).
  `git diff <file>` first; surface to user before editing.

### Tier-X scientific bug

- ❌ **"Fix" the FCD dual-role bug in isolation**. All 8 sites (5
  files) must be migrated together to a corrected
  `extract_observed_features` contract. Tier X requires explicit
  authorization. The bug is dormant while `USE_FCD = False`.

---

## 2.2 IMPORT RULES (precise — no guessing)

| Target | Correct import | Wrong import |
|---|---|---|
| `simulate_gpu_batch` | `from simulation.wc_runner import simulate_gpu_batch` | ❌ `from simulator import simulate_gpu_batch` (module-level) |
| `simulate_single` | `from simulation.wc_runner import simulate_single` | ❌ `from simulator import simulate_single` |
| `compute_delay_matrix` | `from simulation.delays import compute_delay_matrix` | ❌ `from simulator import compute_delay_matrix` (P-2 forbids) |
| `apply_delay`, `detect_delay_key` | `from simulation.delays import ...` | — |
| `warmup_run`, `simulate_with_warmup`, `WarmupResult` | `from simulation.warmup import ...` | — |
| `assert_theta_feature_distinct`, `run_theta_specific_check` | `from simulation.qc import ...` | — |
| `compute_fc`, `fc_to_upper_tri` | `from features.fc import ...` | ❌ `from fc import ...` (root duplicate shadow) |
| `compute_sim_fcd_matrix`, `fcd_to_upper_tri`, `fcd_to_summary_stats` | `from features.fcd import ...` | ❌ `from fcd import ...` |
| `extract_features`, `extract_observed_features`, `worker_extract` | `from features.extraction import ...` | ❌ `from extraction import ...` |
| `ParameterScaler`, `make_stage1_param_scaler`, `make_stage2_param_scaler` | `from inference.scaling import ...` or `from inference import ParameterScaler` | — |
| `FeaturePipeline`, `FCPCAScaler`, `FamilyScaler` | `from inference.feature_pipeline import ...` or `from inference import FeaturePipeline` | — |
| `FeatureEmbedding` | `from inference.embedding import FeatureEmbedding` or `from inference import FeatureEmbedding` | — |
| `run_stage1_snpe`, `run_stage2_snpe`, `save_artifacts`, `load_artifacts` | `from inference import ...` | — |
| `fc_metrics`, `evaluate_validation_stage1/2`, `select_best_model`, `final_test`, `plot_*`, `report_step*`, `print_final_summary` | `from evaluation import ...` | ❌ `import evaluate` (the compat wrapper) |
| `run_pipeline` | `from pipelines import run_pipeline` | — |
| evaluation package (when callers want `evaluate.X` syntax) | `import evaluation as evaluate` (P-1 form) | ❌ `import evaluate` (the wrapper) |
| `cupy`, `torch`, `sbi`, `vbi.models.cupy.wilson_cowan.WC_sde` | **deferred** — inside function body only | ❌ at module top |

If a deferred import is required by signature (e.g.,
`_run_streaming_hrf` needs `cupy` to allocate the stride buffer),
wrap it in `try: import cupy as cp; except Exception:` with a clean
fallback path. See `simulation/wc_runner.py:202-209` for the
canonical pattern.

---

## 2.3 DEPENDENCY RULES (R3 — who can import from whom)

The dependency graph is strict. Verify any new `from X import Y`
against this table before applying.

| Package | May import from | May NOT import from |
|---|---|---|
| `simulation/` | `config`, `bold` | `features/`, `inference/`, `evaluation/`, `pipelines/`, `data_loader` |
| `features/` | `config` | `simulation/`, `inference/`, `evaluation/`, `pipelines/`, `data_loader`, `bold` |
| `inference/` | `config`, `simulation/`, `features/`, `data_loader` (only `inference/training_data.py`) | `evaluation/`, `pipelines/` |
| `evaluation/` | `config`, `simulation/`, `features/`, `inference/` | `pipelines/` |
| `pipelines/` | All packages + `config` + `data_loader` | — |
| `data_loader.py` | `config`, `simulation.delays` (P-2) | `features/`, `inference/`, `evaluation/`, `pipelines/`, `bold` |
| `bold.py` | `config` | All packages |

**Existing dormant violations** (R3 § doc 12):

- `simulation/qc.py:25  from features.fc import compute_fc, fc_to_upper_tri`
  — exists but not exercised at runtime in production; Tier X.

Do not introduce new violations. If a new dependency is needed,
restructure rather than reverse the arrow.

### Circular-import landmines (R4)

- `simulation.delays → simulation.wc_runner` is forbidden.
  `wc_runner` imports `delays`, never the reverse.
- `inference.snpe → inference.stage1` is forbidden.
  `stage1` imports `snpe`.
- If you need shared code: extract to `inference/_utils.py`.

---

## 2.4 SCIENTIFIC INVARIANTS

These are correctness conditions, not style preferences. Violation
breaks the science.

| Invariant | Check |
|---|---|
| FC is raw Pearson r ∈ [-1, 1] — no z-score before PCA | `compute_fc(ts)` returns `np.corrcoef(ts.T)` with NaN→0, diag=0; `FCPCAScaler.fit(fc_train_raw)` takes raw vectors |
| FCD is sliding-window FC correlation **std** (when `USE_FCD=False`, the 5-dim `fcd_to_summary_stats` output of `[mean, std, q25, q50, q75]` is the feature) | `compute_sim_fcd_matrix(bold)` returns `(N, N)`; pipeline path expects 5-dim summary stats |
| `ParameterScaler` is **one instance** for forward + inverse | `inference/stage1.py:run_stage1_snpe` creates one `param_scaler` in step 7, passes it to step 2 (sample) and inference (transform back) |
| `FeaturePipeline` fits on training simulations only (R10) | val/test re-use the fitted pipeline; `pipeline.transform(fc_obs, fcd_obs)` never re-fits |
| `BoldMonitor` uses `xp=np` (INV-1) | `bold.BoldMonitor(..., xp=np, ...)` — never `xp=cp` |
| Per-sim WC params are `(n_nodes, csz)` (INV-2) | `_try_per_sim_params` tiles `chunk[:, i]` to `(n_nodes, csz)`; assert at `wc_runner.py:271` |
| `WC_sde.heunStochastic` stays on GPU (INV-3) | All cupy ops; never moved to numpy |
| BOLD output shape `(T_bold=240, N=115, S=csz)` (INV-4) | `mon.collect(...)` returns `(T_bold, N, S)`; `simulate_gpu_batch` returns `list[ndarray of (T_bold, N)]` per sim |
| n_steps = `int(T_end / dt) = 600,000` (INV-5) | Fixed; do not parameterize |
| NaN mask convention: NaN → 0 in FC loading | `data_loader._load_fc_fcd` replaces NaN with 0; `config.NAN_REGIONS` documents which are NaN |
| Scaled theta space is `[-1, 1]` | `ParameterScaler` maps prior bounds → `[-1, 1]`; SBI trains in scaled space; convert with `.transform()` / `.inverse_transform()` |

---

## 2.5 FILE EDIT AUTHORIZATION MATRIX

| File / dir | Edit allowed? | Conditions |
|---|---|---|
| `config.py` | ❌ NEVER | R1; any change requires explicit user request and a doc trail |
| `inference.py` | ❌ NEVER | R8 — dead monolith |
| `bold.py` | ❌ NEVER | Sole source of `BoldMonitor`; any change must be a deliberate scoped refactor |
| `evaluate.py`, `simulator.py` | ❌ NEVER | Compat wrappers — Tier 5 deletion targets |
| 8 root duplicates | ❌ NEVER | Tier 6 deletion only; `mv` to `_legacy_root/` is OK once authorized |
| Root `__init__.py` | ❌ NEVER | Tier 6 — last to go |
| `main.ipynb cell[2]` (Setup) | ❌ NEVER | All downstream cells depend on its imports |
| `main.ipynb cell[3]`+`cell[4]` (Integrated Debug Cell) | ❌ NEVER | Contract for the staged validation workflow |
| `simulation/wc_runner.py` | ⚠️ With explicit user OK | Contains per-sim param contract + GPU-1 optimization; touching here is high-risk |
| `evaluation/model_selection.py` | ⚠️ With explicit user OK | Scoring weights / formula determine Stage 1 vs 2 winner (R1) |
| `evaluation/validation.py` fix_mean block (lines 134-141) | ⚠️ With explicit user OK | Hardcoded; intentionally distinct from `config.NUISANCE_METHOD` |
| `evaluation/final_test.py` fix_mean block (lines 132-139) | ⚠️ With explicit user OK | Same as above |
| `simulation/qc.py` | ⚠️ With explicit user OK | Has the R3 violation; Tier X to fix |
| All other `simulation/*.py`, `features/*.py`, `inference/*.py`, `evaluation/*.py`, `pipelines/*.py` | ✅ With plan + verify | Standard task protocol (file 28) |
| `data_loader.py` | ✅ With plan + verify | One Tier-2 patch (P-2) already applied at line 278 |
| `main.py`, `pipeline_setup.py`, `debug.py`, `debug_notebook.py` | ✅ With plan + verify | |
| `main.ipynb` cells `5..40` (excluding 2, 3, 4) | ✅ With plan + verify | Notebook protocol (file 28 §3.4) |
| `01_…md` through `30_harness_*.md` | ✅ Append-only | Existing docs are history; create a new numbered doc for new work |

---

## 2.6 PROHIBITED OPERATIONS (process-level)

- ❌ Run `python -m jupyter nbconvert --execute main.ipynb` or any
  programmatic notebook execution.
- ❌ Run `pip install <new-package>` or otherwise mutate the
  installed environment.
- ❌ Add a new top-level config key (HC-3 / HC-4 across multiple
  tasks).
- ❌ Add a new third-party dependency (HC-4).
- ❌ Add a `--no-verify` flag to any git command.
- ❌ Modify pre-commit hook configuration.
- ❌ Rebase, force-push, or amend a published commit.
- ❌ Touch the GitHub Actions / CI configuration (none currently
  exists; do not add one without authorization).

---

## 2.7 REQUIRED PRACTICES (positive constraints)

- ✅ Read every file in the task's read-scope **in full** before
  writing any patch.
- ✅ Show a one-line diff preview per planned edit in a plan, before
  applying.
- ✅ Apply edits one logical concept at a time; do not bundle
  unrelated changes.
- ✅ Run `python -m compileall -q .` after every code change.
- ✅ Run `python debug.py --basic` after any change that touches
  inference, simulation, or evaluation logic. Baseline: PASS=9.
- ✅ Verify P-1, P-2, GPU-1 invariants after every patch (file 29
  Tier 0).
- ✅ Document significant work in a new numbered `NN_*.md` doc
  (preserve append-only history).
- ✅ When unsure, ask. The `🛑 STOP` template in file 30 is the
  required form.

---

**End of constraints. Read 28 (protocols) next.**
