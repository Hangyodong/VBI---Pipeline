# CLAUDE.md — VBI/SBI whole-brain digital-twin pipeline

Project memory. Whole-brain VBI-SBI pipeline inferring region-wise RWW-EIB
parameters (as myelin/gradient basis coefficients) from HCP FC via SNPE-C.

Full detail (env matrix, dataflow, SBI flow, eval, results): `docs/current_pipeline.md`.
**Code/config = source of truth.** Line cites verified 2026-06-23 (branch
`refactor/02-simulation`). config.py ships *mouse* defaults — all active HCP
values come from `main_HCP.py` overrides, NOT config.py.

---

## Active config — what the code actually does

Set in `main_HCP.py` after `setup_pipeline` (overrides config.py):
- Entry `main_HCP.py`. Engine `INFERENCE_MODEL="rwweib2"` (:112) →
  cuBNM `RWWEIB_2CPLSimGroup` (`rww_eib_2cpl.yaml`, `conn_state_vars:[S_E,S_I]`).
- `PARAMETER_MODE` default `"basis_regionwise"` (:130). `REQUIRE_BASIS=1`
  fail-fast guard (:235-242) — wrong mode raises unless `REQUIRE_BASIS=0`.
- `N_REGIONS=360` (:60), cortical slice 381→360 (drop 21 subcortical).
- `SC_DATASET` default `"cabnp381"` (:162); `SC_FILE` default
  `HCP_CABNP381_SC_first100.mat` (:57). FC from `HCP_FC.mat` (var `C`).

### ⚠️ Gotcha defaults (docs were wrong on these)
- `SMOKE` default `"1"` (:47) → **bare `python main_HCP.py` is a TINY toy**
  (N_SUBJECTS=4 / N_TRAIN=2 / N_VAL=1 / N_TEST=1 / N_SIM=64 / GPU_BATCH=64).
  Real run = `SMOKE=0` → 100 / 70 / 10 / 20 / 2000 / 2000.
- `GROUP_AVG_FC` default `0` = **PER-SUBJECT FC** (:124). (Not group-avg.)
- `USE_DELAYS` default `0` = OFF (:179). Delays computed but not fed; sanity
  showed ~0 BOLD-FC change for 5.3× cost (6-100ms ≪ 720ms TR). `=1` to enable.
- `SC_CONDITION` `0` (:186), `GEOMETRY_COUPLING` `0` (:195) — both OFF (baseline).
- SC scale: **effective `maxnorm`** via `main_HCP.py:35`
  `os.environ.setdefault("VBI_SC_SCALE","maxnorm")` (loader bare default is
  `log1p` at `data_loader.py:229`, but main_HCP forces maxnorm first).

## Basis / theta (load-bearing — Do NOT change)
- `basis.npy` on disk `(381,3)=[const,myelin,gradient]`; slice `[:360]`;
  `BASIS_REZSCORE=True` z-scores non-const cols on the slice. `theta_dim=12`
  (4 params × 3 basis).
- Decode (`basis_decoder.py:67-89`): `z = beta @ basis.T`;
  `map = mid + half*tanh(z)`, `mid=(lo+hi)/2`, `half=(hi-lo)/2`. No clip.
- Coeff order: `g_LRE/g_FFI/I_o/sigma`, each `{const,myelin,gradient}`.
- `theta=0` → midpoints `g_LRE=1.5, g_FFI=1.5, I_o=0.5, sigma=0.025`.
- Prior: scaled `BoxUniform[-1,1]^12`; raw coeff `(-2,2)` (`BASIS_COEFF_PRIOR`,
  :154). `ParameterScaler` linear box map (`scaling.py`).

### Bounds — two sets, don't confuse
- **basis mode (active)** `BASIS_BOUNDS` (:151-153): g_LRE(0,`G_BOUND_HIGH`=3.0)
  g_FFI(0,3) I_o(0,1) sigma(0,0.05). Overrides HETERO_BOUNDS at `main_HCP.py:546`
  (also early at :158-159 so the decoder caches narrow bounds).
- homogeneous/direct `HETERO_BOUNDS` (:134): g_LRE(0,9) g_FFI(0,9) I_o(0.15,0.60)
  sigma(0,0.09). NOT used in the basis fit (overwritten).

## Dataflow (one line)
theta_scaled(N_SIM,12) → inverse_transform → raw(-2,2) → `latent_wrap` decode →
`{g_LRE,g_FFI,I_o,sigma}_matrix (S,360)` → `build_param_lists` →
`RWWEIB_2CPLSimGroup(force_gpu, hrf="bw")` → BOLD(T,360) → `compute_fc`
(**raw Pearson r** — print label "Fisher-z" is WRONG, no arctanh anywhere) →
upper-tri `FC_DIM=64,620` → **FeaturePipeline PCA-256 whitened**
(`FC_PCA_DIM=256` :174, `FC_PCA_WHITEN=True` :178; NOT raw passthrough —
docstrings stale) → SNPE-C (`nn.Identity` embedding, `use_embedding=False`;
MAF h128 ×8; batch512; max_epochs200 no early stop; single-round
`RUN_PHASE24=False` :163 → Phase3/4 skipped, s2/posterior_2 alias Phase1).
N_SIM is **per-subject** → real train tensor = 70×2000 = 140k sims.

## Critical files (line cites verified 2026-06-23)
| File | Role |
|---|---|
| `main_HCP.py` | entry; config @101-198; C1 banner @207-234 + guard @235-242; basis dispatch @542-564 (bounds override @546) |
| `basis_decoder.py` | `BasisParamDecoder`, `get_decoder`; decode `:67-89` |
| `param_decoder.py` | `decode_to_param_maps` dispatch + override builder |
| `engine_select.py` | `is_regionwise`, `latent_wrap`, `get_simulate_gpu_batch`; rwweib2→`cuBNM.simulate_rwweib_2cpl` |
| `cuBNM/runner_rwweib_2cpl.py` | `build_param_lists`, `RWWEIB_2CPLSimGroup` |
| `cuBNM/rww_eib_2cpl.yaml` | 2 couplings `conn_state_vars:[S_E,S_I]` |
| `data_loader_hcp.py` | FC/SC load, 381→360 slice, SC scale (maxnorm), group-avg FC |
| `inference/feature_pipeline.py` | FC PCA-256 whiten (`:81-110`) |
| `inference/snpe.py` | SNPE_C; `use_embedding=False`/`nn.Identity` (`:43,107,138`); epochs (`:166-167`) |
| `inference/{priors,scaling,training_data,posterior}.py` | SNPE-C flow |
| `evaluation/{metrics,plots,final_test}.py` | resim/baseline/plots (engine-routed) |
| `basis.npy` | repo-local basis (preferred `/mnt/d/hcp_basis/basis.npy`, may be absent) |

## Launch
```bash
# REAL run (GPU node):
SMOKE=0 PARAMETER_MODE=basis_regionwise python main_HCP.py
# intended target experiment adds (optional):
#   SC_DATASET=cabnp381 SC_FILE=HCP_CABNP381_SC_first100.mat USE_DELAYS=1 GROUP_AVG_FC=1
```

## Safe commands (no training, CPU OK)
```bash
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  python -m pytest test_basis_mode_smoke.py -q          # 7 passed
python test_basis_decoder.py                            # ALL PASS (verify path first)
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  BASIS_PATH=/mnt/d/hcp_basis/basis.npy \
  python tests/smoke/verify_basis_regionwise_rwweib2.py # RUN OK, finite=True
```

## Do NOT
- Change decoder eqs (`mid+half*tanh(basis@beta)`), bounds, coeff order, model
  eqs / `rww_eib_2cpl.yaml` couplings, prior, scaler, training behavior.
- Run `tests/smoke/smoke_e2e_basis_regionwise.py` unattended — it does a tiny
  SNPE-C **train** (approval first).
- Run bare `main_HCP.py` thinking it's a smoke — `SMOKE=0` is the real GPU run;
  `SMOKE=1` is a toy that proves nothing about scale.
- Trust inline "raw passthrough / no PCA" or "Fisher-z" comments — PCA-256 is
  active and FC is raw Pearson r. (Code-doc drift, flagged in current_pipeline.md.)

## Status / latest
- Engine-routing eval bug FIXED (eval/SBC/resim/predictive paths had hardcoded
  WC import → FC corr pinned ±0.0000 fake; now engine-routed via `engine_select`).
- First GPU run (2026-06-19, per-subject FC, delays OFF, N_TRAIN=20): Test FC
  corr **0.1342** [0.128, 0.140], RMSE 0.2151, Val 0.1381, resim 0.0895±0.011.
  Sim cost 14.6 sim/s (delays OFF). Full-scale (70×5000 ≈ 6.7 h) run pending.
- GROUP_AVG_FC=1 expected ceiling ~0.2 (not yet measured).
- Claude scratch shell has **no CUDA** — real training only on the GPU node
  (`force_gpu=True`; `training_data.py` uses cupy).
- Uncommitted: modified code + many untracked docs/tests (see `git status`).
```
