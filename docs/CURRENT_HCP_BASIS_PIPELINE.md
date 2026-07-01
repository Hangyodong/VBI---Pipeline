# Current HCP basis-regionwise Pipeline

> ⚠️ **SUPERSEDED by `docs/current_pipeline.md` (2026-06-23).** 이 문서는 stale
> 줄번호(banner 158-193, dispatch 466-488)와 wrong default(GROUP_AVG_FC=1,
> SC_DATASET=hcp_v73, "no PCA / x=64620")을 포함. 코드 기준 최신본은
> `docs/current_pipeline.md` 참조.

**Generated:** 2026-06-18
**Branch:** `refactor/02-simulation`
**Entry point:** `main_HCP.py`
**Status:** smoke-verified on CPU; full GPU run not yet executed.

Evidence legend: ✅ confirmed by code (file:line) · 🧭 expected from context · ⏳ not yet
verified · ⚠️ remaining risk.

---

## 1. Active experiment summary

| Field | Value | Evidence |
|---|---|---|
| Entry point | `main_HCP.py` | ✅ |
| Inference model | `rwweib2` (RWW-EIB 2-coupling) | ✅ `main_HCP.py:101` |
| Parameter mode | `basis_regionwise` (env `PARAMETER_MODE`) | ✅ `main_HCP.py:119,466` |
| Simulation ROI | 360 cortical-only (Glasser; drop 21 subcortical) | ✅ `main_HCP.py:50` |
| On-disk basis | `(381, 3)` = [const, myelin, gradient] | ✅ `basis_decoder.py:5`, test asserts `(381,3)` |
| Active basis | `(360, 3)` after cortical slice `[:360]` | ✅ `basis_decoder.py:55-56` |
| theta_dim | 12 = n_params(4) × basis_dim(3) | ✅ `basis_decoder.py:39` |
| Regional params | `g_LRE, g_FFI, I_o, sigma` | ✅ `main_HCP.py:122` |
| Active bounds | g_LRE(0,3) g_FFI(0,3) I_o(0,1) sigma(0,0.05) | ✅ `BASIS_BOUNDS` `main_HCP.py:135`, overrides `HETERO_BOUNDS` at `main_HCP.py:470` |
| Target FC | group-average FC (env `GROUP_AVG_FC=1`) | ✅ `main_HCP.py:113`, `data_loader_hcp.py:208-281` |
| Delays | OFF (env `USE_DELAYS=0`) | ✅ `main_HCP.py:149` |

> **Cortical-only caveat.** This is a 360-ROI cortical fit, NOT full 381 ROI
> inference. The 381 basis includes 21 subcortical regions whose myelin/gradient
> values are unreliable; the slice keeps the first 360 (Glasser cortical) rows.
> ⚠️ If 381 is used later, subcortex needs a separate offset / one-hot basis,
> not cortical myelin/gradient.

---

## 2. Repository map (modules relevant to this pipeline)

| File | Responsibility |
|---|---|
| `main_HCP.py` | HCP entry: config, startup banner + fail-fast guard, data split, mode dispatch, training, eval |
| `config.py` / `pipeline_setup.py` | `PipelineConfig`, `setup_pipeline`, global `config` module |
| `data_loader_hcp.py` | HCP loader: FC/SC `.mat` read, 381→360 cortical slice, SC max-norm, group-avg FC, delays |
| `data_loader.py` | shared `_scale_weights`, `three_way_split` |
| `basis_decoder.py` | `BasisParamDecoder`, `get_decoder` — theta(coeffs) → per-region maps |
| `param_decoder.py` | `decode_to_param_maps` dispatch (basis/latent/direct), `make_fixed_overrides_from_param_maps` |
| `engine_select.py` | `INFERENCE_MODEL` → sim module; `is_regionwise()`, `latent_wrap`, `get_simulate_gpu_batch` |
| `cuBNM/simulate_rwweib_2cpl.py` | rwweib2 `simulate_gpu_batch` adapter (trims to `ANALYSIS_BOLD_T`) |
| `cuBNM/runner_rwweib_2cpl.py` | `build_param_lists`, `run_cubnm_rwweib2_batch`, `RWWEIB_2CPLSimGroup` driver |
| `cuBNM/rww_eib_2cpl.yaml` | 2-coupling model def (`conn_state_vars: [S_E, S_I]`) |
| `inference/priors.py` `scaling.py` `training_data.py` `feature_pipeline.py` `snpe.py` `posterior.py` `stage1.py` `embedding.py` | SBI/SNPE-C flow |
| `features/fc.py` | `compute_fc`, `fc_to_upper_tri` |
| `evaluation/metrics.py` `plots.py` | resim/baseline/plots — now region-wise safe |
| `test_basis_mode_smoke.py` `test_basis_decoder.py` | CPU unit/smoke tests (repo root) |
| `tests/smoke/verify_basis_regionwise_rwweib2.py` | non-training basis+rwweib2 smoke (CPU fallback) |
| `tests/smoke/smoke_e2e_basis_regionwise.py` | tiny e2e — **includes tiny SNPE-C training** |

---

## 3. End-to-end pipeline

```
main_HCP.py
 ├─ config (PipelineConfig → setup_pipeline → config module)
 │    N_REGIONS=360, INFERENCE_MODEL="rwweib2", PARAMETER_MODE (env)
 ├─ [C1] startup banner + fail-fast guard          main_HCP.py:158-193
 │    REQUIRE_BASIS=1 (default) → raise unless PARAMETER_MODE=basis_regionwise
 ├─ Step 1: data_loader.load_raw_data() + split     main_HCP.py:402-415
 │    HCP_FC.mat (var C), HCP_SC.mat (h5py) → cortical slice [:360]
 │    SC max-norm + symmetrize; GROUP_AVG_FC → shared group FC target
 ├─ mode dispatch (basis_regionwise branch)         main_HCP.py:466-488
 │    HETERO_BOUNDS = BASIS_BOUNDS; STAGE1_PARAMS = coeff_names() (12)
 │    prior = Uniform(BASIS_COEFF_PRIOR) on 12 coeffs
 ├─ train: collect_training_data(engine=rwweib2)    inference/training_data.py
 │    sample theta_scaled ~ prior → inverse_transform → theta_raw (coeffs)
 │    simulate_gpu_batch = latent_wrap(base)        engine_select.py:120-126
 │      └─ decode_to_param_maps → {param}_matrix overrides → cuBNM 2cpl
 │    BOLD → FeaturePipeline → FC upper-tri (x)
 ├─ SNPE-C train (theta_scaled, x)                  inference/snpe.py
 ├─ posterior sample → inverse_transform → coeffs   inference/posterior.py
 └─ eval: resim / baseline / ppc / plots            evaluation/* (region-wise safe)
      decode coeffs → maps → resim → FC corr vs empirical
```

---

## 4. Basis decoder (`basis_decoder.py`)

**Load** (`from_file`, `basis_decoder.py:48-64`):
- `np.load(BASIS_PATH)` → `(381, 3)` float64.
- If `n_regions(360) < rows(381)` → slice `b[:360]` (cortex). ✅ `:55-56`
- `rezscore=True` re-standardizes each NON-constant column on the slice to
  mean 0 / std 1; constant column (col 0) kept as-is. ✅ `:57-63`

**Decode** (`decode`, `basis_decoder.py:67-89`):
```python
beta = theta.reshape(S, n_params, basis_dim)   # (S, 4, 3)
z    = beta[:, k, :] @ basis.T                 # (S, R)   per param k
mid  = 0.5*(lo+hi);  half = 0.5*(hi-lo)
map  = mid + half * tanh(z)                    # exactly within [lo, hi], no clip
```
- Asserts finite + within bounds; `sigma >= 0`. ✅ `:85-88`
- Single theta `(12,)` → `{p:(360,)}`; batch `(S,12)` → `{p:(S,360)}`. ✅

**Coefficient order** (`coeff_names`, `:92-96`) — 12, basis_dim==3 ⇒ cols
`const, myelin, gradient`:
```
g_LRE_const, g_LRE_myelin, g_LRE_gradient,
g_FFI_const, g_FFI_myelin, g_FFI_gradient,
I_o_const,   I_o_myelin,   I_o_gradient,
sigma_const, sigma_myelin, sigma_gradient
```

**theta = 0 → midpoint maps** (tanh(0)=0 ⇒ map=mid): ✅ verified
`g_LRE=1.5, g_FFI=1.5, I_o=0.5, sigma=0.025` (uniform across regions).

**Bounds caveat.** Two bound sets exist:
- `HETERO_BOUNDS` (homogeneous/direct default) = g_LRE(0,9) g_FFI(0,9)
  I_o(0.15,0.60) sigma(0,0.09). `main_HCP.py:123-124`
- `BASIS_BOUNDS` (basis mode) = g_LRE(0,3) g_FFI(0,3) I_o(0,1) sigma(0,0.05).
  `main_HCP.py:135-136` — **basis mode overrides** HETERO_BOUNDS with these
  at `main_HCP.py:470`. Active basis fit uses the (0,3)/(0,3)/(0,1)/(0,0.05) set.

---

## 5. Simulator integration (rwweib2 → cuBNM 2-coupling)

**Engine resolution** (`engine_select.py:31`): `INFERENCE_MODEL="rwweib2"` →
`cuBNM.simulate_rwweib_2cpl`. Imports deferred to call time (no GPU to import).

**Region-wise wrap** (`engine_select.py:90-126`): in `basis_regionwise`,
`get_simulate_gpu_batch()` returns `latent_wrap(base)`:
1. `decode_to_param_maps(theta, ...)` → `{param:(S,R)}` (dispatches to
   `basis_decoder.get_decoder(config).decode`).
2. `make_fixed_overrides_from_param_maps` → `{g_LRE_matrix, g_FFI_matrix,
   I_o_matrix, sigma_matrix}`, each `(S,R)`.
3. base sim called with **empty** `param_names` + these `fixed_overrides`;
   theta is NOT forwarded as scalar params (so coeff names can't be silently
   ignored).

**Param injection** (`runner_rwweib_2cpl.build_param_lists`):
- `<param>_matrix` in `fixed` with shape `(n_sims, n_nodes)` → assigned
  directly to `param_lists[name]`; wrong shape → `ValueError`. ✅
- scalar per-sim params (from `param_names`) → broadcast `repeat(col, n_nodes)`.
- All RWWEIB2 params are regional-capable (`g_LRE, g_FFI, sigma, I_o, w_E, w_I,
  w_p, J_N, J_i, lambda_IE`).

**Model** (`rww_eib_2cpl.yaml`): two independent couplings —
`globalinput_E = SC @ S_E` (gain `g_LRE·J_N`), `globalinput_I = SC @ S_I`
(gain `g_FFI·J_N·lambda_IE`); `conn_state_vars: [S_E, S_I]`; BOLD driven by
`S_E`. Fixed in this run: `RWWEIB2_FIXED = {w_E:1.0, w_I:0.7, J_i:1.0, w_p:1.4,
J_N:0.15, lambda_IE:1.0}` (`main_HCP.py:108-109`). No FIC, no delays.

**Driver** (`run_cubnm_rwweib2_batch`): `from cubnm.sim import
RWWEIB_2CPLSimGroup`; instantiate with `sc=weights`, `force_gpu`, `hrf="bw"`
(Balloon-Windkessel); inject `param_lists`; BOLD `(n_sims, T, n_nodes)` → list
of `(T, N)`. Adapter trims leading transient to `config.ANALYSIS_BOLD_T`.

> ⚠️ **GPU path.** `simulate_rwweib_2cpl.simulate_gpu_batch` passes
> `force_gpu=True` (`:68`). The full `main_HCP.py` training path therefore needs
> a real GPU. CPU smoke runs only via the runner's `force_gpu=False` (verify
> script auto-detects; e2e smoke monkeypatches the runner).

---

## 6. SBI / VBI flow

| Stage | Module | Detail |
|---|---|---|
| Prior | `inference/priors.py` | scaled prior `BoxUniform(-1,1)^12`; raw coeff bounds `Uniform(-2,2)` (`BASIS_COEFF_PRIOR`) |
| Scaling | `inference/scaling.py` | `ParameterScaler` raw[-2,2] ↔ scaled[-1,1], per-coefficient |
| Train data | `inference/training_data.py` | `collect_training_data(engine="rwweib2")`; region-wise → `latent_wrap`. ⚠️ hard-calls `cupy.free_all_blocks` → GPU-only |
| Features | `inference/feature_pipeline.py` | raw FC upper-tri passthrough (no PCA); FCD off |
| SNPE-C | `inference/snpe.py` | `from sbi.inference import SNPE_C`; MAF density estimator; `append_simulations` + `train` |
| Posterior | `inference/posterior.py` | sample scaled → `inverse_transform` → coeffs; `is_regionwise()` routes resim through wrapped batch decode |

---

## 7. Data flow & object shapes

| Object | Shape | Source |
|---|---|---|
| on-disk FC (`HCP_FC.mat` var `C`) | per-subj `(381,381)` | `data_loader_hcp.py:5-7` |
| on-disk SC (`HCP_SC.mat`) | weights + lengths `(381,381)` | `data_loader_hcp.py:9-12` |
| cortical slice | `[:360, :360]` via `_cortical_slice` | `data_loader_hcp.py:117-135` |
| subject dict | keys `fc, fcd, fc_nan, sc, lengths_mm, delays` | `data_loader_hcp.py:161-164` |
| SC (post) | `(360,360)`, symmetrized + max-norm (`VBI_SC_SCALE=maxnorm`) | `main_HCP.py:35`, `data_loader_hcp.py:148-149` |
| FC target | `(360,360)`, group-avg when `GROUP_AVG_FC=1` | `data_loader_hcp.py:208-281` |
| basis (active) | `(360, 3)` | `basis_decoder.py` |
| theta (coeffs) | `(S, 12)` | decoder |
| decoded maps | `{param:(S, 360)}` | decoder |
| overrides | `{param}_matrix (S, 360)` | `make_fixed_overrides` |
| BOLD (per sim) | `(ANALYSIS_BOLD_T, 360)` | runner/adapter |
| FC feature (x) | upper-tri `FC_DIM = 360·359/2 = 64,620` | `features/fc.py` |

---

## 8. Environment variables

| Var | Default | Effect | Evidence |
|---|---|---|---|
| `PARAMETER_MODE` | `homogeneous` | must be `basis_regionwise` for this experiment | `main_HCP.py:119` |
| `REQUIRE_BASIS` | `1` | `1` → hard-fail unless basis mode; `0` → opt out | `main_HCP.py:186-193` |
| `INFERENCE_MODEL` | set to `rwweib2` in code | sim engine | `main_HCP.py:101` |
| `BASIS_PATH` | `basis.npy` (repo-local) | basis file | `main_HCP.py:133` |
| `GROUP_AVG_FC` | `1` | group-avg vs per-subject FC | `main_HCP.py:113` |
| `USE_DELAYS` | `0` | tract-length delays (~9× cost) | `main_HCP.py:149` |
| `SC_DATASET` | `hcp_v73` | `hcp_v73` \| `cabnp381` | `main_HCP.py:140` |
| `SC_FILE` | `HCP_SC.mat` | SC source | `main_HCP.py:47` |
| `VBI_SC_SCALE` | `maxnorm` | SC scaling | `main_HCP.py:35` |
| `N_SUBJECTS/N_TRAIN/N_VAL/N_TEST` | 100/70/10/20 | split | `main_HCP.py:53-56` |
| `N_SIM` / `GPU_BATCH` | 2000 / 2000 | sims per round / batch | `main_HCP.py:60-61` |

Sim time: `T_END=180000ms` (3min), `T_CUT=60000ms` (1min), `DT=1.0`,
`DECIMATE=720`, `TR_SEC=0.72`. SBI: `N_POSTERIOR=2000`, `N_SBC=200`,
`N_TEST_RESIM=10`, `NDE_HIDDEN=128`, `NDE_TRANSFORMS=8`. (`main_HCP.py:60-85`)

---

## 9. Smoke tests

See `docs/SMOKE_TESTS.md` for full commands/expected output. Summary:

| Test | Scope | GPU? | Training? |
|---|---|---|---|
| `test_basis_mode_smoke.py` | 7 CPU unit checks (decode, bounds, override, wrap) | no | no |
| `test_basis_decoder.py` | decoder unit (CPU) | no | no |
| `tests/smoke/verify_basis_regionwise_rwweib2.py` | import + banner + decode + tiny sim | CPU fallback | no |
| `tests/smoke/smoke_e2e_basis_regionwise.py` | tiny e2e | CPU | ⚠️ tiny SNPE-C train (N_SIM=16) |

All non-training checks: **PASS** (2026-06-18).

---

## 10. Known risks

1. ⚠️ **GPU `force_gpu=True` unverified.** This shell has no CUDA
   (`torch.cuda.is_available()=False`). Full `main_HCP.py` training needs GPU
   (`simulate_rwweib_2cpl.py:68`, `training_data.py` cupy).
2. ⚠️ **`/mnt/d/hcp_basis/basis.npy` absent** → repo-local `basis.npy` used.
   Both are `(381,3)`; equality of the two files not verified.
3. ⏳ **Full `main_HCP.py` real run not executed.** Guard verified; data load
   + SNPE-C train + eval not run end-to-end.
4. ⚠️ **Cortical-only 360 ROI** — not full 381; subcortex basis unreliable.
5. ⚠️ **Uncommitted changes** — 5 modified files + untracked `tests/`,
   `test_basis_mode_smoke.py`.

---

## 11. Next steps

1. Run GPU smoke on a GPU node: `tests/smoke/verify_basis_regionwise_rwweib2.py`
   (auto-uses GPU when available → exercises `force_gpu=True`).
2. (Optional, with approval) tiny e2e: `tests/smoke/smoke_e2e_basis_regionwise.py`.
3. Commit docs + smoke tests.
4. Launch real VBI run on GPU:
   `PARAMETER_MODE=basis_regionwise REQUIRE_BASIS=1 python main_HCP.py`.
