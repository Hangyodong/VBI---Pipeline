# Current HCP basis-regionwise Pipeline (detail)

Reconstructed from **code (source of truth)**, line cites verified 2026-06-23,
branch `refactor/02-simulation`, entry `main_HCP.py`. Supersedes
`CURRENT_HCP_BASIS_PIPELINE.md` (stale line cites + several wrong defaults).
Pointer summary lives in `CLAUDE.md`.

> Note: `config.py` ships **mouse** defaults (N_REGIONS=115, INFERENCE_MODEL="wc",
> USE_DELAYS=True, VELOCITY=1.5, WC STAGE1_PARAMS). All of these are overwritten at
> runtime by `setup_pipeline` then `main_HCP.py:101-198`. Read `main_HCP.py`, not
> `config.py`, for the effective config.

---

## 1. Active experiment
| Field | Value | Evidence (main_HCP.py unless noted) |
|---|---|---|
| Entry | `main_HCP.py` | — |
| Inference model | `rwweib2` (RWW-EIB 2-coupling) | :112 |
| Parameter mode | `basis_regionwise` (default) | :130, dispatch :542 |
| ROI | 360 cortical (slice 381→360) | :60 |
| theta_dim | 12 = 4 params × 3 basis | basis_decoder |
| Active bounds | g_LRE(0,3) g_FFI(0,3) I_o(0,1) sigma(0,0.05) | BASIS_BOUNDS :151-153, override :546 |
| SC dataset | `cabnp381` / `HCP_CABNP381_SC_first100.mat` | :162, :57 |
| FC target | per-subject (GROUP_AVG_FC=0) | :124 |
| Delays | OFF (USE_DELAYS=0) | :179 |
| SC scale | maxnorm (effective; setdefault) | :35 / data_loader.py:218-247 |
| SMOKE | default 1 (toy); 0 = real | :47 |

> Cortical-only caveat: 360-ROI fit, not 381. Subcortex myelin/gradient
> unreliable → 381 later needs a separate offset / one-hot basis.

## 2. Repository map
| File | Responsibility |
|---|---|
| `main_HCP.py` | entry; PipelineConfig (:52-96), config overrides (:101-198), C1 banner+guard (:207-242), per-mode dispatch (homogeneous / latent_regionwise / direct_regionwise :519 / basis_regionwise :542) |
| `pipeline_setup.py` | `PipelineConfig`, `setup_pipeline` (reload/seed/print-patch, `_apply_to_config`) |
| `config.py` | base globals — **mouse defaults, mostly inert/overridden** |
| `data_loader_hcp.py` | FC/SC `.mat` load, 381→360 cortical slice, SC scale, group-avg FC, delay matrix |
| `data_loader.py` | shared `_scale_weights` (VBI_SC_SCALE), `three_way_split` |
| `basis_decoder.py` | `BasisParamDecoder`, `get_decoder(config)`, `coeff_names()` |
| `param_decoder.py` | `decode_to_param_maps` mode dispatch, `make_fixed_overrides_from_param_maps`, direct_* helpers |
| `engine_select.py` | `active_engine`, `is_regionwise`, `latent_wrap`, `get_simulate_gpu_batch` — single source for engine routing |
| `cuBNM/simulate_rwweib_2cpl.py` | adapter → runner; trims transient to ANALYSIS_BOLD_T; gates delays on USE_DELAYS |
| `cuBNM/runner_rwweib_2cpl.py` | `build_param_lists`, `RWWEIB_2CPLSimGroup` |
| `cuBNM/rww_eib_2cpl.yaml` | 2-coupling RWW-EIB codegen (`conn_state_vars:[S_E,S_I]`) |
| `inference/priors.py` | scaled BoxUniform prior |
| `inference/scaling.py` | `ParameterScaler` raw↔[-1,1] |
| `inference/training_data.py` | `collect_training_data` (sample→sim→extract) |
| `inference/feature_pipeline.py` | FC PCA-256 whiten |
| `inference/snpe.py` | `train_snpe` (SNPE_C, MAF, Identity embedding) |
| `inference/posterior.py` | sample, `build_x_obs`, `posterior_predictive_check` |
| `inference/embedding.py`, `inference/sc_channels.py` | SC-conditioning encoder (OFF default) |
| `features/fc.py`, `features/extraction.py` | `compute_fc` (Pearson), `fc_to_upper_tri`, `worker_extract` |
| `evaluation/metrics.py` | `fc_metrics`, `evaluate_subject`, `_resimulate_and_score`, `baseline_eval`, `bootstrap_ci` |
| `evaluation/final_test.py` | `final_test` → `_test_stage1` |
| `evaluation/{validation,plots,reports,sc_diagnostics}.py` | val aggregation, plots, summary, SC diagnostics |
| `simulation/{delays,geometry}.py` | `compute_delay_matrix`, `augment_sc_geometry` |

## 3. End-to-end flow
```
config (cfg :52-96) → setup_pipeline → overrides (:101-198)
  → C1 banner (:207-234) + REQUIRE_BASIS guard (:235-242)
Step1  load_raw_data + three_way_split (seed42, subject-disjoint 70/10/20)
       load_all_subjects: 381→360 slice, SC symmetrize+scale(maxnorm),
       FC nan→0+symmetrize+zero-diag, delays computed, group-avg iff flag
Step1.5 (SC_CONDITION only, OFF) build SC table + fit scaler on train rows
basis dispatch (:542-564): HETERO_BOUNDS=BASIS_BOUNDS; STAGE1_PARAMS=coeff_names(12);
       prior Uniform(-2,2)^12
Step7  fit_param_scaler → ParameterScaler + scaled BoxUniform[-1,1]^12
Step2/3 collect_training_data(engine=rwweib2, latent_wrap):
       per subject: sample theta_scaled(N_SIM,12) → inverse → raw →
       simulate_gpu_batch(sc, chunk, delays, apply_bw) → BOLD →
       ProcessPool worker_extract → (fc_vec, fcd_vec); save features_stage1.npz
Step4  quality filter → FeaturePipeline.fit_transform → x_input (n,256) PCA-white
Step8  train_snpe(theta_scaled, x_input, prior, use_embedding=False) → MAF SNPE_C
Phase3/4 SKIPPED (RUN_PHASE24=False) → posterior_2 aliases Phase1
Step9  validation (evaluate_validation_stage1 + baseline_eval) [model selection only]
Step13 select_best_model
Step14 final_test(test, s1) → FC corr (expected + per-sample) + bootstrap CI
Step14b save_param_maps → param_maps.npz
Save   artifacts.pkl, embedding_net_s1.pt, final summary
```

## 4. Basis decoder
- Load (`basis_decoder.py:48-64`): `np.load` → `(381,3)`; slice `[:360]`; if
  `BASIS_REZSCORE` z-score non-const cols (col 0 = const left as-is).
- Decode (`:67-89`): `beta = theta.reshape(S,4,3)`; per param k:
  `z = beta[:,k,:] @ basis.T` → `map = mid + half*tanh(z)`. Asserts finite +
  in-bounds; sigma ≥ 0. tanh keeps maps within bounds → no clip.
- Coeff order (12): `g_LRE_{const,myelin,gradient}, g_FFI_{...}, I_o_{...},
  sigma_{...}`.
- `theta=0` → `tanh(0)=0` → every param at its bound midpoint
  (g_LRE=1.5, g_FFI=1.5, I_o=0.5, sigma=0.025).

### Bounds (two sets)
| set | g_LRE | g_FFI | I_o | sigma | used? |
|---|---|---|---|---|---|
| BASIS_BOUNDS (active) | (0, 3.0) | (0, 3.0) | (0, 1.0) | (0, 0.05) | ✅ basis fit |
| HETERO_BOUNDS | (0, 9.0) | (0, 9.0) | (0.15, 0.60) | (0, 0.09) | ❌ overwritten |

`G_BOUND_HIGH` env (default 3.0) sets the g upper bound; `=6` re-opens for ablation.

## 5. Simulator integration (rwweib2 → cuBNM 2-coupling)
- Engine resolution (`engine_select`): rwweib2 → `cuBNM.simulate_rwweib_2cpl`,
  imports deferred (no GPU at import time).
- Region-wise wrap (`latent_wrap`): `decode_to_param_maps` → `{param}_matrix`
  overrides → base sim called with **empty param_names + fixed_overrides** (theta
  NOT forwarded as scalars in region-wise mode).
- `build_param_lists`: `<param>_matrix (n_sims, n_nodes)` → `param_lists[name]`;
  wrong shape → ValueError.
- Model (`rww_eib_2cpl.yaml`): E driven by `SC@S_E` (gain g_LRE·J_N), I driven by
  `SC@S_I` (gain g_FFI·J_N·lambda_IE); two independent couplings; BOLD from S_E.
  `RWWEIB2_FIXED = {w_E:1.0, w_I:0.7, J_i:1.0, w_p:1.4, J_N:0.15, lambda_IE:1.0}`
  (:119-120). No FIC.
- Driver: `RWWEIB_2CPLSimGroup(sc=weights, force_gpu, hrf="bw")`; adapter trims
  transient to `ANALYSIS_BOLD_T`. ⚠️ `force_gpu=True` → real GPU required.

## 6. SBI / SNPE-C flow
| Stage | Module | Detail |
|---|---|---|
| Prior | priors.py | scaled BoxUniform[-1,1]^12; raw Uniform(-2,2) |
| Scaling | scaling.py | ParameterScaler raw[-2,2]↔scaled[-1,1] per-coeff, data-free |
| Train data | training_data.py | collect_training_data(rwweib2); region-wise = latent_wrap; ⚠️ cupy → GPU-only |
| Features | feature_pipeline.py :81-110 | **FC PCA-256, whiten=True** (NOT raw passthrough; docstring :1-19 stale) |
| SNPE-C | snpe.py :138 | SNPE_C; embedding `nn.Identity` (use_embedding=False :43,107); MAF h128 ×8; batch512; max_epochs200 no early stop :166-167 |
| Phases | RUN_PHASE24=False :163 | single-round; Phase3/4 skipped; s2/posterior_2 alias Phase1 |
| Posterior | posterior.py | sample scaled → inverse_transform → coeffs; is_regionwise() routes resim through wrapped decode |

## 7. Data flow & shapes
| Object | Shape | Source |
|---|---|---|
| on-disk FC (var C) | per-subj (381,381) | data_loader_hcp:37 |
| on-disk SC (cabnp381) | weight_all / tract_length_all (381,381,100) | data_loader_hcp:173 |
| cortical slice | [:360,:360] | data_loader_hcp:118 |
| SC (post) | (360,360) symmetrized, scaled (maxnorm) | data_loader.py:218 |
| FC target | (360,360); group-avg iff GROUP_AVG_FC=1 | data_loader_hcp:211/272 |
| basis active | (360,3) | basis_decoder |
| theta (coeffs) | (S,12) | decoder |
| decoded maps | {param:(S,360)} | decoder |
| overrides | {param}_matrix (S,360) | make_fixed_overrides |
| BOLD | (ANALYSIS_BOLD_T, 360) | runner/adapter |
| FC feature pre-PCA | upper-tri 64,620 (=360·359/2) | features/fc |
| x_input post-PCA | (n, 256) whitened | feature_pipeline |
| train tensor (real) | (70·2000, *) = 140k sims | collect_training_data |

## 8. Environment variables (full)
| Var | Default | Effect | Cite |
|---|---|---|---|
| SMOKE | 1 | 1=toy sizes, 0=real | :47 |
| PARAMETER_MODE | basis_regionwise | mode | :130 |
| REQUIRE_BASIS | 1 | fail-fast guard | :235 |
| GROUP_AVG_FC | 0 | per-subj vs group FC | :124 |
| USE_DELAYS | 0 | tract delays (5.3×, ~0 FC gain) | :179 |
| SC_DATASET | cabnp381 | hcp_v73 \| cabnp381 | :162 |
| SC_FILE | HCP_CABNP381_SC_first100.mat | SC source | :57 |
| VBI_SC_SCALE | maxnorm (setdefault; loader bare default log1p) | SC scaling | :35 / data_loader.py:229 |
| BASIS_PATH | basis.npy | basis file | :144 |
| G_BOUND_HIGH | 3.0 | g_LRE/g_FFI upper bound | :150 |
| FC_PCA_DIM | 256 | FC PCA comps | :174 |
| FC_PCA_WHITEN | 1 | PCA whitening | :178 |
| SC_CONDITION | 0 | matrix-encoder NPE | :186 |
| SC_CHANNELS | sc_weight,sc_mask | conditioning channels (delay excluded) | :187 |
| GEOMETRY_COUPLING | 0 | homotopic SC aug | :195 |
| GEOM_ALPHA / GEOM_KERNEL / GEOM_RENORM | 0.3 / homotopic / 1 | geometry knobs | :196-198 |
| EMBED_PER_SUBJECT_SC | False | per-subject SC in embedding | :168 |
| N_LAPLACIAN_BASIS | 4 | (latent mode only) | :137 |
| N_SUBJECTS/N_TRAIN/N_VAL/N_TEST | 100/70/10/20 (4/2/1/1 SMOKE) | split | :63-66 |
| N_SIM / GPU_BATCH | 2000/2000 (64/64 SMOKE) | per-subject sims / batch | :70-71 |

Sim time: T_END=180000ms, T_CUT=60000ms, DT=1.0, DECIMATE=720, TR=0.72.
SBI: N_POSTERIOR=2000, N_SBC=200, NDE_HIDDEN=128, NDE_TRANSFORMS=8, N_TEST_RESIM=10.

## 9. Data preprocessing detail
1. Subject selection: `get_target_subjects` = smallest N_SUBJECTS ids common to
   FC&SC, ascending (deterministic). `three_way_split` shuffles (seed42) →
   subject-disjoint 70/10/20.
2. FC: cortical slice `[:360,:360]`, `nan→0`, symmetrize `(fc+fc.T)/2`, zero diag.
   No FC normalization (raw Pearson r kept).
3. SC weights: slice → symmetrize → `_scale_weights` (zero diag, mask, compress,
   **max-norm `w/=w.max()`**, re-apply mask) → `augment_sc_geometry` (no-op
   unless GEOMETRY_COUPLING). Effective compression = maxnorm (via :35 setdefault).
4. tract lengths: slice → symmetrize → zero diag (mm).
5. delays: `compute_delay_matrix(sc, VELOCITY=3.0, lengths_mm)` — always computed,
   only fed to simulator when USE_DELAYS=1.
6. group-avg: if GROUP_AVG_FC, every subject FC overwritten by mean FC over all
   FC subjects; SC stays per-subject.

## 10. Features & metrics
- **Feature**: `compute_fc` = `np.corrcoef(ts.T)`, nan→0, zero-diag → raw Pearson
  r; `fc_to_upper_tri` k=1 → 64,620 vector. **No Fisher-z** (the
  `training_data.py:448` "Fisher-z" print is a wrong label). FCD off (USE_FCD=False).
- **x for SBI** (FC-only path, active): `FeaturePipeline.fit_transform` → PCA
  (n_comp=min(256, n-1, 64620), whiten) fit on TRAIN sims only → x_input (n,256).
  No theta PCA. theta z-scored to [-1,1] by ParameterScaler.
- **Metrics** (evaluation/metrics.py): `fc_metrics` (:47) upper-tri Pearson corr +
  RMSE + MAE over finite ∧ ~nan_mask edges; `bootstrap_ci` (:89, seed42, n=1000);
  `_resimulate_and_score` (:266) routes through `get_simulate_gpu_batch`;
  `evaluate_subject` (:113) reports `fc_corr_mean`, `fc_corr_expected` (score the
  mean of resim FC), `fc_corr_meantheta`; `baseline_eval` (:337) = prior-midpoint
  (basis all-zero coeffs → bound midpoints).
- **final_test** → `_test_stage1` uses s1 posterior; FC corr + per-sample + CI.

## 11. Eval engine-routing bug-fix (history)
- Bug: eval/SBC/resim/predictive paths hardcoded a WC import while training used
  rwweib2 → FC corr pinned `−0.005 ±0.0000` (fake).
- Fix: all eval resim now goes through `engine_select.get_simulate_gpu_batch`
  (latent_wrap for region-wise) — `metrics.py` (`_resimulate_and_score :266`,
  `baseline_eval :337`), `posterior.py` (`posterior_predictive_check`),
  `plots.py`, `diagnostics.py`. Confirmed working: first GPU run gave tight
  positive CI (0.1342 [0.128,0.140]) instead of ±0.0000.

## 12. SC_CONDITION branch (OFF default)
- `x = [row_index | fc_upper_tri(64620)]` (no PCA); encoder =
  `MultiChannelMatrixEmbedding` over per-subject SC table `[FC, sc_weight, sc_mask]`
  (channels from SC_CHANNELS; delay excluded per sanity result).
- SC table built `build_sc_table`; per-channel z-score on TRAIN rows only
  (`ScChannelScaler`); mask channel identity.
- sbi `z_score_x="none"` forced (snpe.py:135-136) so the leading integer index
  column is not corrupted.
- Files: inference/sc_channels.py, inference/embedding.py. Eval x via `build_x_obs`.

## 13. Geometry coupling branch (OFF default)
Homotopic SC augmentation (GEOM_ALPHA=0.3, GEOM_KERNEL=homotopic, renorm on).
Adds a homotopic (i, i+180) prior to coupling to recover cross-hemisphere FC that
tractography SC under-recovers. Structural prior only; theta_dim/equations
unchanged. simulation/geometry.py.

## 14. Intended target dataset (cabnp381 + delays)
`HCP_CABNP381_SC_first100.mat` (44MB): weight_all / tract_length_all (381,381,100)
+ subject_id; FC from HCP_FC.mat. delays = tract_length_mm / VELOCITY(3.0 m/s).
Data consistency PASS: cortical-360 SC-FC corr(log) cabnp +0.142 (> v73 +0.117);
cabnp-vs-v73 SC corr +0.931; delays [6.3, 99.7] ms; 100 common subjects.

## 15. Results & sim-cost history
- First GPU run (2026-06-19, per-subject FC, delays OFF, N_TRAIN=20): Test FC corr
  0.1342 [0.128, 0.140], RMSE 0.2151, Val 0.1381, resim 0.0895±0.011.
- Sim cost: rwweib2+basis 360 ROI = 2000 sims / 137s = 14.6 sim/s (delays OFF);
  delays ON ~1.6 sim/s (~9×). Full 70×5000 ≈ 6.7 h delays-off.
- GROUP_AVG_FC=1 expected ceiling ~0.2 (not yet measured).

## 16. Known risks / next steps
1. Full-scale (SMOKE=0, 70×5000) run pending.
2. `/mnt/d/hcp_basis/basis.npy` may be absent → repo-local fallback (not verified
   byte-identical).
3. Cortical-only 360 (subcortex basis unreliable; 381 needs separate offset basis).
4. Uncommitted: modified code + many untracked docs/tests (see `git status`).
5. Code-doc drift to ignore: inline "raw FC passthrough / no PCA" (feature_pipeline
   docstring), "Fisher-z" (training_data print), config.py mouse defaults.
