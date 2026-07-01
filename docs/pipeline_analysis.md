# Pipeline Analysis — HCP basis-regionwise RWW-EIB-2CPL

**Generated:** 2026-06-22
**Branch:** `refactor/02-simulation`
**Entry point:** `main_HCP.py`
**Status:** 분석 전용 문서 — source code 미수정. empirical FC vs simulated FC corr 향상이 목표.

Evidence legend: ✅ 코드로 확인(file:line) · 🧭 맥락상 추정 · ⏳ 실행 확인 필요 · ⚠️ 위험.

> 이 문서는 [CURRENT_HCP_BASIS_PIPELINE.md](CURRENT_HCP_BASIS_PIPELINE.md)를 보완한다.
> 그 문서가 "무엇을 돌리는가"라면, 여기는 "데이터가 어떻게 흐르고 어디서
> corr이 깎이는가"에 집중한다. 진단은 [performance_diagnosis.md](performance_diagnosis.md),
> 수정 계획은 [corr_improvement_plan.md](corr_improvement_plan.md) 참조.

---

## 1. 프로젝트 구조 요약

활성 파이프라인(HCP basis-regionwise)에 실제로 참여하는 파일만 분류한다. 루트에는
mouse/WC 레거시 파일이 다수 공존하나(`main_mouse.py`, `wc_*.py`, `main.py` 등) 현
실험에는 미참여.

| 역할 | 파일 | 비고 |
|---|---|---|
| **Entry point** | `main_HCP.py` | config + 12-step 셀 스크립트(노트북 export) |
| **Config** | `config.py`, `pipeline_setup.py` | base=MPTP 115; `setup_pipeline`이 HCP로 override |
| **Model/simulator** | `cuBNM/simulate_rwweib_2cpl.py`, `cuBNM/runner_rwweib_2cpl.py`, `cuBNM/rww_eib_2cpl.yaml` | RWW-EIB 2-coupling (SC@S_E, SC@S_I) |
| **Engine routing** | `engine_select.py` | `INFERENCE_MODEL→module`, `latent_wrap`, `is_regionwise` |
| **Data loading** | `data_loader_hcp.py`, `data_loader.py` | FC(scipy)·SC(CAB-NP 3D)·split·scale |
| **SC preprocessing** | `data_loader.py::_scale_weights`, `simulation/delays.py` | maxnorm + delay matrix |
| **FC preprocessing** | `data_loader_hcp.py::_build_subject_data*`, `features/fc.py` | NaN→0·대칭화·diag0·upper-tri |
| **Feature extraction** | `inference/feature_pipeline.py`, `features/fc.py` | FC upper-tri → PCA(whiten) |
| **theta 생성** | `inference/training_data.py`, `inference/priors.py`, `inference/scaling.py` | scaled prior → raw coeffs |
| **Parameter decoding** | `basis_decoder.py`, `param_decoder.py` | coeffs → per-region maps |
| **Inference/training** | `inference/snpe.py` (SNPE-C), `inference/posterior.py` | MAF + amortized posterior |
| **Evaluation/metric** | `evaluation/metrics.py`, `evaluation/validation.py`, `evaluation/final_test.py`, `evaluation/model_selection.py` | resim + FC corr/RMSE |
| **Output saving** | `inference/io.py`(artifacts), `save_param_maps.py`, `evaluation/plots.py` | `output_hcp/` |
| **Docs** | `docs/CURRENT_HCP_BASIS_PIPELINE.md`, `docs/SMOKE_TESTS.md`, 본 문서들 | |

---

## 2. 전체 실행 흐름 (entry → output)

```
main_HCP.py
 ├─ setup_pipeline(cfg)                         pipeline_setup.py:279
 │    config(base MPTP) → HCP override(N_REGIONS=360, T_END=180s …)
 │    seed: np.random.seed + torch.manual_seed (CUDA seed 미설정 ⚠️)
 ├─ config.* 직접 패치                           main_HCP.py:112-169
 │    INFERENCE_MODEL=rwweib2, PARAMETER_MODE=basis_regionwise(env)
 │    GROUP_AVG_FC=env "0"(기본 per-subject ⚠️), USE_DELAYS=env "0"(기본 OFF, :179)
 │    basis_regionwise → HETERO_BOUNDS = BASIS_BOUNDS                main_HCP.py:152-153
 ├─ [C1] 배너 + fail-fast 가드 (REQUIRE_BASIS=1)  main_HCP.py:178-213
 ├─ Step 1: 데이터 로드 + split                  main_HCP.py:422-435
 │    load_raw_data → get_target_subjects → three_way_split(SEED=42)
 │    load_all_subjects → {sid:{fc,sc,lengths_mm,delays,fc_nan}}
 │    basis_regionwise 분기: STAGE1_PARAMS=12 coeff, prior U(-2,2)^12  main_HCP.py:486-508
 ├─ Step 7: param scaler + scaled prior          main_HCP.py:592 (inference/snpe.py:378)
 │    ParameterScaler(raw[-2,2]↔scaled[-1,1]), BoxUniform[-1,1]^12
 ├─ Step 2: 학습데이터 시뮬 (캐시 검사 후)        main_HCP.py:666-680
 │    collect_training_data(engine=rwweib2) + latent_wrap            training_data.py:50
 │      theta_scaled~prior → theta_raw(coeffs) → decode → {param}_matrix
 │      → cuBNM 2cpl(force_gpu, hrf="bw") → BOLD → worker_extract → FC upper-tri
 │    필터: _drain_one_future가 non-finite만 제거 (dead/saturated 통과 ⚠️)  training_data.py:294
 ├─ Step 4: FeaturePipeline.fit_transform        main_HCP.py:742-746
 │    FC upper-tri(64620) → PCA(256, whiten) → x_input  (step5 필터 미사용 ⚠️)
 ├─ Step 8: SNPE-C 학습                          main_HCP.py:762 (snpe.py:42)
 │    MAF(hidden=128, transforms=8), embedding=Identity, batch=512, 200ep
 ├─ Phase 3/4: RUN_PHASE24=False → skip (posterior_2 = Phase1 alias)  main_HCP.py:799,830
 ├─ Step 9: validation                          main_HCP.py:968 (validation.py:33)
 │    evaluate_subject: x_obs→posterior sample→resim→FC corr
 │    baseline_eval_subjects: prior-midpoint(=theta0=bound midpoint) resim
 ├─ Step 13: model selection (stage1만)          main_HCP.py:1063 (model_selection.py:87)
 ├─ Step 14: final test (held-out)              main_HCP.py:1078 (final_test.py:32)
 │    + plot_fc_comparison, save_param_maps(val+test)
 └─ Save: artifacts.pkl, embedding_net_s1.pt, final summary
```

---

## 3. 단계별 메커니즘 상세

### 3.1 데이터 입력

| 항목 | 동작 | Evidence |
|---|---|---|
| FC 출처 | `HCP_FC.mat` var `C` (n,2): col0=id, col1=FC(381,381) | ✅ `data_loader_hcp.py:36-40` |
| SC 출처 | `SC_DATASET=cabnp381` → `HCP_CABNP381_SC_first100.mat` weight/tract_length (381,381,100) | ✅ `data_loader_hcp.py:171-181` |
| subject 선택 | FC∩SC 공통 id 중 작은 것부터 `N_SUBJECTS` | ✅ `data_loader_hcp.py:101-114` |
| split | `three_way_split` = sorted→`RandomState(SEED=42).shuffle` (결정적) | ✅ `data_loader.py:411-440` |
| group-avg FC | `GROUP_AVG_FC` env, **기본 "0"=per-subject** | ⚠️ `main_HCP.py:124` |
| ROI 순서 | FC=HCP Glasser 381→[:360], SC=CAB-NP 381→[:360], 독립 slice (동일성 미검증) | ⚠️ `data_loader_hcp.py:117-127` |

### 3.2 SC preprocessing

- `_scale_weights` (`VBI_SC_SCALE=maxnorm`): diag=0 → raw counts → max-norm → `*sc_mask`.
  log1p **미적용**(maxnorm 분기). ✅ `data_loader.py:229-247`
- 대칭화 `(w+w.T)/2` 후 scale. ✅ `data_loader_hcp.py:197`
- delay = `lengths_mm / VELOCITY(3.0)` ; 1 m/s=1 mm/ms이라 단위변환 없음. ✅ `simulation/delays.py:55`
- NaN/Inf: delay finite assert. ✅ `data_loader_hcp.py:203`
- shape: `(360,360)` per subject.

### 3.3 FC preprocessing

| 항목 | 동작 | Evidence |
|---|---|---|
| Fisher-z | **어디에도 미적용** — empirical/sim 모두 raw Pearson r in [-1,1] | ✅ `features/fc.py:1-11`, `data_loader_hcp.py:190` |
| NaN | `np.nan_to_num(nan=0.0)` + `fc_nan` mask 별도 보관 | ✅ `data_loader_hcp.py:189` |
| 대칭화/diag | `(fc+fc.T)/2`, `fill_diagonal(0)` | ✅ `data_loader_hcp.py:190-191` |
| upper-tri | `np.triu_indices(n,k=1)` (obs/sim 동일 경로) | ✅ `features/fc.py:53` |
| sim FC | `np.corrcoef(ts.T)` → nan→0 → diag0 (동일 raw r) | ✅ `features/fc.py:30-33` |
| scale 차이 | obs/sim 둘 다 raw r → vectorization 일치, scale mismatch 없음 | ✅ |

→ **FC 벡터화·Fisher-z·diagonal·upper-tri는 obs/sim 간 일관**. 이 축에서는 버그 없음.

### 3.4 feature extraction

- raw FC upper-tri(64620) → `FeaturePipeline`.
- `FC_PCA_DIM=256`, whiten=True PCA, train fc_raw에 fit. ✅ `inference/feature_pipeline.py:92-100`
- USE_FCD=False → FCD 미사용, FCD z-score off. ✅
- selected-edge / network-summary / FCD-summary feature **미사용**.
- x_obs도 동일 pipeline.transform 통과(같은 PCA basis). ✅ `inference/posterior.py:149`
- ⚠️ PCA는 **simulated FC에만 fit** → empirical FC는 그 basis로 투영. whiten이라 저분산
  방향 증폭 → empirical OOD 위험 (→ [performance_diagnosis.md](performance_diagnosis.md) D2).

### 3.5 theta / parameter decoding

| 항목 | 값 | Evidence |
|---|---|---|
| inference target | basis coefficient (per-ROI param 아님) | ✅ `basis_decoder.py:3-8` |
| theta_dim | 12 = n_params(4) × basis_dim(3) | ✅ `basis_decoder.py:39` |
| coeff order | `g_LRE_{const,myelin,gradient}, g_FFI_…, I_o_…, sigma_…` | ✅ `basis_decoder.py:92-96` |
| basis shape | disk `(381,3)`→slice `(360,3)`; rezscore로 myelin/gradient 표준화 | ✅ `basis_decoder.py:48-64`, 실측 col0=const 1.0, col1/2 mean≈0 std≈1 |
| 변환 | `z=beta@basis.T; map=mid+half*tanh(z)` (tanh bound, clip 불요) | ✅ `basis_decoder.py:79-86` |
| bounds | basis 모드=BASIS_BOUNDS g(0,3)/I_o(0,1)/sigma(0,0.05) | ✅ `main_HCP.py:146,490` |
| theta=0 | midpoint maps: g=1.5, I_o=0.5, sigma=0.025 | ✅ tanh(0)=0 |
| sim 반영 | `latent_wrap`: decode→`{param}_matrix`→`build_param_lists` per-node 주입 | ✅ `engine_select.py:102-117`, `runner_rwweib_2cpl.py:128-135` |

### 3.6 simulator (cuBNM RWWEIB_2CPL)

| 항목 | 값 | Evidence |
|---|---|---|
| dynamics | E: globalinput=SC@S_E (gain g_LRE·J_N); I: SC@S_I (g_FFI·J_N·λ_IE) | ✅ `runner_rwweib_2cpl.py:5-9` |
| dt | 1.0 ms | ✅ `main_HCP.py:76` |
| 길이/cut | T_END=180s, T_CUT=60s → 분석 120s | ✅ `main_HCP.py:74-75` |
| BOLD T | `ANALYSIS_BOLD_T = (180-60)/0.72 ≈ 166 TR` | ✅ `pipeline_setup.py:127-129` |
| noise | sigma (per-region, 추론), sim_seed=42 **고정** | ⚠️ `simulate_rwweib_2cpl.py:55` |
| global coupling | g_LRE/g_FFI (추론), FIC 없음, J_N=0.15/w_p=1.4/λ_IE=1.0 고정 | ✅ `main_HCP.py:119` |
| delay | USE_DELAYS env 기본 "0"(OFF; delays 미사용) → sc_dist=None | ✅ `main_HCP.py:179`, `simulate_rwweib_2cpl.py:43-48` |
| HRF | `hrf="bw"` (cuBNM Balloon-Windkessel) | ✅ `runner_rwweib_2cpl.py:181-195` |
| sim FC | `compute_fc(bold)=corrcoef` raw r | ✅ `features/fc.py:30` |
| adapter trim | BOLD `[-ANALYSIS_BOLD_T:]` | ✅ `simulate_rwweib_2cpl.py:73-80` |

### 3.7 inference / training (SNPE-C)

| 항목 | 값 | Evidence |
|---|---|---|
| prior | scaled `BoxUniform[-1,1]^12`; raw coeff U(-2,2) | ✅ `inference/priors.py:34`, `main_HCP.py:495-497` |
| scaler | `ParameterScaler` raw[-2,2]↔scaled[-1,1], per-coeff | ✅ `inference/scaling.py:40-48` |
| density est | MAF hidden=128, transforms=8 | ✅ `inference/snpe.py:123-128`, `config NDE_*` |
| embedding | Identity (raw PCA-FC → MAF) | ✅ `main_HCP.py:761`, `snpe.py:106-107` |
| 학습 | batch=512, max 200ep, early-stop 비활성 | ✅ `snpe.py:153-155` |
| append device | x는 CPU 유지(46GB OOM 방지), minibatch만 GPU | ✅ `snpe.py:147-150` |
| posterior 저장 | `build_posterior(estimator)` | ✅ `snpe.py:258` |
| posterior sample | rejection(시간캡60s) 실패→`reject_outside_prior=False` clip | ⚠️ `posterior.py:64-86` |
| mean/MAP/top-k | **미사용** — sample만 사용 | ⚠️ `posterior.py:88-90` |

### 3.8 evaluation

| 항목 | 동작 | Evidence |
|---|---|---|
| corr 계산 | `fc_metrics` upper-tri(k=1), `np.corrcoef` | ✅ `metrics.py:47-72` |
| diagonal | k=1 제외 | ✅ |
| NaN 제거 | `isfinite & ~fc_nan` mask | ✅ `metrics.py:58-60` |
| obs/pred vectorization | 동일 (둘 다 triu k=1, raw r) | ✅ |
| resim 방식 | n_resim posterior draw 각각 sim → **draw별 corr 평균** | ⚠️ `metrics.py:171,247-272` |
| best-sample 선택 | 없음 (FC 행렬 평균/MAP/top-k 없음) | ⚠️ |
| subject-wise corr | `fc_corr_all` 저장, bootstrap CI | ✅ `final_test.py:57-60` |
| train/val/test 분리 | val=Step9, test=Step14(held-out, 1회) | ✅ |
| FC RMSE | mask 후 `sqrt(mean((a-b)^2))` | ✅ `metrics.py:68-71` |
| baseline | prior-midpoint(=theta0=bound midpoint) resim | ✅ `metrics.py:311-324` |
| engine routing | `get_simulate_gpu_batch()` = latent_wrap (학습과 동일 엔진) | ✅ `engine_select.py:120-126` |

→ engine-routing은 [eval-engine-routing-bug] 수정 후 정상. resim FC가 theta를 실제 반영함.

### 3.9 output saving

- `inference.save_artifacts` → `output_hcp/artifacts.pkl` (scaler/pipeline/prior bounds).
- `save_param_maps` → `param_maps.npz` (n_subj, 360, 4) posterior-mean decode.
- plots → `output_hcp/` (단, train-data 그림은 `output_mouse_mptp/`로 새는 잔재 ⚠️ `main_HCP.py:579`).

---

## 4. 핵심 파일/함수 관계도

```
config(setup_pipeline) ─┬─ data_loader_hcp ── subject_data{fc,sc,delays,fc_nan}
                        │
          basis.npy ──► basis_decoder.get_decoder ─┐
                                                    ├─ param_decoder.decode_to_param_maps
        STAGE1_PARAMS(12 coeff)                     │        │
                ▼                                   │        ▼
 inference.scaling.ParameterScaler ◄── prior ──► engine_select.latent_wrap
                │                                            │
                ▼                                            ▼
 training_data.collect_training_data ──► cuBNM.runner_rwweib_2cpl ──► BOLD
                │                                            │
                ▼                                            ▼
 feature_pipeline(PCA) ──► x_input ──► snpe.train_snpe ──► posterior
                                                             │
 empirical FC ──► feature_pipeline.transform ──► x_obs ──────┤
                                                             ▼
                              posterior.infer_subject_raw ──► samples
                                                             │
                              metrics._resimulate_and_score ─► FC corr/RMSE
```

---

## 5. 객체 shape 요약

| 객체 | shape | 출처 |
|---|---|---|
| empirical FC | `(360,360)` raw r | `data_loader_hcp.py` |
| SC | `(360,360)` maxnorm | `_scale_weights` |
| delays | `(360,360)` ms | `compute_delay_matrix` |
| basis(active) | `(360,3)` | `basis_decoder` |
| theta(coeffs) | `(S,12)` | scaler/prior |
| decoded maps | `{param:(S,360)}` | decoder |
| overrides | `{param}_matrix (S,360)` | `make_fixed_overrides` |
| BOLD | `(~166,360)` | runner/adapter |
| FC feature(raw) | `(S,64620)` upper-tri | `features/fc.py` |
| x_input(PCA) | `(S,256)` whiten | `feature_pipeline` |

---

## 6. 관찰된 실측 (run_basis_t20_s2000.log)

N_TRAIN=20, N_SIM=2000(40k sims), per-subject FC, 결과:

| metric | value |
|---|---|
| Test FC corr | **0.1342** [0.1280, 0.1403] |
| Val FC corr | 0.1381 |
| Test FC RMSE | 0.2151 |
| baseline(prior-mid) corr | ≈ 0.00–0.008 |
| per-subject corr 분포 | 0.040 ~ 0.214 (편차 큼) |
| posterior rejection accept | **0.4–1.0%** (OOD 신호) |
| param_maps(이 로그) | g_LRE mean 7.4 (0–9 wide bounds 시절), g_FFI 6.3 |

> param_maps의 g≈7은 **wide HETERO_BOUNDS로 캐시된 decoder** 흔적(현재 코드는
> BASIS_BOUNDS 0–3로 override됨). corr 0.134는 g가 7까지 열렸을 때의 수치임을 유의.

baseline≈0 → model이 0→0.13의 실신호를 만든다(engine-routing fix 정상). 그러나
**0.4–1% accept**와 **per-subject 편차**가 핵심 병목 신호. 상세는
[performance_diagnosis.md](performance_diagnosis.md).
