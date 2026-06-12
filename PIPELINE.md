# HCP VBI-SBI 파이프라인 상세 문서

`main_HCP.py` 기준. RWWEIB_2CPL forward 모델 + SNPE-C amortized 추론으로
HCP 피험자의 뇌 파라미터를 empirical FC로부터 추론한다.

---

## 0. 전체 개요

```
SC (구조연결) ──┐
                ├─► forward 시뮬(RWWEIB_2CPL) ─► sim BOLD ─► sim FC
theta (param) ──┘                                              │
                                                               ▼
empirical FC ──────────────────────────────► SNPE-C 학습 ─► posterior q(theta | FC)
                                                               │
                                            새 피험자 FC ─► theta 추론(amortized)
```

- **목표**: empirical FC를 재현하는 생성 파라미터를 amortized 추론.
- **핵심 지표**: 추론한 theta로 재시뮬한 FC와 empirical FC의 상관 (FC corr).
- **엔진**: cuBNM (GPU). forward 모델은 codegen으로 yaml→CUDA 생성.

---

## 1. 설정 (main_HCP.py 상단)

### 1.1 데이터 설정 (PipelineConfig)
| 항목 | 값 | 의미 |
|---|---|---|
| `N_REGIONS` | 360 | cortical-only (Glasser 360, subcortical 21 드롭) |
| `FC_DIM` | 64620 | 360·359/2 upper-tri |
| `T_END_MS` | 180000 | 시뮬 180초 |
| `T_CUT_MS` | 60000 | 앞 60초 transient 제거 |
| `DT` | 1.0 ms | 적분 스텝 |
| `TR_SEC` | 0.72 | BOLD 샘플링 (HCP TR) |
| `GPU_BATCH` | (env) | 한 GPU 배치 sim 수 |
| `VBI_SC_SCALE` | maxnorm | SC 스케일링 (log1p/maxnorm/sqrt 중) |

### 1.2 모델 / 추론 설정
| 항목 | 값 |
|---|---|
| `INFERENCE_MODEL` | `rwweib2` (RWWEIB_2CPL) |
| `STAGE1_PARAMS` | `[g_LRE, g_FFI, I_o, sigma]` (4개) |
| `STAGE1_PRIOR_LOW` | `[0, 0, 0.15, 0]` |
| `STAGE1_PRIOR_HIGH` | `[9, 9, 0.60, 0.09]` (기존 3×) |
| `RWWEIB2_FIXED` | `w_E=1.0, w_I=0.7, J_i=1.0, w_p=1.4, J_N=0.15, lambda_IE=1.0` |
| `GROUP_AVG_FC` | True (타겟 = 1039명 평균 FC) |
| `USE_DELAYS` | False (tract length 미사용) |
| `USE_FCD` | False (HCP에 FCD 없음) |
| `RUN_PHASE24` | False (Phase2/4 비활성) |

### 1.3 데이터 소스 (data_loader_hcp.py)
- `HCP_FC.mat`: 1039명 × 381×381 FC (scipy C 배열, col 1 = raw Pearson r).
- `HCP_SC.mat`: 1040명 SC (v7.3 h5py; row1 weight, row2 tract length).
- **cortical-only**: 파일은 381이나 `_cortical_slice`로 앞 360만 사용.
- **group-avg FC**: `_compute_group_fc`가 1039명 FC 평균 → 모든 피험자 타겟으로.
- SC 스케일: `_scale_weights` (maxnorm = SC/SC.max).

---

## 2. RWWEIB_2CPL 모델 (cuBNM/rww_eib_2cpl.yaml)

2-population (E/I) reduced Wong-Wang. **두 개의 독립 connectome 결합** —
E는 SC@S_E, I는 SC@S_I로 구동 (이름의 "2CPL").

### 2.1 입력 전류
```
globalinput_E_i = Σ_j SC_ij · S_E_j          # E 장거리 입력
globalinput_I_i = Σ_j SC_ij · S_I_j          # I 장거리 입력

I_E = w_E·I_o + w_p·J_N·S_E + g_LRE·J_N·globalinput_E − J_i·S_I
I_I = w_I·I_o +     J_N·S_E + g_FFI·J_N·lambda_IE·globalinput_I − S_I
```

### 2.2 발화율 + 동역학
```
r_E = H_E(I_E),  r_I = H_I(I_I)              # sigmoid형 transfer (a,b,d 상수)
dS_E/dt = −S_E/τ_E + (1−S_E)·γ_E·r_E + σ·ξ_E
dS_I/dt = −S_I/τ_I +          γ_I·r_I + σ·ξ_I
```
- 상수: a_E=310,b_E=125,d_E=0.16 / a_I=615,b_I=177,d_I=0.087 / γ_E=0.641/1000,
  γ_I=1/1000 / τ_E=100ms, τ_I=10ms.
- `bold_state_var = S_E` → BW(Balloon-Windkessel) HRF로 BOLD 생성.

### 2.3 파라미터 종류
| 종류 | 이름 | 설명 |
|---|---|---|
| global | g_LRE | E 장거리 결합 gain (모델 유일 global_param) |
| regional | g_FFI | I 장거리 결합 gain |
| regional | I_o | 배경 입력 전류 |
| regional | sigma | 노이즈 진폭 |
| regional | w_E, w_I | E/I 외부입력 스케일 (상수→승격) |
| regional | J_i | I→E 억제 weight (FIC 대상) |
| regional | w_p | 국소 E 재귀 |
| regional | J_N, lambda_IE | NMDA 결합, I 장거리 스케일 |

**현재**: g_LRE/g_FFI/I_o/sigma 추론, 나머지 고정. 모든 추론 param은 **노드 공통(homogeneous)** —
한 sim에서 360 노드가 같은 값 (build_param_lists가 스칼라를 노드에 브로드캐스트).

### 2.4 멀티커플링 커널 (cubnm_build)
- codegen이 `conn_state_vars: [S_E, S_I]` 파싱 → `n_conn_state_vars=2`.
- 커널 history 버퍼가 coupling 당 interleave (`*N_CONN + c`). n_conn=1이면 기존과 bit-identical.
- `<name>_matrix` fixed 키로 per-(sim,node) 행렬 주입 가능 (FIC J_i, heterogeneous).

---

## 3. 파이프라인 단계 (main_HCP.py 실행 순서)

### Step 1 — 데이터 분할 + 로드
- `get_target_subjects`: FC·SC 둘 다 있는 1039명 중 id 작은 순 N_SUBJECTS명.
- `three_way_split`: train/val/test (현재 70/10/20, env로 override).
- `load_all_subjects(train+val+test)`: 각 피험자 `{sc, fc, lengths_mm, delays, fc_nan}`.
  group-avg면 모든 피험자 fc = group 평균.

### Step 7 — 파라미터 스케일링 + prior (Step 2 전 실행)
- `step7_fit_param_scaler` → `ParameterScaler` + `prior_scaled`.
- **ParameterScaler** (scaling.py): raw param ↔ [−1,1] 박스.
  - `transform`: scaled = 2·(raw−low)/range − 1
  - `inverse_transform`: raw = (scaled+1)/2·range + low
- **prior_scaled** (priors.py): `BoxUniform([−1,1]^4)` (torch). 균등.
- 샘플 흐름: `theta_scaled ~ U(−1,1)^4` → `inverse_transform` → `theta_raw ~ U(low,high)` (각 param 독립 균등).

### Step 2 — 시뮬레이션 (학습데이터 생성)
`step2_simulate_train` → `collect_training_data`. **핵심 단계.**

```
엔진 선택: rwweib2 → cuBNM.simulate_rwweib_2cpl
가드: scaler.param_names == config.STAGE1_PARAMS (불일치 시 에러)

for 각 train subject:                       # subject 루프
    sc, delays = subject_data[sid]
    theta_s = prior_scaled.sample(n_sim)     # 이 피험자용 n_sim개 랜덤 theta (scaled)
    theta_r = scaler.inverse(theta_s)        # raw 값 (실제 param)
    for batch in ceil(n_sim / GPU_BATCH):    # GPU 배치 분할
        bolds = simulate_gpu_batch(sc, theta_batch, param_names, delays, apply_bw=True)
        future = executor.map(worker_extract, bolds)   # BOLD→FC 병렬 추출
        모음 (theta_s, theta_r, fc, fcd)
```

- **총 sim 수** = n_train × n_sim. 피험자마다 독립 prior 샘플.
- **한 sim** = 4-D prior 박스에서 점 하나 (4 param 동시 랜덤, i.i.d).
- BOLD는 배치마다 worker 프로세스로 streaming → 전체 BOLD 저장 안 함 (메모리 절약).
- **캐시**: `output_hcp/features_stage1.npz`. 로드 시 가드 — theta 행수(n_train·n_sim),
  열수(len(STAGE1_PARAMS)), fc_dim 일치 확인. 불일치면 STALE → 재시뮬.

#### simulate_gpu_batch 내부 (simulate_rwweib_2cpl.py → runner_rwweib_2cpl.py)
```
run_cubnm_rwweib2_batch(sc, theta(4col), pn):
    build_param_lists:                       # theta → param_lists
        regional 기본값 = RWWEIB2_FIXED 오버레이
        theta에 있는 param은 per-sim 값을 노드에 브로드캐스트 (n_sims, n_nodes)
        g_LRE = per-sim 스칼라 (global)
        "<name>_matrix" 키 있으면 per-(sim,node) 행렬로 override (FIC 등)
    RWWEIB_2CPLSimGroup.run()                # GPU 적분
    BW HRF → BOLD (T, 360)
→ T_CUT 트림 → (ANALYSIS_BOLD_T, 360) BOLD 리스트
```

### Step 3 — 피처 요약 + 저장
- `step3_summary_features`: fc_raw/fcd_raw shape, 유한성 출력.
- `save_extracted_features` → `features_stage1.npz` (theta_scaled/raw, fc_raw, fcd_raw, param_names).

### Step 4 — 피처 파이프라인 (raw FC passthrough)
- PCA·z-score 없음. raw FC upper-tri (64620) 그대로 `x_input`.
- RegionTransformer가 SNPE 학습 중 압축 (embedding net).

### Step 8 — Stage 1 추론 (single-round SNPE-C)
`step8_train_snpe`:
- **embedding net** = RegionTransformer: raw FC (64620) → 128-d feature.
  - FC upper-tri를 노드별로 재구성, transformer attention으로 영역 임베딩.
- **density estimator** = SNPE-C (NSF/MAF 등): q(theta_scaled | embedding).
- 학습: (theta_scaled, x_input) 쌍으로 posterior 학습.
- 결과: `posterior` (DirectPosterior), `embedding_net`.

### Phase 3 / 4 — (현재 SKIP, RUN_PHASE24=False)
- Phase 3: attention×gradient로 top-k FC edge 선택 (feature selection).
- Phase 4: 선택 edge로 2차 SBI.
- 현재 비활성 — final_test가 Phase 1 posterior(전체 FC) 사용. posterior_2 = posterior alias.

### Step 9 — Stage 1 검증 (validation)
val 피험자에 대해:
- **posterior 샘플링**: 각 피험자 empirical FC → q(theta|FC) → posterior mean theta.
- **재시뮬(resim)**: 추론 theta로 N_TEST_RESIM회 재시뮬 → sim FC.
- **FC corr / RMSE**: sim FC vs empirical FC.
- **shrinkage**: prior 대비 posterior std 축소 (식별성).
- **baseline**: prior mean theta로 재시뮬 (추론 효과 비교).
- **embedding probing**: 선형 R²로 param별 식별성.
- **ActiveSubspace** (active_sensitivity.py): sbi gradient 민감도 — param별 점수.
- **SBC**: simulation-based calibration (posterior 보정 검사, rank 히스토그램).

### Step 13 — 모델 선택 (validation)
- Stage 1 vs Stage 2 점수 (FC corr 기반, USE_FCD=False면 FCD 제외).
- 더 나은 stage 선택 (현재 Stage 1).

### Step 14 — 최종 테스트
- test 피험자에 대해 선택 stage로 추론 + 재시뮬.
- FC corr / RMSE (bootstrap 95% CI).
- `report_step14` → 최종 요약 표.

---

## 4. 엔진 라우팅 (engine_select.py)

학습/평가/SBC/resim/predictive 전 경로가 `config.INFERENCE_MODEL`로 통일.
| engine | 모듈 |
|---|---|
| `rwweib2` | cuBNM.simulate_rwweib_2cpl |
| `rwweib` | cuBNM.simulate_rwweib (단일결합 FFI) |
| `rwweibdelay` | cuBNM.simulate_rwweib_delay |
| `rww` | cuBNM.simulate_rww (stock Deco) |
| `vbi`/`gpu` | cupy VBI 엔진 |

---

## 5. FIC (feedback inhibition control) — fic_tune.py

J_i(억제 weight)를 영역별로 튜닝해 I_E를 작동점(≈0.377 nA, r_E≈3Hz)에 고정.
- **방법**: online homeostatic (Deco 2014). 결정론적 mean-field를 돌리며
  J_i를 `J_i += η·(mean_I_E − target)`로 반복 조정. numpy로 전 sim 벡터화.
- `compute_fic_ji(sc, theta, pn)` → J_i (n_sims, n_nodes). theta에서 g_LRE/g_FFI/I_o/w_p/w_E/w_I 읽음.
- 사용처: `fixed["J_i_matrix"]`로 cuBNM 주입. (현재 main 파이프라인엔 미적용 — 진단 도구에서 사용.)

---

## 6. 진단 도구 (파이프라인 외부, 수동 실행)

| 도구 | 역할 |
|---|---|
| `sensitivity.py` | forward FC 스윕 — param별 FC 반응(simFC~SC, ~real, 작동점, reorg) |
| `active_sensitivity.py` | sbi ActiveSubspace — gradient 기반 param 식별성 |
| `fc_support_diag.py` | 3-in-1: ①param 민감도 ②민감 edge ③empirical이 sim support 안에 있나. `--prior_scale`, `--io`, `--gsr {demean,eig1}`, `--fic`, `--hetero` 옵션 |
| `eib_tune.py` | EIB per-edge effective connectivity 튜닝 (Hebbian, 비-amortized) |
| `hcp_ceiling.py` | param 랜덤서치로 FC corr 천장 측정 |

---

## 7. 데이터/메트릭 레버 (성능 관련)

진단으로 확인된 핵심 (FC corr이 낮은 이유):
- **per-subject raw FC ~ SC**: ≈0.05 (천장 낮음). → cortical+group으로 0.18.
- **homogeneous param**: 노드 공통이라 노드 차이는 SC뿐 → SC 밖 구조(homotopic) 못 만듦.
- **offset +0.068**: empirical FC의 전역 양의 성분 (global signal). 어떤 param으로도 미도달 →
  전처리(GSR/demean) 영역.
- **random 탐색 천장 ~46%** (support 본체): param 수·prior 폭·노드별 랜덤 무엇도 못 넘음.
  → random이 아닌 **구조화(region-wise 추론)** 필요.

### region-wise (heterogeneous) 방향 — Kong 2021 (vbi/models/pytorch/rww_sde_kong.py)
영역별 param = `C @ coeffs`, C = [myelin, gradient, 1] (공간기저). 추론은 계수 ~10개.
각 영역 다른 값(생물 gradient 조직) + 식별가능. 우리 적용 시 기저 = myelin/gradient(맵 필요)
또는 connectome harmonic(SC 라플라시안 고유벡터, SC만으로).

---

## 8. 실행

```bash
# smoke (빠른 검증)
cd /scratch/home/wog3597/vbi
rm -f output_hcp/features_stage1.npz
N_SUBJECTS=8 N_TRAIN=4 N_VAL=2 N_TEST=2 N_SIM=50 python main_HCP.py

# full
python main_HCP.py

# cuBNM 재빌드 (yaml/커널 변경 시)
cd /scratch/home/wog3597/cubnm_build
python codegen/generate_models.py
pip install -e . --no-build-isolation
```

환경변수 override: `N_SUBJECTS, N_TRAIN, N_VAL, N_TEST, N_SIM, GPU_BATCH, VBI_SC_SCALE`.
