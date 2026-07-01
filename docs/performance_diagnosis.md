# Performance Diagnosis — empirical vs simulated FC corr 저하 원인

**Generated:** 2026-06-22
**Scope:** HCP basis-regionwise RWW-EIB-2CPL. 목표 = empirical FC vs sim FC corr ↑,
RMSE ↓, posterior 안정성 ↑, val/test robust(train 과적합 금지).
**기반:** [pipeline_analysis.md](pipeline_analysis.md) 코드 분석 + `run_basis_t20_s2000.log` 실측.

Confidence: **High**=코드 근거 명확 · **Medium**=가능성 높음, 실행확인 필요 · **Low**=의심.

> 순수 스타일/네이밍/포맷 취향은 제외. 잘못된 동작·corr 저하·shape mismatch·
> data leakage·NaN/Inf·reproducibility를 유발할 수 있는 모든 것을 보고.

---

## 0. 한눈 요약 (우선순위 정렬)

| ID | Issue | Conf | corr 영향 | 종류 |
|---|---|---|---|---|
| **D1** | per-subject FC 타깃이 기본값(GROUP_AVG_FC env "0") — 천장 ~0.05 | High | **HIGH** | config |
| **D2** | posterior rejection accept <1% → empirical FC OOD, clip fallback | High | **HIGH** | conditioning |
| **D3** | draw별 corr 평균(노이즈) — expected-FC/MAP 미사용 | High | MED-HIGH | eval |
| **D4** | Step4가 step5 sim-품질 필터 우회 → degenerate FC가 PCA 오염 | High | MED | feature |
| **D5** | g_LRE/g_FFI bound (0,3)이 posterior 수요(~7)보다 좁아 under-couple | Med-High | MED | bounds |
| **D6** | 모델 미스펙: FIC 없음 + J_N/w_p/λ_IE 고정 → simFC std 천장 | Med | HIGH(천장) | model |
| **D7** | 분석창 120s(166 TR)로 짧음 → simFC 노이즈 | Med | MED | sim |
| **D8** | FC(HCP)·SC(CAB-NP)·basis.npy ROI 순서 동일성 미검증 | Med | HIGH(if 어긋남) | alignment |
| **D9** | whiten PCA가 empirical FC를 OOD로 밀어냄(D2 근원) | Med | MED | feature |
| **D10** | CUDA seed/deterministic 미설정 → posterior 재현 불가 | High | (재현성) | repro |
| **D11** | ~~USE_DELAYS 기본 ON vs 문서 OFF~~ — RESOLVED: 코드 기본 "0"(OFF, `main_HCP.py:179`), 문서와 일치 | — | (해결) | config |
| **D12** | sim_seed=42 고정(학습+resim) — 단일 노이즈 실현 | Low-Med | LOW | sim |

---

## 1. 상세 진단표

| Priority | Issue | Conf | Related files/functions | Evidence | Why corr↓ | Suggested fix | Risk | Validation |
|---|---|---|---|---|---|---|---|---|
| 1 | **D1 per-subject FC 타깃** | High | `main_HCP.py:124`, `data_loader_hcp.py:208-281` | `GROUP_AVG_FC=os.environ.get("GROUP_AVG_FC","0")=="1"` → 기본 per-subject. 로그 per-subj corr 0.04–0.21, baseline≈0 | per-subject raw FC는 SC-FC corr 천장 ~0.05; 한 명 FC는 노이즈 큼. group-avg는 공유 SC-구조 드러내 천장 ~0.18–0.2 | `GROUP_AVG_FC=1`로 실행(설정만). CLAUDE.md가 group-avg를 active로 명시하나 코드 기본은 per-subject | 낮음 — 데이터/평가 의미 안 바뀜, 동일 타깃을 train/val/test 모두에 적용(leak 아님) | val·test corr 동시 상승 확인 (group이면 모든 subj 동일 타깃) |
| 2 | **D2 posterior accept <1%** | High | `posterior.py:64-86`, log "Only 0.442% accepted" | rejection 거의 실패 → `reject_outside_prior=False` clip fallback → 사실상 prior-ish/경계 clip 샘플 | OOD x에서 flow가 신뢰 못할 영역 매핑 → 샘플 품질 낮음 → resim FC가 empirical 못 좇음 | feature conditioning 개선(D9): non-whiten PCA 또는 x z-score, 또는 sim 분포에 empirical 포함되도록 model/bounds 확장 | 중간 — feature 공간 바뀌면 재학습 필요 | accept율 모니터(>30% 목표), corr↑ 동반 확인 |
| 3 | **D3 draw별 corr 평균** | High | `metrics.py:171-205,247-272`, `posterior.py:166-178` | n_resim개 single-run noisy FC 각각 corr→평균. FC 행렬 평균/MAP/top-k 없음 | 120s 단일 sim FC는 노이즈 큼 → 개별 corr 저평가. 여러 draw·노이즈 평균(expected FC)하면 노이즈 상쇄 | resim FC들을 **행렬 평균 후 1회 corr**; posterior-mean/MAP도 별도 sim해 비교 저장 | 낮음 — 평가식만, 더 느슨하지 않음(노이즈 제거는 합법) | 같은 posterior에서 expected-FC corr > draw평균 corr 확인. val/test 동시 |
| 4 | **D4 Step4 필터 우회** | High | `main_HCP.py:742-746` vs `snpe.py:321-337` | main_HCP는 `FeaturePipeline()` 직접 fit_transform. step5의 dead/saturated mask 미적용. training_data는 non-finite만 제거 | std≈0/dead FC가 whiten PCA basis 오염 → x 공간 왜곡 → conditioning 악화 | Step4에서 step5와 동일 mask(finite&notsat&inrange&alive) 적용 후 PCA fit | 낮음 — 유효샘플만 사용 | 필터 전/후 PCA explained-var, accept율, corr 비교 |
| 5 | **D5 coupling bound 협소** | Med-High | `main_HCP.py:146`, log param_maps g_LRE max=9.0 mean=7.4 | wide(0–9) 시절 posterior가 g→~7 선호. 현재 BASIS_BOUNDS g(0,3) cap | 최적 coupling이 3 초과면 under-couple → simFC의 SC-구조 약화 → corr↓ | g_LRE/g_FFI bound을 (0,6) 정도로 재확장 후 재측정(BASIS_BOUNDS만) | 중간 — 너무 키우면 numerical explosion. sigma/tanh가 bound 유지하므로 NaN은 decode 단계서 안 남 | bound (0,3) vs (0,6) ablation: corr·accept·non-finite율 |
| 6 | **D6 모델 미스펙(FIC 없음)** | Med | `main_HCP.py:119`, memory [vbi-fc-misspecification] | FIC/E-I 균형 튜닝 없이 J_N=0.15/w_p=1.4/λ_IE=1.0 고정. simFC std ~0.088 고정(실제 0.17–0.23) | E/I 불균형 regime이면 simFC가 SC를 약하게만 반영 → 패턴 천장 | (생물학 우선 아님) J_N/w_p/λ_IE을 추론셋에 추가하거나 light-FIC로 dynamic range 확대 | 중간-높음 — regime 불안정 가능. non-finite 가드 필수 | forward sweep으로 simFC std·SC-corr 민감 param 확인(sweep_fc/forward_ablation) |
| 7 | **D7 짧은 분석창** | Med | `main_HCP.py:74-75`, `pipeline_setup.py:127` | 120s=166 TR. HCP empirical은 ~14.4분(1200 TR) | 짧은 sim FC는 추정 분산 큼 → corr 저평가·불안정 | T_END↑(예 300s) 또는 D3의 노이즈평균으로 대체(저비용) | sim 비용 ↑ (선형~delays 9×). D3가 더 싸다 | 동일 theta로 120s vs 300s simFC corr 안정성 비교 |
| 8 | **D8 ROI/basis 정렬 미검증** | Med | `data_loader_hcp.py:117-127`, `basis_decoder.py:55-56` | FC=HCP Glasser→[:360], SC=CAB-NP→[:360], basis(381)→[:360] 각각 독립 slice. 코드에 동일순서 assert 없음 | 행 순서 다르면 SC-FC 어긋남 + basis가 myelin/grad을 엉뚱 region에 매핑 → corr 손실(크래시는 안 남) | 세 소스의 region label 1:1 매칭 검증(이미 SC-FC corr +0.142로 부분근거). label 파일로 명시 assert | 낮음(검증), 중간(만약 어긋났다면 수정 필요) | 세 소스 라벨 교차표; 셔플 basis로 corr 떨어지는지 음성통제 |
| 9 | **D9 whiten PCA OOD** | Med | `feature_pipeline.py:92-100` | whiten=True PCA를 **sim FC에만** fit. 저분산 방향 증폭, empirical 투영 폭주 가능 | D2의 근원 — empirical x가 train 분포 밖 → posterior confident-wrong | whiten=False 또는 표준화 후 PCA; 또는 sim+emp 합성에 fit(주의: emp 누수 금지, sim만 fit 권장) | 중간 — 표현력/조건수 변화 | x_obs vs x_train Mahalanobis/percentile; accept율 |
| 10 | **D10 재현성(CUDA seed)** | High | `pipeline_setup.py:330-337` | np.seed + torch.manual_seed만. cuda manual_seed_all/cudnn deterministic 없음. SNPE CUDA 비결정적 | 같은 config/seed라도 posterior·corr 재현 안 됨(목표 #6 위배) | `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`(또는 seed 기록) 추가 | 낮음 — 약간 느려질 수 있음 | 동일 seed 2회 실행 corr 차이 측정 |
| 11 | **D11 USE_DELAYS drift (RESOLVED)** | — | `main_HCP.py:179` | 코드 기본 "0"(OFF). 문서/첫 실험과 일치 — drift 없음 (이전 "기본 ON" 주장은 오독, :169는 VELOCITY) | 없음 | 코드 기준 OFF로 확정 | 없음 | delays ON/OFF corr·비용 ablation (선택) |
| 12 | **D12 sim_seed 고정** | Low-Med | `simulate_rwweib_2cpl.py:55` | 모든 batch sim_seed=42. 학습·resim 동일 노이즈 base | 단일 노이즈 실현 → simFC stochastic 변동 미평균(D3와 결합) | resim 시 seed 다양화 후 평균(D3 expected-FC와 함께) | 낮음 | 노이즈 N회 평균 simFC corr↑ 확인 |

---

## 2. 명시적으로 "버그 아님"으로 확인된 항목 (오탐 방지)

이 축들은 의심 후보였으나 코드 확인 결과 **일관/정상**:

- **Fisher-z 중복/누락 없음** — obs·sim 모두 raw Pearson r, 어디서도 arctanh 미적용. `features/fc.py`, `data_loader_hcp.py:190`. ✅
- **upper-tri/diagonal 불일치 없음** — obs·sim·metric 모두 `triu_indices(n,k=1)`. ✅
- **engine routing 정상** — 학습·resim·SBC·predictive 모두 `INFERENCE_MODEL`→`latent_wrap` 경유(과거 WC 하드코딩 버그 수정됨, [eval-engine-routing-bug]). resim이 theta 실제 반영. ✅
- **theta_dim/coeff order/bounds 일관** — scaler·prior·decoder 길이 12 일치, Step7 가드(`snpe.py:383`)·Step2 가드(`training_data.py:105-114`)가 mismatch 시 fail-loud. ✅
- **NaN/Inf 처리** — load(nan→0+mask), training filter(non-finite drop), metric(isfinite&~fc_nan). 흐름상 누락 없음. ✅
- **train/val/test leakage 없음** — split 결정적, scaler/PCA는 train sim에만 fit, test는 Step14 1회. ✅ (단 D9의 "PCA를 sim FC에 fit" 자체는 정상; empirical 누수 아님.)
- **simulator shape** — `{param}_matrix (S,360)` shape 불일치 시 `runner_rwweib_2cpl.py:133` ValueError. ✅
- **decoder NaN** — tanh bound + finite assert(`basis_decoder.py:85-88`). NaN/Inf 발생 시 즉시 fail. ✅

---

## 3. corr 천장에 대한 종합 해석

```
baseline(prior-mid) corr ≈ 0.00      ← 모델 무튜닝 = SC와 무관
        │  (engine-routing fix로 0이 "가짜 0"이 아님 확인)
        ▼
현재 posterior corr ≈ 0.13–0.14      ← 실신호 있음
        │  병목: D1(per-subject) + D2(<1% accept) + D3(노이즈평균)
        ▼
저비용 천장 추정 ~0.18–0.22          ← group-avg + expected-FC + 필터/conditioning
        │  (joint_opt 과거 delays+joint ~0.21과 정합)
        ▼
구조 천장(모델 미스펙 D6) ~0.2–0.3?  ← FIC/param 확장 없이는 그 위 어려움
```

→ **즉시 ROI 큰 것**: D1(설정), D3(평가식), D4(필터), D2/D9(conditioning). 이들은
train 과적합 위험 없이 val/test corr을 동시에 올린다. D5/D6은 그다음 단계(ablation 필요).
