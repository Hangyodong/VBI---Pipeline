# SC-conditioned amortized NPE — 평가 보고서 + 구현 계획

**Generated:** 2026-06-22
**대상:** `q_φ(θ | x_FC)` → `q_φ(θ | x_FC, z_SC_weight, z_SC_length)` 전환 타당성·계획
**상태:** 평가·계획 전용 — source 미수정(진단 스크립트 외). θ_dim=12·basis decoder 불변.
**근거:** 코드 직접 매핑 + CPU 진단(SC informativeness) + 4-렌즈 적대 검토(audit/science/failure/leak).

Evidence legend: ✅ 타당/확인 · ❌ 안 고침/위험 · ⚠️ 정정 필요 · ➕ 신규.

> 관련: [inference_framework.md](inference_framework.md)(현 추론 루프),
> [performance_diagnosis.md](performance_diagnosis.md)(corr 천장 D1–D12),
> [vbi_paper_vs_code.md](vbi_paper_vs_code.md)(원본 VBI 대비),
> [corr_improvement_plan.md](corr_improvement_plan.md)(in-scope corr 레버 S1–S8).

---

## 0. 검증된 진단 (사실 기반)

CPU 진단, CAB-NP 360-cortical, 100 subjects (`HCP_CABNP381_SC_first100.mat`):

| 지표 | 값 | 함의 |
|---|---|---|
| subject 간 SC corr (log1p) | **0.629** [0.455, 0.770] | 비슷하지 않음 — subject별 다름 |
| subject-PCA 90% 분산 차원 | **68 dim** | 개인차가 고차원·풍부 |
| node strength cross-subject CoV | **0.54** (median 0.52) | ≫0.1 → 강하게 subject-판별적 |
| tract length subject std/mean | **1.585** | 높은 변동 |
| SC support: 전 subject 공통 edge | **1.1%** (≥50%는 11.1%, density 0.144) | topology 자체가 subject마다 다름 |

→ **z_SC는 강한 subject-판별 정보 보유.** 현재 `q(θ|x_FC)`는 SC 혼동 → SC-conditioning이 amortization에
필요. 동시에 **support 1.1%**가 raw-edge SC-PCA를 위험하게 함(§3 R1).

---

## 1. Executive Summary — 제안이 타당한가?

**방향 타당·사실상 필수. v1 feature 설계는 위험. corr 천장은 못 고침.**

| 판정 | 내용 |
|---|---|
| ✅ 과학적으로 옳고 필요 | `q(θ\|x_FC)`는 SC 혼동(marginalized over unknown SC). z_SC는 **known covariate**(추론 대상 아님) → `q(θ\|x_FC,z_SC)`가 통계적으로 옳은 posterior. SC가 강하게 판별적(§0)이라 혼동 실재 |
| ✅ double-counting 아님 | SC-in-simulator = 생성 메커니즘(θ→FC), z_SC = 추론 컨텍스트(어느 맵이 이 FC를 만들었나). 역할 다름 |
| ✅ 여전히 amortized NPE | concat은 MAF context dim만 256→448. 목적·single-round·proposal=prior 불변. test 최적화 없음. θ=12 유지 |
| ✅ digital-twin 목표 정합 | "임의 subject SC+FC → 그 SC에서 FC 맞추는 θ" = per-subject 개인화. per-subject FC 유지가 이 목표엔 옳음 |
| ❌ corr 천장 못 고침 | 이전 CRITICAL 4(per-subject FC OOD·노이즈, 시뮬 manifold 미스펙, whiten-PCA OOD, ~46% NaN 블록) 안 건드림. amortization **품질** 개선이지 corr 레버 아님. 기대 이득 modest |
| ⚠️ v1 feature 위험 | raw-edge SC-PCA(128+64)가 test z_SC OOD 제조 → FC 병을 2채널에 복제 |

**한 줄:** 방향 승인. 단 (a) raw-edge SC-PCA → support-robust 요약 교체, (b) 기존 `EMBED_PER_SUBJECT_SC`
임베딩 활용 검토, (c) distinct 학습 subject 수 대폭↑, (d) corr 레버로 팔지 말 것.

---

## 2. 현재 코드 매핑 (재사용/수정/교체)

| 책임 | 파일:위치 | 현재 동작 | 판정 |
|---|---|---|---|
| SC weight + tract length 로드 | `data_loader_hcp.py:171-206` | CAB-NP 3D → maxnorm SC, length, delay | 재사용(Step 1.5 z_SC 추가) |
| delay 계산 | `simulation/delays.py:28-57` | `length_mm/VELOCITY`(=user 공식, speed=3.0) | 재사용(불변) |
| **delay 시뮬 라우팅** | `simulate_rwweib_2cpl.py:43-71`, `runner_rwweib_2cpl.py:170-203` | `sc_dist=delays, v=1.0` → `RWWEIB_2CPLSimGroup(sc_dist=)` | 재사용 — **rwweib2 delay 지원 확인**(코드 경로 OK; cuBNM 빌드는 GPU서 검증) |
| 엔진 선택 | `engine_select.py` | rwweib2→2cpl, region-wise wrap | 재사용 |
| theta prior/scaling | `priors.py`, `scaling.py` | BoxUniform[-1,1]^12, raw[-2,2] | 재사용(불변) |
| basis decode | `basis_decoder.py`, `param_decoder.py` | mid+half·tanh(Bβ) | 재사용(불변) |
| FC feature | `feature_pipeline.py:88-175` | FC upper-tri→PCA(256, whiten) | 수정(whiten=0 + SC 결합) |
| **학습쌍 조립** | `training_data.py:138-193` | subject loop, (θ_s, x_FC). SC는 sim만, condition엔 없음 | 수정(z_SC append) |
| NPE 학습 | `snpe.py:42-258` | SNPE_C 1-round, MAF, Identity embed | 수정(context dim, NSF 옵션, embed 선택) |
| posterior 샘플 | `posterior.py:36-97` | x_obs=transform(FC) | 수정(z_SC 전달) |
| posterior predictive | `metrics.py:113-205`, `posterior.py:131` | resim FC corr/RMSE + expected-FC(S1) | 수정(x_obs에 z_SC) |
| eval/report | `evaluation/*`, `validation.py`, `final_test.py` | train/val/test 분리 | 재사용(메트릭 확장) |
| **per-subject SC 임베딩(미사용)** | `inference/embedding.py:166-273`, `config.EMBED_PER_SUBJECT_SC` | 학습 SC projection, CPU 검증됨, OFF | **활용 검토** — 제안의 올바른 vehicle |

---

## 3. 주요 위험 + 정정 (직설)

| # | 위험 | 등급 | 정정 |
|---|---|---|---|
| R1 | **raw-edge SC-PCA(128/64)가 test z_SC OOD 제조**. 공통 edge 1.1%뿐 → held-out SC가 PCA null-space로 → FC-OOD를 2채널에 복제 | CRITICAL | raw-edge PCA 금지. **support-robust 요약**: node strength(360), log-total-weight, density, 소수 graph metric → train z-score → 필요시 작은 PCA. 모든 subject에 존재 |
| R2 | **448-dim → Identity+MAF**: 12-dim θ에 과도 conditioning → z_SC 무시/과적합 | HIGH | 기존 `EMBED_PER_SUBJECT_SC`(학습 SC projection, PCA-OOD 없음) 활용 또는 z_SC를 작은 learned embedding 통과(raw concat 금지) |
| R3 | **N_TRAIN=20 distinct SC = 68-dim SC manifold 못 덮음** → flow가 20 SC 암기, held-out 실패 | CRITICAL | θ는 dense(N_SIM/subj)지만 SC는 N_TRAIN번만 샘플 → **distinct subject 수↑**(목표 1040이면 train 수백). N_SIM↑ 아님 |
| R4 | **z_SC ↔ x_FC 중복**: 학습 FC는 SC에서 생성 → 강한 상관 → z_SC 한계정보 작음 | HIGH | **빌드 전 정량화**: train sim에서 z_SC를 x_FC로 회귀(CCA/R²). x_FC가 z_SC 대부분 설명하면 marginal≈0 → 중단 |
| R5 | tract-length raw-PCA: 0-length(없는 edge) vs 짧은 실재 tract 혼동 | MED | length는 present edge만 masked stat(node별 mean/median/percentile) + edge-mask 별도 |
| R6 | SC 채널 whiten 시 OOD 악화 | MED | SC-PCA whiten=0. SC는 sim·empirical 둘 다 real → 본질 in-distribution = 강점, 깨지 말 것 |
| R7 | corr 레버로 오해 | HIGH | de-confound일 뿐. GROUP_AVG_FC·whiten·NaN-mask·misspec 먼저 |

---

## 4. 최소 v1 설계 (안전판)

```
x_FC  = FC upper-tri → (Fisher-z ablation) → PCA(256, whiten=0) → standardize
z_SCw = [node_strength(360), log_total_weight, density, clustering_mean] → z(train)
z_SCl = present-edge length stats [node_median_len(360), global mean/std, delay_max] → z(train)
condition = concat(x_FC, z_SCw, z_SCl)        # dim은 요약 설계로 조정 (raw-edge PCA 아님)
```
- raw-edge PCA(128/64) 폐기 → support-robust 요약(R1/R5).
- z_SC는 learned embedding 통과 권장(R2): 우선순위 = `EMBED_PER_SUBJECT_SC` 재활성 > 작은 MLP > raw concat(최후).
- 모든 SC scaler/PCA train-only fit, `ScConditioner.fit/transform`(FeaturePipeline fitted-guard 모방).
- whiten=0 전 채널(R6).

> **대안(권장 1순위):** concat-PCA 대신 이미 구현·CPU검증된 `inference/embedding.py` per_subject_sc 사용 →
> PCA-OOD(R1) 전부 회피. 단 producer/eval 배선(`main_HCP.py:164-168` 체크리스트) + GPU 검증 필요.

---

## 5. 권장 구현 계획 (staged)

| Stage | 내용 | 수정 파일 | 신규 | 위험 | 테스트 | 산출물 |
|---|---|---|---|---|---|---|
| **0** | baseline 확정 + **z_SC↔x_FC 중복 정량화(R4)** + SC info(완료) | — | `diagnostics/sc_info.py` | 낮음 | CCA/R² | 중복 수치(go/no-go) |
| **1** | 로더에 z_SC 슬롯(Step 1.5) | `data_loader_hcp.py`, `main_HCP.py` | — | 낮음 | shape/finite | `subject_data[sid]["z_sc"]` |
| **2** | `ScConditioner`(support-robust 요약, train-fit/transform, whiten=0) | `feature_pipeline.py` | `inference/sc_conditioner.py` | R1/R5 | train-only fit assert, val 재현성 | frozen conditioner |
| **3** | delay subject별(동작 중) — NaN/inf/0/clip 하드닝 | `simulation/delays.py` | — | 낮음 | finite/clip 단위 | delay sanity |
| **4** | condition=concat(x_FC,z_SC) NPE — embed 선택(per_subject_sc 우선), NSF 옵션 | `snpe.py`, `training_data.py`, `posterior.py`, `metrics.py` | — | R2/R3 | N_TRAIN=5 파일럿 | 학습 posterior |
| **5** | no-leak: SC-PCA train-only, val/test transform-only, cache 동봉 | `main_HCP.py`, `feature_pipeline.py` | — | 누수 | fit-ID⊆train assert | 무누수 보증 |
| **6** | posterior-predictive eval + prior baseline + SC-permutation control | `evaluation/*`, `final_test.py` | — | gaming | 메트릭 단위 | 비교 리포트 |
| **7** | ablation A/B/C/D + 선택 NSF | config | `run_ablation.sh` | parity | frozen artifact 공유 | ablation 표 |

**시퀀싱:** corr 자체는 SC-conditioning이 아니라 whiten=0(S4 완료)·NaN-mask·expected-FC가 올린다.
SC-conditioning은 amortization 정확성 별도 트랙. Stage 0에서 R4가 "x_FC가 z_SC 대부분 설명"이면 **중단**.

---

## 6. 수정/추가 파일 요약

**신규:** `inference/sc_conditioner.py`(ScConditioner), `diagnostics/sc_info.py`(중복·OOD 진단), `run_ablation.sh`.
**수정:** `feature_pipeline.py`(SC 결합, whiten 기본 0), `training_data.py`(z_SC append), `snpe.py`(context
dim·NSF·embed 선택), `posterior.py`/`metrics.py`(x_obs에 z_SC), `main_HCP.py`(Step 1.5 + train-only fit +
config), `delays.py`(하드닝).
**불변:** `basis_decoder.py`, `param_decoder.py`, `engine_select.py`, `simulate_rwweib_2cpl.py`,
`runner_rwweib_2cpl.py`, prior/scaler.

---

## 7. 검증 + ablation 계획

**메트릭(사용자 안 + 강화):**
- 주 메트릭: posterior-predictive **expected-FC corr**(행렬평균 후 1회) + mean/median + **prior-predictive
  baseline 대비 Δ**. best-draw는 **주 메트릭 금지**(보조만). RMSE 동반.
- 진단 triple: **(shrinkage, boundary-mass fraction, rejection accept-rate)** 동시 — shrinkage↑ & boundary↓
  & accept↑만 유효(clip fallback이 가짜 shrinkage 만듦).
- OOD distance: FC·SCw·SCl 각각 별도, **train-only 통계**로(누수 금지).

**ablation A/B/C/D:** FC-only / +SCw / +SCl / both. **동일 frozen (θ, fc_raw, FC-PCA, split)** 공유, SC 블록만
변경(parity). train subject-ID hash 일치 assert.

**결정적 검정(gaming 방지):**
- **paired per-subject Δ = corr(D)−corr(A)** on held-out TEST, **GROUP_AVG_FC=0**, bootstrap **subject-level**.
- **SC-permutation control**: z_SC를 test subject 간 셔플 → 성능 안 떨어지면 flow가 SC 무시/암기 = 실패.
- N_TRAIN 의존성: distinct subject 수 늘리며 held-out corr 곡선(암기 R3 점검).

---

## 8. 결정 필요 (Open Questions)

1. **z_SC vehicle**: (a) 기존 `EMBED_PER_SUBJECT_SC` 재활성(권장) vs (b) 새 `ScConditioner` 요약+concat vs
   (c) 둘 다(ablation)? — R1/R2 갈림.
2. **distinct train subject 수**: 현재 20 → 암기(R3). 목표 1040 대비 몇 명?(200/500/700) GPU 예산 직결.
3. **Stage 0 게이트**: z_SC↔x_FC 중복(R4) 높으면 SC-conditioning 보류?
4. **per-subject FC 유지 확정?** digital-twin엔 맞으나 corr 천장 ~0.05–0.13. SC-conditioning이 못 올림 — 수용?
5. **시퀀싱**: corr in-scope 수정(whiten=0·NaN-mask·expected-FC) 먼저 vs SC-conditioning 평행?
6. **delay GPU 검증**: rwweib2 cuBNM 빌드 sc_dist 지원 코드상 OK, GPU 미검증. delays ON(9× 비용) 확정?

---

## 9. 핵심 결론

- **SC-conditioning은 옳고 필요**(de-confound) — 단 **corr 천장은 못 올린다**. 두 목표를 분리하라:
  amortization 정확성(SC-conditioning) vs corr 향상(whiten=0·NaN-mask·group-avg·misspec).
- **v1 raw-edge SC-PCA는 하지 마라**(R1 OOD 제조). support-robust 요약 + 기존 학습 임베딩.
- **distinct train subject 수가 진짜 병목**(R3) — 20으로는 amortization 일반화 불가.
- **빌드 전 R4(중복) 정량화** = 가장 싼 go/no-go.
