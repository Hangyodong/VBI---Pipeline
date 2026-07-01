# Corr Improvement Plan — 작은 단위 수정 순서

**Generated:** 2026-06-22
**원칙:** train corr만 올리는 hard-coding 금지. **val/test에서 유지되는 robust 향상**만.
평가식을 느슨하게 바꿔 corr이 높아 보이게 하지 않는다. 한 번에 하나씩, 검증 후 다음.
**상태:** 계획 전용 — source code 미수정. "수정 시작" 지시 전까지 코드 변경 없음.

기반: [performance_diagnosis.md](performance_diagnosis.md) D1–D12.

---

> **상태(2026-06-22):** 사용자 결정 = per-subject 유지 + 범위 eval+feature+bounds.
> **S1·S3·S4·S5·S6 구현 완료**(env 기본값=현행 보존), CPU 컴파일·유닛·스모크 PASS,
> 적대적 리뷰 workflow `wr37ifmfw` 진행. GPU 검증·ablation 대기. S7(model)은 범위 외(미수행).
> 구현 내역·실행 명령 = [experiment_log.md](experiment_log.md) §0.

## 0. 수정 우선순위 (저비용·저위험·고ROI 순)

| Step | 대상 | 근거(D) | 위험 | 예상 corr |
|---|---|---|---|---|
| **S1** | 평가: expected-FC(행렬 평균) + MAP/mean resim 비교 | D3,D12 | 낮음 | +0.01~0.03 |
| **S2** | config: GROUP_AVG_FC=1로 천장 확인 | D1 | 낮음 | +0.03~0.06 |
| **S3** | feature: Step4에 step5 sim-품질 필터 적용 | D4 | 낮음 | +0.00~0.02(안정성) |
| **S4** | conditioning: whiten=False/x표준화 → accept율 ↑ | D2,D9 | 중간 | accept↑, corr +0.01~0.03 |
| **S5** | bounds: g_LRE/g_FFI (0,3)→(0,6) ablation | D5 | 중간 | +0.00~0.03 |
| **S6** | reproducibility: CUDA/cudnn seed 고정 | D10 | 낮음 | (재현성) |
| **S7** | sim: 분석창/노이즈평균 또는 model param 확장 | D6,D7 | 중-높음 | 천장 상향 |
| **S8** | alignment: FC/SC/basis ROI 순서 검증 | D8 | 낮음(검증) | (정합성 보증) |

> S1–S3는 **평가/feature/설정**만 건드려 train 과적합 위험이 구조적으로 없다(같은 변경이
> train/val/test에 동일 적용). 먼저 여기서 ceiling을 재측정한 뒤 S4+로 진행.

---

## 1. 단계별 수정 계획표

| Step | Target files | Current behavior | Problem | Proposed change | Expected on corr | Risk | Validation |
|---|---|---|---|---|---|---|---|
| **S1** | `evaluation/metrics.py::_resimulate_and_score`, `evaluate_subject`; `inference/posterior.py` | n_resim draw별 single-sim FC corr 평균 | 단일 sim FC 노이즈로 corr 저평가 | (a) resim FC들을 **행렬 평균** 후 1회 corr 추가 산출, (b) posterior-mean·MAP theta를 별도 resim해 corr 저장. 기존 draw평균은 유지(둘 다 리포트) | 노이즈 상쇄로 +0.01~0.03, val/test 동일 | 낮음 — 더 느슨한 metric 아님. 새 컬럼만 추가 | 같은 posterior에서 expected-FC corr > draw평균 corr; val·test 동시 상승 |
| **S2** | 실행 설정(`GROUP_AVG_FC=1` env) | 기본 per-subject FC | 천장 ~0.05로 낮음 | env로 group-avg 실행(코드 무수정). CLAUDE.md active와 일치시킴 | +0.03~0.06 (천장 ~0.2로) | 낮음 — 동일 타깃 all-subj, leak 아님 | group vs subject 동일 posterior pipeline corr 비교 |
| **S3** | `main_HCP.py:742-746` | `FeaturePipeline()` 직접 fit_transform, 필터 없음 | dead/saturated FC가 whiten PCA 오염 | Step4를 `inference.step5_fit_feature_pipeline` 경로로 교체(또는 동일 mask 적용 후 fit) | 안정성 ↑, accept↑로 +0~0.02 | 낮음 — 유효샘플만. x_obs 차원 동일 유지 필수 | 필터 전/후 PCA EVR·accept·corr; FC_DIM 불변 확인 |
| **S4** | `inference/feature_pipeline.py:98` | `PCA(whiten=True)` sim FC에만 fit | empirical FC OOD → accept <1% | whiten=False 또는 PCA 전 per-feature 표준화(train sim 기준). empirical은 동일 변환 | accept 1%→>30%, corr +0.01~0.03 | 중간 — 표현력/조건수 변화, 재학습 필요 | accept율, x_obs percentile vs x_train, corr |
| **S5** | `main_HCP.py:146` (`BASIS_BOUNDS`) | g_LRE/g_FFI (0,3) | posterior 수요(~7)보다 좁아 under-couple 가능 | (0,6)으로 확장 후 ablation. sigma/I_o는 유지 | +0~0.03 (최적이 3 초과였다면) | 중간 — 과결합 시 dynamic explosion. tanh가 bound 유지하므로 decode NaN은 없음; sim 단계 non-finite 가드로 차단 | (0,3) vs (0,6): corr·accept·non-finite drop율 |
| **S6** | `pipeline_setup.py:330-337` | np+torch.manual_seed만 | CUDA 비결정 → posterior 재현 불가 | `torch.cuda.manual_seed_all(SEED)`, `cudnn.deterministic=True, benchmark=False` 추가(또는 seed 저장) | 재현성(목표 #6) | 낮음 — 약간 느려질 수 있음 | 동일 seed 2회 실행 corr |Δ| 측정 |
| **S7a** | `main_HCP.py:74-75` | T_END=180s | 분석창 120s 노이즈 | (S1로 대체 가능) 필요 시 T_END↑. 우선 S1 노이즈평균이 저비용 | 안정성 | sim 비용 ↑ | 120s vs 300s simFC 안정성 |
| **S7b** | `main_HCP.py:119` `RWWEIB2_FIXED`, `cuBNM/rww_eib_2cpl.yaml` | J_N/w_p/λ_IE 고정, FIC 없음 | simFC std 천장(미스펙) | (생물학 비우선) 민감 param을 추론셋 추가 또는 light-FIC. **non-finite/explosion 가드 필수** | 천장 상향(불확실) | 높음 — regime 불안정 | forward_ablation/sweep_fc로 민감도·안정성 선검증 |
| **S8** | `data_loader_hcp.py`, `basis_decoder.py`, label 파일 | 세 소스 독립 [:360] slice | ROI 순서 동일성 미검증 | 세 소스 region label 1:1 assert 추가(런타임 검증) | 정합성 보증(어긋났다면 corr 회복) | 낮음 | label 교차표; 셔플 basis 음성통제 |

---

## 2. 실험 설계 (각 Step의 검증 절차)

모든 수정은 다음 순서로 검증한다. **destructive 명령·기존 output 삭제 금지**.

1. **import check** — `python -c "import main_HCP"` 수준은 GPU 의존이라 불가. 대신
   `python -m pytest test_basis_mode_smoke.py -q` (7 passed), `python test_basis_decoder.py`.
2. **config consistency** — theta_dim=12, STAGE1_PARAMS len, bounds 일치(Step7 가드 자동).
3. **shape check** — `tests/smoke/verify_basis_regionwise_rwweib2.py` (CPU fallback, finite=True).
4. **one-subject dry run** — 작은 N_SUBJECTS=4/N_TRAIN=2/N_SIM=64 (SMOKE=1) GPU 노드.
5. **small simulation smoke** — SMOKE run end-to-end 크래시/shape 확인.
6. **corr 계산** — empirical vs sim FC, upper-tri, NaN 제거(기존 metric 그대로).
7. **train/val/test corr 분리 저장** — Step9(val)·Step14(test) + per-subject `fc_corr_all`.
8. **FC RMSE 계산** — 기존 `fc_metrics`.
9. **output 경로 확인** — `output_hcp/` (mouse dir 누수 S3 부산물로 같이 점검).

### ablation 목록
- A1: GROUP_AVG_FC ∈ {0,1} (S2)
- A2: eval ∈ {draw평균, expected-FC, MAP} (S1)
- A3: PCA ∈ {whiten, no-whiten, zscore} (S4)
- A4: g bound ∈ {(0,3),(0,6)} (S5)
- A5: filter ∈ {off, on} (S3)
- A6: delays ∈ {off, on} (D11 정의 확정용)

각 ablation은 **동일 seed·split**에서 val/test corr을 비교. train corr 단독 상승은
무시(과적합 신호). val·test 동시 상승만 채택.

---

## 3. 가드레일 (수정 시 반드시 지킬 것)

- decoder 식(`mid+half*tanh(basis@beta)`), coeff order는 **변경 금지**(CLAUDE.md). bound은
  S5에서만, BASIS_BOUNDS 값 자체만 조정.
- 모델 식/`rww_eib_2cpl.yaml` coupling은 S7b 외 불변. S7b도 가드/ablation 선행.
- 평가 metric을 느슨하게(예: |corr|, RMSE 정규화 완화) 바꾸지 않는다. S1은 **노이즈 제거**
  (합법)지 metric 완화가 아니다.
- hard-coded path 신규 금지. 기존 output 삭제·overwrite 금지.
- 각 수정 후 docs(특히 [experiment_log.md](experiment_log.md)) 동시 업데이트.

---

## 4. 권장 진행 순서 (한 줄 요약)

```
S1(expected-FC) → S2(group-avg) → S3(filter) → [재측정: 천장 ~0.2 확인]
   → S4(conditioning) → S5(bounds) → S6(repro) → [S8 alignment 검증]
   → S7(model/sim) 은 위 천장이 막힐 때만, ablation 선행
```

S1–S3만으로 0.13 → ~0.18–0.20을 먼저 노린다(저위험). 그 위는 S4–S7의 단계적 확장.
