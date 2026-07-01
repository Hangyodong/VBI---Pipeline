# Experiment Log — HCP basis-regionwise corr 향상

**Generated:** 2026-06-22
**목적:** corr 향상 수정([corr_improvement_plan.md](corr_improvement_plan.md) S1–S8)의
실험 결과 기록. 아직 수정 미수행 → 신규 행은 `TBD`.

규칙: 동일 seed(42)·split에서 측정. **val·test 동시 상승만 채택**(train 단독 상승=과적합 신호).
Train corr는 현 파이프라인이 직접 산출 안 함(필요 시 S1 expected-FC를 train subset에 적용해 별도 측정).

Corr = empirical vs simulated FC upper-tri Pearson(NaN 제외). 측정식은 `evaluation/metrics.py::fc_metrics`.

**사용자 결정(2026-06-22):** FC 타깃 = **per-subject 유지**(GROUP_AVG_FC=0) · 수정 범위 =
eval+feature+bounds(모델 식 불변) · 구현 = inline + workflow 검증 · 규모 = SMOKE→N_TRAIN=20.

---

## 0. 구현 완료 (2026-06-22) — GPU 검증 대기

모든 변경은 **env 기본값=현행 동작 보존**. 실험은 env var로 구동. 코드 컴파일·CPU 유닛/스모크
PASS(`test_basis_mode_smoke.py` 7 passed, `test_basis_decoder.py` ALL PASS), expected-FC
denoising CPU 검증(per-draw 0.55 → expected 0.90, 합성 데이터). 적대적 리뷰 workflow `wr37ifmfw` 진행.

| ID | 구현 | 파일 | env 노브(기본=현행) |
|---|---|---|---|
| S1 | expected-FC(resim FC 행렬평균 후 1회 score), per-draw와 **병기**(완화 아님) | `evaluation/{metrics,validation,final_test,reports}.py` | (항상 ON, 추가 리포트) |
| S3 | Step4 PCA 전 degenerate FC 필터(theta/fc/fcd 동일 mask) | `main_HCP.py` Step 4 | (항상 ON) |
| S4 | PCA whiten 토글 | `inference/feature_pipeline.py`, `main_HCP.py` | `FC_PCA_WHITEN`(기본 1) |
| S5 | g_LRE/g_FFI 상한 토글 | `main_HCP.py` BASIS_BOUNDS | `G_BOUND_HIGH`(기본 3.0) |
| S6 | CUDA seed + 선택적 deterministic | `pipeline_setup.py`, `inference/snpe.py` | `DETERMINISTIC`(기본 0) |

### GPU 실행 명령 (사용자, GPU 노드)

```bash
# (1) SMOKE — 배선/shape/크래시 확인 (수 분)
SMOKE=1 PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 python main_HCP.py

# (2) baseline 재현 + expected-FC 가시화 (N_TRAIN=20, per-subject, 현행 동작)
SMOKE=0 N_SUBJECTS=50 N_TRAIN=20 N_VAL=10 N_TEST=20 N_SIM=2000 \
  PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 python main_HCP.py

# (3) ablation: conditioning(whiten off)  — A3
#     위 (2)에 FC_PCA_WHITEN=0 추가
# (4) ablation: coupling bound 확장        — A4
#     위 (2)에 G_BOUND_HIGH=6.0 추가
# (5) best-guess 조합
#     FC_PCA_WHITEN=0 G_BOUND_HIGH=6.0  (+ 재현성 원하면 DETERMINISTIC=1)
```

> **캐시 동작:** `output_hcp/features_stage1.npz` 무효화 키 `_cache_meta_now`
> (`main_HCP.py`)는 이제 **`hetero_bounds`·`use_delays`도 담는다**(workflow `wr37ifmfw`
> HIGH finding 수정). 따라서:
> - **A3(whiten)·A5(filter)**: 시뮬 동일, feature/eval만 변경 → features 자동 재사용 OK.
> - **A4(G_BOUND_HIGH)**: bound 변경 → 캐시 자동 **STALE 판정 → 재시뮬**(수동 삭제 불요).
> - **delays on/off**: 마찬가지로 자동 무효화.
> 기존(키 추가 전) sidecar는 새 키가 없어 불일치 → 첫 실행 시 1회 재시뮬(안전 방향).

---

## 1. 결과 표

| Experiment ID | Change | Train corr | Validation corr | Test corr | FC RMSE | Notes |
|---|---|---:|---:|---:|---:|---|
| E0 (baseline 기록) | 현행 코드, N_TRAIN=20 N_SIM=2000, per-subject FC, delays(로그상) | n/a | 0.1381 | 0.1342 | 0.2151 | `run_basis_t20_s2000.log`. accept 0.4–1%, baseline corr≈0, per-subj 0.04–0.21. param_maps g≈7(wide bounds 시절) |
| E0b (prior-mid baseline) | theta=0 = bound midpoint resim | n/a | n/a | ~0.002–0.008 | ~0.19–0.25 | `metrics.baseline_eval`. 모델 무튜닝 천장(SC 무관) |
| **E1 (SC-conditioned)** | SC_CONDITION=1 ch=[sc_weight,sc_mask], N_TRAIN=80 N_SIM=1000, per-subject FC, delays OFF | n/a | 0.1154 / **0.1295(exp)** | 0.1392 / **0.1571(exp)** | 0.2326 | matrix-encoder NPE. **식별성 대폭↑**(shrinkage 0.6–0.9, probing R² ~0.93). accept 여전히 <1%(OOD). E0(FC-only N20 0.134)와 비슷 → SC-cond은 식별성↑이지 corr 천장 못 뚫음 |
| **CEIL (achievable)** | best-sim-FC corr (random θ 3000개 중 subject별 max) | — | — | **0.196** (max 0.256, min 0.136) | — | **천장 진단**: 시뮬이 낼 수 있는 최선 FC가 empirical과 ~0.20 닮음. E1 exp 0.157 = **천장의 ~80%**. OOD z_rms 3.81, corr(emp평균,sim평균)=0.008(구조 어긋남). → 추론은 거의 최적, **천장 자체가 model-level 한계** |
| S1 expected-FC | resim FC 행렬평균 + MAP/mean resim 비교 | TBD | TBD | TBD | TBD | 노이즈 상쇄. metric 완화 아님 |
| S2 group-avg FC | GROUP_AVG_FC=1 | TBD | TBD | TBD | TBD | 천장 ~0.2 기대 |
| S3 sim-품질 필터 | Step4에 step5 mask 적용 | TBD | TBD | TBD | TBD | PCA 오염 제거 |
| S4 conditioning | PCA whiten=False/x표준화 | TBD | TBD | TBD | TBD | accept 1%→>30% 목표 |
| S5 coupling bounds | g_LRE/g_FFI (0,3)→(0,6) | TBD | TBD | TBD | TBD | under-couple 검증 |
| S6 reproducibility | CUDA/cudnn seed 고정 | TBD | TBD | TBD | TBD | 동일 seed 2회 |Δcorr| |
| S7 sim/model | 분석창↑ 또는 param 확장/FIC | TBD | TBD | TBD | TBD | 천장 막힐 때만, ablation 선행 |
| S8 ROI alignment | FC/SC/basis 순서 assert | TBD | TBD | TBD | TBD | 정합성 보증 |

---

## 2. Ablation 표 (계획)

| Ablation | 변형 | 측정 지표 | 채택 기준 |
|---|---|---|---|
| A1 | GROUP_AVG_FC ∈ {0,1} | val·test corr | group이 동시 상승 |
| A2 | eval ∈ {draw평균, expected-FC, MAP} | corr | expected-FC ≥ draw평균 |
| A3 | PCA ∈ {whiten, no-whiten, zscore} | accept율, corr | accept↑ & corr↑ |
| A4 | g bound ∈ {(0,3),(0,6)} | corr, non-finite drop율 | corr↑ & drop율 불변 |
| A5 | filter ∈ {off, on} | PCA EVR, accept, corr | 안정성·corr 비퇴행 |
| A6 | delays ∈ {off, on} | corr, 비용 | 정의 확정(D11) |

---

## 3. 측정 메모

- **accept율**: `run_*.log`의 "Only X% proposal samples are accepted" 라인 추적. <30%면 D2 미해결.
- **per-subject 편차**: `evaluate_subject`의 `fc_corr_all`/subject. 편차 큼 = per-subject FC 노이즈(D1).
- **expected-FC corr 계산법**: 한 subject의 n_resim resim FC 행렬을 평균 → upper-tri → empirical과 1회 corr. draw별 평균 corr과 **별도 컬럼**으로 비교(둘 다 기록).
- **비교 공정성**: 모든 행은 동일 N_SUBJECTS/N_TRAIN/N_SIM/SEED에서. 사이즈 바뀌면 Notes에 명기.

---

## 4. 환경 캐비엇

- Claude scratch 셸엔 GPU 없음 → full run은 GPU 노드에서. CPU에선 smoke/unit만.
- `output_hcp/features_stage1.npz` 캐시는 config 메타 불일치 시 자동 무효화
  (`main_HCP.py:_cache_meta_now`). 모드/타깃/bound 바꾸면 재시뮬됨 — 캐시 stale 오인 주의.
- 기존 output 파일 삭제 금지(목표 #4 재현 비교용).
