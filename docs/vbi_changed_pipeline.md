# 변경된 VBI 파이프라인 정리

## 0. 최종 목표

본 파이프라인의 목표는 임의의 subject에 대해 **SC + Empirical FC**를 입력으로 받아, 해당 subject의 empirical FC를 최대한 재현할 수 있는 **region-wise parameter map**을 추론하는 것이다.

```text
Input:
- SC weight matrix
- tract length matrix
- empirical FC matrix

Output:
- region-wise parameter map Θ
- simulated FC
- FC fitting metrics
```

최종 목표는 다음과 같다.

```text
Given:
SC_s, FC_emp,s

Find:
Θ_s ∈ R^(381×4)

such that:
FC_sim,s = Simulator(SC_s, Θ_s)

Objective:
corr(FC_sim,s, FC_emp,s) → 1
RMSE(FC_sim,s, FC_emp,s) → 0
```

여기서 parameter map은 각 ROI마다 4개 parameter를 가진다.

```text
Θ_s = [
  g_LRE_1 ... g_LRE_381,
  g_FFI_1 ... g_FFI_381,
  I_o_1   ... I_o_381,
  sigma_1 ... sigma_381
]
```

따라서 전체 parameter 수는 다음과 같다.

```text
381 × 4 = 1524
```

cortical-only 360 ROI로 먼저 검증할 경우에는 다음과 같다.

```text
360 × 4 = 1440
```

---

## 1. 기존 파이프라인의 한계

기존 구조는 대체로 다음 형태였다.

```text
θ sample
+ subject SC
→ RWWEIB_2CPL simulation
→ simulated FC

SNPE input  = simulated FC
SNPE target = θ

learns:
q(θ | FC)
```

이 구조에서 SC는 **simulation 생성 과정**에는 사용된다. 그러나 `SC_CONDITION=0`이면 SC는 **추론 모델의 입력**으로 들어가지 않는다.

즉 기존 모델이 배우는 것은 다음과 같다.

```text
q(θ | FC)
```

하지만 최종 목표는 임의 subject의 SC와 empirical FC를 함께 보고 parameter를 추론하는 것이므로, 목표 posterior는 다음이 되어야 한다.

```text
q(Θ | SC, FC)
```

핵심 문제는 다음과 같다.

```text
같은 FC라도 SC가 다르면 필요한 θ가 달라질 수 있다.

예:
강한 SC + 낮은 g_LRE → 특정 FC
약한 SC + 높은 g_LRE → 비슷한 FC
```

FC만 condition으로 사용하면 이 둘을 구분하지 못하고, training SC 분포에 대해 평균화된 흐릿한 posterior를 학습하게 된다.

---

## 2. 변경된 핵심 목표

### 기존 목표

```text
q(θ | FC)
```

### 변경된 목표

```text
q(Θ_381×4 | SC_weight, tract_length, FC_emp)
```

즉 추론 모델은 단순히 FC만 보고 parameter를 예측하는 것이 아니라, subject의 구조 연결성과 functional connectivity를 함께 사용해야 한다.

---

## 3. 변경된 전체 파이프라인

```text
[Input data]
Subject SC weight
Subject tract length
Subject empirical FC

        ↓

[SC/FC preprocessing]
SC weight 정규화
tract length 정리
empirical FC 전처리
sim/empirical FC 전처리 대칭화

        ↓

[Simulation training data 생성]
θ_381×4 샘플링
SC_subject + θ_381×4 → RWWEIB_2CPL simulation
simulated BOLD → simulated FC

        ↓

[Training pair 생성]
x = [SC feature, tract length feature, simulated FC feature]
y = θ_381×4

        ↓

[VBI/SNPE 학습]
qφ(θ_381×4 | SC, FC)

        ↓

[Inference]
SC_new + empirical FC_new 입력
→ posterior qφ(θ | SC_new, FC_emp_new)

        ↓

[Posterior sampling]
θ 후보 여러 개 샘플링

        ↓

[Resimulation + re-ranking]
각 θ로 simulated FC 생성
empirical FC와 corr/RMSE 계산
best θ 선택

        ↓

[Output]
subject-specific 381×4 parameter map
best simulated FC
FC corr / RMSE / posterior uncertainty
```

---

## 4. 기존 파이프라인 vs 변경 파이프라인

| 항목 | 기존 파이프라인 | 변경된 파이프라인 |
|---|---|---|
| 추론 목표 | `q(θ | FC)` | `q(Θ | SC, FC)` |
| 입력 | FC 중심 | SC weight + tract length + FC |
| SC 사용 위치 | simulation 생성에만 사용 | simulation + inference input 모두 사용 |
| 출력 parameter | homogeneous / basis / direct 중 선택 | 최종 목표는 direct region-wise 381×4 |
| parameter dimension | 4, 12, 20, 1440 등 | 1524 또는 cortical-only 1440 |
| FC fitting | posterior mean 재시뮬레이션 | posterior samples 재시뮬레이션 후 best 선택 |
| 목적 함수 | posterior density learning 중심 | posterior inference + FC corr/RMSE re-ranking |
| 일반화 | training SC 분포에 의존 | 임의 SC를 직접 condition으로 사용 |
| 평가 | sim proxy 가능 | empirical posterior predictive corr/RMSE |

---

## 5. 핵심 변경점 1 — SC를 inference input에 포함

기존에는 SC가 simulator에는 들어갔지만, 추론 모델 input에는 빠질 수 있었다.

```text
기존:
θ + SC_subject → simulated FC
SNPE input = simulated FC
SNPE learns q(θ | FC)
```

변경 후에는 다음과 같이 해야 한다.

```text
θ + SC_subject → simulated FC
SNPE input = [SC_subject, tract_length_subject, simulated FC]
SNPE learns q(θ | SC, FC)
```

이렇게 해야 임의 subject의 SC 차이를 반영해서 parameter posterior를 추론할 수 있다.

---

## 6. 핵심 변경점 2 — direct region-wise parameter map

기존 basis 방식은 다음과 같은 구조였다.

```text
12개 coefficient → 381×4 parameter map
```

이 방식은 parameter map을 몇 개의 basis 조합으로 제한한다.

그러나 최종 목표는 각 영역당 4개 parameter를 직접 추론하는 것이다.

```text
direct_regionwise:
θ_dim = 381 × 4 = 1524
```

따라서 최종 목표 모드는 다음에 가깝다.

```text
PARAMETER_MODE=direct_regionwise
ROI_DIM=381
N_PARAMS_PER_ROI=4
THETA_DIM=1524
```

다만 안정적인 개발을 위해서는 1차로 cortical-only 360 ROI에서 검증하는 것이 안전하다.

```text
ROI_DIM=360
THETA_DIM=1440
```

권장 순서:

```text
360 cortical-only smoke test
→ 360 medium run
→ 381 full run
```

---

## 7. 핵심 변경점 3 — posterior mean만 사용하지 않음

기존에는 posterior mean parameter를 사용해 재시뮬레이션했을 가능성이 높다.

```text
posterior mean θ
→ simulation
→ corr 평가
```

변경 후에는 posterior에서 여러 후보를 샘플링하고, 실제 FC fitting 기준으로 선택해야 한다.

```text
posterior samples θ_1 ... θ_K
→ 각각 simulation
→ FC_sim vs FC_emp corr/RMSE 계산
→ best θ 선택
```

최종 저장 항목은 다음과 같이 구성하는 것이 좋다.

```text
posterior mean parameter map
posterior best-corr parameter map
posterior best-RMSE parameter map
posterior uncertainty map
posterior predictive FC distribution
```

---

## 8. 핵심 변경점 4 — forward model 정상화

direct 1524-dim 학습 전에 forward model이 empirical FC와 비슷한 FC를 만들 수 있는 상태인지 먼저 확인해야 한다.

필수 수정/점검 항목:

```text
I_o bound 재설정
FIC 적용 여부 확인
simulated FC std 확인
포화/죽은 simulation 비율 확인
cache key 강화
가짜 val proxy 제거
empirical/sim FC 전처리 대칭화
```

특히 `I_o` prior/bound는 기존처럼 너무 넓게 두면 대부분의 simulation이 포화 또는 죽은 상태에서 생성될 수 있다.

권장 후보:

```text
I_o bound = 0.30 ~ 0.45
```

이 과정을 거치지 않으면 1524개 parameter를 직접 추론해도 학습 데이터 자체가 나빠져 성능이 제한될 수 있다.

---

## 9. 핵심 변경점 5 — cache/evaluation 수정

아래 값이 바뀌면 cache가 반드시 새로 생성되어야 한다.

```text
SC file
SC transform
tract length
USE_DELAYS
I_o bounds
FIC setting
T_END
T_CUT
DT
TR
DECIMATE
sim_seed
parameter mode
ROI count
theta dim
basis/direct mode
train subject ids
```

기존의 sim-vs-sim self-similarity 형태의 `val proxy`는 empirical fitting 성능으로 사용하면 안 된다.

진짜 성능 지표는 다음이어야 한다.

```text
posterior θ → resimulation
→ simulated FC vs empirical FC
→ corr, RMSE
```

---

## 10. 변경된 학습 데이터 구성

각 training sample은 다음과 같이 구성한다.

```text
subject s 선택
SC_s 선택
tract_length_s 선택

θ_s,k ~ prior
θ_s,k shape = 381×4

FC_sim_s,k = Simulator(SC_s, tract_length_s, θ_s,k)

x_s,k = [
    feature(SC_s),
    feature(tract_length_s),
    feature(FC_sim_s,k)
]

y_s,k = θ_s,k
```

SNPE input:

```text
x = SC + length + FC
```

SNPE target:

```text
y = 381×4 parameter map
```

---

## 11. SC/FC feature 구성

SC와 FC를 그대로 upper triangle으로 넣으면 차원이 매우 크다.

381 ROI 기준:

```text
FC upper triangle = 381 × 380 / 2 = 72,390
SC upper triangle = 72,390
length upper triangle = 72,390
```

따라서 초기 구현에서는 PCA 또는 encoder 기반 압축을 사용하는 것이 현실적이다.

권장 1차 구조:

```text
FC feature      = PCA(FC upper triangle)
SC feature      = PCA(log1p SC upper triangle)
Length feature  = PCA(tract length upper triangle or -tract length)

x = concat([FC_feature, SC_feature, Length_feature])
```

초기 구현 예시:

```text
FC_PCA_DIM = 360 or 512
SC_PCA_DIM = 128 or 256
LENGTH_PCA_DIM = 64 or 128
```

장기적으로는 다음 구조도 고려할 수 있다.

```text
Dual encoder:
FC encoder + SC encoder + length encoder

Graph encoder:
SC graph + FC graph를 함께 encoding
```

---

## 12. 변경된 inference 구조

새 subject가 들어오면 다음과 같이 처리한다.

```text
Input:
SC_new
tract_length_new
FC_emp_new

x_new = [
    feature(SC_new),
    feature(tract_length_new),
    feature(FC_emp_new)
]

posterior:
q(θ | x_new)
```

그다음 posterior에서 여러 개의 parameter 후보를 샘플링한다.

```text
θ_1 ... θ_K ~ q(θ | SC_new, FC_emp_new)
```

각 후보에 대해 재시뮬레이션한다.

```text
for each θ_k:
    FC_sim_k = Simulator(SC_new, tract_length_new, θ_k)
    corr_k = corr(FC_sim_k, FC_emp_new)
    rmse_k = RMSE(FC_sim_k, FC_emp_new)
```

최종 선택:

```text
θ_best_corr = argmax corr_k
θ_best_rmse = argmin rmse_k
```

---

## 13. 최종 output 파일

권장 output 구성:

```text
param_maps_mean.npy
param_maps_best_corr.npy
param_maps_best_rmse.npy
posterior_samples.npy
posterior_uncertainty_maps.npy
sim_fc_best_corr.npy
sim_fc_best_rmse.npy
emp_fc.npy
fc_corr.csv
fc_rmse.csv
edgewise_error.npy
networkwise_corr.csv
run_config.json
cache_meta.json
```

---

## 14. 단계별 실행 계획

### Phase 0. 코드 무결성 수정

```text
cache key 강화
val proxy 제거 또는 라벨 변경
held-out empirical FC corr/RMSE metric 추가
config timing 통일
```

### Phase 1. Forward model 정상화

```text
I_o bound = 0.30~0.45
FIC 적용 가능 여부 확인
simulated FC std 확인
포화 simulation 비율 확인
SC/FC 전처리 대칭화
```

### Phase 2. SC-conditioned input 구성

```text
SC weight feature 추가
tract length feature 추가
FC feature와 concat
SC_CONDITION=1
```

추천 input:

```text
x = [
  PCA(FC upper triangle),
  PCA(log1p SC upper triangle),
  PCA(tract length upper triangle)
]
```

### Phase 3. Direct region-wise 학습

```text
PARAMETER_MODE=direct_regionwise
ROI_DIM=360 또는 381
THETA_DIM=1440 또는 1524
SNPE/NSF/MAF 학습
```

권장 순서:

```text
360 cortical-only smoke
→ 360 medium run
→ 381 full run
```

### Phase 4. Posterior re-ranking

```text
N_POSTERIOR_SAMPLES=100~1000
각 sample 재시뮬레이션
corr/RMSE 기준 best 선택
```

### Phase 5. 최종 평가

```text
subject-wise FC corr
subject-wise FC RMSE
network-wise FC corr
edge-wise error map
posterior uncertainty
parameter map visualization
```

---

## 15. 최종 정리

변경된 파이프라인의 핵심은 다음 한 줄로 요약된다.

```text
SC + Empirical FC
→ SC-conditioned VBI/SNPE
→ 381×4 posterior parameter maps
→ posterior samples 재시뮬레이션
→ empirical FC와 corr/RMSE 기준 best map 선택
→ subject-specific digital twin 완성
```

즉 이제 파이프라인은 단순히 다음을 학습하는 것이 아니다.

```text
q(θ | FC)
```

변경된 목표는 다음을 학습하는 것이다.

```text
q(Θ_381×4 | SC_weight, tract_length, FC_emp)
```

최종적으로는 posterior-guided 방식으로 다음 최적화에 가까운 parameter map을 선택한다.

```text
argmax_Θ corr(Simulator(SC, Θ), FC_emp)
```

또는 RMSE 기준으로는 다음과 같다.

```text
argmin_Θ RMSE(Simulator(SC, Θ), FC_emp)
```

이 구조가 임의 subject의 SC와 empirical FC를 입력으로 받아, 그 subject의 empirical FC를 최대한 재현하는 region-wise parameter map을 빠르게 추론하는 VBI 파이프라인이다.
