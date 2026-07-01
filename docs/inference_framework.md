# Inference Framework — HCP basis-regionwise (NPE / single-round amortized SBI)

**Generated:** 2026-06-22
**대상:** `main_HCP.py` · `INFERENCE_MODEL=rwweib2` · `PARAMETER_MODE=basis_regionwise`
**범위:** 추론(inference) 과정 전체 — prior → forward sim → density estimator → posterior → 평가.

Evidence legend: ✅ 코드 확인(file:line) · ⚠️ 위험/제약 · ➕ 신규(S1).

> 관련: [pipeline_analysis.md](pipeline_analysis.md)(전체 데이터 흐름),
> [performance_diagnosis.md](performance_diagnosis.md)(corr 진단),
> [vbi_paper_vs_code.md](vbi_paper_vs_code.md)(원본 VBI 대비).

---

## 0. 한 줄 정의

`p(θ_basis | x_FC)` 추정. **θ = 12차원 basis coefficient**, **x = empirical FC에서 뽑은
feature**. 시뮬레이터로 만든 `(θ, x)` 쌍으로 **normalizing flow(MAF)** 학습 → 새 empirical FC에
대해 posterior를 amortized로 즉시 추론.

```
prior θ_s ──scaler⁻¹──► θ_raw(coeff) ──decode──► param maps ──cuBNM──► BOLD ──► FC ──► x
   │                                                                               │
   └──────────────────── (θ_s, x) 쌍 ──────────► MAF q_φ(θ_s|x) 학습 ◄────────────┘
                                                        │
empirical FC ──feature pipeline──► x_obs ──► q_φ(θ_s|x_obs) ──► θ_s 샘플 ──scaler⁻¹──► θ_raw
```

---

## 1. 두 공간 (핵심 — 절대 섞지 말 것)

| 공간 | 범위 | 용도 | 다리 |
|---|---|---|---|
| **scaled** θ_s | `[-1,1]^12` | SBI/MAF 학습·샘플 | `BoxUniform[-1,1]^12` (`inference/priors.py:34`) |
| **raw** θ_raw (coeff) | `[-2,2]^12` | decode 입력 | `ParameterScaler` (`inference/scaling.py:40-48`) |

변환: `θ_raw = (θ_s+1)/2·(hi-lo)+lo`, `θ_s = 2(θ_raw-lo)/(hi-lo)-1`. lo=-2, hi=+2
(`BASIS_COEFF_PRIOR`, `main_HCP.py`). **같은 scaler 인스턴스**가 학습(forward)·추론(backward) 양쪽
담당 (`scaling.py` docstring 경고). ⚠️ 섞으면 posterior가 조용히 mis-align.

---

## 2. θ → 시뮬레이터 (forward path)

### 2.1 prior 샘플
`θ_s ~ prior_scaled.sample((N_sim,))` → `θ_raw = scaler.inverse_transform(θ_s)`
(`training_data.py:147-151`).

### 2.2 decode (12 coeff → per-ROI param maps)
`engine_select.latent_wrap`가 시뮬 직전 decode (`engine_select.py:102-117`).
param k(∈ g_LRE, g_FFI, I_o, sigma)마다 β_k = θ_raw의 해당 3계수 `[const, myelin, gradient]`:

```
z_k   = B @ β_k                    # B=(360,3) basis,  z_k=(360,)
map_k = mid_k + half_k · tanh(z_k) # mid=(lo+hi)/2, half=(hi-lo)/2
```
(`basis_decoder.py:79-86`) → `{g_LRE,g_FFI,I_o,sigma}_matrix (S,360)`. tanh가 bound 보장 →
clip 불요·NaN 없음. basis col0=const(=1.0), col1=myelin, col2=gradient (rezscore 표준화).

### 2.3 시뮬레이션
`{param}_matrix` → `build_param_lists` per-node 주입 → `cuBNM RWWEIB_2CPLSimGroup(force_gpu,
hrf="bw")` → BOLD `(~166,360)` (`runner_rwweib_2cpl.py:181-211`). theta는 scalar param으로 **안**
넘김(coeff 무시 방지).

### 2.4 feature 추출 → x
BOLD → `compute_fc`(raw Pearson r) → upper-tri(64620) → `FeaturePipeline`: PCA(256, whiten) → **x**
(`feature_pipeline.py:92-100,156-157`). 학습쌍 `(θ_s, x)` 정렬 저장. **θ_s가 SBI 타깃, θ_raw는
시뮬 입력** (`training_data.py:19-27`).

---

## 3. Density estimator 학습 (NPE core)

### 3.1 목적
flow `q_φ(θ_s | x)`로 posterior 근사. 목적함수(SNPE_C, single round = NPE):

```
max_φ  E_{(θ_s,x)~prior×sim} [ log q_φ(θ_s | x) ]
```
prior에서 뽑았으므로 proposal=prior → 보정항 없음(순수 NPE).

### 3.2 아키텍처
| 항목 | 값 | 근거 |
|---|---|---|
| 알고리즘 | `sbi.inference.SNPE_C`, **단일 라운드** | `snpe.py:56,129` |
| density estimator | **MAF** (`posterior_nn(model="maf")`) | `snpe.py:123`, `config NDE_MODEL` |
| hidden / transforms | 128 / 8 | `config NDE_HIDDEN/TRANSFORMS` |
| embedding | `nn.Identity` (RegionTransformer OFF) | `snpe.py:106-107`, `main_HCP.py` |
| 입력 x | PCA-FC 256dim (raw FC 아님) | `main_HCP.py` Step4 |
| 학습 | batch 512, ≤200ep, early-stop off | `snpe.py:153-155` |
| device | x는 CPU 유지(46GB OOM 방지), minibatch만 GPU | `snpe.py:147-150` |

### 3.3 single-round amortization
`append_simulations(θ_s, x)` → `train()` → `build_posterior()` **1회** (`snpe.py:148-258`). posterior는
**임의의 x_obs에 재사용** = amortized. (논문 VBI와 동일 철학; Phase2/4 sequential-유사 단계는
`RUN_PHASE24=False`로 OFF.)

---

## 4. Observation 추론 (backward path)

`infer_subject_raw` (`posterior.py:45-97`):

1. empirical FC → `feature_pipeline.transform` → **x_obs** (학습과 **동일** PCA basis 필수,
   `posterior.py:149`).
2. 샘플:
   ```
   θ_s ~ q_φ(· | x_obs)          # rejection (prior-bounded, 60s cap)
   실패(accept<임계) → reject_outside_prior=False 후 clip[-1,1]   # OOD fallback
   ```
   (`posterior.py:63-86`) — ⚠️ t20 run에서 accept **0.4–1%** = empirical FC가 sim 분포 밖(OOD).
3. `θ_raw = scaler.inverse_transform(θ_s)` → mean/std 산출.

> 추론은 **scaled 공간에서 샘플 → raw로 역변환**. posterior mean/MAP/top-k는 현재 미사용(샘플만).

---

## 5. Posterior 출력 + 검증 (evaluation)

### 5.1 posterior predictive (resim)
`_resimulate_and_score` (`metrics.py:208-272`): posterior 샘플 n_resim개 각각 decode→재시뮬→FC. 두
점수:
- **per-draw**: 각 draw FC corr → 평균 (기존)
- ➕ **expected-FC** (S1): resim FC **행렬 평균 후 1회 score** (노이즈 제거, 같은 `fc_metrics`)

`fc_metrics`: upper-tri(k=1), NaN mask(`~fc_nan`), Pearson + RMSE (`metrics.py:47-72`).

### 5.2 diagnostics
| 진단 | 정의 | 위치 |
|---|---|---|
| shrinkage | `1 - σ_post/σ_prior` (param별 식별성) | `posterior.py:104-117` |
| SBC | prior→sim→rank uniformity(보정 점검) | `diagnostics.simulation_based_calibration` |
| sensitivity | posterior gradient ActiveSubspace eigen | `active_sensitivity.py` |
| ppc | empirical vs predicted FC/FCD corr·RMSE | `posterior.posterior_predictive_check` |
| z-score | 합성데이터 한정 `(θ̂-θ*)/σ_post` — ⚠️ **현재 미구현** | — |

### 5.3 train/val/test 분리
- val(Step9): model selection용 (`validation.py`)
- test(Step14): held-out, **1회만** (`final_test.py`) — bootstrap CI

---

## 6. 수식 한 장 요약

```
θ_s ~ U[-1,1]^12                                   (prior, scaled)
θ_raw = scaler⁻¹(θ_s) ∈ [-2,2]^12                  (coeff)
β_k = θ_raw[3k:3k+3]                               (param k의 const/myelin/grad)
map_k = mid_k + half_k·tanh(B β_k),  B=(360,3)     (per-ROI param, bounded)
BOLD = cuBNM_RWWEIB2(map, SC, delays)              (forward sim, hrf=bw)
x = PCA₂₅₆(triu(corr(BOLD)))                       (feature)
─────────────────────────────────────────────
학습:  max_φ E[log q_φ(θ_s | x)]                   (MAF, single-round NPE)
추론:  θ_s ~ q_φ(·| x_obs);  θ_raw = scaler⁻¹(θ_s) (amortized)
평가:  corr(empFC, mean_i FC(decode(θ_raw^(i))))   (expected-FC)
```

---

## 7. 핵심 불변식 (refactor 시 깨면 안 됨)

1. **θ_dim = 12**, coeff order 고정 (`g_LRE,g_FFI,I_o,sigma` × `const,myelin,gradient`).
2. 학습·추론 **같은 scaler·같은 feature pipeline**(PCA basis).
3. x_sim·x_obs **동일 공간** — ⚠️ 그래서 시계열계 feature(FCD/temporal/spectral)는 **empirical
   BOLD 없으면 conditioning 불가**(HCP empirical = FC matrix only).
4. decode 식 `mid+half·tanh(Bβ)` 불변.
5. single-round(`NUM_ROUNDS=1`) = amortization.

---

## 8. multi-feature 리팩터 접점 (예정)

원본 VBI 스타일 multi-feature 확장은 **§2.4 feature 단계만** 확장한다. §1~3·5(θ·decode·sim·MAF
학습 루프·평가)는 그대로. 추가 feature group(FC PCA / node-strength / network-FC / graph /
FCD / temporal / spectral)은 모두 §7-3 제약을 따른다:
- **FC-derivable**(FC/node-strength/network-FC/graph): sim·empirical 둘 다 계산 → conditioning 가능.
- **timeseries-derivable**(FCD/temporal/spectral): simulated 쪽만 계산 가능 → empirical BOLD 없으면
  conditioning 제외(진단·ablation 한정). 각 group은 config on/off + ablation 구조.
