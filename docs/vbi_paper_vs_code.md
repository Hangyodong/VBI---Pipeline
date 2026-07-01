# VBI 원본(논문/라이브러리) vs 현재 코드 비교

**Generated:** 2026-06-22
**대상:** ins-amu/vbi (eLife 2025, Ziaeemehr et al.) vs 본 repo(HCP RWW-EIB-2cpl 파이프라인)
**근거:** 원본 GitHub 트리/README/소스(web 검증) + 본 repo 직접 매핑 + 사용자 제공 논문 정리.

Evidence legend: ✅ 동일 · ◐ 부분/축소 · ⚠️ 외부화·생략 · ❌ 미구현 · ➕ repo 고유 확장.

> 관련: [pipeline_analysis.md](pipeline_analysis.md)(코드 내부 흐름),
> [performance_diagnosis.md](performance_diagnosis.md)(corr 진단). 본 문서는 "원본 VBI와
> 무엇이 다른가"에 집중.

---

## 0. 핵심 요약

현재 repo = VBI **철학**(SBI + whole-brain model + single-round amortized posterior)을 계승하되
**HCP fMRI × RWW-EIB-2cpl × FC 단일 사례로 특화된 fork**. 즉 "범용 toolkit vs 한 실험".

- `requirements.txt`에 `vbi==0.4.3`이 있으나 **production 미사용**. `debug.py`/`debug_notebook.py`만
  `vbi.models.cupy.wilson_cowan.WC_sde`를 import. 실제 활성 엔진 = 자체 `cuBNM/`(CUDA codegen) +
  `simulation/wc_eib.py`(cupy). → VBI pip 패키지는 **vestigial dependency**.
- 원본의 핵심 자산인 **범용 feature_extraction 카탈로그**(spectral/temporal/info + catch22/tsfel/
  pyspi/hctsa/JIDT)는 **미이식** — repo는 FC upper-tri 1종만 활성.
- repo가 VBI 정신을 가장 충실히 구현한 부분 = **basis reparameterization**(`basis_decoder.py`의
  myelin/gradient tanh 디코드) — 논문 Wong-Wang 264→9 재parameterization과 같은 아이디어.

### 원본 VBI 정체 (web 검증)
- repo: `https://github.com/ins-amu/vbi` · pip `vbi` · INS, Aix-Marseille (Ziaeemehr, Woodman,
  Domide, Petkoski, Jirsa, Hashemi). eLife Reviewed Preprint 106194 / bioRxiv 2025.01.21.633922.
- inference: `sbi` 패키지 래핑(`vbi/sbi_inference.py`, `SNPE/SNLE/SNRE` 제공), 논문은 **단일라운드
  amortized NPE** 사용. density estimator MAF + NSF.

---

## 1. 논문 파이프라인 단계 vs 현재 코드

| 논문 단계 | 논문(VBI) | 현재 코드 | 상태 |
|---|---|---|---|
| 0. 목적 | control param **posterior** 추정 | 동일 (RWW-EIB basis coeff posterior) | ✅ |
| 1.1 구조 입력 | T1 + DW-MRI | 없음 — precomputed SC `.mat` 로드 | ⚠️ |
| 1.2 기능 입력 | EEG/MEG/sEEG/**BOLD** | HCP fMRI **FC만** | ◐ |
| 1.3 prior | param별 plausible range | Uniform coeff U(-2,2)^12 / bound별 | ✅ |
| 2. connectome 구축 | parcellation→tractography→SC+length | 외부 `.mat`(CAB-NP 381→360 cortical) | ⚠️ |
| 2.4 delay | `T=d/v` | `compute_delay_matrix`, v=3.0 m/s | ✅ |
| 3. model assembly | WCo/JR/SL/Epileptor/MPR/WW + BW | RWW-EIB-2cpl만 활성(7 엔진 중) | ◐ |
| 4. prior sampling | θ~p(θ), N_sim | scaled prior BoxUniform | ✅ |
| 5. simulation | multi-backend simulator | cuBNM(CUDA)+cupy. C++/jax/tvbk 없음 | ◐ |
| 5.3 BOLD 매핑 | Balloon-Windkessel | `hrf="bw"` | ✅ |
| 6. 저장/로더 | HDF5/NPZ/PT | NPZ + 캐시 meta | ◐ |
| 7. **feature 추출** | stat/spectral/temporal/connectivity/info + catch22/tsfel/pyspi/hctsa | **FC upper-tri만**. FCD 존재하나 OFF. spectral/temporal/info 미구현 | ❌ |
| 7.1 차원축소 | FC/FCD **PCA** | FC PCA(256,whiten) **+ basis reparam** | ✅➕ |
| 8. train pair | (θ, feature) | 동일 | ✅ |
| 9. density estimator | **MAF + NSF**, `sbi` | **MAF만**(SNPE_C, sbi 0.26.1) | ◐ |
| 9.5 single-round amortized | 핵심 | 단일 라운드. Phase2/4 배선되나 OFF | ✅ |
| 10. observation inference | feature→posterior sample | `infer_subject_raw` | ✅ |
| 11. posterior predictive | θ̂ 재시뮬 비교 | `ppc`, resim FC corr/RMSE | ✅ |
| 12. diagnostics | **z-score** + shrinkage | shrinkage·SBC·ppc. z-score 미명시 | ◐ |
| 13. sensitivity | posterior gradient eigen | `active_sensitivity.py`(sbi ActiveSubspace) | ✅ |

---

## 2. VBI 라이브러리 모듈 vs 현재 repo 모듈

| VBI (ins-amu/vbi) | 책임 | 현재 repo 대응 | 차이 |
|---|---|---|---|
| `vbi/models/{cpp,cupy,jax,numba,pytorch,tvbk}/` | 백엔드별 model | `cuBNM/`(CUDA codegen) + `simulation/wc_eib.py`(cupy) | cuBNM 중심 자체 엔진. C++/jax/tvbk/pytorch-model 없음 |
| `vbi/feature_extraction/` (calc_features, features.json, infodynamics.jar, catch22, hmm) | 범용 시계열 feature 카탈로그 | `features/{fc,fcd,extraction,screening}.py` | FC/FCD 전용 소수 함수. JIDT/catch22/tsfel/pyspi 카탈로그 없음 |
| `vbi/sbi_inference.py` (SNPE/SNLE/SNRE, maf/nsf) | SBI 래퍼 | `inference/snpe.py`(SNPE_C)+`posterior.py` | SNPE_C 단일, MAF only. `inference/`가 패키지로 세분(scaling/priors/feature_pipeline/snpe/posterior/diagnostics/stage1/io) |
| `vbi/cde.py`, `inference.py` | conditional density est | snpe.py에 통합 | — |
| `vbi/dataset/` | 번들 connectome/예제 | 루트 `.mat`(HCP_FC/SC, CAB-NP) | 직접 배치 |
| `vbi/plot.py` | 플롯 | `evaluation/plots.py` | 평가 특화 |
| `vbi/utils.py`, `optional_deps.py` | 유틸 | `pipeline_setup.py`, `config.py`, `engine_select.py` | config/엔진라우팅 인프라가 훨씬 두꺼움 |
| `vbi/papers/`, `tests/` | 재현/테스트 | `tests/smoke/`, `test_*.py`, `docs/` | HCP 전용 smoke |
| — | — | ➕ `basis_decoder.py`, `param_decoder.py`, `region_basis.py` | repo 고유: region-wise basis reparam (latent/direct/basis) |
| — | — | ➕ `engine_select.py` | repo 고유: 7-model 플러그 라우팅 + region-wise wrap |
| — | — | ➕ `evaluation/` 패키지(metrics/validation/final_test/model_selection/reports) | repo 고유: train/val/test 분리 + model selection |

> 주의(원본 모듈명): `vbi/inference/`·`vbi/utils/`는 **디렉토리 아님** — 실제 `sbi_inference.py`/
> `inference.py`, `utils.py` 파일. `vbi/models/jax`는 stub(`__init__.py`만).

---

## 3. 능력 매트릭스

| 축 | VBI 논문/라이브러리 | 현재 repo | 평가 |
|---|---|---|---|
| **Models** | MPR, WW/rww, Jansen-Rit, Wilson-Cowan, VEP/Epileptor, Hopf/SL, Kuramoto, damped | wc, rwweib, **rwweib2(활성)**, rwweibdelay, rww, vbi(cupy) | RWW 계열 특화. JR/SL/Epileptor/MPR/Kuramoto 없음 |
| **Backends** | C++, numba, cupy, jax(stub), pytorch, tvbk | cuBNM(CUDA/Numba codegen), cupy, numpy fallback | 고성능 1종 + cupy |
| **Features** | stat/spectral/temporal/connectivity/info + catch22/tsfel/pyspi/hctsa | **connectivity(FC)만 활성**, FCD OFF | ❌ 최대 격차 |
| **Dim reduction** | FC/FCD PCA; (WW: myelin+FC-gradient 264→9) | FC PCA(256) **+ myelin/gradient basis tanh 360→3계수** | ✅ 논문 Wong-Wang 케이스 직접 모사(가장 닮음) |
| **Inference** | 단일라운드 amortized **NPE**, `sbi` | 단일라운드 **SNPE_C**, sbi 0.26.1 | 사실상 동치(NPE=single-round SNPE) |
| **Density est** | MAF + NSF | **MAF만** | NSF 미사용 |
| **Embedding** | (summary stat) | RegionTransformer 구현됐으나 **OFF**, Identity 사용 | 추가 자산이나 비활성 |
| **Connectome** | T1+DWI→tractography 빌드 | precomputed `.mat` 로드 | 빌드 외부화 |
| **Diagnostics** | z-score, shrinkage, sensitivity | shrinkage·SBC·ppc·ActiveSubspace·embedding-probe (z-score 미명시) | SBC/ppc는 더 풍부 |

---

## 4. 핵심 차이 (narrative)

1. **범위**: VBI = 범용 toolkit(N modality × N model × N feature). repo = HCP fMRI × RWW-EIB-2cpl ×
   FC 단일 파이프라인. "라이브러리 vs 한 실험".
2. **Feature가 최대 격차**: VBI 핵심 자산인 feature_extraction 카탈로그 대신 repo는 FC upper-tri
   1종. 논문 주의점 #3("FC만으론 부족, spatio-temporal 병행 필요")을 정면으로 만남 → corr 천장의
   구조적 원인 중 하나(→ [performance_diagnosis.md](performance_diagnosis.md) D6/D7 인접).
3. **가장 닮은 지점 = basis reparam**: `basis_decoder.py`의 myelin/gradient tanh 디코드 = 논문
   Wong-Wang 264→9 재parameterization과 같은 아이디어. repo가 VBI 정신을 가장 충실히 구현.
4. **엔진**: VBI pip은 vestigial. 실제는 cuBNM(코드젠 GPU) + region-wise wrap. `engine_select.py`/
   `param_decoder.py`는 VBI에 없는 고유 추상화.
5. **connectome 빌드 외부화**: 논문 1~2단계(T1/DWI→tractography) 부재. SC/FC를 완성품으로 받음.
6. **single-round amortized 일치**: repo SNPE_C 단일 라운드 = 논문 amortized NPE와 사실상 동일.

---

## 5. 논문 정리(사용자 spec)에서 정정할 점 (paper 대조)

- **"SNPE-C"는 부정확**: 논문은 단일라운드 **amortized NPE**. SNPE/SNLE/SNRE는 논문이 안 쓰는
  sequential 대안으로만 언급. (단 본 repo는 실제로 `sbi.SNPE_C`를 단일라운드로 써서 NPE와 동치 —
  표기만 다름.)
- **feature lib**: catch22는 경량/직접 통합, tsfel/pyspi/hctsa는 교체형 — 한 묶음 아님.
- **DWI → DW-MRI** 표기. **N_nodes=88**은 논문 일부 사례값(repo는 360 cortical / 381).
- 라이브러리 모듈명: `vbi/inference/`·`vbi/utils/`는 디렉토리 아님(파일).

---

## 6. 시사점 (corr 목표 연결)

- repo가 원본 대비 **버린 것 = feature 다양성**. 현재 corr 천장(~0.13–0.2)은 부분적으로 이 축소
  탓(논문은 spatio-temporal+FC/FCD 병행 시 정확도↑ 보고). per-subject FC 유지 결정 하에서 corr을
  더 올리려면, 향후 **FCD 재활성 / spectral·temporal feature 추가**가 VBI 정신에 부합하는 다음 레버
  (현재 scope=eval+feature+bounds 밖, S7 인접). → [corr_improvement_plan.md](corr_improvement_plan.md).
- repo가 원본 대비 **추가한 것 = region-wise basis 추론 + multi-engine 라우팅 + train/val/test 평가
  인프라**. 이는 VBI보다 진보한 부분.
