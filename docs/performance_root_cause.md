# 성능(FC corr) 저조 — 근본원인 전수 분석

**생성**: 2026-06-23 · **방법**: 7-레이어 병렬 코드조사(read-only) + high-impact 주장 adversarial 검증 + completeness critic (workflow `perf-root-cause`, 57 agent).
**대상**: main_HCP.py SBI 트랙 (rwweib2 + basis_regionwise), test FC corr ~0.14.
**목표(사용자)**: amortized `f(SC,FC)→θ` — 임의 새 (SC,FC)에 high-corr/low-rmse simFC 내는 θ 추론, per-input optim의 빠른 대체.

> 모든 줄번호/숫자는 현재 코드(branch `refactor/02-simulation`) 기준 재검증. CONFIRMED=재독으로 확인, PARTLY=방향 맞으나 일부 과장/오귀속, REFUTED=틀림.

---

## TL;DR — 결론 7개

1. **0.14는 진짜 fit 숫자다** (CONFIRMED). `val FC proxy 0.77~0.82`는 **가짜** — sim FC를 sim FC 평균에 상관시킨 self-similarity, empirical과 무관 (snpe.py:248-260). 성능 지표로 절대 쓰지 마라.
2. **per-node 12-basis 트랙의 천장이 근본적으로 낮다**: per-subject 선형 SC-FC 천장 ~0.05~0.10, basis random-search 천장 0.196, homogeneous 0.03. 0.14는 이미 천장 근처. → **추론 튜닝으로 0.7 불가, 천장(축A)을 바꿔야.**
3. **신규 최대 단서 — I_o operating point 오설정** (CONFIRMED, 정량): basis bound `I_o(0,1)` → 중앙 0.5, prior 샘플 ~0.49. 근데 임계점은 ~0.382. mean-field 결과 학습 (sim,region)의 **2.9%만 임계대, 51.5% 포화, 43% sub-threshold**. → SNPE가 **거의 포화/죽은 FC로 학습됨.** I_o bound를 (0.30,0.45)로 + FIC 배선이 단일 최대 레버.
4. **목표와 구조적 불일치 — 기본 경로는 `f(FC)` 이지 `f(SC,FC)` 아님** (critic #1): `SC_CONDITION=0` 기본(main_HCP.py:186) → posterior가 **FC만** 조건. SC는 추론에 안 들어가고 resim에만 쓰임. 임의 SC 일반화 불가 by construction.
5. **cache 키가 sim 결정 변수 다수 누락** (CONFIRMED): `_cache_meta_now()`에 T_END/T_CUT/DT/TR/DECIMATE/sim_seed, SC_DATASET/SC_FILE/VBI_SC_SCALE, RWWEIB2_FIXED, basis 내용해시, train subject id 전부 **없음**. → I_o/sim길이/operating-point 수정해도 **stale cache 조용히 재사용 → 수정이 학습에 반영 안 됨.** "cache가 문제"라는 직감의 실제 근거.
6. **eib_tune의 0.8은 다른 모델** (CONFIRMED): per-edge W(~64,620 자유도)를 **group FC에 과적합**. 12-dim θ로 amortize 불가(shape 불일치), 일반화 보장 없음. 0.8 vs 0.14 = apples-to-oranges.
7. **사용자 "idx당 fic/eib cache 2개"는 현재 코드에 없음** (CONFIRMED REFUTED): fic_tune은 파일 저장 안 함, eib_tune은 단일 `eib_W.npy` 매번 덮어씀(sid 없음). main_HCP는 어느 것도 안 읽음. → 그 cache 충돌은 main_HCP 0.14의 원인 아님. 단 eib_W.npy 단일파일 덮어쓰기는 EIB 트랙의 실제 위험.

---

## 프레임워크 — 3축

```
달성 corr = min( 축A 천장 , 축B 도달치 )   ,   단 축C가 둘 다/측정을 오염시킬 수 있음
```
- **A 천장**: model+data가 애초에 낼 수 있는 최대 corr (DOF, SC 구조, operating point).
- **B 갭**: 추론/feature/objective가 천장 아래로 잃는 양.
- **C 무결성**: cache/평가/버그가 결과를 오염시키거나 가짜 숫자를 만듦.

집계: A=17, B=24, C=30 (총 71). refuted 3, partly 30.

---

## 축 A — 천장 (model/data 한계)

| id | 상태/검증 | impact | 메커니즘(요약) | 증거 | 수정 |
|---|---|---|---|---|---|
| **io-bound-off-operating-point** | confirmed/partly | **H** | I_o bound(0,1)→중앙0.5, prior~0.49 ≫ 임계 0.382. 학습 2.9%만 임계대, 51.5% 포화. SNPE가 포화 FC로 학습 | main_HCP.py:153; basis_decoder.py:81-83; mean-field r_E 3.08Hz@0.382 vs 25.6Hz@0.5 | `BASIS_BOUNDS['I_o']=(0.30,0.45)` + FIC |
| **no-fic-in-production** | confirmed/partly | **H** | J_i 고정 1.0, homeostatic operating-point 제어 없음 → off-criticality | main_HCP.py:110,119; yaml:124 "No FIC"; runner:31; compute_fic_ji는 standalone에만 | latent_wrap에 J_i_matrix override(USE_FIC) |
| **structural-ceiling-fixed-sc** | confirmed/partly | **H** | SC 토폴로지 고정, θ는 gain만 rescale → SC 없는 엣지(homotopic) 못 만듦 | yaml:33,40 coupling=g·C·S; eib_tune.py:9 W tune | GEOMETRY_COUPLING=1, 또는 EC 추론 |
| **callosal-sc-under-recovery** | confirmed/partly | **H** | tractography가 homotopic 엣지 65% 누락(nnz 0.36), empirical FC homotopic=0.42 vs 전체0.07 | simulation/geometry.py:5-8; 측정 nnz0.359 | GEOMETRY_COUPLING=1 GEOM_ALPHA=0.3 |
| **dof-too-low-12-basis** | confirmed/partly | **H** | θ=12=4param×3 rank-3 고정템플릿(const,myelin,gradient) vs 64,620 FC 엣지 | basis_decoder.py:39; basis_ceiling 0.196 | 맵 추가/per-region/EC |
| **saturation-low-simfc-std** | confirmed/partly | **H** | off-op + J_i1.0 → 포화 → simFC 저분산(legacy mouse 0.088, 현 모델 미측정) vs empirical 0.17~0.23 | mean-field 포화; std 0.088은 legacy | io/fic 수정과 동일 |
| **per-subject-fc-low-ceiling** | confirmed/partly | **H** | per-subject FC 기본(GROUP_AVG_FC=0), 선형 SC-FC 천장 ~0.05~0.10 | main_HCP.py:124; 측정 0.10, 문서 0.05 | GROUP_AVG_FC=1 (단 주의↓) |
| **prior-may-not-cover-empirical** | suspected/partly | H | g(0,3) tanh-bound → g>3 불가. optimum이 위면 support 밖 | basis_decoder.py:82-86; G_BOUND_HIGH | op-point 고친 뒤 G_BOUND_HIGH 스윕 |
| **gsr-empirical-vs-non-gsr-sim** | confirmed/partly | H | empirical FC 35% 음수엣지(GSR 의심), sim BOLD엔 GSR 없음 → manifold 다름 | features/fc.py:30 raw corrcoef; emp 음수비 0.355 | sim/emp 전처리 대칭화(둘 다 GSR 또는 center) |
| g-bound-clip-vs-posterior | suspected/partly | M | (0,3)cap이 under-couple 가능, 단 op-point 고치면 ~7 요구 자체가 artifact일 수 있음 | perf_diag g_LRE mean7.4(wide era) | op-point 먼저, 그 뒤 스윕 |
| fixed-params-off-defaults | suspected/partly | M | w_p1.4/J_N0.15 등 고정값이 op-point 좌우, 틀리면 보정 불가 | main_HCP.py:119-120 | FIC가 흡수 또는 재보정 |
| basis-const-column-symmetry | suspected/true | M | myelin/gradient L-R corr 0.985 → 비대칭/homotopic 특이 변동 못 담음 | 측정 0.986/0.985 | network/eigenmode 맵 추가 |
| homogeneous-baseline-floor | confirmed/partly | M | 동일 모델 homogeneous는 0.016~0.034 = 바닥 | basis_ceiling.py:147,169 | 모델 클래스 바꿔야 |
| only-100-sc-subjects | confirmed | L | group FC는 1039명, SC는 100명 → 모집단/다양성 불일치 | data_loader_hcp.py:211; weight_all(.,.,100) | 동일 100명 group FC |
| delays-default-off | confirmed | L | 6-100ms ≪ 720ms decimate → BOLD-FC에 ~0 효과 | main_HCP.py:179 | 그대로 둠 |
| sc-scaling-negligible | confirmed | L | maxnorm vs log1p 선형 천장 0.100 vs 0.104 동일 | 측정 | maxnorm 유지(천장 레버 아님) |

---

## 축 B — 천장 도달 방해 (추론/feature/objective)

| id | 상태/검증 | impact | 메커니즘(요약) | 증거 | 수정 |
|---|---|---|---|---|---|
| **snpe-loss-decoupled-from-fc-fit** | confirmed/**true** | **H** | loss=posterior density NLL(시뮬 θ 복원), corr/empirical 안 봄 → loss↓여도 corr 0.14 | snpe.py:225 SNPE_C; "FC metrics not exposed" | sim-in-loop fit objective / distill |
| **fc-pca-256-whiten-true-active** | confirmed/**true** | **H** | x=PCA256(whiten), 26% 버림, rank<360인데 256으로 truncate | main_HCP.py:174,178; feat_pipe:104 | FC_PCA_DIM≥360, whiten=0 |
| **eib-W-as-supervised-labels** | confirmed/**true** | **H** | EIB 출력=64620 connectome, 12-dim θ head와 shape 불일치 → 직접 label 불가 | eib_tune.py:137; basis theta_dim12 | 저차원 W 파라미터화 or connectome regressor |
| **eib-fic-not-wired-into-main** | confirmed/partly | **H** | 0.8 경로(eib/fic)가 amortized entry와 완전 단절. main_HCP은 J_i1.0 saturated 모델로 학습 | grep eib/fic in main_HCP=none | W를 target으로 or FIC 주입 |
| **eib-objective-overfit-not-amortizable** | confirmed/partly | **H** | W ~64,620 자유도 = FC 엣지수, group FC에 과적합 → 0.8은 일반화 아님 | eib_tune.py:79,119; docstring 자인 | 상한 진단으로만 취급 |
| pca-fit-sim-only-covariate-shift | confirmed/partly | H | PCA를 sim FC에만 fit→empirical에 적용, sim 부분공간이 empirical에 안 맞음 | main_HCP.py:880; feat_pipe:104 | 학습 embedding or 공동 fit |
| sim-vs-empirical-fc-mismatch | confirmed/partly | H | sim 166TR vs empirical ~1200TR + 전처리 다름 → 분포 mismatch가 OOD의 상류 | main_HCP.py:74-78 | sim 길이↑ + 전처리 대칭 |
| ood-rejection-collapse-clip-bias | confirmed/partly | H | OOD→accept<1%→clip fallback→경계 편향 (단 headline은 rejection 경로) | posterior.py:122-139 | OOD 근본 수정 |
| sim-window-too-short / short-sim-noisy | confirmed/true | M | 166 TR FC = 고분산(SE~0.078) → corr 희석 + 학습쌍 노이즈 | main_HCP.py:75; pipeline_setup:129=166 | T_END≥600000 + seed 평균 |
| single-fixed-sim-seed | confirmed/partly | M | sim_seed=42 고정, noise 미평균 (단 cuBNM noise 구조 일부 반박) | simulate:55; runner:142 | seed 변동/평균 |
| no-fisher-z | confirmed/partly | M | raw r(이분산), arctanh 없음 (sim/emp 둘 다 raw라 상호 일관은 함) | features/fc.py:30 | 양쪽 동일 Fisher-z |
| pca-256-truncation | confirmed/partly | M | 256<rank360, explained_var 0.742 | config.py:174; 측정 0.7416 | dim≥360 |
| point-estimate-mean-of-broad | confirmed/partly | M | posterior-mean θ resim (Jensen: 좋은θ 평균≠좋은θ), MAP/argmax 없음 | metrics.py:203 | FC-argmax-over-draws/refine |
| embedding-identity | confirmed/true | M | embedding=nn.Identity, FC 그래프 구조 미학습 | snpe.py:107; main_HCP:915 | 학습 embedding |
| eib-target-group-fc | confirmed/true | M | eib는 group FC(고정 1개)에 튜닝 → 가장 쉬운 타겟, per-subject 일반화 과장 | eib_tune.py:45 | 동일 타겟서 벤치 |
| fic-eib-interleave-staleness | confirmed/partly | M | J_i 20iter마다, W 매 iter → 19/20 동안 stale J_i로 drift→포화 | eib_tune.py:117-124 | refic_every=1 / W 재정규화 |
| eib-burn-in-short | confirmed/true | M | 튜닝중 ~62TR FC로 Hebbian update → 노이즈 | eib_tune.py:108,113 | FINAL full-len corr만 인용 |
| no-early-stop-overfit | confirmed | L | 200ep 고정, empirical-fit 기반 model selection 없음 | snpe.py:166 | val early stop on empirical proxy |
| upper-tri-loses-graph | confirmed | L | flat vector, 그래프 구조 미보존 | features/fc.py:52 | 구조 인식 encoder |
| hrf-decimate-vs-delays | confirmed | L | decimate720ms ≫ delay → fast timing 불가(소) | main_HCP.py:77 | 그대로 |
| flow-capacity | suspected | L | MAF h128×8, 12-dim엔 충분 | config.py:257 | 고차원 시 NSF |

---

## 축 C — 무결성 (cache/평가/버그)

| id | 상태/검증 | impact | 메커니즘(요약) | 증거 | 수정 |
|---|---|---|---|---|---|
| **val-fc-proxy-sim-vs-simmean** | confirmed/**true** | **H** | 0.77~0.82 = sim FC vs sim-mean self-similarity, empirical 무관. 0.14를 가림 | snpe.py:248-260 | 라벨 변경/제거 |
| **cache-key-omits-simlength-seed** | confirmed | **H** | meta에 T_END/T_CUT/DT/TR/DECIMATE/seed 없음 → sim길이 바꿔도 stale 재사용 | main_HCP.py:687-708 | meta에 추가 |
| **cache-key-omits-sc-dataset-scale** | confirmed/**true** | **H** | SC_DATASET/SC_FILE/VBI_SC_SCALE 없음 → SC 바꿔도 stale | main_HCP.py:687-708 | meta에 추가 |
| **cache-key-omits-rwweib2-fixed** | confirmed/partly | M | RWWEIB2_FIXED(J_i,w_p..) 없음 → operating-point 수정이 cache로 무시됨 | runner:89; meta probe none | meta에 추가 |
| **cache-key-omits-train-ids** | confirmed/**true** | M | theta shape만 검사, train subject id 미검증 → SEED/N 바꿔 같은 길이면 잘못된 sim 재사용 | main_HCP.py:719,728 | train_ids 추가 |
| **cache-key-basis-filename-not-content** | confirmed/partly | M | basis는 파일명만, 내용해시/REZSCORE 없음 → in-place 편집 시 stale | main_HCP.py:676-678 | content hash + rezscore |
| **user-fic-eib-cache-not-in-pipeline** | confirmed/**true** | M | 사용자 "idx당 2 cache"는 코드에 없음; fic 저장 안 함, eib는 단일 eib_W.npy 덮어씀 | fic_tune np.save=none; eib_tune.py:137 | (해당 트랙) per-sid 저장 |
| **no-per-idx-2-cache-exists** | confirmed/**true** | M | eib_W.npy sid 없이 매 run 덮어씀 → 소비자 있으면 마지막 subject W 오용 | eib_tune.py:137 | eib_W_{sid}.npy |
| **bootstrap-pools-all-draws** | confirmed/**true** | M | headline CI[0.128,0.140]는 within-subject draw 분산, subject 단위 아님 → 인위적 좁음 | final_test.py:57-58 | per-subject mean으로 bootstrap |
| nan-zero-fill-degrades | confirmed/partly | M | NaN→0 (단 현 cortical-360 NaN=0% → 영향 거의 없음) | data_loader_hcp.py:138 | 마스크화 (현재 무영향) |
| fixed-sim-seed-no-noise-var | confirmed/partly | M | seed42 고정 → CI가 noise 분산 과소 | runner:142 | seed 변동 |
| degenerate-fallback-corr-zero | confirmed/partly | M | 포화/실패 sim이 corr=0.0(NaN 아님)로 평균에 포함 → 하향 편향 | metrics.py:61-67 | NaN으로 분리 |
| clip-fallback-boundary-bias | confirmed/partly | M | rejection 실패시 clip→경계편향 (단 headline은 rejection) | posterior.py:139 | branch 노출 |
| headline-test-corr-is-real | confirmed | L | 0.14는 진짜 empirical fit (NaN 마스킹 정상) | metrics.py:312-315 | 신뢰 OK |
| engine-routing-fixed | confirmed | L | 옛 WC 하드코딩 버그(−0.005) 수정됨 | metrics.py:274 | 유지 |
| config-timing-defaults-divergent | (critic) | M | config.py/pipeline_setup 타이밍 기본값이 main_HCP override와 다름 | config.py:96-103 vs main:74-78 | 통일 |

### ⚠️ adversarial 검증서 REFUTED (틀린 가설 — 기록)
- **pca-whiten-amplifies-lowvar-ood = FALSE**: sbi가 `z_score_x="independent"`로 PC를 sim std로 재표준화 → flow 입력/OOD가 whiten 플래그와 **무관**. OOD 진짜 원인 = **empirical이 sim span 밖 + 256 truncation**. (whiten=0은 여전히 무해한 정리지만 OOD 해결책 아님.)
- **identifiability-degeneracy = FALSE**: basis는 의도적 식별성-개선 설계, shrinkage 0.6~0.9 정상.
- **fc-nan-mask-46pct = FALSE**: 현 HCP cortical-360 NaN=**0.0000** (100명). "46% NaN"은 stale 문서 주장.

---

## 교차 효과 (compound chains)

1. **flat-prior θ × I_o off-op × no-FIC**: uniform prior draw가 대부분 포화/죽은 영역 → 80k 학습 FC가 저분산 → (a) flow가 empirical 근처 FC 못 봄 (b) PCA 축이 constant 방향 오염 (c) empirical OOD. **하나의 뿌리(operating-point 제어 부재)가 3개 증상 동시 유발.** ← 최우선 수정 표적.
2. **SC 미조건 × per-subject 타겟**: flow는 x=PCA(FC)만 봄 → 새 SC를 추론서 무시하고 resim만 함 → SC 차이 활용 불가. per-subject 낮은 천장과 복합.
3. **group-FC × FC-only 조건**: GROUP_AVG_FC=1 + FC-only면 모든 subject의 x 동일 → posterior가 **상수 θ** → amortization 환상. (의도된 config가 오히려 amortization 깨뜨림.)
4. **short-window × seed고정 × no-FIC drift**: 각 FC가 짧고 비정상적인 단일 noise draw → 학습 라벨 노이즈 + eval 추정 노이즈 동시.
5. **PCA-whiten × PCA-오염 × sim-emp shift × clip**: 4개 부분확인 항목이 사슬로 accept 붕괴→clip→경계편향.

---

## critic가 새로 잡은 누락 원인 (상위)

- **default 경로 = f(FC), f(SC,FC) 아님** (SC_CONDITION=0). 목표와 최상위 불일치.
- **θ가 subject SC와 무관하게 iid prior 샘플** (active/importance 없음) → operating-point 무시.
- **group-FC + FC-only = 상수 θ** (amortization 붕괴).
- **SC축 OOD**: 100 SC subject만, 새 SC 일반화 미검증.
- **PCA가 quality filter 前 오염 sim FC로 fit** (std>1e-4 필터가 하류).
- **empirical-fit 기반 model selection 전무** (early-stop도 sim NLL/가짜 proxy).
- **posterior-predictive 일관성 미검증**: SBC(calibration)는 통과해도 forward FC는 틀릴 수 있음.

---

## 우선순위 (critic 종합) — 무엇부터

1. **조건변수 확정**: `SC_CONDITION` 무엇으로 돌렸나? f(FC)면 목표(amortize over SC) 구조적 불가. 튜닝 前 결정.
2. **타겟모드 degeneracy 점검**: 0.14 낸 run이 GROUP_AVG_FC=? group+FC-only=상수θ(가짜), per-subj=0.05~0.10 천장. 둘 다 문제 — 조합 명시.
3. **operating point를 source에서 고정**: FIC 배선 또는 I_o→(0.30,0.45) 재중심 **학습 前**. flat-prior draw가 비포화로 들어가게. → io-off/saturation/PCA오염/OOD를 **동시 타격** (지배적 사슬). 단일 최고 레버.
4. **empirical-fit 신호를 학습/선택에 도입**: 가짜 val proxy(snpe.py:249) → held-out subject posterior-predictive corr로 교체, early stop/checkpoint 선택에 사용.
5. **cache 키 강화 + config 통일**: T_END/T_CUT/DT/TR/DECIMATE/seed/SC_*/VBI_SC_SCALE/RWWEIB2_FIXED/basis-hash 추가 (전부 누락 확인). 안 하면 위 수정들이 stale cache로 **무시됨.**
6. **그 다음** 측정된 헤드룸 레버: sim 길이↑ + seed 평균, G_BOUND_HIGH 스윕(재sim), multi-round/focused SNPE.

---

## 목표(amortization)에의 함의

- **0.7 원하면 per-node로는 불가** (천장 ~0.2~0.5). EC(node-to-node)가 필요한데 → SBI posterior로 10⁴-dim 불가, **회귀/GNN**로 재설계 필요.
- eib_tune 0.8은 **group FC 과적합 connectome**, 12-dim head로 못 가져옴. amortize하려면 (a) 저차원 W 파라미터화 or (b) connectome regressor.
- **단, 위 #3(operating point)부터 고치면** per-node SBI도 ~0.2 천장까지는 올라갈 것 — 현재 0.14는 operating-point/cache 때문에 천장 아래일 가능성. 그게 사실인지가 첫 측정 대상.

### 결정적 분해 실험 (저비용, 순서대로)
1. `BASIS_BOUNDS['I_o']=(0.30,0.45)` + (가능시 FIC) → `basis_ceiling.py` 천장 재측정. 0.196→? (operating-point가 천장 눌렀나)
2. per-node CMA-ES optim 1 subject (random 아닌 진짜 optim) → 진짜 per-node 천장.
3. `tools/measure_feature_diag.py`(recon_corr/ood) → feature가 천장 아래로 얼마 깎나.
4. SC_CONDITION + GROUP_AVG_FC 조합 명시 후 held-out corr.

각 실험 전 **cache 키에 해당 변수 추가** (안 하면 stale 재사용으로 측정 무효).
