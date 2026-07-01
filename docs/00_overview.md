# 00 — Overview

> ⚠️ **LEGACY (mouse MPTP Wilson-Cowan, 115-region).** 현 HCP RWW-EIB 파이프라인과
> 무관. 현재 파이프라인: `docs/current_pipeline.md` + `CLAUDE.md`.

## 목적
Wilson-Cowan(WC) neural mass 모델을 cuBNM GPU 엔진으로 시뮬하고, SBI(SNPE-C)로 subject별 WC 파라미터를 추론하는 파이프라인. 대상: Mouse MPTP, 115 region, ctr+MPTP 그룹.

## 데이터
- `MPTP_FC_115.mat` — struct (34,3): col0=subject id, col1=FC(115×115, NaN 미측정 포함), col2=FCD
- `MPTP_SC_115.mat` — col0=id, col1=SC weight(uint16 raw→log1p+max-norm), col2=tract length(mm, delay용)
- `participants.tsv` — group(ctr/kh) × treatment(MPTP/veh). 분석 대상 `config.GROUP_FILTER=("ctr","MPTP")` = 8명.
- `atlas_115_labels.txt` — region 라벨
- data_loader가 FC NaN→0 처리, SC 대칭화+스케일, delay matrix 계산.

## 주요 설정 (config.py)
- `N_REGIONS=115`, `VELOCITY_M_PER_S=1.5`, `USE_DELAYS=True`
- `STAGE1_PARAMS=["P","Q","g_e","g_i","c_ei"]` (5개 추론)
- `DT=1.0`ms, `T_END=720_000`ms(=12분, 실제 스캔 길이), `T_CUT=60_000`ms
- `N_SIM=50_000`, `GPU_BATCH=10_000`
- WC 고정값은 `config.WC_FIXED`

## 현재 핵심 이슈
**시뮬 FC가 실제 FC를 잘 재현 못함** (real-FC corr 낮음). 진단 진행 중 — [03_diagnosis_ceiling.md](03_diagnosis_ceiling.md).
- 민감도: c_ee가 최강(0.72)인데 추론셋에서 빠져 고정됨. g_i,Q,c_ie,c_ii는 죽은 knob.
- 천장: delays+합동 탐색으로 ~0.21. c_ee 포함 그룹 천장 측정 중.
- cuBNM 어댑터 P/Q 죽은-컬럼 라우팅 버그 수정됨.

## 환경 주의
- 머신 GPU 2장(H100 NVL), GPU0 종종 타인 점유. MPS 데몬 상시.
- 모듈 수정 후 Jupyter는 커널 재시작 필요.
- 루트에 패키지와 중복된 레거시 .py(bold/fc/fcd/delays/extraction/screening/qc/warmup/wc_runner)가 있음 — 패키지(simulation/features/...) 쪽이 정본.
