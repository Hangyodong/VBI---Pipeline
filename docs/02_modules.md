# 02 — Modules

## 엔트리
- `main_mouse.py` / `main_human.py` — 터미널 실행. PipelineConfig+setup_pipeline→pipeline_setup._apply_to_config
- `main.ipynb` / `main_mouse.ipynb` / `main_human.ipynb` — notebook (동일 로직)
- `pipeline_setup.py` — PipelineConfig 정의, priors, config 적용
- `config.py` — 권위 설정 (WC_FIXED, STAGE1_PARAMS, 시뮬/데이터 경로)
- `data_loader.py` — mat/tsv 로드, SC 스케일, delay matrix, group 필터

## 패키지
### simulation/
- `wc_eib.py` — WC E-I-B 모델 (global 분기 `c_ffi=g_i*(W@I)` 억제결합)
- `wc_runner.py` — 시뮬 러너
- `delays.py` — conduction delay matrix (lengths/velocity)
- `warmup.py`, `qc.py`

### features/
- `fc.py` / `fcd.py` — FC/FCD 계산
- `extraction.py` — feature 추출
- `screening.py` — feature 스크리닝

### inference/
- `stage1.py` — `run_stage1_snpe` 메인 드라이버
- `training_data.py` — 시뮬 batch 생성 (simulate_gpu_batch, GPU_BATCH chunk)
- `snpe.py` — SNPE-C 훈련
- `posterior.py`, `priors.py`, `scaling.py` — posterior/prior/스케일링
- `embedding.py`, `attention_selection.py`, `feature_pipeline.py` — 임베딩/region attention/2차 SBI
- `diagnostics.py`, `io.py`, `_utils.py`

### evaluation/
- `final_test.py` — Step14 최종 테스트
- `validation.py` — validation 지표 (N_VAL=0이면 nan)
- `metrics.py`, `model_selection.py`, `plots.py`, `reports.py`

### pipelines/
- `stage1_stage2.py` — `run_pipeline` 체인

### cuBNM/ (GPU 엔진 어댑터)
- `simulate.py` — `simulate_gpu_batch` (VBI 계약 drop-in, delays→sc_dist 매핑, USE_DELAYS gate)
- `runner_vbi.py` — `run_cubnm_vbi_batch`, `build_param_lists` (theta→WCVBISimGroup param. **`_FIXED_TO_VBI` 매핑 theta 컬럼은 per-sim 적용** — 수정됨)
- `runner.py`, `fc.py`, `benchmark.py`

## 진단/유틸 (루트)
- `sweep_fc.py` — 민감도 sweep
- `joint_opt.py` — 합동 천장 탐색
- `diag_coupling.py` — cuBNM vs VBI 결합 진단
- `simulator.py` — `compute_fc`, `simulate_single`
- `debug.py`, `evaluate.py`, `screening.py` 등

## 레거시 (루트 중복 — 패키지가 정본)
`bold.py, fc.py, fcd.py, delays.py, extraction.py, screening.py, qc.py, warmup.py, wc_runner.py` — 옛 평면 구조 잔재. 패키지(`simulation/`, `features/`) 쪽 사용.
