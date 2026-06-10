# 01 — Runbook

## 풀 파이프라인
```bash
python main_mouse.py          # mouse MPTP (config.py 권위)
python main_human.py          # human (HUMAN_FC/SC placeholder, VELOCITY=1.0)
```
- env override: `N_SIM`, `N_TRAIN`, `N_TEST`, `GPU_BATCH`
- notebook `main.ipynb`/`main_mouse.ipynb`도 동일 PipelineConfig → 결과 일치
- 단계: Phase1(SNPE) → Phase2(attention 선택) → Phase3(2차 SBI/RegionTransformer). `pipelines.run_pipeline`→`inference.run_stage1_snpe`.
- N_VAL=0이면 Step9 validation 지표가 nan으로 나오는 건 정상.
- Step2는 `output_mouse_mptp/features_stage1.npz` 있으면 캐시 로드 — 재시뮬하려면 삭제.

## 진단 도구

### sweep_fc.py — 파라미터 FC 민감도
```bash
python sweep_fc.py --subject sub-419077 --grid 13 --repeats 3
python sweep_fc.py --params g_e,c_ei,c_ee      # 좁혀서 빠르게
```
- 한 param씩 grid sweep → `sensitivity`(FC 변하나) + `realfit_max`(실제 FC 도달 corr) + dead knob 판정
- theta-route(g_e,g_i,c_ei)는 batch 1개, fixed-route(P,Q,c_ee,c_ie,c_ii)는 grid당 call → 좁혀 써라

### joint_opt.py — 합동 천장
```bash
python joint_opt.py --all --workers 8 --params g_e,c_ei,c_ee
python joint_opt.py --subject sub-419077 --params g_e,c_ei,c_ee,P
```
- `--all`: ctr+MPTP 8명 전체. `--params`: 아무 WC param(c_ee/P/Q 포함, 어댑터 수정으로 per-sim 적용됨)
- subject당 한 batch(`--samples` 기본 5000, ≤GPU_BATCH), call 1개 → 그룹 8 call (SC가 subject마다 달라 병합 불가)
- `--workers N`(--gpus 빈값): N process가 GPU 하나 MPS 동시 공유. 큰 batch는 OOM 주의
- `--gpus 0,1`: 멀티 GPU 핀 (GPU0 비었을 때)
- 진행 %/ETA 출력. 결정론적이라 repeats 없음
- 출력: `output_mouse_mptp/joint_opt_<sid>.npz/.png` + `_group.npz` + GROUP SUMMARY 테이블/verdict

## 속도 노브
- `config.DT 1.0` — delays 모델 권장, 0.5 대비 ~2배. (FS_NEURAL 영향, hrf="bw"면 FC 무관)
- call 시간 = 적분길이(T_END/DT) × sim 개수. **sim 개수는 공짜 아님** — H100 SM(~132개)이 ~100 sim에서 포화, 그 위는 거의 선형. (bench_sim.py 실측, sub-419077, delays ON: 100=39s, 1000=125s, 5000=541s → 1000→5000은 4.3×). 줄일 두 노브 = **samples**(선형)와 **call 수**(=SC 수=subject 수).
- **delays ON = ~9× 느림** (bench_sim.py 실측, 크기 무관 8.8~9.9×). delay history buffer 조회 비용. 단 끄면 천장 반토막(0.209→0.116)이라 측정엔 못 끔.
- 측정/벤치: `python bench_sim.py --samples 100,1000,5000 --repeat 2` (warmup으로 GPU init+JIT 제외, delays ON/OFF tax + ETA 출력)
- joint_opt 빠르게: `--samples 1000`(천장엔 충분, 5000 대비 4.3× ↓) + `--workers 8 --gpus 0,1`. `--all` 풀(5000, workers 1) = ~72분.

## 판정 기준 (천장)
- real-FC corr 천장 ≥0.5 → 파라미터 문제(추론으로 해결)
- 0.3~0.5 → 구조적 갭 일부
- <0.3 → 시뮬 구조 미스펙(SC/delays/BOLD/WC) — 추론 고쳐도 소용없음
