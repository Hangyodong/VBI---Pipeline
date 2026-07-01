# 03 — FC 미스펙 진단 (민감도 · 천장)

> ⚠️ **LEGACY (mouse WC).** 현 HCP 진단은 `docs/performance_diagnosis.md` +
> `docs/corr_improvement_plan.md` 참조.

핵심 질문: **WC 시뮬이 실제 ctr+MPTP FC를 재현할 능력이 있나?** 원인이 (1)파라미터 선택 vs (2)시뮬 구조 미스펙인지 가른다.

## 배경 문제
Step14 최종 테스트 real-FC corr ≈ 0.03. 시뮬 FC가 실제 FC 공간패턴 못 맞춤. 시뮬 FC std가 ~0.088에 고정(실제 ~0.17~0.23). amortized posterior가 OOD 실제 데이터에 confidently-wrong.

## 민감도 (sweep_fc.py, sub-419077, delays OFF)
| param | route | sensitivity | realfit_max |
|---|---|---|---|
| **c_ee** | fixed | **0.72** | 0.116 |
| c_ei | theta | 0.33 | 0.028 |
| g_e | theta | 0.14 | -0.002 |
| P | fixed | 0.026 | -0.001 |
| Q | fixed | 0.003 | - |
| g_i | theta | 0.0008 | - |
| c_ie | fixed | 0.0007 | - |
| c_ii | fixed | 0.000 | - |

- 살아있는 knob: **c_ee ≫ c_ei > g_e**
- 죽은 knob: g_i, Q, c_ie, c_ii (P도 거의)
- **추론셋 어긋남**: STAGE1이 죽은 P,Q,g_i 추론 + 최강 c_ee는 고정. → 합리적 셋 = {g_e, c_ei, c_ee}

## 천장 (joint_opt.py)
| 조건 | real-FC corr 천장 |
|---|---|
| delays-OFF, 단일 param | 0.116 |
| delays-ON, joint(g_e,g_i,c_ei), c_ee 제외 | 0.209 / subject |
| delays-ON, joint(g_e,c_ei,**c_ee**), 그룹 8명 | **측정 진행 중** |

delays + 합동이 천장을 ~2배 올림. c_ee 포함 결과가 핵심 판정점.

## cuBNM 어댑터 라우팅 버그 (수정됨)
`cuBNM/runner_vbi.py build_param_lists`:
- **이전**: theta 컬럼에서 g_e/g_i/c_ei만 per-sim 적용. c_ee/P/Q/c_ie/c_ii는 WC_FIXED scalar broadcast → theta에 넣어도 시뮬 무시 (P/Q "죽은 컬럼" 버그: STAGE1이 P,Q 추론해도 시뮬 반영 0)
- **수정**: `_FIXED_TO_VBI` 매핑되는 theta 컬럼 전부 per-sim 적용. → c_ee 합동탐색 가능 + P/Q 라우팅 정상화
- **주의**: P/Q는 적용돼도 민감도 ≈0이라 추론 가치 없음. main 파이프라인 동작 바뀜(P,Q 이제 sim 반영).

## 다음 액션
1. c_ee 포함 그룹 천장 확정 (joint_opt --all --params g_e,c_ei,c_ee)
2. 천장 <0.3 유지면 → 구조 미스펙: SC 스케일링 / delays / BOLD(hrf) / WC 구조 재검토. HRF는 균일 선형 컨볼루션이라 FC 패턴 개선엔 약함(2차).
3. ≥0.3면 → STAGE1을 {g_e, c_ei, c_ee}로 재선택, c_ee 추론 위해 어댑터 theta-route 활용, 임베딩 수정.
