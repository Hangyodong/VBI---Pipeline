#!/usr/bin/env python3
"""
evaluation/metrics.py 패치 — _resimulate_and_score 진행 바 추가

실행:
  cd /scratch/home/wog3597/vbi
  python patch_metrics_resim.py
"""

import shutil
import sys
from pathlib import Path

ROOT   = Path("/scratch/home/wog3597/vbi")
TARGET = ROOT / "evaluation" / "metrics.py"


def main():
    if not TARGET.exists():
        print(f"오류: {TARGET} 없음")
        sys.exit(1)

    src = TARGET.read_text(encoding="utf-8")

    # 이미 패치됐는지 확인
    if "_display_stride" in src or "_rs_done" in src:
        print("✓ 이미 패치됨 — 종료")
        sys.exit(0)

    # 백업
    bak = Path(str(TARGET) + ".bak_resim_bar")
    shutil.copy2(TARGET, bak)
    print(f"백업: {bak}")

    # ──────────────────────────────────────────────────────────────
    # 현재 _resimulate_and_score 루프 구조 확인 후 패치
    #
    # 현재 코드 패턴 (파일에서 확인):
    #   for <idx>, <theta_row> in enumerate(samples_raw):
    #       ...
    #       simulate_single(...)
    #       ...
    #
    # 목표: simulate_single 호출 직후에 \r 진행 바 삽입
    # ──────────────────────────────────────────────────────────────

    # _progress("resim start...") 라인 찾아서 그 뒤에 초기화 코드 삽입
    OLD_RESIM_START = '        _progress(f"resim start{tag}: {n_resim} simulations")'

    NEW_RESIM_START = '''\
        _progress(f"resim start{tag}: {n_resim} simulations")

        # ── resim 진행 바 초기화 ─────────────────────────────────
        import sys as _sys_rs, time as _time_rs
        _rs_bw    = 20
        _rs_t0    = _time_rs.perf_counter()
        _rs_lbl   = str(sid) if sid is not None else tag.strip() or "resim"

        def _fmt_rs(s):
            m = int(s) // 60
            return f"{m:02d}:{int(s) % 60:02d}"'''

    if OLD_RESIM_START not in src:
        print("✗ '_progress(resim start...)' 패턴을 찾지 못함")
        print("  아래 명령으로 실제 패턴 확인 후 알려주세요:")
        print("  grep -n '_progress\\|resim start' evaluation/metrics.py")
        sys.exit(1)

    src = src.replace(OLD_RESIM_START, NEW_RESIM_START, 1)
    print("✓ STEP 1: 진행 바 초기화 코드 삽입")

    # ── simulate_single 호출 직후 진행 바 업데이트 삽입 ──────────
    # 현재 for 루프 안에서 simulate_single 호출 후 fc_corrs.append 등이 있음
    # simulate_single 반환 직후 라인을 찾아 \r 코드 삽입

    # "for <i>, <row> in enumerate(samples_raw):" 패턴 찾기
    import re

    # for 루프 헤더
    loop_m = re.search(
        r'(        for (\w+), (\w+) in enumerate\(samples_raw\):)',
        src
    )
    if not loop_m:
        print("✗ 'for <i>, <row> in enumerate(samples_raw):' 패턴 없음")
        print("  grep -n 'enumerate(samples_raw)' evaluation/metrics.py")
        sys.exit(1)

    loop_var_i   = loop_m.group(2)  # e.g. "idx" or "i"
    loop_var_row = loop_m.group(3)  # e.g. "row" or "theta_row"
    print(f"✓ STEP 2: 루프 변수 확인 — 인덱스={loop_var_i}, 행={loop_var_row}")

    # simulate_single 호출 라인 찾기 (루프 내부)
    sim_m = re.search(
        r'(            bolds = simulate_single\([^\)]+\))',
        src,
        re.DOTALL
    )
    if not sim_m:
        # 더 짧은 패턴으로 재시도
        sim_m = re.search(r'(            \w+ = simulate_single\()', src)

    if not sim_m:
        print("✗ simulate_single 호출을 찾지 못함")
        print("  grep -n 'simulate_single' evaluation/metrics.py")
        sys.exit(1)

    # simulate_single 호출 라인의 끝 (괄호 닫힘) 찾기
    sim_start = sim_m.start()
    # 해당 위치부터 다음 줄 끝까지
    end_of_sim_line = src.find("\n", sim_start)
    # 괄호가 여러 줄에 걸칠 수 있으므로 ) 닫히는 지점 찾기
    depth = 0
    pos = sim_start
    while pos < len(src):
        if src[pos] == '(':
            depth += 1
        elif src[pos] == ')':
            depth -= 1
            if depth == 0:
                end_of_sim_call = pos + 1
                break
        pos += 1

    end_of_sim_line = src.find("\n", end_of_sim_call)
    sim_call_block  = src[sim_start:end_of_sim_line + 1]

    # \r 진행 바 업데이트 코드
    PROGRESS_UPDATE = f'''
            # ── resim \x5cr 진행 바 ──────────────────────────────────
            _rs_done = {loop_var_i} + 1
            _rs_el   = _time_rs.perf_counter() - _rs_t0
            _rs_sp   = _rs_done / _rs_el if _rs_el > 1e-6 else 0.0
            _rs_eta  = (n_resim - _rs_done) / _rs_sp if _rs_sp > 1e-6 else 0.0
            _f       = int(_rs_bw * _rs_done / n_resim)
            _b       = "\\u2593" * _f + "\\u2591" * (_rs_bw - _f)
            _sys_rs.stdout.write(
                f"\\r  [{{_rs_lbl}}] resim"
                f" |{{_b}}| {{_rs_done:>2d}}/{{n_resim}}"
                f" | {{_rs_sp:.1f}} sim/s"
                f" | ETA {{_fmt_rs(_rs_eta)}}   "
            )
            _sys_rs.stdout.flush()
'''

    src = src[:end_of_sim_line + 1] + PROGRESS_UPDATE + src[end_of_sim_line + 1:]
    print("✓ STEP 3: simulate_single 직후 \\r 업데이트 삽입")

    # ── 루프 끝 직후 print() 삽입 ────────────────────────────────
    # _progress("resim done...") 라인 찾기
    done_m = re.search(r'        _progress\(f"resim done[^"]*"\)', src)
    if done_m:
        old_done = done_m.group(0)
        new_done = '        print()  # resim \\r 라인 마무리\n' + old_done
        src = src.replace(old_done, new_done, 1)
        print("✓ STEP 4: 루프 끝 print() 삽입")
    else:
        # 대안: fc_corrs 집계 시작 부분 찾기
        agg_m = re.search(r'\n        if not fc_corrs:', src)
        if agg_m:
            src = src[:agg_m.start()] + "\n        print()  # resim \\r 마무리" + src[agg_m.start():]
            print("✓ STEP 4: 루프 끝 print() 삽입 (대안 위치)")
        else:
            print("  ⚠ 루프 끝 위치 불확실 — print() 수동 추가 필요")

    # ── sid 파라미터 추가 ─────────────────────────────────────────
    # _resimulate_and_score 시그니처에 sid=None 추가
    old_sig_m = re.search(
        r'(def _resimulate_and_score\([^)]+)\)',
        src,
        re.DOTALL
    )
    if old_sig_m and "sid=" not in old_sig_m.group(0):
        old_sig = old_sig_m.group(0)
        new_sig = old_sig[:-1] + ",\n                              sid=None)"
        src = src.replace(old_sig, new_sig, 1)
        print("✓ STEP 5: _resimulate_and_score에 sid=None 파라미터 추가")

    # evaluate_subject 에서 _resimulate_and_score 호출 시 sid= 전달
    # (sid 변수가 evaluate_subject 스코프에 있음)
    call_m = re.search(
        r'(_resimulate_and_score\(\s*\n?\s*n_resim,[^)]+)\)',
        src,
        re.DOTALL
    )
    if call_m and "sid=" not in call_m.group(0):
        old_call = call_m.group(0)
        new_call = old_call[:-1] + ",\n        sid=sid)"
        src = src.replace(old_call, new_call, 1)
        print("✓ STEP 6: evaluate_subject → _resimulate_and_score에 sid=sid 전달")

    # ── 저장 + 컴파일 확인 ───────────────────────────────────────
    TARGET.write_text(src, encoding="utf-8")

    try:
        compile(src, str(TARGET), "exec")
        print("✓ 컴파일 OK")
    except SyntaxError as e:
        print(f"✗ 컴파일 실패: {e}")
        print(f"  라인 {e.lineno}: {e.text}")
        print("백업으로 복구 중...")
        shutil.copy2(bak, TARGET)
        print("복구 완료")
        sys.exit(1)

    # 최종 검증
    final = TARGET.read_text(encoding="utf-8")
    print("\n[검증]")
    for pat, desc in [
        ("_rs_done",       "루프 카운터"),
        ("_sys_rs",        "stdout writer"),
        ("_fmt_rs",        "시간 포맷터"),
        ("sid=None",       "sid 파라미터"),
        ("print()  # resim", "마무리 newline"),
    ]:
        icon = "✓" if pat in final else "✗"
        print(f"  {icon} {desc}")

    print("\n✅ 패치 완료!")
    print("\n다음 단계 — main.ipynb cell[12] 확인:")
    print("  cell[12]가 54줄인데 시각화 코드입니다.")
    print("  간소화 원하시면 알려주세요.")


if __name__ == "__main__":
    main()
