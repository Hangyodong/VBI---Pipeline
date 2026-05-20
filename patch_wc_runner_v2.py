#!/usr/bin/env python3
"""
wc_runner.py 진행 바 패치 스크립트 v2
=======================================
실제 코드 구조 기반 정확한 패치.

실행:
  cd /scratch/home/wog3597/vbi
  python patch_wc_runner_v2.py
"""

import re
import shutil
import sys
from pathlib import Path


ROOT = Path("/scratch/home/wog3597/vbi")
WC   = ROOT / "simulation" / "wc_runner.py"


def main():
    if not WC.exists():
        print(f"오류: {WC} 없음")
        sys.exit(1)

    src = WC.read_text(encoding="utf-8")

    # ── 이미 패치됐는지 확인 ──────────────────────────────────────
    if "_display_stride" in src:
        print("✓ 이미 패치됨 (_display_stride 존재) — 종료")
        sys.exit(0)

    # 백업
    bak = Path(str(WC) + ".bak_v2")
    shutil.copy2(WC, bak)
    print(f"백업: {bak}")

    # ══════════════════════════════════════════════════════════════
    # STEP 1: 시그니처에 label=None, n_total=None 추가
    # ══════════════════════════════════════════════════════════════
    OLD_SIG = (
        "def simulate_gpu_batch(weights, theta_batch, param_names,\n"
        "                       fixed_overrides=None, delays=None, apply_bw=True,\n"
        "                       _allow_fallback=True):"
    )
    NEW_SIG = (
        "def simulate_gpu_batch(weights, theta_batch, param_names,\n"
        "                       fixed_overrides=None, delays=None, apply_bw=True,\n"
        "                       _allow_fallback=True,\n"
        "                       label=None, n_total=None):"
    )
    if OLD_SIG not in src:
        print("✗ 시그니처를 찾지 못함 — wc_runner.py 내용을 확인하세요")
        sys.exit(1)
    src = src.replace(OLD_SIG, NEW_SIG, 1)
    print("✓ STEP 1: 시그니처 수정 완료")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: outputs = [] 직전에 진행 바 초기화 코드 삽입
    # ══════════════════════════════════════════════════════════════
    OLD_INIT = "    outputs = []\n    batch_sz = config.GPU_BATCH"
    NEW_INIT = """\
    outputs = []
    batch_sz = config.GPU_BATCH

    # ── 진행 바 초기화 (label 지정 시 활성화) ────────────────────
    import sys as _sys_pb, time as _time_pb
    if n_total is None:
        n_total = len(theta_batch)
    _bar_width = 20
    _lbl = str(label) if label is not None else ""
    _t_start_pb = _time_pb.perf_counter()
    _n_done = 0

    def _fmt_t(s):
        m = int(s) // 60
        return f"{m:02d}:{int(s) % 60:02d}"

    if label is not None:
        _n_chunks_total = math.ceil(n_total / config.GPU_BATCH) if n_total > 0 else 1
        print(
            f"\\n[Step 2] {_lbl}  N_SIM={n_total:,}  GPU_BATCH={config.GPU_BATCH:,}"
            f"  n_chunks={_n_chunks_total}",
            flush=True,
        )
"""
    if OLD_INIT not in src:
        print("✗ 'outputs = []' 블록을 찾지 못함")
        sys.exit(1)
    src = src.replace(OLD_INIT, NEW_INIT, 1)
    print("✓ STEP 2: 진행 바 초기화 코드 삽입 완료")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: success path 슬라이싱 루프 교체
    # (실제 코드: for i in range(csz): outputs.append(result[:, :, i]))
    # ══════════════════════════════════════════════════════════════
    OLD_SUCCESS = """\
        # Success path: split per-sim outputs
        for i in range(csz):
            outputs.append(result[:, :, i])

        # OPT-5: keep pool warm on H100; trim only above watermark.
        _trim_memory_pool(cp)
        _emit_chunk_log(chunk_i, csz, chunk_start, pre_count)"""

    NEW_SUCCESS = """\
        # Success path: split per-sim outputs + \x5cr progress bar
        _display_stride = max(1, min(512, csz // 50))
        for _si in range(csz):
            outputs.append(result[:, :, _si])
            _n_done += 1
            if label is not None and (
                _n_done % _display_stride == 0 or _n_done >= n_total
            ):
                _elapsed = _time_pb.perf_counter() - _t_start_pb
                _pct   = 100.0 * _n_done / n_total
                _speed = _n_done / _elapsed if _elapsed > 1e-6 else 0.0
                _eta   = (n_total - _n_done) / _speed if _speed > 1e-6 else 0.0
                _f     = int(_bar_width * _n_done / n_total)
                _b     = "\u2593" * _f + "\u2591" * (_bar_width - _f)
                _line  = (
                    f"\\r  {_lbl:<12s}"
                    f" |{_b}|"
                    f" {_n_done:>6,}/{n_total:>6,}"
                    f" |{_pct:5.1f}%"
                    f" |{_speed:5.1f} sim/s"
                    f" | elapsed {_fmt_t(_elapsed)}"
                    f" | ETA {_fmt_t(_eta)}"
                    f"   "
                )
                _sys_pb.stdout.write(_line)
                _sys_pb.stdout.flush()

        # OPT-5: keep pool warm on H100; trim only above watermark.
        _trim_memory_pool(cp)
        _emit_chunk_log(chunk_i, csz, chunk_start, pre_count)"""

    if OLD_SUCCESS not in src:
        print("✗ success path 루프를 찾지 못함")
        print("  아래 패턴이 wc_runner.py에 있는지 확인하세요:")
        print("    # Success path: split per-sim outputs")
        print("    for i in range(csz):")
        sys.exit(1)
    src = src.replace(OLD_SUCCESS, NEW_SUCCESS, 1)
    print("✓ STEP 3: success path 진행 바 삽입 완료")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: fallback path에도 _n_done += 1 삽입
    # (실제 코드: outputs.append(r_single[:, :, 0]) 바로 뒤)
    # ══════════════════════════════════════════════════════════════
    OLD_FALLBACK = "                outputs.append(r_single[:, :, 0])\n                if (r + 1) % max(1, csz // 4) == 0 or r + 1 == csz:"
    NEW_FALLBACK = """\
                outputs.append(r_single[:, :, 0])
                _n_done += 1
                if label is not None and (
                    _n_done % max(1, csz // 50) == 0 or _n_done >= n_total
                ):
                    _elapsed = _time_pb.perf_counter() - _t_start_pb
                    _pct   = 100.0 * _n_done / n_total
                    _speed = _n_done / _elapsed if _elapsed > 1e-6 else 0.0
                    _eta   = (n_total - _n_done) / _speed if _speed > 1e-6 else 0.0
                    _f     = int(_bar_width * _n_done / n_total)
                    _b     = "\u2593" * _f + "\u2591" * (_bar_width - _f)
                    _line  = (
                        f"\\r  {_lbl:<12s}"
                        f" |{_b}| (fallback)"
                        f" {_n_done:>6,}/{n_total:>6,}"
                        f" |{_pct:5.1f}%"
                        f" |{_speed:5.1f} sim/s"
                        f" | elapsed {_fmt_t(_elapsed)}"
                        f"   "
                    )
                    _sys_pb.stdout.write(_line)
                    _sys_pb.stdout.flush()
                if (r + 1) % max(1, csz // 4) == 0 or r + 1 == csz:"""

    if OLD_FALLBACK not in src:
        print("  ⚠ fallback 루프 패턴 불일치 — fallback 진행 바 생략 (선택사항)")
    else:
        src = src.replace(OLD_FALLBACK, NEW_FALLBACK, 1)
        print("✓ STEP 4: fallback path 진행 바 삽입 완료")

    # ══════════════════════════════════════════════════════════════
    # STEP 5: return outputs 직전에 DONE 요약 삽입
    # ══════════════════════════════════════════════════════════════
    OLD_RETURN = "\n    return outputs\n"
    NEW_RETURN = """
    # ── 완료 요약 ─────────────────────────────────────────────────
    if label is not None:
        _total_t = _time_pb.perf_counter() - _t_start_pb
        _spd = len(outputs) / _total_t if _total_t > 1e-6 else 0.0
        print()
        print(
            f"  {_lbl} DONE"
            f" | collected={len(outputs):,}/{n_total:,}"
            f" | dropped={n_total - len(outputs):,}"
            f" | {_spd:.1f} sim/s avg"
            f" | elapsed {_fmt_t(_total_t)}",
            flush=True,
        )

    return outputs
"""
    if OLD_RETURN not in src:
        print("✗ 'return outputs' 를 찾지 못함")
        sys.exit(1)
    src = src.replace(OLD_RETURN, NEW_RETURN, 1)
    print("✓ STEP 5: DONE 요약 코드 삽입 완료")

    # ══════════════════════════════════════════════════════════════
    # 저장 + 컴파일 확인
    # ══════════════════════════════════════════════════════════════
    WC.write_text(src, encoding="utf-8")
    print("\n파일 저장 완료. 컴파일 확인 중...")

    try:
        compile(src, str(WC), "exec")
        print("✓ 컴파일 OK")
    except SyntaxError as e:
        print(f"✗ 컴파일 실패: {e}")
        print(f"  라인 {e.lineno}: {e.text}")
        print("백업으로 복구 중...")
        shutil.copy2(bak, WC)
        print("복구 완료.")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════
    # 최종 검증
    # ══════════════════════════════════════════════════════════════
    final = WC.read_text(encoding="utf-8")
    print("\n[검증]")
    checks = [
        ("label=None",       "label 파라미터"),
        ("n_total=None",     "n_total 파라미터"),
        ("_display_stride",  "_display_stride"),
        ("_n_done += 1",     "_n_done += 1 (per-sim)"),
        ("_fmt_t",           "_fmt_t 헬퍼"),
        ("DONE",             "DONE 요약"),
    ]
    all_ok = True
    for pattern, desc in checks:
        if pattern in final:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} 없음")
            all_ok = False

    print()
    if all_ok:
        print("✅ wc_runner.py 패치 완료!")
        print()
        print("진행 바 테스트 (GPU 있는 경우):")
        print("  python -c \"")
        print("  import numpy as np, config")
        print("  from simulation.wc_runner import simulate_gpu_batch")
        print("  sc = np.random.rand(115,115).astype(np.float64)")
        print("  theta = np.random.rand(20,4).astype(np.float32)")
        print("  theta[:,0] = 1.5; theta[:,1] = 0.5")
        print("  theta[:,2] = 0.3; theta[:,3] = 0.3")
        print("  simulate_gpu_batch(sc, theta, config.STAGE1_PARAMS,")
        print("                     label='test', n_total=20)\"")
    else:
        print("⚠ 일부 항목 없음 — 위 오류 확인 필요")


if __name__ == "__main__":
    main()
