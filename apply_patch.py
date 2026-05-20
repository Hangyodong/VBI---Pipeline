#!/usr/bin/env python3
"""
VBI 코드베이스 패치 스크립트
==============================================
작업 1: main.ipynb에 Pipeline Configuration 셀 추가
작업 2: simulation/wc_runner.py 진행 바 수정 (GPU_BATCH=N_SIM 대응)

실행 방법:
  cd /scratch/home/wog3597/vbi
  python apply_patch.py

또는 경로를 직접 지정:
  python apply_patch.py --root /scratch/home/wog3597/vbi
"""

import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────

def get_root(args_root=None):
    if args_root:
        return Path(args_root)
    # 스크립트 위치 기준으로 추정
    candidates = [
        Path.cwd(),
        Path(__file__).parent,
        Path("/scratch/home/wog3597/vbi"),
    ]
    for c in candidates:
        if (c / "config.py").exists() and (c / "main.ipynb").exists():
            return c
    raise FileNotFoundError(
        "VBI 루트를 찾을 수 없습니다. --root 옵션으로 지정하세요.\n"
        "예: python apply_patch.py --root /scratch/home/wog3597/vbi"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def backup(path: Path, suffix: str) -> Path:
    bak = path.parent / (path.name + suffix)
    shutil.copy2(path, bak)
    print(f"  백업: {bak}")
    return bak


def notebook_cell(cell_type, source_lines):
    """Jupyter 노트북 셀 JSON 객체 생성."""
    src = []
    for i, line in enumerate(source_lines):
        src.append(line if line.endswith("\n") or i == len(source_lines) - 1
                   else line + "\n")
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def read_config_params(config_path: Path) -> dict:
    """config.py에서 실제 파라미터 값을 읽어 반환."""
    spec = importlib.util.spec_from_file_location("_config_read", config_path)
    mod = importlib.util.module_from_spec(spec)
    # 일부 config.py는 torch import 시도 — 없어도 계속 진행
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass

    params = {}
    safe_keys = [
        "N_SIM", "GPU_BATCH", "T_END", "T_CUT", "DT", "DECIMATE",
        "N_TRAIN", "N_VAL", "N_TEST", "SEED",
        "SBI_DEVICE",
        "PCA_DIM_FC", "EMBED_DIM", "EMBED_HIDDEN",
        "NDE_MODEL", "NDE_HIDDEN", "NDE_TRANSFORMS", "N_POSTERIOR",
        "N_TEST_RESIM", "BOOTSTRAP_N",
        "N_SIM_S2", "SENS_THRESHOLD", "SHR_THRESHOLD",
        "TR_SEC", "N_REGIONS", "ANALYSIS_BOLD_T", "FC_DIM",
    ]
    for k in safe_keys:
        v = getattr(mod, k, None)
        if v is not None:
            params[k] = v
    return params


# ──────────────────────────────────────────────────────────────────────────────
# 작업 1: main.ipynb config 셀 삽입
# ──────────────────────────────────────────────────────────────────────────────

def build_config_cell_source(params: dict) -> list:
    """config cell 소스 라인 목록 생성."""

    def v(key, default="???"):
        val = params.get(key, default)
        if isinstance(val, str):
            return f'"{val}"'
        return repr(val)

    lines = [
        "# ================================================================\n",
        "# Pipeline Configuration — config.py 기본값을 여기서 수정하세요.\n",
        "# 이 셀을 먼저 실행하면 이후 모든 셀에 반영됩니다.\n",
        "# WC_FIXED 과학 상수(c_ee, tau_e 등)는 수정하지 마세요.\n",
        "# ================================================================\n",
        "import config\n",
        "\n",
        "# ── 시뮬레이션 ────────────────────────────────────────────────\n",
        f"config.N_SIM        = {params.get('N_SIM', 50000):<10}  # 피험자당 학습 시뮬레이션 수\n",
        f"config.GPU_BATCH    = {params.get('GPU_BATCH', 50000):<10}  # GPU 1회 호출당 시뮬 수 (=N_SIM → 단일 배치)\n",
        f"config.T_END        = {params.get('T_END', 300000.0):<10}  # 전체 시뮬 시간 (ms)\n",
        f"config.T_CUT        = {params.get('T_CUT', 60000.0):<10}  # 과도기 제거 구간 (ms)\n",
        f"config.DT           = {params.get('DT', 0.5):<10}  # 적분 스텝 (ms)\n",
        f"config.DECIMATE     = {params.get('DECIMATE', 2):<10}  # 신경 스텝 서브샘플링 비율\n",
        "\n",
        "# ── 피험자 분할 ───────────────────────────────────────────────\n",
        f"config.N_TRAIN      = {params.get('N_TRAIN', 4):<10}  # 학습 피험자 수\n",
        f"config.N_VAL        = {params.get('N_VAL', 2):<10}  # 검증 피험자 수\n",
        f"config.N_TEST       = {params.get('N_TEST', 2):<10}  # 테스트 피험자 수\n",
        f"config.SEED         = {params.get('SEED', 42):<10}  # 난수 시드\n",
        "\n",
        "# ── GPU / 디바이스 ────────────────────────────────────────────\n",
        f'config.SBI_DEVICE   = {v("SBI_DEVICE", "cuda"):<10}  # "cuda" 또는 "cpu"\n',
        "\n",
        "# ── Feature Pipeline ──────────────────────────────────────────\n",
        f"config.PCA_DIM_FC   = {params.get('PCA_DIM_FC', 300):<10}  # FC PCA 출력 차원\n",
        f"config.EMBED_DIM    = {params.get('EMBED_DIM', 128):<10}  # MLP 임베딩 출력 차원\n",
        f"config.EMBED_HIDDEN = {params.get('EMBED_HIDDEN', 512):<10}  # MLP 임베딩 히든 차원\n",
        "\n",
        "# ── SNPE-C 학습 ───────────────────────────────────────────────\n",
        f'config.NDE_MODEL        = {v("NDE_MODEL", "maf"):<10}  # 정규화 흐름 모델\n',
        f"config.NDE_HIDDEN       = {params.get('NDE_HIDDEN', 128):<10}  # MAF 히든 유닛\n",
        f"config.NDE_TRANSFORMS   = {params.get('NDE_TRANSFORMS', 8):<10}  # MAF 변환 수\n",
        f"config.N_POSTERIOR      = {params.get('N_POSTERIOR', 2000):<10}  # 사후분포 샘플 수\n",
        "\n",
        "# ── 평가 ──────────────────────────────────────────────────────\n",
        f"config.N_TEST_RESIM  = {params.get('N_TEST_RESIM', 50):<10}  # 피험자당 사후 재시뮬 횟수\n",
        f"config.BOOTSTRAP_N   = {params.get('BOOTSTRAP_N', 1000):<10}  # 부트스트랩 CI 반복 수\n",
        "\n",
        "# ── Stage 2 ───────────────────────────────────────────────────\n",
        f"config.N_SIM_S2         = {params.get('N_SIM_S2', 50000):<10}  # Stage 2 피험자당 시뮬 수\n",
        f"config.SENS_THRESHOLD   = {params.get('SENS_THRESHOLD', 0.5):<10}  # theta_bad 감도 임계값\n",
        f"config.SHR_THRESHOLD    = {params.get('SHR_THRESHOLD', 0.2):<10}  # theta_bad 수축 임계값\n",
        "\n",
        "# ── 파생 상수 자동 재계산 (수정 금지) ────────────────────────\n",
        "config.ANALYSIS_BOLD_T = int(\n",
        "    (config.T_END - config.T_CUT)\n",
        "    / (config.DT * config.DECIMATE)\n",
        "    / (config.TR_SEC * 1000.0 / (config.DT * config.DECIMATE))\n",
        ")\n",
        "config.FC_DIM = config.N_REGIONS * (config.N_REGIONS - 1) // 2\n",
        "\n",
        "# ── 현재 설정 출력 ────────────────────────────────────────────\n",
        "_cfg_items = [\n",
        '    ("N_SIM",            config.N_SIM),\n',
        '    ("GPU_BATCH",        config.GPU_BATCH),\n',
        '    ("N_TRAIN/VAL/TEST", f"{config.N_TRAIN}/{config.N_VAL}/{config.N_TEST}"),\n',
        '    ("T_END (ms)",       config.T_END),\n',
        '    ("T_CUT (ms)",       config.T_CUT),\n',
        '    ("DT (ms)",          config.DT),\n',
        '    ("ANALYSIS_BOLD_T",  config.ANALYSIS_BOLD_T),\n',
        '    ("N_REGIONS",        config.N_REGIONS),\n',
        '    ("FC_DIM",           config.FC_DIM),\n',
        '    ("PCA_DIM_FC",       config.PCA_DIM_FC),\n',
        '    ("EMBED_DIM",        config.EMBED_DIM),\n',
        '    ("SBI_DEVICE",       config.SBI_DEVICE),\n',
        '    ("NDE_MODEL",        config.NDE_MODEL),\n',
        '    ("NDE_TRANSFORMS",   config.NDE_TRANSFORMS),\n',
        '    ("N_POSTERIOR",      config.N_POSTERIOR),\n',
        '    ("N_TEST_RESIM",     config.N_TEST_RESIM),\n',
        '    ("N_SIM_S2",         config.N_SIM_S2),\n',
        '    ("SENS_THRESHOLD",   config.SENS_THRESHOLD),\n',
        '    ("SHR_THRESHOLD",    config.SHR_THRESHOLD),\n',
        "]\n",
        'print("=" * 56)\n',
        'print("  Pipeline Configuration")\n',
        'print("=" * 56)\n',
        "for _k, _v in _cfg_items:\n",
        '    print(f"  {_k:<22s} = {_v}")\n',
        "\n",
        "if config.GPU_BATCH >= config.N_SIM:\n",
        '    print(f"\\n  ⚡ 단일 배치 모드")\n',
        '    print(f"     {config.N_SIM:,}개를 1회 GPU 호출로 처리")\n',
        '    print(f"     진행 바 업데이트: 약 {config.N_SIM // 50:,}개마다")\n',
        "else:\n",
        "    _nc = -(-config.N_SIM // config.GPU_BATCH)\n",
        '    print(f"\\n  ℹ  멀티 배치 모드: {_nc}청크 × {config.GPU_BATCH:,}개")\n',
        'print("=" * 56)\n',
    ]
    return lines


def patch_notebook(root: Path):
    nb_path = root / "main.ipynb"
    if not nb_path.exists():
        print(f"  ✗ main.ipynb를 찾을 수 없음: {nb_path}")
        return False

    # config.py에서 실제 파라미터 읽기
    config_path = root / "config.py"
    params = {}
    if config_path.exists():
        try:
            params = read_config_params(config_path)
            print(f"  config.py에서 {len(params)}개 파라미터 읽음")
        except Exception as e:
            print(f"  config.py 읽기 오류 (기본값 사용): {e}")
    else:
        print("  config.py 없음 — 기본값 사용")

    # 백업
    backup(nb_path, ".bak_before_config_cell")

    # 노트북 읽기
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    old_count = len(cells)
    print(f"  현재 셀 수: {old_count}")

    # 이미 config 셀이 있는지 확인
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if "Pipeline Configuration" in src and "config.N_SIM" in src:
            print(f"  ✓ config 셀이 이미 cell[{i}]에 존재 — 건너뜀")
            return True

    # cell[2] 다음에 삽입 (Setup 셀 바로 뒤)
    insert_idx = 3

    md_cell = notebook_cell("markdown", [
        "## Pipeline Configuration\n",
        "\n",
        "config.py 기본값을 여기서 수정하세요. "
        "**이 셀을 먼저 실행**하면 이후 모든 셀에 반영됩니다.\n",
    ])

    code_src = build_config_cell_source(params)
    code_cell = notebook_cell("code", code_src)

    cells.insert(insert_idx, code_cell)
    cells.insert(insert_idx, md_cell)

    nb["cells"] = cells
    new_count = len(cells)

    # 컴파일 확인
    code_src_str = "".join(code_src)
    try:
        compile(code_src_str, "<config_cell>", "exec")
        print("  config 셀 컴파일 OK")
    except SyntaxError as e:
        print(f"  ✗ config 셀 컴파일 실패: {e}")
        shutil.copy2(nb_path.parent / (nb_path.name + ".bak_before_config_cell"), nb_path)
        return False

    # 저장
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  ✓ 셀 삽입 완료: {old_count} → {new_count}개")
    print(f"  ✓ cell[{insert_idx}] markdown: Pipeline Configuration")
    print(f"  ✓ cell[{insert_idx+1}] code: config 파라미터 ({len(code_src)}줄)")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 작업 2: wc_runner.py 진행 바 수정
# ──────────────────────────────────────────────────────────────────────────────

# 삽입할 진행 바 코드 블록
PROGRESS_INIT_CODE = '''\
        # ── 진행 바 초기화 ─────────────────────────────────────────
        import sys as _sys_pb, time as _time_pb
        if n_total is None:
            n_total = len(theta_batch)
        _bar_width = 20
        _lbl = str(label) if label is not None else ""
        _t_start = _time_pb.perf_counter()
        _n_done = 0

        def _fmt_t(s):
            m = int(s) // 60
            return f"{m:02d}:{int(s) % 60:02d}"
'''

PROGRESS_SUBJECT_HEADER = '''\
            # ── 피험자 헤더 ────────────────────────────────────────
            if label is not None:
                _sc_sp = float((_sc > 0).sum()) / _sc.size if hasattr(_sc, 'size') else 0.0
                _dly_max = float(_dly.max()) if _dly is not None and hasattr(_dly, 'max') else 0.0
                _n_chunks_total = -(-len(theta_batch) // max(1, csz))
                print(
                    f"\\n[Step 2] {_lbl}  N_SIM={n_total:,}  GPU_BATCH={csz:,}"
                    f"\\n         n_chunks={_n_chunks_total}",
                    flush=True,
                )
'''

PROGRESS_LOOP_CODE = '''\
                # ── per-sim 진행 바 ─────────────────────────────────
                _display_stride = max(1, min(512, csz // 50))
                for _pb_i in range(len(_result_list)):
                    _n_done += 1
                    if label is not None and (
                        _n_done % _display_stride == 0 or _n_done >= n_total
                    ):
                        _elapsed = _time_pb.perf_counter() - _t_start
                        _pct = 100.0 * _n_done / n_total
                        _speed = _n_done / _elapsed if _elapsed > 1e-6 else 0.0
                        _eta = (n_total - _n_done) / _speed if _speed > 1e-6 else 0.0
                        _f = int(_bar_width * _n_done / n_total)
                        _b = "\\u2593" * _f + "\\u2591" * (_bar_width - _f)
                        _line = (
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
'''

PROGRESS_DONE_CODE = '''\
        # ── 완료 요약 ──────────────────────────────────────────────
        if label is not None:
            _total_t = _time_pb.perf_counter() - _t_start
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
'''


def patch_wc_runner(root: Path):
    wc_path = root / "simulation" / "wc_runner.py"
    if not wc_path.exists():
        print(f"  ✗ wc_runner.py를 찾을 수 없음: {wc_path}")
        return False

    src = wc_path.read_text(encoding="utf-8")

    # ── 이미 패치됐는지 확인 ──────────────────────────────────────
    already_patched = "_display_stride" in src and "_n_done += 1" in src
    has_label_kwarg = "label=None" in src

    if already_patched:
        print("  ✓ 진행 바가 이미 패치됨 (_display_stride 존재)")
        return True

    # 백업
    backup(wc_path, ".bak_before_progress_bar")

    # ── simulate_gpu_batch 시그니처에 label, n_total 추가 ─────────
    if not has_label_kwarg:
        # def simulate_gpu_batch(...): 패턴 찾기
        sig_pattern = re.compile(
            r'(def simulate_gpu_batch\([^)]+)\)',
            re.DOTALL
        )
        m = sig_pattern.search(src)
        if m:
            old_sig = m.group(0)
            # 마지막 ) 앞에 새 kwargs 삽입
            new_sig = old_sig[:-1] + ",\n                          label=None, n_total=None)"
            src = src.replace(old_sig, new_sig, 1)
            print("  simulate_gpu_batch 시그니처에 label, n_total 추가")
        else:
            print("  ✗ simulate_gpu_batch 시그니처를 찾을 수 없음")
            return False

    # ── 함수 본문 시작 직후에 진행 바 초기화 코드 삽입 ───────────
    # "def simulate_gpu_batch" 다음 첫 번째 실제 코드 라인 앞에 삽입
    if "_t_start" not in src:
        # outputs = [] 라인 찾기 (함수 본문 초반)
        outputs_init = re.search(r'(\s+outputs\s*=\s*\[\])', src)
        if outputs_init:
            insert_pt = outputs_init.start()
            src = src[:insert_pt] + PROGRESS_INIT_CODE + src[insert_pt:]
            print("  진행 바 초기화 코드 삽입 (outputs = [] 앞)")
        else:
            print("  ✗ outputs = [] 라인을 찾을 수 없음 — 수동 확인 필요")

    # ── result 슬라이싱 루프 교체 ─────────────────────────────────
    # outputs.extend(...) 또는 outputs.append(result[:, :, i]) 패턴
    slice_patterns = [
        # extend 패턴
        r'(\s+)(outputs\.extend\(\s*result\[.*?\]\s*for\s+\w+\s+in\s+range\(\w+\)\s*\))',
        # append 패턴
        r'(\s+)(for\s+(\w+)\s+in\s+range\((\w+)\)\s*:\s*\n\s+outputs\.append\(result\[:.*?\]\))',
    ]

    replaced = False
    for pat in slice_patterns:
        m = re.search(pat, src, re.DOTALL)
        if m:
            old_block = m.group(0)
            indent = "        "  # 8칸
            # 교체 코드
            new_block = f"""
{indent}_result_list = [result[:, :, _pb_i] for _pb_i in range(result.shape[2] if hasattr(result, 'shape') and len(result.shape) > 2 else len(result))]
{indent}outputs.extend(_result_list)
{PROGRESS_LOOP_CODE}"""
            src = src.replace(old_block, new_block, 1)
            replaced = True
            print("  result 슬라이싱 루프 교체 완료")
            break

    if not replaced:
        # fallback: extend를 찾아 뒤에 진행 코드 삽입
        extend_m = re.search(r'(outputs\.extend\([^\n]+\))', src)
        if extend_m:
            old = extend_m.group(0)
            new = (old +
                   "\n        _result_list = list(outputs[-csz:]) if 'csz' in dir() else []\n" +
                   PROGRESS_LOOP_CODE)
            src = src.replace(old, new, 1)
            replaced = True
            print("  (fallback) extend 뒤에 진행 코드 삽입")

    if not replaced:
        print("  ✗ outputs.extend/append 패턴을 찾을 수 없음")
        print("    simulation/wc_runner.py를 직접 확인하세요.")

    # ── _n_done += csz 제거 ───────────────────────────────────────
    if "_n_done += csz" in src:
        src = src.replace("        _n_done += csz\n", "", 1)
        print("  구버전 _n_done += csz 제거")

    # ── return outputs 직전에 DONE 요약 삽입 ─────────────────────
    if "_total_t" not in src:
        return_pattern = re.search(r'(\n\s{4,8}return outputs)', src)
        if return_pattern:
            insert_pos = return_pattern.start()
            src = src[:insert_pos] + "\n" + PROGRESS_DONE_CODE + src[insert_pos:]
            print("  완료 요약 코드 삽입 (return outputs 앞)")

    # 저장
    wc_path.write_text(src, encoding="utf-8")

    # 컴파일 확인
    try:
        compile(src, str(wc_path), "exec")
        print("  ✓ wc_runner.py 컴파일 OK")
    except SyntaxError as e:
        print(f"  ✗ 컴파일 실패: {e}")
        print("  → 백업으로 복구 중...")
        shutil.copy2(str(wc_path) + ".bak_before_progress_bar", wc_path)
        print("  복구 완료. wc_runner.py를 수동으로 확인하세요.")
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# 검증
# ──────────────────────────────────────────────────────────────────────────────

def verify(root: Path):
    print("\n[검증]")
    ok = True

    # main.ipynb config 셀 확인
    nb_path = root / "main.ipynb"
    if nb_path.exists():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        found_config = False
        for i, c in enumerate(cells):
            src = "".join(c["source"])
            if "Pipeline Configuration" in src and "config.N_SIM" in src:
                found_config = True
                print(f"  ✓ config 셀: cell[{i}] ({c['cell_type']})")
                # 컴파일 확인
                if c["cell_type"] == "code":
                    try:
                        compile(src, f"<cell[{i}]>", "exec")
                        print(f"    컴파일 OK")
                    except SyntaxError as e:
                        print(f"    ✗ 컴파일 실패: {e}")
                        ok = False
                break
        if not found_config:
            print("  ✗ config 셀을 찾을 수 없음")
            ok = False

        # Debug Cell 위치 확인
        for i, c in enumerate(cells):
            src = "".join(c["source"])
            if "## Integrated VBI Pipeline Debug Cell" in src:
                print(f"  ✓ Integrated Debug Cell: cell[{i}] 유지됨")
                break

        print(f"  총 셀 수: {len(cells)}")

    # wc_runner.py 확인
    wc_path = root / "simulation" / "wc_runner.py"
    if wc_path.exists():
        src = wc_path.read_text(encoding="utf-8")
        checks = [
            ("label=None", "label 파라미터"),
            ("n_total=None", "n_total 파라미터"),
            ("_display_stride", "_display_stride 변수"),
            ("_n_done += 1", "_n_done += 1 (per-sim)"),
        ]
        for pattern, desc in checks:
            if pattern in src:
                print(f"  ✓ {desc}")
            else:
                print(f"  ✗ {desc} 없음")
                ok = False
        if "_n_done += csz" in src:
            print("  ✗ 구버전 _n_done += csz 아직 남아있음")
            ok = False

    return ok


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VBI 코드베이스 패치 스크립트")
    parser.add_argument("--root", help="VBI 루트 디렉토리 경로")
    parser.add_argument("--task", choices=["1", "2", "all"], default="all",
                        help="실행할 작업 (기본값: all)")
    parser.add_argument("--verify-only", action="store_true",
                        help="패치 없이 현재 상태만 확인")
    args = parser.parse_args()

    try:
        root = get_root(args.root)
    except FileNotFoundError as e:
        print(f"오류: {e}")
        sys.exit(1)

    print(f"\n VBI 패치 스크립트")
    print(f" 루트 디렉토리: {root}")
    print("=" * 60)

    if args.verify_only:
        verify(root)
        return

    results = {}

    if args.task in ("1", "all"):
        print("\n[작업 1] main.ipynb — Pipeline Configuration 셀 추가")
        results["task1"] = patch_notebook(root)

    if args.task in ("2", "all"):
        print("\n[작업 2] simulation/wc_runner.py — 진행 바 수정")
        results["task2"] = patch_wc_runner(root)

    print()
    verify(root)

    print("\n" + "=" * 60)
    all_ok = all(results.values())
    if all_ok:
        print("✅ 모든 패치 완료")
        print()
        print("다음 단계:")
        print("  1. main.ipynb 열기")
        print("  2. cell[3] (Pipeline Configuration) 먼저 실행")
        print("  3. 나머지 셀 순서대로 실행")
        print()
        print("진행 바 확인 (GPU 있는 경우):")
        print("  python -c \"")
        print("  import config")
        print("  from simulation.wc_runner import simulate_gpu_batch")
        print("  import numpy as np")
        print("  sc = np.random.rand(115,115).astype(np.float64)")
        print("  theta = np.random.rand(100,4).astype(np.float32)")
        print("  simulate_gpu_batch(sc, theta, ['P','Q','g_e','g_i'],")
        print("                     label='test', n_total=100)\"")
    else:
        print("⚠️  일부 패치 실패 — 위 오류 메시지를 확인하세요")
        print("   수동 수정이 필요할 수 있습니다.")


if __name__ == "__main__":
    main()
