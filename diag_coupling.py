"""cuBNM vs VBI 결합(coupling) 진단 + g_e 스윕.

GPU 서버 터미널에서:   python diag_coupling.py
또는 노트북 셀에서:    !python diag_coupling.py

핵심 질문: g_e(전역결합)를 키우면 시뮬 FC 세기(std)가 커지는가?
  - VBI는 커지는데 cuBNM은 0.088 고정  -> cuBNM WCVBI 결합 버그
  - 둘 다 고정                          -> working point 문제(모델 자체)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# ── 1. config + 경로 (main_mouse.py와 동일 규칙) ────────────────────────
import species_configs.mouse_config as _sc
import config
_sc.DATA_DIR   = "/scratch/home/wog3597/vbi"
_sc.FC_PATH    = "/scratch/home/wog3597/vbi/MPTP_FC_115.mat"
_sc.SC_PATH    = "/scratch/home/wog3597/vbi/MPTP_SC_115.mat"
_sc.TSV_PATH   = "/scratch/home/wog3597/vbi/participants.tsv"
_sc.ATLAS_PATH = "/scratch/home/wog3597/vbi/atlas_115_labels.txt"
_sc.BOLD_PATH  = None
_sc.OUTPUT_DIR = "./output_mouse_mptp"
config.apply_species_config(_sc)
for k in ["DATA_DIR", "FC_PATH", "SC_PATH", "TSV_PATH",
          "ATLAS_PATH", "BOLD_PATH", "OUTPUT_DIR"]:
    setattr(config, k, getattr(_sc, k))

import data_loader
from features.fc import compute_fc, fc_to_upper_tri

# ── 2. 데이터 로드 (SC + 관측 FC) ───────────────────────────────────────
out = data_loader.load_raw_data()
df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
subs = data_loader.get_target_subjects(df, fc_ids, sc_ids)
tr, va, te = data_loader.three_way_split(subs)
sd = data_loader.load_all_subjects(
    tr[:1], fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids)
sid = tr[0]
SC = sd[sid]["sc"]
dly = sd[sid]["delays"]
iu = np.triu_indices(config.N_REGIONS, 1)
realv = sd[sid]["fc"][iu]
fin = np.isfinite(realv)
real_z = np.arctanh(np.clip(realv[fin], -0.999, 0.999))
target_std = real_z.std()
print(f"\n[target] 실제 FC std = {target_std:.3f}  (subject {sid})")


def fc_std(bold):
    return float(np.nanstd(fc_to_upper_tri(compute_fc(bold))))


def fc_rho(bold):
    v = fc_to_upper_tri(compute_fc(bold))[fin]
    return float(np.corrcoef(v, real_z)[0, 1])


# ── 3. cuBNM vs VBI 결합 비교 ───────────────────────────────────────────
print("\n" + "=" * 56)
print("  [A] cuBNM vs VBI : g_e(전역결합) 증가 시 FC 세기 변화")
print("=" * 56)
from simulation.wc_runner import simulate_single as sim_vbi
from cuBNM.simulate import simulate_single as sim_cubnm

print(f"{'g_e':>5} | {'VBI_std':>8} {'VBI_ρ':>7} | {'cuBNM_std':>10} {'cuBNM_ρ':>8}")
for ge in [0.0, 1.0, 2.0, 4.0]:
    p = {"g_e": ge, "g_i": 0.5, "c_ei": 12.0}
    try:
        bv = sim_vbi(SC, p, n_repeat=1, delays=dly)[0]
        vs, vr = fc_std(bv), fc_rho(bv)
    except Exception as e:
        vs, vr = float("nan"), float("nan")
        print(f"   VBI 실패: {type(e).__name__}: {e}")
    try:
        bc = sim_cubnm(SC, p, n_repeat=1)[0]
        cs, cr = fc_std(bc), fc_rho(bc)
    except Exception as e:
        cs, cr = float("nan"), float("nan")
        print(f"   cuBNM 실패: {type(e).__name__}: {e}")
    print(f"{ge:>5} | {vs:>8.3f} {vr:>+7.3f} | {cs:>10.3f} {cr:>+8.3f}")

print("\n해석:")
print("  - VBI_std는 g_e↑에 따라 커지는데 cuBNM_std가 0.088 고정 → cuBNM 결합 버그")
print("  - 둘 다 g_e 무관하게 고정          → working point 문제(양쪽 동일)")
print("  - 둘 다 g_e↑에 std↑               → 결합 정상, prior/working point 재탐색")
print(f"\n  (목표 실제 FC std = {target_std:.3f})")
