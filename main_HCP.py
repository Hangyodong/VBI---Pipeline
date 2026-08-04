#!/usr/bin/env python
# coding: utf-8

# # HCP RWW-EIB-FFI VBI-SBI Pipeline (381 region)
# 
# HCP SC/FC 입력. Reduced Wong-Wang E/I + excitatory feedforward inhibition (RWW-EIB-FFI) 추론.
# 사용 subject 수 = `N_SUBJECTS` (subject-id 작은 것부터). 데이터 로더 = `data_loader_hcp`.
# 

# ## Setup
# 
# **Parameters 섹션**의 `PipelineConfig(...)` 인자만 수정하세요.
# 
# 커널 재시작 후 이 셀 하나만 실행하면:
# - 모듈 자동 reload
# - `config` 전체 동기화
# - numpy / torch seed 설정
# - `config.print_config()` 요약 출력
# 
# 셀 출력은 노트북(`.ipynb`) 자체에 저장됩니다.

# In[1]:


# =============================================================
# Setup (HCP + RWW-EIB-FFI) — 파라미터를 여기서 수정하세요
# =============================================================
from pipeline_setup import PipelineConfig, setup_pipeline
import os

# SC scaling = max-normalization only (no log1p). CPU mean-field shows this is
# the ONLY scaling where FIC reaches its target (<I_E>=0.377, 0% saturation) and
# coupling shapes FC best (simFC~SC=0.170 vs 0.027 for log1p). Read by
# data_loader._scale_weights at SC load time.
os.environ.setdefault("VBI_SC_SCALE", "maxnorm")

def _envi(name, default):
    """Int override from env (for quick smoke runs); falls back to default."""
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default

# ── SMOKE toggle ─────────────────────────────────────────────
# SMOKE = tiny end-to-end run (few subjects, few sims) to check the pipeline
# wires + runs without crashing. Flip to False (or env SMOKE=0) for the REAL run.
# Only the SIZES change; dataset/model config (cabnp381 + tract delays +
# per-subject FC) is the same in both. Startup banner prints the active sizes.
SMOKE = (os.environ.get("SMOKE", "1") != "0")
if SMOKE:
    print(">>> SMOKE=1: tiny run (N_SUBJECTS=4/N_TRAIN=2/N_SIM=64). "
          "Set SMOKE=0 for the real run. <<<")

cfg = PipelineConfig(
    # ── Paths (HCP) ──────────────────────────────────────────
    DATA_DIR   = "/scratch/home/wog3597/vbi/HCP_Data",
    OUTPUT_DIR = "./output_hcp",
    FC_FILE    = "HCP_FC.mat",       # var 'C' (n,2): col0 id, col1 FC(381,381)
    SC_FILE    = os.environ.get("SC_FILE", "HCP_CABNP381_SC_first100.mat"),  # CAB-NP 381 SC

    # ── Atlas / regions ──────────────────────────────────────
    N_REGIONS  = 360,                # cortical-only (Glasser 360; drop 21 subcortical). FC_DIM auto = 64620

    # ── Subject pool & split ─────────────────────────────────
    N_SUBJECTS = _envi("N_SUBJECTS", 4 if SMOKE else 100),   # 사용 subject 수 (id 작은 것부터)
    N_TRAIN    = _envi("N_TRAIN", 2 if SMOKE else 70),       # real: was 80; free 10 for validation
    N_VAL      = _envi("N_VAL", 1 if SMOKE else 10),         # enables Step9/13 metrics + baseline
    N_TEST     = _envi("N_TEST", 1 if SMOKE else 20),
    SEED       = 42,

    # ── Simulation ───────────────────────────────────────────
    N_SIM      = _envi("N_SIM", 64 if SMOKE else 1_000),   # per-subject sims (real)
    GPU_BATCH  = _envi("GPU_BATCH", 64 if SMOKE else 2_000),

    # ── Simulation time (ms) — HCP rfMRI TR=0.72s; 875 TR total, 1min cut ─
    # Lengthened from 180s/166 TR toward HCP empirical FC length. T_END_MS MUST be a
    # whole number of SECONDS: runner_rwweib_2cpl does T_END/1000 -> seconds and cuBNM
    # rejects durations not expressible in whole ms (622.08s -> 622080.0000409 ms ->
    # ALL sims fail). 630000 = 630.0s = 875 TR (multiple of both 1000ms and 720ms).
    T_END_MS   = 864_000.0,          # 1200 TR total (864.0 s = HCP run length: 1200 x 0.72s)
    T_CUT_MS   =  30_000.0,          # cut first 30 s transient -> 1158 TR analyzed
    DT         = 1.0,                # RWW Euler step (ms)
    DECIMATE   = 720,                # neural stored dt = DT*DECIMATE = 720ms
    TR_SEC     = 0.72,               # BOLD sampling period (s) -> 1158 TR analysis

    # ── HRF ──────────────────────────────────────────────────
    # human HRF (was mouse A1=3/A2=7). Only used by hrf="vbi"; production = bw.
    HRF_A1 = 6.0, HRF_A2 = 16.0, HRF_L = 1.0, HRF_C = 0.167,
    HRF_LENGTH_SEC = 32.0, HRF_LENGTH_MS = 20_000.0,

    # ── Embedding (RegionTransformer + raw-FC passthrough) ──
    EMBED_DIM = 128,
    REGION_TRANSFORMER_HEADS = 4,
    REGION_TRANSFORMER_LAYERS = 2,
    REGION_TRANSFORMER_D_MODEL = 128,
    SELECTION_K = 300,

    # ── SBI ──────────────────────────────────────────────────
    SBI_DEVICE = "cuda",
    N_POSTERIOR = 2000, N_SBC = 200,
    NDE_HIDDEN = 128, NDE_TRANSFORMS = 8, N_TEST_RESIM = 10,
)

# =============================================================
# Init (수정하지 마세요)
# =============================================================
setup_pipeline(cfg)

import config

# -- RWW-EIB two connectome couplings (RWWEIB_2CPL) ---------------------------
# The equation-literal full WW: E driven by SC@S_E (gain g_LRE), I driven by
# SC@S_I (gain g_FFI), two INDEPENDENT couplings. Built via the multi-coupling
# cuBNM kernel surgery (conn_state_vars=[S_E,S_I]); validated by test_new_models
# (T4: g_FFI>0 -> 2CPL != single-coupling FFI, proving the SC@S_I path is live).
# No FIC, no delays (USE_DELAYS=False -> tract_length unused; delay is a later
# increment via INFERENCE_MODEL="rwweibdelay" + USE_DELAYS=True).
config.INFERENCE_MODEL = "rwweib2"
# Inferred (4): g_LRE/g_FFI (2 coupling gains) + I_o + sigma. Priors 3x the
# original width. Rest fixed (w_E/w_I/J_i/w_p at defaults).
config.STAGE1_PARAMS      = ["g_LRE", "g_FFI", "I_o", "sigma"]
config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
config.STAGE1_PRIOR_LOW   = [0.0, 0.0, 0.15, 0.0]
config.STAGE1_PRIOR_HIGH  = [9.0, 9.0, 0.60, 0.09]   # 3x: g_LRE,g_FFI U(0,9); I_o U(0.15,0.60); sigma U(0,0.09)
config.RWWEIB2_FIXED      = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
                             "J_N": 0.15, "lambda_IE": 1.0}
# Metric/target levers (data analysis: per-subj raw-FC SC-corr ~0.05; cortical-
# only + group-avg raises the achievable ceiling to ~0.2). Cortical-only is set
# via N_REGIONS=360 above; group-avg FC target via the flag below.
config.GROUP_AVG_FC       = (os.environ.get("GROUP_AVG_FC", "0") == "1")  # default per-subject FC; GROUP_AVG_FC=1 -> group-avg
# ── Region-wise latent inference (new mode; default OFF = homogeneous baseline) ──
# PARAMETER_MODE: "homogeneous" (baseline) | "latent_regionwise".
# In latent mode SNPE infers a low-dim latent z (BoxUniform[-1,1]^latent_dim);
# param_decoder turns z into per-region maps for [g_LRE,g_FFI,I_o,sigma]; cuBNM
# receives per-node matrices. See region_basis.py / param_decoder.py / PIPELINE.md.
config.PARAMETER_MODE     = os.environ.get("PARAMETER_MODE", "basis_regionwise")
# 4 region-wise params (g_LRE promoted global->regional; requires cuBNM rebuilt
# from the updated rww_eib_2cpl.yaml).
config.HETERO_PARAMS      = ["g_LRE", "g_FFI", "I_o", "sigma"]
config.HETERO_BOUNDS      = {"g_LRE": (0.0, 9.0), "g_FFI": (0.0, 9.0),
                             "I_o": (0.15, 0.60), "sigma": (0.0, 0.09)}
config.BASIS_TYPE         = "network_laplacian"
config.N_LAPLACIAN_BASIS  = int(os.environ.get("N_LAPLACIAN_BASIS", "4"))
config.USE_NETWORK_BASIS  = True
config.SAVE_PARAM_MAPS    = True
config.NETWORK_LABELS_CSV = None    # optional atlas network labels (row-aligned); None -> const+Laplacian only
# basis_regionwise: SBI infers basis coefficients (theta_dim=n_params x basis_dim);
# BasisParamDecoder maps them via map=mid+half*tanh(basis@beta). Active basis =
# HCP_Data/basis_cortex.npy [const, myelin_z, gradient_z] (360,3) — cortex-only
# FC gradient (extract_gradient_cortex.py). bounds below override HETERO_BOUNDS.
config.BASIS_PATH         = os.environ.get("BASIS_PATH", "HCP_Data/basis_cortex.npy")
config.BASIS_REZSCORE     = True
# S5: coupling upper bound is env-tunable (G_BOUND_HIGH, default 3.0 = current).
# Old wide-bound runs showed the posterior pushing g_LRE/g_FFI toward ~7, so the
# (0,3) cap may under-couple and depress simFC structure. G_BOUND_HIGH=6 re-opens
# it for ablation A4. tanh keeps maps within bounds so decode stays finite.
_G_BOUND_HIGH = float(os.environ.get("G_BOUND_HIGH", "3.0"))
# I_o operating-point bound. Old (0,1) put the decoder midpoint at I_o=0.5, but
# the rWW-EI critical operating point (isolated node, no FIC) is I_o~0.382. With
# (0,1) ~51% of prior-sampled (sim,region) entries land saturated and only ~3%
# in the critical band, so SNPE trains mostly on degenerate/saturated FCs.
# (0.30,0.45) centers the midpoint at 0.375 ~= critical -> the bulk of training
# sims are non-saturated. Revert to old behavior with IO_BOUND_LO=0 IO_BOUND_HI=1.
_IO_LO = float(os.environ.get("IO_BOUND_LO", "0.30"))
_IO_HI = float(os.environ.get("IO_BOUND_HI", "0.45"))
config.BASIS_BOUNDS       = {"g_LRE": (0.0, _G_BOUND_HIGH),
                             "g_FFI": (0.0, _G_BOUND_HIGH),
                             "I_o": (_IO_LO, _IO_HI), "sigma": (0.0, 0.05)}
config.BASIS_COEFF_PRIOR  = (float(os.environ.get("COEFF_PRIOR_LO", "-10.0")),
                             float(os.environ.get("COEFF_PRIOR_HI", "10.0")))
# basis mode uses the narrow BASIS_BOUNDS, NOT the wide homogeneous HETERO_BOUNDS.
# Apply it HERE, before the C1 banner builds/caches the decoder — otherwise the
# decoder caches the wide (0-9) bounds and the later override is a no-op.
if str(getattr(config, "PARAMETER_MODE", "homogeneous")) == "basis_regionwise":
    config.HETERO_BOUNDS = dict(config.BASIS_BOUNDS)
# SC dataset: "hcp_v73" (HCP_SC.mat, h5py) | "cabnp381" (HCP_CABNP381_SC_first100.mat,
# 3D weight_all/tract_length_all; FC still from HCP_FC.mat).
config.SC_DATASET         = os.environ.get("SC_DATASET", "cabnp381")
config.RUN_PHASE24        = False   # P6: Phase2/4 feature-selection unused by final_test (uses Phase1) and hurt NLL (-6.12 vs -8.49) -> skip
# ④ per-subject SC conditioning in the embedding (true amortization). Core is
# implemented + CPU-verified in inference/embedding.py (per_subject_sc path) and
# threaded through inference/snpe.py. Enabling it ALSO needs the producer+eval
# wiring below (see checklist) and GPU validation, so it is OFF by default:
config.EMBED_PER_SUBJECT_SC = False
config.VELOCITY_M_PER_S   = 3.0                # human conduction velocity (m/s)
config.USE_FCD            = False              # HCP FC only (no FCD)
# FC dimensionality reduction: PCA (whitened) replaces the removed RegionTransformer
# embedding. FC rank <= N_REGIONS(360), so 256 comps capture nearly all structure;
# keeps SNPE-C conditioning tractable (avoids the raw-64620 leakage / 0% accept).
config.FC_PCA_DIM         = int(os.environ.get("FC_PCA_DIM", "256"))
# S4: FC PCA whitening toggle (default 1 = current behavior). whiten amplifies
# low-variance PCs and can push empirical FC OOD (rejection accept <1% in the
# t20 run). FC_PCA_WHITEN=0 keeps natural scale -> obs projections stay in-range.
config.FC_PCA_WHITEN      = (os.environ.get("FC_PCA_WHITEN", "1") == "1")
config.USE_DELAYS         = (os.environ.get("USE_DELAYS", "0") == "1")  # default OFF: GPU sanity (delay_sanity_rwweib2.py) showed delays cost 5.3x for ~0 BOLD-FC change (6-100ms << 720ms TR/decimate); USE_DELAYS=1 to re-enable
# ── SC-conditioned amortized NPE (matrix-encoder) — default OFF = FC-only baseline ──
# SC_CONDITION on -> condition the posterior on a learned encoder over multi-channel
# ROIxROI matrices [FC, <SC_CHANNELS>] via MultiChannelMatrixEmbedding (per-subject SC
# looked up by index; theta stays 12 basis coeffs). OFF preserves the FC-only path.
# SC_CHANNELS: subject-constant channels stacked per subject (delay excluded per the
# sanity result; re-add "delay" here to include it). FC is always the per-sim channel.
config.SC_CONDITION       = (os.environ.get("SC_CONDITION", "1") == "1")  # default ON (q(theta|SC,FC)); SC_CONDITION=0 to disable
config.SC_CHANNELS        = tuple(
    os.environ.get("SC_CHANNELS", "sc_weight,sc_mask").split(","))
# ── Geometry coupling (homotopic SC augmentation) — default OFF = baseline ──
# Empirical FC is dominated by homotopic (i,i+180) edges (0.40 vs 0.06 mean, 6.2x)
# that tractography SC under-recovers, capping the SC-driven ceiling (~0.2). Adding
# a homotopic prior to the coupling lets the simulator generate cross-hemisphere FC.
# Structural prior only — theta_dim / model equations unchanged. Distance kernel is
# intentionally absent (tract-length distance was non-informative for FC here).
config.GEOMETRY_COUPLING  = (os.environ.get("GEOMETRY_COUPLING", "0") == "1")
config.GEOM_ALPHA         = float(os.environ.get("GEOM_ALPHA", "0.3"))
config.GEOM_KERNEL        = os.environ.get("GEOM_KERNEL", "homotopic")
config.GEOM_RENORM        = (os.environ.get("GEOM_RENORM", "1") == "1")

import data_loader_hcp as data_loader          # <-- HCP loader (drop-in)
import evaluate
import inference
import simulator

config.print_config()

# ── C1: explicit, fail-fast active-mode banner + basis-mode guard ─────────────
# Without this, an unset PARAMETER_MODE env var silently selects the
# "homogeneous" baseline and the myelin/gradient basis experiment is skipped.
# Always print the active config; by default HARD-FAIL unless basis_regionwise
# is selected. Opt out of the guard with REQUIRE_BASIS=0 to run another mode
# (homogeneous/direct/latent) on purpose.
_pm = str(getattr(config, "PARAMETER_MODE", "homogeneous"))
try:
    if _pm == "basis_regionwise":
        from basis_decoder import get_decoder as _gd
        _theta_dim = _gd(config).theta_dim
    elif _pm == "direct_regionwise":
        _theta_dim = len(config.HETERO_PARAMS) * int(config.N_REGIONS)
    elif _pm == "homogeneous":
        _theta_dim = len(config.STAGE1_PARAMS)
    else:  # latent_regionwise: depends on the per-subject SC basis (set later)
        _theta_dim = "set after Step 1 (SC-dependent)"
except Exception as _e:  # never let the banner crash the run
    _theta_dim = f"?? ({type(_e).__name__}: {_e})"
print("=" * 70)
print("  [HCP] ACTIVE INFERENCE CONFIG")
print("=" * 70)
print(f"    PARAMETER_MODE  : {_pm}")
print(f"    INFERENCE_MODEL : {getattr(config, 'INFERENCE_MODEL', '?')}")
print(f"    N_REGIONS       : {getattr(config, 'N_REGIONS', '?')}")
print(f"    BASIS_PATH      : {getattr(config, 'BASIS_PATH', '?')}")
print(f"    theta_dim       : {_theta_dim}")
print("=" * 70)
_REQUIRE_BASIS = (os.environ.get("REQUIRE_BASIS", "1") == "1")
if _REQUIRE_BASIS and _pm != "basis_regionwise":
    raise RuntimeError(
        f"main_HCP expects the basis_regionwise (myelin/gradient) experiment "
        f"but PARAMETER_MODE={_pm!r}. Run with "
        f"PARAMETER_MODE=basis_regionwise (recommended), or set REQUIRE_BASIS=0 "
        f"to explicitly opt into another mode (homogeneous/direct/latent)."
    )


# ## (Optional) 자원 진단 + 최적 GPU_BATCH 자동 탐색
# 
# GPU/RAM 자원을 점검하고, 짧은 시뮬을 돌려서 OOM 없는 최대 `GPU_BATCH`를 자동으로 찾습니다. 결과가 마음에 들면 Setup 셀의 `GPU_BATCH` 값을 권장값으로 바꾸고 다시 실행하세요.
# 
# - DECIMATE 값에 따라 RAM 사용량이 결정됩니다
# - 일반적으로 1~3분 소요

# import subprocess
# import time
# import numpy as np
# 
# # ── 1. 시스템 자원 확인 ──────────────────────────────────────
# def gpu_info():
#     r = subprocess.run(
#         ["nvidia-smi",
#          "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
#          "--format=csv,noheader,nounits"],
#         capture_output=True, text=True)
#     parts = [p.strip() for p in r.stdout.strip().split(",")]
#     return {
#         "name": parts[0],
#         "vram_total_gb": float(parts[1]) / 1024,
#         "vram_free_gb":  float(parts[2]) / 1024,
#         "vram_used_gb":  float(parts[3]) / 1024,
#         "util_pct":      float(parts[4]),
#     }
# 
# def ram_info():
#     r = subprocess.run(["free", "-g"], capture_output=True, text=True)
#     for line in r.stdout.splitlines():
#         if line.startswith("Mem:"):
#             p = line.split()
#             return {
#                 "total_gb": int(p[1]),
#                 "used_gb":  int(p[2]),
#                 "free_gb":  int(p[3]),
#                 "available_gb": int(p[6]),
#             }
#     return None
# 
# def cpu_count():
#     import os
#     return os.cpu_count()
# 
# g = gpu_info()
# r = ram_info()
# nc = cpu_count()
# 
# print("=" * 70)
# print("  System resources")
# print("=" * 70)
# print(f"  GPU       : {g['name']}")
# print(f"              VRAM total {g['vram_total_gb']:.1f} GB, "
#       f"free {g['vram_free_gb']:.1f} GB, util {g['util_pct']:.0f}%")
# print(f"  RAM       : total {r['total_gb']} GB, "
#       f"available {r['available_gb']} GB")
# print(f"  CPU       : {nc} cores")
# print()
# 
# # ── 2. 현재 설정 + 메모리 추정 ───────────────────────────────
# DT          = cfg.DT
# DECIMATE    = cfg.DECIMATE
# T_END       = cfg.T_END_MS
# N_REG       = 115
# N_SIM       = cfg.N_SIM
# N_TRAIN     = cfg.N_TRAIN
# 
# stored_dt   = DT * DECIMATE
# T_stored    = int(T_END / stored_dt)
# 
# # Streaming BW: only T_bold (TR-downsampled) frames stay in RAM
# # T_bold = T_stored / (TR_sec * 1000 / stored_dt)
# tr_ms = 1000.0  # config.TR_SEC * 1000
# step_per_tr = max(1, int(round(tr_ms / stored_dt)))
# T_bold = T_stored // step_per_tr
# 
# bytes_per_sim = T_bold * N_REG * 4  # streaming = ~100x less than full
# 
# print(f"  Stored dt  : {stored_dt:.1f} ms  "
#       f"(DT={DT}, DECIMATE={DECIMATE})")
# print(f"  T_stored   : {T_stored}  (was used as RAM size; not anymore)")
# print(f"  T_bold     : {T_bold}  (streaming: only TR-downsampled BOLD in RAM)")
# print(f"  RAM/sim    : {bytes_per_sim / 1e6:.3f} MB  (was "
#       f"{T_stored * N_REG * 4 / 1e6:.1f} MB before streaming)")
# print()
# 
# # Streaming makes RAM trivial; the actual bottleneck is now VRAM.
# # We still cap by safe RAM for the unlikely case of huge T_bold.
# safe_ram_gb  = r["available_gb"] * 0.7
# max_batch_ram = int(safe_ram_gb * 1e9 / bytes_per_sim)
# print(f"  Safe RAM   : {safe_ram_gb:.0f} GB "
#       f"(70% of available)")
# print(f"  Max batch by RAM : {max_batch_ram}")
# print()
# 
# # ── 3. 짧은 시뮬로 OOM-free 최대 batch 자동 탐색 ─────────────
# print("=" * 70)
# print("  Probing OOM-free batch sizes (10s simulation)")
# print("=" * 70)
# 
# # 진단 시간 짧게 (10s)
# PROBE_T_END = 10_000.0
# PROBE_T_CUT =  2_000.0
# _saved = (config.T_END, config.T_CUT, config.ANALYSIS_BOLD_T)
# config.T_END             = PROBE_T_END
# config.T_CUT             = PROBE_T_CUT
# config.ANALYSIS_BOLD_T   = int((PROBE_T_END - PROBE_T_CUT) / 1000)
# config.WC_FIXED["t_end"] = PROBE_T_END
# config.WC_FIXED["t_cut"] = PROBE_T_CUT
# 
# # 데이터 1명만 로드
# df, fc_mat, sc_mat, fc_ids, sc_ids, _, _ = data_loader.load_raw_data()
# sids = data_loader.get_target_subjects(df, fc_mat, sc_mat, fc_ids, sc_ids)
# d = data_loader.get_subject_data(sids[0], fc_mat, sc_mat, fc_ids, sc_ids)
# 
# # 시도할 batch 목록: cfg.GPU_BATCH 부근 + 한 단계씩
# candidates = []
# # Streaming이라 RAM 제약이 거의 없음. VRAM 한도까지 시도.
# for b in [500, 1000, 2000, 4000, 8000, 10000, 20000, 40000, 80000]:
#     if b <= max_batch_ram * 1.5:   # RAM 추정의 1.5배까지만
#         candidates.append(b)
# candidates = sorted(set(candidates))
# 
# header = (
#     f"  {'batch':>6}  {'time':>7}  {'s/sim@10s':>10}  "
#     f"{'s/sim@full':>10}  {'full_est_hr':>11}  {'RAM_MB':>8}  "
#     f"VRAM/util"
# )
# print(header)
# print("  " + "-" * 100)
# 
# results = []
# ratio = cfg.T_END_MS / PROBE_T_END   # 10s -> full T_END_MS 환산
# for batch in candidates:
#     try:
#         t0 = time.time()
#         simulator.simulate_single(
#             d["sc"],
#             {"P": 1.5, "Q": 1.0, "g_e": 0.7, "g_i": 0.7},
#             n_repeat=batch,
#             delays=d["delays"],
#             apply_bw=True,
#         )
#         t_probe = time.time() - t0
#         per_sim_10s = t_probe / batch
#         per_sim_full = per_sim_10s * ratio
#         n_batches = N_SIM // batch + (1 if N_SIM % batch else 0)
#         total_hr = per_sim_full * batch * n_batches * N_TRAIN / 3600
#         ram_est_mb = bytes_per_sim * batch / 1e6
#         g = gpu_info()
#         vram_used = g["vram_used_gb"]
#         util = g["util_pct"]
#         print(
#             f"  {batch:>6}  {t_probe:>6.1f}s  "
#             f"{per_sim_10s:>7.3f}s   "
#             f"{per_sim_full:>7.2f}s   "
#             f"{total_hr:>10.2f}hr  "
#             f"{ram_est_mb:>7.1f}MB  "
#             f"VRAM {vram_used:.1f}GB / util {util:.0f}%"
#         )
#         results.append({
#             "batch": batch,
#             "per_sim_full": per_sim_full,
#             "total_hr": total_hr,
#             "ram_mb": ram_est_mb,
#         })
#     except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
#         print(f"  {batch:>6}  RAM OOM ({e.__class__.__name__})")
#         break
#     except Exception as e:
#         print(f"  {batch:>6}  FAILED: {e.__class__.__name__}: {e}")
#         break
# 
# # 원래 설정 복원
# config.T_END             = _saved[0]
# config.T_CUT             = _saved[1]
# config.ANALYSIS_BOLD_T   = _saved[2]
# config.WC_FIXED["t_end"] = _saved[0]
# config.WC_FIXED["t_cut"] = _saved[1]
# 
# # ── 4. 권장값 ───────────────────────────────────────────────
# print()
# print("=" * 70)
# if not results:
#     print("  No successful runs — try lowering DECIMATE further")
# else:
#     # 가장 빠른 batch
#     best = min(results, key=lambda x: x["total_hr"])
#     # 안정성 마진: best의 batch가 max_batch_ram 보다 크면 max_batch_ram에 맞춤
#     rec_batch = min(best["batch"], max_batch_ram)
# 
#     print(f"  Recommended GPU_BATCH = {rec_batch}")
#     print(f"    expected total time  = {best['total_hr']:.2f}hr  "
#           f"({N_SIM} sims x {N_TRAIN} subjects)")
#     print(f"    RAM use             ≈ {best['ram_mb']:.1f} MB  (streaming)")
#     print()
#     print(f"  Setup 셀에서:")
#     print(f"    GPU_BATCH = {rec_batch:_}")
# print("=" * 70)
# 

# ## Step 1. Data split

# In[3]:


df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = (
    data_loader.load_raw_data()
)
subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
train, val, test = data_loader.three_way_split(subjects)

subject_data = data_loader.load_all_subjects(
    train + val + test,
    fc_mat, sc_mat, fc_ids, sc_ids,
    bold_mat, bold_ids,
)

# Result
evaluate.report_step1(train, val, test, subject_data)

# ── Step 1.5: SC-conditioned amortized NPE — per-subject SC table ────────────
# Gated on config.SC_CONDITION (default OFF). When OFF this is a no-op
# (SC_TABLE=None) so every downstream FC-only path is byte-for-byte identical.
# When ON, build a per-subject multi-channel SC table over ALL subjects
# (train+val+test) as a leakage-safe lookup buffer. The ScChannelScaler is FIT
# ON TRAIN ROWS ONLY then applied to the whole table; val/test rows are never
# referenced during training (only their own row is read at eval time), so this
# is not leakage. config.SC_IDMAP maps {sid: row} for both training x and
# eval x_obs.
SC_TABLE = None
if getattr(config, "SC_CONDITION", False):
    from inference.sc_channels import build_sc_table, ScChannelScaler
    _all = list(train) + list(val) + list(test)
    _sctab_raw, _scidmap = build_sc_table(
        subject_data, _all, config.SC_CHANNELS)
    _trows = [_scidmap[int(s)] for s in train]
    _scsc = ScChannelScaler().fit(_sctab_raw[_trows], config.SC_CHANNELS)
    SC_TABLE = _scsc.transform(_sctab_raw)
    config.SC_IDMAP = _scidmap
    print("\n  ===== Step 1.5: SC_CONDITION = ON (SC-conditioned NPE) =====")
    print(f"    SC table         : shape {SC_TABLE.shape}  "
          f"(n_subj={SC_TABLE.shape[0]}, C={SC_TABLE.shape[1]}, "
          f"N={SC_TABLE.shape[2]})")
    print(f"    SC channels      : {list(config.SC_CHANNELS)}")
    print(f"    scaler fit on    : {len(_trows)} TRAIN rows (val/test held out)")
    print("  ============================================================")

# ── Latent region-wise setup (PARAMETER_MODE='latent_regionwise') ────────────
# Replace STAGE1_PARAMS with latent coefficient names and make the prior a
# BoxUniform[-1,1]^latent_dim (scaler becomes identity). The decoder maps z to
# per-region maps for HETERO_PARAMS at sim time (per-subject basis).
if str(getattr(config, "PARAMETER_MODE", "homogeneous")) == "latent_regionwise":
    from region_basis import build_region_basis
    from param_decoder import latent_param_names, latent_dim, per_param_block
    from engine_select import load_network_labels
    _labels = load_network_labels()
    _basis0 = build_region_basis(subject_data[train[0]]["sc"], _labels, config)
    _lnames = latent_param_names(_basis0, config)
    _ld = latent_dim(_basis0, config)
    config.STAGE1_PARAMS      = _lnames
    config.PARAM_NAMES_STAGE1 = _lnames
    config.STAGE1_PRIOR_LOW   = [-1.0] * _ld
    config.STAGE1_PRIOR_HIGH  = [1.0] * _ld
    _block, _nnet, _K = per_param_block(_basis0)
    print("\n  ===== PARAMETER_MODE = latent_regionwise (provisional) =====")
    print(f"    target FC mode   : {'group-avg' if config.GROUP_AVG_FC else 'subject-specific'}")
    print(f"    active hetero    : {config.HETERO_PARAMS}  (all 4 regional; cuBNM rebuilt)")
    print(f"    regions          : {config.N_REGIONS}")
    print(f"    basis            : {config.BASIS_TYPE}  networks={_nnet}  laplacian_K={_K}")
    print(f"    latent dim       : {_ld}  (= {len(config.HETERO_PARAMS)} params x {_block} slots)")
    print(f"    SNPE target      : latent z  (BoxUniform[-1,1]^{_ld})")
    print(f"    decoded map shape: (n_sims, {config.N_REGIONS}) per param")
    print("  ============================================================")
elif str(getattr(config, "PARAMETER_MODE", "homogeneous")) == "direct_regionwise":
    # Direct region-wise control inference: theta = all N_REGIONS x len(HETERO)
    # control params (e.g. 360 x 4 = 1440). No basis. Scaler maps [-1,1] <->
    # per-region physical bounds; the wrapper reshapes theta -> per-region maps.
    from param_decoder import direct_param_names, direct_dim, direct_bounds
    _R = int(config.N_REGIONS)
    _dnames = direct_param_names(config, _R)
    _dd = direct_dim(config, _R)
    _low, _high = direct_bounds(config, _R)
    config.STAGE1_PARAMS      = _dnames
    config.PARAM_NAMES_STAGE1 = _dnames
    config.STAGE1_PRIOR_LOW   = _low
    config.STAGE1_PRIOR_HIGH  = _high
    print("\n  ===== PARAMETER_MODE = direct_regionwise =====")
    print(f"    target FC mode   : {'group-avg' if config.GROUP_AVG_FC else 'subject-specific'}")
    print(f"    control params   : {config.HETERO_PARAMS} (each per-region)")
    print(f"    regions          : {_R}")
    print(f"    theta_control dim : {_dd}  (= {len(config.HETERO_PARAMS)} params x {_R} regions)")
    print(f"    SNPE target      : theta_control (BoxUniform[-1,1]^{_dd} -> physical bounds)")
    print(f"    decoded map shape: (n_sims, {_R}) per param  (reshape, no basis)")
    print("    NOTE: 1440-dim posterior is high-dim; expect low identifiability "
          "(check shrinkage/ActiveSubspace/SBC).")
    print("  ============================================================")
elif str(getattr(config, "PARAMETER_MODE", "homogeneous")) == "basis_regionwise":
    # Identifiable Kong-style: SBI infers basis coefficients (theta_dim =
    # n_params x basis_dim, e.g. 4x3=12). BasisParamDecoder maps them to bounded
    # region-wise maps via mid+half*tanh(basis@beta). bounds = BASIS_BOUNDS.
    config.HETERO_BOUNDS = dict(config.BASIS_BOUNDS)
    from basis_decoder import get_decoder
    _dec = get_decoder(config)
    config.STAGE1_PARAMS      = _dec.coeff_names()
    config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
    _clo, _chi = config.BASIS_COEFF_PRIOR
    config.STAGE1_PRIOR_LOW   = [_clo] * _dec.theta_dim
    config.STAGE1_PRIOR_HIGH  = [_chi] * _dec.theta_dim
    print("\n  ===== PARAMETER_MODE = basis_regionwise (myelin/gradient) =====")
    print(f"    target FC mode   : {'group-avg' if config.GROUP_AVG_FC else 'subject-specific'}")
    print(f"    SC dataset       : {config.SC_DATASET}   delays={config.USE_DELAYS}")
    print(f"    control params   : {config.HETERO_PARAMS}  bounds={config.HETERO_BOUNDS}")
    print(f"    basis            : {config.BASIS_PATH}  (regions={_dec.n_regions}, "
          f"basis_dim={_dec.basis_dim})")
    print(f"    theta dim        : {_dec.theta_dim}  (= {_dec.n_params} params x "
          f"{_dec.basis_dim} basis)  prior Uniform{config.BASIS_COEFF_PRIOR}")
    print(f"    decoded map shape: (n_sims, {_dec.n_regions}) per param  "
          f"(mid+half*tanh(basis@beta))")
    print("  ============================================================")

# ── per-subject myelin/gradient basis (BASIS_PERSUBJECT=1) ────────────────────
# Each subject decodes with its OWN myelin/gradient (myelin/gradient_subjects.npy,
# rows aligned to SC sub_num) instead of the shared group-mean basis.npy. theta
# (coeffs) stays 12-dim; only the spatial template becomes per-subject. Default OFF.
config.BASIS_PERSUBJECT = (os.environ.get("BASIS_PERSUBJECT", "0") == "1")
if config.BASIS_PERSUBJECT and str(getattr(config, "PARAMETER_MODE", "")) == "basis_regionwise":
    from basis_decoder import build_persubject_bases
    _scf = getattr(config, "SC_PATH", None) or config.SC_FILE
    config.PERSUBJECT_BASES = build_persubject_bases(_scf, n_regions=int(config.N_REGIONS))
    _miss = [int(s) for s in subject_data if int(s) not in config.PERSUBJECT_BASES]
    if _miss:
        raise RuntimeError(f"BASIS_PERSUBJECT: no per-subject basis for {_miss[:5]} "
                           f"(have {len(config.PERSUBJECT_BASES)}). Check row alignment.")
    print(f"  [basis] PER-SUBJECT myelin/gradient ON: {len(config.PERSUBJECT_BASES)} subjects; "
          f"each sim/eval decodes with its own subject's basis (rows ~ SC sub_num).")


# ## Train data: weights / tract lengths / empirical FC
# 
# Step 1 직후 실행. Train subject 4명의 SC(weights), SC tract length, empirical FC를 로드하고 출력합니다.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

for sid in train:
    d  = subject_data[sid]
    w  = d["sc"].astype(np.float32)
    lm = d["lengths_mm"].astype(np.float32)
    dl = d["delays"].astype(np.float32)
    fc = d["fc"].astype(np.float32)

    w_pos  = w[w > 0]
    lm_pos = lm[lm > 0]
    dl_pos = dl[dl > 0]
    fc_off = fc[np.eye(fc.shape[0]) == 0]

    print(
        f"  {sid}:"
        f"  SC=[{w_pos.min():.4f}, {w_pos.max():.4f}]"
        f"  length=[{lm_pos.min():.1f}, {lm_pos.max():.1f}]mm"
        f"  delay=[{dl_pos.min():.2f}, {dl_pos.max():.2f}]ms"
        f"  FC=[{fc_off.min():.3f}, {fc_off.max():.3f}]"
        f"  SC_nnz={int((w > 0).sum())}"
    )

n_train = len(train)
fig, axes = plt.subplots(n_train, 4,
                         figsize=(13, 3.0 * n_train),
                         squeeze=False)

for row, sid in enumerate(train):
    d  = subject_data[sid]
    w  = d["sc"].astype(np.float32)
    lm = d["lengths_mm"].astype(np.float32)
    dl = d["delays"].astype(np.float32)
    fc = d["fc"].astype(np.float32)

    im0 = axes[row, 0].imshow(w, cmap="RdBu_r", vmin=0, vmax=1)
    axes[row, 0].set_title(f"{sid}\nSC weights")
    axes[row, 0].set_ylabel("Region")
    plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

    im1 = axes[row, 1].imshow(lm, cmap="viridis")
    axes[row, 1].set_title("Tract length (mm)")
    plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, label="mm")

    im2 = axes[row, 2].imshow(dl, cmap="plasma")
    axes[row, 2].set_title("Delay (ms)")
    plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, label="ms")

    im3 = axes[row, 3].imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[row, 3].set_title("Empirical FC (r)")
    plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, label="r")

for ax in axes.flat:
    ax.set_xlabel("Region")

plt.suptitle(
    "Train subjects — SC / tract lengths / delays / empirical FC",
    fontsize=11, y=1.01,
)
plt.tight_layout()
save_path = "./output_mouse_mptp/train_data_matrices.png"
plt.savefig(save_path, dpi=110, bbox_inches="tight")
plt.show()
print(f"  saved: {save_path}")


# ## Step 7. Parameter preprocessing ([-1, 1] scaling)
# 
# Step 7 runs before step 2 because the simulation needs `prior_scaled` to sample parameters.

# In[6]:


param_scaler, prior_scaled = inference.step7_fit_param_scaler(
    verbose=True
)

# Result
evaluate.report_step7(param_scaler)


# ## Step 2. WC simulation
# 
# Feature extraction (step 3) is interleaved inside this loop.

# In[7]:


import os

diag_bold = None
diag_sid = None

import json as _json


def _cache_meta_now():
    """Cache-identity metadata. Mismatch => stale (never mix modes/targets)."""
    from simulation.geometry import geometry_meta as _geom_meta
    _pm = str(getattr(config, "PARAMETER_MODE", "homogeneous"))
    _rw = _pm in ("latent_regionwise", "direct_regionwise", "basis_regionwise")
    _basis = (config.BASIS_TYPE if _pm == "latent_regionwise"
              else (os.path.basename(config.BASIS_PATH)
                    if _pm == "basis_regionwise" else None))
    _nlap = int(config.N_LAPLACIAN_BASIS) if _pm == "latent_regionwise" else None
    # S5 fix: bounds feed the decoder (map=mid+half*tanh(z)), so a bound change
    # (e.g. G_BOUND_HIGH) changes theta_raw + decoded maps + the simulated FC in
    # features_stage1.npz. It MUST be in the cache identity or an ablation would
    # silently reuse stale (wrong-bound) sims. Same for USE_DELAYS (delays on/off
    # change the simulated FC). Both are env-tunable free variables now.
    _hb = {k: list(v) for k, v in
           sorted((getattr(config, "HETERO_BOUNDS", {}) or {}).items())} if _rw else None
    # --- extra cache-identity fields. Omitting any sim-determining variable lets a
    # stale cache be silently reused after a sim-changing edit (e.g. an I_o-bound /
    # sim-length / SC change) so the edit never reaches the trained posterior.
    # See docs/performance_root_cause.md (cache-key-omits-* findings).
    import hashlib as _hashlib
    _basis_hash = None
    if _pm == "basis_regionwise":
        try:
            _basis_hash = _hashlib.sha1(
                np.load(config.BASIS_PATH).tobytes()).hexdigest()[:12]
        except Exception:
            _basis_hash = "??"
    _rww_fixed = ({k: float(v) for k, v in
                   sorted((getattr(config, "RWWEIB2_FIXED", {}) or {}).items())}
                  if str(config.INFERENCE_MODEL).startswith("rwweib") else None)
    try:                       # `train` is the script-global train split (defined in Step 1)
        _train_ids = sorted(int(s) for s in train)
    except Exception:
        _train_ids = None
    return {
        "target_fc_mode": "group" if config.GROUP_AVG_FC else "subject",
        "PARAMETER_MODE": config.PARAMETER_MODE,
        "N_REGIONS": int(config.N_REGIONS),
        "FC_DIM": int(config.FC_DIM),
        "hetero_params": list(config.HETERO_PARAMS) if _rw else [],
        "hetero_bounds": _hb,
        "use_delays": bool(getattr(config, "USE_DELAYS", False)),
        "basis_type": _basis,
        "n_laplacian_basis": _nlap,
        "latent_dim": len(config.STAGE1_PARAMS) if _rw else None,
        "model": config.INFERENCE_MODEL,
        # Operating-point / regime variables that change the simulated FC but were
        # previously absent from the key (the I_o bound is already covered via
        # hetero_bounds above; these are the remaining free variables).
        "use_fic": bool(getattr(config, "USE_FIC", False)),
        "sim_timing": [config.T_END, config.T_CUT, config.DT,
                       int(config.DECIMATE), config.TR_SEC],
        "sim_seed": int(getattr(config, "SIM_SEED", 42)),
        "sc_dataset": getattr(config, "SC_DATASET", None),
        "sc_file": os.path.basename(str(getattr(config, "SC_FILE", ""))),
        "sc_scale": os.environ.get("VBI_SC_SCALE", "log1p"),
        "rwweib2_fixed": _rww_fixed,
        "basis_rezscore": bool(getattr(config, "BASIS_REZSCORE", False)),
        "basis_persubject": bool(getattr(config, "BASIS_PERSUBJECT", False)),
        "coeff_prior": list(getattr(config, "BASIS_COEFF_PRIOR", (-2.0, 2.0))),
        "basis_hash": _basis_hash,
        "train_ids": _train_ids,
        # Geometry coupling changes the SC fed to the simulator -> changes the
        # simulated FC, so it MUST be in the cache key (re-sim on a geometry change).
        **_geom_meta(config),
        # NOTE: sc_condition / sc_channels are deliberately NOT in the cache key.
        # The Step-2 sim (theta -> FC) is identical regardless of conditioning;
        # SC channels only enter Step 4 (x=[idx|fc]) and Step 8 (encoder), which
        # are post-cache. Keeping them out lets the ablation arms (A FC-only /
        # B,C,D SC variants) SHARE one simulation cache and only re-train the NPE.
        # subj_ids is always saved (producer-side), so any arm can build x.
    }


_feat_path = os.path.join(config.OUTPUT_DIR, "features_stage1.npz")
_meta_path = os.path.join(config.OUTPUT_DIR, "features_stage1_meta.json")
_cache_ok = False
if os.path.exists(_feat_path):
    _loaded = inference.load_extracted_features(
        save_dir=config.OUTPUT_DIR, tag="stage1"
    )
    _ts = _loaded["theta_scaled"]
    _n_exp = len(train) * config.N_SIM
    _p_exp = len(config.STAGE1_PARAMS)
    _meta_old = {}
    if os.path.exists(_meta_path):
        with open(_meta_path) as _mf:
            _meta_old = _json.load(_mf)
    _meta_match = (_meta_old == _cache_meta_now())
    # Reuse only when shapes AND metadata match the CURRENT config. Otherwise the
    # cache is from a different model / split / target-FC / parameter mode.
    if (_ts.shape[0] == _n_exp and _ts.shape[1] == _p_exp
            and _loaded["fc_raw"].shape[1] == config.FC_DIM and _meta_match):
        print(f"  [Step 2 skip] loading saved features: {_feat_path}")
        theta_scaled = _ts
        theta_raw    = _loaded["theta_raw"]
        fc_raw       = _loaded["fc_raw"]
        fcd_raw      = _loaded["fcd_raw"]
        _cache_ok = True
    else:
        print(f"  [Step 2] cache {_feat_path} STALE "
              f"(theta {_ts.shape} vs expected ({_n_exp},{_p_exp}), "
              f"fc_dim {_loaded['fc_raw'].shape[1]} vs {config.FC_DIM}, "
              f"meta_match={_meta_match}) -> re-simulating")
if not _cache_ok:
    _result = inference.step2_simulate_train(
        train, subject_data, prior_scaled, param_scaler,
        n_sim=config.N_SIM, apply_bw=True, verbose=True,
        save_first_sample=True,
        engine=config.INFERENCE_MODEL,  # Step 2 backend = active model (engine_select-consistent)
    )
    theta_scaled = _result["theta_scaled"]
    theta_raw    = _result["theta_raw"]
    fc_raw       = _result["fc_raw"]
    fcd_raw      = _result["fcd_raw"]
    diag_bold    = _result.get("diag_bold")
    diag_sid     = _result.get("diag_sid")

# subj_ids: per-simulation subject-id array aligned with fc_raw rows. Needed by
# the SC_CONDITION x=[idx|fc] path to look up each sim's SC row. Pulled from the
# cache when reused, else from the fresh simulation result.
subj_ids = (_loaded.get("subj_ids") if _cache_ok else _result.get("subj_ids"))
if getattr(config, "SC_CONDITION", False) and subj_ids is None:
    raise RuntimeError(
        "SC_CONDITION needs subj_ids; clear features_stage1.npz and "
        "re-simulate")

evaluate.report_step2(theta_scaled, fc_raw, fcd_raw)


# ### Step 2 진단: train subject 1명의 시뮬 결과 시각화
# 
# feature extraction 루프에서 캐싱한 첫 번째 시뮬 BOLD/FC를 재시뮬 없이 그대로 그립니다.

# # if diag_bold is not None:
#     _one_sim = evaluate.plot_one_simulation(
#         sid=diag_sid,
#         subject_data=subject_data,
#         bold=diag_bold,
#         sim_idx=0,
#     )
# else:
#     print("  [Step 2 진단] diag_bold 없음 (캐시 로드 경로 또는 시뮬 실패) — 진단 출력 스킵")

# ## Step 3. Feature extraction summary
# 
# Step 2의 streaming loop 안에서 이미 추출된 feature를 정리하고, **raw feature를 디스크에 저장**합니다.
# 
# - `fc_raw` : FC upper triangle (Fisher z-transformed)
# - `fcd_raw` : FCD upper triangle (element-wise std of sliding-window FCs)
# 
# 저장 경로: `{OUTPUT_DIR}/features_stage1.npz`  
# Load 방법: `inference.load_extracted_features(tag="stage1")`

# In[8]:


import os
inference.step3_summary_features(fc_raw, fcd_raw, verbose=True)

_feat_path = os.path.join(config.OUTPUT_DIR, "features_stage1.npz")
if not _cache_ok:
    inference.save_extracted_features(
        theta_scaled, theta_raw, fc_raw, fcd_raw,
        param_names=config.STAGE1_PARAMS,
        save_dir=config.OUTPUT_DIR,
        tag="stage1",
        verbose=True,
        subj_ids=subj_ids,   # SC_CONDITION lookup keys (None in FC-only mode)
    )
    with open(_meta_path, "w") as _mf:        # cache-identity sidecar
        _json.dump(_cache_meta_now(), _mf, indent=2)
    print(f"  [cache meta] {_meta_path}: {_cache_meta_now()}")
else:
    print(f"  features_stage1.npz reused — skip save")

evaluate.report_step3(fc_raw, fcd_raw)


# ## Step 4. Feature pipeline (raw FC passthrough)
# 
# PCA/z-score 없음. fc_raw (6555) → x_input 그대로.
# RegionTransformer가 SNPE-C와 jointly train하면서
# attention weight 학습.

# In[9]:


# New pipeline: raw FC is x_input directly
# No PCA, no z-score — RegionTransformer handles feature extraction
from inference.feature_pipeline import FeaturePipeline

# S3: drop degenerate sims (NaN / saturated / out-of-range / dead-flat FC) BEFORE
# fitting the PCA so they don't pollute the FC basis. theta_scaled/fc_raw/fcd_raw
# are row-aligned, so the SAME mask is applied to all three (keeps SNPE pairs
# matched). Mirrors the filter in inference.step5_fit_feature_pipeline; the
# main_HCP Step 4 path previously skipped it.
_n0 = fc_raw.shape[0]
_finite = np.isfinite(fc_raw).all(axis=1)
_notsat = fc_raw.std(axis=1) > 1e-4
_inrng  = (fc_raw.min(axis=1) >= -1.001) & (fc_raw.max(axis=1) <= 1.001)
_alive  = np.abs(fc_raw).mean(axis=1) > 1e-3
_qmask  = _finite & _notsat & _inrng & _alive
_n1 = int(_qmask.sum())
print(f"  [Step 4] sim-quality filter: {_n0:,} -> {_n1:,}  "
      f"(NaN={int((~_finite).sum())}, sat={int((~_notsat).sum())}, "
      f"oor={int((~_inrng).sum())}, dead={int((~_alive).sum())})")
# Relative floor: fail only if the filter dropped most sims (mass degeneration)
# or kept too few for SNPE. An absolute 1000 floor wrongly broke SMOKE runs
# (N_TRAIN=2 x N_SIM=64 = 128 valid sims). max(16, 50% of n0) scales with run size.
_min_keep = max(16, int(0.5 * _n0))
if _n1 < _min_keep:
    raise RuntimeError(
        f"유효 시뮬 {_n1} < {_min_keep} (={_n0}개 중 절반 미만 또는 <16) "
        "— prior/시뮬/bounds 확인 필요 (degenerate FC가 너무 많음).")
if _n1 < _n0:
    theta_scaled = theta_scaled[_qmask]
    theta_raw    = theta_raw[_qmask]
    fc_raw       = fc_raw[_qmask]
    fcd_raw      = fcd_raw[_qmask]
    # SC_CONDITION: subj_ids is row-aligned with fc_raw, so the SAME mask must
    # filter it (keeps each x's index column matched to its FC row).
    if getattr(config, "SC_CONDITION", False) and subj_ids is not None:
        subj_ids = subj_ids[_qmask]

if getattr(config, "SC_CONDITION", False):
    # x = [row_index | fc_upper_tri]: the MultiChannelMatrixEmbedding compresses
    # the raw FC, so NO FC PCA. idx_col references each sim's SC table row; the
    # encoder looks up the subject-constant SC channels by that index.
    idx_col = np.array(
        [config.SC_IDMAP[int(s)] for s in subj_ids], dtype=np.float32)[:, None]
    x_input = np.concatenate([idx_col, fc_raw.astype(np.float32)], axis=1)
    # FeaturePipeline kept for compat (artifacts / eval transform_observed
    # fallback) but NOT used to build x in SC mode.
    feature_pipeline = FeaturePipeline()
    feature_pipeline.fit(fc_raw, fcd_raw, verbose=False)
    print(f"  [Step 4][SC] x=[idx|fc]  x_input shape : {x_input.shape}  "
          f"(theta_scaled {theta_scaled.shape})")
    print(f"  [Step 4][SC] leading col = SC table row index; "
          f"FC upper-tri ({config.FC_DIM}) compressed by encoder (no PCA)")
else:
    feature_pipeline = FeaturePipeline()
    # fit() returns self; fit_transform() returns the (n_kept, fc_out_dim) array
    x_input = feature_pipeline.fit_transform(fc_raw, fcd_raw, verbose=True)

    print(f"  [Step 4] x_input shape : {x_input.shape}  (theta_scaled {theta_scaled.shape})")
    print(f"  [Step 4] FC PCA reduction -> {config.FC_PCA_DIM} comps "
          f"(whiten={config.FC_PCA_WHITEN}), no embedding")
    print(f"  [Step 4] x_input = PCA(FC) fed directly to MAF (no RegionTransformer)")


# ## Step 8. Stage 1 inference (single-round SNPE-C)

# In[10]:


# RegionTransformer embedding REMOVED — raw FC fed straight to MAF (no learned
# embedding, no SC positional encoding). use_embedding=False -> nn.Identity in
# train_snpe. (sc_matrix only fed the old embedding; dropped.)
# SC_CONDITION ON -> swap in a MultiChannelMatrixEmbedding over [FC, SC channels]
# (per-subject SC looked up by the leading idx column of x_input); FC-only path
# is unchanged.
if getattr(config, "SC_CONDITION", False):
    from inference.embedding import MultiChannelMatrixEmbedding
    # C-encoder fix: SC_FUSION=film makes SC MULTIPLICATIVELY gate the FC tokens
    # + adds a direct SC path to the head, so the encoder can no longer ignore
    # the SC channel for free (baseline "add" fusion zeroed it out -> SC-perm
    # delta~0). FC_TOKEN_DROPOUT>0 further discourages FC-only shortcuts. Both
    # are post-cache (Step 8 only) so features_stage1.npz is REUSED (no re-sim).
    _sc_fusion = os.environ.get("SC_FUSION", "add")
    _fc_tok_drop = float(os.environ.get("FC_TOKEN_DROPOUT", "0.0"))
    print(f"\n  [C-encoder] SC fusion = {_sc_fusion!r}  "
          f"fc_token_dropout = {_fc_tok_drop}")
    _emb = MultiChannelMatrixEmbedding(
        fc_input_dim=config.FC_DIM, sc_table=SC_TABLE,
        out_dim=config.EMBED_DIM, n_regions=config.N_REGIONS,
        use_fc_mask=True,
        fusion=_sc_fusion, fc_token_dropout=_fc_tok_drop,
    )
    _USE_EMBEDDING = True
    posterior, embedding_net = inference.step8_train_snpe(
        theta_scaled, x_input, prior_scaled,
        verbose=True,
        fc_raw=fc_raw,
        use_embedding=True,
        embedding_net=_emb,
    )
else:
    _USE_EMBEDDING = False
    posterior, embedding_net = inference.step8_train_snpe(
        theta_scaled, x_input, prior_scaled,
        verbose=True,
        fc_raw=fc_raw,
        use_embedding=_USE_EMBEDDING,
    )

s1 = {
    "posterior":        posterior,
    "embedding_net":    embedding_net,
    "theta_scaled":     theta_scaled,
    "theta_raw":        theta_raw,
    "fc_raw":           fc_raw,
    "fcd_raw":          fcd_raw,
    "x_input":          x_input,
    "param_scaler":     param_scaler,
    "feature_pipeline": feature_pipeline,
    "prior_scaled":     prior_scaled,
    "pca_diagnostic":   None,
}
s1["sc_idmap"] = getattr(config, "SC_IDMAP", None)

# Result
evaluate.report_step8(posterior, embedding_net, theta_scaled, x_input)


# ## Phase 3 — Attention Feature Selection
# Phase 2에서 학습된 RegionTransformer attention과
# posterior gradient를 이용해 파라미터 추론에
# 중요한 FC edge k=300개를 선택.

# In[11]:


# Phase 3: attention × gradient → top-k FC indices
# P6: skipped when RUN_PHASE24=False. final_test reads s1["posterior"]=Phase1,
# so Phase2/4 never affect the test result; Phase4 NLL was worse (-6.12 vs -8.49).
import numpy as np
if getattr(config, "RUN_PHASE24", True):
    from inference.stage1 import run_phase2
    phase3_result = run_phase2(
        phase1=s1,
        fc_obs=subject_data[train[0]]["fc"],
        k=config.SELECTION_K,
        n_samples=200,
        verbose=True,
    )
    fc_selected_indices = phase3_result["fc_selected_indices"]
else:
    fc_selected_indices = np.arange(config.FC_DIM)   # all edges, no selection
    print("[Phase 3] SKIPPED (RUN_PHASE24=False) — using all FC edges")

print(f"[Phase 3] 선택된 FC indices: {len(fc_selected_indices)}개")
print(f"  전체 FC 중 {len(fc_selected_indices)/config.FC_DIM*100:.1f}%")

# 중요도 맵 저장
s1["fc_selected_indices"] = fc_selected_indices


# ## Phase 4 — 2차 SBI (Final Posterior)
# 선택된 k개 FC feature로 RegionTransformer + SNPE-C 재학습.
# value+mask input으로 zero-fill ambiguity 해결.

# In[12]:


# Phase 4: FC(k) + mask → RegionTransformer → SNPE-C
# P6: when RUN_PHASE24=False, alias Phase4 posterior to Phase1 (full FC) so the
# downstream analysis/validation blocks stay defined without 48min of retrain.
if getattr(config, "RUN_PHASE24", True):
    from inference.stage1 import run_phase3
    phase4_result = run_phase3(
        phase1=s1,
        fc_selected_indices=fc_selected_indices,
        sc_matrix=subject_data[train[0]]["sc"],
        verbose=True,
    )
    posterior_2    = phase4_result["posterior"]
    embedding_net2 = phase4_result["embedding_net"]
    print(f"[Phase 4] 완료")
    print(f"  입력: FC({len(fc_selected_indices)}) + mask")
    print(f"  시뮬 재실행: 없음 (Phase 1 데이터 재사용)")
    print(f"  theta: {phase4_result['theta_scaled'].shape}")
    print(f"  fc:    {phase4_result['fc_raw'].shape}")
else:
    posterior_2    = s1["posterior"]        # alias Phase1 (full-FC)
    embedding_net2 = s1["embedding_net"]
    print("[Phase 4] SKIPPED (RUN_PHASE24=False) — posterior aliases Phase1 (full FC)")

s2 = {
    "posterior":           posterior_2,
    "embedding_net":       embedding_net2,
    "fc_selected_indices": fc_selected_indices,
    "theta_scaled":        s1["theta_scaled"],
    "theta_raw":           s1["theta_raw"],
    "fc_raw":              s1["fc_raw"],
    "param_scaler":        s1["param_scaler"],
    "prior_scaled":        s1["prior_scaled"],
}


# ## Phase 2/4 분석 지표
# posterior shrinkage, spatial parameter map,
# prior predictive vs posterior predictive 비교.

# In[ ]:


import numpy as np
import torch
import matplotlib.pyplot as plt
from features.fc import fc_to_upper_tri

# ── 1. Posterior 샘플링 (Phase2: raw FC, Phase4: selected FC) ──
# High-dim region-wise flows (e.g. 1440-dim) blow up GPU memory when sampling
# many draws WITH autograd on (the autoregressive inverse builds a huge graph).
# Sample under no_grad and cap the count for region-wise modes.
_RW = str(getattr(config, "PARAMETER_MODE", "homogeneous")) in (
    "latent_regionwise", "direct_regionwise", "basis_regionwise")
n_samples = min(config.N_POSTERIOR, 512) if _RW else config.N_POSTERIOR
fc_obs_0  = subject_data[train[0]]["fc"]
fc_vec_0  = fc_to_upper_tri(fc_obs_0)

if config.SBI_DEVICE == "cuda":
    torch.cuda.empty_cache()
with torch.no_grad():
    # x_obs MUST match the training conditioning. SC_CONDITION ON -> x=[idx|fc]
    # via build_x_obs (encoder input, dim 1+FC_DIM); OFF -> the FeaturePipeline
    # transform (PCA/whiten) exactly as before (baseline byte-identical).
    if getattr(config, "SC_CONDITION", False):
        from inference.posterior import build_x_obs as _build_x_obs
        _x1 = np.asarray(_build_x_obs(
            feature_pipeline, fc_vec_0, None,
            sid=train[0], fc_matrix=fc_obs_0), dtype=np.float32)
    else:
        _x1 = np.asarray(feature_pipeline.transform(
            fc_vec_0[None, :].astype(np.float32), np.zeros((1, 0), np.float32)),
            dtype=np.float32)
    samples_1 = posterior.sample(
        (n_samples,),
        x=torch.tensor(_x1).to(config.SBI_DEVICE),
        show_progress_bars=False,
        reject_outside_prior=False,
    ).detach().cpu().numpy()

    fc_vec_k  = fc_vec_0[fc_selected_indices]
    # Phase24 OFF: posterior_2 aliases Phase1 (full-FC PCA pipeline) -> reuse _x1.
    # Phase24 ON: posterior_2 trained on selected RAW edges -> pass them raw.
    if getattr(config, "RUN_PHASE24", False):
        _x2 = torch.tensor(fc_vec_k, dtype=torch.float32).unsqueeze(0).to(config.SBI_DEVICE)
    else:
        _x2 = torch.tensor(_x1).to(config.SBI_DEVICE)
    samples_2 = posterior_2.sample(
        (n_samples,),
        x=_x2,
        show_progress_bars=False,
        reject_outside_prior=False,
    ).detach().cpu().numpy()

# scaled prior std = 1 (BoxUniform[-1,1])
prior_std = 1.0
shrink_1 = 1 - (samples_1.std(axis=0) / prior_std) ** 2
shrink_2 = 1 - (samples_2.std(axis=0) / prior_std) ** 2

# ── 2. Shrinkage 요약 ──
print("\n[Shrinkage]")
names = config.STAGE1_PARAMS
_BIG = len(names) > 30          # region-wise (e.g. 1440) -> aggregate, no per-param plot
raw_2  = param_scaler.inverse_transform(samples_2)
mean_2 = raw_2.mean(axis=0)
std_2  = raw_2.std(axis=0)
if not _BIG:
    def _per(s):
        return "  ".join(f"{n}={v:.3f}" for n, v in zip(names, s))
    print(f"  Phase 2 mean: {shrink_1.mean():.3f}  ({_per(shrink_1)})")
    print(f"  Phase 4 mean: {shrink_2.mean():.3f}  ({_per(shrink_2)})")
    print("\n[복원된 파라미터 (posterior mean, raw)]")
    for n, m, s in zip(config.STAGE1_PARAMS, mean_2, std_2):
        print(f"  {n:8s}: mean={m:.3f}  std={s:.3f}")
    plot_data = [(samples_1[:, i], samples_2[:, i], n)
                 for i, n in enumerate(config.STAGE1_PARAMS)]
    fig, axes = plt.subplots(1, len(plot_data),
                             figsize=(5 * len(plot_data), 4), squeeze=False)
    for ax, (d1, d2, label) in zip(axes[0], plot_data):
        ax.boxplot([d1, d2], labels=['Phase2', 'Phase4'])
        ax.set_title(label); ax.set_ylabel('scaled value')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
    plt.suptitle('Posterior: Phase2 vs Phase4'); plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/phase2_vs_phase4_posterior.png",
                dpi=110, bbox_inches='tight')
    plt.close()
    print("[저장] phase2_vs_phase4_posterior.png")
else:
    # region-wise: aggregate shrinkage + restored params per HETERO param
    from param_decoder import group_indices_by_hetero
    print(f"  Phase 2 mean: {shrink_1.mean():.3f}  Phase 4 mean: {shrink_2.mean():.3f}"
          f"  ({len(names)} region-wise params)")
    groups = group_indices_by_hetero(names, list(config.HETERO_PARAMS))
    print("  per-HETERO-param (shrinkage mean[min,max] | restored mean±std over regions):")
    for p, idx in groups.items():
        if not idx:
            continue
        sm = shrink_2[idx]
        print(f"    {p:8s}: shrink {sm.mean():.3f}[{sm.min():.3f},{sm.max():.3f}] | "
              f"raw {mean_2[idx].mean():.4f}±{std_2[idx].mean():.4f}")


# ## Step 9. Stage 1 analysis (validation)
# 
# 9a validation metrics · 9b baseline · 9c MLP probing · 9d SBC · 9e posterior plots

# In[ ]:


s1_val_results, stage1_agg = evaluate.evaluate_validation_stage1(
    val, subject_data, s1, apply_bw=True, verbose=True,
)

baseline_agg = evaluate.baseline_eval_subjects(
    val, subject_data, n_resim=10, verbose=True,
)

# Region-wise (e.g. 1440 params): per-param probing/plots explode (1440 fits /
# subplots). Skip those; ActiveSubspace aggregates internally; SBC ranks still
# computed (used for aggregate diagnostics) but the per-param histogram is skipped.
_BIG_P = len(config.STAGE1_PARAMS) > 30

if not _USE_EMBEDDING:
    print("  [probing] skipped (no embedding — raw FC -> MAF; embedding probe N/A)")
elif not _BIG_P:
    inference.evaluate_embedding_probing(
        s1["embedding_net"], s1["theta_scaled"],
        s1["x_input"], config.STAGE1_PARAMS, verbose=True,
    )
else:
    print(f"  [probing] skipped ({len(config.STAGE1_PARAMS)} region-wise params; "
          f"per-param linear probe not informative at this dim)")

# Library-native (sbi) gradient sensitivity (internally aggregates if region-wise).
try:
    from active_sensitivity import report_active_sensitivity
    report_active_sensitivity(
        s1["posterior"], s1["theta_scaled"], config.STAGE1_PARAMS,
        x_obs=_x1,   # PCA-reduced obs (matches training conditioning dim)
        num_monte_carlo_samples=1000, verbose=True,
    )
except Exception as _e:
    print(f"  [ActiveSubspace] skipped: {type(_e).__name__}: {_e}")

# SBC simulates from the prior and builds x via the FeaturePipeline internally,
# which is not SC-aware (it can't prepend the [idx|fc] column). Skip in
# SC_CONDITION mode — SBC is a calibration diagnostic and does NOT affect the
# posterior-predictive FC corr (the core metric).
if getattr(config, "SC_CONDITION", False):
    print("  [SBC] skipped (SC_CONDITION: SBC needs SC-aware x=[idx|fc]; "
          "diagnostic only, posterior-predictive eval unaffected)")
    sbc_ranks = None
else:
    sbc_ranks = inference.simulation_based_calibration(
        s1["posterior"], s1["prior_scaled"], s1["param_scaler"],
        s1["feature_pipeline"], config.STAGE1_PARAMS,
        weights=subject_data[train[0]]["sc"],
        delays=subject_data[train[0]]["delays"],
        n_sbc=config.N_SBC, n_posterior=1000,
    )
if sbc_ranks is not None and not _BIG_P:
    evaluate.plot_sbc_rank_histogram(sbc_ranks, config.STAGE1_PARAMS)
    evaluate.plot_posteriors(
        stage1_agg["per_subject"],
        config.STAGE1_PARAMS,
        config.STAGE1_PRIOR_LOW,
        config.STAGE1_PRIOR_HIGH,
        title="Stage 1",
    )
elif sbc_ranks is not None:
    import numpy as _np
    _sbc = _np.asarray(sbc_ranks)
    print(f"  [SBC] {_sbc.shape[0]} sims x {_sbc.shape[1]} params ranked "
          f"(per-param histogram skipped for region-wise); "
          f"rank mean={_sbc.mean():.1f}")

# Result
evaluate.report_step9(stage1_agg, baseline_agg)


# ── Phase 4 (posterior_2) validation ─────────────────────
# P6: only when Phase2/4 actually ran. With RUN_PHASE24=False, Phase4 aliases
# Phase1 and fc_selected_indices = all edges, so this is redundant. (It also
# crashes: _SlicedPipeline assumes a 2-D x but evaluate_subject passes a 1-D
# single-subject FC vector -> x[:, idx] IndexError.)
if getattr(config, "RUN_PHASE24", True):
    class _SlicedPipeline:
        def __init__(self, pipe, idx):
            self.pipe, self.idx = pipe, idx
        def transform(self, fc, fcd=None):
            x = self.pipe.transform(fc, fcd)
            x = x[None, :] if x.ndim == 1 else x      # single-subject -> (1, D)
            return x[:, self.idx]

    s2_eval = dict(
        s2,
        feature_pipeline=_SlicedPipeline(
            s1["feature_pipeline"], fc_selected_indices,
        ),
    )
    s2_val_results, stage2_agg = evaluate.evaluate_validation_stage1(
        val, subject_data, s2_eval, apply_bw=True, verbose=True,
    )
    evaluate.report_step9(stage2_agg, baseline_agg)
else:
    stage2_agg = None   # Phase4 == Phase1; nothing extra to validate


# ## Step 13. Model selection (validation)

# In[ ]:


best_stage, scores = evaluate.select_best_model(
    stage1_agg, stage2_agg=None, baseline_agg=None, verbose=True,
)
score_1 = scores["stage1"]
score_2 = scores["stage2"]

# Result
evaluate.report_step13(best_stage, score_1, score_2, stage1_agg, None)


# ## Step 14. Final test

# In[ ]:


test_summary = evaluate.final_test(
    test, subject_data, best_stage, s1, None,
    n_resim=config.N_TEST_RESIM, apply_bw=True, verbose=True,
)
# split into images of 2 subjects each (test_fc_comparison_1.png, _2.png, ...)
_per = test_summary["per_subject"]
_grp = 2
_nimg = (len(_per) + _grp - 1) // _grp
for _i in range(0, len(_per), _grp):
    _k = _i // _grp + 1
    evaluate.plot_fc_comparison(
        _per[_i:_i + _grp],
        save_path=os.path.join(config.OUTPUT_DIR, f"test_fc_comparison_{_k}.png"),
        title=f"Test FC (Stage {best_stage}) — {_k}/{_nimg}",
    )

# Result
evaluate.report_step14(test_summary)

# ── Export test simulated + empirical FC matrices to CSV ──────────────────
# Per test subject: sim_fc_<sid>.csv (expected/mean resim FC), emp_fc_<sid>.csv
# (target), node_fit_<sid>.csv (per-region sim-vs-emp FC-row corr) + a
# node_fit_summary.csv (region x subject). node_fit is the probe for the
# front/back label-order fit asymmetry. Disable with EXPORT_FC_CSV=0.
if os.environ.get("EXPORT_FC_CSV", "1") == "1":
    try:
        from evaluation.export_fc_csv import export_test_fc_csv
        export_test_fc_csv(test_summary, subject_data, config.OUTPUT_DIR,
                           verbose=True)
    except Exception as _e:
        print(f"  [export FC CSV] skipped: {type(_e).__name__}: {_e}")

# ── Region-wise parameter maps (latent_regionwise or direct_regionwise) ─────
if (str(getattr(config, "PARAMETER_MODE", "homogeneous"))
        in ("latent_regionwise", "direct_regionwise", "basis_regionwise")
        and getattr(config, "SAVE_PARAM_MAPS", False)):
    from save_param_maps import save_param_maps as _save_maps
    print("\n  [Step 14b] region-wise parameter maps (val + test subjects)")
    print(f"    mode={config.PARAMETER_MODE}  active params: {config.HETERO_PARAMS} "
          f"(all regional incl g_LRE).")
    _save_maps(
        posterior, feature_pipeline, param_scaler, subject_data,
        list(val) + list(test),
        out_path=os.path.join(config.OUTPUT_DIR, "param_maps.npz"),
        n_post=getattr(config, "N_POSTERIOR", 1000), verbose=True,
    )


# ── Stage 6: SC-conditioned diagnostics (gated; never blocks the summary) ─────
# Decisive tests for whether SC-conditioning is REAL (not memorized/ignored) and
# whether the posterior beats the prior. Only meaningful in SC_CONDITION mode.
#   - boundary mass : posterior mass at the prior box edge (clip-fallback collapse)
#   - OOD distance  : how far the empirical FC sits from the simulated-FC train cloud
#   - median corr   : median-FC analogue of expected-FC
#   - prior-predictive: theta~prior resim corr  -> posterior must beat it (delta>0)
#   - SC-permutation : condition each test subject on a WRONG subject's SC; if corr
#                      does NOT drop vs correct-SC, the encoder ignores/memorizes SC.
if getattr(config, "SC_CONDITION", False) and os.environ.get("RUN_SC_DIAG", "1") == "1":
    try:
        from evaluation import sc_diagnostics as _scd
        from evaluation.metrics import _resimulate_and_score, fc_metrics
        from inference.posterior import infer_subject_raw as _infer
        from features.fc import fc_to_upper_tri as _f2u
        import numpy as _np

        def _exp_corr(_preds, _fcobs, _nan):
            if not _preds:
                return 0.0
            return fc_metrics(_fcobs, _np.mean(_np.stack(_preds, 0), 0),
                              nan_mask=_nan)["corr"]

        _n_d = int(getattr(config, "N_TEST_RESIM", 10))
        _fc_mean, _fc_std = _scd.fit_train_fc_stats(_np.asarray(fc_raw))
        _perm = _scd.permutation_index_map(config.SC_IDMAP)
        # prior-predictive theta (shared across test subjects)
        _th_pp_s = prior_scaled.sample((_n_d,)).cpu().numpy().astype(_np.float32)
        _th_pp_r = param_scaler.inverse_transform(_th_pp_s)

        print("\n" + "=" * 70)
        print("  Stage 6: SC-conditioned diagnostics (test subjects)")
        print("=" * 70)
        _rows = []
        for _r in test_summary["per_subject"]:
            _sid = _r["sid"]; _d = subject_data[_sid]
            _corr_post = float(_r.get("fc_corr_expected", 0.0))
            _bm = _scd.boundary_mass(_r["samples_scaled"])
            _med = _scd.median_fc_corr(_r["fc_preds"], _r["fc_obs"], _d.get("fc_nan"))["corr"]
            _ood = _scd.ood_distance(_f2u(_d["fc"]), _fc_mean, _fc_std)
            # prior-predictive on this subject's SC
            _, _, _, _pp_preds = _resimulate_and_score(
                _n_d, _th_pp_r, list(config.STAGE1_PARAMS), None,
                _d["sc"], _d["delays"], _d["fc"], None, True,
                sid=f"prior:{_sid}", verbose=False, fc_nan=_d.get("fc_nan"))
            _corr_prior = _exp_corr(_pp_preds, _d["fc"], _d.get("fc_nan"))
            # SC-permutation: condition on WRONG subject's SC, resim with OWN SC
            _fcv = _f2u(_d["fc"]).astype(_np.float32)
            _xp = _np.concatenate(
                [_np.array([[float(_perm[int(_sid)])]], _np.float32),
                 _fcv[None, :]], axis=1)
            _sp, _, _, _ = _infer(posterior, _xp, param_scaler,
                                  n_samples=_n_d, verbose=False)
            _, _, _, _perm_preds = _resimulate_and_score(
                _n_d, _sp, list(config.STAGE1_PARAMS), None,
                _d["sc"], _d["delays"], _d["fc"], None, True,
                sid=f"perm:{_sid}", verbose=False, fc_nan=_d.get("fc_nan"))
            _corr_perm = _exp_corr(_perm_preds, _d["fc"], _d.get("fc_nan"))
            _rows.append((_sid, _corr_post, _corr_prior, _corr_perm, _bm,
                          _ood["z_rms"], _med))
            print(f"  {_sid}: corr_post={_corr_post:+.4f} prior={_corr_prior:+.4f} "
                  f"perm={_corr_perm:+.4f}  d(post-prior)={_corr_post-_corr_prior:+.4f} "
                  f"d(post-perm)={_corr_post-_corr_perm:+.4f}  bndry={_bm:.3f} "
                  f"ood_z={_ood['z_rms']:.2f} med={_med:+.4f}")
        if _rows:
            _a = _np.asarray([(r[1], r[2], r[3], r[4], r[5]) for r in _rows])
            print("-" * 70)
            print(f"  MEAN: post={_a[:,0].mean():+.4f} prior={_a[:,1].mean():+.4f} "
                  f"perm={_a[:,2].mean():+.4f}  delta(post-prior)={(_a[:,0]-_a[:,1]).mean():+.4f} "
                  f"delta(post-perm)={(_a[:,0]-_a[:,2]).mean():+.4f}")
            print(f"  interp: delta(post-prior)>0 => posterior beats prior; "
                  f"delta(post-perm)>0 => SC channel USED (not ignored/memorized).")
            print(f"  boundary mass mean={_a[:,3].mean():.3f} (high+low shrink=clip collapse); "
                  f"OOD z_rms mean={_a[:,4].mean():.2f}")
        print("=" * 70)
    except Exception as _e:
        print(f"  [Stage 6 diagnostics] skipped: {type(_e).__name__}: {_e}")


# ## Save outputs and summary

# In[ ]:


inference.save_artifacts(
    os.path.join(config.OUTPUT_DIR, "artifacts.pkl"),
    stage1_param_scaler     = s1["param_scaler"].to_dict(),
    stage1_feature_pipeline = s1["feature_pipeline"],
    stage1_pca_diagnostic   = s1["pca_diagnostic"],
    best_stage              = best_stage,
    n_regions               = config.N_REGIONS,
    fc_dim                  = config.FC_DIM,
    prior_low               = config.STAGE1_PRIOR_LOW,
    prior_high              = config.STAGE1_PRIOR_HIGH,
    param_names_s1          = config.STAGE1_PARAMS,
    feature_config          = {
        # FC is raw upper-tri passthrough (z-scored, no PCA) in this
        # pipeline, so fc_pca may be absent — stay None-safe.
        "pca_dim_fc":  (
            getattr(s1["feature_pipeline"], "fc_pca", None).n_components
            if getattr(s1["feature_pipeline"], "fc_pca", None) is not None
            else None
        ),
        "pca_dim_fcd": None,
    },
)

torch.save(
    s1["embedding_net"].state_dict(),
    os.path.join(config.OUTPUT_DIR, "embedding_net_s1.pt"),
)

evaluate.print_final_summary(
    stage1_agg, None, best_stage, test_summary,
    train, len(s1["theta_scaled"]),
)


# ## cuBNM Benchmark (optional)
# 
# VBI WC_sde+cupy 엔진과 cuBNM `WCSimGroup` 엔진을 같은 SC/theta 배치로 비교.
# 기본 비활성화 (`RUN_CUBNM_BENCHMARK = False`). cubnm 미설치 시 안내 후 skip.

# In[ ]:


# ===== cuBNM Benchmark (optional) =====
RUN_CUBNM_BENCHMARK = False    # set True to run the VBI vs cuBNM benchmark
N_BENCH = 50                   # number of simulations to time

if not RUN_CUBNM_BENCHMARK:
    print("cuBNM benchmark skipped (RUN_CUBNM_BENCHMARK = False)")
else:
    import numpy as np
    try:
        import cubnm  # noqa: F401  — presence check only
    except ImportError:
        print("cubnm not installed. Install with:\n    pip install cubnm")
    else:
        from cuBNM import benchmark as _bench

        # First training subject's SC (from subject_data, built in Step 1)
        _sid0    = train[0]
        _weights = subject_data[_sid0]["sc"]

        # Random theta from the current STAGE1 prior bounds
        _low  = np.asarray(config.STAGE1_PRIOR_LOW,  dtype=np.float32)
        _high = np.asarray(config.STAGE1_PRIOR_HIGH, dtype=np.float32)
        _rng  = np.random.default_rng(config.SEED)
        _theta = _rng.uniform(
            _low, _high, size=(N_BENCH, len(config.STAGE1_PARAMS))
        ).astype(np.float32)

        _res = _bench.benchmark(
            _weights, _theta, list(config.STAGE1_PARAMS),
            duration_s=(config.T_END - config.T_CUT) / 1000.0,
            tr_s=config.TR_SEC,
            dt_ms=config.DT * config.DECIMATE,
            apply_bw=True,
            force_gpu=True,
        )

        # Side-by-side table
        print(f"subject={_sid0}  N_BENCH={N_BENCH}  params={list(config.STAGE1_PARAMS)}")
        print(f"{'engine':8s} {'n_sims':>7s} {'wall_sec':>9s} {'ms/sim':>9s} {'BOLD_shape':>16s}")
        for _eng in ("vbi", "cubnm"):
            _r = _res.get(_eng) or {}
            if "error" in _r:
                print(f"{_eng:8s}  ERROR: {_r['error']}")
            else:
                print(f"{_eng:8s} {_r['n_sims']:>7d} {_r['wall_sec']:>9.2f} "
                      f"{_r['ms_per_sim']:>9.2f} {str(_r['bold_shape']):>16s}")

        _d = _res.get("fc_abs_diff")
        if _d is not None:
            print(f"FC sanity: mean|FC_vbi - FC_cubnm| = {_d:.4f}")
        else:
            print("FC sanity: skipped (one engine produced no BOLD)")

