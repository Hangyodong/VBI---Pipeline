"""Pipeline configuration for Mouse MPTP VBI-SBI (115 region).

All hyperparameters, paths, and prior bounds live here.
Edit this file (not other modules) when tuning the pipeline.

Pipeline overview
-----------------
 1. Data split (train / val / test)
 2. VBI Wilson-Cowan simulation
 3. Feature extraction (FC + FCD)
 4. Feature preprocessing (z-score, train fit only)
 5. Feature embedding (FC PCA + FCD PCA -> MLP)
 6. Embedding quality check (PCA diagnostic + MLP probing)
 7. Parameter preprocessing ([-1, 1] scaling)
 8. Stage 1 inference (single-round SNPE-C on {P, Q, g_e, g_i})
 9. Stage 1 analysis
10. Stage 2 parameter selection (theta_bad + c-params)
11. Stage 2 inference
12. Stage 2 analysis
13. Model selection (validation)
14. Final test
"""
import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = "/scratch/home/wog3597/vbi"
FC_PATH = f"{DATA_DIR}/MPTP_FC_115.mat"
SC_PATH = f"{DATA_DIR}/MPTP_SC_115.mat"
TSV_PATH = f"{DATA_DIR}/participants.tsv"
ATLAS_PATH = f"{DATA_DIR}/atlas_115_labels.txt"
BOLD_PATH = f"{DATA_DIR}/MPTP_BOLD_115.mat"        # optional
OUTPUT_DIR = "./output_mouse_mptp"


# ---------------------------------------------------------------------------
# Data dimensions
# ---------------------------------------------------------------------------

N_REGIONS = 115
FC_DIM = N_REGIONS * (N_REGIONS - 1) // 2          # 6555
FCD_DIM = 5                                        # summary stats: mean,std,q25,q50,q75

GROUP_FILTER = ("ctr", "MPTP")

# FC sources in MPTP_FC_115.mat
#   col 1 (=2nd row) = FC (uses NaN values where unmeasured)
#   col 2 (=3rd row) = FCD (NaN-free 115x115 matrix; used directly)
FC_COL = 1
FCD_COL = 2

# SC source in MPTP_SC_115.mat
#   col 1 (=2nd row) = uint16 raw counts -> internally log1p + max-norm
#   col 2 (=3rd row) = tract length (mm)
SC_WEIGHT_COL = 1        # SC coupling weight
SC_LENGTH_COL = 2        # tract length (mm)
# Backward-compatible aliases (existing modules read SC_COL)
SC_COL = SC_WEIGHT_COL

# NaN mask handling.
# FC col 1 has NaN-affected rows.
# We replace NaN with 0 (rather than masking) so that simulated FC (NaN-free)
# and observed FC stay at the same 6555 dim.
# Constant 0 rows contribute no variance and are effectively ignored by PCA.
NAN_MASK = None
NAN_REGIONS = []


# ---------------------------------------------------------------------------
# Subject split
# ---------------------------------------------------------------------------

N_TRAIN = 4
N_VAL = 2
N_TEST = 2
SEED = 42


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

ENGINE = "gpu"
N_SIM = 50_000              # Stage 1 simulations per subject (H100 94GB)
N_SIM_S2 = 50_000           # Stage 2 simulations per subject
GPU_BATCH = 50_000          # H100 94GB: 50k×115×float32 ≈ 6GB → 배치 1회

DT = 0.5                    # integration step (ms)
T_END = 300_000.0           # total simulation length (ms) - 300s (production)
T_CUT =  60_000.0           # transient cut (ms) - 60s (production)
DECIMATE = 2
FS_NEURAL = 1000.0 / (DT * DECIMATE)

TR_SEC = 1.0
ANALYSIS_BOLD_T = int((T_END - T_CUT) / (DT * DECIMATE) /
                      (TR_SEC * 1000.0 / (DT * DECIMATE)))
FS_BOLD = 1.0 / TR_SEC

# ---------------------------------------------------------------------------
# HRF (TVB MixtureOfGammas)
# ---------------------------------------------------------------------------
# equation: (l*t)^(a1-1)*exp(-l*t)/gamma(a1) - c*(l*t)^(a2-1)*exp(-l*t)/gamma(a2)
# Mouse fMRI: peak ~3s (human: ~6s), undershoot ~7s (human: ~13s)
HRF_A1 = 3.0          # shape of positive gamma  (peak ~ a1/l sec)
HRF_A2 = 7.0          # shape of undershoot gamma
HRF_L = 1.0           # rate parameter
HRF_C = 0.3           # undershoot amplitude ratio (mouse weaker than human)
HRF_LENGTH_SEC = 32.0  # kernel length (sec)


# ---------------------------------------------------------------------------
# Wilson-Cowan fixed parameters (VBI nominal)
# ---------------------------------------------------------------------------

WC_FIXED = {
    "c_ee": 16.0, "c_ei": 12.0,
    "c_ie": 15.0, "c_ii": 3.0,
    "tau_e": 8.0, "tau_i": 8.0,
    "a_e": 1.3, "a_i": 2.0,
    "b_e": 4.0, "b_i": 3.7,
    "noise_amp": 0.01,
    "dt": DT, "t_end": T_END, "t_cut": T_CUT,
    "method": "heun", "decimate": DECIMATE,
    "RECORD_EI": "E", "dtype": "float32",
}

VELOCITY_M_PER_S = 1.5

BW = {
    "tau_s": 0.8, "tau_f": 2.5, "tau_0": 0.7,
    "alpha": 0.32, "epsilon": 0.6,
    "E_0": 0.4, "V_0": 0.02,
    "TE": 0.018, "TR": TR_SEC,
}


# ---------------------------------------------------------------------------
# Stage 1 parameter prior  (1st SBI: global working point + global coupling)
# ---------------------------------------------------------------------------

STAGE1_PARAMS = ["P", "Q", "g_e", "g_i"]
STAGE1_PRIOR_LOW = [0.5, 0.0, 0.0, 0.0]
STAGE1_PRIOR_HIGH = [2.5, 2.0, 1.5, 1.5]

# Explicit alias for the refactored API
PARAM_NAMES_STAGE1 = STAGE1_PARAMS


# ---------------------------------------------------------------------------
# Stage 2: theta_bad (from Stage 1) + local E/I coupling
# ---------------------------------------------------------------------------

LOCAL_EI_PARAMS = ["c_ee", "c_ei", "c_ie", "c_ii"]
PARAM_NAMES_STAGE2_BASE = LOCAL_EI_PARAMS    # to be extended with theta_bad


# ---------------------------------------------------------------------------
# Stage 2 c-parameter prior
# ---------------------------------------------------------------------------

C_PARAM_PRIOR = {
    "c_ee": (12.0, 20.0),
    "c_ei": (8.0, 16.0),
    "c_ie": (10.0, 20.0),
    "c_ii": (1.0, 6.0),
}

DIFFICULT_SHRINKAGE = 0.3
NUISANCE_METHOD = "posterior_sample"   # or "fix_mean"


# ---------------------------------------------------------------------------
# SBI
# ---------------------------------------------------------------------------

SBI_DEVICE = (
    "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
)
N_POSTERIOR = 2000
NDE_HIDDEN     = 128         # 64 → 128
NDE_TRANSFORMS = 8           # 5 → 8 (posterior 표현력↑)
NDE_MODEL = "maf"
USE_MIXED_PRECISION = True


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBED_DIM    = 256           # 128 → 256
EMBED_HIDDEN = 512           # 256 → 512 (H100 Tensor Core 활용)
USE_EMBEDDING: bool = False  # False -> nn.Identity fallback in train_snpe

PCA_DIM_FC = 2000           # FC upper triangle -> 2000 PCs (PCA 품질 개선)
PCA_DIM_FCD = 100           # FCD upper triangle -> 100 PCs
PCA_DIM = PCA_DIM_FC        # alias

PCA_EVR_THRESHOLD = 0.90
PCA_TARGET_EVR = 0.95
PCA_RECON_CORR_THRESH = 0.95


# ---------------------------------------------------------------------------
# FCD computation knobs
# ---------------------------------------------------------------------------

# Observed FCD is loaded directly from FCD_COL; no computation needed.
# Simulated FCD = element-wise std of sliding-window FCs.
FCD_WINDOW_TR = 60
FCD_STRIDE_TR = 3

# Kept only for backward compatibility; new code does not use summary stats.
FCD_SUMMARY_STATS = []
FCD_SUMMARY_DIM = 0


# ---------------------------------------------------------------------------
# Feature toggles
# ---------------------------------------------------------------------------

USE_FC = True
USE_FCD = False  # Disabled by default: FCD computation has issues with
USE_PSD = False             # excluded

# Feature set selector. With empirical FC only (no BOLD time series),
# we must run in "fc_only" mode so that simulated and observed features
# share the same pipeline.
#   "fc_only" : simulated FC upper-tri  vs  empirical FC upper-tri
#   "fc_fcd"  : requires empirical BOLD time series  (raises if absent)
FEATURE_SET = "fc_only"


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------
# "final" : T_END=300s, T_CUT=60s   (production)
# "debug" : T_END=5s,   T_CUT=1s    (smoke tests only)
SIM_MODE = "final"
DEBUG_SIM = False


# ---------------------------------------------------------------------------
# Probing and calibration
# ---------------------------------------------------------------------------

EMB_PROBE_R2_THRESHOLD = 0.5

N_PPC = 50
N_SBC = 200
SBC_BINS = 20

N_TEST_RESIM = 50
BOOTSTRAP_N = 1000


# ---------------------------------------------------------------------------
# Model selection weights
# ---------------------------------------------------------------------------

SELECT_W_FC_CORR = 1.0
SELECT_W_FC_RMSE = 0.5
SELECT_W_FCD_RMSE = 0.5


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

N_CPU = max(1, (os.cpu_count() or 8) - 2)
HAS_BOLD = None


def print_config():
    """Print a short summary of the active configuration."""
    print("=" * 70)
    print(f"  Mouse MPTP - VBI-SBI Pipeline ({N_REGIONS} regions)")
    print("=" * 70)
    print(f"  Engine          : {ENGINE} (SBI: {SBI_DEVICE})")
    print(f"  Split           : train={N_TRAIN} / val={N_VAL} / test={N_TEST}")
    print(
        f"  Sim time        : T_end={T_END / 1000:.0f}s, "
        f"cut={T_CUT / 1000:.0f}s"
    )
    print(f"  Analysis BOLD   : {ANALYSIS_BOLD_T} TR (TR={TR_SEC}s)")
    print(
        f"  Regions         : {N_REGIONS} "
        f"(FC dim={FC_DIM}, FCD dim={FCD_DIM})"
    )
    print(f"  FC source       : col {FC_COL} (NaN -> 0, raw Pearson r)")
    print(f"  FCD source      : col {FCD_COL} (summary stats: mean,std,q25,q50,q75)")
    print(f"  SC source       : col {SC_COL} (raw -> log1p + max-norm)")
    print(f"  Velocity        : {VELOCITY_M_PER_S} m/s")
    print(f"  Stage 1 params  : {STAGE1_PARAMS}")
    print(f"  N_SIM           : {N_SIM} per subject")
    print(f"  GPU batch       : {GPU_BATCH}  (= N_SIM: 배치 1회)")
    if T_END < 100_000:
        print(f"  ⚠ DEBUG mode   : T_end={T_END/1000:.0f}s T_cut={T_CUT/1000:.0f}s (production: 300s/60s)")
    print(f"  PCA             : FC -> {PCA_DIM_FC}, FCD -> {PCA_DIM_FCD}")
    print(f"  Embedding       : MLP {EMBED_HIDDEN} -> {EMBED_DIM}")
    print(f"  Features        : FC={USE_FC} FCD={USE_FCD} PSD={USE_PSD}")
    print(f"  Nuisance method : {NUISANCE_METHOD}")
    print(f"  Mixed precision : {USE_MIXED_PRECISION}")


# ---------------------------------------------------------------------------
# Module-level sanity checks (run on import)
# ---------------------------------------------------------------------------

assert FC_DIM == N_REGIONS * (N_REGIONS - 1) // 2, (
    f"FC_DIM={FC_DIM} inconsistent with N_REGIONS={N_REGIONS} "
    f"(expected {N_REGIONS * (N_REGIONS - 1) // 2})"
)
assert SC_WEIGHT_COL != SC_LENGTH_COL, (
    "SC_WEIGHT_COL and SC_LENGTH_COL must differ"
)
assert STAGE1_PARAMS == PARAM_NAMES_STAGE1, (
    "STAGE1_PARAMS and PARAM_NAMES_STAGE1 must be identical"
)
assert "noise_amp" not in STAGE1_PARAMS, (
    "noise_amp must not be in STAGE1_PARAMS (kept fixed in WC_FIXED)"
)
assert FEATURE_SET in ("fc_only", "fc_fcd"), (
    f"FEATURE_SET must be 'fc_only' or 'fc_fcd', got {FEATURE_SET!r}"
)
assert SIM_MODE in ("debug", "final"), (
    f"SIM_MODE must be 'debug' or 'final', got {SIM_MODE!r}"
)
