"""Pipeline configuration for Mouse MPTP VBI-SBI (115 region).

All hyperparameters, paths, and prior bounds live here.
Edit this file (not other modules) when tuning the pipeline.

Pipeline overview
-----------------
 1. Data split (train / val / test)
 2. WC-EIB simulation (global scalars P, Q, g_e, g_i, c_ei)
 3. Feature extraction (FC upper-tri, raw passthrough — no PCA)
 5. Parameter preprocessing ([-1, 1] scaling)
 6. Phase 1: SNPE-C on raw 6555-dim FC (no embedding)
 7. Phase 2: attention × gradient feature selection (top-k)
 8. Phase 3: SNPE-C on selected FC via RegionTransformer embedding
 9. Validation
10. Final test
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

# Size of the subject pool to use (HCP: smallest N_SUBJECTS subject-ids first).
# train+val+test are drawn from this pool. Ignored by the MPTP group-filter path.
N_SUBJECTS = 100


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

ENGINE = "gpu"
N_SIM = 50_000              # Stage 1 simulations per subject (H100 94GB)
GPU_BATCH = 10_000          # H100: 10k/batch (shared GPU / MPS concurrency headroom)

DT = 1.0                    # integration step (ms) - cubnm recommends >=1.0 with delays (~2x faster)
T_END = 720_000.0           # total simulation length (ms) - 12 min (matches real fMRI scan)
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
# NOTE: these gamma-HRF params ONLY feed the hrf="vbi" MixtureOfGammas path (bold.py).
# Production rwweib2 sims use hrf="bw" (cuBNM Balloon-Windkessel), so changing them does
# NOT affect current FC. Set to HUMAN canonical (was mouse: A1=3 -> peak ~3s) for the
# case the vbi path is ever used on HCP.
HRF_A1 = 6.0          # human: response peak ~6s (a1/l)
HRF_A2 = 16.0         # human: undershoot peak ~16s
HRF_L = 1.0           # rate parameter
HRF_C = 0.167         # human undershoot amplitude ratio (~1/6)
HRF_LENGTH_SEC = 32.0  # kernel length (sec)


# ---------------------------------------------------------------------------
# Wilson-Cowan fixed parameters (VBI nominal)
# ---------------------------------------------------------------------------

WC_FIXED = {
    "c_ee": 16.0,
    "c_ei": 12.0,       # inferred (center) global scalar
    "c_ie": 15.0,
    "c_ii": 3.0,
    "tau_e": 8.0,
    "tau_i": 8.0,
    "a_e": 1.3,   "a_i": 2.0,
    "b_e": 4.0,   "b_i": 3.7,
    "c_e": 1.0,   "c_i": 1.0,
    "alpha_e": 1.0, "alpha_i": 1.0,
    "theta_e": 0.0, "theta_i": 0.0,
    "k_e": 0.994, "k_i": 0.999,
    "r_e": 1.0,   "r_i": 1.0,
    "P": 0.5,     "Q": 0.0,
    "I_ext": 0.0, "lamda": 1.0,
    "rE_max_hz": 20.0, "rI_max_hz": 20.0,
    "noise_amp": 0.005,
    "dt": DT, "t_end": T_END, "t_cut": T_CUT,
    "method": "heun", "decimate": DECIMATE,
    "RECORD_EI": "E", "dtype": "float32",
}

# ---------------------------------------------------------------------------
# RWW-EIB-FFI fixed parameters (tvboptim Reduced Wong-Wang E/I, standard values)
# Used when INFERENCE_MODEL == "rwweib" (cuBNM RWWEIBSimGroup, engine="rwweib").
# Inferred params (g_LRE, g_FFI, sigma) are taken from theta, not from here.
# NOTE a_i = 615.0 (NOT 0.615); inhibitory current uses J_N * S_e.
# ---------------------------------------------------------------------------
# Fixed regional params for RWW-EIB-FFI (the rest — a_E/b_E/d/gamma/tau/w_E/w_I/
# w_II/I_ext — are compile-time constants in cuBNM/rww_eib.yaml, not set here).
# Inferred (g_LRE/g_FFI/sigma/I_o) come from theta, not from this dict.
RWWEIB_FIXED = {
    "w_p": 1.4, "J_N": 0.15, "J_i": 1.0,
}

# Stock cuBNM rWW (Deco 2014). FIC auto-tunes wIE when on (then wIE not inferred).
RWW_DO_FIC = True
RWW_FIXED = {"wIE": 1.0}   # used only when RWW_DO_FIC is False

VELOCITY_M_PER_S = 1.5

# Conduction delays: when True, the cuBNM adapter feeds the per-subject delay
# matrix into the cubnm core (sc_dist + v) so inter-regional delays are
# simulated. When False, delays are ignored (legacy behaviour).
USE_DELAYS = True

BW = {
    "tau_s": 0.8, "tau_f": 2.5, "tau_0": 0.7,
    "alpha": 0.32, "epsilon": 0.6,
    "E_0": 0.4, "V_0": 0.02,
    "TE": 0.018, "TR": TR_SEC,
}


# ---------------------------------------------------------------------------
# Prior bounds (edit directly here)
# ---------------------------------------------------------------------------
# Global scalars P, Q, g_e, g_i, c_ei
PRIOR_P_LOW,   PRIOR_P_HIGH   = 0.0, 10.0
PRIOR_Q_LOW,   PRIOR_Q_HIGH   = 0.0, 10.0
PRIOR_GE_LOW,  PRIOR_GE_HIGH  = 0.0, 10.0
PRIOR_GI_LOW,  PRIOR_GI_HIGH  = 0.0, 10.0
PRIOR_CEI_LOW, PRIOR_CEI_HIGH = 6.0, 18.0

# RWW-EIB-FFI priors (first-fitting ranges). Widen per the notes below if needed:
#   stable dynamics  -> raise PRIOR_SIGMA_HIGH to 0.05
#   weak FC coupling -> raise PRIOR_GLRE/GFFI_HIGH to 5.0
PRIOR_GLRE_LOW,  PRIOR_GLRE_HIGH  = 0.0, 3.0
PRIOR_GFFI_LOW,  PRIOR_GFFI_HIGH  = 0.0, 3.0
PRIOR_SIGMA_LOW, PRIOR_SIGMA_HIGH = 0.0, 0.03
PRIOR_IO_LOW,    PRIOR_IO_HIGH    = 0.3, 0.45   # background current I_o (nom 0.382)

# Stock rWW (Deco 2014) priors. wIE auto-tuned by FIC (not inferred).
PRIOR_G_LOW,    PRIOR_G_HIGH    = 0.0, 7.0      # global coupling
PRIOR_WP_LOW,   PRIOR_WP_HIGH   = 0.0, 2.0      # local excitatory recurrence
PRIOR_JN_LOW,   PRIOR_JN_HIGH   = 0.001, 0.5    # NMDA coupling
PRIOR_RWWSIG_LOW, PRIOR_RWWSIG_HIGH = 0.0, 0.05  # noise


# ---------------------------------------------------------------------------
# Stage 1 parameter prior  —  model selected by INFERENCE_MODEL
# ---------------------------------------------------------------------------
# "wc"     -> Wilson-Cowan (WCVBI, engine="cubnm"): P, Q, g_e, g_i, c_ei
# "rwweib" -> RWW-EIB-FFI (RWWEIBSimGroup, engine="rwweib"): g_LRE, g_FFI, sigma, I_o
# "rww"    -> stock reduced Wong-Wang (rWWSimGroup, engine="rww"): G, w_p, J_N, sigma
INFERENCE_MODEL = "wc"

if INFERENCE_MODEL == "rwweib":
    STAGE1_PARAMS = ["g_LRE", "g_FFI", "sigma", "I_o"]
    STAGE1_PRIOR_LOW  = [PRIOR_GLRE_LOW,  PRIOR_GFFI_LOW,  PRIOR_SIGMA_LOW,  PRIOR_IO_LOW]
    STAGE1_PRIOR_HIGH = [PRIOR_GLRE_HIGH, PRIOR_GFFI_HIGH, PRIOR_SIGMA_HIGH, PRIOR_IO_HIGH]
elif INFERENCE_MODEL == "rww":
    STAGE1_PARAMS = ["G", "w_p", "J_N", "sigma"]
    STAGE1_PRIOR_LOW  = [PRIOR_G_LOW,  PRIOR_WP_LOW,  PRIOR_JN_LOW,  PRIOR_RWWSIG_LOW]
    STAGE1_PRIOR_HIGH = [PRIOR_G_HIGH, PRIOR_WP_HIGH, PRIOR_JN_HIGH, PRIOR_RWWSIG_HIGH]
else:
    # Global scalars: g_e*(SC@E) excitatory, g_i*(SC@I) inhibitory, P, Q drives,
    # c_ei E->I coupling.
    STAGE1_PARAMS = ["P", "Q", "g_e", "g_i", "c_ei"]
    STAGE1_PRIOR_LOW = [
        PRIOR_P_LOW, PRIOR_Q_LOW,
        PRIOR_GE_LOW, PRIOR_GI_LOW, PRIOR_CEI_LOW,
    ]
    STAGE1_PRIOR_HIGH = [
        PRIOR_P_HIGH, PRIOR_Q_HIGH,
        PRIOR_GE_HIGH, PRIOR_GI_HIGH, PRIOR_CEI_HIGH,
    ]

# Explicit alias for the refactored API
PARAM_NAMES_STAGE1 = STAGE1_PARAMS


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
# WC-EIB 파이프라인에서는 Stage 2 미사용. 향후 Stage 2 재설계 시
# prior/threshold 등을 이 위치에 다시 정의한다.

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

EMBED_DIM    = 128
# RegionTransformer is trained jointly with SNPE-C in Phase 3 (raw FC
# passthrough → per-region tokens → Transformer → CLS → EMBED_DIM).
REGION_TRANSFORMER_HEADS   = 4
REGION_TRANSFORMER_LAYERS  = 2
REGION_TRANSFORMER_D_MODEL = 128

SELECTION_K = 300             # Phase 2: top-k FC entries kept for Phase 3


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
# "final" : T_END=720s, T_CUT=60s   (production, matches 12 min fMRI)
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
        print(f"  ⚠ DEBUG mode   : T_end={T_END/1000:.0f}s T_cut={T_CUT/1000:.0f}s (production: 720s/60s)")
    print(f"  Embedding       : RegionTransformer (raw FC passthrough) -> {EMBED_DIM}")
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


# --- species config injection (append to bottom of config.py) ---

def apply_species_config(species_cfg):
    """
    Overwrite config module attributes from a species_configs/*.py module.
    MUST be called before any other module imports config.
    Called from main_mouse.py / main_human.py as the very first operation.
    """
    import sys
    this = sys.modules[__name__]

    _SPECIES_FIELDS = [
        "SPECIES",
        "N_REGIONS",
        "T_END", "T_CUT", "DT", "TR_SEC", "DECIMATE",
        "VELOCITY_M_PER_S",
        "HRF_A1", "HRF_A2", "HRF_C", "HRF_LENGTH_MS",
        "WC_FIXED",
        "STAGE1_PARAMS", "STAGE1_PRIOR_LOW", "STAGE1_PRIOR_HIGH",
    ]
    for k in _SPECIES_FIELDS:
        if hasattr(species_cfg, k):
            setattr(this, k, getattr(species_cfg, k))

    # Recompute derived constants that depend on N_REGIONS or timing
    this.FC_DIM = this.N_REGIONS * (this.N_REGIONS - 1) // 2
    this.ANALYSIS_BOLD_T = int(
        (this.T_END - this.T_CUT)
        / (this.DT * this.DECIMATE)
        / (this.TR_SEC * 1000.0 / (this.DT * this.DECIMATE))
    )
    assert this.FC_DIM == this.N_REGIONS * (this.N_REGIONS - 1) // 2
    assert this.ANALYSIS_BOLD_T > 0
