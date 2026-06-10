# species_configs/human_config.py
# Human entrypoint species config.
#
# ⚠ PLACEHOLDER fields below — N_REGIONS, HRF_*, T_END/T_CUT/DT/TR_SEC/DECIMATE,
#   STAGE1_PARAMS/PRIOR were COPIED FROM mouse_config and are NOT yet validated
#   for a human atlas. Only WC_FIXED and VELOCITY_M_PER_S come from the reference
#   parameters supplied for this task. Set the human atlas/HRF/prior values
#   before running the human pipeline.

SPECIES = "human"

# ⚠ PLACEHOLDER — set human atlas region count
N_REGIONS = 115

# ⚠ PLACEHOLDER — simulation timing (ms), copied from mouse
T_END   = 300_000.0
T_CUT   =  60_000.0
DT      = 0.5
TR_SEC  = 1.0
DECIMATE = 2

# Conduction velocity (tract_conduction_speed)
VELOCITY_M_PER_S = 1.0

# ⚠ PLACEHOLDER — HRF (TVB MixtureOfGammas), copied from mouse.
# Human canonical HRF differs (peak ≈ 6 s, undershoot ≈ 13 s) — update.
HRF_A1        = 3.0
HRF_A2        = 7.0
HRF_C         = 0.3
HRF_LENGTH_MS = 32_000.0

# Wilson-Cowan fixed parameters (Stage 1 non-inferred)
# P, Q, g_e, g_i, c_ei are INFERRED — must NOT appear in WC_FIXED
# Keys mapped to WC_sde.valid_params (wc_ prefix stripped).
# additive_noise_sigma (sigma) -> noise_amp; tract_conduction_speed -> VELOCITY_M_PER_S.
# c_ei (center ~12.0) intentionally omitted: c_ei is inferred (in STAGE1_PARAMS).
WC_FIXED = dict(
    c_ee=16.0, c_ie=15.0, c_ii=3.0,
    r_e=1.0,    r_i=1.0,
    tau_e=8.0,  tau_i=8.0,
    alpha_e=1.0, alpha_i=1.0,
    theta_e=0.0, theta_i=0.0,
    k_e=0.994,  k_i=0.999,
    a_e=1.3,    a_i=2.0,
    b_e=4.0,    b_i=3.7,
    c_e=1.0,    c_i=1.0,
    lamda=1.0,
    noise_amp=0.005,
)

# Inferred parameters (single-stage: P, Q, g_e, g_i, c_ei)
STAGE1_PARAMS     = ["P",  "Q",  "g_e", "g_i", "c_ei"]
STAGE1_PRIOR_LOW  = [0.0,  0.0,  0.0,   0.0,    6.0]
STAGE1_PRIOR_HIGH = [3.0,  3.0,  1.0,   1.0,   18.0]
