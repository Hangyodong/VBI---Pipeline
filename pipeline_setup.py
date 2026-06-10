"""Pipeline setup utilities.

Centralizes configuration override and module reload so that
``main.py`` and ``main.ipynb`` only need to call one or two functions.

Typical use
-----------
>>> from pipeline_setup import PipelineConfig, setup_pipeline
>>> cfg = PipelineConfig(N_SIM=10_000, GPU_BATCH=4_000)
>>> setup_pipeline(cfg)

Or directly with keyword arguments::

>>> setup_pipeline(N_SIM=10_000, GPU_BATCH=4_000)

Cell output is captured by the notebook itself (``.ipynb``), so this
module does not implement file logging.
"""
import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from typing import List


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """User-facing parameters.

    Each field maps to a `config` attribute. Defaults reflect a full
    H100 NVL run; reduce N_SIM / T_END_MS for quick tests.
    """

    # ── Species / atlas ──
    SPECIES: str = "mouse"
    N_REGIONS: int = 115
    VELOCITY_M_PER_S: float = 1.5

    # ── Paths ──
    DATA_DIR: str = "/scratch/home/wog3597/vbi"
    OUTPUT_DIR: str = "./output_mouse_mptp"
    # Data file basenames (resolved against DATA_DIR). Override for human.
    FC_FILE: str = "MPTP_FC_115.mat"
    SC_FILE: str = "MPTP_SC_115.mat"
    TSV_FILE: str = "participants.tsv"
    ATLAS_FILE: str = "atlas_115_labels.txt"
    BOLD_FILE: str = "MPTP_BOLD_115.mat"

    # ── Subject split ──
    N_SUBJECTS: int = 100   # subject pool size (HCP: smallest ids first)
    N_TRAIN: int = 4
    N_VAL: int = 2
    N_TEST: int = 2
    SEED: int = 42

    # ── Simulation ──
    N_SIM: int = 10_000
    GPU_BATCH: int = 10_000

    # ── Simulation time (ms) ──
    # 12 min total (matches real fMRI scan); 60s transient cut → 660s analysis window
    T_END_MS: float = 720_000.0
    T_CUT_MS: float = 60_000.0

    # ── Time discretization ──
    DT: float = 0.5
    DECIMATE: int = 20
    TR_SEC: float = 1.0          # BOLD sampling period (s); cuBNM downsamples to this

    # ── HRF (TVB MixtureOfGammas) ──
    # peak ~ HRF_A1/HRF_L seconds.  Mouse: ~3s  Human: ~6s
    HRF_A1: float = 3.0          # positive gamma shape
    HRF_A2: float = 7.0          # undershoot gamma shape
    HRF_L: float = 1.0           # rate parameter
    HRF_C: float = 0.3           # undershoot ratio
    HRF_LENGTH_SEC: float = 32.0   # kernel length (sec)
    HRF_LENGTH_MS: float = 20_000.0  # TVB Bold hrf_length (ms)

    # ── Prior bounds (global scalars P, Q, g_e, g_i, c_ei) ──
    PRIOR_P_LOW:   float = 0.0
    PRIOR_P_HIGH:  float = 3.0
    PRIOR_Q_LOW:   float = 0.0
    PRIOR_Q_HIGH:  float = 3.0
    PRIOR_GE_LOW:  float = 0.0
    PRIOR_GE_HIGH: float = 1.0
    PRIOR_GI_LOW:  float = 0.0
    PRIOR_GI_HIGH: float = 1.0
    PRIOR_CEI_LOW:  float = 6.0
    PRIOR_CEI_HIGH: float = 18.0

    # ── Stage 1 prior (global scalars P, Q, g_e, g_i, c_ei) ──
    STAGE1_PARAMS: List[str] = field(
        default_factory=lambda: ["P", "Q", "g_e", "g_i", "c_ei"]
    )
    STAGE1_PRIOR_LOW: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 6.0]
    )
    STAGE1_PRIOR_HIGH: List[float] = field(
        default_factory=lambda: [3.0, 3.0, 1.0, 1.0, 18.0]
    )

    # ── Features ──
    USE_FCD: bool = False   # FCD disabled by default (use FC only)

    # ── Embedding (RegionTransformer, Phase 3) ──
    EMBED_DIM: int = 128
    REGION_TRANSFORMER_HEADS: int = 4
    REGION_TRANSFORMER_LAYERS: int = 2
    REGION_TRANSFORMER_D_MODEL: int = 128

    # ── Phase-2 attention × gradient selection ──
    SELECTION_K: int = 300

    # ── SBI ──
    SBI_DEVICE: str = "cuda"
    N_POSTERIOR: int = 2000
    N_SBC: int = 200
    N_TEST_RESIM: int = 50
    NDE_HIDDEN: int = 128
    NDE_TRANSFORMS: int = 8

    @property
    def ANALYSIS_BOLD_T(self) -> int:
        """Number of BOLD TRs after transient cut (= analysis window / TR)."""
        return int((self.T_END_MS - self.T_CUT_MS) / (self.TR_SEC * 1000.0))


# ---------------------------------------------------------------------------
# Module reload
# ---------------------------------------------------------------------------

_PIPELINE_MODULES = (
    "config", "data_loader", "bold", "simulator", "inference", "evaluate",
)


def reload_pipeline_modules():
    """Drop cached pipeline modules so re-import picks up edits."""
    for mod in _PIPELINE_MODULES:
        if mod in sys.modules:
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# Apply configuration
# ---------------------------------------------------------------------------

def _apply_to_config(cfg: PipelineConfig):
    """Push PipelineConfig fields into the global `config` module."""
    import config

    # Species / atlas
    config.SPECIES = cfg.SPECIES
    config.N_REGIONS = cfg.N_REGIONS
    config.FC_DIM = cfg.N_REGIONS * (cfg.N_REGIONS - 1) // 2
    config.VELOCITY_M_PER_S = cfg.VELOCITY_M_PER_S

    # Paths
    config.DATA_DIR = cfg.DATA_DIR
    config.OUTPUT_DIR = cfg.OUTPUT_DIR
    config.FC_PATH = f"{cfg.DATA_DIR}/{cfg.FC_FILE}"
    config.SC_PATH = f"{cfg.DATA_DIR}/{cfg.SC_FILE}"
    config.TSV_PATH = f"{cfg.DATA_DIR}/{cfg.TSV_FILE}"
    config.ATLAS_PATH = f"{cfg.DATA_DIR}/{cfg.ATLAS_FILE}"
    config.BOLD_PATH = f"{cfg.DATA_DIR}/{cfg.BOLD_FILE}"

    # Split
    config.N_SUBJECTS = cfg.N_SUBJECTS
    config.N_TRAIN = cfg.N_TRAIN
    config.N_VAL = cfg.N_VAL
    config.N_TEST = cfg.N_TEST
    config.SEED = cfg.SEED

    # Simulation
    config.N_SIM = cfg.N_SIM
    config.GPU_BATCH = cfg.GPU_BATCH
    config.T_END = cfg.T_END_MS
    config.T_CUT = cfg.T_CUT_MS
    config.ANALYSIS_BOLD_T = cfg.ANALYSIS_BOLD_T
    config.DT = cfg.DT
    config.DECIMATE = cfg.DECIMATE
    config.TR_SEC = cfg.TR_SEC
    config.FS_BOLD = 1.0 / cfg.TR_SEC
    config.FS_NEURAL = 1000.0 / (cfg.DT * cfg.DECIMATE)
    if hasattr(config, "BW"):
        config.BW["TR"] = cfg.TR_SEC
    config.WC_FIXED["t_end"] = cfg.T_END_MS
    config.WC_FIXED["t_cut"] = cfg.T_CUT_MS
    config.WC_FIXED["dt"] = cfg.DT
    config.WC_FIXED["decimate"] = cfg.DECIMATE

    # HRF
    config.HRF_A1 = cfg.HRF_A1
    config.HRF_A2 = cfg.HRF_A2
    config.HRF_L = cfg.HRF_L
    config.HRF_C = cfg.HRF_C
    config.HRF_LENGTH_SEC = cfg.HRF_LENGTH_SEC
    config.HRF_LENGTH_MS = cfg.HRF_LENGTH_MS

    # Prior bounds
    config.PRIOR_P_LOW   = cfg.PRIOR_P_LOW
    config.PRIOR_P_HIGH  = cfg.PRIOR_P_HIGH
    config.PRIOR_Q_LOW   = cfg.PRIOR_Q_LOW
    config.PRIOR_Q_HIGH  = cfg.PRIOR_Q_HIGH
    config.PRIOR_GE_LOW  = cfg.PRIOR_GE_LOW
    config.PRIOR_GE_HIGH = cfg.PRIOR_GE_HIGH
    config.PRIOR_GI_LOW  = cfg.PRIOR_GI_LOW
    config.PRIOR_GI_HIGH = cfg.PRIOR_GI_HIGH
    config.PRIOR_CEI_LOW  = cfg.PRIOR_CEI_LOW
    config.PRIOR_CEI_HIGH = cfg.PRIOR_CEI_HIGH

    # STAGE1_PARAMS / PRIOR_LOW / PRIOR_HIGH — global scalars.
    config.STAGE1_PARAMS = ["P", "Q", "g_e", "g_i", "c_ei"]
    config.STAGE1_PRIOR_LOW = [
        cfg.PRIOR_P_LOW, cfg.PRIOR_Q_LOW,
        cfg.PRIOR_GE_LOW, cfg.PRIOR_GI_LOW,
        cfg.PRIOR_CEI_LOW,
    ]
    config.STAGE1_PRIOR_HIGH = [
        cfg.PRIOR_P_HIGH, cfg.PRIOR_Q_HIGH,
        cfg.PRIOR_GE_HIGH, cfg.PRIOR_GI_HIGH,
        cfg.PRIOR_CEI_HIGH,
    ]
    config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS

    # Embedding / features
    config.USE_FCD = cfg.USE_FCD
    config.EMBED_DIM = cfg.EMBED_DIM
    config.REGION_TRANSFORMER_HEADS   = cfg.REGION_TRANSFORMER_HEADS
    config.REGION_TRANSFORMER_LAYERS  = cfg.REGION_TRANSFORMER_LAYERS
    config.REGION_TRANSFORMER_D_MODEL = cfg.REGION_TRANSFORMER_D_MODEL

    # Phase-2 selection
    config.SELECTION_K = cfg.SELECTION_K

    # SBI
    config.SBI_DEVICE = cfg.SBI_DEVICE
    config.N_POSTERIOR = cfg.N_POSTERIOR
    config.N_SBC = cfg.N_SBC
    config.N_TEST_RESIM = cfg.N_TEST_RESIM
    config.NDE_HIDDEN = cfg.NDE_HIDDEN
    config.NDE_TRANSFORMS = cfg.NDE_TRANSFORMS


# ---------------------------------------------------------------------------
# Print auto-flush patch
# ---------------------------------------------------------------------------

def _patch_print_flush():
    """Make builtins.print always flush, so Jupyter shows output live."""
    import builtins
    import functools
    if getattr(builtins, "_print_patched_for_flush", False):
        return
    _orig_print = builtins.print

    # functools.wraps copies __name__/__module__/__qualname__ from the real
    # print, so libraries that introspect the global print by name (e.g.
    # numba's `@infer_global(print)` -> getattr(print.__module__,
    # print.__name__)) still resolve to builtins.print instead of looking up
    # a non-existent `pipeline_setup._print_with_flush`.
    @functools.wraps(_orig_print)
    def _print_with_flush(*args, **kwargs):
        kwargs.setdefault("flush", True)
        return _orig_print(*args, **kwargs)

    builtins.print = _print_with_flush
    builtins._print_patched_for_flush = True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_pipeline(cfg: PipelineConfig = None, *,
                   seed: bool = True, print_summary: bool = True,
                   force_flush: bool = True, **overrides):
    """One-shot pipeline initialization.

    Parameters
    ----------
    cfg : PipelineConfig, optional
        Full config object. If None, a default one is created and any
        keyword overrides are applied on top.
    seed : bool
        Set numpy + torch random seeds.
    print_summary : bool
        Call `config.print_config()` after applying.
    force_flush : bool
        Patch ``builtins.print`` so every print is forced to flush.
        Ensures Jupyter shows output live for long-running cells.
    **overrides
        Field-name keyword overrides, e.g. ``N_SIM=5000``.

    Returns
    -------
    cfg : PipelineConfig
        The active configuration.
    """
    warnings.filterwarnings("ignore")

    if cfg is None:
        cfg = PipelineConfig()

    # Apply keyword overrides on top of cfg
    if overrides:
        valid_names = {f.name for f in fields(PipelineConfig)}
        for key, value in overrides.items():
            if key not in valid_names:
                raise ValueError(
                    f"Unknown PipelineConfig field: {key!r}"
                )
            setattr(cfg, key, value)

    reload_pipeline_modules()

    import config as _config_mod  # noqa: F401  triggers reload
    _apply_to_config(cfg)

    import config
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if force_flush:
        _patch_print_flush()

    if seed:
        import numpy as np
        np.random.seed(config.SEED)
        try:
            import torch
            torch.manual_seed(config.SEED)
        except ImportError:
            pass

    if print_summary:
        config.print_config()

    return cfg
