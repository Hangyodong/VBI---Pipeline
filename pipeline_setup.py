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
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """User-facing parameters.

    Each field maps to a `config` attribute. Defaults reflect a full
    H100 NVL run; reduce N_SIM / T_END_MS for quick tests.
    """

    # ── Paths ──
    DATA_DIR: str = "/scratch/home/wog3597/vbi"
    OUTPUT_DIR: str = "./output_mouse_mptp"

    # ── Subject split ──
    N_TRAIN: int = 4
    N_VAL: int = 2
    N_TEST: int = 2
    SEED: int = 42

    # ── Simulation ──
    N_SIM: int = 10_000
    N_SIM_S2: int = 10_000
    GPU_BATCH: int = 10_000

    # ── Simulation time (ms) ──
    T_END_MS: float = 30_000.0
    T_CUT_MS: float = 5_000.0

    # ── Time discretization ──
    DT: float = 0.5
    DECIMATE: int = 20

    # ── HRF (TVB MixtureOfGammas) ──
    # peak ~ HRF_A1/HRF_L seconds.  Mouse: ~3s  Human: ~6s
    HRF_A1: float = 3.0          # positive gamma shape
    HRF_A2: float = 7.0          # undershoot gamma shape
    HRF_L: float = 1.0           # rate parameter
    HRF_C: float = 0.3           # undershoot ratio
    HRF_LENGTH_SEC: float = 32.0   # kernel length (sec)
    HRF_LENGTH_MS: float = 20_000.0  # TVB Bold hrf_length (ms)

    # ── Stage 1 prior ──
    STAGE1_PARAMS: List[str] = field(
        default_factory=lambda: ["P", "Q", "g_e", "g_i"]
    )
    STAGE1_PRIOR_LOW: List[float] = field(
        default_factory=lambda: [0.5, 0.0, 0.0, 0.0]
    )
    STAGE1_PRIOR_HIGH: List[float] = field(
        default_factory=lambda: [2.5, 2.0, 1.5, 1.5]
    )

    # ── Stage 2 c-parameter prior ──
    C_PARAM_PRIOR: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "c_ee": (12.0, 20.0),
            "c_ei": (8.0, 16.0),
            "c_ie": (10.0, 20.0),
            "c_ii": (1.0, 6.0),
        }
    )

    # ── Features ──
    USE_FCD: bool = False   # FCD disabled by default (use FC only)

    # ── Embedding ──
    PCA_DIM_FC: int = 200
    PCA_DIM_FCD: int = 100
    EMBED_DIM: int = 64
    EMBED_HIDDEN: int = 256

    # ── SBI ──
    N_POSTERIOR: int = 2000
    N_SBC: int = 200
    N_TEST_RESIM: int = 50

    @property
    def ANALYSIS_BOLD_T(self) -> int:
        """Number of BOLD TRs after transient cut (TR = 1 s)."""
        return int((self.T_END_MS - self.T_CUT_MS) / 1000)


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

    # Paths
    config.DATA_DIR = cfg.DATA_DIR
    config.OUTPUT_DIR = cfg.OUTPUT_DIR
    config.FC_PATH = f"{cfg.DATA_DIR}/MPTP_FC_115.mat"
    config.SC_PATH = f"{cfg.DATA_DIR}/MPTP_SC_115.mat"
    config.TSV_PATH = f"{cfg.DATA_DIR}/participants.tsv"
    config.ATLAS_PATH = f"{cfg.DATA_DIR}/atlas_115_labels.txt"
    config.BOLD_PATH = f"{cfg.DATA_DIR}/MPTP_BOLD_115.mat"

    # Split
    config.N_TRAIN = cfg.N_TRAIN
    config.N_VAL = cfg.N_VAL
    config.N_TEST = cfg.N_TEST
    config.SEED = cfg.SEED

    # Simulation
    config.N_SIM = cfg.N_SIM
    config.N_SIM_S2 = cfg.N_SIM_S2
    config.GPU_BATCH = cfg.GPU_BATCH
    config.T_END = cfg.T_END_MS
    config.T_CUT = cfg.T_CUT_MS
    config.ANALYSIS_BOLD_T = cfg.ANALYSIS_BOLD_T
    config.DT = cfg.DT
    config.DECIMATE = cfg.DECIMATE
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

    # Priors
    config.STAGE1_PARAMS = list(cfg.STAGE1_PARAMS)
    config.STAGE1_PRIOR_LOW = list(cfg.STAGE1_PRIOR_LOW)
    config.STAGE1_PRIOR_HIGH = list(cfg.STAGE1_PRIOR_HIGH)
    config.C_PARAM_PRIOR = dict(cfg.C_PARAM_PRIOR)

    # Embedding
    config.PCA_DIM_FC = cfg.PCA_DIM_FC
    config.PCA_DIM_FCD = cfg.PCA_DIM_FCD
    config.USE_FCD = cfg.USE_FCD
    config.EMBED_DIM = cfg.EMBED_DIM
    config.EMBED_HIDDEN = cfg.EMBED_HIDDEN

    # SBI
    config.N_POSTERIOR = cfg.N_POSTERIOR
    config.N_SBC = cfg.N_SBC
    config.N_TEST_RESIM = cfg.N_TEST_RESIM


# ---------------------------------------------------------------------------
# Print auto-flush patch
# ---------------------------------------------------------------------------

def _patch_print_flush():
    """Make builtins.print always flush, so Jupyter shows output live."""
    import builtins
    if getattr(builtins, "_print_patched_for_flush", False):
        return
    _orig_print = builtins.print

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
