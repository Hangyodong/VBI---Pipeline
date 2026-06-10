"""Simulation engine and helpers.

Submodules
----------
- simulation.delays     : compute_delay_matrix, detect_delay_key, apply_delay
- simulation.wc_runner  : simulate_gpu_batch, simulate_single, _import_wc,
                          _run_streaming_hrf, detect_engine_key
- simulation.warmup     : WarmupResult, warmup_run, simulate_with_warmup
- simulation.qc         : assert_theta_feature_distinct,
                          run_theta_specific_check

All names exposed here are also importable from their submodules. The
``simulator.py`` compatibility wrapper re-exports the same names so
existing callers continue to work unchanged.
"""
from simulation.delays import (
    apply_delay,
    compute_delay_matrix,
    detect_delay_key,
)
from simulation.wc_runner import (
    _apply_engine,
    _import_wc,
    _run_streaming_hrf,
    _try_per_sim_params,
    detect_engine_key,
    normalize_ts,
    simulate_gpu_batch,
    simulate_single,
    to_numpy,
)
from simulation.warmup import (
    WarmupResult,
    simulate_with_warmup,
    warmup_run,
)
from simulation.qc import (
    assert_theta_feature_distinct,
    run_theta_specific_check,
    theta_feature_diff_norm,
)

__all__ = [
    # delays
    "compute_delay_matrix",
    "detect_delay_key",
    "apply_delay",
    # wc_runner (public)
    "simulate_gpu_batch",
    "simulate_single",
    "detect_engine_key",
    "to_numpy",
    "normalize_ts",
    # warmup
    "WarmupResult",
    "warmup_run",
    "simulate_with_warmup",
    # qc
    "assert_theta_feature_distinct",
    "theta_feature_diff_norm",
    "run_theta_specific_check",
]
