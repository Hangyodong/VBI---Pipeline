"""Compatibility wrapper — public API moved to ``simulation/`` and ``features/``.

This file no longer contains any logic. It exists solely so that
existing imports like::

    from simulator import simulate_gpu_batch, compute_fc
    from simulator import worker_extract, extract_observed_features
    import simulator
    simulator.compute_fc(...)

continue to work after the Phase 1+2 refactor.

New code should import from the source packages:

    from simulation.wc_runner import simulate_gpu_batch
    from simulation.warmup    import warmup_run, simulate_with_warmup
    from simulation.delays    import compute_delay_matrix
    from features.fc          import compute_fc, fc_to_upper_tri
    from features.fcd         import compute_sim_fcd_matrix
    from features.extraction  import (
        extract_features, extract_observed_features,
        extract_simulated_features, worker_extract,
    )
"""

# ── simulation core ────────────────────────────────────────────────────────
from simulation.delays import (              # noqa: F401
    apply_delay as _apply_delay,
    compute_delay_matrix,
    detect_delay_key,
)
from simulation.wc_runner import (           # noqa: F401
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
from simulation.warmup import (              # noqa: F401
    WarmupResult,
    simulate_with_warmup,
    warmup_run,
)

# ── features ───────────────────────────────────────────────────────────────
from features.fc import (                    # noqa: F401
    compute_fc,
    fc_to_upper_tri,
)
from features.fcd import (                   # noqa: F401
    compute_sim_fcd_matrix,
    fcd_to_summary_stats,
    fcd_to_upper_tri,
)
from features.extraction import (            # noqa: F401
    extract_features,
    extract_observed_features,
    extract_simulated_features,
    worker_extract,
)
