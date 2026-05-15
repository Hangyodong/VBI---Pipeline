"""Feature computation, extraction, and (future) screening.

Submodules
----------
- features.fc          : compute_fc, fc_to_upper_tri
- features.fcd         : compute_sim_fcd_matrix, fcd_to_upper_tri,
                         fcd_to_summary_stats
- features.extraction  : extract_features, extract_observed_features,
                         extract_simulated_features, worker_extract
- features.screening   : (future) informative-dimension screens

All names exposed at package level are also importable from their
submodules. The simulator.py compatibility wrapper re-exports the same
names so existing callers continue to work.
"""
from features.fc import compute_fc, fc_to_upper_tri
from features.fcd import (
    compute_sim_fcd_matrix,
    fcd_to_summary_stats,
    fcd_to_upper_tri,
)
from features.extraction import (
    extract_features,
    extract_observed_features,
    extract_simulated_features,
    worker_extract,
)

__all__ = [
    # fc
    "compute_fc",
    "fc_to_upper_tri",
    # fcd
    "compute_sim_fcd_matrix",
    "fcd_to_upper_tri",
    "fcd_to_summary_stats",
    # extraction
    "extract_features",
    "extract_observed_features",
    "extract_simulated_features",
    "worker_extract",
]
