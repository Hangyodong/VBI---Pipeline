"""Observed / simulated feature extraction.

Public API
----------
- extract_features(bold)             : simulated BOLD -> (fc_vec, fcd_stats)
                                       always returns both, FCD zeros if disabled.
- extract_observed_features(subject) : empirical -> (fc_vec, fcd_stats)
- extract_simulated_features(bold)   : simulated BOLD -> same shape as observed
                                       (respects FEATURE_SET)
- worker_extract(bold)               : safe wrapper for ProcessPoolExecutor

Feature-set rules
-----------------
- ``config.FEATURE_SET == "fc_only"`` (default):
    observed/simulated return (fc_upper_tri, np.zeros(0)). FCD is ignored.
- ``config.FEATURE_SET == "fc_fcd"``:
    observed must provide either ``subject_data["fcd"]`` or
    empirical ``subject_data["bold"]``; otherwise ``ValueError``.
    simulated returns (fc_upper_tri, fcd_summary_stats).

Observed and simulated feature dimensions are guaranteed to match, since
both routes go through the same FEATURE_SET branch.
"""
import numpy as np

import config
from features.fc import compute_fc, fc_to_upper_tri
from features.fcd import compute_sim_fcd_matrix, fcd_to_summary_stats


# ---------------------------------------------------------------------------
# Combined simulated extraction (legacy API — always returns 5-zeros FCD)
# ---------------------------------------------------------------------------

def extract_features(bold):
    """Simulated BOLD (T, N) -> (fc_vec, fcd_stats).

    fc_vec   : Pearson r upper triangle in [-1, 1]  (FC_DIM,)
    fcd_stats: summary statistics       (5,)  [mean, std, q25, q50, q75]
               zeros if ``config.USE_FCD`` is False.

    This is the legacy entry point used by ``inference.collect_training_data``
    via ``worker_extract``. It always returns a 5-element FCD array so that
    downstream code can assume a fixed shape.
    """
    fc = compute_fc(bold)
    fc_vec = fc_to_upper_tri(fc)
    if config.USE_FCD:
        fcd_mat = compute_sim_fcd_matrix(bold)
        fcd_vec = fcd_to_summary_stats(fcd_mat)
    else:
        fcd_vec = np.zeros(5, dtype=np.float32)
    return fc_vec, fcd_vec


# ---------------------------------------------------------------------------
# FEATURE_SET-aware extraction (observed / simulated symmetric)
# ---------------------------------------------------------------------------

def extract_observed_features(subject_data):
    """Observed subject data -> (fc_vec, fcd_stats).

    Respects ``config.FEATURE_SET``:

    "fc_only" : returns (fc_upper_tri, np.zeros(0)). fcd is ignored.
    "fc_fcd"  : returns (fc_upper_tri, fcd_summary). Requires either
                a precomputed fcd matrix in ``subject_data["fcd"]`` OR
                an empirical BOLD time series in ``subject_data["bold"]``.

    Raises
    ------
    ValueError
        If FEATURE_SET == "fc_fcd" but neither fcd matrix nor BOLD is
        available. Refuses to silently fall back so that simulated and
        observed feature dimensions cannot disagree.
    """
    feature_set = getattr(config, "FEATURE_SET", "fc_only")
    fc = subject_data["fc"]
    fc_vec = fc_to_upper_tri(fc)

    if feature_set == "fc_only" or not getattr(config, "USE_FCD", False):
        return fc_vec, np.zeros(0, dtype=np.float32)

    # FCD branch (fc_fcd mode)
    if "fcd" in subject_data and subject_data["fcd"] is not None:
        fcd = subject_data["fcd"]
        if fcd.ndim == 2:
            fcd_vec = fcd_to_summary_stats(fcd)
        else:
            fcd_vec = np.array([
                fcd.mean(), fcd.std(),
                float(np.percentile(fcd, 25)),
                float(np.percentile(fcd, 50)),
                float(np.percentile(fcd, 75)),
            ], dtype=np.float32)
        return fc_vec, fcd_vec

    if "bold" in subject_data and subject_data["bold"] is not None:
        # Compute simulated-style FCD summary from empirical BOLD
        bold = subject_data["bold"]
        fcd_mat = compute_sim_fcd_matrix(bold)
        fcd_vec = fcd_to_summary_stats(fcd_mat)
        return fc_vec, fcd_vec

    raise ValueError(
        "FEATURE_SET='fc_fcd' requires either subject_data['fcd'] or "
        "an empirical BOLD time series. Switch FEATURE_SET to 'fc_only' "
        "or provide BOLD data."
    )


def extract_simulated_features(bold):
    """Simulated BOLD -> (fc_vec, fcd_stats).

    Uses the same FEATURE_SET branch as extract_observed_features so the
    simulated and observed feature vectors live in the same space.
    """
    feature_set = getattr(config, "FEATURE_SET", "fc_only")
    fc = compute_fc(bold)
    fc_vec = fc_to_upper_tri(fc)

    if feature_set == "fc_only" or not getattr(config, "USE_FCD", False):
        return fc_vec, np.zeros(0, dtype=np.float32)

    fcd_mat = compute_sim_fcd_matrix(bold)
    fcd_vec = fcd_to_summary_stats(fcd_mat)
    return fc_vec, fcd_vec


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def worker_extract(bold):
    """Wrapper for ProcessPoolExecutor; returns None on failure."""
    try:
        return extract_features(bold)
    except Exception:
        return None
