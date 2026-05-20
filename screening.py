"""Informative-feature screening (placeholder for future expansion).

Purpose
-------
Currently the pipeline uses every FC upper-triangle entry. With small
training-subject pools (≤ 4) and 6555-dim FC vectors, much of the input
is noise. This module is the home for *pre-PCA* feature filters that
select informative dimensions before the FeaturePipeline fits PCA.

Planned screens (none enabled by default)
-----------------------------------------
- variance_screen(fc_train_raw, frac_keep)
      drop FC entries with near-zero variance across training simulations.
- theta_correlation_screen(fc_train_raw, theta_train, top_k)
      keep entries whose Pearson r with at least one inferred parameter
      exceeds a threshold.
- group_consistent_screen(fc_train_raw, subject_ids, min_subjects)
      keep entries that show consistent direction across training subjects.

API contract for future screens
-------------------------------
Every screen returns a boolean mask of shape (FC_DIM,) where True means
"keep this dimension". Masks are composable via logical AND.

Activation
----------
Screens are off by default; they will only be wired into
``FeaturePipeline.fit`` once empirical validation shows they improve
posterior shrinkage on validation subjects.

This file currently exports only ``identity_screen`` so that callers can
import a consistent API even before any real screen is implemented.
"""
import numpy as np

import config


def identity_screen(fc_train_raw):
    """Trivial screen: keep every FC dimension.

    Parameters
    ----------
    fc_train_raw : np.ndarray of shape (n_sim, FC_DIM)

    Returns
    -------
    mask : np.ndarray of shape (FC_DIM,) bool
        All True.
    """
    n_feat = fc_train_raw.shape[1]
    expected = config.N_REGIONS * (config.N_REGIONS - 1) // 2
    if n_feat != expected:
        raise ValueError(
            f"FC width {n_feat} != N*(N-1)/2 = {expected}. "
            "Refusing to screen on a non-FC input."
        )
    return np.ones(n_feat, dtype=bool)


def apply_mask(fc_raw, mask):
    """Apply a boolean dimension mask to a (1D or 2D) FC array.

    Parameters
    ----------
    fc_raw : np.ndarray of shape (FC_DIM,) or (n_sim, FC_DIM)
    mask   : np.ndarray of shape (FC_DIM,) bool

    Returns
    -------
    fc_masked : np.ndarray
        Same shape minus the masked-out dimensions.
    """
    if fc_raw.shape[-1] != mask.shape[0]:
        raise ValueError(
            f"FC last-axis dim {fc_raw.shape[-1]} != mask dim {mask.shape[0]}"
        )
    return fc_raw[..., mask]
