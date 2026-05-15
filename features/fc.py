"""FC (Functional Connectivity) computation and vectorization.

Public API
----------
- compute_fc(ts)                  : Pearson FC of a (T, N) time series
- fc_to_upper_tri(fc, nan_mask)   : (N, N) FC matrix -> (N*(N-1)/2,) vector

Both functions return raw Pearson r values in [-1, 1]. No z-scoring is
applied here; per-feature scaling is the job of the inference-stage
FamilyScaler / FeaturePipeline.
"""
import numpy as np

import config


def compute_fc(ts):
    """Pearson FC of a (T, N) time series.

    Parameters
    ----------
    ts : np.ndarray of shape (T, N)
        Time series (rows = samples, cols = regions).

    Returns
    -------
    fc : np.ndarray of shape (N, N)
        Symmetric Pearson FC with zero diagonal. NaNs replaced by 0.
    """
    fc = np.corrcoef(ts.T)
    fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 0.0)
    return fc


def fc_to_upper_tri(fc, nan_mask=None):
    """Convert a (N, N) FC matrix to its upper-triangle vector.

    Parameters
    ----------
    fc : np.ndarray of shape (N, N)
    nan_mask : np.ndarray of shape (N, N), optional
        Boolean mask. If supplied, masked entries are dropped from the
        returned vector. Defaults to ``config.NAN_MASK`` (usually None,
        so the full upper triangle is returned).

    Returns
    -------
    vec : np.ndarray of shape (N*(N-1)/2,) float32
        Raw Pearson r values in [-1, 1].
    """
    n = fc.shape[0]
    iu = np.triu_indices(n, k=1)
    if nan_mask is None:
        nan_mask = getattr(config, "NAN_MASK", None)
    if nan_mask is not None and nan_mask.shape == fc.shape:
        valid = ~nan_mask[iu]
        vec = fc[iu[0][valid], iu[1][valid]]
    else:
        vec = fc[iu]
    return vec.astype(np.float32)
