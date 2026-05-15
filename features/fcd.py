"""FCD (Functional Connectivity Dynamics) computation and reduction.

Public API
----------
- compute_sim_fcd_matrix(bold, window_tr, stride_tr) : sliding-window
      element-wise std of FCs -> (N, N) FCD matrix
- fcd_to_upper_tri(fcd_matrix, nan_mask)             : (N, N) -> vector
- fcd_to_summary_stats(fcd_matrix, nan_mask)         : (N, N) -> (5,)
      summary [mean, std, q25, q50, q75]

FCD is disabled by default in this project (``config.USE_FCD = False``)
because empirical BOLD time series are not currently available. These
functions are kept for future activation.
"""
import numpy as np

import config


def compute_sim_fcd_matrix(bold, window_tr=None, stride_tr=None):
    """Simulated BOLD -> FCD-like (N, N) matrix.

    Defined as the element-wise standard deviation across sliding-window
    FCs. Captures dynamic variability and is distinct from the static FC.

    Parameters
    ----------
    bold : np.ndarray of shape (T, N)
    window_tr : int, optional
        Window length in TRs. Defaults to ``config.FCD_WINDOW_TR``.
    stride_tr : int, optional
        Window stride in TRs. Defaults to ``config.FCD_STRIDE_TR``.

    Returns
    -------
    fcd_matrix : np.ndarray of shape (N, N) float32
        Symmetric, zero-diagonal. Returns zeros if the BOLD series is
        shorter than ``window_tr + stride_tr``.
    """
    window_tr = window_tr or config.FCD_WINDOW_TR
    stride_tr = stride_tr or config.FCD_STRIDE_TR
    t_len, n_nodes = bold.shape

    if t_len < window_tr + stride_tr:
        return np.zeros((n_nodes, n_nodes), dtype=np.float32)

    starts = np.arange(0, t_len - window_tr + 1, stride_tr)
    fcs = []
    for s in starts:
        seg = bold[s:s + window_tr]
        if seg.std() < 1e-8:
            fcs.append(np.zeros((n_nodes, n_nodes), dtype=np.float32))
            continue
        fc_seg = np.corrcoef(seg.T)
        fcs.append(np.nan_to_num(fc_seg, nan=0.0).astype(np.float32))

    fcs = np.stack(fcs)
    fcd_matrix = fcs.std(axis=0)
    fcd_matrix = (fcd_matrix + fcd_matrix.T) / 2
    np.fill_diagonal(fcd_matrix, 0.0)
    return fcd_matrix.astype(np.float32)


def fcd_to_upper_tri(fcd_matrix, nan_mask=None):
    """Convert FCD matrix (N, N) to its upper-triangle vector."""
    n = fcd_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    if nan_mask is None:
        nan_mask = getattr(config, "NAN_MASK", None)
    if nan_mask is not None and nan_mask.shape == fcd_matrix.shape:
        valid = ~nan_mask[iu]
        vec = fcd_matrix[iu[0][valid], iu[1][valid]]
    else:
        vec = fcd_matrix[iu]
    return vec.astype(np.float32)


def fcd_to_summary_stats(fcd_matrix, nan_mask=None):
    """FCD matrix (N, N) -> summary statistics vector (5,).

    Returns [mean, std, q25, q50, q75] of the upper-triangle values.
    Much lower dimensional than full upper-tri (5 vs 6555) and avoids
    the poor PCA explained-variance caused by raw FCD spread.
    """
    vec = fcd_to_upper_tri(fcd_matrix, nan_mask=nan_mask)
    return np.array([
        vec.mean(),
        vec.std(),
        float(np.percentile(vec, 25)),
        float(np.percentile(vec, 50)),
        float(np.percentile(vec, 75)),
    ], dtype=np.float32)
