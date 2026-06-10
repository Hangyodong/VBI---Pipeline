"""FC computation for cuBNM BOLD output.

Mirrors the scientific convention of ``features/fc.py`` WITHOUT importing it
(kept import-free at module top to avoid cross-package coupling):

    FC = Pearson r  (np.corrcoef)
    NaN -> 0        (np.nan_to_num)
    diagonal -> 0   (np.fill_diagonal)

Raw r values in [-1, 1]; no z-scoring.
"""
import numpy as np


def compute_fc(ts):
    """Pearson FC of a (T, N) time series → (N, N), NaN→0, diag→0.

    Identical convention to ``features/fc.py:compute_fc``.
    """
    ts = np.asarray(ts)
    fc = np.corrcoef(ts.T)
    fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 0.0)
    return fc


def fc_abs_diff(fc_a, fc_b):
    """Mean absolute difference between two (N, N) FC matrices (upper-tri).

    Returns ``float`` or ``np.nan`` if shapes mismatch.
    """
    fc_a = np.asarray(fc_a)
    fc_b = np.asarray(fc_b)
    if fc_a.shape != fc_b.shape or fc_a.ndim != 2:
        return float("nan")
    n = fc_a.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(np.abs(fc_a[iu] - fc_b[iu])))
