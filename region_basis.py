"""Region basis for latent region-wise parameter decoding (no myelin/gradient).

Builds a spatial basis over regions from:
  - a constant/base term (always),
  - atlas network labels -> one-hot (optional, if labels provided),
  - SC graph Laplacian eigenvectors (connectome harmonics, from the subject SC).

Used by ``param_decoder.decode_latent_to_param_maps`` to turn a low-dim latent
coefficient vector ``z`` into full region-wise parameter maps. Myelin / functional
gradient maps are intentionally NOT used.
"""
import numpy as np


def _laplacian_basis(sc, K):
    """Top-K non-trivial eigenvectors of the symmetric normalized Laplacian.

    L = I - D^{-1/2} W D^{-1/2}. The smoothest eigenvector (smallest eigenvalue,
    ~constant) is dropped (the constant/base term already covers it). Columns are
    z-scored to unit std and sign-fixed deterministically (largest-|component| > 0).
    """
    W = np.asarray(sc, dtype=np.float64).copy()
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    if K <= 0:
        return np.zeros((n, 0), dtype=np.float64)
    deg = W.sum(1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    L = np.eye(n) - (dinv[:, None] * W * dinv[None, :])
    L = (L + L.T) / 2.0
    evals, evecs = np.linalg.eigh(L)           # ascending
    # drop the first (smoothest/constant-like); take next K
    cols = evecs[:, 1:1 + K]
    if cols.shape[1] < K:                       # pad if SC too small/degenerate
        cols = np.pad(cols, ((0, 0), (0, K - cols.shape[1])))
    # z-score columns to stable scale
    std = cols.std(0)
    std[std == 0] = 1.0
    cols = (cols - cols.mean(0)) / std
    # deterministic sign: make the largest-|component| positive
    for k in range(cols.shape[1]):
        j = np.argmax(np.abs(cols[:, k]))
        if cols[j, k] < 0:
            cols[:, k] = -cols[:, k]
    return np.ascontiguousarray(cols)


def build_region_basis(sc, labels=None, config=None):
    """Build the region basis dict.

    Parameters
    ----------
    sc : (n_regions, n_regions) structural connectivity (subject SC).
    labels : optional sequence of length n_regions of network labels (aligned by
             ROW ORDER, not numeric id). If None, only constant + Laplacian basis.
    config : object with attributes N_LAPLACIAN_BASIS (int), USE_NETWORK_BASIS (bool).

    Returns
    -------
    dict with:
        network_ids      : (n_regions,) int network index per region (or None)
        network_names    : list of unique network names (or None)
        network_onehot   : (n_regions, n_networks) float (or (n_regions, 0))
        laplacian_basis  : (n_regions, K) float
        basis_info       : dict (n_regions, n_networks, K, has_networks)
    """
    sc = np.asarray(sc, dtype=np.float64)
    n = sc.shape[0]
    K = int(getattr(config, "N_LAPLACIAN_BASIS", 4)) if config is not None else 4
    use_net = bool(getattr(config, "USE_NETWORK_BASIS", True)) if config is not None else True

    network_ids = None
    network_names = None
    if use_net and labels is not None:
        labels = list(labels)
        if len(labels) != n:
            raise ValueError(f"labels length {len(labels)} != n_regions {n}")
        network_names = sorted(set(labels))
        name_to_idx = {nm: i for i, nm in enumerate(network_names)}
        network_ids = np.array([name_to_idx[l] for l in labels], dtype=int)
        onehot = np.zeros((n, len(network_names)), dtype=np.float64)
        onehot[np.arange(n), network_ids] = 1.0
    else:
        onehot = np.zeros((n, 0), dtype=np.float64)

    lap = _laplacian_basis(sc, K)

    assert np.all(np.isfinite(onehot)) and np.all(np.isfinite(lap)), "non-finite basis"
    return {
        "network_ids": network_ids,
        "network_names": network_names,
        "network_onehot": onehot,
        "laplacian_basis": lap,
        "basis_info": {
            "n_regions": n,
            "n_networks": onehot.shape[1],
            "K": lap.shape[1],
            "has_networks": onehot.shape[1] > 0,
        },
    }
