"""Decode a low-dim latent coefficient vector z into region-wise parameter maps.

Latent layout (per heterogeneous parameter p, in this order):
    [ base_p (1) , network_coeffs_p (n_networks) , laplacian_coeffs_p (K) ]
concatenated over all params in ``config.HETERO_PARAMS``.

Decoding (z assumed in scaled space [-1, 1], i.e. BoxUniform draws):
    base_phys = lo + (hi-lo) * (base + 1)/2                 # base -> [lo, hi]
    var[i]    = tanh( (onehot @ net_coeffs)[i] + (lap @ lap_coeffs)[i] )   # [-1, 1]
    param[i]  = clip( base_phys + (hi-lo)/2 * var[i], lo, hi )             # bounded

Homogeneous baseline is the special case net_coeffs = lap_coeffs = 0
(=> param[i] = base_phys for all i).
"""
import numpy as np

# Physiological / prior bounds for the heterogeneous parameters.
DEFAULT_BOUNDS = {
    "g_LRE": (0.0, 9.0),
    "g_FFI": (0.0, 9.0),
    "I_o":   (0.15, 0.60),
    "sigma": (0.0, 0.09),
}


def _bounds(config):
    b = dict(DEFAULT_BOUNDS)
    if config is not None:
        b.update(getattr(config, "HETERO_BOUNDS", {}) or {})
    return b


def _hetero_params(config):
    if config is not None and getattr(config, "HETERO_PARAMS", None):
        return list(config.HETERO_PARAMS)
    return ["g_LRE", "g_FFI", "I_o", "sigma"]


def per_param_block(basis):
    """Latent slots per parameter: 1 (base) + n_networks + K (laplacian)."""
    n_net = basis["network_onehot"].shape[1]
    K = basis["laplacian_basis"].shape[1]
    return 1 + n_net + K, n_net, K


def latent_param_names(basis, config):
    """Names for each latent coefficient (for STAGE1_PARAMS in latent mode)."""
    _, n_net, K = per_param_block(basis)
    names = []
    for p in _hetero_params(config):
        names.append(f"{p}_base")
        names += [f"{p}_net{j}" for j in range(n_net)]
        names += [f"{p}_lap{j}" for j in range(K)]
    return names


def latent_dim(basis, config):
    block, _, _ = per_param_block(basis)
    return block * len(_hetero_params(config))


def decode_latent_to_param_maps(z_raw, basis, config):
    """z_raw (n_sims, latent_dim) -> {param: (n_sims, n_regions)} region-wise maps."""
    z = np.asarray(z_raw, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    n_sims = z.shape[0]
    onehot = basis["network_onehot"]            # (R, n_net)
    lap = basis["laplacian_basis"]              # (R, K)
    R = lap.shape[0]
    block, n_net, K = per_param_block(basis)
    params = _hetero_params(config)
    if z.shape[1] != block * len(params):
        raise ValueError(
            f"z dim {z.shape[1]} != expected {block*len(params)} "
            f"(block {block} x {len(params)} params)")
    bounds = _bounds(config)

    out = {}
    off = 0
    for p in params:
        lo, hi = bounds[p]
        half = (hi - lo) / 2.0
        b = z[:, off]                            # (n_sims,)
        net_c = z[:, off + 1: off + 1 + n_net]   # (n_sims, n_net)
        lap_c = z[:, off + 1 + n_net: off + block]  # (n_sims, K)
        off += block

        base_phys = lo + (hi - lo) * (b + 1.0) / 2.0          # (n_sims,)
        term = net_c @ onehot.T + lap_c @ lap.T               # (n_sims, R)
        var = np.tanh(term)                                   # [-1, 1]
        pmap = base_phys[:, None] + half * var
        pmap = np.clip(pmap, lo, hi)
        out[p] = np.ascontiguousarray(pmap, dtype=np.float64)

    # strict checks
    for p, m in out.items():
        lo, hi = bounds[p]
        assert m.shape == (n_sims, R), f"{p} map shape {m.shape} != ({n_sims},{R})"
        assert np.all(np.isfinite(m)), f"{p} map has non-finite values"
        assert m.min() >= lo - 1e-9 and m.max() <= hi + 1e-9, f"{p} out of [{lo},{hi}]"
    if "sigma" in out:
        assert out["sigma"].min() >= 0.0, "negative sigma"
    return out


def make_fixed_overrides_from_param_maps(param_maps):
    """{param: (n_sims, n_regions)} -> {'<param>_matrix': map} for cuBNM injection."""
    return {f"{p}_matrix": np.ascontiguousarray(m, dtype=np.float64)
            for p, m in param_maps.items()}
