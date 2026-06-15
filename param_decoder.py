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


# ---------------------------------------------------------------------------
# Direct region-wise (theta_control = all N_REGIONS x len(HETERO) params,
# no basis). theta layout: [p0_r0..p0_rN, p1_r0..p1_rN, ...] for p in HETERO.
# ---------------------------------------------------------------------------

def direct_param_names(config, n_regions):
    """Flat names g_LRE_r000.. for the direct (1440-dim) control vector."""
    names = []
    for p in _hetero_params(config):
        names += [f"{p}_r{i:03d}" for i in range(n_regions)]
    return names


def direct_dim(config, n_regions):
    return len(_hetero_params(config)) * int(n_regions)


def direct_bounds(config, n_regions):
    """(low, high) lists of length direct_dim; per-region bounds per param."""
    b = _bounds(config)
    low, high = [], []
    for p in _hetero_params(config):
        lo, hi = b[p]
        low += [lo] * int(n_regions)
        high += [hi] * int(n_regions)
    return low, high


def decode_direct_to_param_maps(theta_phys, config, n_regions):
    """Reshape a PHYSICAL theta (n_sims, len(HETERO)*n_regions) -> per-param maps.

    No basis: the parameter scaler already mapped [-1,1] -> physical bounds, so
    this is a pure reshape (with a safety clip to the bounds).
    """
    z = np.asarray(theta_phys, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    R = int(n_regions)
    params = _hetero_params(config)
    if z.shape[1] != len(params) * R:
        raise ValueError(f"theta dim {z.shape[1]} != {len(params)*R} "
                         f"({len(params)} params x {R} regions)")
    b = _bounds(config)
    out = {}
    for k, p in enumerate(params):
        lo, hi = b[p]
        m = np.clip(z[:, k * R:(k + 1) * R], lo, hi)
        out[p] = np.ascontiguousarray(m, dtype=np.float64)
        assert m.shape == (z.shape[0], R) and np.all(np.isfinite(m))
    if "sigma" in out:
        assert out["sigma"].min() >= 0.0
    return out


def decode_to_param_maps(theta, basis, config, n_regions=None):
    """Dispatch decode by config.PARAMETER_MODE.

    direct_regionwise : theta is PHYSICAL (n_sims, len(HETERO)*n_regions) -> reshape.
    latent_regionwise : theta is z in [-1,1] (n_sims, latent_dim) -> basis decode.
    """
    mode = str(getattr(config, "PARAMETER_MODE", "homogeneous"))
    if mode == "direct_regionwise":
        R = n_regions if n_regions is not None else basis["laplacian_basis"].shape[0]
        return decode_direct_to_param_maps(theta, config, R)
    return decode_latent_to_param_maps(theta, basis, config)


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


def group_indices_by_hetero(names, hetero):
    """Map each HETERO param -> list of indices in `names` whose name is that
    param or starts with '<param>_' (region-wise / coefficient names). Used to
    aggregate 1440-line per-coefficient reports into per-parameter summaries."""
    groups = {p: [] for p in hetero}
    for i, nm in enumerate(names):
        for p in hetero:
            if nm == p or str(nm).startswith(p + "_"):
                groups[p].append(i)
                break
    return groups


def make_fixed_overrides_from_param_maps(param_maps):
    """{param: (n_sims, n_regions)} -> {'<param>_matrix': map} for cuBNM injection."""
    return {f"{p}_matrix": np.ascontiguousarray(m, dtype=np.float64)
            for p, m in param_maps.items()}
