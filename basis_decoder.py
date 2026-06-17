"""HCP myelin/gradient basis decoder for region-wise parameter maps.

VBI/SBI infers only BASIS COEFFICIENTS (theta), not per-ROI parameters. The
decoder turns those coefficients into bounded region-wise parameter maps using a
fixed spatial basis (e.g. basis.npy = [constant, myelin, gradient], shape
(n_regions, basis_dim)). This is the identifiable Kong-2021-style parameterization
(theta_dim = n_params x basis_dim, e.g. 4 x 3 = 12) — NOT the unidentifiable
direct N_REGIONS x n_params inference.

Decode (per parameter p):
    beta_p = theta.reshape(n_params, basis_dim)[p]          # (basis_dim,)
    z      = basis @ beta_p                                 # (n_regions,)
    map_p  = mid + half * tanh(z),  mid=(lo+hi)/2, half=(hi-lo)/2

tanh bounds z -> [-1,1] so map_p is exactly within [lo, hi] (no clip needed).
basis is used ONLY here (pre-simulation); it is never fed into the FC features.
"""
import os
import numpy as np

# default coefficient prior bounds (Uniform) and parameter physical bounds
DEFAULT_COEFF_PRIOR = (-2.0, 2.0)
DEFAULT_BOUNDS = {
    "g_LRE": (0.0, 3.0),
    "g_FFI": (0.0, 3.0),
    "I_o":   (0.0, 1.0),
    "sigma": (0.0, 0.05),
}


class BasisParamDecoder:
    """theta (basis coefficients) -> {param: region-wise map}."""

    def __init__(self, basis, params, bounds=None):
        self.basis = np.ascontiguousarray(np.asarray(basis, dtype=np.float64))  # (R, B)
        self.n_regions, self.basis_dim = self.basis.shape
        self.params = list(params)
        self.n_params = len(self.params)
        self.theta_dim = self.n_params * self.basis_dim
        self.bounds = dict(DEFAULT_BOUNDS)
        if bounds:
            self.bounds.update(bounds)
        for p in self.params:
            if p not in self.bounds:
                raise ValueError(f"no bounds for param {p!r}")

    # ----- construction -----
    @classmethod
    def from_file(cls, path, params, bounds=None, n_regions=None, rezscore=True):
        """Load basis.npy. If n_regions < basis rows, slice the first n_regions
        (cortex). rezscore re-standardizes non-constant columns on the slice."""
        b = np.load(path).astype(np.float64)
        if b.ndim != 2:
            raise ValueError(f"basis {path} must be 2D, got {b.shape}")
        if n_regions is not None and n_regions < b.shape[0]:
            b = b[:n_regions].copy()
        if rezscore:
            for j in range(b.shape[1]):
                col = b[:, j]
                if np.allclose(col, col[0]):           # constant column -> keep
                    continue
                s = col.std()
                b[:, j] = (col - col.mean()) / (s if s > 0 else 1.0)
        return cls(b, params, bounds)

    # ----- decode -----
    def decode(self, theta):
        """theta (theta_dim,) -> {p:(R,)}, or (S, theta_dim) -> {p:(S, R)}."""
        t = np.asarray(theta, dtype=np.float64)
        single = (t.ndim == 1)
        if single:
            t = t[None, :]
        S = t.shape[0]
        if t.shape[1] != self.theta_dim:
            raise ValueError(f"theta dim {t.shape[1]} != {self.theta_dim} "
                             f"({self.n_params} params x {self.basis_dim} basis)")
        beta = t.reshape(S, self.n_params, self.basis_dim)   # (S, P, B)
        out = {}
        for k, p in enumerate(self.params):
            lo, hi = self.bounds[p]
            mid = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
            z = beta[:, k, :] @ self.basis.T                 # (S, R)
            m = mid + half * np.tanh(z)
            out[p] = m[0] if single else np.ascontiguousarray(m)
            assert np.all(np.isfinite(out[p]))
            assert out[p].min() >= lo - 1e-9 and out[p].max() <= hi + 1e-9
        if "sigma" in out:
            assert np.asarray(out["sigma"]).min() >= 0.0
        return out

    # ----- SBI helpers -----
    def coeff_names(self):
        cols = ["const"] + [f"b{j}" for j in range(1, self.basis_dim)]
        if self.basis_dim == 3:
            cols = ["const", "myelin", "gradient"]
        return [f"{p}_{c}" for p in self.params for c in cols]

    def prior_bounds(self, lo=DEFAULT_COEFF_PRIOR[0], hi=DEFAULT_COEFF_PRIOR[1]):
        """Uniform coefficient prior bounds (lists of length theta_dim)."""
        return [lo] * self.theta_dim, [hi] * self.theta_dim

    def make_fixed_overrides(self, param_maps):
        """{param:(...,R)} -> {'<param>_matrix': (...,R)} for cuBNM injection."""
        return {f"{p}_matrix": np.ascontiguousarray(np.asarray(m, dtype=np.float64))
                for p, m in param_maps.items()}


_DECODER_CACHE = {}


def get_decoder(config):
    """Cached BasisParamDecoder built from config (BASIS_PATH, N_REGIONS,
    HETERO_PARAMS, HETERO_BOUNDS)."""
    path = getattr(config, "BASIS_PATH", "basis.npy")
    R = int(config.N_REGIONS)
    params = tuple(config.HETERO_PARAMS)
    bounds = getattr(config, "HETERO_BOUNDS", None)
    key = (path, R, params)
    if key not in _DECODER_CACHE:
        _DECODER_CACHE[key] = BasisParamDecoder.from_file(
            path, list(params), bounds=bounds, n_regions=R,
            rezscore=bool(getattr(config, "BASIS_REZSCORE", True)))
    return _DECODER_CACHE[key]


def qc_decode(decoder, thetas, save_dir=None, tag="qc", verbose=True):
    """Decode theta sample(s), report per-param map stats, save param_maps/*.npy."""
    t = np.asarray(thetas, dtype=np.float64)
    if t.ndim == 1:
        t = t[None, :]
    maps = decoder.decode(t)                                # {p:(S,R)}
    if verbose:
        print(f"  [QC] decoded {t.shape[0]} theta(s) x basis_dim={decoder.basis_dim} "
              f"-> {decoder.n_regions} regions; bounds={decoder.bounds}")
        for p in decoder.params:
            m = np.asarray(maps[p]); lo, hi = decoder.bounds[p]
            print(f"    {p:7s}: min={m.min():.4f} max={m.max():.4f} "
                  f"mean={m.mean():.4f} std={m.std():.4f}  bound=[{lo},{hi}]")
    if save_dir:
        d = os.path.join(save_dir, "param_maps")
        os.makedirs(d, exist_ok=True)
        for p in decoder.params:
            np.save(os.path.join(d, f"{tag}_{p}.npy"), np.asarray(maps[p]))
        if verbose:
            print(f"    saved -> {d}/{tag}_<param>.npy")
    return maps
