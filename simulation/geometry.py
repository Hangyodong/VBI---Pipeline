"""Geometry-based structural augmentation for the coupling matrix.

Motivation (measured on HCP cortical-360):
  - Empirical FC is dominated by HOMOTOPIC connectivity: the (i, i+N/2) left<->right
    same-region edges average 0.399 vs the 0.064 whole-matrix mean (6.2x stronger).
  - Tractography SC under-recovers these callosal connections, so an SC-driven
    simulator cannot produce the homotopic FC structure -> caps the achievable
    FC-corr ceiling (~0.2, vs the ~0.84 individual-vs-group FC similarity).
  - Tract-length distance decay was checked and is NOT informative here
    (corr(length, FC) ~ 0.016; exp(-L/lambda) vs FC ~ 0), so the distance kernel
    is intentionally omitted; only homotopic augmentation is provided.

This adds a fixed STRUCTURAL prior to the coupling matrix (NOT an inferred
parameter; theta_dim and the model equations are unchanged). The simulator then
propagates same-region inter-hemispheric coupling, letting it generate homotopic
FC. Gated on ``config.GEOMETRY_COUPLING`` (default OFF -> identity).

Region-order assumption: the cortical parcellation is ordered first-half = left
hemisphere, second-half = right (verified: FC[i, i+180] ~ 0.40). If a future atlas
interleaves hemispheres, pass an explicit homotopic pairing instead.
"""
import numpy as np


def homotopic_kernel(n_regions):
    """(N, N) symmetric binary kernel with 1 on (i, i+N/2) homotopic pairs.

    Assumes first-half = left, second-half = right (N even). For odd N or a
    midline region count mismatch, only the matched first-half pairs are set.
    """
    n = int(n_regions)
    K = np.zeros((n, n), dtype=np.float64)
    h = n // 2
    for i in range(h):
        K[i, i + h] = 1.0
        K[i + h, i] = 1.0
    return K


def augment_sc_geometry(sc, config):
    """Add a gated geometry prior to a (already-scaled) SC coupling matrix.

    SC_eff = SC + alpha * K   (optionally re-maxnorm'd to keep the coupling
    magnitude/regime bounded). When ``config.GEOMETRY_COUPLING`` is False this is
    an exact no-op (returns ``sc`` unchanged) so the baseline is preserved.

    Knobs (read from config; all have safe defaults):
      GEOMETRY_COUPLING : bool  -> enable (default False)
      GEOM_ALPHA        : float -> homotopic weight (default 0.0)
      GEOM_KERNEL       : str   -> "homotopic" (only supported kernel)
      GEOM_RENORM       : bool  -> re-divide by max after adding (default True)
    """
    if not bool(getattr(config, "GEOMETRY_COUPLING", False)):
        return sc
    alpha = float(getattr(config, "GEOM_ALPHA", 0.0))
    if alpha == 0.0:
        return sc
    kind = str(getattr(config, "GEOM_KERNEL", "homotopic")).lower()
    sc = np.asarray(sc, dtype=np.float64).copy()
    n = sc.shape[0]
    if kind == "homotopic":
        K = homotopic_kernel(n)
    else:
        raise ValueError(f"unknown GEOM_KERNEL {kind!r} (only 'homotopic')")
    np.fill_diagonal(K, 0.0)
    sc_eff = sc + alpha * K
    np.fill_diagonal(sc_eff, 0.0)
    if bool(getattr(config, "GEOM_RENORM", True)):
        mx = float(sc_eff.max())
        if mx > 0:
            sc_eff = sc_eff / mx
    return sc_eff


def geometry_meta(config):
    """Cache-identity fragment so an SC-geometry change invalidates the sim cache."""
    if not bool(getattr(config, "GEOMETRY_COUPLING", False)):
        return {"geometry": None}
    return {
        "geometry": str(getattr(config, "GEOM_KERNEL", "homotopic")),
        "geom_alpha": float(getattr(config, "GEOM_ALPHA", 0.0)),
        "geom_renorm": bool(getattr(config, "GEOM_RENORM", True)),
    }
