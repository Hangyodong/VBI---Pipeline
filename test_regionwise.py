#!/usr/bin/env python
"""Unit checks for latent region-wise decoding (CPU, no GPU). Run: python test_regionwise.py"""
import numpy as np
from region_basis import build_region_basis
import param_decoder as pd


class _Cfg:
    N_LAPLACIAN_BASIS = 4
    USE_NETWORK_BASIS = True
    HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
    HETERO_BOUNDS = {"g_LRE": (0, 9), "g_FFI": (0, 9),
                     "I_o": (0.15, 0.60), "sigma": (0, 0.09)}


def main():
    cfg = _Cfg()
    R = 40
    rng = np.random.RandomState(0)
    sc = rng.rand(R, R); sc = (sc + sc.T) / 2; np.fill_diagonal(sc, 0)

    # 1. basis (no labels)
    b = build_region_basis(sc, labels=None, config=cfg)
    assert b["laplacian_basis"].shape == (R, 4)
    assert pd.latent_dim(b, cfg) == 4 * (1 + 0 + 4) == 20

    # 2. decode shape + bounds
    z = rng.uniform(-1, 1, size=(7, 20))
    maps = pd.decode_latent_to_param_maps(z, b, cfg)
    for p, m in maps.items():
        lo, hi = cfg.HETERO_BOUNDS[p]
        assert m.shape == (7, R)
        assert m.min() >= lo - 1e-9 and m.max() <= hi + 1e-9
    assert maps["sigma"].min() >= 0

    # 3. homogeneous special case (net/lap=0 -> uniform per region)
    m0 = pd.decode_latent_to_param_maps(np.zeros((1, 20)), b, cfg)
    for p, m in m0.items():
        assert (m.max() - m.min()) < 1e-9

    # 4. fixed overrides
    ov = pd.make_fixed_overrides_from_param_maps(maps)
    assert set(ov) == {"g_LRE_matrix", "g_FFI_matrix", "I_o_matrix", "sigma_matrix"}
    for v in ov.values():
        assert v.shape == (7, R)

    # 5. labels -> onehot
    labels = ["DMN" if i < 20 else "VIS" for i in range(R)]
    b2 = build_region_basis(sc, labels=labels, config=cfg)
    assert b2["network_onehot"].shape == (R, 2)
    assert pd.latent_dim(b2, cfg) == 4 * (1 + 2 + 4) == 28

    print("test_regionwise: ALL PASS")


if __name__ == "__main__":
    main()
