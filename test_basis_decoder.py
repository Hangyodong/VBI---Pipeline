#!/usr/bin/env python
"""Unit checks for basis_decoder (CPU). Run: python test_basis_decoder.py"""
import os
import numpy as np
from basis_decoder import BasisParamDecoder, qc_decode, DEFAULT_BOUNDS

PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]


def main():
    dec = BasisParamDecoder.from_file("basis.npy", PARAMS, n_regions=360, rezscore=True)
    assert dec.n_regions == 360 and dec.basis_dim == 3 and dec.theta_dim == 12
    assert np.allclose(dec.basis[:, 0], 1.0)                       # const kept
    assert abs(dec.basis[:, 1].std() - 1.0) < 1e-6                 # rezscored

    rng = np.random.RandomState(0)
    m = dec.decode(rng.uniform(-2, 2, 12))                         # single
    for p in PARAMS:
        lo, hi = DEFAULT_BOUNDS[p]
        assert m[p].shape == (360,)
        assert m[p].min() >= lo - 1e-9 and m[p].max() <= hi + 1e-9
    assert m["sigma"].min() >= 0

    mb = dec.decode(rng.uniform(-2, 2, (5, 12)))                   # batch
    for p in PARAMS:
        assert mb[p].shape == (5, 360)

    m0 = dec.decode(np.zeros(12))                                  # z=0 -> uniform mid
    for p in PARAMS:
        lo, hi = DEFAULT_BOUNDS[p]
        assert (m0[p].max() - m0[p].min()) < 1e-9
        assert abs(m0[p][0] - (lo + hi) / 2) < 1e-9

    lo, hi = dec.prior_bounds()
    assert len(lo) == 12 and lo[0] == -2 and hi[0] == 2
    assert dec.coeff_names()[:3] == ["g_LRE_const", "g_LRE_myelin", "g_LRE_gradient"]

    ov = dec.make_fixed_overrides(mb)
    assert set(ov) == {f"{p}_matrix" for p in PARAMS}

    qc_decode(dec, rng.uniform(-2, 2, (3, 12)), save_dir="output_hcp",
              tag="qctest", verbose=False)
    assert os.path.exists("output_hcp/param_maps/qctest_g_LRE.npy")

    print("test_basis_decoder: ALL PASS")


if __name__ == "__main__":
    main()
