#!/usr/bin/env python
"""CPU-only smoke tests for the basis_regionwise (myelin/gradient) mode.

No GPU, no cuBNM core, no long simulation. Verifies:
  * basis.npy shape + cortical slicing/rezscore behaviour
  * theta_dim == n_params * basis_dim == 12
  * theta=zeros decodes to each parameter's bound MIDPOINT map
  * decoded maps stay inside parameter bounds (and sigma >= 0)
  * runner_rwweib_2cpl.build_param_lists accepts per-region <param>_matrix
    overrides (pure-numpy path, importable without a GPU)
  * the wrapped batch simulator (engine_select.latent_wrap) actually DECODES
    theta into per-region <param>_matrix overrides in basis mode — the same
    path baseline_eval / posterior_predictive_check / plot_one_simulation now use

Run:
    PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
        python -m pytest test_basis_mode_smoke.py -q
    # or:
    python test_basis_mode_smoke.py
"""
import os

import numpy as np

import config

PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
BOUNDS = {"g_LRE": (0.0, 3.0), "g_FFI": (0.0, 3.0),
          "I_o": (0.0, 1.0), "sigma": (0.0, 0.05)}
MIDPOINTS = {p: 0.5 * (lo + hi) for p, (lo, hi) in BOUNDS.items()}  # 1.5/1.5/0.5/0.025
N_REGIONS = 360
BASIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basis.npy")


def _setup_basis_config():
    """Configure the (mouse-default) config module for basis_regionwise.

    Mirrors main_HCP.py's basis_regionwise setup block, minus the GPU bits.
    """
    config.PARAMETER_MODE = "basis_regionwise"
    config.INFERENCE_MODEL = "rwweib2"
    config.N_REGIONS = N_REGIONS
    config.HETERO_PARAMS = list(PARAMS)
    config.HETERO_BOUNDS = dict(BOUNDS)
    config.BASIS_PATH = BASIS_PATH
    config.BASIS_REZSCORE = True


# ---------------------------------------------------------------------------
# 1. basis shape + slicing
# ---------------------------------------------------------------------------

def test_basis_shape_and_slice():
    b = np.load(BASIS_PATH)
    assert b.shape == (381, 3), f"on-disk basis must be (381,3), got {b.shape}"

    from basis_decoder import BasisParamDecoder
    dec = BasisParamDecoder.from_file(
        BASIS_PATH, PARAMS, bounds=BOUNDS, n_regions=N_REGIONS, rezscore=True)
    assert dec.basis.shape == (N_REGIONS, 3), "cortical slice -> (360,3)"
    assert np.allclose(dec.basis[:, 0], 1.0), "col0 stays constant after rezscore"
    assert abs(dec.basis[:, 1].std() - 1.0) < 1e-6, "myelin col rezscored to std 1"
    assert abs(dec.basis[:, 2].std() - 1.0) < 1e-6, "gradient col rezscored to std 1"


# ---------------------------------------------------------------------------
# 2. theta_dim == 12
# ---------------------------------------------------------------------------

def test_theta_dim_is_12():
    _setup_basis_config()
    from basis_decoder import get_decoder
    dec = get_decoder(config)
    assert dec.n_params == 4 and dec.basis_dim == 3
    assert dec.theta_dim == 12, f"theta_dim must be 4*3=12, got {dec.theta_dim}"
    names = dec.coeff_names()
    assert len(names) == 12
    assert names[:3] == ["g_LRE_const", "g_LRE_myelin", "g_LRE_gradient"]


# ---------------------------------------------------------------------------
# 3. theta=zeros -> midpoint maps
# ---------------------------------------------------------------------------

def test_zeros_decode_to_midpoint_maps():
    _setup_basis_config()
    from basis_decoder import get_decoder
    dec = get_decoder(config)
    maps = dec.decode(np.zeros(dec.theta_dim))      # single theta -> {p:(R,)}
    for p in PARAMS:
        m = np.asarray(maps[p])
        assert m.shape == (N_REGIONS,)
        assert (m.max() - m.min()) < 1e-9, f"{p} map should be uniform at z=0"
        assert abs(float(m[0]) - MIDPOINTS[p]) < 1e-9, (
            f"{p} z=0 -> {m[0]} != midpoint {MIDPOINTS[p]}")


# ---------------------------------------------------------------------------
# 4. decoded maps within bounds
# ---------------------------------------------------------------------------

def test_decoded_maps_within_bounds():
    _setup_basis_config()
    from basis_decoder import get_decoder
    dec = get_decoder(config)
    rng = np.random.RandomState(0)
    theta = rng.uniform(-2.0, 2.0, (5, dec.theta_dim))
    maps = dec.decode(theta)
    for p in PARAMS:
        m = np.asarray(maps[p])
        lo, hi = BOUNDS[p]
        assert m.shape == (5, N_REGIONS)
        assert np.all(np.isfinite(m))
        assert m.min() >= lo - 1e-9 and m.max() <= hi + 1e-9, f"{p} out of bounds"
    assert np.asarray(maps["sigma"]).min() >= 0.0


# ---------------------------------------------------------------------------
# 5. runner accepts per-region matrix overrides (CPU, no cubnm core)
# ---------------------------------------------------------------------------

def test_runner_accepts_matrix_overrides():
    _setup_basis_config()
    from cuBNM.runner_rwweib_2cpl import build_param_lists  # pure-numpy import

    n_sims, n_nodes = 2, N_REGIONS
    theta = np.zeros((n_sims, 0), dtype=np.float64)   # no scalar params
    overrides = {f"{p}_matrix": np.full((n_sims, n_nodes), MIDPOINTS[p])
                 for p in PARAMS}
    pl = build_param_lists(theta, [], n_nodes, fixed=overrides)
    for p in PARAMS:
        assert pl[p].shape == (n_sims, n_nodes)
        assert np.allclose(pl[p], MIDPOINTS[p]), f"{p}_matrix override not applied"

    # wrong-shape matrix must be rejected loudly
    bad = {"g_LRE_matrix": np.zeros((n_sims, n_nodes + 1))}
    try:
        build_param_lists(theta, [], n_nodes, fixed=bad)
        raise AssertionError("expected ValueError for mismatched matrix shape")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 6. wrapped batch sim decodes theta -> overrides (path used by all eval fixes)
# ---------------------------------------------------------------------------

def test_latent_wrap_decodes_to_overrides():
    _setup_basis_config()
    import engine_select
    from basis_decoder import get_decoder

    assert engine_select.is_regionwise() is True

    captured = {}

    def stub_base_sim(weights, theta, param_names=None, fixed_overrides=None,
                      delays=None, apply_bw=True, label=None, n_total=None, **kw):
        captured["fixed_overrides"] = fixed_overrides
        captured["param_names"] = param_names
        return [None] * np.asarray(theta).shape[0]

    wrapped = engine_select.latent_wrap(stub_base_sim)
    weights = np.zeros((N_REGIONS, N_REGIONS), dtype=np.float64)
    rng = np.random.RandomState(1)
    theta = rng.uniform(-2.0, 2.0, (3, 12)).astype(np.float64)

    wrapped(weights, theta, param_names=list(PARAMS))

    ov = captured["fixed_overrides"]
    # theta is NOT forwarded as scalar params; only decoded maps are injected.
    assert captured["param_names"] == []
    assert set(ov) == {f"{p}_matrix" for p in PARAMS}

    dec = get_decoder(config)
    expected = dec.decode(theta)               # {p:(3,R)}
    for p in PARAMS:
        assert ov[f"{p}_matrix"].shape == (3, N_REGIONS)
        assert np.allclose(ov[f"{p}_matrix"], expected[p]), (
            f"{p} override != decoder output")


# ---------------------------------------------------------------------------
# 7. is_regionwise toggles with PARAMETER_MODE
# ---------------------------------------------------------------------------

def test_is_regionwise_toggle():
    import engine_select
    config.PARAMETER_MODE = "basis_regionwise"
    assert engine_select.is_regionwise() is True
    config.PARAMETER_MODE = "homogeneous"
    assert engine_select.is_regionwise() is False
    _setup_basis_config()   # restore for any later test


def main():
    tests = [
        test_basis_shape_and_slice,
        test_theta_dim_is_12,
        test_zeros_decode_to_midpoint_maps,
        test_decoded_maps_within_bounds,
        test_runner_accepts_matrix_overrides,
        test_latent_wrap_decodes_to_overrides,
        test_is_regionwise_toggle,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print("test_basis_mode_smoke: ALL PASS")


if __name__ == "__main__":
    main()
