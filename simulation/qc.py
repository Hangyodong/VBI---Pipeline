"""Simulation quality control.

Public API
----------
- assert_theta_feature_distinct(theta_a, theta_b, fc_a, fc_b, atol)
      AssertionError if different thetas produce nearly identical FCs.
      Used by debug_notebook F section to catch batch-mean regressions.

- theta_feature_diff_norm(fc_a, fc_b)
      L2 norm of the feature difference. Convenience.

- run_theta_specific_check(weights, delays, param_names, theta_a, theta_b,
                            apply_bw=True, atol=1e-3, verbose=True)
      End-to-end check: simulate two contrastive thetas, compute FCs,
      assert distinct. Returns a small dict with the measured norm.

Why this lives in simulation/
-----------------------------
QC is *about* the simulator's behaviour, not feature math. It would
also fit under tests/ but is small and used from debug_notebook, so we
keep it accessible as a runtime helper.
"""
import numpy as np

from features.fc import compute_fc, fc_to_upper_tri
from simulation.wc_runner import simulate_gpu_batch


def theta_feature_diff_norm(fc_a, fc_b):
    """L2 norm of the upper-triangle FC difference."""
    v_a = fc_to_upper_tri(fc_a) if fc_a.ndim == 2 else fc_a
    v_b = fc_to_upper_tri(fc_b) if fc_b.ndim == 2 else fc_b
    return float(np.linalg.norm(v_a - v_b))


def assert_theta_feature_distinct(theta_a, theta_b, fc_a, fc_b, atol=1e-3):
    """Raise AssertionError if different thetas produced near-identical FCs.

    Parameters
    ----------
    theta_a, theta_b : array-like  raw theta vectors (only used for the
                                   error message; values are not checked)
    fc_a, fc_b       : (N, N) FC matrix or (FC_DIM,) upper-tri vector
    atol             : float       minimum required ||Δfeature||₂

    Raises
    ------
    AssertionError if ``||fc_a - fc_b||₂ <= atol``. The message includes
    the diff norm and suggests checking ``simulate_gpu_batch`` for a
    batch-mean regression.
    """
    diff = theta_feature_diff_norm(fc_a, fc_b)
    if diff <= atol:
        raise AssertionError(
            f"theta-feature pairing failed: ||Δfeature||₂ = {diff:.6f} "
            f"<= atol={atol}.\n"
            f"  theta_a = {np.asarray(theta_a)}\n"
            f"  theta_b = {np.asarray(theta_b)}\n"
            "Different theta produced near-identical features. "
            "Check simulate_gpu_batch for batch-mean regression."
        )
    return diff


def run_theta_specific_check(
    weights, delays, param_names,
    theta_a, theta_b,
    apply_bw=True, atol=1e-3, verbose=True,
):
    """End-to-end theta-specific simulation check.

    Returns
    -------
    dict
        {
          "fc_a"     : (N, N),
          "fc_b"     : (N, N),
          "diff"     : float,    L2 norm of upper-tri difference
          "v_a_std"  : float,
          "v_b_std"  : float,
        }
    """
    theta_batch = np.stack([
        np.asarray(theta_a, dtype=np.float32),
        np.asarray(theta_b, dtype=np.float32),
    ], axis=0)
    if verbose:
        print(f"    theta_raw_batch:\n{theta_batch}")
        print("    starting GPU simulation (2 contrastive thetas) ...")

    bolds = simulate_gpu_batch(
        weights, theta_batch, param_names,
        delays=delays, apply_bw=apply_bw,
    )
    if len(bolds) != 2:
        raise AssertionError(f"expected 2 BOLDs, got {len(bolds)}")

    fc_a = compute_fc(bolds[0])
    fc_b = compute_fc(bolds[1])
    v_a = fc_to_upper_tri(fc_a)
    v_b = fc_to_upper_tri(fc_b)
    diff = float(np.linalg.norm(v_a - v_b))

    if verbose:
        print(f"    BOLD shapes : {[b.shape for b in bolds]}")
        print(
            f"    FC0 range   : [{v_a.min():.3f}, {v_a.max():.3f}]  "
            f"std={v_a.std():.4f}"
        )
        print(
            f"    FC1 range   : [{v_b.min():.3f}, {v_b.max():.3f}]  "
            f"std={v_b.std():.4f}"
        )
        print(f"    ||FC0 - FC1||₂ = {diff:.4f}   (need > {atol})")

    # The actual assert (raises AssertionError with a helpful message)
    assert_theta_feature_distinct(theta_a, theta_b, fc_a, fc_b, atol=atol)

    if verbose:
        print("    ✓ theta-feature pairing OK")

    return {
        "fc_a": fc_a, "fc_b": fc_b,
        "diff": diff,
        "v_a_std": float(v_a.std()), "v_b_std": float(v_b.std()),
    }
