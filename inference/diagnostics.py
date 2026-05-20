"""Posterior diagnostics: SBC and embedding linear probing.

Public API
----------
- simulation_based_calibration(...) -> (n_sbc, n_params) rank array
- evaluate_embedding_probing(embedding_net, theta_scaled, x_input, ...)
       -> dict mapping param_name -> {r2_mean, r2_std}, plus "_pass" key

SBC interpretation
------------------
For a well-calibrated posterior, ranks should be uniformly distributed.
Systematic deviation reveals miscalibration.

Probing interpretation
----------------------
``embedding_net(x)`` should be linearly informative about scaled theta.
R² > 0.7 = OK, > config.EMB_PROBE_R2_THRESHOLD = warn, otherwise fail.
This is the post-inference half of the "embedding quality check"
(the pre-inference half is the PCA diagnostic in step 6).
"""
import time

import numpy as np

import config
from inference._utils import _progress

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Simulation-based calibration
# ---------------------------------------------------------------------------

def simulation_based_calibration(posterior, prior_scaled, param_scaler,
                                 feature_pipeline, param_names,
                                 weights, delays,
                                 fixed_overrides=None,
                                 n_sbc=None, n_posterior=None,
                                 verbose=True):
    """Run SBC: prior sample -> simulate -> rank posterior samples."""
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fc_to_upper_tri,
        fcd_to_upper_tri, simulate_single,
    )

    n_sbc = n_sbc or config.N_SBC
    n_posterior = n_posterior or 1000

    if verbose:
        print(
            f"  SBC: {n_sbc} simulations, "
            f"{n_posterior} posterior samples each"
        )

    ranks = []
    t0 = time.time()
    for k in range(n_sbc):
        theta_scaled = prior_scaled.sample().cpu().numpy()
        theta_raw = param_scaler.inverse_transform(theta_scaled[None, :])[0]

        params = dict(fixed_overrides or {})
        for j, name in enumerate(param_names):
            params[name] = float(theta_raw[j])

        try:
            bolds = simulate_single(
                weights, params, n_repeat=1, delays=delays,
            )
            bold = bolds[0]
            fc_vec = fc_to_upper_tri(compute_fc(bold))
            fcd_vec = fcd_to_upper_tri(compute_sim_fcd_matrix(bold))

            x_obs = feature_pipeline.transform(fc_vec, fcd_vec)
            x_t = torch.tensor(x_obs, dtype=torch.float32)
            samples_scaled = (
                posterior.sample(
                    (n_posterior,), x=x_t, show_progress_bars=False,
                ).cpu().numpy()
            )
            rank = (samples_scaled < theta_scaled).sum(axis=0)
            ranks.append(rank)
        except Exception:
            continue

        if verbose and (k + 1) in {
            max(1, n_sbc // 4),
            max(1, n_sbc // 2),
            max(1, 3 * n_sbc // 4),
            n_sbc,
        }:
            pct = (k + 1) / n_sbc * 100
            _progress(
                f"SBC {k + 1}/{n_sbc} ({pct:.0f}%)  "
                f"({time.time() - t0:.1f}s)"
            )

    ranks = np.array(ranks)
    if verbose:
        print(f"    SBC done ({time.time() - t0:.1f}s)")
    return ranks


# ---------------------------------------------------------------------------
# Embedding probing
# ---------------------------------------------------------------------------

def evaluate_embedding_probing(embedding_net, theta_scaled, x_input,
                               param_names, n_samples=None,
                               verbose=True):
    """5-fold linear regression R² from embedding to scaled parameters."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    if verbose:
        print("\n  [Embedding probing - linear R²]")

    n_samples = n_samples or min(2000, len(theta_scaled))
    rng = np.random.RandomState(config.SEED)
    idx = rng.choice(len(theta_scaled), n_samples, replace=False)

    embedding_net.eval()
    has_params = any(
        p.requires_grad for p in embedding_net.parameters()
    )
    device = (
        next(embedding_net.parameters()).device if has_params else "cpu"
    )

    with torch.no_grad():
        x_t = torch.tensor(
            x_input[idx], dtype=torch.float32, device=device,
        )
        embs = embedding_net(x_t).cpu().numpy()

    theta_sub = theta_scaled[idx]
    probe = {}
    for i, name in enumerate(param_names):
        y = theta_sub[:, i]
        try:
            scores = cross_val_score(
                LinearRegression(), embs, y, cv=5, scoring="r2",
            )
            r2_mean = float(np.mean(scores))
            r2_std = float(np.std(scores))
        except Exception:
            r2_mean, r2_std = 0.0, 0.0
        probe[name] = {"r2_mean": r2_mean, "r2_std": r2_std}
        if r2_mean > 0.7:
            mark = "  OK"
        elif r2_mean > config.EMB_PROBE_R2_THRESHOLD:
            mark = "  WARN"
        else:
            mark = "  FAIL"
        if verbose:
            print(
                f"    {name:6s}: R² = {r2_mean:.4f} ± {r2_std:.4f}{mark}"
            )

    if verbose:
        mean_r2 = float(np.mean(
            [v["r2_mean"] for v in probe.values()]
        ))
        print(f"    Mean R²: {mean_r2:.4f}")

    probe["_pass"] = bool(
        np.mean([v["r2_mean"] for v in probe.values()])
        > config.EMB_PROBE_R2_THRESHOLD
    )
    return probe
