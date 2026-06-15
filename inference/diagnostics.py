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
        fcd_to_upper_tri,
    )
    from engine_select import get_simulate_gpu_batch  # honor INFERENCE_MODEL
    simulate_gpu_batch = get_simulate_gpu_batch()

    n_sbc = n_sbc or config.N_SBC
    n_posterior = n_posterior or 1000

    if verbose:
        print(
            f"  SBC: {n_sbc} simulations, "
            f"{n_posterior} posterior samples each"
        )

    t0 = time.time()

    # ── 1) sample ALL theta up front (scaled + raw) ──────────────────
    theta_scaled_all = (
        prior_scaled.sample((n_sbc,)).cpu().numpy().astype(np.float32)
    )
    theta_raw_all = param_scaler.inverse_transform(theta_scaled_all)

    # ── 2) batch-simulate ALL n_sbc draws via cuBNM in one GPU call ──
    #     (replaces the old per-iteration VBI simulate_single loop)
    bolds_all = simulate_gpu_batch(
        weights, theta_raw_all, param_names=list(param_names),
        fixed_overrides=fixed_overrides, delays=delays, apply_bw=True,
        label="SBC", n_total=n_sbc,
    )
    if verbose:
        _progress(
            f"SBC: simulated {len(bolds_all)}/{n_sbc} via cuBNM "
            f"({time.time() - t0:.1f}s); ranking posteriors ..."
        )

    # ── 3) per-draw feature extraction + posterior rank ──────────────
    ranks = []
    _first_err = None
    _dev = getattr(config, "SBI_DEVICE", "cpu")
    for k in range(min(n_sbc, len(bolds_all))):
        try:
            bold = bolds_all[k]
            fc_vec = fc_to_upper_tri(compute_fc(bold))
            fcd_vec = fcd_to_upper_tri(compute_sim_fcd_matrix(bold))

            x_obs = feature_pipeline.transform(fc_vec, fcd_vec)
            # x_t MUST live on the posterior's device (SBI_DEVICE); a CPU
            # tensor against a CUDA posterior raises and would silently drop
            # every SBC draw.
            x_t = torch.tensor(
                np.atleast_2d(x_obs), dtype=torch.float32, device=_dev,
            )
            with torch.no_grad():   # high-dim flows OOM if sampling tracks grad
                samples_scaled = (
                    posterior.sample(
                        (n_posterior,), x=x_t, show_progress_bars=False,
                        # SBC ranks need prior-bounded samples → keep rejection;
                        # cap time so a leaky iteration can't stall the loop.
                        max_sampling_time=getattr(
                            config, "POSTERIOR_MAX_SAMPLING_TIME", 60.0),
                        return_partial_on_timeout=True,
                    ).cpu().numpy()
                )
            rank = (samples_scaled < theta_scaled_all[k]).sum(axis=0)
            ranks.append(rank)
        except Exception as e:           # don't silently swallow ALL draws
            if _first_err is None:
                _first_err = e
            continue

        if verbose and (k + 1) in {
            max(1, n_sbc // 4),
            max(1, n_sbc // 2),
            max(1, 3 * n_sbc // 4),
            n_sbc,
        }:
            pct = (k + 1) / n_sbc * 100
            _progress(
                f"SBC rank {k + 1}/{n_sbc} ({pct:.0f}%)  "
                f"({time.time() - t0:.1f}s)"
            )

    ranks = np.array(ranks)
    if verbose:
        n_fail = n_sbc - len(ranks)
        if n_fail > 0 and _first_err is not None:
            print(
                f"    SBC: {n_fail}/{n_sbc} draws failed during ranking "
                f"(first error: {type(_first_err).__name__}: {_first_err})"
            )
        print(f"    SBC done — {len(ranks)}/{n_sbc} ranks "
              f"({time.time() - t0:.1f}s)")
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
