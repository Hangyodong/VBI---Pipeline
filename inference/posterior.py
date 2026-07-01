"""Posterior sampling, shrinkage, correlation, and predictive check.

Public API
----------
- transform_observed(fc_obs_raw, fcd_obs_raw, feature_pipeline)
- infer_subject_raw(posterior, x_obs_input, param_scaler, n_samples)
       -> (samples_raw, means_raw, stds_raw, samples_scaled)
- compute_shrinkage_scaled(samples_scaled)
- compute_shrinkage_raw(samples_raw, prior_low, prior_high)
- posterior_correlation(samples)
- posterior_predictive_check(...)

Sampling convention
-------------------
SBI returns samples in **scaled** space [-1, 1]; we then map them back
to raw space via ``param_scaler.inverse_transform`` for interpretation
and re-simulation. The two-space split is critical: never mix scaled
posterior samples with raw simulation inputs.
"""
import numpy as np

import config

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Observed feature transform
# ---------------------------------------------------------------------------

def transform_observed(fc_obs_raw, fcd_obs_raw, feature_pipeline):
    """Apply the fitted FeaturePipeline to observed features."""
    return feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)


def build_x_obs(feature_pipeline, fc_obs_raw, fcd_obs_raw,
                sid=None, fc_matrix=None):
    """Build the conditioning vector ``x_obs`` for one observation.

    Two modes, gated on ``config.SC_CONDITION``:

    * SC_CONDITION OFF (default): identical to the old path —
      ``feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)`` (PCA/whiten).
    * SC_CONDITION ON: the MultiChannelMatrixEmbedding compresses the raw FC,
      so x is ``[row_index | fc_upper_tri]`` with fc_upper_tri of length
      ``config.FC_DIM`` and ``row_index = config.SC_IDMAP[int(sid)]``.

    ``fc_obs_raw`` here is already the FC **upper-tri vector** of length
    ``config.FC_DIM`` (see ``features.extraction.extract_observed_features``,
    which returns ``fc_to_upper_tri(fc)``). It is used directly as the FC
    channel; ``fc_matrix`` (the (N, N) subject FC) is accepted for callers
    that only have the matrix and is converted to the upper-tri vector here.

    Returns a ``float32`` array of shape ``(1, 1 + FC_DIM)`` in SC mode,
    otherwise whatever ``feature_pipeline.transform`` returns (unchanged).
    """
    if not bool(getattr(config, "SC_CONDITION", False)):
        return feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)

    # SC_CONDITION ON: x = [row_index | fc_upper_tri].
    if sid is None:
        raise RuntimeError("build_x_obs needs sid when SC_CONDITION=True")
    idmap = getattr(config, "SC_IDMAP", None)
    if idmap is None:
        raise RuntimeError(
            "build_x_obs needs config.SC_IDMAP when SC_CONDITION=True "
            "(set in main_HCP Step 1.5)")

    if fc_obs_raw is not None:
        fc_vec = np.asarray(fc_obs_raw, dtype=np.float32).reshape(-1)
    elif fc_matrix is not None:
        from features.fc import fc_to_upper_tri
        fc_vec = np.asarray(
            fc_to_upper_tri(fc_matrix), dtype=np.float32).reshape(-1)
    else:
        raise RuntimeError(
            "build_x_obs needs fc_obs_raw or fc_matrix when SC_CONDITION=True")
    if fc_vec.shape[0] != int(config.FC_DIM):
        raise RuntimeError(
            f"build_x_obs: fc upper-tri length {fc_vec.shape[0]} != "
            f"config.FC_DIM {int(config.FC_DIM)}")

    row = float(idmap[int(sid)])
    idx_col = np.array([[row]], dtype=np.float32)            # (1, 1)
    return np.concatenate(
        [idx_col, fc_vec[None, :].astype(np.float32)], axis=1)  # (1, 1+FC_DIM)


# ---------------------------------------------------------------------------
# Posterior sampling at one observation
# ---------------------------------------------------------------------------

def infer_subject_raw(posterior, x_obs_input, param_scaler,
                      n_samples=None, verbose=False):
    """Sample the amortized posterior at one observation.

    Returns
    -------
    samples_raw    : (n_samples, n_params)  raw parameter values
    means_raw      : (n_params,)            posterior means in raw space
    stds_raw       : (n_params,)            posterior stds in raw space
    samples_scaled : (n_samples, n_params)  scaled [-1, 1] samples
    """
    n_samples = n_samples or config.N_POSTERIOR
    _dev = getattr(config, "SBI_DEVICE", "cpu")
    x_t = torch.tensor(x_obs_input, dtype=torch.float32).to(_dev)
    # no_grad: high-dim region-wise flows (e.g. 1440-dim) OOM if the autoregressive
    # inverse tracks gradients during sampling.
    if _dev == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        try:
            # Prefer rejection (prior-bounded samples → correct metrics), with a
            # time cap so heavy leakage can't hang the run.
            samples_scaled = (
                posterior.sample(
                    (n_samples,), x=x_t, show_progress_bars=False,
                    max_sampling_time=getattr(
                        config, "POSTERIOR_MAX_SAMPLING_TIME", 60.0),
                    return_partial_on_timeout=True,
                )
                .cpu().numpy().astype(np.float32)
            )
        except RuntimeError:
            # Near-zero acceptance (empirical FC OOD) -> direct flow sampling,
            # clipped to the prior box [-1, 1].
            samples_scaled = (
                posterior.sample(
                    (n_samples,), x=x_t, show_progress_bars=False,
                    reject_outside_prior=False,
                )
                .detach().cpu().numpy().astype(np.float32)
            )
            samples_scaled = np.clip(samples_scaled, -1.0, 1.0)

    samples_raw = param_scaler.inverse_transform(samples_scaled)
    means_raw = samples_raw.mean(axis=0)
    stds_raw = samples_raw.std(axis=0)

    if verbose and len(param_scaler.param_names) <= 30:
        for i, name in enumerate(param_scaler.param_names):
            print(
                f"    {name:8s} = {means_raw[i]:.4f} ± {stds_raw[i]:.4f}"
            )
    return samples_raw, means_raw, stds_raw, samples_scaled


# ---------------------------------------------------------------------------
# Shrinkage and correlation
# ---------------------------------------------------------------------------

def compute_shrinkage_scaled(samples_scaled):
    """Posterior shrinkage in the scaled space (prior_std = 2/sqrt(12))."""
    prior_std = 2.0 / np.sqrt(12.0)
    post_std = samples_scaled.std(axis=0)
    return np.clip(1.0 - post_std / prior_std, 0.0, 1.0)


def compute_shrinkage_raw(samples_raw, prior_low, prior_high):
    """Posterior shrinkage in the raw parameter space."""
    prior_low = np.asarray(prior_low)
    prior_high = np.asarray(prior_high)
    prior_std = (prior_high - prior_low) / np.sqrt(12.0)
    post_std = samples_raw.std(axis=0)
    return np.clip(1.0 - post_std / prior_std, 0.0, 1.0)


def posterior_correlation(samples):
    """Posterior correlation matrix; identity if dim < 2."""
    if samples.shape[1] < 2:
        return np.eye(samples.shape[1])
    return np.corrcoef(samples.T)


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------

def posterior_predictive_check(sid, subject_data, posterior,
                               fc_obs_raw, fcd_obs_raw,
                               param_scaler, feature_pipeline,
                               param_names, fixed_overrides=None,
                               n_predictive=None, apply_bw=True,
                               verbose=True):
    """Posterior predictive simulation + comparison to observed FC/FCD."""
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
    )
    from engine_select import get_simulate_single  # honor INFERENCE_MODEL
    simulate_single = get_simulate_single()

    n_predictive = n_predictive or config.N_PPC
    d = subject_data[sid]
    sc = d["sc"]
    dly = d["delays"]

    # SC_CONDITION OFF -> identical to before (FeaturePipeline.transform).
    # SC_CONDITION ON  -> x = [row_index | fc_upper_tri] (encoder compresses FC).
    x_obs = build_x_obs(
        feature_pipeline, fc_obs_raw, fcd_obs_raw,
        sid=sid, fc_matrix=d["fc"],
    )
    samples_raw, means_raw, stds_raw, samples_scaled = infer_subject_raw(
        posterior, x_obs, param_scaler,
        n_samples=n_predictive, verbose=False,
    )

    fc_obs_full = d["fc"]
    iu = np.triu_indices(fc_obs_full.shape[0], k=1)
    # Original-NaN positions were replaced with 0 at load time, so they are
    # finite — isfinite() alone cannot exclude them. Use the recorded NaN
    # mask to drop those ~46% entries from corr/RMSE.
    fc_nan_vec = (
        d["fc_nan"][iu] if "fc_nan" in d else np.zeros(iu[0].shape, dtype=bool)
    )

    fc_corrs, fc_rmses, fcd_rmses = [], [], []

    from engine_select import is_regionwise
    if is_regionwise():
        # M1 fix: region-wise samples are basis/latent coefficients (e.g.
        # ``g_LRE_const``), NOT scalar params. simulate_single would silently
        # ignore them. Reuse the decoded (latent_wrap-wrapped) batch path that
        # evaluation._resimulate_and_score already implements correctly.
        from evaluation.metrics import _resimulate_and_score
        n_ppc = min(n_predictive, len(samples_raw))
        fc_corrs, fc_rmses, fcd_rmses, _ = _resimulate_and_score(
            n_ppc, samples_raw, param_names, fixed_overrides,
            sc, dly, fc_obs_full, fcd_obs_raw, apply_bw,
            sid=sid, verbose=verbose, fc_nan=d.get("fc_nan"),
        )
    else:
        for i in range(min(n_predictive, len(samples_raw))):
            params = dict(fixed_overrides or {})
            for j, name in enumerate(param_names):
                params[name] = float(samples_raw[i, j])
            try:
                bolds = simulate_single(
                    sc, params, n_repeat=1, delays=dly, apply_bw=apply_bw,
                )
                bold = bolds[0]
                fc_pred = compute_fc(bold)

                obs_vec = fc_obs_full[iu]
                pred_vec = fc_pred[iu]
                mask = np.isfinite(obs_vec) & np.isfinite(pred_vec) & ~fc_nan_vec
                if (mask.sum() > 10
                        and obs_vec[mask].std() > 0
                        and pred_vec[mask].std() > 0):
                    r = float(
                        np.corrcoef(obs_vec[mask], pred_vec[mask])[0, 1]
                    )
                    rmse = float(np.sqrt(
                        ((obs_vec[mask] - pred_vec[mask]) ** 2).mean()
                    ))
                    fc_corrs.append(r)
                    fc_rmses.append(rmse)

                fcd_mat = compute_sim_fcd_matrix(bold)
                fcd_pred_vec = fcd_to_upper_tri(fcd_mat)
                fcd_rmses.append(float(np.sqrt(
                    ((fcd_obs_raw - fcd_pred_vec) ** 2).mean()
                )))
            except Exception:
                continue

    out = {
        "samples_raw": samples_raw,
        "samples_scaled": samples_scaled,
        "means_raw": means_raw,
        "stds_raw": stds_raw,
        "fc_corr_mean": float(np.mean(fc_corrs)) if fc_corrs else 0.0,
        "fc_corr_std": float(np.std(fc_corrs)) if fc_corrs else 0.0,
        "fc_rmse_mean": float(np.mean(fc_rmses)) if fc_rmses else 1.0,
        "fcd_rmse_mean": float(np.mean(fcd_rmses)) if fcd_rmses else 1.0,
    }
    if verbose:
        print(
            f"    FC  corr = {out['fc_corr_mean']:.4f} ± "
            f"{out['fc_corr_std']:.4f}, "
            f"RMSE = {out['fc_rmse_mean']:.4f}"
        )
        print(f"    FCD RMSE = {out['fcd_rmse_mean']:.4f}")
    return out
