"""Per-subject evaluation metrics + re-simulation scoring.

Public API
----------
- fc_metrics(fc_obs, fc_pred, nan_mask)   : corr / rmse / mae
- fcd_vec_rmse(fcd_obs_vec, fcd_pred_vec) : RMSE
- bootstrap_ci(values, n, alpha)          : mean + percentile CI
- evaluate_subject(sid, ...)              : posterior sampling + re-sim
                                            -> per-subject result dict
- baseline_eval(sid, ...)                 : prior-midpoint baseline
- baseline_eval_subjects(subjects, ...)   : aggregated baseline

Internal helpers
----------------
- _resimulate_and_score(...)              : shared by evaluate_subject /
                                            baseline / final_test
- _progress(msg)                          : timestamped print

USE_FCD handling
----------------
When ``config.USE_FCD`` is False, FCD is not computed and the result's
``fcd_rmse_mean`` is 0.0. The model-selection score in
``evaluation.model_selection`` excludes the FCD term in that case.
"""
import sys
import time

import numpy as np

import config


# ---------------------------------------------------------------------------
# Progress printing
# ---------------------------------------------------------------------------

def _progress(msg):
    """Print a timestamped progress message and flush."""
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def fc_metrics(fc_obs, fc_pred, nan_mask=None):
    """FC correlation, RMSE, and MAE on the upper triangle.

    ``nan_mask`` (N, N bool) flags original-NaN FC entries that were replaced
    with 0 at load time. Those are finite, so isfinite() cannot drop them —
    they are excluded explicitly here when the mask is provided.
    """
    n = fc_obs.shape[0]
    iu = np.triu_indices(n, k=1)
    a = fc_obs[iu]
    b = fc_pred[iu]
    mask = np.isfinite(a) & np.isfinite(b)
    if nan_mask is not None:
        mask &= ~np.asarray(nan_mask, dtype=bool)[iu]
    if mask.sum() < 2:
        return {"corr": 0.0, "rmse": 1.0, "mae": 1.0}
    a, b = a[mask], b[mask]
    if a.std() > 0 and b.std() > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    else:
        r = 0.0
    return {
        "corr": r,
        "rmse": float(np.sqrt(((a - b) ** 2).mean())),
        "mae": float(np.abs(a - b).mean()),
    }


def fcd_vec_rmse(fcd_obs_vec, fcd_pred_vec):
    """RMSE between two FCD upper-triangle vectors (NaN-safe)."""
    a = np.asarray(fcd_obs_vec, dtype=np.float64)
    b = np.asarray(fcd_pred_vec, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 1.0
    return float(np.sqrt(((a[mask] - b[mask]) ** 2).mean()))


# Deprecated alias
fcd_summary_rmse = fcd_vec_rmse


def bootstrap_ci(values, n=None, alpha=0.05):
    """Bootstrap mean + (alpha/2, 1-alpha/2) percentile CI."""
    n = n or config.BOOTSTRAP_N
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        mean = float(values.mean()) if len(values) else 0.0
        return mean, 0.0, 0.0
    rng = np.random.RandomState(42)
    boots = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n)
    ]
    return (
        float(np.mean(boots)),
        float(np.percentile(boots, 100 * alpha / 2)),
        float(np.percentile(boots, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Per-subject evaluation
# ---------------------------------------------------------------------------

def evaluate_subject(sid, subject_data, posterior, param_scaler,
                     feature_pipeline, param_names,
                     fixed_overrides=None, n_resim=None,
                     apply_bw=True, verbose=True):
    """Posterior sampling + re-simulation + FC/FCD comparison."""
    from simulator import extract_observed_features
    from inference import infer_subject_raw, compute_shrinkage_scaled
    from inference.posterior import build_x_obs

    n_resim = n_resim or config.N_TEST_RESIM
    d = subject_data[sid]
    sc = d["sc"]
    dly = d["delays"]
    fc_obs_full = d["fc"]

    if verbose:
        _progress(f"evaluating {sid} (posterior sampling)")

    fc_obs_raw, fcd_obs_raw = extract_observed_features(d)
    # SC_CONDITION OFF -> identical to before (FeaturePipeline.transform).
    # SC_CONDITION ON  -> x = [row_index | fc_upper_tri] (encoder compresses FC).
    x_obs_input = build_x_obs(
        feature_pipeline, fc_obs_raw, fcd_obs_raw,
        sid=sid, fc_matrix=d["fc"],
    )

    t_infer = time.time()
    samples_raw, means_raw, stds_raw, samples_scaled = infer_subject_raw(
        posterior, x_obs_input, param_scaler,
        n_samples=n_resim, verbose=False,
    )
    infer_elapsed = time.time() - t_infer
    shrink = compute_shrinkage_scaled(samples_scaled)
    if verbose:
        _progress(
            f"infer_subject_raw  n_samples={n_resim}  "
            f"({infer_elapsed:.2f}s)"
        )

    if verbose:
        print(f"  [{sid}] posterior:")
        if len(param_names) <= 30:
            for i, name in enumerate(param_names):
                tag = " (low shrinkage)" if (
                    shrink[i] < config.DIFFICULT_SHRINKAGE
                ) else ""
                print(
                    f"    {name:8s} = {means_raw[i]:.4f} ± "
                    f"{stds_raw[i]:.4f}  shrinkage={shrink[i]:.3f}{tag}"
                )
        else:
            # region-wise: aggregate shrinkage per HETERO param (mean over regions)
            import numpy as _np
            from param_decoder import group_indices_by_hetero
            hp = list(getattr(config, "HETERO_PARAMS", []))
            groups = group_indices_by_hetero(param_names, hp)
            for p, idx in groups.items():
                if not idx:
                    continue
                sm = _np.asarray(shrink)[idx]
                print(f"    {p:8s} : shrinkage mean={sm.mean():.3f} "
                      f"[{sm.min():.3f},{sm.max():.3f}] over {len(idx)} regions "
                      f"(low={int((sm < config.DIFFICULT_SHRINKAGE).sum())})")

    fc_corrs, fc_rmses, fcd_rmses, fc_preds = _resimulate_and_score(
        n_resim, samples_raw, param_names, fixed_overrides,
        sc, dly, fc_obs_full, fcd_obs_raw, apply_bw,
        sid=sid, verbose=verbose, fc_nan=d.get("fc_nan"),
    )

    # S1: expected-FC. Average the resim FC matrices over posterior draws (and
    # their noise realizations), then score ONCE with the SAME fc_metrics + NaN
    # mask. This denoises the single-run estimator — a stronger estimator, NOT a
    # metric relaxation. Reported ALONGSIDE the per-draw mean (both kept).
    if fc_preds:
        fc_pred_mean = np.mean(np.stack(fc_preds, axis=0), axis=0)
        m_exp = fc_metrics(fc_obs_full, fc_pred_mean, nan_mask=d.get("fc_nan"))
        fc_corr_expected = m_exp["corr"]
        fc_rmse_expected = m_exp["rmse"]
    else:
        fc_corr_expected, fc_rmse_expected = 0.0, 1.0

    # ADDITIVE: posterior-MEAN-theta resim (the subject "digital twin" point
    # estimate). Unlike expected-FC (which averages FC over DISTINCT posterior
    # draws), this fixes ONE theta = mean of the posterior draws and resimulates
    # it n_resim times to noise-average the SAME param set, then scores the mean
    # FC once. Wrapped in try/except so it can NEVER crash evaluate_subject — on
    # any failure the two keys fall back to 0.0 / 1.0. Does NOT replace anything.
    fc_corr_meantheta, fc_rmse_meantheta = 0.0, 1.0
    try:
        theta_mean = np.asarray(samples_raw).mean(axis=0)        # (n_params,)
        _, _, _, fc_preds_mt = _resimulate_and_score(
            n_resim, np.tile(theta_mean[None, :], (n_resim, 1)),
            param_names, fixed_overrides,
            sc, dly, fc_obs_full, fcd_obs_raw, apply_bw,
            sid=sid, verbose=False, fc_nan=d.get("fc_nan"),
        )
        if fc_preds_mt:
            m_mt = fc_metrics(
                fc_obs_full, np.mean(np.stack(fc_preds_mt, 0), 0),
                nan_mask=d.get("fc_nan"),
            )
            fc_corr_meantheta = m_mt["corr"]
            fc_rmse_meantheta = m_mt["rmse"]
        if verbose:
            print(
                f"    FC corr(mean-theta) = {fc_corr_meantheta:.4f}  "
                f"(posterior-mean theta, {len(fc_preds_mt)} resim)"
            )
    except Exception as e:
        fc_corr_meantheta, fc_rmse_meantheta = 0.0, 1.0
        if verbose:
            print(f"    FC corr(mean-theta) skipped: {e}", flush=True)

    result = {
        "sid": sid,
        "samples_raw": samples_raw,
        "samples_scaled": samples_scaled,
        "means_raw": means_raw,
        "stds_raw": stds_raw,
        "shrinkage_scaled": shrink,
        "fc_obs": fc_obs_full,
        "fc_preds": fc_preds,
        "fc_corr_mean": float(np.mean(fc_corrs)) if fc_corrs else 0.0,
        "fc_corr_std": float(np.std(fc_corrs)) if fc_corrs else 0.0,
        "fc_corr_all": fc_corrs,
        "fc_corr_expected": float(fc_corr_expected),
        "fc_rmse_expected": float(fc_rmse_expected),
        "fc_corr_meantheta": float(fc_corr_meantheta),
        "fc_rmse_meantheta": float(fc_rmse_meantheta),
        "fc_rmse_mean": float(np.mean(fc_rmses)) if fc_rmses else 1.0,
        "fc_rmse_std": float(np.std(fc_rmses)) if fc_rmses else 0.0,
        "fcd_rmse_mean": (
            float(np.mean(fcd_rmses)) if fcd_rmses else 0.0
        ),
        "fcd_rmse_std": float(np.std(fcd_rmses)) if fcd_rmses else 0.0,
        "param_names": param_names,
    }
    if verbose:
        print(
            f"    FC corr      = {result['fc_corr_mean']:.4f} ± "
            f"{result['fc_corr_std']:.4f}"
        )
        print(
            f"    FC corr(exp) = {result['fc_corr_expected']:.4f}  "
            f"(expected-FC: avg {len(fc_preds)} resim FC, score once)"
        )
        print(f"    FC RMSE      = {result['fc_rmse_mean']:.4f}")
        if getattr(config, "USE_FCD", True):
            print(f"    FCD vec RMSE = {result['fcd_rmse_mean']:.4f}")
    return result


def _resimulate_and_score(n_resim, samples_raw, param_names,
                          fixed_overrides, sc, dly,
                          fc_obs_full, fcd_obs_raw, apply_bw,
                          sid=None, verbose=True, fc_nan=None):
    """Re-simulate from posterior samples; score FC (+ FCD)."""
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
    )
    from engine_select import get_simulate_gpu_batch  # honor INFERENCE_MODEL
    simulate_gpu_batch = get_simulate_gpu_batch()

    fc_corrs, fc_rmses, fcd_rmses, fc_preds = [], [], [], []
    t_resim = time.time()
    use_fcd = bool(getattr(config, "USE_FCD", True))
    if verbose:
        tag = f" ({sid})" if sid else ""
        _progress(f"resim start{tag}: {n_resim} simulations")

    theta_resim = np.asarray(samples_raw[:n_resim], dtype=np.float32)
    batch_label = f"resim({sid})" if sid else "resim"
    batch_kwargs = dict(
        delays=dly, apply_bw=apply_bw,
        label=batch_label, n_total=n_resim, subject_id=sid,
    )
    if fixed_overrides:
        batch_kwargs["fixed_overrides"] = fixed_overrides
    try:
        bold_list = simulate_gpu_batch(
            sc, theta_resim, param_names, **batch_kwargs,
        )
    except TypeError:
        # Older simulate_gpu_batch signature without label/n_total.
        bold_list = simulate_gpu_batch(
            sc, theta_resim, param_names,
            delays=dly, apply_bw=apply_bw,
            **({"fixed_overrides": fixed_overrides}
               if fixed_overrides else {}),
        )

    for i, bold in enumerate(bold_list):
        try:
            if bold is None:
                continue
            bold = np.asarray(bold)
            if bold.size == 0:
                continue
            fc_pred = compute_fc(bold)
            fc_preds.append(fc_pred)
            m = fc_metrics(fc_obs_full, fc_pred, nan_mask=fc_nan)
            fc_corrs.append(m["corr"])
            fc_rmses.append(m["rmse"])
            if use_fcd:
                fcd_pred_vec = fcd_to_upper_tri(
                    compute_sim_fcd_matrix(bold),
                )
                fcd_rmses.append(fcd_vec_rmse(fcd_obs_raw, fcd_pred_vec))
        except Exception as e:
            print(f"\n      resim {i} failed: {e}", flush=True)
            continue
    if verbose:
        _progress(
            f"resim done: {len(fc_corrs)}/{n_resim} OK  "
            f"({time.time() - t_resim:.1f}s)"
        )
    return fc_corrs, fc_rmses, fcd_rmses, fc_preds


# ---------------------------------------------------------------------------
# Baseline (prior midpoint)
# ---------------------------------------------------------------------------

def baseline_eval(sid, subject_data, n_resim=10, apply_bw=True,
                  verbose=True):
    """Prior-midpoint baseline simulation.

    The baseline theta is the prior midpoint in *raw* space, 0.5*(low+high)
    per dimension, for every PARAMETER_MODE:

    * homogeneous   -> 0.5*(lo+hi) per scalar param (legacy path below).
    * basis_regionwise -> the coefficient prior is symmetric (e.g. U(-2, 2)),
      so the midpoint is the ALL-ZERO coefficient vector. Decoding it gives
      tanh(basis @ 0) = 0, i.e. every parameter maps to its bound MIDPOINT:
      g_LRE=1.5 for (0,3), g_FFI=1.5 for (0,3), I_o=0.5 for (0,1),
      sigma=0.025 for (0,0.05).
    * direct/latent_regionwise -> 0.5*(lo+hi) per latent/region dim.

    M1 fix: region-wise thetas are coefficient/latent names (e.g. ``g_LRE_const``)
    that the simulator does NOT recognise as scalar params — passing them to
    ``simulate_single`` would be silently ignored (defaults used), producing a
    misleading baseline. So region-wise modes are routed through the SAME decoded
    (latent_wrap-wrapped) batch simulator used by ``_resimulate_and_score``.
    """
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
        extract_observed_features,
    )
    from engine_select import is_regionwise

    d = subject_data[sid]
    fc_obs_full = d["fc"]
    fc_obs_raw, fcd_obs_raw = extract_observed_features(d)
    use_fcd = bool(getattr(config, "USE_FCD", True))

    if is_regionwise():
        # Decoded baseline: raw prior-midpoint theta -> wrapped batch sim.
        low = np.asarray(config.STAGE1_PRIOR_LOW, dtype=np.float64)
        high = np.asarray(config.STAGE1_PRIOR_HIGH, dtype=np.float64)
        theta_mid = 0.5 * (low + high)                       # (theta_dim,)
        theta_batch = np.tile(
            theta_mid[None, :], (max(1, int(n_resim)), 1)).astype(np.float32)
        # _resimulate_and_score uses get_simulate_gpu_batch() == latent_wrap,
        # which decodes theta -> per-region maps before simulating.
        fc_corrs, fc_rmses, fcd_rmses, _ = _resimulate_and_score(
            theta_batch.shape[0], theta_batch, list(config.STAGE1_PARAMS),
            None, d["sc"], d["delays"], fc_obs_full, fcd_obs_raw, apply_bw,
            sid=f"baseline:{sid}", verbose=verbose, fc_nan=d.get("fc_nan"),
        )
    else:
        from engine_select import get_simulate_single  # honor INFERENCE_MODEL
        simulate_single = get_simulate_single()

        params = {}
        for n, lo, hi in zip(
            config.STAGE1_PARAMS,
            config.STAGE1_PRIOR_LOW,
            config.STAGE1_PRIOR_HIGH,
        ):
            params[n] = 0.5 * (lo + hi)
        params.update({"c_ee": 16.0, "c_ei": 12.0, "c_ie": 15.0, "c_ii": 3.0})

        fc_corrs, fc_rmses, fcd_rmses = [], [], []
        for _ in range(n_resim):
            try:
                bolds = simulate_single(
                    d["sc"], params, n_repeat=1,
                    delays=d["delays"], apply_bw=apply_bw,
                )
                bold = bolds[0]
                fc_pred = compute_fc(bold)
                m = fc_metrics(fc_obs_full, fc_pred, nan_mask=d.get("fc_nan"))
                fc_corrs.append(m["corr"])
                fc_rmses.append(m["rmse"])
                if use_fcd:
                    fcd_pred_vec = fcd_to_upper_tri(
                        compute_sim_fcd_matrix(bold)
                    )
                    fcd_rmses.append(fcd_vec_rmse(fcd_obs_raw, fcd_pred_vec))
            except Exception:
                continue

    out = {
        "fc_corr_mean": float(np.mean(fc_corrs)) if fc_corrs else 0.0,
        "fc_rmse_mean": float(np.mean(fc_rmses)) if fc_rmses else 1.0,
        "fcd_rmse_mean": (
            float(np.mean(fcd_rmses)) if fcd_rmses else 0.0
        ),
    }
    if verbose:
        msg = (
            f"  [baseline] {sid}: "
            f"FC corr={out['fc_corr_mean']:.4f}, "
            f"FC RMSE={out['fc_rmse_mean']:.4f}"
        )
        if use_fcd:
            msg += f", FCD RMSE={out['fcd_rmse_mean']:.4f}"
        print(msg)
    return out


def baseline_eval_subjects(subjects, subject_data, n_resim=10,
                           apply_bw=True, verbose=True):
    """Run baseline_eval for a list of subjects; return aggregated dict."""
    if verbose:
        _progress(
            f"baseline eval start: {len(subjects)} subjects x "
            f"{n_resim} resims"
        )
    t0 = time.time()
    results = []
    for s_idx, sid in enumerate(subjects):
        if verbose:
            _progress(
                f"baseline [{s_idx + 1}/{len(subjects)}] {sid}  "
                f"(elapsed {time.time() - t0:.1f}s)"
            )
        results.append(
            baseline_eval(sid, subject_data,
                          n_resim=n_resim, apply_bw=apply_bw,
                          verbose=verbose)
        )
    if verbose:
        _progress(f"baseline eval done ({time.time() - t0:.1f}s)")
    return {
        "fc_corr_mean": float(np.mean(
            [r["fc_corr_mean"] for r in results]
        )),
        "fc_rmse_mean": float(np.mean(
            [r["fc_rmse_mean"] for r in results]
        )),
        "fcd_rmse_mean": float(np.mean(
            [r["fcd_rmse_mean"] for r in results]
        )),
    }
