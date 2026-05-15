"""Validation/test evaluation, model selection, and plotting.

Metrics
-------
- FC correlation (Pearson over upper triangle)
- FC RMSE
- FCD vector RMSE
- Posterior shrinkage (scaled space)
- Bootstrap confidence intervals (test set)
"""
import os
import time

import matplotlib
matplotlib.use("Agg")  # noqa: E402 - backend must be set before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Progress printing
# ---------------------------------------------------------------------------

def _progress(msg):
    """Print a timestamped progress message and flush."""
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def fc_metrics(fc_obs, fc_pred, nan_mask=None):
    """FC correlation, RMSE, and MAE on the full upper triangle."""
    n = fc_obs.shape[0]
    iu = np.triu_indices(n, k=1)
    a = fc_obs[iu]
    b = fc_pred[iu]
    mask = np.isfinite(a) & np.isfinite(b)
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
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
        extract_observed_features, simulate_single,
    )
    from inference import infer_subject_raw, compute_shrinkage_scaled

    n_resim = n_resim or config.N_TEST_RESIM
    d = subject_data[sid]
    sc = d["sc"]
    dly = d["delays"]
    fc_obs_full = d["fc"]

    if verbose:
        _progress(f"evaluating {sid} (posterior sampling)")

    fc_obs_raw, fcd_obs_raw = extract_observed_features(d)
    x_obs_input = feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)

    samples_raw, means_raw, stds_raw, samples_scaled = infer_subject_raw(
        posterior, x_obs_input, param_scaler,
        n_samples=n_resim, verbose=False,
    )
    shrink = compute_shrinkage_scaled(samples_scaled)

    if verbose:
        print(f"  [{sid}] posterior:")
        for i, name in enumerate(param_names):
            tag = " (low shrinkage)" if (
                shrink[i] < config.DIFFICULT_SHRINKAGE
            ) else ""
            print(
                f"    {name:6s} = {means_raw[i]:.4f} ± "
                f"{stds_raw[i]:.4f}  shrinkage={shrink[i]:.3f}{tag}"
            )

    fc_corrs, fc_rmses, fcd_rmses, fc_preds = _resimulate_and_score(
        n_resim, samples_raw, param_names, fixed_overrides,
        sc, dly, fc_obs_full, fcd_obs_raw, apply_bw,
        sid=sid, verbose=verbose,
    )

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
        print(f"    FC RMSE      = {result['fc_rmse_mean']:.4f}")
        if getattr(config, "USE_FCD", True):
            print(f"    FCD vec RMSE = {result['fcd_rmse_mean']:.4f}")
    return result


def _resimulate_and_score(n_resim, samples_raw, param_names,
                          fixed_overrides, sc, dly,
                          fc_obs_full, fcd_obs_raw, apply_bw,
                          sid=None, verbose=True):
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
        simulate_single,
    )

    fc_corrs, fc_rmses, fcd_rmses, fc_preds = [], [], [], []
    t_resim = time.time()
    use_fcd = bool(getattr(config, "USE_FCD", True))
    if verbose:
        tag = f" ({sid})" if sid else ""
        _progress(f"resim start{tag}: {n_resim} simulations")

    for i in range(n_resim):
        params = dict(fixed_overrides or {})
        for j, name in enumerate(param_names):
            params[name] = float(samples_raw[i, j])
        try:
            bolds = simulate_single(
                sc, params, n_repeat=1, delays=dly, apply_bw=apply_bw,
            )
            bold = bolds[0]
            fc_pred = compute_fc(bold)
            fc_preds.append(fc_pred)
            m = fc_metrics(fc_obs_full, fc_pred)
            fc_corrs.append(m["corr"])
            fc_rmses.append(m["rmse"])
            if use_fcd:
                fcd_pred_vec = fcd_to_upper_tri(
                    compute_sim_fcd_matrix(bold),
                )
                fcd_rmses.append(fcd_vec_rmse(fcd_obs_raw, fcd_pred_vec))
        except Exception as e:
            print(f"      resim {i} failed: {e}", flush=True)
            continue
        if verbose and (i + 1) in {
            max(1, n_resim // 4),
            max(1, n_resim // 2),
            max(1, 3 * n_resim // 4),
            n_resim,
        }:
            pct = (i + 1) / n_resim * 100
            elapsed = time.time() - t_resim
            _progress(
                f"resim {i + 1}/{n_resim} ({pct:.0f}%)  "
                f"({elapsed:.1f}s)"
            )
    if verbose:
        _progress(
            f"resim done: {len(fc_corrs)}/{n_resim} OK  "
            f"({time.time() - t_resim:.1f}s)"
        )
    return fc_corrs, fc_rmses, fcd_rmses, fc_preds


def baseline_eval(sid, subject_data, n_resim=10, apply_bw=True,
                  verbose=True):
    """Prior-midpoint baseline simulation."""
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
        extract_observed_features, simulate_single,
    )

    d = subject_data[sid]
    fc_obs_full = d["fc"]
    fc_obs_raw, fcd_obs_raw = extract_observed_features(d)

    params = {}
    for n, lo, hi in zip(
        config.STAGE1_PARAMS,
        config.STAGE1_PRIOR_LOW,
        config.STAGE1_PRIOR_HIGH,
    ):
        params[n] = 0.5 * (lo + hi)
    params.update({"c_ee": 16.0, "c_ei": 12.0, "c_ie": 15.0, "c_ii": 3.0})

    fc_corrs, fc_rmses, fcd_rmses = [], [], []
    use_fcd = bool(getattr(config, "USE_FCD", True))
    for _ in range(n_resim):
        try:
            bolds = simulate_single(
                d["sc"], params, n_repeat=1,
                delays=d["delays"], apply_bw=apply_bw,
            )
            bold = bolds[0]
            fc_pred = compute_fc(bold)
            m = fc_metrics(fc_obs_full, fc_pred)
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


# ---------------------------------------------------------------------------
# Validation aggregates
# ---------------------------------------------------------------------------

def evaluate_validation_stage1(val_subjects, subject_data, stage1_result,
                               apply_bw=True, verbose=True):
    """Evaluate Stage 1 on all validation subjects."""
    if verbose:
        print("\n" + "=" * 65)
        print("  Step 9. Stage 1 validation")
        print("=" * 65)
        _progress(
            f"validation start: {len(val_subjects)} subjects x "
            f"{config.N_TEST_RESIM} resims each"
        )

    results = []
    t0 = time.time()
    for s_idx, sid in enumerate(val_subjects):
        if verbose:
            _progress(
                f"[{s_idx + 1}/{len(val_subjects)}] {sid}  "
                f"(elapsed {time.time() - t0:.1f}s)"
            )
        r = evaluate_subject(
            sid, subject_data,
            posterior=stage1_result["posterior"],
            param_scaler=stage1_result["param_scaler"],
            feature_pipeline=stage1_result["feature_pipeline"],
            param_names=config.STAGE1_PARAMS,
            fixed_overrides=None,
            n_resim=config.N_TEST_RESIM,
            apply_bw=apply_bw, verbose=verbose,
        )
        results.append(r)

    agg = _aggregate_validation(
        results, stage=1, param_names=config.STAGE1_PARAMS,
    )
    if verbose:
        _progress(
            f"Stage 1 validation done ({time.time() - t0:.1f}s)"
        )
        _print_validation_summary(agg, label="Stage 1 validation")
    return agg


def evaluate_validation_stage2(val_subjects, subject_data, stage2_result,
                               stage1_result, apply_bw=True,
                               verbose=True):
    """Evaluate Stage 2 on all validation subjects."""
    if verbose:
        print("\n" + "=" * 65)
        print("  Step 12. Stage 2 validation")
        print("=" * 65)
        _progress(
            f"validation start: {len(val_subjects)} subjects x "
            f"{config.N_TEST_RESIM} resims each"
        )

    s2_posterior = stage2_result["posterior"]
    s2_param_scaler = stage2_result["param_scaler"]
    s2_pipeline = stage2_result["feature_pipeline"]
    s2_params = stage2_result["stage2_params"]
    nuisance = stage2_result["nuisance_params"]

    s1_posterior = stage1_result["posterior"]
    s1_param_scaler = stage1_result["param_scaler"]
    s1_pipeline = stage1_result["feature_pipeline"]

    from inference import infer_subject_raw
    from simulator import extract_observed_features

    results = []
    t0 = time.time()
    for s_idx, sid in enumerate(val_subjects):
        if verbose:
            _progress(
                f"[{s_idx + 1}/{len(val_subjects)}] {sid}  "
                f"(elapsed {time.time() - t0:.1f}s)"
            )
        d = subject_data[sid]
        fc_obs_raw, fcd_obs_raw = extract_observed_features(d)
        x_s1 = s1_pipeline.transform(fc_obs_raw, fcd_obs_raw)
        _, s1_means_raw, _, _ = infer_subject_raw(
            s1_posterior, x_s1, s1_param_scaler,
            n_samples=2000, verbose=False,
        )
        s1_lookup = dict(zip(config.STAGE1_PARAMS, s1_means_raw))
        fixed_for_s2 = {n: float(s1_lookup[n]) for n in nuisance}

        r = evaluate_subject(
            sid, subject_data,
            posterior=s2_posterior,
            param_scaler=s2_param_scaler,
            feature_pipeline=s2_pipeline,
            param_names=s2_params,
            fixed_overrides=fixed_for_s2,
            n_resim=config.N_TEST_RESIM,
            apply_bw=apply_bw, verbose=verbose,
        )
        r["fixed_from_s1"] = fixed_for_s2
        results.append(r)

    agg = _aggregate_validation(
        results, stage=2, param_names=s2_params,
    )
    agg["nuisance_params"] = nuisance
    if verbose:
        _progress(
            f"Stage 2 validation done ({time.time() - t0:.1f}s)"
        )
        _print_validation_summary(agg, label="Stage 2 validation")
    return agg


def _aggregate_validation(results, stage, param_names):
    return {
        "stage": stage,
        "fc_corr_mean": float(np.mean(
            [r["fc_corr_mean"] for r in results]
        )),
        "fc_rmse_mean": float(np.mean(
            [r["fc_rmse_mean"] for r in results]
        )),
        "fcd_rmse_mean": float(np.mean(
            [r["fcd_rmse_mean"] for r in results]
        )),
        "shrinkage_mean": np.mean(
            [r["shrinkage_scaled"] for r in results], axis=0,
        ),
        "per_subject": results,
        "param_names": param_names,
    }


def _print_validation_summary(agg, label):
    print(f"\n  [{label} aggregate]")
    print(f"    FC corr   : {agg['fc_corr_mean']:.4f}")
    print(f"    FC RMSE   : {agg['fc_rmse_mean']:.4f}")
    print(f"    FCD RMSE  : {agg['fcd_rmse_mean']:.4f}")
    shrink = dict(zip(
        agg["param_names"], agg["shrinkage_mean"].round(3),
    ))
    print(f"    Shrinkage : {shrink}")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def compute_selection_score(val_agg, baseline_agg=None):
    """Weighted score; if baseline given, normalize each metric by it."""
    if baseline_agg is not None:
        fc_corr_norm = (
            val_agg["fc_corr_mean"] - baseline_agg["fc_corr_mean"]
        )
        fc_rmse_norm = (
            (baseline_agg["fc_rmse_mean"] - val_agg["fc_rmse_mean"])
            / max(baseline_agg["fc_rmse_mean"], 1e-8)
        )
        fcd_rmse_norm = (
            (baseline_agg["fcd_rmse_mean"] - val_agg["fcd_rmse_mean"])
            / max(baseline_agg["fcd_rmse_mean"], 1e-8)
        )
    else:
        fc_corr_norm = val_agg["fc_corr_mean"]
        fc_rmse_norm = -val_agg["fc_rmse_mean"]
        fcd_rmse_norm = -val_agg["fcd_rmse_mean"]

    return (
        config.SELECT_W_FC_CORR * fc_corr_norm
        + config.SELECT_W_FC_RMSE * fc_rmse_norm
        + config.SELECT_W_FCD_RMSE * fcd_rmse_norm
    )


def select_best_model(stage1_agg, stage2_agg=None, baseline_agg=None,
                      verbose=True):
    """Choose Stage 1 vs Stage 2 by validation selection score."""
    score_1 = compute_selection_score(stage1_agg, baseline_agg)
    if stage2_agg is not None:
        score_2 = compute_selection_score(stage2_agg, baseline_agg)
    else:
        score_2 = -np.inf

    if verbose:
        print("\n" + "=" * 65)
        print("  Step 13. Model selection (validation)")
        print("=" * 65)
        print(f"  Stage 1 score : {score_1:+.4f}")
        if stage2_agg is not None:
            print(f"  Stage 2 score : {score_2:+.4f}")

    if stage2_agg is None:
        best = 1
    else:
        best = 2 if score_2 > score_1 else 1

    if verbose:
        print(f"\n  => Selected: Stage {best}")
        _print_selection_table(stage1_agg, stage2_agg)
    return best, score_1, score_2


def _print_selection_table(stage1_agg, stage2_agg):
    print("\n  +----------------+----------+----------+----------+")
    print("  |                |  Stage 1 |  Stage 2 |  delta   |")
    print("  +----------------+----------+----------+----------+")
    if stage2_agg is not None:
        d_corr = stage2_agg["fc_corr_mean"] - stage1_agg["fc_corr_mean"]
        d_rmse = stage2_agg["fc_rmse_mean"] - stage1_agg["fc_rmse_mean"]
        d_fcd = stage2_agg["fcd_rmse_mean"] - stage1_agg["fcd_rmse_mean"]
        print(
            f"  | FC corr        |  {stage1_agg['fc_corr_mean']:>+.4f} "
            f"|  {stage2_agg['fc_corr_mean']:>+.4f} "
            f"|  {d_corr:>+.4f} |"
        )
        print(
            f"  | FC RMSE        |  {stage1_agg['fc_rmse_mean']:.4f} "
            f"|  {stage2_agg['fc_rmse_mean']:.4f} "
            f"|  {d_rmse:>+.4f} |"
        )
        print(
            f"  | FCD vec RMSE   |  {stage1_agg['fcd_rmse_mean']:.4f} "
            f"|  {stage2_agg['fcd_rmse_mean']:.4f} "
            f"|  {d_fcd:>+.4f} |"
        )
    else:
        print(
            f"  | FC corr        |  {stage1_agg['fc_corr_mean']:>+.4f} "
            f"|   N/A    |   N/A    |"
        )
    print("  +----------------+----------+----------+----------+")


# ---------------------------------------------------------------------------
# Final test
# ---------------------------------------------------------------------------

def final_test(test_subjects, subject_data, best_stage,
               stage1_result, stage2_result=None,
               n_resim=None, apply_bw=True, verbose=True):
    """Evaluate the selected model on the held-out test set."""
    n_resim = n_resim or config.N_TEST_RESIM

    if verbose:
        print("\n" + "=" * 65)
        print(
            f"  Step 14. Final test (Stage {best_stage}, "
            f"n_resim={n_resim})"
        )
        print("=" * 65)
        _progress(
            f"final test start: {len(test_subjects)} subjects x "
            f"{n_resim} resims (Stage {best_stage})"
        )

    t0 = time.time()
    if best_stage == 1:
        results = _test_stage1(
            test_subjects, subject_data, stage1_result,
            n_resim, apply_bw, verbose,
        )
    else:
        results = _test_stage2(
            test_subjects, subject_data,
            stage1_result, stage2_result,
            n_resim, apply_bw, verbose,
        )

    all_fc_corrs = [v for r in results for v in r["fc_corr_all"]]
    fc_corr_boot = bootstrap_ci(all_fc_corrs)
    fc_rmse_boot = bootstrap_ci([r["fc_rmse_mean"] for r in results])
    fcd_rmse_boot = bootstrap_ci([r["fcd_rmse_mean"] for r in results])

    test_summary = {
        "best_stage": best_stage,
        "per_subject": results,
        "fc_corr_boot_ci": fc_corr_boot,
        "fc_rmse_boot_ci": fc_rmse_boot,
        "fcd_rmse_boot_ci": fcd_rmse_boot,
    }
    if verbose:
        _progress(f"final test done ({time.time() - t0:.1f}s)")
        _print_test_summary(test_summary)
    return test_summary


def _test_stage1(test_subjects, subject_data, stage1_result,
                 n_resim, apply_bw, verbose):
    results = []
    t0 = time.time()
    for s_idx, sid in enumerate(test_subjects):
        if verbose:
            _progress(
                f"[{s_idx + 1}/{len(test_subjects)}] {sid}  "
                f"(elapsed {time.time() - t0:.1f}s)"
            )
        r = evaluate_subject(
            sid, subject_data,
            posterior=stage1_result["posterior"],
            param_scaler=stage1_result["param_scaler"],
            feature_pipeline=stage1_result["feature_pipeline"],
            param_names=config.STAGE1_PARAMS,
            fixed_overrides=None,
            n_resim=n_resim, apply_bw=apply_bw, verbose=verbose,
        )
        results.append(r)
    return results


def _test_stage2(test_subjects, subject_data,
                 stage1_result, stage2_result,
                 n_resim, apply_bw, verbose):
    from inference import infer_subject_raw
    from simulator import extract_observed_features

    s1_posterior = stage1_result["posterior"]
    s1_param_scaler = stage1_result["param_scaler"]
    s1_pipeline = stage1_result["feature_pipeline"]
    s2_posterior = stage2_result["posterior"]
    s2_param_scaler = stage2_result["param_scaler"]
    s2_pipeline = stage2_result["feature_pipeline"]
    stage2_params = stage2_result["stage2_params"]
    nuisance = stage2_result["nuisance_params"]

    results = []
    t0 = time.time()
    for s_idx, sid in enumerate(test_subjects):
        if verbose:
            _progress(
                f"[{s_idx + 1}/{len(test_subjects)}] {sid}  "
                f"(elapsed {time.time() - t0:.1f}s)"
            )
        d = subject_data[sid]
        fc_obs_raw, fcd_obs_raw = extract_observed_features(d)
        x_s1 = s1_pipeline.transform(fc_obs_raw, fcd_obs_raw)
        _, s1_means_raw, _, _ = infer_subject_raw(
            s1_posterior, x_s1, s1_param_scaler,
            n_samples=2000, verbose=False,
        )
        s1_lookup = dict(zip(config.STAGE1_PARAMS, s1_means_raw))
        fixed_for_s2 = {n: float(s1_lookup[n]) for n in nuisance}

        r = evaluate_subject(
            sid, subject_data,
            posterior=s2_posterior,
            param_scaler=s2_param_scaler,
            feature_pipeline=s2_pipeline,
            param_names=stage2_params,
            fixed_overrides=fixed_for_s2,
            n_resim=n_resim, apply_bw=apply_bw, verbose=verbose,
        )
        r["fixed_from_s1"] = fixed_for_s2
        results.append(r)
    return results


def _print_test_summary(test_summary):
    print("\n  Test results (bootstrap 95% CI)")
    m, lo, hi = test_summary["fc_corr_boot_ci"]
    print(f"    FC corr   : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    m, lo, hi = test_summary["fc_rmse_boot_ci"]
    print(f"    FC RMSE   : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    m, lo, hi = test_summary["fcd_rmse_boot_ci"]
    print(f"    FCD RMSE  : {m:.4f}  [{lo:.4f}, {hi:.4f}]")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_posteriors(results, param_names, prior_low, prior_high,
                    title="Stage 1", save_path=None):
    """Histogram of posterior samples per subject x parameter."""
    n_subj = len(results)
    n_p = len(param_names)
    fig, axes = plt.subplots(
        n_subj, n_p, figsize=(3 * n_p, 3 * n_subj), squeeze=False,
    )
    for r_idx, res in enumerate(results):
        samples = res["samples_raw"]
        for c, name in enumerate(param_names):
            ax = axes[r_idx, c]
            ax.hist(
                samples[:, c], bins=50,
                color="steelblue", alpha=0.6, density=True,
            )
            ax.set_xlim(prior_low[c], prior_high[c])
            ax.axvline(
                res["means_raw"][c], color="red",
                linestyle="--", lw=1,
            )
            ax.set_title(f"{res['sid']} | {name}")
            ax.set_xlabel(name)

    plt.suptitle(f"Posteriors - {title}", fontsize=12)
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR,
        f"posterior_{title.lower().replace(' ', '_')}.png",
    )
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


def plot_fc_comparison(results, save_path=None, title="FC comparison"):
    """Observed FC vs mean predicted FC, side-by-side per subject."""
    n_subj = len(results)
    fig, axes = plt.subplots(
        n_subj, 2, figsize=(8, 4 * n_subj), squeeze=False,
    )
    for r_idx, res in enumerate(results):
        axes[r_idx, 0].imshow(res["fc_obs"], cmap="RdBu_r",
                              vmin=-1, vmax=1)
        axes[r_idx, 0].set_title(f"{res['sid']}\nObserved FC")
        if res["fc_preds"]:
            fc_mean_pred = np.mean(res["fc_preds"], axis=0)
            axes[r_idx, 1].imshow(
                fc_mean_pred, cmap="RdBu_r", vmin=-1, vmax=1,
            )
            axes[r_idx, 1].set_title(
                f"Predicted (mean)\n"
                f"corr={res['fc_corr_mean']:.3f}, "
                f"RMSE={res['fc_rmse_mean']:.3f}"
            )

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR, "fc_comparison.png",
    )
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# Two-stage aliases for the refactored main.py interface
# ---------------------------------------------------------------------------

def plot_posteriors_two_stage(results, param_names,
                              prior_low, prior_high,
                              title="Stage 1+2", save_path=None):
    """Alias of plot_posteriors for the two-stage API."""
    return plot_posteriors(
        results, param_names, prior_low, prior_high,
        title=title, save_path=save_path,
    )


def plot_fc_comparison_two_stage(results, save_path=None,
                                 title="FC comparison (two-stage)"):
    """Alias of plot_fc_comparison for the two-stage API."""
    return plot_fc_comparison(results, save_path=save_path, title=title)


def print_summary_two_stage(results_s1, results_s2=None,
                            theta_bad=None):
    """Print a compact summary of Stage 1 / Stage 1+2 validation results.

    Parameters
    ----------
    results_s1 : list[dict]  per-subject Stage 1 results
                            (from evaluate_subject_stage1 or analogue)
    results_s2 : list[dict] or None  Stage 2 results, if available
    theta_bad  : list[str] or None   params chosen for Stage 2
    """
    print("\n" + "=" * 70)
    print("  Two-stage summary")
    print("=" * 70)
    if results_s1:
        fc_corr_s1 = np.mean([r.get("fc_corr_mean", 0.0)
                              for r in results_s1])
        fc_rmse_s1 = np.mean([r.get("fc_rmse_mean", 1.0)
                              for r in results_s1])
        print(
            f"  Stage 1 : FC corr = {fc_corr_s1:.4f}, "
            f"FC RMSE = {fc_rmse_s1:.4f}  "
            f"(n={len(results_s1)})"
        )
    if theta_bad is not None:
        print(f"  theta_bad selected : {theta_bad}")
    if results_s2:
        fc_corr_s2 = np.mean([r.get("fc_corr_mean", 0.0)
                              for r in results_s2])
        fc_rmse_s2 = np.mean([r.get("fc_rmse_mean", 1.0)
                              for r in results_s2])
        print(
            f"  Stage 2 : FC corr = {fc_corr_s2:.4f}, "
            f"FC RMSE = {fc_rmse_s2:.4f}  "
            f"(n={len(results_s2)})"
        )
        if results_s1:
            d_corr = fc_corr_s2 - fc_corr_s1
            d_rmse = fc_rmse_s1 - fc_rmse_s2
            print(
                f"  Δ corr  = {d_corr:+.4f}   "
                f"Δ rmse  = {d_rmse:+.4f}   "
                f"(positive = Stage 2 better)"
            )


def evaluate_all_two_stage(*args, **kwargs):
    """Alias for the existing evaluate_all_two_stage if present.

    If the module already defines a function with the same name, we
    delegate to it. Otherwise this is a thin wrapper around the per-
    subject evaluation helper.
    """
    # If a real implementation exists later in this file, leave
    # this stub unused; Python takes the last definition.
    raise NotImplementedError(
        "evaluate_all_two_stage: pipeline-specific implementation "
        "should override this. See main.py for the two-stage flow."
    )


def plot_sbc_rank_histogram(ranks, param_names=None, save_path=None):
    """SBC rank histogram per parameter."""
    if ranks is None or len(ranks) == 0:
        print("  no SBC ranks")
        return
    param_names = param_names or config.STAGE1_PARAMS
    n_p = ranks.shape[1]
    n_bins = config.SBC_BINS

    fig, axes = plt.subplots(
        1, n_p, figsize=(3 * n_p, 3), squeeze=False,
    )
    n_post = ranks.max() + 1
    expected = len(ranks) / n_bins

    for i, name in enumerate(param_names):
        ax = axes[0, i]
        ax.hist(
            ranks[:, i], bins=n_bins, range=(0, n_post),
            edgecolor="black", color="lightblue",
        )
        ax.axhline(
            expected, color="red", linestyle="--",
            label=f"uniform ({expected:.1f})",
        )
        ax.set_title(name)
        ax.set_xlabel("rank")
        ax.legend(fontsize=7)

    plt.suptitle("SBC rank histograms", fontsize=12)
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR, "sbc_ranks.png",
    )
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


def plot_pca_diagnostic(pca_diag, save_path=None):
    """Bar plot of the top-5 PCs' explained variance ratio."""
    fc_diag = pca_diag.get("fc_pca", pca_diag)
    if "explained_variance_top5" not in fc_diag:
        return
    evr_top = fc_diag["explained_variance_top5"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(evr_top)), evr_top, color="steelblue")
    ax.set_xlabel("PC index")
    ax.set_ylabel("EVR")
    ax.set_title(
        f"PCA top-5 EVR  "
        f"(cum EVR = {fc_diag['explained_variance_sum']:.4f})"
    )
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR, "pca_evr.png",
    )
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_final_summary(stage1_agg, stage2_agg, best_stage,
                        test_summary, train_subjects, n_train_sim):
    """Compact end-of-run summary."""
    print("\n" + "=" * 95)
    print("  Pipeline complete - final summary")
    print("=" * 95)
    print(f"  Train subjects  : {len(train_subjects)}")
    print(f"  Stage 1 sims    : {n_train_sim}")
    print(f"  Selected stage  : {best_stage}")
    print()
    s2_corr = stage2_agg["fc_corr_mean"] if stage2_agg else 0.0
    s2_rmse = stage2_agg["fc_rmse_mean"] if stage2_agg else 0.0
    s2_fcd = stage2_agg["fcd_rmse_mean"] if stage2_agg else 0.0
    print("  +----------------+----------+----------+----------+")
    print("  | Metric         |  Val S1  |  Val S2  |  Test    |")
    print("  +----------------+----------+----------+----------+")
    print(
        f"  | FC corr        |  {stage1_agg['fc_corr_mean']:>+.4f} "
        f"|  {s2_corr:>+.4f} "
        f"|  {test_summary['fc_corr_boot_ci'][0]:>+.4f} |"
    )
    print(
        f"  | FC RMSE        |  {stage1_agg['fc_rmse_mean']:.4f} "
        f"|  {s2_rmse:.4f} "
        f"|  {test_summary['fc_rmse_boot_ci'][0]:.4f} |"
    )
    print(
        f"  | FCD vec RMSE   |  {stage1_agg['fcd_rmse_mean']:.4f} "
        f"|  {s2_fcd:.4f} "
        f"|  {test_summary['fcd_rmse_boot_ci'][0]:.4f} |"
    )
    print("  +----------------+----------+----------+----------+")
    print("\n  Test bootstrap 95% CI:")
    _, lo, hi = test_summary["fc_corr_boot_ci"]
    print(f"    FC corr   : [{lo:.4f}, {hi:.4f}]")
    _, lo, hi = test_summary["fc_rmse_boot_ci"]
    print(f"    FC RMSE   : [{lo:.4f}, {hi:.4f}]")
    _, lo, hi = test_summary["fcd_rmse_boot_ci"]
    print(f"    FCD RMSE  : [{lo:.4f}, {hi:.4f}]")


# ---------------------------------------------------------------------------
# Per-step result reporters (for notebook cells)
# ---------------------------------------------------------------------------

def report_step1(train, val, test, subject_data):
    """Step 1 summary table + SC sparsity bar plot."""
    print("\n" + "=" * 60)
    print("  Step 1 result")
    print("=" * 60)
    print(f"  train ({len(train)}): {train}")
    print(f"  val   ({len(val)}): {val}")
    print(f"  test  ({len(test)}): {test}")

    sids = train + val + test
    sc_edges = [int((subject_data[s]["sc"] > 0).sum()) for s in sids]
    fc_nan = [int(subject_data[s]["fc_nan"].sum()) for s in sids]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].bar(range(len(sids)), sc_edges, color="steelblue")
    axes[0].set_xticks(range(len(sids)))
    axes[0].set_xticklabels(sids, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("SC nonzero edges")
    axes[0].set_title("SC sparsity per subject")

    axes[1].bar(range(len(sids)), fc_nan, color="indianred")
    axes[1].set_xticks(range(len(sids)))
    axes[1].set_xticklabels(sids, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("FC NaN count")
    axes[1].set_title("FC NaN per subject")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step1_subjects.png")
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step2(theta_scaled, fc_raw, fcd_raw):
    """Step 2 summary: shapes + theta distribution histogram."""
    print("\n" + "=" * 60)
    print("  Step 2 result")
    print("=" * 60)
    print(f"  theta_scaled : {theta_scaled.shape}")
    print(f"  fc_raw       : {fc_raw.shape}  "
          f"(finite={np.all(np.isfinite(fc_raw))})")
    print(f"  fcd_raw      : {fcd_raw.shape}  "
          f"(finite={np.all(np.isfinite(fcd_raw))})")

    # 시뮬 실패 (예: OOM) -> 빈 배열인 경우
    if theta_scaled.ndim < 2 or theta_scaled.shape[0] == 0:
        print()
        print("  !! Step 2 collected 0 samples — simulation failed.")
        print("  !! Common causes:")
        print("     - GPU_BATCH too large (OOM)")
        print("       Setup 셀에서 GPU_BATCH를 줄여서 다시 실행하세요.")
        print("     - VBI WC engine error (check earlier batch logs)")
        return

    n_params = theta_scaled.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(3 * n_params, 3),
                             squeeze=False)
    for i in range(n_params):
        axes[0, i].hist(theta_scaled[:, i], bins=40,
                        color="steelblue", alpha=0.7)
        axes[0, i].set_title(f"{config.STAGE1_PARAMS[i]} (scaled)")
        axes[0, i].set_xlim(-1, 1)
    plt.suptitle("Step 2 - sampled theta distribution", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step2_theta_hist.png")
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def plot_one_simulation(sid, subject_data, theta_raw, param_names,
                        sim_idx=0, save_name="step2_one_sim.png"):
    """Re-simulate one sample from Step 2's theta and plot BOLD + FC.

    Useful as a sanity check after Step 2: pick one of the 10000
    simulated parameter sets, re-run the WC + HRF on the same subject,
    and visualise the resulting BOLD time series and simulated FC.

    Parameters
    ----------
    sid          : str       subject id (one of `train`)
    subject_data : dict      output of data_loader.load_all_subjects
    theta_raw    : np.ndarray (N_sim, n_params)  Step 2 raw params
    param_names  : list[str]  config.STAGE1_PARAMS
    sim_idx      : int        which simulation (0..N_sim-1)
    save_name    : str        output filename
    """
    from simulator import simulate_single, compute_fc

    d = subject_data[sid]
    params = {n: float(theta_raw[sim_idx, i])
              for i, n in enumerate(param_names)}

    print(f"  [one-sim plot] sid={sid}  sim_idx={sim_idx}")
    print(f"    params = {params}")

    bolds = simulate_single(
        d["sc"], params, n_repeat=1,
        delays=d["delays"], apply_bw=True,
    )
    bold = bolds[0]                       # (T_bold, N)
    fc_sim = compute_fc(bold)              # (N, N)
    fc_obs = d["fc"]

    # FC correlation between sim and obs (off-diagonal)
    iu = np.triu_indices(fc_sim.shape[0], k=1)
    fc_corr = float(np.corrcoef(fc_sim[iu], fc_obs[iu])[0, 1])

    print(f"    BOLD shape : {bold.shape}")
    print(f"    BOLD range : [{bold.min():.3f}, {bold.max():.3f}]  "
          f"std={bold.std():.3f}")
    print(f"    Sim FC vs Obs FC: Pearson r = {fc_corr:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) BOLD time series (first 5 regions)
    t = np.arange(bold.shape[0]) * config.TR_SEC
    for i in range(min(5, bold.shape[1])):
        axes[0, 0].plot(t, bold[:, i], lw=0.8, alpha=0.8,
                        label=f"region {i}")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("BOLD")
    axes[0, 0].set_title(f"{sid}  sim {sim_idx} — BOLD (5 regions)")
    axes[0, 0].legend(fontsize=7, ncol=5)

    # (b) BOLD heatmap (all regions)
    im = axes[0, 1].imshow(bold.T, aspect="auto", cmap="RdBu_r",
                           vmin=-np.abs(bold).max(),
                           vmax=np.abs(bold).max())
    axes[0, 1].set_xlabel("TR")
    axes[0, 1].set_ylabel("Region")
    axes[0, 1].set_title(f"BOLD all regions ({bold.shape[1]} x {bold.shape[0]})")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # (c) Simulated FC
    im2 = axes[1, 0].imshow(fc_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_xlabel("Region")
    axes[1, 0].set_ylabel("Region")
    axes[1, 0].set_title(f"Simulated FC  (sim {sim_idx})")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label="r")

    # (d) Observed FC + similarity
    im3 = axes[1, 1].imshow(fc_obs, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_xlabel("Region")
    axes[1, 1].set_ylabel("Region")
    axes[1, 1].set_title(
        f"Empirical FC ({sid})\nSim vs Obs corr = {fc_corr:.3f}"
    )
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, label="r")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")
    return {"bold": bold, "fc_sim": fc_sim, "fc_corr": fc_corr,
            "params": params}


def report_step3(fc_raw, fcd_raw):
    """Step 3 summary: FC and FCD value distributions."""
    print("\n" + "=" * 60)
    print("  Step 3 result")
    print("=" * 60)
    print(f"  FC  raw : shape={fc_raw.shape}, "
          f"min={float(fc_raw.min()):.3f}, "
          f"max={float(fc_raw.max()):.3f}, "
          f"mean={float(fc_raw.mean()):.3f}")
    print(f"  FCD raw : shape={fcd_raw.shape}, "
          f"min={float(fcd_raw.min()):.3f}, "
          f"max={float(fcd_raw.max()):.3f}, "
          f"mean={float(fcd_raw.mean()):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(fc_raw.flatten(), bins=80, color="steelblue", alpha=0.7)
    axes[0].set_title("FC raw distribution")
    axes[0].set_xlabel("value")
    axes[1].hist(fcd_raw.flatten(), bins=80, color="seagreen", alpha=0.7)
    axes[1].set_title("FCD raw distribution")
    axes[1].set_xlabel("value")
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step3_feature_dist.png")
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step4(scalers, fc_raw, fcd_raw):
    """Step 4 summary: FCD z-score (FC has none — already in [-1, 1])."""
    print("\n" + "=" * 60)
    print("  Step 4 result")
    print("=" * 60)
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print(f"  FC     : raw Pearson r [-1, 1], no z-score")
    if use_fcd:
        fcd_z = scalers["fcd_z"].transform(fcd_raw)
        print(f"  FCD z  : mean={float(fcd_z.mean()):.4f}, "
              f"std={float(fcd_z.std()):.4f}")
    else:
        print(f"  FCD    : disabled (USE_FCD=False)")
        fcd_z = None

    # Plot raw FC + raw/z FCD (if enabled)
    n_rows = 2 if use_fcd else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows),
                             squeeze=False)
    axes[0, 0].hist(fc_raw.flatten(), bins=80,
                    color="steelblue", alpha=0.7)
    axes[0, 0].set_title("FC raw (Pearson r)")
    axes[0, 0].axvline(0, color="gray", ls=":", lw=0.5)
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5,
                    "FC z-score: disabled\n(already in [-1, 1])",
                    ha="center", va="center", fontsize=11,
                    color="gray")
    if use_fcd:
        axes[1, 0].hist(fcd_raw.flatten(), bins=80,
                        color="seagreen", alpha=0.7)
        axes[1, 0].set_title("FCD raw")
        axes[1, 1].hist(fcd_z.flatten(), bins=80,
                        color="seagreen", alpha=0.7)
        axes[1, 1].set_title("FCD z-scored")
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step4_zscore.png")
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step5(pipeline, x_input):
    """Step 5 summary: FC PCA dimensions + FCD summary stats."""
    print("\n" + "=" * 60)
    print("  Step 5 result")
    print("=" * 60)
    print(f"  FC  PCA  : {pipeline.fc_dim} -> "
          f"{pipeline.fc_pca.n_components}")
    print(f"  FCD      : {pipeline.fcd_dim} dims "
          f"(summary stats, z-scored, no PCA)")
    print(f"  x_input  : {x_input.shape}")

    evr_fc = pipeline.fc_pca.explained_variance_ratio_

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    ax.plot(np.cumsum(evr_fc), color="steelblue", lw=2)
    ax.axhline(0.9, color="red", ls="--", lw=1, label="90%")
    ax.axhline(0.95, color="orange", ls="--", lw=1, label="95%")
    ax.set_xlabel("PC index")
    ax.set_ylabel("cumulative EVR")
    ax.set_title(f"FC PCA (sum = {evr_fc.sum():.4f})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step5_pca_evr.png")
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step6(pca_diagnostic):
    """Step 6 summary: FC PCA diagnostic + FCD summary stats info."""
    print("\n" + "=" * 60)
    print("  Step 6 result")
    print("=" * 60)

    # FC PCA diagnostic
    d = pca_diagnostic.get("fc_pca", {})
    if d:
        evr_ok = "PASS" if d.get("pca_pass_evr") else "FAIL"
        rec_ok = "PASS" if d.get("pca_pass_recon") else "FAIL"
        print(f"  FC:")
        print(f"    n_components     : {d['n_components']}")
        print(f"    cum EVR          : "
              f"{d['explained_variance_sum']:.4f} [{evr_ok}]")
        print(f"    recon corr (train) : "
              f"{d['recon_corr_train_mean']:.4f} [{rec_ok}]")

    # FCD: summary stats (no PCA)
    d2 = pca_diagnostic.get("fcd_pca", {})
    if d2:
        print(f"  FCD:")
        print(f"    type   : {d2.get('type', 'summary_stats')}")
        print(f"    dims   : {d2.get('dims', [])}")
        if "train_mean" in d2:
            print(f"    mean   : {d2['train_mean']:.4f}")
            print(f"    std    : {d2['train_std']:.4f}")


def report_step7(param_scaler):
    """Step 7 summary: prior bounds and scaled range."""
    print("\n" + "=" * 60)
    print("  Step 7 result")
    print("=" * 60)
    print("  Prior bounds:")
    for name, lo, hi in zip(param_scaler.param_names,
                            param_scaler.low, param_scaler.high):
        print(f"    {name:6s} : [{float(lo):7.3f}, {float(hi):7.3f}]"
              f"  -> [-1, 1]")


def report_step8(posterior, embedding_net, theta_scaled, x_input):
    """Step 8 summary: trained estimator info + sample posterior."""
    print("\n" + "=" * 60)
    print("  Step 8 result")
    print("=" * 60)
    print(f"  theta_scaled : {theta_scaled.shape}")
    print(f"  x_input      : {x_input.shape}")
    has_params = any(p.requires_grad
                     for p in embedding_net.parameters())
    if has_params:
        n_p = sum(p.numel() for p in embedding_net.parameters())
        print(f"  embedding net params: {n_p:,}")
    print(f"  posterior     : {type(posterior).__name__}")


def report_step9(stage1_agg, baseline_agg):
    """Step 9 summary: Stage 1 vs baseline metrics."""
    print("\n" + "=" * 60)
    print("  Step 9 result")
    print("=" * 60)
    print("                       Stage 1     Baseline    delta")
    print("  -----------------------------------------------------")
    d_fc = stage1_agg["fc_corr_mean"] - baseline_agg["fc_corr_mean"]
    d_rmse = baseline_agg["fc_rmse_mean"] - stage1_agg["fc_rmse_mean"]
    d_fcd = baseline_agg["fcd_rmse_mean"] - stage1_agg["fcd_rmse_mean"]
    print(f"  FC  corr           {stage1_agg['fc_corr_mean']:>+8.4f}  "
          f"{baseline_agg['fc_corr_mean']:>+8.4f}  {d_fc:>+8.4f}")
    print(f"  FC  RMSE           {stage1_agg['fc_rmse_mean']:>8.4f}  "
          f"{baseline_agg['fc_rmse_mean']:>8.4f}  {d_rmse:>+8.4f}")
    print(f"  FCD RMSE           {stage1_agg['fcd_rmse_mean']:>8.4f}  "
          f"{baseline_agg['fcd_rmse_mean']:>8.4f}  {d_fcd:>+8.4f}")
    print()
    print("  Shrinkage per param:")
    for name, s in zip(stage1_agg["param_names"],
                       stage1_agg["shrinkage_mean"]):
        mark = "OK" if s >= config.DIFFICULT_SHRINKAGE else "LOW"
        print(f"    {name:6s} : {float(s):.4f}  [{mark}]")


def report_step10(difficult, val_shrinkage):
    """Step 10 summary: which params go into Stage 2."""
    print("\n" + "=" * 60)
    print("  Step 10 result")
    print("=" * 60)
    print(f"  Shrinkage threshold : {config.DIFFICULT_SHRINKAGE}")
    print(f"  Difficult params    : {difficult}")
    print(f"  c-params to add     : "
          f"{list(config.C_PARAM_PRIOR.keys())}")
    print(f"  Stage 2 targets     : "
          f"{difficult + list(config.C_PARAM_PRIOR.keys())}")


def report_step11(s2):
    """Step 11 summary: Stage 2 parameter targets."""
    print("\n" + "=" * 60)
    print("  Step 11 result")
    print("=" * 60)
    print(f"  Stage 2 params   : {s2['stage2_params']}")
    print(f"  Nuisance params  : {s2['nuisance_params']}")
    print(f"  theta_scaled     : {s2['theta_scaled'].shape}")
    print(f"  x_input          : {s2['x_input'].shape}")


def report_step12(stage2_agg):
    """Step 12 summary: Stage 2 metrics."""
    print("\n" + "=" * 60)
    print("  Step 12 result")
    print("=" * 60)
    print(f"  FC  corr   : {stage2_agg['fc_corr_mean']:>+.4f}")
    print(f"  FC  RMSE   : {stage2_agg['fc_rmse_mean']:.4f}")
    print(f"  FCD RMSE   : {stage2_agg['fcd_rmse_mean']:.4f}")
    print()
    print("  Shrinkage per param:")
    for name, s in zip(stage2_agg["param_names"],
                       stage2_agg["shrinkage_mean"]):
        print(f"    {name:6s} : {float(s):.4f}")


def report_step13(best_stage, score_1, score_2,
                  stage1_agg, stage2_agg):
    """Step 13 summary: selection scores + winner."""
    print("\n" + "=" * 60)
    print("  Step 13 result")
    print("=" * 60)
    print(f"  Stage 1 score : {score_1:>+.4f}")
    if stage2_agg is not None:
        print(f"  Stage 2 score : {score_2:>+.4f}")
    print(f"\n  Selected: Stage {best_stage}")


def report_step14(test_summary):
    """Step 14 summary: test metrics with bootstrap CI."""
    print("\n" + "=" * 60)
    print("  Step 14 result")
    print("=" * 60)
    for label, key in [("FC  corr ", "fc_corr_boot_ci"),
                       ("FC  RMSE ", "fc_rmse_boot_ci"),
                       ("FCD RMSE ", "fcd_rmse_boot_ci")]:
        m, lo, hi = test_summary[key]
        print(f"  {label}: {m:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}]")
