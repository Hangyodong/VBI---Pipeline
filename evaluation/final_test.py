"""Final test evaluation on the held-out test set.

Public API
----------
- final_test(test_subjects, subject_data, best_stage,
             stage1_result, stage2_result, n_resim, apply_bw, verbose)
       -> test_summary dict

Rules
-----
- Test set is used ONLY here, ONLY for the model that was selected on
  validation. It must not be touched during validation, tuning, or
  hyperparameter search.
- When ``config.USE_FCD`` is False, FCD bootstrap CI is still computed
  (it's all-zeros, harmless) but ``_print_test_summary`` suppresses
  the FCD row.
"""
import time

import config
from evaluation.metrics import (
    _progress,
    bootstrap_ci,
    evaluate_subject,
)


# ---------------------------------------------------------------------------
# Final test driver
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


# ---------------------------------------------------------------------------
# Per-stage helpers
# ---------------------------------------------------------------------------

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
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n  Test results (bootstrap 95% CI)")
    m, lo, hi = test_summary["fc_corr_boot_ci"]
    print(f"    FC corr   : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    m, lo, hi = test_summary["fc_rmse_boot_ci"]
    print(f"    FC RMSE   : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    if use_fcd:
        m, lo, hi = test_summary["fcd_rmse_boot_ci"]
        print(f"    FCD RMSE  : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
