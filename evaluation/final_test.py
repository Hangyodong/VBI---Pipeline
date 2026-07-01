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

def final_test(test_subjects, subject_data, best_stage=1,
               stage1_result=None, stage2_result=None,
               n_resim=None, apply_bw=True, verbose=True):
    """Evaluate the Stage 1 model on the held-out test set.

    The WC-EIB pipeline runs Stage 1 only; ``best_stage`` is always 1.
    ``stage2_result`` is accepted for call-site compatibility but unused.
    """
    n_resim = n_resim or config.N_TEST_RESIM

    if verbose:
        print("\n" + "=" * 65)
        print(f"  Step 14. Final test (Stage 1, n_resim={n_resim})")
        print("=" * 65)
        _progress(
            f"final test start: {len(test_subjects)} subjects x "
            f"{n_resim} resims (Stage 1)"
        )

    t0 = time.time()
    results = _test_stage1(
        test_subjects, subject_data, stage1_result,
        n_resim, apply_bw, verbose,
    )

    all_fc_corrs = [v for r in results for v in r["fc_corr_all"]]
    fc_corr_boot = bootstrap_ci(all_fc_corrs)
    fc_rmse_boot = bootstrap_ci([r["fc_rmse_mean"] for r in results])
    fcd_rmse_boot = bootstrap_ci([r["fcd_rmse_mean"] for r in results])
    # S1: expected-FC bootstrap over per-subject expected-FC scores (one per
    # subject, computed from the averaged resim FC). Reported alongside.
    fc_corr_exp_boot = bootstrap_ci(
        [r.get("fc_corr_expected", 0.0) for r in results])
    fc_rmse_exp_boot = bootstrap_ci(
        [r.get("fc_rmse_expected", 1.0) for r in results])
    # ADDITIVE: bootstrap over per-subject posterior-mean-theta FC corr (the
    # subject "digital twin" point-estimate). Reported alongside per-draw/exp.
    fc_corr_mt_boot = bootstrap_ci(
        [r.get("fc_corr_meantheta", 0.0) for r in results])

    test_summary = {
        "best_stage": 1,
        "per_subject": results,
        "fc_corr_boot_ci": fc_corr_boot,
        "fc_rmse_boot_ci": fc_rmse_boot,
        "fcd_rmse_boot_ci": fcd_rmse_boot,
        "fc_corr_expected_boot_ci": fc_corr_exp_boot,
        "fc_rmse_expected_boot_ci": fc_rmse_exp_boot,
        "fc_corr_meantheta_boot_ci": fc_corr_mt_boot,
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


def _print_test_summary(test_summary):
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n  Test results (bootstrap 95% CI)")
    m, lo, hi = test_summary["fc_corr_boot_ci"]
    print(f"    FC corr       : {m:.4f}  [{lo:.4f}, {hi:.4f}]  (per-draw mean)")
    if "fc_corr_expected_boot_ci" in test_summary:
        m, lo, hi = test_summary["fc_corr_expected_boot_ci"]
        print(f"    FC corr(exp)  : {m:.4f}  [{lo:.4f}, {hi:.4f}]  (expected-FC)")
    if "fc_corr_meantheta_boot_ci" in test_summary:
        m, lo, hi = test_summary["fc_corr_meantheta_boot_ci"]
        print(f"    FC corr(mean-θ): {m:.4f}  [{lo:.4f}, {hi:.4f}]  "
              f"(posterior-mean theta)")
    m, lo, hi = test_summary["fc_rmse_boot_ci"]
    print(f"    FC RMSE       : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    if "fc_rmse_expected_boot_ci" in test_summary:
        m, lo, hi = test_summary["fc_rmse_expected_boot_ci"]
        print(f"    FC RMSE(exp)  : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
    if use_fcd:
        m, lo, hi = test_summary["fcd_rmse_boot_ci"]
        print(f"    FCD RMSE  : {m:.4f}  [{lo:.4f}, {hi:.4f}]")
