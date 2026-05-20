"""Validation evaluation for Stage 1 and Stage 2.

Public API
----------
- evaluate_validation_stage1(val_subjects, subject_data, stage1_result, ...)
       -> (per_subject_results, agg_dict)
- evaluate_validation_stage2(val_subjects, subject_data, stage2_result,
                             stage1_result, ...)
       -> (per_subject_results, agg_dict)

Both functions return ``(results, agg)`` so main.py can do::

    val_results, val_agg = evaluate.evaluate_validation_stage1(...)

Validation rules
----------------
- Validation set is used ONLY for model selection (Stage 1 vs Stage 1+2)
  and θ_bad picking. It is NOT used for final reporting.
- Test set is never touched here.
"""
import time

import numpy as np

import config
from evaluation.metrics import _progress, evaluate_subject


# ---------------------------------------------------------------------------
# Stage 1 validation
# ---------------------------------------------------------------------------

def evaluate_validation_stage1(val_subjects, subject_data, stage1_result,
                               apply_bw=True, verbose=True):
    """Evaluate Stage 1 on all validation subjects.

    Returns
    -------
    results : list[dict]
        Per-subject evaluation dicts from ``evaluate_subject``.
    agg : dict
        Aggregated FC corr, FC RMSE, FCD RMSE, shrinkage_mean across
        subjects, plus ``per_subject`` and ``param_names``.
    """
    if not val_subjects:
        print("[Inference] N_VAL=0 — validation 스킵")
        return [], {}
    n_val = len(val_subjects)
    if verbose:
        print(
            f"\n[Inference] Stage 1 Validation"
            f"  subjects={n_val}"
            f"  N_POSTERIOR={getattr(config, 'N_POSTERIOR', 2000):,}"
            f"  N_TEST_RESIM={getattr(config, 'N_TEST_RESIM', 50)}",
            flush=True,
        )

    results = []
    t0 = time.time()
    for s_idx, sid in enumerate(val_subjects):
        if verbose:
            print(
                f"\n  [{sid}]  {s_idx + 1}/{n_val}"
                f"  sampling posterior ...",
                flush=True,
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
        if verbose:
            shrink = r.get("shrinkage_scaled", [])
            sk = "[" + ",".join(f"{float(v):.2f}" for v in shrink) + "]"
            print(
                f"  [{sid}]  {s_idx + 1}/{n_val}  DONE"
                f"  FC_corr={r.get('fc_corr_mean', 0.0):.3f}"
                f"  FC_rmse={r.get('fc_rmse_mean', 0.0):.3f}"
                f"  shrink={sk}",
                flush=True,
            )

    agg = _aggregate_validation(
        results, stage=1, param_names=config.STAGE1_PARAMS,
    )
    if verbose:
        el = time.time() - t0
        m, s = divmod(int(el), 60)
        print(
            f"\n[Inference] Stage 1 Validation DONE"
            f"  mean FC_corr={agg.get('fc_corr_mean', 0):.3f}"
            f"  mean FC_rmse={agg.get('fc_rmse_mean', 0):.3f}"
            f"  elapsed={m:02d}:{s:02d}",
            flush=True,
        )
        _print_validation_summary(agg, label="Stage 1 validation")
    return results, agg


# ---------------------------------------------------------------------------
# Stage 2 validation
# ---------------------------------------------------------------------------

def evaluate_validation_stage2(val_subjects, subject_data, stage2_result,
                               stage1_result, apply_bw=True,
                               verbose=True):
    """Evaluate Stage 2 on all validation subjects.

    Nuisance Stage 1 parameters are fixed at their Stage 1 posterior
    mean for each subject (deterministic, not sampled), then Stage 2
    infers theta_bad + c-params on top.

    Returns
    -------
    results : list[dict]
    agg : dict   (includes ``nuisance_params``)
    """
    if not val_subjects:
        print("[Inference] N_VAL=0 — validation 스킵")
        return [], {}
    n_val = len(val_subjects)
    if verbose:
        print(
            f"\n[Inference] Stage 2 Validation"
            f"  subjects={n_val}"
            f"  N_POSTERIOR={getattr(config, 'N_POSTERIOR', 2000):,}"
            f"  N_TEST_RESIM={getattr(config, 'N_TEST_RESIM', 50)}",
            flush=True,
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
            print(
                f"\n  [{sid}]  {s_idx + 1}/{n_val}"
                f"  sampling posterior ...",
                flush=True,
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
        if verbose:
            shrink = r.get("shrinkage_scaled", [])
            sk = "[" + ",".join(f"{float(v):.2f}" for v in shrink) + "]"
            print(
                f"  [{sid}]  {s_idx + 1}/{n_val}  DONE"
                f"  FC_corr={r.get('fc_corr_mean', 0.0):.3f}"
                f"  FC_rmse={r.get('fc_rmse_mean', 0.0):.3f}"
                f"  shrink={sk}",
                flush=True,
            )

    agg = _aggregate_validation(
        results, stage=2, param_names=s2_params,
    )
    agg["nuisance_params"] = nuisance
    if verbose:
        el = time.time() - t0
        m, s = divmod(int(el), 60)
        print(
            f"\n[Inference] Stage 2 Validation DONE"
            f"  mean FC_corr={agg.get('fc_corr_mean', 0):.3f}"
            f"  mean FC_rmse={agg.get('fc_rmse_mean', 0):.3f}"
            f"  elapsed={m:02d}:{s:02d}",
            flush=True,
        )
        _print_validation_summary(agg, label="Stage 2 validation")
    return results, agg


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

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
        "shrinkage_per_param": np.mean(
            [r["shrinkage_scaled"] for r in results], axis=0,
        ),
        "per_subject": results,
        "param_names": param_names,
    }


def _print_validation_summary(agg, label):
    print(f"\n  [{label} aggregate]")
    print(f"    FC corr   : {agg['fc_corr_mean']:.4f}")
    print(f"    FC RMSE   : {agg['fc_rmse_mean']:.4f}")
    if getattr(config, "USE_FCD", True):
        print(f"    FCD RMSE  : {agg['fcd_rmse_mean']:.4f}")
    shrink = dict(zip(
        agg["param_names"], agg["shrinkage_mean"].round(3),
    ))
    print(f"    Shrinkage : {shrink}")
