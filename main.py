"""main.py — Mouse MPTP VBI-SBI pipeline entry point.

Pipeline (production)
---------------------
 1. Load raw data           (FC / SC / tract length / participants.tsv)
 2. Train / val / test split
 3. Load per-subject data dicts
 4. Stage 1 simulation + feature extraction
 5. Feature pipeline + parameter scaler
 6. Stage 1 SNPE-C training
 7. Stage 1 validation analysis
 8. θ_bad selection (sensitivity high ∧ shrinkage low)
 9. Optional Stage 2 SNPE-C training (if θ_bad non-empty)
10. Stage 2 validation analysis
11. Model selection on VALIDATION ONLY
12. Final test on TEST SET ONLY (no model selection here)
13. Save artifacts + final summary

Rules
-----
- Train  : SBI training simulations only
- Val    : Stage 1 vs Stage 1+2 selection, θ_bad picking
- Test   : final evaluation of selected model only — never used for tuning
- Default FEATURE_SET = "fc_only" (no FCD without empirical BOLD)
- ParameterScaler maps raw ↔ [-1, 1]. SBI trains in scaled space,
  VBI simulation receives raw parameters.
"""
import os
import warnings

import numpy as np

import config
import data_loader
import evaluate
import inference

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Tunables (override via env or PipelineConfig in main.ipynb)
# ---------------------------------------------------------------------------

# Whether to attempt Stage 2 at all. Stage 2 is auto-skipped if θ_bad
# turns out to be empty.
RUN_STAGE2 = True

# Sensitivity / shrinkage thresholds for θ_bad selection.
SENS_THRESHOLD = 0.5
SHR_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Step 1 + 2 + 3 — Data loading & split
# ---------------------------------------------------------------------------

def step_data_split():
    """Load raw data, pick target subjects, do 4:2:2 split, bundle dicts."""
    print("\n" + "=" * 70)
    print("  Step 1-3. Data loading + Train/Val/Test split")
    print("=" * 70)

    out = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out

    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    train, val, test = data_loader.three_way_split(subjects)

    subject_data = data_loader.load_all_subjects(
        train + val + test,
        fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids,
    )

    n = config.N_REGIONS
    for sid, d in subject_data.items():
        assert d["fc"].shape == (n, n), f"{sid} fc shape"
        assert d["sc"].shape == (n, n), f"{sid} sc shape"

    return train, val, test, subject_data


# ---------------------------------------------------------------------------
# Step 4-6 — Stage 1 simulation + features + SNPE training
# ---------------------------------------------------------------------------

def stage1_pipeline(train, subject_data, n_sim=None):
    """Run Stage 1: simulate, fit feature pipeline + scaler, train SNPE.

    Returns
    -------
    stage1 : dict
        {posterior, feature_pipeline, param_scaler, prior_scaled,
         param_names, embedding_net, theta_scaled, x_input,
         fc_raw, fcd_raw}
    """
    print("\n" + "=" * 70)
    print("  Stage 1: simulation -> features -> SNPE-C")
    print("=" * 70)

    n_sim = n_sim if n_sim is not None else config.N_SIM
    return inference.run_stage1_snpe(
        train_subjects=train, subject_data=subject_data,
        n_sim=n_sim, apply_bw=True, verbose=True,
    )


# ---------------------------------------------------------------------------
# Step 7 — Stage 1 validation
# ---------------------------------------------------------------------------

def stage1_validation(val_subjects, subject_data, stage1):
    """Evaluate Stage 1 posterior on validation subjects."""
    print("\n" + "=" * 70)
    print("  Stage 1 validation")
    print("=" * 70)
    val_results, val_agg = evaluate.evaluate_validation_stage1(
        val_subjects, subject_data, stage1,
        apply_bw=True, verbose=True,
    )
    return val_results, val_agg


# ---------------------------------------------------------------------------
# Step 8 — θ_bad selection
# ---------------------------------------------------------------------------

def select_theta_bad_from_val(val_agg, param_names,
                              sens_threshold=SENS_THRESHOLD,
                              shr_threshold=SHR_THRESHOLD):
    """Pick θ_bad from validation aggregate.

    val_agg is expected to contain:
        "shrinkage_per_param" : array-like length n_params
        optionally "sensitivity_per_param" : same length

    If sensitivity is unavailable, we conservatively fall back to
    "shrinkage low only" (a Stage 1 parameter that was poorly inferred).
    The conservative fallback is documented so callers know it.
    """
    print("\n" + "=" * 70)
    print("  θ_bad selection")
    print("=" * 70)

    shrinkage = np.asarray(val_agg.get("shrinkage_per_param", []))
    sensitivity = val_agg.get("sensitivity_per_param", None)

    if len(shrinkage) != len(param_names):
        raise ValueError(
            f"shrinkage_per_param length {len(shrinkage)} != "
            f"param_names length {len(param_names)}"
        )

    if sensitivity is None:
        print(
            "  [warn] no per-parameter sensitivity in val_agg — "
            "using shrinkage-low-only criterion."
        )
        theta_bad = [
            n for n, s in zip(param_names, shrinkage)
            if s < shr_threshold
        ]
    else:
        sensitivity = np.asarray(sensitivity)
        theta_bad = inference.select_theta_bad(
            sensitivity, shrinkage, param_names=param_names,
            sens_threshold=sens_threshold,
            shrinkage_threshold=shr_threshold,
        )

    print(
        f"  Stage 1 params       : {list(param_names)}\n"
        f"  shrinkage_per_param  : "
        f"{[f'{s:.3f}' for s in shrinkage]}"
    )
    if sensitivity is not None:
        print(
            f"  sensitivity_per_param: "
            f"{[f'{s:.3f}' for s in sensitivity]}"
        )
    print(f"  → θ_bad = {theta_bad}")
    return theta_bad


# ---------------------------------------------------------------------------
# Step 9 — Stage 2 (optional)
# ---------------------------------------------------------------------------

def stage2_pipeline(train, val_subjects, subject_data, stage1, theta_bad,
                    n_sim=None):
    """Run Stage 2 if θ_bad is non-empty, otherwise return None."""
    if not theta_bad:
        print("\n  Stage 2 skipped — θ_bad is empty.")
        return None
    if not RUN_STAGE2:
        print("\n  Stage 2 skipped — RUN_STAGE2 = False.")
        return None

    print("\n" + "=" * 70)
    print("  Stage 2: simulation -> features -> SNPE-C")
    print("=" * 70)
    print(f"  Inferring          : {theta_bad + config.LOCAL_EI_PARAMS}")

    n_sim = n_sim if n_sim is not None else config.N_SIM_S2
    val_shrinkage = np.zeros(len(config.STAGE1_PARAMS))
    # Force "difficult" set to be exactly theta_bad
    for j, name in enumerate(config.STAGE1_PARAMS):
        if name in theta_bad:
            val_shrinkage[j] = 0.0   # below threshold
        else:
            val_shrinkage[j] = 1.0   # above threshold
    return inference.run_stage2_snpe(
        train_subjects=train, subject_data=subject_data,
        stage1_result=stage1, val_shrinkage=val_shrinkage,
        n_sim=n_sim, apply_bw=True, verbose=True,
    )


# ---------------------------------------------------------------------------
# Step 10 — Stage 2 validation
# ---------------------------------------------------------------------------

def stage2_validation(val_subjects, subject_data, stage1, stage2):
    if stage2 is None:
        return None, None
    print("\n" + "=" * 70)
    print("  Stage 2 validation")
    print("=" * 70)
    val_results, val_agg = evaluate.evaluate_validation_stage2(
        val_subjects, subject_data, stage2, stage1,
        apply_bw=True, verbose=True,
    )
    return val_results, val_agg


# ---------------------------------------------------------------------------
# Step 11 — Model selection (validation only)
# ---------------------------------------------------------------------------

def select_model(stage1_agg, stage2_agg, baseline_agg=None):
    print("\n" + "=" * 70)
    print("  Model selection (validation only)")
    print("=" * 70)
    best, _ = evaluate.select_best_model(
        stage1_agg=stage1_agg, stage2_agg=stage2_agg,
        baseline_agg=baseline_agg, verbose=True,
    )
    print(f"  → best model: Stage {best}")
    return best


# ---------------------------------------------------------------------------
# Step 12 — Final test (selected model only)
# ---------------------------------------------------------------------------

def run_final_test(test_subjects, subject_data, stage1, stage2, best_stage):
    print("\n" + "=" * 70)
    print(f"  Final test on test set (Stage {best_stage})")
    print("=" * 70)
    return evaluate.final_test(
        test_subjects=test_subjects, subject_data=subject_data,
        best_stage=best_stage, stage1_result=stage1,
        stage2_result=stage2, n_resim=config.N_TEST_RESIM,
        apply_bw=True, verbose=True,
    )


# ---------------------------------------------------------------------------
# Step 13 — Save & summary
# ---------------------------------------------------------------------------

def save_and_summarize(stage1, stage2, stage1_agg, stage2_agg,
                       best_stage, test_summary, theta_bad,
                       train_subjects):
    save_path = os.path.join(config.OUTPUT_DIR, "pipeline_artifacts.pkl")
    inference.save_artifacts(
        save_path,
        stage1=stage1, stage2=stage2,
        stage1_val_agg=stage1_agg, stage2_val_agg=stage2_agg,
        best_stage=best_stage, theta_bad=theta_bad,
        test_summary=test_summary,
    )
    print(f"\n  saved: {save_path}")
    evaluate.print_final_summary(
        stage1_agg=stage1_agg, stage2_agg=stage2_agg,
        best_stage=best_stage, test_summary=test_summary,
        train_subjects=train_subjects, n_train_sim=config.N_SIM,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.random.seed(config.SEED)
    try:
        import torch
        torch.manual_seed(config.SEED)
    except ImportError:
        pass

    config.print_config()

    # ── data ──
    train, val, test, subject_data = step_data_split()

    # ── Stage 1 ──
    stage1 = stage1_pipeline(train, subject_data)
    s1_val_results, s1_val_agg = stage1_validation(
        val, subject_data, stage1,
    )

    # ── θ_bad ──
    theta_bad = select_theta_bad_from_val(
        s1_val_agg, param_names=config.STAGE1_PARAMS,
    )

    # ── Stage 2 (optional) ──
    stage2 = stage2_pipeline(train, val, subject_data, stage1, theta_bad)
    s2_val_results, s2_val_agg = stage2_validation(
        val, subject_data, stage1, stage2,
    )

    # ── Model selection ──
    best_stage = select_model(s1_val_agg, s2_val_agg)

    # ── Final test ──
    test_summary = run_final_test(
        test, subject_data, stage1, stage2, best_stage,
    )

    # ── Save ──
    save_and_summarize(
        stage1, stage2, s1_val_agg, s2_val_agg,
        best_stage, test_summary, theta_bad,
        train_subjects=train,
    )


if __name__ == "__main__":
    main()
