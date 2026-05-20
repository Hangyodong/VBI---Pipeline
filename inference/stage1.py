"""Stage 1 end-to-end driver.

Public API
----------
- run_stage1_snpe(train_subjects, subject_data, ...) -> dict

Chains step 2 (simulate) → step 3 (feature summary) → step 4-5 (scalers,
PCA) → step 6 (PCA diagnostic) → step 7 (parameter scaler) → step 8
(SNPE-C training) into a single call. Each step is also available
individually from ``inference.snpe`` / ``inference.training_data`` so
notebook users can break the call apart.

Returned dict keys
------------------
- posterior, embedding_net
- theta_scaled, theta_raw, fc_raw, fcd_raw, x_input
- param_scaler, feature_pipeline, prior_scaled
- pca_diagnostic
"""
import config
from inference.snpe import (
    step4_fit_feature_scalers,
    step5_fit_feature_pipeline,
    step6_pca_diagnostic,
    step7_fit_param_scaler,
    step8_train_snpe,
)
from inference.training_data import (
    step2_simulate_train,
    step3_summary_features,
)


def run_stage1_snpe(train_subjects, subject_data, n_sim=None,
                    apply_bw=True, verbose=True):
    """End-to-end Stage 1 inference (steps 2 - 8)."""
    n_sim = n_sim or config.N_SIM

    if verbose:
        print("\n" + "=" * 65)
        print("  Stage 1: single-round SNPE-C")
        print(f"  Params: {config.STAGE1_PARAMS}, n_sim={n_sim}/subject")
        print("=" * 65)

    # Step 7: parameter scaler (needed before simulation)
    param_scaler, prior_scaled = step7_fit_param_scaler(verbose=verbose)

    # Step 2 + 3: simulate and extract features
    theta_s, theta_r, fc_raw, fcd_raw = step2_simulate_train(
        train_subjects, subject_data, prior_scaled, param_scaler,
        n_sim=n_sim, apply_bw=apply_bw, verbose=verbose,
    )
    step3_summary_features(fc_raw, fcd_raw, verbose=verbose)

    # Step 4 + 5: scale and PCA
    step4_fit_feature_scalers(fc_raw, fcd_raw, verbose=verbose)
    pipeline, x_input = step5_fit_feature_pipeline(
        fc_raw, fcd_raw, verbose=verbose,
    )

    # Step 6: PCA diagnostic
    pca_diag = step6_pca_diagnostic(
        pipeline, fc_raw, fcd_raw, verbose=verbose,
    )

    # Step 8: train SNPE-C with jointly-optimized MLP embedding
    posterior, embedding_net = step8_train_snpe(
        theta_s, x_input, prior_scaled, verbose=verbose,
    )

    return {
        "posterior": posterior,
        "embedding_net": embedding_net,
        "theta_scaled": theta_s,
        "theta_raw": theta_r,
        "fc_raw": fc_raw,
        "fcd_raw": fcd_raw,
        "x_input": x_input,
        "param_scaler": param_scaler,
        "feature_pipeline": pipeline,
        "prior_scaled": prior_scaled,
        "pca_diagnostic": pca_diag,
    }
