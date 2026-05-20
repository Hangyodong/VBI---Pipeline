"""Inference package: scaling, priors, feature pipeline, SNPE-C training,
two-stage flow, posterior diagnostics, and persistence.

Submodules
----------
- inference.scaling           : ParameterScaler, make_stage*_param_scaler
- inference.priors            : make_scaled_prior
- inference.feature_pipeline  : FamilyScaler, FCPCAScaler, FeaturePipeline
- inference.embedding         : FeatureEmbedding (MLP)
- inference.training_data     : collect_training_data, step2/3,
                                save/load_extracted_features
- inference.snpe              : train_snpe, step4/5/6/7/8
- inference.stage1            : run_stage1_snpe
- inference.stage2            : run_stage2_snpe, build_stage2_param_set,
                                select_theta_bad, select_difficult_params
- inference.posterior         : transform_observed, infer_subject_raw,
                                compute_shrinkage_*, posterior_correlation,
                                posterior_predictive_check
- inference.diagnostics       : simulation_based_calibration,
                                evaluate_embedding_probing
- inference.io                : save_artifacts, load_artifacts

All public names are re-exported at the package level. The old
``inference.py`` module is removed: this package replaces it, and
imports like ``from inference import ParameterScaler`` continue to work.
"""
# --- _utils (internal, but re-exported for legacy callers) ----------------
from inference._utils import _progress

# --- scaling --------------------------------------------------------------
from inference.scaling import (
    ParameterScaler,
    make_stage1_param_scaler,
    make_stage2_param_scaler,
)

# --- priors ---------------------------------------------------------------
from inference.priors import make_scaled_prior

# --- feature pipeline -----------------------------------------------------
from inference.feature_pipeline import (
    FamilyScaler,
    FCPCAScaler,
    FeaturePipeline,
)

# --- embedding ------------------------------------------------------------
from inference.embedding import FeatureEmbedding

# --- training data --------------------------------------------------------
from inference.training_data import (
    _drain_one_future,
    collect_training_data,
    load_extracted_features,
    save_extracted_features,
    step2_simulate_train,
    step3_summary_features,
)

# --- snpe / step 4-8 ------------------------------------------------------
from inference.snpe import (
    _print_pca_diagnostic,
    step4_fit_feature_scalers,
    step5_fit_feature_pipeline,
    step6_pca_diagnostic,
    step7_fit_param_scaler,
    step8_train_snpe,
    train_snpe,
)

# --- stage drivers --------------------------------------------------------
from inference.stage1 import run_stage1_snpe
from inference.stage2 import (
    _build_nuisance_array,
    build_stage2_param_set,
    collect_stage2_data,
    run_stage2_snpe,
    select_difficult_params,
    select_theta_bad,
)

# --- posterior ------------------------------------------------------------
from inference.posterior import (
    compute_shrinkage_raw,
    compute_shrinkage_scaled,
    infer_subject_raw,
    posterior_correlation,
    posterior_predictive_check,
    transform_observed,
)

# --- diagnostics ----------------------------------------------------------
from inference.diagnostics import (
    evaluate_embedding_probing,
    simulation_based_calibration,
)

# --- io -------------------------------------------------------------------
from inference.io import load_artifacts, save_artifacts


__all__ = [
    # scaling
    "ParameterScaler", "make_stage1_param_scaler",
    "make_stage2_param_scaler",
    # priors
    "make_scaled_prior",
    # feature pipeline
    "FamilyScaler", "FCPCAScaler", "FeaturePipeline",
    # embedding
    "FeatureEmbedding",
    # training data
    "collect_training_data",
    "step2_simulate_train", "step3_summary_features",
    "save_extracted_features", "load_extracted_features",
    # snpe / steps
    "train_snpe",
    "step4_fit_feature_scalers", "step5_fit_feature_pipeline",
    "step6_pca_diagnostic", "step7_fit_param_scaler",
    "step8_train_snpe",
    # stage drivers
    "run_stage1_snpe",
    "run_stage2_snpe",
    "build_stage2_param_set",
    "select_theta_bad", "select_difficult_params",
    "collect_stage2_data",
    # posterior
    "transform_observed", "infer_subject_raw",
    "compute_shrinkage_scaled", "compute_shrinkage_raw",
    "posterior_correlation", "posterior_predictive_check",
    # diagnostics
    "simulation_based_calibration", "evaluate_embedding_probing",
    # io
    "save_artifacts", "load_artifacts",
]
