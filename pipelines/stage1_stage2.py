"""Stage 1 SNPE-C pipeline driver (WC-EIB).

End-to-end orchestration: data → Stage 1 → validation → final test on
the held-out test set → save + summary.

Stage 2 is currently undefined for the WC-EIB pipeline. The driver
runs Stage 1 only; ``best_stage`` is fixed to 1 until a Stage 2 design
is reintroduced.

Public API
----------
- run_pipeline(n_sim=None, verbose=True) -> artifacts dict

Rules enforced here
-------------------
- Train  : SBI training simulations only.
- Val    : Stage 1 validation only. Validation subjects never enter
           training.
- Test   : final evaluation on the held-out test set only. Test
           subjects never enter training or validation.
- ParameterScaler maps raw ↔ [-1, 1]. SBI trains in scaled space,
  VBI simulation receives raw parameters.
- USE_FCD is consulted by ``evaluation.model_selection``; FCD is
  excluded from the selection score when False (default for this
  project, since empirical BOLD is not yet available).
"""
import os
import warnings

import numpy as np

import config
import data_loader
import evaluation as evaluate
import inference

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Step 1-3 — Data loading & split
# ---------------------------------------------------------------------------

def step_data_split(verbose=True):
    """Load raw data, pick target subjects, do 4:2:2 split, bundle dicts."""
    if verbose:
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

def stage1_pipeline(train, subject_data, n_sim=None, verbose=True):
    """Run Stage 1: simulate, fit feature pipeline + scaler, train SNPE."""
    if verbose:
        print("\n" + "=" * 70)
        print("  Stage 1: simulation -> features -> SNPE-C")
        print("=" * 70)
    n_sim = n_sim if n_sim is not None else config.N_SIM
    return inference.run_stage1_snpe(
        train_subjects=train, subject_data=subject_data,
        n_sim=n_sim, apply_bw=True, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Step 7 — Stage 1 validation
# ---------------------------------------------------------------------------

def stage1_validation(val_subjects, subject_data, stage1, verbose=True):
    """Evaluate Stage 1 posterior on validation subjects."""
    if verbose:
        print("\n" + "=" * 70)
        print("  Stage 1 validation")
        print("=" * 70)
    val_results, val_agg = evaluate.evaluate_validation_stage1(
        val_subjects, subject_data, stage1,
        apply_bw=True, verbose=verbose,
    )
    return val_results, val_agg


# ---------------------------------------------------------------------------
# Step 12 — Final test (Stage 1 only)
# ---------------------------------------------------------------------------

def run_final_test(test_subjects, subject_data, stage1, verbose=True):
    if verbose:
        print("\n" + "=" * 70)
        print("  Final test on test set (Stage 1)")
        print("=" * 70)
    return evaluate.final_test(
        test_subjects=test_subjects, subject_data=subject_data,
        best_stage=1, stage1_result=stage1,
        stage2_result=None, n_resim=config.N_TEST_RESIM,
        apply_bw=True, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Step 13 — Save & summary
# ---------------------------------------------------------------------------

def save_and_summarize(stage1, stage1_agg, test_summary,
                       train_subjects, verbose=True):
    save_path = os.path.join(config.OUTPUT_DIR, "pipeline_artifacts.pkl")
    inference.save_artifacts(
        save_path,
        stage1=stage1,
        stage1_val_agg=stage1_agg,
        best_stage=1,
        test_summary=test_summary,
    )
    if verbose:
        print(f"\n  saved: {save_path}")
    evaluate.print_final_summary(
        stage1_agg=stage1_agg, stage2_agg=None,
        best_stage=1, test_summary=test_summary,
        train_subjects=train_subjects, n_train_sim=config.N_SIM,
    )


# ---------------------------------------------------------------------------
# Top-level pipeline driver
# ---------------------------------------------------------------------------

def run_pipeline(n_sim=None, verbose=True):
    """End-to-end Stage 1 SNPE-C pipeline (WC-EIB).

    Parameters
    ----------
    n_sim : int or None
        Per-subject Stage 1 simulation count. Defaults to ``config.N_SIM``.
    verbose : bool

    Returns
    -------
    artifacts : dict
        Keys: train, val, test, subject_data, stage1,
        stage1_val_agg, best_stage, test_summary.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.random.seed(config.SEED)
    try:
        import torch
        torch.manual_seed(config.SEED)
    except ImportError:
        pass

    if verbose:
        config.print_config()

    # ── data ──
    train, val, test, subject_data = step_data_split(verbose=verbose)

    # ── Stage 1 ──
    stage1 = stage1_pipeline(
        train, subject_data, n_sim=n_sim, verbose=verbose,
    )
    s1_val_results, s1_val_agg = stage1_validation(
        val, subject_data, stage1, verbose=verbose,
    )

    # ── Final test (Stage 1 only) ──
    test_summary = run_final_test(
        test, subject_data, stage1, verbose=verbose,
    )

    # ── Save & summary ──
    save_and_summarize(
        stage1, s1_val_agg, test_summary,
        train_subjects=train, verbose=verbose,
    )

    return {
        "train": train, "val": val, "test": test,
        "subject_data": subject_data,
        "stage1": stage1,
        "stage1_val_agg": s1_val_agg,
        "best_stage": 1,
        "test_summary": test_summary,
    }
