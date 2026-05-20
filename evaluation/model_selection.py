"""Model selection from validation aggregates.

Public API
----------
- compute_selection_score(val_agg, baseline_agg=None) -> float
- select_best_model(stage1_agg, stage2_agg=None, baseline_agg=None)
       -> (best_stage_int, scores_dict)

USE_FCD handling
----------------
When ``config.USE_FCD`` is False (the default for this project, since
empirical BOLD is currently unavailable), the FCD term is **excluded
from the selection score**. Mixing zero-valued FCD RMSEs from both
stages would still tie the metric to noise, so we drop the term
entirely. The selection table also hides FCD rows in this case.

Selection score formula
-----------------------
With baseline (recommended)::

    fc_corr_norm  = val_agg.fc_corr  - baseline.fc_corr
    fc_rmse_norm  = (baseline.fc_rmse - val_agg.fc_rmse) / baseline.fc_rmse
    fcd_rmse_norm = (baseline.fcd_rmse - val_agg.fcd_rmse) / baseline.fcd_rmse
                                                          [only if USE_FCD]
    score = W_FC_CORR * fc_corr_norm
          + W_FC_RMSE * fc_rmse_norm
          + W_FCD_RMSE * fcd_rmse_norm   [only if USE_FCD]

Without baseline::

    score = W_FC_CORR * fc_corr - W_FC_RMSE * fc_rmse
                                - W_FCD_RMSE * fcd_rmse   [only if USE_FCD]
"""
import numpy as np

import config


# ---------------------------------------------------------------------------
# Selection score
# ---------------------------------------------------------------------------

def compute_selection_score(val_agg, baseline_agg=None):
    """Weighted validation score. FCD term excluded when USE_FCD=False.

    The score is computed in the same direction for all metrics:
    higher = better. FC correlation is positive-better; FC/FCD RMSE
    are negative-better, so we flip their signs.
    """
    use_fcd = bool(getattr(config, "USE_FCD", True))

    if baseline_agg is not None:
        fc_corr_norm = (
            val_agg["fc_corr_mean"] - baseline_agg["fc_corr_mean"]
        )
        fc_rmse_norm = (
            (baseline_agg["fc_rmse_mean"] - val_agg["fc_rmse_mean"])
            / max(baseline_agg["fc_rmse_mean"], 1e-8)
        )
        if use_fcd:
            fcd_rmse_norm = (
                (baseline_agg["fcd_rmse_mean"] - val_agg["fcd_rmse_mean"])
                / max(baseline_agg["fcd_rmse_mean"], 1e-8)
            )
        else:
            fcd_rmse_norm = 0.0
    else:
        fc_corr_norm = val_agg["fc_corr_mean"]
        fc_rmse_norm = -val_agg["fc_rmse_mean"]
        fcd_rmse_norm = (
            -val_agg["fcd_rmse_mean"] if use_fcd else 0.0
        )

    score = (
        config.SELECT_W_FC_CORR * fc_corr_norm
        + config.SELECT_W_FC_RMSE * fc_rmse_norm
    )
    if use_fcd:
        score += config.SELECT_W_FCD_RMSE * fcd_rmse_norm
    return float(score)


# ---------------------------------------------------------------------------
# Best-model picker
# ---------------------------------------------------------------------------

def select_best_model(stage1_agg, stage2_agg=None, baseline_agg=None,
                      verbose=True):
    """Choose Stage 1 vs Stage 2 by validation selection score.

    Returns
    -------
    best : int        1 or 2
    scores : dict     {"stage1": float, "stage2": float | None}
    """
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
        if not getattr(config, "USE_FCD", True):
            print(
                "  [note] USE_FCD=False — FCD term excluded from score."
            )

    if stage2_agg is None:
        best = 1
    else:
        best = 2 if score_2 > score_1 else 1

    if verbose:
        print(f"\n  => Selected: Stage {best}")
        _print_selection_table(stage1_agg, stage2_agg)
    scores = {
        "stage1": score_1,
        "stage2": float(score_2) if stage2_agg is not None else None,
    }
    return best, scores


def _print_selection_table(stage1_agg, stage2_agg):
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n  +----------------+----------+----------+----------+")
    print("  |                |  Stage 1 |  Stage 2 |  delta   |")
    print("  +----------------+----------+----------+----------+")
    if stage2_agg is not None:
        d_corr = stage2_agg["fc_corr_mean"] - stage1_agg["fc_corr_mean"]
        d_rmse = stage2_agg["fc_rmse_mean"] - stage1_agg["fc_rmse_mean"]
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
        if use_fcd:
            d_fcd = (
                stage2_agg["fcd_rmse_mean"]
                - stage1_agg["fcd_rmse_mean"]
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
