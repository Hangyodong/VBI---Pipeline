"""Evaluation package: metrics, validation, model selection, final test,
plots, reports.

Submodules
----------
- evaluation.metrics          : fc_metrics, fcd_vec_rmse, bootstrap_ci,
                                evaluate_subject, baseline_eval,
                                baseline_eval_subjects,
                                _resimulate_and_score
- evaluation.validation       : evaluate_validation_stage1/2
                                (return (results, agg) tuple)
- evaluation.model_selection  : compute_selection_score,
                                select_best_model
- evaluation.final_test       : final_test
- evaluation.plots            : plot_posteriors, plot_fc_comparison,
                                plot_sbc_rank_histogram,
                                plot_pca_diagnostic, plot_one_simulation,
                                plot_*_two_stage
- evaluation.reports          : print_summary_two_stage,
                                evaluate_all_two_stage,
                                print_final_summary,
                                report_step1 ... report_step14

All public names are re-exported at the package level. The thin
``evaluate.py`` module re-exports them too so legacy callers
(``from evaluate import ...`` / ``evaluate.X``) continue to work.
"""
# metrics
from evaluation.metrics import (
    _progress,
    _resimulate_and_score,
    baseline_eval,
    baseline_eval_subjects,
    bootstrap_ci,
    evaluate_subject,
    fc_metrics,
    fcd_summary_rmse,
    fcd_vec_rmse,
)

# validation
from evaluation.validation import (
    _aggregate_validation,
    _print_validation_summary,
    evaluate_validation_stage1,
    evaluate_validation_stage2,
)

# model selection
from evaluation.model_selection import (
    _print_selection_table,
    compute_selection_score,
    select_best_model,
)

# final test
from evaluation.final_test import (
    _print_test_summary,
    _test_stage1,
    _test_stage2,
    final_test,
)

# plots
from evaluation.plots import (
    plot_fc_comparison,
    plot_fc_comparison_two_stage,
    plot_one_simulation,
    plot_pca_diagnostic,
    plot_posteriors,
    plot_posteriors_two_stage,
    plot_sbc_rank_histogram,
)

# reports
from evaluation.reports import (
    evaluate_all_two_stage,
    print_final_summary,
    print_summary_two_stage,
    report_step1,
    report_step2,
    report_step3,
    report_step4,
    report_step5,
    report_step6,
    report_step7,
    report_step8,
    report_step9,
    report_step10,
    report_step11,
    report_step12,
    report_step13,
    report_step14,
)


__all__ = [
    # metrics
    "fc_metrics", "fcd_vec_rmse", "fcd_summary_rmse", "bootstrap_ci",
    "evaluate_subject", "baseline_eval", "baseline_eval_subjects",
    # validation
    "evaluate_validation_stage1", "evaluate_validation_stage2",
    # model selection
    "compute_selection_score", "select_best_model",
    # final test
    "final_test",
    # plots
    "plot_posteriors", "plot_fc_comparison",
    "plot_posteriors_two_stage", "plot_fc_comparison_two_stage",
    "plot_sbc_rank_histogram", "plot_pca_diagnostic",
    "plot_one_simulation",
    # reports
    "print_summary_two_stage", "evaluate_all_two_stage",
    "print_final_summary",
    "report_step1", "report_step2", "report_step3", "report_step4",
    "report_step5", "report_step6", "report_step7", "report_step8",
    "report_step9", "report_step10", "report_step11", "report_step12",
    "report_step13", "report_step14",
]
