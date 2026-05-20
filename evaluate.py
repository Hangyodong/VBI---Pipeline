"""Compatibility wrapper — public API moved to ``evaluation/`` package.

This file no longer contains any logic. It exists solely so that
existing imports like::

    from evaluate import evaluate_validation_stage1, select_best_model
    from evaluate import final_test, print_final_summary
    import evaluate
    evaluate.plot_posteriors_two_stage(...)

continue to work after the Phase 4 refactor.

New code should import from the source submodules:

    from evaluation.metrics          import fc_metrics, evaluate_subject
    from evaluation.validation       import evaluate_validation_stage1
    from evaluation.model_selection  import select_best_model
    from evaluation.final_test       import final_test
    from evaluation.plots            import plot_posteriors
    from evaluation.reports          import print_final_summary, report_step1
"""

# Re-export every public name from the evaluation package.
from evaluation import *                          # noqa: F401, F403
from evaluation import (                          # noqa: F401
    # Internal helpers that some legacy callers reach into directly.
    _aggregate_validation,
    _print_selection_table,
    _print_test_summary,
    _print_validation_summary,
    _progress,
    _resimulate_and_score,
    _test_stage1,
    _test_stage2,
)
