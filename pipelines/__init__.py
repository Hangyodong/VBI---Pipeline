"""Pipeline drivers — full end-to-end run orchestration.

Submodules
----------
- pipelines.stage1_stage2 : the production two-stage SNPE-C pipeline
                             (Stage 1, θ_bad, optional Stage 2, model
                             selection on validation, final test on
                             test set, save + summary)

Each submodule exposes a ``run_pipeline(...)`` function that takes a
config-like object (or uses defaults from ``config``) and returns the
artifacts dict (or writes to ``config.OUTPUT_DIR``). ``main.py`` is a
thin entry point that just calls into here.
"""
from pipelines.stage1_stage2 import run_pipeline

__all__ = ["run_pipeline"]
