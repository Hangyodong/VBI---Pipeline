"""main.py — Mouse MPTP VBI-SBI pipeline entry point.

This file is intentionally **thin**. All orchestration lives in
``pipelines.stage1_stage2``; ``main.py`` only:

  1. parses the small set of command-line / environment knobs
  2. calls ``pipelines.run_pipeline(...)``

For interactive use, prefer ``main.ipynb`` (the notebook re-runs each
step in its own cell and produces the figures + reports inline).

Usage
-----
    python main.py                        # full run with config defaults
    N_SIM=5000 python main.py             # override n_sim from env
    RUN_STAGE2=0 python main.py           # skip Stage 2 entirely

Pipeline (production)
---------------------
 1. Load raw data           (FC / SC / tract length / participants.tsv)
 2. Train / val / test split (4:2:2)
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
"""
import os

from pipelines import run_pipeline


def _env_int(key, default):
    v = os.environ.get(key)
    return int(v) if v is not None and v != "" else default


def _env_bool(key, default):
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return v.lower() not in ("0", "false", "no", "off")


def _env_float(key, default):
    v = os.environ.get(key)
    return float(v) if v is not None and v != "" else default


def main():
    """Parse env knobs, hand off to pipelines.run_pipeline."""
    n_sim = _env_int("N_SIM", None)
    n_sim_s2 = _env_int("N_SIM_S2", None)
    run_stage2 = _env_bool("RUN_STAGE2", True)
    sens_threshold = _env_float("SENS_THRESHOLD", 0.5)
    shr_threshold = _env_float("SHR_THRESHOLD", 0.2)

    return run_pipeline(
        n_sim=n_sim,
        n_sim_s2=n_sim_s2,
        run_stage2=run_stage2,
        sens_threshold=sens_threshold,
        shr_threshold=shr_threshold,
        verbose=True,
    )


if __name__ == "__main__":
    main()
