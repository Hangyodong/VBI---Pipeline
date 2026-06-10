#!/usr/bin/env python
"""Benchmark HCP RWW-EIB-FFI cuBNM sim time — measure + extrapolate full run.

Mirrors main_HCP.py's config (TR=0.72, DT=1ms, 5min/1min cut, 381 region,
rwweib engine, delays OFF) and times one GPU batch of ``--samples`` sims on a
real subject's SC. The first GPU call initializes the cuBNM session (~1.5 s);
it is excluded via a warmup call so the reported numbers are steady-state.

Sim cost depends on N_REGIONS, T_END/DT and (for delays) history lookups —
NOT on parameter VALUES — so a midpoint theta replicated to fill the batch
times identically to a real Step-2 batch.

Usage
-----
    python bench_hcp.py                      # 2000 samples, 1 subject, extrapolate to N_TRAIN
    python bench_hcp.py --samples 2000 --repeat 3
    python bench_hcp.py --samples 1000,2000,4000
"""
import argparse
import time

import numpy as np

from pipeline_setup import PipelineConfig, setup_pipeline


def _build_config(n_sim):
    """Same PipelineConfig + post-overrides as main_HCP.py (Setup cell)."""
    cfg = PipelineConfig(
        DATA_DIR   = "/scratch/home/wog3597/vbi",
        OUTPUT_DIR = "./output_hcp",
        FC_FILE    = "HCP_FC.mat",
        SC_FILE    = "HCP_SC.mat",
        N_REGIONS  = 381,
        N_SUBJECTS = 100,
        N_TRAIN    = 80,
        N_VAL      = 0,
        N_TEST     = 20,
        SEED       = 42,
        N_SIM      = n_sim,
        GPU_BATCH  = n_sim,           # one grp.run per batch
        T_END_MS   = 300_000.0,       # 5 min
        T_CUT_MS   =  60_000.0,       # cut first 1 min
        DT         = 1.0,             # RWW Euler step (ms)
        DECIMATE   = 720,             # neural stored dt = 720 ms
        TR_SEC     = 0.72,            # BOLD sampling period (s)
    )
    setup_pipeline(cfg, print_summary=False)

    import config
    # RWW-EIB-FFI overrides (identical to main_HCP.py)
    config.INFERENCE_MODEL    = "rwweib"
    config.STAGE1_PARAMS      = ["g_LRE", "g_FFI", "sigma", "I_o"]
    config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS
    config.STAGE1_PRIOR_LOW   = [0.0, 0.0, 0.0,  0.30]
    config.STAGE1_PRIOR_HIGH  = [3.0, 3.0, 0.03, 0.45]
    config.VELOCITY_M_PER_S   = 3.0
    config.USE_FCD            = False
    config.USE_DELAYS         = False             # ~9x cost if enabled
    return cfg, config


def _midpoint_theta(n, config):
    """n identical mid-prior rows over STAGE1_PARAMS (value-independent cost)."""
    lo = np.asarray(config.STAGE1_PRIOR_LOW, dtype=np.float64)
    hi = np.asarray(config.STAGE1_PRIOR_HIGH, dtype=np.float64)
    row = (lo + hi) / 2.0
    return np.repeat(row[None, :], n, axis=0)


def _load_one_sc(config):
    """Load the first training subject's SC (+delays) via data_loader_hcp."""
    import data_loader_hcp as dl
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = dl.load_raw_data()
    subjects = dl.get_target_subjects(df, fc_ids, sc_ids)
    sid = subjects[0]
    data = dl.load_all_subjects(
        [sid], fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids
    )
    d = data[sid]
    return sid, d["sc"].astype(np.float32), d["delays"].astype(np.float32)


def _fmt(s):
    """Seconds -> human string."""
    if s < 90:
        return f"{s:.1f}s"
    if s < 5400:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="2000",
                    help="comma sample counts to sweep (each = one GPU batch)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat each timing and take the median")
    args = ap.parse_args()

    sample_counts = [int(x) for x in str(args.samples).split(",")]
    max_n = max(sample_counts)

    cfg, config = _build_config(max_n)
    from cuBNM.simulate_rwweib import simulate_gpu_batch

    sid, weights, delays = _load_one_sc(config)
    pn = list(config.STAGE1_PARAMS)

    print("=" * 66)
    print("  HCP RWW-EIB-FFI sim benchmark")
    print("=" * 66)
    print(f"  subject     : {sid}")
    print(f"  N_REGIONS   : {config.N_REGIONS}")
    print(f"  T_END/T_CUT : {config.T_END/1000:.0f}s / {config.T_CUT/1000:.0f}s"
          f"  (DT={config.DT}ms, {int(config.T_END/config.DT):,} steps)")
    print(f"  TR / BOLD T : {config.TR_SEC}s / {config.ANALYSIS_BOLD_T} TR")
    print(f"  delays      : {'ON' if config.USE_DELAYS else 'OFF'}")
    print(f"  params      : {pn}")
    print(f"  N_TRAIN     : {config.N_TRAIN} subjects")
    print("=" * 66)

    # Warmup — init GPU session (excluded from timing).
    print("  warmup (GPU session init) ...", flush=True)
    _ = simulate_gpu_batch(
        weights, _midpoint_theta(1, config), param_names=pn,
        delays=delays, apply_bw=True, label=None,
    )

    for n in sample_counts:
        theta = _midpoint_theta(n, config)
        ts = []
        for _ in range(args.repeat):
            t0 = time.time()
            simulate_gpu_batch(
                weights, theta, param_names=pn,
                delays=delays, apply_bw=True, label=None,
            )
            ts.append(time.time() - t0)
        per_batch = float(np.median(ts))
        per_sim = per_batch / n

        # Per subject: real N_SIM split into ceil(N_SIM/GPU_BATCH) batches.
        n_batches_real = -(-config.N_SIM // config.GPU_BATCH)
        per_subject = per_batch * (config.N_SIM / n)  # scale this n -> real N_SIM
        full_run = per_subject * config.N_TRAIN

        print()
        print(f"  [{n:,} sims/batch]  repeat={args.repeat}")
        print(f"    per batch     : {_fmt(per_batch)}   "
              f"(raw: {', '.join(f'{t:.1f}' for t in ts)})")
        print(f"    per sim       : {per_sim*1000:.2f} ms   "
              f"({1/per_sim:,.0f} sims/s)")
        print(f"    -> per subject (N_SIM={config.N_SIM:,}, "
              f"{n_batches_real} batch): {_fmt(per_subject)}")
        print(f"    -> full Step 2 (x{config.N_TRAIN} train): {_fmt(full_run)}")

    print()
    print("  note: GPU sim time only. Feature extraction (FC) runs on the CPU")
    print("        pool in parallel and overlaps, so real Step 2 wall-clock is")
    print("        ~this or slightly higher, not the sum.")


if __name__ == "__main__":
    main()
