#!/usr/bin/env python
"""Benchmark cuBNM WC sim — ETA estimate + delays-ON/OFF cost.

Answers two things:
  1. ETA       — how long a full ``joint_opt.py --all`` run takes
                 (per-call time x group-subject count).
  2. DELAY TAX — how much conduction delays add vs running without them.

The sim cost depends on N_REGIONS, T_END/DT and (for delays) the history
buffer lookups — NOT on the parameter VALUES. So we replicate one BASELINE
theta row to fill the batch; timing is identical to a real joint_opt batch.

The first GPU call initializes the session (~1.5 s) — excluded via a warmup
call so the reported numbers are steady-state.

Usage
-----
    python bench_sim.py                       # default subject, 5000 samples
    python bench_sim.py --samples 100,1000,5000
    python bench_sim.py --subject sub-419077 --repeat 2
"""
import argparse
import time

import numpy as np

import config
from joint_opt import load_group, ALWAYS_THETA
from sweep_fc import BASELINE, FIXED_PARAMS


def _theta_batch(n, cols):
    """n identical BASELINE rows over the given theta columns."""
    row = [BASELINE[c] for c in cols]
    return np.asarray([row] * n, dtype=np.float64)


def _time_call(weights, cols, theta_batch, baseline_fixed, delays, repeat):
    """Median wall-time of `repeat` simulate_gpu_batch calls (s)."""
    from cuBNM.simulate import simulate_gpu_batch
    ts = []
    for _ in range(repeat):
        t0 = time.time()
        simulate_gpu_batch(
            weights, theta_batch, param_names=cols,
            delays=delays, apply_bw=True, fixed_overrides=baseline_fixed,
        )
        ts.append(time.time() - t0)
    return float(np.median(ts)), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None,
                    help="subject id to bench (default: first group subject)")
    ap.add_argument("--samples", default="5000",
                    help="comma sample counts to sweep (each <= GPU_BATCH)")
    ap.add_argument("--params", default="g_e,c_ei,c_ee",
                    help="searched params (only affects theta column count)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="timed calls per config (median taken)")
    args = ap.parse_args()

    sample_sizes = [int(s) for s in args.samples.split(",") if s]
    gpu_cap = int(getattr(config, "GPU_BATCH", 10_000))
    for s in sample_sizes:
        if s > gpu_cap:
            raise SystemExit(f"--samples {s} > GPU_BATCH {gpu_cap}")

    search = [p for p in args.params.split(",") if p]
    cols = ALWAYS_THETA + [p for p in search if p not in ALWAYS_THETA]
    baseline_fixed = {p: BASELINE[p] for p in FIXED_PARAMS if p not in search}

    # one subject for sc/delays
    subjects, data = load_group(args.subject)
    n_group = len(load_group(None)[0])      # full group size for ETA
    sid = subjects[0]
    d = data[sid]
    weights, delays = d["sc"], d["delays"]

    print("=" * 70)
    print(f"  BENCH — subject {sid}")
    print(f"  N_REGIONS={config.N_REGIONS}  T_END={config.T_END:.0f}ms  "
          f"DT={config.DT}ms  steps={int(config.T_END / config.DT):,}")
    print(f"  USE_DELAYS={getattr(config, 'USE_DELAYS', False)}  "
          f"theta cols={cols}  GPU_BATCH={gpu_cap}")
    print(f"  group size (for ETA)={n_group}")
    print("=" * 70)

    # warmup: pay the one-time GPU session init + per-path JIT compile
    # outside the timing. BOTH delays ON and OFF compile separate kernels,
    # so warm each path or the first timed call of each is inflated.
    print("\n  [warmup] GPU init + JIT compile (ON + OFF paths) ...")
    warm = _theta_batch(min(sample_sizes), cols)
    _time_call(weights, cols, warm, baseline_fixed, None, 1)    # OFF path
    _time_call(weights, cols, warm, baseline_fixed, delays, 1)  # ON path
    print("  [warmup] done\n")

    print(f"  {'samples':>8s}  {'OFF (s)':>9s}  {'ON (s)':>9s}  "
          f"{'tax x':>7s}  {'tax +s':>8s}")
    print("  " + "-" * 50)

    on_time_at_default = None
    for s in sample_sizes:
        tb = _theta_batch(s, cols)
        t_off, _ = _time_call(weights, cols, tb, baseline_fixed, None,
                              args.repeat)
        t_on, _ = _time_call(weights, cols, tb, baseline_fixed, delays,
                             args.repeat)
        tax = t_on / t_off if t_off > 0 else float("nan")
        print(f"  {s:8d}  {t_off:9.2f}  {t_on:9.2f}  {tax:7.2f}  "
              f"{t_on - t_off:+8.2f}")
        on_time_at_default = t_on  # last (largest) sample = real joint_opt cfg

    # ---- ETA projection for `joint_opt.py --all` -----------------------
    print("\n" + "=" * 70)
    print("  ETA — joint_opt.py --all (1 call/subject, sequential)")
    print("=" * 70)
    per = on_time_at_default
    print(f"  per-subject (delays ON, {sample_sizes[-1]} samples): "
          f"{per:.1f}s")
    total = per * n_group
    print(f"  group x{n_group} sequential (--workers 1): "
          f"{total:.0f}s = {total / 60:.1f} min")
    print(f"  + CPU FC scoring per subject (not GPU-timed here) adds a tail")
    for w in (2, 4, 8):
        if w <= n_group:
            print(f"  ~parallel --workers {w}: "
                  f"{total / w:.0f}s = {total / w / 60:.1f} min "
                  f"(ideal, ignores GPU/MPS contention)")
    print()


if __name__ == "__main__":
    main()
