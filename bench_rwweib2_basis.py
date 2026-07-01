#!/usr/bin/env python
"""Per-simulation timing for the basis_regionwise + rwweib2 (360 ROI) pipeline,
matching the REAL main_HCP config (CAB-NP 381 SC -> 360 cortical, T_END=180s /
T_CUT=60s, DT=1ms, TR=0.72s). Measures both delays-OFF and delays-ON so you see
the delay tax.

cuBNM runs a whole BATCH in parallel, so "time per sim" = wall / n_sims and DROPS
as the batch grows until the GPU saturates. Sim cost depends on N_REGIONS,
T_END/DT and (delays) history lookups — NOT on parameter/SC VALUES — so a decoded
random theta batch times like a real Step-2 batch.

The first GPU call inits the cuBNM session (~1 s) + JIT; a warmup excludes it.

Usage
-----
    python bench_rwweib2_basis.py                      # ON+OFF, default sizes, real cabnp SC
    python bench_rwweib2_basis.py --sizes 2000 --repeat 3
    python bench_rwweib2_basis.py --delays on          # only delays ON
    python bench_rwweib2_basis.py --synthetic          # random SC (no data load)
    python bench_rwweib2_basis.py --extrapolate 2000   # ETA/round from largest batch
"""
import argparse
import os
import time

import numpy as np

from pipeline_setup import PipelineConfig, setup_pipeline

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")

# ── production config (mirror main_HCP.py: CAB-NP 381 -> 360 cortical) ────────
_cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat",
    SC_FILE=os.environ.get("SC_FILE", "HCP_CABNP381_SC_first100.mat"),
    N_REGIONS=360, N_SUBJECTS=100, N_TRAIN=70, N_VAL=10, N_TEST=20, SEED=42,
    N_SIM=2_000, GPU_BATCH=2_000,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
)
setup_pipeline(_cfg, print_summary=False)

import config
config.INFERENCE_MODEL = "rwweib2"
config.PARAMETER_MODE = "basis_regionwise"
config.N_REGIONS = 360
config.HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
config.BASIS_BOUNDS = {"g_LRE": (0.0, 3.0), "g_FFI": (0.0, 3.0),
                       "I_o": (0.0, 1.0), "sigma": (0.0, 0.05)}
config.HETERO_BOUNDS = dict(config.BASIS_BOUNDS)
config.BASIS_COEFF_PRIOR = (-2.0, 2.0)
config.BASIS_PATH = os.environ.get("BASIS_PATH", "basis.npy")
config.BASIS_REZSCORE = True
config.RWWEIB2_FIXED = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
                        "J_N": 0.15, "lambda_IE": 1.0}
config.SC_DATASET = os.environ.get("SC_DATASET", "cabnp381")
config.VELOCITY_M_PER_S = 3.0
config.USE_FCD = False

DURATION_S = float(os.environ.get("DURATION_S", config.T_END / 1000.0))
BURN_IN_S = float(os.environ.get("BURN_IN_S", config.T_CUT / 1000.0))
TR_S = float(config.TR_SEC)
DT_MS = float(config.DT)


def _load_sc_delays(synthetic):
    """Return (SC 360x360, delays 360x360) — real CAB-NP subject 0, or synthetic."""
    if not synthetic:
        import data_loader_hcp as dl
        df, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi = dl.load_raw_data()
        sid = dl.get_target_subjects(df, fc_ids, sc_ids)[0]
        d = dl.load_all_subjects([sid], fc_mat, sc_mat, fc_ids, sc_ids, bm, bi)[sid]
        print(f"  SC: real CAB-NP subject {sid}  delay[max]={d['delays'].max():.1f}ms")
        return (np.ascontiguousarray(d["sc"].astype(np.float64)),
                np.ascontiguousarray(d["delays"].astype(np.float64)))
    n = config.N_REGIONS
    rng = np.random.RandomState(7)
    W = np.abs(rng.randn(n, n)); W = (W + W.T) / 2.0; np.fill_diagonal(W, 0.0); W /= W.max()
    L = rng.uniform(10, 100, (n, n)); L = (L + L.T) / 2.0; np.fill_diagonal(L, 0.0)
    print("  SC: synthetic random (delays ~ uniform 10-100ms)")
    return np.ascontiguousarray(W), np.ascontiguousarray(L / config.VELOCITY_M_PER_S)


def _overrides(n_sims):
    from basis_decoder import get_decoder
    from param_decoder import decode_to_param_maps, make_fixed_overrides_from_param_maps
    dec = get_decoder(config)
    theta = np.random.RandomState(0).uniform(-2.0, 2.0, (n_sims, dec.theta_dim))
    maps = decode_to_param_maps(theta, None, config, n_regions=config.N_REGIONS)
    return make_fixed_overrides_from_param_maps(maps)


def _time_batch(run, W, delays, n_sims, force_gpu):
    ov = _overrides(n_sims)
    t0 = time.perf_counter()
    bolds = run(W, np.zeros((n_sims, 0)), [], duration_s=DURATION_S, tr_s=TR_S,
                dt_ms=DT_MS, burn_in_s=BURN_IN_S, fixed=ov, force_gpu=force_gpu,
                hrf="bw", sc_dist=delays, velocity=(1.0 if delays is not None else None))
    wall = time.perf_counter() - t0
    assert len(bolds) == n_sims and np.isfinite(np.asarray(bolds[0])).all()
    return wall


def _sweep(run, W, delays, sizes, repeat, force_gpu, tag):
    print(f"\n  === delays {tag} ===")
    print(f"  {'n_sim':>6} | {'wall(s)':>9} | {'per-sim(ms)':>12} | {'sims/s':>8}")
    print("  " + "-" * 46)
    rows = []
    for n in sizes:
        best = min(_time_batch(run, W, delays, n, force_gpu) for _ in range(repeat))
        rows.append((n, best))
        print(f"  {n:>6} | {best:>9.3f} | {best/n*1e3:>12.3f} | {n/best:>8.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,10,100,1000,2000")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--delays", choices=["both", "on", "off"], default="both")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--extrapolate", type=int, default=0)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    from cuBNM.runner_rwweib_2cpl import run_cubnm_rwweib2_batch as run
    try:
        import torch; has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    force_gpu = (not args.cpu) and has_gpu

    print("=" * 60)
    print("  bench rwweib2 + basis_regionwise (CAB-NP 360, per-sim timing)")
    print("=" * 60)
    print(f"  device   : {'GPU' if force_gpu else 'CPU'}  (cuda={has_gpu})")
    print(f"  duration : {DURATION_S}s  burn_in: {BURN_IN_S}s  TR:{TR_S}s  DT:{DT_MS}ms")
    W, delays = _load_sc_delays(args.synthetic)

    # warmup (excluded)
    print("  warmup ...")
    _time_batch(run, W, None, 1, force_gpu)

    results = {}
    if args.delays in ("both", "off"):
        results["off"] = _sweep(run, W, None, sizes, args.repeat, force_gpu, "OFF")
    if args.delays in ("both", "on"):
        results["on"] = _sweep(run, W, delays, sizes, args.repeat, force_gpu, "ON")

    if "off" in results and "on" in results:
        print("\n  === delay tax (ON/OFF per-sim) ===")
        for (n, woff), (_, won) in zip(results["off"], results["on"]):
            print(f"    n={n:>5}: {won/woff:.1f}x   (OFF {woff/n*1e3:.2f}ms -> ON {won/n*1e3:.2f}ms per sim)")

    if args.extrapolate:
        key = "on" if "on" in results else "off"
        n_big, wall_big = max(results[key], key=lambda r: r[0])
        per = wall_big / n_big
        eta = per * args.extrapolate
        print(f"\n  extrapolate (delays {key}, projection): {args.extrapolate} sims @ "
              f"{per*1e3:.2f}ms/sim -> {eta:.0f}s ({eta/60:.1f}min, {eta/3600:.2f}h)")
        print(f"  full run = N_TRAIN x N_SIM sims (e.g. 70x2000=140k -> "
              f"{per*140000/3600:.1f}h at this per-sim)")
    print("\n  DONE")


if __name__ == "__main__":
    main()
