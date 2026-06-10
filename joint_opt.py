#!/usr/bin/env python
"""Joint parameter optimization — true real-FC fit ceiling.

The one-at-a-time sweep (``sweep_fc.py``) misses interactions: a knob that
looks dead at one operating point (e.g. g_i) may matter when other params
move too. This script searches the *joint* space to answer two questions:

  1. CEILING   — what is the best real-FC corr the simulator can reach at
                 ALL, when several params move together?
  2. WHICH     — which parameter combination achieves it?

Single batched call per subject
-------------------------------
Every searched parameter is now routed through the theta columns
(``cuBNM.runner_vbi.build_param_lists`` applies any theta column whose name
maps via ``_FIXED_TO_VBI`` per-sim — this also fixed the old P/Q "dead
column" bug). So the whole joint search runs as ONE GPU batch of
``--samples`` random draws per subject. Non-searched params stay at
``BASELINE``.

NOTE: sample count is NOT free. The GPU saturates its SMs near ~100 sims,
so above that cost grows ~linearly with ``--samples`` (bench_sim.py,
delays ON: 100=39s, 1000=125s, 5000=541s/subject). delays ON costs ~9x
vs OFF. ``--all`` at 5000/workers 1 takes ~72 min. For a ceiling estimate
``--samples 1000`` is usually enough (4.3x faster) and ``--workers 8
--gpus 0,1`` parallelizes the per-subject calls.

cuBNM is deterministic for a fixed ``sim_seed``, so there is no noise to
average out — the single batched corr per sample IS the value (no repeats).

Group mode (``--all``)
----------------------
Runs the search for EVERY subject in ``config.GROUP_FILTER`` (ctr+MPTP).
Per-subject SC weights differ, so ``WCVBISimGroup`` (one SC per group)
cannot merge subjects into one call — the group floor is 1 call/subject.
``--workers 1`` runs them sequentially; ``--workers N`` runs N concurrent
processes that share the single GPU (small batches only — big batches
already fill the GPU and would OOM). ``--gpus 0,1`` pins workers onto
multiple GPUs.

Usage
-----
    python joint_opt.py --all                          # ctr+MPTP, g_e,g_i,c_ei
    python joint_opt.py --all --params g_e,c_ei,c_ee   # include the top knob
    python joint_opt.py --subject sub-419077 --params g_e,c_ei,c_ee,P
"""
import argparse
import os
import threading
import time

import numpy as np

import config
from sweep_fc import (
    BASELINE, THETA_PARAMS, FIXED_PARAMS, SWEEP_RANGE, _masked_corr,
)

# params that the cuBNM adapter always reads from the theta columns; we keep
# them present (at BASELINE if not searched) so the operating point is fixed.
ALWAYS_THETA = ["g_e", "g_i", "c_ei"]


# --------------------------------------------------------------------------
# Data loading (whole group, mat files read once)
# --------------------------------------------------------------------------

def load_group(subject=None):
    """Load SC/FC/delays for the ctr+MPTP group (config.GROUP_FILTER)."""
    import data_loader
    df, fc_mat, sc_mat, fc_ids, sc_ids, _, _ = data_loader.load_raw_data()
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    if subject is not None:
        if subject not in subjects:
            raise SystemExit(f"subject {subject} not in group {subjects}")
        subjects = [subject]
    data = {
        sid: data_loader.get_subject_data(sid, fc_mat, sc_mat, fc_ids, sc_ids)
        for sid in subjects
    }
    return subjects, data


# --------------------------------------------------------------------------
# Simulation (one batched call over arbitrary theta-routed params)
# --------------------------------------------------------------------------

def _sim(weights, param_names, theta_batch, baseline_fixed, delays):
    """One cuBNM batch -> list of FC matrices, one per theta row."""
    from cuBNM.simulate import simulate_gpu_batch
    from simulator import compute_fc
    bolds = simulate_gpu_batch(
        weights, theta_batch, param_names=param_names,
        delays=delays, apply_bw=True, fixed_overrides=baseline_fixed,
    )
    return [compute_fc(np.asarray(b)) for b in bolds]


# --------------------------------------------------------------------------
# Per-subject joint search
# --------------------------------------------------------------------------

def run_subject(sid, d, args, quiet=False):
    """Joint search for one subject. Returns a result dict and saves npz/png."""
    weights, delays = d["sc"], d["delays"]
    fc_real = d["fc"]
    N = fc_real.shape[0]
    iu = np.triu_indices(N, k=1)
    real_vec = fc_real[iu]
    nan_vec = (d["fc_nan"][iu] if "fc_nan" in d
               else np.zeros(iu[0].shape, dtype=bool))

    search = [p for p in args.params.split(",") if p]
    cols = ALWAYS_THETA + [p for p in search if p not in ALWAYS_THETA]
    # Non-searched fixed-route params held at BASELINE via fixed_overrides;
    # searched ones travel through the theta columns instead.
    baseline_fixed = {p: BASELINE[p] for p in FIXED_PARAMS if p not in search}

    rng = np.random.RandomState(args.seed)

    def _say(msg):
        if not quiet:
            print(msg)

    _say("=" * 70)
    _say(f"  Joint optimization — subject {sid}")
    _say(f"  search params : {search}")
    _say(f"  theta columns : {cols}")
    _say(f"  samples       : {args.samples}   GPU calls: 1   "
         f"(one batched sim-duration)")
    _say(f"  valid FC entries: {int((~nan_vec).sum())}/{len(nan_vec)}")
    _say("=" * 70)

    # Build one batch: searched params random in SWEEP_RANGE, rest at BASELINE.
    rows, samp_overrides = [], []
    for _ in range(args.samples):
        ov = {p: float(rng.uniform(*SWEEP_RANGE[p])) for p in search}
        vals = dict(BASELINE); vals.update(ov)
        rows.append([vals[c] for c in cols])
        samp_overrides.append(ov)
    theta_batch = np.asarray(rows, dtype=np.float64)

    t0 = time.time()
    fc_list = _sim(weights, cols, theta_batch, baseline_fixed, delays)
    corrs = np.array([
        _masked_corr(fc[iu], real_vec, nan_vec) for fc in fc_list
    ], dtype=float)
    corrs = np.where(np.isfinite(corrs), corrs, -1.0)
    _say(f"  batch done: best_corr={np.nanmax(corrs):+.4f}  "
         f"({time.time() - t0:.0f}s)")

    order = np.argsort(-corrs)
    _say(f"\n  Top {args.topk} candidates:")
    for r in order[:args.topk]:
        ps = "  ".join(f"{k}={samp_overrides[r].get(k, BASELINE[k]):.3f}"
                       for k in search)
        _say(f"    corr={corrs[r]:+.4f}   {ps}")

    best_i = int(order[0])
    best_corr = float(corrs[best_i])
    best_p = {k: samp_overrides[best_i].get(k, BASELINE[k]) for k in search}

    # ---- per-param influence on fit -------------------------------------
    param_mat = np.array([[samp_overrides[i].get(k, BASELINE[k])
                           for k in search] for i in range(len(corrs))])
    influence = {}
    for j, k in enumerate(search):
        col = param_mat[:, j]
        influence[k] = (abs(np.corrcoef(col, corrs)[0, 1])
                        if col.std() > 0 else float("nan"))

    # ---- save -----------------------------------------------------------
    out = f"{args.out}_{sid}"
    np.savez(
        out + ".npz",
        subject=sid, param_names=np.array(search),
        params=param_mat, corrs=corrs,
        best_params=np.array([best_p[k] for k in search]),
        best_corr=best_corr,
    )
    _say(f"\n  saved: {out}.npz")
    try:
        _plot(param_mat, corrs, search, sid, best_corr, out + ".png")
        _say(f"  saved: {out}.png")
    except Exception as e:  # noqa: BLE001
        _say(f"  [plot skipped] {type(e).__name__}: {e}")

    if not quiet:
        _verdict_one(sid, best_corr, best_p, search, influence)

    return {
        "subject": sid, "best_corr": best_corr,
        "best_params": best_p, "keys": search, "influence": influence,
    }


def _verdict_one(sid, best_corr, best_p, keys, influence):
    print("\n" + "=" * 70)
    print(f"  VERDICT — {sid}")
    print("=" * 70)
    print(f"  best joint real-FC corr = {best_corr:+.4f}")
    print("  at: " + "  ".join(f"{k}={best_p[k]:.3f}" for k in keys))
    print("\n  param influence on fit (|corr(value, fit)|):")
    for k in keys:
        print(f"    {k:6s}: {influence[k]:.3f}")
    print()
    if best_corr < 0.3:
        print("  => Ceiling LOW (<0.30). Param re-selection alone won't fix it.")
        print("     Simulator misspecification: revisit SC scaling, conduction")
        print("     delays, BOLD model, or WC structure.")
    elif best_corr < 0.5:
        print("  => Ceiling MODERATE. Param re-selection helps but a structural")
        print("     gap remains (likely delays / SC / noise).")
    else:
        print("  => Ceiling OK (>=0.50). Parameterization is the main problem —")
        print("     infer the influential params + fix the embedding.")


# --------------------------------------------------------------------------
# Parallel worker (one subject pinned to one GPU)
# --------------------------------------------------------------------------

def _worker(payload):
    """Process entry: pin GPU, run one subject. Returns result dict."""
    sid, d, args, gpu = payload
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        return run_subject(sid, d, args, quiet=True)
    except Exception as e:  # noqa: BLE001
        return {"subject": sid, "error": f"{type(e).__name__}: {e}"}


def _heartbeat(out_prefix, subjects, t0, stop, every):
    """Periodic same-terminal progress: elapsed + completed-npz count.

    Counts this-run npz (mtime >= t0) so the per-batch opaque GPU call still
    shows liveness even before the first subject finishes.
    """
    n = len(subjects)
    while not stop.wait(every):
        done = 0
        for s in subjects:
            p = f"{out_prefix}_{s}.npz"
            if os.path.exists(p) and os.path.getmtime(p) >= t0 - 1:
                done += 1
        el = time.time() - t0
        msg = f"  [heartbeat] {done}/{n} = {100 * done // n}%  elapsed {el:.0f}s"
        if done:
            msg += f"  eta {el / done * (n - done):.0f}s"
        print(msg, flush=True)


def _run_group(subjects, data, args):
    """Run all subjects, sequential or process-parallel, return result list."""
    gpus = [g for g in args.gpus.split(",") if g] if args.gpus else []
    n = len(subjects)
    t0 = time.time()

    # Background heartbeat (same terminal) — survives the long opaque batches.
    stop = threading.Event()
    hb = None
    if args.heartbeat > 0:
        hb = threading.Thread(
            target=_heartbeat,
            args=(args.out, subjects, t0, stop, args.heartbeat),
            daemon=True,
        )
        hb.start()

    results = []
    try:
        if args.workers <= 1:
            for i, sid in enumerate(subjects):
                print(f"\n########## subject {i + 1}/{n} ({100 * i // n}% done):"
                      f" {sid} ##########")
                results.append(run_subject(sid, data[sid], args))
                el = time.time() - t0
                eta = el / (i + 1) * (n - i - 1)
                print(f"  progress: {i + 1}/{n} = {100 * (i + 1) // n}%  "
                      f"(elapsed {el:.0f}s, eta {eta:.0f}s)")
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp

            payloads = [
                (sid, data[sid], args, (gpus[i % len(gpus)] if gpus else None))
                for i, sid in enumerate(subjects)
            ]
            mode = (f"pinned to gpus {gpus}" if gpus
                    else "sharing single GPU 0")
            print(f"\n  parallel: {n} subjects, {args.workers} workers {mode}")
            if not gpus and args.workers > 1:
                print("  (1 GPU shared by all workers — if OOM, lower --workers)")
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=ctx) as ex:
                for res in ex.map(_worker, payloads):
                    results.append(res)
                    k = len(results)
                    el = time.time() - t0
                    eta = el / k * (n - k)
                    tag = res.get("error", f"corr={res.get('best_corr'):+.4f}")
                    print(f"  [done {k}/{n} = {100 * k // n}%] {res['subject']}"
                          f"  {tag}  (elapsed {el:.0f}s, eta {eta:.0f}s)")
    finally:
        stop.set()
        if hb is not None:
            hb.join(timeout=1)
    return results


def _group_summary(results, args):
    """Print aggregate table + group verdict, save combined npz."""
    ok = [r for r in results if "error" not in r]
    print("\n" + "=" * 70)
    print(f"  GROUP SUMMARY — {config.GROUP_FILTER[0]}+{config.GROUP_FILTER[1]}"
          f"  ({len(ok)}/{len(results)} ok)")
    print("=" * 70)
    keys = ok[0]["keys"] if ok else []
    print(f"  {'subject':16s} {'best_corr':>10s}   "
          + "  ".join(f"{k:>7s}" for k in keys))
    for r in results:
        if "error" in r:
            print(f"  {r['subject']:16s}   ERROR: {r['error']}")
            continue
        bp = r["best_params"]
        print(f"  {r['subject']:16s} {r['best_corr']:+10.4f}   "
              + "  ".join(f"{bp[k]:7.3f}" for k in keys))

    if ok:
        corrs = np.array([r["best_corr"] for r in ok])
        print("\n  best real-FC corr:  "
              f"mean={corrs.mean():+.4f}  median={np.median(corrs):+.4f}  "
              f"min={corrs.min():+.4f}  max={corrs.max():+.4f}")
        print("\n  mean param influence across group:")
        for k in keys:
            vals = [r["influence"][k] for r in ok
                    if np.isfinite(r["influence"][k])]
            mv = np.mean(vals) if vals else float("nan")
            print(f"    {k:6s}: {mv:.3f}")

        np.savez(
            f"{args.out}_group.npz",
            subjects=np.array([r["subject"] for r in ok]),
            best_corrs=corrs, param_names=np.array(keys),
            best_params=np.array([[r["best_params"][k] for k in keys]
                                  for r in ok]),
        )
        print(f"\n  saved: {args.out}_group.npz")

        gm = corrs.mean()
        print("\n" + "=" * 70)
        print("  GROUP VERDICT")
        print("=" * 70)
        if gm < 0.3:
            print("  => Group ceiling LOW (<0.30). Simulator misspecification —")
            print("     SC scaling / delays / BOLD / WC structure, not params.")
        elif gm < 0.5:
            print("  => Group ceiling MODERATE. Params help but structural gap "
                  "remains.")
        else:
            print("  => Group ceiling OK (>=0.50). Parameterization is the main "
                  "lever.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None,
                    help="single subject id (default: first group subject)")
    ap.add_argument("--all", action="store_true",
                    help="run every subject in config.GROUP_FILTER (ctr+MPTP)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent processes for --all (1 = sequential; "
                         "N with no --gpus = N subjects share GPU 0)")
    ap.add_argument("--gpus", default="",
                    help="comma GPU ids to pin workers onto (multi-GPU only); "
                         "empty = all workers share the single default GPU")
    ap.add_argument("--params", default="g_e,g_i,c_ei",
                    help="params to jointly search (comma). Any WC param now "
                         "routes per-sim, so c_ee/P/Q/c_ie/c_ii are allowed.")
    ap.add_argument("--samples", type=int, default=5000,
                    help="random joint draws, run as ONE GPU batch (cost ~flat "
                         "up to config.GPU_BATCH, so big is free)")
    ap.add_argument("--topk", type=int, default=5,
                    help="top candidates to print")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heartbeat", type=float, default=10.0,
                    help="seconds between same-terminal progress prints "
                         "(completed-npz count + eta); 0 disables")
    ap.add_argument("--out", default="output_mouse_mptp/joint_opt")
    args = ap.parse_args()

    search = [p for p in args.params.split(",") if p]
    valid = set(THETA_PARAMS) | set(FIXED_PARAMS)
    for p in search:
        assert p in valid, f"{p} not a known WC param {sorted(valid)}"

    gpu_cap = int(getattr(config, "GPU_BATCH", 50_000))
    if args.samples > gpu_cap:
        raise SystemExit(
            f"--samples {args.samples} exceeds config.GPU_BATCH {gpu_cap}; "
            f"joint_opt sends one batch per call. Lower --samples."
        )
    print(f"  search={search}  samples/subject={args.samples}  "
          f"GPU calls/subject=1")

    if args.all:
        subjects, data = load_group(None)
        print(f"\n  group {config.GROUP_FILTER}: {len(subjects)} subjects "
              f"-> {subjects}")
        results = _run_group(subjects, data, args)
        _group_summary(results, args)
    else:
        subjects, data = load_group(args.subject)
        run_subject(subjects[0], data[subjects[0]], args)


def _plot(param_mat, corrs, keys, sid, best, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(keys)
    fig, axes = plt.subplots(1, k + 1, figsize=(3.2 * (k + 1), 3),
                             squeeze=False)
    for j, key in enumerate(keys):
        ax = axes[0, j]
        ax.scatter(param_mat[:, j], corrs, s=6, alpha=0.3)
        ax.set_xlabel(key)
        ax.set_ylabel("real-FC corr")
        ax.grid(alpha=0.3)
    ax = axes[0, k]
    ax.hist(corrs[np.isfinite(corrs)], bins=40)
    ax.axvline(best, c="red", ls="--", label=f"best={best:.3f}")
    ax.set_xlabel("real-FC corr")
    ax.set_title("fit distribution")
    ax.legend()
    fig.suptitle(f"Joint search — {sid}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
