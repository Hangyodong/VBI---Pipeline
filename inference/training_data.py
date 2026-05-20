"""Training-data collection for SBI: simulate + extract features.

Public API
----------
- collect_training_data(...)       : sample theta from scaled prior,
                                     inverse-transform to raw, simulate,
                                     extract features
- step2_simulate_train(...)        : Stage 1 wrapper (chains the call)
- step3_summary_features(...)      : print/inspect feature shapes
- save_extracted_features(...)
- load_extracted_features(...)

Internal helpers
----------------
- _drain_one_future(...)           : merge ProcessPoolExecutor results

Flow
----
1. Sample ``theta_scaled ~ prior_scaled`` (BoxUniform on [-1, 1])
2. ``theta_raw = scaler.inverse_transform(theta_scaled)``
3. ``BOLD = simulate_gpu_batch(SC, theta_raw, ...)``    <-- raw values
4. ``fc_vec, fcd_vec = extract_features(BOLD)``         <-- in worker
5. Return ``(theta_scaled, theta_raw, fc_raw, fcd_raw)``

theta_scaled is what SBI trains on. theta_raw is what VBI runs on.
The pair (theta_scaled[i], fc_raw[i]) is always aligned because
``simulate_gpu_batch`` uses per-simulation parameter arrays.
"""
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import config
from inference._utils import _progress


# ---------------------------------------------------------------------------
# Training-data collection
# ---------------------------------------------------------------------------

def collect_training_data(subjects, subject_data, prior_scaled,
                          theta_scaler, param_names, n_sim,
                          fixed_overrides=None, apply_bw=True,
                          verbose=True, save_first_sample=False):
    """Run simulations and extract features for the training set.

    When ``save_first_sample=True``, caches the first simulated BOLD
    (first batch, first subject) and returns a dict containing the
    standard outputs plus ``diag_bold`` / ``diag_sid`` so a downstream
    diagnostic cell can plot without re-simulating. Returns the legacy
    4-tuple when ``save_first_sample=False`` (default).
    """
    from simulator import simulate_gpu_batch, worker_extract
    import cupy as cp

    all_theta_s = []
    all_theta_r = []
    all_fc = []
    all_fcd = []
    _first_bold = None
    _first_sid = None
    t0 = time.time()

    n_subj = len(subjects)
    if verbose:
        print(
            f"  Training data collection: {n_subj} subjects x "
            f"{n_sim} = {n_subj * n_sim} sims"
        )

    with ProcessPoolExecutor(max_workers=config.N_CPU) as executor:
        for s_idx, sid in enumerate(subjects):
            batch_sz = config.GPU_BATCH
            n_batches = (n_sim + batch_sz - 1) // batch_sz

            d = subject_data[sid]
            sc = d["sc"]
            dly = d["delays"]
            subj_n0 = len(all_theta_s)

            theta_s = (
                prior_scaled.sample((n_sim,))
                .cpu().numpy().astype(np.float32)
            )
            theta_r = theta_scaler.inverse_transform(theta_s)

            future_queue = []
            t_sub = time.time()

            if verbose:
                sc_sparsity = float((sc > 0).sum()) / sc.size
                delay_max = float(dly.max()) if dly is not None else 0.0
                n_chunks = math.ceil(n_sim / config.GPU_BATCH)
                print(
                    f"\n[Step 2] Subject {s_idx + 1}/{n_subj}  {sid}"
                    f"  N_SIM={n_sim:,}  GPU_BATCH={config.GPU_BATCH:,}"
                    f"\n         sc_sparsity={sc_sparsity:.3f}"
                    f"  delay_max={delay_max:.1f}ms  n_chunks={n_chunks}",
                    flush=True,
                )

            for b_idx in range(n_batches):
                start = b_idx * batch_sz
                end = min(start + batch_sz, n_sim)
                chunk_r = theta_r[start:end]
                chunk_s = theta_s[start:end]

                try:
                    bolds = simulate_gpu_batch(
                        sc, chunk_r, param_names=param_names,
                        fixed_overrides=fixed_overrides,
                        delays=dly, apply_bw=apply_bw,
                        label=str(sid), n_total=n_sim,
                    )
                except Exception as e:
                    print(f"  batch {b_idx} failed: {e}")
                    continue

                if save_first_sample and _first_bold is None and bolds:
                    _first_bold = np.array(bolds[0], copy=True)
                    _first_sid = sid

                future = executor.map(worker_extract, bolds, chunksize=16)
                future_queue.append((chunk_s, chunk_r, future))

                while len(future_queue) >= 2:
                    _drain_one_future(
                        future_queue.pop(0),
                        all_theta_s, all_theta_r, all_fc, all_fcd,
                    )

                if verbose:
                    elapsed = time.time() - t_sub
                    n_done = len(all_theta_s)
                    total = n_subj * n_sim
                    pct = n_done / max(total, 1) * 100
                    _progress(
                        f"batch {b_idx + 1}/{n_batches}  "
                        f"sim {end}/{n_sim}  "
                        f"total {n_done}/{total} ({pct:.1f}%)  "
                        f"({elapsed:.1f}s)"
                    )

            for queued in future_queue:
                _drain_one_future(
                    queued,
                    all_theta_s, all_theta_r, all_fc, all_fcd,
                )

            cp.get_default_memory_pool().free_all_blocks()
            if verbose:
                # L4 — per-subject done (Task A).
                elapsed = time.time() - t_sub
                subj_n = len(all_theta_s) - subj_n0
                fc_dim = all_fc[-1].size if all_fc else 0
                _progress(
                    f"[Subject {s_idx + 1}/{n_subj}] done  "
                    f"{subj_n}/{n_sim} collected  "
                    f"fc=({subj_n},{fc_dim})  {elapsed:.2f} s"
                )

    theta_s = np.array(all_theta_s, dtype=np.float32)
    theta_r = np.array(all_theta_r, dtype=np.float32)
    fc_raw = np.array(all_fc, dtype=np.float32)
    fcd_raw = np.array(all_fcd, dtype=np.float32)

    if verbose:
        # L5 — final summary (Task A).
        total_collected = int(fc_raw.shape[0]) if fc_raw.ndim >= 1 else 0
        total_requested = n_subj * n_sim
        mb = fc_raw.nbytes / 1e6
        if total_collected > 0:
            fc_min = float(np.nanmin(fc_raw))
            fc_max = float(np.nanmax(fc_raw))
            fc_mean = float(np.nanmean(fc_raw))
        else:
            fc_min = fc_max = fc_mean = float("nan")
        dropped = total_requested - total_collected
        elapsed = time.time() - t0
        print()
        print("  [Step 2] Done")
        print(
            f"    total collected : {total_collected} / {total_requested}"
        )
        print(
            f"    fc_raw          : {fc_raw.shape}  "
            f"{fc_raw.dtype}  ~{mb:.2f} MB"
        )
        print(
            f"    fc range        : "
            f"min={fc_min:.4f}  max={fc_max:.4f}  mean={fc_mean:.4f}"
        )
        print(f"    dropped (NaN/Inf): {dropped}")
        print(f"    elapsed         : {elapsed:.2f} s")
    if save_first_sample:
        return {
            "theta_scaled": theta_s,
            "theta_raw": theta_r,
            "fc_raw": fc_raw,
            "fcd_raw": fcd_raw,
            "diag_bold": _first_bold,
            "diag_sid": _first_sid,
        }
    return theta_s, theta_r, fc_raw, fcd_raw


def _drain_one_future(queued, all_theta_s, all_theta_r, all_fc, all_fcd):
    """Append finite results from one future onto the accumulator lists."""
    tcs, tcr, future = queued
    for i, res in enumerate(future):
        if res is None:
            continue
        fc_vec, fcd_vec = res
        if not (np.all(np.isfinite(fc_vec))
                and np.all(np.isfinite(fcd_vec))):
            continue
        all_theta_s.append(tcs[i].tolist())
        all_theta_r.append(tcr[i].tolist())
        all_fc.append(fc_vec)
        all_fcd.append(fcd_vec)


# ---------------------------------------------------------------------------
# Stage 1 wrappers
# ---------------------------------------------------------------------------

def step2_simulate_train(train_subjects, subject_data, prior_scaled,
                         param_scaler, n_sim=None, apply_bw=True,
                         verbose=True, save_first_sample=False):
    """Step 2. Simulate WC for the training set + sample features.

    Returns ``(theta_scaled, theta_raw, fc_raw, fcd_raw)`` by default.
    When ``save_first_sample=True``, returns a dict containing the same
    arrays plus ``diag_bold`` / ``diag_sid`` for downstream diagnostic
    plotting without re-simulation.

    Although the name says "simulate", FC/FCD extraction (step 3) is
    interleaved with simulation by design: BOLD outputs are streamed
    through worker processes batch-by-batch, so extracted features come
    out of the same loop. This avoids storing all BOLD timeseries.
    """
    n_sim = n_sim or config.N_SIM
    if verbose:
        # L1 — header (Task A): one-line banner + a config / per-step block.
        n_steps = int(config.T_END / config.DT)
        _progress(
            f"[Step 2] WC simulation  "
            f"N_SIM={n_sim}  subjects={len(train_subjects)}"
        )
        print(
            f"    config: GPU_BATCH={config.GPU_BATCH}  "
            f"T_end={config.T_END:.0f}ms  T_cut={config.T_CUT:.0f}ms  "
            f"dt={config.DT}ms"
        )
        print(
            f"            n_steps={n_steps}  "
            f"T_bold={config.ANALYSIS_BOLD_T}  N={config.N_REGIONS}"
        )
        print(f"            params: {', '.join(config.STAGE1_PARAMS)}")
    out = collect_training_data(
        train_subjects, subject_data, prior_scaled,
        theta_scaler=param_scaler,
        param_names=config.STAGE1_PARAMS,
        n_sim=n_sim,
        fixed_overrides=None,
        apply_bw=apply_bw,
        verbose=verbose,
        save_first_sample=save_first_sample,
    )
    if isinstance(out, dict):
        theta_s = out["theta_scaled"]
    else:
        theta_s = out[0]
    if theta_s.ndim < 2 or theta_s.shape[0] == 0:
        raise RuntimeError(
            "Step 2 collected 0 samples — all simulation batches failed. "
            "Most common cause: GPU_BATCH is too large and VBI OOMs. "
            "Try setting GPU_BATCH to a smaller value in the Setup cell "
            "(e.g. 4000 instead of 10000) and re-run from Setup."
        )
    return out


def step3_summary_features(fc_raw, fcd_raw, verbose=True):
    """Step 3. Summarize extracted feature shapes."""
    if verbose:
        print("\n  [Step 3] Feature summary")
        print(f"    FC  raw : {fc_raw.shape}  "
              f"(finite={np.all(np.isfinite(fc_raw))})")
        print(f"    FCD raw : {fcd_raw.shape}  "
              f"(finite={np.all(np.isfinite(fcd_raw))})")
    return {
        "fc_dim": fc_raw.shape[1],
        "fcd_dim": fcd_raw.shape[1],
        "n_samples": fc_raw.shape[0],
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_extracted_features(theta_scaled, theta_raw, fc_raw, fcd_raw,
                            param_names=None, save_dir=None,
                            tag="stage1", verbose=True):
    """Save raw extracted features (pre-embedding) to disk."""
    save_dir = save_dir or config.OUTPUT_DIR
    param_names = param_names or config.STAGE1_PARAMS
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"features_{tag}.npz")
    np.savez_compressed(
        path,
        theta_scaled=theta_scaled,
        theta_raw=theta_raw,
        fc_raw=fc_raw,
        fcd_raw=fcd_raw,
        param_names=np.array(param_names),
    )

    if verbose:
        n_mb = os.path.getsize(path) / 1024 / 1024
        print(f"\n  [save_features] {path}  ({n_mb:.1f} MB)")
        print(f"    theta_scaled : {theta_scaled.shape}")
        print(f"    theta_raw    : {theta_raw.shape}")
        print(f"    fc_raw       : {fc_raw.shape}  (Fisher-z upper tri)")
        print(f"    fcd_raw      : {fcd_raw.shape}  (FCD upper tri)")
    return path


def load_extracted_features(save_dir=None, tag="stage1"):
    """Load features saved by `save_extracted_features`."""
    save_dir = save_dir or config.OUTPUT_DIR
    path = os.path.join(save_dir, f"features_{tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"extracted-features file not found: {path!r}. "
            "Run save_extracted_features(...) first (Step 3 / Step 11 "
            "in main.ipynb) or pass an explicit save_dir / tag."
        )
    data = np.load(path, allow_pickle=True)
    return {
        "theta_scaled": data["theta_scaled"],
        "theta_raw": data["theta_raw"],
        "fc_raw": data["fc_raw"],
        "fcd_raw": data["fcd_raw"],
        "param_names": list(data["param_names"]),
    }
