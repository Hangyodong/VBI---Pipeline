"""Stage 2 inference + θ_bad selection.

Public API
----------
- select_theta_bad(sensitivity, shrinkage, ...) -> list[str]
       θ_bad = (sensitivity high) AND (shrinkage low)
- select_difficult_params(shrinkage, ...) -> list[str]  (legacy, shrinkage-only)
- build_stage2_param_set(shrinkage, sensitivity=None, ...)
       -> (stage2_params, nuisance_params)
- run_stage2_snpe(train_subjects, subject_data, stage1_result, ...)
       -> dict

Stage 2 design
--------------
Stage 2 re-infers θ_bad (sensitive but poorly identified in Stage 1)
plus the local E/I coupling parameters (c_ee, c_ei, c_ie, c_ii).
Nuisance Stage 1 parameters (the rest) are either fixed at their
posterior mean or sampled from the Stage 1 posterior, controlled by
``config.NUISANCE_METHOD``.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import config
from inference._utils import _progress
from inference.feature_pipeline import FeaturePipeline
from inference.priors import make_scaled_prior
from inference.scaling import make_stage2_param_scaler
from inference.snpe import _print_pca_diagnostic, train_snpe
from inference.training_data import _drain_one_future

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# θ_bad selection
# ---------------------------------------------------------------------------

def select_difficult_params(shrinkage_per_param, param_names=None,
                            threshold=None):
    """Return Stage 1 parameters whose shrinkage falls below threshold.

    Deprecated by ``select_theta_bad`` which combines sensitivity and
    shrinkage. Kept for backward compatibility.
    """
    param_names = param_names or config.STAGE1_PARAMS
    threshold = threshold or config.DIFFICULT_SHRINKAGE
    return [
        name for name, s in zip(param_names, shrinkage_per_param)
        if s < threshold
    ]


def select_theta_bad(sensitivity_per_param, shrinkage_per_param,
                     param_names=None,
                     sens_threshold=0.5, shrinkage_threshold=0.2):
    """Select θ_bad = sensitivity high AND shrinkage low.

    Output에 중요한데 1차에서 잘 식별되지 않은 parameter가 2차 후보.

    Parameters
    ----------
    sensitivity_per_param : array-like  per-parameter sensitivity score
    shrinkage_per_param   : array-like  per-parameter shrinkage score
    param_names           : list[str] or None  (default Stage 1 params)
    sens_threshold        : sensitivity >= this counts as "high"
    shrinkage_threshold   : shrinkage   <  this counts as "low"

    Returns
    -------
    theta_bad : list[str]
    """
    param_names = param_names or config.STAGE1_PARAMS
    if len(sensitivity_per_param) != len(param_names):
        raise ValueError(
            f"sensitivity len {len(sensitivity_per_param)} != "
            f"n_params {len(param_names)}"
        )
    if len(shrinkage_per_param) != len(param_names):
        raise ValueError(
            f"shrinkage len {len(shrinkage_per_param)} != "
            f"n_params {len(param_names)}"
        )
    theta_bad = []
    for name, sens, shr in zip(
        param_names, sensitivity_per_param, shrinkage_per_param
    ):
        if sens >= sens_threshold and shr < shrinkage_threshold:
            theta_bad.append(name)
    return theta_bad


def build_stage2_param_set(shrinkage_per_param, param_names=None,
                           sensitivity_per_param=None,
                           sens_threshold=0.5,
                           shrinkage_threshold=None):
    """Build Stage 2 parameter set.

    Returns (stage2_params, nuisance_params) where::

        stage2_params = theta_bad + local E/I coupling
        theta_bad = (sensitivity high) AND (shrinkage low)   if sensitivity given
                  = (shrinkage low)                          if sensitivity None
        nuisance  = Stage 1 params not in theta_bad
    """
    param_names = param_names or config.STAGE1_PARAMS
    shr_thresh = (
        shrinkage_threshold if shrinkage_threshold is not None
        else config.DIFFICULT_SHRINKAGE
    )
    if sensitivity_per_param is not None:
        difficult = select_theta_bad(
            sensitivity_per_param, shrinkage_per_param,
            param_names=param_names,
            sens_threshold=sens_threshold,
            shrinkage_threshold=shr_thresh,
        )
    else:
        difficult = select_difficult_params(
            shrinkage_per_param, param_names=param_names,
            threshold=shr_thresh,
        )
    nuisance = [p for p in param_names if p not in difficult]
    c_params = list(config.LOCAL_EI_PARAMS)
    stage2_params = difficult + c_params

    print("\n  Stage 2 configuration:")
    print(
        f"    theta_bad (high sens & low shrinkage): {difficult}"
    )
    print(f"    Nuisance (fix or sample from Stage 1)     : {nuisance}")
    print(f"    c-params to add (LOCAL_EI_PARAMS)         : {c_params}")
    print(f"    => Stage 2 inference targets              : {stage2_params}")
    return stage2_params, nuisance


# ---------------------------------------------------------------------------
# Stage 2 data collection
# ---------------------------------------------------------------------------

def collect_stage2_data(train_subjects, subject_data,
                        stage2_prior_scaled,
                        stage2_params, nuisance_params,
                        stage2_param_scaler,
                        stage1_posterior, stage1_param_scaler,
                        stage1_feature_pipeline, x_obs_dict_s1,
                        n_sim=None, nuisance_method=None,
                        apply_bw=True, verbose=True):
    """Collect Stage 2 training simulations with nuisance handling."""
    from simulator import simulate_gpu_batch, worker_extract
    import cupy as cp

    n_sim = n_sim or config.N_SIM_S2
    nuisance_method = nuisance_method or config.NUISANCE_METHOD
    n_subj = len(train_subjects)

    if verbose:
        print(f"\n  Stage 2 data collection ({nuisance_method})")
        print(f"    Targets : {stage2_params}")
        print(f"    Nuisance: {nuisance_params}")

    all_param_names = stage2_params + nuisance_params
    all_theta_s, all_theta_r, all_fc, all_fcd = [], [], [], []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=config.N_CPU) as executor:
        for s_idx, sid in enumerate(train_subjects):
            if verbose:
                print(
                    f"\n  [{s_idx + 1}/{n_subj}] {sid}"
                )
            d = subject_data[sid]
            sc, dly = d["sc"], d["delays"]

            theta_s2_s = (
                stage2_prior_scaled.sample((n_sim,))
                .cpu().numpy().astype(np.float32)
            )
            theta_s2_r = stage2_param_scaler.inverse_transform(theta_s2_s)

            nuis_raw_per_sim = _build_nuisance_array(
                nuisance_params, n_sim, sid,
                stage1_posterior, stage1_param_scaler,
                x_obs_dict_s1, nuisance_method,
            )
            theta_combined_raw = np.concatenate(
                [theta_s2_r, nuis_raw_per_sim], axis=1,
            ).astype(np.float32)

            batch_sz = config.GPU_BATCH
            n_batches = (n_sim + batch_sz - 1) // batch_sz
            future_queue = []
            t_sub = time.time()

            for b_idx in range(n_batches):
                start = b_idx * batch_sz
                end = min(start + batch_sz, n_sim)
                chunk_raw = theta_combined_raw[start:end]
                chunk_s2_s = theta_s2_s[start:end]
                chunk_s2_r = theta_s2_r[start:end]

                try:
                    bolds = simulate_gpu_batch(
                        sc, chunk_raw,
                        param_names=all_param_names,
                        fixed_overrides=None, delays=dly,
                        apply_bw=apply_bw,
                    )
                except Exception as e:
                    print(f"  batch {b_idx} failed: {e}")
                    continue

                future = executor.map(worker_extract, bolds, chunksize=16)
                future_queue.append((chunk_s2_s, chunk_s2_r, future))

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
                elapsed = time.time() - t_sub
                print(
                    f"    done: {len(all_theta_s)}  ({elapsed:.1f}s)"
                )
                sys.stdout.flush()

    theta_s = np.array(all_theta_s, dtype=np.float32)
    theta_r = np.array(all_theta_r, dtype=np.float32)
    fc_raw = np.array(all_fc, dtype=np.float32)
    fcd_raw = np.array(all_fcd, dtype=np.float32)

    if verbose:
        print(
            f"\n  Stage 2 collected: theta={theta_s.shape}, "
            f"fc={fc_raw.shape}  ({time.time() - t0:.1f}s)"
        )
    return theta_s, theta_r, fc_raw, fcd_raw


def _build_nuisance_array(nuisance_params, n_sim, sid,
                          stage1_posterior, stage1_param_scaler,
                          x_obs_dict_s1, nuisance_method):
    """Return (n_sim, n_nuisance) raw nuisance values for Stage 2 sims."""
    if not nuisance_params:
        return np.empty((n_sim, 0), dtype=np.float32)

    s1_names = config.STAGE1_PARAMS
    x_obs_t = torch.tensor(x_obs_dict_s1[sid], dtype=torch.float32)

    if nuisance_method == "fix_mean":
        samples_scaled = (
            stage1_posterior
            .sample((1000,), x=x_obs_t, show_progress_bars=False)
            .cpu().numpy()
        )
        samples_raw = stage1_param_scaler.inverse_transform(samples_scaled)
        means_raw = samples_raw.mean(axis=0)
        idx_nuis = [s1_names.index(p) for p in nuisance_params]
        return np.tile(means_raw[idx_nuis][None, :], (n_sim, 1))

    # posterior_sample
    samples_scaled = (
        stage1_posterior
        .sample((n_sim,), x=x_obs_t, show_progress_bars=False)
        .cpu().numpy()
    )
    samples_raw = stage1_param_scaler.inverse_transform(samples_scaled)
    idx_nuis = [s1_names.index(p) for p in nuisance_params]
    return samples_raw[:, idx_nuis]


# ---------------------------------------------------------------------------
# Stage 2 driver
# ---------------------------------------------------------------------------

def run_stage2_snpe(train_subjects, subject_data, stage1_result,
                    val_shrinkage, n_sim=None, apply_bw=True,
                    verbose=True):
    """End-to-end Stage 2 inference (steps 10 - 11)."""
    n_sim = n_sim or config.N_SIM_S2

    stage2_params, nuisance_params = build_stage2_param_set(val_shrinkage)
    s2_param_scaler = make_stage2_param_scaler(stage2_params)
    s2_prior_scaled = make_scaled_prior(
        len(stage2_params), device=config.SBI_DEVICE,
    )

    s1_posterior = stage1_result["posterior"]
    s1_param_scaler = stage1_result["param_scaler"]
    s1_pipeline = stage1_result["feature_pipeline"]

    from simulator import extract_observed_features
    x_obs_s1 = {}
    for sid in train_subjects:
        fc_obs, fcd_obs = extract_observed_features(subject_data[sid])
        x_obs_s1[sid] = s1_pipeline.transform(fc_obs, fcd_obs)

    theta_s, theta_r, fc_raw, fcd_raw = collect_stage2_data(
        train_subjects, subject_data, s2_prior_scaled,
        stage2_params, nuisance_params,
        stage2_param_scaler=s2_param_scaler,
        stage1_posterior=s1_posterior,
        stage1_param_scaler=s1_param_scaler,
        stage1_feature_pipeline=s1_pipeline,
        x_obs_dict_s1=x_obs_s1,
        n_sim=n_sim,
        nuisance_method=config.NUISANCE_METHOD,
        apply_bw=apply_bw, verbose=verbose,
    )

    s2_pipeline = FeaturePipeline()
    s2_pipeline.fit(fc_raw, fcd_raw, verbose=verbose)
    x_input = s2_pipeline.transform(fc_raw, fcd_raw)

    pca_diag = s2_pipeline.diagnostic(fc_raw, fcd_raw)
    if verbose:
        _print_pca_diagnostic(pca_diag, header="Stage 2 PCA diagnostic")

    posterior, embedding_net = train_snpe(
        theta_s, x_input, s2_prior_scaled,
        embedding_net=None, proposal=None, verbose=verbose,
    )

    return {
        "posterior": posterior,
        "embedding_net": embedding_net,
        "stage2_params": stage2_params,
        "nuisance_params": nuisance_params,
        "param_scaler": s2_param_scaler,
        "prior_scaled": s2_prior_scaled,
        "feature_pipeline": s2_pipeline,
        "theta_scaled": theta_s,
        "theta_raw": theta_r,
        "fc_raw": fc_raw,
        "fcd_raw": fcd_raw,
        "x_input": x_input,
        "pca_diagnostic": pca_diag,
    }
