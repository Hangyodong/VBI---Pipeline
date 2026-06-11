"""Drop-in adapter: route VBI ``simulate_gpu_batch`` calls to cuBNM RWWEIB_2CPL.

Same signature/return as ``cuBNM/simulate_rwweib.py``, but drives the
TWO-connectome-coupling model (``RWWEIB_2CPLSimGroup`` via
``cuBNM.runner_rwweib_2cpl.run_cubnm_rwweib2_batch``) — E driven by SC @ S_E,
I driven by SC @ S_I. Selected by ``engine="rwweib2"`` in
``inference/training_data.py`` and ``engine_select.py``.

Importable without a GPU: heavy imports (cubnm, runner) are deferred.

Scope: ``apply_bw=False`` (raw neural) is NOT supported -> raises.
"""

_DELAY_WARNED = False


def simulate_gpu_batch(weights, theta_batch, param_names=None,
                       delays=None, apply_bw=True, fixed_overrides=None,
                       label=None, n_total=None, **kwargs):
    """cuBNM RWWEIB_2CPL implementation of the VBI ``simulate_gpu_batch`` contract.

    Returns a list of (config.ANALYSIS_BOLD_T, N) float32 BOLD arrays — one
    per row of ``theta_batch``.
    """
    global _DELAY_WARNED
    import numpy as np
    import config
    from cuBNM.runner_rwweib_2cpl import run_cubnm_rwweib2_batch

    if not apply_bw:
        raise NotImplementedError(
            "cuBNM.simulate_rwweib_2cpl.simulate_gpu_batch supports apply_bw=True "
            "(BOLD) only; the raw-neural path is not implemented."
        )

    pn = list(param_names) if param_names is not None else []

    # Conduction delays (delay matrix fed as sc_dist with v=1.0). Gate on USE_DELAYS.
    sc_dist = None
    velocity = None
    use_delays = bool(getattr(config, "USE_DELAYS", False))
    has_delays = delays is not None and np.any(np.asarray(delays) > 0)
    if use_delays and has_delays:
        sc_dist = np.asarray(delays, dtype=np.float64)
        velocity = 1.0
        if not _DELAY_WARNED:
            print("[cuBNM-RWWEIB2CPL] conduction delays ENABLED "
                  "(delay matrix fed as sc_dist, v=1.0).", flush=True)
            _DELAY_WARNED = True
    elif has_delays and not _DELAY_WARNED:
        print("[cuBNM-RWWEIB2CPL] WARNING: delays present but config.USE_DELAYS "
              "is False; running WITHOUT delays.", flush=True)
        _DELAY_WARNED = True

    sim_seed = 42
    if isinstance(fixed_overrides, dict):
        _s = fixed_overrides.get("seed", None)
        if _s is not None:
            sim_seed = int(_s)

    if label is not None:
        _n = n_total if n_total is not None else len(theta_batch)
        print(f"[cuBNM-RWWEIB2CPL] {label}: running {len(theta_batch):,} sims "
              f"(of {_n:,}) on GPU ...", flush=True)

    bolds = run_cubnm_rwweib2_batch(
        weights, theta_batch, pn,
        sim_seed=sim_seed, force_gpu=True, hrf="bw",
        fixed=fixed_overrides,
        sc_dist=sc_dist, velocity=velocity,
    )

    # Trim leading transient to match VBI post-T_CUT length (ANALYSIS_BOLD_T, N).
    T = int(config.ANALYSIS_BOLD_T)
    out = []
    for b in bolds:
        b = np.asarray(b, dtype=np.float32)
        if b.shape[0] > T:
            b = b[-T:]
        out.append(np.ascontiguousarray(b, dtype=np.float32))
    return out


def simulate_single(weights, params_dict, n_repeat=1, delays=None,
                    apply_bw=True, **kwargs):
    """cuBNM RWWEIB_2CPL drop-in for VBI ``simulator.simulate_single``."""
    import numpy as np
    import config

    pn = list(config.STAGE1_PARAMS)
    params_dict = dict(params_dict or {})
    missing = [n for n in pn if n not in params_dict]
    if missing:
        raise ValueError(
            f"simulate_single: params_dict missing inferred params {missing} "
            f"(expected all of config.STAGE1_PARAMS={pn})."
        )
    theta_row = np.array([float(params_dict[n]) for n in pn], dtype=np.float64)
    n = max(1, int(n_repeat))
    theta_batch = np.tile(theta_row[None, :], (n, 1))
    fixed = {k: v for k, v in params_dict.items() if k not in pn}
    return simulate_gpu_batch(
        weights, theta_batch, param_names=pn,
        delays=delays, apply_bw=apply_bw, fixed_overrides=fixed,
    )
