"""Drop-in adapter: route VBI ``simulate_gpu_batch`` calls to cuBNM WCVBI.

The VBI pipeline (``inference/training_data.py``) calls::

    simulate_gpu_batch(sc, theta_batch, param_names=..., fixed_overrides=...,
                       delays=..., apply_bw=True, label=..., n_total=...)
        -> list of (T_bold=240, N=115) float32 BOLD arrays

This module exposes a ``simulate_gpu_batch`` with the SAME signature and
return format, but internally runs the custom ``WCVBISimGroup`` (via
``cuBNM.runner_vbi.run_cubnm_vbi_batch``) instead of the cupy VBI engine —
~9.5x faster at 10k sims. The inference files only need to swap the import.

Importable without a GPU: all heavy imports (cupy via runner, cubnm) are
deferred inside the function body.

Scope / limitations (raises or warns, never silently wrong):
- ``apply_bw=False`` (raw neural output) is NOT supported -> raises.
- Conduction ``delays`` are not yet mapped onto cuBNM ``sc_dist`` -> a
  one-time warning is emitted and delays are ignored.
"""

_DELAY_WARNED = False


def simulate_gpu_batch(weights, theta_batch, param_names=None,
                       delays=None, apply_bw=True, fixed_overrides=None,
                       label=None, n_total=None, **kwargs):
    """cuBNM WCVBI implementation of the VBI ``simulate_gpu_batch`` contract.

    Returns a list of (config.ANALYSIS_BOLD_T, N) float32 BOLD arrays — one
    per row of ``theta_batch`` — matching the original output format.
    """
    global _DELAY_WARNED
    import numpy as np
    import config
    from cuBNM.runner_vbi import run_cubnm_vbi_batch

    if not apply_bw:
        raise NotImplementedError(
            "cuBNM.simulate.simulate_gpu_batch supports apply_bw=True (BOLD) "
            "only; the raw-neural path is not implemented in the cuBNM adapter."
        )

    pn = list(param_names) if param_names is not None else []

    # Conduction delays. cubnm's core computes per-edge delay = sc_dist / v.
    # We already hold the delay matrix (ms), so feeding it as sc_dist with
    # v=1.0 reproduces those delays exactly. Gate on config.USE_DELAYS.
    sc_dist = None
    velocity = None
    use_delays = bool(getattr(config, "USE_DELAYS", False))
    has_delays = delays is not None and np.any(np.asarray(delays) > 0)
    if use_delays and has_delays:
        sc_dist = np.asarray(delays, dtype=np.float64)   # delay matrix in ms
        velocity = 1.0                                    # delay = sc_dist / 1
        if not _DELAY_WARNED:
            print(
                "[cuBNM.simulate] conduction delays ENABLED "
                "(delay matrix fed as sc_dist, v=1.0).",
                flush=True,
            )
            _DELAY_WARNED = True
    elif has_delays and not _DELAY_WARNED:
        print(
            "[cuBNM.simulate] WARNING: delays present but config.USE_DELAYS "
            "is False; running WITHOUT delays.",
            flush=True,
        )
        _DELAY_WARNED = True

    # seed: honour an explicit fixed_overrides['seed'] (matches VBI), else 42.
    sim_seed = 42
    if isinstance(fixed_overrides, dict):
        _s = fixed_overrides.get("seed", None)
        if _s is not None:
            sim_seed = int(_s)

    if label is not None:
        _n = n_total if n_total is not None else len(theta_batch)
        print(
            f"[cuBNM-WCVBI] {label}: running {len(theta_batch):,} sims "
            f"(of {_n:,}) on GPU ...",
            flush=True,
        )

    bolds = run_cubnm_vbi_batch(
        weights, theta_batch, pn,
        sim_seed=sim_seed, force_gpu=True, hrf="bw",
        fixed=fixed_overrides,
        sc_dist=sc_dist, velocity=velocity,
    )

    # cuBNM sim_bold keeps the full duration (burn-in not trimmed from the
    # returned array); trim the leading transient so the shape matches VBI's
    # post-T_CUT length (config.ANALYSIS_BOLD_T, N).
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
    """cuBNM drop-in for VBI ``simulator.simulate_single``.

    Splits ``params_dict`` into the inferred Stage-1 parameters (theta,
    by ``config.STAGE1_PARAMS``) and everything else (fixed WC overrides),
    then runs an ``n_repeat``-row cuBNM batch. Returns a list of
    ``n_repeat`` (config.ANALYSIS_BOLD_T, N) float32 BOLD arrays — matching
    the VBI ``simulate_single`` return format.
    """
    import numpy as np
    import config

    pn = list(config.STAGE1_PARAMS)
    params_dict = dict(params_dict or {})
    missing = [n for n in pn if n not in params_dict]
    if missing:
        raise ValueError(
            f"simulate_single: params_dict is missing inferred params "
            f"{missing} (expected all of config.STAGE1_PARAMS={pn})."
        )
    theta_row = np.array([float(params_dict[n]) for n in pn], dtype=np.float64)
    n = max(1, int(n_repeat))
    theta_batch = np.tile(theta_row[None, :], (n, 1))
    # everything not inferred = fixed WC override (e.g. baseline c_ee/c_ei…)
    fixed = {k: v for k, v in params_dict.items() if k not in pn}
    return simulate_gpu_batch(
        weights, theta_batch, param_names=pn,
        delays=delays, apply_bw=apply_bw, fixed_overrides=fixed,
    )
