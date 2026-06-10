"""Runner for the RWW-EIB-FFI cuBNM model (``RWWEIBSimGroup``).

Mirrors ``cuBNM/runner_vbi.py`` but drives the custom ``RWWEIB`` model defined
in ``cuBNM/rww_eib.yaml`` — VBI-based Reduced Wong-Wang E/I with
excitatory-driven feedforward inhibition (single connectome coupling
``globalinput = SC @ S_e`` feeds both populations with independent gains
``g_LRE`` and ``g_FFI``).

``RWWEIBSimGroup`` only exists after ``cuBNM/rww_eib.yaml`` has been fed through
the upstream cuBNM codegen driver and the extension rebuilt from source. The
installed 0.1.0 wheel does NOT ship that codegen driver, so until the model is
generated+built this module raises ``ModelNotBuiltError`` (see BUILD_RWWEIB.md).

Parameter mapping (VBI theta / RWWEIB_FIXED  ->  RWWEIB regional param):
  inferred  g_LRE -> g_LRE   (regional, per-sim scalar broadcast to all nodes)
  inferred  g_FFI -> g_FFI   (regional, per-sim scalar broadcast to all nodes)
  inferred  sigma -> sigma   (regional, per-sim scalar broadcast to all nodes)
  fixed     RWWEIB_FIXED[k]  -> regional param k (broadcast to all nodes)
All params are regional; the model declares NO global_param (the long-range
gain is applied in step_equations as ``g_LRE * J_N * globalinput``).
"""

import numpy as np


class ModelNotBuiltError(RuntimeError):
    """Raised when RWWEIBSimGroup is not yet generated + compiled into cubnm."""


# RWWEIB regional params + fallback defaults (mirror cuBNM/rww_eib.yaml).
# config.RWWEIB_FIXED overrides these; the 3 inferred params (g_LRE/g_FFI/sigma)
# are filled from theta when present, else fall back to these defaults.
_RWWEIB_REGIONAL_DEFAULT = {
    "g_LRE": 1.0, "g_FFI": 1.0, "sigma": 0.01,
    "J_N": 0.15, "J_i": 1.0, "w_p": 1.4,
    "W_e": 1.0, "W_i": 0.7, "I_o": 0.382, "I_ext": 0.0,
    "a_e": 310.0, "b_e": 125.0, "d_e": 0.16,
    "gamma_e": 0.641 / 1000.0, "tau_e": 100.0,
    "a_i": 615.0, "b_i": 177.0, "d_i": 0.087,
    "gamma_i": 1.0 / 1000.0, "tau_i": 10.0,
}

# Which inferred theta columns map to which regional param.
_INFERRED = ("g_LRE", "g_FFI", "sigma")


def _import_rwweib():
    """Import RWWEIBSimGroup or raise a clear, actionable error."""
    try:
        from cubnm.sim import RWWEIBSimGroup  # generated class
        return RWWEIBSimGroup
    except Exception as e:  # noqa: BLE001
        raise ModelNotBuiltError(
            "RWWEIBSimGroup is not available in the installed cubnm. Generate it "
            "from cuBNM/rww_eib.yaml with the upstream cuBNM codegen driver, then "
            "rebuild the extension from source with --no-build-isolation "
            "(see cuBNM/BUILD_RWWEIB.md). "
            f"Underlying import error: {type(e).__name__}: {e}"
        ) from e


def _theta_column(theta, pn, name):
    if name in pn:
        return np.asarray(theta, dtype=np.float64)[:, pn.index(name)]
    return None


def build_param_lists(theta_batch, param_names, n_nodes, fixed=None):
    """Build RWWEIB ``param_lists`` {name: (n_sims, n_nodes) float64}.

    Every regional param is set explicitly. Fixed defaults come from
    ``config.RWWEIB_FIXED`` (falling back to ``_RWWEIB_REGIONAL_DEFAULT``);
    the inferred g_LRE/g_FFI/sigma are taken per-sim from ``theta_batch`` and
    broadcast to all nodes.
    """
    import config

    rwweib_fixed = dict(getattr(config, "RWWEIB_FIXED", {}) or {})
    if fixed:
        for k, v in dict(fixed).items():
            if k in _RWWEIB_REGIONAL_DEFAULT:
                rwweib_fixed[k] = v

    theta = np.asarray(theta_batch, dtype=np.float64)
    n_sims = theta.shape[0]
    pn = list(param_names)

    # Start from yaml defaults, overlay config.RWWEIB_FIXED.
    regional = dict(_RWWEIB_REGIONAL_DEFAULT)
    for k, v in rwweib_fixed.items():
        if k in regional:
            regional[k] = float(v)

    # Broadcast every regional default to (n_sims, n_nodes).
    param_lists = {
        name: np.full((n_sims, n_nodes), dval, dtype=np.float64)
        for name, dval in regional.items()
    }

    # Override the inferred params with per-sim theta values (broadcast to nodes).
    for name in _INFERRED:
        col = _theta_column(theta, pn, name)
        if col is not None:
            param_lists[name] = np.ascontiguousarray(
                np.repeat(col[:, None], n_nodes, axis=1), dtype=np.float64
            )

    return param_lists


def run_cubnm_rwweib_batch(weights, theta_batch, param_names,
                           duration_s=None, tr_s=None, dt_ms=None,
                           fixed=None, force_gpu=True, sim_seed=42,
                           burn_in_s=None, hrf="bw",
                           sc_dist=None, velocity=None):
    """Run an RWWEIB batch — same signature/return as runner_vbi.run_cubnm_batch.

    hrf : {"bw", "vbi"}
        "bw"  -> cuBNM built-in Balloon-Windkessel BOLD (fast).
        "vbi" -> take the neural S_e series and convolve with VBI's exact
                 MixtureOfGammas HRF (bold.BoldMonitor), identical BOLD model
                 to the cupy VBI engine.
    Returns list[np.ndarray] — one (T, N) BOLD array per simulation.
    """
    RWWEIBSimGroup = _import_rwweib()
    import config

    if duration_s is None:
        duration_s = float(config.T_END) / 1000.0
    if tr_s is None:
        tr_s = float(config.TR_SEC)
    if dt_ms is None:
        dt_ms = float(config.DT)
    if burn_in_s is None:
        burn_in_s = float(config.T_CUT) / 1000.0

    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    n_nodes = weights.shape[0]
    theta = np.asarray(theta_batch, dtype=np.float64)
    n_sims = theta.shape[0]

    use_delay = sc_dist is not None
    if use_delay:
        sc_dist = np.ascontiguousarray(np.asarray(sc_dist, dtype=np.float64))
        if sc_dist.shape != (n_nodes, n_nodes):
            raise ValueError(f"sc_dist shape {sc_dist.shape} != ({n_nodes}, {n_nodes})")
        if velocity is None:
            velocity = float(config.VELOCITY_M_PER_S)

    use_vbi_hrf = (str(hrf).lower() == "vbi")
    neural_dt_ms = float(config.DT) * float(config.DECIMATE)

    grp = RWWEIBSimGroup(
        duration=float(duration_s),
        TR=float(tr_s),
        sc=weights,
        sc_dist=(sc_dist if use_delay else None),
        dt=str(dt_ms),
        bold_remove_s=(0.0 if use_vbi_hrf else float(burn_in_s)),
        do_fc=(not use_vbi_hrf),
        do_fcd=False,
        gof_terms=([] if use_vbi_hrf else ["+fc_corr"]),
        force_gpu=bool(force_gpu),
        sim_seed=int(sim_seed),
        states_ts=use_vbi_hrf,
        states_sampling=(neural_dt_ms / 1000.0 if use_vbi_hrf else None),
    )
    grp.N = n_sims

    param_lists = build_param_lists(theta, list(param_names), n_nodes, fixed)
    for k, v in param_lists.items():
        grp.param_lists[k] = v

    if use_delay:
        grp.param_lists["v"] = np.full(n_sims, float(velocity), dtype=np.float64)

    grp.run()

    if not use_vbi_hrf:
        sim_bold = np.asarray(grp.sim_bold)
        if sim_bold.ndim == 3:
            return [np.ascontiguousarray(sim_bold[i]) for i in range(sim_bold.shape[0])]
        return [np.ascontiguousarray(sim_bold.reshape(-1, n_nodes))]

    # ── VBI MixtureOfGammas HRF on cuBNM neural S_e ───────────────────────
    from bold import BoldMonitor  # repo-root module (read-only use)

    # sim_states['S_e'] shape (N_sims, T_neural, nodes); full (un-trimmed) series.
    Se = np.asarray(grp.sim_states["S_e"], dtype=np.float32)  # (S, T, N)
    n_steps = Se.shape[1]
    mon = BoldMonitor(
        nn=n_nodes, ns=n_sims, dt_ms=neural_dt_ms,
        xp=np, use_gpu=False,
        period_ms=float(tr_s) * 1000.0,
        hrf_length_ms=getattr(config, "HRF_LENGTH_MS", 32_000.0),
    )
    t_cut_ms = float(burn_in_s) * 1000.0
    for i in range(n_steps):
        mon.step(i, Se[:, i, :].T, t_cut_ms=t_cut_ms)   # frame (nn, ns)
    bold = mon.collect(mean_subtract=True)              # (T_bold, N, S)
    return [np.ascontiguousarray(bold[:, :, s]) for s in range(bold.shape[2])]
