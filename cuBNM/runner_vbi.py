"""Runner for the VBI-equivalent cuBNM model (``WCVBISimGroup``).

Same interface as :mod:`cuBNM.runner` (``run_cubnm_batch``) but targets the
custom ``WCVBI`` model defined in ``cuBNM/wc_vbi.yaml`` — which exposes every
VBI WC_EIB parameter (tau / alpha / theta / k / r / c / lamda + the FFI gate
g_i) as a settable regional parameter, so both engines run identical dynamics.

Build prerequisite
------------------
``WCVBISimGroup`` only exists after the YAML has been fed through the upstream
cuBNM codegen driver and the extension rebuilt from source. The installed
0.1.0 sdist does NOT ship that codegen driver (only pre-generated sources +
mako templates), so until the model is generated+built this module raises a
clear ``ModelNotBuiltError`` instead of silently falling back.

Parameter mapping (VBI WC_FIXED / inferred  ->  WCVBI param)
-----------------------------------------------------------
    g_e   (inferred)  -> G        (global_param)
    c_ei  (inferred)  -> c_ei     (regional, I->E inhibition)
    g_i   (inferred)  -> g_i      (regional, FFI inhibitory gate)
    c_ee/c_ie/c_ii    -> c_ee/c_ie/c_ii
    P / Q             -> P_E / P_I
    noise_amp         -> sigma_E, sigma_I
    lamda             -> lamda
    alpha_e/alpha_i   -> alpha_E / alpha_I   (sigmoid gain)
    theta_e/theta_i   -> theta_E / theta_I   (sigmoid threshold)
    k_e/k_i           -> k_e / k_i
    r_e/r_i           -> r_e / r_i
    c_e/c_i           -> c_e / c_i
    tau_e/tau_i       -> tau_E / tau_I
"""
import numpy as np


class ModelNotBuiltError(RuntimeError):
    """Raised when WCVBISimGroup is not yet generated + compiled into cubnm."""


# VBI WC_FIXED key -> WCVBI regional param name. (Inferred g_e/c_ei/g_i handled
# separately.) Fallback defaults match cuBNM/wc_vbi.yaml.
_FIXED_TO_VBI = {
    "c_ee":   "c_ee",
    "c_ie":   "c_ie",
    "c_ii":   "c_ii",
    "P":      "P_E",
    "Q":      "P_I",
    "lamda":  "lamda",
    "alpha_e": "alpha_E", "alpha_i": "alpha_I",
    "theta_e": "theta_E", "theta_i": "theta_I",
    "k_e": "k_e", "k_i": "k_i",
    "r_e": "r_e", "r_i": "r_i",
    "c_e": "c_e", "c_i": "c_i",
    "tau_e": "tau_E", "tau_i": "tau_I",
}

# WCVBI regional params + fallback defaults (mirror wc_vbi.yaml).
_VBI_REGIONAL_DEFAULT = {
    "c_ee": 11.0, "c_ei": 12.0, "c_ie": 10.0, "c_ii": 1.0,
    "P_E": 0.5, "P_I": 0.0, "sigma_E": 0.02, "sigma_I": 0.02,
    "g_i": 0.0, "lamda": 1.0,
    "alpha_E": 1.2, "alpha_I": 2.0, "theta_E": 2.0, "theta_I": 3.5,
    "k_e": 1.0, "k_i": 1.0, "r_e": 1.0, "r_i": 1.0,
    "c_e": 1.0, "c_i": 1.0, "tau_E": 10.0, "tau_I": 10.0,
}


def _import_wcvbi():
    """Import WCVBISimGroup or raise a clear, actionable error."""
    try:
        from cubnm.sim import WCVBISimGroup  # generated class
        return WCVBISimGroup
    except Exception as e:  # noqa: BLE001
        raise ModelNotBuiltError(
            "WCVBISimGroup is not available in the installed cubnm. Generate it "
            "from cuBNM/wc_vbi.yaml with the upstream cuBNM codegen driver "
            "(codegen/ + recipes/ are NOT in the installed 0.1.0 sdist), then "
            "rebuild the extension from source with --no-build-isolation. "
            f"Underlying import error: {type(e).__name__}: {e}"
        ) from e


def _theta_column(theta, pn, name):
    if name in pn:
        return np.asarray(theta, dtype=np.float64)[:, pn.index(name)]
    return None


def build_param_lists(theta_batch, param_names, n_nodes, fixed=None):
    """Build the WCVBI ``param_lists`` from a VBI theta batch.

    Returns dict {'G': (n_sims,), 'c_ee': (n_sims, n_nodes), ...} float64,
    with every regional WC_FIXED value set explicitly (no kernel constants).
    """
    import config

    wc_fixed = dict(getattr(config, "WC_FIXED", {}) or {})
    if fixed:
        # Explicit per-call overrides win over config.WC_FIXED. Only keys
        # recognized below (_FIXED_TO_VBI / noise_amp) take effect; unknown
        # keys (seed, weights, num_sim, ...) are ignored harmlessly.
        for k, v in dict(fixed).items():
            if k in _FIXED_TO_VBI or k == "noise_amp":
                wc_fixed[k] = v
    theta = np.asarray(theta_batch, dtype=np.float64)
    n_sims = theta.shape[0]
    pn = list(param_names)

    # Start from yaml defaults, then overlay explicit WC_FIXED values.
    regional = dict(_VBI_REGIONAL_DEFAULT)
    for vbi_key, vbi_name in _FIXED_TO_VBI.items():
        if vbi_key in wc_fixed:
            regional[vbi_name] = float(wc_fixed[vbi_key])
    if "noise_amp" in wc_fixed:
        regional["sigma_E"] = float(wc_fixed["noise_amp"])
        regional["sigma_I"] = float(wc_fixed["noise_amp"])

    param_lists = {}

    # Global coupling G  <-  inferred g_e (fallback 0.5)
    g_e = _theta_column(theta, pn, "g_e")
    if g_e is None:
        g_e = np.full(n_sims, 0.5, dtype=np.float64)
    param_lists["G"] = np.ascontiguousarray(g_e, dtype=np.float64)

    # Broadcast every regional default to (n_sims, n_nodes)
    for name, dval in regional.items():
        param_lists[name] = np.full((n_sims, n_nodes), dval, dtype=np.float64)

    # Inferred c_ei (I->E) per-sim scalar broadcast to all nodes
    c_ei = _theta_column(theta, pn, "c_ei")
    if c_ei is not None:
        param_lists["c_ei"] = np.ascontiguousarray(
            np.repeat(c_ei[:, None], n_nodes, axis=1), dtype=np.float64
        )

    # Inferred g_i (FFI gate) per-sim scalar broadcast to all nodes
    g_i = _theta_column(theta, pn, "g_i")
    if g_i is not None:
        param_lists["g_i"] = np.ascontiguousarray(
            np.repeat(g_i[:, None], n_nodes, axis=1), dtype=np.float64
        )

    # Per-sim routing for any fixed-route WC param supplied via theta.
    # Previously these (c_ee, P->P_E, Q->P_I, c_ie, c_ii, ...) were only
    # applied as a single scalar broadcast from WC_FIXED, so theta columns
    # for them never reached the simulator (the long-standing P/Q "dead
    # column" bug, and why c_ee could not be jointly searched). When a
    # theta column exists for one of these, override its regional broadcast
    # with the per-sim values.
    for vbi_key, vbi_name in _FIXED_TO_VBI.items():
        col = _theta_column(theta, pn, vbi_key)
        if col is not None:
            param_lists[vbi_name] = np.ascontiguousarray(
                np.repeat(col[:, None], n_nodes, axis=1), dtype=np.float64
            )

    return param_lists


def run_cubnm_batch(weights, theta_batch, param_names,
                    duration_s=None, tr_s=None, dt_ms=None,
                    fixed=None, force_gpu=True, sim_seed=42,
                    burn_in_s=None, hrf="bw",
                    sc_dist=None, velocity=None):
    """Run a WCVBI batch — same signature/return as cuBNM.runner.run_cubnm_batch.

    Defaults (None) read config: duration=T_END/1000, tr=TR_SEC, dt=DT,
    burn_in=T_CUT/1000.

    hrf : {"bw", "vbi"}
        "bw"  -> cuBNM's built-in Balloon-Windkessel BOLD (fast).
        "vbi" -> drop cuBNM BOLD; take the neural E time series and convolve
                 it with VBI's exact MixtureOfGammas HRF (bold.BoldMonitor),
                 so the BOLD model is identical to the VBI engine. The neural
                 series is sampled at config.DT*config.DECIMATE ms and the full
                 (un-trimmed) series is fed with t_cut_ms=burn_in so the HRF
                 stock pre-fills during the transient exactly like VBI.

    Returns list[np.ndarray]  — one (T, N) BOLD array per simulation.
    """
    WCVBISimGroup = _import_wcvbi()
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

    # Conduction delays: when an sc_dist matrix is provided, cubnm's core
    # computes per-edge delay = sc_dist / v (so feeding the delay matrix in ms
    # with v=1.0 reproduces it exactly). do_delay is enabled by passing
    # sc_dist to the SimGroup constructor; v is set as a per-sim param below.
    use_delay = sc_dist is not None
    if use_delay:
        sc_dist = np.ascontiguousarray(np.asarray(sc_dist, dtype=np.float64))
        if sc_dist.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"sc_dist shape {sc_dist.shape} != ({n_nodes}, {n_nodes})"
            )
        if velocity is None:
            velocity = float(config.VELOCITY_M_PER_S)

    use_vbi_hrf = (str(hrf).lower() == "vbi")
    # neural sampling for the VBI HRF path (ms) == VBI's stored neural step
    neural_dt_ms = float(config.DT) * float(config.DECIMATE)

    grp = WCVBISimGroup(
        duration=float(duration_s),
        TR=float(tr_s),
        sc=weights,
        sc_dist=(sc_dist if use_delay else None),
        dt=str(dt_ms),
        # vbi-hrf: keep the full neural series (no trim) so the HRF stock
        # pre-fills during the transient; bw: trim transient from BOLD.
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

    # Conduction velocity per simulation (only meaningful when do_delay).
    if use_delay:
        grp.param_lists["v"] = np.full(n_sims, float(velocity), dtype=np.float64)

    grp.run()

    if not use_vbi_hrf:
        sim_bold = np.asarray(grp.sim_bold)
        if sim_bold.ndim == 3:
            return [np.ascontiguousarray(sim_bold[i]) for i in range(sim_bold.shape[0])]
        return [np.ascontiguousarray(sim_bold.reshape(-1, n_nodes))]

    # ── VBI MixtureOfGammas HRF on cuBNM neural E ─────────────────────────
    from bold import BoldMonitor  # repo-root module (read-only use)

    # sim_states['E'] shape (N_sims, T_neural, nodes); full (un-trimmed) series.
    E = np.asarray(grp.sim_states["E"], dtype=np.float32)   # (S, T, N)
    n_steps = E.shape[1]
    mon = BoldMonitor(
        nn=n_nodes, ns=n_sims, dt_ms=neural_dt_ms,
        xp=np, use_gpu=False,
        period_ms=float(tr_s) * 1000.0,
        hrf_length_ms=getattr(config, "HRF_LENGTH_MS", 32_000.0),
    )
    t_cut_ms = float(burn_in_s) * 1000.0
    for i in range(n_steps):
        mon.step(i, E[:, i, :].T, t_cut_ms=t_cut_ms)   # frame (nn, ns)
    bold = mon.collect(mean_subtract=True)             # (T_bold, N, S)
    return [np.ascontiguousarray(bold[:, :, s]) for s in range(bold.shape[2])]


# Alias: explicit VBI-named entry point (== run_cubnm_batch).
run_cubnm_vbi_batch = run_cubnm_batch
