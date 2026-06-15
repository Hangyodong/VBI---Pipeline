"""Runner for the RWW-EIB two-coupling cuBNM model (``RWWEIB_2CPLSimGroup``).

Mirrors ``cuBNM/runner_rwweib.py`` but drives the TWO-connectome-coupling model
defined in ``cuBNM/rww_eib_2cpl.yaml`` — VBI-based Reduced Wong-Wang E/I with two
independent connectome couplings (the equation-literal full WW):

    globalinput_E = SC @ S_E   (long-range excitation,  gain g_LRE)
    globalinput_I = SC @ S_I   (long-range I-coupling,  gain g_FFI * lambda_IE)

Unlike the single-coupling FFI variant (``runner_rwweib.py``), here the I
population is driven by ``SC @ S_I`` (its own conn_state_var), NOT by the shared
``SC @ S_E``. This requires the cuBNM codegen to support ``conn_state_vars:
[S_E, S_I]`` (two global-input arrays). If your cubnm build's codegen only
supports a single coupling, this model cannot be generated — see
``cuBNM/cubnm_build`` / BUILD notes and verify codegen multi-coupling support
BEFORE building.

``RWWEIB_2CPLSimGroup`` only exists after ``cuBNM/rww_eib_2cpl.yaml`` is fed
through the upstream cuBNM codegen driver and the extension rebuilt from source.

Parameter mapping (theta / RWWEIB2_FIXED  ->  RWWEIB_2CPL param):
  inferred  g_LRE -> g_LRE   (global, per-sim scalar)
  inferred  g_FFI -> g_FFI   (regional, per-sim scalar broadcast to all nodes)
  inferred  sigma -> sigma   (regional, per-sim scalar broadcast to all nodes)
  inferred  I_o   -> I_o     (regional, per-sim scalar broadcast to all nodes)
  fixed     RWWEIB2_FIXED[k] -> regional param k (broadcast to all nodes)

The two long-range gains drive DIFFERENT couplings (SC@S_E vs SC@S_I), so they
are inferred directly and independently (no r_FFI reparam — that workaround was
for the single-coupling FFI model where both gains scaled the same globalinput).
NO FIC: J_i is a fixed regional param, not FIC-tuned.
"""

import numpy as np


class ModelNotBuiltError(RuntimeError):
    """Raised when RWWEIB_2CPLSimGroup is not yet generated + compiled into cubnm."""


# RWWEIB_2CPL params + fallback defaults (mirror cuBNM/rww_eib_2cpl.yaml).
# g_LRE is the model's single global_param (shape (n_sims,)); everything else is
# a regional_param (shape (n_sims, nodes)). Inferred: g_LRE (global) +
# g_FFI/sigma/I_o (regional). Fixed regional: w_p/J_N/J_i/lambda_IE.
_GLOBAL_DEFAULT = {}                              # g_LRE promoted to regional (0 globals)
_RWWEIB2_REGIONAL_DEFAULT = {
    "g_LRE": 1.0,                                 # promoted global -> regional (per-node)
    "g_FFI": 1.0, "sigma": 0.01, "I_o": 0.382,
    "w_E": 1.0, "w_I": 0.7,                       # promoted from constants (inferrable)
    "w_p": 1.4, "J_N": 0.15, "J_i": 1.0, "lambda_IE": 1.0,
}

# All params are regional now (g_LRE promoted global->regional); any param in
# param_names is read per-sim from theta by the regional loop in build_param_lists.


def _import_rwweib2():
    """Import RWWEIB_2CPLSimGroup or raise a clear, actionable error."""
    try:
        from cubnm.sim import RWWEIB_2CPLSimGroup  # generated class
        return RWWEIB_2CPLSimGroup
    except Exception as e:  # noqa: BLE001
        raise ModelNotBuiltError(
            "RWWEIB_2CPLSimGroup is not available in the installed cubnm. Generate "
            "it from cuBNM/rww_eib_2cpl.yaml with the upstream cuBNM codegen driver "
            "(requires conn_state_vars=[S_E, S_I] / two-coupling support), then "
            "rebuild the extension from source with --no-build-isolation. "
            f"Underlying import error: {type(e).__name__}: {e}"
        ) from e


def _theta_column(theta, pn, name):
    if name in pn:
        return np.asarray(theta, dtype=np.float64)[:, pn.index(name)]
    return None


def build_param_lists(theta_batch, param_names, n_nodes, fixed=None):
    """Build RWWEIB_2CPL ``param_lists`` {name: (n_sims, n_nodes) float64}.

    Every regional param is set explicitly. Fixed defaults come from
    ``config.RWWEIB2_FIXED`` (falling back to ``_RWWEIB2_REGIONAL_DEFAULT``);
    any param present in ``param_names`` (incl g_LRE, now a regional_param) is
    taken per-sim from ``theta_batch`` and broadcast to all nodes. A precomputed
    per-(sim,node) ``<name>_matrix`` in ``fixed`` overrides node-by-node.
    """
    import config

    rwweib2_fixed = dict(getattr(config, "RWWEIB2_FIXED", {}) or {})
    if fixed:
        for k, v in dict(fixed).items():
            if k in _RWWEIB2_REGIONAL_DEFAULT:
                rwweib2_fixed[k] = v

    theta = np.asarray(theta_batch, dtype=np.float64)
    n_sims = theta.shape[0]
    pn = list(param_names)

    # Start from yaml defaults, overlay config.RWWEIB2_FIXED.
    regional = dict(_RWWEIB2_REGIONAL_DEFAULT)
    for k, v in rwweib2_fixed.items():
        if k in regional:
            regional[k] = float(v)

    # Broadcast every regional default to (n_sims, n_nodes).
    param_lists = {
        name: np.full((n_sims, n_nodes), dval, dtype=np.float64)
        for name, dval in regional.items()
    }

    # Override ANY regional param present in param_names with per-sim theta
    # (broadcast to nodes). This covers the inferred params (g_FFI/sigma/I_o)
    # and also lets a sensitivity sweep vary a normally-fixed regional param
    # (w_p/J_N/J_i/lambda_IE) by listing it in param_names.
    for name in regional:
        col = _theta_column(theta, pn, name)
        if col is not None:
            param_lists[name] = np.ascontiguousarray(
                np.repeat(col[:, None], n_nodes, axis=1), dtype=np.float64
            )

    # g_LRE is now a regional_param (promoted from global): handled by the
    # regional loop above (theta) and the matrix loop below (g_LRE_matrix).

    # Per-(sim,node) override matrices: any fixed key "<name>_matrix" with shape
    # (n_sims, n_nodes) replaces that regional param node-by-node. Covers FIC
    # (J_i_matrix) and HETEROGENEOUS region-wise params incl. g_LRE.
    if fixed:
        for name in regional:
            key = f"{name}_matrix"
            if key in fixed:
                arr = np.asarray(fixed[key], dtype=np.float64)
                if arr.shape != (n_sims, n_nodes):
                    raise ValueError(f"{key} shape {arr.shape} != ({n_sims},{n_nodes})")
                param_lists[name] = np.ascontiguousarray(arr)

    return param_lists


def run_cubnm_rwweib2_batch(weights, theta_batch, param_names,
                            duration_s=None, tr_s=None, dt_ms=None,
                            fixed=None, force_gpu=True, sim_seed=42,
                            burn_in_s=None, hrf="bw",
                            sc_dist=None, velocity=None):
    """Run an RWWEIB_2CPL batch — same signature/return as runner_rwweib.

    hrf : {"bw", "vbi"}
        "bw"  -> cuBNM built-in Balloon-Windkessel BOLD (fast).
        "vbi" -> take the neural S_E series and convolve with VBI's exact
                 MixtureOfGammas HRF (bold.BoldMonitor).
    Returns list[np.ndarray] — one (T, N) BOLD array per simulation.
    """
    RWWEIB_2CPLSimGroup = _import_rwweib2()
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

    grp = RWWEIB_2CPLSimGroup(
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

    # ── VBI MixtureOfGammas HRF on cuBNM neural S_E ───────────────────────
    from bold import BoldMonitor  # repo-root module (read-only use)

    # sim_states['S_E'] shape (N_sims, T_neural, nodes); full (un-trimmed) series.
    Se = np.asarray(grp.sim_states["S_E"], dtype=np.float32)  # (S, T, N)
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
