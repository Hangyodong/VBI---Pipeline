"""Runner for cuBNM's STOCK reduced Wong-Wang (``rWWSimGroup``, Deco 2014).

Unlike RWWEIB, ``rWWSimGroup`` ships compiled in cubnm (no build needed). This
adapter mirrors ``runner_rwweib.py`` but drives the stock rWW:

  global_param   : G                        (long-range excitatory coupling)
  regional_params: w_p, J_N, wIE, sigma
  feedback inhibition control (FIC): config.RWW_DO_FIC (default True). When on,
  wIE is auto-tuned by cubnm (NOT inferred); when off, wIE is a fixed/inferred
  regional param.

theta -> param_lists mapping (by config.STAGE1_PARAMS):
  G               -> global   (n_sims,)
  w_p, J_N, sigma -> regional  (n_sims, nodes)   [inferred if in theta, else fixed]
  wIE             -> regional  (only when do_fic is False)
"""
import numpy as np

# Stock rWW regional defaults (cubnm rww.yaml).
_RWW_REGIONAL_DEFAULT = {"w_p": 1.4, "J_N": 0.15, "wIE": 1.0, "sigma": 0.01}
_GLOBAL_DEFAULT = {"G": 1.0}


def _import_rww():
    from cubnm.sim import rWWSimGroup       # stock model, always present
    return rWWSimGroup


def _theta_column(theta, pn, name):
    if name in pn:
        return np.asarray(theta, dtype=np.float64)[:, pn.index(name)]
    return None


def build_param_lists(theta_batch, param_names, n_nodes, fixed=None, do_fic=True):
    """Build stock-rWW ``param_lists`` from a theta batch.

    G is the global_param (n_sims,); w_p/J_N/wIE/sigma are regional. With
    do_fic=True, wIE is left to cubnm's FIC (not set here).
    """
    import config

    rww_fixed = dict(getattr(config, "RWW_FIXED", {}) or {})
    if fixed:
        for k, v in dict(fixed).items():
            if k in _RWW_REGIONAL_DEFAULT or k == "G":
                rww_fixed[k] = v

    theta = np.asarray(theta_batch, dtype=np.float64)
    n_sims = theta.shape[0]
    pn = list(param_names)

    regional = dict(_RWW_REGIONAL_DEFAULT)
    for k, v in rww_fixed.items():
        if k in regional:
            regional[k] = float(v)

    param_lists = {}
    # Regional params (broadcast fixed defaults, then override with theta).
    for name, dval in regional.items():
        # When FIC is on, cubnm sets wIE itself — skip it here.
        if name == "wIE" and do_fic:
            continue
        col = _theta_column(theta, pn, name)
        if col is not None:
            param_lists[name] = np.ascontiguousarray(
                np.repeat(col[:, None], n_nodes, axis=1), dtype=np.float64
            )
        else:
            param_lists[name] = np.full((n_sims, n_nodes), dval, dtype=np.float64)

    # Global coupling G.
    g = _theta_column(theta, pn, "G")
    if g is None:
        g = np.full(n_sims, float(rww_fixed.get("G", _GLOBAL_DEFAULT["G"])),
                    dtype=np.float64)
    param_lists["G"] = np.ascontiguousarray(g, dtype=np.float64)

    return param_lists


def run_cubnm_rww_batch(weights, theta_batch, param_names,
                        duration_s=None, tr_s=None, dt_ms=None,
                        fixed=None, force_gpu=True, sim_seed=42,
                        burn_in_s=None, hrf="bw",
                        sc_dist=None, velocity=None):
    """Run a stock-rWW batch — same signature/return as runner_rwweib."""
    rWWSimGroup = _import_rww()
    import config

    do_fic = bool(getattr(config, "RWW_DO_FIC", True))

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

    grp = rWWSimGroup(
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
        do_fic=do_fic,
    )
    grp.N = n_sims

    param_lists = build_param_lists(theta, list(param_names), n_nodes, fixed, do_fic)
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

    # ── VBI MixtureOfGammas HRF on cuBNM neural r_E ───────────────────────
    from bold import BoldMonitor
    rE = np.asarray(grp.sim_states["r_E"], dtype=np.float32)   # (S, T, N)
    n_steps = rE.shape[1]
    mon = BoldMonitor(
        nn=n_nodes, ns=n_sims, dt_ms=neural_dt_ms, xp=np, use_gpu=False,
        period_ms=float(tr_s) * 1000.0,
        hrf_length_ms=getattr(config, "HRF_LENGTH_MS", 32_000.0),
    )
    t_cut_ms = float(burn_in_s) * 1000.0
    for i in range(n_steps):
        mon.step(i, rE[:, i, :].T, t_cut_ms=t_cut_ms)
    bold = mon.collect(mean_subtract=True)
    return [np.ascontiguousarray(bold[:, :, s]) for s in range(bold.shape[2])]
