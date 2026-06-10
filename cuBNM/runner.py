"""Thin adapter: run cuBNM ``WCSimGroup`` with VBI-style SC/theta inputs.

The current VBI engine entry point is::

    simulation.wc_runner.simulate_gpu_batch(weights, theta_batch, param_names)
        -> list of (T, N) BOLD arrays

This module exposes the equivalent::

    cuBNM.runner.run_cubnm_batch(weights, theta_batch, param_names)
        -> list of (T, N) BOLD arrays

so the two engines can be benchmarked on the same inputs.

Design constraints
------------------
- ``cubnm`` is imported lazily (inside functions only), so this module is
  importable without a GPU / without cubnm installed.
- No top-level import from simulation/, features/, inference/, etc.
- ``config`` is read lazily for WC_FIXED defaults.

Parameter mapping (VBI WC_sde  ->  cuBNM WCSimGroup)
---------------------------------------------------
cuBNM WC has ONE global coupling ``G`` (long-range, E only) and regional
``c_EE, c_IE, c_EI, c_II, P_E, P_I, sigma_E, sigma_I``. The VBI inferred
params are (g_e, g_i, c_ei). Mapping used here:

    g_e   -> G          (global coupling on SC @ E)
    c_ei  -> c_EI       (E->I coupling, inferred)
    g_i   -> (NO cuBNM equivalent: cuBNM has a single global coupling)
    WC_FIXED c_ee  -> c_EE
    WC_FIXED c_ie  -> c_IE
    WC_FIXED c_ii  -> c_II
    WC_FIXED P     -> P_E
    WC_FIXED Q     -> P_I
    WC_FIXED noise_amp -> sigma_E, sigma_I
    (tau_e/tau_i, alpha_*, theta_* are NOT exposed by cuBNM WC -> dropped)

See module ``benchmark`` and the notebook cell for the uncertainty notes.
"""
import numpy as np


# VBI WC_FIXED key  ->  cuBNM regional param key.
#
# Semantic fix (condition-equalization audit): the VBI WC-EIB equations are
#   x_e = ... - c_ei * I ...   (c_ei = I->E inhibition)
#   x_i = ...   c_ie * E ...   (c_ie = E->I excitation)
# cuBNM WC uses
#   S^E[... - c_IE * I ...]    (c_IE = I->E)
#   S^I[  c_EI * E ...]        (c_EI = E->I)
# so the correct mapping is  c_ei -> c_IE  and  c_ie -> c_EI.
# The previous mapping (c_ei -> c_EI, c_ie -> c_IE) swapped them, injecting the
# inferred I->E coupling into cuBNM's E->I slot. ``legacy_ei_swap=True`` keeps
# the old (wrong) mapping only for before/after A-B comparison in the notebook.
_FIXED_TO_CUBNM = {
    "c_ee": "c_EE",
    "c_ie": "c_EI",   # VBI c_ie (E->I)  ==  cuBNM c_EI
    "c_ii": "c_II",
    "P":    "P_E",
    "Q":    "P_I",
}

# Old, semantically-wrong mapping (c_ie -> c_IE); inferred c_ei -> c_EI.
_FIXED_TO_CUBNM_LEGACY = {
    "c_ee": "c_EE",
    "c_ie": "c_IE",
    "c_ii": "c_II",
    "P":    "P_E",
    "Q":    "P_I",
}

# cuBNM regional params + their fallback defaults (cubnm WC default_params)
_CUBNM_REGIONAL = {
    "c_EE": 16.0, "c_IE": 12.0, "c_EI": 15.0, "c_II": 3.0,
    "P_E": 0.5, "P_I": 0.0, "sigma_E": 5e-5, "sigma_I": 5e-5,
}


def _theta_column(theta_batch, param_names, name):
    """Return the (n_sims,) column for ``name`` or None if absent."""
    if name in param_names:
        return np.asarray(theta_batch, dtype=np.float64)[:, param_names.index(name)]
    return None


def build_param_lists(theta_batch, param_names, n_nodes, fixed=None,
                      legacy_ei_swap=False):
    """Build the cuBNM ``param_lists`` dict from a VBI theta batch.

    ``legacy_ei_swap=True`` reproduces the old (semantically wrong) c_ei/c_ie
    mapping so the notebook can show the FC diff before vs after the fix.

    Returns
    -------
    dict
        {'G': (n_sims,), 'c_EE': (n_sims, n_nodes), ...} float64.
    """
    import config

    fixed = dict(getattr(config, "WC_FIXED", {}))
    if isinstance(fixed, dict):
        fixed = dict(fixed)
    theta = np.asarray(theta_batch, dtype=np.float64)
    n_sims = theta.shape[0]
    pn = list(param_names)

    # Mapping selection: correct vs legacy (wrong) c_ei/c_ie convention.
    fixed_to_cubnm = _FIXED_TO_CUBNM_LEGACY if legacy_ei_swap else _FIXED_TO_CUBNM
    # Inferred c_ei target slot: I->E (c_IE) is correct; legacy used E->I (c_EI).
    cei_target = "c_EI" if legacy_ei_swap else "c_IE"

    # Per-node regional defaults from WC_FIXED (mapped) or cubnm defaults.
    regional_default = dict(_CUBNM_REGIONAL)
    for vbi_key, cub_key in fixed_to_cubnm.items():
        if vbi_key in config.WC_FIXED:
            regional_default[cub_key] = float(config.WC_FIXED[vbi_key])
    if "noise_amp" in config.WC_FIXED:
        regional_default["sigma_E"] = float(config.WC_FIXED["noise_amp"])
        regional_default["sigma_I"] = float(config.WC_FIXED["noise_amp"])

    param_lists = {}

    # Global coupling G  <-  g_e (fallback 0.5)
    g_e = _theta_column(theta, pn, "g_e")
    if g_e is None:
        g_e = np.full(n_sims, 0.5, dtype=np.float64)
    param_lists["G"] = np.ascontiguousarray(g_e, dtype=np.float64)

    # Regional params: start from defaults broadcast to (n_sims, n_nodes)
    for cub_key, dval in _CUBNM_REGIONAL.items():
        param_lists[cub_key] = np.full(
            (n_sims, n_nodes), regional_default[cub_key], dtype=np.float64
        )

    # Inferred c_ei (I->E) -> cuBNM c_IE (per-sim scalar broadcast to all nodes)
    c_ei = _theta_column(theta, pn, "c_ei")
    if c_ei is not None:
        param_lists[cei_target] = np.ascontiguousarray(
            np.repeat(c_ei[:, None], n_nodes, axis=1), dtype=np.float64
        )

    return param_lists


def run_cubnm_batch(weights, theta_batch, param_names,
                    duration_s=None, tr_s=None, dt_ms=None,
                    fixed=None, force_gpu=True, sim_seed=42,
                    burn_in_s=None, legacy_ei_swap=False):
    """Run a batch of WC simulations through cuBNM ``WCSimGroup``.

    Parameters
    ----------
    weights : (N, N) array
        SC coupling matrix (same one passed to simulate_gpu_batch).
    theta_batch : (n_sims, n_params) array
        Per-sim inferred parameters, columns ordered by ``param_names``.
    param_names : list[str]
        e.g. ``config.STAGE1_PARAMS`` == ['g_e', 'g_i', 'c_ei'].
    duration_s : float | None
        TOTAL sim length (seconds), == config.T_END/1000 when None. cuBNM
        ``WCSimGroup(duration=...)`` is the full length (NOT post-burn-in).
    tr_s : float | None
        BOLD TR (seconds); config.TR_SEC when None.
    dt_ms : float | None
        Integration step (ms); config.DT when None (0.5 ms, matches VBI).
    burn_in_s : float | None
        Transient discarded from BOLD (seconds) -> ``bold_remove_s``.
        config.T_CUT/1000 when None. duration - burn_in == analyzed length.
    sim_seed : int
        Noise seed (default 42 to match the VBI benchmark seed).
    legacy_ei_swap : bool
        If True, use the old (wrong) c_ei->c_EI / c_ie->c_IE mapping. For
        before/after FC comparison only.

    Returns
    -------
    list[np.ndarray]
        One (T, N) BOLD array per simulation — same shape contract as
        ``simulate_gpu_batch``.
    """
    from cubnm.sim import WCSimGroup  # lazy
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

    grp = WCSimGroup(
        duration=float(duration_s),
        TR=float(tr_s),
        sc=weights,
        dt=str(dt_ms),
        bold_remove_s=float(burn_in_s),
        do_fc=True,
        do_fcd=False,
        gof_terms=["+fc_corr"],
        force_gpu=bool(force_gpu),
        sim_seed=int(sim_seed),
    )
    grp.N = n_sims  # number of parallel simulations

    param_lists = build_param_lists(
        theta, list(param_names), n_nodes, fixed,
        legacy_ei_swap=legacy_ei_swap,
    )
    for k, v in param_lists.items():
        grp.param_lists[k] = v

    grp.run()

    # sim_bold is reshaped to (N_sims, T, n_nodes); split into per-sim (T, N)
    sim_bold = np.asarray(grp.sim_bold)
    if sim_bold.ndim == 3:
        return [np.ascontiguousarray(sim_bold[i]) for i in range(sim_bold.shape[0])]
    # Defensive: single-sim flat case
    return [np.ascontiguousarray(sim_bold.reshape(-1, n_nodes))]
