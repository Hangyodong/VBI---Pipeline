"""WC warmup and warm-start batch simulation.

Public API
----------
- WarmupResult              : stores warmed-up WC state + Bold Monitor
- warmup_run(...)           : run a warmup simulation
- simulate_with_warmup(...) : batch simulation starting from warmup

Why warmup
----------
The Bold Monitor needs the outer stock buffer (~5000 steps) filled
before it can emit valid BOLD frames. Without warmup, the first
``HRF_LENGTH_MS`` of every simulation is wasted on stock-fill. Warming
up once per subject and reusing the state across all training
simulations recovers that time.

BoldMonitor placement
---------------------
Like in ``wc_runner._run_streaming_hrf``, the BoldMonitor here is forced
to ``xp=np`` (CPU). The cupy NVRTC compilation issue with mismatched
CUDA headers would otherwise break warm-up too.
"""
import numpy as np

import config
from simulation.delays import apply_delay as _apply_delay
from simulation.wc_runner import _apply_engine, _import_wc


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class WarmupResult:
    """Stores the outcome of a warmup simulation.

    Attributes
    ----------
    x0           : array (2*N, S)  WC state after warmup (GPU)
    bold_monitor : BoldMonitor     with pre-filled stock buffers (CPU)
    sc           : np.ndarray (N, N)
    delays       : np.ndarray or None
    params       : dict            WC parameter dict used
    t_warmup_ms  : float           warmup duration (ms)
    """

    def __init__(self, x0, bold_monitor, sc, delays, params, t_warmup_ms):
        self.x0 = x0
        self.bold_monitor = bold_monitor
        self.sc = sc
        self.delays = delays
        self.params = params
        self.t_warmup_ms = t_warmup_ms

    def __repr__(self):
        nn = self.sc.shape[0]
        ns = self.x0.shape[1] if self.x0 is not None else 0
        stock_filled = self.bold_monitor._stock_count
        K = self.bold_monitor._K
        return (
            f"WarmupResult(\n"
            f"  t_warmup   = {self.t_warmup_ms:.0f} ms\n"
            f"  WC state   = {self.x0.shape}  (2*N={2*nn}, S={ns})\n"
            f"  stock      = {stock_filled}/{K} steps filled "
            f"({'ready' if stock_filled >= K else 'NOT ready'})\n"
            f"  params     = {list(self.params.keys())}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Warmup run
# ---------------------------------------------------------------------------

def warmup_run(sc, params_dict, t_warmup_ms=None, delays=None, ns=1):
    """Run a warmup simulation to stabilise WC state and BOLD monitor.

    Runs the WC model for ``t_warmup_ms`` ms with a single parameter set,
    filling the BoldMonitor's interim and outer stock buffers. The
    resulting ``WarmupResult`` can be passed to ``simulate_with_warmup``
    so that every subsequent batch simulation starts from a stable,
    pre-warmed state.

    Parameters
    ----------
    sc          : np.ndarray (N, N)  structural connectivity
    params_dict : dict               WC parameters (e.g. mean of prior)
    t_warmup_ms : float              warmup duration; defaults to
                                     ``config.HRF_LENGTH_MS * 2`` so the
                                     BOLD monitor stock fills completely.
    delays      : np.ndarray or None
    ns          : int                number of parallel sims (usually 1)

    Returns
    -------
    WarmupResult
    """
    from bold import BoldMonitor
    wc_cls = _import_wc()

    hrf_length_ms = getattr(config, "HRF_LENGTH_MS", 20_000.0)
    t_warmup_ms = t_warmup_ms or (hrf_length_ms * 2.0)

    params = dict(config.WC_FIXED)
    params["weights"] = sc.astype(np.float64)
    params["num_sim"] = ns
    params["t_end"] = t_warmup_ms
    params["t_cut"] = 0.0       # no transient cut during warmup
    _apply_engine(params)
    params["seed"] = None
    for k, v in params_dict.items():
        params[k] = float(v)
    _apply_delay(params, delays)

    model = wc_cls(params)
    model.prepare_input()
    model.set_initial_state()

    dt_full = config.DT
    n_steps = int(np.ceil(t_warmup_ms / dt_full))
    nn = sc.shape[0]

    mon = BoldMonitor(
        nn=nn, ns=ns,
        dt_ms=dt_full,
        xp=np,                          # force CPU
        period_ms=config.TR_SEC * 1000.0,
        hrf_length_ms=hrf_length_ms,
        verbose=True,
    )

    print(
        f"  [warmup] t={t_warmup_ms:.0f}ms  "
        f"n_steps={n_steps}  "
        f"stock_target={mon._K} steps"
    )

    for i in range(n_steps):
        t_curr = i * dt_full
        model.x0 = model.heunStochastic(model.x0, t_curr)
        E_w = model.x0[:nn, :]
        E_cpu = E_w.get() if hasattr(E_w, "get") else np.asarray(E_w)
        mon.step(i, E_cpu, t_cut_ms=0.0)

    result = WarmupResult(
        x0=model.x0,
        bold_monitor=mon,
        sc=sc,
        delays=delays,
        params=params_dict,
        t_warmup_ms=t_warmup_ms,
    )

    print(
        f"  [warmup] done  "
        f"stock filled={mon._stock_count}/{mon._K}  "
        f"({'ready' if mon._stock_count >= mon._K else 'NOT ready'})"
    )
    return result


# ---------------------------------------------------------------------------
# Warm-started batch simulation
# ---------------------------------------------------------------------------

def simulate_with_warmup(warmup, theta_batch, param_names,
                         fixed_overrides=None):
    """Simulate a batch starting from a pre-warmed state.

    Each simulation in the batch starts from ``warmup.x0`` and uses a
    cloned BoldMonitor with the same stock contents.

    Per-simulation parameter arrays are used (never batch mean). A shape
    assert catches regressions.

    Parameters
    ----------
    warmup          : WarmupResult
    theta_batch     : np.ndarray (B, n_params)
    param_names     : list[str]
    fixed_overrides : dict or None

    Returns
    -------
    list of (T_bold, N) BOLD arrays, one per simulation in the batch
    """
    from bold import BoldMonitor
    wc_cls = _import_wc()

    nn = warmup.sc.shape[0]
    B = len(theta_batch)
    dt_full = config.DT
    hrf_length_ms = getattr(config, "HRF_LENGTH_MS", 20_000.0)
    t_end_ms = config.T_END
    t_cut_ms = config.T_CUT
    n_steps = int(np.ceil(t_end_ms / dt_full))

    overrides = dict(fixed_overrides or {})
    theta_batch_arr = np.asarray(theta_batch, dtype=np.float32)

    params = dict(config.WC_FIXED)
    params["weights"] = warmup.sc.astype(np.float64)
    params["num_sim"] = B
    params["t_end"] = t_end_ms
    params["t_cut"] = t_cut_ms
    _apply_engine(params)
    params["seed"] = None
    params.update(overrides)
    # Per-simulation parameter arrays (replaces batch-mean)
    for i, name in enumerate(param_names):
        params[name] = theta_batch_arr[:, i].astype(np.float32)
    # Sanity check
    for name in param_names:
        v = params[name]
        assert hasattr(v, "shape") and v.shape == (B,), (
            f"simulate_with_warmup: per-sim param {name!r} has shape "
            f"{getattr(v, 'shape', None)}, expected ({B},)."
        )
    _apply_delay(params, warmup.delays)

    model = wc_cls(params)
    model.prepare_input()

    # Start from warmed-up x0 (broadcast to B sims, GPU-safe)
    x0_warm = warmup.x0
    if x0_warm.shape[1] == 1:
        if hasattr(x0_warm, "get"):
            import cupy as _cp
            x0_warm = _cp.broadcast_to(x0_warm, (2 * nn, B)).copy()
        else:
            x0_warm = np.broadcast_to(x0_warm, (2 * nn, B)).copy()
    model.x0 = x0_warm

    # Clone BoldMonitor — always CPU (numpy)
    w_mon = warmup.bold_monitor
    mon = BoldMonitor(
        nn=nn, ns=B,
        dt_ms=dt_full,
        xp=np,                          # force CPU
        period_ms=config.TR_SEC * 1000.0,
        hrf_length_ms=hrf_length_ms,
    )
    # Copy stock from warmup (already numpy after warmup_run fix)
    if w_mon._stock.shape[2] == 1:
        mon._stock = np.broadcast_to(
            w_mon._stock, (w_mon._K, nn, B)
        ).copy()
    else:
        mon._stock = w_mon._stock.copy()
    mon._stock_pos = w_mon._stock_pos
    mon._stock_count = w_mon._stock_count
    mon._interim_stock = np.zeros_like(mon._interim_stock)

    for i in range(n_steps):
        t_curr = i * dt_full
        model.x0 = model.heunStochastic(model.x0, t_curr)
        E_s = model.x0[:nn, :]
        E_cpu = E_s.get() if hasattr(E_s, "get") else np.asarray(E_s)
        mon.step(i, E_cpu, t_cut_ms=t_cut_ms)

    bold = mon.collect(mean_subtract=True)   # (T_bold, N, B)
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return [bold[:, :, i] for i in range(B)]
