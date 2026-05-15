"""Wilson-Cowan simulation + feature extraction.

Provides VBI WC backend wrappers, delay computation, and feature
extractors for both simulated and observed data.

Feature dimensions (115 region)
-------------------------------
FC  : 6555 upper triangle (115 * 114 / 2)
FCD : 6555 upper triangle of the FCD-like matrix

Observed data
-------------
FC  -> file column 1 (NaN -> 0)
FCD -> file column 2 (used directly; no computation)

Simulated data
--------------
FC  -> compute_fc(BOLD)            (Pearson over the full BOLD window)
FCD -> compute_sim_fcd_matrix(BOLD) (element-wise std of sliding-window FCs)
"""
import numpy as np

import config
from bold import balloon_windkessel


# ---------------------------------------------------------------------------
# VBI WC import (lazy)
# ---------------------------------------------------------------------------

_WC_SDE_CLASS = None


def _import_wc():
    """Import the cupy WC class on first use."""
    global _WC_SDE_CLASS
    if _WC_SDE_CLASS is None:
        from vbi.models.cupy.wilson_cowan import WC_sde
        _WC_SDE_CLASS = WC_sde
    return _WC_SDE_CLASS


def to_numpy(x):
    """Move cupy arrays to numpy, leave numpy untouched."""
    return x.get() if hasattr(x, "get") else np.asarray(x)


def normalize_ts(ts, n_nodes, num_sim=None):
    """Reshape a VBI WC output to (T, N) or (T, N, S)."""
    ts = to_numpy(ts)
    if ts.ndim == 3:
        candidates = [(1, 2), (2, 1), (0, 2), (2, 0), (0, 1), (1, 0)]
        for an, as_ in candidates:
            if ts.shape[an] == n_nodes and (
                num_sim is None or ts.shape[as_] == num_sim
            ):
                at = 3 - an - as_
                return np.transpose(ts, (at, an, as_))
        if ts.shape[1] == n_nodes:
            return ts
        return np.transpose(ts, (1, 2, 0))
    if ts.ndim == 2 and ts.shape[0] == n_nodes:
        return ts.T
    return ts


# ---------------------------------------------------------------------------
# Delay matrix
# ---------------------------------------------------------------------------

def compute_delay_matrix(weights, velocity_m_per_s, lengths_mm=None):
    """Convert tract weights into millisecond delays."""
    if velocity_m_per_s is None or velocity_m_per_s <= 0:
        return None
    if lengths_mm is None:
        eps = 1e-3
        lengths_mm = 1.0 / (weights + eps)
        np.fill_diagonal(lengths_mm, 0)
        if lengths_mm.max() > 0:
            # Scale to mouse-typical mean ~10 mm
            lengths_mm = lengths_mm / lengths_mm.max() * 10.0
    delays = lengths_mm / float(velocity_m_per_s)
    np.fill_diagonal(delays, 0)
    return delays.astype(np.float64)


# ---------------------------------------------------------------------------
# VBI delay key detection
# ---------------------------------------------------------------------------

_DELAY_KEY = None
_DELAY_KEY_CHECKED = False
_ENGINE_KEY = None
_ENGINE_KEY_CHECKED = False


def detect_delay_key():
    """Find the parameter name that the WC class accepts for delays."""
    global _DELAY_KEY, _DELAY_KEY_CHECKED
    if _DELAY_KEY_CHECKED:
        return _DELAY_KEY
    _DELAY_KEY_CHECKED = True
    try:
        wc_cls = _import_wc()
        m = wc_cls({})
        valid = set(getattr(m, "valid_params", []))
        for key in ("delays", "delay_matrix", "tract_lengths"):
            if key in valid:
                _DELAY_KEY = key
                return key
        for key in ("velocity", "speed", "conduction_velocity"):
            if key in valid:
                _DELAY_KEY = key
                return key
    except Exception:
        pass
    return None


def detect_engine_key():
    """Find the parameter key that VBI WC uses for GPU/CPU selection."""
    global _ENGINE_KEY, _ENGINE_KEY_CHECKED
    if _ENGINE_KEY_CHECKED:
        return _ENGINE_KEY
    _ENGINE_KEY_CHECKED = True
    try:
        wc_cls = _import_wc()
        m = wc_cls({})
        valid = set(getattr(m, "valid_params", []))
        for key in ("engine", "backend", "device", "use_gpu", "mode"):
            if key in valid:
                _ENGINE_KEY = key
                return key
    except Exception:
        pass
    return None


def _apply_engine(params):
    """Inject the GPU engine key into WC parameters."""
    key = detect_engine_key()
    if key is None:
        # valid_params에 없으면 그냥 'engine' 키로 시도
        params["engine"] = config.ENGINE
    else:
        params[key] = config.ENGINE


def _apply_delay(params, delays_precomputed):
    """Inject the delay into VBI WC parameters using the detected key."""
    key = detect_delay_key()
    if key is None or delays_precomputed is None:
        return
    if key in ("delays", "delay_matrix", "tract_lengths"):
        params[key] = delays_precomputed
    elif key in ("velocity", "speed", "conduction_velocity"):
        params[key] = float(config.VELOCITY_M_PER_S)


# ---------------------------------------------------------------------------
# GPU batch simulation
# ---------------------------------------------------------------------------

def _run_streaming_hrf(model, n_nodes, num_sim, dt_ms, apply_bw):
    """Run VBI WC step-by-step with TVB Bold Monitor on GPU.

    Uses ``bold.BoldMonitor`` which mirrors the exact TVB Bold Monitor
    algorithm (interim stock → outer stock → HRF dot-product at TR).

    Returns
    -------
    bold : np.ndarray  (T_bold, N, S)   if apply_bw=True
    e    : np.ndarray  (T_stored, N, S) if apply_bw=False
    """
    from bold import BoldMonitor

    xp = model.xp
    dt_full = model.dt       # integration step (ms)
    t_cut = model.t_cut      # transient cutoff (ms)
    decimate = model.decimate
    nn = n_nodes
    ns = num_sim

    n_steps = int(np.ceil(model.t_end / dt_full))

    if not apply_bw:
        # accumulate decimated E in RAM only
        valid = int(np.floor((model.t_end - t_cut) / (dt_full * decimate)))
        e_out = np.zeros((valid, nn, ns), dtype=np.float32)
        buf_idx = 0
        for i in range(n_steps):
            t_curr = i * dt_full
            model.x0 = model.heunStochastic(model.x0, t_curr)
            if t_curr > t_cut and i % decimate == 0 and buf_idx < valid:
                E_i = model.x0[:nn, :]
                e_out[buf_idx] = (
                    E_i.get() if hasattr(E_i, "get") else np.asarray(E_i)
                )
                buf_idx += 1
        return e_out[:buf_idx]

    # TVB Bold Monitor — runs on GPU (xp = cupy)
    mon = BoldMonitor(
        nn=nn, ns=ns,
        dt_ms=dt_full,              # integration step, NOT decimated
        xp=xp,
        period_ms=config.TR_SEC * 1000.0,
        hrf_length_ms=getattr(config, "HRF_LENGTH_MS", 20_000.0),
    )

    for i in range(n_steps):
        t_curr = i * dt_full
        model.x0 = model.heunStochastic(model.x0, t_curr)
        E_i = model.x0[:nn, :]     # (N, S) on GPU
        mon.step(i, E_i, t_cut_ms=t_cut)

    return mon.collect(mean_subtract=True)


def _try_per_sim_params(params, chunk, param_names):
    """Try to set per-simulation parameter arrays.

    Returns
    -------
    success : bool   True if WC accepted per-sim arrays
    """
    # Each parameter becomes an array of length num_sim.
    for i, name in enumerate(param_names):
        params[name] = chunk[:, i].astype(np.float32)
    return True


def simulate_gpu_batch(weights, theta_batch, param_names,
                       fixed_overrides=None, delays=None, apply_bw=True,
                       _allow_fallback=True):
    """Simulate a batch of parameter sets on the GPU.

    Each row of `theta_batch` corresponds to one simulation
    (num_sim = len(chunk)). Parameters are passed as per-simulation
    arrays so the (theta_i, x_i) training pairs stay correctly aligned.

    If the underlying WC implementation does NOT support per-simulation
    parameter arrays, the function falls back to a per-theta loop
    (each row becomes a separate num_sim=1 simulation). This is slower
    but preserves label-feature alignment.

    Returns a list of BOLD arrays (one per simulation).
    """
    import cupy as cp
    wc_cls = _import_wc()

    overrides = dict(fixed_overrides or {})
    n_nodes = weights.shape[0]
    n_total = len(theta_batch)
    outputs = []
    batch_sz = config.GPU_BATCH
    dt_ms = config.DT * config.DECIMATE

    # Verify per-sim parameter support on the very first chunk.
    # If it fails (TypeError / shape error) we fall back to per-theta loop.
    array_param_supported = None

    for start in range(0, n_total, batch_sz):
        end = min(start + batch_sz, n_total)
        chunk = np.asarray(theta_batch[start:end], dtype=np.float32)
        csz = len(chunk)

        params = dict(config.WC_FIXED)
        params["weights"] = weights.astype(np.float64)
        params["num_sim"] = csz
        _apply_engine(params)
        params["seed"] = None
        params.update(overrides)
        _apply_delay(params, delays)

        # Per-simulation parameter arrays (CRITICAL — replaces batch mean)
        _try_per_sim_params(params, chunk, param_names)

        # Sanity check: parameter values must be vectors of length csz,
        # not scalars. Catches accidental regressions to batch-mean.
        for name in param_names:
            v = params[name]
            assert hasattr(v, "shape") and v.shape == (csz,), (
                f"Per-simulation parameter {name!r} has wrong shape "
                f"{getattr(v, 'shape', None)}, expected ({csz},). "
                "This guards against batch-mean regressions."
            )

        try:
            model = wc_cls(params)
            model.prepare_input()
            model.set_initial_state()
            result = _run_streaming_hrf(
                model, n_nodes, csz, dt_ms, apply_bw=apply_bw,
            )
            if array_param_supported is None:
                array_param_supported = True
        except Exception as e:
            if not _allow_fallback or array_param_supported is True:
                raise
            print(
                f"  ⚠ Per-sim parameter arrays not supported by WC "
                f"({type(e).__name__}: {e}). "
                f"Falling back to per-theta loop (slower but correct)."
            )
            array_param_supported = False
            # Fallback: run each theta as a separate num_sim=1 simulation
            for r in range(csz):
                p_single = dict(config.WC_FIXED)
                p_single["weights"] = weights.astype(np.float64)
                p_single["num_sim"] = 1
                _apply_engine(p_single)
                p_single["seed"] = None
                p_single.update(overrides)
                _apply_delay(p_single, delays)
                for i, name in enumerate(param_names):
                    p_single[name] = float(chunk[r, i])
                m_single = wc_cls(p_single)
                m_single.prepare_input()
                m_single.set_initial_state()
                r_single = _run_streaming_hrf(
                    m_single, n_nodes, 1, dt_ms, apply_bw=apply_bw,
                )
                outputs.append(r_single[:, :, 0])
            cp.get_default_memory_pool().free_all_blocks()
            continue

        # Success path: split per-sim outputs
        for i in range(csz):
            outputs.append(result[:, :, i])

        cp.get_default_memory_pool().free_all_blocks()

    return outputs


def simulate_single(weights, params_dict, n_repeat=1, delays=None,
                    apply_bw=True):
    """Simulate one parameter set, optionally repeated `n_repeat` times.

    Uses the same streaming BW as simulate_gpu_batch — neural time-series
    never accumulate in RAM.
    """
    import cupy as cp
    wc_cls = _import_wc()

    params = dict(config.WC_FIXED)
    params["weights"] = weights.astype(np.float64)
    params["num_sim"] = n_repeat
    _apply_engine(params)
    params["seed"] = None
    for k, v in params_dict.items():
        params[k] = float(v)
    _apply_delay(params, delays)

    model = wc_cls(params)
    model.prepare_input()
    model.set_initial_state()

    dt_ms = config.DT * config.DECIMATE
    result = _run_streaming_hrf(
        model, weights.shape[0], n_repeat, dt_ms, apply_bw=apply_bw,
    )
    # result shape: (T_bold, N, S)
    out = [result[:, :, i] for i in range(n_repeat)]

    cp.get_default_memory_pool().free_all_blocks()
    return out


# ---------------------------------------------------------------------------
# Warmup simulation
# ---------------------------------------------------------------------------

class WarmupResult:
    """Stores the outcome of a warmup simulation.

    Attributes
    ----------
    x0        : array  (2*N, S)  WC state after warmup (on GPU)
    bold_monitor : BoldMonitor   monitor with pre-filled stock buffers
    sc        : np.ndarray (N, N)  structural connectivity used
    delays    : np.ndarray or None
    params    : dict              WC parameter dict used
    t_warmup_ms : float           warmup duration (ms)
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


def warmup_run(sc, params_dict, t_warmup_ms=None, delays=None, ns=1):
    """Run a warmup simulation to stabilise WC state and BOLD monitor.

    Runs the WC model for ``t_warmup_ms`` ms with a single parameter set,
    filling the BoldMonitor's interim and outer stock buffers. The resulting
    ``WarmupResult`` can be passed to ``simulate_gpu_batch`` so that every
    subsequent batch simulation starts from a stable, pre-warmed state.

    Parameters
    ----------
    sc          : np.ndarray (N, N)  structural connectivity
    params_dict : dict               WC parameters (e.g. mean of prior)
    t_warmup_ms : float              warmup duration; default
                                     ``config.HRF_LENGTH_MS * 2`` so that
                                     the BOLD monitor stock is fully filled
    delays      : np.ndarray or None
    ns          : int                number of parallel sims (usually 1)

    Returns
    -------
    WarmupResult
    """
    import cupy as cp
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
        xp=model.xp,
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
        mon.step(i, model.x0[:nn, :], t_cut_ms=0.0)

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


def simulate_with_warmup(warmup, theta_batch, param_names,
                         fixed_overrides=None):
    """Simulate a batch starting from a pre-warmed state.

    Each simulation in the batch starts from ``warmup.x0`` and uses a
    cloned BoldMonitor with the same stock contents.

    Parameters
    ----------
    warmup       : WarmupResult
    theta_batch  : np.ndarray  (B, n_params)
    param_names  : list[str]
    fixed_overrides : dict or None

    Returns
    -------
    list of (T_bold, N) BOLD arrays, one per simulation in the batch
    """
    import cupy as cp
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

    # Start from warmed-up x0 (broadcast to B sims)
    x0_warm = warmup.x0                    # (2*N, 1) or (2*N, ns_warm)
    if x0_warm.shape[1] == 1:
        import cupy as cp_mod
        x0_warm = cp_mod.broadcast_to(x0_warm, (2 * nn, B)).copy()
    model.x0 = x0_warm

    # Clone BoldMonitor with same stock state, but ns=B
    w_mon = warmup.bold_monitor
    mon = BoldMonitor(
        nn=nn, ns=B,
        dt_ms=dt_full,
        xp=model.xp,
        period_ms=config.TR_SEC * 1000.0,
        hrf_length_ms=hrf_length_ms,
    )
    # Copy stock from warmup (broadcast ns dimension)
    if w_mon._stock.shape[2] == 1:
        import cupy as cp_mod
        mon._stock = cp_mod.broadcast_to(
            w_mon._stock, (w_mon._K, nn, B)
        ).copy()
    else:
        mon._stock = w_mon._stock.copy()
    mon._stock_pos = w_mon._stock_pos
    mon._stock_count = w_mon._stock_count
    mon._interim_stock = model.xp.zeros_like(mon._interim_stock)

    for i in range(n_steps):
        t_curr = i * dt_full
        model.x0 = model.heunStochastic(model.x0, t_curr)
        mon.step(i, model.x0[:nn, :], t_cut_ms=t_cut_ms)

    bold = mon.collect(mean_subtract=True)   # (T_bold, N, B)
    cp.get_default_memory_pool().free_all_blocks()
    return [bold[:, :, i] for i in range(B)]


# ---------------------------------------------------------------------------
# Compatibility wrappers — feature functions moved to ``features/``
# (Phase 1 of repository refactor.)
#
# Existing imports like::
#     from simulator import compute_fc, fc_to_upper_tri
#     from simulator import extract_features, worker_extract
# continue to work via these re-exports. New code should import directly
# from the ``features`` package.
# ---------------------------------------------------------------------------
from features.fc import (                # noqa: E402, F401
    compute_fc,
    fc_to_upper_tri,
)
from features.fcd import (               # noqa: E402, F401
    compute_sim_fcd_matrix,
    fcd_to_summary_stats,
    fcd_to_upper_tri,
)
from features.extraction import (        # noqa: E402, F401
    extract_features,
    extract_observed_features,
    extract_simulated_features,
    worker_extract,
)
