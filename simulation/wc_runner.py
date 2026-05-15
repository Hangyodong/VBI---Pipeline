"""VBI Wilson-Cowan execution core.

Public API
----------
- simulate_gpu_batch(weights, theta_batch, param_names, ...) -> list[BOLD]
- simulate_single(weights, params_dict, n_repeat, delays, apply_bw)
- _import_wc()           : lazy import of vbi.models.cupy.wilson_cowan.WC_sde
- detect_engine_key()    : auto-detect VBI engine parameter name
- to_numpy(x)            : cupy -> numpy
- normalize_ts(...)      : reshape WC raw output to (T, N) or (T, N, S)
- _run_streaming_hrf(...): step-by-step integration + Bold Monitor

Critical invariants
-------------------
1. **Per-simulation parameters, never batch mean.** Each row of
   ``theta_batch`` corresponds to exactly one simulation. ``params[name]``
   is a vector of length num_sim, not a scalar. A shape assert catches
   any regression to ``chunk[:, i].mean()`` patterns.
2. **Fallback preserves alignment.** If VBI cannot accept per-sim
   parameter arrays, we drop to a per-theta loop with num_sim=1. This is
   slower but the (theta_i, BOLD_i) pair stays correct.
3. **BoldMonitor runs on CPU (numpy).** WC heunStochastic stays on GPU
   (cupy); E values are transferred to CPU at each interim step. This
   avoids cupy NVRTC compilation issues with mismatched CUDA headers.
"""
import numpy as np

import config
from simulation.delays import apply_delay as _apply_delay


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


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

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
# VBI engine-key detection
# ---------------------------------------------------------------------------

_ENGINE_KEY = None
_ENGINE_KEY_CHECKED = False


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
        params["engine"] = config.ENGINE
    else:
        params[key] = config.ENGINE


# ---------------------------------------------------------------------------
# Streaming integration with TVB Bold Monitor
# ---------------------------------------------------------------------------

def _run_streaming_hrf(model, n_nodes, num_sim, dt_ms, apply_bw):
    """Run VBI WC step-by-step with TVB Bold Monitor on CPU.

    Uses ``bold.BoldMonitor`` with ``xp=np`` so the HRF math stays on
    CPU. WC heunStochastic runs on GPU (cupy); only the per-step E
    values are transferred down.

    Returns
    -------
    bold : np.ndarray  (T_bold, N, S)   if apply_bw=True
    e    : np.ndarray  (T_stored, N, S) if apply_bw=False
    """
    from bold import BoldMonitor

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

    # TVB Bold Monitor — always on CPU (numpy) to avoid cupy NVRTC issues.
    mon = BoldMonitor(
        nn=nn, ns=ns,
        dt_ms=dt_full,
        xp=np,                          # force CPU
        period_ms=config.TR_SEC * 1000.0,
        hrf_length_ms=getattr(config, "HRF_LENGTH_MS", 20_000.0),
    )

    for i in range(n_steps):
        t_curr = i * dt_full
        model.x0 = model.heunStochastic(model.x0, t_curr)
        E_i = model.x0[:nn, :]
        E_cpu = E_i.get() if hasattr(E_i, "get") else np.asarray(E_i)
        mon.step(i, E_cpu, t_cut_ms=t_cut)

    return mon.collect(mean_subtract=True)


# ---------------------------------------------------------------------------
# Per-simulation parameter injection
# ---------------------------------------------------------------------------

def _try_per_sim_params(params, chunk, param_names):
    """Inject per-simulation parameter arrays into VBI WC params.

    Each parameter becomes a (num_sim,) array so the i-th simulation
    sees the i-th theta row. **Never** a batch-mean scalar.

    Returns
    -------
    success : bool   Always True here; the caller's try/except handles
                     actual VBI-side rejection.
    """
    for i, name in enumerate(param_names):
        params[name] = chunk[:, i].astype(np.float32)
    return True


# ---------------------------------------------------------------------------
# GPU batch simulation
# ---------------------------------------------------------------------------

def simulate_gpu_batch(weights, theta_batch, param_names,
                       fixed_overrides=None, delays=None, apply_bw=True,
                       _allow_fallback=True):
    """Simulate a batch of parameter sets on the GPU.

    Each row of ``theta_batch`` corresponds to one simulation
    (num_sim = len(chunk)). Parameters are passed as per-simulation
    arrays so the (theta_i, x_i) training pairs stay correctly aligned.

    If the underlying WC implementation does NOT support per-simulation
    parameter arrays, we fall back to a per-theta loop (each row becomes
    a separate num_sim=1 simulation). This is slower but preserves
    label-feature alignment.

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
            short_msg = str(e).strip().splitlines()[0][:200]
            print(
                f"  ⚠ Per-sim parameter arrays not supported by WC "
                f"({type(e).__name__}: {short_msg}). "
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
                try:
                    m_single = wc_cls(p_single)
                    m_single.prepare_input()
                    m_single.set_initial_state()
                    r_single = _run_streaming_hrf(
                        m_single, n_nodes, 1, dt_ms, apply_bw=apply_bw,
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"Per-theta fallback also failed at theta {r}: "
                        f"{type(e2).__name__}: {str(e2)[:200]}\n"
                        "Fix: pip install nvidia-cuda-nvrtc-cu12 "
                        "--force-reinstall"
                    ) from e2
                outputs.append(r_single[:, :, 0])
                if (r + 1) % max(1, csz // 4) == 0 or r + 1 == csz:
                    print(f"    [fallback] {r + 1}/{csz} theta done")
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            continue

        # Success path: split per-sim outputs
        for i in range(csz):
            outputs.append(result[:, :, i])

        cp.get_default_memory_pool().free_all_blocks()

    return outputs


# ---------------------------------------------------------------------------
# Single-theta simulation
# ---------------------------------------------------------------------------

def simulate_single(weights, params_dict, n_repeat=1, delays=None,
                    apply_bw=True):
    """Simulate one parameter set, optionally repeated ``n_repeat`` times.

    Uses the same streaming BW as ``simulate_gpu_batch`` — neural
    time-series never accumulate in RAM.
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
