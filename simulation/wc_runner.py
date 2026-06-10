"""VBI Wilson-Cowan execution core.

Public API
----------
- simulate_gpu_batch(weights, theta_batch, param_names, ...) -> list[BOLD]
- simulate_single(weights, params_dict, n_repeat, delays, apply_bw)
- _import_wc()           : lazy import of simulation.wc_eib.WC_EIB
- detect_engine_key()    : auto-detect WC engine parameter name
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
3. **BoldMonitor runs on GPU (cupy) when available.** WC heunStochastic
   produces E on GPU; the BoldMonitor stock buffer + HRF kernel ride on
   the same cupy backend, so no per-step host transfer is needed. The
   HRF kernel itself is computed once in numpy (TVB MixtureOfGammas →
   no NVRTC) and then transferred onto cupy. Falls back to per-step
   numpy transfer when cupy is unavailable.
"""
import math
import time

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
        from simulation.wc_eib import WC_EIB
        _WC_SDE_CLASS = WC_EIB
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
# H100-targeted optimizations (OPT-1/OPT-3/OPT-5)
# ---------------------------------------------------------------------------
# OPT-5 watermark: keep cupy memory pool warm between chunks on a 94 GB H100
# NVL — eager free_all_blocks() forces re-allocation each chunk.  Only trim
# when the pool exceeds 80 GB total.
_POOL_TRIM_WATERMARK_BYTES = 80 * 1024 ** 3


def _trim_memory_pool(cp):
    """OPT-5: free cupy pool blocks only when above the watermark."""
    try:
        pool = cp.get_default_memory_pool()
        if pool.total_bytes() > _POOL_TRIM_WATERMARK_BYTES:
            pool.free_all_blocks()
    except Exception:
        pass


def _alloc_stride_buffers(cp, stride, nn, ns):
    """OPT-1: stride-sized GPU staging buffer.

    Returns ``(gpu_buf, cpu_buf, pinned_handle)``. With the BoldMonitor
    running on cupy, the CPU staging buffer + pinned host pages are no
    longer needed — both extra slots are returned as ``None``. Signature
    is preserved for backward compatibility with callers / GPU-1
    invariant grep.
    """
    gpu_buf = cp.empty((stride, nn, ns), dtype=cp.float32)
    return gpu_buf, None, None


# ---------------------------------------------------------------------------
# Streaming integration with TVB Bold Monitor
# ---------------------------------------------------------------------------

def _run_streaming_hrf(model, n_nodes, num_sim, dt_ms, apply_bw,
                       progress_fn=None):
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

    # +1 so the loop reaches the final boundary step (step_idx = t_end/dt).
    # Without it the BOLD monitor emits its last TR frame one period early
    # (e.g. T=239 instead of 240 for t_end=300s, t_cut=60s, TR=1s), making
    # the VBI BOLD T off-by-one vs cuBNM (= config.ANALYSIS_BOLD_T).
    n_steps = int(round(model.t_end / dt_full)) + 1

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

    # TVB Bold Monitor on GPU when cupy is available (HRF + stock buffers
    # ride on cupy). The HRF kernel is computed once in numpy and then
    # transferred onto cupy — no NVRTC compilation triggered.
    mon = BoldMonitor(
        nn=nn, ns=ns,
        dt_ms=dt_full,
        xp=np,                          # legacy default; overridden by use_gpu
        use_gpu=True,                   # → cupy backend if available
        period_ms=config.TR_SEC * 1000.0,
        hrf_length_ms=getattr(config, "HRF_LENGTH_MS", 20_000.0),
    )

    # OPT-1: stride-batched GPU staging. With the BoldMonitor on cupy, the
    # per-step host transfer (gpu_buf.get) and the pinned CPU buffer are
    # no longer needed — gpu_buf slices are fed straight into mon.step().
    # _alloc_stride_buffers keeps its signature (cpu_buf returned as None
    # for backward compat) so the GPU-1 invariant text is preserved.
    try:
        import cupy as cp  # only available when running on GPU
        stride = max(1, int(getattr(mon, "_interim_istep", 1)))
        gpu_buf, _cpu_buf, _pinned_handle = _alloc_stride_buffers(
            cp, stride, nn, ns
        )
        use_gpu_buffer = bool(getattr(mon, "_on_gpu", False))
    except Exception:
        # cupy unavailable (e.g. CPU-only test environment) — fall back to
        # the per-step transfer path. mon will also have fallen back to
        # numpy internally, so passing E_cpu is correct.
        use_gpu_buffer = False

    if use_gpu_buffer:
        i = 0
        while i < n_steps:
            burst = min(stride, n_steps - i)
            for j in range(burst):
                t_curr = (i + j) * dt_full
                model.x0 = model.heunStochastic(model.x0, t_curr)
                gpu_buf[j] = model.x0[:nn, :]
            # No host transfer — BoldMonitor runs on cupy now.
            for j in range(burst):
                mon.step(i + j, gpu_buf[j], t_cut_ms=t_cut)
            i += burst
            if progress_fn is not None:
                progress_fn(i, n_steps)
    else:
        for i in range(n_steps):
            t_curr = i * dt_full
            model.x0 = model.heunStochastic(model.x0, t_curr)
            E_i = model.x0[:nn, :]
            E_cpu = (
                E_i.get() if hasattr(E_i, "get") else np.asarray(E_i)
            )
            mon.step(i, E_cpu, t_cut_ms=t_cut)
            if progress_fn is not None and i % 1000 == 0:
                progress_fn(i + 1, n_steps)

    return mon.collect(mean_subtract=True)


# ---------------------------------------------------------------------------
# Per-simulation parameter injection
# ---------------------------------------------------------------------------

def _expand_global(chunk, param_names, n_nodes):
    """Expand global scalars (P, Q, g_e, g_i) to per-sim arrays.

    chunk        : (n_sim, n_params)
    returns dict : {name: (n_nodes, n_sim) float32}

    Broadcast each scalar column to (n_nodes, n_sim) so it satisfies the
    per-sim (n_nodes, csz) shape contract enforced by ``simulate_gpu_batch``
    (INV-2). WC-EIB reads g_e/g_i as global scalars and broadcasts P/Q.
    """
    n_sim = chunk.shape[0]
    params_out = {}
    for i, name in enumerate(param_names):
        col = chunk[:, i].astype(np.float32)              # (n_sim,)
        tiled = np.broadcast_to(col[None, :], (n_nodes, n_sim))
        params_out[name] = np.ascontiguousarray(tiled, dtype=np.float32)
    return params_out


def _try_per_sim_params(params, chunk, param_names, n_nodes=None):
    """Inject per-simulation parameter arrays into the WC parameter dict.

    WC-EIB (and its VBI-compatible interface) expects parameters as
    shape ``(n_nodes, n_sim)``. Global scalars P, Q, g_e, g_i, c_ei are
    broadcast to per-node-per-sim arrays.

    Returns
    -------
    effective_names : list[str]
        The parameter keys whose shape should be asserted by the caller.
    """
    weights = params["weights"]
    n_nodes = n_nodes or weights.shape[0]

    expanded = _expand_global(chunk, param_names, n_nodes)
    for k, v in expanded.items():
        params[k] = v
    return list(param_names)


# ---------------------------------------------------------------------------
# GPU batch simulation
# ---------------------------------------------------------------------------

def simulate_gpu_batch(weights, theta_batch, param_names,
                       fixed_overrides=None, delays=None, apply_bw=True,
                       _allow_fallback=True,
                       label=None, n_total=None):
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

    # ── 진행 바 초기화 (label 지정 시 활성화) ────────────────────
    import sys as _sys_pb, time as _time_pb
    if n_total is None:
        n_total = len(theta_batch)
    _bar_width = 20
    _lbl = str(label) if label is not None else ""
    _t_start_pb = _time_pb.perf_counter()
    _n_done = 0

    def _fmt_t(s):
        m = int(s) // 60
        return f"{m:02d}:{int(s) % 60:02d}"

    if label is not None:
        _n_chunks_total = math.ceil(n_total / config.GPU_BATCH) if n_total > 0 else 1
        print(
            f"\n[Step 2] {_lbl}  N_SIM={n_total:,}  GPU_BATCH={config.GPU_BATCH:,}"
            f"  n_chunks={_n_chunks_total}",
            flush=True,
        )

    dt_ms = config.DT * config.DECIMATE

    # Verify per-sim parameter support on the very first chunk.
    array_param_supported = None

    # L3 — per-chunk progress accounting (Task A). Local helper so the
    # success and fallback paths emit identical lines.
    n_chunks = math.ceil(n_total / batch_sz) if n_total > 0 else 0

    def _emit_chunk_log(idx, sz, t_start, pre):
        elapsed = time.time() - t_start
        valid = sum(int(np.isfinite(b).all()) for b in outputs[pre:])
        running = len(outputs)
        if sz > 0 and valid == 0:
            print(
                f"  WARN chunk {idx:>3d}/{n_chunks:<3d}  "
                f"({sz:>4d} sims) ... all dropped  total {running:>6d}"
            )
        else:
            print(
                f"  chunk {idx:>3d}/{n_chunks:<3d}  "
                f"({sz:>4d} sims) ...  {elapsed:5.2f} s  "
                f"got {valid:>4d}  total {running:>6d}"
            )

    for chunk_i, start in enumerate(range(0, n_total, batch_sz), start=1):
        chunk_start = time.time()
        pre_count = len(outputs)
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
        effective_names = _try_per_sim_params(
            params, chunk, param_names, n_nodes=n_nodes,
        )

        # Sanity check: parameter arrays must be (n_nodes, csz), not
        # scalar. Catches accidental regressions to batch-mean.
        n_nodes_chk = weights.shape[0]
        for name in effective_names:
            v = params[name]
            assert (
                hasattr(v, "shape") and v.shape == (n_nodes_chk, csz)
            ), (
                f"Per-simulation parameter {name!r} has wrong shape "
                f"{getattr(v, 'shape', None)}, expected "
                f"({n_nodes_chk}, {csz}). "
                "This guards against batch-mean regressions."
            )

        try:
            model = wc_cls(params)
            model.prepare_input()
            model.set_initial_state()
            if label is not None:
                import sys as _sys_hrf, time as _time_hrf
                _hrf_base = _n_done

                def _pfn(steps_done, steps_total):
                    _cp = steps_done / steps_total if steps_total else 0
                    _est = _hrf_base + int(_cp * csz)
                    _el = _time_hrf.perf_counter() - _t_start_pb
                    _sp = _est / _el if _el > 1e-6 else 0.0
                    _eta = (n_total - _est) / _sp if _sp > 1e-6 else 0.0
                    _f = int(_bar_width * _est / n_total)
                    _b = "▓" * _f + "░" * (_bar_width - _f)
                    _sys_hrf.stdout.write(
                        f"\r  {_lbl:<12s}"
                        f" |{_b}|"
                        f" ~{_est:>6,}/{n_total:>6,}"
                        f" |{100.0*_est/n_total if n_total else 0:5.1f}%"
                        f" |{_sp:5.1f} sim/s"
                        f" | elapsed {int(_el)//60:02d}:{int(_el)%60:02d}"
                        f" | ETA {int(_eta)//60:02d}:{int(_eta)%60:02d}"
                        f"   "
                    )
                    _sys_hrf.stdout.flush()
            else:
                _pfn = None
            result = _run_streaming_hrf(
                model, n_nodes, csz, dt_ms, apply_bw=apply_bw,
                progress_fn=_pfn,
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
                _try_per_sim_params(
                    p_single, chunk[r:r + 1], param_names,
                    n_nodes=n_nodes,
                )
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
                _n_done += 1
                if label is not None and (
                    _n_done % max(1, csz // 50) == 0 or _n_done >= n_total
                ):
                    _elapsed = _time_pb.perf_counter() - _t_start_pb
                    _pct   = 100.0 * _n_done / n_total
                    _speed = _n_done / _elapsed if _elapsed > 1e-6 else 0.0
                    _eta   = (n_total - _n_done) / _speed if _speed > 1e-6 else 0.0
                    _f     = int(_bar_width * _n_done / n_total)
                    _b     = "▓" * _f + "░" * (_bar_width - _f)
                    _line  = (
                        f"\r  {_lbl:<12s}"
                        f" |{_b}| (fallback)"
                        f" {_n_done:>6,}/{n_total:>6,}"
                        f" |{_pct:5.1f}%"
                        f" |{_speed:5.1f} sim/s"
                        f" | elapsed {_fmt_t(_elapsed)}"
                        f"   "
                    )
                    _sys_pb.stdout.write(_line)
                    _sys_pb.stdout.flush()
                if (r + 1) % max(1, csz // 4) == 0 or r + 1 == csz:
                    print(f"    [fallback] {r + 1}/{csz} theta done")
            # OPT-5: keep pool warm on H100; trim only above watermark.
            _trim_memory_pool(cp)
            _emit_chunk_log(chunk_i, csz, chunk_start, pre_count)
            continue

        # Success path: split per-sim outputs + \r progress bar
        _display_stride = max(1, min(512, csz // 50))
        for _si in range(csz):
            outputs.append(result[:, :, _si])
            _n_done += 1
            if label is not None and (
                _n_done % _display_stride == 0 or _n_done >= n_total
            ):
                _elapsed = _time_pb.perf_counter() - _t_start_pb
                _pct   = 100.0 * _n_done / n_total
                _speed = _n_done / _elapsed if _elapsed > 1e-6 else 0.0
                _eta   = (n_total - _n_done) / _speed if _speed > 1e-6 else 0.0
                _f     = int(_bar_width * _n_done / n_total)
                _b     = "▓" * _f + "░" * (_bar_width - _f)
                _line  = (
                    f"\r  {_lbl:<12s}"
                    f" |{_b}|"
                    f" {_n_done:>6,}/{n_total:>6,}"
                    f" |{_pct:5.1f}%"
                    f" |{_speed:5.1f} sim/s"
                    f" | elapsed {_fmt_t(_elapsed)}"
                    f" | ETA {_fmt_t(_eta)}"
                    f"   "
                )
                _sys_pb.stdout.write(_line)
                _sys_pb.stdout.flush()

        # OPT-5: keep pool warm on H100; trim only above watermark.
        _trim_memory_pool(cp)
        _emit_chunk_log(chunk_i, csz, chunk_start, pre_count)

    # ── 완료 요약 ─────────────────────────────────────────────────
    if label is not None:
        _total_t = _time_pb.perf_counter() - _t_start_pb
        _spd = len(outputs) / _total_t if _total_t > 1e-6 else 0.0
        print()
        print(
            f"  {_lbl} DONE"
            f" | collected={len(outputs):,}/{n_total:,}"
            f" | dropped={n_total - len(outputs):,}"
            f" | {_spd:.1f} sim/s avg"
            f" | elapsed {_fmt_t(_total_t)}",
            flush=True,
        )

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

    # OPT-5: keep pool warm on H100; trim only above watermark.
    _trim_memory_pool(cp)
    return out
