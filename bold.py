"""BOLD signal generation using TVB Bold Monitor algorithm.

Provides ``BoldMonitor``: a stateful object that mirrors TVB's Bold Monitor
but works with VBI's cupy Wilson-Cowan step-by-step output.

TVB Bold Monitor internals (from monitors.py):
    _stock_sample_rate = 2^-2 = 0.25 /ms  → interim period = 4 ms
    _stock_steps = stock_sample_rate * hrf_length  (e.g. 0.25 * 20000 = 5000)
    interim_istep = interim_period / dt   (e.g. 4 / 0.5 = 8 steps)
    stock shape   = (_stock_steps, N, S)
    sample: dot(rolled_hrf, stock.transpose(1,2,0,3))

We use MixtureOfGammas instead of FirstOrderVolterra for the HRF so that
mouse-specific timing (peak ~3 s) can be set via config.HRF_A1 etc.
"""
import numpy as np

import config


# ---------------------------------------------------------------------------
# HRF kernel
# ---------------------------------------------------------------------------

def tvb_hrf(dt_ms, hrf_length_ms=None, a_1=None, a_2=None, rate=1.0, c=None):
    """Compute TVB MixtureOfGammas HRF kernel.

    Parameters match TVB Bold Monitor conventions:

    dt_ms         : sampling interval in ms  (interim period, e.g. 4 ms)
    hrf_length_ms : kernel duration in ms    (e.g. 20000 ms = 20 s)
    a_1           : positive gamma shape  (peak ~ a_1/rate s)
    a_2           : undershoot gamma shape
    rate          : rate parameter (1/s)
    c             : undershoot amplitude ratio

    Returns
    -------
    hrf_rev : np.ndarray (K,)  HRF reversed for dot-product (TVB convention)
    K       : int              number of stock steps
    """
    from tvb.datatypes.equations import MixtureOfGammas

    a_1 = a_1 if a_1 is not None else getattr(config, "HRF_A1", 3.0)
    a_2 = a_2 if a_2 is not None else getattr(config, "HRF_A2", 7.0)
    c = c if c is not None else getattr(config, "HRF_C", 0.3)
    hrf_length_ms = hrf_length_ms or getattr(config, "HRF_LENGTH_MS", 20_000.0)

    # Stock sample rate: TVB uses 2^-2 = 0.25 /ms = period 4 ms
    stock_sample_rate = 2.0 ** -2          # /ms
    K = int(np.ceil(stock_sample_rate * hrf_length_ms))
    stock_time_max = hrf_length_ms / 1000.0  # s
    stock_time_step = stock_time_max / K    # s
    stock_time = np.arange(0.0, stock_time_max, stock_time_step)  # s

    m = MixtureOfGammas()
    m.parameters["a_1"] = a_1
    m.parameters["a_2"] = a_2
    m.parameters["l"] = rate
    m.parameters["c"] = c

    G = m.evaluate(stock_time).astype(np.float32)
    G_rev = G[::-1].copy()  # reversed for dot-product (TVB convention)
    return G_rev, K


# ---------------------------------------------------------------------------
# Stateful BOLD Monitor (mirrors TVB Bold.sample)
# ---------------------------------------------------------------------------

class BoldMonitor:
    """TVB Bold Monitor algorithm for VBI step-by-step integration.

    Usage
    -----
    mon = BoldMonitor(nn=115, ns=10000, dt_ms=0.5, xp=cupy)
    for i in range(n_steps):
        x0 = model.heunStochastic(x0, t)
        E  = x0[:nn, :]
        bold_frame = mon.step(i, E)   # returns (N, S) or None
    bold = mon.collect()              # (T_bold, N, S) numpy array
    """

    def __init__(self, nn, ns, dt_ms, xp=np, use_gpu=False,
                 period_ms=None, hrf_length_ms=None, verbose=False):
        """
        nn            : number of brain regions
        ns            : number of parallel simulations
        dt_ms         : integration step (ms), e.g. 0.5
        xp            : numpy or cupy (legacy; defaults to numpy)
        use_gpu       : if True, override xp with cupy (lazy import). Falls
                        back to numpy when cupy is unavailable. The HRF
                        kernel is still computed in numpy first, then
                        transferred onto the active backend.
        period_ms     : BOLD sampling period (ms), default config.TR_SEC*1000
        hrf_length_ms : HRF duration (ms), default config.HRF_LENGTH_MS
        verbose       : print monitor configuration on init
        """
        self.nn = nn
        self.ns = ns
        self.dt_ms = dt_ms
        # Resolve active backend: use_gpu=True dispatches to cupy. Note
        # that ``import cupy`` may succeed even when the CUDA runtime is
        # unhealthy (driver too old, no GPU visible, MIG mode, etc.); the
        # failure only surfaces at the first allocation. Probe with a
        # 1-element zero alloc to detect that case and fall back cleanly.
        if use_gpu:
            try:
                import cupy as _cp
                _ = _cp.zeros(1, dtype=_cp.float32)
                self.xp = _cp
                self._on_gpu = True
            except Exception:
                self.xp = np
                self._on_gpu = False
        else:
            self.xp = xp
            self._on_gpu = (
                xp is not np and getattr(xp, "__name__", "") == "cupy"
            )
        self.period_ms = period_ms or (config.TR_SEC * 1000.0)
        self.hrf_length_ms = hrf_length_ms or getattr(
            config, "HRF_LENGTH_MS", 20_000.0
        )

        # ── TVB Bold Monitor internals ───────────────────────────────
        # interim period = 4 ms (TVB's _stock_sample_rate = 0.25/ms)
        self._interim_period_ms = 4.0
        self._interim_istep = int(round(self._interim_period_ms / dt_ms))

        # HRF kernel + stock size — computed in numpy, then transferred
        # onto self.xp. tvb_hrf() returns CPU numpy values; no NVRTC.
        _hrf_np, self._K = tvb_hrf(
            dt_ms=self._interim_period_ms,
            hrf_length_ms=self.hrf_length_ms,
        )
        self._hrf_rev = self.xp.asarray(_hrf_np)
        # GPU version of reversed HRF: shape (1, K) on self.xp.
        self._hrf_gpu = self._hrf_rev[np.newaxis, :]

        # interim stock: accumulate `_interim_istep` raw steps then average
        self._interim_stock = self.xp.zeros(
            (self._interim_istep, nn, ns), dtype=self.xp.float32
        )
        self._interim_pos = 0  # write index in interim stock

        # outer stock: circular buffer of K averaged interim samples
        self._stock = self.xp.zeros(
            (self._K, nn, ns), dtype=self.xp.float32,
        )
        self._stock_pos = 0    # write index in outer stock
        self._stock_count = 0  # how many times outer stock was written

        # BOLD output period
        self._istep = int(round(self.period_ms / dt_ms))

        # output frames
        self._bold_frames = []

        if verbose:
            print(
                f"  [BoldMonitor] dt={dt_ms}ms  "
                f"interim_period={self._interim_period_ms}ms  "
                f"interim_istep={self._interim_istep}  "
                f"K(stock_steps)={self._K}  "
                f"period={self.period_ms}ms  "
                f"istep={self._istep}  "
                f"on_gpu={self._on_gpu}"
            )

    def step(self, step_idx, E, t_cut_ms=0.0):
        """Process one integration step.

        Parameters
        ----------
        step_idx : int   global integration step counter
        E        : array (nn, ns) excitatory activity (on xp device)
        t_cut_ms : float transient cutoff — no output before this time

        Returns
        -------
        bold_frame : (nn, ns) numpy array at TR steps, else None
        """
        xp = self.xp
        t_curr_ms = step_idx * self.dt_ms

        # Coerce E onto the active backend. cp.asarray(cupy_array) is a
        # no-op view; cp.asarray(numpy_array) does a host→device transfer;
        # np.asarray(numpy_array) is a no-op view. We do NOT try to demote
        # a cupy array onto numpy here — that path is the CPU fallback,
        # exercised only when cupy is unavailable.
        E_xp = xp.asarray(E)

        # 1. Interim stock: accumulate raw E
        interim_idx = step_idx % self._interim_istep
        self._interim_stock[interim_idx] = xp.clip(E_xp, 0.0, 1.0)

        # 2. At interim period: average → push to outer stock
        if step_idx > 0 and step_idx % self._interim_istep == 0:
            avg = self._interim_stock.mean(axis=0)  # (N, S)
            self._stock[self._stock_pos] = avg
            self._stock_pos = (self._stock_pos + 1) % self._K
            self._stock_count += 1

        # 3. At TR period: dot(rolled_hrf, stock) → BOLD frame
        if (step_idx > 0
                and step_idx % self._istep == 0
                and t_curr_ms > t_cut_ms
                and self._stock_count >= self._K):
            # Roll stock so oldest is at index 0 (TVB convention)
            rolled = xp.roll(
                self._stock, -self._stock_pos, axis=0
            )                                    # (K, N, S)
            # dot: (1, K) @ (K, N*S) -> (1, N*S) -> (N, S)
            flat = rolled.reshape(self._K, self.nn * self.ns)
            bold_t = self._hrf_gpu @ flat        # (1, N*S)
            bold_t = bold_t.reshape(self.nn, self.ns)
            frame = (
                bold_t.get() if hasattr(bold_t, "get")
                else np.asarray(bold_t)
            )
            self._bold_frames.append(frame)
            return frame
        return None

    def collect(self, mean_subtract=True):
        """Return all recorded BOLD frames as (T_bold, N, S) array.

        Parameters
        ----------
        mean_subtract : bool  subtract per-region mean (for FC computation)
        """
        if not self._bold_frames:
            return np.zeros((0, self.nn, self.ns), dtype=np.float32)
        bold = np.stack(self._bold_frames, axis=0)   # (T, N, S)
        if mean_subtract and bold.shape[0] > 1:
            bold = bold - bold.mean(axis=0, keepdims=True)
        target = config.ANALYSIS_BOLD_T
        if bold.shape[0] > target:
            bold = bold[:target]
        return bold.astype(np.float32)


# ---------------------------------------------------------------------------
# Backward-compatible API
# ---------------------------------------------------------------------------

def balloon_windkessel(neural, dt_ms=None, bw_params=None, tr_sec=None,
                       use_gpu=True):
    """Apply TVB HRF convolution to a pre-computed neural time series.

    Drop-in for the old Balloon-Windkessel function. Used when the full
    E time series is available (apply_bw=False path + post-processing).

    Parameters
    ----------
    neural : (T, N) or (T, N, S) float32
    dt_ms  : stored step in ms  (default config.DT * config.DECIMATE)
    tr_sec : output TR in s     (default config.TR_SEC)
    """
    dt_ms = dt_ms or (config.DT * config.DECIMATE)
    tr_sec = tr_sec or config.TR_SEC

    if hasattr(neural, "get"):
        neural = neural.get()
    neural = np.asarray(neural, dtype=np.float32)
    squeeze = neural.ndim == 2
    if squeeze:
        neural = neural[:, :, None]
    T, N, S = neural.shape

    # Build monitor on CPU
    mon = BoldMonitor(
        nn=N, ns=S, dt_ms=dt_ms,
        xp=np, period_ms=tr_sec * 1000.0,
    )
    E_centered = neural - neural.mean(axis=0, keepdims=True)
    for i in range(T):
        mon.step(i, E_centered[i].T if S > 1 else E_centered[i, :, 0:1].T,
                 t_cut_ms=0.0)

    bold = mon.collect(mean_subtract=True)   # (T_bold, N, S)
    if squeeze:
        bold = bold[:, :, 0]
    return bold
