"""
WC-EIB: Wilson-Cowan with Excitatory-Inhibitory Balance
cupy GPU 구현. VBI WC_sde와 동일한 인터페이스 유지.

방정식 (global scalars):
  c_lre = g_e * (W @ E)           # E 기반 장거리 흥분
  c_ffi = g_i * (W @ I)           # I 기반 장거리 억제 (Σ_l w_kl I_l)

  x_e = alpha_e*(c_ee*E - c_ei*I + P - theta_e + c_lre)
  x_i = alpha_i*(c_ie*E - c_ii*I + Q - theta_i + lamda*c_ffi)

  dE/dt = (1/tau_e)*(-E + (k_e - r_e*E)*S(x_e))
  dI/dt = (1/tau_i)*(-I + (k_i - r_i*I)*S(x_i))

  S(x) = c / (1 + exp(-a*(x-b)))
"""
import numpy as np
import os

try:
    import cupy as cp
except ImportError:
    cp = np


class WC_EIB:
    """WC-EIB dynamics — cupy GPU implementation.

    VBI ``WC_sde``-compatible surface:
        - ``prepare_input()``
        - ``set_initial_state()``
        - ``heunStochastic(x, t)``
        - ``valid_params`` attribute
    """

    valid_params = [
        "c_ee", "c_ei", "c_ie", "c_ii",
        "tau_e", "tau_i",
        "a_e", "a_i", "b_e", "b_i", "c_e", "c_i",
        "alpha_e", "alpha_i",
        "theta_e", "theta_i",
        "k_e", "k_i", "r_e", "r_i",
        "P", "Q", "I_ext", "lamda",
        "g_e", "g_i",
        "rE_max_hz", "rI_max_hz",
        "weights", "num_sim",
        "dt", "t_end", "t_cut",
        "noise_amp", "seed",
        "engine", "decimate",
        "same_initial_state", "dtype",
        "method", "RECORD_EI",
        "delays",
    ]

    def __init__(self, par: dict = {}):
        self._par = self._default_params()
        self._par.update(par)
        for k, v in self._par.items():
            setattr(self, k, v)

        if self._par.get("engine", "cpu") in ("gpu", "cuda"):
            self.xp = cp
        else:
            self.xp = np

        if self.seed is not None:
            self.xp.random.seed(self.seed)

        self.PREPARE_INPUT = False

    def _default_params(self):
        return {
            "c_ee": 10.0,
            "c_ei": 6.0,
            "c_ie": 10.0,
            "c_ii": 1.0,
            "tau_e": 10.0,
            "tau_i": 10.0,
            "a_e": 1.0,   "a_i": 1.0,
            "b_e": 0.0,   "b_i": 0.0,
            "c_e": 1.0,   "c_i": 1.0,
            "alpha_e": 1.2, "alpha_i": 2.0,
            "theta_e": 2.0, "theta_i": 3.5,
            "k_e": 1.0,   "k_i": 1.0,
            "r_e": 1.0,   "r_i": 1.0,
            "P": 0.5,     "Q": 0.0,
            "I_ext": 0.0, "lamda": 1.0,
            "g_e": 1.0,
            "g_i": 0.0,
            "rE_max_hz": 20.0,
            "rI_max_hz": 20.0,
            "method": "heun",
            "RECORD_EI": "E",
            "weights": None,
            "num_sim": 1,
            "dt": 0.5,
            "t_end": 300000.0,
            "t_cut": 60000.0,
            "noise_amp": 0.01,
            "seed": None,
            "engine": "gpu",
            "decimate": 20,
            "same_initial_state": False,
            "dtype": "float32",
            "delays": None,
        }

    def prepare_input(self):
        assert self.weights is not None
        xp = self.xp
        ns = self.num_sim
        self.W = xp.array(self.weights, dtype=xp.float32)
        self.nn = self.W.shape[0]
        nn, ns_ = self.nn, ns

        def _to_xp(v, shape):
            """Promote a scalar or (nn, ns) array onto the active xp."""
            arr = xp.array(v, dtype=xp.float32)
            if arr.ndim == 0:
                return xp.full(shape, float(arr), dtype=xp.float32)
            return arr

        self._c_ee    = _to_xp(self.c_ee,    (nn, ns_))
        self._c_ei    = _to_xp(self.c_ei,    (nn, ns_))
        self._c_ie    = _to_xp(self.c_ie,    (nn, ns_))
        self._c_ii    = _to_xp(self.c_ii,    (nn, ns_))
        self._tau_e   = _to_xp(self.tau_e,   (nn, ns_))
        self._tau_i   = _to_xp(self.tau_i,   (nn, ns_))
        self._a_e     = _to_xp(self.a_e,     (nn, ns_))
        self._a_i     = _to_xp(self.a_i,     (nn, ns_))
        self._b_e     = _to_xp(self.b_e,     (nn, ns_))
        self._b_i     = _to_xp(self.b_i,     (nn, ns_))
        self._c_e     = _to_xp(self.c_e,     (nn, ns_))
        self._c_i     = _to_xp(self.c_i,     (nn, ns_))
        self._alpha_e = _to_xp(self.alpha_e, (nn, ns_))
        self._alpha_i = _to_xp(self.alpha_i, (nn, ns_))
        self._theta_e = _to_xp(self.theta_e, (nn, ns_))
        self._theta_i = _to_xp(self.theta_i, (nn, ns_))
        self._k_e     = _to_xp(self.k_e,     (nn, ns_))
        self._k_i     = _to_xp(self.k_i,     (nn, ns_))
        self._r_e     = _to_xp(self.r_e,     (nn, ns_))
        self._r_i     = _to_xp(self.r_i,     (nn, ns_))
        self._P       = _to_xp(self.P,       (nn, ns_))
        self._Q       = _to_xp(self.Q,       (nn, ns_))
        self._I_ext   = _to_xp(self.I_ext,   (nn, ns_))
        self._lamda   = float(self.lamda)

        # g_e, g_i: global scalars. Kept as xp arrays so a per-sim
        # (n_nodes, n_sim) vector broadcasts against WE (n_nodes, n_sim);
        # a 0-d scalar broadcasts too.
        self._g_e = xp.asarray(self.g_e, dtype=xp.float32)
        self._g_i = xp.asarray(self.g_i, dtype=xp.float32)

        self._inv_tau_e = 1.0 / self._tau_e
        self._inv_tau_i = 1.0 / self._tau_i
        self._dt = float(self.dt)
        self._noise_amp = float(self.noise_amp)

        self._prepare_delays(nn, ns_)

        self.PREPARE_INPUT = True

    def _prepare_delays(self, nn, ns_):
        """Build the conduction-delay ring buffer (discrete-delay form).

        Coupling becomes  WE = Σ_k W_k @ E(t - k·dt), where each W_k holds
        only the edges whose integer delay (ceil(delay_ms/dt)) equals k.
        This avoids the (N, N, n_sim) gather temporary that the naive
        per-edge approach would allocate every step.
        """
        xp = self.xp
        delays = self.delays
        self.use_delay = (
            delays is not None and np.any(np.asarray(delays) > 0)
        )
        if not self.use_delay:
            self._E_buf = None
            self._max_delay = 0
            self._delayed_WE = None
            return

        delays_ms = np.asarray(delays, dtype=np.float64)
        delay_steps = np.ceil(delays_ms / self._dt).astype(np.int64)
        np.fill_diagonal(delay_steps, 0)
        self._max_delay = int(delay_steps.max())
        self._buf_size = self._max_delay + 1

        # One masked weight matrix per unique positive delay step.
        # k == 0 carries only zero-weight (diagonal / unconnected) edges,
        # so it contributes nothing and is skipped.
        W_np = np.asarray(self.weights, dtype=np.float32)
        ks = [int(k) for k in np.unique(delay_steps) if k > 0]
        Wk = np.stack(
            [W_np * (delay_steps == k).astype(np.float32) for k in ks]
        )                                       # (n_uniq, nn, nn)
        self._delay_ks = np.asarray(ks, dtype=np.int64)
        self._Wk = xp.asarray(Wk, dtype=xp.float32)
        self._E_buf = xp.zeros(
            (self._buf_size, nn, ns_), dtype=xp.float32
        )
        self._buf_ptr = 0
        self._delayed_WE = xp.zeros((nn, ns_), dtype=xp.float32)

    def _update_delayed_WE(self, E_now):
        """Push current E into the ring buffer and refresh ``_delayed_WE``.

        Called once per integration step (the delayed input is fixed for
        both Heun stages of that step).
        """
        xp = self.xp
        self._E_buf[self._buf_ptr] = E_now
        acc = self._delayed_WE
        acc[...] = 0.0
        for idx in range(self._delay_ks.shape[0]):
            k = int(self._delay_ks[idx])
            past = (self._buf_ptr - k) % self._buf_size
            acc += self._Wk[idx] @ self._E_buf[past]
        self._buf_ptr = (self._buf_ptr + 1) % self._buf_size

    def set_initial_state(self):
        xp = self.xp
        nn, ns = self.nn, self.num_sim
        if self.same_initial_state:
            y = np.random.rand(2 * nn).astype(np.float32)
            self.x0 = xp.array(np.tile(y[:, None], (1, ns)))
        else:
            self.x0 = xp.array(
                np.random.rand(2 * nn, ns).astype(np.float32)
            )

    def _sigmoid(self, x, a, b, c):
        return c / (1.0 + self.xp.exp(-a * (x - b)))

    def derivative(self, x, t):
        xp = self.xp
        nn = self.nn
        E = x[:nn, :]
        I = x[nn:, :]

        if self.use_delay:
            WE = self._delayed_WE          # E(t - τ_ij), fixed for k1/k2
        else:
            WE = self.W @ E
        # global scalars: g_e*(SC@E) excitatory, g_i*(SC@I) inhibitory.
        # Inhibitory long-range coupling rides on I (no delay buffer for I).
        c_lre = self._g_e * WE
        c_ffi = self._g_i * (self.W @ I)

        x_e = self._alpha_e * (
            self._c_ee * E
            - self._c_ei * I
            + self._P
            + self._I_ext
            - self._theta_e
            + c_lre
        )
        x_i = self._alpha_i * (
            self._c_ie * E
            - self._c_ii * I
            + self._Q
            - self._theta_i
            + self._lamda * c_ffi
        )

        S_e = self._sigmoid(x_e, self._a_e, self._b_e, self._c_e)
        S_i = self._sigmoid(x_i, self._a_i, self._b_i, self._c_i)

        dE = self._inv_tau_e * (-E + (self._k_e - self._r_e * E) * S_e)
        dI = self._inv_tau_i * (-I + (self._k_i - self._r_i * I) * S_i)

        return xp.vstack([dE, dI])

    def heunStochastic(self, x, t):
        xp = self.xp
        if self.use_delay:
            self._update_delayed_WE(x[:self.nn, :])
        coeff = self._noise_amp * xp.sqrt(self._dt)
        dw = xp.random.normal(size=x.shape).astype(xp.float32)
        k1 = self.derivative(x, t)
        x_pred = x + self._dt * k1 + coeff * dw
        k2 = self.derivative(x_pred, t + self._dt)
        return x + self._dt * (k1 + k2) / 2.0 + coeff * dw
