"""Fast numerical FIC (feedback inhibition control) for RWWEIB_2CPL.

Computes a per-(sim, node) inhibitory weight J_i that holds each region's
excitatory input current I_E near the target operating point (~0.377 nA,
r_E ~ 3 Hz), so the network is critical rather than saturated. This is the
operating-point fix the sensitivity analysis showed is required (simFC~SC
collapses 0.30 -> 0.05 once saturated).

Method: online homeostatic tuning (Deco 2014 / EI_Tuning notebook Part 1),
vectorized across ALL sims at once in numpy — run the DETERMINISTIC mean-field
dynamics once while nudging J_i every few steps toward the I_E target. Returns
J_i (n_sims, n_nodes) to inject into cuBNM via the "J_i_matrix" fixed key.

Deterministic (noise-free) tuning is intentional: FIC sets the fixed point, the
noisy FC simulation runs afterward in cuBNM with the tuned J_i.
"""
import numpy as np

# rWW constants (match cuBNM rww_eib_2cpl.yaml)
_A_E, _B_E, _D_E = 310.0, 125.0, 0.16
_A_I, _B_I, _D_I = 615.0, 177.0, 0.087
_GAMMA_E, _GAMMA_I = 0.641 / 1000.0, 1.0 / 1000.0
_TAU_E, _TAU_I = 100.0, 10.0
_W_E, _W_I = 1.0, 0.7
_I_E_TARGET = 0.377            # b_a_ratio_E - 0.026 = 125/310 - 0.026


def _H(a, b, d, I):
    x = a * I - b
    # guard the removable singularity at x->0
    return x / (1.0 - np.exp(-d * x) + 1e-12)


def _col(theta, pn, name, default, n):
    if name in pn:
        return np.asarray(theta, float)[:, pn.index(name)].copy()
    return np.full(n, default, float)


def compute_fic_ji(sc, theta, param_names, J_N=0.15,
                   n_steps=4000, adjust_every=25, eta=0.4, dt=1.0,
                   ji_init=1.0, ji_clip=(0.02, 40.0), verbose=True):
    """Return J_i (n_sims, n_nodes) tuned so I_E ~ target for every sim/node.

    Parameters read per-sim from theta (by name in param_names), else default:
    g_LRE, g_FFI, I_o, w_p. sigma is ignored (deterministic tuning).
    """
    sc = np.ascontiguousarray(np.asarray(sc, float))
    N = sc.shape[0]
    theta = np.asarray(theta, float)
    S = theta.shape[0]
    pn = list(param_names)

    gL = _col(theta, pn, "g_LRE", 1.0, S)[:, None]      # (S,1)
    gF = _col(theta, pn, "g_FFI", 1.0, S)[:, None]
    Io = _col(theta, pn, "I_o", 0.382, S)[:, None]
    wp = _col(theta, pn, "w_p", 1.4, S)[:, None]
    wE = _col(theta, pn, "w_E", _W_E, S)[:, None]       # promoted params (else default)
    wI = _col(theta, pn, "w_I", _W_I, S)[:, None]

    SE = np.full((S, N), 0.15)
    SI = np.full((S, N), 0.15)
    Ji = np.full((S, N), float(ji_init))
    acc = np.zeros((S, N)); nacc = 0

    for t in range(n_steps):
        cE = SE @ sc                                     # (S,N): sum_j SC_ij S_E_j
        cI = SI @ sc
        IE = wE * Io + wp * J_N * SE + gL * J_N * cE - Ji * SI
        II = wI * Io + J_N * SE + gF * J_N * cI - SI
        SE = np.clip(SE + dt * (_GAMMA_E * (1 - SE) * _H(_A_E, _B_E, _D_E, IE) - SE / _TAU_E), 0, 1)
        SI = np.clip(SI + dt * (_GAMMA_I * _H(_A_I, _B_I, _D_I, II) - SI / _TAU_I), 0, 1)
        acc += IE; nacc += 1
        if (t + 1) % adjust_every == 0:
            mIE = acc / nacc                             # mean I_E over window
            Ji = np.clip(Ji + eta * (mIE - _I_E_TARGET), ji_clip[0], ji_clip[1])
            acc[:] = 0; nacc = 0

    if verbose:
        err = np.abs(mIE - _I_E_TARGET)
        print(f"  [FIC] {S} sims x {N} nodes tuned: <I_E>={mIE.mean():.3f} "
              f"(target {_I_E_TARGET:.3f}), |err| mean={err.mean():.3f} "
              f"max={err.max():.3f}; J_i range=[{Ji.min():.2f},{Ji.max():.2f}]",
              flush=True)
    return np.ascontiguousarray(Ji)
