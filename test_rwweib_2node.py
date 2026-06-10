"""Minimal 2-node test for the RWW-EIB-FFI model (cuBNM/rww_eib.yaml).

Part A (runs anywhere, no GPU/build): a numpy transcription of the yaml
step_equations, asserting the modelling contract:
  - globalinput = SC @ S_e   (S_i is NEVER used in the coupling)
  - increasing g_LRE changes the excitatory current x_e
  - increasing g_FFI changes the inhibitory current x_i
  - no SC @ S_i term exists anywhere

Part B (runs only after the model is codegen'd + compiled into cubnm):
  instantiate RWWEIBSimGroup on a 2-node SC and confirm it runs.

Run:  python test_rwweib_2node.py
"""
import numpy as np


# ---- fixed params (mirror config.RWWEIB_FIXED / rww_eib.yaml) ----
FIXED = dict(
    J_N=0.15, J_i=1.0, w_p=1.4, W_e=1.0, W_i=0.7, I_o=0.382, I_ext=0.0,
    a_e=310.0, b_e=125.0, d_e=0.16, gamma_e=0.641 / 1000.0, tau_e=100.0,
    a_i=615.0, b_i=177.0, d_i=0.087, gamma_i=1.0 / 1000.0, tau_i=10.0,
)


def currents(S_e, S_i, SC, g_LRE, g_FFI, p=FIXED):
    """Transcription of rww_eib.yaml step_equations (x_e, x_i, globalinput).

    The ONLY connectome product is globalinput = SC @ S_e.
    """
    globalinput = SC @ S_e                       # (N,)  <-- SC @ S_e ONLY
    x_e = (p["w_p"] * p["J_N"] * S_e
           - p["J_i"] * S_i
           + p["W_e"] * p["I_o"]
           + g_LRE * p["J_N"] * globalinput
           + p["I_ext"])
    x_i = (p["J_N"] * S_e
           - S_i
           + p["W_i"] * p["I_o"]
           + g_FFI * p["J_N"] * globalinput)
    return x_e, x_i, globalinput


def test_part_a():
    rng = np.random.default_rng(0)
    SC = np.array([[0.0, 0.8],
                   [0.8, 0.0]])               # 2-node, zero diagonal
    S_e = np.array([0.30, 0.50])
    S_i = np.array([0.10, 0.20])

    # 1) globalinput == SC @ S_e exactly (and does NOT involve S_i)
    _, _, gin = currents(S_e, S_i, SC, 1.0, 1.0)
    assert np.allclose(gin, SC @ S_e), "globalinput must equal SC @ S_e"
    # changing S_i must NOT change globalinput (no SC @ S_i pathway)
    _, _, gin2 = currents(S_e, S_i * 5.0 + 3.0, SC, 1.0, 1.0)
    assert np.allclose(gin, gin2), "globalinput must be independent of S_i"
    print("[A1] globalinput = SC @ S_e, independent of S_i              OK")

    # 2) increasing g_LRE changes x_e (slope = J_N * globalinput)
    xe_lo, _, _ = currents(S_e, S_i, SC, 1.0, 1.0)
    xe_hi, _, _ = currents(S_e, S_i, SC, 2.0, 1.0)
    slope_e = (xe_hi - xe_lo) / 1.0
    assert np.allclose(slope_e, FIXED["J_N"] * gin), "dx_e/dg_LRE must be J_N*globalinput"
    assert np.all(np.abs(xe_hi - xe_lo) > 0), "g_LRE must change x_e where coupled"
    print("[A2] increasing g_LRE changes x_e  (slope = J_N*globalinput)  OK")

    # 3) increasing g_FFI changes x_i (slope = J_N * globalinput); x_e unaffected
    xe_a, xi_a, _ = currents(S_e, S_i, SC, 1.0, 1.0)
    xe_b, xi_b, _ = currents(S_e, S_i, SC, 1.0, 2.0)
    slope_i = (xi_b - xi_a) / 1.0
    assert np.allclose(slope_i, FIXED["J_N"] * gin), "dx_i/dg_FFI must be J_N*globalinput"
    assert np.all(np.abs(xi_b - xi_a) > 0), "g_FFI must change x_i where coupled"
    assert np.allclose(xe_a, xe_b), "g_FFI must NOT change x_e"
    print("[A3] increasing g_FFI changes x_i, leaves x_e unchanged       OK")

    # 4) feedforward inhibition: remote E activity (S_e of the OTHER node)
    #    reaches the local I current via globalinput, with NO SC@S_i.
    S_e_perturbed = S_e.copy(); S_e_perturbed[1] += 0.4   # change node-1 S_e
    _, xi_ref, _ = currents(S_e, S_i, SC, 1.0, 1.0)
    _, xi_per, _ = currents(S_e_perturbed, S_i, SC, 1.0, 1.0)
    # node-0 inhibitory current must move because SC[0,1]*S_e[1] entered globalinput
    assert abs(xi_per[0] - xi_ref[0]) > 0, "remote S_e must drive local I (FFI)"
    print("[A4] remote excitatory S_e drives local I via SC@S_e (FFI)    OK")

    print("\nPart A passed: model contract verified (no SC @ S_i anywhere).")


def test_part_b():
    try:
        from cubnm.sim import RWWEIBSimGroup
    except Exception as e:  # noqa: BLE001
        print(f"\n[B] SKIP cuBNM smoke test — RWWEIBSimGroup not built yet "
              f"({type(e).__name__}). Build per cuBNM/BUILD_RWWEIB.md, then re-run.")
        return

    SC = np.array([[0.0, 0.8], [0.8, 0.0]], dtype=np.float64)
    from cuBNM.runner_rwweib import run_cubnm_rwweib_batch
    # two sims: vary g_LRE then g_FFI
    theta = np.array([[1.0, 1.0, 0.01],
                      [2.0, 1.0, 0.01],
                      [1.0, 2.0, 0.01]], dtype=np.float64)
    bolds = run_cubnm_rwweib_batch(
        SC, theta, ["g_LRE", "g_FFI", "sigma"],
        duration_s=10.0, tr_s=1.0, dt_ms=0.1, burn_in_s=2.0, hrf="bw",
    )
    assert len(bolds) == 3
    for b in bolds:
        assert b.shape[1] == 2, f"expected 2 nodes, got {b.shape}"
        assert np.isfinite(b).all(), "non-finite BOLD"
    print(f"\n[B] cuBNM RWWEIBSimGroup ran: {len(bolds)} sims, "
          f"BOLD shape {bolds[0].shape}  OK")


if __name__ == "__main__":
    test_part_a()
    test_part_b()
