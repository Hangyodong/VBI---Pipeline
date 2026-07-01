#!/usr/bin/env python
"""operating_point_check.py — CPU validation of the I_o operating-point fix.

The fix (main_HCP.py:157-161) narrows the basis I_o bound from (0,1) [midpoint
0.5] to (0.30,0.45) [midpoint 0.375 ~= critical ~0.382]. Claim: with (0,1) the
prior draws land mostly in the SATURATED regime, so SNPE trains on degenerate
FCs; (0.30,0.45) keeps them near the healthy E/I operating point.

This validates that claim WITHOUT a GPU by solving the rWW-EIB *isolated-node*
(no coupling, no noise) steady state directly from the yaml equations/constants
(cuBNM/rww_eib_2cpl.yaml), then:

  (A) sweeps I_o -> steady-state r_E (operating-point curve): locate critical I_o,
      confirm r_E(0.382) ~ healthy few-Hz vs r_E(0.5) saturated.
  (B) draws the prior (beta ~ U(-2,2)^12), decodes the REAL basis to per-region
      I_o maps under OLD (0,1) vs NEW (0.30,0.45) bounds, and reports the fraction
      of (draw,region) entries that are saturated / healthy / sub-threshold.

CAVEAT (conservative): isolated-node ignores the long-range coupling
(+g_LRE*J_N*SC@S_E) which adds excitation -> the real network saturates at least
as much. So the OLD saturation fraction here is a LOWER bound on the live run.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

# ── rWW-EIB constants (cuBNM/rww_eib_2cpl.yaml) ───────────────────────────────
a_E, b_E, d_E = 310.0, 125.0, 0.16
a_I, b_I, d_I = 615.0, 177.0, 0.087
gamma_E, gamma_I = 0.641 / 1000.0, 1.0 / 1000.0
tau_E, tau_I = 100.0, 10.0
# RWWEIB2_FIXED (main_HCP.py:119-120)
w_E, w_I, J_i, w_p, J_N, lambda_IE = 1.0, 0.7, 1.0, 1.4, 0.15, 1.0

CRIT_IO = 0.382        # yaml default I_o (Deco 2014 isolated-node critical point)
OLD_BOUND = (0.0, 1.0)
NEW_BOUND = (0.30, 0.45)


def _H(aIb, d):
    """Transfer fn x/(1-exp(-d x)) with removable singularity at x=0 -> 1/d."""
    out = np.empty_like(aIb)
    small = np.abs(d * aIb) < 1e-8
    out[~small] = aIb[~small] / (1.0 - np.exp(-d * aIb[~small]))
    out[small] = 1.0 / d
    return out


def steady_state(I_o, n_steps=8000, dt=1.0):
    """Deterministic (sigma=0), isolated (globalinput=0) steady state per I_o.
    Integrates the yaml ODE with Euler (dt=1ms, same as the simulator).
    I_o: array (M,). returns dict of (M,) steady-state r_E,r_I,S_E,S_I."""
    I_o = np.asarray(I_o, dtype=np.float64)
    S_E = np.full_like(I_o, 0.001)
    S_I = np.full_like(I_o, 0.001)
    dt_gE, dt_gI = dt * gamma_E, dt * gamma_I
    dt_iE, dt_iI = dt / tau_E, dt / tau_I
    for _ in range(n_steps):
        I_Ecur = w_E * I_o + w_p * J_N * S_E - J_i * S_I          # globalinput_E=0
        I_Icur = w_I * I_o + J_N * S_E - S_I                       # globalinput_I=0
        r_E = _H(a_E * I_Ecur - b_E, d_E)
        r_I = _H(a_I * I_Icur - b_I, d_I)
        S_E = S_E + dt_gE * (1.0 - S_E) * r_E - dt_iE * S_E
        S_I = S_I + dt_gI * r_I - dt_iI * S_I
        np.clip(S_E, 0.0, 1.0, out=S_E)
        np.clip(S_I, 0.0, 1.0, out=S_I)
    # final rates at the converged state
    I_Ecur = w_E * I_o + w_p * J_N * S_E - J_i * S_I
    I_Icur = w_I * I_o + J_N * S_E - S_I
    return {"r_E": _H(a_E * I_Ecur - b_E, d_E), "r_I": _H(a_I * I_Icur - b_I, d_I),
            "S_E": S_E, "S_I": S_I}


def part_A():
    print("=" * 72)
    print("  (A) ISOLATED-NODE OPERATING-POINT CURVE  r_E(I_o)")
    print("=" * 72)
    ios = np.linspace(0.0, 1.0, 51)
    ss = steady_state(ios)
    print(f"  {'I_o':>6} | {'r_E(Hz)':>9} | {'r_I(Hz)':>9} | {'S_E':>6} | {'S_I':>6}")
    print("  " + "-" * 52)
    for i, io in enumerate(ios):
        if i % 5 == 0 or abs(io - 0.375) < 0.011 or abs(io - 0.40) < 0.011:
            mark = ""
            if abs(io - 0.50) < 0.011:
                mark = "  <- OLD midpoint"
            if abs(io - 0.375) < 0.011:
                mark = "  <- NEW midpoint (~critical)"
            print(f"  {io:>6.3f} | {ss['r_E'][i]:>9.3f} | {ss['r_I'][i]:>9.3f} | "
                  f"{ss['S_E'][i]:>6.3f} | {ss['S_I'][i]:>6.3f}{mark}")
    anchors = steady_state(np.array([CRIT_IO, 0.375, 0.50]))
    print("  " + "-" * 52)
    print(f"  anchor r_E:  I_o=0.382 -> {anchors['r_E'][0]:.3f} Hz   "
          f"I_o=0.375 -> {anchors['r_E'][1]:.3f} Hz   "
          f"I_o=0.500 -> {anchors['r_E'][2]:.3f} Hz")
    return anchors


def _decode_io_maps(bound, n_draws=200, seed=0):
    """Draw prior beta ~ U(-2,2)^12, decode REAL basis to per-region I_o maps
    under the given I_o bound. Returns (n_draws, 360) I_o array."""
    from basis_decoder import BasisParamDecoder
    params = ["g_LRE", "g_FFI", "I_o", "sigma"]
    bounds = {"g_LRE": (0.0, 3.0), "g_FFI": (0.0, 3.0),
              "I_o": tuple(bound), "sigma": (0.0, 0.05)}
    dec = BasisParamDecoder.from_file(
        os.environ.get("BASIS_PATH", "basis.npy"), params, bounds=bounds,
        n_regions=360, rezscore=True)
    rng = np.random.RandomState(seed)
    beta = rng.uniform(-2.0, 2.0, (n_draws, dec.theta_dim))
    return dec.decode(beta)["I_o"]            # (n_draws, 360)


def part_B(sat_hz=10.0, healthy_lo=1.0):
    print("=" * 72)
    print(f"  (B) PRIOR SATURATION FRACTION  (saturated := r_E > {sat_hz:.0f} Hz, "
          f"sub-threshold := r_E < {healthy_lo:.0f} Hz)")
    print("=" * 72)
    print(f"  {'bound':>16} | {'I_o mean':>8} {'std':>6} | "
          f"{'%sub':>6} {'%healthy':>8} {'%satur':>7} | {'r_E mean':>8} {'med':>6}")
    print("  " + "-" * 70)
    rows = {}
    for name, bd in [("OLD (0,1)", OLD_BOUND), ("NEW (0.30,0.45)", NEW_BOUND)]:
        io_maps = _decode_io_maps(bd).ravel()
        r_E = steady_state(io_maps)["r_E"]
        sub = float(np.mean(r_E < healthy_lo)) * 100
        sat = float(np.mean(r_E > sat_hz)) * 100
        heal = 100.0 - sub - sat
        rows[name] = (sat, heal, sub)
        print(f"  {name:>16} | {io_maps.mean():>8.3f} {io_maps.std():>6.3f} | "
              f"{sub:>5.1f}% {heal:>7.1f}% {sat:>6.1f}% | "
              f"{r_E.mean():>8.2f} {np.median(r_E):>6.2f}")
    print("=" * 72)
    return rows


def main():
    anc = part_A()
    rows = part_B()
    print("  VERDICT:")
    healthy = anc["r_E"][1] < 8.0 and anc["r_E"][2] > 15.0
    print(f"  - operating point: r_E(0.375 new mid)={anc['r_E'][1]:.2f}Hz "
          f"{'HEALTHY' if anc['r_E'][1] < 8 else 'NOT healthy'}, "
          f"r_E(0.5 old mid)={anc['r_E'][2]:.2f}Hz "
          f"{'SATURATED' if anc['r_E'][2] > 15 else 'not saturated'}")
    old_sat = rows["OLD (0,1)"][0]; new_sat = rows["NEW (0.30,0.45)"][0]
    print(f"  - prior saturation: OLD {old_sat:.1f}%  ->  NEW {new_sat:.1f}%  "
          f"(Δ {old_sat - new_sat:+.1f} pts)")
    if healthy and new_sat < old_sat - 10:
        print("  => I_o FIX VALIDATED (isolated-node, lower bound): the new bound "
              "moves the prior off the saturated regime toward the healthy point.")
    else:
        print("  => CHECK: premise not reproduced as expected — inspect curve above.")
    print("  NOTE: coupling (+g_LRE*J_N*SC@S_E) only ADDS excitation, so the live "
          "network saturates >= these isolated-node numbers.")


if __name__ == "__main__":
    main()
