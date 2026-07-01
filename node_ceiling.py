#!/usr/bin/env python
"""node_ceiling.py — S0 decisive gate: per-subject NODE-param forward ceiling.

Question (gates the whole direct/enriched-node track): can the FULL node DOF
(360x4=1440, direct per-region) fit per-subject empFC meaningfully better than
the 12-coeff myelin/gradient basis? If NOT, node-1524 amortization is dead
weight and effort must move to edge-level (geometry/EC). NO SNPE training here —
pure forward optimization (the inference ceiling, not the inference).

Method (per test subject):
  1. basis-12 ceiling: random search over basis coeffs (narrow HEALTHY bounds)
     -> best corr + decoded 1440 physical map = CMA WARM START.
  2. direct-1440 ceiling: sep-CMA (diagonal) in unit [0,1]^1440 space, warm-
     started from (1), one GPU batch / generation, objective = -corr(simFC,empFC).
  3. DENOISE both: re-sim the winner as K identical-theta rows -> average FC ->
     expected-FC corr (controls per-sim noise; the honest number).
  4. B4 noise floor: pairwise corr among the K single-sim FCs of one theta =
     simFC reproducibility -> an upper bound on any corr(sim,emp).

CRITIC FIXES baked in:
  - BOUNDS TRAP: config.HETERO_BOUNDS is set to the narrow HEALTHY set BEFORE any
    direct_bounds()/decode call (else direct pulls the WIDE saturated DEFAULT
    g(0,9)/I_o(0.15,0.60) and the comparison is meaningless).
  - OPTIMISM CONTROL: argmax-over-population corr (OPTIMISTIC) is ALWAYS
    co-reported with the denoised best-resim; the denoised number is the verdict.
  - CONVERGENCE: per-generation best/sigma logged; a LOW direct number is only
    trustworthy if sigma collapsed (else "no evidence within budget", not KILL).
  - empFC test-retest reliability is NOT in HCP_FC.mat (one FC/subject); the only
    measurable floor here is simFC reproducibility (B4). corr->1 is unreachable.

Usage (GPU node):
  python node_ceiling.py --subjects 100307,106016,108020 --basis-draws 3000 \
      --cma-gen 25 --denoise-k 16
  IO_BOUND_LO=0 IO_BOUND_HI=1 python node_ceiling.py   # (deliberately) old saturated bound
"""
import argparse
import os
import numpy as np

from pipeline_setup import PipelineConfig, setup_pipeline

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")

# ── production config (mirror main_HCP.py basis_regionwise + cabnp + 864 TR) ──
_cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat",
    SC_FILE=os.environ.get("SC_FILE", "HCP_CABNP381_SC_first100.mat"),
    N_REGIONS=360, N_SUBJECTS=int(os.environ.get("N_SUBJECTS", "10")),
    N_TRAIN=7, N_VAL=1, N_TEST=2, SEED=42, N_SIM=2_000, GPU_BATCH=2_000,
    T_END_MS=630_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,  # 864 TR total, 780 analyzed
)
setup_pipeline(_cfg, print_summary=False)

import config
config.INFERENCE_MODEL = "rwweib2"
config.N_REGIONS = 360
config.HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
# ── HEALTHY narrow bounds — set BEFORE any direct_bounds()/decode (BOUNDS TRAP) ──
_G_HI = float(os.environ.get("G_BOUND_HIGH", "3.0"))
_IO_LO = float(os.environ.get("IO_BOUND_LO", "0.30"))
_IO_HI = float(os.environ.get("IO_BOUND_HI", "0.45"))
_HEALTHY = {"g_LRE": (0.0, _G_HI), "g_FFI": (0.0, _G_HI),
            "I_o": (_IO_LO, _IO_HI), "sigma": (0.0, 0.05)}
config.HETERO_BOUNDS = dict(_HEALTHY)   # <-- consumed by param_decoder._bounds / direct_bounds
config.BASIS_BOUNDS = dict(_HEALTHY)
config.BASIS_COEFF_PRIOR = (-2.0, 2.0)
config.BASIS_PATH = os.environ.get("BASIS_PATH", "basis.npy")
config.BASIS_REZSCORE = True
config.RWWEIB2_FIXED = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
                        "J_N": 0.15, "lambda_IE": 1.0}
config.SC_DATASET = os.environ.get("SC_DATASET", "cabnp381")
config.GROUP_AVG_FC = (os.environ.get("GROUP_AVG_FC", "0") == "1")  # default per-subject (user decision)
config.VELOCITY_M_PER_S = 3.0
config.USE_FCD = False
config.USE_DELAYS = (os.environ.get("USE_DELAYS", "0") == "1")

import data_loader_hcp as data_loader
import engine_select
from basis_decoder import BasisParamDecoder
import param_decoder
from features.fc import compute_fc, fc_to_upper_tri

R = 360
HETERO = config.HETERO_PARAMS
IU = np.triu_indices(R, k=1)


def _masked_corr(a, b, nan_mask=None):
    a = np.asarray(a); b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if nan_mask is not None:
        m &= ~nan_mask
    if m.sum() < 3:
        return -1.0
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def _fc_vecs(bolds):
    return [fc_to_upper_tri(compute_fc(np.asarray(b))) for b in bolds]


def basis_warmstart(sc, real_vec, nan_vec, simulate_batch, n_draws, rng):
    """Random-search the 12 basis coeffs (HEALTHY bounds) -> best corr + decoded
    1440 physical warm-start vector (HETERO order)."""
    dec = BasisParamDecoder.from_file(config.BASIS_PATH, HETERO, bounds=_HEALTHY,
                                      n_regions=R, rezscore=True)
    config.PARAMETER_MODE = "basis_regionwise"
    config.STAGE1_PARAMS = dec.coeff_names()
    beta = rng.uniform(-2.0, 2.0, (n_draws, dec.theta_dim))
    bolds = simulate_batch(sc, beta, config.STAGE1_PARAMS, delays=None, apply_bw=True)
    corrs = np.array([_masked_corr(v, real_vec, nan_vec) for v in _fc_vecs(bolds)])
    k = int(np.nanargmax(np.where(np.isfinite(corrs), corrs, -1.0)))
    maps = dec.decode(beta[k][None, :])                      # {p:(1,R)}
    x0 = np.concatenate([np.asarray(maps[p]).ravel() for p in HETERO])  # (1440,) physical
    return float(corrs[k]), x0


def cma_direct(sc, real_vec, nan_vec, simulate_batch, x0_phys, n_gen, rng):
    """sep-CMA over unit [0,1]^1440, warm-started from x0_phys. Returns best corr,
    best physical theta, convergence trace."""
    import cma
    config.PARAMETER_MODE = "direct_regionwise"
    config.STAGE1_PARAMS = param_decoder.direct_param_names(config, R)
    lo, hi = param_decoder.direct_bounds(config, R)
    lo = np.asarray(lo); hi = np.asarray(hi); span = np.maximum(hi - lo, 1e-12)
    u0 = np.clip((np.asarray(x0_phys) - lo) / span, 0.0, 1.0)
    es = cma.CMAEvolutionStrategy(u0.tolist(), 0.2, {
        "bounds": [0.0, 1.0], "CMA_diagonal": True, "verbose": -9,
        "maxiter": n_gen, "seed": 1})
    best_corr, best_phys = -1.0, lo + u0 * span
    trace = []
    gen = 0
    while not es.stop():
        U = np.asarray(es.ask())                              # (pop, 1440) unit
        Theta = lo + U * span                                 # physical
        bolds = simulate_batch(sc, Theta, config.STAGE1_PARAMS, delays=None, apply_bw=True)
        corrs = np.array([_masked_corr(v, real_vec, nan_vec) for v in _fc_vecs(bolds)])
        corrs = np.where(np.isfinite(corrs), corrs, -1.0)
        es.tell(U.tolist(), (-corrs).tolist())
        k = int(np.argmax(corrs))
        if corrs[k] > best_corr:
            best_corr, best_phys = float(corrs[k]), Theta[k].copy()
        sig = float(np.mean(es.sigma * es.sigma_vec.scaling)) if hasattr(es, "sigma_vec") else float(es.sigma)
        trace.append((gen, float(corrs.max()), best_corr, sig))
        gen += 1
        print(f"      [cma] gen {gen:2d}/{n_gen}  batch_best={corrs.max():+.4f}  "
              f"running_best={best_corr:+.4f}  sigma~{sig:.4f}", flush=True)
    return best_corr, best_phys, trace


def denoise_and_floor(sc, theta_phys, mode, real_vec, nan_vec, simulate_batch, K):
    """Re-sim one theta as K identical rows -> (expected-FC corr, B4 sim-repro floor).
    expected-FC = corr(mean of K sim FCs, empFC). B4 = mean pairwise corr among the
    K single-sim FCs (simFC reproducibility = upper bound on any sim-vs-emp corr)."""
    config.PARAMETER_MODE = mode
    config.STAGE1_PARAMS = (param_decoder.direct_param_names(config, R)
                            if mode == "direct_regionwise"
                            else BasisParamDecoder.from_file(
                                config.BASIS_PATH, HETERO, bounds=_HEALTHY,
                                n_regions=R, rezscore=True).coeff_names())
    Theta = np.tile(np.asarray(theta_phys)[None, :], (K, 1))
    bolds = simulate_batch(sc, Theta, config.STAGE1_PARAMS, delays=None, apply_bw=True)
    vecs = np.array(_fc_vecs(bolds))                          # (K, FC_DIM)
    exp_corr = _masked_corr(vecs.mean(0), real_vec, nan_vec)
    # B4: pairwise sim-sim corr (finite, non-nan edges)
    fin = np.isfinite(vecs).all(0) & ~nan_vec
    pij = []
    for i in range(K):
        for j in range(i + 1, K):
            a, b = vecs[i, fin], vecs[j, fin]
            if a.std() > 1e-9 and b.std() > 1e-9:
                pij.append(np.corrcoef(a, b)[0, 1])
    return float(exp_corr), (float(np.mean(pij)) if pij else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="")
    ap.add_argument("--n-subjects", type=int, default=3)
    ap.add_argument("--basis-draws", type=int, default=3000)
    ap.add_argument("--cma-gen", type=int, default=25)
    ap.add_argument("--denoise-k", type=int, default=16)
    args = ap.parse_args()

    df, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi = data_loader.load_raw_data()
    sids = ([int(s) for s in args.subjects.split(",") if s] if args.subjects
            else data_loader.get_target_subjects(df, fc_ids, sc_ids)[:args.n_subjects])
    subj = data_loader.load_all_subjects(sids, fc_mat, sc_mat, fc_ids, sc_ids, bm, bi)
    simulate_batch = engine_select.get_simulate_gpu_batch()   # latent_wrap routes by config.PARAMETER_MODE

    print("=" * 74)
    print("  NODE-param forward ceiling: direct-1440 (CMA) vs basis-12 (random)")
    print(f"  bounds(HEALTHY): g(0,{_G_HI}) I_o({_IO_LO},{_IO_HI}) sigma(0,0.05) | "
          f"864 TR (780 analyzed) | per-{'group' if config.GROUP_AVG_FC else 'subject'} FC")
    print(f"  subjects={sids}  basis_draws={args.basis_draws}  cma_gen={args.cma_gen}  K={args.denoise_k}")
    print("  NOTE: empFC test-retest NOT in data (1 FC/subj); only simFC repro (B4) measurable.")
    print("=" * 74)

    rng = np.random.RandomState(0)
    rows = []
    for sid in sids:
        d = subj[sid]
        sc = d["sc"].astype(np.float64)
        real_vec = np.asarray(d["fc"])[IU]
        nan_vec = (np.asarray(d.get("fc_nan", np.zeros((R, R))))[IU] > 0)
        print(f"\n  subject {sid}:")
        bc, x0 = basis_warmstart(sc, real_vec, nan_vec, simulate_batch, args.basis_draws, rng)
        print(f"    basis-12 random best (argmax) = {bc:+.4f}")
        dc, dtheta, trace = cma_direct(sc, real_vec, nan_vec, simulate_batch, x0, args.cma_gen, rng)
        sig_last = trace[-1][3] if trace else float("nan")
        converged = (sig_last < 0.05)
        b_exp, b_floor = denoise_and_floor(sc, x0, "basis_regionwise", real_vec, nan_vec, simulate_batch, args.denoise_k)
        d_exp, d_floor = denoise_and_floor(sc, dtheta, "direct_regionwise", real_vec, nan_vec, simulate_batch, args.denoise_k)
        print(f"    >>> basis  argmax={bc:+.4f}  DENOISED={b_exp:+.4f}  (sim-repro floor {b_floor:.3f})")
        print(f"    >>> direct argmax={dc:+.4f}  DENOISED={d_exp:+.4f}  (sim-repro floor {d_floor:.3f})  "
              f"sigma_last={sig_last:.4f} {'CONVERGED' if converged else 'NOT-converged'}")
        rows.append((sid, bc, b_exp, dc, d_exp, d_exp - b_exp, converged))

    print("\n" + "=" * 74)
    print("  SUMMARY (DENOISED = verdict number; argmax = optimistic)")
    print(f"  {'sid':>8} | {'basis_dn':>8} | {'direct_dn':>9} | {'delta':>7} | conv")
    for sid, bc, be, dc, de, dl, cv in rows:
        print(f"  {sid:>8} | {be:>8.4f} | {de:>9.4f} | {dl:>+7.4f} | {'Y' if cv else 'n'}")
    deltas = np.array([r[5] for r in rows]); convs = [r[6] for r in rows]
    md = float(np.mean(deltas))
    print("  " + "-" * 56)
    print(f"  mean direct-vs-basis DENOISED delta = {md:+.4f}  (converged: {sum(convs)}/{len(convs)})")
    if md >= 0.05 and all(convs):
        print("  => GATE PASS: node DOF beyond the 3-mode basis genuinely helps -> "
              "escalate to an IDENTIFIABLE enriched/Laplacian basis (NOT raw direct).")
    elif md < 0.05 and all(convs):
        print("  => GATE FAIL: full node DOF ~= basis-12 -> node-1440 amortization is dead "
              "weight; the 3 spatial modes already capture node heterogeneity. Ship basis-12 "
              "amortized; the only ceiling-raiser left is EDGE-level (geometry/EC).")
    else:
        print("  => INCONCLUSIVE: CMA not converged (sigma high) -> raise --cma-gen / budget "
              "before concluding. A LOW direct number here is NOT trustworthy (one-sided).")
    print("=" * 74)


if __name__ == "__main__":
    main()
