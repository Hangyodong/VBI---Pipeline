#!/usr/bin/env python
"""Forward sweep of RWWEIB2 FIXED params to RAISE the achievable FC-corr ceiling.

Motivation (docs/experiment_log.md CEIL): run #1's expected-FC corr (0.157) is
~80% of the achievable ceiling (~0.196). The ceiling — the best corr ANY theta
could reach — is a SIMULATOR limit, not an inference one. This script searches
the fixed-parameter regime (J_N / w_p / lambda_IE / w_E / w_I) for a setting
whose simulated-FC manifold aligns BETTER with empirical FC, i.e. a higher
achievable ceiling. theta_dim stays 12, the basis decoder + model equations are
unchanged — only the fixed scalar regime is swept (in-scope).

Method per config:
  * sample N_THETA basis coeffs ~ U(-2,2)^12 (SAME set across configs, fixed seed)
  * decode -> region maps -> simulate on each subject's SC with the swept FIXED
  * ceiling(subject) = max over sims of corr(sim_FC, empirical_FC)  (best-case theta)
  * config score = mean ceiling over subjects
Rank configs; the current regime (J_N=0.15, w_p=1.4, lambda_IE=1.0) is the baseline.

Run on a GPU node:
  SC_DATASET=cabnp381 SC_FILE=HCP_CABNP381_SC_first100.mat \
    python forward_sweep_ceiling.py
Env knobs: SWEEP_N_THETA (300), SWEEP_N_SUBJ (5), SWEEP_JN ("0.10,0.15,0.25"),
SWEEP_WP ("1.0,1.4,1.8"), SWEEP_LAMBDA ("1.0"). A higher mean-ceiling regime is a
candidate to RE-RUN main_HCP in (set config.RWWEIB2_FIXED there) — that lifts the
ceiling the amortized posterior is then trained against.
"""
import itertools
import os

import numpy as np

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")
os.environ.setdefault("SC_DATASET", "cabnp381")
os.environ.setdefault("SC_FILE", "HCP_CABNP381_SC_first100.mat")

from pipeline_setup import PipelineConfig, setup_pipeline

setup_pipeline(PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat", SC_FILE=os.environ["SC_FILE"], N_REGIONS=360,
    N_SUBJECTS=100, N_TRAIN=80, N_VAL=10, N_TEST=10, SEED=42,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
), print_summary=False)

import config
config.INFERENCE_MODEL = "rwweib2"
config.PARAMETER_MODE = "basis_regionwise"
config.SC_DATASET = "cabnp381"
config.HETERO_PARAMS = ["g_LRE", "g_FFI", "I_o", "sigma"]
_g_hi = float(os.environ.get("G_BOUND_HIGH", "3.0"))
config.BASIS_BOUNDS = {"g_LRE": (0.0, _g_hi), "g_FFI": (0.0, _g_hi),
                       "I_o": (0.0, 1.0), "sigma": (0.0, 0.05)}
config.HETERO_BOUNDS = dict(config.BASIS_BOUNDS)
config.BASIS_PATH = os.environ.get("BASIS_PATH", "basis.npy")
config.BASIS_REZSCORE = True
config.USE_DELAYS = False

_BASE_FIXED = {"w_E": 1.0, "w_I": 0.7, "J_i": 1.0, "w_p": 1.4,
               "J_N": 0.15, "lambda_IE": 1.0}

from basis_decoder import get_decoder
from engine_select import get_simulate_gpu_batch
from features.fc import compute_fc, fc_to_upper_tri
from simulation.geometry import augment_sc_geometry
import data_loader_hcp as dl

_dec = get_decoder(config)
config.STAGE1_PARAMS = _dec.coeff_names()
config.PARAM_NAMES_STAGE1 = config.STAGE1_PARAMS


def _grid():
    # Defaults isolate the GEOMETRY (homotopic) lever at the baseline J_N/w_p.
    # Widen SWEEP_JN / SWEEP_WP to also sweep the dynamical regime.
    jn = [float(x) for x in os.environ.get("SWEEP_JN", "0.15").split(",")]
    wp = [float(x) for x in os.environ.get("SWEEP_WP", "1.4").split(",")]
    lam = [float(x) for x in os.environ.get("SWEEP_LAMBDA", "1.0").split(",")]
    ga = [float(x) for x in os.environ.get("SWEEP_GEOM_ALPHA", "0.0,0.3,0.6,1.0").split(",")]
    return list(itertools.product(jn, wp, lam, ga))


def main():
    n_theta = int(os.environ.get("SWEEP_N_THETA", "300"))
    n_subj = int(os.environ.get("SWEEP_N_SUBJ", "5"))
    config.N_SIM = n_theta
    config.GPU_BATCH = n_theta

    # subjects (first n_subj of the train pool) + empirical FC
    df, fc_mat, sc_mat, fc_ids, sc_ids, *_ = dl.load_raw_data()
    subs = dl.get_target_subjects(df, fc_ids, sc_ids)
    train, _, _ = dl.three_way_split(subs)
    use = list(train)[:n_subj]
    data = dl.load_all_subjects(use, fc_mat, sc_mat, fc_ids, sc_ids)
    emp = {s: fc_to_upper_tri(data[s]["fc"]).astype(np.float32) for s in use}

    # one fixed theta set (same across configs for a fair regime comparison)
    rng = np.random.default_rng(0)
    theta = rng.uniform(-2.0, 2.0, size=(n_theta, _dec.theta_dim)).astype(np.float32)

    sim = get_simulate_gpu_batch()        # basis_regionwise -> latent_wrap
    pn = list(config.STAGE1_PARAMS)
    grid = _grid()
    print("=" * 78)
    print(f"  Forward sweep — achievable FC-corr ceiling  "
          f"(N_THETA={n_theta}, subjects={len(use)}, {len(grid)} configs)")
    print(f"  baseline regime: J_N=0.15 w_p=1.4 lambda_IE=1.0  "
          f"(run #1 achievable ceiling ~0.196)")
    print("=" * 78)

    results = []
    for jn, wp, lam, ga in grid:
        config.RWWEIB2_FIXED = {**_BASE_FIXED, "J_N": jn, "w_p": wp,
                                "lambda_IE": lam}
        config.GEOMETRY_COUPLING = (ga > 0.0)
        config.GEOM_ALPHA = ga
        ceils = []
        for s in use:
            # apply the (gated) homotopic prior to the base SC for this alpha
            sc_aug = augment_sc_geometry(data[s]["sc"], config)
            try:
                bolds = sim(sc_aug, theta, param_names=pn,
                            delays=data[s]["delays"], apply_bw=True,
                            label=None, n_total=n_theta)
            except Exception as e:
                print(f"  J_N={jn} w_p={wp} lam={lam} geom={ga}  sim FAIL: {e}")
                ceils = None
                break
            E = emp[s]; ec = E - E.mean(); en = np.sqrt((ec ** 2).sum()) + 1e-9
            best = -1.0
            for b in bolds:
                v = fc_to_upper_tri(compute_fc(np.asarray(b)))
                vc = v - v.mean(); c = float((vc @ ec) / (np.sqrt((vc ** 2).sum()) * en + 1e-9))
                if c > best:
                    best = c
            ceils.append(best)
        if ceils is None:
            continue
        m = float(np.mean(ceils))
        results.append((m, jn, wp, lam, ga, ceils))
        flag = "  <== baseline" if (jn == 0.15 and wp == 1.4 and lam == 1.0 and ga == 0.0) else ""
        print(f"  J_N={jn:<5} w_p={wp:<4} lam={lam:<4} geom_a={ga:<4}  "
              f"ceiling mean={m:.4f}  [{min(ceils):.3f},{max(ceils):.3f}]{flag}")

    if results:
        results.sort(reverse=True)
        print("-" * 78)
        bm, bj, bw, bl, bg, _ = results[0]
        base = next((r for r in results
                     if r[1] == 0.15 and r[2] == 1.4 and r[3] == 1.0 and r[4] == 0.0), None)
        print(f"  BEST: J_N={bj} w_p={bw} lambda_IE={bl} GEOM_ALPHA={bg}  ceiling={bm:.4f}")
        if base:
            print(f"  vs baseline ceiling {base[0]:.4f}  ->  delta={bm - base[0]:+.4f}")
        print(f"  next: if delta>0, re-run main_HCP with those FIXED + "
              f"GEOMETRY_COUPLING={int(bg>0)} GEOM_ALPHA={bg} to lift the ceiling "
              f"the posterior trains against.")
    print("=" * 78)


if __name__ == "__main__":
    main()
