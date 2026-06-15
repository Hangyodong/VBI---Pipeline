#!/usr/bin/env python
"""Forward-model ablation: SC transform x delays x FIC for RWWEIB_2CPL.

Isolates FORWARD-model quality (NO SNPE/inference): for each setting, simulate FC
per subject at a FIXED homogeneous operating point and compare sim-FC to empirical
FC, SC weight, and -tract_length. RWWEIB_2CPL equations + BOLD are unchanged; only
SC preprocessing, conduction delays, and FIC (J_i calibration) vary.

Settings:
  A0  raw_maxnorm    , delays off, FIC off   (baseline = current pipeline)
  A1  log1p_maxnorm  , delays off, FIC off
  A2  log1p_maxnorm  , delays ON , FIC off
  A3  log1p_maxnorm  , delays off, FIC ON
  A4  log1p_maxnorm  , delays ON , FIC ON

Delay units: delay_ms = tract_length_mm / velocity_m_per_s  (1 m/s = 1 mm/ms).
FIC: fic_tune calibrates per-node J_i to the operating point (I_E target). g_FFI
stays a FIXED forward param here (NOT inferred, NOT FIC-calibrated).
Raw SC weights + tract lengths are loaded directly from HCP_SC.mat (not the
data_loader-scaled matrices), so the SC transform is applied cleanly once.

GPU required.  python forward_ablation.py [--n_subjects 8 --t_s 120]
Outputs: output_hcp/forward_ablation_sc_delay_fic.csv + a markdown table.
"""
import argparse, csv, os, time
import numpy as np

os.environ.setdefault("VBI_SC_SCALE", "maxnorm")
from pipeline_setup import PipelineConfig, setup_pipeline

_cfg = PipelineConfig(
    DATA_DIR="/scratch/home/wog3597/vbi", OUTPUT_DIR="./output_hcp",
    FC_FILE="HCP_FC.mat", SC_FILE="HCP_SC.mat",
    N_REGIONS=360, N_SUBJECTS=10, N_TRAIN=6, N_VAL=2, N_TEST=2, SEED=42,
    T_END_MS=180_000.0, T_CUT_MS=60_000.0, DT=1.0, DECIMATE=720, TR_SEC=0.72,
)
setup_pipeline(_cfg)
import config
config.INFERENCE_MODEL = "rwweib2"
config.VELOCITY_M_PER_S = 3.0    # match main_HCP human value (delay_ms = len_mm / 3.0)

from cuBNM.runner_rwweib_2cpl import run_cubnm_rwweib2_batch
from cuBNM.fc import compute_fc

PARAMS = {"g_LRE": 1.0, "g_FFI": 1.0, "I_o": 0.32, "sigma": 0.01,
          "w_p": 1.4, "J_N": 0.15, "lambda_IE": 1.0}
PN = ["g_LRE", "g_FFI", "I_o", "sigma", "w_p", "J_N", "lambda_IE"]
SETTINGS = [
    ("A0", "raw_maxnorm",   False, False),
    ("A1", "log1p_maxnorm", False, False),
    ("A2", "log1p_maxnorm", True,  False),
    ("A3", "log1p_maxnorm", False, True),
    ("A4", "log1p_maxnorm", True,  True),
]
N = 360


def load_raw(n_subjects):
    """Per-subject raw SC weight, tract length (mm), empirical FC — cortical 360."""
    import scipy.io as sio, h5py
    C = sio.loadmat(config.FC_PATH if hasattr(config, "FC_PATH") else "HCP_FC.mat")["C"]
    ids = [int(np.asarray(C[i, 0]).flatten()[0]) for i in range(C.shape[0])]
    fcid = {s: i for i, s in enumerate(ids)}
    f = h5py.File(config.SC_PATH if hasattr(config, "SC_PATH") else "HCP_SC.mat", "r")
    refs = f["data"]; scidx = {}
    for j in range(refs.shape[1]):
        a = np.asarray(f[refs[0, j]][()]).flatten()
        scidx[int("".join(chr(int(x)) for x in a))] = j
    subs = sorted(set(ids) & set(scidx))[:n_subjects]
    out = {}
    for s in subs:
        W = np.asarray(f[refs[1, scidx[s]]][()]).astype(np.float64)
        W = (W + W.T) / 2.0; W = W[:N, :N]; np.fill_diagonal(W, 0.0)
        L = np.asarray(f[refs[2, scidx[s]]][()]).astype(np.float64)
        L = (L + L.T) / 2.0; L = L[:N, :N]; np.fill_diagonal(L, 0.0)
        FC = np.asarray(C[fcid[s], 1]).astype(np.float64)[:N, :N]
        out[s] = {"W": W, "L": L, "FC": FC}
    return subs, out


def sc_transform(W, mode):
    W = W.copy(); np.fill_diagonal(W, 0.0)
    if mode == "log1p_maxnorm":
        W = np.log1p(W)
    W = W / (W.max() + 1e-12); np.fill_diagonal(W, 0.0)
    return np.ascontiguousarray(W)


def _corr(a, b, m):
    m = m & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or a[m].std() <= 0 or b[m].std() <= 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_subjects", type=int, default=8)
    ap.add_argument("--t_s", type=float, default=120.0)
    ap.add_argument("--burn_s", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    subs, D = load_raw(args.n_subjects)
    iu = np.triu_indices(N, 1)
    vel = float(config.VELOCITY_M_PER_S)
    theta = np.array([[PARAMS[p] for p in PN]], float)
    print(f"forward ablation: {len(subs)} subjects, sim {args.t_s}s, velocity {vel} m/s")

    rows = []
    for rid, sctf, use_delays, use_fic in SETTINGS:
        t0 = time.time(); status = "ok"; note = []
        per = {k: [] for k in ("fc_all", "fc_nz", "ssc_all", "ssc_nz", "slen_nz",
                               "esc_all", "esc_nz", "elen_nz", "rmse")}
        try:
            for sid in subs:
                d = D[sid]
                Wt = sc_transform(d["W"], sctf)
                rawv = d["W"][iu]; Lv = d["L"][iu]; ev = d["FC"][iu]
                fin = np.isfinite(ev); nz = (rawv > 0) & fin
                sc_dist = vel_v = None
                if use_delays:
                    delay = d["L"] / vel; np.fill_diagonal(delay, 0.0)  # ms (1 m/s=1 mm/ms)
                    sc_dist = np.ascontiguousarray(delay); vel_v = 1.0
                fixed = {"seed": args.seed}
                if use_fic:
                    from fic_tune import compute_fic_ji
                    fixed["J_i_matrix"] = compute_fic_ji(
                        Wt, theta, PN, J_N=PARAMS["J_N"], verbose=False)
                bolds = run_cubnm_rwweib2_batch(
                    Wt, theta, PN, duration_s=args.t_s, burn_in_s=args.burn_s,
                    fixed=fixed, force_gpu=True, hrf="bw",
                    sc_dist=sc_dist, velocity=vel_v)
                sv = compute_fc(np.asarray(bolds[0]))[iu]
                per["fc_all"].append(_corr(sv, ev, fin))
                per["fc_nz"].append(_corr(sv, ev, nz))
                per["ssc_all"].append(_corr(sv, rawv, fin))
                per["ssc_nz"].append(_corr(sv, rawv, nz))
                per["slen_nz"].append(_corr(sv, -Lv, nz))
                per["esc_all"].append(_corr(rawv, ev, fin))
                per["esc_nz"].append(_corr(rawv, ev, nz))
                per["elen_nz"].append(_corr(-Lv, ev, nz))
                per["rmse"].append(float(np.sqrt(np.nanmean((sv[fin] - ev[fin]) ** 2))))
        except Exception as e:  # noqa: BLE001
            status = f"FAIL:{type(e).__name__}"; note.append(str(e)[:140])
        m = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in per.items()}
        rt = time.time() - t0
        rows.append(dict(
            run_id=rid, sc_transform=sctf, use_delays=use_delays, use_fic=use_fic,
            n_subjects=len(subs), n_sim=1,
            fc_corr_all=m["fc_all"], fc_corr_nz=m["fc_nz"],
            sim_sc_corr_all=m["ssc_all"], sim_sc_corr_nz=m["ssc_nz"],
            sim_length_corr_nz=m["slen_nz"], emp_sc_corr_all=m["esc_all"],
            emp_sc_corr_nz=m["esc_nz"], emp_length_corr_nz=m["elen_nz"],
            fc_rmse=m["rmse"], val_proxy="NA(forward-only)",
            runtime_sec=round(rt, 1), status=status, notes=";".join(note)))
        print(f"  [{rid}] {sctf} dly={int(use_delays)} fic={int(use_fic)} -> "
              f"sim~emp(nz)={m['fc_nz']:+.3f} sim~SC(nz)={m['ssc_nz']:+.3f} "
              f"sim~-len(nz)={m['slen_nz']:+.3f}  ({rt:.0f}s) {status}", flush=True)

    cols = ["run_id", "sc_transform", "use_delays", "use_fic", "n_subjects", "n_sim",
            "fc_corr_all", "fc_corr_nz", "sim_sc_corr_all", "sim_sc_corr_nz",
            "sim_length_corr_nz", "emp_sc_corr_all", "emp_sc_corr_nz",
            "emp_length_corr_nz", "fc_rmse", "val_proxy", "runtime_sec", "status", "notes"]
    out = os.path.join(config.OUTPUT_DIR, "forward_ablation_sc_delay_fic.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n| run | sc | dly | fic | sim~emp(all) | sim~emp(nz) | sim~SC(nz) | "
          "sim~-len(nz) | emp~SC(nz) | emp~-len(nz) | rmse | st |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run_id']} | {r['sc_transform'][:5]} | {int(r['use_delays'])} | "
              f"{int(r['use_fic'])} | {r['fc_corr_all']:+.3f} | {r['fc_corr_nz']:+.3f} | "
              f"{r['sim_sc_corr_nz']:+.3f} | {r['sim_length_corr_nz']:+.3f} | "
              f"{r['emp_sc_corr_nz']:+.3f} | {r['emp_length_corr_nz']:+.3f} | "
              f"{r['fc_rmse']:.3f} | {r['status']} |")
    print(f"\nsaved: {out}")
    a0 = rows[0]
    print("\nvs A0 baseline (sim~emp nz):")
    for r in rows[1:]:
        dd = r["fc_corr_nz"] - a0["fc_corr_nz"]
        print(f"  {r['run_id']}: {dd:+.3f} ({'better' if dd > 0 else 'worse/equal'})")


if __name__ == "__main__":
    main()
