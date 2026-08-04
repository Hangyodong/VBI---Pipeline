"""Export test-subject simulated + empirical FC matrices to CSV.

For each test subject, writes under ``<OUTPUT_DIR>/fc_csv/``:

  - ``sim_fc_<sid>.csv``    expected simulated FC = mean over the N_TEST_RESIM
                            resimulated FCs, full (N, N) matrix.
  - ``emp_fc_<sid>.csv``    empirical / observed FC target, full (N, N) matrix.
  - ``node_fit_<sid>.csv``  per-region correlation between the sim and emp FC
                            row (diagonal excluded), length-N vector.

Plus one cross-subject summary:

  - ``node_fit_summary.csv``  region x subject node-fit corr table + a ``mean``
                              column (mean over subjects, ignoring NaN).

The per-region FC-row correlation quantifies *which nodes* are reproduced
well vs poorly — the direct probe for the front/back label-order asymmetry
(issue #1). A high value means that region's whole connectivity profile is
recovered by the model; a low value means it is not.
"""
import os
import csv

import numpy as np


def _node_fit_corr(sim_fc, emp_fc, nan_mask=None):
    """Per-region Pearson corr between sim and emp FC rows (diag excluded).

    Returns a length-N vector; entries with <2 valid off-diagonal edges or a
    zero-variance row are NaN.
    """
    n = sim_fc.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    nm = None if nan_mask is None else np.asarray(nan_mask, dtype=bool)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False                       # drop self (diagonal)
        if nm is not None:
            keep &= ~nm[i]
        a = np.asarray(emp_fc[i], dtype=np.float64)[keep]
        b = np.asarray(sim_fc[i], dtype=np.float64)[keep]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 2 and a[m].std() > 0 and b[m].std() > 0:
            out[i] = float(np.corrcoef(a[m], b[m])[0, 1])
    return out


def export_test_fc_csv(test_summary, subject_data, out_dir, verbose=True):
    """Write per-test-subject sim/emp FC + node-fit CSVs.

    Parameters
    ----------
    test_summary : dict
        Output of ``evaluation.final_test.final_test`` — must carry
        ``per_subject`` with ``sid`` / ``fc_obs`` / ``fc_preds`` per entry.
    subject_data : dict
        sid -> subject record (used only for the optional ``fc_nan`` mask).
    out_dir : str
        Base output directory; CSVs land in ``<out_dir>/fc_csv/``.

    Returns
    -------
    str | None
        The CSV directory, or None when there is nothing to write.
    """
    per = test_summary.get("per_subject", []) if test_summary else []
    if not per:
        if verbose:
            print("  [export] no test per_subject results — CSV export skipped")
        return None

    csv_dir = os.path.join(out_dir, "fc_csv")
    os.makedirs(csv_dir, exist_ok=True)

    node_tbl = {}                              # sid -> node-fit vector
    for r in per:
        sid = r["sid"]
        preds = r.get("fc_preds") or []
        if not preds:
            continue
        emp = np.asarray(r["fc_obs"], dtype=np.float64)
        sim = np.mean(
            np.stack([np.asarray(p, dtype=np.float64) for p in preds], axis=0),
            axis=0,
        )
        nan_mask = (subject_data.get(sid, {}) or {}).get("fc_nan")

        np.savetxt(os.path.join(csv_dir, f"sim_fc_{sid}.csv"),
                   sim, delimiter=",", fmt="%.6f")
        np.savetxt(os.path.join(csv_dir, f"emp_fc_{sid}.csv"),
                   emp, delimiter=",", fmt="%.6f")
        nf = _node_fit_corr(sim, emp, nan_mask)
        np.savetxt(os.path.join(csv_dir, f"node_fit_{sid}.csv"),
                   nf, delimiter=",", fmt="%.6f")
        node_tbl[sid] = nf

    if node_tbl:
        sids = list(node_tbl.keys())
        mat = np.column_stack([node_tbl[s] for s in sids])      # (N, S)
        mean_col = np.nanmean(mat, axis=1)
        path = os.path.join(csv_dir, "node_fit_summary.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["region"] + [str(s) for s in sids] + ["mean"])
            for i in range(mat.shape[0]):
                row = [i] + [f"{mat[i, j]:.6f}" for j in range(len(sids))]
                row.append(f"{mean_col[i]:.6f}")
                w.writerow(row)

    if verbose:
        print(f"  [export] test FC CSVs -> {csv_dir}")
        print(f"           sim_fc / emp_fc / node_fit per subject "
              f"({len(node_tbl)} subjects) + node_fit_summary.csv")
    return csv_dir
