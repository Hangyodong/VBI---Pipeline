"""HCP data loader — drop-in replacement for ``data_loader`` for HCP SC/FC.

HCP file layout (differs from MPTP):

  HCP_FC.mat  (MATLAB v7, scipy): variable ``C`` of shape (n_subj, 2)
      col 0 -> subject id (int, e.g. 100206)
      col 1 -> FC matrix (381, 381) float64        (NO FCD column)

  HCP_SC.mat  (MATLAB v7.3, h5py): variable ``data`` of shape (3, n_subj)
      row 0 -> subject id (uint16 char codes -> str "100206")
      row 1 -> SC weights (raw streamline counts, 381x381)
      row 2 -> SC tract lengths (mm, 381x381)

Exposes the same function names the pipeline Step-1 cell calls, so a notebook
only needs ``import data_loader_hcp as data_loader``:

    load_raw_data() -> get_target_subjects() -> three_way_split() -> load_all_subjects()

Subject selection: the ``config.N_SUBJECTS`` smallest subject ids (ascending),
present in BOTH FC and SC. ``three_way_split`` (reused) then shuffles those.
No participants.tsv / group filter, no FCD, no empirical BOLD.
"""
import numpy as np
import scipy.io as sio

import config
# reuse format-agnostic helpers from the MPTP loader
from data_loader import _scale_weights, three_way_split, four_way_split  # noqa: F401
from simulation.delays import compute_delay_matrix


# ---------------------------------------------------------------------------
# Low-level HCP readers
# ---------------------------------------------------------------------------

def _load_hcp_fc(path):
    """Load HCP_FC.mat -> (C object array, fc_ids:list[int])."""
    C = sio.loadmat(path)["C"]                       # (n, 2)
    fc_ids = [int(np.asarray(C[i, 0]).flatten()[0]) for i in range(C.shape[0])]
    return C, fc_ids


def _decode_sc_ids(path):
    """Read HCP_SC.mat subject ids (uint16 char codes) -> list[int]."""
    import h5py
    with h5py.File(path, "r") as f:
        d = f["data"]                                # (3, n) refs
        ids = []
        for j in range(d.shape[1]):
            a = np.asarray(f[d[0, j]][()]).flatten()
            ids.append(int("".join(chr(int(x)) for x in a)))
    return ids


# ---------------------------------------------------------------------------
# Pipeline-facing API (mirrors data_loader)
# ---------------------------------------------------------------------------

def load_raw_data():
    """Load HCP FC (scipy) + SC ids (h5py). SC matrices are read lazily later.

    Returns the same 7-tuple shape as data_loader.load_raw_data:
        (df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids)
    here df=None, fc_mat=C array, sc_mat=SC_PATH (str, opened per-subject),
    bold_mat=bold_ids=None.
    """
    print("  [HCP data loading]")
    C, fc_ids = _load_hcp_fc(config.FC_PATH)
    sc_ids = _decode_sc_ids(config.SC_PATH)
    n = config.N_REGIONS
    expected = n * (n - 1) // 2
    assert config.FC_DIM == expected, (
        f"config.FC_DIM={config.FC_DIM} but n*(n-1)/2={expected} "
        f"(N_REGIONS={n}); set N_REGIONS=381 for HCP."
    )
    fc0 = np.asarray(C[0, config.FC_COL])
    # cortical-only: raw file may be larger than N_REGIONS (e.g. 381 raw vs 360
    # cortical); _cortical_slice keeps the first N_REGIONS. Require raw >= n.
    assert fc0.shape[0] >= n and fc0.shape[0] == fc0.shape[1], (
        f"HCP FC matrix is {fc0.shape}, need square with size >= N_REGIONS={n}."
    )
    if fc0.shape[0] > n:
        print(f"  [cortical-only] slicing raw FC/SC {fc0.shape[0]} -> first {n} regions")
    print(f"  FC: {C.shape[0]} subjects x {fc0.shape} (FC col {config.FC_COL}, no FCD)")
    print(f"  SC: {len(sc_ids)} subjects (v7.3; weights col 1, lengths col 2)")
    return None, C, config.SC_PATH, fc_ids, sc_ids, None, None


def get_target_subjects(df, fc_ids, sc_ids):
    """The ``config.N_SUBJECTS`` smallest subject ids present in both FC and SC.

    ``df`` is ignored (HCP has no participants.tsv / group filter).
    """
    n_req = int(getattr(config, "N_SUBJECTS", 100))
    common = sorted(set(int(x) for x in fc_ids) & set(int(x) for x in sc_ids))
    targets = common[:n_req]
    print(
        f"  HCP: {len(common)} subjects in both FC & SC -> "
        f"first {len(targets)} by subject id (ascending): "
        f"{targets[0]}..{targets[-1]}"
    )
    return targets


def _cortical_slice(mat, n):
    """Cortical-only: keep the first ``n`` regions (drop trailing subcortical).

    HCP 381 = 360 Glasser cortical + 21 subcortical (last 21). The subcortical
    nodes are SC hubs but FC-weak (deep, low-SNR BOLD), which depresses SC-FC
    correspondence. Setting N_REGIONS=360 slices them off. No-op if mat is
    already <= n.
    """
    if mat.shape[0] > n:
        return mat[:n, :n]
    return mat


def _build_subject_data(sid, fc_mat, fc_index, h5f, sc_refs, sc_index, n):
    """Bundle FC / SC / lengths / delays for one HCP subject."""
    # ---- FC (scipy C, col config.FC_COL) ----
    fc_raw = _cortical_slice(
        np.asarray(fc_mat[fc_index[sid], config.FC_COL]).astype(np.float64), n)
    assert fc_raw.shape == (n, n), f"{sid}: FC shape {fc_raw.shape} != ({n},{n})"
    fc_nan = np.isnan(fc_raw)
    fc = np.nan_to_num(fc_raw, nan=0.0)
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)

    # HCP has no FCD; supply zeros (USE_FCD must be False).
    fcd = np.zeros((n, n), dtype=np.float64)

    # ---- SC weights (row 1) + tract lengths (row 2), h5py deref ----
    col = sc_index[sid]
    w_raw = _cortical_slice(np.asarray(h5f[sc_refs[1, col]][()]).astype(np.float64), n)
    assert w_raw.shape == (n, n), f"{sid}: SC weight shape {w_raw.shape}"
    sc = (w_raw + w_raw.T) / 2.0
    sc = _scale_weights(sc).astype(np.float64)

    len_raw = _cortical_slice(np.asarray(h5f[sc_refs[2, col]][()]).astype(np.float64), n)
    assert len_raw.shape == (n, n), f"{sid}: SC length shape {len_raw.shape}"
    lengths_mm = (len_raw + len_raw.T) / 2.0
    np.fill_diagonal(lengths_mm, 0.0)
    lengths_mm = lengths_mm.astype(np.float32)

    delays = compute_delay_matrix(sc, config.VELOCITY_M_PER_S, lengths_mm=lengths_mm)

    assert sc.shape == (n, n) and delays.shape == (n, n)
    assert np.isfinite(delays).all(), f"{sid}: non-finite delays"
    return {
        "fc": fc, "fcd": fcd, "fc_nan": fc_nan,
        "sc": sc, "lengths_mm": lengths_mm, "delays": delays,
    }


def _compute_group_fc(fc_mat, fc_ids, n):
    """Group-averaged FC (cortical-sliced) over ALL HCP subjects.

    Per-subject FC is noisy -> SC-FC correspondence ~0.05. Averaging across
    subjects cancels idiosyncratic noise and reveals the shared SC-shaped
    structure (group SC-FC ~0.18). Using the group FC as the inference target
    raises the achievable ceiling for an SC-driven model.
    """
    acc = np.zeros((n, n), dtype=np.float64)
    cnt = 0
    for i in range(len(fc_ids)):
        m = _cortical_slice(np.asarray(fc_mat[i, config.FC_COL]).astype(np.float64), n)
        m = np.nan_to_num(m, nan=0.0)
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        acc += m
        cnt += 1
    return acc / max(cnt, 1)


def load_all_subjects(subjects, fc_mat, sc_mat, fc_ids, sc_ids,
                      bold_mat=None, bold_ids=None):
    """Load the listed HCP subjects into a ``{sid: data}`` dict.

    ``sc_mat`` is the HCP_SC.mat path; SC matrices for ``subjects`` only are
    dereferenced from the v7.3 file (memory-lean — never loads all 1040).

    When ``config.GROUP_AVG_FC`` is set, every subject's FC target is replaced
    by the group-averaged FC (computed over all subjects); SC stays per-subject.
    """
    import h5py
    n = config.N_REGIONS
    fc_index = {int(s): i for i, s in enumerate(fc_ids)}
    data = {}
    with h5py.File(sc_mat, "r") as f:
        sc_refs = f["data"]                          # (3, nsub) refs
        sc_index = {}
        for j in range(sc_refs.shape[1]):
            a = np.asarray(f[sc_refs[0, j]][()]).flatten()
            sc_index[int("".join(chr(int(x)) for x in a))] = j
        for sid in subjects:
            data[sid] = _build_subject_data(
                int(sid), fc_mat, fc_index, f, sc_refs, sc_index, n
            )
            d = data[sid]
            n_edges = int((d["sc"] > 0).sum())
            lpos = d["lengths_mm"][d["lengths_mm"] > 0]
            dl = d["delays"][d["delays"] > 0]
            print(
                f"    {sid}: SC nnz={n_edges}, FC NaN={int(d['fc_nan'].sum())}, "
                f"len=[{lpos.min():.1f},{lpos.max():.1f}]mm, "
                f"delay=[{(dl.min() if dl.size else 0):.2f},"
                f"{(dl.max() if dl.size else 0):.2f}]ms"
            )

    if getattr(config, "GROUP_AVG_FC", False):
        gfc = _compute_group_fc(fc_mat, fc_ids, n)
        gnan = np.zeros((n, n), dtype=bool)
        for sid in subjects:
            data[sid]["fc"] = gfc.copy()
            data[sid]["fc_nan"] = gnan.copy()
        print(f"  [GROUP_AVG_FC] every subject's FC target -> group mean "
              f"over {len(fc_ids)} subjects (cortical-only n={n}). "
              f"FC range=[{gfc.min():.3f},{gfc.max():.3f}] mean={gfc.mean():.3f}")
    return data
