"""Data loading, SC scaling, and train/val/test splitting.

The FC `.mat` file contains a (n_subjects, 3) object array per subject:

    col 0 -> subject ID (e.g. "sub-419077")
    col 1 -> FC matrix (115, 115), float64, NaN-affected
    col 2 -> FCD matrix (115, 115), float64, NaN-free

The SC `.mat` file is structured similarly:

    col 0 -> subject ID
    col 1 -> SC raw counts (uint16)
    col 2 -> SC log-transformed (float64)

Pipeline modules read these by `config.FC_COL`, `config.FCD_COL`,
and `config.SC_COL`.
"""
import os

import numpy as np
import pandas as pd
import scipy.io as sio

import config


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_mat(path):
    """Load a `.mat` file and return its `data` field."""
    try:
        return sio.loadmat(path)["data"]
    except Exception as e:
        raise RuntimeError(f"mat load failed: {path}\n{e}")


def _parse_ids(mat):
    """Extract subject ID strings from column 0 of a mat array."""
    ids = []
    for i in range(mat.shape[0]):
        cell = mat[i, 0]
        if isinstance(cell, np.ndarray):
            ids.append(str(cell.flatten()[0]))
        else:
            ids.append(str(cell))
    return ids


def load_atlas_labels(path=None):
    """Parse `atlas_115_labels.txt` into a list of dicts."""
    path = path or config.ATLAS_PATH
    if not os.path.exists(path):
        print(f"  ⚠️  atlas file not found: {path}")
        return None
    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                labels.append({
                    "idx": int(parts[0]),
                    "abbrev": parts[1],
                    "name": parts[2],
                })
    return labels


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_data():
    """Load FC/SC mat files, TSV, and optional BOLD timeseries.

    Returns
    -------
    df : pandas.DataFrame
        participants.tsv contents.
    fc_mat, sc_mat : np.ndarray
        Object arrays from the mat files.
    fc_ids, sc_ids : list[str]
    bold_mat, bold_ids : np.ndarray or None
    """
    print("  [data loading]")
    df = pd.read_csv(config.TSV_PATH, sep="\t")
    fc_mat = _load_mat(config.FC_PATH)
    sc_mat = _load_mat(config.SC_PATH)
    fc_ids = _parse_ids(fc_mat)
    sc_ids = _parse_ids(sc_mat)

    print(f"  FC: {fc_mat.shape[0]} subjects x {fc_mat.shape[1]} cols")
    print(f"    FC  <- col {config.FC_COL} (NaN -> 0)")
    print(f"    FCD <- col {config.FCD_COL} (direct use, no computation)")
    print(
        f"  SC: {sc_mat.shape[0]} subjects x {sc_mat.shape[1]} cols  "
        f"(weights=col {config.SC_WEIGHT_COL}, "
        f"lengths=col {config.SC_LENGTH_COL})"
    )

    _check_shapes(fc_mat, sc_mat)
    _record_nan_mask(fc_mat)
    _check_fcd_nan(fc_mat)

    bold_mat, bold_ids = _load_optional_bold()
    return df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids


def _check_shapes(fc_mat, sc_mat):
    """Strict region-count consistency check."""
    n = config.N_REGIONS
    expected = n * (n - 1) // 2
    assert config.FC_DIM == expected, (
        f"config.FC_DIM={config.FC_DIM} but n*(n-1)/2={expected} "
        f"with N_REGIONS={n}"
    )

    fc_test = fc_mat[0, config.FC_COL]
    assert fc_test.shape == (n, n), (
        f"FC col {config.FC_COL} shape {fc_test.shape} != ({n}, {n}). "
        f"Update config.N_REGIONS or the data path."
    )
    sc_w = sc_mat[0, config.SC_WEIGHT_COL]
    assert sc_w.shape == (n, n), (
        f"SC weight col {config.SC_WEIGHT_COL} shape {sc_w.shape} "
        f"!= ({n}, {n})"
    )
    if sc_mat.shape[1] > config.SC_LENGTH_COL:
        sc_l = sc_mat[0, config.SC_LENGTH_COL]
        assert sc_l.shape == (n, n), (
            f"SC length col {config.SC_LENGTH_COL} shape {sc_l.shape} "
            f"!= ({n}, {n})"
        )
        # Must be distinct matrices (raw counts vs tract length)
        a = np.asarray(sc_w, dtype=np.float64)
        b = np.asarray(sc_l, dtype=np.float64)
        assert not np.allclose(a, b), (
            f"SC weight col {config.SC_WEIGHT_COL} and length col "
            f"{config.SC_LENGTH_COL} are identical — column assignment "
            f"is wrong."
        )


def _record_nan_mask(fc_mat):
    """Record the NaN mask for diagnostics (does not gate features)."""
    n = config.N_REGIONS
    nan_mask = np.zeros((n, n), dtype=bool)
    for i in range(fc_mat.shape[0]):
        fc_use = fc_mat[i, config.FC_COL].astype(np.float64)
        if fc_use.shape == (n, n):
            nan_mask |= np.isnan(fc_use)
    nan_regions = np.where(nan_mask.any(axis=1))[0]
    print(
        f"  NaN regions in FC col {config.FC_COL}: {len(nan_regions)} "
        f"(replaced with 0, mask not applied)"
    )

    # FC_DIM stays 6555 so simulated and observed FC have matching dim.
    # NaN regions are constant 0 -> contribute no variance to PCA.
    config.NAN_MASK = None
    config.NAN_REGIONS = nan_regions.tolist()


def _check_fcd_nan(fc_mat):
    fcd_nan_total = 0
    for i in range(fc_mat.shape[0]):
        fcd = fc_mat[i, config.FCD_COL].astype(np.float64)
        fcd_nan_total += int(np.isnan(fcd).sum())
    if fcd_nan_total > 0:
        print(
            f"  ⚠️  FCD col {config.FCD_COL} has {fcd_nan_total} NaNs - "
            f"replaced with 0"
        )
    else:
        print(f"  ✓ FCD col {config.FCD_COL}: no NaN")


def _load_optional_bold():
    if not os.path.exists(config.BOLD_PATH):
        print("  BOLD file not present")
        config.HAS_BOLD = False
        return None, None
    try:
        bold_mat = _load_mat(config.BOLD_PATH)
        bold_ids = _parse_ids(bold_mat)
        config.HAS_BOLD = True
        print(f"  ✓ BOLD timeseries: {len(bold_ids)} subjects")
        return bold_mat, bold_ids
    except Exception as e:
        print(f"  ⚠️  BOLD load failed: {e}")
        config.HAS_BOLD = False
        return None, None


# ---------------------------------------------------------------------------
# Group filtering
# ---------------------------------------------------------------------------

def get_target_subjects(df, fc_ids, sc_ids):
    """Return subjects matching `config.GROUP_FILTER` present in both mats."""
    group, treatment = config.GROUP_FILTER
    candidates = df[
        (df["group"] == group) & (df["treatment"] == treatment)
    ]["participant_id"].tolist()
    targets = [s for s in candidates if s in fc_ids and s in sc_ids]
    print(
        f"  {group}+{treatment}: {len(candidates)} candidates -> "
        f"{len(targets)} retained"
    )
    return targets


# ---------------------------------------------------------------------------
# SC scaling
# ---------------------------------------------------------------------------

def _scale_weights(weights):
    """Apply log1p + max-norm + sparse mask to SC."""
    weights = weights.copy().astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    sc_mask = (weights > 0).astype(np.float32)
    if sc_mask.sum() == 0:
        raise RuntimeError("SC has no positive edges.")
    weights = np.log1p(weights + 0.5)
    wmax = float(np.max(weights))
    if wmax > 0:
        weights = weights / wmax
    return weights * sc_mask


# ---------------------------------------------------------------------------
# Per-subject loading
# ---------------------------------------------------------------------------

def _load_fc_fcd(fc_mat, fc_ids, sid):
    """Load FC (col 1) and FCD (col 2) for one subject.

    Returns
    -------
    fc : np.ndarray
        FC matrix with NaN replaced by 0, symmetrized, zero diagonal.
    fcd : np.ndarray
        FCD matrix, symmetrized, zero diagonal.
    fc_nan : np.ndarray
        Boolean mask of original NaN positions (diagnostics only).
    """
    idx = fc_ids.index(sid)

    fc_raw = fc_mat[idx, config.FC_COL].copy().astype(np.float64)
    nan_pos = np.isnan(fc_raw)
    fc = np.nan_to_num(fc_raw, nan=0.0)
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)

    fcd_raw = fc_mat[idx, config.FCD_COL].copy().astype(np.float64)
    fcd = np.nan_to_num(fcd_raw, nan=0.0)
    fcd = (fcd + fcd.T) / 2.0
    np.fill_diagonal(fcd, 0.0)

    return fc, fcd, nan_pos


def get_subject_data(sid, fc_mat, sc_mat, fc_ids, sc_ids,
                     bold_mat=None, bold_ids=None):
    """Bundle FC, FCD, SC, tract lengths, delays, and optional BOLD.

    Returned dict keys
    ------------------
    fc          : (N, N) Pearson FC, NaN→0, symmetrized, zero diagonal
    fcd         : (N, N) FCD matrix (used only if USE_FCD)
    fc_nan      : (N, N) bool, original NaN positions (diagnostic)
    sc          : (N, N) coupling weight, log1p + max-norm
    lengths_mm  : (N, N) tract length (mm)        — for delay calc
    delays      : (N, N) delay (ms) = lengths/velocity
    bold        : (T, N) optional ROI BOLD time series
    """
    from simulator import compute_delay_matrix

    fc, fcd, fc_nan = _load_fc_fcd(fc_mat, fc_ids, sid)

    sc_idx = sc_ids.index(sid)
    n = config.N_REGIONS

    # SC weights (col SC_WEIGHT_COL): raw counts → log1p + max-norm
    sc_raw = sc_mat[sc_idx, config.SC_WEIGHT_COL]
    assert sc_raw.shape == (n, n), (
        f"{sid}: SC weight shape {sc_raw.shape} != ({n}, {n})"
    )
    sc = sc_raw.copy().astype(np.float64)
    sc = (sc + sc.T) / 2.0
    sc = _scale_weights(sc).astype(np.float64)

    # Tract lengths (col SC_LENGTH_COL): mm
    has_length_col = sc_mat.shape[1] > config.SC_LENGTH_COL
    if has_length_col:
        len_raw = sc_mat[sc_idx, config.SC_LENGTH_COL]
        assert len_raw.shape == (n, n), (
            f"{sid}: SC length shape {len_raw.shape} != ({n}, {n})"
        )
        lengths_mm = len_raw.copy().astype(np.float64)
        lengths_mm = (lengths_mm + lengths_mm.T) / 2.0
        np.fill_diagonal(lengths_mm, 0.0)
        lengths_mm = lengths_mm.astype(np.float32)
    else:
        print(
            f"  ⚠️ {sid}: tract length column missing — "
            f"using 1/sc proxy as fallback (degrades delays)."
        )
        lengths_mm = None

    # Delays (ms). 1 m/s == 1 mm/ms so no unit conversion needed.
    delays = compute_delay_matrix(
        sc, config.VELOCITY_M_PER_S, lengths_mm=lengths_mm,
    )

    # Per-subject runtime asserts
    assert fc.shape == (n, n), f"{sid}: FC shape {fc.shape}"
    assert sc.shape == (n, n), f"{sid}: SC shape {sc.shape}"
    if lengths_mm is not None:
        assert lengths_mm.shape == (n, n), (
            f"{sid}: lengths shape {lengths_mm.shape}"
        )
        # SC weight (∈[0,1]) and length (mm-scale, max ~20mm) must differ
        assert not np.allclose(sc, lengths_mm), (
            f"{sid}: SC weights and lengths are identical "
            f"(column assignment likely wrong)"
        )
        lpos = lengths_mm[lengths_mm > 0]
        if lpos.size > 0:
            lmean = float(lpos.mean())
            assert 0.1 < lmean < 100.0, (
                f"{sid}: length mean {lmean:.3f} not in plausible "
                f"mm-scale (0.1, 100). Wrong SC_LENGTH_COL?"
            )
    assert delays.shape == (n, n), f"{sid}: delays shape {delays.shape}"
    assert np.isfinite(delays).all(), f"{sid}: non-finite delays"

    data = {
        "fc":         fc,
        "fcd":        fcd,
        "fc_nan":     fc_nan,
        "sc":         sc,
        "lengths_mm": lengths_mm if lengths_mm is not None
        else np.zeros((n, n), dtype=np.float32),
        "delays":     delays,
    }
    if bold_mat is not None and bold_ids is not None and sid in bold_ids:
        bold = bold_mat[bold_ids.index(sid), 1].copy().astype(np.float64)
        data["bold"] = bold
    return data


def load_all_subjects(subjects, fc_mat, sc_mat, fc_ids, sc_ids,
                      bold_mat=None, bold_ids=None):
    """Load all listed subjects into a `{sid: data}` dict."""
    data = {}
    for sid in subjects:
        data[sid] = get_subject_data(
            sid, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids,
        )
        _print_subject_info(sid, data[sid])
    return data


def _print_subject_info(sid, d):
    n_edges = int((d["sc"] > 0).sum())
    nan_cnt = int(d["fc_nan"].sum())
    fcd_lo = float(d["fcd"].min())
    fcd_hi = float(d["fcd"].max())
    msg = (
        f"    {sid}: SC nonzero={n_edges}, FC NaN={nan_cnt}, "
        f"FCD range=[{fcd_lo:.3f}, {fcd_hi:.3f}]"
    )
    if "lengths_mm" in d:
        lm = d["lengths_mm"]
        lpos = lm[lm > 0]
        msg += f"  length=[{lpos.min():.1f}, {lpos.max():.1f}]mm"
    delays = d["delays"]
    if delays is not None and (delays > 0).any():
        dmin = float(delays[delays > 0].min())
        dmax = float(delays.max())
        msg += f"  delay=[{dmin:.2f}, {dmax:.2f}]ms"
    if "bold" in d:
        msg += f"  +BOLD{d['bold'].shape}"
    print(msg)


# ---------------------------------------------------------------------------
# Subject split
# ---------------------------------------------------------------------------

def three_way_split(subjects, n_train=None, n_val=None, n_test=None,
                    seed=None):
    """Deterministic train/val/test split using `config.SEED`."""
    n_train = config.N_TRAIN if n_train is None else n_train
    n_val = config.N_VAL if n_val is None else n_val
    n_test = config.N_TEST if n_test is None else n_test
    seed = config.SEED if seed is None else seed

    total = n_train + n_val + n_test
    if len(subjects) < total:
        raise ValueError(
            f"Not enough subjects: {len(subjects)} < required {total} "
            f"(train={n_train}, val={n_val}, test={n_test})"
        )

    rng = np.random.RandomState(seed)
    shuffled = sorted(subjects)
    rng.shuffle(shuffled)

    i = 0
    train = shuffled[i:i + n_train]
    i += n_train
    val = shuffled[i:i + n_val]
    i += n_val
    test = shuffled[i:i + n_test]

    print(f"  train ({len(train)}): {train}")
    print(f"  val   ({len(val)}):   {val}")
    print(f"  test  ({len(test)}):  {test}")
    return train, val, test


def four_way_split(subjects, **kwargs):
    """Deprecated alias for `three_way_split`."""
    valid = {
        k: v for k, v in kwargs.items()
        if k in ("n_train", "n_val", "n_test", "seed")
    }
    return three_way_split(subjects, **valid)
