"""Per-subject multi-channel SC table for SC-conditioned amortized NPE.

Pure-numpy (no torch, no GPU). Builds a per-subject multi-channel structural
connectivity (SC) table that a matrix encoder looks up by subject index, plus a
leakage-safe per-channel z-score scaler fit on TRAIN rows only.

Channels (fixed order, see :data:`CHANNELS`):
    sc_weight : maxnorm SC weights in [0, 1]
    delay     : tract-length delays in ms
    sc_mask   : binary 0/1 connectivity mask (sc > 0)

The scaler z-scores sc_weight and delay using statistics computed over the
off-diagonal NONZERO entries of the TRAIN table only (zeros and the diagonal are
excluded). The sc_mask channel is left untouched (identity), since it is binary.
"""

from __future__ import annotations

import numpy as np

# Fixed channel order. Do not reorder — encoders depend on this layout.
CHANNELS = ("sc_weight", "delay", "sc_mask")

# Floor for per-channel std to avoid divide-by-zero.
_STD_FLOOR = 1e-8


def channel_index(name, channels=CHANNELS):
    """Return the integer channel index for ``name`` within ``channels``."""
    channels = tuple(channels)
    if name not in channels:
        raise KeyError(
            f"unknown channel {name!r}; available channels: {channels}"
        )
    return channels.index(name)


def _build_one_channel(name, entry):
    """Build a single (N, N) channel array for one subject's data ``entry``."""
    sc = np.asarray(entry["sc"], dtype=np.float64)
    if name == "sc_weight":
        return sc
    if name == "delay":
        return np.asarray(entry["delays"], dtype=np.float64)
    if name == "sc_mask":
        return (sc > 0).astype(np.float64)
    raise KeyError(f"unknown channel {name!r}")


def build_sc_table(subject_data, subjects, channels=CHANNELS):
    """Build a per-subject multi-channel SC table.

    Parameters
    ----------
    subject_data : dict
        ``{sid: {"sc": (N, N) maxnorm float, "delays": (N, N) ms float, ...}}``.
    subjects : list
        Ordered list of sids to include (e.g. the train list). Defines row order.
    channels : tuple of str, optional
        Channel order (defaults to :data:`CHANNELS`).

    Returns
    -------
    table : np.ndarray
        ``float32`` array of shape ``(len(subjects), C, N, N)`` with
        ``C = len(channels)`` and channel order ``channels``.
    idmap : dict
        ``{int(sid): row_index}``.
    """
    channels = tuple(channels)
    subjects = list(subjects)
    if len(subjects) == 0:
        raise ValueError("subjects list is empty")

    # Determine N from the first subject's SC and validate consistency.
    first_sid = subjects[0]
    if first_sid not in subject_data:
        raise KeyError(f"subject {first_sid!r} not found in subject_data")
    n_ref = np.asarray(subject_data[first_sid]["sc"]).shape[0]

    n_subj = len(subjects)
    n_chan = len(channels)
    table = np.empty((n_subj, n_chan, n_ref, n_ref), dtype=np.float32)
    idmap = {}

    for row, sid in enumerate(subjects):
        if sid not in subject_data:
            raise KeyError(f"subject {sid!r} not found in subject_data")
        entry = subject_data[sid]
        for c, name in enumerate(channels):
            mat = _build_one_channel(name, entry)
            # Square + (N, N) + matches reference N across all subjects.
            assert mat.ndim == 2, (
                f"subject {sid!r} channel {name!r} is not 2D: shape {mat.shape}"
            )
            assert mat.shape[0] == mat.shape[1], (
                f"subject {sid!r} channel {name!r} not square: shape {mat.shape}"
            )
            assert mat.shape == (n_ref, n_ref), (
                f"subject {sid!r} channel {name!r} shape {mat.shape} != "
                f"({n_ref}, {n_ref})"
            )
            assert np.all(np.isfinite(mat)), (
                f"subject {sid!r} channel {name!r} has non-finite entries"
            )
            table[row, c] = mat.astype(np.float32)
        idmap[int(sid)] = row

    return table, idmap


class ScChannelScaler:
    """Per-channel z-score scaler fit on TRAIN rows only (no leakage).

    sc_weight and delay are z-scored using mean/std over the off-diagonal
    NONZERO entries of the TRAIN table (zeros and diagonal excluded). The
    sc_mask channel is left untouched (identity), since it is binary.

    After :meth:`fit`, statistics are frozen; :meth:`transform` never refits.
    """

    def __init__(self):
        self.channels_ = None
        self.mean_ = None  # (C,) float64
        self.std_ = None   # (C,) float64
        self.identity_ = None  # (C,) bool — True = channel left untouched

    @staticmethod
    def _edge_values(channel_table):
        """Off-diagonal nonzero entries across all rows of a (S, N, N) stack."""
        s, n, _ = channel_table.shape
        off_diag = ~np.eye(n, dtype=bool)
        mask = np.broadcast_to(off_diag, channel_table.shape)
        vals = channel_table[mask]
        return vals[vals != 0]

    def fit(self, train_table, channels=CHANNELS):
        """Fit per-channel stats on ``train_table`` rows only. Returns self.

        Parameters
        ----------
        train_table : np.ndarray
            ``(S, C, N, N)`` table of TRAIN rows only.
        channels : tuple of str, optional
            Channel order matching ``train_table`` (defaults to :data:`CHANNELS`).
        """
        train_table = np.asarray(train_table)
        if train_table.ndim != 4:
            raise ValueError(
                f"train_table must be 4D (S, C, N, N); got {train_table.shape}"
            )
        channels = tuple(channels)
        n_chan = train_table.shape[1]
        if n_chan != len(channels):
            raise ValueError(
                f"train_table has {n_chan} channels but channels has "
                f"{len(channels)} names"
            )

        self.channels_ = channels
        mean = np.zeros(n_chan, dtype=np.float64)
        std = np.ones(n_chan, dtype=np.float64)
        identity = np.zeros(n_chan, dtype=bool)

        for c, name in enumerate(channels):
            if name == "sc_mask":
                # Binary channel: identity transform.
                mean[c] = 0.0
                std[c] = 1.0
                identity[c] = True
                continue
            chan = train_table[:, c, :, :].astype(np.float64)
            vals = self._edge_values(chan)
            if vals.size == 0:
                # No edges to estimate from: fall back to identity.
                mean[c] = 0.0
                std[c] = 1.0
            else:
                mean[c] = float(vals.mean())
                std[c] = max(float(vals.std()), _STD_FLOOR)

        self.mean_ = mean
        self.std_ = std
        self.identity_ = identity
        return self

    def transform(self, table):
        """Apply frozen per-channel z-score. Never refits. Returns float32."""
        if self.mean_ is None or self.std_ is None or self.identity_ is None:
            raise RuntimeError("ScChannelScaler.transform called before fit()")
        table = np.asarray(table)
        if table.ndim != 4:
            raise ValueError(
                f"table must be 4D (S, C, N, N); got {table.shape}"
            )
        n_chan = table.shape[1]
        if n_chan != len(self.mean_):
            raise ValueError(
                f"table has {n_chan} channels but scaler fit on {len(self.mean_)}"
            )

        out = table.astype(np.float32, copy=True)
        for c in range(n_chan):
            if self.identity_[c]:
                continue
            out[:, c, :, :] = (
                (table[:, c, :, :].astype(np.float64) - self.mean_[c])
                / self.std_[c]
            ).astype(np.float32)
        return out

    def fit_transform(self, train_table, channels=CHANNELS):
        """Fit on ``train_table`` then transform and return it (float32)."""
        return self.fit(train_table, channels=channels).transform(train_table)

    def to_dict(self):
        """Serialize scaler state to a plain dict for persistence."""
        if self.mean_ is None:
            raise RuntimeError("ScChannelScaler.to_dict called before fit()")
        return {
            "channels_": list(self.channels_),
            "mean_": np.asarray(self.mean_, dtype=np.float64).tolist(),
            "std_": np.asarray(self.std_, dtype=np.float64).tolist(),
            "identity_": np.asarray(self.identity_, dtype=bool).tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        """Reconstruct a frozen scaler from :meth:`to_dict` output."""
        obj = cls()
        obj.channels_ = tuple(d["channels_"])
        obj.mean_ = np.asarray(d["mean_"], dtype=np.float64)
        obj.std_ = np.asarray(d["std_"], dtype=np.float64)
        obj.identity_ = np.asarray(d["identity_"], dtype=bool)
        return obj


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8
    n_subj = 5
    sparsity = 0.30

    subject_data = {}
    sids = [100 + i for i in range(n_subj)]
    for sid in sids:
        # Random symmetric ~30% sparse SC in [0, 1] (maxnorm-like).
        raw = rng.random((n, n))
        keep = rng.random((n, n)) < sparsity
        keep = np.triu(keep, k=1)
        keep = keep | keep.T  # symmetric, zero diagonal
        sc = raw * keep
        sc = np.triu(sc, k=1)
        sc = sc + sc.T  # symmetric
        if sc.max() > 0:
            sc = sc / sc.max()  # maxnorm
        mask = (sc > 0).astype(np.float64)
        delays = rng.random((n, n)) * mask  # ms, only where SC present
        delays = np.triu(delays, k=1)
        delays = delays + delays.T
        subject_data[sid] = {"sc": sc, "delays": delays}

    # Build full table over all 5 subjects.
    table, idmap = build_sc_table(subject_data, sids)
    assert table.shape == (5, 3, 8, 8), f"bad table shape {table.shape}"
    assert table.dtype == np.float32, f"bad dtype {table.dtype}"
    assert idmap == {sid: i for i, sid in enumerate(sids)}, f"bad idmap {idmap}"

    # mask channel must be binary.
    mi = channel_index("sc_mask")
    mask_chan = table[:, mi, :, :]
    assert np.all((mask_chan == 0) | (mask_chan == 1)), "mask not binary"
    assert np.all(np.isfinite(table)), "table has non-finite entries"

    # Fit scaler on TRAIN rows [0, 1, 2] only; transform all 5.
    train_rows = [0, 1, 2]
    train_table = table[train_rows]
    scaler = ScChannelScaler().fit(train_table)
    scaled = scaler.transform(table)

    # scaler.mean_ for sc_weight must equal the train-only off-diagonal nonzero
    # edge mean computed independently here.
    wi = channel_index("sc_weight")
    di = channel_index("delay")

    def _edge_mean(stack):
        off = ~np.eye(stack.shape[-1], dtype=bool)
        m = np.broadcast_to(off, stack.shape)
        v = stack[m]
        v = v[v != 0]
        return float(v.mean()), v

    w_mean_ref, _ = _edge_mean(train_table[:, wi, :, :].astype(np.float64))
    d_mean_ref, _ = _edge_mean(train_table[:, di, :, :].astype(np.float64))
    assert np.isclose(scaler.mean_[wi], w_mean_ref), (
        f"sc_weight mean mismatch {scaler.mean_[wi]} != {w_mean_ref}"
    )
    assert np.isclose(scaler.mean_[di], d_mean_ref), (
        f"delay mean mismatch {scaler.mean_[di]} != {d_mean_ref}"
    )

    # Transformed sc_weight/delay must be ~mean0 over TRAIN edges. Select edges
    # by the (untouched) mask channel so that transformed values landing exactly
    # on 0 are not spuriously dropped.
    train_edge_mask = scaled[train_rows][:, mi, :, :] > 0  # original nonzero edges
    w_train_edges = scaled[train_rows][:, wi, :, :][train_edge_mask].astype(np.float64)
    d_train_edges = scaled[train_rows][:, di, :, :][train_edge_mask].astype(np.float64)
    assert abs(float(w_train_edges.mean())) < 1e-4, (
        f"transformed sc_weight train-edge mean not ~0: {w_train_edges.mean()}"
    )
    assert abs(float(d_train_edges.mean())) < 1e-4, (
        f"transformed delay train-edge mean not ~0: {d_train_edges.mean()}"
    )

    # mask channel must still be 0/1 after transform (identity).
    scaled_mask = scaled[:, mi, :, :]
    assert np.all((scaled_mask == 0) | (scaled_mask == 1)), "mask altered by scaler"
    assert np.array_equal(scaled_mask, mask_chan), "mask not identity-preserved"

    # Round-trip persistence.
    restored = ScChannelScaler.from_dict(scaler.to_dict())
    assert np.allclose(restored.mean_, scaler.mean_)
    assert np.allclose(restored.std_, scaler.std_)
    assert np.array_equal(restored.identity_, scaler.identity_)
    assert np.allclose(restored.transform(table), scaled, equal_nan=True)

    print("sc_channels self-test: ALL PASS")
