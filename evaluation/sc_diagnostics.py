"""SC-conditioning evaluation-diagnostic primitives (pure numpy).

Small, dependency-light diagnostics for an SC-conditioned amortized NPE
pipeline (HCP whole-brain SBI). NONE of these import torch / cupy / GPU —
they operate on plain numpy arrays so they can be imported and unit-tested
anywhere, including a CUDA-less shell.

Public API
----------
- boundary_mass(samples_scaled, thresh, per_param)
    Fraction of scaled-space [-1,1] entries piled at the prior box edge.
    A high value flags clip-fallback / posterior collapse onto the boundary.

- median_fc_corr(fc_preds, fc_obs, nan_mask)
    Element-wise MEDIAN FC across posterior draws, scored ONCE vs observed
    FC on the upper triangle (k=1). Median analogue of the existing
    expected-FC=mean path in ``evaluation.metrics``. Masking mirrors
    ``evaluation.metrics.fc_metrics`` exactly.

- ood_distance(x_obs_fc, train_fc_mean, train_fc_std)
    How far one observed FC feature vector sits from the simulated-FC
    training cloud (per-feature standardized deviation summaries). Quantifies
    the documented out-of-distribution / low-acceptance problem.

- fit_train_fc_stats(fc_train)
    Per-feature mean/std of TRAINING simulated FC (std floor 1e-8). Caller
    must pass train rows only (TRAIN-only by contract).

- permutation_index_map(idmap, seed_offset)
    Deterministic derangement of a {sid: row} map for the SC-permutation
    control (condition each test subject on a WRONG subject's SC).
"""
import numpy as np

# Shared with fit_train_fc_stats / ood_distance to avoid divide-by-zero.
_STD_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# 1. Boundary-mass collapse detector
# ---------------------------------------------------------------------------

def boundary_mass(samples_scaled, thresh=0.98, per_param=False):
    """Fraction of scaled-space entries piled at the prior box edge.

    ``samples_scaled``: (n_samples, n_params) array in scaled space [-1, 1]
    (sbi ``BoxUniform`` convention). Returns the fraction of all entries with
    ``abs(value) > thresh``. A large fraction means probability mass is piled
    at the prior box edge -> clip-fallback collapse.

    If ``per_param`` is True, returns ``(overall_fraction, per_param_fraction)``
    where ``per_param_fraction`` is a length-``n_params`` array of the
    boundary fraction within each parameter column.
    """
    x = np.asarray(samples_scaled, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Empty / no-entry edge cases.
    if x.size == 0:
        if per_param:
            n_params = x.shape[1] if x.ndim == 2 else 0
            return 0.0, np.zeros(n_params, dtype=np.float64)
        return 0.0

    over = np.abs(x) > thresh
    overall = float(over.mean())

    if per_param:
        per = over.mean(axis=0).astype(np.float64)
        return overall, per
    return overall


# ---------------------------------------------------------------------------
# 2. Median-FC correlation (median analogue of expected-FC=mean)
# ---------------------------------------------------------------------------

def median_fc_corr(fc_preds, fc_obs, nan_mask=None):
    """Score the element-wise MEDIAN predicted FC against observed FC.

    ``fc_preds``: list (or stack) of (N, N) predicted FC matrices (posterior
    draws). Their element-wise median -> one (N, N) matrix -> scored once
    against ``fc_obs`` on the upper triangle (k=1) with Pearson corr + RMSE.

    Masking mirrors ``evaluation.metrics.fc_metrics`` exactly: upper triangle
    (k=1), keep entries where both sides are finite AND (if provided) not
    flagged by ``nan_mask`` (N, N bool). Empty ``fc_preds`` -> the same
    degenerate fallback used there: ``{"corr": 0.0, "rmse": 1.0}``.
    """
    if fc_preds is None or len(fc_preds) == 0:
        return {"corr": 0.0, "rmse": 1.0}

    stack = np.asarray(fc_preds, dtype=np.float64)
    # Element-wise median across draws -> (N, N). nanmedian so per-element
    # NaNs in some draws don't poison the whole entry.
    fc_med = np.nanmedian(stack, axis=0)

    fc_obs = np.asarray(fc_obs, dtype=np.float64)
    n = fc_obs.shape[0]
    iu = np.triu_indices(n, k=1)
    a = fc_obs[iu]
    b = fc_med[iu]
    mask = np.isfinite(a) & np.isfinite(b)
    if nan_mask is not None:
        mask &= ~np.asarray(nan_mask, dtype=bool)[iu]
    if mask.sum() < 2:
        return {"corr": 0.0, "rmse": 1.0}
    a, b = a[mask], b[mask]
    if a.std() > 0 and b.std() > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    else:
        r = 0.0
    return {"corr": r, "rmse": float(np.sqrt(((a - b) ** 2).mean()))}


# ---------------------------------------------------------------------------
# 3. Out-of-distribution distance vs simulated-FC training cloud
# ---------------------------------------------------------------------------

def ood_distance(x_obs_fc, train_fc_mean, train_fc_std):
    """Standardized-deviation summaries of ONE observation vs the train cloud.

    ``x_obs_fc``: (D,) or (1, D) FC feature vector of a single observation.
    ``train_fc_mean`` / ``train_fc_std``: (D,) per-feature mean/std computed
    from TRAINING simulated FC only. A std floor of 1e-8 is applied internally
    to avoid divide-by-zero on degenerate features.

    Returns floats:
      - ``z_rms``  : sqrt(mean(z**2)) of standardized deviations.
      - ``z_p95``  : 95th percentile of |z|.
      - ``frac_gt3``: fraction of features with |z| > 3.
    """
    x = np.asarray(x_obs_fc, dtype=np.float64).reshape(-1)
    mu = np.asarray(train_fc_mean, dtype=np.float64).reshape(-1)
    sd = np.asarray(train_fc_std, dtype=np.float64).reshape(-1)
    sd = np.maximum(sd, _STD_FLOOR)

    if x.size == 0:
        return {"z_rms": 0.0, "z_p95": 0.0, "frac_gt3": 0.0}

    z = (x - mu) / sd
    az = np.abs(z)
    return {
        "z_rms": float(np.sqrt(np.mean(z ** 2))),
        "z_p95": float(np.percentile(az, 95)),
        "frac_gt3": float(np.mean(az > 3)),
    }


# ---------------------------------------------------------------------------
# 4. Per-feature train-FC statistics (TRAIN-only by contract)
# ---------------------------------------------------------------------------

def fit_train_fc_stats(fc_train):
    """Per-feature mean/std of training simulated FC.

    ``fc_train``: (n_train_sims, D) training simulated FC upper-tri rows.
    Returns ``(mean (D,), std (D,))`` with the std floored at 1e-8. TRAIN-only
    by contract -- the caller is responsible for passing only train rows.
    """
    x = np.asarray(fc_train, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), _STD_FLOOR)
    return mean, std


# ---------------------------------------------------------------------------
# 5. Deterministic derangement of an index map (SC-permutation control)
# ---------------------------------------------------------------------------

def permutation_index_map(idmap, seed_offset=0):
    """Map each subject to a DIFFERENT subject's row (a derangement).

    ``idmap``: ``{sid: row}``. Returns a NEW dict ``{sid: permuted_row}`` in
    which no sid keeps its own row (when ``n > 1``). Used for the
    SC-permutation control: condition each test subject on a WRONG subject's
    SC; if the FC correlation does NOT drop, the encoder is ignoring SC.

    Deterministic: keys are sorted, then permuted with
    ``np.random.RandomState(12345 + seed_offset)`` (NO global random / date
    state). ``n == 1`` -> returns ``idmap`` unchanged (no derangement exists).
    """
    sids = sorted(idmap.keys())
    n = len(sids)
    if n <= 1:
        return dict(idmap)

    rng = np.random.RandomState(12345 + seed_offset)
    rows = [idmap[s] for s in sids]

    # Rejection-sample a permutation until it is a derangement (no fixed
    # point). Expected tries ~ e, so this terminates quickly; the RNG is
    # seeded so the outcome is fully deterministic.
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break

    return {sids[i]: rows[perm[i]] for i in range(n)}


# ---------------------------------------------------------------------------
# CPU self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # --- 1. boundary_mass: known fraction near +-1 -------------------------
    # 10 params x 100 samples; force exactly the first 20% of entries to edge.
    samp = rng.uniform(-0.5, 0.5, size=(100, 10))   # interior
    flat = samp.reshape(-1)
    n_edge = int(0.20 * flat.size)
    flat[:n_edge] = 1.0                              # pile at +1
    samp = flat.reshape(100, 10)
    frac = boundary_mass(samp, thresh=0.98)
    assert abs(frac - 0.20) < 1e-9, frac
    overall, per = boundary_mass(samp, thresh=0.98, per_param=True)
    assert abs(overall - 0.20) < 1e-9, overall
    assert per.shape == (10,)
    assert abs(float(per.mean()) - overall) < 1e-9
    # all-interior -> 0, all-edge -> 1
    assert boundary_mass(rng.uniform(-0.5, 0.5, (50, 4))) == 0.0
    assert boundary_mass(np.ones((5, 3))) == 1.0
    # edge cases
    assert boundary_mass(np.empty((0, 4))) == 0.0
    _o, _p = boundary_mass(np.empty((0, 4)), per_param=True)
    assert _o == 0.0 and _p.shape == (4,)

    # --- 2. median_fc_corr: perfect match -> ~1, noisy -> lower ------------
    N = 12
    A = rng.randn(N, N)
    fc_obs = (A + A.T) / 2.0
    np.fill_diagonal(fc_obs, 1.0)
    # perfect: every draw == observed -> median == observed -> corr ~ 1
    m_perfect = median_fc_corr([fc_obs.copy(), fc_obs.copy(), fc_obs.copy()],
                               fc_obs)
    assert m_perfect["corr"] > 0.999, m_perfect
    assert m_perfect["rmse"] < 1e-9, m_perfect
    # noisy draws -> lower corr than perfect
    noisy = [fc_obs + rng.randn(N, N) * 0.5 for _ in range(15)]
    m_noisy = median_fc_corr(noisy, fc_obs)
    assert m_noisy["corr"] < m_perfect["corr"], (m_noisy, m_perfect)
    # empty -> documented fallback
    assert median_fc_corr([], fc_obs) == {"corr": 0.0, "rmse": 1.0}
    # nan_mask honored (drop a block, still finite numbers -> no crash)
    nm = np.zeros((N, N), dtype=bool)
    nm[0, 1] = nm[1, 0] = True
    m_masked = median_fc_corr([fc_obs.copy()], fc_obs, nan_mask=nm)
    assert m_masked["corr"] > 0.999, m_masked

    # --- 3. ood_distance: in-distribution small, shifted large ------------
    D = 500
    mu = rng.randn(D)
    sd = np.abs(rng.randn(D)) + 0.5
    x_in = mu + sd * rng.randn(D)          # ~1 sigma away on average
    x_out = mu + sd * (rng.randn(D) + 8.0)  # shifted ~8 sigma
    d_in = ood_distance(x_in, mu, sd)
    d_out = ood_distance(x_out, mu, sd)
    assert d_in["z_rms"] < 2.0, d_in
    assert d_out["z_rms"] > d_in["z_rms"], (d_in, d_out)
    assert d_out["z_rms"] > 5.0, d_out
    assert d_in["frac_gt3"] < 0.05, d_in
    assert d_out["frac_gt3"] > 0.9, d_out
    assert d_out["z_p95"] > d_in["z_p95"], (d_in, d_out)
    # (1, D) input shape also accepted
    assert ood_distance(x_in.reshape(1, D), mu, sd)["z_rms"] == d_in["z_rms"]
    # std-floor: degenerate train std must not blow up to inf/nan
    d_floor = ood_distance(np.array([1.0, 2.0]), np.array([0.0, 0.0]),
                           np.array([0.0, 0.0]))
    assert np.isfinite(d_floor["z_rms"]), d_floor

    # --- 4. fit_train_fc_stats: shapes + std floor ------------------------
    fc_train = rng.randn(40, D)
    m_t, s_t = fit_train_fc_stats(fc_train)
    assert m_t.shape == (D,) and s_t.shape == (D,)
    assert np.all(s_t >= _STD_FLOOR)
    # constant column -> std floored, not zero
    const_train = np.ones((10, 3)) * 7.0
    _, s_c = fit_train_fc_stats(const_train)
    assert np.allclose(s_c, _STD_FLOOR), s_c
    # round-trip: stats from train + in-train obs -> moderate z_rms
    obs_row = fc_train[0]
    d_rt = ood_distance(obs_row, m_t, s_t)
    assert d_rt["z_rms"] < 3.0, d_rt

    # --- 5. permutation_index_map: derangement + deterministic ------------
    idmap = {sid: sid * 100 for sid in [5, 1, 9, 3, 7]}
    p1 = permutation_index_map(idmap)
    p2 = permutation_index_map(idmap)
    # every sid mapped to a DIFFERENT subject's row (no fixed point)
    for sid in idmap:
        assert p1[sid] != idmap[sid], sid
    # all rows used exactly once (it's a permutation)
    assert sorted(p1.values()) == sorted(idmap.values())
    # deterministic across two calls
    assert p1 == p2, (p1, p2)
    # seed_offset changes the outcome (very likely for n=5)
    p3 = permutation_index_map(idmap, seed_offset=1)
    assert p3 != p1
    # n == 1 -> unchanged
    one = {42: 999}
    assert permutation_index_map(one) == one
    # n == 0 -> unchanged (empty)
    assert permutation_index_map({}) == {}
    # n == 2 -> the unique derangement (swap)
    two = {2: "a", 8: "b"}
    p_two = permutation_index_map(two)
    assert p_two == {2: "b", 8: "a"}, p_two

    print("sc_diagnostics self-test: ALL PASS")
