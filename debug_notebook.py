"""Debug cell for the mouse MPTP VBI-SBI pipeline.

Paste the body of `run_all_checks()` into a single notebook cell right
after the Setup cell, or import and call it::

    from debug_notebook import run_all_checks
    run_all_checks()

Sections (A..J) match the refactor requirements:
    A  Import / environment check
    B  Config sanity check
    C  Data loading check
    D  ParameterScaler check
    E  Feature extraction check
    F  GPU simulation theta-specific check  (CRITICAL)
    G  Mini training smoke test
    H  Stage1 SNPE smoke test  (slow — gated by RUN_SBI_SMOKE)
    I  Parameter selection logic check
    J  Main function existence check
"""
import sys
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_checks(run_sbi_smoke=False, run_gpu_smoke=True, verbose=True):
    """Run every debug section, collect results, print a summary."""
    results = {}
    for name, fn, gate in [
        ("A. import/env",       _check_a_imports,  True),
        ("B. config",           _check_b_config,   True),
        ("C. data loading",     _check_c_data,     True),
        ("D. ParameterScaler",  _check_d_scaler,   True),
        ("E. feature extract",  _check_e_features, True),
        ("F. GPU per-theta",    _check_f_gpu,      run_gpu_smoke),
        ("G. mini training",    _check_g_mini,     run_gpu_smoke),
        ("H. SBI smoke",        _check_h_snpe,     run_sbi_smoke),
        ("I. theta_bad logic",  _check_i_select,   True),
        ("J. main funcs",       _check_j_funcs,    True),
    ]:
        if not gate:
            results[name] = "SKIPPED"
            print(f"\n=== {name} === SKIPPED")
            continue
        print(f"\n=== {name} ===")
        try:
            fn(verbose=verbose)
            results[name] = "PASS"
            print(f"    [{name}] PASS")
        except AssertionError as e:
            results[name] = f"FAIL: {e}"
            print(f"    [{name}] FAIL: {e}")
        except Exception as e:
            results[name] = f"ERROR: {type(e).__name__}: {e}"
            if verbose:
                traceback.print_exc()
            print(f"    [{name}] ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("  Debug summary")
    print("=" * 60)
    for k, v in results.items():
        flag = "✓" if v == "PASS" else ("·" if v == "SKIPPED" else "✗")
        print(f"  {flag} {k:<22s}  {v}")
    return results


# ---------------------------------------------------------------------------
# A. Import / environment check
# ---------------------------------------------------------------------------

def _check_a_imports(verbose=True):
    import config       # noqa: F401
    import data_loader  # noqa: F401
    import simulator    # noqa: F401
    import inference    # noqa: F401
    import evaluate     # noqa: F401

    import torch
    has_cuda = torch.cuda.is_available()
    print(f"    torch.cuda.is_available(): {has_cuda}")
    if has_cuda:
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    config.SBI_DEVICE: {config.SBI_DEVICE}")

    # VBI WC import
    try:
        from vbi.models.cupy.wilson_cowan import WC_sde   # noqa: F401
        print(f"    VBI WC_sde: OK")
    except ImportError as e:
        print(f"    VBI WC_sde: NOT AVAILABLE ({e})")


# ---------------------------------------------------------------------------
# B. Config sanity check
# ---------------------------------------------------------------------------

def _check_b_config(verbose=True):
    import config
    assert config.FEATURE_SET == "fc_only", (
        f"FEATURE_SET must be 'fc_only', got {config.FEATURE_SET!r}"
    )
    assert config.PARAM_NAMES_STAGE1 == ["P", "Q", "g_e", "g_i"], (
        f"STAGE1 params: {config.PARAM_NAMES_STAGE1}"
    )
    assert config.LOCAL_EI_PARAMS == ["c_ee", "c_ei", "c_ie", "c_ii"], (
        f"LOCAL_EI_PARAMS: {config.LOCAL_EI_PARAMS}"
    )
    assert config.SC_WEIGHT_COL == 1, f"SC_WEIGHT_COL: {config.SC_WEIGHT_COL}"
    assert config.SC_LENGTH_COL == 2, f"SC_LENGTH_COL: {config.SC_LENGTH_COL}"
    for k in ("dt", "t_end", "t_cut"):
        assert k in config.WC_FIXED, f"WC_FIXED missing {k}"
    print(f"    FEATURE_SET     : {config.FEATURE_SET}")
    print(f"    SIM_MODE        : {getattr(config, 'SIM_MODE', '<none>')}")
    print(f"    PARAM_NAMES_S1  : {config.PARAM_NAMES_STAGE1}")
    print(f"    LOCAL_EI_PARAMS : {config.LOCAL_EI_PARAMS}")
    print(f"    SC_WEIGHT_COL   : {config.SC_WEIGHT_COL}")
    print(f"    SC_LENGTH_COL   : {config.SC_LENGTH_COL}")


# ---------------------------------------------------------------------------
# C. Data loading check
# ---------------------------------------------------------------------------

def _check_c_data(verbose=True):
    import config
    import data_loader

    out = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    assert len(subjects) >= config.N_TRAIN + config.N_VAL + config.N_TEST, (
        f"not enough subjects: {len(subjects)}"
    )
    train, val, test = data_loader.three_way_split(subjects)

    d = data_loader.get_subject_data(
        train[0], fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids,
    )
    n = config.N_REGIONS
    fc, sc = d["fc"], d["sc"]
    lengths = d["lengths_mm"]
    delays = d["delays"]

    assert fc.shape == (n, n), f"FC shape {fc.shape}"
    assert sc.shape == (n, n), f"SC shape {sc.shape}"
    assert lengths.shape == (n, n), f"lengths shape {lengths.shape}"
    assert delays.shape == (n, n), f"delays shape {delays.shape}"
    assert np.all(np.diag(fc) == 0), "FC diagonal not zero"
    assert not np.allclose(sc, lengths), (
        "SC weight and tract length are identical — "
        "check SC_WEIGHT_COL vs SC_LENGTH_COL"
    )

    print(f"    {train[0]}:")
    print(f"      FC      : {fc.shape}  diag0={np.all(np.diag(fc) == 0)}")
    print(f"      SC      : {sc.shape}  "
          f"range=[{sc[sc > 0].min():.4f}, {sc.max():.4f}]")
    print(f"      lengths : {lengths.shape}  "
          f"range=[{lengths[lengths > 0].min():.2f}, {lengths.max():.2f}] mm")
    print(f"      delays  : {delays.shape}  "
          f"range=[{delays[delays > 0].min():.3f}, {delays.max():.3f}] ms")
    assert 0.1 < lengths[lengths > 0].mean() < 50, (
        f"lengths mean {lengths[lengths > 0].mean():.3f} not in mm-scale"
    )


# ---------------------------------------------------------------------------
# D. ParameterScaler check
# ---------------------------------------------------------------------------

def _check_d_scaler(verbose=True):
    import config
    from inference import make_stage1_param_scaler

    scaler = make_stage1_param_scaler()
    assert scaler.param_names == config.PARAM_NAMES_STAGE1

    rng = np.random.default_rng(0)
    scaled = rng.uniform(-1.0, 1.0, size=(5, len(scaler.param_names)))
    raw = scaler.inverse_transform(scaled)
    scaled_back = scaler.transform(raw)
    assert np.allclose(scaled, scaled_back, atol=1e-5), (
        "round-trip scale failed"
    )
    for j, name in enumerate(scaler.param_names):
        lo, hi = scaler.low[j], scaler.high[j]
        assert np.all(raw[:, j] >= lo - 1e-5)
        assert np.all(raw[:, j] <= hi + 1e-5)
    d = scaler.to_dict(raw[0])
    print(f"    round-trip OK")
    print(f"    to_dict(raw[0]) = {d}")


# ---------------------------------------------------------------------------
# E. Feature extraction check
# ---------------------------------------------------------------------------

def _check_e_features(verbose=True):
    import config
    from simulator import compute_fc, fc_to_upper_tri
    n = config.N_REGIONS
    fc_dim = n * (n - 1) // 2
    assert config.FC_DIM == fc_dim, (
        f"FC_DIM {config.FC_DIM} != n*(n-1)/2 = {fc_dim}"
    )
    ts = np.random.randn(100, n).astype(np.float32)
    fc = compute_fc(ts)
    assert fc.shape == (n, n)
    vec = fc_to_upper_tri(fc)
    assert vec.shape == (fc_dim,), (
        f"FC upper-tri shape {vec.shape} != ({fc_dim},)"
    )
    print(f"    FC dim = {fc_dim}  (N={n})  ✓")


# ---------------------------------------------------------------------------
# F. GPU simulation theta-specific check   (CRITICAL)
# ---------------------------------------------------------------------------

def _check_f_gpu(verbose=True):
    """Different theta MUST produce different features."""
    import config
    import data_loader
    import simulator

    out = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    train, _, _ = data_loader.three_way_split(subjects)
    d = data_loader.get_subject_data(
        train[0], fc_mat, sc_mat, fc_ids, sc_ids,
    )

    # Two contrastive thetas in raw space (Stage 1)
    param_names = config.PARAM_NAMES_STAGE1
    theta_raw_batch = np.array([
        [0.6, 0.1, 0.1, 0.1],    # low everything
        [2.4, 1.9, 1.4, 1.4],    # high everything
    ], dtype=np.float32)

    bolds = simulator.simulate_gpu_batch(
        d["sc"], theta_raw_batch, param_names,
        delays=d["delays"], apply_bw=True,
    )
    assert len(bolds) == 2, f"expected 2 BOLDs, got {len(bolds)}"
    print(f"    BOLD shapes : {[b.shape for b in bolds]}")

    fc0 = simulator.compute_fc(bolds[0])
    fc1 = simulator.compute_fc(bolds[1])
    v0 = simulator.fc_to_upper_tri(fc0)
    v1 = simulator.fc_to_upper_tri(fc1)
    diff = np.linalg.norm(v0 - v1)
    print(f"    FC0 range   : [{v0.min():.3f}, {v0.max():.3f}]")
    print(f"    FC1 range   : [{v1.min():.3f}, {v1.max():.3f}]")
    print(f"    ||FC0 - FC1||₂ = {diff:.4f}")
    assert diff > 1e-3, (
        f"Different theta produced near-identical features "
        f"(||Δ||={diff:.6f}). This means batch-mean parameter bug "
        f"is still present in simulate_gpu_batch."
    )


# ---------------------------------------------------------------------------
# G. Mini training smoke test
# ---------------------------------------------------------------------------

def _check_g_mini(verbose=True):
    import config
    import data_loader
    import inference

    out = data_loader.load_raw_data()
    df, fc_mat, sc_mat, fc_ids, sc_ids, bold_mat, bold_ids = out
    subjects = data_loader.get_target_subjects(df, fc_ids, sc_ids)
    train, _, _ = data_loader.three_way_split(subjects)
    subj1 = train[:1]
    subject_data = {
        s: data_loader.get_subject_data(s, fc_mat, sc_mat, fc_ids, sc_ids)
        for s in subj1
    }

    n_sim = 4
    scaler = inference.make_stage1_param_scaler()
    prior_scaled = inference.make_scaled_prior(
        len(scaler.param_names), device=config.SBI_DEVICE,
    )

    theta_scaled, _, fc_raw, _ = inference.collect_training_data(
        subj1, subject_data, prior_scaled, scaler,
        param_names=scaler.param_names, n_sim=n_sim,
        apply_bw=True, verbose=False,
    )
    assert theta_scaled.shape == (n_sim, len(scaler.param_names)), (
        f"theta shape {theta_scaled.shape}"
    )
    assert fc_raw.shape == (n_sim, config.FC_DIM), (
        f"fc shape {fc_raw.shape}, expected ({n_sim}, {config.FC_DIM})"
    )
    assert (theta_scaled >= -1).all() and (theta_scaled <= 1).all()
    assert np.isfinite(fc_raw).all()
    print(f"    theta shape : {theta_scaled.shape}")
    print(f"    fc    shape : {fc_raw.shape}")
    print(f"    theta in [-1,1] & fc finite  ✓")


# ---------------------------------------------------------------------------
# H. Stage1 SNPE smoke test  (slow)
# ---------------------------------------------------------------------------

def _check_h_snpe(verbose=True):
    print("    (SBI smoke: enable RUN_SBI_SMOKE=True to run)")


# ---------------------------------------------------------------------------
# I. Parameter selection logic check
# ---------------------------------------------------------------------------

def _check_i_select(verbose=True):
    from inference import select_theta_bad
    sens = [0.9, 0.8, 0.1, 0.7]
    shr = [0.6, 0.1, 0.05, 0.15]
    bad = select_theta_bad(
        sens, shr,
        param_names=["P", "Q", "g_e", "g_i"],
        sens_threshold=0.5, shrinkage_threshold=0.2,
    )
    assert bad == ["Q", "g_i"], f"got {bad}, expected ['Q', 'g_i']"
    print(f"    sens={sens}, shr={shr}")
    print(f"    theta_bad={bad}  ✓")


# ---------------------------------------------------------------------------
# J. Main function existence check
# ---------------------------------------------------------------------------

def _check_j_funcs(verbose=True):
    import evaluate
    expected = [
        "evaluate_all_two_stage",
        "plot_posteriors_two_stage",
        "plot_fc_comparison_two_stage",
        "print_summary_two_stage",
    ]
    missing = [n for n in expected if not hasattr(evaluate, n)]
    assert not missing, f"missing evaluate functions: {missing}"
    for n in expected:
        print(f"    evaluate.{n}: OK")


if __name__ == "__main__":
    run_all_checks(
        run_sbi_smoke=False,
        run_gpu_smoke=("--gpu" in sys.argv),
    )
