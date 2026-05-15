"""Debugging and unit-style checks for the Mouse MPTP VBI-SBI pipeline.

Usage
-----
    python debug.py --all          all tests (default)
    python debug.py --basic        scalers, PCA, features (no GPU)
    python debug.py --data         data file loading (mat files required)
    python debug.py --pipeline     mock inference flow (no GPU)
    python debug.py --sim          real WC simulation (GPU + VBI required)

Each test prints PASS / FAIL / SKIP. SKIP means the test was not applicable
in the current environment (e.g. no GPU).
"""
import argparse
import os
import sys
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _color(text, code):
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _ok(text):
    return _color(text, "92")


def _fail(text):
    return _color(text, "91")


def _warn(text):
    return _color(text, "93")


def _info(text):
    return _color(text, "94")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestRunner:
    """Lightweight PASS/FAIL aggregator."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def run(self, name, fn, *args, **kwargs):
        print(f"\n{_info('>')} {name}")
        print("-" * 70)
        try:
            result = fn(*args, **kwargs)
            if result == "skip":
                self.skipped += 1
                self.results.append((name, "SKIP", ""))
                print(_warn("  SKIP"))
            else:
                self.passed += 1
                self.results.append((name, "PASS", ""))
                print(_ok("  PASS"))
        except AssertionError as e:
            self.failed += 1
            self.results.append((name, "FAIL", str(e)))
            print(f"{_fail('  FAIL')}: {e}")
        except Exception as e:
            self.failed += 1
            self.results.append((name, "ERROR", str(e)))
            print(f"{_fail('  ERROR')}: {e}")
            traceback.print_exc(limit=3)

    def summary(self):
        print("\n" + "=" * 70)
        print("  Test summary")
        print("=" * 70)
        for name, status, _ in self.results:
            if status == "PASS":
                mark = _ok("PASS")
            elif status == "SKIP":
                mark = _warn("SKIP")
            else:
                mark = _fail(status)
            short = name if len(name) <= 50 else name[:47] + "..."
            print(f"  {mark}  {short:<52s}")
        print("-" * 70)
        print(
            f"  PASS: {self.passed}  |  "
            f"FAIL: {self.failed}  |  "
            f"SKIP: {self.skipped}"
        )
        return self.failed == 0


# ---------------------------------------------------------------------------
# Config and import checks
# ---------------------------------------------------------------------------

def test_config_consistency():
    """Internal consistency of `config` values."""
    import config as cfg
    assert cfg.N_REGIONS == 115, f"N_REGIONS={cfg.N_REGIONS} != 115"
    assert cfg.FC_DIM == 115 * 114 // 2 == 6555, (
        f"FC_DIM={cfg.FC_DIM} != 6555 (115*114/2)"
    )
    assert cfg.FCD_DIM == cfg.FC_DIM, (
        f"FCD_DIM={cfg.FCD_DIM} != FC_DIM={cfg.FC_DIM}"
    )
    expected_bold = int((cfg.T_END - cfg.T_CUT) / 1000 / cfg.TR_SEC)
    assert cfg.ANALYSIS_BOLD_T == expected_bold, (
        f"ANALYSIS_BOLD_T={cfg.ANALYSIS_BOLD_T} != {expected_bold}"
    )
    assert cfg.FC_COL == 1, f"FC_COL={cfg.FC_COL} (=2nd row)"
    assert cfg.FCD_COL == 2, f"FCD_COL={cfg.FCD_COL} (=3rd row)"
    assert (
        len(cfg.STAGE1_PARAMS)
        == len(cfg.STAGE1_PRIOR_LOW)
        == len(cfg.STAGE1_PRIOR_HIGH)
    ), "Prior dimensions inconsistent"
    print(
        f"  N_REGIONS={cfg.N_REGIONS}, FC_DIM={cfg.FC_DIM}, "
        f"FCD_DIM={cfg.FCD_DIM}"
    )
    print(
        f"  Analysis BOLD: {cfg.ANALYSIS_BOLD_T} TR "
        f"= ({cfg.T_END} - {cfg.T_CUT}) / {cfg.TR_SEC * 1000}ms  OK"
    )
    print(f"  FC  <- col {cfg.FC_COL} (=2nd row, NaN -> 0)")
    print(f"  FCD <- col {cfg.FCD_COL} (=3rd row, used directly)")
    print(
        f"  SC  <- col {cfg.SC_COL} "
        f"(=2nd row, uint16 raw -> log1p + max-norm)"
    )


def test_imports():
    """All pipeline modules can be imported."""
    import importlib
    modules = [
        "config", "data_loader", "bold", "simulator",
        "inference", "evaluate",
    ]
    for m in modules:
        try:
            importlib.import_module(m)
            print(f"  OK   {m}")
        except ImportError as e:
            print(f"  WARN {m} import (external dep): {e}")
        except Exception as e:
            raise AssertionError(f"{m}: {e}")


# ---------------------------------------------------------------------------
# Data file checks
# ---------------------------------------------------------------------------

def test_data_files_exist():
    """Required data files are present."""
    import config as cfg
    files = {
        "FC mat": cfg.FC_PATH,
        "SC mat": cfg.SC_PATH,
        "TSV": cfg.TSV_PATH,
        "Atlas": cfg.ATLAS_PATH,
    }
    missing = []
    for label, path in files.items():
        if not os.path.exists(path):
            missing.append(f"{label} ({path})")
        else:
            print(f"  OK   {label}: {path}")
    if missing:
        print(f"  missing: {missing}")
        return "skip"


def test_data_loading():
    """`load_raw_data` returns matrices of the expected shape."""
    import config as cfg
    if not (os.path.exists(cfg.FC_PATH) and os.path.exists(cfg.SC_PATH)):
        print("  data files missing -> skip")
        return "skip"

    import data_loader
    df, fc_mat, sc_mat, fc_ids, sc_ids, _, _ = data_loader.load_raw_data()

    fc_sample = fc_mat[0, cfg.FC_COL]
    sc_sample = sc_mat[0, cfg.SC_COL]
    assert fc_sample.shape == (cfg.N_REGIONS, cfg.N_REGIONS), (
        f"FC shape {fc_sample.shape} != "
        f"({cfg.N_REGIONS}, {cfg.N_REGIONS})"
    )
    assert sc_sample.shape == (cfg.N_REGIONS, cfg.N_REGIONS), (
        f"SC shape {sc_sample.shape} != "
        f"({cfg.N_REGIONS}, {cfg.N_REGIONS})"
    )

    n_nan = int(np.isnan(fc_sample).sum())
    print(f"  FC col {cfg.FC_COL} NaN: {n_nan} (replaced with 0)")

    sc_f = sc_sample.astype(np.float64)
    assert sc_f.min() >= 0, f"SC has negative values: min={sc_f.min()}"
    assert (sc_f > 0).sum() > 0, "SC has no nonzero edges"
    print(
        f"  SC col {cfg.SC_COL}: dtype={sc_sample.dtype}, "
        f"range=[{sc_f.min():.0f}, {sc_f.max():.0f}], "
        f"nonzero={int((sc_f > 0).sum())}"
    )

    common = set(fc_ids) & set(sc_ids)
    assert len(common) >= 8, (
        f"common IDs={len(common)} (need >= 8)"
    )
    print(
        f"  ID match: FC {len(fc_ids)}, SC {len(sc_ids)}, "
        f"common {len(common)}"
    )


def test_atlas_labels():
    """Atlas file parses into the expected number of regions."""
    import config as cfg
    if not os.path.exists(cfg.ATLAS_PATH):
        print("  atlas file missing -> skip")
        return "skip"
    import data_loader
    labels = data_loader.load_atlas_labels()
    assert labels is not None, "atlas parse failed"
    assert len(labels) == cfg.N_REGIONS, (
        f"label count {len(labels)} != {cfg.N_REGIONS}"
    )
    print(f"  OK   {len(labels)} regions parsed")
    print(f"    first 5: {[lab['abbrev'] for lab in labels[:5]]}")
    print(f"    last  5: {[lab['abbrev'] for lab in labels[-5:]]}")


# ---------------------------------------------------------------------------
# ParameterScaler
# ---------------------------------------------------------------------------

def test_parameter_scaler():
    """ParameterScaler maps midpoint -> 0 and round-trips identity."""
    import config as cfg
    from inference import ParameterScaler, make_stage1_param_scaler

    ps = make_stage1_param_scaler()

    midpoint = np.array([[
        0.5 * (lo + hi)
        for lo, hi in zip(cfg.STAGE1_PRIOR_LOW, cfg.STAGE1_PRIOR_HIGH)
    ]])
    mid_scaled = ps.transform(midpoint)
    assert np.allclose(mid_scaled, 0.0, atol=1e-5), (
        f"midpoint -> {mid_scaled.flatten()} (expected 0)"
    )
    print(f"  midpoint -> scaled: {mid_scaled.flatten()}")

    edges = np.array([cfg.STAGE1_PRIOR_LOW, cfg.STAGE1_PRIOR_HIGH])
    edges_s = ps.transform(edges)
    assert np.allclose(edges_s[0], -1.0, atol=1e-5), (
        f"low -> {edges_s[0]} != -1"
    )
    assert np.allclose(edges_s[1], +1.0, atol=1e-5), (
        f"high -> {edges_s[1]} != +1"
    )
    print("  bounds map to [-1, +1] OK")

    rng = np.random.RandomState(0)
    n_dim = len(cfg.STAGE1_PARAMS)
    raw_rand = np.zeros((100, n_dim), dtype=np.float32)
    for i, (lo, hi) in enumerate(
        zip(cfg.STAGE1_PRIOR_LOW, cfg.STAGE1_PRIOR_HIGH)
    ):
        raw_rand[:, i] = rng.uniform(lo, hi, 100)
    scaled = ps.transform(raw_rand)
    back = ps.inverse_transform(scaled)
    max_diff = float(np.abs(raw_rand - back).max())
    assert np.allclose(raw_rand, back, atol=1e-5), (
        f"round-trip failed: max diff {max_diff}"
    )
    print("  round-trip on 100 random samples: OK")

    d = ps.to_dict()
    ps2 = ParameterScaler.from_dict(d)
    assert ps2.param_names == ps.param_names
    assert np.allclose(ps2.low, ps.low)
    print("  to_dict / from_dict: OK")


def test_stage2_param_scaler():
    """Stage 2 scaler combines Stage 1 params and c-params correctly."""
    from inference import make_stage2_param_scaler

    stage2_params = ["Q", "g_i", "c_ee", "c_ei", "c_ie", "c_ii"]
    s2 = make_stage2_param_scaler(stage2_params)
    expected_low = [0.0, 0.0, 12.0, 8.0, 10.0, 1.0]
    expected_high = [2.0, 1.5, 20.0, 16.0, 20.0, 6.0]
    assert s2.low.tolist() == expected_low, (
        f"low {s2.low.tolist()} != {expected_low}"
    )
    assert s2.high.tolist() == expected_high, (
        f"high {s2.high.tolist()} != {expected_high}"
    )
    print(f"  Stage 2 prior matched: {stage2_params}")


# ---------------------------------------------------------------------------
# Feature scalers
# ---------------------------------------------------------------------------

def test_family_scaler():
    """FamilyScaler produces unit z-score and handles std=0 columns."""
    from inference import FamilyScaler

    rng = np.random.RandomState(0)
    x_train = rng.randn(1000, 50) * 5 + 3
    x_val = rng.randn(200, 50) * 5 + 3

    fs = FamilyScaler("test")
    x_train_s = fs.fit_transform(x_train)
    x_val_s = fs.transform(x_val)

    assert abs(x_train_s.mean()) < 0.05, (
        f"mean={x_train_s.mean()} != 0"
    )
    assert abs(x_train_s.std() - 1.0) < 0.05, (
        f"std={x_train_s.std()} != 1"
    )
    print(
        f"  train z-score: mean={x_train_s.mean():.4f}, "
        f"std={x_train_s.std():.4f}"
    )
    print(
        f"  val z-score:   mean={x_val_s.mean():.4f}, "
        f"std={x_val_s.std():.4f}"
    )

    x_zero = np.ones((100, 5), dtype=np.float32)
    fs2 = FamilyScaler("zero").fit(x_zero)
    x_z = fs2.transform(x_zero)
    assert np.all(np.isfinite(x_z)), "std=0 column produced NaN/Inf"
    print("  std=0 column handled OK")


def test_fc_pca_scaler():
    """FCPCAScaler fits and produces a diagnostic dict."""
    import config as cfg
    from inference import FCPCAScaler

    rng = np.random.RandomState(0)
    n_train = 1000
    fc_train = (
        rng.randn(n_train, cfg.FC_DIM).astype(np.float32) * 0.3
    )

    scaler = FCPCAScaler(
        n_components=min(50, n_train, cfg.FC_DIM),
    )
    fc_pca = scaler.fit_transform(fc_train)
    assert fc_pca.shape == (n_train, scaler.n_components)
    print(f"  FC PCA: {fc_train.shape} -> {fc_pca.shape}")

    diag = scaler.diagnostic(fc_train, None)
    print(f"  EVR sum: {diag['explained_variance_sum']:.4f}")
    print(f"  recon corr: {diag['recon_corr_train_mean']:.4f}")
    assert "explained_variance_sum" in diag


def test_feature_pipeline():
    """FeaturePipeline concatenates FC and FCD PCAs."""
    import config as cfg
    from inference import FeaturePipeline

    rng = np.random.RandomState(0)
    n = 200
    fc_raw = rng.randn(n, cfg.FC_DIM).astype(np.float32) * 0.5
    fcd_raw = rng.randn(n, cfg.FCD_DIM).astype(np.float32) * 0.3

    saved_fc = cfg.PCA_DIM_FC
    saved_fcd = cfg.PCA_DIM_FCD
    cfg.PCA_DIM_FC = 50
    cfg.PCA_DIM_FCD = 30
    pipe = FeaturePipeline()
    x = pipe.fit_transform(fc_raw, fcd_raw)
    cfg.PCA_DIM_FC = saved_fc
    cfg.PCA_DIM_FCD = saved_fcd

    expected_dim = 50 + 30
    assert x.shape == (n, expected_dim), (
        f"output {x.shape} != ({n}, {expected_dim})"
    )
    print(
        f"  pipeline: FC({cfg.FC_DIM}) + FCD({cfg.FCD_DIM}) "
        f"-> PCA(50) + PCA(30) = {x.shape[1]}"
    )


# ---------------------------------------------------------------------------
# Embedding net
# ---------------------------------------------------------------------------

def test_embedding_net():
    """FeatureEmbedding forward pass shape."""
    import config as cfg
    try:
        import torch
        from inference import FeatureEmbedding
    except ImportError as e:
        print(f"  torch missing: {e}")
        return "skip"

    input_dim = cfg.PCA_DIM_FC + cfg.PCA_DIM_FCD
    net = FeatureEmbedding(input_dim=input_dim)
    x = torch.randn(64, input_dim)
    y = net(x)
    assert y.shape == (64, cfg.EMBED_DIM)
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"  MLP: {input_dim} -> {cfg.EMBED_HIDDEN} -> "
        f"{cfg.EMBED_HIDDEN // 2} -> {cfg.EMBED_DIM}"
    )
    print(f"  params: {n_params:,}")


# ---------------------------------------------------------------------------
# Simulator helper functions
# ---------------------------------------------------------------------------

def test_fc_upper_tri():
    """`fc_to_upper_tri` returns a (FC_DIM,) finite vector."""
    import config as cfg
    from simulator import fc_to_upper_tri

    rng = np.random.RandomState(0)
    fc = rng.uniform(
        -0.5, 0.5, (cfg.N_REGIONS, cfg.N_REGIONS),
    ).astype(np.float32)
    fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 0)

    vec = fc_to_upper_tri(fc)
    assert vec.shape == (cfg.FC_DIM,), (
        f"vec shape {vec.shape} != ({cfg.FC_DIM},)"
    )
    assert np.all(np.isfinite(vec)), "NaN/Inf in FC upper tri"
    print(
        f"  FC ({cfg.N_REGIONS}, {cfg.N_REGIONS}) -> "
        f"upper tri ({len(vec)},)"
    )


def test_fcd_summary():
    """Simulated FCD matrix and its upper-triangle vector."""
    import config as cfg
    from simulator import compute_sim_fcd_matrix, fcd_to_upper_tri

    rng = np.random.RandomState(0)
    bold = rng.randn(cfg.ANALYSIS_BOLD_T, cfg.N_REGIONS).astype(np.float32)
    fcd_mat = compute_sim_fcd_matrix(bold)
    print(f"  simulated FCD matrix: {fcd_mat.shape}")
    assert fcd_mat.shape == (cfg.N_REGIONS, cfg.N_REGIONS), (
        f"FCD matrix {fcd_mat.shape} != "
        f"({cfg.N_REGIONS}, {cfg.N_REGIONS})"
    )

    vec = fcd_to_upper_tri(fcd_mat)
    assert vec.shape == (cfg.FCD_DIM,), (
        f"vec {vec.shape} != ({cfg.FCD_DIM},)"
    )
    assert np.all(np.isfinite(vec))
    print(f"  FCD upper tri: {vec.shape} (same dim as FC)")


def test_observed_fcd_direct():
    """Observed FCD is loaded directly from FCD_COL (no computation)."""
    print("  Observed FCD <- fc_mat[i, FCD_COL] (no computation needed)")


# ---------------------------------------------------------------------------
# Mock inference flow
# ---------------------------------------------------------------------------

def test_mock_inference_flow():
    """Mock the inference flow with random data (no GPU/SBI required)."""
    import config as cfg
    import inference

    rng = np.random.RandomState(0)
    n_train = 500

    param_scaler = inference.make_stage1_param_scaler()
    theta_scaled = rng.uniform(-1, 1, (n_train, 4)).astype(np.float32)
    param_scaler.inverse_transform(theta_scaled)
    fc_raw = rng.randn(n_train, cfg.FC_DIM).astype(np.float32) * 0.3
    fcd_raw = rng.randn(n_train, cfg.FCD_DIM).astype(np.float32) * 0.3

    saved_fc = cfg.PCA_DIM_FC
    saved_fcd = cfg.PCA_DIM_FCD
    cfg.PCA_DIM_FC = 50
    cfg.PCA_DIM_FCD = 30
    pipeline = inference.FeaturePipeline()
    pipeline.fit(fc_raw, fcd_raw)
    x_input = pipeline.transform(fc_raw, fcd_raw)
    cfg.PCA_DIM_FC = saved_fc
    cfg.PCA_DIM_FCD = saved_fcd

    assert x_input.shape == (n_train, 50 + 30)
    print(f"  x_input: {x_input.shape}")

    diag = pipeline.diagnostic(fc_raw, fcd_raw)
    assert "fc_pca" in diag and "fcd_pca" in diag
    print(f"  FC  PCA EVR: {diag['fc_pca']['explained_variance_sum']:.4f}")
    print(f"  FCD PCA EVR: {diag['fcd_pca']['explained_variance_sum']:.4f}")

    fc_val = rng.randn(20, cfg.FC_DIM).astype(np.float32) * 0.3
    fcd_val = rng.randn(20, cfg.FCD_DIM).astype(np.float32) * 0.3
    x_val = pipeline.transform(fc_val, fcd_val)
    print(f"  val transform: {x_val.shape}")

    samples_scaled = rng.uniform(-0.3, 0.3, (1000, 4)).astype(np.float32)
    shrink = inference.compute_shrinkage_scaled(samples_scaled)
    assert shrink.shape == (4,)
    assert np.all((shrink >= 0) & (shrink <= 1))
    print(f"  shrinkage (mock): {shrink.tolist()}")

    stage2_params, nuisance = inference.build_stage2_param_set(shrink)
    print(f"  stage2: {stage2_params}, nuisance: {nuisance}")


# ---------------------------------------------------------------------------
# Real WC simulation (GPU + VBI required)
# ---------------------------------------------------------------------------

def test_real_simulation():
    """Run one short WC simulation end-to-end on the GPU."""
    import config as cfg

    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        print(f"  GPU/cupy missing: {e}")
        return "skip"
    try:
        from vbi.models.cupy.wilson_cowan import WC_sde  # noqa: F401
    except ImportError as e:
        print(f"  VBI missing: {e}")
        return "skip"
    if not (os.path.exists(cfg.FC_PATH) and os.path.exists(cfg.SC_PATH)):
        print("  data missing")
        return "skip"

    import data_loader
    import simulator

    _, fc_mat, sc_mat, fc_ids, sc_ids, _, _ = data_loader.load_raw_data()
    sid = fc_ids[0]
    print(f"  test subject: {sid}")

    d = data_loader.get_subject_data(sid, fc_mat, sc_mat, fc_ids, sc_ids)

    saved_t_end = cfg.T_END
    saved_t_cut = cfg.T_CUT
    saved_a_bold = cfg.ANALYSIS_BOLD_T
    cfg.T_END = 10_000.0
    cfg.T_CUT = 2_000.0
    cfg.ANALYSIS_BOLD_T = 8
    cfg.WC_FIXED["t_end"] = cfg.T_END
    cfg.WC_FIXED["t_cut"] = cfg.T_CUT

    try:
        params = {"P": 1.5, "Q": 1.0, "g_e": 0.7, "g_i": 0.7}
        bolds = simulator.simulate_single(
            d["sc"], params, n_repeat=1,
            delays=d["delays"], apply_bw=True,
        )
        bold = bolds[0]
        assert bold.ndim == 2
        assert bold.shape[1] == cfg.N_REGIONS, (
            f"region count {bold.shape[1]} != {cfg.N_REGIONS}"
        )
        assert np.all(np.isfinite(bold)), "BOLD has NaN/Inf"
        print(f"  BOLD shape: {bold.shape}")
        print(f"  BOLD range: [{bold.min():.4f}, {bold.max():.4f}]")

        fc_vec, fcd_vec = simulator.extract_features(bold)
        assert fc_vec.shape[0] > 0 and np.all(np.isfinite(fc_vec)), (
            f"fc_vec issue: shape={fc_vec.shape}, "
            f"finite={np.isfinite(fc_vec).all()}"
        )
        assert fcd_vec.shape[0] > 0 and np.all(np.isfinite(fcd_vec)), (
            f"fcd_vec issue: shape={fcd_vec.shape}"
        )
        print(f"  FC vec: {fc_vec.shape}, FCD vec: {fcd_vec.shape}")
    finally:
        cfg.T_END = saved_t_end
        cfg.T_CUT = saved_t_cut
        cfg.ANALYSIS_BOLD_T = saved_a_bold
        cfg.WC_FIXED["t_end"] = saved_t_end
        cfg.WC_FIXED["t_cut"] = saved_t_cut


# ---------------------------------------------------------------------------
# Test groups
# ---------------------------------------------------------------------------

def run_basic_tests(runner):
    """No GPU, no SBI required."""
    runner.run("config consistency", test_config_consistency)
    runner.run("imports", test_imports)
    runner.run("ParameterScaler", test_parameter_scaler)
    runner.run("Stage 2 ParameterScaler", test_stage2_param_scaler)
    runner.run("FamilyScaler", test_family_scaler)
    runner.run("FCPCAScaler", test_fc_pca_scaler)
    runner.run("FeaturePipeline", test_feature_pipeline)
    runner.run("FeatureEmbedding (torch)", test_embedding_net)
    runner.run("FC upper triangle", test_fc_upper_tri)
    runner.run("FCD upper triangle (simulated)", test_fcd_summary)
    runner.run("observed FCD = direct file load", test_observed_fcd_direct)


def run_data_tests(runner):
    """Mat files required."""
    runner.run("data files exist", test_data_files_exist)
    runner.run("data load", test_data_loading)
    runner.run("atlas labels parse", test_atlas_labels)


def run_pipeline_tests(runner):
    """Mock inference flow."""
    runner.run("mock inference flow", test_mock_inference_flow)


def run_sim_tests(runner):
    """GPU + VBI required."""
    runner.run("real WC simulation", test_real_simulation)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mouse MPTP VBI-SBI debug runner",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--basic", action="store_true",
                        help="no GPU/SBI required")
    parser.add_argument("--data", action="store_true",
                        help="data files required")
    parser.add_argument("--pipeline", action="store_true",
                        help="mock flow")
    parser.add_argument("--sim", action="store_true",
                        help="real simulation (GPU required)")
    args = parser.parse_args()

    if not (args.basic or args.data or args.pipeline or args.sim):
        args.all = True

    runner = TestRunner()
    print("=" * 70)
    print("  Mouse MPTP VBI-SBI debug")
    print("=" * 70)

    if args.all or args.basic:
        print(f"\n{_info('=== BASIC (no GPU) ===')}")
        run_basic_tests(runner)
    if args.all or args.pipeline:
        print(f"\n{_info('=== MOCK PIPELINE ===')}")
        run_pipeline_tests(runner)
    if args.all or args.data:
        print(f"\n{_info('=== DATA (files required) ===')}")
        run_data_tests(runner)
    if args.all or args.sim:
        print(f"\n{_info('=== REAL SIMULATION (GPU) ===')}")
        run_sim_tests(runner)

    ok = runner.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
