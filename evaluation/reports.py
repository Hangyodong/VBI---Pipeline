"""Notebook-style report functions and final summary.

Public API
----------
- print_summary_two_stage(results_s1, results_s2, theta_bad)
- evaluate_all_two_stage(*args, **kwargs)         (stub)
- print_final_summary(stage1_agg, stage2_agg, best_stage, test_summary,
                      train_subjects, n_train_sim)
- report_step1 ... report_step14

These functions are used by main.ipynb cells. FCD rows are suppressed
when ``config.USE_FCD`` is False.
"""
import os

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Two-stage compact summary
# ---------------------------------------------------------------------------

def print_summary_two_stage(results_s1, results_s2=None,
                            theta_bad=None):
    """Print a compact summary of Stage 1 / Stage 1+2 validation results."""
    print("\n" + "=" * 70)
    print("  Two-stage summary")
    print("=" * 70)
    if results_s1:
        fc_corr_s1 = np.mean([r.get("fc_corr_mean", 0.0)
                              for r in results_s1])
        fc_rmse_s1 = np.mean([r.get("fc_rmse_mean", 1.0)
                              for r in results_s1])
        print(
            f"  Stage 1 : FC corr = {fc_corr_s1:.4f}, "
            f"FC RMSE = {fc_rmse_s1:.4f}  "
            f"(n={len(results_s1)})"
        )
    if theta_bad is not None:
        print(f"  theta_bad selected : {theta_bad}")
    if results_s2:
        fc_corr_s2 = np.mean([r.get("fc_corr_mean", 0.0)
                              for r in results_s2])
        fc_rmse_s2 = np.mean([r.get("fc_rmse_mean", 1.0)
                              for r in results_s2])
        print(
            f"  Stage 2 : FC corr = {fc_corr_s2:.4f}, "
            f"FC RMSE = {fc_rmse_s2:.4f}  "
            f"(n={len(results_s2)})"
        )
        if results_s1:
            d_corr = fc_corr_s2 - fc_corr_s1
            d_rmse = fc_rmse_s1 - fc_rmse_s2
            print(
                f"  Δ corr  = {d_corr:+.4f}   "
                f"Δ rmse  = {d_rmse:+.4f}   "
                f"(positive = Stage 2 better)"
            )


def evaluate_all_two_stage(*args, **kwargs):
    """Stub kept for legacy notebook calls."""
    raise NotImplementedError(
        "evaluate_all_two_stage: pipeline-specific implementation "
        "should override this. See main.py for the two-stage flow."
    )


# ---------------------------------------------------------------------------
# Final summary (after Step 14)
# ---------------------------------------------------------------------------

def print_final_summary(stage1_agg, stage2_agg, best_stage,
                        test_summary, train_subjects, n_train_sim):
    """Compact end-of-run summary.

    FCD rows are hidden when ``config.USE_FCD`` is False.
    """
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n" + "=" * 95)
    print("  Pipeline complete - final summary")
    print("=" * 95)
    print(f"  Train subjects  : {len(train_subjects)}")
    print(f"  Stage 1 sims    : {n_train_sim}")
    print(f"  Selected stage  : {best_stage}")
    print()
    s2_corr = stage2_agg["fc_corr_mean"] if stage2_agg else 0.0
    s2_rmse = stage2_agg["fc_rmse_mean"] if stage2_agg else 0.0
    s2_fcd = stage2_agg["fcd_rmse_mean"] if stage2_agg else 0.0
    print("  +----------------+----------+----------+----------+")
    print("  | Metric         |  Val S1  |  Val S2  |  Test    |")
    print("  +----------------+----------+----------+----------+")
    print(
        f"  | FC corr        |  {stage1_agg['fc_corr_mean']:>+.4f} "
        f"|  {s2_corr:>+.4f} "
        f"|  {test_summary['fc_corr_boot_ci'][0]:>+.4f} |"
    )
    print(
        f"  | FC RMSE        |  {stage1_agg['fc_rmse_mean']:.4f} "
        f"|  {s2_rmse:.4f} "
        f"|  {test_summary['fc_rmse_boot_ci'][0]:.4f} |"
    )
    if use_fcd:
        print(
            f"  | FCD vec RMSE   |  {stage1_agg['fcd_rmse_mean']:.4f} "
            f"|  {s2_fcd:.4f} "
            f"|  {test_summary['fcd_rmse_boot_ci'][0]:.4f} |"
        )
    print("  +----------------+----------+----------+----------+")
    print("\n  Test bootstrap 95% CI:")
    _, lo, hi = test_summary["fc_corr_boot_ci"]
    print(f"    FC corr   : [{lo:.4f}, {hi:.4f}]")
    _, lo, hi = test_summary["fc_rmse_boot_ci"]
    print(f"    FC RMSE   : [{lo:.4f}, {hi:.4f}]")
    if use_fcd:
        _, lo, hi = test_summary["fcd_rmse_boot_ci"]
        print(f"    FCD RMSE  : [{lo:.4f}, {hi:.4f}]")


# ---------------------------------------------------------------------------
# Per-step reports (notebook helpers)
# ---------------------------------------------------------------------------

def report_step1(train, val, test, subject_data):
    """Step 1 summary table + SC sparsity bar plot."""
    print("\n" + "=" * 60)
    print("  Step 1 result")
    print("=" * 60)
    print(f"  train ({len(train)}): {train}")
    print(f"  val   ({len(val)}): {val}")
    print(f"  test  ({len(test)}): {test}")

    sids = train + val + test
    sc_edges = [int((subject_data[s]["sc"] > 0).sum()) for s in sids]
    fc_nan = [int(subject_data[s]["fc_nan"].sum()) for s in sids]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].bar(range(len(sids)), sc_edges, color="steelblue")
    axes[0].set_xticks(range(len(sids)))
    axes[0].set_xticklabels(sids, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("SC nonzero edges")
    axes[0].set_title("SC sparsity per subject")

    axes[1].bar(range(len(sids)), fc_nan, color="indianred")
    axes[1].set_xticks(range(len(sids)))
    axes[1].set_xticklabels(sids, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("FC NaN count")
    axes[1].set_title("FC NaN per subject")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step1_subjects.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step2(theta_scaled, fc_raw, fcd_raw):
    """Step 2 summary: shapes + theta distribution histogram."""
    print("\n" + "=" * 60)
    print("  Step 2 result")
    print("=" * 60)
    print(f"  theta_scaled : {theta_scaled.shape}")
    print(f"  fc_raw       : {fc_raw.shape}  "
          f"(finite={np.all(np.isfinite(fc_raw))})")
    print(f"  fcd_raw      : {fcd_raw.shape}  "
          f"(finite={np.all(np.isfinite(fcd_raw))})")

    if theta_scaled.ndim < 2 or theta_scaled.shape[0] == 0:
        print()
        print("  !! Step 2 collected 0 samples — simulation failed.")
        print("  !! Common causes:")
        print("     - GPU_BATCH too large (OOM)")
        print("       Setup 셀에서 GPU_BATCH를 줄여서 다시 실행하세요.")
        print("     - VBI WC engine error (check earlier batch logs)")
        return

    n_params = theta_scaled.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(3 * n_params, 3),
                             squeeze=False)
    for i in range(n_params):
        axes[0, i].hist(theta_scaled[:, i], bins=40,
                        color="steelblue", alpha=0.7)
        axes[0, i].set_title(f"{config.STAGE1_PARAMS[i]} (scaled)")
        axes[0, i].set_xlim(-1, 1)
    plt.suptitle("Step 2 - sampled theta distribution", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step2_theta_hist.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step3(fc_raw, fcd_raw):
    """Step 3 summary: FC and FCD value distributions."""
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n" + "=" * 60)
    print("  Step 3 result")
    print("=" * 60)
    print(f"  FC  raw : shape={fc_raw.shape}, "
          f"min={float(fc_raw.min()):.3f}, "
          f"max={float(fc_raw.max()):.3f}, "
          f"mean={float(fc_raw.mean()):.3f}")
    if use_fcd:
        print(f"  FCD raw : shape={fcd_raw.shape}, "
              f"min={float(fcd_raw.min()):.3f}, "
              f"max={float(fcd_raw.max()):.3f}, "
              f"mean={float(fcd_raw.mean()):.3f}")

    n_cols = 2 if use_fcd else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 3.5),
                             squeeze=False)
    axes[0, 0].hist(fc_raw.flatten(), bins=80,
                    color="steelblue", alpha=0.7)
    axes[0, 0].set_title("FC raw distribution")
    axes[0, 0].set_xlabel("value")
    if use_fcd:
        axes[0, 1].hist(fcd_raw.flatten(), bins=80,
                        color="seagreen", alpha=0.7)
        axes[0, 1].set_title("FCD raw distribution")
        axes[0, 1].set_xlabel("value")
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step3_feature_dist.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step4(scalers, fc_raw, fcd_raw):
    """Step 4 summary: FCD z-score (FC has none — already in [-1, 1])."""
    print("\n" + "=" * 60)
    print("  Step 4 result")
    print("=" * 60)
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print(f"  FC     : raw Pearson r [-1, 1], no z-score")
    if use_fcd:
        fcd_z = scalers["fcd_z"].transform(fcd_raw)
        print(f"  FCD z  : mean={float(fcd_z.mean()):.4f}, "
              f"std={float(fcd_z.std()):.4f}")
    else:
        print(f"  FCD    : disabled (USE_FCD=False)")
        fcd_z = None

    n_rows = 2 if use_fcd else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.5 * n_rows),
                             squeeze=False)
    axes[0, 0].hist(fc_raw.flatten(), bins=80,
                    color="steelblue", alpha=0.7)
    axes[0, 0].set_title("FC raw (Pearson r)")
    axes[0, 0].axvline(0, color="gray", ls=":", lw=0.5)
    axes[0, 1].axis("off")
    axes[0, 1].text(0.5, 0.5,
                    "FC z-score: disabled\n(already in [-1, 1])",
                    ha="center", va="center", fontsize=11,
                    color="gray")
    if use_fcd:
        axes[1, 0].hist(fcd_raw.flatten(), bins=80,
                        color="seagreen", alpha=0.7)
        axes[1, 0].set_title("FCD raw")
        axes[1, 1].hist(fcd_z.flatten(), bins=80,
                        color="seagreen", alpha=0.7)
        axes[1, 1].set_title("FCD z-scored")
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step4_zscore.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step5(pipeline, x_input):
    """Step 5 summary: FC PCA dimensions + FCD summary stats."""
    print("\n" + "=" * 60)
    print("  Step 5 result")
    print("=" * 60)
    print(f"  FC  PCA  : {pipeline.fc_dim} -> "
          f"{pipeline.fc_pca.n_components}")
    print(f"  FCD      : {pipeline.fcd_dim} dims "
          f"(summary stats, z-scored, no PCA)")
    print(f"  x_input  : {x_input.shape}")

    evr_fc = pipeline.fc_pca.explained_variance_ratio_

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    ax.plot(np.cumsum(evr_fc), color="steelblue", lw=2)
    ax.axhline(0.9, color="red", ls="--", lw=1, label="90%")
    ax.axhline(0.95, color="orange", ls="--", lw=1, label="95%")
    ax.set_xlabel("PC index")
    ax.set_ylabel("cumulative EVR")
    ax.set_title(f"FC PCA (sum = {evr_fc.sum():.4f})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "step5_pca_evr.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")


def report_step6(pca_diagnostic):
    """Step 6 summary: FC PCA diagnostic + FCD summary stats info."""
    print("\n" + "=" * 60)
    print("  Step 6 result")
    print("=" * 60)
    d = pca_diagnostic.get("fc_pca", {})
    if d:
        evr_ok = "PASS" if d.get("pca_pass_evr") else "FAIL"
        rec_ok = "PASS" if d.get("pca_pass_recon") else "FAIL"
        print(f"  FC:")
        print(f"    n_components     : {d['n_components']}")
        print(f"    cum EVR          : "
              f"{d['explained_variance_sum']:.4f} [{evr_ok}]")
        print(f"    recon corr (train) : "
              f"{d['recon_corr_train_mean']:.4f} [{rec_ok}]")
    d2 = pca_diagnostic.get("fcd_pca", {})
    if d2:
        print(f"  FCD:")
        print(f"    type   : {d2.get('type', 'summary_stats')}")
        print(f"    dims   : {d2.get('dims', [])}")
        if "train_mean" in d2:
            print(f"    mean   : {d2['train_mean']:.4f}")
            print(f"    std    : {d2['train_std']:.4f}")


def report_step7(param_scaler):
    """Step 7 summary: prior bounds and scaled range."""
    print("\n" + "=" * 60)
    print("  Step 7 result")
    print("=" * 60)
    print("  Prior bounds:")
    for name, lo, hi in zip(param_scaler.param_names,
                            param_scaler.low, param_scaler.high):
        print(f"    {name:6s} : [{float(lo):7.3f}, {float(hi):7.3f}]"
              f"  -> [-1, 1]")


def report_step8(posterior, embedding_net, theta_scaled, x_input):
    """Step 8 summary: trained estimator info + sample posterior."""
    print("\n" + "=" * 60)
    print("  Step 8 result")
    print("=" * 60)
    print(f"  theta_scaled : {theta_scaled.shape}")
    print(f"  x_input      : {x_input.shape}")
    has_params = any(p.requires_grad
                     for p in embedding_net.parameters())
    if has_params:
        n_p = sum(p.numel() for p in embedding_net.parameters())
        print(f"  embedding net params: {n_p:,}")
    print(f"  posterior     : {type(posterior).__name__}")


def report_step9(stage1_agg, baseline_agg):
    """Step 9 summary: Stage 1 vs baseline metrics."""
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n" + "=" * 60)
    print("  Step 9 result")
    print("=" * 60)
    print("                       Stage 1     Baseline    delta")
    print("  -----------------------------------------------------")
    d_fc = stage1_agg["fc_corr_mean"] - baseline_agg["fc_corr_mean"]
    d_rmse = baseline_agg["fc_rmse_mean"] - stage1_agg["fc_rmse_mean"]
    print(f"  FC  corr           {stage1_agg['fc_corr_mean']:>+8.4f}  "
          f"{baseline_agg['fc_corr_mean']:>+8.4f}  {d_fc:>+8.4f}")
    print(f"  FC  RMSE           {stage1_agg['fc_rmse_mean']:>8.4f}  "
          f"{baseline_agg['fc_rmse_mean']:>8.4f}  {d_rmse:>+8.4f}")
    if use_fcd:
        d_fcd = (
            baseline_agg["fcd_rmse_mean"] - stage1_agg["fcd_rmse_mean"]
        )
        print(
            f"  FCD RMSE           {stage1_agg['fcd_rmse_mean']:>8.4f}  "
            f"{baseline_agg['fcd_rmse_mean']:>8.4f}  {d_fcd:>+8.4f}"
        )
    print()
    print("  Shrinkage per param:")
    for name, s in zip(stage1_agg["param_names"],
                       stage1_agg["shrinkage_mean"]):
        mark = "OK" if s >= config.DIFFICULT_SHRINKAGE else "LOW"
        print(f"    {name:6s} : {float(s):.4f}  [{mark}]")


def report_step10(difficult, val_shrinkage):
    """Step 10 summary: which params go into Stage 2."""
    print("\n" + "=" * 60)
    print("  Step 10 result")
    print("=" * 60)
    print(f"  Shrinkage threshold : {config.DIFFICULT_SHRINKAGE}")
    print(f"  Difficult params    : {difficult}")
    print(f"  c-params to add     : "
          f"{list(config.C_PARAM_PRIOR.keys())}")
    print(f"  Stage 2 targets     : "
          f"{difficult + list(config.C_PARAM_PRIOR.keys())}")


def report_step11(s2):
    """Step 11 summary: Stage 2 parameter targets."""
    print("\n" + "=" * 60)
    print("  Step 11 result")
    print("=" * 60)
    print(f"  Stage 2 params   : {s2['stage2_params']}")
    print(f"  Nuisance params  : {s2['nuisance_params']}")
    print(f"  theta_scaled     : {s2['theta_scaled'].shape}")
    print(f"  x_input          : {s2['x_input'].shape}")


def report_step12(stage2_agg):
    """Step 12 summary: Stage 2 metrics."""
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n" + "=" * 60)
    print("  Step 12 result")
    print("=" * 60)
    print(f"  FC  corr   : {stage2_agg['fc_corr_mean']:>+.4f}")
    print(f"  FC  RMSE   : {stage2_agg['fc_rmse_mean']:.4f}")
    if use_fcd:
        print(f"  FCD RMSE   : {stage2_agg['fcd_rmse_mean']:.4f}")
    print()
    print("  Shrinkage per param:")
    for name, s in zip(stage2_agg["param_names"],
                       stage2_agg["shrinkage_mean"]):
        print(f"    {name:6s} : {float(s):.4f}")


def report_step13(best_stage, score_1, score_2,
                  stage1_agg, stage2_agg):
    """Step 13 summary: selection scores + winner."""
    print("\n" + "=" * 60)
    print("  Step 13 result")
    print("=" * 60)
    print(f"  Stage 1 score : {score_1:>+.4f}")
    if stage2_agg is not None:
        print(f"  Stage 2 score : {score_2:>+.4f}")
    print(f"\n  Selected: Stage {best_stage}")


def report_step14(test_summary):
    """Step 14 summary: test metrics with bootstrap CI."""
    use_fcd = bool(getattr(config, "USE_FCD", True))
    print("\n" + "=" * 60)
    print("  Step 14 result")
    print("=" * 60)
    rows = [("FC  corr ", "fc_corr_boot_ci"),
            ("FC  RMSE ", "fc_rmse_boot_ci")]
    if use_fcd:
        rows.append(("FCD RMSE ", "fcd_rmse_boot_ci"))
    for label, key in rows:
        m, lo, hi = test_summary[key]
        print(f"  {label}: {m:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}]")
