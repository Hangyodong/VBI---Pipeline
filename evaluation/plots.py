"""Plotting helpers — posteriors, FC comparison, SBC, PCA, one-sim.

Public API
----------
- plot_posteriors(results, param_names, prior_low, prior_high, title, save)
- plot_fc_comparison(results, save_path, title)
- plot_posteriors_two_stage(...)  : alias for two-stage API
- plot_fc_comparison_two_stage(...)
- plot_sbc_rank_histogram(ranks, param_names, save_path)
- plot_pca_diagnostic(pca_diag, save_path)
- plot_one_simulation(sid, subject_data, theta_raw, param_names, ...)

All figures are written to ``config.OUTPUT_DIR`` by default.
Matplotlib backend is forced to ``Agg`` so this module is import-safe
on headless servers.
"""
import os

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Posterior plots
# ---------------------------------------------------------------------------

def plot_posteriors(results, param_names, prior_low, prior_high,
                    title="Stage 1", save_path=None):
    """Histogram of posterior samples per subject x parameter."""
    n_subj = len(results)
    n_p = len(param_names)
    fig, axes = plt.subplots(
        n_subj, n_p, figsize=(3 * n_p, 3 * n_subj), squeeze=False,
    )
    for r_idx, res in enumerate(results):
        samples = res["samples_raw"]
        for c, name in enumerate(param_names):
            ax = axes[r_idx, c]
            ax.hist(
                samples[:, c], bins=50,
                color="steelblue", alpha=0.6, density=True,
            )
            ax.set_xlim(prior_low[c], prior_high[c])
            ax.axvline(
                res["means_raw"][c], color="red",
                linestyle="--", lw=1,
            )
            ax.set_title(f"{res['sid']} | {name}")
            ax.set_xlabel(name)

    plt.suptitle(f"Posteriors - {title}", fontsize=12)
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR,
        f"posterior_{title.lower().replace(' ', '_')}.png",
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# FC comparison plots
# ---------------------------------------------------------------------------

def plot_fc_comparison(results, save_path=None, title="FC comparison"):
    """Observed FC vs mean predicted FC, side-by-side per subject."""
    n_subj = len(results)
    fig, axes = plt.subplots(
        n_subj, 2, figsize=(8, 4 * n_subj), squeeze=False,
    )
    for r_idx, res in enumerate(results):
        axes[r_idx, 0].imshow(res["fc_obs"], cmap="RdBu_r",
                              vmin=-1, vmax=1)
        axes[r_idx, 0].set_title(f"{res['sid']}\nObserved FC")
        if res["fc_preds"]:
            fc_mean_pred = np.mean(res["fc_preds"], axis=0)
            axes[r_idx, 1].imshow(
                fc_mean_pred, cmap="RdBu_r", vmin=-1, vmax=1,
            )
            axes[r_idx, 1].set_title(
                f"Predicted (mean)\n"
                f"corr={res['fc_corr_mean']:.3f}, "
                f"RMSE={res['fc_rmse_mean']:.3f}"
            )

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR, "fc_comparison.png",
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# Two-stage aliases
# ---------------------------------------------------------------------------

def plot_posteriors_two_stage(results, param_names,
                              prior_low, prior_high,
                              title="Stage 1+2", save_path=None):
    """Alias of plot_posteriors for the two-stage API."""
    return plot_posteriors(
        results, param_names, prior_low, prior_high,
        title=title, save_path=save_path,
    )


def plot_fc_comparison_two_stage(results, save_path=None,
                                 title="FC comparison (two-stage)"):
    """Alias of plot_fc_comparison for the two-stage API."""
    return plot_fc_comparison(results, save_path=save_path, title=title)


# ---------------------------------------------------------------------------
# SBC rank histogram
# ---------------------------------------------------------------------------

def plot_sbc_rank_histogram(ranks, param_names=None, save_path=None):
    """SBC rank histogram per parameter."""
    if ranks is None or len(ranks) == 0:
        print("  no SBC ranks")
        return
    param_names = param_names or config.STAGE1_PARAMS
    n_p = ranks.shape[1]
    n_bins = config.SBC_BINS

    fig, axes = plt.subplots(
        1, n_p, figsize=(3 * n_p, 3), squeeze=False,
    )
    n_post = ranks.max() + 1
    expected = len(ranks) / n_bins

    for i, name in enumerate(param_names):
        ax = axes[0, i]
        ax.hist(
            ranks[:, i], bins=n_bins, range=(0, n_post),
            edgecolor="black", color="lightblue",
        )
        ax.axhline(
            expected, color="red", linestyle="--",
            label=f"uniform ({expected:.1f})",
        )
        ax.set_title(name)
        ax.set_xlabel("rank")
        ax.legend(fontsize=7)

    plt.suptitle("SBC rank histograms", fontsize=12)
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR, "sbc_ranks.png",
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# PCA diagnostic plot
# ---------------------------------------------------------------------------

def plot_pca_diagnostic(pca_diag, save_path=None):
    """Bar plot of the top-5 PCs' explained variance ratio."""
    fc_diag = pca_diag.get("fc_pca", pca_diag)
    if "explained_variance_top5" not in fc_diag:
        return
    evr_top = fc_diag["explained_variance_top5"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(evr_top)), evr_top, color="steelblue")
    ax.set_xlabel("PC index")
    ax.set_ylabel("EVR")
    ax.set_title(
        f"PCA top-5 EVR  "
        f"(cum EVR = {fc_diag['explained_variance_sum']:.4f})"
    )
    plt.tight_layout()
    save_path = save_path or os.path.join(
        config.OUTPUT_DIR, "pca_evr.png",
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# ---------------------------------------------------------------------------
# One-simulation diagnostic
# ---------------------------------------------------------------------------

def plot_one_simulation(sid, subject_data, theta_raw=None, param_names=None,
                        sim_idx=0, save_name="step2_one_sim.png", bold=None):
    """Plot BOLD + FC for one simulated sample.

    If ``bold`` is supplied (e.g. cached during feature extraction),
    no re-simulation is performed. Otherwise this falls back to the
    legacy behavior: re-run the WC + HRF for ``theta_raw[sim_idx]``
    on subject ``sid`` and plot the resulting BOLD time series and
    simulated FC. Useful as a sanity check after Step 2.
    """
    from simulator import compute_fc

    d = subject_data[sid]

    if bold is None:
        from simulator import simulate_single
        if theta_raw is None or param_names is None:
            raise ValueError(
                "plot_one_simulation: when bold is None, theta_raw and "
                "param_names are required for re-simulation."
            )
        params = {n: float(theta_raw[sim_idx, i])
                  for i, n in enumerate(param_names)}
        print(f"  [one-sim plot] sid={sid}  sim_idx={sim_idx}")
        print(f"    params = {params}")
        bolds = simulate_single(
            d["sc"], params, n_repeat=1,
            delays=d["delays"], apply_bw=True,
        )
        bold = bolds[0]
    else:
        bold = np.asarray(bold)
        print(f"  [one-sim plot] sid={sid}  (cached BOLD, no re-sim)")

    fc_sim = compute_fc(bold)
    fc_obs = d["fc"]

    iu = np.triu_indices(fc_sim.shape[0], k=1)
    fc_corr = float(np.corrcoef(fc_sim[iu], fc_obs[iu])[0, 1])
    fc_rmse = float(np.sqrt(((fc_sim[iu] - fc_obs[iu]) ** 2).mean()))

    print(f"    BOLD shape : {bold.shape}")
    print(f"    BOLD range : [{bold.min():.3f}, {bold.max():.3f}]  "
          f"std={bold.std():.3f}")
    print(f"    Sim FC vs Obs FC: Pearson r = {fc_corr:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    t = np.arange(bold.shape[0]) * config.TR_SEC
    for i in range(min(5, bold.shape[1])):
        axes[0, 0].plot(t, bold[:, i], lw=0.8, alpha=0.8,
                        label=f"region {i}")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("BOLD")
    axes[0, 0].set_title(f"{sid}  sim {sim_idx} — BOLD (5 regions)")
    axes[0, 0].legend(fontsize=7, ncol=5)

    im = axes[0, 1].imshow(bold.T, aspect="auto", cmap="RdBu_r",
                           vmin=-np.abs(bold).max(),
                           vmax=np.abs(bold).max())
    axes[0, 1].set_xlabel("TR")
    axes[0, 1].set_ylabel("Region")
    axes[0, 1].set_title(
        f"BOLD all regions ({bold.shape[1]} x {bold.shape[0]})"
    )
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(fc_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_xlabel("Region")
    axes[1, 0].set_ylabel("Region")
    axes[1, 0].set_title(
        f"Simulated FC  (sim {sim_idx})\n"
        f"corr={fc_corr:.3f}  rmse={fc_rmse:.3f}"
    )
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label="r")

    im3 = axes[1, 1].imshow(fc_obs, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_xlabel("Region")
    axes[1, 1].set_ylabel("Region")
    axes[1, 1].set_title(
        f"Empirical FC ({sid})\nSim vs Obs corr = {fc_corr:.3f}"
    )
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, label="r")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, save_name)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved: {save_path}")
    return {"bold": bold, "fc_sim": fc_sim, "fc_corr": fc_corr,
            "params": params}
