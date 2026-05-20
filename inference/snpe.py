"""SNPE-C training and the feature/scaler preparation steps.

Public API
----------
- train_snpe(theta_scaled, x_input, prior_scaled, ...) -> (posterior, embedding_net)

Step functions (notebook-friendly)
----------------------------------
- step4_fit_feature_scalers(fc_raw, fcd_raw)   -> {fc_z, fcd_z}
- step5_fit_feature_pipeline(fc_raw, fcd_raw)  -> (pipeline, x_input)
- step6_pca_diagnostic(pipeline, ...)          -> pca_diag dict
- step7_fit_param_scaler()                     -> (param_scaler, prior_scaled)
- step8_train_snpe(theta_scaled, x_input, ...)  -> (posterior, embedding_net)

Internal
--------
- _print_pca_diagnostic(pca_diag, header)
"""
import threading
import time

import numpy as np

import config
from inference._utils import _progress
from inference.embedding import FeatureEmbedding
from inference.feature_pipeline import FamilyScaler, FeaturePipeline
from inference.priors import make_scaled_prior
from inference.scaling import make_stage1_param_scaler

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# SNPE-C trainer
# ---------------------------------------------------------------------------

def train_snpe(theta_scaled, x_input, prior_scaled, embedding_net=None,
               proposal=None, verbose=True, fc_raw=None):
    """Train SNPE-C jointly with the embedding network.

    Works with sbi 0.22+ where ``posterior_nn`` moved from
    ``sbi.utils`` to ``sbi.neural_nets``.

    ``fc_raw`` (optional): raw upper-tri FC vectors aligned with
    ``theta_scaled``/``x_input``. When provided, a single proxy FC
    self-correlation is printed after training as a sanity hint
    (epoch-by-epoch FC metrics are not exposed by sbi).
    """
    from sbi.inference import SNPE_C

    # posterior_nn moved between sbi versions — try the three known
    # locations and surface a clear error if all three fail.
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        try:
            from sbi.utils import posterior_nn
        except ImportError:
            try:
                from sbi.utils.get_nn_models import posterior_nn
            except ImportError as _exc:
                raise ImportError(
                    "Could not locate `posterior_nn` in the installed sbi. "
                    "Tried these import paths in order: "
                    "(1) `from sbi.neural_nets import posterior_nn` "
                    "(sbi >= 0.22), "
                    "(2) `from sbi.utils import posterior_nn` "
                    "(sbi 0.18 - 0.21), "
                    "(3) `from sbi.utils.get_nn_models import posterior_nn` "
                    "(sbi < 0.18). Last error: " + repr(_exc)
                ) from _exc

    theta_t = torch.tensor(theta_scaled, dtype=torch.float32)
    x_t = torch.tensor(x_input, dtype=torch.float32)

    if embedding_net is None:
        if config.USE_EMBEDDING:
            embedding_net = FeatureEmbedding(input_dim=x_input.shape[1])
            _n_params = sum(
                p.numel() for p in embedding_net.parameters()
                if p.requires_grad
            )
            _embed_desc = (
                f"input={x_input.shape[1]}"
                f" -> [{config.EMBED_HIDDEN}->ReLU->Drop,"
                f" {config.EMBED_DIM}]"
            )
        else:
            import torch.nn as nn
            embedding_net = nn.Identity()
            _n_params = 0
            _embed_desc = (
                f"Identity — no embedding"
                f" (MAF input={x_input.shape[1]})"
            )
    else:
        _n_params = sum(
            p.numel() for p in embedding_net.parameters()
            if p.requires_grad
        )
        _embed_desc = (
            f"input={x_input.shape[1]}"
            f" -> [{config.EMBED_HIDDEN}->ReLU->Drop,"
            f" {config.EMBED_DIM}]"
        )

    density_estimator = posterior_nn(
        model=config.NDE_MODEL,
        embedding_net=embedding_net,
        hidden_features=config.NDE_HIDDEN,
        num_transforms=config.NDE_TRANSFORMS,
    )
    inferer = SNPE_C(
        prior=prior_scaled,
        density_estimator=density_estimator,
        device=config.SBI_DEVICE,
    )

    if config.USE_MIXED_PRECISION and config.SBI_DEVICE == "cuda":
        try:
            torch.set_float32_matmul_precision("high")   # H100 Tensor Core
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    inferer.append_simulations(theta_t, x_t, proposal=proposal)

    # ---- E7 / E8 — training-config + architecture banner ------------------
    _train_batch = 512        # 2048 → 512 (정규화 효과 / 과적합 완화)
    _max_epochs  = 200        # 300 → 200
    _stop_after  = _max_epochs  # early stop 비활성화
    if verbose:
        _progress("[Step 8] SNPE-C training")
        print(
            f"           data   : theta={tuple(theta_t.shape)}  "
            f"x={tuple(x_t.shape)}"
        )
        print(f"           embed  : {_embed_desc}")
        print(f"                    trainable params: {_n_params:,}")
        print(
            f"           MAF    : model={config.NDE_MODEL}  "
            f"hidden={config.NDE_HIDDEN}  "
            f"transforms={config.NDE_TRANSFORMS}"
        )
        print(
            f"           train  : batch={_train_batch}  "
            f"stop_after={_stop_after}  max_epochs={_max_epochs}"
        )
        print(
            f"                    device={config.SBI_DEVICE}  "
            f"(PCA frozen | MLP+MAF jointly trained)"
        )

    # Heartbeat: inferer.train() can run silently for many minutes.
    # Daemon thread prints a periodic alive-ping with the current epoch
    # read from inferer._summary (sbi >= 0.22) so the user sees progress.
    _hb_stop = threading.Event()

    def _heartbeat():
        interval = 15.0  # seconds
        t_hb = time.time()
        while not _hb_stop.wait(interval):
            summary = getattr(inferer, "_summary", None) or {}
            tr = list(summary.get("training_loss", []))
            va = list(summary.get("validation_loss", []))
            ep = max(len(tr), len(va))
            elapsed = time.time() - t_hb
            if ep > 0:
                tr_last = tr[-1] if tr else float("nan")
                va_last = va[-1] if va else float("nan")
                print(
                    f"           ... training: epoch {ep}/{_max_epochs}  "
                    f"train={tr_last:.4f}  val={va_last:.4f}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
            else:
                print(
                    f"           ... training: warming up  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

    _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    _hb_thread.start()

    t0 = time.time()
    try:
        estimator = inferer.train(
            training_batch_size=_train_batch,
            stop_after_epochs=_stop_after,
            max_num_epochs=_max_epochs,
            show_train_summary=False,
        )
    finally:
        _hb_stop.set()
        _hb_thread.join(timeout=1.0)
    train_elapsed = time.time() - t0

    # ---- E9 / E10 — post-hoc per-epoch summary (Approach A) ---------------
    if verbose:
        _print_epoch_table(
            inferer, max_epochs_cap=_max_epochs, stop_after=_stop_after,
        )
        _print_train_summary(
            inferer, train_elapsed=train_elapsed,
            max_epochs_cap=_max_epochs, stop_after=_stop_after,
        )
        try:
            if fc_raw is not None:
                _n_val = max(1, int(len(x_input) * 0.1))
                _fc_val = fc_raw[-_n_val:]
                _fc_mean_pred = _fc_val.mean(axis=0)
                _corrs = [
                    float(np.corrcoef(_fc_val[i], _fc_mean_pred)[0, 1])
                    for i in range(min(50, _n_val))
                ]
                _mean_corr = float(np.mean(_corrs))
                _rmse = float(np.sqrt(
                    ((_fc_val[:min(50, _n_val)] - _fc_mean_pred) ** 2).mean()
                ))
                print(
                    f"  [Step 8] val FC proxy (n={min(50, _n_val)}): "
                    f"fc_corr={_mean_corr:.4f}  fc_rmse={_rmse:.4f}"
                )
            else:
                print(
                    "  [Step 8] epoch별 FC 지표: 학습 완료 후 "
                    "validation 단계에서 확인"
                )
        except Exception as _e:
            print(f"  [Step 8] FC proxy 계산 불가: {_e}")

    posterior = inferer.build_posterior(estimator)
    # ---- E11 — posterior built -------------------------------------------
    if verbose:
        _progress(
            f"[Step 8] posterior built  device={config.SBI_DEVICE}"
        )
    return posterior, embedding_net


# ---------------------------------------------------------------------------
# Step 4-8 cells
# ---------------------------------------------------------------------------

def step4_fit_feature_scalers(fc_raw, fcd_raw, verbose=True):
    """Step 4. Fit FC and FCD z-score scalers (train data only)."""
    if verbose:
        # E1 — Step 4 header: input shapes + USE_FCD scaler decision.
        fcd_choice = "FamilyScaler" if config.USE_FCD else "None"
        _progress(
            f"[Step 4] Feature scalers  "
            f"fc_raw={fc_raw.shape}  fcd_raw={fcd_raw.shape}"
        )
        print(
            f"           USE_FCD={config.USE_FCD} -> "
            f"fc_z=None  fcd_z={fcd_choice}"
        )
    fc_z = None  # FC: no z-score (already Pearson r in [-1, 1])
    if config.USE_FCD:
        fcd_z = FamilyScaler(name="FCD").fit(fcd_raw)
    else:
        fcd_z = None
    if verbose:
        print(f"    FC  z: disabled (FC is Pearson r in [-1, 1])")
        if fcd_z is not None:
            print(f"    FCD z: mean ~ {float(fcd_z.mean_.mean()):.4f}, "
                  f"std ~ {float(fcd_z.std_.mean()):.4f}")
        else:
            print(f"    FCD z: disabled (USE_FCD=False)")
        _progress("[Step 4] done")
    return {"fc_z": fc_z, "fcd_z": fcd_z}


def step5_fit_feature_pipeline(fc_raw, fcd_raw, verbose=True):
    """Step 5. Fit FC PCA (+ optional FCD scaler) on train simulations.

    Progress is printed by FeaturePipeline / FCPCAScaler when verbose=True.
    """
    if verbose:
        # E2 — Step 5 header.
        _progress("[Step 5] Feature pipeline fit")
        print(
            f"           input: fc_raw={fc_raw.shape}  "
            f"fcd_raw={fcd_raw.shape}"
        )
        print(
            f"           PCA_DIM_FC={config.PCA_DIM_FC}  "
            f"USE_FCD={config.USE_FCD}"
        )
        # E3 — PCA start (wrap-around log just before pipeline.fit).
        print(
            f"  fitting FCPCAScaler: {fc_raw.shape} -> "
            f"PCA(n={config.PCA_DIM_FC}) ..."
        )
    pipeline = FeaturePipeline()
    t_pca = time.time()
    _n0 = fc_raw.shape[0]
    _finite = np.isfinite(fc_raw).all(axis=1)
    _notsat = fc_raw.std(axis=1) > 1e-4
    _inrng  = (fc_raw.min(axis=1) >= -1.001) & (fc_raw.max(axis=1) <= 1.001)
    _alive  = np.abs(fc_raw).mean(axis=1) > 1e-3
    _mask = _finite & _notsat & _inrng & _alive
    _n1 = int(_mask.sum())
    print(f"  [Step 5] 필터: {_n0:,} -> {_n1:,}  "
          f"(NaN={int((~_finite).sum())}, sat={int((~_notsat).sum())}, "
          f"oor={int((~_inrng).sum())}, dead={int((~_alive).sum())})",
          flush=True)
    if _n1 < 1000:
        raise RuntimeError(f"유효 시뮬 {_n1}개 < 1000. prior/시뮬 확인 필요.")
    fc_raw = fc_raw[_mask]
    if fcd_raw is not None and fcd_raw.shape[0] == _n0:
        fcd_raw = fcd_raw[_mask]
    try: pipeline._train_mask = _mask
    except: pass
    pipeline.fit(fc_raw, fcd_raw, verbose=verbose)
    pca_elapsed = time.time() - t_pca
    if verbose:
        # E4 — PCA done. EVR pulled from the fitted sklearn PCA object;
        # HC-11: graceful fallback if the attribute path is unavailable.
        try:
            evr = pipeline.fc_pca.pca.explained_variance_ratio_
            n_comp = int(pipeline.fc_pca.pca.n_components_)
            top5_str = "[" + ", ".join(
                f"{float(v):.4f}" for v in evr[:5]
            ) + "]"
            cum = float(evr.sum())
            _progress(
                f"[Step 5] PCA done  {pca_elapsed:.2f} s  "
                f"n_components={n_comp}"
            )
            print(f"           EVR top-5: {top5_str}")
            print(f"           EVR cumulative: {cum:.4f}")
        except Exception as exc:
            _progress(
                f"[Step 5] PCA done  {pca_elapsed:.2f} s  "
                f"EVR: unavailable ({exc!r})"
            )
    t_xf = time.time()
    x_input = pipeline.transform(fc_raw, fcd_raw)
    xf_elapsed = time.time() - t_xf
    if verbose:
        # E5 — Transform done: x_input shape/dtype/range + time.
        _progress(f"[Step 5] transform done  {xf_elapsed:.2f} s")
        print(
            f"           x_input: {x_input.shape}  {x_input.dtype}"
        )
        print(
            f"           range: min={float(x_input.min()):.4f}  "
            f"max={float(x_input.max()):.4f}  "
            f"mean={float(x_input.mean()):.4f}"
        )
        _progress("[Step 5] done")
    return pipeline, x_input


def step6_pca_diagnostic(pipeline, fc_raw, fcd_raw, verbose=True):
    """Step 6. PCA quality check (pre-inference embedding quality)."""
    if verbose:
        _progress("[Step 6] PCA diagnostic")
    pca_diag = pipeline.diagnostic(fc_raw, fcd_raw)
    if verbose:
        _print_pca_diagnostic(pca_diag, header="Step 6 - PCA diagnostic")
        # E6 — PASS/WARN summary derived from pca_diag dict.
        d_fc = pca_diag.get("fc_pca", {})
        evr_sum = float(d_fc.get("explained_variance_sum", float("nan")))
        recon = float(d_fc.get("recon_corr_train_mean", float("nan")))
        evr_status = "PASS" if d_fc.get("pca_pass_evr") else "WARN"
        recon_status = "PASS" if d_fc.get("pca_pass_recon") else "WARN"
        print(
            f"           EVR @ threshold: {evr_sum:.4f}  "
            f"(threshold={config.PCA_EVR_THRESHOLD:.2f}) {evr_status}"
        )
        print(
            f"           recon corr: {recon:.4f}  "
            f"(threshold={config.PCA_RECON_CORR_THRESH:.2f}) {recon_status}"
        )
        _progress("[Step 6] done")
    return pca_diag


def step7_fit_param_scaler(verbose=True):
    """Step 7. Build Stage 1 parameter scaler and scaled prior."""
    if verbose:
        print("\n  [Step 7] Parameter scaling ([-1, 1])")
    param_scaler = make_stage1_param_scaler()
    prior_scaled = make_scaled_prior(
        len(config.STAGE1_PARAMS), device=config.SBI_DEVICE,
    )
    if verbose:
        for name, lo, hi in zip(config.STAGE1_PARAMS,
                                config.STAGE1_PRIOR_LOW,
                                config.STAGE1_PRIOR_HIGH):
            print(f"    {name:6s} : [{lo}, {hi}] -> [-1, 1]")
    return param_scaler, prior_scaled


def step8_train_snpe(theta_scaled, x_input, prior_scaled, verbose=True,
                     fc_raw=None):
    """Step 8. Train single-round amortized SNPE-C."""
    t_step8 = time.time()
    posterior, embedding_net = train_snpe(
        theta_scaled, x_input, prior_scaled,
        embedding_net=None, proposal=None, verbose=verbose,
        fc_raw=fc_raw,
    )
    # E12 — Step 8 summary: total elapsed + posterior + embedding device.
    if verbose:
        try:
            emb_device = next(embedding_net.parameters()).device
        except Exception:
            emb_device = config.SBI_DEVICE
        in_dim = int(x_input.shape[1])
        hidden = config.EMBED_HIDDEN
        out_dim = config.EMBED_DIM
        _progress(f"[Step 8] done  {time.time() - t_step8:.2f} s")
        print(
            f"           posterior: SNPE-C  device={config.SBI_DEVICE}"
        )
        print(
            f"           embedding: FeatureEmbedding("
            f"{in_dim}->{hidden}->{hidden // 2}->{out_dim})"
        )
        print(f"                      device={emb_device}")
    return posterior, embedding_net


# ---------------------------------------------------------------------------
# Internal: SNPE-C per-epoch + final-summary printers (Approach A — reads
# inferer._summary populated by sbi>=0.22; no patching or subclassing).
# ---------------------------------------------------------------------------

def _print_epoch_table(inferer, max_epochs_cap, stop_after):
    """E9: compact per-epoch table from sbi's training summary.

    Throttled output: first 20 epochs verbatim, then every 10th, plus the
    BEST-loss epoch and the final epoch. A "..." marker separates
    non-contiguous picks. Read-only access to ``inferer._summary``.
    """
    summary = getattr(inferer, "_summary", None) or {}
    train_loss = list(summary.get("training_loss", []))
    val_loss = list(summary.get("validation_loss", []))
    n_epochs = max(len(train_loss), len(val_loss))
    if n_epochs == 0:
        print("           (no per-epoch summary available from sbi)")
        return
    if val_loss:
        best_idx = int(np.argmin(val_loss))
    else:
        best_idx = int(np.argmin(train_loss))

    keep = set(range(min(20, n_epochs)))
    keep.update(range(19, n_epochs, 10))
    keep.add(best_idx)
    keep.add(n_epochs - 1)
    keep_sorted = sorted(keep)

    prev = None
    for i in keep_sorted:
        if prev is not None and i - prev > 1:
            print("           ...")
        tr = train_loss[i] if i < len(train_loss) else float("nan")
        va = val_loss[i] if i < len(val_loss) else float("nan")
        marker = "  <- BEST" if i == best_idx else ""
        print(
            f"           epoch {i + 1:>3d}/{max_epochs_cap}  "
            f"train_loss={tr:.4f}  val_loss={va:.4f}{marker}"
        )
        prev = i

    _epochs_run = n_epochs
    if stop_after >= max_epochs_cap:
        print(
            f"  [full training: {_epochs_run} / {max_epochs_cap} epochs]"
        )
    else:
        print(
            f"  [early stop at epoch {_epochs_run}"
            f" - no improvement for {stop_after} epochs]"
        )


def _print_train_summary(inferer, train_elapsed, max_epochs_cap, stop_after):
    """E10: SNPE training-done summary line + epoch / best-loss block."""
    summary = getattr(inferer, "_summary", None) or {}
    val_loss = list(summary.get("validation_loss", []))
    epochs_run_list = summary.get("epochs_trained", [])
    if epochs_run_list:
        epochs_run = int(epochs_run_list[-1])
    else:
        epochs_run = int(getattr(inferer, "epoch", len(val_loss)))
    best_list = summary.get("best_validation_loss", [])
    if best_list:
        best = float(best_list[-1])
    elif val_loss:
        best = float(min(val_loss))
    else:
        best = float("nan")
    if stop_after >= max_epochs_cap:
        early_note = "full training (early stop disabled)"
    elif epochs_run < max_epochs_cap:
        early_note = f"early stop: {stop_after} patience"
    else:
        early_note = "reached max_epochs"
    _progress(f"[Step 8] training done  {train_elapsed:.2f} s")
    print(
        f"           epochs: {epochs_run} / {max_epochs_cap}  "
        f"({early_note})"
    )
    print(f"           best val loss: {best:.4f}")


# ---------------------------------------------------------------------------
# Internal: PCA diagnostic printer
# ---------------------------------------------------------------------------

def _print_pca_diagnostic(pca_diag, header="PCA diagnostic"):
    import config
    def _fmt(v):
        try: return f"{float(v):.4f}"
        except: return "N/A"
    print(f"  [{header}]")
    d_fc = pca_diag.get("fc_pca", {}) or {}
    if d_fc:
        ok = d_fc.get("pca_pass_evr") and d_fc.get("pca_pass_recon")
        print(f"    FC  PCA  : n={d_fc.get('n_components','N/A')}, "
              f"EVR={_fmt(d_fc.get('explained_variance_sum'))}, "
              f"recon={_fmt(d_fc.get('recon_corr_train_mean'))}  "
              f"[{'OK' if ok else 'FAIL'}]")
    if not bool(getattr(config, "USE_FCD", False)):
        print("    FCD      : disabled (USE_FCD=False)")
        return
    d_fcd = pca_diag.get("fcd_pca", {}) or {}
    if not d_fcd:
        print("    FCD PCA  : (no data)"); return
    ok = d_fcd.get("pca_pass_evr") and d_fcd.get("pca_pass_recon")
    print(f"    FCD PCA  : n={d_fcd.get('n_components','N/A')}, "
          f"EVR={_fmt(d_fcd.get('explained_variance_sum'))}, "
          f"recon={_fmt(d_fcd.get('recon_corr_train_mean'))}  "
          f"[{'OK' if ok else 'FAIL'}]")
