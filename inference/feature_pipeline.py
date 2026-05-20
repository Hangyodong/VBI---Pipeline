"""Feature scalers and pipeline (FC PCA + optional FCD z-score).

Public API
----------
- FamilyScaler        : per-feature z-score, train fit only
- FCPCAScaler         : FC raw upper triangle -> PCA (no z-score)
- FeaturePipeline     : FC PCA (+ optional FCD z-score) combined

Why these are in inference/, not features/
------------------------------------------
``features/`` modules compute raw features from data; they do not learn
anything from the training distribution. Scalers and PCA, in contrast,
**must** be fitted on training simulations only and frozen before being
applied to validation/test/observed data — this is an inference-stage
concern.

Refusing silent fallback
------------------------
``FeaturePipeline.transform`` raises ``ValueError`` on dimension
mismatch. We never broadcast / pad / truncate / replace with x_train.mean.
Simulated and observed feature vectors must live in the same space.
"""
import time

import numpy as np

import config


# ---------------------------------------------------------------------------
# Per-feature z-score
# ---------------------------------------------------------------------------

class FamilyScaler:
    """Per-feature z-score. Fit only on training data."""

    def __init__(self, name="feature"):
        self.name = name
        self.mean_ = None
        self.std_ = None
        self.fitted = False

    def fit(self, x_train):
        x = np.asarray(x_train, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        self.fitted = True
        return self

    def transform(self, x):
        if not self.fitted:
            raise RuntimeError(f"{self.name} scaler not fitted")
        x = np.asarray(x, dtype=np.float32)
        squeeze = (x.ndim == 1)
        if squeeze:
            x = x[None]
        out = ((x - self.mean_) / self.std_).astype(np.float32)
        return out[0] if squeeze else out

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)


# ---------------------------------------------------------------------------
# FC PCA (no z-score)
# ---------------------------------------------------------------------------

class FCPCAScaler:
    """FC raw upper triangle -> PCA (no z-score). Train fit only.

    FC is already a Pearson correlation in [-1, 1] so per-feature
    z-scoring is unnecessary and would distort the variance structure.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components or config.PCA_DIM
        self.pca = None
        self.fitted = False

    def _make_pca(self, n_comp):
        from sklearn.decomposition import PCA
        # randomized solver: ~30x faster for n_components << n_features
        return PCA(n_components=n_comp, svd_solver="randomized",
                   random_state=42)

    def fit(self, fc_train_raw, verbose=True):
        """Fit PCA. verbose=True prints shape, time, EVR."""
        n_samples, n_feat = fc_train_raw.shape
        n_comp = min(self.n_components, n_samples, n_feat)
        if n_comp != self.n_components:
            if verbose:
                print(
                    f"    [PCA] n_components capped: "
                    f"{self.n_components} -> {n_comp} "
                    f"(data {fc_train_raw.shape})"
                )
            self.n_components = n_comp
        if verbose:
            print(
                f"    [PCA] fitting  "
                f"input=({n_samples:,} x {n_feat:,})  "
                f"n_components={n_comp}  solver=randomized ..."
            )
        t0 = time.time()
        self.pca = self._make_pca(n_comp)
        self.pca.fit(fc_train_raw)
        self.fitted = True
        if verbose:
            evr = self.pca.explained_variance_ratio_
            cum = float(evr.cumsum()[-1]) * 100
            top5 = ", ".join(f"{v * 100:.1f}%" for v in evr[:5])
            print(
                f"    [PCA] done  ({time.time() - t0:.1f}s)\n"
                f"      cumulative EVR       : {cum:.2f}%\n"
                f"      top-5 PC EVR         : {top5}\n"
                f"      output shape         : ({n_samples:,} x {n_comp})"
            )
        return self

    def transform(self, fc_raw):
        if not self.fitted:
            raise RuntimeError("FCPCAScaler not fitted")
        return self.pca.transform(np.atleast_2d(fc_raw)).astype(np.float32)

    def fit_transform(self, fc_train_raw, verbose=True):
        return self.fit(fc_train_raw, verbose=verbose).transform(fc_train_raw)

    def inverse_transform(self, fc_pca):
        return self.pca.inverse_transform(fc_pca).astype(np.float32)

    @property
    def explained_variance_ratio_(self):
        return self.pca.explained_variance_ratio_

    def diagnostic(self, fc_train_raw, fc_val_raw=None):
        """PCA quality diagnostic: EVR, reconstruction corr, val shift."""
        evr = self.pca.explained_variance_ratio_
        cum_evr = float(evr.cumsum()[-1])

        x_pca = self.pca.transform(fc_train_raw)
        x_recon = self.pca.inverse_transform(x_pca)
        recon_corrs = [
            float(np.corrcoef(fc_train_raw[i], x_recon[i])[0, 1])
            for i in range(min(len(fc_train_raw), 500))
        ]
        recon_corr_train = float(np.mean(recon_corrs))

        result = {
            "n_components": self.n_components,
            "explained_variance_sum": cum_evr,
            "explained_variance_top5": evr[:5].tolist(),
            "recon_corr_train_mean": recon_corr_train,
            "pca_pass_evr": cum_evr >= config.PCA_EVR_THRESHOLD,
            "pca_pass_recon": (
                recon_corr_train >= config.PCA_RECON_CORR_THRESH
            ),
        }

        if fc_val_raw is not None and len(fc_val_raw) > 0:
            x_val_pca = self.pca.transform(fc_val_raw)
            tr_mean = x_pca.mean(axis=0)[:5]
            tr_std = x_pca.std(axis=0)[:5]
            val_mean = x_val_pca.mean(axis=0)[:5]
            shift = np.abs(val_mean - tr_mean) / (tr_std + 1e-8)
            result["pca_train_val_shift_top5"] = shift.tolist()
            result["pca_train_val_max_shift"] = float(shift.max())
            result["pca_train_val_overlap_ok"] = float(shift.max()) < 2.0
        return result


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """FC PCA (+ optional FCD z-score), fitted on training set.

    FC (6555-dim) -> PCA -> config.PCA_DIM_FC
    FCD (5-dim)   -> z-score          (only if config.USE_FCD)
    Concatenated: (PCA_DIM_FC,) or (PCA_DIM_FC + 5,)
    """

    def __init__(self):
        self.fc_pca = FCPCAScaler(n_components=config.PCA_DIM_FC)
        self.fcd_z = FamilyScaler(name="FCD") if config.USE_FCD else None
        self.use_fcd = bool(config.USE_FCD)
        self.fc_dim = None
        self.fcd_dim = None
        self.input_dim = None
        self.fitted = False

    def fit(self, fc_train_raw, fcd_train_raw, verbose=True):
        """Fit FC PCA (+ optional FCD scaler). verbose prints progress."""
        self.fc_dim = fc_train_raw.shape[1]
        if verbose:
            print(
                f"\n  [FeaturePipeline] FC PCA  "
                f"({fc_train_raw.shape[0]:,} samples x "
                f"{fc_train_raw.shape[1]:,} features -> "
                f"{self.fc_pca.n_components} PCs)"
            )
        self.fc_pca.fit(fc_train_raw, verbose=verbose)
        if self.use_fcd:
            if verbose:
                print(
                    f"\n  [FeaturePipeline] FCD z-score  "
                    f"({fcd_train_raw.shape[0]:,} samples x "
                    f"{fcd_train_raw.shape[1]} features)"
                )
            self.fcd_z.fit(fcd_train_raw)
            self.fcd_dim = fcd_train_raw.shape[1]
            self.input_dim = self.fc_pca.n_components + self.fcd_dim
            if verbose:
                print(
                    f"    [FCD] mean={fcd_train_raw.mean():.4f}  "
                    f"std={fcd_train_raw.std():.4f}"
                )
        else:
            self.fcd_dim = 0
            self.input_dim = self.fc_pca.n_components
        self.fitted = True
        if verbose:
            print(
                f"\n  [FeaturePipeline] done  output_dim={self.input_dim}"
            )
        return self

    def transform(self, fc_raw, fcd_raw):
        if not self.fitted:
            raise RuntimeError("FeaturePipeline not fitted")
        fc_2d = np.atleast_2d(fc_raw)
        if fc_2d.shape[1] != self.fc_dim:
            raise ValueError(
                f"FC input dim mismatch: got {fc_2d.shape[1]}, "
                f"pipeline was fitted on {self.fc_dim}. "
                f"Refusing to silently broadcast — simulated and observed "
                f"features must share the same dimension."
            )
        fc_pca = self.fc_pca.transform(fc_2d)
        if self.use_fcd:
            fcd_2d = np.atleast_2d(fcd_raw)
            if fcd_2d.shape[1] != self.fcd_dim:
                raise ValueError(
                    f"FCD input dim mismatch: got {fcd_2d.shape[1]}, "
                    f"pipeline was fitted on {self.fcd_dim}."
                )
            fcd_scaled = self.fcd_z.transform(fcd_2d)
            out = np.concatenate([fc_pca, fcd_scaled], axis=1)
        else:
            out = fc_pca
        out = out.astype(np.float32)
        if out.shape[1] != self.input_dim:
            raise ValueError(
                f"Output dim mismatch: got {out.shape[1]}, "
                f"expected {self.input_dim}"
            )
        return out[0] if fc_raw.ndim == 1 else out

    def fit_transform(self, fc_train_raw, fcd_train_raw, verbose=True):
        self.fit(fc_train_raw, fcd_train_raw, verbose=verbose)
        return self.transform(fc_train_raw, fcd_train_raw)

    def diagnostic(self, fc_train_raw, fcd_train_raw=None,
                   fc_val_raw=None, fcd_val_raw=None):
        d_fc = self.fc_pca.diagnostic(fc_train_raw, fc_val_raw)
        if self.use_fcd:
            d_fcd = {
                "n_components": self.fcd_dim,
                "type": "summary_stats (no PCA)",
                "dims": ["mean", "std", "q25", "q50", "q75"],
            }
            if fcd_train_raw is not None:
                d_fcd["train_mean"] = float(fcd_train_raw.mean())
                d_fcd["train_std"] = float(fcd_train_raw.std())
        else:
            d_fcd = {"type": "disabled", "n_components": 0}
        return {"fc_pca": d_fc, "fcd_pca": d_fcd}
