"""Feature scalers and pipeline (raw FC passthrough + optional FCD z-score).

Public API
----------
- FamilyScaler        : per-feature z-score, train fit only
- FeaturePipeline     : raw FC passthrough (+ optional FCD z-score) combined

FC is fed to SNPE-C as the raw 6555-dim upper triangle (no PCA, no
z-score) — the RegionTransformer embedding (Phase 3) learns its own
projection. Only the optional FCD family is z-scored, and that scaler
**must** be fitted on training simulations only and frozen before being
applied to validation/test/observed data — an inference-stage concern.

Refusing silent fallback
------------------------
``FeaturePipeline.transform`` raises ``ValueError`` on dimension
mismatch. We never broadcast / pad / truncate / replace with x_train.mean.
Simulated and observed feature vectors must live in the same space.
"""
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
# Combined pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Raw FC passthrough (+ optional FCD z-score), fitted on training set.

        FC (6555-dim) -> raw passthrough  (output_dim == fc_dim)
        FCD (5-dim)   -> z-score          (only if config.USE_FCD)
    Concatenated: (FC_dim,) or (FC_dim + FCD_dim,)
    """

    def __init__(self):
        self.fcd_z = FamilyScaler(name="FCD") if config.USE_FCD else None
        self.use_fcd = bool(config.USE_FCD)
        self.fc_dim = None          # raw FC input dim (e.g. 64620)
        self.fcd_dim = None
        self.input_dim = None
        self.fitted = False
        # Optional FC dimensionality reduction (PCA). config.FC_PCA_DIM>0 enables
        # it: raw FC (fc_dim) -> n_components, whitened. Replaces the removed
        # RegionTransformer embedding. 0 (default) = passthrough (other pipelines).
        self.fc_pca_dim = int(getattr(config, "FC_PCA_DIM", 0) or 0)
        self.pca = None
        self.fc_out_dim = None      # FC dim AFTER pca (== fc_dim if passthrough)

    def fit(self, fc_train_raw, fcd_train_raw, verbose=True):
        """Record FC dim (raw passthrough) + fit optional FCD scaler."""
        fc_train_raw = np.asarray(fc_train_raw, dtype=np.float32)
        self.fc_dim = fc_train_raw.shape[1]
        if self.fc_pca_dim > 0:
            from sklearn.decomposition import PCA
            # PCA components <= min(requested, n_samples-1, n_features)
            n_comp = int(min(self.fc_pca_dim,
                             max(1, fc_train_raw.shape[0] - 1),
                             self.fc_dim))
            # S4: whitening is configurable. whiten=True divides each PC by its
            # singular value -> low-variance directions get amplified, which can
            # push an OOD empirical FC far outside the simulated-FC training
            # distribution (rejection-sampling acceptance collapses). whiten=False
            # keeps the natural scale so observed FC projections stay bounded.
            _whiten = bool(getattr(config, "FC_PCA_WHITEN", True))
            self.pca = PCA(n_components=n_comp, whiten=_whiten,
                           svd_solver="auto", random_state=0).fit(fc_train_raw)
            fc_out_dim = int(self.pca.n_components_)
            if verbose:
                ev = float(self.pca.explained_variance_ratio_.sum())
                print(
                    f"\n  [FeaturePipeline] FC PCA  "
                    f"({fc_train_raw.shape[0]:,} samples)  "
                    f"{self.fc_dim:,} -> {fc_out_dim} comps "
                    f"(whiten={_whiten}, explained var={ev:.3f})"
                )
        else:
            fc_out_dim = self.fc_dim
            if verbose:
                print(
                    f"\n  [FeaturePipeline] FC passthrough  "
                    f"({fc_train_raw.shape[0]:,} samples x "
                    f"{fc_train_raw.shape[1]:,} features, raw upper-tri)"
                )
        self.fc_out_dim = fc_out_dim
        if self.use_fcd:
            fcd_train_raw = np.asarray(fcd_train_raw, dtype=np.float32)
            if verbose:
                print(
                    f"\n  [FeaturePipeline] FCD z-score  "
                    f"({fcd_train_raw.shape[0]:,} samples x "
                    f"{fcd_train_raw.shape[1]} features)"
                )
            self.fcd_z.fit(fcd_train_raw)
            self.fcd_dim = fcd_train_raw.shape[1]
            self.input_dim = fc_out_dim + self.fcd_dim
            if verbose:
                print(
                    f"    [FCD] mean={fcd_train_raw.mean():.4f}  "
                    f"std={fcd_train_raw.std():.4f}"
                )
        else:
            self.fcd_dim = 0
            self.input_dim = fc_out_dim
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
        fc_out = fc_2d.astype(np.float32)
        if self.pca is not None:
            fc_out = self.pca.transform(fc_out).astype(np.float32)  # -> (n, n_comp)
        if self.use_fcd:
            fcd_2d = np.atleast_2d(fcd_raw)
            if fcd_2d.shape[1] != self.fcd_dim:
                raise ValueError(
                    f"FCD input dim mismatch: got {fcd_2d.shape[1]}, "
                    f"pipeline was fitted on {self.fcd_dim}."
                )
            fcd_scaled = self.fcd_z.transform(fcd_2d)
            out = np.concatenate([fc_out, fcd_scaled], axis=1)
        else:
            out = fc_out
        out = out.astype(np.float32)
        if out.shape[1] != self.input_dim:
            raise ValueError(
                f"Output dim mismatch: got {out.shape[1]}, "
                f"expected {self.input_dim}"
            )
        return out[0] if (hasattr(fc_raw, "ndim") and fc_raw.ndim == 1) else out

    def fit_transform(self, fc_train_raw, fcd_train_raw, verbose=True):
        self.fit(fc_train_raw, fcd_train_raw, verbose=verbose)
        return self.transform(fc_train_raw, fcd_train_raw)

    def diagnostic(self, fc_train_raw, fcd_train_raw=None,
                   fc_val_raw=None, fcd_val_raw=None):
        if self.pca is not None:
            d_fc = {"type": "pca_whiten", "n_components": int(self.pca.n_components_),
                    "fc_dim_in": self.fc_dim,
                    "explained_var": float(self.pca.explained_variance_ratio_.sum())}
        else:
            d_fc = {"type": "passthrough", "n_components": self.fc_dim}
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


# ---------------------------------------------------------------------------
# Selected-index persistence (Phase-2 → Phase-3 hand-off)
# ---------------------------------------------------------------------------

def save_selected_indices(indices, path):
    """Persist Phase-2 selected FC indices for the Phase-3 re-fit."""
    import os
    arr = np.asarray(indices, dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.save(path, arr)
    return path


def load_selected_indices(path):
    """Load Phase-2 selected FC indices saved by ``save_selected_indices``."""
    return np.load(path).astype(np.int64)
