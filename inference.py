"""Scaling, embedding, SNPE-C training, and two-stage inference.

Module layout
-------------
- ParameterScaler  : prior-based [-1, 1] mapping (data-free)
- FamilyScaler     : per-feature z-score, train fit only
- FCPCAScaler      : FC z-score + PCA, train fit only
- FeaturePipeline  : FC PCA + FCD PCA combined
- FeatureEmbedding : MLP head jointly trained with SBI
- run_stage1_snpe  : end-to-end Stage 1 SNPE-C
- run_stage2_snpe  : Stage 2 with nuisance posterior_sample / fix_mean
- Diagnostics      : shrinkage, posterior correlation, SBC, MLP probing
"""
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False

import config


# ---------------------------------------------------------------------------
# Progress printing
# ---------------------------------------------------------------------------

def _progress(msg):
    """Print a timestamped progress message and flush."""
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Parameter scaling
# ---------------------------------------------------------------------------

class ParameterScaler:
    """Map raw parameters to/from the [-1, 1] scaled box defined by prior."""

    def __init__(self, param_names, prior_low, prior_high):
        self.param_names = list(param_names)
        self.low = np.asarray(prior_low, dtype=np.float32)
        self.high = np.asarray(prior_high, dtype=np.float32)
        self.range = self.high - self.low
        if (self.range <= 0).any():
            raise ValueError("prior range must be positive")

    def transform(self, theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=np.float32)
        scaled = 2.0 * (theta_raw - self.low) / self.range - 1.0
        return scaled.astype(np.float32)

    def inverse_transform(self, theta_scaled):
        theta_scaled = np.asarray(theta_scaled, dtype=np.float32)
        raw = (theta_scaled + 1.0) / 2.0 * self.range + self.low
        return raw.astype(np.float32)

    def to_dict(self, theta=None):
        """Return either the scaler config or a param-name->value mapping.

        If `theta` is None: returns scaler metadata (param_names, low, high).
        If `theta` is an array of length n_params: returns
            {param_name: value} mapping (raw values, not scaled).
        """
        if theta is None:
            return {
                "param_names": self.param_names,
                "low": self.low.tolist(),
                "high": self.high.tolist(),
            }
        theta = np.asarray(theta, dtype=np.float32).ravel()
        if theta.shape[0] != len(self.param_names):
            raise ValueError(
                f"theta length {theta.shape[0]} != "
                f"n_params {len(self.param_names)}"
            )
        return {n: float(v) for n, v in zip(self.param_names, theta)}

    def subset(self, param_names):
        """Return a new ParameterScaler over a subset of parameters."""
        idx = []
        for n in param_names:
            if n not in self.param_names:
                raise ValueError(f"{n} not in {self.param_names}")
            idx.append(self.param_names.index(n))
        return ParameterScaler(
            [self.param_names[i] for i in idx],
            self.low[idx].tolist(),
            self.high[idx].tolist(),
        )

    @classmethod
    def from_dict(cls, d):
        return cls(d["param_names"], d["low"], d["high"])


def make_stage1_param_scaler():
    """ParameterScaler covering the Stage 1 prior."""
    return ParameterScaler(
        config.STAGE1_PARAMS,
        config.STAGE1_PRIOR_LOW,
        config.STAGE1_PRIOR_HIGH,
    )


def make_stage2_param_scaler(stage2_params):
    """ParameterScaler over selected Stage 1 params + c-params."""
    s1_lookup = {
        n: (low, high) for n, low, high in zip(
            config.STAGE1_PARAMS,
            config.STAGE1_PRIOR_LOW,
            config.STAGE1_PRIOR_HIGH,
        )
    }
    low, high = [], []
    for name in stage2_params:
        if name in s1_lookup:
            lo, hi = s1_lookup[name]
        elif name in config.C_PARAM_PRIOR:
            lo, hi = config.C_PARAM_PRIOR[name]
        else:
            raise ValueError(f"Unknown parameter: {name}")
        low.append(lo)
        high.append(hi)
    return ParameterScaler(stage2_params, low, high)


# ---------------------------------------------------------------------------
# Scaled prior (BoxUniform [-1, 1])
# ---------------------------------------------------------------------------

def make_scaled_prior(n_dim):
    """Build the scaled-space BoxUniform prior used by SNPE."""
    from sbi.utils import BoxUniform
    return BoxUniform(
        low=torch.full((n_dim,), -1.0, dtype=torch.float32),
        high=torch.full((n_dim,), +1.0, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Feature scalers
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

    def fit(self, fc_train_raw):
        n_comp = min(self.n_components, *fc_train_raw.shape)
        if n_comp != self.n_components:
            self.n_components = n_comp
        self.pca = self._make_pca(n_comp)
        self.pca.fit(fc_train_raw)
        self.fitted = True
        return self

    def transform(self, fc_raw):
        if not self.fitted:
            raise RuntimeError("FCPCAScaler not fitted")
        return self.pca.transform(np.atleast_2d(fc_raw)).astype(np.float32)

    def fit_transform(self, fc_train_raw):
        return self.fit(fc_train_raw).transform(fc_train_raw)

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


class FeaturePipeline:
    """FC PCA (+ optional FCD summary stats), fitted on training set.

    FC (6555-dim) -> z-score -> PCA -> config.PCA_DIM_FC
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

    def fit(self, fc_train_raw, fcd_train_raw):
        self.fc_pca.fit(fc_train_raw)
        self.fc_dim = fc_train_raw.shape[1]
        if self.use_fcd:
            self.fcd_z.fit(fcd_train_raw)
            self.fcd_dim = fcd_train_raw.shape[1]
            self.input_dim = self.fc_pca.n_components + self.fcd_dim
        else:
            self.fcd_dim = 0
            self.input_dim = self.fc_pca.n_components
        self.fitted = True
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

    def fit_transform(self, fc_train_raw, fcd_train_raw):
        self.fit(fc_train_raw, fcd_train_raw)
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


# ---------------------------------------------------------------------------
# Embedding network
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    _EMBED_BASE = torch.nn.Module
else:
    _EMBED_BASE = object


class FeatureEmbedding(_EMBED_BASE):
    """MLP head jointly trained with the SBI density estimator."""

    def __init__(self, input_dim, hidden_dim=None, out_dim=None):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for FeatureEmbedding")
        super().__init__()
        hidden_dim = hidden_dim or config.EMBED_HIDDEN
        out_dim = out_dim or config.EMBED_DIM
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training data collection (Stage 1)
# ---------------------------------------------------------------------------

def collect_training_data(subjects, subject_data, prior_scaled,
                          theta_scaler, param_names, n_sim,
                          fixed_overrides=None, apply_bw=True,
                          verbose=True):
    """Run simulations and extract features for the training set."""
    from simulator import simulate_gpu_batch, worker_extract
    import cupy as cp

    all_theta_s = []
    all_theta_r = []
    all_fc = []
    all_fcd = []
    t0 = time.time()

    n_subj = len(subjects)
    if verbose:
        print(
            f"  Training data collection: {n_subj} subjects x "
            f"{n_sim} = {n_subj * n_sim} sims"
        )

    with ProcessPoolExecutor(max_workers=config.N_CPU) as executor:
        for s_idx, sid in enumerate(subjects):
            batch_sz = config.GPU_BATCH
            n_batches = (n_sim + batch_sz - 1) // batch_sz

            if verbose:
                _progress(
                    f"[{s_idx + 1}/{n_subj}] {sid}  "
                    f"({n_batches} batches x {batch_sz} sims)"
                )
            d = subject_data[sid]
            sc = d["sc"]
            dly = d["delays"]

            theta_s = (
                prior_scaled.sample((n_sim,))
                .cpu().numpy().astype(np.float32)
            )
            theta_r = theta_scaler.inverse_transform(theta_s)

            future_queue = []
            t_sub = time.time()

            for b_idx in range(n_batches):
                start = b_idx * batch_sz
                end = min(start + batch_sz, n_sim)
                chunk_r = theta_r[start:end]
                chunk_s = theta_s[start:end]

                try:
                    bolds = simulate_gpu_batch(
                        sc, chunk_r, param_names=param_names,
                        fixed_overrides=fixed_overrides,
                        delays=dly, apply_bw=apply_bw,
                    )
                except Exception as e:
                    print(f"  batch {b_idx} failed: {e}")
                    continue

                future = executor.map(worker_extract, bolds, chunksize=16)
                future_queue.append((chunk_s, chunk_r, future))

                while len(future_queue) >= 2:
                    _drain_one_future(
                        future_queue.pop(0),
                        all_theta_s, all_theta_r, all_fc, all_fcd,
                    )

                if verbose:
                    elapsed = time.time() - t_sub
                    n_done = len(all_theta_s)
                    total = n_subj * n_sim
                    pct = n_done / max(total, 1) * 100
                    _progress(
                        f"batch {b_idx + 1}/{n_batches}  "
                        f"sim {end}/{n_sim}  "
                        f"total {n_done}/{total} ({pct:.1f}%)  "
                        f"({elapsed:.1f}s)"
                    )

            for queued in future_queue:
                _drain_one_future(
                    queued,
                    all_theta_s, all_theta_r, all_fc, all_fcd,
                )

            cp.get_default_memory_pool().free_all_blocks()
            if verbose:
                elapsed = time.time() - t_sub
                _progress(
                    f"{sid} done: {len(all_theta_s)} samples  "
                    f"({elapsed:.1f}s)"
                )

    theta_s = np.array(all_theta_s, dtype=np.float32)
    theta_r = np.array(all_theta_r, dtype=np.float32)
    fc_raw = np.array(all_fc, dtype=np.float32)
    fcd_raw = np.array(all_fcd, dtype=np.float32)

    if verbose:
        print(
            f"\n  Total collected: theta={theta_s.shape}, "
            f"fc={fc_raw.shape}, fcd={fcd_raw.shape}  "
            f"({time.time() - t0:.1f}s)"
        )
    return theta_s, theta_r, fc_raw, fcd_raw


def _drain_one_future(queued, all_theta_s, all_theta_r, all_fc, all_fcd):
    """Append finite results from one future onto the accumulator lists."""
    tcs, tcr, future = queued
    for i, res in enumerate(future):
        if res is None:
            continue
        fc_vec, fcd_vec = res
        if not (np.all(np.isfinite(fc_vec))
                and np.all(np.isfinite(fcd_vec))):
            continue
        all_theta_s.append(tcs[i].tolist())
        all_theta_r.append(tcr[i].tolist())
        all_fc.append(fc_vec)
        all_fcd.append(fcd_vec)


# ---------------------------------------------------------------------------
# SNPE-C training
# ---------------------------------------------------------------------------

def train_snpe(theta_scaled, x_input, prior_scaled, embedding_net=None,
               proposal=None, verbose=True):
    """Train SNPE-C jointly with the embedding network.

    Works with sbi 0.22+ where `posterior_nn` moved from
    ``sbi.utils`` to ``sbi.neural_nets``.
    """
    from sbi.inference import SNPE_C

    # posterior_nn moved between versions — try both locations
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        try:
            from sbi.utils import posterior_nn
        except ImportError:
            from sbi.utils.get_nn_models import posterior_nn

    theta_t = torch.tensor(theta_scaled, dtype=torch.float32)
    x_t = torch.tensor(x_input, dtype=torch.float32)

    if embedding_net is None:
        embedding_net = FeatureEmbedding(input_dim=x_input.shape[1])

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
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    inferer.append_simulations(theta_t, x_t, proposal=proposal)

    t0 = time.time()
    # show_train_summary=False to keep cell output short.
    # We print start / end timestamps and final loss separately.
    estimator = inferer.train(
        training_batch_size=512,
        stop_after_epochs=20,
        max_num_epochs=300,
        show_train_summary=False,
    )
    if verbose:
        _progress(
            f"[Step 8] SNPE training done ({config.SBI_DEVICE}): "
            f"{time.time() - t0:.1f}s"
        )

    posterior = inferer.build_posterior(estimator)
    return posterior, embedding_net


# ---------------------------------------------------------------------------
# Stage 1 - per-step functions (steps 2 - 8)
# ---------------------------------------------------------------------------

def step2_simulate_train(train_subjects, subject_data, prior_scaled,
                         param_scaler, n_sim=None, apply_bw=True,
                         verbose=True):
    """Step 2. Simulate WC for the training set + sample features.

    Returns ``(theta_scaled, theta_raw, fc_raw, fcd_raw)``.

    Although the name says "simulate", FC/FCD extraction (step 3) is
    interleaved with simulation by design: BOLD outputs are streamed
    through worker processes batch-by-batch, so extracted features come
    out of the same loop. This avoids storing all BOLD timeseries.
    """
    n_sim = n_sim or config.N_SIM
    if verbose:
        _progress(
            f"[Step 2] WC simulation start  "
            f"(n_sim={n_sim}, GPU_BATCH={config.GPU_BATCH}, "
            f"subjects={len(train_subjects)})"
        )
    out = collect_training_data(
        train_subjects, subject_data, prior_scaled,
        theta_scaler=param_scaler,
        param_names=config.STAGE1_PARAMS,
        n_sim=n_sim,
        fixed_overrides=None,
        apply_bw=apply_bw,
        verbose=verbose,
    )
    theta_s, theta_r, fc_raw, fcd_raw = out
    if theta_s.ndim < 2 or theta_s.shape[0] == 0:
        raise RuntimeError(
            "Step 2 collected 0 samples — all simulation batches failed. "
            "Most common cause: GPU_BATCH is too large and VBI OOMs. "
            "Try setting GPU_BATCH to a smaller value in the Setup cell "
            "(e.g. 4000 instead of 10000) and re-run from Setup."
        )
    return out


def step3_summary_features(fc_raw, fcd_raw, verbose=True):
    """Step 3. Summarize extracted feature shapes.

    Feature extraction itself happens inside step 2's streaming loop
    (FC upper triangle + FCD upper triangle, both NaN-safe). This
    function only reports shapes and finite-value diagnostics.
    """
    if verbose:
        print("\n  [Step 3] Feature summary")
        print(f"    FC  raw : {fc_raw.shape}  "
              f"(finite={np.all(np.isfinite(fc_raw))})")
        print(f"    FCD raw : {fcd_raw.shape}  "
              f"(finite={np.all(np.isfinite(fcd_raw))})")
    return {
        "fc_dim": fc_raw.shape[1],
        "fcd_dim": fcd_raw.shape[1],
        "n_samples": fc_raw.shape[0],
    }


def save_extracted_features(theta_scaled, theta_raw, fc_raw, fcd_raw,
                            param_names=None, save_dir=None,
                            tag="stage1", verbose=True):
    """Save raw extracted features (pre-embedding) to disk.

    Saves the following arrays as a single compressed .npz file:

    - theta_scaled : parameters in [-1, 1] space (n_sim, n_params)
    - theta_raw    : parameters in original units (n_sim, n_params)
    - fc_raw       : FC upper triangle, Fisher z-transformed (n_sim, fc_dim)
    - fcd_raw      : FCD upper triangle, raw values (n_sim, fcd_dim)
    - param_names  : list of parameter names

    Parameters
    ----------
    theta_scaled, theta_raw, fc_raw, fcd_raw : np.ndarray
        Output of `step2_simulate_train`.
    param_names : list[str], optional
        Defaults to `config.STAGE1_PARAMS`.
    save_dir : str, optional
        Defaults to `config.OUTPUT_DIR`.
    tag : str
        File name tag, e.g. "stage1" or "stage2".

    Returns
    -------
    path : str
        Full path of the written .npz file.
    """
    import os
    save_dir = save_dir or config.OUTPUT_DIR
    param_names = param_names or config.STAGE1_PARAMS
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"features_{tag}.npz")
    np.savez_compressed(
        path,
        theta_scaled=theta_scaled,
        theta_raw=theta_raw,
        fc_raw=fc_raw,
        fcd_raw=fcd_raw,
        param_names=np.array(param_names),
    )

    if verbose:
        n_mb = os.path.getsize(path) / 1024 / 1024
        print(f"\n  [save_features] {path}  ({n_mb:.1f} MB)")
        print(f"    theta_scaled : {theta_scaled.shape}")
        print(f"    theta_raw    : {theta_raw.shape}")
        print(f"    fc_raw       : {fc_raw.shape}  (Fisher-z upper tri)")
        print(f"    fcd_raw      : {fcd_raw.shape}  (FCD upper tri)")
    return path


def load_extracted_features(save_dir=None, tag="stage1"):
    """Load features saved by `save_extracted_features`.

    Returns
    -------
    dict with keys: theta_scaled, theta_raw, fc_raw, fcd_raw, param_names.
    """
    import os
    save_dir = save_dir or config.OUTPUT_DIR
    path = os.path.join(save_dir, f"features_{tag}.npz")
    data = np.load(path, allow_pickle=True)
    return {
        "theta_scaled": data["theta_scaled"],
        "theta_raw": data["theta_raw"],
        "fc_raw": data["fc_raw"],
        "fcd_raw": data["fcd_raw"],
        "param_names": list(data["param_names"]),
    }


def step4_fit_feature_scalers(fc_raw, fcd_raw, verbose=True):
    """Step 4. Fit FC and FCD z-score scalers (train data only).

    Returns the per-family ``FamilyScaler`` objects. Note that the
    feature pipeline (step 5) wraps these scalers together with PCA;
    this function is exposed mainly so that the notebook can show the
    z-score step as its own cell.
    """
    if verbose:
        _progress("[Step 4] Fit feature z-score scalers (train only)")
    # FC: no z-score (already Pearson r in [-1, 1])
    fc_z = None
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
    """Step 5. Fit FC PCA + FCD PCA and concatenate.

    Returns ``(pipeline, x_input)``. The pipeline owns its own
    FamilyScalers; the ones fitted in step 4 are kept only for
    diagnostics and are not re-used here.
    """
    if verbose:
        _progress(
            f"[Step 5] Fit FC + FCD PCA pipeline  "
            f"(FC: {fc_raw.shape} -> {config.PCA_DIM_FC} PCs, "
            f"FCD: {fcd_raw.shape} -> {config.PCA_DIM_FCD} PCs)"
        )
    pipeline = FeaturePipeline()
    pipeline.fit(fc_raw, fcd_raw)
    x_input = pipeline.transform(fc_raw, fcd_raw)
    if verbose:
        print(f"    FC  PCA  : {fc_raw.shape[1]} -> "
              f"{pipeline.fc_pca.n_components}")
        print(f"    FCD      : {fcd_raw.shape[1]} dims "
              f"(summary stats, z-scored)")
        print(f"    x_input  : {x_input.shape}")
        _progress("[Step 5] done")
    return pipeline, x_input


def step6_pca_diagnostic(pipeline, fc_raw, fcd_raw, verbose=True):
    """Step 6. PCA quality check (pre-inference embedding quality).

    Computes EVR, reconstruction correlation, and train/val shift for
    both FC and FCD PCAs. MLP linear probing (the post-inference half
    of the embedding quality check) lives in step 9 because it
    requires the trained embedding network.
    """
    if verbose:
        _progress("[Step 6] PCA diagnostic")
    pca_diag = pipeline.diagnostic(fc_raw, fcd_raw)
    if verbose:
        _print_pca_diagnostic(pca_diag, header="Step 6 - PCA diagnostic")
        _progress("[Step 6] done")
    return pca_diag


def step7_fit_param_scaler(verbose=True):
    """Step 7. Build Stage 1 parameter scaler and scaled prior.

    No data needed: this depends only on the prior bounds.
    Returns ``(param_scaler, prior_scaled)``.
    """
    if verbose:
        print("\n  [Step 7] Parameter scaling ([-1, 1])")
    param_scaler = make_stage1_param_scaler()
    prior_scaled = make_scaled_prior(len(config.STAGE1_PARAMS))
    if verbose:
        for name, lo, hi in zip(config.STAGE1_PARAMS,
                                config.STAGE1_PRIOR_LOW,
                                config.STAGE1_PRIOR_HIGH):
            print(f"    {name:6s} : [{lo}, {hi}] -> [-1, 1]")
    return param_scaler, prior_scaled


def step8_train_snpe(theta_scaled, x_input, prior_scaled, verbose=True):
    """Step 8. Train single-round amortized SNPE-C.

    Trains the MLP embedding network jointly with the density
    estimator. Returns ``(posterior, embedding_net)``.
    """
    if verbose:
        n_samples = len(theta_scaled)
        _progress(
            f"[Step 8] SNPE-C training start  "
            f"(samples={n_samples}, x_dim={x_input.shape[1]}, "
            f"device={config.SBI_DEVICE})"
        )
    return train_snpe(
        theta_scaled, x_input, prior_scaled,
        embedding_net=None, proposal=None, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Stage 1 entry point (atomic wrapper around steps 2 - 8)
# ---------------------------------------------------------------------------

def run_stage1_snpe(train_subjects, subject_data, n_sim=None,
                    apply_bw=True, verbose=True):
    """End-to-end Stage 1 inference (steps 2 - 8).

    Convenience wrapper that chains ``step2_simulate_train`` through
    ``step8_train_snpe``. Notebook users may prefer calling the
    individual ``stepN_*`` functions one cell at a time; this function
    is for scripts that want the whole stage in one call.
    """
    n_sim = n_sim or config.N_SIM

    if verbose:
        print("\n" + "=" * 65)
        print("  Stage 1: single-round SNPE-C")
        print(f"  Params: {config.STAGE1_PARAMS}, n_sim={n_sim}/subject")
        print("=" * 65)

    # Step 7: parameter scaler (needed before simulation)
    param_scaler, prior_scaled = step7_fit_param_scaler(verbose=verbose)

    # Step 2 + 3: simulate and extract features
    theta_s, theta_r, fc_raw, fcd_raw = step2_simulate_train(
        train_subjects, subject_data, prior_scaled, param_scaler,
        n_sim=n_sim, apply_bw=apply_bw, verbose=verbose,
    )
    step3_summary_features(fc_raw, fcd_raw, verbose=verbose)

    # Step 4 + 5: scale and PCA
    step4_fit_feature_scalers(fc_raw, fcd_raw, verbose=verbose)
    pipeline, x_input = step5_fit_feature_pipeline(
        fc_raw, fcd_raw, verbose=verbose,
    )

    # Step 6: PCA diagnostic (pre-inference half of embedding quality)
    pca_diag = step6_pca_diagnostic(
        pipeline, fc_raw, fcd_raw, verbose=verbose,
    )

    # Step 8: train SNPE-C with jointly-optimized MLP embedding
    posterior, embedding_net = step8_train_snpe(
        theta_s, x_input, prior_scaled, verbose=verbose,
    )

    return {
        "posterior": posterior,
        "embedding_net": embedding_net,
        "theta_scaled": theta_s,
        "theta_raw": theta_r,
        "fc_raw": fc_raw,
        "fcd_raw": fcd_raw,
        "x_input": x_input,
        "param_scaler": param_scaler,
        "feature_pipeline": pipeline,
        "prior_scaled": prior_scaled,
        "pca_diagnostic": pca_diag,
    }


def _print_pca_diagnostic(pca_diag, header="PCA diagnostic"):
    d_fc = pca_diag["fc_pca"]
    print(f"\n  [{header}]")
    mark_fc = (
        "OK" if d_fc["pca_pass_evr"] and d_fc["pca_pass_recon"] else "FAIL"
    )
    print(
        f"    FC  PCA  : n={d_fc['n_components']}, "
        f"EVR={d_fc['explained_variance_sum']:.4f}, "
        f"recon={d_fc['recon_corr_train_mean']:.4f}  [{mark_fc}]"
    )
    d_fcd = pca_diag.get("fcd_pca", {})
    if not d_fcd:
        return
    if d_fcd.get("type", "").startswith("summary_stats"):
        # FCD is summary stats — no PCA diagnostic
        dims = ", ".join(d_fcd.get("dims", []))
        print(
            f"    FCD      : {d_fcd['n_components']} summary stats "
            f"({dims})"
        )
        if "train_mean" in d_fcd:
            print(
                f"               mean={d_fcd['train_mean']:.4f}, "
                f"std={d_fcd['train_std']:.4f}"
            )
    else:
        # Legacy: FCD PCA diagnostic
        mark_fcd = (
            "OK"
            if d_fcd.get("pca_pass_evr") and d_fcd.get("pca_pass_recon")
            else "FAIL"
        )
        print(
            f"    FCD PCA  : n={d_fcd['n_components']}, "
            f"EVR={d_fcd['explained_variance_sum']:.4f}, "
            f"recon={d_fcd['recon_corr_train_mean']:.4f}  [{mark_fcd}]"
        )


# ---------------------------------------------------------------------------
# Posterior sampling and shrinkage
# ---------------------------------------------------------------------------

def transform_observed(fc_obs_raw, fcd_obs_raw, feature_pipeline):
    """Apply the fitted FeaturePipeline to observed features."""
    return feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)


def infer_subject_raw(posterior, x_obs_input, param_scaler,
                      n_samples=None, verbose=False):
    """Sample the amortized posterior at one observation."""
    n_samples = n_samples or config.N_POSTERIOR
    x_t = torch.tensor(x_obs_input, dtype=torch.float32)
    samples_scaled = (
        posterior.sample((n_samples,), x=x_t, show_progress_bars=False)
        .cpu().numpy().astype(np.float32)
    )

    samples_raw = param_scaler.inverse_transform(samples_scaled)
    means_raw = samples_raw.mean(axis=0)
    stds_raw = samples_raw.std(axis=0)

    if verbose:
        for i, name in enumerate(param_scaler.param_names):
            print(
                f"    {name:6s} = {means_raw[i]:.4f} ± {stds_raw[i]:.4f}"
            )
    return samples_raw, means_raw, stds_raw, samples_scaled


def compute_shrinkage_scaled(samples_scaled):
    """Posterior shrinkage in the scaled space (prior_std = 2/sqrt(12))."""
    prior_std = 2.0 / np.sqrt(12.0)
    post_std = samples_scaled.std(axis=0)
    return np.clip(1.0 - post_std / prior_std, 0.0, 1.0)


def compute_shrinkage_raw(samples_raw, prior_low, prior_high):
    """Posterior shrinkage in the raw parameter space."""
    prior_low = np.asarray(prior_low)
    prior_high = np.asarray(prior_high)
    prior_std = (prior_high - prior_low) / np.sqrt(12.0)
    post_std = samples_raw.std(axis=0)
    return np.clip(1.0 - post_std / prior_std, 0.0, 1.0)


def posterior_correlation(samples):
    """Posterior correlation matrix; identity if dim < 2."""
    if samples.shape[1] < 2:
        return np.eye(samples.shape[1])
    return np.corrcoef(samples.T)


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------

def posterior_predictive_check(sid, subject_data, posterior,
                               fc_obs_raw, fcd_obs_raw,
                               param_scaler, feature_pipeline,
                               param_names, fixed_overrides=None,
                               n_predictive=None, apply_bw=True,
                               verbose=True):
    """Posterior predictive simulation + comparison to observed FC/FCD."""
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fcd_to_upper_tri,
        simulate_single,
    )

    n_predictive = n_predictive or config.N_PPC
    d = subject_data[sid]
    sc = d["sc"]
    dly = d["delays"]

    x_obs = feature_pipeline.transform(fc_obs_raw, fcd_obs_raw)
    samples_raw, means_raw, stds_raw, samples_scaled = infer_subject_raw(
        posterior, x_obs, param_scaler,
        n_samples=n_predictive, verbose=False,
    )

    fc_obs_full = d["fc"]
    iu = np.triu_indices(fc_obs_full.shape[0], k=1)

    fc_corrs, fc_rmses, fcd_rmses = [], [], []
    for i in range(min(n_predictive, len(samples_raw))):
        params = dict(fixed_overrides or {})
        for j, name in enumerate(param_names):
            params[name] = float(samples_raw[i, j])
        try:
            bolds = simulate_single(
                sc, params, n_repeat=1, delays=dly, apply_bw=apply_bw,
            )
            bold = bolds[0]
            fc_pred = compute_fc(bold)

            obs_vec = fc_obs_full[iu]
            pred_vec = fc_pred[iu]
            mask = np.isfinite(obs_vec) & np.isfinite(pred_vec)
            if (mask.sum() > 10
                    and obs_vec[mask].std() > 0
                    and pred_vec[mask].std() > 0):
                r = float(
                    np.corrcoef(obs_vec[mask], pred_vec[mask])[0, 1]
                )
                rmse = float(np.sqrt(
                    ((obs_vec[mask] - pred_vec[mask]) ** 2).mean()
                ))
                fc_corrs.append(r)
                fc_rmses.append(rmse)

            fcd_mat = compute_sim_fcd_matrix(bold)
            fcd_pred_vec = fcd_to_upper_tri(fcd_mat)
            fcd_rmses.append(float(np.sqrt(
                ((fcd_obs_raw - fcd_pred_vec) ** 2).mean()
            )))
        except Exception:
            continue

    out = {
        "samples_raw": samples_raw,
        "samples_scaled": samples_scaled,
        "means_raw": means_raw,
        "stds_raw": stds_raw,
        "fc_corr_mean": float(np.mean(fc_corrs)) if fc_corrs else 0.0,
        "fc_corr_std": float(np.std(fc_corrs)) if fc_corrs else 0.0,
        "fc_rmse_mean": float(np.mean(fc_rmses)) if fc_rmses else 1.0,
        "fcd_rmse_mean": float(np.mean(fcd_rmses)) if fcd_rmses else 1.0,
    }
    if verbose:
        print(
            f"    FC  corr = {out['fc_corr_mean']:.4f} ± "
            f"{out['fc_corr_std']:.4f}, "
            f"RMSE = {out['fc_rmse_mean']:.4f}"
        )
        print(f"    FCD RMSE = {out['fcd_rmse_mean']:.4f}")
    return out


# ---------------------------------------------------------------------------
# Simulation-based calibration
# ---------------------------------------------------------------------------

def simulation_based_calibration(posterior, prior_scaled, param_scaler,
                                 feature_pipeline, param_names,
                                 weights, delays,
                                 fixed_overrides=None,
                                 n_sbc=None, n_posterior=None,
                                 verbose=True):
    """Run SBC: prior sample -> simulate -> rank posterior samples."""
    from simulator import (
        compute_fc, compute_sim_fcd_matrix, fc_to_upper_tri,
        fcd_to_upper_tri, simulate_single,
    )

    n_sbc = n_sbc or config.N_SBC
    n_posterior = n_posterior or 1000

    if verbose:
        print(
            f"  SBC: {n_sbc} simulations, "
            f"{n_posterior} posterior samples each"
        )

    ranks = []
    t0 = time.time()
    for k in range(n_sbc):
        theta_scaled = prior_scaled.sample().cpu().numpy()
        theta_raw = param_scaler.inverse_transform(theta_scaled[None, :])[0]

        params = dict(fixed_overrides or {})
        for j, name in enumerate(param_names):
            params[name] = float(theta_raw[j])

        try:
            bolds = simulate_single(
                weights, params, n_repeat=1, delays=delays,
            )
            bold = bolds[0]
            fc_vec = fc_to_upper_tri(compute_fc(bold))
            fcd_vec = fcd_to_upper_tri(compute_sim_fcd_matrix(bold))

            x_obs = feature_pipeline.transform(fc_vec, fcd_vec)
            x_t = torch.tensor(x_obs, dtype=torch.float32)
            samples_scaled = (
                posterior.sample(
                    (n_posterior,), x=x_t, show_progress_bars=False,
                ).cpu().numpy()
            )
            rank = (samples_scaled < theta_scaled).sum(axis=0)
            ranks.append(rank)
        except Exception:
            continue

        # 25% 진행마다만 출력 (총 4회)
        if verbose and (k + 1) in {
            max(1, n_sbc // 4),
            max(1, n_sbc // 2),
            max(1, 3 * n_sbc // 4),
            n_sbc,
        }:
            pct = (k + 1) / n_sbc * 100
            _progress(
                f"SBC {k + 1}/{n_sbc} ({pct:.0f}%)  "
                f"({time.time() - t0:.1f}s)"
            )

    ranks = np.array(ranks)
    if verbose:
        print(f"    SBC done ({time.time() - t0:.1f}s)")
    return ranks


# ---------------------------------------------------------------------------
# Embedding probing
# ---------------------------------------------------------------------------

def evaluate_embedding_probing(embedding_net, theta_scaled, x_input,
                               param_names, n_samples=None,
                               verbose=True):
    """5-fold linear regression R² from embedding to scaled parameters."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    if verbose:
        print("\n  [Embedding probing - linear R²]")

    n_samples = n_samples or min(2000, len(theta_scaled))
    rng = np.random.RandomState(config.SEED)
    idx = rng.choice(len(theta_scaled), n_samples, replace=False)

    embedding_net.eval()
    has_params = any(
        p.requires_grad for p in embedding_net.parameters()
    )
    device = (
        next(embedding_net.parameters()).device if has_params else "cpu"
    )

    with torch.no_grad():
        x_t = torch.tensor(
            x_input[idx], dtype=torch.float32, device=device,
        )
        embs = embedding_net(x_t).cpu().numpy()

    theta_sub = theta_scaled[idx]
    probe = {}
    for i, name in enumerate(param_names):
        y = theta_sub[:, i]
        try:
            scores = cross_val_score(
                LinearRegression(), embs, y, cv=5, scoring="r2",
            )
            r2_mean = float(np.mean(scores))
            r2_std = float(np.std(scores))
        except Exception:
            r2_mean, r2_std = 0.0, 0.0
        probe[name] = {"r2_mean": r2_mean, "r2_std": r2_std}
        if r2_mean > 0.7:
            mark = "  OK"
        elif r2_mean > config.EMB_PROBE_R2_THRESHOLD:
            mark = "  WARN"
        else:
            mark = "  FAIL"
        if verbose:
            print(
                f"    {name:6s}: R² = {r2_mean:.4f} ± {r2_std:.4f}{mark}"
            )

    if verbose:
        mean_r2 = float(np.mean(
            [v["r2_mean"] for v in probe.values()]
        ))
        print(f"    Mean R²: {mean_r2:.4f}")

    probe["_pass"] = bool(
        np.mean([v["r2_mean"] for v in probe.values()])
        > config.EMB_PROBE_R2_THRESHOLD
    )
    return probe


# ---------------------------------------------------------------------------
# Stage 2 parameter selection
# ---------------------------------------------------------------------------

def select_difficult_params(shrinkage_per_param, param_names=None,
                            threshold=None):
    """Return Stage 1 parameters whose shrinkage falls below threshold.

    Deprecated by select_theta_bad() which uses sensitivity AND shrinkage.
    Kept for backward compatibility.
    """
    param_names = param_names or config.STAGE1_PARAMS
    threshold = threshold or config.DIFFICULT_SHRINKAGE
    return [
        name for name, s in zip(param_names, shrinkage_per_param)
        if s < threshold
    ]


def select_theta_bad(sensitivity_per_param, shrinkage_per_param,
                     param_names=None,
                     sens_threshold=0.5, shrinkage_threshold=0.2):
    """Select θ_bad = sensitivity high AND shrinkage low.

    "Output에 중요한데 1차에서 잘 식별되지 않은 parameter"가 2차 후보.

    Parameters
    ----------
    sensitivity_per_param : array-like  per-parameter sensitivity score
    shrinkage_per_param   : array-like  per-parameter shrinkage score
    param_names           : list[str] or None  (default Stage 1 params)
    sens_threshold        : sensitivity >= this counts as "high"
    shrinkage_threshold   : shrinkage   <  this counts as "low"

    Returns
    -------
    theta_bad : list[str]
    """
    param_names = param_names or config.STAGE1_PARAMS
    if len(sensitivity_per_param) != len(param_names):
        raise ValueError(
            f"sensitivity len {len(sensitivity_per_param)} != "
            f"n_params {len(param_names)}"
        )
    if len(shrinkage_per_param) != len(param_names):
        raise ValueError(
            f"shrinkage len {len(shrinkage_per_param)} != "
            f"n_params {len(param_names)}"
        )
    theta_bad = []
    for name, sens, shr in zip(
        param_names, sensitivity_per_param, shrinkage_per_param
    ):
        if sens >= sens_threshold and shr < shrinkage_threshold:
            theta_bad.append(name)
    return theta_bad


def build_stage2_param_set(shrinkage_per_param, param_names=None,
                           sensitivity_per_param=None,
                           sens_threshold=0.5,
                           shrinkage_threshold=None):
    """Build Stage 2 parameter set.

    Returns (stage2_params, nuisance_params) where::

        stage2_params = theta_bad + local E/I coupling
        theta_bad = (sensitivity high) AND (shrinkage low)   if sensitivity given
                  = (shrinkage low)                          if sensitivity is None
        nuisance  = Stage 1 params not in theta_bad
    """
    param_names = param_names or config.STAGE1_PARAMS
    shr_thresh = (
        shrinkage_threshold if shrinkage_threshold is not None
        else config.DIFFICULT_SHRINKAGE
    )
    if sensitivity_per_param is not None:
        difficult = select_theta_bad(
            sensitivity_per_param, shrinkage_per_param,
            param_names=param_names,
            sens_threshold=sens_threshold,
            shrinkage_threshold=shr_thresh,
        )
    else:
        difficult = select_difficult_params(
            shrinkage_per_param, param_names=param_names,
            threshold=shr_thresh,
        )
    nuisance = [p for p in param_names if p not in difficult]
    c_params = list(config.LOCAL_EI_PARAMS)   # ["c_ee","c_ei","c_ie","c_ii"]
    stage2_params = difficult + c_params

    print("\n  Stage 2 configuration:")
    print(
        f"    theta_bad (high sens & low shrinkage): {difficult}"
    )
    print(f"    Nuisance (fix or sample from Stage 1)     : {nuisance}")
    print(f"    c-params to add (LOCAL_EI_PARAMS)         : {c_params}")
    print(f"    => Stage 2 inference targets              : {stage2_params}")
    return stage2_params, nuisance


# ---------------------------------------------------------------------------
# Stage 2 data collection and inference
# ---------------------------------------------------------------------------

def collect_stage2_data(train_subjects, subject_data,
                        stage2_prior_scaled,
                        stage2_params, nuisance_params,
                        stage2_param_scaler,
                        stage1_posterior, stage1_param_scaler,
                        stage1_feature_pipeline, x_obs_dict_s1,
                        n_sim=None, nuisance_method=None,
                        apply_bw=True, verbose=True):
    """Collect Stage 2 training simulations with nuisance handling."""
    from simulator import simulate_gpu_batch, worker_extract
    import cupy as cp

    n_sim = n_sim or config.N_SIM_S2
    nuisance_method = nuisance_method or config.NUISANCE_METHOD

    if verbose:
        print(f"\n  Stage 2 data collection ({nuisance_method})")
        print(f"    Targets : {stage2_params}")
        print(f"    Nuisance: {nuisance_params}")

    all_param_names = stage2_params + nuisance_params
    all_theta_s, all_theta_r, all_fc, all_fcd = [], [], [], []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=config.N_CPU) as executor:
        for s_idx, sid in enumerate(train_subjects):
            if verbose:
                print(
                    f"\n  [{s_idx + 1}/{len(train_subjects)}] {sid}"
                )
            d = subject_data[sid]
            sc, dly = d["sc"], d["delays"]

            theta_s2_s = (
                stage2_prior_scaled.sample((n_sim,))
                .cpu().numpy().astype(np.float32)
            )
            theta_s2_r = stage2_param_scaler.inverse_transform(theta_s2_s)

            nuis_raw_per_sim = _build_nuisance_array(
                nuisance_params, n_sim, sid,
                stage1_posterior, stage1_param_scaler,
                x_obs_dict_s1, nuisance_method,
            )
            theta_combined_raw = np.concatenate(
                [theta_s2_r, nuis_raw_per_sim], axis=1,
            ).astype(np.float32)

            batch_sz = config.GPU_BATCH
            n_batches = (n_sim + batch_sz - 1) // batch_sz
            future_queue = []
            t_sub = time.time()

            for b_idx in range(n_batches):
                start = b_idx * batch_sz
                end = min(start + batch_sz, n_sim)
                chunk_raw = theta_combined_raw[start:end]
                chunk_s2_s = theta_s2_s[start:end]
                chunk_s2_r = theta_s2_r[start:end]

                try:
                    bolds = simulate_gpu_batch(
                        sc, chunk_raw,
                        param_names=all_param_names,
                        fixed_overrides=None, delays=dly,
                        apply_bw=apply_bw,
                    )
                except Exception as e:
                    print(f"  batch {b_idx} failed: {e}")
                    continue

                future = executor.map(worker_extract, bolds, chunksize=16)
                future_queue.append((chunk_s2_s, chunk_s2_r, future))

                while len(future_queue) >= 2:
                    _drain_one_future(
                        future_queue.pop(0),
                        all_theta_s, all_theta_r, all_fc, all_fcd,
                    )

                if verbose:
                    elapsed = time.time() - t_sub
                    n_done = len(all_theta_s)
                    total = n_subj * n_sim
                    pct = n_done / max(total, 1) * 100
                    _progress(
                        f"batch {b_idx + 1}/{n_batches}  "
                        f"sim {end}/{n_sim}  "
                        f"total {n_done}/{total} ({pct:.1f}%)  "
                        f"({elapsed:.1f}s)"
                    )

            for queued in future_queue:
                _drain_one_future(
                    queued,
                    all_theta_s, all_theta_r, all_fc, all_fcd,
                )

            cp.get_default_memory_pool().free_all_blocks()
            if verbose:
                elapsed = time.time() - t_sub
                print(
                    f"    done: {len(all_theta_s)}  ({elapsed:.1f}s)"
                )
                sys.stdout.flush()

    theta_s = np.array(all_theta_s, dtype=np.float32)
    theta_r = np.array(all_theta_r, dtype=np.float32)
    fc_raw = np.array(all_fc, dtype=np.float32)
    fcd_raw = np.array(all_fcd, dtype=np.float32)

    if verbose:
        print(
            f"\n  Stage 2 collected: theta={theta_s.shape}, "
            f"fc={fc_raw.shape}  ({time.time() - t0:.1f}s)"
        )
    return theta_s, theta_r, fc_raw, fcd_raw


def _build_nuisance_array(nuisance_params, n_sim, sid,
                          stage1_posterior, stage1_param_scaler,
                          x_obs_dict_s1, nuisance_method):
    """Return (n_sim, n_nuisance) raw nuisance values for Stage 2 sims."""
    if not nuisance_params:
        return np.empty((n_sim, 0), dtype=np.float32)

    s1_names = config.STAGE1_PARAMS
    x_obs_t = torch.tensor(x_obs_dict_s1[sid], dtype=torch.float32)

    if nuisance_method == "fix_mean":
        samples_scaled = (
            stage1_posterior
            .sample((1000,), x=x_obs_t, show_progress_bars=False)
            .cpu().numpy()
        )
        samples_raw = stage1_param_scaler.inverse_transform(samples_scaled)
        means_raw = samples_raw.mean(axis=0)
        idx_nuis = [s1_names.index(p) for p in nuisance_params]
        return np.tile(means_raw[idx_nuis][None, :], (n_sim, 1))

    # posterior_sample
    samples_scaled = (
        stage1_posterior
        .sample((n_sim,), x=x_obs_t, show_progress_bars=False)
        .cpu().numpy()
    )
    samples_raw = stage1_param_scaler.inverse_transform(samples_scaled)
    idx_nuis = [s1_names.index(p) for p in nuisance_params]
    return samples_raw[:, idx_nuis]


def run_stage2_snpe(train_subjects, subject_data, stage1_result,
                    val_shrinkage, n_sim=None, apply_bw=True,
                    verbose=True):
    """End-to-end Stage 2 inference (steps 10 - 11)."""
    n_sim = n_sim or config.N_SIM_S2

    stage2_params, nuisance_params = build_stage2_param_set(val_shrinkage)
    s2_param_scaler = make_stage2_param_scaler(stage2_params)
    s2_prior_scaled = make_scaled_prior(len(stage2_params))

    s1_posterior = stage1_result["posterior"]
    s1_param_scaler = stage1_result["param_scaler"]
    s1_pipeline = stage1_result["feature_pipeline"]

    from simulator import extract_observed_features
    x_obs_s1 = {}
    for sid in train_subjects:
        fc_obs, fcd_obs = extract_observed_features(subject_data[sid])
        x_obs_s1[sid] = s1_pipeline.transform(fc_obs, fcd_obs)

    theta_s, theta_r, fc_raw, fcd_raw = collect_stage2_data(
        train_subjects, subject_data, s2_prior_scaled,
        stage2_params, nuisance_params,
        stage2_param_scaler=s2_param_scaler,
        stage1_posterior=s1_posterior,
        stage1_param_scaler=s1_param_scaler,
        stage1_feature_pipeline=s1_pipeline,
        x_obs_dict_s1=x_obs_s1,
        n_sim=n_sim,
        nuisance_method=config.NUISANCE_METHOD,
        apply_bw=apply_bw, verbose=verbose,
    )

    s2_pipeline = FeaturePipeline()
    s2_pipeline.fit(fc_raw, fcd_raw)
    x_input = s2_pipeline.transform(fc_raw, fcd_raw)

    pca_diag = s2_pipeline.diagnostic(fc_raw, fcd_raw)
    if verbose:
        _print_pca_diagnostic(pca_diag, header="Stage 2 PCA diagnostic")

    posterior, embedding_net = train_snpe(
        theta_s, x_input, s2_prior_scaled,
        embedding_net=None, proposal=None, verbose=verbose,
    )

    return {
        "posterior": posterior,
        "embedding_net": embedding_net,
        "stage2_params": stage2_params,
        "nuisance_params": nuisance_params,
        "param_scaler": s2_param_scaler,
        "prior_scaled": s2_prior_scaled,
        "feature_pipeline": s2_pipeline,
        "theta_scaled": theta_s,
        "theta_raw": theta_r,
        "fc_raw": fc_raw,
        "fcd_raw": fcd_raw,
        "x_input": x_input,
        "pca_diagnostic": pca_diag,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(path, **kwargs):
    """Pickle scalers, pipelines, and metadata to disk."""
    with open(path, "wb") as f:
        pickle.dump(kwargs, f)


def load_artifacts(path):
    """Load a previously saved artifacts dict."""
    with open(path, "rb") as f:
        return pickle.load(f)
