"""Phase-2 attention × gradient feature selection.

Given a Phase-1 posterior backed by a ``RegionTransformerEmbedding``
network, score every FC upper-triangle entry by

    importance[p, i, j] = |grad(E[A_p | FC]) w.r.t. FC_{ij}|
                          x mean_layers( attention(i <-> j) )

Take the per-parameter top-k union to obtain the reduced FC index set
used to re-train SBI in Phase 3.

Public API
----------
- ``AttentionSelector(posterior, embedding_net, param_names, ...)``
  - ``get_attention_maps(fc_obs)``  → (n_layers, N, N)
  - ``get_gradients(fc_obs, n_samples)`` → (n_params, n_features)
  - ``get_importance(fc_obs, n_samples)`` → (n_params, N, N)
  - ``select_top_k(importance, k)`` → flat indices (≤ k unique union)

Notes
-----
Attention weights live on the (N+1, N+1) token grid because of the CLS
token. We slice off the CLS row/column and symmetrize before scoring.
"""
import numpy as np

import config


class AttentionSelector:
    """Attention × gradient feature scorer."""

    def __init__(self, posterior, embedding_net, param_names,
                 n_regions=None, device=None):
        import torch
        self._torch = torch
        self.posterior = posterior
        self.embedding = embedding_net
        self.param_names = list(param_names)
        self.n_params = len(self.param_names)
        self.n_regions = int(n_regions or config.N_REGIONS)
        self.device = device or str(
            next(embedding_net.parameters()).device
        )

        rr, cc = np.triu_indices(self.n_regions, k=1)
        self.triu_rows = rr
        self.triu_cols = cc
        self.fc_dim = rr.shape[0]

    def _vectorize(self, fc_obs):
        """Return flat upper-tri (fc_dim,) the embedding expects.

        Accepts a full ``(N, N)`` FC matrix, a flat ``(fc_dim,)`` vector,
        or a batched ``(1, fc_dim)`` row; always returns ``(fc_dim,)``.
        """
        arr = np.asarray(fc_obs, dtype=np.float32)
        if arr.ndim == 2 and arr.shape == (self.n_regions, self.n_regions):
            return arr[self.triu_rows, self.triu_cols]
        flat = arr.ravel()
        if flat.shape[0] != self.fc_dim:
            raise ValueError(
                f"fc_obs has {flat.shape[0]} entries; expected a "
                f"({self.n_regions},{self.n_regions}) matrix or a flat "
                f"vector of length {self.fc_dim}."
            )
        return flat

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_cls(attn):
        """Drop the CLS row/column from a (B, T, T) attention tensor."""
        # tokens are [CLS, r0, r1, ...] (see RegionTransformerEmbedding)
        return attn[:, 1:, 1:]

    def get_attention_maps(self, fc_obs):
        """Run a single forward pass and return (n_layers, N, N) attentions."""
        torch = self._torch
        self.embedding.eval()
        x = torch.tensor(
            self._vectorize(fc_obs)[None],
            device=self.device,
        )
        with torch.no_grad():
            _ = self.embedding(x)
        maps = []
        for layer in getattr(self.embedding, "layers", []):
            a = layer.attn
            if a is None:
                continue
            a = self._strip_cls(a)             # (B, N, N)
            maps.append(a.mean(dim=0).detach().cpu().numpy())  # (N, N)
        if not maps:
            raise RuntimeError(
                "Embedding network exposes no attention weights — only "
                "RegionTransformerEmbedding-style nets are supported."
            )
        return np.stack(maps, axis=0)          # (n_layers, N, N)

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def get_gradients(self, fc_obs, n_samples=200):
        """Per-parameter |∂E[A_p | FC] / ∂FC_k| over an MC posterior sample.

        Uses ``posterior.sample`` conditioned on ``fc_obs`` to estimate
        the posterior mean, then differentiates through the embedding
        + flow log-prob path. When the posterior does not support
        gradient pass-through we fall back to a finite-difference
        estimate around ``fc_obs``.
        """
        torch = self._torch
        fc_t = torch.tensor(
            self._vectorize(fc_obs)[None],
            device=self.device,
        )
        fc_t.requires_grad_(True)

        # Estimate posterior mean as a differentiable function of fc_t by
        # running the embedding manually and asking the underlying flow
        # for an expectation when available. Sbi's API is heterogeneous,
        # so we MC-sample and average — and rely on the embedding for the
        # gradient path. We model the gradient of the posterior mean as
        #   dE[A_p|x] / dx ≈ J_emb^T · g_p
        # where g_p is the average gradient of the flow log-prob with
        # respect to its embedded input. In practice this reduces to
        # differentiating the parameter estimate w.r.t. the embedding
        # output, which is what we approximate below.
        try:
            theta = self.posterior.sample(
                (int(n_samples),), x=fc_t, show_progress_bars=False,
                reject_outside_prior=False,  # selection heuristic: speed first
            )
            theta_mean = theta.mean(dim=0)     # (n_params,)
        except Exception:
            theta_mean = None

        grads = np.zeros((self.n_params, self.fc_dim), dtype=np.float32)
        if theta_mean is not None and theta_mean.requires_grad:
            for p in range(self.n_params):
                fc_t.grad = None
                g = torch.autograd.grad(
                    theta_mean[p], fc_t, retain_graph=True,
                    allow_unused=True,
                )[0]
                if g is None:
                    continue
                grads[p] = g.detach().cpu().numpy().ravel()
        else:
            # Finite-difference fallback around fc_obs.
            eps = 1e-3
            fc_base = self._vectorize(fc_obs)
            with torch.no_grad():
                base = (
                    self.posterior.sample(
                        (int(n_samples),),
                        x=torch.tensor(fc_base[None], device=self.device),
                        show_progress_bars=False,
                        reject_outside_prior=False,  # heuristic: speed first
                    )
                    .mean(dim=0).detach().cpu().numpy()
                )
            for k in range(self.fc_dim):
                fc_pert = fc_base.copy()
                fc_pert[k] += eps
                with torch.no_grad():
                    pert = (
                        self.posterior.sample(
                            (int(n_samples),),
                            x=torch.tensor(
                                fc_pert[None], device=self.device,
                            ),
                            show_progress_bars=False,
                            reject_outside_prior=False,  # heuristic: speed
                        )
                        .mean(dim=0).detach().cpu().numpy()
                    )
                grads[:, k] = (pert - base) / eps

        return np.abs(grads)                   # (n_params, fc_dim)

    # ------------------------------------------------------------------
    # Importance + selection
    # ------------------------------------------------------------------

    def get_importance(self, fc_obs, n_samples=200):
        """Combine attention × gradient into per-parameter (N, N) maps."""
        attn = self.get_attention_maps(fc_obs)       # (n_layers, N, N)
        attn = attn.mean(axis=0)                     # (N, N)
        attn = 0.5 * (attn + attn.T)                 # symmetrize
        grads = self.get_gradients(fc_obs, n_samples=n_samples)
        n_params = grads.shape[0]
        N = self.n_regions
        importance = np.zeros((n_params, N, N), dtype=np.float32)
        rows = self.triu_rows
        cols = self.triu_cols

        # edge-level attention (upper-tri), L1 normalized
        eps = 1e-8
        attn_edge = attn[rows, cols]                       # (fc_dim,)
        attn_edge = attn_edge / (attn_edge.sum() + eps)    # L1 normalize

        for p in range(n_params):
            g = np.abs(grads[p])                           # (fc_dim,)
            g = g / (g.sum() + eps)                        # L1 normalize
            imp_edge = g * attn_edge                       # (fc_dim,)
            mat = np.zeros((N, N), dtype=np.float32)
            mat[rows, cols] = imp_edge
            mat[cols, rows] = imp_edge
            importance[p] = mat
        return importance

    def select_top_k(self, importance, k=None):
        """Per-parameter top-k upper-tri entries, returned as a union."""
        k = int(k or config.SELECTION_K)
        rows = self.triu_rows
        cols = self.triu_cols
        scores = importance[:, rows, cols]           # (n_params, fc_dim)
        per_param_k = max(1, k // max(1, scores.shape[0]))
        chosen = set()
        for p in range(scores.shape[0]):
            top = np.argpartition(-scores[p], per_param_k)[:per_param_k]
            chosen.update(top.tolist())
        # If the union is too small, top-up with the globally largest
        # importance entries until we reach k.
        if len(chosen) < k:
            agg = scores.max(axis=0)
            order = np.argsort(-agg)
            for idx in order:
                if len(chosen) >= k:
                    break
                chosen.add(int(idx))
        indices = np.array(sorted(chosen), dtype=np.int64)
        if indices.shape[0] > k:
            agg = scores.max(axis=0)
            sub_order = np.argsort(-agg[indices])
            indices = np.sort(indices[sub_order[:k]])
        return indices
