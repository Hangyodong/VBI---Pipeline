"""Embedding networks for SBI/SNPE-C.

Public API
----------
- RegionTransformerEmbedding(input_dim, out_dim, n_regions, ...)
      Transformer over per-region tokens reconstructed from a flattened
      FC upper-triangle vector (or a subset of upper-tri entries when
      ``triu_indices`` is provided). Exposes per-layer attention weights
      via the ``last_attn`` attribute so Phase-2 feature selection
      (``inference.attention_selection``) can read them off.
- make_embedding_net(embed_type, input_dim, out_dim=None, ...)
      Factory dispatching on ``embed_type`` ("region_transformer",
      "identity"). Returns the corresponding ``torch.nn.Module``.

The torch dependency is guarded so this module can be imported on
torch-less systems (CPU-only environments without ML stack).
"""
import math

import numpy as np

import config

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    _EMBED_BASE = nn.Module
else:
    _EMBED_BASE = object


# ---------------------------------------------------------------------------
# Transformer encoder layer that captures attention weights
# ---------------------------------------------------------------------------

class _AttnCapturingEncoderLayer(_EMBED_BASE):
    """Minimal Transformer encoder block that stores its attention weights.

    PyTorch's ``nn.TransformerEncoderLayer`` only exposes attention
    weights via the ``need_weights`` kwarg on the underlying
    ``nn.MultiheadAttention``. We replicate the pre-LN block with
    ``need_weights=True`` so AttentionSelector can read ``self.attn``.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required")
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.attn = None   # last attention weights, shape (B, T, T)

    def forward(self, x):
        # Pre-LN residual self-attention with attention capture.
        h = self.norm1(x)
        attn_out, attn_w = self.self_attn(
            h, h, h, need_weights=True, average_attn_weights=True,
        )
        self.attn = attn_w  # (B, T, T) averaged over heads
        x = x + self.dropout1(attn_out)
        h = self.norm2(x)
        h = self.linear2(self.dropout_ff(F.relu(self.linear1(h))))
        x = x + self.dropout2(h)
        return x


# ---------------------------------------------------------------------------
# RegionTransformer embedding
# ---------------------------------------------------------------------------

class RegionTransformerEmbedding(_EMBED_BASE):
    """FC vector → per-region tokens → Transformer → CLS → out_dim.

    Parameters
    ----------
    input_dim : int
        Length of the FC feature vector fed in at forward time. Equal to
        ``len(triu_indices)`` when explicit indices are passed; otherwise
        the canonical ``n_regions * (n_regions - 1) // 2`` upper-triangle
        length.
    out_dim : int
        Embedding output dimensionality.
    n_regions : int
        Atlas size N (number of region tokens).
    d_model : int
        Transformer hidden width per token.
    n_heads, n_layers : int
        Transformer width / depth.
    sc_matrix : (N, N) numpy array or None
        Optional structural connectivity matrix; used as a positional
        encoding via a linear projection of each region's SC row.
    triu_indices : (k, 2) or 2-tuple of (k,) numpy arrays, optional
        Explicit (row, col) indices in the (N, N) FC matrix that the
        flattened input represents. Required when running on a
        Phase-3 reduced feature set. When ``None``, falls back to the
        full upper triangle.
    """

    def __init__(self, input_dim, out_dim=None, n_regions=None,
                 d_model=None, n_heads=None, n_layers=None,
                 sc_matrix=None, triu_indices=None,
                 use_mask=False, sc_table=None, per_subject_sc=False):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for RegionTransformerEmbedding")
        super().__init__()

        self.n_regions = int(n_regions or config.N_REGIONS)
        self.out_dim   = int(out_dim   or config.EMBED_DIM)
        self.d_model   = int(d_model   or config.REGION_TRANSFORMER_D_MODEL)
        self.n_heads   = int(n_heads   or config.REGION_TRANSFORMER_HEADS)
        self.n_layers  = int(n_layers  or config.REGION_TRANSFORMER_LAYERS)

        # Resolve (row, col) indices the flat input represents.
        if triu_indices is None:
            rr, cc = np.triu_indices(self.n_regions, k=1)
        elif isinstance(triu_indices, tuple) and len(triu_indices) == 2:
            rr = np.asarray(triu_indices[0], dtype=np.int64)
            cc = np.asarray(triu_indices[1], dtype=np.int64)
        else:
            triu_indices = np.asarray(triu_indices, dtype=np.int64)
            if triu_indices.ndim != 2 or triu_indices.shape[1] != 2:
                raise ValueError(
                    "triu_indices must be a (k,2) array or a 2-tuple of (k,)"
                )
            rr = triu_indices[:, 0]
            cc = triu_indices[:, 1]
        if rr.shape[0] != input_dim:
            raise ValueError(
                f"input_dim={input_dim} does not match the number of "
                f"triu indices ({rr.shape[0]})."
            )
        self.register_buffer("triu_rows", torch.tensor(rr, dtype=torch.long))
        self.register_buffer("triu_cols", torch.tensor(cc, dtype=torch.long))

        self.input_dim = int(input_dim)

        # When use_mask is set, each region token is [value_row | mask_row]
        # (2N wide) so the Transformer can distinguish a true-zero edge from
        # an absent (unselected) edge in the Phase-4 sparse FC input.
        self.use_mask = bool(use_mask)
        if self.use_mask:
            token_input_dim = 2 * self.n_regions   # value + mask
        else:
            token_input_dim = self.n_regions

        self.token_proj = nn.Linear(token_input_dim, self.d_model)

        # ── Per-subject SC conditioning (④) ──────────────────────────────
        # When per_subject_sc is on, the SC positional encoding is selected
        # PER SAMPLE instead of a single fixed matrix. The forward input x then
        # carries a leading subject-index column: x = [subj_idx | fc_vec].
        #   - train: subj_idx -> row of the registered sc_table (one per subject)
        #   - eval : set self.override_sc to the test subject's SC (broadcast to
        #            the whole batch); subj_idx is then ignored.
        # This gives true amortization over subjects with different SC, instead
        # of a constant bias. Storage is cheap: sc_table is (n_subj, N, N).
        self.per_subject_sc = bool(per_subject_sc)
        self.override_sc = None     # set at eval time to a (N, N) tensor
        if self.per_subject_sc:
            if sc_table is None:
                raise ValueError("per_subject_sc=True requires sc_table (n_subj, N, N)")
            sct = torch.tensor(np.asarray(sc_table, dtype=np.float32), dtype=torch.float32)
            if sct.ndim != 3 or sct.shape[1:] != (self.n_regions, self.n_regions):
                raise ValueError(
                    f"sc_table shape {tuple(sct.shape)} != (n_subj, "
                    f"{self.n_regions}, {self.n_regions})"
                )
            self.register_buffer("sc_table", sct)
            self.sc_matrix = None
            self.sc_proj = nn.Linear(self.n_regions, self.d_model)
        elif sc_matrix is not None:
            sc_t = torch.tensor(
                np.asarray(sc_matrix, dtype=np.float32),
                dtype=torch.float32,
            )
            if sc_t.shape != (self.n_regions, self.n_regions):
                raise ValueError(
                    f"sc_matrix shape {tuple(sc_t.shape)} != "
                    f"({self.n_regions}, {self.n_regions})"
                )
            self.register_buffer("sc_matrix", sc_t)
            self.sc_proj = nn.Linear(self.n_regions, self.d_model)
        else:
            self.sc_matrix = None
            self.sc_proj = None

        self.cls = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        self.layers = nn.ModuleList([
            _AttnCapturingEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dim_feedforward=self.d_model * 4,
                dropout=0.1,
            )
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.out_dim)

    @property
    def last_attn(self):
        """List of last-forward attention weights per layer (B, T, T)."""
        return [layer.attn for layer in self.layers]

    def _scatter_to_matrix(self, x):
        """Reconstruct a symmetric (N, N) matrix from flattened upper-tri."""
        B = x.shape[0]
        N = self.n_regions
        mat = x.new_zeros(B, N, N)
        rows = self.triu_rows
        cols = self.triu_cols
        # in-place index assignment is fine for forward; we don't need
        # higher-order gradients through the scatter.
        mat[:, rows, cols] = x
        mat[:, cols, rows] = x
        return mat

    def _scatter_to_matrix_with_mask(self, x):
        """Scatter upper-tri values to (B,N,N) + binary mask (B,N,N)."""
        B = x.shape[0]
        rows, cols = self.triu_rows, self.triu_cols
        N = self.n_regions
        val = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
        msk = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
        val[:, rows, cols] = x
        val[:, cols, rows] = x
        sel = (x.abs() > 0).float()
        msk[:, rows, cols] = sel
        msk[:, cols, rows] = sel
        return val, msk

    def forward(self, x):
        # x: (B, input_dim)  — or (B, 1 + input_dim) when per_subject_sc
        # (leading column = subject index into sc_table).
        B = x.shape[0]
        subj_idx = None
        if self.per_subject_sc:
            subj_idx = x[:, 0].long()                # (B,)
            x = x[:, 1:]                             # strip index -> (B, input_dim)
        if self.use_mask:
            val_mat, msk_mat = self._scatter_to_matrix_with_mask(x)
            combined = torch.cat([val_mat, msk_mat], dim=2)  # (B,N,2N)
            tok = self.token_proj(combined)                   # (B,N,d_model)
        else:
            mat = self._scatter_to_matrix(x)             # (B, N, N)
            tok = self.token_proj(mat)                   # (B, N, d_model)
        if self.per_subject_sc:
            if self.override_sc is not None:
                # eval: one subject for the whole batch
                sc_sel = self.override_sc.to(tok.dtype).to(tok.device)
                sc_sel = sc_sel.unsqueeze(0).expand(B, -1, -1)   # (B,N,N)
            else:
                sc_sel = self.sc_table[subj_idx]                 # (B,N,N) per-sample
            tok = tok + self.sc_proj(sc_sel)                     # (B,N,d_model)
        elif self.sc_proj is not None:
            sc_pe = self.sc_proj(self.sc_matrix)     # (N, d_model)
            tok = tok + sc_pe.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)             # (B, 1, d_model)
        h = torch.cat([cls, tok], dim=1)             # (B, N+1, d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h[:, 0, :])                 # (B, out_dim)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_embedding_net(embed_type, input_dim, out_dim=None,
                       sc_matrix=None, triu_indices=None,
                       use_mask=False, **kw):
    """Factory for SBI embedding networks.

    embed_type
        - "region_transformer" : ``RegionTransformerEmbedding`` (default)
        - "identity"           : ``nn.Identity``
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for make_embedding_net")

    et = (embed_type or "region_transformer").lower()
    out_dim = out_dim or config.EMBED_DIM

    if et == "region_transformer":
        return RegionTransformerEmbedding(
            input_dim=input_dim,
            out_dim=out_dim,
            sc_matrix=sc_matrix,
            triu_indices=triu_indices,
            use_mask=use_mask,
            **kw,
        )
    if et == "multichannel_matrix":
        sc_table = kw.pop("sc_table", None)
        if sc_table is None:
            raise ValueError(
                "embed_type='multichannel_matrix' requires sc_table "
                "(n_subj, C, N, N)"
            )
        return MultiChannelMatrixEmbedding(
            fc_input_dim=input_dim,
            sc_table=sc_table,
            out_dim=out_dim,
            **kw,
        )
    if et == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown embed_type: {embed_type!r}")


# ---------------------------------------------------------------------------
# Multi-channel matrix embedding (SC-conditioned amortized NPE)
# ---------------------------------------------------------------------------

class MultiChannelMatrixEmbedding(_EMBED_BASE):
    """Conditioning encoder over a stack of ROIxROI matrices.

    Encodes a multi-channel matrix stack ``[FC, sc_weight, delay, sc_mask]``
    into a fixed-dim embedding for an SC-conditioned amortized NPE.

    The only per-simulation varying channel is **FC**, fed in as a flat
    upper-triangle vector. The subject-constant channels (sc_weight, delay,
    sc_mask, ...) are NEVER materialized per simulation: they are looked up
    from a registered ``sc_table`` by a LEADING subject-index column in the
    forward input ``x``. This keeps the training tensor memory-cheap (each
    sim only stores its FC vector + an integer subject index).

    Architecture (mirrors RegionTransformerEmbedding)
    -------------------------------------------------
    FC upper-tri vector → scatter to (B,N,N) [+ binary mask → (B,N,2N)]
    → ``token_proj`` Linear → (B,N,d_model) FC tokens.
    Each SC channel ``c`` → ``sc_proj[c]`` Linear((N)→d_model), added to the
    region tokens. A CLS token is prepended, the stack is run through
    ``_AttnCapturingEncoderLayer`` blocks, layer-normed, and the CLS row is
    projected to ``out_dim`` by ``head``.

    Parameters
    ----------
    fc_input_dim : int
        FC upper-tri vector length (e.g. 64620 for N=360).
    sc_table : (n_subj, C, N, N) array-like
        Per-subject SC channel stack. Registered as a float32 buffer.
        ``C = n_sc_channels`` (inferred from ``sc_table.shape[1]`` by
        default). Channel order is the caller's responsibility, e.g.
        ``[sc_weight, delay, sc_mask]``.
    out_dim, d_model, n_heads, n_layers : int, optional
        Defaults pulled from ``config`` like RegionTransformerEmbedding.
    n_regions : int, optional
        Atlas size N. Defaults to ``config.N_REGIONS``. Must match the
        ``sc_table`` spatial dims and the FC upper-tri length.
    n_sc_channels : int, optional
        Number of SC channels C. Defaults to ``sc_table.shape[1]``.
    use_fc_mask : bool
        When True (default) FC tokens are ``[fc_value_row | fc_mask_row]``
        (2N wide) like the existing ``use_mask`` path, so the Transformer
        can tell a true-zero edge from an absent one.

    Eval-time override
    ------------------
    Set ``self.override_sc_channels`` to a ``(C, N, N)`` tensor (the test
    subject's SC channels). When set, the leading subject index is ignored
    and that stack is broadcast over the whole batch.
    """

    def __init__(self, fc_input_dim, sc_table, out_dim=None, n_regions=None,
                 d_model=None, n_heads=None, n_layers=None,
                 n_sc_channels=None, use_fc_mask=True):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for MultiChannelMatrixEmbedding")
        super().__init__()

        self.n_regions = int(n_regions or config.N_REGIONS)
        self.out_dim   = int(out_dim   or config.EMBED_DIM)
        self.d_model   = int(d_model   or config.REGION_TRANSFORMER_D_MODEL)
        self.n_heads   = int(n_heads   or config.REGION_TRANSFORMER_HEADS)
        self.n_layers  = int(n_layers  or config.REGION_TRANSFORMER_LAYERS)

        # ── SC table (subject-constant channels) ─────────────────────────
        sct = torch.tensor(np.asarray(sc_table, dtype=np.float32),
                           dtype=torch.float32)
        if sct.ndim != 4:
            raise ValueError(
                f"sc_table must be (n_subj, C, N, N); got ndim={sct.ndim} "
                f"shape={tuple(sct.shape)}"
            )
        C = int(n_sc_channels) if n_sc_channels is not None else int(sct.shape[1])
        if sct.shape[1] != C:
            raise ValueError(
                f"n_sc_channels={C} != sc_table.shape[1]={sct.shape[1]}"
            )
        if sct.shape[2:] != (self.n_regions, self.n_regions):
            raise ValueError(
                f"sc_table shape {tuple(sct.shape)} spatial dims "
                f"!= ({self.n_regions}, {self.n_regions})"
            )
        self.n_sc_channels = C
        self.register_buffer("sc_table", sct)        # (n_subj, C, N, N) float32

        # ── FC upper-tri indices ─────────────────────────────────────────
        rr, cc = np.triu_indices(self.n_regions, k=1)
        if rr.shape[0] != int(fc_input_dim):
            raise ValueError(
                f"fc_input_dim={fc_input_dim} does not match the number of "
                f"upper-tri indices for n_regions={self.n_regions} "
                f"({rr.shape[0]})."
            )
        self.fc_input_dim = int(fc_input_dim)
        self.register_buffer("triu_rows", torch.tensor(rr, dtype=torch.long))
        self.register_buffer("triu_cols", torch.tensor(cc, dtype=torch.long))

        # ── Token / channel projections ──────────────────────────────────
        self.use_fc_mask = bool(use_fc_mask)
        if self.use_fc_mask:
            token_input_dim = 2 * self.n_regions   # value + mask
        else:
            token_input_dim = self.n_regions
        self.token_proj = nn.Linear(token_input_dim, self.d_model)
        self.sc_proj = nn.ModuleList([
            nn.Linear(self.n_regions, self.d_model) for _ in range(C)
        ])

        self.cls = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.layers = nn.ModuleList([
            _AttnCapturingEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dim_feedforward=self.d_model * 4,
                dropout=0.1,
            )
            for _ in range(self.n_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.out_dim)

        # Eval-time override: (C, N, N) tensor for one test subject.
        self.override_sc_channels = None

    @property
    def last_attn(self):
        """List of last-forward attention weights per layer (B, T, T)."""
        return [layer.attn for layer in self.layers]

    def _scatter_to_matrix(self, x):
        """Reconstruct a symmetric (B,N,N) matrix from flattened upper-tri."""
        B = x.shape[0]
        N = self.n_regions
        mat = x.new_zeros(B, N, N)
        rows = self.triu_rows
        cols = self.triu_cols
        mat[:, rows, cols] = x
        mat[:, cols, rows] = x
        return mat

    def _scatter_to_matrix_with_mask(self, x):
        """Scatter upper-tri values to (B,N,N) + binary mask (B,N,N)."""
        B = x.shape[0]
        rows, cols = self.triu_rows, self.triu_cols
        N = self.n_regions
        val = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
        msk = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
        val[:, rows, cols] = x
        val[:, cols, rows] = x
        sel = (x.abs() > 0).float()
        msk[:, rows, cols] = sel
        msk[:, cols, rows] = sel
        return val, msk

    def forward(self, x):
        # x: (B, 1 + fc_input_dim); leading col = subject index into sc_table.
        B = x.shape[0]
        subj_idx = x[:, 0].long()                    # (B,)
        fc = x[:, 1:]                                 # (B, fc_input_dim)

        # FC tokens.
        if self.use_fc_mask:
            val_mat, msk_mat = self._scatter_to_matrix_with_mask(fc)
            combined = torch.cat([val_mat, msk_mat], dim=2)   # (B,N,2N)
            tok = self.token_proj(combined)                    # (B,N,d_model)
        else:
            mat = self._scatter_to_matrix(fc)                  # (B,N,N)
            tok = self.token_proj(mat)                         # (B,N,d_model)

        # SC channels (subject-constant): table lookup or eval override.
        if self.override_sc_channels is not None:
            sc = self.override_sc_channels.to(tok.dtype).to(tok.device)
            sc = sc.unsqueeze(0).expand(B, -1, -1, -1)         # (B,C,N,N)
        else:
            sc = self.sc_table[subj_idx]                       # (B,C,N,N)
        for c in range(self.n_sc_channels):
            tok = tok + self.sc_proj[c](sc[:, c])              # (B,N,d_model)

        # CLS + Transformer.
        cls = self.cls.expand(B, -1, -1)                       # (B,1,d_model)
        h = torch.cat([cls, tok], dim=1)                       # (B,N+1,d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h[:, 0, :])                           # (B, out_dim)


# ---------------------------------------------------------------------------
# CPU self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__" and _TORCH_AVAILABLE:
    torch.manual_seed(0)
    np.random.seed(0)

    N = 8
    fc_input_dim = N * (N - 1) // 2     # 28
    C = 3
    n_subj = 5
    B = 4
    out_dim = 16

    sc_table = np.random.randn(n_subj, C, N, N).astype(np.float32)
    net = MultiChannelMatrixEmbedding(
        fc_input_dim=fc_input_dim,
        sc_table=sc_table,
        out_dim=out_dim,
        n_regions=N,
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_sc_channels=C,
        use_fc_mask=True,
    )

    # Build x = [subj_idx | fc_vec] with subj_idx in 0..n_subj-1.
    subj_idx = torch.randint(0, n_subj, (B, 1)).float()
    fc_vec = torch.randn(B, fc_input_dim)
    x = torch.cat([subj_idx, fc_vec], dim=1)        # (B, 1 + fc_input_dim)

    out = net(x)
    assert out.shape == (B, out_dim), f"bad shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite output"

    loss = out.pow(2).mean()
    loss.backward()
    assert net.token_proj.weight.grad is not None, "no grad on token_proj"
    assert torch.isfinite(net.token_proj.weight.grad).all(), "token_proj grad not finite"
    assert net.sc_proj[0].weight.grad is not None, "no grad on sc_proj[0]"
    assert torch.isfinite(net.sc_proj[0].weight.grad).all(), "sc_proj[0] grad not finite"

    # Eval path: set override_sc_channels to a (C, N, N) tensor.
    net.zero_grad()
    net.eval()
    net.override_sc_channels = torch.randn(C, N, N)
    with torch.no_grad():
        out_eval = net(x)
    assert out_eval.shape == (B, out_dim), f"bad eval shape {tuple(out_eval.shape)}"
    assert torch.isfinite(out_eval).all(), "non-finite eval output"

    # use_fc_mask=False variant also works.
    net2 = MultiChannelMatrixEmbedding(
        fc_input_dim=fc_input_dim, sc_table=sc_table, out_dim=out_dim,
        n_regions=N, d_model=32, n_heads=4, n_layers=1, use_fc_mask=False,
    )
    out2 = net2(x)
    assert out2.shape == (B, out_dim) and torch.isfinite(out2).all()

    # Factory branch.
    net3 = make_embedding_net(
        "multichannel_matrix", input_dim=fc_input_dim, out_dim=out_dim,
        sc_table=sc_table, n_regions=N, d_model=32, n_heads=4, n_layers=1,
    )
    assert isinstance(net3, MultiChannelMatrixEmbedding)
    assert net3(x).shape == (B, out_dim)

    print("MultiChannelMatrixEmbedding self-test: ALL PASS")
