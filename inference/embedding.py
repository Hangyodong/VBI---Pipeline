"""Embedding network for SBI/SNPE-C.

Public API
----------
- FeatureEmbedding(input_dim, hidden_dim=None, out_dim=None)
      MLP head jointly trained with the SBI density estimator.

The torch dependency is guarded so this module can be imported on
torch-less systems (CPU-only environments without ML stack). The class
itself raises ImportError if instantiated without torch.
"""
import config

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


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
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)
