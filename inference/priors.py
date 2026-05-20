"""Prior distributions used by SBI/SNPE-C.

Public API
----------
- make_scaled_prior(n_dim) : BoxUniform([-1, 1]^n_dim) on torch tensors

Why scaled space
----------------
SBI's density estimators train more stably when the prior is bounded
to a unit box. The raw-space prior bounds live in ``config`` and are
honored by ParameterScaler (see ``inference.scaling``).
"""


def make_scaled_prior(n_dim, device="cpu"):
    """Build the scaled-space BoxUniform prior used by SNPE.

    Parameters
    ----------
    n_dim : int
        Number of inferred parameters (Stage 1: 4, Stage 2: len(theta_bad)+4).
    device : str or torch.device, optional
        Device for the underlying low/high tensors (default ``"cpu"``).
        Pass ``config.SBI_DEVICE`` so the prior lives on the same device
        as the SNPE inferer and avoids cross-device sampling overhead.

    Returns
    -------
    sbi.utils.BoxUniform on torch tensors in [-1, 1]^n_dim.
    """
    import torch
    from sbi.utils import BoxUniform
    dev = torch.device(device)
    return BoxUniform(
        low=torch.full((n_dim,), -1.0, dtype=torch.float32, device=dev),
        high=torch.full((n_dim,), +1.0, dtype=torch.float32, device=dev),
    )
