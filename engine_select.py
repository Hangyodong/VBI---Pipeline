"""Single source of truth for the active simulation engine.

The bug this fixes: training (``inference/training_data.py``) routed the
forward model by an explicit ``engine=`` arg (so HCP trained on RWW-EIB-FFI),
but every *evaluation* path — final-test resim, SBC, posterior-predictive,
plots, baseline — hardcoded ``from cuBNM.simulate import ...`` (the WCVBI
adapter). Result: the posterior was trained on RWW but scored against Wilson-
Cowan, which doesn't even have the RWW params (g_LRE/g_FFI/I_o). The resim
ignored the sampled theta -> identical FC every draw -> FC corr pinned and
invariant (the ``± 0.0000`` signature).

Resolve the engine from ``config.INFERENCE_MODEL`` instead, so resim/SBC/
predictive use the SAME model as training. Imports are deferred to call time
so importing this module needs no GPU.

Mapping (mirrors inference/training_data.py engine routing):
  INFERENCE_MODEL   module                    model
  "wc" / "cubnm"    cuBNM.simulate            Wilson-Cowan (WCVBI)
  "rwweib"          cuBNM.simulate_rwweib     RWW-EIB-FFI
  "rww"             cuBNM.simulate_rww        stock reduced Wong-Wang
  "vbi" / "gpu"     simulator                 cupy VBI engine
"""
import importlib

import config

_MODEL_TO_MODULE = {
    "wc":     "cuBNM.simulate",
    "cubnm":  "cuBNM.simulate",
    "rwweib": "cuBNM.simulate_rwweib",
    "rwweib2": "cuBNM.simulate_rwweib_2cpl",
    "rwweibdelay": "cuBNM.simulate_rwweib_delay",
    "rww":    "cuBNM.simulate_rww",
    "vbi":    "simulator",
    "gpu":    "simulator",
}


def active_engine():
    """Return the lower-cased active INFERENCE_MODEL (default 'wc')."""
    return str(getattr(config, "INFERENCE_MODEL", "wc")).lower()


def _module():
    m = active_engine()
    if m not in _MODEL_TO_MODULE:
        raise ValueError(
            f"engine_select: unknown config.INFERENCE_MODEL {m!r}; "
            f"expected one of {sorted(_MODEL_TO_MODULE)}"
        )
    return importlib.import_module(_MODEL_TO_MODULE[m])


def load_network_labels():
    """Row-aligned atlas network labels from config.NETWORK_LABELS_CSV, or None.

    Reads the first column of each non-empty line (aligned by ROW ORDER, not id).
    """
    path = getattr(config, "NETWORK_LABELS_CSV", None)
    if not path:
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line.split(",")[0])
    return rows or None


def latent_wrap(base_sim):
    """Wrap a base simulate_gpu_batch so latent z (theta) is decoded to per-region
    parameter maps before simulation (PARAMETER_MODE='latent_regionwise').

    The wrapper builds the region basis from the passed SC (so it is correct
    per-subject), decodes z -> param maps, and injects them as '<param>_matrix'
    fixed overrides. theta is otherwise NOT used as physical parameters.
    """
    import numpy as np
    from region_basis import build_region_basis
    from param_decoder import (decode_latent_to_param_maps,
                               make_fixed_overrides_from_param_maps)

    def wrapped(weights, theta_z, param_names=None, fixed_overrides=None,
                delays=None, apply_bw=True, label=None, n_total=None, **kw):
        basis = build_region_basis(weights, labels=load_network_labels(), config=config)
        maps = decode_latent_to_param_maps(theta_z, basis, config)
        ov = dict(fixed_overrides or {})
        ov.update(make_fixed_overrides_from_param_maps(maps))
        n = np.asarray(theta_z).shape[0]
        return base_sim(weights, np.zeros((n, 0), dtype=np.float64),
                        param_names=[], fixed_overrides=ov, delays=delays,
                        apply_bw=apply_bw, label=label, n_total=n_total, **kw)
    return wrapped


def get_simulate_gpu_batch():
    """Return the active engine's ``simulate_gpu_batch`` (latent-wrapped if
    PARAMETER_MODE='latent_regionwise')."""
    base = _module().simulate_gpu_batch
    if str(getattr(config, "PARAMETER_MODE", "homogeneous")) == "latent_regionwise":
        return latent_wrap(base)
    return base


def get_simulate_single():
    """Return the active engine's ``simulate_single``."""
    return _module().simulate_single
