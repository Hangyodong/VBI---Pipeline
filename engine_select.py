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


def get_simulate_gpu_batch():
    """Return the active engine's ``simulate_gpu_batch``."""
    return _module().simulate_gpu_batch


def get_simulate_single():
    """Return the active engine's ``simulate_single``."""
    return _module().simulate_single
