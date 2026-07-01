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


def _regionwise_modes():
    return ("latent_regionwise", "direct_regionwise", "basis_regionwise")


def is_regionwise():
    """True when config.PARAMETER_MODE infers a region-wise control vector
    (latent/direct/basis) whose theta MUST be decoded into per-region parameter
    maps before simulation.

    Single-simulation evaluation paths use this to route through the decoded
    (latent_wrap-wrapped) batch simulator instead of passing coefficient-named
    thetas (e.g. ``g_LRE_const``) as scalar params — which the simulator
    silently ignores, falling back to defaults and giving identical, misleading
    FC every draw.
    """
    return (str(getattr(config, "PARAMETER_MODE", "homogeneous"))
            in _regionwise_modes())


def latent_wrap(base_sim):
    """Wrap a base simulate_gpu_batch so the region-wise control vector (theta) is
    decoded to per-region parameter maps before simulation.

    latent_regionwise : theta is a low-dim latent z -> region basis decode.
    direct_regionwise : theta is the PHYSICAL 1440 control vector -> reshape.
    Either way the maps are injected as '<param>_matrix' fixed overrides; theta is
    NOT passed as physical scalar parameters to the base simulator.
    """
    import numpy as np
    from param_decoder import decode_to_param_maps, make_fixed_overrides_from_param_maps

    def wrapped(weights, theta, param_names=None, fixed_overrides=None,
                delays=None, apply_bw=True, label=None, n_total=None,
                subject_id=None, **kw):
        n_regions = np.asarray(weights).shape[0]
        mode = str(getattr(config, "PARAMETER_MODE", "homogeneous"))
        if mode == "basis_regionwise" and getattr(config, "BASIS_PERSUBJECT", False):
            # per-subject myelin/gradient basis: decode THIS subject's basis.
            import re as _re
            from basis_decoder import get_decoder_for_basis

            def _to_sid(c):
                # robust: int, "100206", "resim(100206)", "baseline:100206" -> 100206
                if c is None:
                    return None
                if isinstance(c, (int, np.integer)):
                    return int(c)
                m = _re.search(r"\d+", str(c))
                return int(m.group()) if m else None
            sid = _to_sid(subject_id)
            if sid is None:
                sid = _to_sid(label)
            bases = getattr(config, "PERSUBJECT_BASES", None) or {}
            if sid is not None and sid in bases:
                dec = get_decoder_for_basis(
                    bases[int(sid)], list(config.HETERO_PARAMS),
                    getattr(config, "HETERO_BOUNDS", None),
                    rezscore=bool(getattr(config, "BASIS_REZSCORE", True)))
                maps = dec.decode(theta)
            else:
                # SBC / non-subject sims have no sid -> shared group basis (warn once).
                if not getattr(config, "_PERSUBJ_FALLBACK_WARNED", False):
                    print("  [basis] WARN: BASIS_PERSUBJECT on but subject_id unresolved "
                          f"(subject_id={subject_id!r} label={label!r}) -> shared-basis "
                          "fallback (expected only for SBC / non-subject sims).")
                    config._PERSUBJ_FALLBACK_WARNED = True
                maps = decode_to_param_maps(theta, None, config, n_regions=n_regions)
        else:
            basis = None
            if mode == "latent_regionwise":
                from region_basis import build_region_basis
                basis = build_region_basis(weights, labels=load_network_labels(), config=config)
            maps = decode_to_param_maps(theta, basis, config, n_regions=n_regions)
        ov = dict(fixed_overrides or {})
        ov.update(make_fixed_overrides_from_param_maps(maps))
        n = np.asarray(theta).shape[0]
        return base_sim(weights, np.zeros((n, 0), dtype=np.float64),
                        param_names=[], fixed_overrides=ov, delays=delays,
                        apply_bw=apply_bw, label=label, n_total=n_total, **kw)
    return wrapped


def get_simulate_gpu_batch():
    """Return the active engine's ``simulate_gpu_batch`` (region-wise-wrapped if
    PARAMETER_MODE is latent_regionwise or direct_regionwise)."""
    base = _module().simulate_gpu_batch
    if str(getattr(config, "PARAMETER_MODE", "homogeneous")) in _regionwise_modes():
        return latent_wrap(base)
    return base


def get_simulate_single():
    """Return the active engine's ``simulate_single``."""
    return _module().simulate_single
