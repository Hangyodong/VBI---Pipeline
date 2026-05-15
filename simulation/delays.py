"""Tract length → conduction delay computation, plus VBI delay-key detection.

Public API
----------
- compute_delay_matrix(weights, velocity_m_per_s, lengths_mm) : (N,N) ms
- detect_delay_key()  : auto-detect the VBI WC parameter name for delay
- apply_delay(params, delays)  : inject delay into WC params via the
                                 detected key

Design notes
------------
- Tract length matrix is used ONLY for delay. SC weights are coupling
  weights — never used as a length proxy unless ``lengths_mm`` is
  explicitly None (legacy fallback). The fallback prints no warning here
  (the caller, ``data_loader.get_subject_data``, prints it instead).
- 1 m/s == 1 mm/ms, so ``delays_ms = lengths_mm / velocity_m_per_s``
  needs no extra unit conversion.
"""
import numpy as np

import config


# ---------------------------------------------------------------------------
# Delay matrix
# ---------------------------------------------------------------------------

def compute_delay_matrix(weights, velocity_m_per_s, lengths_mm=None):
    """Convert tract weights / lengths into millisecond delays.

    Parameters
    ----------
    weights : (N, N) ndarray
        SC coupling weights. Used ONLY as a length proxy when
        ``lengths_mm`` is None (legacy fallback).
    velocity_m_per_s : float
        Conduction velocity. ``None`` or ``<=0`` returns ``None``.
    lengths_mm : (N, N) ndarray, optional
        Tract lengths in mm. **Preferred input.** If supplied, weights
        are ignored.

    Returns
    -------
    delays : (N, N) float64  or  None
    """
    if velocity_m_per_s is None or velocity_m_per_s <= 0:
        return None
    if lengths_mm is None:
        eps = 1e-3
        lengths_mm = 1.0 / (weights + eps)
        np.fill_diagonal(lengths_mm, 0)
        if lengths_mm.max() > 0:
            # Scale to mouse-typical mean ~10 mm
            lengths_mm = lengths_mm / lengths_mm.max() * 10.0
    delays = lengths_mm / float(velocity_m_per_s)
    np.fill_diagonal(delays, 0)
    return delays.astype(np.float64)


# ---------------------------------------------------------------------------
# VBI WC delay-key detection
# ---------------------------------------------------------------------------

_DELAY_KEY = None
_DELAY_KEY_CHECKED = False


def detect_delay_key():
    """Find the parameter name that the VBI WC class accepts for delays.

    Returns ``"delays"`` / ``"delay_matrix"`` / ``"tract_lengths"`` /
    ``"velocity"`` / ... or ``None`` if no recognised key is supported.
    The result is cached for subsequent calls.
    """
    global _DELAY_KEY, _DELAY_KEY_CHECKED
    if _DELAY_KEY_CHECKED:
        return _DELAY_KEY
    _DELAY_KEY_CHECKED = True
    try:
        from simulation.wc_runner import _import_wc
        wc_cls = _import_wc()
        m = wc_cls({})
        valid = set(getattr(m, "valid_params", []))
        for key in ("delays", "delay_matrix", "tract_lengths"):
            if key in valid:
                _DELAY_KEY = key
                return key
        for key in ("velocity", "speed", "conduction_velocity"):
            if key in valid:
                _DELAY_KEY = key
                return key
    except Exception:
        pass
    return None


def apply_delay(params, delays_precomputed):
    """Inject the delay into VBI WC parameters using the detected key.

    Parameters
    ----------
    params : dict
        WC parameter dict. Mutated in place.
    delays_precomputed : (N, N) ndarray or None
        Output of ``compute_delay_matrix``. If None, this is a no-op.
    """
    key = detect_delay_key()
    if key is None or delays_precomputed is None:
        return
    if key in ("delays", "delay_matrix", "tract_lengths"):
        params[key] = delays_precomputed
    elif key in ("velocity", "speed", "conduction_velocity"):
        params[key] = float(config.VELOCITY_M_PER_S)


# Backward-compatible private alias used by simulator.py / wc_runner.py
_apply_delay = apply_delay
