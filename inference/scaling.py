"""Parameter scaling between raw parameter space and SBI's [-1, 1] box.

Public API
----------
- ParameterScaler              : raw <-> scaled mapping, with subset/to_dict
- make_stage1_param_scaler()   : scaler for Stage 1 prior
- make_stage2_param_scaler(p)  : scaler for theta_bad + c-params

Why this exists
---------------
SBI's SNPE-C is trained in scaled parameter space [-1, 1] for numerical
stability, but VBI Wilson-Cowan simulation expects raw parameter values
(e.g. P=1.5, g_e=0.5). ParameterScaler is the only bridge between the
two spaces; it must be the SAME instance that does forward (raw→scaled)
during training-data collection and backward (scaled→raw) during
posterior sampling, otherwise the posterior is silently mis-aligned.

Design notes
------------
- The mapping is **data-free**: it depends only on the prior box.
- ``to_dict(theta)`` returns ``{param_name: value}`` for raw arrays,
  matching the dict form WC_sde expects.
- ``subset(names)`` produces a scaler over a parameter subset (used by
  Stage 2 where only theta_bad + c-params are inferred).
"""
import numpy as np

import config


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
