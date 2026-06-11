"""Library-native parameter sensitivity via sbi's ActiveSubspace.

VBI is built on `sbi`, which ships gradient-based sensitivity analysis
(`sbi.analysis.ActiveSubspace`). This finds the directions in PARAMETER space
that a property is most sensitive to — using the trained density estimator,
so NO new simulations are needed (unlike the forward sweep in sensitivity.py).

Two senses of sensitivity:
  identifiability  : which params the posterior is most peaked along, given an
                     observation x_obs  (find_directions on the posterior).
  property-driven  : which params most drive a scalar property of the sim
                     (e.g. simFC~SC), via add_property + find_directions.

Eigenvalue_k  = how sensitive along active direction k (larger = more).
Eigenvector_k = the parameter combination of that direction; its squared
                components, summed weighted by eigenvalues, give a per-parameter
                sensitivity score.

Usage (after Step 8, with the trained posterior in scope):

    from active_sensitivity import report_active_sensitivity
    report_active_sensitivity(
        s1["posterior"], s1["theta_scaled"], config.STAGE1_PARAMS,
        x_obs=x_obs_for_one_subject,           # identifiability mode
    )
    # or property-driven:
    report_active_sensitivity(
        s1["posterior"], s1["theta_scaled"], config.STAGE1_PARAMS,
        property_values=simfc_sc_per_sim,      # e.g. corr(simFC, SC) per train sim
    )
"""
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _per_param_scores(e_vals, e_vecs, n_keep=None):
    """Aggregate active-subspace eigval/eigvec into a per-parameter score.

    score_p = sum_k eigval_k * eigvec_k[p]^2   (eigvecs are columns), normalized.
    """
    ev = np.asarray(e_vals, dtype=float).ravel()
    V = np.asarray(e_vecs, dtype=float)          # (n_params, n_dirs), columns=dirs
    if V.shape[0] != ev.shape[0] and V.shape[1] == ev.shape[0]:
        pass  # already (n_params, n_dirs)
    ev = np.abs(ev)
    if n_keep:
        order = np.argsort(-ev)[:n_keep]
        ev = ev[order]; V = V[:, order]
    score = (V ** 2) @ ev                        # (n_params,)
    s = score.sum()
    return score / s if s > 0 else score


def report_active_sensitivity(posterior, theta_scaled, param_names,
                              x_obs=None, property_values=None,
                              num_monte_carlo_samples=2000, verbose=True):
    """Run sbi ActiveSubspace and report per-parameter sensitivity.

    Provide EITHER x_obs (identifiability of the posterior at that observation)
    OR property_values (per-sim scalar property to attribute to parameters).
    """
    from sbi.analysis import ActiveSubspace

    asub = ActiveSubspace(posterior)
    mode = None

    if property_values is not None:
        theta_t = torch.as_tensor(np.asarray(theta_scaled), dtype=torch.float32)
        prop_t = torch.as_tensor(np.asarray(property_values, dtype=np.float32)).reshape(-1, 1)
        asub.add_property(theta_t, prop_t)
        e_vals, e_vecs = asub.find_directions(num_monte_carlo_samples=num_monte_carlo_samples)
        mode = "property-driven"
    else:
        if x_obs is not None:
            try:
                posterior.set_default_x(
                    torch.as_tensor(np.asarray(x_obs), dtype=torch.float32).reshape(1, -1))
            except Exception:
                pass
        e_vals, e_vecs = asub.find_directions(
            posterior_log_prob_as_property=True,
            num_monte_carlo_samples=num_monte_carlo_samples)
        mode = "identifiability (posterior log-prob)"

    e_vals = np.asarray(e_vals.detach().cpu()).ravel()
    e_vecs = np.asarray(e_vecs.detach().cpu())
    # sbi returns eigvecs as columns ordered by ascending eigval -> reorder desc
    order = np.argsort(-np.abs(e_vals))
    e_vals = e_vals[order]; e_vecs = e_vecs[:, order]

    scores = _per_param_scores(e_vals, e_vecs)

    if verbose:
        print("=" * 64)
        print(f"  ActiveSubspace sensitivity — mode: {mode}")
        print("=" * 64)
        evn = np.abs(e_vals) / (np.abs(e_vals).max() + 1e-12)
        print("  active-direction eigenvalues (norm): "
              + "  ".join(f"{v:.3f}" for v in evn))
        print("  top active direction (param loadings):")
        top = e_vecs[:, 0]
        for i, p in enumerate(param_names):
            print(f"    {p:8s} {top[i]:+.3f}")
        print("  per-parameter sensitivity score (sum=1):")
        for i, p in enumerate(param_names):
            bar = "#" * int(round(scores[i] * 40))
            print(f"    {p:8s} {scores[i]:.3f}  {bar}")
        print("=" * 64)
    return {"eigenvalues": e_vals, "eigenvectors": e_vecs,
            "param_scores": dict(zip(param_names, scores)), "mode": mode}
