"""Side-by-side timing harness: VBI WC engine vs cuBNM WCSimGroup.

Runs both engines on the SAME (SC, theta) batch, records wall-clock time and
ms/sim, and computes a sanity FC difference between the two BOLD outputs.

All heavy imports (cupy via simulation.wc_runner, cubnm via cuBNM.runner) are
deferred inside functions, so this module imports without a GPU.
"""
import time

import numpy as np


def _run_vbi(weights, theta_batch, param_names, apply_bw=True):
    """Run the current VBI engine. Returns (bold_list, wall_sec) or raises."""
    from simulation.wc_runner import simulate_gpu_batch  # lazy
    t0 = time.perf_counter()
    bold = simulate_gpu_batch(
        weights, theta_batch, list(param_names), apply_bw=apply_bw,
    )
    return bold, time.perf_counter() - t0


def _run_cubnm(weights, theta_batch, param_names,
               duration_s, tr_s, dt_ms, force_gpu=True):
    """Run the cuBNM engine. Returns (bold_list, wall_sec) or raises."""
    from cuBNM.runner import run_cubnm_batch  # lazy
    t0 = time.perf_counter()
    bold = run_cubnm_batch(
        weights, theta_batch, list(param_names),
        duration_s=duration_s, tr_s=tr_s, dt_ms=dt_ms, force_gpu=force_gpu,
    )
    return bold, time.perf_counter() - t0


def _summarize(bold_list, wall_sec):
    n = len(bold_list) if bold_list is not None else 0
    shape = tuple(np.asarray(bold_list[0]).shape) if n else None
    return {
        "n_sims": n,
        "wall_sec": float(wall_sec),
        "ms_per_sim": float(wall_sec * 1000.0 / n) if n else float("nan"),
        "bold_shape": shape,
    }


def benchmark(weights, theta_batch, param_names,
              duration_s=600.0, tr_s=1.0, dt_ms=0.1,
              apply_bw=True, force_gpu=True):
    """Benchmark VBI vs cuBNM on one SC + theta batch.

    Returns
    -------
    dict
        {
          'vbi':   {n_sims, wall_sec, ms_per_sim, bold_shape}  or  {'error': ...},
          'cubnm': {n_sims, wall_sec, ms_per_sim, bold_shape}  or  {'error': ...},
          'fc_abs_diff': float | None,   # mean |FC_vbi - FC_cubnm| (first sim)
        }

    Each engine is wrapped independently: a failure in one (e.g. cubnm not
    runnable in this env) does not abort the other.
    """
    from cuBNM.fc import compute_fc, fc_abs_diff  # lazy (no heavy deps)

    out = {"vbi": None, "cubnm": None, "fc_abs_diff": None}
    vbi_bold = cubnm_bold = None

    # ── VBI engine ──
    try:
        vbi_bold, vbi_wall = _run_vbi(weights, theta_batch, param_names, apply_bw)
        out["vbi"] = _summarize(vbi_bold, vbi_wall)
    except Exception as e:  # noqa: BLE001 — keep the other engine alive
        out["vbi"] = {"error": f"{type(e).__name__}: {e}"}

    # ── cuBNM engine ──
    try:
        cubnm_bold, cub_wall = _run_cubnm(
            weights, theta_batch, param_names,
            duration_s=duration_s, tr_s=tr_s, dt_ms=dt_ms, force_gpu=force_gpu,
        )
        out["cubnm"] = _summarize(cubnm_bold, cub_wall)
    except Exception as e:  # noqa: BLE001
        out["cubnm"] = {"error": f"{type(e).__name__}: {e}"}

    # ── FC sanity diff (first sim of each, same convention as features/fc.py) ──
    if vbi_bold and cubnm_bold:
        try:
            fc_v = compute_fc(vbi_bold[0])
            fc_c = compute_fc(cubnm_bold[0])
            out["fc_abs_diff"] = fc_abs_diff(fc_v, fc_c)
        except Exception as e:  # noqa: BLE001
            out["fc_abs_diff"] = None
            out["fc_error"] = f"{type(e).__name__}: {e}"

    return out
