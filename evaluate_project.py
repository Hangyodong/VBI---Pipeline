#!/usr/bin/env python3
"""Project scorecard — grade the VBI-SBI pipeline per aspect.

Each aspect is graded by an automated check that introspects the repo
(filesystem, config values, code patterns) or reads result artifacts.
Where a fact cannot be measured live (e.g. the simulator FC ceiling when
no joint_opt artifact exists), the check falls back to the recorded value
and marks the evidence as "(recorded)".

Run
---
    python evaluate_project.py            # colored scorecard
    python evaluate_project.py --no-color
    python evaluate_project.py --json     # machine-readable

No GPU / torch / cuBNM imports — pure introspection. numpy optional.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))

# Letter grade -> GPA points (4.0 scale).
GPA = {
    "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7, "D": 1.0, "F": 0.0, "?": None,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _read(path):
    """Return file text or '' if unreadable."""
    try:
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _exists(path):
    return os.path.exists(os.path.join(ROOT, path))


def grade_from_thresholds(value, bands):
    """bands: list of (min_value, grade) sorted high->low. First hit wins."""
    for thresh, letter in bands:
        if value >= thresh:
            return letter
    return bands[-1][1]


# --------------------------------------------------------------------------
# checks — each returns (grade, evidence_str)
# --------------------------------------------------------------------------

def check_code_structure():
    pkgs = ["inference", "evaluation", "pipelines", "simulation",
            "features", "data_loader.py", "config.py", "bold.py"]
    present = [p for p in pkgs if _exists(p)]
    inits = sum(
        1 for p in pkgs
        if p.endswith(".py") or _exists(os.path.join(p, "__init__.py"))
    )
    frac = len(present) / len(pkgs)
    grade = grade_from_thresholds(
        frac, [(1.0, "A"), (0.9, "A-"), (0.75, "B"), (0.5, "C"), (0.0, "F")]
    )
    if inits < len([p for p in pkgs if not p.endswith(".py")]):
        grade = "B+"  # missing some package __init__
    ev = f"{len(present)}/{len(pkgs)} modules present, package __init__ ok"
    return grade, ev


def check_split_hygiene():
    driver = _read("pipelines/stage1_stage2.py")
    dl = _read("data_loader.py")
    signals = {
        "three_way_split": "three_way_split" in driver or "three_way_split" in dl,
        "shape_assert": "assert" in driver and "shape" in driver,
        "leak_guard_doc": "never enter training" in driver.lower()
                          or "never enter" in driver.lower(),
    }
    hits = sum(signals.values())
    grade = grade_from_thresholds(
        hits, [(3, "A"), (2, "B+"), (1, "C"), (0, "F")]
    )
    ev = "+".join(k for k, v in signals.items() if v) or "no guards found"
    return grade, ev


def check_diagnostics():
    infn = _read("inference/__init__.py")
    diag = _read("inference/diagnostics.py")
    blob = infn + diag
    feats = {
        "SBC": "simulation_based_calibration" in blob,
        "probing": "evaluate_embedding_probing" in blob,
        "predictive": "posterior_predictive_check" in blob,
        "shrinkage": "compute_shrinkage" in blob,
    }
    hits = sum(feats.values())
    grade = grade_from_thresholds(
        hits, [(4, "A-"), (3, "B+"), (2, "B-"), (1, "C"), (0, "D")]
    )
    ev = ", ".join(k for k, v in feats.items() if v) or "none"
    return grade, ev


def check_inference_method():
    """2-stage design concerns: arbitrary fc_obs default, no-resim slice,
    and conflicting stage2 definitions across modules."""
    stage1 = _read("inference/stage1.py")
    driver = _read("pipelines/stage1_stage2.py")
    issues = []
    # fc_obs defaults to a *simulated* sample, not the real target.
    if re.search(r"fc_obs\s*=\s*p1\[.fc_raw.\]\[0\]", stage1):
        issues.append("fc_obs default = sim[0] (not real subject)")
    # Phase 3 re-uses Phase-1 sims by slicing only (no resimulation).
    if "no re-simulation" in stage1.lower() or "no re-simulation" in stage1:
        issues.append("Phase3 = feature slice, no resim (info-losing)")
    # Conflicting stage2 definition: driver disables it.
    if "Stage 2 is currently undefined" in driver:
        issues.append("stage2 conflict: driver disables Phase2/3")
    n = len(issues)
    grade = grade_from_thresholds(
        n, [(3, "C-"), (2, "C"), (1, "B-"), (0, "B+")]
    )
    ev = " | ".join(issues) if issues else "no structural concerns detected"
    return grade, ev


def _active_model():
    """Resolve the inference model. config.py holds the default, but each
    entrypoint may override it (main_HCP.py sets 'rwweib'). Honor the
    VBI_MODEL env override first so the score matches the run you mean."""
    env = os.environ.get("VBI_MODEL")
    if env:
        return env
    cfg = _read("config.py")
    m = re.search(r"^INFERENCE_MODEL\s*=\s*[\"'](\w+)[\"']", cfg, re.M)
    return m.group(1) if m else "wc"


# Per-model ceiling artifact: the npz holding best real-FC corr per subject.
# WC (mouse) is measured by joint_opt. RWW-EIB-FFI (HCP) has no ceiling tool
# yet, so it stays unmeasured rather than inheriting WC's number.
_CEILING_ARTIFACT = {
    "wc": "output_mouse_mptp/joint_opt_group.npz",
    "rwweib": "output_hcp/joint_opt_group.npz",   # not produced yet
    "rww": "output_hcp/joint_opt_group.npz",
}


def check_scientific_validity():
    """Simulator's ceiling: max real-FC correlation reachable by joint
    parameter search. <0.30 => structural misspecification.

    Model-aware: WC and RWW are different forward models with different
    ceilings. WC's recorded 0.216 does NOT transfer to RWW."""
    model = _active_model()
    path = os.path.join(ROOT, _CEILING_ARTIFACT.get(model, ""))
    mean = None
    src = ""
    if path and os.path.exists(path):
        try:
            import numpy as np
            d = np.load(path, allow_pickle=True)
            mean = float(np.mean(d["best_corrs"]))
            src = f"measured n={len(d['best_corrs'])}"
        except Exception:
            mean = None
    # Fall back to a recorded value ONLY for WC (the model it was measured on).
    if mean is None and model == "wc":
        mean = 0.2156  # recorded group ceiling (joint g_e,c_ei,c_ee, delays ON)
        src = "recorded"
    if mean is None:
        # RWW switched in but no ceiling measured — do not inherit WC's grade.
        ev = (f"model={model}: real-FC ceiling NOT measured "
              f"(no {os.path.basename(_CEILING_ARTIFACT.get(model, '?'))}; "
              f"WC's 0.216 does not transfer) — needs a ceiling run")
        return "?", ev
    grade = grade_from_thresholds(
        mean, [(0.6, "A"), (0.5, "B"), (0.4, "C"), (0.3, "C-"), (0.0, "D")]
    )
    ev = (f"model={model}: real-FC ceiling = {mean:.3f} ({src}); "
          f"<0.30 = struct. misspec")
    return grade, ev


def check_repo_hygiene():
    baks = glob.glob(os.path.join(ROOT, "*.bak*"))
    baks += glob.glob(os.path.join(ROOT, "**", "*.bak*"), recursive=True)
    baks = set(baks)
    mat = glob.glob(os.path.join(ROOT, "*.mat"))
    mat_gb = sum(os.path.getsize(m) for m in mat) / 1e9
    oneoff = [f for f in (
        "diag_coupling.py", "sweep_fc.py", "joint_opt.py", "sc_scale_test.py",
        "wc_neural_fc.py", "wc_opsweep.py", "wc_regime_test.py", "bench_sim.py",
        "bench_hcp.py", "debug.py", "debug_notebook.py")
        if _exists(f)]
    penalty = (len(baks) > 5) + (mat_gb > 0.5) + (len(oneoff) > 4)
    grade = grade_from_thresholds(
        3 - penalty, [(3, "A"), (2, "B"), (1, "C"), (0, "D")]
    )
    ev = (f"{len(baks)} .bak, {mat_gb:.1f}GB .mat in-tree, "
          f"{len(oneoff)} one-off scripts at root")
    return grade, ev


def check_config_validation():
    cfg = _read("config.py")
    nval = re.search(r"^N_VAL\s*=\s*(\d+)", cfg, re.M)
    nval = int(nval.group(1)) if nval else None
    use_fcd = re.search(r"^USE_FCD\s*=\s*(True|False)", cfg, re.M)
    use_fcd = use_fcd.group(1) if use_fcd else "?"
    problems = []
    if nval == 0:
        problems.append("N_VAL=0 -> validation metrics NaN, no model-sel signal")
    if use_fcd == "False":
        problems.append("USE_FCD=False -> FCD PCA branch is dead weight")
    n = len(problems)
    grade = grade_from_thresholds(
        n, [(2, "C"), (1, "B-"), (0, "A-")]
    )
    ev = (f"N_VAL={nval}, USE_FCD={use_fcd}"
          + ("; " + "; ".join(problems) if problems else " (ok)"))
    return grade, ev


# --------------------------------------------------------------------------
# registry: (aspect, weight, check_fn)
# --------------------------------------------------------------------------

ASPECTS = [
    ("Code structure / modularity", 1.0, check_code_structure),
    ("Train/val/test hygiene",      1.0, check_split_hygiene),
    ("Diagnostics infrastructure",  1.0, check_diagnostics),
    ("Inference methodology",       1.5, check_inference_method),
    ("Scientific validity",         2.0, check_scientific_validity),
    ("Repo hygiene",                0.5, check_repo_hygiene),
    ("Config / validation setup",   1.0, check_config_validation),
]


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

_COLORS = {
    "A": "\033[92m", "B": "\033[96m", "C": "\033[93m",
    "D": "\033[91m", "F": "\033[91m", "?": "\033[90m",
}
_RESET = "\033[0m"


def _color(grade, use_color):
    if not use_color:
        return grade
    return f"{_COLORS.get(grade[0], '')}{grade}{_RESET}"


def run(use_color=True):
    rows = []
    for aspect, weight, fn in ASPECTS:
        try:
            grade, ev = fn()
        except Exception as e:  # a broken check never kills the report
            grade, ev = "?", f"check error: {e}"
        rows.append({"aspect": aspect, "weight": weight,
                     "grade": grade, "gpa": GPA.get(grade),
                     "evidence": ev})
    return rows


def print_table(rows, use_color=True):
    wa = max(len(r["aspect"]) for r in rows)
    line = "=" * (wa + 58)
    print(line)
    print("  VBI-SBI PIPELINE SCORECARD")
    print(line)
    print(f"  {'Aspect':<{wa}}  {'Grade':<6} {'Wt':<4} Evidence")
    print("  " + "-" * (wa + 54))
    for r in rows:
        print(f"  {r['aspect']:<{wa}}  "
              f"{_color(r['grade'], use_color):<{6 + (9 if use_color else 0)}} "
              f"{r['weight']:<4} {r['evidence']}")
    print("  " + "-" * (wa + 54))
    graded = [(r["gpa"], r["weight"]) for r in rows if r["gpa"] is not None]
    if graded:
        wsum = sum(w for _, w in graded)
        gpa = sum(g * w for g, w in graded) / wsum
        letter = min(GPA.items(),
                     key=lambda kv: abs((kv[1] or 99) - gpa)
                     if kv[1] is not None else 99)[0]
        print(f"  {'WEIGHTED GPA':<{wa}}  "
              f"{_color(letter, use_color):<{6 + (9 if use_color else 0)}} "
              f"     {gpa:.2f} / 4.00")
    print(line)
    sci = next((r for r in rows if r["aspect"].startswith("Scientific")), None)
    if sci and sci["grade"] == "?":
        print("  verdict: solid engineering; forward-model validity UNMEASURED")
        print("  for the active model. Run a real-FC ceiling before trusting")
        print("  any posterior — WC's old number does not carry over to RWW.")
    else:
        print("  verdict: solid engineering on an unresolved forward-model gap.")
        print("  scientific validity is the binding constraint — fix the")
        print("  simulator (FC ceiling >0.30) before scaling inference.")
    print(line)


def main():
    ap = argparse.ArgumentParser(description="Grade the VBI-SBI pipeline.")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = run(use_color=not args.no_color)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print_table(rows, use_color=not args.no_color)


if __name__ == "__main__":
    main()
