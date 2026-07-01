#!/usr/bin/env python
"""Stage 7 — parse SC-conditioning ablation logs into one comparison table.

Usage:
    python tools/ablation_summary.py A:logs/ablation_A.log B:logs/ablation_B.log ...

For each arm it extracts (from the main_HCP stdout the run already prints):
  - test FC corr (per-draw mean)        : final-test / final-summary line
  - test FC corr (expected-FC)          : the S1 expected-FC line
  - test FC RMSE
  - Stage-6 means (post / prior / perm) + delta(post-prior), delta(post-perm)

Decisive readouts:
  delta(post-prior) > 0  -> posterior beats the prior-predictive baseline
  corr(D) - corr(A)      -> value added by SC conditioning vs FC-only baseline
  delta(post-perm) > 0   -> the SC channel is actually USED (not ignored/memorized)
"""
import re
import sys


def _last(pat, text, grp=1):
    """Return the last regex match group as float, or None."""
    ms = re.findall(pat, text)
    if not ms:
        return None
    g = ms[-1]
    g = g[grp - 1] if isinstance(g, tuple) else g
    try:
        return float(g)
    except (TypeError, ValueError):
        return None


def parse(path):
    try:
        with open(path) as f:
            t = f.read()
    except OSError as e:
        return {"error": str(e)}
    num = r"([-+]?\d+\.?\d*)"
    return {
        "corr_perdraw": _last(rf"FC corr\s+:\s+{num}\s+\[.*per-draw", t)
        or _last(rf"FC\s+corr\s+:\s+{num}\s+\[", t),
        "corr_exp": _last(rf"FC corr\(exp\)\s+:\s+{num}", t),
        "rmse": _last(rf"FC RMSE\s+:\s+{num}", t),
        # Stage-6 MEAN line
        "post": _last(rf"MEAN:\s*post={num}", t),
        "prior": _last(rf"MEAN:.*prior={num}", t),
        "perm": _last(rf"MEAN:.*perm={num}", t),
        "d_prior": _last(rf"delta\(post-prior\)={num}", t),
        "d_perm": _last(rf"delta\(post-perm\)={num}", t),
    }


def _f(v, w=8, p=4):
    return (f"{v:+.{p}f}".rjust(w)) if isinstance(v, float) else "n/a".rjust(w)


def main(argv):
    if not argv:
        print("usage: ablation_summary.py NAME:log NAME:log ...")
        return 1
    arms = []
    for a in argv:
        name, _, path = a.partition(":")
        arms.append((name or path, parse(path or name)))

    hdr = (f"  {'arm':3} | {'corr(draw)':>10} | {'corr(exp)':>10} | {'RMSE':>8} | "
           f"{'d(post-prior)':>13} | {'d(post-perm)':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    base_exp = None
    for name, r in arms:
        if "error" in r:
            print(f"  {name:3} | (log not read: {r['error']})")
            continue
        if name == "A":
            base_exp = r.get("corr_exp")
        print(f"  {name:3} | {_f(r['corr_perdraw'],10)} | {_f(r['corr_exp'],10)} | "
              f"{_f(r['rmse'],8)} | {_f(r['d_prior'],13)} | {_f(r['d_perm'],12)}")
    # headline contrasts vs A
    print("  " + "-" * (len(hdr) - 2))
    for name, r in arms:
        if name == "A" or "error" in r:
            continue
        ce = r.get("corr_exp")
        if isinstance(ce, float) and isinstance(base_exp, float):
            print(f"  corr(exp) {name} - A = {ce - base_exp:+.4f}  "
                  f"(SC value vs FC-only baseline)")
    print("\n  reads: d(post-prior)>0 = posterior beats prior; "
          "corr(X)-A>0 = SC channel adds value; d(post-perm)>0 = SC actually used.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
