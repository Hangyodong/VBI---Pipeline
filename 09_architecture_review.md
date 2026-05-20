# 09 — Architecture Review

**Date:** 2026-05-18
**Branch:** refactor/02-simulation
**Reviewer:** Claude (claude-sonnet-4-6)

---

## 1. Current Structure Summary

The repository contains two overlapping layers: a **clean five-package modular system** and
a **set of root-level legacy/duplicate/compat files** that pre-date the refactor. Both layers
are present simultaneously.

### Authoritative packages (new, correct)

| Package | Role | Files |
|---|---|---|
| `simulation/` | GPU WC engine, delays, warmup, QC | 4 modules + `__init__.py` |
| `features/` | FC/FCD computation, batch extraction | 4 modules + `__init__.py` |
| `inference/` | Scaling, priors, pipeline, embedding, SNPE, IO | 12 modules + `__init__.py` |
| `evaluation/` | Metrics, validation, model selection, plots, reports | 6 modules + `__init__.py` |
| `pipelines/` | 14-step orchestration | 1 module + `__init__.py` |

### Root-level modules (legitimate, not legacy)

| File | Role |
|---|---|
| `config.py` | Single source of truth for all constants |
| `data_loader.py` | Data I/O, SC preprocessing, subject split |
| `bold.py` | Balloon-Windkessel BoldMonitor / HRF kernel |
| `pipeline_setup.py` | Interactive config override + module reload helper |
| `main.py` | Thin CLI entry point |
| `debug.py` | Smoke tests (no GPU: `--basic`; full: `--all`) |
| `debug_notebook.py` | Notebook debug cell helpers |

### Root-level legacy / compat / duplicate files (problematic)

| File | Category | Authoritative location |
|---|---|---|
| `inference.py` | OLD MONOLITH (55 KB, dead code) | `inference/` package |
| `simulator.py` | COMPAT WRAPPER (no logic) | `simulation/` + `features/` |
| `evaluate.py` | COMPAT WRAPPER (no logic) | `evaluation/` package |
| `wc_runner.py` | ROOT DUPLICATE | `simulation/wc_runner.py` |
| `fc.py` | ROOT DUPLICATE | `features/fc.py` |
| `fcd.py` | ROOT DUPLICATE | `features/fcd.py` |
| `extraction.py` | ROOT DUPLICATE | `features/extraction.py` |
| `screening.py` | ROOT DUPLICATE | `features/screening.py` |
| `delays.py` | ROOT DUPLICATE | `simulation/delays.py` |
| `warmup.py` | ROOT DUPLICATE | `simulation/warmup.py` |
| `qc.py` | ROOT DUPLICATE | `simulation/qc.py` |
| `__init__.py` (root) | PACKAGE LEAK (mirrors `simulation/__init__.py`) | `simulation/__init__.py` |

Total root-level problem files: **12**

---

## 2. Major Risks

### Risk 1 — `inference.py` monolith is dead but misleading (HIGH)

Python 3 resolves `import inference` to `inference/` package (package beats module).
However, `inference.py` is 55 KB of real logic still on disk. Editors and static
analyzers (pylint, pyright, mypy) often cache stale resolutions and may point to
`inference.py` instead of `inference/`. Any developer who opens `inference.py` and
edits it will get no runtime effect, causing silent divergence.

**Likelihood of confusion:** high on any new contributor or AI assistant without full context.

### Risk 2 — Root duplicate files can shadow package submodules (HIGH)

A bare `import fc`, `import wc_runner`, `import delays`, etc. resolves to the root
duplicate, not the package submodule. The root files and package files may drift over
time if the root is accidentally edited. Currently the content appears identical, but
that invariant is not enforced anywhere.

**Symptom if it breaks:** subtle wrong behavior with no import error — both files are
importable, but one is stale.

### Risk 3 — Root `__init__.py` makes repo importable as `vbi` package (MEDIUM)

If `/scratch/home/wog3597` is ever added to `sys.path` (e.g., by a notebook or a
parent package), `import vbi` succeeds and exposes `simulation` names at the top level.
This mirrors `simulation/__init__.py` and is not the intended public interface.

**Likelihood of triggering:** low in normal use; elevated if Jupyter adds parent dirs.

### Risk 4 — `__pycache__` staleness after refactor (MEDIUM)

Multiple `.pyc` caches may exist for both the root duplicates and the package submodules.
After any rename/move, stale caches can cause Python to import the wrong compiled version.
The `refactor/02-simulation` branch history shows several refactor commits; caches may
not have been purged.

### Risk 5 — Compat wrappers (`simulator.py`, `evaluate.py`) delay cleanup (LOW)

These files contain no logic, only re-exports. They are safe today but extend the window
during which callers can use legacy import paths. Callers that import from `simulator`
will never naturally migrate without an explicit audit.

---

## 3. Import Conflict Analysis

### 3.1 `inference.py` vs `inference/` package

```
Python 3 resolution order:
  1. sys.modules cache
  2. Built-in modules
  3. Frozen modules
  4. sys.path entries — for each entry:
       a. <entry>/inference/  (package directory with __init__.py)  ← WINS
       b. <entry>/inference.py
```

**Conclusion:** `inference/` always wins at runtime. `inference.py` is unreachable via
normal `import inference`. However, tools that resolve imports by filename-matching
(VS Code Pylance in some configs, static type-checkers with non-standard paths) may
incorrectly link to `inference.py`.

### 3.2 Root duplicate vs package submodule

For files like `fc.py` / `features/fc.py`:

```
import fc              → resolves to root fc.py           (risky bare import)
from fc import ...     → resolves to root fc.py           (risky bare import)
from features import fc → resolves to features/fc.py      (correct)
from features.fc import compute_fc → correct              (correct)
```

All eight root duplicates follow this pattern. If the repo root is in `sys.path`
(which it always is when run from root), the bare name hits the root file first.

### 3.3 Root `__init__.py` vs `simulation/__init__.py`

```
import vbi              → resolves only if parent dir in sys.path
from simulation import  → always resolves to simulation/__init__.py (correct path)
```

The root `__init__.py` only becomes a conflict if the parent directory
(`/scratch/home/wog3597`) is in `sys.path`.

### 3.4 Confirmed safe paths

| Import statement | Resolves to | Safe? |
|---|---|---|
| `from simulation import simulate_gpu_batch` | `simulation/__init__.py` | Yes |
| `from features import compute_fc` | `features/__init__.py` | Yes |
| `from inference import ParameterScaler` | `inference/__init__.py` | Yes |
| `from evaluation import fc_metrics` | `evaluation/__init__.py` | Yes |
| `from pipelines import run_pipeline` | `pipelines/__init__.py` | Yes |
| `import inference` | `inference/__init__.py` | Yes |
| `import fc` | `fc.py` (ROOT DUPLICATE) | Risky |
| `import wc_runner` | `wc_runner.py` (ROOT DUPLICATE) | Risky |
| `from simulator import simulate_gpu_batch` | `simulator.py` (compat wrapper) | Avoid |

---

## 4. Root-Level Duplicate File Analysis

### 4.1 Exact-duplicate candidates (same content as package version)

Based on the refactor history (Phase 1–4 commits), these root files were created
as copies before the package was built. They should be identical to their package
counterparts, but content has not been diff'd yet.

| Root file | Package counterpart | Drift risk |
|---|---|---|
| `wc_runner.py` | `simulation/wc_runner.py` | Medium — both files were modified in `refactor/02-simulation` branch |
| `delays.py` | `simulation/delays.py` | Low — delays logic is stable |
| `warmup.py` | `simulation/warmup.py` | Low — warmup logic is stable |
| `qc.py` | `simulation/qc.py` | Low — QC logic is stable |
| `fc.py` | `features/fc.py` | Low — FC computation is stable |
| `fcd.py` | `features/fcd.py` | Low — FCD is disabled in production |
| `extraction.py` | `features/extraction.py` | Low — extraction logic is stable |
| `screening.py` | `features/screening.py` | Low — screening is a future stub |

**Action required:** Run `diff` on each pair before deleting root copies (see Section 8).

### 4.2 Compat wrappers (thin re-exports, no logic)

| Root file | What it re-exports | Can be removed when... |
|---|---|---|
| `simulator.py` | `simulation.*` + `features.*` | All callers use `simulation.*` / `features.*` |
| `evaluate.py` | `evaluation.*` | All callers use `evaluation.*` |

### 4.3 Dead monolith

| Root file | Size | Status | Risk of deleting now |
|---|---|---|---|
| `inference.py` | 55 KB | Dead code; package wins | Must verify no direct import exists first |

### 4.4 Package leak

| Root file | Mirrors | Side effect if present |
|---|---|---|
| `__init__.py` | `simulation/__init__.py` | Makes repo root importable as a package |

---

## 5. Recommended Analysis Order

Analyze files in this order to build understanding from the outside in and from
stable to volatile:

### Phase A — Configuration and entry points (read first)

1. `config.py` — all constants; must be understood before any module
2. `main.py` — CLI entry, shows top-level call
3. `pipelines/stage1_stage2.py` — 14-step driver; shows full orchestration

### Phase B — Data and I/O (foundational, no imports from packages)

4. `data_loader.py` — SC/FC loading, subject split
5. `bold.py` — BoldMonitor HRF (imported only by simulation/)

### Phase C — Simulation package (innermost, no upward dependencies)

6. `simulation/delays.py` — delay matrix computation
7. `simulation/warmup.py` — warm-start logic
8. `simulation/wc_runner.py` — GPU WC engine (most complex; imports delays, bold)
9. `simulation/qc.py` — QC guards

### Phase D — Features package

10. `features/fc.py` — FC computation
11. `features/fcd.py` — FCD (disabled in production)
12. `features/extraction.py` — batch extraction

### Phase E — Inference package (depends on simulation + features)

13. `inference/scaling.py` — ParameterScaler
14. `inference/priors.py` — SBI prior construction
15. `inference/feature_pipeline.py` — FamilyScaler, PCA
16. `inference/embedding.py` — MLP head
17. `inference/training_data.py` — step2 simulate, step3 extract
18. `inference/snpe.py` — steps 4–8, SNPE-C training
19. `inference/stage1.py` — stage 1 driver
20. `inference/posterior.py` — posterior ops
21. `inference/stage2.py` — stage 2 driver
22. `inference/diagnostics.py` — SBC, MLP probe
23. `inference/io.py` — artifact save/load

### Phase F — Evaluation package

24. `evaluation/metrics.py` — fc_metrics, bootstrap_ci
25. `evaluation/validation.py` — validate stage 1/2
26. `evaluation/model_selection.py` — scoring and selection
27. `evaluation/final_test.py` — test set evaluation
28. `evaluation/plots.py` — visualization
29. `evaluation/reports.py` — console report functions

### Phase G — Root legacy files (analyze to confirm drift / no callers)

30. Root duplicates — diff against package counterparts
31. `inference.py` — scan for any `import inference.py`-specific patterns
32. `simulator.py`, `evaluate.py` — check who calls them
33. Root `__init__.py` — check if parent dir ever added to sys.path

---

## 6. Recommended Cleanup Order

Execute cleanup in this order (each phase depends on the prior being verified):

### Stage 1 — Verify no callers of root duplicates

Before any deletion, grep the entire repo for bare imports of root files:
```bash
grep -rn "^import fc\b\|^from fc import\|^import wc_runner\|^from wc_runner import" \
     --include="*.py" --include="*.ipynb" .
grep -rn "^import delays\|^import warmup\|^import qc\b\|^import fcd\b\|^import extraction\|^import screening\b" \
     --include="*.py" --include="*.ipynb" .
grep -rn "^import simulator\|^from simulator import" --include="*.py" --include="*.ipynb" .
grep -rn "^import evaluate\b\|^from evaluate import" --include="*.py" --include="*.ipynb" .
```

### Stage 2 — Diff root duplicates vs package counterparts

```bash
diff delays.py simulation/delays.py
diff warmup.py simulation/warmup.py
diff qc.py simulation/qc.py
diff fc.py features/fc.py
diff fcd.py features/fcd.py
diff extraction.py features/extraction.py
diff screening.py features/screening.py
diff wc_runner.py simulation/wc_runner.py
```

If a root file differs from its package counterpart, port any unique changes to the
package version before deleting the root file.

### Stage 3 — Delete safe root duplicates (no callers, no drift)

Order of deletion (least risky first):
1. `screening.py` — future stub, no logic
2. `fcd.py` — FCD disabled in production, stable code
3. `delays.py` — stable, no callers expected
4. `warmup.py` — stable
5. `qc.py` — stable
6. `fc.py` — stable
7. `extraction.py` — stable

### Stage 4 — Delete or neutralize `wc_runner.py` (root)

`wc_runner.py` was modified in the `refactor/02-simulation` branch. Diff carefully
before removing. Verify the package version is the authoritative one.

### Stage 5 — Remove compat wrappers after caller migration

1. Audit callers of `simulator.py` and `evaluate.py`
2. Update each caller to use `simulation.*` / `features.*` / `evaluation.*` directly
3. Delete `simulator.py` and `evaluate.py`

### Stage 6 — Remove root `__init__.py`

Verify that nothing imports the repo root as a package (`import vbi`), then delete.

### Stage 7 — Remove `inference.py` monolith (last)

```bash
python -c "import inference; print(inference.__file__)"
# Must print: .../inference/__init__.py NOT .../inference.py
grep -rn "inference\.py" --include="*.py" --include="*.md" .
```

If the above confirms the monolith is never loaded, delete `inference.py`.

---

## 7. Files That Require Manual Confirmation Before Deletion

Do not delete these without explicit review:

| File | Reason for caution |
|---|---|
| `inference.py` | 55 KB of logic; must confirm 100% of it is replicated in `inference/`; diff is large |
| `wc_runner.py` (root) | Modified in current branch; may contain changes not yet merged into `simulation/wc_runner.py` |
| `evaluate.py` | Compat wrapper, but `main.ipynb` may import from it; notebook must be grep-checked |
| `simulator.py` | Compat wrapper, but `debug.py` and `debug_notebook.py` may use it |
| `__init__.py` (root) | Mirrors `simulation/__init__.py`; check if parent `sys.path` manipulation exists anywhere |
| `debug.py` | Not a duplicate, but imports from multiple packages; must not be deleted, only audited for bad import patterns |
| `debug_notebook.py` | Same as `debug.py` — keep, only audit |
| `main.ipynb` | Large notebook; cells may use bare imports from root duplicates; audit before removing root files |

---

## 8. Next Commands to Run

Run these in order. Do not edit Python files until each check passes.

### 8.1 Confirm Python resolves packages correctly

```bash
python -c "import inference; print(inference.__file__)"
# Expected: .../vbi/inference/__init__.py

python -c "from simulation import simulate_gpu_batch; print('OK')"
python -c "from features import compute_fc; print('OK')"
python -c "from evaluation import fc_metrics; print('OK')"
python -c "from pipelines import run_pipeline; print('OK')"
```

### 8.2 Purge stale pycache

```bash
find /scratch/home/wog3597/vbi -name "*.pyc" -delete
find /scratch/home/wog3597/vbi -name "__pycache__" -type d -exec rm -rf {} +
```

### 8.3 Grep for bare imports of root duplicates

```bash
grep -rn "^import fc\b\|^from fc import" --include="*.py" .
grep -rn "^import fcd\b\|^from fcd import" --include="*.py" .
grep -rn "^import wc_runner\|^from wc_runner import" --include="*.py" .
grep -rn "^import delays\b\|^from delays import" --include="*.py" .
grep -rn "^import warmup\b\|^from warmup import" --include="*.py" .
grep -rn "^import qc\b\|^from qc import" --include="*.py" .
grep -rn "^import extraction\b\|^from extraction import" --include="*.py" .
grep -rn "^import screening\b\|^from screening import" --include="*.py" .
grep -rn "^import simulator\b\|^from simulator import" --include="*.py" .
grep -rn "^import evaluate\b\|^from evaluate import" --include="*.py" .
```

Also grep the notebook (JSON format):
```bash
grep -n "\"import fc\|from fc import\|import wc_runner\|from wc_runner\|import evaluate\b\|from evaluate import" main.ipynb
```

### 8.4 Diff root duplicates vs package counterparts

```bash
diff delays.py simulation/delays.py     && echo "delays: identical"
diff warmup.py simulation/warmup.py     && echo "warmup: identical"
diff qc.py simulation/qc.py             && echo "qc: identical"
diff fc.py features/fc.py               && echo "fc: identical"
diff fcd.py features/fcd.py             && echo "fcd: identical"
diff extraction.py features/extraction.py && echo "extraction: identical"
diff screening.py features/screening.py && echo "screening: identical"
diff wc_runner.py simulation/wc_runner.py && echo "wc_runner: identical"
```

### 8.5 Check root `__init__.py` vs `simulation/__init__.py`

```bash
diff __init__.py simulation/__init__.py
```

### 8.6 Compile-check all package modules (no GPU needed)

```bash
python -m py_compile config.py data_loader.py bold.py main.py pipeline_setup.py
python -m py_compile simulation/wc_runner.py simulation/delays.py simulation/warmup.py simulation/qc.py
python -m py_compile features/fc.py features/fcd.py features/extraction.py features/screening.py
python -m py_compile inference/scaling.py inference/priors.py inference/feature_pipeline.py inference/embedding.py
python -m py_compile inference/training_data.py inference/snpe.py inference/stage1.py inference/stage2.py
python -m py_compile inference/posterior.py inference/diagnostics.py inference/io.py
python -m py_compile evaluation/metrics.py evaluation/validation.py evaluation/model_selection.py
python -m py_compile evaluation/final_test.py evaluation/plots.py evaluation/reports.py
python -m py_compile pipelines/stage1_stage2.py
echo "All modules compile-clean."
```

### 8.7 Check dependency rule violations (R3 in 07_refactor_rules.md)

Verify `simulation/` does not import from `features/`, `inference/`, or `evaluation/`:
```bash
grep -n "from features\|from inference\|from evaluation" simulation/*.py
grep -n "from features\|from inference\|from evaluation" features/*.py
grep -n "from evaluation" inference/*.py
```

Each of these should return no output.

### 8.8 Check for circular import risks within `inference/`

```bash
grep -n "from inference.stage1\|import stage1" inference/snpe.py
grep -n "from inference.snpe\|import snpe" inference/stage1.py inference/stage2.py | head
```

`snpe.py` must not import `stage1`; `stage1`/`stage2` may import `snpe`.

---

## Summary

The modular package structure (`simulation/`, `features/`, `inference/`, `evaluation/`,
`pipelines/`) is sound and the dependency hierarchy (R3) is well-defined. The primary
technical debt is the 12-file root-level legacy layer. Of these, `inference.py` poses
the highest confusion risk (55 KB of dead code), followed by the 8 root duplicate files
whose content may have drifted. No Python code should be edited until the diff and grep
checks in Section 8 are complete.
