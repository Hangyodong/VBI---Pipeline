# 28 — Claude Code Harness: Task Protocols

Part of the Claude Code Harness for VBI-SBI pipeline.
Generated: 2026-05-18  Branch: refactor/02-simulation
Repo: /scratch/home/wog3597/vbi

## HOW TO USE THIS FILE

This file is a **reference of standard operating procedures**. When
the user assigns a task, identify which protocol matches (§3.1 import
migration, §3.2 logging addition, etc.) and follow the numbered
checklist verbatim. Do **not** improvise — the protocols encode
hard-won lessons from session history (docs 19–25).

Every protocol ends with a **STOP-condition cross-reference** to file
30. If any stop condition triggers during the protocol, halt and use
the §5.6 STOP template in file 30.

---

## 3.1 PROTOCOL: IMPORT MIGRATION (P-1, P-2-style patches)

Used for: replacing legacy `import evaluate` / `from simulator
import X` with the correct package path.

Reference precedents: P-1 (doc 18-19), P-2 (doc 20-21).

```
1. READ the target file in full (no skim).
2. IDENTIFY the exact line(s) — capture line number and surrounding
   context (3 lines above, 3 below).
3. VERIFY the target symbol exists in the new package:
     python -c "from <new.pkg> import <Symbol>; print(<Symbol>)"
4. VERIFY identity with the old source (object equality):
     python -c "
     from <new.pkg> import <Symbol> as new
     from <old> import <Symbol> as old
     print(new is old, type(new).__name__)
     "
   For evaluation (P-1) this also requires
   `from <new.pkg> import _private_helper_a, _private_helper_b, ...`
   to confirm wildcard coverage.
5. RUN T-0 invariants from file 29 §4.1 to confirm current state.
6. APPLY only the targeted line(s). Use the Edit tool with a
   precise old_string / new_string anchor. Do NOT touch surrounding
   whitespace.
7. Run compile check:  python -m compileall -q .
8. Run the file-29 §4.1 Tier 0 verification suite.
9. Document the result in a new doc:
     NN_patchN_result.md
   following the doc-19 / doc-21 template:
     - Changed files
     - Exact line changed
     - Diff summary
     - Tests run + results
     - Whether rollback is needed
     - Recommended next patch
```

**Stop-condition crosscheck**: SC-8 (no Proceed), SC-11–17 (any
verification fails), SC-14–16 (invariants break).

---

## 3.2 PROTOCOL: ADDING LOGGING / PRINT STATEMENTS

Used for: per-step progress logs (Step 2 simulation, Step 4–8 SNPE).

Reference precedents: Step 2 logging in `inference/training_data.py`
+ `simulation/wc_runner.py`; Step 4–8 logging in `inference/snpe.py`.

```
1. READ the target file in full.
2. IDENTIFY existing convention:
     - inference/snpe.py + inference/training_data.py use _progress()
     - simulation/wc_runner.py uses plain print()
     Use the file's existing convention; do not mix.
3. PLAN every log line. Each line must be ≤ 100 characters wide.
4. INSERT logs only between existing lines — never inside an
   expression, never split a statement.
5. Special-case patterns:
   a. PCA EVR  (must use the fitted PCA object):
        pipeline.fc_pca.pca.explained_variance_ratio_
      Wrap in try/except AttributeError → print "EVR: unavailable".
   b. Trainable parameter count:
        sum(p.numel() for p in net.parameters() if p.requires_grad)
      Format with f"{n:,}" (thousands separator).
   c. SNPE per-epoch table (sbi 0.26.1):
        Check `inferer._summary` AFTER training. It exposes
          training_loss[], validation_loss[],
          epoch_durations_sec[], epochs_trained[], best_validation_loss[]
        Use Approach A (post-hoc table). Never call .train() twice.
        If _summary is missing (older sbi): fall back to Approach B,
        a 30 s heartbeat thread (`threading.Thread(daemon=True)`).
   d. GPU buffer event logs (inside simulate_gpu_batch):
        chunk index, csz, elapsed (time.time() delta), valid count
        (np.isfinite(b).all() summed), running total (len(outputs))
6. COMPILE-check after each function edited:
     python -m compileall -q <file>
7. IMPORT-smoke:
     python -c "import <module>; print('import OK')"
8. Public-API signatures must be byte-identical before/after —
   verify with inspect.signature() (file 29 §4.2).
9. Document line additions per file in the result doc.
```

**Forbidden in §3.2**:
- ❌ Calling `inferer.train()` more than once.
- ❌ Subclassing or monkey-patching sbi internals (HC-7 across
  multiple tasks).
- ❌ Redirecting sbi's own stdout / stderr.
- ❌ Importing tqdm or any other progress library (HC-3).
- ❌ Adding new required keyword arguments to public functions
  (HC-1 across multiple tasks).

**Stop-condition crosscheck**: SC-8, SC-11–13, SC-23 (new config key
attempt), SC-24 (new dependency attempt).

---

## 3.3 PROTOCOL: GPU OPTIMIZATION (`simulation/wc_runner.py`)

Used for: HBM/PCIe/sync optimizations inside `_run_streaming_hrf`
and `simulate_gpu_batch`.

Reference precedent: GPU-1 (doc-driven session in this branch).

```
0. PRE-FLIGHT (mandatory):
   a. Verify GPU-1 invariants are present (file 29 §4.1 T-0e):
        _alloc_stride_buffers(cp, ...) — module-level helper
        _trim_memory_pool(cp) — module-level helper
        NO module-scope `import cupy` (must stay deferred inside
        function bodies).
   b. Run debug.py --basic baseline (file 29 §4.5) to capture
      the existing PASS=9 / FAIL=3 baseline before edit.
   c. If on a GPU host, run the QC check (file 29 §4.3 T-2b)
      to capture pre-edit BOLD output. Save the diff value.

1. READ simulation/wc_runner.py in full (~480 lines now).
2. READ bold.py to confirm BoldMonitor behavior:
     interim_istep = int(round(4.0 / dt_ms))   # = 8 for dt=0.5
     mon.step(i, E_cpu, t_cut_ms) accumulates 8 raw frames then
       averages → outer stock → HRF convolve at every 1 TR.
   Any optimization that changes the per-step E value seen by
   mon.step(i, ...) is a SCIENTIFIC change (atol must be checked).
3. READ vbi/models/cupy/wilson_cowan.py WC_sde.heunStochastic to
   confirm whether matmuls are involved (they are NOT — TF32 hints
   give no benefit on this kernel).
4. PLAN the optimization. State explicitly:
     - Whether the BOLD output is bit-identical (preferred) or
       within an atol (must measure).
     - Which sync/transfer counts change before vs after.
     - Whether the lazy cupy import pattern is preserved.
     - Whether the per-theta fallback path is unaffected.
5. APPLY edits. Keep helpers parameterized with cp (do not import at
   module top). Keep `_run_streaming_hrf(model, n_nodes, num_sim,
   dt_ms, apply_bw)` signature unchanged.
6. POST-VERIFY (mandatory):
   a. python -m compileall -q simulation/wc_runner.py
   b. python -c "import simulation.wc_runner; print('OK')"
      (must succeed even without GPU — lazy import)
   c. On GPU host: run_theta_specific_check (file 29 §4.3 T-2b)
      MUST return pass=True. For bit-identical optimizations
      diff=0.0 is expected; for within-atol diff < 1e-4 is required.
   d. 50-sim timing benchmark (file 29 §4.3 T-2c) — record
      ms/sim before vs after.
7. DOCUMENT in a new doc, including:
     - Which OPT-N items were IMPLEMENTED / SKIPPED / PARTIAL
     - Pre/post benchmark numbers
     - Constraint compliance checklist (HC-1..HC-15)
```

**Forbidden in §3.3**:
- ❌ Add module-level `import cupy` (breaks no-GPU import).
- ❌ Move `BoldMonitor` to GPU.
- ❌ Change `BoldMonitor.step()` itself; optimize only the loop
  feeding it.
- ❌ Replace `mon.step(i+j, E_buf[j])` with `mon.step(i+j, E_last)`
  — that variant changes the 8-sample window mean and violates
  HC-1's atol contract.
- ❌ Patch VBI source files.

**Stop-condition crosscheck**: SC-7 (VBI file edit), SC-9 (unstaged
M-status on wc_runner.py), SC-10 (param shape break), SC-17 (QC
pass=False), SC-20 (module-scope deferred import).

---

## 3.4 PROTOCOL: NOTEBOOK CELL EDITING (`main.ipynb`)

Used for: any scientific cell edit. **Never** touches `cell[2]`,
`cell[3]`, `cell[4]`.

Reference precedents: doc 23 (insertion), 24 (Section I expansion),
25 (cleanup).

```
1. BACKUP first (HC-11):
     cp main.ipynb main.ipynb.bak_$(date +%Y%m%d_%H%M%S)
2. READ the notebook JSON. List every cell:
     for each i: cell_type, line count, first 80 chars of source.
3. IDENTIFY the target cell(s). Cell numbering is 0-based:
     cell[0]  markdown — title
     cell[1]  markdown — Setup heading
     cell[2]  code     — Setup  ← NEVER EDIT
     cell[3]  markdown — Integrated Debug heading  ← NEVER EDIT
     cell[4]  code     — Integrated Debug Cell    ← NEVER EDIT
     cell[5]  markdown — Optional GPU batch tuner heading
     cell[6]  markdown-typed-as-code — GPU batch tuner
     cell[7..40] — scientific cells (Step 1..14 markdown + code pairs)
4. WRITE a one-shot Python helper (e.g., _edit_cell_N.py) that
   loads the JSON, edits the target cell's source string only,
   and writes back. Helper template:
     nb = json.loads(Path("main.ipynb").read_text())
     cell = nb["cells"][N]
     assert cell["cell_type"] == "code"
     src = "".join(cell["source"])
     # ... edit src ...
     compile(src, f"<cell[{N}]>", "exec")   # T2-9 syntax gate
     lines = src.splitlines(keepends=True)
     if lines and lines[-1].endswith("\n"):
         lines[-1] = lines[-1].rstrip("\n")  # ipynb convention
     cell["source"] = lines
     Path("main.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
5. RUN the helper. VERIFY:
     - Cell count unchanged (or +N if inserting per spec).
     - Byte size of main.ipynb changed (backup ≠ new).
     - All other cells unmodified (diff cell[0..N-1] and cell[N+1..end]
       byte-for-byte if possible).
6. COMPILE-check every modified cell:
     compile(modified_src, f"<cell[{N}]>", "exec")
7. CLEAN UP: rm _edit_cell_N.py
8. NEVER execute the notebook programmatically (HC-13).
9. Document in a new doc.
```

**Forbidden in §3.4**:
- ❌ Touch cell[2] (Setup).
- ❌ Touch cell[3] or cell[4] (Integrated Debug Cell + heading).
- ❌ Reorder cells.
- ❌ Change cell_type from `markdown` to `code` or vice versa.
- ❌ Mutate cell metadata (outputs, execution_count) outside the
  normal write-back semantics.

**Stop-condition crosscheck**: SC-8, SC-12, SC-13 (cell compile
fails), SC-31 (file delete proposal includes a backup).

---

## 3.5 PROTOCOL: COLORMAP CHANGES (`evaluation/plots.py`)

Used for: aligning heatmap colormaps to the C-1..C-4 contract.

Reference precedent: doc-driven Task B in colormap session
(no-op confirmed) and recent re-verification in doc 25.

### Required final state

| Matrix | cmap | vmin | vmax |
|---|---|---|---|
| SC weight (`sc`, `weights`) | `'RdBu_r'` | `-np.abs(sc).max()` | `+np.abs(sc).max()` |
| Tract length (`lengths_mm`) | `'viridis'` | `0` | `np.nanmax(lengths_mm)` |
| FC (`fc`, `fc_obs`, `fc_pred`, `fc_mean`) | `'RdBu_r'` | `-1` | `+1` |
| FCD (`fcd`) | `'RdBu_r'` | `-1` | `+1` |

```
1. READ evaluation/plots.py AND evaluation/reports.py in full.
2. LIST every imshow/pcolormesh/heatmap call. Report:
     file:line, function name, data variable, current cmap,
     current vmin/vmax.
3. CLASSIFY each call: SC / tract length / FC / FCD / other.
   "Other" includes BOLD timeseries — leave at matplotlib default.
4. APPLY the minimum change: only the `cmap=`, `vmin=`, `vmax=`
   keyword arguments. Never restructure the function.
5. Non-heatmaps (histograms, bars, lines) → do not touch.
   plot_pca_diagnostic EVR bar plot → do not touch (T3-8).
   plot_posteriors histogram subplots → do not touch (T3-9).
6. After edits, REPORT every heatmap before/after — even
   non-edited ones (T3-6).
7. Compile check.
```

**Current state** (doc 25 §3): all FC sites in `evaluation/plots.py`
already conform; no SC/length/FCD heatmaps exist in `plots.py` or
`reports.py`. Notebook `cell[10]` plots SC weights with `cmap='hot_r'`
— but **`cell[10]` is out of scope for this protocol**.

**Stop-condition crosscheck**: SC-8 (no Proceed).

---

## 3.6 PROTOCOL: ADDING A NEW SOURCE FILE

Used for: extending a package (rarely needed; prefer edit-in-place).

```
1. CONFIRM the new file does NOT violate R3 (file 27 §2.3). For
   example, a new file in `features/` may not import from
   `simulation/`.
2. CREATE the file under the correct package directory. Never add a
   root-level `.py` (creates a duplicate-shadow risk like the 8 root
   duplicates already present).
3. REGISTER the new module in the package's `__init__.py`:
     from .<new_module> import <PublicSymbol>
   And add to `__all__` if the package uses one.
4. ADD docstrings: module header + every public function/class.
5. Compile-check the whole tree:
     python -m compileall -q .
6. Smoke-import:
     python -c "from <pkg> import <PublicSymbol>; print('OK')"
7. Update file 26 §1.4 FILE STATUS LEGEND with the new file.
   (This is the only doc edit allowed in a code-protocol — it
   keeps the harness in sync.)
```

**Forbidden in §3.6**:
- ❌ Add root-level `.py` files (extends the duplicate-shadow
  surface).
- ❌ Skip the `__init__.py` registration (creates a hidden import).

---

## 3.7 PROTOCOL: CODE ERROR-PROOFING PASS

Used for: defensive hardening only. Never bundle with logic work.

Reference precedent: doc 25 last task — 8 fixes across 5 files.

The 10 fix categories (each with explicit guard):

| Rule | Pattern | Fix |
|---|---|---|
| T1-1 | Unused top-level import (verified zero references) | Remove it (do not touch deferred imports inside function bodies) |
| T1-2 | Bare `except:` clause | `except Exception as e:` + `print(type(e).__name__, e)` before re-raise / continue |
| T1-3 | `print(var)` where `var` not defined on every code path | Wrap in try-guard or move below definition |
| T1-4 | `for x in iterable:` where iterable could be empty | `if not iterable: return / raise / continue` with clear message |
| T1-5 | Pickle / `.npz` load without existence check | `if not os.path.exists(path): raise FileNotFoundError(...)` |
| T1-6 | `config.X` for optional attribute | `getattr(config, "X", default)` |
| T1-7 | Numpy → cupy → torch dtype boundary | Print one-line warning (do not cast silently) |
| T1-8 | sbi multi-path import (try/except chain) | Final `except ImportError as e:` with a message listing all paths tried |
| T1-9 | Write to `OUTPUT_DIR` without makedirs | `os.makedirs(os.path.dirname(path), exist_ok=True)` before write |
| T1-10 | GPU resource path lacking pool trim on exit | `finally: _trim_memory_pool(cp)` — only in functions that already allocate cupy |

```
1. SCAN the in-scope files (one package at a time) for each pattern.
2. PROPOSE every fix in a single plan with one-line diffs. Group
   by file.
3. WAIT for "Proceed" authorization.
4. APPLY one fix at a time. After each, run compile-check.
5. Run debug.py --basic — PASS count must remain 9.
6. Document in a new doc with a per-rule fix table.
```

**Forbidden in §3.7**:
- ❌ Refactor logic, rename functions, split/merge functions.
- ❌ Remove or rewrite existing print statements.
- ❌ Bundle error-proofing with other work.

**Stop-condition crosscheck**: SC-8, SC-36 (>5 files).

---

## 3.8 PROTOCOL: ROOT-LEGACY-FILE OPERATIONS

Used for: archiving or deleting any of the 8 root duplicates,
`evaluate.py`, `simulator.py`, `inference.py`, or root `__init__.py`.

Reference precedent: doc 22 (decision report; no execution).

```
0. PRE-FLIGHT — these constraints are absolute:
   - simulator.py: keep until all 11 deferred `from simulator import`
     callers in inference/ + evaluation/ migrate (Tier 3 + Tier 4).
   - evaluate.py:  keep until main.ipynb cells 163, 165 (notebook
     references) and debug_notebook.py:82, 341 migrate (Tier 4b).
   - inference.py: full per-name diff audit against
     inference/__init__.py __all__ required (Tier 7).
   - 8 root duplicates: byte-identical to packages today (Tier 6).
   - root __init__.py: must confirm zero `import vbi` callers
     anywhere on the filesystem (Tier 6, last).

1. RUN the pre-flight grep for each target:
     grep -rn "^import <name>\b\|^from <name> import" \
       --include="*.py" --include="*.ipynb" .
   AND
     diff <root_dup>.py <pkg>/<root_dup>.py
   Both must return zero (for duplicates) or have approved migration
   path (for wrappers / monolith).
2. CONFIRM the user's exact authorization:
     "Proceed with Tier 6 archive of <names>"  — or
     "Proceed with delete of <name>"
3. Prefer mv into _legacy_root/ over rm:
     mkdir -p _legacy_root
     mv <name>.py _legacy_root/
4. POST-VERIFY:
     python -m compileall -q .
     python -c "import simulation, features, inference, evaluation, pipelines; print('OK')"
     python debug.py --basic | tail -1   # PASS=9 still
5. Document in a new doc.
```

**Stop-condition crosscheck**: SC-5, SC-8, SC-31 (file delete).

---

## 3.9 PROTOCOL: WRITING A NEW DOC

Used for: every significant task should end with a new numbered
`NN_*.md` doc.

```
1. Choose the next free number (currently 31, 32, ...).
2. Use the standard header:
     # NN — <Title>
     **Date:** YYYY-MM-DD
     **Author:** <name>
     **Status:** <Plan only / Applied / Verified>
     **Predecessor docs:** ...
     **Branch:** refactor/02-simulation
3. Sections (adapt per task):
     1. Goal
     2. Files in scope / Files NOT in scope
     3. Plan / Diffs
     4. Verification commands run + outputs
     5. Invariant re-check (P-1, P-2, GPU-1)
     6. Summary table
     7. Next-step recommendation
4. End with: a one-line "Awaiting user instruction" or "Complete"
   marker.
```

---

## 3.10 PROTOCOL: ANSWERING A "WHY?" QUESTION

Used for: design / history / decision-rationale questions that
don't require code changes.

```
1. Read the relevant existing doc(s) — chain 01 → 16 → 22 covers
   most architecture / refactor decisions.
2. Cite specific doc + section, not vague memory.
3. Quote exact file:line for any code claim.
4. Do not propose changes unless explicitly asked.
5. If memory says "X exists" and the user is about to act,
   verify by grep/import that X still exists today.
```

---

## 3.11 PROMPT TEMPLATE: STANDARD TASK HANDOFF

When the user asks for a task or you draft a self-prompt, use this
template:

```
CONTEXT:
  Working dir: /scratch/home/wog3597/vbi
  Branch: refactor/02-simulation
  Applied patches: P-1, P-2, GPU-1 (+ any new ones)

TASK:
  <one-sentence goal>

SCOPE (files to modify):
  - <file 1>
  - <file 2>

OUT OF SCOPE (do not touch):
  - config.py (R1)
  - inference.py (R8)
  - 8 root duplicates
  - bold.py
  - main.ipynb cell[2,3,4]
  - <other task-specific exclusions>

CONSTRAINTS:
  - <relevant subset of file 27>
  - Each <metric> ≤ <bound>
  - Backwards compatibility required for <surface>

PROCEDURE:
  Step 1 — READ <file list>
  Step 2 — PLAN (one-line diff previews; wait for OK)
  Step 3 — APPLY
  Step 4 — VERIFY (file 29 Tier 0 + relevant Tier)
  Step 5 — INVARIANT RE-CHECK (P-1, P-2, GPU-1)
  Step 6 — SUMMARY TABLE
  Step 7 — DOC

STOP if any of:
  - <stop conditions from file 30 relevant to the task>
```

---

## 3.12 PROTOCOL: EMERGENCY ROLLBACK

If verification fails after an edit, rollback immediately rather
than trying to patch-on-top.

```
1. Identify the affected file(s):
     git status
     git diff <file>
2. For tracked files:
     git checkout -- <file>
3. For files that have a .bak_* sibling (notebook edits):
     cp <file>.bak_<timestamp> <file>
4. Re-run Tier 0 verification (file 29 §4.1) to confirm clean
   baseline restored.
5. Document the rollback and the diagnosed root cause.
6. Do not retry the same edit without surfacing the root cause to
   the user.
```

---

**End of task protocols. Read 29 (verification) next.**
