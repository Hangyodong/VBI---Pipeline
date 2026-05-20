# 08 — Claude Project Upload Guide

This guide describes which files to upload to a Claude Project for effective
AI-assisted development on this codebase, in what order, and what to ask first.

---

## Upload Order: Markdown Documentation First

Upload these files **before any Python code**. They give Claude the architectural
context needed to reason correctly about any Python file.

### Tier 1 — Upload First (essential context)

| File | Why |
|---|---|
| `01_repo_overview.md` | Architecture, scientific assumptions, package vs. legacy structure |
| `02_repo_tree.md` | Full annotated file tree with legacy markers |
| `03_module_index.md` | Module-by-module function tables with deps |
| `04_data_flow.md` | End-to-end data flow with shapes and formulas |
| `06_known_errors.md` | Import conflict risks; prevents Claude from suggesting broken patterns |
| `07_refactor_rules.md` | Rules that must not be violated during edits |

### Tier 2 — Upload Second (config and entry points)

| File | Why |
|---|---|
| `config.py` | All numerical constants; required to understand any pipeline question |
| `main.py` | Thin entry point; shows the top-level call |
| `pipelines/stage1_stage2.py` | Full 14-step orchestration; shows the big picture |
| `pipelines/__init__.py` | Short; shows what `run_pipeline` is |

### Tier 3 — Upload on Demand (by topic)

Upload these only when working on a specific area:

| File | When to upload |
|---|---|
| `simulation/wc_runner.py` | GPU simulation bugs, per-sim param issues, BOLD shape problems |
| `simulation/delays.py` | Delay / velocity / tract length questions |
| `simulation/warmup.py` | Warm-start or BoldMonitor questions |
| `features/fc.py` | FC computation or vectorization questions |
| `features/fcd.py` | FCD computation questions (when USE_FCD=True) |
| `features/extraction.py` | Batch extraction or parallelism questions |
| `inference/scaling.py` | ParameterScaler or prior bounds questions |
| `inference/feature_pipeline.py` | PCA / z-score / FeaturePipeline questions |
| `inference/snpe.py` | SNPE-C training, MLP, step4-8 questions |
| `inference/stage1.py` | Stage 1 end-to-end flow questions |
| `inference/stage2.py` | Stage 2, theta_bad selection, nuisance sampling |
| `inference/posterior.py` | Posterior sampling, shrinkage, PPC |
| `evaluation/metrics.py` | FC/FCD metric computation questions |
| `evaluation/validation.py` | Validation flow questions |
| `evaluation/model_selection.py` | Model selection score questions |
| `data_loader.py` | Data loading, SC preprocessing, subject split |
| `bold.py` | BoldMonitor, HRF kernel questions |

### Files NOT to Upload Initially

| File | Reason |
|---|---|
| `inference.py` (root) | Dead code; shadows the package; will confuse Claude |
| `wc_runner.py` (root) | Duplicate; use `simulation/wc_runner.py` instead |
| `fc.py` (root) | Duplicate; use `features/fc.py` instead |
| `fcd.py` (root) | Duplicate; use `features/fcd.py` instead |
| `extraction.py` (root) | Duplicate; use `features/extraction.py` instead |
| `delays.py` (root) | Duplicate; use `simulation/delays.py` instead |
| `warmup.py` (root) | Duplicate; use `simulation/warmup.py` instead |
| `qc.py` (root) | Duplicate; use `simulation/qc.py` instead |
| `screening.py` (root) | Duplicate; use `features/screening.py` instead |
| `simulator.py` | Compat wrapper; no logic inside |
| `evaluate.py` | Compat wrapper; no logic inside |
| `debug.py` | Large file; upload only when debugging test failures |
| `main.ipynb` | Very large (1.2 MB); upload only if notebook cells are the focus |
| `MPTP_FC_115.mat` | Binary data; Claude cannot parse `.mat` files |
| `MPTP_SC_115.mat` | Binary data |

---

## Recommended First Claude Project Question

After uploading Tier 1 + Tier 2 files, start with:

```
I am working on the Mouse MPTP VBI-SBI pipeline.
I have uploaded the architecture docs and config.py.

Please confirm your understanding of:
1. What data is the input (SC, FC, participants.tsv)?
2. What parameters are being inferred (Stage 1 and Stage 2)?
3. What is the SBI feature vector (shape and content)?
4. What root-level files are legacy/duplicate and should not be edited?
5. Which package takes priority when both inference.py and inference/ exist?

Do not write any code yet. Just confirm your understanding of the architecture.
```

---

## Debugging Prompt Template

Use this template when reporting a bug:

```
## Bug Report

**Error message:**
<paste full traceback here>

**What I was trying to do:**
<one sentence>

**Which step of the 14-step pipeline:**
Step N: <step name>

**Relevant config values:**
N_REGIONS = 115
FEATURE_SET = fc_only / fc_fcd
USE_FCD = True / False
N_SIM = ___

**Files involved (already uploaded):**
- simulation/wc_runner.py
- inference/training_data.py
- <etc.>

**What I have already tried:**
<any fixes attempted>

**Question:**
What is the root cause and the minimal fix?
Rules: do not change scientific behavior, use package imports (not root duplicates),
       do not add comments that describe what the code does.
```

---

## Quick Reference: Import Pattern Check

If Claude suggests an import, verify it follows these rules:

| Suggested import | Correct? | Use instead |
|---|---|---|
| `from simulation.wc_runner import simulate_gpu_batch` | Yes | — |
| `from features.fc import compute_fc` | Yes | — |
| `from inference import ParameterScaler` | Yes | — |
| `from evaluation import fc_metrics` | Yes | — |
| `from pipelines import run_pipeline` | Yes | — |
| `from simulator import simulate_gpu_batch` | Avoid | Use `simulation.wc_runner` |
| `from fc import compute_fc` | Wrong | Use `features.fc` |
| `import inference` then `inference.ParameterScaler` | Yes (resolves to package) | — |
| `from inference import ParameterScaler` (from `inference.py`) | Never — dead file | Use `inference/` package |
