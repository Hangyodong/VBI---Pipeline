# 26 — Claude Code Harness: Context

Part of the Claude Code Harness for VBI-SBI pipeline.
Generated: 2026-05-18  Branch: refactor/02-simulation
Repo: /scratch/home/wog3597/vbi

## HOW TO USE THIS FILE

Read this **first** at the start of every new session. It orients you
in five minutes. After reading, you must understand: (a) what the
project does scientifically, (b) the package architecture, (c) which
files are authoritative vs. legacy, (d) what patches have already
been applied, (e) which failures are known and out of scope.

Then read 27 (constraints), 28 (protocols), 29 (verification), 30
(stop conditions) in order — before producing any plan.

---

## 1.1 PROJECT IDENTITY

This codebase is a **whole-brain digital twin pipeline for the mouse
MPTP model of Parkinson's disease**. It performs **simulation-based
inference (SBI)** of neural-mass-model parameters from empirical
functional connectivity (FC) data.

- **Forward model:** Wilson-Cowan SDE network (115 mouse-atlas
  regions, mouse SC + tract-length matrices, TVB BoldMonitor HRF).
- **Inverse problem:** SNPE-C (sbi 0.26.1) jointly with a small MLP
  feature-embedding network on PCA-reduced FC features.
- **Two-stage design:** Stage 1 infers global parameters `[P, Q,
  g_e, g_i]`; Stage 2 re-infers θ_bad ∪ local E/I couplings
  `[c_ee, c_ei, c_ie, c_ii]` after Stage 1 identifies which globals
  are poorly identified.
- **One-line summary:** "Given empirical FC, find Wilson-Cowan
  parameters that reproduce it under a TVB Bold monitor — with
  amortized neural posterior estimation."

---

## 1.2 REPOSITORY ROOT AND EXECUTION MODEL

| Item | Value |
|---|---|
| Absolute path | `/scratch/home/wog3597/vbi` |
| Branch | `refactor/02-simulation` |
| Last known commit | `a5b93bd  test: simulation import compat verified` |
| Python | 3.13.9 |
| GPU | NVIDIA H100 NVL (Hopper, sm_90, 95,830 MiB VRAM) |
| CUDA runtime | 12.5 (driver 555.42.02) |
| Notebook | `main.ipynb` (41 cells) |
| Production entry | `python main.py` → `pipelines.run_pipeline(...)` |
| Notebook entry | `setup_pipeline(cfg)` in cell[2], then run cells |

**Always run from the repo root.** `cwd == /scratch/home/wog3597/vbi`.
Never run from a subdirectory (`pipelines/` etc.) — the package
imports break under sub-directory execution.

---

## 1.3 PACKAGE ARCHITECTURE

```
            ┌──────────────────┐
            │     config.py    │ (single source of truth for constants)
            └──────────────────┘
                     ▲                      every package imports config
                     │
    ┌────────────────┼──────────────────────────────┐
    │                │                              │
┌───┴──────┐   ┌─────┴────┐   ┌────────────┐   ┌────┴────────┐
│simulation│ → │ features │ → │ inference  │ → │ evaluation  │
└──────────┘   └──────────┘   └────────────┘   └─────────────┘
    │                              ▲                ▲
    │                              │                │
    │                       ┌──────┴───────────┐    │
    └────────────────────── │ data_loader.py   │    │
                            └──────────────────┘    │
    ┌─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  pipelines/  (only package allowed  │
│   to import from all four)          │
└─────────────────────────────────────┘
         ▲
         │
    main.py / main.ipynb
```

| Package | Role | Key public names |
|---|---|---|
| `simulation/` | Wilson-Cowan GPU engine, delays, warmup, QC | `simulate_gpu_batch`, `simulate_single`, `compute_delay_matrix`, `warmup_run` |
| `features/` | FC, FCD, feature extraction (CPU-only math) | `compute_fc`, `fc_to_upper_tri`, `compute_sim_fcd_matrix`, `fcd_to_summary_stats`, `extract_features` |
| `inference/` | Parameter scaling, PCA, embedding MLP, SNPE-C | `ParameterScaler`, `FeaturePipeline`, `FCPCAScaler`, `FamilyScaler`, `FeatureEmbedding`, `run_stage1_snpe`, `run_stage2_snpe`, `save_artifacts`, `load_artifacts` |
| `evaluation/` | Metrics, validation, model selection, final test, plots | `fc_metrics`, `fcd_vec_rmse`, `evaluate_validation_stage1/2`, `select_best_model`, `final_test`, `plot_*`, `report_step1..14`, `print_final_summary` |
| `pipelines/` | Orchestration only (Stage 1 → 2 → eval) | `run_pipeline` |

**Sibling non-package modules:**

| File | Role |
|---|---|
| `config.py` | All numerical constants — never edit values without explicit user OK |
| `data_loader.py` | Mat-file loading, train/val/test split, per-subject dict assembly |
| `bold.py` | TVB-style BoldMonitor — sole source of `BoldMonitor`; imported by simulation submodules |
| `main.py` | Thin CLI wrapper around `pipelines.run_pipeline` |
| `main.ipynb` | 41-cell notebook: Setup → Integrated Debug Cell → 14 scientific steps |
| `debug.py` | Lightweight test harness (`python debug.py --basic`) |
| `pipeline_setup.py` | `PipelineConfig` dataclass + `setup_pipeline()` |

---

## 1.4 FILE STATUS LEGEND

Four categories. Apply this lens to every file before editing it.

| Icon | Category | Treatment |
|---|---|---|
| ✅ | AUTHORITATIVE | Production package source — read, plan, edit with verify |
| ⚠️ | COMPAT WRAPPER | Re-exports only, no logic — never edit; will be deleted after Tier 5 |
| 🔴 | DEAD CODE | Unreachable at runtime (package wins) — never edit; delete only after audit |
| 📋 | GENERATED / ARTIFACT | Output, backup, cache — do not commit, do not edit |

| File / dir | Status |
|---|---|
| `config.py` | ✅ AUTHORITATIVE — but values are frozen (R1) |
| `data_loader.py` | ✅ AUTHORITATIVE |
| `bold.py` | ✅ AUTHORITATIVE — sole source of `BoldMonitor` |
| `main.py`, `main.ipynb`, `pipeline_setup.py`, `debug.py` | ✅ AUTHORITATIVE |
| `simulation/__init__.py`, `simulation/wc_runner.py`, `simulation/delays.py`, `simulation/warmup.py`, `simulation/qc.py` | ✅ AUTHORITATIVE |
| `features/__init__.py`, `features/fc.py`, `features/fcd.py`, `features/extraction.py`, `features/screening.py` | ✅ AUTHORITATIVE |
| `inference/__init__.py`, `inference/scaling.py`, `inference/priors.py`, `inference/feature_pipeline.py`, `inference/embedding.py`, `inference/training_data.py`, `inference/snpe.py`, `inference/stage1.py`, `inference/stage2.py`, `inference/posterior.py`, `inference/diagnostics.py`, `inference/io.py`, `inference/_utils.py` | ✅ AUTHORITATIVE |
| `evaluation/__init__.py`, `evaluation/metrics.py`, `evaluation/validation.py`, `evaluation/model_selection.py`, `evaluation/final_test.py`, `evaluation/plots.py`, `evaluation/reports.py` | ✅ AUTHORITATIVE |
| `pipelines/__init__.py`, `pipelines/stage1_stage2.py` | ✅ AUTHORITATIVE |
| `simulator.py` | ⚠️ COMPAT WRAPPER (re-exports `simulation.*` + `features.*`) |
| `evaluate.py` | ⚠️ COMPAT WRAPPER (re-exports `evaluation`) |
| `inference.py` | 🔴 DEAD CODE — 55 KB monolith; package wins; has a `NameError` in `collect_stage2_data` |
| `fc.py`, `fcd.py`, `wc_runner.py`, `delays.py`, `warmup.py`, `qc.py`, `extraction.py`, `screening.py` (root) | 🔴 DEAD CODE — byte-identical to `features/*` or `simulation/*` (verified `diff` exit 0); zero callers |
| `__init__.py` (root) | 🔴 DEAD CODE — makes repo loadable as `vbi`; no callers |
| `debug_notebook.py` | ⚠️ LEGACY — notebook no longer imports it; keep on disk |
| `*.bak_*`, `output_mouse_mptp/*`, `__pycache__/*`, `*.npz`, `*.pkl`, `*.pt`, `patch2_remaining_simulator_imports.txt` | 📋 GENERATED |
| `MPTP_FC_115.mat`, `MPTP_SC_115.mat`, `participants.tsv`, `atlas_115_labels.txt` | 📋 DATA (read-only inputs) |
| `01_..._md` through `30_harness_*.md` | 📋 DOCS (read-only history) |

---

## 1.5 APPLIED PATCHES (permanent — verified — do not re-apply)

| ID | File | Line | Before | After | Verified by |
|---|---|---|---|---|---|
| **P-1** | `pipelines/stage1_stage2.py` | 38 | `import evaluate` | `import evaluation as evaluate` | doc 19; `pl.evaluate.__name__ == 'evaluation'` |
| **P-2** | `data_loader.py` | 278 | `from simulator import compute_delay_matrix` | `from simulation.delays import compute_delay_matrix` | doc 21; `simulator.compute_delay_matrix is simulation.delays.compute_delay_matrix` |
| **GPU-1** | `simulation/wc_runner.py` | (multiple) | per-step `.get()` × 600 K, eager `free_all_blocks()` | stride-batched `.get(out=cpu_buf)` + pinned host buffer + watermarked pool trim | helpers `_alloc_stride_buffers(cp, ...)`, `_trim_memory_pool(cp)` at module scope; lazy cupy import preserved |
| **NB-1** | `main.ipynb` | cell 3 + 4 | (old `## DEBUG CHECKS` + `run_all_checks()` cells removed) | `## Integrated VBI Pipeline Debug Cell` + 657-line staged debug runner with 9 stage gates | doc 23, doc 25; compile-check passes |
| **NB-2** | `main.ipynb` | cell 4 Section I | single `if RUN_FEATURE_EXTRACTION:` covering 3 subsections | split into `RUN_FEATURE_EXTRACTION` (I.1-3) and `RUN_FEATURE_EMBEDDING` (I.4-8) | doc 24, doc 25 |
| **DBG-1** | `debug.py` | line 488 + `run_basic_tests` | (no patch invariants test) | new `test_patch_invariants()` asserting P-1 + P-2 | doc 25; `python debug.py --basic` PASS count = 9 |

### Patch verification one-liners

```bash
# P-1
python -c "import pipelines.stage1_stage2 as pl; assert pl.evaluate.__name__=='evaluation'; print('P-1 OK')"

# P-2
python -c "
import inspect, data_loader
src = inspect.getsource(data_loader.get_subject_data)
assert 'from simulation.delays import compute_delay_matrix' in src
assert 'from simulator import compute_delay_matrix' not in src
print('P-2 OK')"

# GPU-1
python -c "
import inspect
from simulation import wc_runner
src = inspect.getsource(wc_runner)
assert '_alloc_stride_buffers' in src and '_trim_memory_pool' in src
print('GPU-1 OK')"

# debug.py PASS count
python debug.py --basic | tail -1   # → 'PASS: 9  |  FAIL: 3  |  SKIP: 0'
```

---

## 1.6 KNOWN PRE-EXISTING FAILURES (do not fix without authorization)

These failures are **catalogued, scoped, and intentionally untouched**.
A new session must not "fix" them as part of unrelated work.

| Failure | Sites | Why dormant | Authorization needed to fix |
|---|---|---|---|
| **FCD dual-role / dim mismatch** | 8 sites in 5 files (`inference/posterior.py:123,160`, `inference/diagnostics.py:77`, `inference/stage2.py:322`, `evaluation/metrics.py:123,211,252`, `evaluation/validation.py:135`, `evaluation/final_test.py:133`) | `config.USE_FCD = False` in production | "Proceed with Tier X" — must bundle all 8 sites together |
| **R3 violation** | `simulation/qc.py:25` (`from features.fc import …`) | Not exercised at runtime in package code path | "Proceed with R3 fix" — requires moving `qc.py` or aliasing |
| **`inference.py` `n_subj` NameError** | `inference.py:1391` in `collect_stage2_data` | Monolith is unreachable (package wins) — proves the monolith never ran | Never fix; delete `inference.py` in Tier 7 instead |
| **`debug.py --basic` FAILs** | 3 tests: `config consistency` (FCD_DIM=5 != FC_DIM=6555), `FeaturePipeline` ((200,50)≠(200,80) output dim), `FCD upper triangle (simulated)` ((6555,)≠(5,)) | All three trace to the same FCD dual-role surface | Bundled into Tier X authorization |

**Baseline acceptable `debug.py --basic` result: PASS=9, FAIL=3, SKIP=0.**
Any deviation from this is a regression you introduced.

---

## 1.7 KEY CONSTANTS — NEVER CHANGE WITHOUT EXPLICIT USER REQUEST

(All values verified live against `config.py`. R1 forbids silent change.)

| Constant | Value | Meaning | Source |
|---|---|---|---|
| `N_REGIONS` | 115 | Mouse atlas region count | `config.py:50` |
| `FC_DIM` | 6555 | FC upper-triangle vector length (`N*(N-1)/2`) | `config.py:51` |
| `FCD_DIM` | 5 | FCD summary-stats vector length (mean/std/q25/q50/q75) | `config.py` |
| `ANALYSIS_BOLD_T` | 240 | BOLD frames after transient cut (`(T_END − T_CUT)/(DT·DECIMATE)/(TR·1000/(DT·DECIMATE))`) | `config.py:105` — derived; never hardcode (R1) |
| `N_SIM` | 50,000 | Stage 1 simulations per subject | `config.py:94` |
| `N_SIM_S2` | 50,000 | Stage 2 simulations per subject | `config.py` |
| `GPU_BATCH` | 50,000 | Sims per GPU launch | `config.py:96` |
| `SEED` | 42 | Global RNG seed (numpy + torch) | `config.py` |
| `DT` | 0.5 | Integration step (ms) | `config.py:98` |
| `DECIMATE` | 2 | Sub-sample factor for stored neural output | `config.py` |
| `T_END` | 300,000.0 | Total sim length (ms) — 300 s | `config.py:99` |
| `T_CUT` | 60,000.0 | Transient cutoff (ms) — 60 s | `config.py:100` |
| `TR_SEC` | 1.0 | BOLD output TR (s) | `config.py` |
| `VELOCITY_M_PER_S` | 1.5 | Conduction velocity for delay matrix | `config.py:137` |
| `STAGE1_PARAMS` | `["P", "Q", "g_e", "g_i"]` | Stage 1 inferred globals | `config.py:151` |
| `STAGE1_PRIOR_LOW` | `[0.5, 0.0, 0.0, 0.0]` | Lower bounds (raw space) | `config.py` |
| `STAGE1_PRIOR_HIGH` | `[2.5, 2.0, 1.5, 1.5]` | Upper bounds (raw space) | `config.py` |
| `LOCAL_EI_PARAMS` | `["c_ee", "c_ei", "c_ie", "c_ii"]` | Stage 2 local couplings | `config.py` |
| `PCA_DIM_FC` | 300 | FC PCA components | `config.py:203` |
| `PCA_DIM_FCD` | 100 | FCD PCA components (unused while USE_FCD=False) | `config.py` |
| `PCA_EVR_THRESHOLD` | 0.90 | PCA pass criterion (cum. EVR) | `config.py:207` |
| `PCA_RECON_CORR_THRESH` | 0.95 | PCA pass criterion (recon r) | `config.py:209` |
| `EMBED_DIM` | 128 | FeatureEmbedding MLP output dim | `config.py:200` |
| `EMBED_HIDDEN` | 512 | FeatureEmbedding MLP hidden dim | `config.py:201` |
| `NDE_MODEL` | `"maf"` | sbi density estimator | `config.py:192` |
| `NDE_HIDDEN` | 128 | MAF hidden features | `config.py:190` |
| `NDE_TRANSFORMS` | 8 | MAF transform count | `config.py:191` |
| `N_POSTERIOR` | 2000 | Posterior samples per subject | `config.py` |
| `N_SBC` | 200 | SBC sample count | `config.py` |
| `N_TEST_RESIM` | 50 | Resimulations per test subject | `config.py` |
| `SBI_DEVICE` | `"cuda"` if torch+cuda available else `"cpu"` | sbi training device | `config.py:186` |
| `USE_FCD` | `False` | FCD feature toggle (dormant Tier X bug surface) | `config.py` |
| `SELECT_W_FC_CORR` | 1.0 | Model-selection weight: FC corr | `config.py` |
| `SELECT_W_FC_RMSE` | 0.5 | Model-selection weight: FC RMSE | `config.py` |
| `SELECT_W_FCD_RMSE` | 0.5 | Model-selection weight: FCD RMSE | `config.py` |
| `NUISANCE_METHOD` | `"posterior_sample"` (training); `"fix_mean"` (val/test, hardcoded) | Stage 2 nuisance handling | `config.py` + `evaluation/validation.py:134-141` |
| `WC_FIXED` | `{c_ee:16, c_ei:12, c_ie:15, c_ii:3, tau_e:8, tau_i:8, a_e:1.3, a_i:2.0, b_e:4, b_i:3.7, noise_amp:0.01, ...}` | WC default parameters | `config.py:125-138` |

---

## 1.8 GPU HARDWARE PROFILE

| Item | Value |
|---|---|
| GPU | NVIDIA H100 NVL |
| Arch | Hopper (sm_90) |
| VRAM | 95,830 MiB (~93.6 GB free typical) |
| HBM3 bandwidth | ~3.35 TB/s |
| PCIe | Gen5 x16 (~128 GB/s bidirectional) |
| FP32 peak | ~60 TFLOPS (dense) |
| FP16/BF16 Tensor Core peak | ~1,979 TFLOPS |
| SMs | 132 |
| L2 cache | 51 MB |
| Shared memory / SM | 228 KB (configurable) |
| ECC | Enabled |
| MIG | Disabled |
| CUDA runtime | 12.5 |
| Driver | 555.42.02 |
| cupy | 13.4.0 |
| sbi | 0.26.1 |
| torch | 2.6.0+cu124 |

### Pre-GPU-1 baseline (from `nvidia-smi` during production run)

| Metric | Value | Interpretation |
|---|---|---|
| GPU utilization | 15 % | Heavily under-utilized |
| Memory utilization | 9 % | HBM3 idle most of the time |
| Power draw | 135 W / 400 W | ~33 % of TDP — host-sync-bound, not compute-bound |
| Inner loop calls | `.get()` × 600,000 per simulation | PCIe sync was the dominant cost |

### Post-GPU-1 expectations

| Metric | Expected change |
|---|---|
| PCIe syncs per simulation | 600,000 → 75,000 (stride = `interim_istep` = 8) |
| GPU utilization | 15 % → 40–60 % |
| Wall-clock per simulation | −30 % to −50 % |
| BOLD output | bit-identical (mathematically equivalent reordering only) |

### Optimization recipe applied in GPU-1 (preserve, do not unwind)

1. **Stride-batched GPU staging**: `gpu_buf = cp.empty((stride, nn, ns))`,
   accumulate 8 consecutive E-snapshots on HBM3, then one DMA.
2. **Pinned host buffer**: `cp.cuda.alloc_pinned_memory(...)` →
   `np.frombuffer(...)`. Falls back to plain numpy if pinned
   alloc fails.
3. **Watermarked pool trim**: `_trim_memory_pool(cp)` only frees the
   cupy pool when `pool.total_bytes() > 80 GB`. Eager freeing on H100
   is wasteful.
4. **Lazy cupy import preserved**: `import cupy` stays inside
   function bodies, never at module scope (allows non-GPU import).

---

**End of context file. Read 27 next.**
