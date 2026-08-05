# VBI-SBI Brain Parameter Inference Pipeline

Whole-brain parameter inference with simulation-based inference (SNPE-C).
GPU forward simulation via **cuBNM**, amortized posterior via **sbi**.

Infers region-wise RWW-EIB parameters — encoded as **myelin/gradient basis
coefficients** — from HCP functional connectivity (FC).

Entrypoint: **`main_HCP.py`** — HCP human, **RWWEIB_2CPL** model, 360 cortical
regions.

> **Source of truth = code + config.** This repo carries only the code and
> assets a full `main_HCP.py` run actually reaches — 60 tracked files. Dead
> engine adapters, the benchmark harness, standalone experiments, and the
> superseded whole-brain assets are kept locally but not published (see
> `.gitignore`, bottom block, for the exact list and how to restore one).
>
> Consequence: **only `INFERENCE_MODEL=rwweib2` ships.** `engine_select.py`
> still routes `wc`/`cubnm`/`rww`/`rwweib`/`rwweibdelay`, but their adapter
> modules are not in the repo, so selecting one raises `ModuleNotFoundError`
> in a fresh clone. Same for `RUN_CUBNM_BENCHMARK = True`.

---

## Data

All data lives under **`HCP_Data/`** (`config.DATA_DIR`). Large `.mat` files
exceed GitHub's 100MB limit and are **not** committed — obtain them separately
and drop them into `HCP_Data/`.

### In-repo (committed)
| File | Size | Contents |
|------|------|----------|
| `HCP_Data/HCP_CABNP381_SC_first100.mat` | 43M | **active SC** — CAB-NP 381-region, first 100 subjects (`SC_DATASET=cabnp381`) |
| `HCP_Data/basis_cortex.npy` | 9K | **active basis** — `(360,3) = [const, myelin_z, gradient_z]`, cortex-only |
| `HCP_Data/gradient_subjects_cortex.npy` | 288K | `(100,360)` cortex-only principal FC gradient (built by `extract_gradient_cortex.py`) |
| `HCP_Data/myelin_subjects.npy` | 300K | `(100,381)` per-subject myelin (T1w/T2w) maps — basis input |

### External (NOT committed — get separately)
| File | Size | Contents |
|------|------|----------|
| `HCP_Data/HCP_FC.mat` (var `C`) | 1.1G | **active FC** target — per-subject 381-region FC. **Required** — nothing runs without it |
| `HCP_Data/HCP_SC.mat` | 224M | full HCP SC (only for `SC_DATASET=hcp_v73`) |

Cortical-only: 381 → first **360** Glasser regions (21 subcortical dropped).
SC scaling via `VBI_SC_SCALE` (`main_HCP.py` forces `maxnorm`).

The cortex-only gradient matters: the whole-brain gradient's leading component
is driven by the 21 subcortical regions that then get sliced away, so
`basis_cortex.npy` re-derives it on the 360 cortical regions alone.

---

## HCP pipeline (`main_HCP.py`)

### Model — RWWEIB_2CPL (cuBNM)
Two-population reduced Wong-Wang with **two independent connectome couplings**
(E driven by `SC@S_E`, I driven by `SC@S_I`):

```
I_E = w_E·I_o + w_p·J_N·S_E + g_LRE·J_N·(SC@S_E) − J_i·S_I
I_I = w_I·I_o +     J_N·S_E + g_FFI·J_N·λ_IE·(SC@S_I) − S_I
dS_E/dt = −S_E/τ_E + (1−S_E)·γ_E·H_E(I_E) + σ·ξ
dS_I/dt = −S_I/τ_I +          γ_I·H_I(I_I) + σ·ξ
```
BOLD = Balloon-Windkessel HRF on `S_E`.

### Parameterization — `basis_regionwise` (default)
Four region-wise params (`g_LRE, g_FFI, I_o, sigma`) are **not** inferred
directly. Instead each is a linear combination of 3 basis maps
`[const, myelin, gradient]`, so the inferred vector `theta` has **12 = 4×3**
coefficients (`theta_dim=12`).

Decode (`basis_decoder.py`): per param,
```
z   = beta · basis.Tᵀ          # (S,360) region map from 3 coeffs
map = mid + half·tanh(z)        # mid=(lo+hi)/2, half=(hi-lo)/2
```
Basis bounds (`BASIS_BOUNDS`):
`g_LRE(0,3) g_FFI(0,3) I_o(0.30,0.45) sigma(0,0.05)`.
Prior: scaled `BoxUniform[-1,1]^12`; raw coeff `(-10,10)`. `theta=0` → param
midpoints. Bounds/coeff-order are load-bearing — see `basis_decoder.py`.

**`I_o` bound is not cosmetic.** The rWW-EI critical operating point (isolated
node, no FIC) is `I_o≈0.382`. The old `(0,1)` bound put the decoder midpoint at
0.5, so ~51% of prior-sampled `(sim, region)` entries landed saturated and only
~3% in the critical band — SNPE trained mostly on degenerate FCs. `(0.30,0.45)`
centres the midpoint at 0.375. Revert with `IO_BOUND_LO=0 IO_BOUND_HI=1`.

### Dataflow (one line)
```
theta(12) → decode → {g_LRE,g_FFI,I_o,sigma}(S,360) → RWWEIB_2CPLSimGroup(GPU)
  → BOLD(1200 TR,360) → compute_fc (raw Pearson r) → upper-tri (64,620)
  → FeaturePipeline PCA-256 whitened → SNPE-C (MAF)
```
N_SIM is **per-subject** → real train tensor = N_TRAIN × N_SIM = 70 × 1000.
Sim length: `T_END=864s` (1200 TR @ TR=0.72s), `T_CUT=30s` → **1158 TR analyzed**.

### ⚠️ Default gotchas (env overrides in `main_HCP.py`)
- `SMOKE=1` is the **default** → bare `python main_HCP.py` is a **tiny toy**
  (4/2/1/1 subjects, 64 sims). Real run = **`SMOKE=0`** (100/70/10/20, 1000 sims).
- `SC_CONDITION=1` is now the **default** → posterior is `q(theta | FC, SC)` via
  `MultiChannelMatrixEmbedding`. `SC_CONDITION=0` restores the FC-only baseline.
- `GROUP_AVG_FC=0` → **per-subject** FC (not group-averaged).
- `USE_DELAYS=0` → delays OFF (computed but not fed; ~0 BOLD-FC effect for 5.3× cost).
- `GEOMETRY_COUPLING=0` → OFF.

### Run
```bash
# REAL run (GPU node) — ~2-3 h end to end. This is the configuration that
# produced the best measured result; the defaults are already correct:
SMOKE=0 python main_HCP.py

# smoke / CPU-safe checks (no training):
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  python -m pytest test_basis_mode_smoke.py -q
```

Env overrides:

| Group | Vars |
|---|---|
| sizes | `SMOKE, N_SUBJECTS, N_TRAIN, N_VAL, N_TEST, N_SIM, GPU_BATCH` |
| data | `SC_DATASET, SC_FILE, BASIS_PATH, GROUP_AVG_FC, VBI_SC_SCALE` |
| model | `PARAMETER_MODE, USE_DELAYS, G_BOUND_HIGH, IO_BOUND_LO, IO_BOUND_HI, COEFF_PRIOR_LO, COEFF_PRIOR_HI` |
| inference | `SC_CONDITION, SC_FUSION, FC_TOKEN_DROPOUT, FC_PCA_DIM, FC_PCA_WHITEN` |
| output | `EXPORT_FC_CSV` |

### What a real run writes to `output_hcp/`
| Artifact | Contents |
|---|---|
| `test_fc_comparison_{1..10}.png` | per test subject: observed FC vs **mean of `N_TEST_RESIM` resimulated FCs**, with `corr` / `RMSE` in the title (2 subjects per image) |
| `fc_csv/sim_fc_<sid>.csv`, `emp_fc_<sid>.csv` | the same two matrices as raw CSV |
| `fc_csv/node_fit_<sid>.csv`, `node_fit_summary.csv` | per-region sim-vs-emp FC-row correlation — which nodes are recovered and which are not |
| `features_stage1.npz` | cached `(theta_scaled, fc_raw)` simulation pairs; a matching cache key **skips Step 2 entirely** |

### Measured results

Test-set FC correlation, 20 held-out subjects, bootstrap 95% CI. All runs:
70 train subjects, `N_SIM=1000` (69,988 sims), `N_TEST_RESIM=10`, stage 1.

| Run | Change from default | per-draw mean | expected-FC | mean-θ |
|---|---|---|---|---|
| 2026-07-02 | **default — best** | **0.3013** [0.2895, 0.3133] | **0.3827** [0.3547, 0.4098] | **0.3143** |
| 2026-07-01 | default | 0.2780 [0.2651, 0.2911] | 0.3609 [0.3298, 0.3877] | 0.3019 |
| 2026-07-03 | `G_BOUND_HIGH=6` | 0.2583 [0.2434, 0.2733] | 0.3586 [0.3229, 0.3942] | 0.2570 |
| 2026-07-02 | `SC_FUSION=film` | 0.0752 [0.0647, 0.0861] | 0.1672 [0.1429, 0.1913] | 0.0281 |

Read the three columns as three estimators of the same thing, not as three
results: *per-draw mean* scores each posterior draw's resim separately and
averages the 20×10 scores; *expected-FC* averages the 10 resim FC matrices
first and scores once, so it reads highest; *mean-θ* fixes one θ = the
posterior mean and scores that.

**Simulation is deterministic, which changes what these mean.** cuBNM
pre-computes ONE noise array per `SimGroup` — `bnm.cu` sizes it
`nodes × bw_it × inner_it × n_noise`, with no `n_sims` dimension, and the
kernel's `noise_idx` is a function of (timestep, inner step, node) only, never
of the simulation index. Every sim in a batch therefore sees the identical
noise stream, and `sim_seed` is hardcoded to 42. Consequences:

- Resimulating the same θ gives a bit-identical BOLD and FC. Reruns are
  reproducible; there is no run-to-run scatter to average away.
- The expected-FC / per-draw gap (0.38 vs 0.30) is **not** noise cancellation.
  It is averaging over 10 *different* posterior draws, which yields a smoother
  posterior-predictive mean FC that correlates better with empirical FC than
  any single draw does.
- The per-subject `±` on the per-draw score is entirely posterior spread, with
  no noise component in it.
- `fc_corr_meantheta` resims one fixed θ `N_TEST_RESIM` times, so it produces
  N identical FCs and averages them — a no-op costing N× the sims it needs.
  The code comment there still describes it as noise-averaging.

Per-subject spread is wide — the 2026-07-03 run ranged from 0.16 to 0.53
expected-FC across its 20 test subjects.

**Two ablations that did not work.** `G_BOUND_HIGH=6` re-opens the coupling
bound on the theory that the `(0,3)` cap under-couples; it measured worse, so
`3.0` stays the default. `SC_FUSION=film` measured much worse — the FiLM gate
is a sound fix for the "SGD zeroes out the additive SC branch" problem in
principle, but in practice it cost more than the SC signal was worth here.
Both runs completed cleanly with no errors; these are real results, not
crashes. `add` remains the default fusion.

Roughly 0.38 expected-FC appears to be this model's ceiling, not the
inference's: posterior-predictive FC sits close to what the simulator can
produce at all. Raising it means changing the generative model (operating
point, FIC, criticality), not the posterior.

> The right-hand panel is **not** an optimizer fit. This is amortized SBI: the
> posterior is trained once on simulated `(theta, FC)` pairs, then applied to a
> held-out subject's real FC in a single forward pass; the panel is a
> resimulation from those posterior draws. There is no per-subject gradient
> descent on FC error, and the reported `corr` is model-limited, not
> inference-limited.

### Docker
```bash
./docker/build.sh                          # ~150 MB context, 30-60 min first build
CUBNM_SRC=/path/to/cubnm ./docker/build.sh # if the fork is not at ../cubnm_build

docker run --rm --gpus all -e SMOKE=0 \
  -v "$PWD/HCP_Data:/app/HCP_Data" -v "$PWD/output_hcp:/app/output_hcp" vbi-hcp:latest
```
Two stages: the first compiles the cuBNM fork with nvcc (no GPU needed at build
time), the second is a CUDA runtime image with the wheel plus the 60 tracked
repo files. Expect ~8-10 GB. `build.sh` stages both source trees as
`git archive` tarballs, so `output_hcp/` never enters the context; it refuses to
build if the fork's tree is dirty, since `git archive` would silently omit
uncommitted kernel changes. Data is mounted, never baked in.

`cupy`, `brainspace` and `tvb` are left out of the image — every import of them
is lazy, and all 50 library modules were verified to import with those three
names blocked. `cupy` alone would add ~1.5 GB for a Wilson-Cowan path that
`rwweib2` never takes.

Two caveats worth knowing before you rely on the image for reproduction. The
RWWEIB_2CPL model exists only in a **local fork of cuBNM that is not published
anywhere** — not on PyPI, and its two kernel commits are not on the upstream
remote, so the image cannot be rebuilt without that working copy. And cuBNM's
own Dockerfile warns that identical builds on different hardware can generate
different noise, so containerising does not by itself guarantee the numbers in
the results table reproduce.

### cuBNM rebuild (after yaml / kernel changes)
```bash
cd /scratch/home/wog3597/cubnm_build
python codegen/generate_models.py
pip install -e . --no-build-isolation
```
The multi-coupling kernel surgery (`conn_state_vars` support) lives in a
separate cuBNM fork (`cubnm_build/`), required to build RWWEIB_2CPL.

---

## Key files

| File | Purpose |
|------|---------|
| `main_HCP.py`                 | HCP pipeline driver; config @50-210, basis dispatch |
| `basis_decoder.py`            | `BasisParamDecoder`, `get_decoder`; myelin/gradient decode |
| `param_decoder.py`            | `decode_to_param_maps` dispatch |
| `engine_select.py`            | route active model (rwweib2 / rwweib / rww / vbi) |
| `data_loader_hcp.py`          | FC/SC load, 381→360 slice, SC scale, group-avg FC |
| `extract_gradient_cortex.py`  | build `basis_cortex.npy` — cortex-only principal FC gradient |
| `cuBNM/rww_eib_2cpl.yaml`     | RWWEIB_2CPL model spec (2 couplings, `conn_state_vars:[S_E,S_I]`); codegen input |
| `cuBNM/runner_rwweib_2cpl.py` | `build_param_lists`, `run_cubnm_rwweib2_batch`; imports the generated `cubnm.sim.RWWEIB_2CPLSimGroup` |
| `cuBNM/simulate_rwweib_2cpl.py` | `simulate_gpu_batch` — the only engine adapter shipped |
| `inference/feature_pipeline.py` | FC PCA-256 whiten |
| `inference/embedding.py`      | `MultiChannelMatrixEmbedding` — SC-conditioned encoder (`add` default; `film` implemented but measured worse) |
| `inference/snpe.py`           | SNPE-C; MAF |
| `evaluation/`                 | validation/test metrics, plots (engine-routed) |
| `evaluation/export_fc_csv.py` | test sim/emp FC + per-region node-fit CSV export |

---

## Style
Library modules target `pycodestyle --max-line-length=88`. `main_HCP.py` is
converted from a notebook and does not conform.
