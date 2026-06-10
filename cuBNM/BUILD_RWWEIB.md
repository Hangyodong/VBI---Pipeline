# Building the RWW-EIB-FFI model into cuBNM

`cuBNM/rww_eib.yaml` defines a custom model (`RWWEIB`). The installed cubnm
wheel (v0.1.0) is **precompiled and ships no codegen driver**, so the generated
class `cubnm.sim.RWWEIBSimGroup` does not exist until you regenerate + rebuild
cubnm **from source** (same procedure that produced `WCVBISimGroup`).

`nvcc` is available at `/scratch/app/cuda/12.5/bin/nvcc`. Build on a GPU node.

## 1. Get the cuBNM source with the codegen driver

You need the cuBNM **source tree** (not the installed wheel) — the one used to
build `WCVBISimGroup`. Candidates: the `tvboptim_env` conda env, or clone:

```bash
git clone https://github.com/amnsbr/cuBNM.git
cd cuBNM
git checkout v0.1.0      # match the installed version (cubnm 0.1.0)
```

The codegen pieces live under `src/cubnm/codegen/` (recipes + mako templates)
and `src/cubnm/sim/`.

## 2. Drop in the recipe

Copy the recipe into the codegen recipes dir (same place `wc_vbi.yaml` /
`rww.yaml` live):

```bash
cp /scratch/home/wog3597/vbi/cuBNM/rww_eib.yaml  src/cubnm/codegen/recipes/
```

## 3. Run codegen

Use the same codegen entry point that generated WCVBI (check the source's
`codegen/` README / `model_specs.py`). Typically:

```bash
python -m cubnm.codegen            # or: python src/cubnm/codegen/generate.py
```

This emits `src/cubnm/sim/rwweib.py` + the C++/CUDA sources for `RWWEIB`.
Confirm the generated class name is **`RWWEIBSimGroup`** (model_name `RWWEIB`
+ `SimGroup`). The runner/adapter import exactly this name.

## 4. Rebuild from source

```bash
CC=gcc CXX=g++ pip install -e . --no-build-isolation -v
# GPU build; ensure CUDA on PATH:
export PATH=/scratch/app/cuda/12.5/bin:$PATH
```

## 5. Verify

```bash
python -c "from cubnm.sim import RWWEIBSimGroup as G; \
print(G.global_param_names, G.regional_param_names, G.state_names)"
# expect regional: ['g_LRE','g_FFI','sigma','J_N',...]  states: ['S_e','S_i']

python /scratch/home/wog3597/vbi/test_rwweib_2node.py   # Part B should now run
```

## Caveats / things to check during the build

1. **No `global_param`.** This model declares all params as `regional_param`
   (the long-range gain is applied in step_equations: `g_LRE * J_N * globalinput`).
   If the codegen requires ≥1 `global_param`, either (a) the kernel does not —
   fine, or (b) promote `g_LRE` to `global_param` (shape `(n_sims,)`). If you do
   the latter, update `cuBNM/runner_rwweib.py:build_param_lists` to set
   `param_lists["g_LRE"]` as a 1-D `(n_sims,)` array instead of `(n_sims, nodes)`.

2. **Single noise channel.** Only `noise_e` is declared (noise on `S_e` only,
   matching the tvboptim example). If the codegen/kernel expects one noise per
   state var (`n_noise == n_state`), add a `noise_i` channel and append
   `+ sqrt_dt * noise_i * sigma` to the `S_i` update — **document that this is a
   cuBNM stochastic variant, not the exact tvboptim (S_e-only) noise**.

3. **Transfer-function singularity.** `H = (a*x - b)/(1 - exp(-d(a*x-b)))` has a
   removable singularity at `a*x == b`. We use direct division (as upstream
   `rww`). If the build's codegen guards it (epsilon / Taylor branch), keep that
   guard; additive noise prevents an exact hit in practice.

4. **`conn_state_var: S_e`** and `bold_state_var: S_e` — single coupling, no
   `SC @ S_i`, no second `conn_state_var`. No CUDA-core change required.

## Using it after the build

In `config.py` set `INFERENCE_MODEL = "rwweib"` (switches `STAGE1_PARAMS` to
`["g_LRE","g_FFI","sigma"]` + priors). In the notebook Step 2, pass
`engine="rwweib"` to `collect_training_data(...)`.
