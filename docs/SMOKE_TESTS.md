# Smoke Tests — basis_regionwise + rwweib2

**Generated:** 2026-06-18

Lightweight, evidence-based checks for the HCP `basis_regionwise` + `rwweib2`
pipeline. None require a full VBI/SNPE run. Three are strictly non-training;
one (`smoke_e2e_basis_regionwise.py`) includes a **tiny** SNPE-C training pass
and must not be run accidentally.

> ⚠️ **Do NOT run full training to "smoke test".** Full `main_HCP.py` collects
> `N_SIM=2000` sims/round and trains SNPE-C on `FC_DIM=64,620` features — that
> is the real run, not a smoke. Use the scripts below instead.

---

## Test inventory

| # | File | Purpose | CPU/GPU | Training |
|---|---|---|---|---|
| 1 | `test_basis_mode_smoke.py` | 7 CPU unit checks of basis mode mechanics | CPU only | none |
| 2 | `test_basis_decoder.py` | `BasisParamDecoder` unit checks | CPU only | none |
| 3 | `tests/smoke/verify_basis_regionwise_rwweib2.py` | import + banner + decode + smallest cuBNM 2cpl run | GPU if present, else CPU fallback | none |
| 4 | `tests/smoke/smoke_e2e_basis_regionwise.py` | tiny end-to-end inference mechanics | CPU (forced) | ⚠️ tiny SNPE-C (N_SIM=16) |

---

## 1. `test_basis_mode_smoke.py` (CPU, no training)

**Verifies** (`test_basis_mode_smoke.py:1-19`):
- on-disk basis `(381,3)`; cortical slice → `(360,3)`; rezscore keeps const col,
  std→1 on myelin/gradient.
- `theta_dim == 4*3 == 12`; `coeff_names()[:3] == [g_LRE_const, g_LRE_myelin,
  g_LRE_gradient]`.
- `theta=zeros` → uniform **midpoint** maps (1.5 / 1.5 / 0.5 / 0.025).
- random theta → maps within bounds, `sigma >= 0`.
- `runner_rwweib_2cpl.build_param_lists` accepts `<param>_matrix` overrides;
  rejects wrong shape with `ValueError`.
- `engine_select.latent_wrap` decodes theta into `<param>_matrix` overrides and
  passes empty `param_names` (the exact path baseline_eval / ppc / plots use).
- `is_regionwise()` toggles with `PARAMETER_MODE`.

**Command:**
```bash
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  python -m pytest test_basis_mode_smoke.py -q
```
**Expected:** `7 passed`. **PASS** = all 7. **FAIL** = any assertion error.

---

## 2. `test_basis_decoder.py` (CPU, no training)

**Verifies** (`test_basis_decoder.py`): decoder built from `basis.npy`
(`n_regions=360`): `theta_dim==12`, const col kept, myelin rezscored; single +
batch decode shapes `(360,)` / `(5,360)`; bounds; `z=0`→midpoint; `prior_bounds`
returns `([-2]*12,[2]*12)`; `make_fixed_overrides` keys = `{param}_matrix`.

**Command:**
```bash
python test_basis_decoder.py
```
**Expected:** `test_basis_decoder: ALL PASS`.

---

## 3. `tests/smoke/verify_basis_regionwise_rwweib2.py` (non-training)

Mirrors `main_HCP.py` basis config (no data deps), runs 5 stages:
1. `from cubnm.sim import RWWEIB_2CPLSimGroup` import check.
2. startup banner + guard (`REQUIRE_BASIS` logic).
3. decoder shape check — active basis `(360,3)`, theta_dim 12, z=0 midpoints.
4. simulator matrix-override check — `{g_LRE,g_FFI,I_o,sigma}_matrix` present,
   `param_lists == decoded maps`.
5. smallest real cuBNM 2cpl run — 2 sims, 360 nodes, finite BOLD. Uses GPU when
   `torch.cuda.is_available()`, else CPU fallback.

> Has a `sys.path` repo-root injection so it runs from repo root.

**Command:**
```bash
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  BASIS_PATH=/mnt/d/hcp_basis/basis.npy \
  python tests/smoke/verify_basis_regionwise_rwweib2.py
```
(`BASIS_PATH` falls back to repo-local `basis.npy` if `/mnt/d/...` absent — the
script prints a NOTE.)

**Expected tail:**
```
########## (5) SMALL SMOKE RUN ##########
  torch.cuda.is_available() = <True|False>
  RUN OK  (GPU|CPU): 2 sims, BOLD[0] shape=(2, 360), finite=True
########## DONE ##########
```
**PASS** = `RUN OK` + `finite=True`, all earlier `present=True` / midpoints
exact. **GPU note:** CPU `RUN OK` does NOT prove the `force_gpu=True` path; run
on a GPU node to exercise it.

---

## 4. `tests/smoke/smoke_e2e_basis_regionwise.py` (⚠️ tiny training — approval first)

**Why gated:** runs a real SNPE-C training pass (tiny: `N_SIM=16`,
`N_REGIONS=12`, CPU-forced cuBNM via runner monkeypatch). It exercises the full
inference chain: prior/scaler → sample basis coeffs → wrapped 2cpl sim (decodes
coeffs → maps) → FC → `FeaturePipeline` → SNPE-C train → posterior → decode →
`_resimulate_and_score` FC corr. It bypasses the production collector
(`training_data.collect_training_data` hard-calls cupy → GPU-only) by driving
the same components minus the cupy wrapper.

**Command (run only with explicit approval):**
```bash
python tests/smoke/smoke_e2e_basis_regionwise.py
```
**Expected:** prints `fc_raw` finite, posterior `samples_raw=(32,12)`, decoded
posterior-mean maps, and `resim: k/10 OK  FC corr mean=...`. **PASS** =
completes without error and produces a finite FC corr. Runtime: tens of seconds
to a few minutes on CPU (tiny). It is NOT a full run, but it IS training — keep
it out of unattended/CI loops unless intended.

---

## Quick "is everything wired" sequence (all safe, no training)

```bash
# syntax
python -m py_compile engine_select.py main_HCP.py basis_decoder.py \
  evaluation/metrics.py evaluation/plots.py inference/posterior.py \
  tests/smoke/verify_basis_regionwise_rwweib2.py \
  tests/smoke/smoke_e2e_basis_regionwise.py

# unit + smoke
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  python -m pytest test_basis_mode_smoke.py -q
python test_basis_decoder.py

# end-to-end mechanics (non-training)
PARAMETER_MODE=basis_regionwise INFERENCE_MODEL=rwweib2 \
  BASIS_PATH=/mnt/d/hcp_basis/basis.npy \
  python tests/smoke/verify_basis_regionwise_rwweib2.py
```

Last verified **2026-06-18** on a CPU-only shell: py_compile OK, pytest `7
passed`, decoder `ALL PASS`, verify script `RUN OK (CPU) ... finite=True`.
