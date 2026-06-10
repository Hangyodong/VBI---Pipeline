# cuBNM 0.1.0 ↔ numpy 2.x incompatibility — local rebuild notes

## Symptom
Running any `WCSimGroup.run()` (incl. cuBNM's own test instance) raised:

```
Error parsing arguments
TypeError: argument 2 must be numpy.ndarray, not numpy.ndarray
SystemError: <built-in function run_simulations> returned a result with an exception set
```

## Root cause
- Installed `numpy` is **2.3.5**.
- The cuBNM 0.1.0 **prebuilt wheel** (`cubnm/core.cpython-313-...so`) was compiled
  against **numpy 1.x** (`.so` references `numpy.core._multiarray_umath`).
- numpy 2.x relocated that module to `numpy._core._multiarray_umath`, so the
  extension's `import_array()` leaves `_ARRAY_API` NULL → every `PyArg_ParseTuple`
  `O!`/`PyArray_Type` check fails → "argument N must be numpy.ndarray, not numpy.ndarray".
- cuBNM's `pyproject.toml` build-requires pins `numpy<2`, so even a normal
  source build (`--no-binary cubnm`) recompiles against numpy 1.x → same break.

## Fix (local, reproducible)
1. Build deps in the env (so `--no-build-isolation` uses the installed numpy 2.3.5):
   ```
   pip install "versioneer[toml]"
   ```
2. Patch `src/ext/core.cpp` for numpy 2.x (numpy 1.x-only API removed) — add this
   shim immediately after `#include <numpy/arrayobject.h>`:
   ```cpp
   #ifndef PyArray_DOUBLE
   #define PyArray_DOUBLE NPY_DOUBLE
   #endif
   #ifndef PyArray_BOOL
   #define PyArray_BOOL NPY_BOOL
   #endif
   #ifndef PyArray_INT
   #define PyArray_INT NPY_INT
   #endif
   static inline void*    cubnm_PyArray_DATA(PyObject* o)        { return PyArray_DATA((PyArrayObject*)o); }
   static inline npy_intp cubnm_PyArray_DIM (PyObject* o, int i) { return PyArray_DIM ((PyArrayObject*)o, i); }
   static inline npy_intp cubnm_PyArray_SIZE(PyObject* o)        { return PyArray_SIZE((PyArrayObject*)o); }
   #define PyArray_DATA(obj)    cubnm_PyArray_DATA((PyObject*)(obj))
   #define PyArray_DIM(obj, i)  cubnm_PyArray_DIM ((PyObject*)(obj), (i))
   #define PyArray_SIZE(obj)    cubnm_PyArray_SIZE((PyObject*)(obj))
   ```
   (numpy 2.0 removed the `PyArray_DOUBLE/BOOL/INT` enum aliases and changed
   `PyArray_DATA/DIM/SIZE` to take `const PyArrayObject*` instead of `PyObject*`.)
3. Rebuild against numpy 2.x:
   ```
   pip install <patched-cubnm-src-dir> --no-build-isolation --no-deps --force-reinstall
   ```

## Verified
- `.so` now references `numpy._core._multiarray_umath` (numpy 2.x).
- `WCSimGroup.run()` executes; CPU run produced `sim_bold (N_sims, T, nodes)`.
- GPU run requires an actual CUDA device (the sandbox shell here exposes none:
  `CUDA_VISIBLE_DEVICES` empty, no `/dev/nvidia*`). On the H100 node, set
  `force_gpu=True` (the benchmark default).

## ⚠ Caveat
This is a **local rebuild**. A plain `pip install cubnm` will restore the broken
numpy-1.x wheel. Re-apply this patch + `--no-build-isolation` rebuild after any
cubnm reinstall, or keep a patched wheel.
