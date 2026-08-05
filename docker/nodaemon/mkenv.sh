#!/usr/bin/env bash
# Stage A — build the runtime environment natively, no container involved.
#
# Creates a self-contained conda env, compiles the cuBNM fork against it, and
# verifies the result. docker/nodaemon/mkimage.py then packages this env into
# an image layer. See docker/nodaemon/README.md for why this path exists.
#
#   ./mkenv.sh                       # builds into /var/tmp/vbi-img-build/env
#   BUILD=/somewhere ./mkenv.sh
set -euo pipefail

BUILD="${BUILD:-/var/tmp/vbi-img-build}"
ENVP="$BUILD/env"
CUBNM_SRC="${CUBNM_SRC:-/scratch/home/wog3597/cubnm_build}"
mkdir -p "$BUILD"

# The host's ~/.local/lib/python3.13/site-packages lands on the new env's
# sys.path too, because both are python 3.13. Without this, pip reports every
# dependency as "already satisfied" and installs nothing, and the interpreter
# then imports the host's copies. Needed for every python/pip call below.
export PYTHONNOUSERSITE=1

echo "=== [1/4] conda env: python 3.13 + gsl ==="
[ -x "$ENVP/bin/python" ] || conda create -y -p "$ENVP" python=3.13 -c conda-forge
conda install -y -p "$ENVP" -c conda-forge gsl=2.7
PY="$ENVP/bin/python"
"$PY" -V

echo "=== [2/4] python stack ==="
# ONE resolve, with torch pinned to +cu124 and download.pytorch.org as an EXTRA
# index. Installing torch separately from sbi lets pip re-resolve torch off
# PyPI while satisfying sbi, which silently swaps in a cu13 build needing a far
# newer driver than the cu124 stack this pipeline was validated on.
"$PY" -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0+cu124" \
    "numpy==2.3.5" "scipy==1.17.1" "pandas==2.3.3" "matplotlib==3.10.6" \
    "scikit-learn==1.7.2" "sbi==0.26.1" "h5py==3.15.1" "tqdm==4.67.1"

echo "=== [3/4] build the cuBNM fork ==="
[ -d "$CUBNM_SRC/.git" ] || { echo "no git repo at $CUBNM_SRC" >&2; exit 1; }
[ -z "$(git -C "$CUBNM_SRC" status --porcelain)" ] || {
    echo "cuBNM fork has uncommitted changes; git archive would not see them" >&2
    exit 1
}
CUBNM_COMMIT=$(git -C "$CUBNM_SRC" rev-parse --short HEAD)
rm -rf "$BUILD/cubnm-src"; mkdir -p "$BUILD/cubnm-src"
git -C "$CUBNM_SRC" archive HEAD | tar -x -C "$BUILD/cubnm-src"

# setup.py links libgsl.a / libgslcblas.a and -lcudart_static, which it finds
# through these paths, and turns on the GPU build if nvcc is merely on PATH —
# no GPU is needed to compile.
export CPATH="$ENVP/include:${CPATH:-}"
export LIBRARY_PATH="$ENVP/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENVP/lib:${LD_LIBRARY_PATH:-}"
export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+fork.${CUBNM_COMMIT}"
command -v nvcc >/dev/null || { echo "nvcc not on PATH — would build CPU-only" >&2; exit 1; }

"$PY" -m pip install --no-cache-dir build
(cd "$BUILD/cubnm-src" && "$PY" -m build --wheel .)
"$PY" -m pip install --no-cache-dir "$BUILD"/cubnm-src/dist/*.whl

echo "=== [4/4] verify ==="
"$PY" - <<'PY'
import sys, glob, os
assert not any("/.local/" in p for p in sys.path), "host user-site leaked into sys.path"
import numpy, scipy, sklearn, pandas, h5py, matplotlib, tqdm, sbi, torch, cubnm
from cubnm.sim import RWWEIB_2CPLSimGroup
assert torch.__version__.startswith("2.6.0+cu124"), f"torch is {torch.__version__}"
assert torch.version.cuda == "12.4", f"cuda is {torch.version.cuda}"
sp = os.path.dirname(numpy.__file__).rsplit("/numpy", 1)[0]
stray = [os.path.basename(p) for p in glob.glob(f"{sp}/nvidia*") if "cu13" in p]
assert not stray, f"cu13 wheels leaked in: {stray}"
# GPU_ENABLED really compiled in, rather than a silent CPU-only fallback
so = glob.glob(f"{sp}/cubnm/core*.so")[0]
blob = open(so, "rb").read()
for sym in (b"cudaMalloc", b"cudaGetDeviceCount", b"RWWEIB_2CPL"):
    assert sym in blob, f"{sym.decode()} missing from {so} — CPU-only build?"
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
print(f"  sbi {sbi.__version__}  numpy {numpy.__version__}  scipy {scipy.__version__}")
print("  cubnm built with GPU support, RWWEIB_2CPL present")
PY
echo "=== stage A done ==="; du -sh "$ENVP"
