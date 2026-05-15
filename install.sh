#!/usr/bin/env bash
# =============================================================================
# Mouse MPTP VBI-SBI pipeline — dependency installer
#
# Usage:
#   bash install.sh            # CUDA 12.x (default)
#   bash install.sh --cuda11   # CUDA 11.x
#   bash install.sh --cpu      # CPU only (simulation will be slow)
#   bash install.sh --dry-run  # print commands without running them
#
# Requires: Python 3.10+, pip
# Tested on: H100 NVL, CUDA 12.6, Ubuntu 22.04
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

CUDA_TAG="cu121"          # used for PyTorch index
CUPY_PKG="cupy-cuda12x"   # cupy wheel name
CPU_ONLY=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --cuda11)  CUDA_TAG="cu118"; CUPY_PKG="cupy-cuda11x" ;;
        --cpu)     CPU_ONLY=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

run() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        echo "+ $*"
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------

PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")

echo "============================================================"
echo "  Mouse MPTP VBI-SBI — install"
echo "  Python  : $PY_VER"
echo "  CUDA tag: $CUDA_TAG"
echo "  CPU only: $CPU_ONLY"
echo "  Dry run : $DRY_RUN"
echo "============================================================"

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python >= 3.10 required (got $PY_VER)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: pip upgrade
# ---------------------------------------------------------------------------

echo ""
echo "--- Step 1: upgrade pip / setuptools / wheel ---"
run pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Step 2: Core scientific stack
# ---------------------------------------------------------------------------

echo ""
echo "--- Step 2: core scientific stack ---"
run pip install \
    "numpy>=1.26,<3.0" \
    "scipy>=1.11" \
    "pandas>=2.0" \
    "matplotlib>=3.7"

# ---------------------------------------------------------------------------
# Step 3: PyTorch (GPU vs CPU)
# ---------------------------------------------------------------------------

echo ""
echo "--- Step 3: PyTorch ---"
if $CPU_ONLY; then
    run pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    run pip install torch \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# ---------------------------------------------------------------------------
# Step 4: scikit-learn
# ---------------------------------------------------------------------------

echo ""
echo "--- Step 4: scikit-learn ---"
run pip install "scikit-learn>=1.3"

# ---------------------------------------------------------------------------
# Step 5: SBI (sbi==0.26.1)
# ---------------------------------------------------------------------------

echo ""
echo "--- Step 5: sbi ---"
run pip install "sbi==0.26.1"

# ---------------------------------------------------------------------------
# Step 6: VBI Wilson-Cowan backend (vbi==0.4.3)
# ---------------------------------------------------------------------------

echo ""
echo "--- Step 6: vbi ---"
run pip install "vbi==0.4.3"

# ---------------------------------------------------------------------------
# Step 7: cupy (skipped if --cpu)
# ---------------------------------------------------------------------------

if $CPU_ONLY; then
    echo ""
    echo "--- Step 7: cupy SKIPPED (--cpu mode) ---"
else
    echo ""
    echo "--- Step 7: cupy ($CUPY_PKG) ---"
    run pip install "$CUPY_PKG"
fi

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if ! $DRY_RUN; then
    echo ""
    echo "============================================================"
    echo "  Verification"
    echo "============================================================"
    python - << 'PYEOF'
import sys

def check(label, fn):
    try:
        fn()
        print(f"  OK   {label}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")

check("numpy",        lambda: __import__("numpy"))
check("scipy",        lambda: __import__("scipy"))
check("pandas",       lambda: __import__("pandas"))
check("matplotlib",   lambda: __import__("matplotlib"))
check("torch",        lambda: __import__("torch"))
check("sklearn",      lambda: __import__("sklearn"))
check("sbi",          lambda: __import__("sbi"))
check("vbi",          lambda: __import__("vbi"))

try:
    import cupy as cp
    dev = cp.cuda.runtime.getDeviceCount()
    print(f"  OK   cupy  (devices={dev})")
except Exception as e:
    print(f"  SKIP cupy: {e}")

try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  OK   torch CUDA — {name} ({mem:.1f} GB)")
    else:
        print("  WARN torch: CUDA not available (CPU only)")
except Exception as e:
    print(f"  FAIL torch cuda check: {e}")

try:
    from vbi.models.cupy.wilson_cowan import WC_sde  # noqa: F401
    print("  OK   vbi WC_sde (cupy backend)")
except ImportError:
    print("  WARN vbi WC_sde cupy backend not available")
    try:
        from vbi.models.wilson_cowan import WC_sde  # noqa: F401
        print("  OK   vbi WC_sde (numpy fallback)")
    except ImportError as e:
        print(f"  FAIL vbi WC_sde: {e}")

print()
print("  Run  python debug.py --basic  to verify the pipeline itself.")
PYEOF
fi

echo ""
echo "Done."
