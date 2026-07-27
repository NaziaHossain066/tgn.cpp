#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?Usage: $0 cpu|cu126}"
case "$VARIANT" in
    cpu|cu126) ;;
    *)
        echo "Unknown variant '$VARIANT', expected 'cpu' or 'cu126'."
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
PYTHON_DIR="$PROJECT_ROOT/python"
BUILD_DIR="$PYTHON_DIR/.wheel_build"
DIST_DIR="$PYTHON_DIR/dist"

if [ ! -x "$PYTHON_DIR/.venv/bin/python" ]; then
    echo "python/.venv not found. Run 'make python' first to set up the dev environment."
    exit 1
fi

if [ "$VARIANT" = "cu126" ] && ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found on PATH. The CUDA-enabled torch installed in .venv needs a" \
         "matching CUDA toolkit to configure, even though this script only links" \
         "against prebuilt libraries (no .cu files are compiled)."
    echo "On the Mila cluster: module load cuda/12.6.0"
    echo "Elsewhere: install a CUDA toolkit matching the installed torch build."
    exit 1
fi

cd "$PYTHON_DIR"

# Swap torch in the existing venv directly, leaving pyproject.toml untouched
case "$VARIANT" in
    cpu)   TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu" ;;
    cu126) TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126" ;;
esac
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX_URL" \
    --reinstall "torch==2.10.0"

# Tag the version per variant so cpu/cu126 wheels never collide; restored on exit
ORIG_VERSION_CONTENT="$(cat VERSION)"
BASE_VERSION="$(printf '%s' "$ORIG_VERSION_CONTENT" | sed -E 's/^VERSION = "(.*)"$/\1/')"
restore_version() { printf '%s\n' "$ORIG_VERSION_CONTENT" > VERSION; }

RAW_WHEEL_DIR="$(mktemp -d)"
trap 'restore_version; rm -rf "$RAW_WHEEL_DIR"' EXIT

printf 'VERSION = "%s+%s"\n' "$BASE_VERSION" "$VARIANT" > VERSION

# --no-build-isolation reuses .venv's torch; -DTGN_ARCH=x86-64-v3 keeps the wheel portable
SKBUILD_CMAKE_ARGS="-DTGN_BUILD_PYTHON=ON;-DTGN_ARCH=x86-64-v3" \
    uv build --wheel \
        --python .venv/bin/python \
        --no-build-isolation \
        -C build-dir="$BUILD_DIR" \
        -o "$RAW_WHEEL_DIR"

RAW_WHEEL="$(ls "$RAW_WHEEL_DIR"/*.whl)"

# Exclude libtorch/CUDA so the wheel never bundles a second libtorch copy
uv tool run --from auditwheel auditwheel repair \
    --exclude 'libtorch*' \
    --exclude 'libc10*' \
    --exclude 'libcu*' \
    --exclude 'libnccl*' \
    --exclude 'libnvshmem*' \
    --exclude 'libnvJitLink*' \
    --exclude 'libgomp*' \
    -w "$DIST_DIR" \
    "$RAW_WHEEL"

echo "Self-contained $VARIANT wheel written to $DIST_DIR/"
