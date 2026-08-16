#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 tgbl-wiki"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

DATASET_NAME="$1"
OUTPUT_PATH="$PROJECT_ROOT/data/$DATASET_NAME.tguf"

uv run --no-project \
    --with py-tgb \
    --with numpy \
    --with "torch @ https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl" \
    --with tqdm \
    --with pandas==2.2.3 \
    --with-editable "$PROJECT_ROOT/python" \
    python "$PROJECT_ROOT/tools/download_tgb_to_tguf.py" \
    --name "$DATASET_NAME" \
    --output "$OUTPUT_PATH"
