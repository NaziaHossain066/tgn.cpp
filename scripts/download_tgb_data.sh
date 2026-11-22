#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 tgbl-wiki"
    exit 1
fi

# Reuse the Python dev env from `make python` (tguf + `data` group)
PYTHON="$PROJECT_ROOT/python/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Python env not found at $PYTHON. Run 'make python' first."
    exit 1
fi

DATASET_NAME="$1"
OUTPUT_PATH="$PROJECT_ROOT/data/$DATASET_NAME.tguf"

"$PYTHON" "$PROJECT_ROOT/tools/download_tgb_to_tguf.py" \
    --name "$DATASET_NAME" \
    --output "$OUTPUT_PATH"
