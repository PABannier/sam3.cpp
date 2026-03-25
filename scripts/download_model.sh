#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/models"

mkdir -p "${MODEL_DIR}"

echo "=== Downloading SAM3 checkpoint from HuggingFace ==="
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

python3 -c "
from huggingface_hub import hf_hub_download
import os
dst = '${MODEL_DIR}'
print('Downloading sam3.pt ...')
hf_hub_download('facebook/sam3', 'sam3.pt', local_dir=dst)
print(f'Saved to {dst}/sam3.pt')
"

echo ""
echo "=== Converting to ggml format ==="
python3 "${PROJECT_DIR}/convert_sam3_to_ggml.py" \
    --model "${MODEL_DIR}/sam3.pt" \
    --output "${MODEL_DIR}/sam3-f16.ggml" \
    --ftype 1

echo ""
echo "Done. Model saved to ${MODEL_DIR}/sam3-f16.ggml"
ls -lh "${MODEL_DIR}/sam3-f16.ggml"
