#!/usr/bin/env bash
# Download Depth Anything 3 checkpoint for OpenBot 3D perception.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models/depth-anything-3"

mkdir -p "${MODEL_DIR}"

echo "Downloading Depth Anything 3 to ${MODEL_DIR} ..."

# Prefer hf-mirror for users in mainland China.
if command -v huggingface-cli >/dev/null 2>&1; then
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    huggingface-cli download ByteDance-Seed/Depth-Anything-3 \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
else
    echo "huggingface-cli not found. Please install: pip install huggingface-hub"
    exit 1
fi

echo "Download complete: ${MODEL_DIR}"
