#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export BUILD_FILE="${BUILD_FILE:-${CODE_ROOT}/builds/v1a_seed_variation_wan22.yaml}"
export RUN_ID="${RUN_ID:-v1a_seed_variation_wan22_4gpu_$(date +%Y%m%d_%H%M%S)}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"

exec bash "${SCRIPT_DIR}/run_four_gpu_generation.sh"
