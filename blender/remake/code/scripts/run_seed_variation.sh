#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export BUILD_FILE="${BUILD_FILE:-${CODE_ROOT}/builds/v1a_seed_variation_wan22.yaml}"
export RUN_ID="${RUN_ID:-v1a_seed_variation_wan22_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_full_generation.sh"
