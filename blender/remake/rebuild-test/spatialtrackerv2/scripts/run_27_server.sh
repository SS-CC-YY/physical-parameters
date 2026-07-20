#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
MAX_JOBS="${MAX_JOBS:-}"
VIDEOS_ROOT="${VIDEOS_ROOT:-}"

eval "$(conda shell.bash hook)"
conda activate "$ROOT/env"
cd "$ROOT"

manifest_args=(--strict)
if [[ -n "$VIDEOS_ROOT" ]]; then
  manifest_args+=(--videos-root "$VIDEOS_ROOT")
fi
python scripts/build_manifest.py "${manifest_args[@]}"
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/preflight.py

args=()
if [[ -n "$MAX_JOBS" ]]; then
  args+=(--max-jobs "$MAX_JOBS")
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_batch.py "${args[@]}"
