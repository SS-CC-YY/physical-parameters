#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEOS_ROOT="${VIDEOS_ROOT:?Set VIDEOS_ROOT to the directory containing the 27 V1A MP4 files}"
GPU_ID="${GPU_ID:-0}"
MAX_JOBS="${MAX_JOBS:-}"
AUDIT_JSONL="${AUDIT_JSONL:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/v1a_hybrid}"
RUNTIME_MANIFEST="$ROOT/work/v1a_hybrid_manifest.jsonl"

eval "$(conda shell.bash hook)"
conda activate "$ROOT/env"
cd "$ROOT"

python scripts/build_manifest.py \
  --videos-root "$VIDEOS_ROOT" \
  --output "$RUNTIME_MANIFEST" \
  --strict

args=(
  --videos "$VIDEOS_ROOT"
  --manifest "$RUNTIME_MANIFEST"
  --output-root "$OUTPUT_ROOT"
)
if [[ -n "$AUDIT_JSONL" ]]; then
  args+=(--audit-jsonl "$AUDIT_JSONL")
fi
if [[ -n "$MAX_JOBS" ]]; then
  args+=(--max-jobs "$MAX_JOBS")
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_hybrid_v1a.py "${args[@]}"
