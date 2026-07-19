#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$CODE_ROOT/.." && pwd)"

VIDEOS_DIR="${1:-$WORKSPACE_ROOT/rebuild-test/test-videos/videos}"
OUTPUT_DIR="${2:-$WORKSPACE_ROOT/rebuild-test/outputs/reconstruction_handoff_run}"
WORKERS="${3:-4}"
OVERLAY_COUNT="${4:-5}"
PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" "$SCRIPT_DIR/check_reconstruction_handoff.py" \
  --workspace-root "$WORKSPACE_ROOT" \
  --videos "$VIDEOS_DIR" \
  --output "$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/run_reconstruction_mvp.py" \
  --workspace-root "$WORKSPACE_ROOT" \
  --videos "$VIDEOS_DIR" \
  --output "$OUTPUT_DIR" \
  --workers "$WORKERS" \
  --overlay-count "$OVERLAY_COUNT"

echo "Completed. Open: $OUTPUT_DIR/report/index.html"
