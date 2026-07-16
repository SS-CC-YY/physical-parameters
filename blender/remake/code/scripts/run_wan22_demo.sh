#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMAKE_ROOT="${REMAKE_ROOT:-$(cd "${CODE_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_FILE="${BUILD_FILE:-${CODE_ROOT}/builds/v1a_wan22_demo.yaml}"
RUN_ID="${RUN_ID:-v1a_wan22_demo_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REMAKE_ROOT}/outputs/${RUN_ID}}"

if [[ "${DETACHED:-0}" == "1" ]]; then
  mkdir -p "${RUN_DIR}"
  DETACHED=0 RUN_ID="${RUN_ID}" RUN_DIR="${RUN_DIR}" nohup bash "$0" >"${RUN_DIR}/master.log" 2>&1 &
  echo "pid=$!"
  echo "log=${RUN_DIR}/master.log"
  echo "run_dir=${RUN_DIR}"
  exit 0
fi

export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

prepare_args=(
  -m remake_benchmark prepare
  --build "${BUILD_FILE}"
  --run-dir "${RUN_DIR}"
  --workspace-root "${REMAKE_ROOT}"
)
sequence_args=(
  -m remake_benchmark sequence
  --run-dir "${RUN_DIR}"
)

if [[ -n "${MAX_JOBS:-}" ]]; then
  prepare_args+=(--max-jobs "${MAX_JOBS}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  sequence_args+=(--dry-run)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  prepare_args+=(--overwrite)
  sequence_args+=(--overwrite)
fi
if [[ "${FAIL_FAST:-1}" == "1" ]]; then
  sequence_args+=(--fail-fast)
fi
if [[ "${STOP_ON_INVALID:-0}" == "1" ]]; then
  sequence_args+=(--stop-on-invalid)
fi

"${PYTHON_BIN}" "${prepare_args[@]}"
"${PYTHON_BIN}" "${sequence_args[@]}"

echo "run_dir=${RUN_DIR}"
