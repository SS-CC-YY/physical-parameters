#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMAKE_ROOT="${REMAKE_ROOT:-$(cd "${CODE_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_FILE="${BUILD_FILE:-${CODE_ROOT}/builds/all_experiments_wan22_generation.yaml}"
RUN_ID="${RUN_ID:-all13_wan22_generation_$(date +%Y%m%d_%H%M%S)}"
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

resolved_build="${RUN_DIR}/resolved_build.yaml"
manifest="${RUN_DIR}/manifest.jsonl"
if [[ -f "${resolved_build}" && -f "${manifest}" ]]; then
  echo "resume prepared run: ${RUN_DIR}"
elif [[ -e "${resolved_build}" || -e "${manifest}" ]]; then
  echo "error: incomplete prepared run; use a new RUN_ID after inspecting ${RUN_DIR}" >&2
  exit 2
else
  "${PYTHON_BIN}" -m remake_benchmark prepare \
    --build "${BUILD_FILE}" \
    --run-dir "${RUN_DIR}" \
    --workspace-root "${REMAKE_ROOT}"
fi

generation_args=(
  -m remake_benchmark generate
  --run-dir "${RUN_DIR}"
)
if [[ -n "${MAX_JOBS:-}" ]]; then
  generation_args+=(--max-jobs "${MAX_JOBS}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  generation_args+=(--start-index "${START_INDEX}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  generation_args+=(--dry-run)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  generation_args+=(--overwrite)
fi
if [[ "${FAIL_FAST:-1}" == "1" ]]; then
  generation_args+=(--fail-fast)
fi

"${PYTHON_BIN}" "${generation_args[@]}"
echo "generation-only run complete: ${RUN_DIR}"
