#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/outputs/closed_api_cost20}"
RUN_TAG="${RUN_TAG:-cost20_$(date +%Y%m%d_%H%M%S)}"
BATCH_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
SEEDANCE_RUN="${BATCH_DIR}/seedance20"
KLING_RUN="${BATCH_DIR}/kling3_std"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
DETACHED="${DETACHED:-1}"
CONFIRM_BILLABLE_20="${CONFIRM_BILLABLE_20:-NO}"
PROVIDERS="${PROVIDERS:-both}"
SEEDANCE_CONCURRENCY="${SEEDANCE_CONCURRENCY:-1}"
KLING_CONCURRENCY="${KLING_CONCURRENCY:-1}"

case "${PROVIDERS}" in
  both)
    RUN_SEEDANCE=1
    RUN_KLING=1
    ;;
  seedance)
    RUN_SEEDANCE=1
    RUN_KLING=0
    ;;
  kling)
    RUN_SEEDANCE=0
    RUN_KLING=1
    ;;
  *)
    echo "PROVIDERS must be one of: both, seedance, kling"
    exit 2
    ;;
esac

validate_concurrency() {
  local name="$1"
  local value="$2"
  local maximum="$3"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]] || (( value > maximum )); then
    echo "${name} must be an integer in [1, ${maximum}], got: ${value}"
    exit 2
  fi
}

# These hard caps match the account quotas confirmed for this benchmark. The
# default remains one so another account never gains concurrency implicitly.
validate_concurrency SEEDANCE_CONCURRENCY "${SEEDANCE_CONCURRENCY}" 3
validate_concurrency KLING_CONCURRENCY "${KLING_CONCURRENCY}" 5

# REST generation is remote; hide local GPUs from all child processes so this
# test never competes with Wan2.2 jobs on CUDA_VISIBLE_DEVICES=4,5,6,7.
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${BATCH_DIR}"

if [[ "${DRY_RUN_ONLY}" != "1" ]]; then
  if [[ "${RUN_SEEDANCE}" == "1" && -z "${ARK_API_KEY:-}" ]]; then
    echo "ARK_API_KEY is missing; configure a newly rotated key on the server."
    exit 2
  fi
  if [[ "${RUN_KLING}" == "1" && -z "${KLING_API_KEY:-}" ]]; then
    echo "KLING_API_KEY is missing; configure a newly rotated key on the server."
    exit 2
  fi
fi

if [[ "${DETACHED}" == "1" ]]; then
  if [[ -f "${BATCH_DIR}/master.pid" ]]; then
    EXISTING_PID="$(cat "${BATCH_DIR}/master.pid")"
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
      echo "Batch is already running with PID ${EXISTING_PID}: ${BATCH_DIR}"
      exit 3
    fi
  fi
  DETACHED=0 DRY_RUN_ONLY="${DRY_RUN_ONLY}" PYTHON="${PYTHON_BIN}" \
    RUN_TAG="${RUN_TAG}" OUTPUT_ROOT="${OUTPUT_ROOT}" WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
    CONFIRM_BILLABLE_20="${CONFIRM_BILLABLE_20}" PROVIDERS="${PROVIDERS}" \
    SEEDANCE_CONCURRENCY="${SEEDANCE_CONCURRENCY}" KLING_CONCURRENCY="${KLING_CONCURRENCY}" \
    nohup bash "$0" >>"${BATCH_DIR}/master.log" 2>&1 &
  MASTER_PID=$!
  echo "${MASTER_PID}" >"${BATCH_DIR}/master.pid"
  echo "Started closed-API cost20 batch: PID=${MASTER_PID} RUN_TAG=${RUN_TAG}"
  echo "Log: ${BATCH_DIR}/master.log"
  exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent duplicate billable submissions."
  exit 4
fi
exec 9>"${BATCH_DIR}/.batch.lock"
if ! flock -n 9; then
  echo "Another process already owns this batch lock: ${BATCH_DIR}"
  exit 3
fi

prepare_run() {
  local build_file="$1"
  local run_dir="$2"
  if [[ -f "${run_dir}/resolved_build.yaml" && -f "${run_dir}/manifest.jsonl" ]]; then
    return
  fi
  if [[ -e "${run_dir}/resolved_build.yaml" || -e "${run_dir}/manifest.jsonl" ]]; then
    echo "Refusing partially prepared run directory: ${run_dir}"
    exit 5
  fi
  "${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" prepare \
    --build "${build_file}" \
    --run-dir "${run_dir}" \
    --workspace-root "${WORKSPACE_ROOT}"
}

prepare_run \
  "${WORKSPACE_ROOT}/code/builds/v1a_seedance20_cn_api_cost20.yaml" \
  "${SEEDANCE_RUN}"
prepare_run \
  "${WORKSPACE_ROOT}/code/builds/v1a_kling3_std_cn_api_cost20.yaml" \
  "${KLING_RUN}"

"${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/verify_closed_api_cost20.py" \
  --seedance-manifest "${SEEDANCE_RUN}/manifest.jsonl" \
  --kling-manifest "${KLING_RUN}/manifest.jsonl" \
  --workspace-root "${WORKSPACE_ROOT}" \
  --output-dir "${BATCH_DIR}" \
  --expected-jobs 20

"${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" generate \
  --run-dir "${SEEDANCE_RUN}" \
  --max-jobs 20 \
  --dry-run
"${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" generate \
  --run-dir "${KLING_RUN}" \
  --max-jobs 20 \
  --dry-run

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo "Dry-run complete; no external API request was sent: ${BATCH_DIR}"
  exit 0
fi

MAX_JOBS=1
PHASE=canary
if [[ "${CONFIRM_BILLABLE_20}" == "YES" ]]; then
  MAX_JOBS=20
  PHASE=full20
fi

run_provider() {
  local provider="$1"
  local run_dir="$2"
  local requested_concurrency="$3"
  local provider_log="${BATCH_DIR}/${provider}_${PHASE}.log"
  local generate_rc=0
  local summary_rc=0
  local concurrency=1
  if [[ "${PHASE}" == "full20" ]]; then
    concurrency="${requested_concurrency}"
  fi
  {
    echo "Starting ${provider} phase=${PHASE} max_jobs=${MAX_JOBS} concurrency=${concurrency}"
    "${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" generate \
      --run-dir "${run_dir}" \
      --max-jobs "${MAX_JOBS}" \
      --concurrency "${concurrency}" \
      --fail-fast || generate_rc=$?
    "${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" summarize-api \
      --run-dir "${run_dir}" || summary_rc=$?
    if [[ "${generate_rc}" -ne 0 ]]; then
      return "${generate_rc}"
    fi
    if [[ "${summary_rc}" -ne 0 ]]; then
      return "${summary_rc}"
    fi
  } >"${provider_log}" 2>&1
}

# With PROVIDERS=both the providers run in parallel. Canary is always one task;
# full20 uses the explicit per-provider concurrency values above.
SEEDANCE_PID=""
KLING_PID=""
if [[ "${RUN_SEEDANCE}" == "1" ]]; then
  run_provider seedance "${SEEDANCE_RUN}" "${SEEDANCE_CONCURRENCY}" &
  SEEDANCE_PID=$!
fi
if [[ "${RUN_KLING}" == "1" ]]; then
  run_provider kling "${KLING_RUN}" "${KLING_CONCURRENCY}" &
  KLING_PID=$!
fi

FAILURES=0
if [[ -n "${SEEDANCE_PID}" ]] && ! wait "${SEEDANCE_PID}"; then
  echo "Seedance ${PHASE} failed; see ${BATCH_DIR}/seedance_${PHASE}.log"
  FAILURES=$((FAILURES + 1))
fi
if [[ -n "${KLING_PID}" ]] && ! wait "${KLING_PID}"; then
  echo "Kling ${PHASE} failed; see ${BATCH_DIR}/kling_${PHASE}.log"
  FAILURES=$((FAILURES + 1))
fi

if [[ "${FAILURES}" -ne 0 ]]; then
  exit 1
fi
if [[ "${PHASE}" == "canary" ]]; then
  echo "Selected one-job canary phase completed (PROVIDERS=${PROVIDERS}). Inspect videos, *.api.json, and account balances."
  echo "Continue the same batch only after review:"
  echo "  PYTHON='${PYTHON_BIN}' RUN_TAG=${RUN_TAG} PROVIDERS=${PROVIDERS} CONFIRM_BILLABLE_20=YES bash code/scripts/run_closed_api_cost20_background.sh"
else
  echo "Selected 20-job API batch phase completed (PROVIDERS=${PROVIDERS}): ${BATCH_DIR}"
fi
