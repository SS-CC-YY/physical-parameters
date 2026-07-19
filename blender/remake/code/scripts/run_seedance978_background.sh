#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-python3}"
BUILD_FILE="${WORKSPACE_ROOT}/code/builds/standard_ball_factorized978_seedance20_generation.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/outputs/seedance978}"
CONFIRM_BILLABLE_978="${CONFIRM_BILLABLE_978:-NO}"
DETACHED="${DETACHED:-1}"
SEEDANCE_CONCURRENCY="${SEEDANCE_CONCURRENCY:-3}"

RUN_ID_WAS_EXPLICIT=0
if [[ -n "${RUN_ID+x}" && -n "${RUN_ID}" ]]; then
  RUN_ID_WAS_EXPLICIT=1
fi
RUN_ID="${RUN_ID:-seedance20_factorized978_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
CANARY_MARKER="${RUN_DIR}/.canary_success"
RESOLVED_BUILD="${RUN_DIR}/resolved_build.yaml"
MANIFEST="${RUN_DIR}/manifest.jsonl"

PHASE=canary
MAX_JOBS=1
CONCURRENCY=1
if [[ "${CONFIRM_BILLABLE_978}" == "YES" ]]; then
  PHASE=full978
  MAX_JOBS=978
  CONCURRENCY="${SEEDANCE_CONCURRENCY}"
elif [[ "${CONFIRM_BILLABLE_978}" != "NO" ]]; then
  echo "error: CONFIRM_BILLABLE_978 must be exactly NO or YES" >&2
  exit 2
fi

if [[ ! "${SEEDANCE_CONCURRENCY}" =~ ^[1-3]$ ]]; then
  echo "error: SEEDANCE_CONCURRENCY must be an integer in [1, 3]" >&2
  exit 2
fi

# A full run may only continue a reviewed canary directory. Requiring RUN_ID to
# be supplied by the caller prevents a confirmed invocation from silently
# creating a fresh, fully billable run.
if [[ "${PHASE}" == "full978" ]]; then
  if [[ "${RUN_ID_WAS_EXPLICIT}" != "1" ]]; then
    echo "error: full978 requires an explicit RUN_ID from the completed canary" >&2
    exit 2
  fi
  if [[ ! -d "${RUN_DIR}" || ! -f "${RESOLVED_BUILD}" || ! -f "${MANIFEST}" || ! -f "${CANARY_MARKER}" ]]; then
    echo "error: full978 must reuse a successful canary RUN_ID; required files are missing in ${RUN_DIR}" >&2
    exit 2
  fi
fi

if [[ -z "${ARK_API_KEY:-}" ]]; then
  echo "error: ARK_API_KEY is missing" >&2
  exit 2
fi

# REST generation is remote and must not reserve a local GPU.
export CUDA_VISIBLE_DEVICES=""

if [[ "${DETACHED}" == "1" ]]; then
  mkdir -p "${RUN_DIR}"
  if [[ -s "${RUN_DIR}/master.pid" ]]; then
    EXISTING_PID="$(<"${RUN_DIR}/master.pid")"
    if [[ "${EXISTING_PID}" =~ ^[0-9]+$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
      echo "error: this run is already active as PID ${EXISTING_PID}: ${RUN_DIR}" >&2
      exit 3
    fi
  fi
  DETACHED=0 \
    PYTHON="${PYTHON_BIN}" \
    WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_ID="${RUN_ID}" \
    CONFIRM_BILLABLE_978="${CONFIRM_BILLABLE_978}" \
    SEEDANCE_CONCURRENCY="${SEEDANCE_CONCURRENCY}" \
    nohup bash "$0" >>"${RUN_DIR}/master.log" 2>&1 &
  MASTER_PID=$!
  printf '%s\n' "${MASTER_PID}" >"${RUN_DIR}/master.pid"
  echo "Started Seedance phase=${PHASE}: PID=${MASTER_PID} RUN_ID=${RUN_ID}"
  echo "Log: ${RUN_DIR}/master.log"
  exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "error: flock is required to prevent duplicate billable submissions" >&2
  exit 4
fi
mkdir -p "${RUN_DIR}"
exec 9>"${RUN_DIR}/.seedance978.lock"
if ! flock -n 9; then
  echo "error: another process owns the Seedance run lock: ${RUN_DIR}" >&2
  exit 3
fi

if [[ -f "${RESOLVED_BUILD}" && -f "${MANIFEST}" ]]; then
  echo "Resuming prepared Seedance run: ${RUN_DIR}"
elif [[ -e "${RESOLVED_BUILD}" || -e "${MANIFEST}" ]]; then
  echo "error: refusing partially prepared run directory: ${RUN_DIR}" >&2
  exit 5
elif [[ "${PHASE}" == "full978" ]]; then
  echo "error: full978 cannot prepare a new run; reuse the canary RUN_ID" >&2
  exit 5
else
  "${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" prepare \
    --build "${BUILD_FILE}" \
    --run-dir "${RUN_DIR}" \
    --workspace-root "${WORKSPACE_ROOT}"
fi

TOTAL_JOBS="$(awk 'NF {count++} END {print count+0}' "${MANIFEST}")"
if [[ "${TOTAL_JOBS}" != "978" ]]; then
  echo "error: expected a frozen 978-job manifest, found ${TOTAL_JOBS}: ${MANIFEST}" >&2
  exit 5
fi

FIRST_JOB_ID="$("${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    first = next(line for line in handle if line.strip())
print(json.loads(first)["job_id"])
PY
)"

verify_canary_success() {
  local expected_marker
  local video="${RUN_DIR}/videos/${FIRST_JOB_ID}.mp4"
  local metadata="${RUN_DIR}/metadata/${FIRST_JOB_ID}.json"
  local api_metadata="${RUN_DIR}/metadata/${FIRST_JOB_ID}.api.json"

  if [[ ! -s "${CANARY_MARKER}" ]]; then
    echo "error: canary success marker is missing: ${CANARY_MARKER}" >&2
    return 1
  fi
  expected_marker="$(<"${CANARY_MARKER}")"
  if [[ "${expected_marker}" != "${FIRST_JOB_ID}" ]]; then
    echo "error: canary marker does not match the first manifest job" >&2
    return 1
  fi
  if [[ ! -s "${video}" || ! -s "${metadata}" || ! -s "${api_metadata}" ]]; then
    echo "error: canary video or metadata is incomplete for ${FIRST_JOB_ID}" >&2
    return 1
  fi
  "${PYTHON_BIN}" - "${api_metadata}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    record = json.load(handle)
if record.get("status") != "succeeded" or not record.get("task_id"):
    raise SystemExit("canary API metadata is not a succeeded task")
PY
}

verify_full_success() {
  "${PYTHON_BIN}" - "${RUN_DIR}" "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest = Path(sys.argv[2])
jobs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
problems = []
for job in jobs:
    job_id = job["job_id"]
    video = run_dir / "videos" / f"{job_id}.mp4"
    metadata = run_dir / "metadata" / f"{job_id}.json"
    api_metadata = run_dir / "metadata" / f"{job_id}.api.json"
    if not video.is_file() or video.stat().st_size == 0:
        problems.append(f"{job_id}: missing/empty video")
        continue
    if not metadata.is_file() or metadata.stat().st_size == 0:
        problems.append(f"{job_id}: missing canonical metadata")
    try:
        record = json.loads(api_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        problems.append(f"{job_id}: missing/invalid API metadata")
        continue
    if record.get("status") != "succeeded" or not record.get("task_id"):
        problems.append(f"{job_id}: API task is not succeeded")

if problems:
    preview = "\n".join(problems[:20])
    suffix = "" if len(problems) <= 20 else f"\n... and {len(problems) - 20} more"
    raise SystemExit(f"full-run artifact verification failed for {len(problems)} job(s):\n{preview}{suffix}")
if len(jobs) != 978:
    raise SystemExit(f"full-run verification expected 978 manifest jobs, found {len(jobs)}")
print("Verified 978 non-empty videos with canonical and succeeded API metadata.")
PY
}

if [[ "${PHASE}" == "full978" ]]; then
  verify_canary_success
fi

GENERATE_RC=0
"${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" generate \
  --run-dir "${RUN_DIR}" \
  --max-jobs "${MAX_JOBS}" \
  --concurrency "${CONCURRENCY}" \
  --fail-fast || GENERATE_RC=$?

"${PYTHON_BIN}" "${WORKSPACE_ROOT}/code/scripts/remake_benchmark.py" summarize-api \
  --run-dir "${RUN_DIR}" || true

if [[ "${GENERATE_RC}" -ne 0 ]]; then
  echo "error: Seedance ${PHASE} stopped after a generation failure; resume with the same RUN_ID after inspection" >&2
  exit "${GENERATE_RC}"
fi

if [[ "${PHASE}" == "canary" ]]; then
  # The runner only returns success after writing a non-empty video and both
  # provider and canonical metadata. Mark exactly that first manifest job.
  if [[ ! -s "${RUN_DIR}/videos/${FIRST_JOB_ID}.mp4" || ! -s "${RUN_DIR}/metadata/${FIRST_JOB_ID}.json" || ! -s "${RUN_DIR}/metadata/${FIRST_JOB_ID}.api.json" ]]; then
    echo "error: canary command returned without complete artifacts" >&2
    exit 6
  fi
  printf '%s\n' "${FIRST_JOB_ID}" >"${CANARY_MARKER}"
  verify_canary_success
  echo "Seedance one-job canary succeeded: ${FIRST_JOB_ID}"
  echo "Inspect the video and account usage, then continue this exact run:"
  echo "  RUN_ID='${RUN_ID}' CONFIRM_BILLABLE_978=YES bash code/scripts/run_seedance978_background.sh"
else
  verify_full_success
  echo "Seedance factorized978 generation completed: ${RUN_DIR}"
fi
