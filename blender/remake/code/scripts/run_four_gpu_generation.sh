#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMAKE_ROOT="${REMAKE_ROOT:-$(cd "${CODE_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_FILE="${BUILD_FILE:?BUILD_FILE must point to a generation-only build}"
RUN_ID="${RUN_ID:?RUN_ID must be set}"
RUN_ROOT="${RUN_ROOT:-${REMAKE_ROOT}/outputs/${RUN_ID}}"
GPU_IDS_RAW="${GPU_IDS:-4,5,6,7}"

normalized_gpu_ids="${GPU_IDS_RAW//,/ }"
read -r -a gpu_ids <<<"${normalized_gpu_ids}"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "error: exactly four unique GPU ids are required; got: ${GPU_IDS_RAW}" >&2
  exit 2
fi
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "error: invalid GPU id: ${gpu_id}" >&2
    exit 2
  fi
  if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
    echo "error: duplicate GPU id: ${gpu_id}" >&2
    exit 2
  fi
  seen_gpu_ids[${gpu_id}]=1
done
if [[ ! -f "${BUILD_FILE}" ]]; then
  echo "error: build file does not exist: ${BUILD_FILE}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/control" "${RUN_ROOT}/shards"
export PYTHONPATH="${CODE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

build_marker="${RUN_ROOT}/build_file.txt"
resolved_build_file="$(cd "$(dirname "${BUILD_FILE}")" && pwd)/$(basename "${BUILD_FILE}")"
if [[ -f "${build_marker}" ]]; then
  existing_build_file="$(<"${build_marker}")"
  if [[ "${existing_build_file}" != "${resolved_build_file}" ]]; then
    echo "error: RUN_ID already belongs to a different build: ${existing_build_file}" >&2
    exit 2
  fi
else
  printf '%s\n' "${resolved_build_file}" >"${build_marker}"
fi

control_dir="${RUN_ROOT}/control"
if [[ -f "${control_dir}/resolved_build.yaml" && -f "${control_dir}/manifest.jsonl" ]]; then
  echo "resume four-GPU run: ${RUN_ROOT}"
elif [[ -e "${control_dir}/resolved_build.yaml" || -e "${control_dir}/manifest.jsonl" ]]; then
  echo "error: incomplete controller manifest; inspect ${control_dir} and use a new RUN_ID" >&2
  exit 2
else
  GPU_ID="${gpu_ids[0]}" "${PYTHON_BIN}" -m remake_benchmark prepare \
    --build "${BUILD_FILE}" \
    --run-dir "${control_dir}" \
    --workspace-root "${REMAKE_ROOT}"
fi

total_jobs="$(awk 'NF {count++} END {print count+0}' "${control_dir}/manifest.jsonl")"
if (( total_jobs < ${#gpu_ids[@]} )); then
  echo "error: ${total_jobs} jobs cannot be split across ${#gpu_ids[@]} GPUs" >&2
  exit 2
fi

base_jobs=$((total_jobs / ${#gpu_ids[@]}))
remainder=$((total_jobs % ${#gpu_ids[@]}))
start_index=0
plan_tmp="${RUN_ROOT}/shards.tsv.tmp"
printf 'gpu_id\tstart_index\tmax_jobs\tpid\tlog\n' >"${plan_tmp}"

for position in "${!gpu_ids[@]}"; do
  gpu_id="${gpu_ids[${position}]}"
  max_jobs="${base_jobs}"
  if (( position < remainder )); then
    max_jobs=$((max_jobs + 1))
  fi
  shard_dir="${RUN_ROOT}/shards/gpu-${gpu_id}"
  pid_file="${shard_dir}/worker.pid"
  log_file="${shard_dir}/master.log"
  mkdir -p "${shard_dir}"

  existing_pid=""
  if [[ -s "${pid_file}" ]]; then
    existing_pid="$(<"${pid_file}")"
  fi
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    process_run_dir="$(tr '\0' '\n' <"/proc/${existing_pid}/environ" 2>/dev/null | awk -F= '$1 == "RUN_DIR" {sub(/^RUN_DIR=/, ""); print; exit}' || true)"
    if [[ "${process_run_dir}" == "${shard_dir}" ]]; then
      echo "GPU ${gpu_id}: worker already running as PID ${existing_pid}"
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${gpu_id}" "${start_index}" "${max_jobs}" "${existing_pid}" "${log_file}" >>"${plan_tmp}"
      start_index=$((start_index + max_jobs))
      continue
    fi
    echo "warning: stale PID file points to another process; it will not be touched: ${existing_pid}" >&2
  fi

  nohup env \
    BUILD_FILE="${resolved_build_file}" \
    RUN_ID="${RUN_ID}_gpu-${gpu_id}" \
    RUN_DIR="${shard_dir}" \
    REMAKE_ROOT="${REMAKE_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    GPU_ID="${gpu_id}" \
    START_INDEX="${start_index}" \
    MAX_JOBS="${max_jobs}" \
    DETACHED=0 \
    FAIL_FAST="${FAIL_FAST:-1}" \
    bash "${SCRIPT_DIR}/run_full_generation.sh" >"${log_file}" 2>&1 &
  worker_pid=$!
  printf '%s\n' "${worker_pid}" >"${pid_file}"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${gpu_id}" "${start_index}" "${max_jobs}" "${worker_pid}" "${log_file}" >>"${plan_tmp}"
  echo "GPU ${gpu_id}: PID ${worker_pid}, jobs [${start_index}, $((start_index + max_jobs)))"
  start_index=$((start_index + max_jobs))
done

mv "${plan_tmp}" "${RUN_ROOT}/shards.tsv"
echo "total_jobs=${total_jobs}"
echo "run_root=${RUN_ROOT}"
echo "status: bash code/scripts/check_four_gpu_generation.sh '${RUN_ROOT}'"
echo "collect after completion: python code/scripts/collect_sharded_run.py '${RUN_ROOT}'"
