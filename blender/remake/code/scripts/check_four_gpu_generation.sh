#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-${RUN_ROOT:-}}"
if [[ -z "${RUN_ROOT}" ]]; then
  echo "usage: bash code/scripts/check_four_gpu_generation.sh outputs/<four_gpu_run>" >&2
  exit 2
fi
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
if [[ ! -f "${RUN_ROOT}/control/manifest.jsonl" ]]; then
  echo "error: controller manifest not found under ${RUN_ROOT}" >&2
  exit 2
fi

expected="$(awk 'NF {count++} END {print count+0}' "${RUN_ROOT}/control/manifest.jsonl")"
generated=0
printf '%-8s %-10s %-12s %-12s %s\n' GPU PID STATUS VIDEOS LAST_LOG_LINE
for shard_dir in "${RUN_ROOT}"/shards/gpu-*; do
  [[ -d "${shard_dir}" ]] || continue
  gpu_id="${shard_dir##*/gpu-}"
  pid="-"
  status="stopped"
  if [[ -s "${shard_dir}/worker.pid" ]]; then
    pid="$(<"${shard_dir}/worker.pid")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      process_run_dir="$(tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null | awk -F= '$1 == "RUN_DIR" {sub(/^RUN_DIR=/, ""); print; exit}' || true)"
      if [[ "${process_run_dir}" == "${shard_dir}" ]]; then
        status="running"
      else
        status="pid-reused"
      fi
    elif [[ -f "${shard_dir}/run_summary.json" ]]; then
      status="finished"
    fi
  fi
  count=0
  if [[ -d "${shard_dir}/videos" ]]; then
    count="$(find "${shard_dir}/videos" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l)"
  fi
  generated=$((generated + count))
  last_line="$(tail -n 1 "${shard_dir}/master.log" 2>/dev/null || true)"
  printf '%-8s %-10s %-12s %-12s %s\n' "${gpu_id}" "${pid}" "${status}" "${count}" "${last_line}"
done
echo "generated=${generated}/${expected}"
if (( generated == expected )); then
  echo "all videos are present; collect with:"
  echo "python code/scripts/collect_sharded_run.py '${RUN_ROOT}'"
fi
