#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WAN_REPO="${WAN_REPO:-../Wan2.2}"
CKPT_DIR="${CKPT_DIR:-../models/Wan2.2-I2V-A14B}"
OUTDIR="${OUTDIR:-outputs/v1_wan22_i2v_a14b}"
GPU_ID="${GPU_ID:-7}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-121}"
PROMPT_MODE="${PROMPT_MODE:-explicit}"
CAMERAS="${CAMERAS:-CAM_Side}"
EXPERIMENTS="${EXPERIMENTS:-v1_A v1_B v1_C v1_D}"
MAX_JOBS="${MAX_JOBS:-}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "${OUTDIR}"

MANIFEST="${OUTDIR}/manifest.jsonl"
RUN_LOG="${OUTDIR}/run.log"
PID_FILE="${OUTDIR}/run.pid"

python code/v1_wan22_i2v_full/build_v1_manifest.py \
  --seeds-root blender/seeds \
  --output "${MANIFEST}" \
  --prompt-mode "${PROMPT_MODE}" \
  --experiments ${EXPERIMENTS} \
  --cameras ${CAMERAS}

RUN_CMD=(
  python code/v1_wan22_i2v_full/run_v1_wan22_official.py
  --manifest "${MANIFEST}"
  --wan-repo "${WAN_REPO}"
  --ckpt-dir "${CKPT_DIR}"
  --outdir "${OUTDIR}"
  --gpu-id "${GPU_ID}"
  --size "${SIZE}"
  --frame-num "${FRAME_NUM}"
)

if [[ -n "${MAX_JOBS}" ]]; then
  RUN_CMD+=(--max-jobs "${MAX_JOBS}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  RUN_CMD+=(--overwrite)
fi

printf '%q ' "${RUN_CMD[@]}" > "${OUTDIR}/run_command.sh"
printf '\n' >> "${OUTDIR}/run_command.sh"

nohup "${RUN_CMD[@]}" > "${RUN_LOG}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "started pid ${PID}"
echo "log: ${RUN_LOG}"
echo "pid file: ${PID_FILE}"
echo "monitor: tail -f ${RUN_LOG}"

