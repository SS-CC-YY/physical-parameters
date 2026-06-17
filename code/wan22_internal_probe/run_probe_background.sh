#!/usr/bin/env bash
# Run Wan2.2 V1-A internal-state probing in the background.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="${STAMP:-$(date +'%Y%m%d_%H%M%S')}"
OUT_ROOT="${OUT_ROOT:-outputs/wan22_internal_probe_${STAMP}}"
MASTER_LOG="${OUT_ROOT}/master.log"
PID_FILE="${OUT_ROOT}/master.pid"

if [[ "${WAN22_PROBE_DETACHED:-0}" != "1" ]]; then
  mkdir -p "${OUT_ROOT}"
  env WAN22_PROBE_DETACHED=1 OUT_ROOT="${OUT_ROOT}" STAMP="${STAMP}" \
    nohup bash "$0" "$@" > "${MASTER_LOG}" 2>&1 &
  PID="$!"
  echo "${PID}" > "${PID_FILE}"
  echo "started Wan2.2 internal probe run"
  echo "pid: ${PID}"
  echo "out_root: ${OUT_ROOT}"
  echo "log: ${MASTER_LOG}"
  echo "monitor: tail -f ${MASTER_LOG}"
  exit 0
fi

WAN_REPO="${WAN_REPO:-../Wan2.2}"
CKPT_DIR="${CKPT_DIR:-../../models/Wan2.2-I2V-A14B}"
GPU_ID="${GPU_ID:-7}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-121}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
PROMPT_MODE="${PROMPT_MODE:-visual_trace}"
CAMERA="${CAMERA:-CAM_Side}"
SEEDS="${SEEDS:-11 22 33 44 55}"
FRAME_NAME="${FRAME_NAME:-frame_10.png}"
USE_CONDITIONING_ROOT="${USE_CONDITIONING_ROOT:-1}"
CONDITIONING_MODE="${CONDITIONING_MODE:-center_10f}"
CONDITIONING_ROOT="${CONDITIONING_ROOT:-${OUT_ROOT}/conditioning_images}"
MAKE_CONDITIONING_IMAGES="${MAKE_CONDITIONING_IMAGES:-1}"
MAX_JOBS="${MAX_JOBS:-}"
OVERWRITE="${OVERWRITE:-0}"
CLASS_REGEX="${CLASS_REGEX:-(Block)}"
MODULE_REGEX="${MODULE_REGEX:-^wan\\.modules\\.model}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"

mkdir -p "${OUT_ROOT}/meta"

log() {
  printf '[%s] %s\n' "$(date +'%F %T')" "$*"
}

run_and_record() {
  local label="$1"
  shift
  log "START ${label}: $*"
  "$@"
  local status="$?"
  log "END ${label}: status=${status}"
  echo "${label},${status}" >> "${OUT_ROOT}/meta/status.csv"
  return 0
}

make_conditioning_images() {
  if [[ "${USE_CONDITIONING_ROOT}" != "1" || "${MAKE_CONDITIONING_IMAGES}" != "1" ]]; then
    log "SKIP make_conditioning_images: USE_CONDITIONING_ROOT=${USE_CONDITIONING_ROOT}, MAKE_CONDITIONING_IMAGES=${MAKE_CONDITIONING_IMAGES}"
    return 0
  fi
  python code/v1_wan22_i2v_prompt_trail/make_motion_trail_images.py \
    --seeds-root blender/seeds \
    --renders-root blender/renders \
    --output-root "${CONDITIONING_ROOT}" \
    --experiments v1_A \
    --cameras "${CAMERA}" \
    --modes "${CONDITIONING_MODE}" \
    --allow-missing
}

build_manifest() {
  local cmd=(
    python code/wan22_internal_probe/build_probe_manifest.py
    --seeds-root blender/seeds
    --output "${OUT_ROOT}/manifest.jsonl"
    --camera "${CAMERA}"
    --prompt-mode "${PROMPT_MODE}"
    --base-seeds ${SEEDS}
    --frame-name "${FRAME_NAME}"
  )
  if [[ "${USE_CONDITIONING_ROOT}" == "1" ]]; then
    cmd+=(
      --conditioning-root "${CONDITIONING_ROOT}"
      --conditioning-mode "${CONDITIONING_MODE}"
    )
  fi
  if [[ -n "${MAX_JOBS}" ]]; then
    cmd+=(--max-jobs "${MAX_JOBS}")
  fi
  "${cmd[@]}"
}

collect_features() {
  local cmd=(
    python code/wan22_internal_probe/collect_wan22_hidden_features.py
    --manifest "${OUT_ROOT}/manifest.jsonl"
    --outdir "${OUT_ROOT}"
    --wan-repo "${WAN_REPO}"
    --ckpt-dir "${CKPT_DIR}"
    --gpu-id "${GPU_ID}"
    --size "${SIZE}"
    --frame-num "${FRAME_NUM}"
    --sample-steps "${SAMPLE_STEPS}"
    --class-regex "${CLASS_REGEX}"
    --module-regex "${MODULE_REGEX}"
  )
  if [[ -n "${MAX_JOBS}" ]]; then
    cmd+=(--max-jobs "${MAX_JOBS}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

evaluate_video_outputs() {
  python code/v1_wan22_i2v_full/evaluate_v1_physics.py \
    --manifest "${OUT_ROOT}/manifest.jsonl" \
    --generated-root "${OUT_ROOT}" \
    --outdir "${OUT_ROOT}/eval_video"
}

train_probe() {
  python code/wan22_internal_probe/train_linear_probe.py \
    --manifest "${OUT_ROOT}/manifest.jsonl" \
    --features-dir "${OUT_ROOT}/features" \
    --outdir "${OUT_ROOT}/probe" \
    --target-key target_param_value \
    --alpha "${RIDGE_ALPHA}"
}

compare_outputs() {
  python code/wan22_internal_probe/compare_video_and_probe.py \
    --video-summary "${OUT_ROOT}/eval_video/summary.csv" \
    --probe-summary "${OUT_ROOT}/probe/probe_summary.csv" \
    --outdir "${OUT_ROOT}/comparison"
}

collect_meta() {
  {
    echo "timestamp=${STAMP}"
    echo "out_root=${OUT_ROOT}"
    echo "wan_repo=${WAN_REPO}"
    echo "ckpt_dir=${CKPT_DIR}"
    echo "gpu_id=${GPU_ID}"
    echo "size=${SIZE}"
    echo "frame_num=${FRAME_NUM}"
    echo "sample_steps=${SAMPLE_STEPS}"
    echo "prompt_mode=${PROMPT_MODE}"
    echo "camera=${CAMERA}"
    echo "frame_name=${FRAME_NAME}"
    echo "use_conditioning_root=${USE_CONDITIONING_ROOT}"
    echo "conditioning_mode=${CONDITIONING_MODE}"
    echo "conditioning_root=${CONDITIONING_ROOT}"
    echo "make_conditioning_images=${MAKE_CONDITIONING_IMAGES}"
    echo "seeds=${SEEDS}"
    echo "max_jobs=${MAX_JOBS}"
    echo "class_regex=${CLASS_REGEX}"
    echo "module_regex=${MODULE_REGEX}"
  } > "${OUT_ROOT}/meta/run_config.txt"
  git rev-parse HEAD > "${OUT_ROOT}/meta/git_head.txt" 2>/dev/null || true
  git status --short > "${OUT_ROOT}/meta/git_status.txt" 2>/dev/null || true
  python - <<'PY' > "${OUT_ROOT}/meta/python_env.txt" 2>&1
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
except Exception as exc:
    print("torch_error:", exc)
PY
  nvidia-smi > "${OUT_ROOT}/meta/nvidia_smi.txt" 2>/dev/null || true
}

package_results() {
  tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
  log "PACKAGED ${OUT_ROOT}.tar.gz"
}

log "Wan2.2 internal probe run started"
collect_meta
echo "label,status" > "${OUT_ROOT}/meta/status.csv"

run_and_record make_conditioning_images make_conditioning_images
run_and_record build_manifest build_manifest
run_and_record collect_features collect_features
run_and_record evaluate_video_outputs evaluate_video_outputs
run_and_record train_probe train_probe
run_and_record compare_outputs compare_outputs
run_and_record package_results package_results

log "Wan2.2 internal probe run finished"
