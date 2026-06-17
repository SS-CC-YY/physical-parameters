#!/usr/bin/env bash
# Run V1 Wan2.2 I2V strict-prompt + motion-trail ablations in the background.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="${STAMP:-$(date +'%Y%m%d_%H%M%S')}"
OUT_ROOT="${OUT_ROOT:-outputs/v1_wan22_prompt_trail_${STAMP}}"
MASTER_LOG="${OUT_ROOT}/master.log"
PID_FILE="${OUT_ROOT}/master.pid"

if [[ "${V1_TRAIL_DETACHED:-0}" != "1" ]]; then
  mkdir -p "${OUT_ROOT}"
  env V1_TRAIL_DETACHED=1 OUT_ROOT="${OUT_ROOT}" STAMP="${STAMP}" \
    nohup bash "$0" "$@" > "${MASTER_LOG}" 2>&1 &
  PID="$!"
  echo "${PID}" > "${PID_FILE}"
  echo "started V1 Wan2.2 prompt/trail ablation"
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
CAMERAS="${CAMERAS:-CAM_Side}"
EXPERIMENTS="${EXPERIMENTS:-v1_A v1_B v1_C v1_D}"
CONDITIONING_MODES="${CONDITIONING_MODES:-raw_frame10 center_10f}"
MAX_JOBS="${MAX_JOBS:-}"
OVERWRITE="${OVERWRITE:-0}"
MAKE_OVERLAYS="${MAKE_OVERLAYS:-0}"
OVERLAY_MAX_JOBS="${OVERLAY_MAX_JOBS:-}"
PACKAGE_SAMPLE_VIDEOS="${PACKAGE_SAMPLE_VIDEOS:-6}"

CONDITIONING_ROOT="${OUT_ROOT}/conditioning_images"
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

mode_outdir() {
  local mode="$1"
  echo "${OUT_ROOT}/${mode}"
}

make_conditioning_images() {
  python code/v1_wan22_i2v_prompt_trail/make_motion_trail_images.py \
    --seeds-root blender/seeds \
    --renders-root blender/renders \
    --output-root "${CONDITIONING_ROOT}" \
    --experiments ${EXPERIMENTS} \
    --cameras ${CAMERAS} \
    --modes ${CONDITIONING_MODES} \
    --allow-missing
}

build_manifest() {
  local mode="$1"
  local outdir
  outdir="$(mode_outdir "${mode}")"
  mkdir -p "${outdir}"
  local cmd=(
    python code/v1_wan22_i2v_prompt_trail/build_v1_strict_manifest.py
    --seeds-root blender/seeds
    --conditioning-root "${CONDITIONING_ROOT}"
    --conditioning-mode "${mode}"
    --output "${outdir}/manifest.jsonl"
    --experiments ${EXPERIMENTS}
    --cameras ${CAMERAS}
    --prompt-mode "${PROMPT_MODE}"
  )
  if [[ -n "${MAX_JOBS}" ]]; then
    cmd+=(--max-jobs "${MAX_JOBS}")
  fi
  "${cmd[@]}"
}

generate_mode() {
  local mode="$1"
  local outdir
  outdir="$(mode_outdir "${mode}")"
  local cmd=(
    python code/v1_wan22_i2v_full/run_v1_wan22_official.py
    --manifest "${outdir}/manifest.jsonl"
    --wan-repo "${WAN_REPO}"
    --ckpt-dir "${CKPT_DIR}"
    --outdir "${outdir}"
    --gpu-id "${GPU_ID}"
    --size "${SIZE}"
    --frame-num "${FRAME_NUM}"
    --sample-steps "${SAMPLE_STEPS}"
  )
  if [[ -n "${MAX_JOBS}" ]]; then
    cmd+=(--max-jobs "${MAX_JOBS}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

evaluate_mode() {
  local mode="$1"
  local outdir
  outdir="$(mode_outdir "${mode}")"
  python code/v1_wan22_i2v_full/evaluate_v1_physics.py \
    --manifest "${outdir}/manifest.jsonl" \
    --generated-root "${outdir}" \
    --outdir "${outdir}/eval"
}

visualize_mode() {
  local mode="$1"
  local outdir
  outdir="$(mode_outdir "${mode}")"
  local cmd=(
    python code/v1_wan22_i2v_full/visualize_v1_eval.py
    --generated-root "${outdir}"
    --eval-dir "${outdir}/eval"
    --manifest "${outdir}/manifest.jsonl"
  )
  if [[ "${MAKE_OVERLAYS}" == "1" ]]; then
    cmd+=(--make-overlays)
    if [[ -n "${OVERLAY_MAX_JOBS}" ]]; then
      cmd+=(--overlay-max-jobs "${OVERLAY_MAX_JOBS}")
    fi
  fi
  "${cmd[@]}"
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
    echo "cameras=${CAMERAS}"
    echo "experiments=${EXPERIMENTS}"
    echo "conditioning_modes=${CONDITIONING_MODES}"
    echo "max_jobs=${MAX_JOBS}"
  } > "${OUT_ROOT}/meta/run_config.txt"
  git rev-parse HEAD > "${OUT_ROOT}/meta/git_head.txt" 2>/dev/null || true
  git status --short > "${OUT_ROOT}/meta/git_status.txt" 2>/dev/null || true
  nvidia-smi > "${OUT_ROOT}/meta/nvidia_smi.txt" 2>/dev/null || true
}

package_results() {
  local package_dir="${OUT_ROOT}/package"
  mkdir -p "${package_dir}/sample_videos" "${package_dir}/runs"
  cp -r "${OUT_ROOT}/meta" "${package_dir}/" 2>/dev/null || true
  cp "${MASTER_LOG}" "${package_dir}/master.log" 2>/dev/null || true
  for mode in ${CONDITIONING_MODES}; do
    local outdir
    outdir="$(mode_outdir "${mode}")"
    local dst="${package_dir}/runs/${mode}"
    mkdir -p "${dst}"
    cp "${outdir}/manifest.jsonl" "${dst}/" 2>/dev/null || true
    cp "${outdir}/generation_log.jsonl" "${dst}/" 2>/dev/null || true
    cp "${outdir}/run_summary.json" "${dst}/" 2>/dev/null || true
    cp -r "${outdir}/logs" "${dst}/" 2>/dev/null || true
    cp -r "${outdir}/eval" "${dst}/" 2>/dev/null || true
    if [[ -d "${outdir}/videos" ]]; then
      mkdir -p "${package_dir}/sample_videos/${mode}"
      find "${outdir}/videos" -maxdepth 1 -name "*.mp4" | sort | head -n "${PACKAGE_SAMPLE_VIDEOS}" | while read -r video; do
        cp "${video}" "${package_dir}/sample_videos/${mode}/" 2>/dev/null || true
      done
    fi
  done
  tar -czf "${OUT_ROOT}.tar.gz" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")/package"
  log "PACKAGED ${OUT_ROOT}.tar.gz"
}

summarize_ablation() {
  python code/v1_wan22_i2v_prompt_trail/summarize_ablation.py \
    --out-root "${OUT_ROOT}" \
    --modes ${CONDITIONING_MODES} \
    --output "${OUT_ROOT}/ablation_summary.json"
}

log "V1 Wan2.2 prompt/trail ablation started"
collect_meta
echo "label,status" > "${OUT_ROOT}/meta/status.csv"

run_and_record make_conditioning_images make_conditioning_images
for mode in ${CONDITIONING_MODES}; do
  run_and_record "${mode}_build_manifest" build_manifest "${mode}"
  run_and_record "${mode}_generate" generate_mode "${mode}"
  run_and_record "${mode}_evaluate" evaluate_mode "${mode}"
  run_and_record "${mode}_visualize" visualize_mode "${mode}"
done
run_and_record summarize_ablation summarize_ablation
run_and_record package_results package_results
log "V1 Wan2.2 prompt/trail ablation finished"
