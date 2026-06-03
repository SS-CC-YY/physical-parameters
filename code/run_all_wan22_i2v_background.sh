#!/usr/bin/env bash
# Run Wan2.2 I2V generation, evaluation, visualization, and packaging for V1/V2/V3.
#
# Usage:
#   bash code/run_all_wan22_i2v_background.sh
#
# Useful overrides:
#   GPU_ID=7 FRAME_NUM=121 SIZE='832*480' PROMPT_MODE=explicit bash code/run_all_wan22_i2v_background.sh
#   VERSIONS='v1 v2' MAX_JOBS=1 bash code/run_all_wan22_i2v_background.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="${STAMP:-$(date +'%Y%m%d_%H%M%S')}"
OUT_ROOT="${OUT_ROOT:-outputs/wan22_i2v_all_${STAMP}}"
MASTER_LOG="${OUT_ROOT}/master.log"
PID_FILE="${OUT_ROOT}/master.pid"

if [[ "${WAN22_ALL_DETACHED:-0}" != "1" ]]; then
  mkdir -p "${OUT_ROOT}"
  env WAN22_ALL_DETACHED=1 OUT_ROOT="${OUT_ROOT}" STAMP="${STAMP}" \
    nohup bash "$0" "$@" > "${MASTER_LOG}" 2>&1 &
  PID="$!"
  echo "${PID}" > "${PID_FILE}"
  echo "started all-version Wan2.2 run"
  echo "pid: ${PID}"
  echo "out_root: ${OUT_ROOT}"
  echo "log: ${MASTER_LOG}"
  echo "monitor: tail -f ${MASTER_LOG}"
  exit 0
fi

WAN_REPO="${WAN_REPO:-../Wan2.2}"
CKPT_DIR="${CKPT_DIR:-../models/Wan2.2-I2V-A14B}"
GPU_ID="${GPU_ID:-7}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-121}"
PROMPT_MODE="${PROMPT_MODE:-explicit}"
CAMERAS="${CAMERAS:-CAM_Side}"
VERSIONS="${VERSIONS:-v1 v2 v3}"
MAX_JOBS="${MAX_JOBS:-}"
OVERWRITE="${OVERWRITE:-0}"
MAKE_OVERLAYS="${MAKE_OVERLAYS:-0}"
OVERLAY_MAX_JOBS="${OVERLAY_MAX_JOBS:-}"
PACKAGE_SAMPLE_VIDEOS="${PACKAGE_SAMPLE_VIDEOS:-6}"

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

version_outdir() {
  local version="$1"
  echo "${OUT_ROOT}/${version}_wan22_i2v_a14b"
}

build_manifest() {
  local version="$1"
  local outdir
  outdir="$(version_outdir "${version}")"
  mkdir -p "${outdir}"
  python "code/${version}_wan22_i2v_full/build_${version}_manifest.py" \
    --seeds-root blender/seeds \
    --output "${outdir}/manifest.jsonl" \
    --prompt-mode "${PROMPT_MODE}" \
    --cameras ${CAMERAS}
}

generate_version() {
  local version="$1"
  local outdir
  outdir="$(version_outdir "${version}")"
  local cmd=(
    python "code/${version}_wan22_i2v_full/run_${version}_wan22_official.py"
    --manifest "${outdir}/manifest.jsonl"
    --wan-repo "${WAN_REPO}"
    --ckpt-dir "${CKPT_DIR}"
    --outdir "${outdir}"
    --gpu-id "${GPU_ID}"
    --size "${SIZE}"
    --frame-num "${FRAME_NUM}"
  )
  if [[ -n "${MAX_JOBS}" ]]; then
    cmd+=(--max-jobs "${MAX_JOBS}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"
}

evaluate_version() {
  local version="$1"
  local outdir
  outdir="$(version_outdir "${version}")"
  python "code/${version}_wan22_i2v_full/evaluate_${version}_physics.py" \
    --manifest "${outdir}/manifest.jsonl" \
    --generated-root "${outdir}" \
    --outdir "${outdir}/eval"
}

visualize_version() {
  local version="$1"
  local outdir
  outdir="$(version_outdir "${version}")"
  local cmd=(
    python "code/${version}_wan22_i2v_full/visualize_${version}_eval.py"
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
    echo "prompt_mode=${PROMPT_MODE}"
    echo "cameras=${CAMERAS}"
    echo "versions=${VERSIONS}"
    echo "max_jobs=${MAX_JOBS}"
    echo "make_overlays=${MAKE_OVERLAYS}"
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
  local package_dir="${OUT_ROOT}/package"
  mkdir -p "${package_dir}/sample_videos" "${package_dir}/runs"
  cp -r "${OUT_ROOT}/meta" "${package_dir}/" 2>/dev/null || true
  cp "${MASTER_LOG}" "${package_dir}/master.log" 2>/dev/null || true

  for version in ${VERSIONS}; do
    local outdir
    outdir="$(version_outdir "${version}")"
    local dst="${package_dir}/runs/${version}"
    mkdir -p "${dst}"
    cp "${outdir}/manifest.jsonl" "${dst}/" 2>/dev/null || true
    cp "${outdir}/generation_log.jsonl" "${dst}/" 2>/dev/null || true
    cp "${outdir}/run_summary.json" "${dst}/" 2>/dev/null || true
    cp -r "${outdir}/logs" "${dst}/" 2>/dev/null || true
    cp -r "${outdir}/eval" "${dst}/" 2>/dev/null || true
    if [[ -d "${outdir}/videos" ]]; then
      mkdir -p "${package_dir}/sample_videos/${version}"
      find "${outdir}/videos" -maxdepth 1 -name "*.mp4" | sort | head -n "${PACKAGE_SAMPLE_VIDEOS}" | while read -r video; do
        cp "${video}" "${package_dir}/sample_videos/${version}/" 2>/dev/null || true
      done
    fi
  done

  find "${OUT_ROOT}" -maxdepth 4 -type f | sort > "${package_dir}/meta/output_filelist.txt" 2>/dev/null || true
  du -sh "${OUT_ROOT}" > "${package_dir}/meta/output_size.txt" 2>/dev/null || true

  local tarball="${OUT_ROOT}.tar.gz"
  tar -czf "${tarball}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")/package"
  log "PACKAGED ${tarball}"
  echo "${tarball}" > "${OUT_ROOT}/meta/package_path.txt"
}

log "Wan2.2 all-version run started"
collect_meta
echo "label,status" > "${OUT_ROOT}/meta/status.csv"

for version in ${VERSIONS}; do
  log "========== ${version}: build manifest =========="
  run_and_record "${version}_build_manifest" build_manifest "${version}"

  log "========== ${version}: generate =========="
  run_and_record "${version}_generate" generate_version "${version}"

  log "========== ${version}: evaluate =========="
  run_and_record "${version}_evaluate" evaluate_version "${version}"

  log "========== ${version}: visualize =========="
  run_and_record "${version}_visualize" visualize_version "${version}"
done

log "========== package results =========="
run_and_record "package_results" package_results
log "Wan2.2 all-version run finished"

