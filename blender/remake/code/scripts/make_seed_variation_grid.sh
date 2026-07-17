#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-${RUN_DIR:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "usage: bash code/scripts/make_seed_variation_grid.sh outputs/<seed_variation_run>" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg is not available in PATH" >&2
  exit 2
fi

RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
VIDEOS_DIR="${RUN_DIR}/videos"
PREFIX="v1_A__g9p81__baseline__standard_ball__CAM_Side"
new_seeds=(341867882 1750912582 265635392 135883006)
legacy_seeds=(36 37 38 39)
seed_values=()
if [[ -n "${SEEDS:-}" ]]; then
  normalized_seeds="${SEEDS//,/ }"
  read -r -a seed_values <<<"${normalized_seeds}"
else
  new_complete=1
  for seed in "${new_seeds[@]}"; do
    [[ -s "${VIDEOS_DIR}/${PREFIX}__seed-${seed}.mp4" ]] || new_complete=0
  done
  if (( new_complete )); then
    seed_values=("${new_seeds[@]}")
  else
    legacy_complete=1
    for seed in "${legacy_seeds[@]}"; do
      [[ -s "${VIDEOS_DIR}/${PREFIX}__seed-${seed}.mp4" ]] || legacy_complete=0
    done
    if (( legacy_complete )); then
      seed_values=("${legacy_seeds[@]}")
    else
      seed_values=("${new_seeds[@]}")
    fi
  fi
fi
if [[ "${#seed_values[@]}" -ne 4 ]]; then
  echo "error: SEEDS must contain exactly four comma- or space-separated integers" >&2
  exit 2
fi
ffmpeg_inputs=()
for seed in "${seed_values[@]}"; do
  video="${VIDEOS_DIR}/${PREFIX}__seed-${seed}.mp4"
  if [[ ! -s "${video}" ]]; then
    echo "error: missing non-empty video: ${video}" >&2
    exit 2
  fi
  ffmpeg_inputs+=(-i "${video}")
done

OUTPUT="${RUN_DIR}/seed_variation_grid.mp4"
ffmpeg -y "${ffmpeg_inputs[@]}" \
  -filter_complex \
  "[0:v]scale=416:240:force_original_aspect_ratio=decrease,pad=416:240:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[v0];\
[1:v]scale=416:240:force_original_aspect_ratio=decrease,pad=416:240:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[v1];\
[2:v]scale=416:240:force_original_aspect_ratio=decrease,pad=416:240:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[v2];\
[3:v]scale=416:240:force_original_aspect_ratio=decrease,pad=416:240:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[v3];\
[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black:shortest=1[v]" \
  -map "[v]" -an -r 16 -movflags +faststart "${OUTPUT}"

echo "grid=${OUTPUT}"
echo "layout: top-left=seed${seed_values[0]} top-right=seed${seed_values[1]} bottom-left=seed${seed_values[2]} bottom-right=seed${seed_values[3]}"
