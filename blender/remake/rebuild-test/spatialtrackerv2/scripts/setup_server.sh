#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UPSTREAM="$ROOT/upstream/SpaTrackerV2"
ENV_PREFIX="$ROOT/env"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda is required" >&2
  exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$UPSTREAM/requirements.txt"

export HF_HOME="$ROOT/models/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

cache = os.environ["HUGGINGFACE_HUB_CACHE"]
for repo in [
    "Yuxihenry/SpatialTrackerV2_Front",
    "Yuxihenry/SpatialTrackerV2-Offline",
]:
    print(f"downloading {repo} -> {cache}")
    snapshot_download(repo_id=repo, cache_dir=cache)
PY

echo "SpatialTrackerV2 environment and weights are ready under: $ROOT"
echo "activate with: conda activate '$ENV_PREFIX'"
