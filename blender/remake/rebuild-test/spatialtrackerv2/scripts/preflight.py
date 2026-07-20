#!/usr/bin/env python3
"""Fail-fast checks before spending GPU time."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REMAKE_ROOT = PACKAGE_ROOT.parents[1]
MANIFEST = PACKAGE_ROOT / "manifests" / "v1a_seedance27.jsonl"


def main() -> None:
    modules = ["torch", "torchvision", "cv2", "decord", "numpy", "matplotlib", "huggingface_hub"]
    failures = []
    versions = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError("Missing/broken Python dependencies:\n  - " + "\n  - ".join(failures))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; SpatialTrackerV2 full inference requires an NVIDIA GPU")
    upstream = PACKAGE_ROOT / "upstream" / "SpaTrackerV2"
    sys.path.insert(0, str(upstream))
    try:
        from models.SpaTrackV2.models.predictor import Predictor  # noqa: F401
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"SpatialTrackerV2 upstream import failed: {exc}") from exc
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Build the manifest first: {MANIFEST}")
    jobs = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = []
    for job in jobs:
        for key in ["video", "first_frame", "calibration", "anchors"]:
            path = REMAKE_ROOT / job[key]
            if not path.is_file():
                missing.append(str(path))
    if len(jobs) != 27 or missing:
        raise RuntimeError(f"Manifest jobs={len(jobs)}, missing files={len(missing)}")
    report = {
        "python": sys.executable,
        "versions": versions,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "manifest_jobs": len(jobs),
        "model_cache": str(PACKAGE_ROOT / "models" / "huggingface"),
        "upstream_inference": str(upstream / "inference.py"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
