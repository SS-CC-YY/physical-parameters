#!/usr/bin/env python3
"""Fail-fast checks before spending GPU time."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REMAKE_ROOT = PACKAGE_ROOT.parents[1]
MANIFEST = PACKAGE_ROOT / "manifests" / "v1a_seedance27.jsonl"


def module_version(module: object) -> object:
    """Read an eagerly defined version without invoking module __getattr__."""

    return vars(module).get("__version__", "installed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--expected-jobs",
        type=int,
        default=None,
        help="Optional exact manifest-size assertion; omitted for arbitrary/978 manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = [
        "torch",
        "torchvision",
        "cv2",
        "decord",
        "numpy",
        "scipy",
        "matplotlib",
        "huggingface_hub",
        "utils3d",
        "pycolmap",
        "pyceres",
    ]
    failures = []
    versions = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            # Some dependencies (notably EasternJournalist/utils3d) implement a
            # dynamic module-level __getattr__.  Calling getattr for an absent
            # __version__ then tries to import ``utils3d.__version__`` and turns
            # a successful package import into a false dependency failure.
            versions[name] = module_version(module)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError("Missing/broken Python dependencies:\n  - " + "\n  - ".join(failures))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; SpatialTrackerV2 full inference requires an NVIDIA GPU")
    upstream = Path(
        os.environ.get(
            "SPATIALTRACKERV2_ROOT",
            str(PACKAGE_ROOT / "upstream" / "SpaTrackerV2"),
        )
    ).expanduser().resolve()
    sys.path.insert(0, str(upstream))
    try:
        from models.SpaTrackV2.models.predictor import Predictor
        from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
        from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"SpatialTrackerV2 upstream import failed: {exc}") from exc
    required_forward_parameters = {
        "video",
        "depth",
        "unc_metric",
        "intrs",
        "extrs",
        "queries",
        "iters_track",
        "full_point",
        "fps",
        "fixed_cam",
        "query_no_BA",
        "stage",
        "support_frame",
        "replace_ratio",
    }
    forward_parameters = set(inspect.signature(Predictor.forward).parameters)
    missing_forward = sorted(required_forward_parameters - forward_parameters)
    if missing_forward:
        raise RuntimeError(
            "The external SpaTrackerV2 checkout is incompatible with the benchmark "
            f"adapter; Predictor.forward is missing {missing_forward}"
        )
    for model_class in (Predictor, VGGT4Track):
        if not callable(getattr(model_class, "from_pretrained", None)):
            raise RuntimeError(f"{model_class.__name__}.from_pretrained is unavailable")
    manifest = args.manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Build the manifest first: {manifest}")
    jobs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = []
    invalid = []
    import cv2

    for job in jobs:
        for key in ["video", "first_frame", "calibration"]:
            value = job.get(key)
            if value is None:
                missing.append(f"manifest job {job.get('job_id')} missing key {key}")
                continue
            path = Path(value)
            if not path.is_absolute():
                path = REMAKE_ROOT / path
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                with path.open("rb") as handle:
                    prefix = handle.read(128)
            except OSError as exc:
                invalid.append(f"{path}: cannot read ({exc})")
                continue
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                invalid.append(f"{path}: unresolved Git LFS pointer")
                continue
            if key == "first_frame" and cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
                invalid.append(f"{path}: PNG/image is not decodable")
            elif key == "calibration":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(value, dict):
                        raise ValueError("root is not an object")
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    invalid.append(f"{path}: invalid calibration JSON ({exc})")
            elif key == "video":
                capture = cv2.VideoCapture(str(path))
                ok, _ = capture.read() if capture.isOpened() else (False, None)
                capture.release()
                if not ok:
                    invalid.append(f"{path}: MP4 is not decodable")
    if args.expected_jobs is not None and len(jobs) != args.expected_jobs:
        raise RuntimeError(f"Manifest jobs={len(jobs)}, expected={args.expected_jobs}")
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:20])
        raise RuntimeError(f"Manifest jobs={len(jobs)}, missing files={len(missing)}:\n{preview}")
    if invalid:
        preview = "\n".join(f"  - {item}" for item in invalid[:20])
        raise RuntimeError(f"Manifest jobs={len(jobs)}, invalid files={len(invalid)}:\n{preview}")
    try:
        if not (upstream / ".git").exists():
            raise FileNotFoundError("upstream is a vendored snapshot, not a standalone checkout")
        git_prefix = ["git", "-c", f"safe.directory={upstream}", "-C", str(upstream)]
        upstream_commit = subprocess.check_output(
            [*git_prefix, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        upstream_dirty = bool(
            subprocess.check_output(
                [*git_prefix, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        upstream_commit = None
        upstream_dirty = None
    report = {
        "python": sys.executable,
        "versions": versions,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "manifest_jobs": len(jobs),
        "manifest": str(manifest),
        "spatialtrackerv2_root": str(upstream),
        "spatialtrackerv2_git_commit": upstream_commit,
        "spatialtrackerv2_git_dirty": upstream_dirty,
        "adapter_private_api_signature_check": "passed",
        "model_cache": str(PACKAGE_ROOT / "models" / "huggingface"),
        "upstream_inference": str(upstream / "inference.py"),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
