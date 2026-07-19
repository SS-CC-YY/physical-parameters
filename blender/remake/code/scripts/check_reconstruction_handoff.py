#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent


REQUIRED_IMPORTS = {
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight a handed-off reconstruction MVP run.")
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument(
        "--videos",
        type=Path,
        default=WORKSPACE_ROOT / "rebuild-test" / "test-videos" / "videos",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expect-count", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    workspace_root = args.workspace_root.resolve()
    videos_dir = args.videos.resolve()
    output_root = None if args.output is None else args.output.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    missing_imports = [
        f"{module} (install package {package})"
        for module, package in REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_imports:
        errors.append("missing Python dependencies: " + ", ".join(missing_imports))

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "workspace_root": str(workspace_root),
        "videos_dir": str(videos_dir),
        "output_root": None if output_root is None else str(output_root),
    }
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required")
    if not workspace_root.is_dir():
        errors.append(f"workspace root does not exist: {workspace_root}")
    if not videos_dir.is_dir():
        errors.append(f"videos directory does not exist: {videos_dir}")
    if output_root is not None and (output_root / ".reconstruction_mvp.lock").exists():
        errors.append(
            f"output lock exists: {output_root / '.reconstruction_mvp.lock'}; "
            "verify no reconstruction process is running before removing it"
        )

    if not missing_imports and videos_dir.is_dir() and workspace_root.is_dir():
        sys.path.insert(0, str(CODE_ROOT / "src"))
        import cv2  # noqa: PLC0415

        from remake_benchmark.reconstruction.mvp import (  # noqa: PLC0415
            conditioning_image_for_job,
            parse_video_job,
        )

        videos = sorted(videos_dir.glob("*.mp4"))
        report["video_count"] = len(videos)
        if not videos:
            errors.append(f"no MP4 files directly inside: {videos_dir}")
        if args.expect_count is not None and len(videos) != args.expect_count:
            errors.append(f"expected {args.expect_count} MP4 files, found {len(videos)}")

        jobs = []
        media_profiles: set[tuple[int, int, float]] = set()
        conditioning_images: set[str] = set()
        for video in videos:
            try:
                job = parse_video_job(video)
                jobs.append(job)
            except Exception as exc:
                errors.append(f"invalid filename {video.name}: {exc}")
                continue
            supported = (
                job["experiment_id"] == "v1_A"
                and job["factors"]["object_id"] == "standard_ball"
                and job["factors"]["camera"] == "CAM_Side"
                and "gravity_g" in job["targets"]
            )
            if not supported:
                errors.append(
                    f"unsupported MVP profile in {video.name}; expected "
                    "v1_A/standard_ball/CAM_Side/gravity_g"
                )
            try:
                conditioning = conditioning_image_for_job(workspace_root, job)
                conditioning_images.add(str(conditioning))
                image = cv2.imread(str(conditioning), cv2.IMREAD_COLOR)
                if image is None:
                    errors.append(
                        f"conditioning image cannot be decoded (possibly an unfetched Git LFS pointer): {conditioning}"
                    )
                sibling_paths = list(conditioning.parent.parent.glob(f"*/{conditioning.name}"))
                decodable_siblings = sum(
                    cv2.imread(str(sibling), cv2.IMREAD_COLOR) is not None for sibling in sibling_paths
                )
                if decodable_siblings < 3:
                    warnings.append(
                        f"only {decodable_siblings} decodable same-scene object first frames found for {conditioning}; "
                        "tracking will lose the sibling-variation prior"
                    )
            except Exception as exc:
                errors.append(str(exc))

            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                errors.append(f"cannot open MP4: {video}")
                continue
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
            ok, _ = capture.read()
            capture.release()
            if not ok:
                errors.append(f"MP4 has no decodable first frame: {video}")
            if not math.isfinite(fps) or fps <= 0:
                warnings.append(f"MP4 backend did not report a usable FPS; timestamp fallback will be used: {video.name}")
            media_profiles.add((width, height, fps if math.isfinite(fps) else 0.0))

        report.update(
            {
                "job_count": len(jobs),
                "scenes": sorted({job["factors"]["scene_id"] for job in jobs}),
                "targets_gravity_m_s2": sorted(
                    {job["targets"]["gravity_g"] for job in jobs if "gravity_g" in job["targets"]}
                ),
                "conditioning_image_count": len(conditioning_images),
                "media_profiles_width_height_fps": [list(item) for item in sorted(media_profiles)],
            }
        )

    report["warnings"] = sorted(set(warnings))
    report["errors"] = sorted(set(errors))
    report["status"] = "ready" if not errors else "not_ready"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
