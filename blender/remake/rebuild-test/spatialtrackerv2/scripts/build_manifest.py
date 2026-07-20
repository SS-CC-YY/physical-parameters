#!/usr/bin/env python3
"""Build the fixed 27-video V1A/Seedance SpatialTrackerV2 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REMAKE_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "v1a_seedance27.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "manifests" / "v1a_seedance27.jsonl"


def rel(path: Path) -> str:
    return path.resolve().relative_to(REMAKE_ROOT.resolve()).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=REMAKE_ROOT / "seedanceVideos.tar" / "videos",
    )
    parser.add_argument("--strict", action="store_true", help="Fail unless all 27 inputs exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    missing = []
    alias = config.get("calibration_scene_aliases", {})

    for scene in config["scenes"]:
        for camera in config["cameras"]:
            job_id = (
                f"{config['experiment_id']}__{config['factor_token']}__{scene}__"
                f"{config['object_id']}__{camera}__seed-{config['seed']}"
            )
            video = args.videos_root / f"{job_id}.mp4"
            first_frame = (
                REMAKE_ROOT
                / "first_frames_v1_0"
                / "images"
                / config["experiment_id"]
                / scene
                / config["object_id"]
                / f"{camera}.png"
            )
            calibration_scene = alias.get(scene, scene)
            calibration = (
                REMAKE_ROOT
                / "code"
                / "assets"
                / "v1a_calibration"
                / calibration_scene
                / f"{camera}.json"
            )
            anchor = None
            if calibration.is_file():
                calibration_data = json.loads(calibration.read_text(encoding="utf-8"))
                anchor = calibration.parent.parent / calibration_data["reference_geometry_anchors"]["path"]
            required = [video, first_frame, calibration] + ([anchor] if anchor is not None else [])
            missing.extend(str(path) for path in required if not path.is_file())
            rows.append(
                {
                    "job_id": job_id,
                    "experiment_id": config["experiment_id"],
                    "factor": config["factor"],
                    "factor_token": config["factor_token"],
                    "physical_value": config["physical_value"],
                    "physical_unit": config["physical_unit"],
                    "scene": scene,
                    "camera": camera,
                    "object_id": config["object_id"],
                    "seed": config["seed"],
                    "source_fps": config["source_fps"],
                    "video": rel(video),
                    "first_frame": rel(first_frame),
                    "calibration": rel(calibration),
                    "anchors": rel(anchor) if anchor is not None else None,
                }
            )

    if len(rows) != 27:
        raise RuntimeError(f"Expected 27 jobs, built {len(rows)}")
    if missing and args.strict:
        preview = "\n".join(f"  - {item}" for item in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} required files:\n{preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"manifest: {args.output}")
    print(f"jobs: {len(rows)}")
    print(f"missing required files: {len(missing)}")


if __name__ == "__main__":
    main()
