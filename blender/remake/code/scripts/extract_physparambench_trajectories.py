#!/usr/bin/env python3
"""Extract standardized per-frame trajectories for any benchmark model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument(
        "--camera-audit",
        type=Path,
        required=True,
        help=(
            "Model-specific camera_motion_audit.jsonl produced by "
            "audit_camera_motion.py; never reuse another model's audit."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=REMAKE_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=CODE_ROOT / "assets" / "fixed_camera_ball_calibration",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "side", "main", "top", "robustness"),
        default="all",
    )
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--dynamic-frame-stride", type=int, default=1)
    parser.add_argument("--isolated-process", action="store_true")
    parser.add_argument("--skip-background-rigidity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--spatialtracker-root",
        type=Path,
        default=None,
        help="Path to the separately cloned official SpaTrackerV2 repository.",
    )
    args = parser.parse_args()
    videos = args.videos.resolve()
    output = args.output.resolve()
    camera_audit = args.camera_audit.resolve()
    if not camera_audit.is_file():
        parser.error(f"--camera-audit does not exist: {camera_audit}")
    if videos == output:
        parser.error("--output must not be the same directory as --videos")
    if args.spatialtracker_root is not None:
        root = args.spatialtracker_root.expanduser().resolve()
        if not (root / "inference.py").is_file() or not (root / "models").is_dir():
            parser.error(f"not an official SpaTrackerV2 checkout: {root}")
        os.environ["SPATIALTRACKERV2_ROOT"] = str(root)

    # Delay the OpenCV-heavy pipeline import so --help remains available in a
    # lightweight shell used only to inspect commands.
    from remake_benchmark.reconstruction.trajectory_pipeline import (  # noqa: E402
        run_seedance978_trajectory_extraction,
    )

    result = run_seedance978_trajectory_extraction(
        workspace_root=args.workspace_root.resolve(),
        videos_dir=videos,
        manifest_path=args.manifest.resolve(),
        audit_jsonl=camera_audit,
        calibration_root=args.calibration_root.resolve(),
        output_root=output,
        spatialtracker_script=(
            REMAKE_ROOT
            / "rebuild-test"
            / "spatialtrackerv2"
            / "scripts"
            / "run_batch.py"
        ).resolve(),
        phase=args.phase,
        job_ids=args.job_id,
        max_jobs=args.max_jobs,
        overwrite=args.overwrite,
        assess_background_rigidity=not args.skip_background_rigidity,
        workers=args.workers,
        run_dynamic=not args.skip_dynamic,
        dynamic_frame_stride=args.dynamic_frame_stride,
        isolated_process=args.isolated_process,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
