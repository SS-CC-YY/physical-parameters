#!/usr/bin/env python3
"""Extract one auditable per-frame object trajectory and overlay per video."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.trajectory_pipeline import (  # noqa: E402
    run_seedance978_trajectory_extraction,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REMAKE_ROOT)
    parser.add_argument("--videos", type=Path, default=REMAKE_ROOT / "seedanceVideos.tar" / "videos")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl",
    )
    parser.add_argument(
        "--camera-audit",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "camera_motion.jsonl",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=CODE_ROOT / "assets" / "fixed_camera_ball_calibration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REMAKE_ROOT / "analysis" / "seedance978_tracks",
    )
    parser.add_argument("--phase", choices=["all", "side", "main", "top", "robustness"], default="all")
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--dynamic-frame-stride", type=int, default=1)
    parser.add_argument("--isolated-process", action="store_true")
    parser.add_argument("--skip-background-rigidity", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all 978 inputs and write routing/runtime manifests without tracking.",
    )
    parser.add_argument(
        "--spatialtracker-root",
        type=Path,
        default=None,
        help="Path to a separately cloned official SpaTrackerV2 repository.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.spatialtracker_root is not None:
        root = args.spatialtracker_root.expanduser().resolve()
        if not (root / "inference.py").is_file() or not (root / "models").is_dir():
            raise FileNotFoundError(f"not an official SpaTrackerV2 checkout: {root}")
        os.environ["SPATIALTRACKERV2_ROOT"] = str(root)
    result = run_seedance978_trajectory_extraction(
        workspace_root=args.workspace_root.resolve(),
        videos_dir=args.videos.resolve(),
        manifest_path=args.manifest.resolve(),
        audit_jsonl=args.camera_audit.resolve(),
        calibration_root=args.calibration_root.resolve(),
        output_root=args.output.resolve(),
        spatialtracker_script=(
            REMAKE_ROOT / "rebuild-test" / "spatialtrackerv2" / "scripts" / "run_batch.py"
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
