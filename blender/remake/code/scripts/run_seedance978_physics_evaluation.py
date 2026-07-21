#!/usr/bin/env python3
"""Evaluate the complete frozen Seedance set: Side first, then Main and Top."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.seedance978 import run_seedance978_evaluation  # noqa: E402


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
        "--registry",
        type=Path,
        default=REMAKE_ROOT / "benchmark_release_v1_0" / "experiment_registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REMAKE_ROOT / "analysis" / "seedance978_physics",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "side", "main", "top", "robustness"],
        default="all",
        help="all executes three resumable phases in the required Side -> Main -> Top order.",
    )
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--run-dynamic",
        action="store_true",
        help="Run SpaTrackerV2 for changed/borderline camera videos after the cheap validity pre-gate.",
    )
    parser.add_argument("--dynamic-frame-stride", type=int, default=1)
    parser.add_argument("--isolated-process", action="store_true")
    parser.add_argument("--overlay-count", type=int, default=12)
    parser.add_argument(
        "--skip-background-rigidity",
        action="store_true",
        help="Faster diagnostic mode; not recommended for final validity labels.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    phases = ["side", "main", "top"] if args.phase == "all" else [args.phase]
    result = None
    for phase in phases:
        print(f"=== Seedance 978 evaluation phase: {phase} ===", flush=True)
        result = run_seedance978_evaluation(
            workspace_root=args.workspace_root.resolve(),
            videos_dir=args.videos.resolve(),
            manifest_path=args.manifest.resolve(),
            audit_jsonl=args.camera_audit.resolve(),
            calibration_root=args.calibration_root.resolve(),
            registry_path=args.registry.resolve(),
            output_root=args.output.resolve(),
            spatialtracker_script=(
                REMAKE_ROOT / "rebuild-test" / "spatialtrackerv2" / "scripts" / "run_batch.py"
            ).resolve(),
            phase=phase,
            max_jobs=args.max_jobs,
            overwrite=args.overwrite,
            assess_background_rigidity=not args.skip_background_rigidity,
            overlay_count=args.overlay_count,
            run_dynamic=args.run_dynamic,
            dynamic_frame_stride=args.dynamic_frame_stride,
            isolated_process=args.isolated_process,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
