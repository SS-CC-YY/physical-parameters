#!/usr/bin/env python3
"""Build the video-first, human-readable PhysParamBench evidence package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.evidence_report import build_evidence_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render frozen videos, overlays, 2-D/3-D trajectories and concise "
            "inverse-physics formula cards. This command never reruns or changes grading."
        )
    )
    parser.add_argument("--all-jobs", type=Path, required=True, help="Frozen evaluation all_jobs.csv.")
    parser.add_argument(
        "--simple-report",
        type=Path,
        required=True,
        help="Compact paper report containing parameter_scans.jsonl and video_gate.csv.",
    )
    parser.add_argument("--evaluation-root", type=Path, required=True, help="Root containing jobs/<job_id>/result.json and trajectories.")
    parser.add_argument("--tracks-root", type=Path, help="Trajectory extraction root; needed for object_track_overlay.mp4 and dynamic 3-D native evidence.")
    parser.add_argument("--videos-root", type=Path, required=True, help="Directory containing <job_id>.mp4 originals.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "experiment_registry.json",
    )
    parser.add_argument("--primary-seed", type=int, default=341867882)
    parser.add_argument(
        "--media-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="hardlink avoids duplicating hundreds of videos; falls back safely when unsupported.",
    )
    parser.add_argument(
        "--skip-montages",
        action="store_true",
        help="Build all HTML/images but skip side-by-side MP4 creation (useful for a quick CPU smoke test).",
    )
    args = parser.parse_args()
    summary = build_evidence_report(
        all_jobs_csv=args.all_jobs.resolve(),
        simple_report_root=args.simple_report.resolve(),
        evaluation_root=args.evaluation_root.resolve(),
        tracks_root=None if args.tracks_root is None else args.tracks_root.resolve(),
        videos_root=args.videos_root.resolve(),
        output=args.output.resolve(),
        registry_path=args.registry.resolve(),
        media_mode=args.media_mode,
        build_montages=not args.skip_montages,
        primary_seed=args.primary_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
