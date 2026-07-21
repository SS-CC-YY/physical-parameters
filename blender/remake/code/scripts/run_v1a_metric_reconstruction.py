#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.v1a_metric import (  # noqa: E402
    load_v1a_config,
    run_v1a_metric_batch,
)


def _parser() -> argparse.ArgumentParser:
    workspace = CODE_ROOT.parent
    parser = argparse.ArgumentParser(
        description="Reconstruct calibrated metric V1A object trajectories from generated MP4 files."
    )
    parser.add_argument("--workspace-root", type=Path, default=workspace)
    parser.add_argument("--videos", type=Path, required=True, help="Directory containing V1A standard-ball MP4 files.")
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=CODE_ROOT / "assets" / "v1a_calibration",
        help="Directory produced by export_v1a_calibration.py.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_ROOT / "configs" / "reconstruction" / "v1a_metric_v1.yaml",
    )
    parser.add_argument("--overlay-count", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--glob",
        default="*.mp4",
        help="Optional deterministic filename glob, e.g. 'v1_A__g9p81__indoor[34]__standard_ball__CAM_*__seed-*.mp4'.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_v1a_metric_batch(
        workspace_root=args.workspace_root.resolve(),
        videos_dir=args.videos.resolve(),
        calibration_root=args.calibration_root.resolve(),
        output_root=args.output.resolve(),
        config=load_v1a_config(args.config),
        overlay_count=args.overlay_count,
        limit=args.limit,
        video_glob=args.glob,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
