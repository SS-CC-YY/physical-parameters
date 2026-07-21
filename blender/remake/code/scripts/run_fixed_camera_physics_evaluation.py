#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.physics_evaluation import (  # noqa: E402
    run_physics_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track the standard ball under a fixed-camera assumption, recover a metric trajectory "
            "from first-frame GT size plus K/R/t, fit experiment parameters, and score against registry GT."
        )
    )
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=CODE_ROOT / "assets" / "fixed_camera_ball_calibration",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=WORKSPACE_ROOT / "benchmark_release_v1_0" / "experiment_registry.json",
    )
    parser.add_argument("--glob", default="*.mp4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--one-per-experiment",
        action="store_true",
        help="Smoke-test mode: retain only the first matching video for each experiment.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    aggregate = run_physics_batch(
        args.videos.resolve(),
        calibration_root=args.calibration_root.resolve(),
        registry_path=args.registry.resolve(),
        output_root=args.output.resolve(),
        video_glob=args.glob,
        limit=args.limit,
        one_per_experiment=args.one_per_experiment,
        overwrite=args.overwrite,
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
