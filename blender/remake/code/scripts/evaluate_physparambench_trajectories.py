#!/usr/bin/env python3
"""Fit all 13 experiment equations from frozen trajectory artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Standardized extraction root containing jobs/<job_id>/track_result.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "experiment_registry.json",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "side", "main", "top", "robustness"),
        default="all",
    )
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    tracks = args.tracks.resolve()
    output = args.output.resolve()
    if tracks == output:
        parser.error("--output must not be the same directory as --tracks")
    # Keep --help usable in a light environment; the trajectory evaluator's
    # OpenCV dependency is imported only for an actual run.
    from remake_benchmark.reconstruction.trajectory_pipeline import (  # noqa: E402
        evaluate_extracted_seedance978,
    )

    result = evaluate_extracted_seedance978(
        extraction_root=tracks,
        manifest_path=args.manifest.resolve(),
        registry_path=args.registry.resolve(),
        output_root=output,
        phase=args.phase,
        job_ids=args.job_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
