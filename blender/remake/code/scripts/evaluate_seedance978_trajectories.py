#!/usr/bin/env python3
"""Fit and score physics from frozen trajectory CSV artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.trajectory_pipeline import (  # noqa: E402
    evaluate_extracted_seedance978,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
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
        "--output",
        type=Path,
        default=REMAKE_ROOT / "analysis" / "seedance978_physics_from_tracks",
    )
    parser.add_argument("--phase", choices=["all", "side", "main", "top", "robustness"], default="all")
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = evaluate_extracted_seedance978(
        extraction_root=args.tracks.resolve(),
        manifest_path=args.manifest.resolve(),
        registry_path=args.registry.resolve(),
        output_root=args.output.resolve(),
        phase=args.phase,
        job_ids=args.job_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
