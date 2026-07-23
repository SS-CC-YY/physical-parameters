#!/usr/bin/env python3
"""Build simple WorldBench-style tables from one or more evaluated models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.simple_metrics import (  # noqa: E402
    build_simple_physics_report,
    parse_model_arguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=EVALUATION_ROOT",
        help="Repeat for every model to compare.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "experiment_registry.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing per-job results for an explicitly incomplete progress report.",
    )
    args = parser.parse_args()
    summary = build_simple_physics_report(
        parse_model_arguments(args.model),
        output=args.output,
        registry_path=args.registry,
        manifest_path=args.manifest,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
