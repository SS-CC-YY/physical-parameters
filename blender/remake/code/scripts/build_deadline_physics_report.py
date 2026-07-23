#!/usr/bin/env python3
"""Build a compact four-model, human-readable PhysParamBench report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.deadline_report import (  # noqa: E402
    build_deadline_report,
    parse_model_arguments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=EVALUATION_ROOT",
        help=(
            "Repeat once per model. EVALUATION_ROOT must contain all_jobs.csv "
            "and preferably jobs/<job_id>/result.json."
        ),
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
        help="Frozen 978-job universe used for completeness checks.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a visibly incomplete progress report; never use it for the final paper table.",
    )
    args = parser.parse_args()
    summary = build_deadline_report(
        parse_model_arguments(args.model),
        output=args.output,
        registry_path=args.registry,
        manifest_path=args.manifest,
        expected_manifest_jobs=978,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
