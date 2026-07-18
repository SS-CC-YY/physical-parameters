from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from remake_benchmark.core.errors import BenchmarkError
from remake_benchmark.orchestration import (
    evaluate_run,
    generate_run,
    prepare_run,
    run_sequential,
    summarize_api_run,
)


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remake-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="resolve a build and create canonical manifest.jsonl")
    prepare_parser.add_argument("--build", type=Path, required=True)
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--workspace-root", type=Path, default=None)
    prepare_parser.add_argument("--max-jobs", type=int, default=None)
    prepare_parser.add_argument("--no-check-inputs", action="store_true")
    prepare_parser.add_argument("--overwrite", action="store_true")

    generate_parser = subparsers.add_parser("generate", help="run model adapter jobs from a prepared run")
    generate_parser.add_argument("--run-dir", type=Path, required=True)
    _add_generation_options(generate_parser)

    sequence_parser = subparsers.add_parser(
        "sequence",
        help="for each prepared job: generate, physics-gate/evaluate, visualize, then continue",
    )
    sequence_parser.add_argument("--run-dir", type=Path, required=True)
    _add_generation_options(sequence_parser)
    sequence_parser.add_argument("--stop-on-invalid", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate canonical output videos")
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)

    api_summary_parser = subparsers.add_parser(
        "summarize-api", help="write per-video cost and timing tables for a closed-API run"
    )
    api_summary_parser.add_argument("--run-dir", type=Path, required=True)

    run_parser = subparsers.add_parser(
        "run", help="prepare and generate one build; evaluation is an explicit later step"
    )
    run_parser.add_argument("--build", type=Path, required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--workspace-root", type=Path, default=None)
    _add_generation_options(run_parser)
    return parser


def _positive_int_or_none(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise BenchmarkError(f"{name} must be positive")


def dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "prepare":
        _positive_int_or_none(args.max_jobs, "--max-jobs")
        resolved, jobs = prepare_run(
            args.build,
            args.run_dir,
            workspace_root_override=args.workspace_root,
            max_jobs=args.max_jobs,
            check_inputs=not args.no_check_inputs,
            overwrite=args.overwrite,
        )
        return {"build_id": resolved["build_id"], "jobs": len(jobs), "run_dir": str(args.run_dir.resolve())}
    if args.command == "generate":
        _positive_int_or_none(args.max_jobs, "--max-jobs")
        return generate_run(
            args.run_dir,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs,
            start_index=args.start_index,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
        )
    if args.command == "sequence":
        _positive_int_or_none(args.max_jobs, "--max-jobs")
        return run_sequential(
            args.run_dir,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs,
            start_index=args.start_index,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
            stop_on_invalid=args.stop_on_invalid,
        )
    if args.command == "evaluate":
        return evaluate_run(args.run_dir)
    if args.command == "summarize-api":
        return summarize_api_run(args.run_dir)
    if args.command == "run":
        _positive_int_or_none(args.max_jobs, "--max-jobs")
        resolved, jobs = prepare_run(
            args.build,
            args.run_dir,
            workspace_root_override=args.workspace_root,
            max_jobs=args.max_jobs,
            overwrite=args.overwrite,
        )
        generation = generate_run(
            args.run_dir,
            dry_run=args.dry_run,
            start_index=args.start_index,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
        )
        return {"build_id": resolved["build_id"], "jobs": len(jobs), "generation": generation}
    raise AssertionError(args.command)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = dispatch(args)
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False))
