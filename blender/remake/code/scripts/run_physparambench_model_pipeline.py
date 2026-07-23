#!/usr/bin/env python3
"""Run camera audit -> trajectory extraction -> fitting -> simple report for one model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Model-specific root; camera_audit/, tracks/, evaluation/, report/ are created here.",
    )
    parser.add_argument("--workspace-root", type=Path, default=REMAKE_ROOT)
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
        "--calibration-root",
        type=Path,
        default=CODE_ROOT / "assets" / "fixed_camera_ball_calibration",
    )
    parser.add_argument(
        "--camera-audit",
        type=Path,
        default=None,
        help="Reuse this model-specific audit instead of creating output-root/camera_audit.",
    )
    parser.add_argument(
        "--spatialtracker-root",
        type=Path,
        default=None,
        help="Separately cloned official SpaTrackerV2 checkout, required for dynamic routes.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "audit", "extract", "fit", "report"),
        default="all",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "side", "main", "top", "robustness"),
        default="all",
    )
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-audit", action="store_true")
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help=(
            "Extraction preview only: leave moving-camera jobs pending. "
            "It cannot be combined with the full all->fit pipeline."
        ),
    )
    parser.add_argument("--skip-background-rigidity", action="store_true")
    parser.add_argument(
        "--shared-dynamic-process",
        action="store_true",
        help="Disable per-job process isolation; isolation is the safer default.",
    )
    parser.add_argument("--audit-contact-sheets", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow report generation before all manifest jobs finish.",
    )
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _audit_video_inventory(
    videos: Path,
    manifest_path: Path,
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    expected: list[str] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("job_id"):
            raise ValueError(f"invalid manifest row at {manifest_path}:{line_number}")
        expected.append(f"{value['job_id']}.mp4")
    if len(expected) != len(set(expected)):
        raise ValueError(f"duplicate job_id in manifest: {manifest_path}")
    actual_paths = sorted(videos.glob("*.mp4"))
    actual = {path.name for path in actual_paths}
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    unexpected = sorted(actual - expected_set)
    empty = sorted(path.name for path in actual_paths if path.stat().st_size <= 0)
    if unexpected:
        raise ValueError(
            f"video directory contains {len(unexpected)} unexpected MP4 file(s); "
            f"first: {unexpected[:3]}"
        )
    if empty:
        raise ValueError(
            f"video directory contains {len(empty)} empty MP4 file(s); first: {empty[:3]}"
        )
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"video directory is missing {len(missing)} frozen task(s); "
            f"first: {missing[:3]}. Use --allow-partial only for a progress run."
        )
    return {
        "expected": len(expected),
        "found": len(actual),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "empty": len(empty),
        "complete": not missing and not unexpected and not empty,
        "missing_examples": missing[:10],
    }


def main() -> None:
    args = _parser().parse_args()
    videos = args.videos.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not videos.is_dir():
        raise FileNotFoundError(f"video directory does not exist: {videos}")
    if videos == output_root:
        raise ValueError("--output-root must not be the input video directory")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    video_inventory = _audit_video_inventory(
        videos,
        manifest_path,
        allow_partial=args.allow_partial,
    )
    print(
        "VIDEO INVENTORY "
        f"{video_inventory['found']}/{video_inventory['expected']} "
        f"complete={video_inventory['complete']}",
        flush=True,
    )

    audit_root = output_root / "camera_audit"
    tracks_root = output_root / "tracks"
    evaluation_root = output_root / "evaluation"
    report_root = output_root / "report"
    audit_jsonl = (
        args.camera_audit.expanduser().resolve()
        if args.camera_audit is not None
        else audit_root / "camera_motion_audit.jsonl"
    )
    stages = ("audit", "extract", "fit", "report") if args.stage == "all" else (args.stage,)
    if (
        args.skip_dynamic
        and "extract" in stages
        and ("fit" in stages or "report" in stages)
    ):
        raise ValueError(
            "--skip-dynamic may only be used with --stage extract. A complete fit/report "
            "requires SpaTrackerV2 outputs for every moving-camera job."
        )
    stage_results: dict[str, Any] = {}

    if "audit" in stages:
        if args.camera_audit is not None:
            if not audit_jsonl.is_file():
                raise FileNotFoundError(audit_jsonl)
            print(f"REUSE camera audit: {audit_jsonl}", flush=True)
            stage_results["audit"] = {"status": "reused", "camera_audit": str(audit_jsonl)}
        elif audit_jsonl.is_file() and not args.overwrite_audit:
            print(f"SKIP camera audit: {audit_jsonl}", flush=True)
            stage_results["audit"] = {"status": "reused", "camera_audit": str(audit_jsonl)}
        else:
            command = [
                sys.executable,
                str(CODE_ROOT / "scripts" / "audit_camera_motion.py"),
                "--videos",
                str(videos),
                "--output",
                str(audit_root),
                "--workers",
                str(args.audit_workers),
            ]
            if not args.audit_contact_sheets:
                command.append("--no-contact-sheets")
            print("START camera audit", flush=True)
            subprocess.run(command, check=True)
            stage_results["audit"] = {"status": "completed", "camera_audit": str(audit_jsonl)}

    if "extract" in stages:
        if not audit_jsonl.is_file():
            raise FileNotFoundError(
                f"camera audit missing: {audit_jsonl}; run --stage audit first"
            )
        spatialtracker_root = (
            args.spatialtracker_root.expanduser().resolve()
            if args.spatialtracker_root is not None
            else None
        )
        if not args.skip_dynamic:
            if spatialtracker_root is None:
                raise ValueError(
                    "--spatialtracker-root is required unless --skip-dynamic is used"
                )
            if not (spatialtracker_root / "inference.py").is_file():
                raise FileNotFoundError(
                    f"not an official SpaTrackerV2 checkout: {spatialtracker_root}"
                )
            os.environ["SPATIALTRACKERV2_ROOT"] = str(spatialtracker_root)
        from remake_benchmark.reconstruction.trajectory_pipeline import (  # noqa: E402
            run_seedance978_trajectory_extraction,
        )

        print("START trajectory extraction", flush=True)
        stage_results["extract"] = run_seedance978_trajectory_extraction(
            workspace_root=args.workspace_root.expanduser().resolve(),
            videos_dir=videos,
            manifest_path=manifest_path,
            audit_jsonl=audit_jsonl,
            calibration_root=args.calibration_root.expanduser().resolve(),
            output_root=tracks_root,
            spatialtracker_script=(
                REMAKE_ROOT
                / "rebuild-test"
                / "spatialtrackerv2"
                / "scripts"
                / "run_batch.py"
            ).resolve(),
            phase=args.phase,
            job_ids=args.job_id,
            max_jobs=args.max_jobs,
            overwrite=args.overwrite,
            assess_background_rigidity=not args.skip_background_rigidity,
            workers=args.workers,
            run_dynamic=not args.skip_dynamic,
            dynamic_frame_stride=1,
            isolated_process=not args.shared_dynamic_process,
            dry_run=False,
        )

    if "fit" in stages:
        from remake_benchmark.reconstruction.trajectory_pipeline import (  # noqa: E402
            evaluate_extracted_seedance978,
        )

        if not tracks_root.is_dir():
            raise FileNotFoundError(f"trajectory root missing: {tracks_root}")
        print("START equation fitting and parameter recovery", flush=True)
        stage_results["fit"] = evaluate_extracted_seedance978(
            extraction_root=tracks_root,
            manifest_path=manifest_path,
            registry_path=args.registry.expanduser().resolve(),
            output_root=evaluation_root,
            phase=args.phase,
            job_ids=args.job_id,
            overwrite=args.overwrite,
        )

    if "report" in stages:
        from remake_benchmark.reconstruction.simple_metrics import (  # noqa: E402
            build_simple_physics_report,
        )

        if not evaluation_root.is_dir():
            raise FileNotFoundError(f"evaluation root missing: {evaluation_root}")
        print("START simple paper report", flush=True)
        stage_results["report"] = build_simple_physics_report(
            {args.model_name: evaluation_root},
            output=report_root,
            registry_path=args.registry.expanduser().resolve(),
            manifest_path=manifest_path,
            allow_partial=args.allow_partial,
        )

    summary = {
        "schema_version": "1.0.0",
        "model_name": args.model_name,
        "videos": str(videos),
        "video_inventory": video_inventory,
        "output_root": str(output_root),
        "camera_audit": str(audit_jsonl),
        "tracks": str(tracks_root),
        "evaluation": str(evaluation_root),
        "report": str(report_root),
        "stage_results": stage_results,
    }
    _write_json(output_root / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
