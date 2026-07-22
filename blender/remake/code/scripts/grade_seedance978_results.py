#!/usr/bin/env python3
"""Assign auditable G0--G4 grades from frozen Seedance trajectory results."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.grading_visuals import (  # noqa: E402
    write_grade_funnel,
    write_motion_type_plot,
    write_parameter_response_plot,
    write_trajectory_comparison,
    write_trajectory_fit_evidence,
    write_video_grade_card,
)
from remake_benchmark.reconstruction.hierarchical_grading import (  # noqa: E402
    aggregate_experiment_scan_grades,
    build_cross_scene_parameter_scan_grades,
    build_parameter_scan_grades,
    grade_video_result,
    load_json,
    missing_video_grade,
    read_trajectory_csv,
    resolve_trajectory_path,
    scan_grade_summary_row,
    video_grade_summary_row,
    write_csv,
    write_json,
    write_jsonl,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _manifest_job(row: dict[str, Any]) -> dict[str, Any]:
    factors = row.get("factors", {})
    return {
        "experiment_id": row.get("experiment_id"),
        "parameter_tuple_id": factors.get("parameter_tuple_id"),
        "scene_id": factors.get("scene_id"),
        "object_id": factors.get("object_id", "standard_ball"),
        "camera_name": factors.get("camera"),
        "seed": int(row.get("seed", 0)),
        "video_name": f"{row['job_id']}.mp4",
    }


def _load_adjudications(path: Path | None, known_job_ids: set[str]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_jsonl(path.resolve())
    output: dict[str, dict[str, Any]] = {}
    allowed = {"pass", "fail", "indeterminate"}
    for line_number, row in enumerate(rows, 1):
        job_id = str(row.get("job_id") or "")
        if not job_id:
            raise ValueError(f"{path}:{line_number}: adjudication requires job_id")
        if job_id in output:
            raise ValueError(f"{path}:{line_number}: duplicate adjudication for {job_id}")
        if job_id not in known_job_ids:
            raise ValueError(f"{path}:{line_number}: unknown adjudication job_id {job_id}")
        scope = str(row.get("decision_scope") or "g0_generation_validity")
        if scope != "g0_generation_validity":
            raise ValueError(
                f"{path}:{line_number}: decision_scope must be g0_generation_validity; "
                "manual overrides cannot assign G1/G2/G3/G4"
            )
        status = str(row.get("g0_status") or "").lower()
        if status not in allowed:
            raise ValueError(
                f"{path}:{line_number}: g0_status must be one of {sorted(allowed)}"
            )
        evidence_paths = row.get("evidence_paths", {})
        if not isinstance(evidence_paths, dict):
            raise ValueError(f"{path}:{line_number}: evidence_paths must be a JSON object")
        reason_codes = row.get("reason_codes", [])
        if not isinstance(reason_codes, (str, list, tuple)):
            raise ValueError(f"{path}:{line_number}: reason_codes must be a string or list")
        copied = dict(row)
        copied["job_id"] = job_id
        copied["decision_scope"] = scope
        copied["g0_status"] = status
        output[job_id] = copied
    return output


def _grade_video_with_optional_adjudication(
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    motion_profile: dict[str, Any],
    adjudication_override: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "policy": policy,
        "motion_profile": motion_profile,
    }
    if adjudication_override is not None:
        parameters = inspect.signature(grade_video_result).parameters
        if "adjudication_override" not in parameters:
            raise RuntimeError(
                "--adjudication-jsonl was supplied, but "
                "hierarchical_grading.grade_video_result does not yet expose "
                "adjudication_override=. Add that keyword-only interface before using "
                "manual G0 decisions; no override was silently applied."
            )
        kwargs["adjudication_override"] = adjudication_override
    return grade_video_result(result, rows, **kwargs)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _merge_evidence_rows(
    existing: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge stage-specific evidence without erasing paths from earlier runs."""

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for source in (existing, current):
        for row in source:
            key = (str(row.get("entity_scope") or ""), str(row.get("entity_id") or ""))
            if not all(key):
                continue
            if key not in merged:
                merged[key] = {}
                order.append(key)
            target = merged[key]
            for name, value in row.items():
                # A no-visuals or partial-stage rerun must not erase an already
                # generated artifact path with an empty cell.
                if value not in (None, "") or name not in target:
                    target[str(name)] = value
    return [merged[key] for key in order]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True, help="Directory containing jobs/<job_id>/result.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl")
    parser.add_argument("--registry", type=Path, default=CODE_ROOT / "assets" / "seedance978_evaluation" / "experiment_registry.json")
    parser.add_argument("--policy", type=Path, default=CODE_ROOT / "configs" / "evaluations" / "hierarchical_physics_grading_v1.json")
    parser.add_argument("--motion-profile", type=Path, default=CODE_ROOT / "configs" / "evaluations" / "motion_type_profiles_v1.json")
    parser.add_argument(
        "--adjudication-jsonl",
        type=Path,
        default=None,
        help="Optional frozen human G0 adjudications; never allowed to assign G1/G2/G3/G4",
    )
    parser.add_argument("--stage", choices=["video", "response", "all"], default="all")
    parser.add_argument("--job-id", action="append", default=None, help="Limit video evidence rendering; response grading still audits all available jobs")
    parser.add_argument("--scan-id", action="append", default=None, help="Render only selected atomic or cross-scene parameter scans")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--all-scan-visuals", action="store_true", help="Also render non-canonical edge scans")
    return parser


def main() -> None:
    args = _parser().parse_args()
    evaluation_root = args.evaluation_root.resolve()
    output = (args.output or evaluation_root / "grading_v1").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _read_jsonl(args.manifest.resolve())
    registry = load_json(args.registry.resolve())
    policy = load_json(args.policy.resolve())
    motion_profile = load_json(args.motion_profile.resolve())
    requested = set(args.job_id or [])
    requested_scans = set(args.scan_id or [])
    known = {str(row["job_id"]) for row in manifest}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown job ids: {unknown}")
    adjudications = _load_adjudications(args.adjudication_jsonl, known)

    write_json(output / "grading_config.json", {
        "schema_version": "1.0.0",
        "policy": policy,
        "motion_type_profile": motion_profile,
        "sources": {
            "evaluation_root": str(evaluation_root),
            "manifest": str(args.manifest.resolve()),
            "registry": str(args.registry.resolve()),
            "policy": str(args.policy.resolve()),
            "motion_profile": str(args.motion_profile.resolve()),
            "adjudication_jsonl": (
                str(args.adjudication_jsonl.resolve())
                if args.adjudication_jsonl is not None
                else None
            ),
        },
        "adjudication_record_count": len(adjudications),
    })

    results_by_id: dict[str, dict[str, Any]] = {}
    grades_by_id: dict[str, dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    if args.stage in {"video", "all"}:
        for ordinal, manifest_row in enumerate(manifest, 1):
            job_id = str(manifest_row["job_id"])
            result_path = evaluation_root / "jobs" / job_id / "result.json"
            rows: list[dict[str, Any]] = []
            if result_path.is_file():
                result = load_json(result_path)
                results_by_id[job_id] = result
                trajectory_path = resolve_trajectory_path(result)
                if trajectory_path is not None:
                    rows = read_trajectory_csv(trajectory_path)
                grade = _grade_video_with_optional_adjudication(
                    result,
                    rows,
                    policy=policy,
                    motion_profile=motion_profile,
                    adjudication_override=adjudications.get(job_id),
                )
            else:
                grade = missing_video_grade(job_id, _manifest_job(manifest_row), policy)
                if job_id in adjudications:
                    grade["adjudication_override"] = {
                        "applied": False,
                        "reason": "evaluation_result_missing",
                        "record": adjudications[job_id],
                    }
            job_output = output / "jobs" / job_id
            generated: dict[str, Any] = {}
            render_this = not args.no_visuals and (not requested or job_id in requested)
            if render_this:
                generated["grade_card"] = write_video_grade_card(job_output / "grade_card.png", grade)
                if rows:
                    generated["motion_type_plot"] = write_motion_type_plot(job_output / "motion_type_plot.png", rows, grade)
                    result = results_by_id.get(job_id)
                    if result is not None:
                        generated["trajectory_fit"] = write_trajectory_fit_evidence(job_output, rows, result, grade)
            grade["generated_evidence_paths"] = generated
            write_json(job_output / "grade.json", grade)
            grades_by_id[job_id] = grade
            evidence_rows.append({
                "entity_scope": "single_video",
                "entity_id": job_id,
                "grade": grade.get("video_stage"),
                "grade_json": str(job_output / "grade.json"),
                "grade_card": (generated.get("grade_card") or {}).get("png") if isinstance(generated.get("grade_card"), dict) else None,
                "motion_type_plot": (generated.get("motion_type_plot") or {}).get("png") if isinstance(generated.get("motion_type_plot"), dict) else None,
                "trajectory_fit_plot": ((generated.get("trajectory_fit") or {}).get("trajectory_fit_plot") or {}).get("png") if isinstance(generated.get("trajectory_fit"), dict) else None,
            })
            print(f"[video {ordinal}/{len(manifest)}] {job_id} -> {grade['video_stage']}", flush=True)
        write_csv(output / "video_grades.csv", [video_grade_summary_row(grades_by_id[str(row["job_id"])]) for row in manifest])
        write_jsonl(output / "video_grades.jsonl", [grades_by_id[str(row["job_id"])] for row in manifest])
    else:
        for manifest_row in manifest:
            job_id = str(manifest_row["job_id"])
            result_path = evaluation_root / "jobs" / job_id / "result.json"
            grade_path = output / "jobs" / job_id / "grade.json"
            if result_path.is_file():
                results_by_id[job_id] = load_json(result_path)
            if grade_path.is_file():
                grades_by_id[job_id] = load_json(grade_path)
            else:
                grades_by_id[job_id] = missing_video_grade(job_id, _manifest_job(manifest_row), policy)

    scans: list[dict[str, Any]] = []
    cross_scene_scans: list[dict[str, Any]] = []
    experiment_grades: list[dict[str, Any]] = []
    cross_scene_experiment_grades: list[dict[str, Any]] = []
    if args.stage in {"response", "all"}:
        scans = build_parameter_scan_grades(
            manifest_rows=manifest,
            results_by_id=results_by_id,
            grades_by_id=grades_by_id,
            registry=registry,
            policy=policy,
        )
        for ordinal, scan in enumerate(scans, 1):
            scan_output = output / "groups" / str(scan["scan_id"])
            artifacts: dict[str, Any] = {}
            if not args.no_visuals and (args.all_scan_visuals or scan["scan_id"] in requested_scans):
                artifacts["parameter_response"] = write_parameter_response_plot(scan_output / "parameter_response.png", scan)
                artifacts["trajectory_comparison"] = write_trajectory_comparison(scan_output / "trajectory_comparison.png", scan, results_by_id)
            scan["artifacts"] = artifacts
            write_json(scan_output / "group_grade.json", scan)
            evidence_rows.append({
                "entity_scope": "parameter_scan",
                "entity_id": scan["scan_id"],
                "grade": scan["response_grade"],
                "grade_json": str(scan_output / "group_grade.json"),
                "parameter_response_plot": (artifacts.get("parameter_response") or {}).get("png") if isinstance(artifacts.get("parameter_response"), dict) else None,
                "trajectory_comparison_plot": (artifacts.get("trajectory_comparison") or {}).get("png") if isinstance(artifacts.get("trajectory_comparison"), dict) else None,
            })
            print(f"[scan {ordinal}/{len(scans)}] {scan['scan_id']} -> {scan['response_grade']}", flush=True)
        experiment_grades = aggregate_experiment_scan_grades(scans, registry=registry)
        cross_scene_scans = build_cross_scene_parameter_scan_grades(scans, policy=policy)
        known_scan_ids = {str(scan["scan_id"]) for scan in scans + cross_scene_scans}
        unknown_scan_ids = sorted(requested_scans - known_scan_ids)
        if unknown_scan_ids:
            raise ValueError(f"unknown scan ids: {unknown_scan_ids}")
        for ordinal, scan in enumerate(cross_scene_scans, 1):
            scan_output = output / "groups_aggregate" / str(scan["scan_id"])
            artifacts: dict[str, Any] = {}
            if not args.no_visuals and (not requested_scans or scan["scan_id"] in requested_scans):
                artifacts["parameter_response"] = write_parameter_response_plot(scan_output / "parameter_response.png", scan)
                artifacts["trajectory_comparison"] = write_trajectory_comparison(scan_output / "trajectory_comparison.png", scan, results_by_id)
            scan["artifacts"] = artifacts
            write_json(scan_output / "group_grade.json", scan)
            evidence_rows.append({
                "entity_scope": "cross_scene_parameter_scan",
                "entity_id": scan["scan_id"],
                "grade": scan["response_grade"],
                "grade_json": str(scan_output / "group_grade.json"),
                "parameter_response_plot": (artifacts.get("parameter_response") or {}).get("png") if isinstance(artifacts.get("parameter_response"), dict) else None,
                "trajectory_comparison_plot": (artifacts.get("trajectory_comparison") or {}).get("png") if isinstance(artifacts.get("trajectory_comparison"), dict) else None,
            })
            print(f"[cross-scene {ordinal}/{len(cross_scene_scans)}] {scan['scan_id']} -> {scan['response_grade']}", flush=True)
        cross_scene_experiment_grades = aggregate_experiment_scan_grades(
            cross_scene_scans,
            registry=registry,
        )
        write_jsonl(output / "parameter_scan_grades.jsonl", scans)
        write_csv(output / "parameter_scan_grades.csv", [scan_grade_summary_row(scan) for scan in scans])
        write_jsonl(output / "experiment_response_grades.jsonl", experiment_grades)
        write_jsonl(output / "cross_scene_parameter_grades.jsonl", cross_scene_scans)
        write_csv(output / "cross_scene_parameter_grades.csv", [scan_grade_summary_row(scan) for scan in cross_scene_scans])
        write_jsonl(output / "cross_scene_experiment_response_grades.jsonl", cross_scene_experiment_grades)

    if not args.no_visuals and grades_by_id:
        write_grade_funnel(output / "figures" / "grade_funnel.png", list(grades_by_id.values()), cross_scene_scans or scans)
    evidence_index_path = output / "evidence_index.csv"
    evidence_rows = _merge_evidence_rows(
        _read_csv_rows(evidence_index_path),
        evidence_rows,
    )
    write_csv(evidence_index_path, evidence_rows)
    video_counts = Counter(str(value.get("video_stage")) for value in grades_by_id.values())
    scan_counts = Counter(str(value.get("response_grade")) for value in scans if value.get("canonical_for_target"))
    cross_scene_counts = Counter(str(value.get("response_grade")) for value in cross_scene_scans)
    summary = {
        "schema_version": "1.0.0",
        "policy_id": policy.get("policy_id"),
        "adjudication_record_count": len(adjudications),
        "video_entity_count": len(grades_by_id),
        "video_stage_counts": dict(video_counts),
        "parameter_scan_count": len(scans),
        "canonical_parameter_scan_count": sum(bool(scan.get("canonical_for_target")) for scan in scans),
        "canonical_response_grade_counts": dict(scan_counts),
        "experiment_response_entity_count": len(experiment_grades),
        "cross_scene_parameter_scan_count": len(cross_scene_scans),
        "cross_scene_response_grade_counts": dict(cross_scene_counts),
        "cross_scene_experiment_response_entity_count": len(cross_scene_experiment_grades),
        "scope_warning": "Do not combine per-video G0/G1 and per-scan G2/G3/G4 into one ordinal sample-level column.",
        "output": str(output),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
