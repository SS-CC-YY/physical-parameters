#!/usr/bin/env python3
"""Package a small, auditable PhysParamBench v1.3.1 evidence bundle.

The default selection contains the 18 rollout cases used by the advisor report:

* the same two V2-E parameter settings across all five models;
* matched background, viewpoint, and seed comparisons; and
* one accepted SpatialTrackerV2 reconstruction.

Paths to videos and tracking artifacts are read from each evaluation result, so
the script does not require five separate video-root arguments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_DIRS = {
    "Wan2.2": "wan2_2",
    "Seedance2.0": "seedance2_0",
    "Cosmos-Predict2.5-14B": "cosmos_predict2_5_14b",
    "Helios": "helios",
    "LongLive2.0": "longlive2_0",
}

ALL_MODELS = tuple(MODEL_DIRS)


@dataclass(frozen=True)
class EvidenceCase:
    group: str
    model: str
    job_id: str
    note: str


@dataclass
class ResolvedCase:
    case: EvidenceCase
    result: dict[str, Any]
    artifacts: dict[str, Path]
    missing: list[str]


def core_cases() -> list[EvidenceCase]:
    cases: list[EvidenceCase] = []

    for model in ALL_MODELS:
        cases.append(
            EvidenceCase(
                group="01_five_model_e080",
                model=model,
                job_id="v2_E__g9p80_e0p80__baseline__standard_ball__CAM_Side__seed-341867882",
                note="Same task across all five models; target g=9.8, e=0.80.",
            )
        )
        cases.append(
            EvidenceCase(
                group="02_five_model_e058",
                model=model,
                job_id="v2_E__g9p80_e0p58__baseline__standard_ball__CAM_Side__seed-341867882",
                note="Low-restitution endpoint paired with group 01; target g=9.8, e=0.58.",
            )
        )

    cases.extend(
        [
            EvidenceCase(
                group="03_background",
                model="Cosmos-Predict2.5-14B",
                job_id="v1_A__g14p70__baseline__standard_ball__CAM_Side__seed-341867882",
                note="Matched background pair: baseline.",
            ),
            EvidenceCase(
                group="03_background",
                model="Cosmos-Predict2.5-14B",
                job_id="v1_A__g14p70__indoor3__standard_ball__CAM_Side__seed-341867882",
                note="Matched background pair: indoor3.",
            ),
            EvidenceCase(
                group="04_viewpoint",
                model="Seedance2.0",
                job_id="v2_A__g9p81__outdoor4__standard_ball__CAM_Side__seed-341867882",
                note="Matched viewpoint pair: side.",
            ),
            EvidenceCase(
                group="04_viewpoint",
                model="Seedance2.0",
                job_id="v2_A__g9p81__outdoor4__standard_ball__CAM_Main__seed-341867882",
                note="Matched viewpoint pair: main.",
            ),
        ]
    )

    for seed in (135883006, 265635392, 341867882):
        cases.append(
            EvidenceCase(
                group="05_seed",
                model="Seedance2.0",
                job_id=f"v3_B__eL_low__baseline__standard_ball__CAM_Side__seed-{seed}",
                note="Matched seed comparison for the same V3-B physical specification.",
            )
        )

    cases.append(
        EvidenceCase(
            group="06_dynamic_3d_accepted",
            model="LongLive2.0",
            job_id="v2_A__g4p90__indoor4__standard_ball__CAM_Top__seed-341867882",
            note="SpatialTrackerV2 route accepted by the dynamic-3D gate; target g=4.9.",
        )
    )
    return cases


def rejected_3d_case() -> EvidenceCase:
    return EvidenceCase(
        group="07_dynamic_3d_rejected",
        model="LongLive2.0",
        job_id="v1_A__g2p00__baseline__standard_ball__CAM_Top__seed-341867882",
        note="Dynamic reconstruction rejected by its evidence-quality gate; this is not a physics-model failure.",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _path(raw: Any, relative_to: Path | None = None) -> Path | None:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
        return None
    candidate = Path(str(raw))
    if not candidate.is_absolute() and relative_to is not None:
        candidate = relative_to / candidate
    return candidate


def _first_file(candidates: Iterable[Path | None]) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _result_path(evaluation_root: Path, case: EvidenceCase) -> Path:
    return (
        evaluation_root
        / "models"
        / MODEL_DIRS[case.model]
        / "evaluation"
        / "jobs"
        / case.job_id
        / "result.json"
    )


def _append_existing(
    artifacts: dict[str, Path],
    name: str,
    candidates: Iterable[Path | None],
) -> None:
    found = _first_file(candidates)
    if found is not None:
        artifacts[name] = found


def resolve_case(evaluation_root: Path, case: EvidenceCase) -> ResolvedCase:
    result_path = _result_path(evaluation_root, case)
    if not result_path.is_file():
        return ResolvedCase(
            case=case,
            result={},
            artifacts={},
            missing=[f"result_json:{result_path}"],
        )

    result = _read_json(result_path)
    eval_job = result_path.parent
    job = result.get("job") if isinstance(result.get("job"), Mapping) else {}
    visuals = (
        result.get("visual_evidence")
        if isinstance(result.get("visual_evidence"), Mapping)
        else {}
    )

    source_extraction = _path(result.get("source_extraction"), eval_job)
    source_trajectory = _path(result.get("source_trajectory"), eval_job)
    track_job = source_extraction.parent if source_extraction is not None else None
    track_result: dict[str, Any] = {}
    if source_extraction is not None and source_extraction.is_file():
        try:
            track_result = _read_json(source_extraction)
        except RuntimeError:
            track_result = {}

    artifacts: dict[str, Path] = {"result.json": result_path}
    _append_existing(
        artifacts,
        "original.mp4",
        [
            _path(job.get("video_path"), eval_job),
            _path(track_result.get("source_video"), track_job),
        ],
    )
    _append_existing(
        artifacts,
        "object_track_overlay.mp4",
        [
            _path(track_result.get("object_track_overlay"), track_job),
            _path(track_result.get("native_object_track_overlay"), track_job),
            _path(visuals.get("overlay_video"), eval_job),
            None if track_job is None else track_job / "object_track_overlay.mp4",
            None if track_job is None else track_job / "validity_object_track_overlay.mp4",
            eval_job / "object_track_overlay.mp4",
            eval_job / "validity_object_track_overlay.mp4",
        ],
    )
    _append_existing(
        artifacts,
        "trajectory_plot.png",
        [
            _path(visuals.get("trajectory_plot"), eval_job),
            eval_job / "trajectory_plot.png",
        ],
    )
    _append_existing(
        artifacts,
        "trajectory_frames.csv",
        [
            _path(result.get("trajectory_csv"), eval_job),
            source_trajectory,
            _path(track_result.get("trajectory_frames_csv"), track_job),
            None if track_job is None else track_job / "trajectory_frames.csv",
            eval_job / "trajectory_frames.csv",
        ],
    )
    _append_existing(
        artifacts,
        "track_result.json",
        [source_extraction],
    )

    # These are useful for the accepted/rejected SpatialTrackerV2 examples.
    _append_existing(
        artifacts,
        "native_trajectory.csv",
        [
            _path(track_result.get("native_trajectory_csv"), track_job),
        ],
    )
    _append_existing(
        artifacts,
        "native_tracking.csv",
        [
            _path(track_result.get("native_tracking_csv"), track_job),
        ],
    )
    _append_existing(
        artifacts,
        "trajectory_3d.png",
        [
            None if track_job is None else track_job / "trajectory_3d.png",
            None if track_job is None else track_job / "trajectory_world.png",
            eval_job / "trajectory_3d.png",
        ],
    )

    required = (
        "result.json",
        "original.mp4",
        "object_track_overlay.mp4",
        "trajectory_plot.png",
        "trajectory_frames.csv",
    )
    missing = [name for name in required if name not in artifacts]
    return ResolvedCase(case=case, result=result, artifacts=artifacts, missing=missing)


def appendix_cases(evaluation_root: Path) -> list[EvidenceCase]:
    path = evaluation_root / "report" / "case_selection.csv"
    if not path.is_file():
        raise RuntimeError(f"--include-appendix requires {path}")
    cases: list[EvidenceCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = str(row.get("model") or "")
            job_id = str(row.get("job_id") or "")
            case_type = str(row.get("case_type") or "case")
            if model not in MODEL_DIRS or not job_id:
                continue
            cases.append(
                EvidenceCase(
                    group=f"appendix_{case_type}",
                    model=model,
                    job_id=job_id,
                    note=(
                        "Automatically selected audit case. A parameter-close candidate "
                        "may still have fit_status=model_mismatch."
                    ),
                )
            )
    return cases


def deduplicate_cases(cases: Sequence[EvidenceCase]) -> list[EvidenceCase]:
    output: list[EvidenceCase] = []
    seen: set[tuple[str, str, str]] = set()
    for case in cases:
        key = (case.group, case.model, case.job_id)
        if key not in seen:
            seen.add(key)
            output.append(case)
    return output


def _parameter_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    parameters = metrics.get("parameters") if isinstance(metrics, Mapping) else None
    if not isinstance(parameters, Mapping):
        return {}
    output: dict[str, Any] = {}
    for name, payload in parameters.items():
        if not isinstance(payload, Mapping):
            continue
        output[str(name)] = {
            key: payload.get(key)
            for key in (
                "target",
                "estimate",
                "estimate_raw",
                "absolute_error",
                "benchmark_span_normalized_absolute_error",
                "target_range_status",
            )
        }
    return output


def _manifest_row(resolved: ResolvedCase) -> dict[str, Any]:
    result = resolved.result
    fit = result.get("fit") if isinstance(result.get("fit"), Mapping) else {}
    dynamic = (
        result.get("dynamic_3d_inclusion")
        if isinstance(result.get("dynamic_3d_inclusion"), Mapping)
        else {}
    )
    return {
        "group": resolved.case.group,
        "model": resolved.case.model,
        "job_id": resolved.case.job_id,
        "note": resolved.case.note,
        "reconstruction_route": result.get("reconstruction_route"),
        "fit_status": fit.get("status"),
        "fit_reason": fit.get("reason"),
        "dynamic_3d_decision": dynamic.get("decision"),
        "parameter_summary_json": json.dumps(
            _parameter_summary(result), ensure_ascii=False, sort_keys=True
        ),
        "original_included": "original.mp4" in resolved.artifacts,
        "overlay_included": "object_track_overlay.mp4" in resolved.artifacts,
        "trajectory_plot_included": "trajectory_plot.png" in resolved.artifacts,
        "trajectory_csv_included": "trajectory_frames.csv" in resolved.artifacts,
        "track_result_included": "track_result.json" in resolved.artifacts,
        "missing_required": ";".join(resolved.missing),
    }


REPORT_FILES = (
    "lineage_audit.json",
    "unified_run_summary.json",
    "report/paper_main_table.csv",
    "report/scan_response_channels.csv",
    "report/background_summary.csv",
    "report/view_summary.csv",
    "report/seed_stability.csv",
    "report/composition_summary.csv",
    "report/case_selection.csv",
    "experiment_evidence/experiment_evidence_matrix.csv",
    "experiment_evidence/experiment_evidence_matrix.svg",
    "experiment_evidence/experiment_conclusion_summary.csv",
    "experiment_evidence/model_evidence_summary.csv",
    "experiment_evidence/EXPERIMENT_CONCLUSIONS_ZH.md",
)


README = """\
# PhysParamBench v1.3.1 advisor evidence bundle

This archive is a fixed-rule evidence subset, not a hand-picked leaderboard.

Each case directory can contain:

- `original.mp4`: generated rollout;
- `object_track_overlay.mp4`: object detection and trajectory overlay;
- `trajectory_plot.png`: observed/fitted trajectory visualization;
- `trajectory_frames.csv`: per-frame trajectory used by the evaluator;
- `track_result.json`: extraction/reconstruction metadata;
- `result.json`: physics fit, target, estimate, and metric record.

The default 18 cases cover:

1. the same V2-E task across five models at `e=0.80`;
2. the same task across five models at `e=0.58`;
3. matched background, viewpoint, and random-seed comparisons; and
4. an accepted SpatialTrackerV2 dynamic-3D reconstruction.

Important interpretation rules:

- one rollout illustrates an aggregate result; it does not define a model rank;
- `model_mismatch` is not a successful full-rule fit even when one parameter is close;
- a rejected dynamic reconstruction is evaluator-insufficient evidence, not a
  video-model physics failure;
- missing estimates are not zero-valued estimates.

See `manifest.csv` for the fit state and parameter summary of every case.
"""


def _write_manifest(path: Path, resolved: Sequence[ResolvedCase]) -> None:
    rows = [_manifest_row(item) for item in resolved]
    fieldnames = list(rows[0]) if rows else ["group", "model", "job_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(
    evaluation_root: Path,
    output: Path | None,
    resolved: Sequence[ResolvedCase],
) -> dict[str, Any]:
    summary = {
        "schema_version": "1.0.0",
        "evaluation_root": str(evaluation_root),
        "output": None if output is None else str(output),
        "case_count": len(resolved),
        "group_counts": {},
        "model_counts": {},
        "required_artifacts_complete": sum(not item.missing for item in resolved),
        "missing_required_count": sum(bool(item.missing) for item in resolved),
        "trajectory_plot_count": sum(
            "trajectory_plot.png" in item.artifacts for item in resolved
        ),
        "trajectory_csv_count": sum(
            "trajectory_frames.csv" in item.artifacts for item in resolved
        ),
        "missing": [
            {
                "group": item.case.group,
                "model": item.case.model,
                "job_id": item.case.job_id,
                "items": item.missing,
            }
            for item in resolved
            if item.missing
        ],
    }
    for item in resolved:
        group_counts = summary["group_counts"]
        model_counts = summary["model_counts"]
        group_counts[item.case.group] = group_counts.get(item.case.group, 0) + 1
        model_counts[item.case.model] = model_counts.get(item.case.model, 0) + 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def create_archive(
    evaluation_root: Path,
    output: Path,
    resolved: Sequence[ResolvedCase],
    *,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise RuntimeError(f"output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()

    archive_root = "physparambench_v131_advisor_evidence"
    with tempfile.TemporaryDirectory(prefix="physparambench-evidence-") as temp_text:
        temp = Path(temp_text)
        readme = temp / "README.md"
        manifest = temp / "manifest.csv"
        readme.write_text(README, encoding="utf-8")
        _write_manifest(manifest, resolved)

        with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
            archive.add(readme, arcname=f"{archive_root}/README.md", recursive=False)
            archive.add(
                manifest, arcname=f"{archive_root}/manifest.csv", recursive=False
            )

            for relative in REPORT_FILES:
                source = evaluation_root / relative
                if source.is_file():
                    archive.add(
                        source,
                        arcname=f"{archive_root}/aggregate/{relative}",
                        recursive=False,
                    )

            added: set[str] = set()
            for item in resolved:
                model_dir = MODEL_DIRS[item.case.model]
                case_root = (
                    f"{archive_root}/cases/{item.case.group}/"
                    f"{model_dir}/{item.case.job_id}"
                )
                for name, source in item.artifacts.items():
                    arcname = f"{case_root}/{name}"
                    if arcname in added:
                        continue
                    added.add(arcname)
                    archive.add(source, arcname=arcname, recursive=False)

        if output.exists():
            output.unlink()
        shutil.move(str(partial), str(output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the fixed PhysParamBench v1.3.1 advisor evidence bundle."
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
        help="five_models_unified_v131 directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .tar.gz path. A timestamped path is used when omitted.",
    )
    parser.add_argument(
        "--include-3d-rejected",
        action="store_true",
        help="Add one dynamic-3D quality-gate rejection as evaluator-insufficient evidence.",
    )
    parser.add_argument(
        "--include-appendix",
        action="store_true",
        help="Also package the cases listed in report/case_selection.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate paths without creating an archive.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Create the archive even when a result, original video, overlay, "
            "trajectory plot, or trajectory CSV is missing."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    if not evaluation_root.is_dir():
        raise SystemExit(f"error: evaluation root does not exist: {evaluation_root}")

    cases = core_cases()
    if args.include_3d_rejected:
        cases.append(rejected_3d_case())
    if args.include_appendix:
        cases.extend(appendix_cases(evaluation_root))
    cases = deduplicate_cases(cases)

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = (
            evaluation_root.parent
            / f"physparambench_v131_advisor_evidence_{stamp}.tar.gz"
        )
    else:
        output = args.output.expanduser().resolve()

    resolved = [resolve_case(evaluation_root, case) for case in cases]
    summary = _print_summary(
        evaluation_root,
        None if args.dry_run else output,
        resolved,
    )
    if args.dry_run:
        return
    if summary["missing_required_count"] and not args.allow_missing:
        raise SystemExit(
            "error: required artifacts are missing; rerun with --dry-run to inspect "
            "the list, or use --allow-missing to package the available evidence"
        )

    create_archive(
        evaluation_root,
        output,
        resolved,
        overwrite=bool(args.overwrite),
    )
    print(f"archive: {output}")
    print(f"size_bytes: {output.stat().st_size}")


if __name__ == "__main__":
    main()
