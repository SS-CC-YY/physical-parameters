#!/usr/bin/env python3
"""Export per-model PhysParamBench v1.3.1 media-review bundles.

This exporter is intentionally separate from ``export_v131_advisor_evidence``:
the advisor bundle contains a small fixed evidence subset, while this command
can package every primary-fit job or all 978 jobs for one or more models.

Only explicitly resolved evaluation artifacts are archived.  Model weights,
Hugging Face caches, source video directories, and tracking directories are
never traversed recursively.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from export_v131_advisor_evidence import (
    EvidenceCase,
    MODEL_DIRS,
    resolve_case,
)


EXPECTED_SCOPE_COUNTS = {
    "primary": 126,
    "all": 978,
}
MODEL_ALIASES = {
    **{name.lower(): name for name in MODEL_DIRS},
    **{slug.lower(): name for name, slug in MODEL_DIRS.items()},
    "wan": "Wan2.2",
    "wan22": "Wan2.2",
    "seedance": "Seedance2.0",
    "cosmos": "Cosmos-Predict2.5-14B",
    "longlive": "LongLive2.0",
}

TOP_LEVEL_EVALUATION_FILES = (
    "all_jobs.csv",
    "evaluation_metadata_all.json",
    "summary.json",
)

UNIFIED_REPORT_FILES = (
    "lineage_audit.json",
    "unified_run_summary.json",
    "report/paper_main_table.csv",
    "report/scan_response_channels.csv",
    "report/background_summary.csv",
    "report/view_summary.csv",
    "report/seed_stability.csv",
    "report/composition_summary.csv",
    "experiment_evidence/experiment_evidence_matrix.csv",
    "experiment_evidence/experiment_conclusion_summary.csv",
)


@dataclass
class ReviewJob:
    model: str
    job_id: str
    result: dict[str, Any]
    artifacts: dict[str, Path]
    missing: list[str]


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
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _resolve_models(values: Sequence[str]) -> list[str]:
    if not values:
        return list(MODEL_DIRS)
    models: list[str] = []
    for raw in values:
        for token in raw.split(","):
            key = token.strip().lower()
            if not key:
                continue
            model = MODEL_ALIASES.get(key)
            if model is None:
                choices = ", ".join(MODEL_DIRS.values())
                raise ValueError(f"unknown model {token!r}; use one of: {choices}")
            if model not in models:
                models.append(model)
    if not models:
        raise ValueError("--model did not contain a model name")
    return models


def _selected(result: Mapping[str, Any], scope: str) -> bool:
    if scope == "all":
        return True
    job = result.get("job") if isinstance(result.get("job"), Mapping) else {}
    return (
        str(job.get("scene_id")) == "baseline"
        and str(job.get("camera_name")) == "CAM_Side"
    )


def _add_distinct_validity_overlay(
    artifacts: dict[str, Path],
    result: Mapping[str, Any],
    result_path: Path,
) -> None:
    eval_job = result_path.parent
    source_extraction = _path(result.get("source_extraction"), eval_job)
    track_job = source_extraction.parent if source_extraction is not None else None
    validity_overlay = _first_file(
        (
            None if track_job is None else track_job / "validity_object_track_overlay.mp4",
            eval_job / "validity_object_track_overlay.mp4",
        )
    )
    if validity_overlay is None:
        return
    existing = artifacts.get("object_track_overlay.mp4")
    try:
        same = (
            existing is not None
            and existing.resolve(strict=True) == validity_overlay.resolve(strict=True)
        )
    except OSError:
        same = False
    if not same:
        artifacts["validity_object_track_overlay.mp4"] = validity_overlay


def collect_jobs(
    evaluation_root: Path,
    model: str,
    *,
    scope: str,
) -> list[ReviewJob]:
    model_slug = MODEL_DIRS[model]
    jobs_root = evaluation_root / "models" / model_slug / "evaluation" / "jobs"
    if not jobs_root.is_dir():
        raise RuntimeError(f"missing evaluation jobs directory: {jobs_root}")

    output: list[ReviewJob] = []
    for result_path in sorted(jobs_root.glob("*/result.json")):
        result = _read_json(result_path)
        if not _selected(result, scope):
            continue
        job_id = result_path.parent.name
        case = EvidenceCase(
            group=f"review_{scope}",
            model=model,
            job_id=job_id,
            note=f"{scope} media review export.",
        )
        resolved = resolve_case(evaluation_root, case)
        artifacts = dict(resolved.artifacts)
        _add_distinct_validity_overlay(artifacts, result, result_path)
        output.append(
            ReviewJob(
                model=model,
                job_id=job_id,
                result=result,
                artifacts=artifacts,
                missing=list(resolved.missing),
            )
        )
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


def _manifest_row(item: ReviewJob) -> dict[str, Any]:
    result = item.result
    job = result.get("job") if isinstance(result.get("job"), Mapping) else {}
    validity = (
        result.get("video_generation_validity")
        if isinstance(result.get("video_generation_validity"), Mapping)
        else {}
    )
    fit = result.get("fit") if isinstance(result.get("fit"), Mapping) else {}
    fit_validity = (
        fit.get("fit_validity")
        if isinstance(fit.get("fit_validity"), Mapping)
        else {}
    )
    dynamic = (
        result.get("dynamic_3d_inclusion")
        if isinstance(result.get("dynamic_3d_inclusion"), Mapping)
        else {}
    )
    return {
        "model": item.model,
        "job_id": item.job_id,
        "experiment_id": job.get("experiment_id"),
        "parameter_tuple_id": job.get("parameter_tuple_id"),
        "scene_id": job.get("scene_id"),
        "camera_name": job.get("camera_name"),
        "seed": job.get("seed"),
        "reconstruction_route": result.get("reconstruction_route"),
        "generation_validity_status": validity.get("status"),
        "generation_failure_codes": ";".join(
            str(value) for value in validity.get("failure_codes", [])
        ),
        "generation_warning_codes": ";".join(
            str(value) for value in validity.get("warning_codes", [])
        ),
        "fit_status": fit.get("status"),
        "fit_reason": fit.get("reason"),
        "fit_validity_status": fit_validity.get("status"),
        "fit_validity_category": fit_validity.get("category"),
        "dynamic_3d_decision": dynamic.get("decision"),
        "parameter_summary_json": json.dumps(
            _parameter_summary(result), ensure_ascii=False, sort_keys=True
        ),
        "original_included": "original.mp4" in item.artifacts,
        "overlay_included": "object_track_overlay.mp4" in item.artifacts,
        "validity_overlay_included": (
            "validity_object_track_overlay.mp4" in item.artifacts
        ),
        "trajectory_plot_included": "trajectory_plot.png" in item.artifacts,
        "trajectory_csv_included": "trajectory_frames.csv" in item.artifacts,
        "track_result_included": "track_result.json" in item.artifacts,
        "missing_required": ";".join(item.missing),
        "manual_validity_label": "",
        "manual_first_failure_time_s": "",
        "manual_notes": "",
    }


def _write_manifest(path: Path, jobs: Sequence[ReviewJob]) -> None:
    rows = [_manifest_row(item) for item in jobs]
    fieldnames = list(rows[0]) if rows else ["model", "job_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _real_file(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve artifact {path}: {exc}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"artifact is not a regular file: {resolved}")
    return resolved


def _total_bytes(jobs: Sequence[ReviewJob]) -> int:
    seen: set[str] = set()
    total = 0
    for item in jobs:
        for source in item.artifacts.values():
            try:
                real = _real_file(source)
            except RuntimeError:
                continue
            key = str(real)
            if key in seen:
                continue
            seen.add(key)
            total += real.stat().st_size
    return total


def _human_bytes(value: int) -> str:
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


README = """\
# PhysParamBench v1.3.1 per-model review bundle

This archive is intended for generation-validity and inverse-fit auditing.
Every job directory can contain:

- `original.mp4`: the generated rollout;
- `object_track_overlay.mp4`: the trajectory extraction overlay;
- `validity_object_track_overlay.mp4`: a separate validity overlay, when present;
- `trajectory_plot.png`: observed and fitted trajectory;
- `trajectory_frames.csv`: frame-level trajectory passed to the evaluator;
- `track_result.json`: extraction/reconstruction metadata;
- `result.json`: validity, fit, target, estimate, and metric record.

Review order:

1. inspect the original video for object duplication, disappearance, identity
   switching, non-rigid deformation, penetration, or a scene cut;
2. inspect the overlay to distinguish a generation failure from a detector or
   tracker failure;
3. inspect the trajectory plot and fit state only when the rollout is valid.

An obvious generation-validity failure must not be reported merely as a
physics-equation mismatch. Missing parameter estimates are not zero.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_file(
    archive: tarfile.TarFile,
    source: Path,
    arcname: str,
    added: set[str],
) -> None:
    if arcname in added:
        return
    real = _real_file(source)
    archive.add(real, arcname=arcname, recursive=False)
    added.add(arcname)


def create_archive(
    evaluation_root: Path,
    output: Path,
    model: str,
    jobs: Sequence[ReviewJob],
    *,
    scope: str,
    overwrite: bool,
) -> tuple[int, str]:
    if output.exists() and not overwrite:
        raise RuntimeError(f"output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()

    model_slug = MODEL_DIRS[model]
    archive_root = f"physparambench_v131_{model_slug}_{scope}_review"
    with tempfile.TemporaryDirectory(prefix="physparambench-model-review-") as text:
        temp = Path(text)
        readme = temp / "README.md"
        manifest = temp / "manifest.csv"
        readme.write_text(README, encoding="utf-8")
        _write_manifest(manifest, jobs)

        with tarfile.open(
            partial,
            "w:gz",
            compresslevel=1,
            dereference=True,
        ) as archive:
            added: set[str] = set()
            _add_file(archive, readme, f"{archive_root}/README.md", added)
            _add_file(archive, manifest, f"{archive_root}/manifest.csv", added)

            evaluation = evaluation_root / "models" / model_slug / "evaluation"
            for relative in TOP_LEVEL_EVALUATION_FILES:
                source = evaluation / relative
                if source.is_file():
                    _add_file(
                        archive,
                        source,
                        f"{archive_root}/aggregate/model/{relative}",
                        added,
                    )
            for relative in UNIFIED_REPORT_FILES:
                source = evaluation_root / relative
                if source.is_file():
                    _add_file(
                        archive,
                        source,
                        f"{archive_root}/aggregate/unified/{relative}",
                        added,
                    )

            for item in jobs:
                job_root = f"{archive_root}/jobs/{item.job_id}"
                for name, source in sorted(item.artifacts.items()):
                    _add_file(
                        archive,
                        source,
                        f"{job_root}/{name}",
                        added,
                    )

    with tarfile.open(partial, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        required = {
            f"{archive_root}/README.md",
            f"{archive_root}/manifest.csv",
        }
        if not required.issubset(names):
            raise RuntimeError(f"archive verification failed: {partial}")

    if output.exists():
        output.unlink()
    shutil.move(str(partial), str(output))
    digest = _sha256(output)
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return len(members), digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
        help="Unified five-model v1.3.1 output root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for separate per-model archives.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model display name or slug. Repeat the option or pass comma-separated "
            "names. Defaults to all five models."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("primary", "all"),
        default="primary",
        help=(
            "primary = baseline + CAM_Side; all = every evaluated job "
            "(normally 978 per model)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve media and report counts/bytes without creating archives.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected job lacks a required original/overlay/plot/CSV.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    if not evaluation_root.is_dir():
        raise SystemExit(f"error: evaluation root does not exist: {evaluation_root}")
    try:
        models = _resolve_models(args.model)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            evaluation_root.parent
            / f"physparambench_v131_model_review_{args.scope}_{stamp}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()

    summaries: list[dict[str, Any]] = []
    selections: dict[str, list[ReviewJob]] = {}
    for model in models:
        jobs = collect_jobs(evaluation_root, model, scope=args.scope)
        selections[model] = jobs
        summary = {
            "model": model,
            "model_slug": MODEL_DIRS[model],
            "scope": args.scope,
            "selected_jobs": len(jobs),
            "jobs_with_missing_required": sum(bool(item.missing) for item in jobs),
            "original_videos": sum("original.mp4" in item.artifacts for item in jobs),
            "overlay_videos": sum(
                "object_track_overlay.mp4" in item.artifacts for item in jobs
            ),
            "trajectory_plots": sum(
                "trajectory_plot.png" in item.artifacts for item in jobs
            ),
            "estimated_uncompressed_bytes": _total_bytes(jobs),
        }
        summary["expected_jobs"] = EXPECTED_SCOPE_COUNTS[args.scope]
        summary["inventory_complete"] = (
            len(jobs) == EXPECTED_SCOPE_COUNTS[args.scope]
        )
        summary["estimated_uncompressed_size"] = _human_bytes(
            int(summary["estimated_uncompressed_bytes"])
        )
        summaries.append(summary)

    print(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "evaluation_root": str(evaluation_root),
                "output_dir": None if args.dry_run else str(output_dir),
                "scope": args.scope,
                "models": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.strict:
        incomplete_inventory = [
            (
                f"{model}: selected {len(jobs)}, "
                f"expected {EXPECTED_SCOPE_COUNTS[args.scope]}"
            )
            for model, jobs in selections.items()
            if len(jobs) != EXPECTED_SCOPE_COUNTS[args.scope]
        ]
        incomplete_artifacts = [
            f"{item.model}/{item.job_id}: {item.missing}"
            for jobs in selections.values()
            for item in jobs
            if item.missing
        ]
        if incomplete_inventory or incomplete_artifacts:
            raise SystemExit(
                "error: strict review export validation failed; first entries:\n- "
                + "\n- ".join((incomplete_inventory + incomplete_artifacts)[:20])
            )
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        slug = MODEL_DIRS[model]
        output = output_dir / f"{slug}_{args.scope}_review.tar.gz"
        member_count, digest = create_archive(
            evaluation_root,
            output,
            model,
            selections[model],
            scope=args.scope,
            overwrite=bool(args.overwrite),
        )
        print(
            json.dumps(
                {
                    "model": model,
                    "archive": str(output),
                    "archive_bytes": output.stat().st_size,
                    "archive_size": _human_bytes(output.stat().st_size),
                    "archive_members": member_count,
                    "sha256": digest,
                    "sha256_file": str(output.with_name(output.name + ".sha256")),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
