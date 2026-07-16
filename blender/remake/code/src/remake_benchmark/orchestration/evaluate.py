from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remake_benchmark.core.errors import ConfigError
from remake_benchmark.core.io import read_jsonl, read_yaml, write_json, write_jsonl
from remake_benchmark.evaluators import evaluate_video


def _one_config(configs: list[dict[str, Any]], evaluator_id: str, *, required: bool = True) -> dict[str, Any] | None:
    matches = [item for item in configs if item.get("id") == evaluator_id]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ConfigError(f"evaluation profile requires exactly one '{evaluator_id}' evaluator")
    return matches[0]


def _overlay_job_ids(jobs: list[dict[str, Any]], count: int, seed: int) -> set[str]:
    """Choose overlay videos across both scene and camera instead of side view only."""
    if count <= 0:
        return set()
    by_view_scene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        key = (str(job["factors"].get("camera")), str(job["factors"].get("scene_id")))
        by_view_scene[key].append(job)
    cameras = sorted({key[0] for key in by_view_scene})
    if not cameras:
        return set()
    rng = random.Random(seed)
    base, extra = divmod(count, len(cameras))
    selected: list[dict[str, Any]] = []
    for camera_index, camera in enumerate(cameras):
        target = base + (1 if camera_index < extra else 0)
        camera_groups = [key for key in sorted(by_view_scene) if key[0] == camera]
        representatives = [
            by_view_scene[key][group_index % len(by_view_scene[key])]
            for group_index, key in enumerate(camera_groups)
        ]
        if target < len(representatives):
            representatives = rng.sample(representatives, target)
        elif target > len(representatives):
            chosen = {job["job_id"] for job in representatives}
            remaining = [
                job
                for job in jobs
                if str(job["factors"].get("camera")) == camera and job["job_id"] not in chosen
            ]
            representatives.extend(rng.sample(remaining, min(target - len(representatives), len(remaining))))
        selected.extend(representatives[:target])
    return {job["job_id"] for job in selected[:count]}


def _error_row(
    job: dict[str, Any],
    evaluator_id: str,
    error: Exception | str,
    *,
    version: str,
    status: str = "error",
) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "evaluator_id": evaluator_id,
        "evaluator_version": version,
        "status": status,
        "quality_flags": ["missing_video" if status == "invalid" else "evaluator_error"],
        "metrics": {},
        "artifacts": {},
        "factors": dict(job["factors"]),
        "error": str(error),
    }


def _ordered_rows(
    jobs: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for job in jobs:
        for evaluator_id in ("basic_video", "v1a_freefall"):
            row = rows_by_key.get((job["job_id"], evaluator_id))
            if row is not None:
                output.append(row)
    return output


def _evaluate_selected(run_dir: Path, selected_job_ids: set[str] | None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    resolved = read_yaml(run_dir / "resolved_build.yaml")
    jobs = read_jsonl(run_dir / "manifest.jsonl")
    selected_jobs = jobs if selected_job_ids is None else [job for job in jobs if job["job_id"] in selected_job_ids]
    if selected_job_ids is not None and len(selected_jobs) != len(selected_job_ids):
        known = {job["job_id"] for job in jobs}
        missing = sorted(selected_job_ids - known)
        raise ConfigError(f"job id(s) not found in manifest: {missing}")

    evaluator_configs = resolved["evaluation"]["evaluators"]
    basic_config = _one_config(evaluator_configs, "basic_video") or {}
    freefall_config = _one_config(evaluator_configs, "v1a_freefall", required=False)
    eval_dir = run_dir / "eval"
    sample_path = eval_dir / "sample_metrics.jsonl"
    existing = read_jsonl(sample_path) if selected_job_ids is not None and sample_path.is_file() else []
    rows_by_key = {(row["job_id"], row["evaluator_id"]): row for row in existing}

    for job in selected_jobs:
        video_path = run_dir / "videos" / f"{job['job_id']}.mp4"
        try:
            result = evaluate_video(job, video_path, basic_config)
            row = {
                "job_id": job["job_id"],
                "evaluator_id": "basic_video",
                "evaluator_version": str(basic_config.get("version", "1.0.0")),
                **result,
                "artifacts": {},
                "factors": dict(job["factors"]),
            }
        except Exception as exc:
            row = _error_row(
                job,
                "basic_video",
                exc,
                version=str(basic_config.get("version", "1.0.0")),
            )
        rows_by_key[(job["job_id"], "basic_video")] = row

    if freefall_config is not None:
        from remake_benchmark.evaluators.freefall import evaluate_freefall_job

        all_freefall_jobs = [job for job in jobs if "v1a_freefall" in job.get("evaluation_tags", [])]
        overlay_ids = _overlay_job_ids(
            all_freefall_jobs,
            int(freefall_config.get("overlay_video_count", 9)),
            int(freefall_config.get("selection_seed", 36)),
        )
        workspace_root = Path(resolved["workspace_root"])
        freefall_root = eval_dir / "freefall"
        for job in selected_jobs:
            if "v1a_freefall" not in job.get("evaluation_tags", []):
                continue
            video_path = run_dir / "videos" / f"{job['job_id']}.mp4"
            version = str(freefall_config.get("version", "2.0.0"))
            if not video_path.is_file():
                row = _error_row(job, "v1a_freefall", f"video not found: {video_path}", version=version, status="invalid")
            else:
                try:
                    row = evaluate_freefall_job(
                        job,
                        video_path,
                        workspace_root,
                        freefall_root,
                        freefall_config,
                        make_overlay=job["job_id"] in overlay_ids,
                    )
                except Exception as exc:
                    row = _error_row(job, "v1a_freefall", exc, version=version)
            rows_by_key[(job["job_id"], "v1a_freefall")] = row

    all_results = _ordered_rows(jobs, rows_by_key)
    basic_rows = [row for row in all_results if row["evaluator_id"] == "basic_video"]
    freefall_rows = [row for row in all_results if row["evaluator_id"] == "v1a_freefall"]
    freefall_aggregate: dict[str, Any] | None = None
    if freefall_config is not None:
        from remake_benchmark.evaluators.freefall import (
            aggregate_freefall,
            write_freefall_html_report,
            write_freefall_summary_csv,
        )

        freefall_root = eval_dir / "freefall"
        freefall_aggregate = aggregate_freefall(freefall_rows)
        write_freefall_summary_csv(freefall_root / "summary.csv", freefall_rows)
        write_json(freefall_root / "aggregate.json", freefall_aggregate)
        write_freefall_html_report(eval_dir / "report" / "index.html", freefall_rows, freefall_aggregate)

    counts = {
        "ok": sum(row["status"] == "ok" for row in all_results),
        "invalid": sum(row["status"] == "invalid" for row in all_results),
        "error": sum(row["status"] == "error" for row in all_results),
        "total": len(all_results),
    }
    selected_rows = [row for row in all_results if row["job_id"] in {job["job_id"] for job in selected_jobs}]
    selected_counts = {
        "ok": sum(row["status"] == "ok" for row in selected_rows),
        "invalid": sum(row["status"] == "invalid" for row in selected_rows),
        "error": sum(row["status"] == "error" for row in selected_rows),
        "total": len(selected_rows),
    }
    write_jsonl(sample_path, all_results)
    write_json(
        eval_dir / "aggregate.json",
        {
            **counts,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "completed_manifest_jobs": len({row["job_id"] for row in all_results}),
            "planned_manifest_jobs": len(jobs),
            "per_evaluator": {
                "basic_video": {
                    "ok": sum(row["status"] == "ok" for row in basic_rows),
                    "invalid": sum(row["status"] == "invalid" for row in basic_rows),
                    "error": sum(row["status"] == "error" for row in basic_rows),
                },
                "v1a_freefall": freefall_aggregate,
            },
        },
    )
    return {**counts, "selected": selected_counts}


def evaluate_job(run_dir: Path, job_id: str) -> dict[str, Any]:
    """Evaluate exactly one generated job and upsert it into cumulative reports."""
    return _evaluate_selected(run_dir, {job_id})


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    return _evaluate_selected(run_dir, None)
