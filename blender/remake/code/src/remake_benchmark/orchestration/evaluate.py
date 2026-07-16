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
    if count <= 0:
        return set()
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_scene[str(job["factors"]["scene_id"])].append(job)
    representatives: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(by_scene):
        scene_jobs = by_scene[scene]
        representatives.append(scene_jobs[scene_index % len(scene_jobs)])
    rng = random.Random(seed)
    if count < len(representatives):
        representatives = rng.sample(representatives, count)
    elif count > len(representatives):
        chosen = {job["job_id"] for job in representatives}
        remaining = [job for job in jobs if job["job_id"] not in chosen]
        representatives.extend(rng.sample(remaining, min(count - len(representatives), len(remaining))))
    return {job["job_id"] for job in representatives[:count]}


def _error_row(job: dict[str, Any], evaluator_id: str, error: Exception | str, status: str = "error") -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "evaluator_id": evaluator_id,
        "evaluator_version": "1.0.0",
        "status": status,
        "quality_flags": ["missing_video" if status == "invalid" else "evaluator_error"],
        "metrics": {},
        "artifacts": {},
        "factors": dict(job["factors"]),
        "error": str(error),
    }


def evaluate_run(run_dir: Path) -> dict[str, int]:
    run_dir = run_dir.resolve()
    resolved = read_yaml(run_dir / "resolved_build.yaml")
    jobs = read_jsonl(run_dir / "manifest.jsonl")
    evaluator_configs = resolved["evaluation"]["evaluators"]
    basic_config = _one_config(evaluator_configs, "basic_video")
    freefall_config = _one_config(evaluator_configs, "v1a_freefall", required=False)
    all_results: list[dict[str, Any]] = []
    basic_rows: list[dict[str, Any]] = []

    for job in jobs:
        video_path = run_dir / "videos" / f"{job['job_id']}.mp4"
        try:
            result = evaluate_video(job, video_path, basic_config or {})
            row = {
                "job_id": job["job_id"],
                "evaluator_id": "basic_video",
                "evaluator_version": str((basic_config or {}).get("version", "1.0.0")),
                **result,
                "artifacts": {},
                "factors": dict(job["factors"]),
            }
        except Exception as exc:
            row = _error_row(job, "basic_video", exc)
        basic_rows.append(row)
        all_results.append(row)

    freefall_rows: list[dict[str, Any]] = []
    freefall_aggregate: dict[str, Any] | None = None
    if freefall_config is not None:
        from remake_benchmark.evaluators.freefall import (
            aggregate_freefall,
            evaluate_freefall_job,
            write_freefall_html_report,
            write_freefall_summary_csv,
        )

        freefall_jobs = [job for job in jobs if "v1a_freefall" in job.get("evaluation_tags", [])]
        overlay_ids = _overlay_job_ids(
            freefall_jobs,
            int(freefall_config.get("overlay_video_count", 5)),
            int(freefall_config.get("selection_seed", 36)),
        )
        freefall_root = run_dir / "eval" / "freefall"
        workspace_root = Path(resolved["workspace_root"])
        for job in freefall_jobs:
            video_path = run_dir / "videos" / f"{job['job_id']}.mp4"
            if not video_path.is_file():
                row = _error_row(job, "v1a_freefall", f"video not found: {video_path}", status="invalid")
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
                    row = _error_row(job, "v1a_freefall", exc)
            freefall_rows.append(row)
            all_results.append(row)
        freefall_aggregate = aggregate_freefall(freefall_rows)
        write_freefall_summary_csv(freefall_root / "summary.csv", freefall_rows)
        write_json(freefall_root / "aggregate.json", freefall_aggregate)
        write_freefall_html_report(run_dir / "eval" / "report" / "index.html", freefall_rows, freefall_aggregate)

    counts = {
        "ok": sum(row["status"] == "ok" for row in all_results),
        "invalid": sum(row["status"] == "invalid" for row in all_results),
        "error": sum(row["status"] == "error" for row in all_results),
        "total": len(all_results),
    }
    eval_dir = run_dir / "eval"
    write_jsonl(eval_dir / "sample_metrics.jsonl", all_results)
    write_json(
        eval_dir / "aggregate.json",
        {
            **counts,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
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
    return counts
