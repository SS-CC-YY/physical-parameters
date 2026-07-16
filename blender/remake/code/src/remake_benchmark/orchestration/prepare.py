from __future__ import annotations

from pathlib import Path
from typing import Any

from remake_benchmark.core.build import resolve_build
from remake_benchmark.core.errors import ConfigError
from remake_benchmark.core.io import write_json, write_jsonl, write_yaml
from remake_benchmark.core.manifest import build_jobs


def prepare_run(
    build_path: Path,
    run_dir: Path,
    *,
    workspace_root_override: Path | None = None,
    max_jobs: int | None = None,
    check_inputs: bool = True,
    overwrite: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.resolve()
    resolved_path = run_dir / "resolved_build.yaml"
    manifest_path = run_dir / "manifest.jsonl"
    if not overwrite and (resolved_path.exists() or manifest_path.exists()):
        raise ConfigError(f"run is already prepared: {run_dir}; pass --overwrite to replace build and manifest")
    resolved = resolve_build(build_path, workspace_root_override)
    all_jobs = build_jobs(resolved, check_inputs=check_inputs)
    jobs = all_jobs if max_jobs is None else all_jobs[:max_jobs]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(resolved_path, resolved)
    write_jsonl(manifest_path, jobs)
    selected_scenes = list(dict.fromkeys(str(job["factors"]["scene_id"]) for job in all_jobs))
    manifest_scenes = list(dict.fromkeys(str(job["factors"]["scene_id"]) for job in jobs))
    write_json(
        run_dir / "manifest.selection.json",
        {
            "build_id": resolved["build_id"],
            "scene_selection_seed": jobs[0]["factors"].get("scene_selection_seed"),
            "selected_scenes": selected_scenes,
            "manifest_scenes": manifest_scenes,
            "target_assignment": jobs[0]["factors"].get("target_assignment"),
            "planned_job_count": len(all_jobs),
            "job_count": len(jobs),
        },
    )
    return resolved, jobs
