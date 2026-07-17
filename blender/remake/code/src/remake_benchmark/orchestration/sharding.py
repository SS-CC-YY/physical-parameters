from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from remake_benchmark.core.errors import ConfigError
from remake_benchmark.core.io import read_jsonl, write_json


def split_job_ranges(total_jobs: int, gpu_ids: Sequence[str | int]) -> list[dict[str, int | str]]:
    if total_jobs <= 0:
        raise ConfigError("total_jobs must be positive")
    normalized = [str(gpu_id) for gpu_id in gpu_ids]
    if not normalized:
        raise ConfigError("at least one GPU id is required")
    if len(set(normalized)) != len(normalized):
        raise ConfigError("GPU ids must be unique")
    if total_jobs < len(normalized):
        raise ConfigError("there must be at least one job per GPU")

    base, remainder = divmod(total_jobs, len(normalized))
    start_index = 0
    ranges: list[dict[str, int | str]] = []
    for position, gpu_id in enumerate(normalized):
        max_jobs = base + (1 if position < remainder else 0)
        ranges.append(
            {
                "gpu_id": gpu_id,
                "start_index": start_index,
                "max_jobs": max_jobs,
            }
        )
        start_index += max_jobs
    return ranges


def _unique_sources(shard_dirs: list[Path], relative_directory: str, suffix: str) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for shard_dir in shard_dirs:
        source_dir = shard_dir / relative_directory
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.glob(f"*{suffix}")):
            job_id = source.name[: -len(suffix)]
            if job_id in sources:
                raise ConfigError(
                    f"duplicate {relative_directory} output for {job_id}: {sources[job_id]} and {source}"
                )
            sources[job_id] = source.resolve()
    return sources


def _hardlink_without_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise ConfigError(f"refusing to overwrite an existing collected result: {destination}")
    try:
        os.link(source, destination)
    except OSError as exc:
        raise ConfigError(
            f"could not hard-link {source} to {destination}; keep the run root and shards on one filesystem"
        ) from exc


def _copy_canonical_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ConfigError(f"missing controller file: {source}")
    if destination.exists():
        if source.read_bytes() == destination.read_bytes():
            return
        raise ConfigError(f"refusing to overwrite a different canonical file: {destination}")
    shutil.copy2(source, destination)


def collect_sharded_run(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    control_dir = run_root / "control"
    manifest = read_jsonl(control_dir / "manifest.jsonl")
    expected_ids = [str(job["job_id"]) for job in manifest]
    if len(expected_ids) != len(set(expected_ids)):
        raise ConfigError("controller manifest contains duplicate job ids")
    expected = set(expected_ids)

    shard_dirs = sorted(path for path in (run_root / "shards").glob("gpu-*") if path.is_dir())
    if not shard_dirs:
        raise ConfigError(f"no GPU shard directories found under {run_root / 'shards'}")

    videos = _unique_sources(shard_dirs, "videos", ".mp4")
    metadata = _unique_sources(shard_dirs, "metadata", ".json")
    empty_videos = sorted(job_id for job_id, path in videos.items() if path.stat().st_size == 0)
    unexpected_videos = sorted(set(videos) - expected)
    unexpected_metadata = sorted(set(metadata) - expected)
    missing_videos = sorted(expected - set(videos))
    missing_metadata = sorted(expected - set(metadata))
    if empty_videos or unexpected_videos or unexpected_metadata or missing_videos or missing_metadata:
        details = {
            "empty_videos": empty_videos[:10],
            "unexpected_videos": unexpected_videos[:10],
            "unexpected_metadata": unexpected_metadata[:10],
            "missing_videos": missing_videos[:10],
            "missing_metadata": missing_metadata[:10],
        }
        raise ConfigError(f"sharded run is incomplete or inconsistent: {details}")

    _copy_canonical_file(control_dir / "manifest.jsonl", run_root / "manifest.jsonl")
    _copy_canonical_file(control_dir / "resolved_build.yaml", run_root / "resolved_build.yaml")
    selection_path = control_dir / "manifest.selection.json"
    if selection_path.is_file():
        _copy_canonical_file(selection_path, run_root / "manifest.selection.json")

    for job_id in expected_ids:
        _hardlink_without_overwrite(videos[job_id], run_root / "videos" / f"{job_id}.mp4")
        _hardlink_without_overwrite(metadata[job_id], run_root / "metadata" / f"{job_id}.json")

    gpu_ids = [shard_dir.name.removeprefix("gpu-") for shard_dir in shard_dirs]
    summary: dict[str, Any] = {
        "sharded": True,
        "gpu_ids": gpu_ids,
        "shards": len(shard_dirs),
        "ok": len(expected_ids),
        "skip": 0,
        "error": 0,
        "dry_run": 0,
        "total": len(expected_ids),
        "videos_dir": str(run_root / "videos"),
        "metadata_dir": str(run_root / "metadata"),
    }
    write_json(run_root / "run_summary.json", summary)
    return summary
