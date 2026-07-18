from __future__ import annotations

import os
import shlex
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remake_benchmark.core.errors import BenchmarkError, ConfigError
from remake_benchmark.core.io import append_jsonl, read_json, read_jsonl, read_yaml, write_json
from remake_benchmark.models import get_adapter
from remake_benchmark.models.wan22 import Wan22Adapter

from .wan_session import PersistentWan22Session


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enforce_max_billable_jobs_per_run(model: dict[str, Any], job_count: int) -> None:
    safety = model.get("safety", {})
    if not isinstance(safety, dict) or safety.get("max_billable_jobs_per_run") is None:
        return
    try:
        maximum = int(safety["max_billable_jobs_per_run"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("model safety.max_billable_jobs_per_run must be a positive integer") from exc
    if maximum <= 0:
        raise ConfigError("model safety.max_billable_jobs_per_run must be a positive integer")
    if job_count > maximum:
        raise ConfigError(
            f"refusing to run {job_count} jobs: model safety.max_billable_jobs_per_run is {maximum}"
        )


def generate_run(
    run_dir: Path,
    *,
    dry_run: bool = False,
    max_jobs: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
    fail_fast: bool = False,
) -> dict[str, int]:
    run_dir = run_dir.resolve()
    resolved = read_yaml(run_dir / "resolved_build.yaml")
    all_jobs = read_jsonl(run_dir / "manifest.jsonl")
    _enforce_max_billable_jobs_per_run(resolved["model"], len(all_jobs))
    jobs = all_jobs[start_index:]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    if overwrite and resolved["model"].get("safety", {}).get("max_billable_jobs_per_run") is not None:
        raise ConfigError("--overwrite is disabled for billable API runs; use the saved task ID or a new run tag")
    workspace_root = Path(resolved["workspace_root"])
    adapter = get_adapter(resolved["model"], workspace_root)
    videos_dir = run_dir / "videos"
    metadata_dir = run_dir / "metadata"
    logs_dir = run_dir / "logs"
    for directory in (videos_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.jsonl"
    counts = {"ok": 0, "skip": 0, "error": 0, "dry_run": 0, "total": len(jobs)}
    execution_mode = str(resolved["model"].get("runtime", {}).get("execution_mode", "subprocess"))
    persistent = execution_mode == "persistent_worker"
    if persistent and not isinstance(adapter, Wan22Adapter):
        raise ConfigError("persistent_worker execution currently requires the Wan2.2 adapter")
    needs_worker = persistent and not dry_run and any(
        overwrite
        or not (videos_dir / f"{job['job_id']}.mp4").is_file()
        or (videos_dir / f"{job['job_id']}.mp4").stat().st_size == 0
        for job in jobs
    )
    session_context = (
        PersistentWan22Session(adapter, run_dir)
        if needs_worker
        else nullcontext(None)
    )

    with session_context as session:
        for offset, job in enumerate(jobs, start_index):
            job_id = job["job_id"]
            output_video = videos_dir / f"{job_id}.mp4"
            invocation = None
            command_path = logs_dir / f"{job_id}.command.json"
            stdout_path = logs_dir / f"{job_id}.stdout.log"
            stderr_path = logs_dir / f"{job_id}.stderr.log"
            try:
                adapter.validate(job, dry_run=dry_run)
                invocation = adapter.build_invocation(job, output_video)
                write_json(
                    command_path,
                    {
                        "job_id": job_id,
                        "execution_mode": execution_mode,
                        "command": invocation.command,
                        "cwd": str(invocation.cwd),
                        "environment": invocation.environment,
                    },
                )
                if dry_run:
                    print(f"[{offset + 1}] {shlex.join(invocation.command)}")
                    append_jsonl(state_path, {"timestamp": _timestamp(), "status": "dry_run", "job_id": job_id})
                    counts["dry_run"] += 1
                    continue
                output_ready = output_video.is_file() and output_video.stat().st_size > 0
                primary_metadata_ready = (metadata_dir / f"{job_id}.json").is_file()
                provider_metadata_ready = True
                if invocation.result_metadata is not None:
                    provider_metadata_ready = False
                    if invocation.result_metadata.is_file():
                        try:
                            provider_record = read_json(invocation.result_metadata)
                            provider_metadata_ready = bool(
                                isinstance(provider_record, dict)
                                and provider_record.get("status") == "succeeded"
                                and provider_record.get("task_id")
                            )
                        except (OSError, ValueError):
                            provider_metadata_ready = False
                if (
                    output_ready
                    and provider_metadata_ready
                    and (invocation.result_metadata is None or primary_metadata_ready)
                    and not overwrite
                ):
                    append_jsonl(
                        state_path,
                        {"timestamp": _timestamp(), "status": "skip", "job_id": job_id, "output_video": str(output_video)},
                    )
                    counts["skip"] += 1
                    continue

                append_jsonl(state_path, {"timestamp": _timestamp(), "status": "running", "job_id": job_id})
                if session is not None:
                    elapsed = session.generate(job, output_video, stdout_path)
                    actual_stderr_path = logs_dir / "persistent_worker.stderr.log"
                else:
                    environment = os.environ.copy()
                    environment.update(invocation.environment)
                    started = time.time()
                    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                        "w", encoding="utf-8"
                    ) as stderr_handle:
                        process = subprocess.run(
                            invocation.command,
                            cwd=invocation.cwd,
                            env=environment,
                            stdout=stdout_handle,
                            stderr=stderr_handle,
                            text=True,
                            check=False,
                        )
                    elapsed = time.time() - started
                    if process.returncode != 0:
                        raise BenchmarkError(
                            f"model process failed (returncode={process.returncode}); see {stderr_path}"
                        )
                    actual_stderr_path = stderr_path
                if not output_video.is_file() or output_video.stat().st_size == 0:
                    raise BenchmarkError(f"model produced no video; see {actual_stderr_path}")
                provider_metrics = None
                if invocation.result_metadata is not None and invocation.result_metadata.is_file():
                    provider_metrics = read_json(invocation.result_metadata)
                metadata = {
                    **job,
                    "model": adapter.provenance(),
                    "execution_mode": execution_mode,
                    "output_video": str(output_video),
                    "command_record": str(command_path),
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(actual_stderr_path),
                    "elapsed_seconds": elapsed,
                    "completed_at": _timestamp(),
                }
                if provider_metrics is not None:
                    metadata["provider_metrics"] = provider_metrics
                    metadata["provider_metrics_file"] = str(invocation.result_metadata)
                write_json(metadata_dir / f"{job_id}.json", metadata)
                state_record = {
                    "timestamp": _timestamp(),
                    "status": "ok",
                    "job_id": job_id,
                    "output_video": str(output_video),
                    "elapsed_seconds": elapsed,
                }
                if provider_metrics is not None:
                    state_record.update(
                        {
                            "provider": provider_metrics.get("provider"),
                            "provider_task_id": provider_metrics.get("task_id"),
                            "provider_cost": provider_metrics.get("cost"),
                        }
                    )
                append_jsonl(
                    state_path,
                    state_record,
                )
                counts["ok"] += 1
            except Exception as exc:
                counts["error"] += 1
                error_record = {"timestamp": _timestamp(), "status": "error", "job_id": job_id, "error": str(exc)}
                if invocation is not None and invocation.result_metadata is not None:
                    error_record["provider_metrics_file"] = str(invocation.result_metadata)
                append_jsonl(
                    state_path,
                    error_record,
                )
                print(f"[{offset + 1}] ERROR {job_id}: {exc}")
                if fail_fast:
                    raise

    write_json(run_dir / "run_summary.json", counts)
    if counts["error"]:
        raise BenchmarkError(f"{counts['error']} generation job(s) failed")
    return counts
