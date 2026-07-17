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
from remake_benchmark.core.io import append_jsonl, read_jsonl, read_yaml, write_json
from remake_benchmark.models import get_adapter
from remake_benchmark.models.wan22 import Wan22Adapter

from .evaluate import evaluate_job
from .wan_session import PersistentWan22Session


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_subprocess(invocation: Any, stdout_path: Path, stderr_path: Path) -> float:
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
    if process.returncode != 0:
        raise BenchmarkError(f"model process exited with {process.returncode}; see {stderr_path}")
    return time.time() - started


def run_sequential(
    run_dir: Path,
    *,
    dry_run: bool = False,
    max_jobs: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
    fail_fast: bool = False,
    stop_on_invalid: bool = False,
) -> dict[str, Any]:
    """Generate one job, evaluate/upsert it, then and only then continue."""
    run_dir = run_dir.resolve()
    resolved = read_yaml(run_dir / "resolved_build.yaml")
    if str(resolved.get("workflow", "generation_only")) == "generation_only":
        raise ConfigError(
            "this build is generation_only; use 'generate' now and run evaluation explicitly later"
        )
    all_jobs = read_jsonl(run_dir / "manifest.jsonl")
    jobs = all_jobs[start_index:]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    workspace_root = Path(resolved["workspace_root"])
    adapter = get_adapter(resolved["model"], workspace_root)
    videos_dir = run_dir / "videos"
    metadata_dir = run_dir / "metadata"
    logs_dir = run_dir / "logs"
    for directory in (videos_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.jsonl"
    counts: dict[str, Any] = {
        "planned": len(jobs),
        "generated": 0,
        "generation_skipped": 0,
        "generation_errors": 0,
        "evaluated": 0,
        "evaluation_invalid": 0,
        "evaluation_errors": 0,
        "dry_run": 0,
    }

    execution_mode = str(resolved["model"].get("runtime", {}).get("execution_mode", "subprocess"))
    persistent = execution_mode == "persistent_worker"
    if persistent and not isinstance(adapter, Wan22Adapter):
        raise ConfigError("persistent_worker execution currently requires the wan22_official adapter")
    needs_generation = any(
        overwrite
        or not (videos_dir / f"{job['job_id']}.mp4").is_file()
        or (videos_dir / f"{job['job_id']}.mp4").stat().st_size == 0
        for job in jobs
    )
    session_context = (
        PersistentWan22Session(adapter, run_dir)
        if persistent and needs_generation and not dry_run and isinstance(adapter, Wan22Adapter)
        else nullcontext(None)
    )

    with session_context as session:
        for index, job in enumerate(jobs, start=start_index):
            job_id = job["job_id"]
            output_video = videos_dir / f"{job_id}.mp4"
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
                        "reference_subprocess_command": invocation.command,
                        "cwd": str(invocation.cwd),
                        "environment": invocation.environment,
                    },
                )
                if dry_run:
                    print(f"[{index + 1}] {shlex.join(invocation.command)}")
                    append_jsonl(state_path, {"timestamp": _timestamp(), "status": "dry_run", "job_id": job_id})
                    counts["dry_run"] += 1
                    continue

                exists = output_video.is_file() and output_video.stat().st_size > 0
                if exists and not overwrite:
                    append_jsonl(
                        state_path,
                        {"timestamp": _timestamp(), "status": "generation_skip", "job_id": job_id, "output_video": str(output_video)},
                    )
                    counts["generation_skipped"] += 1
                else:
                    append_jsonl(state_path, {"timestamp": _timestamp(), "status": "generation_running", "job_id": job_id})
                    if session is not None:
                        elapsed = session.generate(job, output_video, stdout_path)
                    else:
                        elapsed = _run_subprocess(invocation, stdout_path, stderr_path)
                    if not output_video.is_file() or output_video.stat().st_size == 0:
                        raise BenchmarkError("model returned without a non-empty output video")
                    metadata = {
                        **job,
                        "model": adapter.provenance(),
                        "execution_mode": execution_mode,
                        "output_video": str(output_video),
                        "command_record": str(command_path),
                        "stdout_log": str(stdout_path),
                        "stderr_log": str(stderr_path) if session is None else str(logs_dir / "persistent_worker.stderr.log"),
                        "elapsed_seconds": elapsed,
                        "completed_at": _timestamp(),
                    }
                    write_json(metadata_dir / f"{job_id}.json", metadata)
                    append_jsonl(
                        state_path,
                        {
                            "timestamp": _timestamp(),
                            "status": "generation_ok",
                            "job_id": job_id,
                            "output_video": str(output_video),
                            "elapsed_seconds": elapsed,
                        },
                    )
                    counts["generated"] += 1

                append_jsonl(state_path, {"timestamp": _timestamp(), "status": "evaluation_running", "job_id": job_id})
                evaluation = evaluate_job(run_dir, job_id)
                selected = evaluation["selected"]
                counts["evaluated"] += 1
                counts["evaluation_invalid"] += int(selected["invalid"] > 0)
                counts["evaluation_errors"] += int(selected["error"] > 0)
                status = "evaluation_error" if selected["error"] else "evaluation_invalid" if selected["invalid"] else "evaluation_ok"
                append_jsonl(
                    state_path,
                    {"timestamp": _timestamp(), "status": status, "job_id": job_id, "selected_counts": selected},
                )
                print(
                    f"[{index + 1}/{start_index + len(jobs)}] {job_id}: generation complete; "
                    f"evaluation ok={selected['ok']} invalid={selected['invalid']} error={selected['error']}"
                )
                if selected["error"] and fail_fast:
                    raise BenchmarkError(f"evaluation failed for {job_id}")
                if selected["invalid"] and stop_on_invalid:
                    raise BenchmarkError(f"quality/physics gate marked {job_id} invalid")
            except Exception as exc:
                if not output_video.is_file() or output_video.stat().st_size == 0:
                    counts["generation_errors"] += 1
                append_jsonl(
                    state_path,
                    {"timestamp": _timestamp(), "status": "error", "job_id": job_id, "error": str(exc)},
                )
                print(f"[{index + 1}] ERROR {job_id}: {exc}")
                if fail_fast or stop_on_invalid:
                    write_json(run_dir / "sequential_summary.json", counts)
                    raise

    write_json(run_dir / "sequential_summary.json", counts)
    if counts["generation_errors"] or counts["evaluation_errors"]:
        raise BenchmarkError(
            f"sequential run completed with {counts['generation_errors']} generation and "
            f"{counts['evaluation_errors']} evaluation error(s)"
        )
    return counts
