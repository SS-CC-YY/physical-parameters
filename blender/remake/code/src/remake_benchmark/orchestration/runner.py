from __future__ import annotations

import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remake_benchmark.core.errors import BenchmarkError, ConfigError
from remake_benchmark.core.io import append_jsonl, read_jsonl, read_yaml, write_json
from remake_benchmark.models import get_adapter


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    jobs = read_jsonl(run_dir / "manifest.jsonl")[start_index:]
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
    counts = {"ok": 0, "skip": 0, "error": 0, "dry_run": 0, "total": len(jobs)}

    for offset, job in enumerate(jobs, start_index):
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
                {"job_id": job_id, "command": invocation.command, "cwd": str(invocation.cwd), "environment": invocation.environment},
            )
            if dry_run:
                print(f"[{offset + 1}] {shlex.join(invocation.command)}")
                append_jsonl(state_path, {"timestamp": _timestamp(), "status": "dry_run", "job_id": job_id})
                counts["dry_run"] += 1
                continue
            if output_video.is_file() and output_video.stat().st_size > 0 and not overwrite:
                append_jsonl(
                    state_path,
                    {"timestamp": _timestamp(), "status": "skip", "job_id": job_id, "output_video": str(output_video)},
                )
                counts["skip"] += 1
                continue

            append_jsonl(state_path, {"timestamp": _timestamp(), "status": "running", "job_id": job_id})
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
            if process.returncode != 0 or not output_video.is_file() or output_video.stat().st_size == 0:
                raise BenchmarkError(
                    f"model process failed or produced no video (returncode={process.returncode}); see {stderr_path}"
                )
            metadata = {
                **job,
                "model": adapter.provenance(),
                "output_video": str(output_video),
                "command_record": str(command_path),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "elapsed_seconds": elapsed,
                "completed_at": _timestamp(),
            }
            write_json(metadata_dir / f"{job_id}.json", metadata)
            append_jsonl(
                state_path,
                {
                    "timestamp": _timestamp(),
                    "status": "ok",
                    "job_id": job_id,
                    "output_video": str(output_video),
                    "elapsed_seconds": elapsed,
                },
            )
            counts["ok"] += 1
        except Exception as exc:
            counts["error"] += 1
            append_jsonl(
                state_path,
                {"timestamp": _timestamp(), "status": "error", "job_id": job_id, "error": str(exc)},
            )
            print(f"[{offset + 1}] ERROR {job_id}: {exc}")
            if fail_fast:
                raise

    write_json(run_dir / "run_summary.json", counts)
    if counts["error"]:
        raise BenchmarkError(f"{counts['error']} generation job(s) failed")
    return counts
