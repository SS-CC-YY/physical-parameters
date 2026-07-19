from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from remake_benchmark.core.errors import BenchmarkError, ConfigError
from remake_benchmark.core.io import append_jsonl, read_json, read_jsonl, read_yaml, write_json
from remake_benchmark.models import get_adapter
from remake_benchmark.models.wan22 import Wan22Adapter

from .wan_session import PersistentWan22Session


_WORKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BUILTIN_CONCURRENCY_CAPS = {"seedance_ark_api": 3, "kling_api_v1": 5}


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


def _runner_artifact_paths(run_dir: Path, worker_id: str | None) -> tuple[Path, Path]:
    if worker_id is None:
        return run_dir / "run_state.jsonl", run_dir / "run_summary.json"
    if not _WORKER_ID_PATTERN.fullmatch(worker_id):
        raise ConfigError(
            "worker_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-' (maximum 64 characters)"
        )
    return (
        run_dir / f"run_state.{worker_id}.jsonl",
        run_dir / f"run_summary.{worker_id}.json",
    )


def _enforce_concurrency(model: dict[str, Any], concurrency: int, execution_mode: str) -> None:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ConfigError("concurrency must be a positive integer")
    if concurrency == 1:
        return
    if execution_mode == "persistent_worker":
        raise ConfigError("persistent_worker generation does not support concurrency greater than one")
    safety = model.get("safety", {})
    builtin_maximum = _BUILTIN_CONCURRENCY_CAPS.get(str(model.get("adapter", "")))
    configured = safety.get("max_concurrent_jobs") if isinstance(safety, dict) else None
    if configured is None:
        configured = builtin_maximum
    if configured is None:
        raise ConfigError("this model has no safety.max_concurrent_jobs; concurrency must remain one")
    try:
        maximum = int(configured)
    except (TypeError, ValueError) as exc:
        raise ConfigError("model safety.max_concurrent_jobs must be a positive integer") from exc
    if maximum <= 0:
        raise ConfigError("model safety.max_concurrent_jobs must be a positive integer")
    if builtin_maximum is not None:
        maximum = min(maximum, builtin_maximum)
    if concurrency > maximum:
        raise ConfigError(
            f"requested concurrency {concurrency} exceeds model safety.max_concurrent_jobs {maximum}"
        )


def _run_bounded_jobs(
    job_indices: list[int],
    *,
    concurrency: int,
    fail_fast: bool,
    run_one: Callable[[int], dict[str, int]],
) -> tuple[list[dict[str, int]], list[tuple[int, Exception]], list[int]]:
    """Run at most ``concurrency`` jobs and stop replenishing after a failure."""

    remaining = deque(job_indices)
    results: list[dict[str, int]] = []
    failures: list[tuple[int, Exception]] = []
    active: dict[Future[dict[str, int]], int] = {}
    stopped = False

    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(job_indices)))) as executor:
        while remaining and len(active) < concurrency:
            index = remaining.popleft()
            active[executor.submit(run_one, index)] = index

        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                index = active.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append((index, exc))
                    if fail_fast:
                        stopped = True
            while remaining and len(active) < concurrency and not stopped:
                index = remaining.popleft()
                active[executor.submit(run_one, index)] = index

    return results, failures, list(remaining)


def _generate_parallel_run(
    run_dir: Path,
    *,
    job_indices: list[int],
    concurrency: int,
    overwrite: bool,
    fail_fast: bool,
) -> dict[str, int]:
    def run_one(index: int) -> dict[str, int]:
        return generate_run(
            run_dir,
            max_jobs=1,
            start_index=index,
            overwrite=overwrite,
            fail_fast=True,
            concurrency=1,
            _worker_id=f"parallel-{index:05d}",
        )

    results, failures, not_started = _run_bounded_jobs(
        job_indices,
        concurrency=concurrency,
        fail_fast=fail_fast,
        run_one=run_one,
    )
    counts = {
        "ok": sum(result.get("ok", 0) for result in results),
        "skip": sum(result.get("skip", 0) for result in results),
        "error": len(failures),
        "dry_run": sum(result.get("dry_run", 0) for result in results),
        "not_started": len(not_started),
        "total": len(job_indices),
        "concurrency": concurrency,
    }
    main_state_path = run_dir / "run_state.jsonl"
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    not_started_set = set(not_started)
    for index in job_indices:
        if index in not_started_set:
            append_jsonl(
                main_state_path,
                {
                    "timestamp": _timestamp(),
                    "status": "not_started_after_fail_fast",
                    "manifest_index": index,
                    "job_id": manifest[index].get("job_id"),
                },
            )
            continue
        worker_state_path, _ = _runner_artifact_paths(run_dir, f"parallel-{index:05d}")
        worker_records = read_jsonl(worker_state_path) if worker_state_path.is_file() else []
        if worker_records:
            record = dict(worker_records[-1])
            record["parallel_worker_state"] = str(worker_state_path)
            append_jsonl(main_state_path, record)
    write_json(run_dir / "run_summary.json", counts)
    for index, exc in failures:
        print(f"[{index + 1}] PARALLEL ERROR: {exc}")
    if failures:
        first_index, first_error = failures[0]
        raise BenchmarkError(
            f"{len(failures)} parallel generation job(s) failed; first failure at "
            f"manifest index {first_index}: {first_error}"
        )
    return counts


def generate_run(
    run_dir: Path,
    *,
    dry_run: bool = False,
    max_jobs: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
    fail_fast: bool = False,
    concurrency: int = 1,
    _worker_id: str | None = None,
) -> dict[str, int]:
    run_dir = run_dir.resolve()
    if start_index < 0:
        raise ConfigError("start_index must be non-negative")
    state_path, summary_path = _runner_artifact_paths(run_dir, _worker_id)
    resolved = read_yaml(run_dir / "resolved_build.yaml")
    all_jobs = read_jsonl(run_dir / "manifest.jsonl")
    _enforce_max_billable_jobs_per_run(resolved["model"], len(all_jobs))
    jobs = all_jobs[start_index:]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    execution_mode = str(resolved["model"].get("runtime", {}).get("execution_mode", "subprocess"))
    _enforce_concurrency(resolved["model"], concurrency, execution_mode)
    if overwrite and resolved["model"].get("safety", {}).get("max_billable_jobs_per_run") is not None:
        raise ConfigError("--overwrite is disabled for billable API runs; use the saved task ID or a new run tag")
    workspace_root = Path(resolved["workspace_root"])
    adapter = get_adapter(resolved["model"], workspace_root)
    if concurrency > 1 and not dry_run and len(jobs) > 1:
        return _generate_parallel_run(
            run_dir,
            job_indices=list(range(start_index, start_index + len(jobs))),
            concurrency=concurrency,
            overwrite=overwrite,
            fail_fast=fail_fast,
        )
    videos_dir = run_dir / "videos"
    metadata_dir = run_dir / "metadata"
    logs_dir = run_dir / "logs"
    for directory in (videos_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "skip": 0, "error": 0, "dry_run": 0, "total": len(jobs)}
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

    write_json(summary_path, counts)
    if counts["error"]:
        raise BenchmarkError(f"{counts['error']} generation job(s) failed")
    return counts
