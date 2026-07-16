from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, TextIO

from remake_benchmark.core.errors import BenchmarkError
from remake_benchmark.core.io import write_json
from remake_benchmark.models.wan22 import Wan22Adapter


RESULT_PREFIX = "__REMAKE_WAN22_RESULT__"


class PersistentWan22Session:
    """A JSONL controller for one server-side Wan2.2 process."""

    def __init__(self, adapter: Wan22Adapter, run_dir: Path):
        self.adapter = adapter
        self.run_dir = run_dir
        self.process: subprocess.Popen[str] | None = None
        self.stderr_handle: TextIO | None = None

    def __enter__(self) -> "PersistentWan22Session":
        invocation = self.adapter.persistent_worker_invocation()
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            logs_dir / "persistent_worker.command.json",
            {
                "command": invocation.command,
                "cwd": str(invocation.cwd),
                "environment": invocation.environment,
            },
        )
        environment = os.environ.copy()
        environment.update(invocation.environment)
        self.stderr_handle = (logs_dir / "persistent_worker.stderr.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            invocation.command,
            cwd=invocation.cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        return self

    def generate(self, job: dict[str, Any], output_video: Path, stdout_path: Path) -> float:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise BenchmarkError("persistent Wan2.2 worker is not running")
        if self.process.poll() is not None:
            raise BenchmarkError(
                f"persistent Wan2.2 worker exited with code {self.process.returncode}; "
                f"see {self.run_dir / 'logs' / 'persistent_worker.stderr.log'}"
            )
        request = self.adapter.persistent_request(job, output_video)
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("a", encoding="utf-8") as output_log:
            while True:
                line = self.process.stdout.readline()
                if line == "":
                    raise BenchmarkError(
                        f"persistent Wan2.2 worker stopped before responding; "
                        f"see {self.run_dir / 'logs' / 'persistent_worker.stderr.log'}"
                    )
                if not line.startswith(RESULT_PREFIX):
                    output_log.write(line)
                    output_log.flush()
                    continue
                response = json.loads(line[len(RESULT_PREFIX) :])
                if response.get("job_id") != job["job_id"]:
                    raise BenchmarkError(
                        f"persistent worker response mismatch: expected {job['job_id']}, got {response.get('job_id')}"
                    )
                if not response.get("ok"):
                    hint = ""
                    if response.get("error_type") in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
                        hint = " Set WAN_OFFLOAD_MODEL=true and start a new run."
                    raise BenchmarkError(f"Wan2.2 worker failed: {response.get('error')}.{hint}")
                return float(response["elapsed_seconds"])

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        process = self.process
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if self.stderr_handle is not None:
            self.stderr_handle.close()
