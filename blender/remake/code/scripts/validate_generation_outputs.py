#!/usr/bin/env python3
"""Validate canonical generation videos against a frozen JSONL manifest.

The validator deliberately uses only the Python standard library.  It always
checks the canonical ``videos/<job_id>.mp4`` contract and can additionally use
an installed ``ffprobe`` binary to verify that every file contains a readable
video stream.  It does not import or execute any model code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {line_number} is invalid JSON: {exc}") from exc
            if not isinstance(job, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            job_id = job.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError(f"manifest line {line_number} has no non-empty job_id")
            if Path(job_id).name != job_id or "/" in job_id or "\\" in job_id:
                raise ValueError(f"manifest line {line_number} has unsafe job_id: {job_id!r}")
            if job_id in seen:
                raise ValueError(f"manifest contains duplicate job_id: {job_id}")
            seen.add(job_id)
            jobs.append(job)
    if not jobs:
        raise ValueError(f"manifest contains no jobs: {path}")
    return jobs


def _ratio(value: Any) -> float | None:
    if not isinstance(value, str) or not value or value == "0/0":
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _probe_one(ffprobe: str, video: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(video),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        message = process.stderr.strip() or f"ffprobe exited with {process.returncode}"
        return {"ok": False, "error": message}
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"ffprobe returned invalid JSON: {exc}"}
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        return {"ok": False, "error": "no video stream"}
    stream = streams[0]
    format_record = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    try:
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
    except (TypeError, ValueError):
        width = height = 0
    if width <= 0 or height <= 0:
        return {"ok": False, "error": "video stream has invalid dimensions"}
    duration: float | None
    try:
        duration = float(format_record["duration"])
    except (KeyError, TypeError, ValueError):
        duration = None
    try:
        frame_count = int(stream["nb_frames"])
    except (KeyError, TypeError, ValueError):
        frame_count = None
    return {
        "ok": True,
        "codec": stream.get("codec_name"),
        "width": width,
        "height": height,
        "fps": _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate")),
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def validate_generation_outputs(
    manifest_path: Path,
    videos_dir: Path,
    *,
    metadata_dir: Path | None = None,
    expected_jobs: int | None = None,
    allow_extra: bool = False,
    ffprobe_mode: str = "auto",
    ffprobe_workers: int | None = None,
) -> dict[str, Any]:
    """Return a machine-readable validation report without mutating inputs."""

    manifest_path = manifest_path.resolve()
    videos_dir = videos_dir.resolve()
    jobs = _read_manifest(manifest_path)
    expected_ids = [str(job["job_id"]) for job in jobs]
    expected_set = set(expected_ids)
    issues: list[dict[str, Any]] = []

    if expected_jobs is not None and len(jobs) != expected_jobs:
        issues.append(
            {
                "kind": "unexpected_manifest_job_count",
                "expected": expected_jobs,
                "actual": len(jobs),
            }
        )
    if not videos_dir.is_dir():
        actual_files: dict[str, Path] = {}
        issues.append({"kind": "videos_directory_missing", "path": str(videos_dir)})
    else:
        actual_files = {path.stem: path for path in videos_dir.glob("*.mp4") if path.is_file()}

    missing = sorted(expected_set - set(actual_files))
    extra = sorted(set(actual_files) - expected_set)
    empty = sorted(job_id for job_id in expected_set & set(actual_files) if actual_files[job_id].stat().st_size == 0)
    if missing:
        issues.append({"kind": "missing_videos", "count": len(missing), "job_ids": missing})
    if empty:
        issues.append({"kind": "empty_videos", "count": len(empty), "job_ids": empty})
    if extra and not allow_extra:
        issues.append({"kind": "unexpected_videos", "count": len(extra), "job_ids": extra})

    metadata_summary: dict[str, Any] | None = None
    if metadata_dir is not None:
        metadata_dir = metadata_dir.resolve()
        if not metadata_dir.is_dir():
            metadata_files: dict[str, Path] = {}
            issues.append({"kind": "metadata_directory_missing", "path": str(metadata_dir)})
        else:
            metadata_files = {
                path.stem: path for path in metadata_dir.glob("*.json") if path.is_file()
            }
        missing_metadata = sorted(expected_set - set(metadata_files))
        extra_metadata = sorted(set(metadata_files) - expected_set)
        invalid_metadata: list[dict[str, str]] = []
        for job_id in expected_ids:
            path = metadata_files.get(job_id)
            if path is None:
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                invalid_metadata.append({"job_id": job_id, "error": str(exc)})
                continue
            if not isinstance(value, dict):
                invalid_metadata.append({"job_id": job_id, "error": "metadata is not an object"})
            elif value.get("job_id") != job_id:
                invalid_metadata.append({"job_id": job_id, "error": "job_id field differs"})
            elif value.get("status") not in {"succeeded", "ok"}:
                invalid_metadata.append(
                    {"job_id": job_id, "error": f"non-success status {value.get('status')!r}"}
                )
        if missing_metadata:
            issues.append(
                {
                    "kind": "missing_metadata",
                    "count": len(missing_metadata),
                    "job_ids": missing_metadata,
                }
            )
        if extra_metadata and not allow_extra:
            issues.append(
                {
                    "kind": "unexpected_metadata",
                    "count": len(extra_metadata),
                    "job_ids": extra_metadata,
                }
            )
        if invalid_metadata:
            issues.append(
                {
                    "kind": "invalid_metadata",
                    "count": len(invalid_metadata),
                    "records": invalid_metadata,
                }
            )
        metadata_summary = {
            "directory": str(metadata_dir),
            "found": len(metadata_files),
            "missing_count": len(missing_metadata),
            "extra_count": len(extra_metadata),
            "invalid_count": len(invalid_metadata),
        }

    if ffprobe_mode not in {"auto", "off", "required"}:
        raise ValueError(f"invalid ffprobe_mode: {ffprobe_mode!r}")
    ffprobe = None if ffprobe_mode == "off" else shutil.which("ffprobe")
    if ffprobe_mode == "required" and ffprobe is None:
        issues.append({"kind": "ffprobe_missing", "message": "ffprobe is required but not in PATH"})

    probes: dict[str, dict[str, Any]] = {}
    probe_ids = [job_id for job_id in expected_ids if job_id in actual_files and job_id not in empty]
    if ffprobe is not None and probe_ids:
        workers = ffprobe_workers or min(8, max(1, os.cpu_count() or 1))

        def run_probe(job_id: str) -> tuple[str, dict[str, Any]]:
            return job_id, _probe_one(ffprobe, actual_files[job_id])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for job_id, probe in executor.map(run_probe, probe_ids):
                probes[job_id] = probe
        failed_probes = sorted(job_id for job_id, probe in probes.items() if not probe.get("ok"))
        if failed_probes:
            issues.append(
                {
                    "kind": "unreadable_videos",
                    "count": len(failed_probes),
                    "job_ids": failed_probes,
                }
            )

    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "valid": not issues,
        "manifest": str(manifest_path),
        "videos_dir": str(videos_dir),
        "manifest_jobs": len(jobs),
        "videos_found": len(actual_files),
        "nonempty_expected_videos": len(expected_set & set(actual_files)) - len(empty),
        "missing_count": len(missing),
        "empty_count": len(empty),
        "extra_count": len(extra),
        "ffprobe_mode": ffprobe_mode,
        "ffprobe_available": ffprobe is not None,
        "ffprobe_checked": len(probes),
        "issues": issues,
    }
    if metadata_summary is not None:
        report["metadata"] = metadata_summary
    if probes:
        report["media"] = probes
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate videos/<job_id>.mp4 against a frozen canonical manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="optional metadata directory; when supplied, require one successful JSON per job",
    )
    parser.add_argument("--expected-jobs", type=int, default=None)
    parser.add_argument("--allow-extra", action="store_true")
    parser.add_argument("--ffprobe", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--ffprobe-workers", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.expected_jobs is not None and args.expected_jobs <= 0:
        raise SystemExit("error: --expected-jobs must be positive")
    if args.ffprobe_workers is not None and args.ffprobe_workers <= 0:
        raise SystemExit("error: --ffprobe-workers must be positive")
    try:
        report = validate_generation_outputs(
            args.manifest,
            args.videos,
            metadata_dir=args.metadata,
            expected_jobs=args.expected_jobs,
            allow_extra=args.allow_extra,
            ffprobe_mode=args.ffprobe,
            ffprobe_workers=args.ffprobe_workers,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
