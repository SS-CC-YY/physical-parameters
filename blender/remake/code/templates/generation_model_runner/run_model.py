#!/usr/bin/env python3
"""Load one local I2V model once and consume a frozen canonical manifest.

Only the two functions marked ``TODO(PROVIDER)`` are model-specific.  The
remaining code is shared output-contract plumbing and intentionally has no
third-party dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|private[_-]?key|secret)",
    re.IGNORECASE,
)
PUBLIC_MODEL_FIELDS = (
    "schema_version",
    "model_id",
    "model_revision",
    "repo_path",
    "checkpoint_path",
    "device",
    "generation",
)


def load_model_once(config: Mapping[str, Any]) -> Any:
    """TODO(PROVIDER): import and load the official model/checkpoint once."""

    raise NotImplementedError(
        "TODO(PROVIDER): implement load_model_once(config) for the selected local model"
    )


def generate_video(
    model: Any,
    job: Mapping[str, Any],
    input_image: Path,
    output_video: Path,
    config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """TODO(PROVIDER): generate one job and write MP4 to output_video.

    ``output_video`` is a temporary path ending in ``.mp4``.  The common loop
    validates that it is non-empty and atomically publishes it under the
    canonical job ID.  Return only non-sensitive, JSON-compatible metadata.
    """

    del model, job, input_image, output_video, config
    raise NotImplementedError(
        "TODO(PROVIDER): implement generate_video(...) for the selected local model"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered + "\n")
        handle.flush()


def _assert_no_secret_keys(value: Any, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if SECRET_KEY_PATTERN.search(key_text):
                raise ValueError(
                    f"{location}.{key_text}: secret-like config keys are forbidden; "
                    "install/download the local model separately"
                )
            _assert_no_secret_keys(item, f"{location}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_secret_keys(item, f"{location}[{index}]")


def _read_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    _assert_no_secret_keys(value)
    model_id = value.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("config.model_id must be a non-empty string")
    return value


def _public_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in PUBLIC_MODEL_FIELDS if key in config}


def _iter_manifest(path: Path, max_jobs: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    emitted = 0
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
            if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
                raise ValueError(f"manifest line {line_number} has unsafe job_id: {job_id!r}")
            if job_id in seen:
                raise ValueError(f"manifest contains duplicate job_id: {job_id}")
            seen.add(job_id)
            emitted += 1
            yield line_number, job
            if max_jobs is not None and emitted >= max_jobs:
                return


def _safe_input_image(bundle_root: Path, job: Mapping[str, Any]) -> Path:
    inputs = job.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("job.inputs must be an object")
    raw_image = inputs.get("image")
    if not isinstance(raw_image, str) or not raw_image:
        raise ValueError("job.inputs.image must be a non-empty relative path")
    relative = Path(raw_image)
    if relative.is_absolute():
        raise ValueError(f"absolute conditioning-image path is forbidden: {raw_image}")
    resolved = (bundle_root / relative).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError(f"conditioning-image path leaves bundle root: {raw_image}") from exc
    if not resolved.is_file():
        raise ValueError(f"conditioning image does not exist: {resolved}")
    return resolved


def _successful_result(video: Path, metadata: Path) -> bool:
    if not video.is_file() or video.stat().st_size <= 0 or not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("status") == "succeeded"


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_generation(
    bundle_root: Path,
    output_root: Path,
    config_path: Path,
    *,
    max_jobs: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    manifest_path = bundle_root / "manifest.jsonl"
    if not bundle_root.is_dir():
        raise ValueError(f"bundle root does not exist: {bundle_root}")
    if not manifest_path.is_file():
        raise ValueError(f"frozen manifest does not exist: {manifest_path}")
    if max_jobs is not None and max_jobs <= 0:
        raise ValueError("max_jobs must be positive")

    config = _read_config(config_path)
    model_record = _public_model_config(config)
    manifest_sha256 = _sha256(manifest_path)
    videos_dir = output_root / "videos"
    metadata_dir = output_root / "metadata"
    logs_dir = output_root / "logs"
    for directory in (videos_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "run_state.jsonl"
    summary_path = output_root / "run_summary.json"

    run_started = _utc_now()
    counts: dict[str, Any] = {
        "schema_version": "1.0.0",
        "model_id": config["model_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "started_at": run_started,
        "ok": 0,
        "skip": 0,
        "error": 0,
        "total": 0,
        "resume": resume,
        "max_jobs": max_jobs,
    }

    load_started = time.monotonic()
    try:
        model = load_model_once(config)
    except Exception as exc:
        counts.update(
            {
                "status": "fatal_model_load_error",
                "error": 1,
                "fatal_error_type": type(exc).__name__,
                "fatal_error": str(exc),
                "completed_at": _utc_now(),
                "elapsed_seconds": round(time.monotonic() - load_started, 6),
            }
        )
        _atomic_write_json(summary_path, counts)
        raise
    model_load_seconds = time.monotonic() - load_started

    try:
        for line_number, job in _iter_manifest(manifest_path, max_jobs):
            counts["total"] += 1
            job_id = str(job["job_id"])
            output_video = videos_dir / f"{job_id}.mp4"
            output_metadata = metadata_dir / f"{job_id}.json"
            output_log = logs_dir / f"{job_id}.log"
            if resume and _successful_result(output_video, output_metadata):
                counts["skip"] += 1
                _append_jsonl(
                    state_path,
                    {"timestamp": _utc_now(), "status": "skip", "job_id": job_id},
                )
                continue
            if not resume and (output_video.exists() or output_metadata.exists()):
                counts["error"] += 1
                message = "canonical output already exists; pass --resume or use a new output root"
                _write_log(output_log, [f"status=error", f"message={message}"])
                _append_jsonl(
                    state_path,
                    {"timestamp": _utc_now(), "status": "error", "job_id": job_id, "error": message},
                )
                continue

            started_at = _utc_now()
            started = time.monotonic()
            temporary_video = videos_dir / f".{job_id}.{uuid.uuid4().hex}.tmp.mp4"
            _append_jsonl(
                state_path,
                {"timestamp": started_at, "status": "running", "job_id": job_id},
            )
            try:
                input_image = _safe_input_image(bundle_root, job)
                provider_metadata = generate_video(
                    model, job, input_image, temporary_video, config
                )
                if provider_metadata is not None and not isinstance(provider_metadata, Mapping):
                    raise TypeError("generate_video must return a mapping or None")
                if not temporary_video.is_file() or temporary_video.stat().st_size <= 0:
                    raise RuntimeError("provider produced no non-empty MP4 at the supplied output path")
                os.replace(temporary_video, output_video)
                completed_at = _utc_now()
                elapsed = time.monotonic() - started
                provider_record = dict(provider_metadata or {})
                metadata = {
                    "schema_version": "1.0.0",
                    "job_id": job_id,
                    "status": "succeeded",
                    "manifest_sha256": manifest_sha256,
                    "input": {
                        "image": str(job["inputs"]["image"]),
                        "image_sha256": _sha256(input_image),
                    },
                    "prompt": {
                        "canonical": str(job.get("prompt", "")),
                        "canonical_negative": str(job.get("negative_prompt", "")),
                        "provider_text": provider_record.get(
                            "provider_prompt", str(job.get("prompt", ""))
                        ),
                        "provider_negative": provider_record.get(
                            "provider_negative_prompt", str(job.get("negative_prompt", ""))
                        ),
                        "transform_id": provider_record.get("prompt_transform_id", "identity"),
                    },
                    "seed": {
                        "requested": job.get("seed"),
                        "supported": provider_record.get("seed_supported"),
                        "sent": provider_record.get("seed_sent"),
                        "effective": provider_record.get("effective_seed"),
                    },
                    "model": model_record,
                    "requested_generation": dict(job.get("generation", {})),
                    "actual_generation": provider_record.get("actual_generation", {}),
                    "media_probe": provider_record.get("media_probe", {}),
                    "output_video": f"videos/{job_id}.mp4",
                    "output_sha256": _sha256(output_video),
                    "timing": {
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "wall_seconds": round(elapsed, 6),
                    },
                    "provider": provider_record,
                    "postprocess": provider_record.get("postprocess", []),
                    "error": None,
                }
                _assert_no_secret_keys(metadata, "metadata")
                _atomic_write_json(output_metadata, metadata)
                _write_log(
                    output_log,
                    [
                        "status=ok",
                        f"job_id={job_id}",
                        f"manifest_line={line_number}",
                        f"elapsed_seconds={elapsed:.6f}",
                        f"output_video={output_video}",
                    ],
                )
                counts["ok"] += 1
                _append_jsonl(
                    state_path,
                    {
                        "timestamp": completed_at,
                        "status": "succeeded",
                        "job_id": job_id,
                        "output_video": str(output_video),
                        "elapsed_seconds": round(elapsed, 6),
                    },
                )
            except Exception as exc:
                if temporary_video.exists():
                    temporary_video.unlink()
                elapsed = time.monotonic() - started
                counts["error"] += 1
                error_type = type(exc).__name__
                error_message = str(exc)
                failed_metadata = {
                    "schema_version": "1.0.0",
                    "job_id": job_id,
                    "status": "failed",
                    "manifest_sha256": manifest_sha256,
                    "input": {
                        "image": str(job.get("inputs", {}).get("image", "")),
                        "image_sha256": None,
                    },
                    "prompt": {
                        "canonical": str(job.get("prompt", "")),
                        "canonical_negative": str(job.get("negative_prompt", "")),
                        "provider_text": None,
                        "provider_negative": None,
                        "transform_id": None,
                    },
                    "seed": {
                        "requested": job.get("seed"),
                        "supported": None,
                        "sent": None,
                        "effective": None,
                    },
                    "model": model_record,
                    "requested_generation": dict(job.get("generation", {})),
                    "actual_generation": {},
                    "media_probe": {},
                    "output_video": f"videos/{job_id}.mp4",
                    "output_sha256": None,
                    "timing": {
                        "started_at": started_at,
                        "completed_at": _utc_now(),
                        "wall_seconds": round(elapsed, 6),
                    },
                    "provider": {},
                    "postprocess": [],
                    "error": {"type": error_type, "message": error_message},
                }
                _assert_no_secret_keys(failed_metadata, "metadata")
                _atomic_write_json(output_metadata, failed_metadata)
                _write_log(
                    output_log,
                    [
                        "status=error",
                        f"job_id={job_id}",
                        f"manifest_line={line_number}",
                        f"error_type={error_type}",
                        f"error={error_message}",
                        "traceback:",
                        traceback.format_exc(),
                    ],
                )
                _append_jsonl(
                    state_path,
                    {
                        "timestamp": _utc_now(),
                        "status": "failed",
                        "job_id": job_id,
                        "error_type": error_type,
                        "error": error_message,
                    },
                )
                print(f"ERROR {job_id}: {error_type}: {error_message}", file=sys.stderr, flush=True)
    finally:
        completed_at = _utc_now()
        counts.update(
            {
                "status": "complete" if counts["error"] == 0 else "complete_with_errors",
                "model_load_seconds": round(model_load_seconds, 6),
                "completed_at": completed_at,
            }
        )
        _atomic_write_json(summary_path, counts)
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Template load-once runner for a frozen canonical I2V manifest."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = run_generation(
            args.bundle_root,
            args.output_root,
            args.config,
            max_jobs=args.max_jobs,
            resume=args.resume,
        )
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
