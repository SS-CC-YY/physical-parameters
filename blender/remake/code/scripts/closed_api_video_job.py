#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_FAILURES = {"failed", "cancelled", "expired"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_existing(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _submission_marker_path(metrics_out: Path) -> Path:
    return metrics_out.with_suffix(metrics_out.suffix + ".submit.json")


def _reserve_submission(args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    """Atomically reserve the one allowed POST for this run/job.

    The marker is intentionally permanent. If the process loses the POST response,
    an automatic retry could create a second billable task, so a later run refuses
    to submit again until an operator reconciles the provider console.
    """

    marker = _submission_marker_path(args.metrics_out)
    marker.parent.mkdir(parents=True, exist_ok=True)
    reservation = {
        "schema_version": "1.0.0",
        "provider": args.provider,
        "model_id": args.model,
        "reserved_at": _timestamp(),
        "metrics_file": str(args.metrics_out),
        "external_task_id": args.external_task_id or None,
        "warning": "Do not delete before reconciling any uncertain submission with the provider console.",
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(reservation, indent=2, ensure_ascii=False) + "\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"submission marker exists but no saved provider task_id: {marker}. "
            "Automatic POST retry is blocked to prevent duplicate billing; reconcile the provider "
            "console before manually starting a new run tag."
        ) from exc
    metrics.update(
        {
            "status": "submitting",
            "submission_state": "intent_reserved",
            "submission_marker": str(marker),
            "submission_intent_at": reservation["reserved_at"],
            "submission_may_have_succeeded": True,
        }
    )
    _atomic_json(args.metrics_out, metrics)


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    retry_reads: int = 4,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "remake-video-benchmark/0.1",
        },
    )
    attempts = retry_reads if method in {"GET", "HEAD"} else 1
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError(f"API returned a non-object response from {url}")
            return value
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            if method in {"GET", "HEAD"} and exc.code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise RuntimeError(f"API HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if method in {"GET", "HEAD"} and attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise RuntimeError(f"API network error from {url}: {exc}") from exc
    raise AssertionError("unreachable")


def _download(url: str, output: Path) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "remake-video-benchmark/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
        if temporary.stat().st_size == 0:
            raise RuntimeError("provider returned an empty video")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return time.monotonic() - started


def _image_value(path: Path, encoding: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if encoding == "raw_base64":
        return encoded
    if encoding != "data_uri":
        raise ValueError(f"unsupported image encoding: {encoding}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{encoded}"


def _combined_prompt(prompt: str, negative_prompt: str, strategy: str) -> str:
    negative_prompt = negative_prompt.strip()
    if not negative_prompt or strategy in {"ignore", "separate"}:
        return prompt
    if strategy == "append":
        return f"{prompt}\nAvoid: {negative_prompt}"
    raise ValueError(f"unsupported negative prompt strategy: {strategy}")


def _observe(metrics: dict[str, Any], status: str, started: float) -> None:
    timing = metrics.setdefault("timing", {})
    elapsed = float(timing.get("resume_offset_seconds", 0) or 0) + time.monotonic() - started
    history = metrics.setdefault("poll_history", [])
    if not history or history[-1]["status"] != status:
        history.append(
            {
                "status": status,
                "observed_at": _timestamp(),
                "elapsed_seconds": round(elapsed, 3),
            }
        )
    if status in {"running", "processing"} and "first_processing_seconds" not in timing:
        timing["first_processing_seconds"] = round(elapsed, 3)


def _finalize_timing(metrics: dict[str, Any], started: float) -> None:
    timing = metrics.setdefault("timing", {})
    wall = float(timing.get("resume_offset_seconds", 0) or 0) + time.monotonic() - started
    timing["wall_seconds"] = round(wall, 3)
    first_processing = timing.get("first_processing_seconds")
    submitted = timing.get("submit_seconds")
    provider_completed = timing.get("provider_completed_seconds")
    if first_processing is not None and submitted is not None:
        timing["queue_seconds"] = round(max(0.0, float(first_processing) - float(submitted)), 3)
    if first_processing is not None and provider_completed is not None:
        timing["processing_seconds"] = round(
            max(0.0, float(provider_completed) - float(first_processing)), 3
        )


def _configured_rate(args: argparse.Namespace, generic_name: str, legacy_name: str) -> float | None:
    value = getattr(args, generic_name, None)
    if value is None:
        value = getattr(args, legacy_name, None)
    return None if value is None else float(value)


def _seedance_cost(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    usage = task.get("usage") if isinstance(task.get("usage"), dict) else {}
    tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", tokens) or tokens)
    rate = _configured_rate(args, "price_per_million_tokens", "usd_per_million_tokens")
    estimated = None if rate is None or tokens <= 0 else tokens * rate / 1_000_000
    return {
        "basis": "usage.completion_tokens_x_configured_list_rate",
        "currency": args.currency,
        "completion_tokens": tokens,
        "total_tokens": total_tokens,
        "price_per_million_tokens": rate,
        "list_cost": None if estimated is None else round(estimated, 6),
        "note": "List cost; prepaid packages or account discounts can change cash balance deduction.",
    }


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _kling_cost(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Any]:
    deduction = task.get("final_unit_deduction")
    credits = _number(deduction)
    if credits is None and isinstance(deduction, dict):
        for key in ("amount", "credits", "units", "value"):
            credits = _number(deduction.get(key))
            if credits is not None:
                break
    configured = None
    price_per_second = _configured_rate(args, "price_per_second", "usd_per_second")
    if price_per_second is not None:
        configured = args.duration * price_per_second
    return {
        "basis": "provider_final_unit_deduction",
        "currency": args.currency,
        "credits": None if credits is None else round(credits, 6),
        "final_unit_deduction": deduction,
        "configured_list_cost": None if configured is None else round(configured, 6),
        "price_per_second": price_per_second,
        "note": "Credits are provider quota units, not a cash currency. Reconcile with the account balance.",
    }


def _record_seedance_terminal(
    args: argparse.Namespace, metrics: dict[str, Any], task: dict[str, Any]
) -> None:
    provider_status = str(task.get("status", "unknown")).lower()
    metrics["provider_status"] = provider_status
    metrics["status"] = (
        "provider_succeeded_pending_download"
        if provider_status == "succeeded"
        else "provider_terminal_failure"
    )
    metrics["pipeline_status"] = metrics["status"]
    metrics["usage"] = task.get("usage", {})
    metrics["cost"] = _seedance_cost(args, task)
    metrics["output_duration_seconds"] = task.get("duration")
    metrics["provider_task_raw"] = task
    metrics["provider_task"] = {
        key: task.get(key)
        for key in (
            "id", "model", "status", "created_at", "updated_at", "seed", "resolution", "ratio",
            "duration", "framespersecond", "service_tier", "generate_audio", "draft", "priority",
        )
        if key in task
    }
    _atomic_json(args.metrics_out, metrics)


def _record_kling_terminal(
    args: argparse.Namespace, metrics: dict[str, Any], task: dict[str, Any]
) -> None:
    provider_status = str(task.get("task_status", "unknown")).lower()
    metrics["provider_status"] = provider_status
    metrics["status"] = (
        "provider_succeeded_pending_download"
        if provider_status == "succeed"
        else "provider_terminal_failure"
    )
    metrics["pipeline_status"] = metrics["status"]
    metrics["cost"] = _kling_cost(args, task)
    metrics["provider_task_raw"] = task
    metrics["provider_task"] = {
        key: task.get(key)
        for key in (
            "task_id",
            "task_status",
            "task_status_msg",
            "created_at",
            "updated_at",
            "final_unit_deduction",
        )
        if key in task
    }
    result = task.get("task_result") if isinstance(task.get("task_result"), dict) else {}
    outputs = result.get("videos") if isinstance(result.get("videos"), list) else []
    video = next((item for item in outputs if isinstance(item, dict) and item.get("url")), None)
    if video:
        metrics["output_duration_seconds"] = video.get("duration")
    _atomic_json(args.metrics_out, metrics)


def _run_seedance(args: argparse.Namespace, api_key: str, metrics: dict[str, Any], started: float) -> None:
    task_id = None
    existing = _read_existing(args.metrics_out)
    if existing and existing.get("provider") == "seedance" and existing.get("model_id") == args.model:
        if existing.get("status") not in TERMINAL_FAILURES:
            task_id = existing.get("task_id")
            if task_id:
                metrics.update(existing)
                metrics["resumed_at"] = _timestamp()
                metrics["status"] = "resuming"
                previous_timing = existing.get("timing") if isinstance(existing.get("timing"), dict) else {}
                metrics.setdefault("timing", {})["resume_offset_seconds"] = float(
                    previous_timing.get("wall_seconds", previous_timing.get("resume_offset_seconds", 0)) or 0
                )
                _atomic_json(args.metrics_out, metrics)

    if not task_id:
        payload = {
            "model": args.model,
            "content": [
                {"type": "text", "text": _combined_prompt(args.prompt, args.negative_prompt, args.negative_prompt_strategy)},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_value(args.image, args.image_encoding)},
                    "role": "first_frame",
                },
            ],
            "resolution": args.resolution,
            "ratio": args.ratio,
            "duration": args.duration,
            "watermark": args.watermark,
            "generate_audio": args.generate_audio,
        }
        if args.send_seed:
            payload["seed"] = args.seed
        if args.send_camera_fixed:
            payload["camera_fixed"] = args.camera_fixed
        metrics["submitted_payload_controls"] = {
            "seed_applied": bool(args.send_seed),
            "camera_fixed_applied": bool(args.send_camera_fixed),
        }
        _reserve_submission(args, metrics)
        submit_started = time.monotonic()
        response = _request_json(
            "POST",
            f"{args.api_base.rstrip('/')}/contents/generations/tasks",
            api_key=api_key,
            payload=payload,
        )
        metrics["timing"]["submit_request_seconds"] = round(time.monotonic() - submit_started, 3)
        task_id = response.get("id")
        if not task_id:
            raise RuntimeError(f"Seedance create response has no task id: {response}")
        metrics.update(
            {
                "task_id": str(task_id),
                "status": "submitted",
                "submitted_at": _timestamp(),
                "submission_state": "task_id_saved",
                "submission_may_have_succeeded": False,
                "submission_resolved_at": _timestamp(),
            }
        )
        metrics["timing"]["submit_seconds"] = round(time.monotonic() - started, 3)
        _atomic_json(args.metrics_out, metrics)

    deadline = time.monotonic() + args.timeout_seconds
    final_task: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        task = _request_json(
            "GET",
            f"{args.api_base.rstrip('/')}/contents/generations/tasks/{urllib.parse.quote(str(task_id), safe='')}",
            api_key=api_key,
        )
        status = str(task.get("status", "unknown")).lower()
        metrics["status"] = status
        _observe(metrics, status, started)
        _atomic_json(args.metrics_out, metrics)
        if status == "succeeded":
            final_task = task
            timing = metrics.setdefault("timing", {})
            timing["provider_completed_seconds"] = round(
                float(timing.get("resume_offset_seconds", 0) or 0) + time.monotonic() - started,
                3,
            )
            _record_seedance_terminal(args, metrics, task)
            break
        if status in TERMINAL_FAILURES:
            timing = metrics.setdefault("timing", {})
            timing["provider_completed_seconds"] = round(
                float(timing.get("resume_offset_seconds", 0) or 0) + time.monotonic() - started,
                3,
            )
            _record_seedance_terminal(args, metrics, task)
            raise RuntimeError(f"Seedance task {task_id} ended with {status}: {task.get('error')}")
        time.sleep(args.poll_interval_seconds)
    if final_task is None:
        raise TimeoutError(f"Seedance task {task_id} exceeded {args.timeout_seconds}s timeout")

    content = final_task.get("content") if isinstance(final_task.get("content"), dict) else {}
    video_url = content.get("video_url")
    if not video_url:
        raise RuntimeError(f"Seedance task {task_id} succeeded without content.video_url")
    metrics["timing"]["download_seconds"] = round(_download(str(video_url), args.output), 3)


def _kling_data(response: dict[str, Any], operation: str) -> dict[str, Any]:
    code = int(response.get("code", 0) or 0)
    if code not in {0, 200}:
        raise RuntimeError(f"Kling {operation} failed: {response}")
    data = response.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"Kling {operation} response has no data object: {response}")
    return data


def _run_kling(args: argparse.Namespace, api_key: str, metrics: dict[str, Any], started: float) -> None:
    task_id = None
    existing = _read_existing(args.metrics_out)
    if existing and existing.get("provider") == "kling" and existing.get("model_id") == args.model:
        if existing.get("status") not in TERMINAL_FAILURES:
            task_id = existing.get("task_id")
            if task_id:
                metrics.update(existing)
                metrics["resumed_at"] = _timestamp()
                metrics["status"] = "resuming"
                previous_timing = existing.get("timing") if isinstance(existing.get("timing"), dict) else {}
                metrics.setdefault("timing", {})["resume_offset_seconds"] = float(
                    previous_timing.get("wall_seconds", previous_timing.get("resume_offset_seconds", 0)) or 0
                )
                _atomic_json(args.metrics_out, metrics)

    if not task_id:
        payload = {
            "model_name": args.model,
            "image": _image_value(args.image, args.image_encoding),
            "image_tail": "",
            "prompt": args.prompt,
            "negative_prompt": "" if args.negative_prompt_strategy == "ignore" else args.negative_prompt,
            "duration": str(args.duration),
            "mode": args.mode,
            "sound": args.sound,
            "callback_url": "",
            "external_task_id": args.external_task_id,
        }
        metrics["submitted_payload_controls"] = {
            "seed_applied": False,
            "camera_fixed_applied": False,
            "stochasticity_control": "provider_managed",
        }
        _reserve_submission(args, metrics)
        submit_started = time.monotonic()
        response = _request_json(
            "POST",
            f"{args.api_base.rstrip('/')}/v1/videos/image2video",
            api_key=api_key,
            payload=payload,
        )
        metrics["timing"]["submit_request_seconds"] = round(time.monotonic() - submit_started, 3)
        data = _kling_data(response, "create task")
        task_id = data.get("task_id")
        metrics["request_id"] = response.get("request_id")
        metrics["provider_submit_data"] = data
        if not task_id:
            raise RuntimeError(f"Kling create response has no task id: {response}")
        metrics.update(
            {
                "submission_state": "task_id_saved",
                "submission_may_have_succeeded": False,
                "submission_resolved_at": _timestamp(),
            }
        )

    metrics.update(
        {
            "task_id": str(task_id),
            "external_task_id": args.external_task_id,
            "status": "submitted",
            "submitted_at": metrics.get("submitted_at", _timestamp()),
        }
    )
    metrics["timing"].setdefault("submit_seconds", round(time.monotonic() - started, 3))
    _atomic_json(args.metrics_out, metrics)

    deadline = time.monotonic() + args.timeout_seconds
    final_task: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = _request_json(
            "GET",
            f"{args.api_base.rstrip('/')}/v1/videos/image2video/{urllib.parse.quote(str(task_id), safe='')}",
            api_key=api_key,
        )
        task = _kling_data(response, "query task")
        status = str(task.get("task_status", "unknown")).lower()
        metrics["status"] = status
        _observe(metrics, status, started)
        _atomic_json(args.metrics_out, metrics)
        if status == "succeed":
            final_task = task
            timing = metrics.setdefault("timing", {})
            timing["provider_completed_seconds"] = round(
                float(timing.get("resume_offset_seconds", 0) or 0) + time.monotonic() - started,
                3,
            )
            _record_kling_terminal(args, metrics, task)
            break
        if status in TERMINAL_FAILURES:
            timing = metrics.setdefault("timing", {})
            timing["provider_completed_seconds"] = round(
                float(timing.get("resume_offset_seconds", 0) or 0) + time.monotonic() - started,
                3,
            )
            _record_kling_terminal(args, metrics, task)
            raise RuntimeError(f"Kling task {task_id} ended with {status}: {task.get('task_status_msg')}")
        time.sleep(args.poll_interval_seconds)
    if final_task is None:
        raise TimeoutError(f"Kling task {task_id} exceeded {args.timeout_seconds}s timeout")

    result = final_task.get("task_result") if isinstance(final_task.get("task_result"), dict) else {}
    outputs = result.get("videos") if isinstance(result.get("videos"), list) else []
    video = next((item for item in outputs if isinstance(item, dict) and item.get("url")), None)
    if not video or not video.get("url"):
        raise RuntimeError(f"Kling task {task_id} succeeded without a video output")
    metrics["timing"]["download_seconds"] = round(_download(str(video["url"]), args.output), 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one resumable Seedance or Kling I2V API task")
    parser.add_argument("--provider", choices=("seedance", "kling"), required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-encoding", choices=("data_uri", "raw_base64"), default="data_uri")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--negative-prompt-strategy", choices=("append", "ignore", "separate"), default="append"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--ratio", default="adaptive")
    parser.add_argument("--mode", choices=("std", "pro"), default="std")
    parser.add_argument("--sound", choices=("on", "off"), default="off")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--external-task-id", default="")
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--currency", default="USD")
    parser.add_argument(
        "--price-per-million-tokens",
        "--usd-per-million-tokens",
        dest="price_per_million_tokens",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--price-per-second", "--usd-per-second", dest="price_per_second", type=float, default=None
    )
    parser.add_argument("--camera-fixed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--send-seed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--send-camera-fixed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"required API key environment variable is not set: {args.api_key_env}")
    if not args.image.is_file():
        raise SystemExit(f"conditioning image not found: {args.image}")
    if args.poll_interval_seconds <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("poll interval and timeout must be positive")
    if args.provider == "kling" and not args.external_task_id:
        raise SystemExit("--external-task-id is required for Kling duplicate-submission protection")
    if args.provider == "kling" and (args.send_seed or args.send_camera_fixed):
        raise SystemExit("Kling I2V does not accept canonical seed or camera_fixed request fields")

    started = time.monotonic()
    metrics: dict[str, Any] = {
        "schema_version": "1.0.0",
        "provider": args.provider,
        "model_id": args.model,
        "status": "starting",
        "started_at": _timestamp(),
        "request": {
            "resolution": args.resolution,
            "duration_seconds": args.duration,
            "ratio": args.ratio if args.provider == "seedance" else None,
            "mode": args.mode if args.provider == "kling" else None,
            "sound": args.sound if args.provider == "kling" else None,
            "canonical_seed": args.seed,
            "seed_applied": bool(args.send_seed and args.provider == "seedance"),
            "camera_fixed_requested": args.camera_fixed if args.provider == "seedance" else None,
            "camera_fixed_applied": bool(args.send_camera_fixed and args.provider == "seedance"),
            "stochasticity_control": (
                "canonical_seed" if args.send_seed and args.provider == "seedance" else "provider_managed"
            ),
            "generate_audio": args.generate_audio if args.provider == "seedance" else None,
            "watermark": args.watermark,
            "image_file": str(args.image),
        },
        "timing": {},
        "poll_history": [],
    }
    try:
        if args.provider == "seedance":
            _run_seedance(args, api_key, metrics, started)
        else:
            _run_kling(args, api_key, metrics, started)
        metrics["output_file_size_bytes"] = args.output.stat().st_size
        metrics["status"] = "succeeded"
        metrics["pipeline_status"] = "succeeded"
        metrics["completed_at"] = _timestamp()
        _finalize_timing(metrics, started)
        _atomic_json(args.metrics_out, metrics)
        print(json.dumps({"provider": args.provider, "task_id": metrics.get("task_id"), "cost": metrics.get("cost")}, ensure_ascii=False))
    except Exception as exc:
        metrics["status"] = "client_error"
        metrics["pipeline_status"] = "client_error"
        metrics["error"] = str(exc)
        metrics["failed_at"] = _timestamp()
        _finalize_timing(metrics, started)
        _atomic_json(args.metrics_out, metrics)
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
