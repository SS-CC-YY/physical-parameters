from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from remake_benchmark.core.errors import ConfigError
from remake_benchmark.core.io import read_json, read_jsonl, write_json


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _preferred_cost(cost: dict[str, Any]) -> tuple[float | None, str]:
    credits = _number(cost.get("credits"))
    if credits is not None:
        return credits, "provider_final_unit_deduction"
    cash = _number(cost.get("cash_amount")) or 0.0
    unit_equivalent = _number(cost.get("package_unit_list_equivalent"))
    if unit_equivalent is None:
        unit_equivalent = _number(cost.get("package_unit_list_equivalent_usd"))
    if cash > 0 or (unit_equivalent is not None and unit_equivalent > 0):
        return cash + max(unit_equivalent or 0.0, 0.0), "provider_cash_plus_package_unit_equivalent"
    list_cost = _number(cost.get("list_cost"))
    if list_cost is None:
        list_cost = _number(cost.get("list_cost_usd"))
    if list_cost is not None:
        return list_cost, "usage_tokens_x_list_rate"
    configured = _number(cost.get("configured_list_cost"))
    if configured is None:
        configured = _number(cost.get("configured_list_cost_usd"))
    if configured is not None:
        return configured, "configured_list_rate"
    provider_list = _number(cost.get("provider_list_price"))
    if provider_list is not None and provider_list > 0:
        return provider_list, "provider_list_price"
    return None, "unavailable"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def summarize_api_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metadata_dir = run_dir / "metadata"
    api_files = {path.name[: -len(".api.json")]: path for path in metadata_dir.glob("*.api.json")}
    manifest_path = run_dir / "manifest.jsonl"
    manifest = read_jsonl(manifest_path) if manifest_path.is_file() else []
    manifest_by_id = {
        str(job.get("job_id")): job
        for job in manifest
        if isinstance(job, dict) and job.get("job_id")
    }
    job_ids = list(manifest_by_id)
    job_ids.extend(sorted(job_id for job_id in api_files if job_id not in manifest_by_id))
    if not job_ids:
        raise ConfigError(f"no manifest jobs or closed-API metrics found in {run_dir}")

    rows: list[dict[str, Any]] = []
    for job_id in job_ids:
        path = api_files.get(job_id)
        metrics = read_json(path) if path is not None else {}
        if not isinstance(metrics, dict):
            metrics = {}
        timing = metrics.get("timing") if isinstance(metrics.get("timing"), dict) else {}
        request = metrics.get("request") if isinstance(metrics.get("request"), dict) else {}
        cost = metrics.get("cost") if isinstance(metrics.get("cost"), dict) else {}
        preferred_cost, preferred_basis = _preferred_cost(cost)
        rows.append(
            {
                "job_id": job_id,
                "metrics_present": path is not None,
                "provider": metrics.get("provider"),
                "model_id": metrics.get("model_id"),
                "task_id": metrics.get("task_id"),
                "status": metrics.get("status", "not_started_or_missing_metrics"),
                "provider_status": metrics.get("provider_status"),
                "submission_state": metrics.get("submission_state"),
                "submission_may_have_succeeded": metrics.get("submission_may_have_succeeded"),
                "requested_duration_seconds": request.get("duration_seconds"),
                "output_duration_seconds": metrics.get("output_duration_seconds"),
                "output_file_size_bytes": metrics.get("output_file_size_bytes"),
                "resolution": request.get("resolution"),
                "wall_seconds": timing.get("wall_seconds"),
                "submit_request_seconds": timing.get("submit_request_seconds"),
                "queue_seconds": timing.get("queue_seconds"),
                "processing_seconds": timing.get("processing_seconds"),
                "download_seconds": timing.get("download_seconds"),
                "completion_tokens": cost.get("completion_tokens"),
                "total_tokens": cost.get("total_tokens"),
                "credits": cost.get("credits"),
                "cash_amount": cost.get("cash_amount"),
                "package_units": cost.get("package_units"),
                "provider_list_price": cost.get("provider_list_price"),
                "preferred_cost": preferred_cost,
                "preferred_cost_usd": (
                    preferred_cost if str(cost.get("currency", "")).upper() == "USD" else None
                ),
                "preferred_cost_basis": preferred_basis,
                "currency": cost.get("currency"),
                "started_at": metrics.get("started_at"),
                "completed_at": metrics.get("completed_at"),
                "metrics_file": str(path) if path is not None else str(metadata_dir / f"{job_id}.api.json"),
            }
        )

    fieldnames = list(rows[0])
    csv_path = run_dir / "api_cost_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    succeeded = [row for row in rows if row["status"] == "succeeded"]
    metrics_rows = [row for row in rows if row["metrics_present"]]
    currencies = sorted({str(row["currency"]) for row in metrics_rows if row["currency"]})
    currency = currencies[0] if len(currencies) == 1 else None
    costs = [_number(row["preferred_cost"]) for row in metrics_rows]
    costs = [value for value in costs if value is not None]
    cash_amounts = [_number(row["cash_amount"]) for row in metrics_rows]
    cash_amounts = [value for value in cash_amounts if value is not None]
    package_units = [_number(row["package_units"]) for row in metrics_rows]
    package_units = [value for value in package_units if value is not None]
    completion_tokens = [_number(row["completion_tokens"]) for row in metrics_rows]
    completion_tokens = [value for value in completion_tokens if value is not None]
    total_tokens = [_number(row["total_tokens"]) for row in metrics_rows]
    total_tokens = [value for value in total_tokens if value is not None]
    credits = [_number(row["credits"]) for row in metrics_rows]
    credits = [value for value in credits if value is not None]
    walls = [_number(row["wall_seconds"]) for row in succeeded]
    walls = [value for value in walls if value is not None]
    queues = [_number(row["queue_seconds"]) for row in succeeded]
    queues = [value for value in queues if value is not None]
    processing = [_number(row["processing_seconds"]) for row in succeeded]
    processing = [value for value in processing if value is not None]
    starts = [_datetime(row["started_at"]) for row in succeeded]
    starts = [value for value in starts if value is not None]
    completions = [_datetime(row["completed_at"]) for row in succeeded]
    completions = [value for value in completions if value is not None]
    batch_span = None
    if starts and completions:
        batch_span = max(0.0, (max(completions) - min(starts)).total_seconds())
    can_sum_money = len(currencies) <= 1
    total_cost = round(sum(costs), 6) if costs and can_sum_money else None
    mean_cost = round(mean(costs), 6) if costs and can_sum_money else None
    summary = {
        "schema_version": "1.1.0",
        "run_dir": str(run_dir),
        "jobs": len(rows),
        "expected_manifest_jobs": len(manifest_by_id) if manifest_by_id else None,
        "metrics_found": len(metrics_rows),
        "missing_metrics": len(rows) - len(metrics_rows),
        "succeeded": len(succeeded),
        "failed_or_incomplete": len(rows) - len(succeeded),
        "currency": currency,
        "currencies": currencies,
        "total_preferred_cost": total_cost,
        "mean_preferred_cost": mean_cost,
        "total_cash_amount": round(sum(cash_amounts), 6) if cash_amounts and can_sum_money else None,
        "total_package_units": round(sum(package_units), 6) if package_units else 0.0,
        "total_completion_tokens": int(sum(completion_tokens)) if completion_tokens else 0,
        "total_tokens": int(sum(total_tokens)) if total_tokens else 0,
        "mean_completion_tokens": round(mean(completion_tokens), 3) if completion_tokens else None,
        "total_credits": round(sum(credits), 6) if credits else 0.0,
        "total_preferred_cost_usd": total_cost if str(currency).upper() == "USD" else None,
        "mean_preferred_cost_usd": mean_cost if str(currency).upper() == "USD" else None,
        "mean_wall_seconds": round(mean(walls), 3) if walls else None,
        "median_wall_seconds": round(median(walls), 3) if walls else None,
        "p90_wall_seconds": round(_percentile(walls, 0.90), 3) if walls else None,
        "min_wall_seconds": round(min(walls), 3) if walls else None,
        "max_wall_seconds": round(max(walls), 3) if walls else None,
        "median_queue_seconds": round(median(queues), 3) if queues else None,
        "p90_queue_seconds": round(_percentile(queues, 0.90), 3) if queues else None,
        "median_processing_seconds": round(median(processing), 3) if processing else None,
        "p90_processing_seconds": round(_percentile(processing, 0.90), 3) if processing else None,
        "batch_span_seconds": round(batch_span, 3) if batch_span is not None else None,
        "throughput_jobs_per_hour": (
            round(len(succeeded) * 3600.0 / batch_span, 3) if batch_span and succeeded else None
        ),
        "cost_basis_note": (
            "Seedance reports returned token usage; actual CNY deduction must be reconciled with the "
            "account bill. Kling reports final_unit_deduction as Credits; Credits are quota units, not cash."
        ),
        "timing_basis_note": (
            "Queue and processing fields are client-observed polling estimates. They can be blank when "
            "the first observed terminal state is success and can differ by up to the polling interval."
        ),
        "csv": str(csv_path),
        "rows": rows,
    }
    write_json(run_dir / "api_cost_time_summary.json", summary)
    return {key: value for key, value in summary.items() if key != "rows"}
