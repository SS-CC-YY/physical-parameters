"""Compact, model-agnostic PhysParamBench report for paper deadlines.

The report deliberately separates three questions:

1. parameter fidelity on the clean baseline Side view;
2. trajectory/dynamics fidelity of the fitted equation;
3. matched robustness comparisons for background, view and composition.

It consumes frozen ``all_jobs.csv`` plus ``jobs/<job_id>/result.json`` files.
It never runs tracking, reconstruction or a video model.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "1.0.0"
ACCURATE_NAE = 0.25
DIRECTIONAL_CONCORDANCE = 0.50
ACCURATE_CONCORDANCE = 0.50
MINIMUM_ACCURATE_RESPONSE_SLOPE = 0.20


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _median(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _number(item)) is not None]
    return float(statistics.median(finite)) if finite else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _number(item)) is not None]
    return float(statistics.fmean(finite)) if finite else None


def _normalized_rmse(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _number(item)) is not None]
    if not finite:
        return None
    return math.sqrt(sum(value * value for value in finite) / len(finite))


def _quartiles(values: Iterable[Any]) -> tuple[float | None, float | None]:
    finite = [value for item in values if (value := _number(item)) is not None]
    if not finite:
        return None, None
    return _percentile(finite, 0.25), _percentile(finite, 0.75)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _job_id(row: Mapping[str, Any]) -> str:
    direct = str(row.get("job_id") or "").strip()
    if direct:
        return direct
    return Path(str(row.get("video_name") or "")).stem


def _result_path(root: Path, job_id: str) -> Path | None:
    candidates = (
        root / "jobs" / job_id / "result.json",
        root / job_id / "result.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def _registry_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for experiment in registry.get("experiments", []):
        if isinstance(experiment, Mapping) and experiment.get("id"):
            output[str(experiment["id"])] = dict(experiment)
    return output


def _anchor_index(spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(anchor["id"]): dict(anchor)
        for anchor in spec.get("anchor_tuples", [])
        if isinstance(anchor, Mapping) and anchor.get("id") is not None
    }


def _scene_group(scene_id: str) -> str:
    value = scene_id.lower()
    if value == "baseline":
        return "clean"
    if value.startswith("indoor"):
        return "indoor"
    if value.startswith("outdoor"):
        return "outdoor"
    return "other"


def _parameter_family(name: str) -> str:
    value = name.lower()
    if "gravity" in value:
        return "gravity"
    if "restitution" in value:
        return "restitution"
    if "friction" in value:
        return "friction"
    if "drag" in value:
        return "drag"
    if "damping" in value or "decay" in value:
        return "damping"
    if "magnetic" in value:
        return "magnetic"
    return name


def _nuisance_signature(
    spec: Mapping[str, Any],
    parameter_name: str,
    anchor: Mapping[str, Any],
) -> str:
    nuisance = {
        str(item["name"]): anchor.get(str(item["name"]))
        for item in spec.get("hidden_parameters", [])
        if isinstance(item, Mapping)
        and item.get("name")
        and str(item["name"]) != parameter_name
    }
    return json.dumps(
        nuisance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _registry_scan_design(
    registry: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, str], int],
    dict[tuple[str, str], str],
    dict[tuple[str, str], int],
]:
    """Freeze OAT branches from registry order, never from model outcomes."""

    branch_level_counts: dict[tuple[str, str, str], int] = {}
    canonical: dict[tuple[str, str], str] = {}
    parameter_counts: dict[tuple[str, str], int] = {}
    for spec in registry.get("experiments", []):
        if not isinstance(spec, Mapping) or not spec.get("id"):
            continue
        experiment_id = str(spec["id"])
        parameters = [
            str(item["name"])
            for item in spec.get("hidden_parameters", [])
            if isinstance(item, Mapping) and item.get("name")
        ]
        anchors = [
            anchor
            for anchor in spec.get("anchor_tuples", [])
            if isinstance(anchor, Mapping)
        ]
        for parameter_name in parameters:
            signatures_in_order: list[str] = []
            targets_by_signature: dict[str, set[float]] = defaultdict(set)
            for anchor in anchors:
                target = _number(anchor.get(parameter_name))
                if target is None:
                    continue
                signature = _nuisance_signature(spec, parameter_name, anchor)
                if signature not in targets_by_signature:
                    signatures_in_order.append(signature)
                targets_by_signature[signature].add(target)
            for signature in signatures_in_order:
                branch_level_counts[
                    (experiment_id, parameter_name, signature)
                ] = len(targets_by_signature[signature])
            if signatures_in_order:
                # max() keeps the first registry branch on an exact tie.
                canonical[(experiment_id, parameter_name)] = max(
                    signatures_in_order,
                    key=lambda signature: len(targets_by_signature[signature]),
                )
            parameter_counts[(experiment_id, parameter_name)] = len(parameters)
    return branch_level_counts, canonical, parameter_counts


def _walk_fit_series(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"observed", "predicted"}.issubset(value):
            yield value
            return
        fit_series = value.get("fit_series")
        if isinstance(fit_series, Mapping):
            yield from _walk_fit_series(fit_series)
        for key, child in value.items():
            if key == "fit_series":
                continue
            yield from _walk_fit_series(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_fit_series(child)


def _clean_series(value: Mapping[str, Any]) -> dict[str, Any] | None:
    observed = value.get("observed")
    predicted = value.get("predicted")
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
        return None
    if not isinstance(predicted, Sequence) or isinstance(predicted, (str, bytes)):
        return None
    times = value.get("time_s")
    if not isinstance(times, Sequence) or isinstance(times, (str, bytes)):
        times = list(range(min(len(observed), len(predicted))))
    cleaned: list[tuple[float, float, float]] = []
    for time_value, observed_value, predicted_value in zip(times, observed, predicted):
        time_s = _number(time_value)
        obs = _number(observed_value)
        pred = _number(predicted_value)
        if time_s is not None and obs is not None and pred is not None:
            cleaned.append((time_s, obs, pred))
    if not cleaned:
        return None
    return {
        "series_name": str(value.get("series_name") or "q"),
        "time_s": [item[0] for item in cleaned],
        "observed": [item[1] for item in cleaned],
        "predicted": [item[2] for item in cleaned],
    }


def _series_fingerprint(series: Mapping[str, Any]) -> str:
    compact = {
        "name": series["series_name"],
        "t": [round(float(value), 9) for value in series["time_s"]],
        "o": [round(float(value), 9) for value in series["observed"]],
        "p": [round(float(value), 9) for value in series["predicted"]],
    }
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def _legacy_fit_diagnostics(value: Any) -> dict[str, Any]:
    candidates: list[Mapping[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if any(key in item for key in ("fit_nrmse", "fit_rmse", "fit_r2")):
                candidates.append(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)

    visit(value)
    if not candidates:
        return {}
    best = max(candidates, key=lambda item: int(_number(item.get("fit_points")) or 0))
    return {
        "trajectory_rmse": _number(best.get("fit_rmse")),
        "trajectory_nrmse": _number(best.get("fit_nrmse")),
        "trajectory_r2": _number(best.get("fit_r2")),
        "fit_point_count": int(_number(best.get("fit_points")) or 0),
    }


def trajectory_metrics(fit: Mapping[str, Any]) -> dict[str, Any]:
    """Compute comparable equation-fit metrics from all canonical fit series."""

    diagnostics = fit.get("diagnostics", {})
    seen: set[str] = set()
    series: list[dict[str, Any]] = []
    malformed_series_count = 0
    for candidate in _walk_fit_series(diagnostics):
        observed = candidate.get("observed")
        predicted = candidate.get("predicted")
        times = candidate.get("time_s")
        if (
            not isinstance(observed, Sequence)
            or isinstance(observed, (str, bytes))
            or not isinstance(predicted, Sequence)
            or isinstance(predicted, (str, bytes))
            or len(observed) != len(predicted)
            or (
                isinstance(times, Sequence)
                and not isinstance(times, (str, bytes))
                and len(times) != len(observed)
            )
        ):
            malformed_series_count += 1
            continue
        cleaned = _clean_series(candidate)
        if cleaned is None:
            continue
        fingerprint = _series_fingerprint(cleaned)
        if fingerprint not in seen:
            seen.add(fingerprint)
            series.append(cleaned)
    if not series:
        legacy = _legacy_fit_diagnostics(diagnostics)
        metric_points = _number(
            fit.get("fit_input", {}).get("metric_point_count")
            if isinstance(fit.get("fit_input"), Mapping)
            else None
        )
        count = int(legacy.get("fit_point_count") or 0)
        legacy.update(
            {
                "trajectory_mae": None,
                "trajectory_nmae": None,
                "trajectory_pearson_r": None,
                "fit_series_count": 0,
                "fit_coverage": (
                    min(1.0, count / metric_points)
                    if metric_points and count
                    else None
                ),
                "fit_start_s": None,
                "fit_end_s": None,
                "trajectory_metric_source": (
                    "legacy_best_segment" if legacy else "missing"
                ),
                "malformed_fit_series_count": malformed_series_count,
            }
        )
        return legacy

    native_squared: list[float] = []
    native_absolute: list[float] = []
    normalized_squared: list[float] = []
    normalized_absolute: list[float] = []
    normalized_centered_squared: list[float] = []
    correlations: list[tuple[int, float]] = []
    times: set[float] = set()
    for item in series:
        observed = [float(value) for value in item["observed"]]
        predicted = [float(value) for value in item["predicted"]]
        residual = [obs - pred for obs, pred in zip(observed, predicted)]
        native_squared.extend(value * value for value in residual)
        native_absolute.extend(abs(value) for value in residual)
        robust_range = _percentile(observed, 0.95) - _percentile(observed, 0.05)
        if robust_range <= 1e-12:
            robust_range = max(observed) - min(observed)
        if robust_range > 1e-12:
            normalized_squared.extend((value / robust_range) ** 2 for value in residual)
            normalized_absolute.extend(abs(value) / robust_range for value in residual)
            center = statistics.fmean(observed)
            normalized_centered_squared.extend(
                ((value - center) / robust_range) ** 2 for value in observed
            )
        if len(observed) >= 2:
            mean_observed = statistics.fmean(observed)
            mean_predicted = statistics.fmean(predicted)
            covariance = sum(
                (obs - mean_observed) * (pred - mean_predicted)
                for obs, pred in zip(observed, predicted)
            )
            denominator = math.sqrt(
                sum((obs - mean_observed) ** 2 for obs in observed)
                * sum((pred - mean_predicted) ** 2 for pred in predicted)
            )
            if denominator > 1e-12:
                correlations.append((len(observed), covariance / denominator))
        times.update(round(float(value), 9) for value in item["time_s"])

    point_count = len(native_squared)
    rmse = math.sqrt(sum(native_squared) / point_count) if point_count else None
    mae = statistics.fmean(native_absolute) if native_absolute else None
    nrmse = (
        math.sqrt(sum(normalized_squared) / len(normalized_squared))
        if normalized_squared
        else None
    )
    nmae = statistics.fmean(normalized_absolute) if normalized_absolute else None
    r2 = (
        1.0 - sum(normalized_squared) / sum(normalized_centered_squared)
        if normalized_squared and sum(normalized_centered_squared) > 1e-12
        else None
    )
    pearson = (
        sum(count * value for count, value in correlations)
        / sum(count for count, _ in correlations)
        if correlations
        else None
    )
    metric_points = _number(
        fit.get("fit_input", {}).get("metric_point_count")
        if isinstance(fit.get("fit_input"), Mapping)
        else None
    )
    all_times = [float(value) for item in series for value in item["time_s"]]
    return {
        "trajectory_rmse": rmse if len(series) == 1 else None,
        "trajectory_mae": mae if len(series) == 1 else None,
        "trajectory_nrmse": nrmse,
        "trajectory_nmae": nmae,
        "trajectory_r2": r2,
        "trajectory_pearson_r": pearson,
        "fit_point_count": point_count,
        "fit_series_count": len(series),
        "fit_coverage": (
            min(1.0, len(times) / metric_points)
            if metric_points and times
            else None
        ),
        "fit_start_s": min(all_times) if all_times else None,
        "fit_end_s": max(all_times) if all_times else None,
        "trajectory_metric_source": "canonical_fit_series",
        "malformed_fit_series_count": malformed_series_count,
        "trajectory_native_metric_note": (
            "single_observable_native_units"
            if len(series) == 1
            else "native_rmse_not_pooled_across_multiple_observables"
        ),
    }


def _motion_interval(fit: Mapping[str, Any]) -> dict[str, Any]:
    segmentation = fit.get("segmentation", {})
    frames: list[int] = []
    compact_indices: list[int] = []
    if isinstance(segmentation, Mapping):
        for collection_name in ("segments", "cycles", "events"):
            collection = segmentation.get(collection_name)
            if not isinstance(collection, Sequence):
                continue
            for item in collection:
                if not isinstance(item, Mapping):
                    continue
                for key in (
                    "source_frame",
                    "start_source_frame",
                    "middle_source_frame",
                    "end_source_frame",
                ):
                    value = _number(item.get(key))
                    if value is not None:
                        frames.append(int(value))
                for key in ("index", "start_index", "middle_index", "end_index"):
                    value = _number(item.get(key))
                    if value is not None:
                        compact_indices.append(int(value))
    fit_input = fit.get("fit_input", {})
    if isinstance(fit_input, Mapping):
        source = fit_input.get("source_frame_indices")
        if isinstance(source, Sequence):
            source_frames = [
                int(value) for item in source if (value := _number(item)) is not None
            ]
            if not frames and compact_indices:
                frames.extend(
                    source_frames[index]
                    for index in compact_indices
                    if 0 <= index < len(source_frames)
                )
            if not frames:
                frames.extend(source_frames)
    return {
        "motion_start_frame": min(frames) if frames else None,
        "motion_end_frame": max(frames) if frames else None,
        "motion_frame_span": max(frames) - min(frames) + 1 if frames else None,
    }


def _parameter_estimates(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    parameter_name: str,
) -> tuple[float | None, float | None, str]:
    fit = result.get("fit", {}) if isinstance(result.get("fit"), Mapping) else {}
    raw = fit.get("raw_parameter_estimates", {})
    accepted = fit.get("parameter_estimates", {})
    raw_value = _number(raw.get(parameter_name)) if isinstance(raw, Mapping) else None
    observed = fit.get("parameter_observed", {})
    attribution = fit.get("parameter_attribution", {})
    accepted_candidate = (
        _number(accepted.get(parameter_name)) if isinstance(accepted, Mapping) else None
    )
    observed_flag = (
        bool(observed.get(parameter_name))
        if isinstance(observed, Mapping) and parameter_name in observed
        else accepted_candidate is not None
    )
    attribution_item = (
        attribution.get(parameter_name, {}) if isinstance(attribution, Mapping) else {}
    )
    attribution_status = (
        str(attribution_item.get("status") or "").lower()
        if isinstance(attribution_item, Mapping)
        else ""
    )
    accepted_value = (
        accepted_candidate
        if observed_flag and attribution_status not in {"fail", "rejected", "indeterminate"}
        else None
    )
    if raw_value is not None:
        return raw_value, accepted_value, "result.fit.raw_parameter_estimates"
    flattened_raw = _number(row.get(f"{parameter_name}__raw_estimate"))
    if flattened_raw is not None:
        return flattened_raw, _number(row.get(f"{parameter_name}__estimate")), "all_jobs.raw"
    raw_json = row.get("raw_parameter_estimates_json")
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            raw_mapping = json.loads(raw_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_mapping = {}
        if isinstance(raw_mapping, Mapping):
            raw_json_value = _number(raw_mapping.get(parameter_name))
            if raw_json_value is not None:
                return raw_json_value, accepted_value, "all_jobs.raw_json"
    diagnostics = fit.get("diagnostics", {})
    candidates = (
        diagnostics.get("candidate_parameter_estimates", {})
        if isinstance(diagnostics, Mapping)
        else {}
    )
    if isinstance(candidates, Mapping):
        candidate_value = _number(candidates.get(parameter_name))
        if candidate_value is not None:
            return candidate_value, accepted_value, "result.fit.candidate_parameter_estimates"
    if accepted_value is not None:
        return accepted_value, accepted_value, "result.fit.parameter_estimates_legacy_or_accepted"
    flattened = _number(row.get(f"{parameter_name}__estimate"))
    if flattened is not None:
        return flattened, flattened, "all_jobs.accepted"
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), Mapping) else {}
    parameters = metrics.get("parameters", {}) if isinstance(metrics, Mapping) else {}
    item = parameters.get(parameter_name, {}) if isinstance(parameters, Mapping) else {}
    metric_estimate = _number(item.get("estimate_raw")) if isinstance(item, Mapping) else None
    return metric_estimate, metric_estimate, "result.metrics" if metric_estimate is not None else "missing"


def _artifact_path(result: Mapping[str, Any], key: str) -> str | None:
    if key == "trajectory_plot":
        visuals = result.get("visual_evidence", {})
        value = visuals.get(key) if isinstance(visuals, Mapping) else None
    elif key == "video_path":
        job = result.get("job", {})
        value = job.get(key) if isinstance(job, Mapping) else None
    else:
        value = result.get(key)
    return str(value) if value else None


def load_parameter_rows(
    model_name: str,
    evaluation_root: Path,
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load one long-format row per video and hidden parameter."""

    evaluation_root = evaluation_root.resolve()
    all_jobs = evaluation_root if evaluation_root.is_file() else evaluation_root / "all_jobs.csv"
    if not all_jobs.is_file():
        raise FileNotFoundError(f"missing all_jobs.csv for {model_name}: {all_jobs}")
    root = all_jobs.parent
    registry_by_id = _registry_index(registry)
    output: list[dict[str, Any]] = []
    for source in _read_csv(all_jobs):
        experiment_id = str(source.get("experiment_id") or "")
        if experiment_id not in registry_by_id:
            continue
        spec = registry_by_id[experiment_id]
        tuple_id = str(source.get("parameter_tuple_id") or "")
        anchor = _anchor_index(spec).get(tuple_id)
        if anchor is None:
            continue
        job_id = _job_id(source)
        result_path = _result_path(root, job_id)
        result = _read_json(result_path) if result_path is not None else {}
        job = result.get("job", {}) if isinstance(result.get("job"), Mapping) else {}
        scene_id = str(job.get("scene_id") or source.get("scene_id") or "")
        camera_name = str(job.get("camera_name") or source.get("camera_name") or "")
        seed = str(job.get("seed") or source.get("seed") or "")
        fit = result.get("fit", {}) if isinstance(result.get("fit"), Mapping) else {}
        trajectory = trajectory_metrics(fit)
        interval = _motion_interval(fit)
        parameters = [
            dict(item)
            for item in spec.get("hidden_parameters", [])
            if isinstance(item, Mapping) and item.get("name")
        ]
        for parameter in parameters:
            parameter_name = str(parameter["name"])
            result_metrics = (
                result.get("metrics", {}) if isinstance(result.get("metrics"), Mapping) else {}
            )
            result_parameters = (
                result_metrics.get("parameters", {})
                if isinstance(result_metrics.get("parameters"), Mapping)
                else {}
            )
            metric_parameter = (
                result_parameters.get(parameter_name, {})
                if isinstance(result_parameters.get(parameter_name), Mapping)
                else {}
            )
            metric_target = (
                _number(metric_parameter.get("gt"))
                if result.get("target_lookup_performed") is True
                else None
            )
            row_target = _number(source.get(f"{parameter_name}__gt"))
            target = (
                metric_target
                if metric_target is not None
                else row_target
                if row_target is not None
                else _number(anchor.get(parameter_name))
            )
            target_source = (
                "result.metrics"
                if metric_target is not None
                else "all_jobs"
                if row_target is not None
                else "registry_post_fit_lookup"
            )
            if target is None:
                continue
            valid_range = parameter.get("valid_range", [])
            if not isinstance(valid_range, Sequence) or len(valid_range) != 2:
                continue
            lower, upper = (_number(value) for value in valid_range)
            if lower is None or upper is None or upper <= lower:
                continue
            candidate_estimate, accepted_estimate, estimate_source = _parameter_estimates(
                source, result, parameter_name
            )
            attribution_mapping = (
                fit.get("parameter_attribution", {})
                if isinstance(fit.get("parameter_attribution"), Mapping)
                else {}
            )
            attribution_item = (
                attribution_mapping.get(parameter_name, {})
                if isinstance(attribution_mapping.get(parameter_name), Mapping)
                else {}
            )
            attribution_status = (
                str(attribution_item.get("status") or "").lower() or None
            )
            attribution_reason_codes = attribution_item.get("reason_codes", [])
            if not isinstance(attribution_reason_codes, Sequence) or isinstance(
                attribution_reason_codes, (str, bytes, bytearray)
            ):
                attribution_reason_codes = []
            estimate = accepted_estimate
            absolute_error = abs(estimate - target) if estimate is not None else None
            relative_error = (
                absolute_error / abs(target)
                if absolute_error is not None and abs(target) > 1e-12
                else None
            )
            nae = absolute_error / (upper - lower) if absolute_error is not None else None
            candidate_absolute_error = (
                abs(candidate_estimate - target)
                if candidate_estimate is not None
                else None
            )
            candidate_nae = (
                candidate_absolute_error / (upper - lower)
                if candidate_absolute_error is not None
                else None
            )
            nuisance_signature = _nuisance_signature(spec, parameter_name, anchor)
            fit_status = str(fit.get("status") or source.get("fit_status") or "")
            nrmse = trajectory.get("trajectory_nrmse")
            r2 = trajectory.get("trajectory_r2")
            if candidate_estimate is None:
                simple_status = "UNAVAILABLE"
            elif accepted_estimate is None:
                simple_status = "CANDIDATE_REJECTED"
            elif fit_status == "model_mismatch" or (
                nrmse is not None and float(nrmse) > 0.35
            ) or (r2 is not None and float(r2) < 0.20):
                simple_status = "TRAJECTORY_MISMATCH"
            elif nae is not None and nae <= ACCURATE_NAE:
                simple_status = "NUMERICALLY_ACCURATE"
            else:
                simple_status = "PARAMETER_INACCURATE"
            output.append(
                {
                    "model": model_name,
                    "job_id": job_id,
                    "video_name": str(job.get("video_name") or source.get("video_name") or ""),
                    "experiment_id": experiment_id,
                    "design_tier": experiment_id.split("_", 1)[0],
                    "parameter_count": len(parameters),
                    "parameter_tuple_id": tuple_id,
                    "parameter_name": parameter_name,
                    "parameter_family": _parameter_family(parameter_name),
                    "unit": parameter.get("unit"),
                    "scene_id": scene_id,
                    "scene_group": _scene_group(scene_id),
                    "camera_name": camera_name,
                    "seed": seed,
                    "target": target,
                    "target_source": target_source,
                    "estimate": estimate,
                    "candidate_estimate": candidate_estimate,
                    "accepted_estimate": accepted_estimate,
                    "candidate_estimate_source": estimate_source,
                    "valid_min": lower,
                    "valid_max": upper,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "range_normalized_absolute_error": nae,
                    "candidate_absolute_error": candidate_absolute_error,
                    "candidate_range_normalized_absolute_error": candidate_nae,
                    "parameter_score_0_100": (
                        100.0 * max(0.0, 1.0 - nae) if nae is not None else None
                    ),
                    "nuisance_signature": nuisance_signature,
                    "fit_status": fit_status or None,
                    "parameter_attribution_status": attribution_status,
                    "parameter_attribution_reason_codes": [
                        str(code) for code in attribution_reason_codes
                    ],
                    "rule_family_status": (
                        fit.get("rule_family_evaluation", {}).get("status")
                        if isinstance(fit.get("rule_family_evaluation"), Mapping)
                        else source.get("rule_family_status")
                    ),
                    "generation_validity_status": (
                        result.get("video_generation_validity", {}).get("status")
                        if isinstance(result.get("video_generation_validity"), Mapping)
                        else source.get("generation_validity_status")
                    ),
                    "simple_status": simple_status,
                    **trajectory,
                    **interval,
                    "result_json": str(result_path) if result_path is not None else None,
                    "trajectory_csv": _artifact_path(result, "trajectory_csv"),
                    "trajectory_plot": _artifact_path(result, "trajectory_plot"),
                    "video_path": _artifact_path(result, "video_path"),
                    "registry_sha256": (
                        result.get("evaluation_lineage", {}).get("registry_sha256")
                        if isinstance(result.get("evaluation_lineage"), Mapping)
                        else None
                    ),
                    "evaluator_source_sha256": (
                        result.get("evaluation_lineage", {}).get("evaluator_source_sha256")
                        if isinstance(result.get("evaluation_lineage"), Mapping)
                        else None
                    ),
                    "physics_fitter_source_sha256": (
                        result.get("evaluation_lineage", {}).get("physics_fitter_source_sha256")
                        if isinstance(result.get("evaluation_lineage"), Mapping)
                        else None
                    ),
                }
            )
    return output


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for position in range(cursor, end):
            output[ordered[position]] = rank
        cursor = end
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - mean_left) ** 2 for value in left)
        * sum((value - mean_right) ** 2 for value in right)
    )
    return numerator / denominator if denominator > 1e-12 else None


def _response_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one OAT channel after taking the median across seeds per level."""

    by_target: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        target = _number(row.get("target"))
        if target is not None:
            by_target[target].append(row)
    seed_sets: list[set[str]] = []
    level_seed_counts: dict[str, int] = {}
    for target, target_rows in sorted(by_target.items()):
        seeds = {
            str(row.get("seed") or "")
            for row in target_rows
            if _number(row.get("estimate")) is not None
        }
        seed_sets.append(seeds)
        level_seed_counts[str(target)] = len(seeds)
    common_seeds = set.intersection(*seed_sets) if seed_sets else set()
    levels: list[dict[str, Any]] = []
    level_seed_iqrs: list[float] = []
    level_seed_mads: list[float] = []
    for target, target_rows in sorted(by_target.items()):
        paired_rows = [
            row
            for row in target_rows
            if str(row.get("seed") or "") in common_seeds
            and _number(row.get("estimate")) is not None
        ]
        estimate_values = [
            float(row["estimate"]) for row in paired_rows
            if _number(row.get("estimate")) is not None
        ]
        estimate = _median(estimate_values)
        if estimate is None:
            continue
        if len(estimate_values) >= 2:
            q25, q75 = _quartiles(estimate_values)
            if q25 is not None and q75 is not None:
                level_seed_iqrs.append(q75 - q25)
            center = float(statistics.median(estimate_values))
            level_seed_mads.append(
                float(statistics.median(abs(value - center) for value in estimate_values))
            )
        width = _median(
            float(row["valid_max"]) - float(row["valid_min"]) for row in paired_rows
        )
        levels.append(
            {
                "target": target,
                "estimate": estimate,
                "range_nae": (
                    abs(estimate - target) / width
                    if width is not None and width > 0
                    else None
                ),
                "trajectory_nrmse": _median(
                    row.get("trajectory_nrmse") for row in paired_rows
                ),
                "trajectory_r2": _median(
                    row.get("trajectory_r2") for row in paired_rows
                ),
                "fit_coverage": _median(row.get("fit_coverage") for row in paired_rows),
            }
        )
    targets = [float(row["target"]) for row in levels]
    estimates = [float(row["estimate"]) for row in levels]
    concordant = 0
    comparable = 0
    slopes: list[float] = []
    for left in range(len(levels)):
        for right in range(left + 1, len(levels)):
            target_delta = targets[right] - targets[left]
            if abs(target_delta) <= 1e-12:
                continue
            estimate_delta = estimates[right] - estimates[left]
            comparable += 1
            if target_delta * estimate_delta > 0:
                concordant += 1
            slopes.append(estimate_delta / target_delta)
    direction = concordant / comparable if comparable else None
    spearman = _pearson(_ranks(targets), _ranks(estimates)) if len(levels) >= 2 else None
    median_nae = _median(row.get("range_nae") for row in levels)
    fit_gate_pass_fraction = 1.0 if levels else None
    if len({float(row["target"]) for row in levels}) < 2:
        grade = "INSUFFICIENT"
    elif (
        direction is not None
        and direction > ACCURATE_CONCORDANCE
        and _median(slopes) is not None
        and float(_median(slopes)) >= MINIMUM_ACCURATE_RESPONSE_SLOPE
        and median_nae is not None
        and median_nae <= ACCURATE_NAE
        and fit_gate_pass_fraction is not None
        and fit_gate_pass_fraction >= 0.50
    ):
        grade = "ACCURATE"
    elif (
        direction is not None
        and direction > DIRECTIONAL_CONCORDANCE
        and _median(slopes) is not None
        and float(_median(slopes)) > 0.0
    ):
        grade = "DIRECTIONAL_ONLY"
    else:
        grade = "NO_PARAMETER_RESPONSE"
    return {
        "expected_level_count": len(by_target),
        "available_level_count": len(levels),
        "seed_count": len({str(row.get("seed")) for row in rows}),
        "common_seed_count": len(common_seeds),
        "common_seeds": sorted(common_seeds),
        "level_seed_counts": level_seed_counts,
        "median_seed_iqr": _median(level_seed_iqrs),
        "median_seed_mad": _median(level_seed_mads),
        "pairwise_direction_accuracy": direction,
        "spearman_rho": spearman,
        "theil_sen_response_slope": _median(slopes),
        "median_range_nae": median_nae,
        "normalized_parameter_rmse": _normalized_rmse(
            row.get("range_nae") for row in levels
        ),
        "median_trajectory_nrmse": _median(
            row.get("trajectory_nrmse") for row in levels
        ),
        "median_trajectory_r2": _median(row.get("trajectory_r2") for row in levels),
        "median_fit_coverage": _median(row.get("fit_coverage") for row in levels),
        "fit_gate_pass_fraction": fit_gate_pass_fraction,
        "response_grade": grade,
    }


def build_scan_rows(
    parameter_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any] | None = None,
    *,
    model_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    branch_counts, canonical_signatures, parameter_counts = (
        _registry_scan_design(registry) if registry is not None else ({}, {}, {})
    )
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in parameter_rows:
        key = (
            str(row["model"]),
            str(row["experiment_id"]),
            str(row["parameter_name"]),
            str(row["scene_id"]),
            str(row["camera_name"]),
            str(row["nuisance_signature"]),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        model, experiment, parameter, scene, camera, nuisance = key
        response = _response_metrics(rows)
        expected_from_registry = branch_counts.get((experiment, parameter, nuisance))
        if expected_from_registry is not None:
            response["expected_level_count"] = expected_from_registry
        output.append(
            {
                "model": model,
                "experiment_id": experiment,
                "parameter_name": parameter,
                "parameter_family": _parameter_family(parameter),
                "scene_id": scene,
                "scene_group": _scene_group(scene),
                "camera_name": camera,
                "nuisance_signature": nuisance,
                "parameter_count": int(rows[0]["parameter_count"]),
                "canonical_scan": (
                    canonical_signatures.get((experiment, parameter)) == nuisance
                    if registry is not None
                    else False
                ),
                "canonical_selection_source": (
                    "frozen_registry_design" if registry is not None else "legacy_fallback"
                ),
                **response,
            }
        )
    if registry is None:
        channels: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in output:
            channels[
                (
                    str(row["model"]),
                    str(row["experiment_id"]),
                    str(row["parameter_name"]),
                    str(row["scene_id"]),
                    str(row["camera_name"]),
                )
            ].append(row)
        for rows in channels.values():
            max(
                rows,
                key=lambda row: (
                    int(row["expected_level_count"]),
                    str(row["nuisance_signature"]),
                ),
            )["canonical_scan"] = True
    else:
        existing_primary = {
            (
                str(row["model"]),
                str(row["experiment_id"]),
                str(row["parameter_name"]),
            )
            for row in output
            if row["scene_id"] == "baseline"
            and row["camera_name"] == "CAM_Side"
            and row.get("canonical_scan") is True
        }
        empty_response = _response_metrics([])
        for model in model_names or sorted({str(row["model"]) for row in parameter_rows}):
            for (experiment, parameter), nuisance in canonical_signatures.items():
                if (str(model), experiment, parameter) in existing_primary:
                    continue
                output.append(
                    {
                        "model": str(model),
                        "experiment_id": experiment,
                        "parameter_name": parameter,
                        "parameter_family": _parameter_family(parameter),
                        "scene_id": "baseline",
                        "scene_group": "clean",
                        "camera_name": "CAM_Side",
                        "nuisance_signature": nuisance,
                        "parameter_count": parameter_counts[(experiment, parameter)],
                        "canonical_scan": True,
                        "canonical_selection_source": "frozen_registry_design_missing_observation",
                        **{
                            **empty_response,
                            "expected_level_count": branch_counts[
                                (experiment, parameter, nuisance)
                            ],
                        },
                    }
                )
    return output


def build_model_summary(
    parameter_rows: Sequence[Mapping[str, Any]],
    scan_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    models = sorted({str(row["model"]) for row in parameter_rows})
    output: list[dict[str, Any]] = []
    for model in models:
        primary = [
            row for row in parameter_rows
            if row["model"] == model
            and row["scene_id"] == "baseline"
            and row["camera_name"] == "CAM_Side"
        ]
        available = [row for row in primary if _number(row.get("estimate")) is not None]
        job_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in primary:
            job_groups[str(row["job_id"])].append(row)
        jobs = {
            job_id: rows[0] for job_id, rows in job_groups.items()
        }
        scans = [
            row for row in scan_rows
            if row["model"] == model
            and row["scene_id"] == "baseline"
            and row["camera_name"] == "CAM_Side"
            and row.get("canonical_scan") is True
        ]
        counts: dict[str, int] = defaultdict(int)
        for row in scans:
            counts[str(row["response_grade"])] += 1
        usable_jobs = [
            rows
            for rows in job_groups.values()
            if any(_number(row.get("estimate")) is not None for row in rows)
            and not any(
                row.get("simple_status") == "TRAJECTORY_MISMATCH"
                for row in rows
            )
        ]
        evaluable_scans = [
            row for row in scans if row["response_grade"] != "INSUFFICIENT"
        ]
        responsive_scans = [
            row for row in scans
            if row["response_grade"] in {"ACCURATE", "DIRECTIONAL_ONLY"}
        ]
        scans_by_experiment: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in scans:
            scans_by_experiment[str(row["experiment_id"])].append(row)
        experiment_naes = [
            value
            for rows in scans_by_experiment.values()
            if (value := _median(row.get("median_range_nae") for row in rows))
            is not None
        ]
        experiment_trajectory_nrmse = [
            value
            for rows in scans_by_experiment.values()
            if (value := _median(row.get("median_trajectory_nrmse") for row in rows))
            is not None
        ]
        experiment_trajectory_r2 = [
            value
            for rows in scans_by_experiment.values()
            if (value := _median(row.get("median_trajectory_r2") for row in rows))
            is not None
        ]
        experiment_fit_coverage = [
            value
            for rows in scans_by_experiment.values()
            if (value := _median(row.get("median_fit_coverage") for row in rows))
            is not None
        ]
        output.append(
            {
                "model": model,
                "primary_job_count": len(jobs),
                "valid_motion_job_count": len(usable_jobs),
                "valid_motion_rate": len(usable_jobs) / len(jobs) if jobs else None,
                "primary_parameter_observations": len(primary),
                "available_parameter_observations": len(available),
                "parameter_coverage": len(available) / len(primary) if primary else None,
                "mean_range_nae": _mean(
                    experiment_naes
                ),
                "median_range_nae": _median(
                    experiment_naes
                ),
                "normalized_parameter_rmse": _normalized_rmse(
                    experiment_naes
                ),
                "within_25pct_range_fraction": (
                    sum(
                        float(value) <= ACCURATE_NAE
                        for value in experiment_naes
                    )
                    / len(experiment_naes)
                    if experiment_naes
                    else None
                ),
                "median_trajectory_nrmse": _median(experiment_trajectory_nrmse),
                "median_trajectory_nmae": _median(
                    row.get("trajectory_nmae") for row in jobs.values()
                ),
                "median_trajectory_r2": _median(experiment_trajectory_r2),
                "median_fit_coverage": _median(experiment_fit_coverage),
                "macro_aggregation": "equal_weight_over_13_experiments",
                "primary_scan_count": len(scans),
                "evaluable_scan_count": len(evaluable_scans),
                "directional_response_rate_over_all_channels": (
                    len(responsive_scans) / len(scans) if scans else None
                ),
                "accurate_scan_rate_over_all_channels": (
                    counts["ACCURATE"] / len(scans) if scans else None
                ),
                "accurate_scan_count": counts["ACCURATE"],
                "directional_only_scan_count": counts["DIRECTIONAL_ONLY"],
                "no_response_scan_count": counts["NO_PARAMETER_RESPONSE"],
                "insufficient_scan_count": counts["INSUFFICIENT"],
            }
        )
    return output


def build_experiment_summary(
    scan_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Macro summary over the 13 systems, retaining each parameter grade."""

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scan_rows:
        if (
            row["scene_id"] == "baseline"
            and row["camera_name"] == "CAM_Side"
            and row.get("canonical_scan") is True
        ):
            groups[(str(row["model"]), str(row["experiment_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, experiment), rows in sorted(groups.items()):
        evaluable = [row for row in rows if row["response_grade"] != "INSUFFICIENT"]
        responsive = [
            row
            for row in rows
            if row["response_grade"] in {"DIRECTIONAL_ONLY", "ACCURATE"}
        ]
        output.append(
            {
                "model": model,
                "experiment_id": experiment,
                "parameter_channel_count": len(rows),
                "evaluable_channel_count": len(evaluable),
                "directional_or_accurate_channel_count": len(responsive),
                "directional_response_rate": (
                    len(responsive) / len(rows) if rows else None
                ),
                "accurate_channel_count": sum(
                    row["response_grade"] == "ACCURATE" for row in rows
                ),
                "median_range_nae": _median(
                    row.get("median_range_nae") for row in rows
                ),
                "median_trajectory_r2": _median(
                    row.get("median_trajectory_r2") for row in rows
                ),
                "median_trajectory_nrmse": _median(
                    row.get("median_trajectory_nrmse") for row in rows
                ),
                "parameter_grade_vector": "; ".join(
                    f'{row["parameter_name"]}={row["response_grade"]}'
                    for row in sorted(rows, key=lambda item: str(item["parameter_name"]))
                ),
            }
        )
    return output


def build_background_comparison(
    parameter_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in parameter_rows:
        if row["scene_id"] == "baseline" and row["camera_name"] == "CAM_Side":
            key = (
                str(row["model"]),
                str(row["experiment_id"]),
                str(row["parameter_tuple_id"]),
                str(row["parameter_name"]),
                str(row["seed"]),
            )
            baseline[key] = row
    pairs: list[dict[str, Any]] = []
    for row in parameter_rows:
        if row["scene_id"] == "baseline" or row["camera_name"] != "CAM_Side":
            continue
        key = (
            str(row["model"]),
            str(row["experiment_id"]),
            str(row["parameter_tuple_id"]),
            str(row["parameter_name"]),
            str(row["seed"]),
        )
        clean = baseline.get(key)
        if clean is None:
            continue
        clean_estimate = _number(clean.get("estimate"))
        scene_estimate = _number(row.get("estimate"))
        width = float(row["valid_max"]) - float(row["valid_min"])
        shift = (
            abs(scene_estimate - clean_estimate) / width
            if clean_estimate is not None and scene_estimate is not None and width > 0
            else None
        )
        pairs.append(
            {
                "model": row["model"],
                "experiment_id": row["experiment_id"],
                "parameter_tuple_id": row["parameter_tuple_id"],
                "parameter_name": row["parameter_name"],
                "seed": row["seed"],
                "scene_id": row["scene_id"],
                "scene_group": row["scene_group"],
                "baseline_job_id": clean["job_id"],
                "scene_job_id": row["job_id"],
                "target": row["target"],
                "baseline_estimate": clean_estimate,
                "scene_estimate": scene_estimate,
                "matched_parameter_shift_norm": shift,
                "baseline_range_nae": clean.get("range_normalized_absolute_error"),
                "scene_range_nae": row.get("range_normalized_absolute_error"),
                "delta_range_nae": (
                    float(row["range_normalized_absolute_error"])
                    - float(clean["range_normalized_absolute_error"])
                    if _number(row.get("range_normalized_absolute_error")) is not None
                    and _number(clean.get("range_normalized_absolute_error")) is not None
                    else None
                ),
            }
        )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[(str(row["model"]), str(row["scene_group"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (model, scene_group), rows in sorted(groups.items()):
        available = [
            row for row in rows
            if _number(row.get("matched_parameter_shift_norm")) is not None
        ]
        channel_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            channel_groups[
                (str(row["experiment_id"]), str(row["parameter_name"]))
            ].append(row)
        channel_rows = [
            {
                "shift": _median(
                    row.get("matched_parameter_shift_norm") for row in items
                ),
                "baseline_nae": _median(row.get("baseline_range_nae") for row in items),
                "scene_nae": _median(row.get("scene_range_nae") for row in items),
                "delta_nae": _median(row.get("delta_range_nae") for row in items),
            }
            for items in channel_groups.values()
        ]
        available_channels = [
            row for row in channel_rows if _number(row.get("shift")) is not None
        ]
        shift_q25, shift_q75 = _quartiles(
            row.get("shift") for row in available_channels
        )
        summary.append(
            {
                "model": model,
                "scene_group": scene_group,
                "matched_pair_count": len(rows),
                "available_pair_count": len(available),
                "pair_coverage": len(available) / len(rows) if rows else None,
                "parameter_channel_count": len(channel_rows),
                "available_parameter_channel_count": len(available_channels),
                "channel_coverage": (
                    len(available_channels) / len(channel_rows)
                    if channel_rows
                    else None
                ),
                "baseline_median_range_nae": _median(
                    row.get("baseline_nae") for row in available_channels
                ),
                "scene_median_range_nae": _median(
                    row.get("scene_nae") for row in available_channels
                ),
                "median_delta_range_nae": _median(
                    row.get("delta_nae") for row in available_channels
                ),
                "median_matched_parameter_shift_norm": _median(
                    row.get("shift") for row in available_channels
                ),
                "matched_parameter_shift_q25": shift_q25,
                "matched_parameter_shift_q75": shift_q75,
                "within_25pct_consistency_fraction": (
                    sum(
                        float(row["shift"]) <= ACCURATE_NAE
                        for row in available_channels
                    )
                    / len(available_channels)
                    if available_channels
                    else None
                ),
            }
        )
    return pairs, summary


def build_view_comparison(
    parameter_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    side: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in parameter_rows:
        if row["camera_name"] == "CAM_Side":
            key = (
                str(row["model"]),
                str(row["experiment_id"]),
                str(row["parameter_tuple_id"]),
                str(row["parameter_name"]),
                str(row["scene_id"]),
                str(row["seed"]),
            )
            side[key] = row
    pairs: list[dict[str, Any]] = []
    for row in parameter_rows:
        if row["camera_name"] not in {"CAM_Main", "CAM_Top"}:
            continue
        key = (
            str(row["model"]),
            str(row["experiment_id"]),
            str(row["parameter_tuple_id"]),
            str(row["parameter_name"]),
            str(row["scene_id"]),
            str(row["seed"]),
        )
        reference = side.get(key)
        if reference is None:
            continue
        side_estimate = _number(reference.get("estimate"))
        view_estimate = _number(row.get("estimate"))
        width = float(row["valid_max"]) - float(row["valid_min"])
        shift = (
            abs(view_estimate - side_estimate) / width
            if side_estimate is not None and view_estimate is not None and width > 0
            else None
        )
        pairs.append(
            {
                "model": row["model"],
                "experiment_id": row["experiment_id"],
                "parameter_tuple_id": row["parameter_tuple_id"],
                "parameter_name": row["parameter_name"],
                "scene_id": row["scene_id"],
                "seed": row["seed"],
                "camera_name": row["camera_name"],
                "side_job_id": reference["job_id"],
                "view_job_id": row["job_id"],
                "target": row["target"],
                "side_estimate": side_estimate,
                "view_estimate": view_estimate,
                "matched_parameter_shift_norm": shift,
                "side_range_nae": reference.get("range_normalized_absolute_error"),
                "view_range_nae": row.get("range_normalized_absolute_error"),
            }
        )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        groups[(str(row["model"]), str(row["camera_name"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (model, camera), rows in sorted(groups.items()):
        available = [
            row for row in rows
            if _number(row.get("matched_parameter_shift_norm")) is not None
        ]
        channel_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            channel_groups[
                (str(row["experiment_id"]), str(row["parameter_name"]))
            ].append(row)
        channel_rows = [
            {
                "shift": _median(
                    row.get("matched_parameter_shift_norm") for row in items
                ),
                "side_nae": _median(row.get("side_range_nae") for row in items),
                "view_nae": _median(row.get("view_range_nae") for row in items),
            }
            for items in channel_groups.values()
        ]
        available_channels = [
            row for row in channel_rows if _number(row.get("shift")) is not None
        ]
        shift_q25, shift_q75 = _quartiles(
            row.get("shift") for row in available_channels
        )
        summary.append(
            {
                "model": model,
                "camera_name": camera,
                "matched_pair_count": len(rows),
                "available_pair_count": len(available),
                "pair_coverage": len(available) / len(rows) if rows else None,
                "parameter_channel_count": len(channel_rows),
                "available_parameter_channel_count": len(available_channels),
                "channel_coverage": (
                    len(available_channels) / len(channel_rows)
                    if channel_rows
                    else None
                ),
                "side_median_range_nae": _median(
                    row.get("side_nae") for row in available_channels
                ),
                "view_median_range_nae": _median(
                    row.get("view_nae") for row in available_channels
                ),
                "median_matched_parameter_shift_norm": _median(
                    row.get("shift") for row in available_channels
                ),
                "matched_parameter_shift_q25": shift_q25,
                "matched_parameter_shift_q75": shift_q75,
                "within_25pct_consistency_fraction": (
                    sum(
                        float(row["shift"]) <= ACCURATE_NAE
                        for row in available_channels
                    )
                    / len(available_channels)
                    if available_channels
                    else None
                ),
            }
        )
    return pairs, summary


def build_composition_summary(
    scan_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary = [
        row for row in scan_rows
        if row["scene_id"] == "baseline"
        and row["camera_name"] == "CAM_Side"
        and row.get("canonical_scan") is True
    ]
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        groups[
            (
                str(row["model"]),
                str(row["parameter_family"]),
                int(row["parameter_count"]),
            )
        ].append(row)
    baselines: dict[tuple[str, str], float] = {}
    for (model, family, count), rows in groups.items():
        if count == 1:
            value = _median(row.get("median_range_nae") for row in rows)
            if value is not None:
                baselines[(model, family)] = value
    output: list[dict[str, Any]] = []
    for (model, family, count), rows in sorted(groups.items()):
        available = [
            row for row in rows if _number(row.get("median_range_nae")) is not None
        ]
        median_nae = _median(
            row.get("median_range_nae") for row in available
        )
        reference = baselines.get((model, family))
        output.append(
            {
                "model": model,
                "parameter_family": family,
                "combined_parameter_count": count,
                "parameter_observation_count": len(rows),
                "available_parameter_observation_count": len(available),
                "parameter_coverage": len(available) / len(rows) if rows else None,
                "median_range_nae": median_nae,
                "normalized_parameter_rmse": _normalized_rmse(
                    row.get("median_range_nae") for row in available
                ),
                "within_25pct_range_fraction": (
                    sum(
                        float(row["median_range_nae"]) <= ACCURATE_NAE
                        for row in available
                    )
                    / len(available)
                    if available
                    else None
                ),
                "single_parameter_reference_median_nae": reference,
                "delta_vs_single_parameter_median_nae": (
                    median_nae - reference
                    if median_nae is not None and reference is not None
                    else None
                ),
            }
        )
    return output


def _format(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _format_percent(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{100.0 * number:.1f}%"


def _lineage_label(value: Any) -> str:
    return {
        "consistent": "同一新版评测版本",
        "legacy_no_lineage": "旧结果兼容模式",
        "mixed_do_not_pool": "版本混合：不可合并",
    }.get(str(value), str(value or "—"))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _write_bar_svg(
    path: Path,
    *,
    title: str,
    rows: Sequence[tuple[str, float | None]],
    axis_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = [float(value) for _, value in rows if value is not None and math.isfinite(value)]
    minimum = min([0.0, *valid]) if valid else 0.0
    maximum = max([0.25, *valid]) if valid else 1.0
    if math.isclose(minimum, maximum):
        maximum = minimum + 1.0
    left, right, top, row_height = 260, 80, 62, 31
    width = 960
    height = max(180, top + row_height * len(rows) + 58)
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * plot_width

    zero = x(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        "<style>text{font-family:Arial,sans-serif;fill:#202124}"
        ".title{font-size:20px;font-weight:600}.label{font-size:13px}"
        ".axis{font-size:12px;fill:#5f6368}.grid{stroke:#d9dce1;stroke-width:1}"
        ".bar{fill:#3f6fb6}.na{fill:#b9bec7}</style>",
        f'<text class="title" x="18" y="30">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = minimum + (maximum - minimum) * tick / 5
        xpos = x(value)
        parts.append(
            f'<line class="grid" x1="{xpos:.2f}" y1="{top - 12}" '
            f'x2="{xpos:.2f}" y2="{height - 38}"/>'
        )
        parts.append(
            f'<text class="axis" text-anchor="middle" x="{xpos:.2f}" '
            f'y="{height - 20}">{value:.2f}</text>'
        )
    parts.append(
        f'<line x1="{zero:.2f}" y1="{top - 12}" x2="{zero:.2f}" '
        f'y2="{height - 38}" stroke="#73777f" stroke-width="1.4"/>'
    )
    for index, (label, value) in enumerate(rows):
        ypos = top + index * row_height
        parts.append(
            f'<text class="label" text-anchor="end" x="{left - 10}" '
            f'y="{ypos + 15}">{html.escape(label)}</text>'
        )
        if value is None or not math.isfinite(float(value)):
            parts.append(
                f'<rect class="na" x="{zero:.2f}" y="{ypos + 3}" width="18" height="16"/>'
            )
            parts.append(
                f'<text class="axis" x="{zero + 24:.2f}" y="{ypos + 15}">N/A</text>'
            )
            continue
        endpoint = x(float(value))
        bar_x = min(zero, endpoint)
        bar_width = max(1.5, abs(endpoint - zero))
        parts.append(
            f'<rect class="bar" x="{bar_x:.2f}" y="{ypos + 3}" '
            f'width="{bar_width:.2f}" height="16"/>'
        )
        text_x = endpoint + (7 if float(value) >= 0 else -7)
        anchor = "start" if float(value) >= 0 else "end"
        parts.append(
            f'<text class="label" text-anchor="{anchor}" x="{text_x:.2f}" '
            f'y="{ypos + 15}">{float(value):.3f}</text>'
        )
    parts.append(
        f'<text class="axis" text-anchor="middle" x="{left + plot_width / 2:.2f}" '
        f'y="{height - 3}">{html.escape(axis_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _write_human_reports(
    output: Path,
    *,
    model_summary: Sequence[Mapping[str, Any]],
    experiment_summary: Sequence[Mapping[str, Any]],
    background: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
    composition: Sequence[Mapping[str, Any]],
) -> None:
    summary_rows = [
        [
            row["model"],
            _format_percent(row.get("job_universe_coverage")),
            _lineage_label(row.get("lineage_state")),
            _format_percent(row["valid_motion_rate"]),
            _format_percent(row["directional_response_rate_over_all_channels"]),
            _format_percent(row["accurate_scan_rate_over_all_channels"]),
            _format(row["median_range_nae"]),
            _format(row["median_trajectory_r2"]),
            _format(row["median_trajectory_nrmse"]),
            f'{row["insufficient_scan_count"]}/{row["primary_scan_count"]}',
        ]
        for row in model_summary
    ]
    background_rows = [
        [
            row["model"],
            row["scene_group"],
            _format_percent(row["channel_coverage"]),
            _format(row["baseline_median_range_nae"]),
            _format(row["scene_median_range_nae"]),
            (
                f'{_format(row["median_matched_parameter_shift_norm"])} '
                f'[{_format(row["matched_parameter_shift_q25"])}, '
                f'{_format(row["matched_parameter_shift_q75"])}]'
            ),
        ]
        for row in background
    ]
    view_rows = [
        [
            row["model"],
            row["camera_name"],
            _format_percent(row["channel_coverage"]),
            (
                f'{_format(row["median_matched_parameter_shift_norm"])} '
                f'[{_format(row["matched_parameter_shift_q25"])}, '
                f'{_format(row["matched_parameter_shift_q75"])}]'
            ),
            _format_percent(row["within_25pct_consistency_fraction"]),
        ]
        for row in views
    ]
    composition_rows = [
        [
            row["model"],
            row["parameter_family"],
            row["combined_parameter_count"],
            _format_percent(row["parameter_coverage"]),
            _format(row["median_range_nae"]),
            _format(row["delta_vs_single_parameter_median_nae"]),
        ]
        for row in composition
    ]
    experiment_rows = [
        [
            row["model"],
            row["experiment_id"],
            f'{row["evaluable_channel_count"]}/{row["parameter_channel_count"]}',
            _format_percent(row["directional_response_rate"]),
            row["accurate_channel_count"],
            _format(row["median_range_nae"]),
            _format(row["median_trajectory_r2"]),
            row["parameter_grade_vector"],
        ]
        for row in experiment_summary
    ]
    markdown = f"""# PhysParamBench 截稿版定量结果

## 读图原则

- 主参数结果只使用 `baseline + CAM_Side`。Side 视角先去掉静止前缀和停止尾帧；碰撞、分区摩擦、斜坡转地面等事件按实验方程分段，再合并为完整运动证据。
- 参数精度与轨迹精度分开报告，不压成一个不可解释的总分。
- `NAE = |p_hat-p*|/(p_max-p_min)` 是跨物理量主指标；Normalized parameter RMSE 对大误差更敏感。
- Trajectory NRMSE 衡量方程解释轨迹的程度；R² 衡量解释方差；NMAE 对少量抖动更稳健；coverage 防止只拟合一小段。
- 13 个实验包含 24 个参数通道，不把 24 个通道误称为 24 个实验。每个 target level 先跨 seed 取中位数，再判断参数响应。
- 参数扫描按 `Accurate / Directional only / No response / Insufficient` 汇总：成对方向一致率大于 50% 才算正响应；Accurate 还要求 Theil–Sen slope 至少 0.2、median NAE 不超过 25%，并有足够的方程门控通过证据。

## 1. 干净背景 Side 主结果

{_markdown_table(
    ["模型", "978任务覆盖", "评测版本", "有效运动率", "方向响应率", "准确通道率", "Median NAE", "Trajectory R²", "Trajectory NRMSE", "证据不足/24通道"],
    summary_rows,
)}

![主参数误差](figures/figure1-parameter-fidelity.svg)

![轨迹方程拟合](figures/figure2-dynamics-fidelity.svg)

### 13 个实验系统

{_markdown_table(
    ["模型", "实验", "可评/参数通道", "方向响应率", "准确通道", "Median NAE", "Trajectory R²", "参数 grade vector"],
    experiment_rows,
)}

## 2. 复杂背景是否改变结果

同模型、实验、参数、seed 和 Side 视角与 baseline 配对。`matched shift` 越小越一致。

{_markdown_table(
    ["模型", "场景", "参数通道覆盖", "Baseline NAE", "复杂场景 NAE", "Matched shift median [IQR]"],
    background_rows,
)}

![复杂背景影响](figures/figure3-background-effect.svg)

## 3. 视角是否改变恢复参数

同模型、实验、参数、场景和 seed 下，用 Main/Top 与 Side 的恢复参数做配对一致性；它衡量参数稳定性，不假设三个独立生成视频逐帧同步。

{_markdown_table(
    ["模型", "视角", "参数通道覆盖", "相对 Side 的 matched shift median [IQR]", "25% 范围内一致率"],
    view_rows,
)}

![视角一致性](figures/figure4-view-consistency.svg)

## 4. 多参数组合是否削弱单参数理解

在 baseline Side 中，按同一参数族比较 1/2/3 个目标参数实验。`delta vs single` 为正表示组合后误差增大；只作描述性结论，不冒充因果证明。

{_markdown_table(
    ["模型", "参数族", "组合参数数", "覆盖", "Median NAE", "相对单参数 ΔNAE"],
    composition_rows,
)}

![组合参数影响](figures/figure5-composition-effect.svg)

## 输出解释

- `side_parameter_results.csv`：每条视频、每个参数的 target、直接反推值和所有轨迹指标。
- `primary_24_channel_summary.csv`：每个模型在 baseline Side 上的 24 个参数通道。
- `experiment_summary.csv`：每个模型的 13 个实验系统汇总。
- `side_scan_summary.csv`：包含所有 Side 场景和 OAT 分支的完整扫描审计表。
- `background_pairs.csv` / `view_pairs.csv`：可追溯的逐对比较；对应 summary CSV 是论文表格。
- `result.json`、轨迹 CSV 和轨迹图路径保留在长表中，便于逐条复核。
"""
    (output / "REPORT_ZH.md").write_text(markdown, encoding="utf-8")

    style = """
body{font-family:Arial,"Microsoft YaHei",sans-serif;color:#202124;max-width:1180px;
margin:32px auto;padding:0 20px;line-height:1.55}h1,h2{color:#15345d}
table{border-collapse:collapse;width:100%;margin:12px 0 24px;font-size:14px}
th,td{border:1px solid #d7dce3;padding:7px 9px;text-align:left}
th{background:#eef3f9}img{max-width:100%;height:auto;margin:8px 0 24px}
.note{background:#f3f6fa;border-left:4px solid #3f6fb6;padding:10px 14px}
code{background:#f2f3f5;padding:1px 4px}
"""
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PhysParamBench 截稿版结果</title><style>{style}</style></head><body>
<h1>PhysParamBench 截稿版定量结果</h1>
<div class="note">主结论只由 baseline Side 的参数反推给出；背景、视角和组合复杂度是三项独立稳健性分析。参数误差与轨迹拟合质量不合成一个分数。</div>
<h2>1. 干净背景 Side 主结果</h2>
{_html_table(["模型","978任务覆盖","评测版本","有效运动率","方向响应率","准确通道率","Median NAE","Trajectory R²","Trajectory NRMSE","证据不足/24通道"], summary_rows)}
<img src="figures/figure1-parameter-fidelity.svg" alt="各模型 baseline Side 参数 NAE">
<img src="figures/figure2-dynamics-fidelity.svg" alt="各模型轨迹 NRMSE">
<h3>13 个实验系统</h3>
{_html_table(["模型","实验","可评/参数通道","方向响应率","准确通道","Median NAE","Trajectory R²","参数 grade vector"], experiment_rows)}
<h2>2. 复杂背景与干净背景</h2>
{_html_table(["模型","场景","参数通道覆盖","Baseline NAE","复杂场景 NAE","Matched shift median [IQR]"], background_rows)}
<img src="figures/figure3-background-effect.svg" alt="复杂背景配对参数偏移">
<h2>3. Main/Top 相对 Side 的参数一致性</h2>
{_html_table(["模型","视角","参数通道覆盖","Matched shift median [IQR]","25% 范围内一致率"], view_rows)}
<img src="figures/figure4-view-consistency.svg" alt="不同视角参数一致性">
<h2>4. 多参数组合影响</h2>
{_html_table(["模型","参数族","组合参数数","覆盖","Median NAE","相对单参数 ΔNAE"], composition_rows)}
<img src="figures/figure5-composition-effect.svg" alt="组合参数数与误差变化">
<h2>指标为何这样选</h2>
<ul><li><b>Range NAE</b> 可跨 g、e、μ、β 比较，且目标为 0 时仍稳定。</li>
<li><b>Normalized parameter RMSE</b> 对少数很大的参数错误更敏感。</li>
<li><b>Trajectory NRMSE / NMAE / R²</b> 分别反映相对轨迹误差、抗离群误差和方程解释度。</li>
<li><b>方向一致率 + Spearman + Theil–Sen slope</b> 回答参数变大时生成运动是否沿正确方向变化。</li>
<li><b>Coverage</b> 防止只拟合短暂的好看片段。</li></ul>
<p>详细结果：<a href="side_parameter_results.csv">Side 参数长表</a> ·
<a href="primary_24_channel_summary.csv">主 24 参数通道</a> ·
<a href="experiment_summary.csv">13 实验系统</a> ·
<a href="background_comparison.csv">背景汇总</a> ·
<a href="view_consistency.csv">视角汇总</a> ·
<a href="composition_effect.csv">组合参数汇总</a> ·
<a href="input_completeness.csv">输入完整性</a> ·
<a href="background_pairs.csv">背景逐对审计</a> ·
<a href="view_pairs.csv">视角逐对审计</a> ·
<a href="REPORT_ZH.md">Markdown 报告</a></p>
</body></html>"""
    (output / "index.html").write_text(html_text, encoding="utf-8")


def build_deadline_report(
    models: Mapping[str, Path],
    *,
    output: Path,
    registry_path: Path,
    manifest_path: Path | None = None,
    expected_manifest_jobs: int | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Build the compact report for one or more model evaluation roots."""

    if not models:
        raise ValueError("at least one model evaluation root is required")
    registry = _read_json(registry_path.resolve())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameter_rows: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for name, root in models.items():
        model = str(name).strip()
        if not model:
            raise ValueError("model name cannot be empty")
        if model in inputs:
            raise ValueError(f"duplicate model name: {model}")
        inputs[model] = str(Path(root).resolve())
        loaded = load_parameter_rows(model, Path(root), registry)
        if not loaded:
            raise ValueError(
                f"model {model!r} has no rows matching the frozen registry: {root}"
            )
        parameter_rows.extend(loaded)
    lineage_audit: dict[str, dict[str, Any]] = {}
    lineage_tokens: set[tuple[str, ...]] = set()
    for model in inputs:
        model_rows = [row for row in parameter_rows if row["model"] == model]
        has_legacy = any(
            not row.get("registry_sha256")
            and not row.get("evaluator_source_sha256")
            and not row.get("physics_fitter_source_sha256")
            for row in model_rows
        )
        lineage_tuples = {
            (
                str(row.get("registry_sha256") or ""),
                str(row.get("evaluator_source_sha256") or ""),
                str(row.get("physics_fitter_source_sha256") or ""),
            )
            for row in model_rows
            if row.get("registry_sha256")
            or row.get("evaluator_source_sha256")
            or row.get("physics_fitter_source_sha256")
        }
        if len(lineage_tuples) > 1 or (has_legacy and lineage_tuples):
            raise ValueError(
                f"model {model!r} mixes legacy or multiple evaluator lineages; "
                "rerun trajectory-only evaluation into one clean output root"
            )
        if lineage_tuples:
            token = ("new", *next(iter(lineage_tuples)))
            lineage_state = "consistent"
        else:
            token = ("legacy",)
            lineage_state = "legacy_no_lineage"
        lineage_tokens.add(token)
        lineage_audit[model] = {
            "lineage_state": lineage_state,
            "lineage_cohort_count": len(lineage_tuples),
            "lineage_token": token,
        }
    if len(lineage_tokens) > 1:
        raise ValueError(
            "model inputs use different evaluator lineages; rerun all models with "
            "the same trajectory-only evaluator before cross-model pooling"
        )
    scan_rows = build_scan_rows(
        parameter_rows,
        registry,
        model_names=list(inputs),
    )
    model_summary = build_model_summary(parameter_rows, scan_rows)
    experiment_summary = build_experiment_summary(scan_rows)
    background_pairs, background_summary = build_background_comparison(parameter_rows)
    view_pairs, view_summary = build_view_comparison(parameter_rows)
    composition = build_composition_summary(scan_rows)

    expected_ids: set[str] = set()
    if manifest_path is not None:
        manifest_rows = _read_jsonl(manifest_path.resolve())
        manifest_ids = [str(row.get("job_id") or "").strip() for row in manifest_rows]
        if any(not value for value in manifest_ids):
            raise ValueError("manifest contains a row without job_id")
        if len(set(manifest_ids)) != len(manifest_ids):
            raise ValueError("manifest contains duplicate job_id values")
        if expected_manifest_jobs is not None and len(manifest_rows) != expected_manifest_jobs:
            raise ValueError(
                f"manifest must contain exactly {expected_manifest_jobs} jobs, "
                f"found {len(manifest_rows)}"
            )
        registry_experiments = set(_registry_index(registry))
        manifest_experiments: set[str] = set()
        for index, row in enumerate(manifest_rows, 1):
            experiment_id = str(row.get("experiment_id") or "")
            factors = row.get("factors")
            if experiment_id not in registry_experiments:
                raise ValueError(
                    f"manifest row {index} has unknown experiment_id={experiment_id!r}"
                )
            if not isinstance(factors, Mapping) or any(
                not str(factors.get(key) or "").strip()
                for key in ("parameter_tuple_id", "scene_id", "object_id", "camera")
            ):
                raise ValueError(f"manifest row {index} has incomplete factors")
            if row.get("seed") is None:
                raise ValueError(f"manifest row {index} has no seed")
            manifest_experiments.add(experiment_id)
        if manifest_experiments != registry_experiments:
            raise ValueError(
                "manifest experiment coverage does not match the frozen registry"
            )
        expected_ids = set(manifest_ids)
    completeness_rows: list[dict[str, Any]] = []
    for model in inputs:
        model_rows = [row for row in parameter_rows if row["model"] == model]
        observed_ids = {str(row["job_id"]) for row in model_rows}
        missing_ids = sorted(expected_ids - observed_ids)
        unexpected_ids = sorted(observed_ids - expected_ids) if expected_ids else []
        lineage_state = str(lineage_audit[model]["lineage_state"])
        completeness = {
            "model": model,
            "expected_job_count": len(expected_ids) if expected_ids else None,
            "observed_job_count": len(observed_ids),
            "job_universe_coverage": (
                len(observed_ids & expected_ids) / len(expected_ids)
                if expected_ids
                else None
            ),
            "missing_job_count": len(missing_ids),
            "unexpected_job_count": len(unexpected_ids),
            "lineage_state": lineage_state,
            "lineage_cohort_count": lineage_audit[model]["lineage_cohort_count"],
            "missing_job_ids": missing_ids,
            "unexpected_job_ids": unexpected_ids,
        }
        completeness_rows.append(completeness)
        if expected_ids and not allow_partial and (missing_ids or unexpected_ids):
            raise ValueError(
                f"model {model!r} does not match the frozen manifest: "
                f"missing={len(missing_ids)}, unexpected={len(unexpected_ids)}; "
                "finish evaluation or pass --allow-partial for a progress-only report"
            )
        summary_row = next(
            (row for row in model_summary if row["model"] == model),
            None,
        )
        if summary_row is not None:
            summary_row.update(
                {
                    "job_universe_coverage": completeness["job_universe_coverage"],
                    "missing_job_count": completeness["missing_job_count"],
                    "lineage_state": lineage_state,
                }
            )

    side_parameters = [
        row for row in parameter_rows if row["camera_name"] == "CAM_Side"
    ]
    _write_csv(output / "side_parameter_results.csv", side_parameters)
    _write_csv(output / "side_scan_summary.csv", [
        row for row in scan_rows if row["camera_name"] == "CAM_Side"
    ])
    primary_channels = [
        row
        for row in scan_rows
        if row["scene_id"] == "baseline"
        and row["camera_name"] == "CAM_Side"
        and row.get("canonical_scan") is True
    ]
    _write_csv(output / "primary_24_channel_summary.csv", primary_channels)
    _write_csv(output / "experiment_summary.csv", experiment_summary)
    _write_csv(output / "model_summary.csv", model_summary)
    _write_csv(output / "input_completeness.csv", completeness_rows)
    _write_csv(output / "background_pairs.csv", background_pairs)
    _write_csv(output / "background_comparison.csv", background_summary)
    _write_csv(output / "view_pairs.csv", view_pairs)
    _write_csv(output / "view_consistency.csv", view_summary)
    _write_csv(output / "composition_effect.csv", composition)

    _write_bar_svg(
        output / "figures" / "figure1-parameter-fidelity.svg",
        title="Baseline Side: median range-normalized parameter error",
        rows=[
            (str(row["model"]), _number(row.get("median_range_nae")))
            for row in model_summary
        ],
        axis_label="Median parameter NAE (lower is better)",
    )
    _write_bar_svg(
        output / "figures" / "figure2-dynamics-fidelity.svg",
        title="Baseline Side: median trajectory equation-fit error",
        rows=[
            (str(row["model"]), _number(row.get("median_trajectory_nrmse")))
            for row in model_summary
        ],
        axis_label="Median trajectory NRMSE (lower is better)",
    )
    _write_bar_svg(
        output / "figures" / "figure3-background-effect.svg",
        title="Matched background effect relative to clean baseline",
        rows=[
            (
                f'{row["model"]} · {row["scene_group"]}',
                _number(row.get("median_matched_parameter_shift_norm")),
            )
            for row in background_summary
        ],
        axis_label="Median matched parameter shift / valid range",
    )
    _write_bar_svg(
        output / "figures" / "figure4-view-consistency.svg",
        title="Matched view effect relative to CAM_Side",
        rows=[
            (
                f'{row["model"]} · {row["camera_name"]}',
                _number(row.get("median_matched_parameter_shift_norm")),
            )
            for row in view_summary
        ],
        axis_label="Median matched parameter shift / valid range",
    )
    _write_bar_svg(
        output / "figures" / "figure5-composition-effect.svg",
        title="Change from single-parameter experiments",
        rows=[
            (
                f'{row["model"]} · {row["parameter_family"]} · {row["combined_parameter_count"]}p',
                _number(row.get("delta_vs_single_parameter_median_nae")),
            )
            for row in composition
            if int(row["combined_parameter_count"]) > 1
        ],
        axis_label="Δ median NAE versus one-parameter reference",
    )
    _write_human_reports(
        output,
        model_summary=model_summary,
        experiment_summary=experiment_summary,
        background=background_summary,
        views=view_summary,
        composition=composition,
    )
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "policy": {
            "primary_scope": "baseline_CAM_Side",
            "parameter_metric": "range_normalized_absolute_error",
            "accurate_nae_threshold": ACCURATE_NAE,
            "directional_pairwise_threshold": DIRECTIONAL_CONCORDANCE,
            "accurate_pairwise_threshold": ACCURATE_CONCORDANCE,
            "minimum_accurate_theil_sen_slope": MINIMUM_ACCURATE_RESPONSE_SLOPE,
            "seed_aggregation": "median_estimate_within_target_level",
            "canonical_parameter_channel_count": 24,
            "parameter_and_trajectory_metrics_are_not_collapsed": True,
        },
        "models": inputs,
        "manifest": str(manifest_path.resolve()) if manifest_path is not None else None,
        "expected_manifest_jobs": expected_manifest_jobs,
        "allow_partial": allow_partial,
        "input_completeness": completeness_rows,
        "parameter_row_count": len(parameter_rows),
        "side_parameter_row_count": len(side_parameters),
        "scan_row_count": len(scan_rows),
        "primary_parameter_channel_count": len(primary_channels),
        "experiment_summary_row_count": len(experiment_summary),
        "background_pair_count": len(background_pairs),
        "view_pair_count": len(view_pairs),
        "output": str(output),
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_model_arguments(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = str(value).partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"--model must be NAME=EVALUATION_ROOT, got: {value!r}")
        if name.strip() in output:
            raise ValueError(f"duplicate --model name: {name.strip()}")
        output[name.strip()] = Path(os.path.expandvars(os.path.expanduser(path.strip())))
    return output
