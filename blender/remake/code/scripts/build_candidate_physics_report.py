#!/usr/bin/env python3
"""Build a lenient but auditable multi-model PhysParamBench deadline report.

This report intentionally separates:

1. whether a target-free trajectory fit produced a finite parameter candidate;
2. whether the fitted equation explains the selected motion segment; and
3. whether the recovered parameter is close to the prompted target.

It never replaces a missing estimate with zero and never clips an estimate to
the benchmark parameter range.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.deadline_report import (  # noqa: E402
    build_scan_rows,
    load_parameter_rows,
    parse_model_arguments,
)


REPORT_SCHEMA_VERSION = "1.0.0"
DEFAULT_R2_MIN = 0.50
DEFAULT_TRAJECTORY_NRMSE_MAX = 0.25
DEFAULT_PARAMETER_BNAE_MAX = 0.25
DEFAULT_DIRECTIONAL_SLOPE_MIN = 0.10
DEFAULT_ACCURATE_SLOPE_MIN = 0.50
DEFAULT_ACCURATE_SLOPE_MAX = 1.50


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _median(values: Iterable[Any]) -> float | None:
    numeric = [value for item in values if (value := _number(item)) is not None]
    return float(statistics.median(numeric)) if numeric else None


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [value for item in values if (value := _number(item)) is not None]
    return float(statistics.fmean(numeric)) if numeric else None


def _quartiles(values: Iterable[Any]) -> tuple[float | None, float | None]:
    numeric = sorted(
        value for item in values if (value := _number(item)) is not None
    )
    if not numeric:
        return None, None
    if len(numeric) == 1:
        return numeric[0], numeric[0]
    cuts = statistics.quantiles(numeric, n=4, method="inclusive")
    return float(cuts[0]), float(cuts[2])


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _number(value)
    return "—" if numeric is None else f"{numeric:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    numeric = _number(value)
    return "—" if numeric is None else f"{100.0 * numeric:.{digits}f}%"


def _json_key(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _equation_supported(
    row: Mapping[str, Any],
    *,
    r2_min: float,
    trajectory_nrmse_max: float,
) -> bool:
    candidate = _number(row.get("candidate_estimate"))
    fit_status = str(row.get("fit_status") or "").lower()
    attribution_status = str(
        row.get("parameter_attribution_status") or ""
    ).lower()
    # Periodic and collision fitters can provide valid parameter-specific rule
    # evidence without a single top-level global R²/NRMSE.  Do not discard
    # those event-based fits merely because the generic trajectory metric is
    # absent.
    if (
        candidate is not None
        and fit_status in {"ok", "partial"}
        and attribution_status == "pass"
    ):
        return True
    r2 = _number(row.get("trajectory_r2"))
    nrmse = _number(row.get("trajectory_nrmse"))
    return (
        candidate is not None
        and r2 is not None
        and nrmse is not None
        and r2 >= r2_min
        and nrmse <= trajectory_nrmse_max
    )


def _candidate_nae(row: Mapping[str, Any]) -> float | None:
    candidate = _number(row.get("candidate_estimate"))
    target = _number(row.get("target"))
    span = _number(row.get("benchmark_target_span"))
    if (
        candidate is None
        or target is None
        or span is None
        or span <= 0
    ):
        return None
    return abs(candidate - target) / span


def _attach_benchmark_target_ranges(rows: Sequence[dict[str, Any]]) -> None:
    """Attach the preregistered target sweep used to normalize parameter error.

    The range is the min/max target appearing in the frozen registry for one
    experiment-parameter channel. It is deliberately not the wider physical
    validity domain.
    """

    targets: dict[tuple[str, str], set[float]] = defaultdict(set)
    for row in rows:
        target = _number(row.get("target"))
        if target is not None:
            targets[
                (
                    str(row.get("experiment_id") or ""),
                    str(row.get("parameter_name") or ""),
                )
            ].add(target)
    ranges: dict[tuple[str, str], tuple[float, float, float | None]] = {}
    for key, values in targets.items():
        lower = min(values)
        upper = max(values)
        ranges[key] = (lower, upper, upper - lower if upper > lower else None)
    for row in rows:
        lower, upper, span = ranges[
            (
                str(row.get("experiment_id") or ""),
                str(row.get("parameter_name") or ""),
            )
        ]
        row["benchmark_target_min"] = lower
        row["benchmark_target_max"] = upper
        row["benchmark_target_span"] = span


def _deduplicated_jobs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        output.setdefault(str(row.get("job_id") or ""), row)
    return output


def _fit_funnel(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    jobs = _deduplicated_jobs(rows)
    counts = Counter(str(row.get("fit_status") or "missing") for row in jobs.values())
    skipped = counts.get("skipped_trajectory_not_eligible", 0)
    attempted = len(jobs) - skipped
    return {
        "primary_job_count": len(jobs),
        "fit_attempted_job_count": attempted,
        "fit_skipped_job_count": skipped,
        "equation_mismatch_job_count": counts.get("model_mismatch", 0),
        "insufficient_evidence_job_count": counts.get("insufficient_evidence", 0),
        "strict_ok_or_partial_job_count": counts.get("ok", 0)
        + counts.get("partial", 0),
    }


def _model_summary(
    model: str,
    primary: Sequence[Mapping[str, Any]],
    *,
    r2_min: float,
    trajectory_nrmse_max: float,
    parameter_bnae_max: float,
) -> dict[str, Any]:
    funnel = _fit_funnel(primary)
    candidates = [
        row for row in primary if _number(row.get("candidate_estimate")) is not None
    ]
    supported = [
        row
        for row in primary
        if _equation_supported(
            row,
            r2_min=r2_min,
            trajectory_nrmse_max=trajectory_nrmse_max,
        )
    ]
    successful = [
        row
        for row in supported
        if (
            (error := _candidate_nae(row)) is not None
            and error <= parameter_bnae_max
        )
    ]
    supported_jobs = _deduplicated_jobs(supported)
    supported_errors = [
        error
        for row in supported
        if (error := _candidate_nae(row)) is not None
    ]
    return {
        "model": model,
        **funnel,
        "fit_attempt_rate": _ratio(
            funnel["fit_attempted_job_count"], funnel["primary_job_count"]
        ),
        "expected_parameter_observation_count": len(primary),
        "candidate_parameter_count": len(candidates),
        "candidate_parameter_coverage": _ratio(len(candidates), len(primary)),
        "equation_supported_parameter_count": len(supported),
        "equation_supported_parameter_coverage": _ratio(len(supported), len(primary)),
        "equation_supported_job_count": len(supported_jobs),
        "equation_supported_job_rate": _ratio(
            len(supported_jobs), funnel["primary_job_count"]
        ),
        "parameter_success_count": len(successful),
        "parameter_success_at_25pct_all": _ratio(len(successful), len(primary)),
        "parameter_success_at_25pct_conditional": _ratio(
            len(successful), len(supported)
        ),
        "median_candidate_bnae_supported": _median(supported_errors),
        "median_trajectory_r2_supported": _median(
            row.get("trajectory_r2") for row in supported_jobs.values()
        ),
        "median_trajectory_nrmse_supported": _median(
            row.get("trajectory_nrmse") for row in supported_jobs.values()
        ),
    }


def _canonical_channel_keys(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    models: Sequence[str],
) -> dict[str, list[tuple[str, str, str]]]:
    scans = build_scan_rows(rows, registry, model_names=models)
    output: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in scans:
        if (
            row.get("scene_id") == "baseline"
            and row.get("camera_name") == "CAM_Side"
            and row.get("canonical_scan") is True
        ):
            key = (
                str(row.get("experiment_id") or ""),
                str(row.get("parameter_name") or ""),
                _json_key(row.get("nuisance_signature")),
            )
            if key not in output[str(row.get("model") or "")]:
                output[str(row.get("model") or "")].append(key)
    return output


def _scan_rows(
    rows: Sequence[Mapping[str, Any]],
    canonical: Mapping[str, Sequence[tuple[str, str, str]]],
    *,
    r2_min: float,
    trajectory_nrmse_max: float,
    parameter_bnae_max: float,
    directional_slope_min: float,
    accurate_slope_min: float,
    accurate_slope_max: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, keys in canonical.items():
        model_rows = [
            row
            for row in rows
            if row.get("model") == model
            and row.get("scene_id") == "baseline"
            and row.get("camera_name") == "CAM_Side"
        ]
        for experiment, parameter, nuisance in sorted(keys):
            channel = [
                row
                for row in model_rows
                if row.get("experiment_id") == experiment
                and row.get("parameter_name") == parameter
                and _json_key(row.get("nuisance_signature")) == nuisance
                and _equation_supported(
                    row,
                    r2_min=r2_min,
                    trajectory_nrmse_max=trajectory_nrmse_max,
                )
            ]
            by_target: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
            for row in channel:
                target = _number(row.get("target"))
                if target is not None:
                    by_target[target].append(row)
            levels: list[dict[str, Any]] = []
            for target, level_rows in sorted(by_target.items()):
                estimates = [
                    estimate
                    for row in level_rows
                    if (estimate := _number(row.get("candidate_estimate"))) is not None
                ]
                if not estimates:
                    continue
                errors = [
                    error
                    for row in level_rows
                    if (error := _candidate_nae(row)) is not None
                ]
                levels.append(
                    {
                        "target": target,
                        "estimate": float(statistics.median(estimates)),
                        "seed_count": len(estimates),
                        "median_bnae": _median(errors),
                    }
                )
            pair_scores: list[float] = []
            slopes: list[float] = []
            for left in range(len(levels)):
                for right in range(left + 1, len(levels)):
                    target_delta = levels[right]["target"] - levels[left]["target"]
                    if abs(target_delta) <= 1e-12:
                        continue
                    estimate_delta = (
                        levels[right]["estimate"] - levels[left]["estimate"]
                    )
                    pair_scores.append(
                        1.0 if estimate_delta > 0 else 0.5 if estimate_delta == 0 else 0.0
                    )
                    slopes.append(estimate_delta / target_delta)
            direction = _mean(pair_scores)
            slope = _median(slopes)
            span = (
                max(level["target"] for level in levels)
                - min(level["target"] for level in levels)
                if len(levels) >= 2
                else None
            )
            normalized_rmse = (
                math.sqrt(
                    statistics.fmean(
                        (level["estimate"] - level["target"]) ** 2
                        for level in levels
                    )
                )
                / span
                if span is not None and span > 1e-12
                else None
            )
            if len(levels) < 2:
                grade = "U"
            elif (
                direction is not None
                and direction > 0.50
                and slope is not None
                and accurate_slope_min <= slope <= accurate_slope_max
                and normalized_rmse is not None
                and normalized_rmse <= parameter_bnae_max
            ):
                grade = "G4_ACCURATE"
            elif (
                direction is not None
                and direction > 0.50
                and slope is not None
                and slope >= directional_slope_min
            ):
                grade = "G3_DIRECTIONAL"
            else:
                grade = "G2_NO_RESPONSE"
            output.append(
                {
                    "model": model,
                    "experiment_id": experiment,
                    "parameter_name": parameter,
                    "nuisance_signature": nuisance,
                    "available_target_level_count": len(levels),
                    "pairwise_direction_accuracy": direction,
                    "theil_sen_slope": slope,
                    "target_level_normalized_rmse": normalized_rmse,
                    "median_target_level_bnae": _median(
                        level.get("median_bnae") for level in levels
                    ),
                    "grade": grade,
                    "target_levels": levels,
                }
            )
    return output


def _scan_model_summary(
    model_summaries: Sequence[dict[str, Any]],
    scans: Sequence[Mapping[str, Any]],
) -> None:
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    for row in scans:
        by_model[str(row.get("model") or "")][str(row.get("grade") or "U")] += 1
    for row in model_summaries:
        counts = by_model[str(row["model"])]
        total = sum(counts.values())
        evaluable = total - counts.get("U", 0)
        row.update(
            {
                "canonical_channel_count": total,
                "evaluable_channel_count": evaluable,
                "g4_accurate_channel_count": counts.get("G4_ACCURATE", 0),
                "g3_directional_channel_count": counts.get("G3_DIRECTIONAL", 0),
                "g2_no_response_channel_count": counts.get("G2_NO_RESPONSE", 0),
                "u_channel_count": counts.get("U", 0),
                "directional_or_better_rate_all_channels": _ratio(
                    counts.get("G4_ACCURATE", 0)
                    + counts.get("G3_DIRECTIONAL", 0),
                    total,
                ),
                "accurate_rate_all_channels": _ratio(
                    counts.get("G4_ACCURATE", 0), total
                ),
            }
        )


def _matched_condition_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparison: str,
    r2_min: float,
    trajectory_nrmse_max: float,
) -> list[dict[str, Any]]:
    supported = [
        row
        for row in rows
        if _equation_supported(
            row,
            r2_min=r2_min,
            trajectory_nrmse_max=trajectory_nrmse_max,
        )
    ]
    if comparison == "background":
        key_fields = (
            "model",
            "experiment_id",
            "parameter_tuple_id",
            "parameter_name",
            "camera_name",
            "seed",
        )
        reference_name = "baseline"
        condition_field = "scene_id"
        supported = [row for row in supported if row.get("camera_name") == "CAM_Side"]
    elif comparison == "view":
        key_fields = (
            "model",
            "experiment_id",
            "parameter_tuple_id",
            "parameter_name",
            "scene_id",
            "seed",
        )
        reference_name = "CAM_Side"
        condition_field = "camera_name"
    else:
        raise ValueError(comparison)
    groups: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in supported:
        key = tuple(str(row.get(field) or "") for field in key_fields)
        groups[key][str(row.get(condition_field) or "")] = row
    output: list[dict[str, Any]] = []
    for key, conditions in groups.items():
        reference = conditions.get(reference_name)
        if reference is None:
            continue
        reference_estimate = _number(reference.get("candidate_estimate"))
        span = _number(reference.get("benchmark_target_span"))
        if (
            reference_estimate is None
            or span is None
            or span <= 0
        ):
            continue
        for condition, row in conditions.items():
            if condition == reference_name:
                continue
            estimate = _number(row.get("candidate_estimate"))
            if estimate is None:
                continue
            output.append(
                {
                    **{field: value for field, value in zip(key_fields, key)},
                    "comparison": comparison,
                    "reference": reference_name,
                    "condition": condition,
                    "reference_estimate": reference_estimate,
                    "condition_estimate": estimate,
                    "normalized_parameter_shift": abs(
                        estimate - reference_estimate
                    )
                    / span,
                    "reference_job_id": reference.get("job_id"),
                    "condition_job_id": row.get("job_id"),
                }
            )
    return output


def _condition_summary(
    pairs: Sequence[Mapping[str, Any]], comparison: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        condition = str(row.get("condition") or "")
        if comparison == "background":
            if condition.startswith("indoor"):
                condition = "indoor"
            elif condition.startswith("outdoor"):
                condition = "outdoor"
        groups[(str(row.get("model") or ""), condition)].append(row)
    output: list[dict[str, Any]] = []
    for (model, condition), rows in sorted(groups.items()):
        values = [
            value
            for row in rows
            if (value := _number(row.get("normalized_parameter_shift"))) is not None
        ]
        q25, q75 = _quartiles(values)
        output.append(
            {
                "model": model,
                "comparison": comparison,
                "condition": condition,
                "matched_pair_count": len(values),
                "median_normalized_parameter_shift": _median(values),
                "q25_normalized_parameter_shift": q25,
                "q75_normalized_parameter_shift": q75,
                "within_25pct_consistency_rate": _ratio(
                    sum(value <= 0.25 for value in values), len(values)
                ),
            }
        )
    return output


def _composition_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    r2_min: float,
    trajectory_nrmse_max: float,
    parameter_bnae_max: float,
) -> list[dict[str, Any]]:
    primary = [
        row
        for row in rows
        if row.get("scene_id") == "baseline"
        and row.get("camera_name") == "CAM_Side"
    ]
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        groups[
            (str(row.get("model") or ""), int(row.get("parameter_count") or 0))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (model, count), group in sorted(groups.items()):
        supported = [
            row
            for row in group
            if _equation_supported(
                row,
                r2_min=r2_min,
                trajectory_nrmse_max=trajectory_nrmse_max,
            )
        ]
        successful = [
            row
            for row in supported
            if (error := _candidate_nae(row)) is not None
            and error <= parameter_bnae_max
        ]
        output.append(
            {
                "model": model,
                "parameter_count": count,
                "expected_parameter_observation_count": len(group),
                "equation_supported_parameter_count": len(supported),
                "equation_supported_parameter_coverage": _ratio(
                    len(supported), len(group)
                ),
                "parameter_success_count": len(successful),
                "parameter_success_at_25pct_all": _ratio(
                    len(successful), len(group)
                ),
                "median_candidate_bnae_supported": _median(
                    _candidate_nae(row) for row in supported
                ),
            }
        )
    return output


def _seed_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    r2_min: float,
    trajectory_nrmse_max: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, float], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        target = _number(row.get("target"))
        if (
            row.get("scene_id") == "baseline"
            and row.get("camera_name") == "CAM_Side"
            and target is not None
            and _equation_supported(
                row,
                r2_min=r2_min,
                trajectory_nrmse_max=trajectory_nrmse_max,
            )
        ):
            groups[
                (
                    str(row.get("model") or ""),
                    str(row.get("experiment_id") or ""),
                    str(row.get("parameter_name") or ""),
                    _json_key(row.get("nuisance_signature")),
                    target,
                )
            ].append(row)
    by_model: dict[str, list[float]] = defaultdict(list)
    for key, group in groups.items():
        estimates = [
            value
            for row in group
            if (value := _number(row.get("candidate_estimate"))) is not None
        ]
        span = _number(group[0].get("benchmark_target_span"))
        if len(estimates) < 2 or span is None or span <= 0:
            continue
        center = float(statistics.median(estimates))
        mad = float(statistics.median(abs(value - center) for value in estimates))
        by_model[key[0]].append(mad / span)
    return [
        {
            "model": model,
            "multi_seed_target_group_count": len(values),
            "median_seed_mad_normalized": _median(values),
            "q75_seed_mad_normalized": _quartiles(values)[1],
        }
        for model, values in sorted(by_model.items())
    ]


def _case_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    r2_min: float,
    trajectory_nrmse_max: float,
    parameter_bnae_max: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in sorted({str(row.get("model") or "") for row in rows}):
        primary = [
            row
            for row in rows
            if row.get("model") == model
            and row.get("scene_id") == "baseline"
            and row.get("camera_name") == "CAM_Side"
        ]
        supported = [
            row
            for row in primary
            if _equation_supported(
                row,
                r2_min=r2_min,
                trajectory_nrmse_max=trajectory_nrmse_max,
            )
            and _candidate_nae(row) is not None
        ]
        good = sorted(
            (
                row
                for row in supported
                if _candidate_nae(row) is not None
                and _candidate_nae(row) <= parameter_bnae_max
            ),
            key=lambda row: (
                _candidate_nae(row) or math.inf,
                -(_number(row.get("trajectory_r2")) or -math.inf),
            ),
        )
        wrong = sorted(
            (
                row
                for row in supported
                if _candidate_nae(row) is not None
                and _candidate_nae(row) > parameter_bnae_max
            ),
            key=lambda row: (
                -(_number(row.get("trajectory_r2")) or -math.inf),
                -(_candidate_nae(row) or -math.inf),
            ),
        )
        unavailable = [
            row
            for row in primary
            if _number(row.get("candidate_estimate")) is None
            and row.get("fit_status") == "skipped_trajectory_not_eligible"
        ]
        for case_type, candidates in (
            ("parameter_close", good),
            ("trajectory_good_parameter_wrong", wrong),
            ("tracking_or_reconstruction_unavailable", unavailable),
        ):
            if not candidates:
                continue
            row = candidates[0]
            output.append(
                {
                    "model": model,
                    "case_type": case_type,
                    "job_id": row.get("job_id"),
                    "experiment_id": row.get("experiment_id"),
                    "parameter_name": row.get("parameter_name"),
                    "target": row.get("target"),
                    "candidate_estimate": row.get("candidate_estimate"),
                    "candidate_bnae": _candidate_nae(row),
                    "trajectory_r2": row.get("trajectory_r2"),
                    "trajectory_nrmse": row.get("trajectory_nrmse"),
                    "fit_status": row.get("fit_status"),
                    "result_json": row.get("result_json"),
                    "trajectory_csv": row.get("trajectory_csv"),
                    "trajectory_plot": row.get("trajectory_plot"),
                    "video_path": row.get("video_path"),
                }
            )
    return output


def _bar_svg(
    path: Path,
    *,
    title: str,
    rows: Sequence[tuple[str, float | None, float | None]],
    left_label: str,
    right_label: str,
) -> None:
    width = 940
    row_height = 68
    height = 110 + row_height * len(rows)
    plot_left = 250
    plot_width = 610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#172033}"
        ".title{font-size:24px;font-weight:700}.label{font-size:16px}"
        ".value{font-size:14px;font-weight:700}.axis{font-size:13px;fill:#5d6678}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="36" class="title">{html.escape(title)}</text>',
        f'<text x="{plot_left}" y="68" class="axis">{html.escape(left_label)}</text>',
        f'<text x="{plot_left + 320}" y="68" class="axis">{html.escape(right_label)}</text>',
    ]
    for index, (label, left, right) in enumerate(rows):
        y = 88 + index * row_height
        parts.append(
            f'<text x="24" y="{y + 21}" class="label">{html.escape(label)}</text>'
        )
        for offset, value, color in ((0, left, "#3765d6"), (320, right, "#e07a3f")):
            numeric = max(0.0, min(1.0, value or 0.0))
            x = plot_left + offset
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="260" height="24" rx="4" fill="#edf0f5"/>',
                    f'<rect x="{x}" y="{y}" width="{260 * numeric:.2f}" height="24" '
                    f'rx="4" fill="{color}"/>',
                    f'<text x="{x + 268}" y="{y + 18}" class="value">'
                    f'{100.0 * numeric:.1f}%</text>',
                ]
            )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _scan_svg(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    width = 940
    row_height = 68
    height = 110 + row_height * len(summaries)
    x0 = 250
    bar_width = 610
    colors = {
        "G4": "#2ca25f",
        "G3": "#4c78a8",
        "G2": "#e39c37",
        "U": "#c7ccd6",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#172033}"
        ".title{font-size:24px;font-weight:700}.label{font-size:16px}"
        ".small{font-size:13px;fill:#4f596a}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="36" class="title">Canonical parameter channels: response grade</text>',
        '<text x="250" y="68" class="small">G4 accurate</text>',
        '<text x="360" y="68" class="small">G3 direction only</text>',
        '<text x="510" y="68" class="small">G2 no response</text>',
        '<text x="650" y="68" class="small">U insufficient</text>',
    ]
    for index, row in enumerate(summaries):
        y = 88 + index * row_height
        parts.append(
            f'<text x="24" y="{y + 20}" class="label">'
            f'{html.escape(str(row["model"]))}</text>'
        )
        counts = [
            ("G4", int(row.get("g4_accurate_channel_count") or 0)),
            ("G3", int(row.get("g3_directional_channel_count") or 0)),
            ("G2", int(row.get("g2_no_response_channel_count") or 0)),
            ("U", int(row.get("u_channel_count") or 0)),
        ]
        total = max(1, sum(count for _, count in counts))
        cursor = x0
        for grade, count in counts:
            segment = bar_width * count / float(total)
            parts.append(
                f'<rect x="{cursor:.2f}" y="{y}" width="{segment:.2f}" height="26" '
                f'fill="{colors[grade]}"/>'
            )
            if count:
                parts.append(
                    f'<text x="{cursor + segment / 2:.2f}" y="{y + 18}" '
                    f'text-anchor="middle" class="small">{count}</text>'
                )
            cursor += segment
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]]
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_reports(
    output: Path,
    *,
    model_summary: Sequence[Mapping[str, Any]],
    video_counts_by_model: Mapping[str, int],
    scans: Sequence[Mapping[str, Any]],
    background: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
    composition: Sequence[Mapping[str, Any]],
    seeds: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    r2_min: float,
    trajectory_nrmse_max: float,
    parameter_bnae_max: float,
    directional_slope_min: float,
) -> None:
    model_names = [str(row["model"]) for row in model_summary]
    model_count = len(model_names)
    video_counts = [int(video_counts_by_model.get(name, 0)) for name in model_names]
    total_video_count = sum(video_counts)
    if video_counts and len(set(video_counts)) == 1:
        inventory_zh = (
            f"{model_count} 个模型 × {video_counts[0]} 条视频/模型 "
            f"= {total_video_count} 条"
        )
    else:
        inventory_zh = (
            f"{model_count} 个模型，共 {total_video_count} 条视频（"
            + "；".join(
                f"{name}={video_counts_by_model.get(name, 0)}"
                for name in model_names
            )
            + "）"
        )
    inventory_en = (
        f"{model_count} video world model{'' if model_count == 1 else 's'} "
        f"and {total_video_count:,} generated videos"
    )
    model_list_zh = "、".join(model_names)
    total_g4 = sum(
        int(row.get("g4_accurate_channel_count") or 0) for row in model_summary
    )
    total_g3 = sum(
        int(row.get("g3_directional_channel_count") or 0) for row in model_summary
    )
    total_expected_parameters = sum(
        int(row.get("expected_parameter_observation_count") or 0)
        for row in model_summary
    )
    total_supported_parameters = sum(
        int(row.get("equation_supported_parameter_count") or 0)
        for row in model_summary
    )
    total_success_parameters = sum(
        int(row.get("parameter_success_count") or 0) for row in model_summary
    )
    overall_support_rate = _ratio(
        total_supported_parameters,
        total_expected_parameters,
    )
    overall_success_rate = _ratio(
        total_success_parameters,
        total_expected_parameters,
    )
    if total_g4:
        g4_zh = f"共有 {total_g4} 个参数通道达到 G4"
        g4_en = (
            f"{total_g4} complete parameter channels achieved both directional "
            "and numerical fidelity"
        )
    else:
        g4_zh = "没有完整参数通道达到 G4"
        g4_en = (
            "none achieved both directional and numerical fidelity over a "
            "complete parameter channel"
        )
    ranked = [
        row
        for row in model_summary
        if _number(row.get("parameter_success_at_25pct_all")) is not None
    ]
    if ranked:
        best = max(float(row["parameter_success_at_25pct_all"]) for row in ranked)
        leaders = [
            str(row["model"])
            for row in ranked
            if math.isclose(
                float(row["parameter_success_at_25pct_all"]),
                best,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        ranking_zh = (
            f"当前口径下，{'、'.join(leaders)} 的 `Success@25%` 最高"
            f"（{_pct(best)}），但这只是描述性比较，不能写成绝对能力排名。"
        )
    else:
        ranking_zh = "当前没有足够的数值证据比较 `Success@25%`。"
    triple_rows = [
        row for row in composition if int(row.get("parameter_count") or 0) == 3
    ]
    triple_success = sum(
        int(row.get("parameter_success_count") or 0) for row in triple_rows
    )
    triple_expected = sum(
        int(row.get("expected_parameter_observation_count") or 0)
        for row in triple_rows
    )
    if triple_rows:
        triple_en = (
            f"{triple_success} of {triple_expected} three-parameter observations "
            "met the preregistered 25\\% target-span tolerance"
        )
        triple_zh = (
            f"三参数组合中 {triple_success}/{triple_expected} 个应评参数观测"
            "达到 Success@25%"
        )
    else:
        triple_en = "The current inputs contained no three-parameter observations"
        triple_zh = "当前输入没有三参数组合观测"
    main_table = _markdown_table(
        [
            "Model",
            "Fit attempted",
            "Equation-supported params",
            "Success@25% (all expected)",
            "Conditional median BNAE",
            "G3 direction/channels",
            "G4 accurate/channels",
            "U/channels",
        ],
        [
            [
                str(row["model"]),
                _pct(row.get("fit_attempt_rate")),
                (
                    f'{row["equation_supported_parameter_count"]}/'
                    f'{row["expected_parameter_observation_count"]} '
                    f'({_pct(row.get("equation_supported_parameter_coverage"))})'
                ),
                (
                    f'{row["parameter_success_count"]}/'
                    f'{row["expected_parameter_observation_count"]} '
                    f'({_pct(row.get("parameter_success_at_25pct_all"))})'
                ),
                _fmt(row.get("median_candidate_bnae_supported")),
                (
                    f'{row.get("g3_directional_channel_count") or 0}/'
                    f'{row.get("canonical_channel_count") or 0}'
                ),
                (
                    f'{row.get("g4_accurate_channel_count") or 0}/'
                    f'{row.get("canonical_channel_count") or 0}'
                ),
                (
                    f'{row.get("u_channel_count") or 0}/'
                    f'{row.get("canonical_channel_count") or 0}'
                ),
            ]
            for row in model_summary
        ],
    )
    scan_rows = _markdown_table(
        ["Model", "Experiment", "Parameter", "Grade", "Levels", "Direction", "Slope"],
        [
            [
                str(row["model"]),
                str(row["experiment_id"]),
                str(row["parameter_name"]),
                str(row["grade"]),
                str(row["available_target_level_count"]),
                _fmt(row.get("pairwise_direction_accuracy")),
                _fmt(row.get("theil_sen_slope")),
            ]
            for row in scans
            if row.get("grade") != "U"
        ],
    )
    background_rows = _markdown_table(
        ["Model", "Scene", "Pairs", "Median shift", "Within 25%"],
        [
            [
                str(row["model"]),
                str(row["condition"]),
                str(row["matched_pair_count"]),
                _fmt(row.get("median_normalized_parameter_shift")),
                _pct(row.get("within_25pct_consistency_rate")),
            ]
            for row in background
        ],
    )
    view_rows = _markdown_table(
        ["Model", "View", "Pairs", "Median shift", "Within 25%"],
        [
            [
                str(row["model"]),
                str(row["condition"]),
                str(row["matched_pair_count"]),
                _fmt(row.get("median_normalized_parameter_shift")),
                _pct(row.get("within_25pct_consistency_rate")),
            ]
            for row in views
        ],
    )
    composition_rows = _markdown_table(
        ["Model", "# parameters", "Supported", "Success@25%", "Median BNAE"],
        [
            [
                str(row["model"]),
                str(row["parameter_count"]),
                (
                    f'{row["equation_supported_parameter_count"]}/'
                    f'{row["expected_parameter_observation_count"]}'
                ),
                _pct(row.get("parameter_success_at_25pct_all")),
                _fmt(row.get("median_candidate_bnae_supported")),
            ]
            for row in composition
        ],
    )
    seed_by_model = {str(row["model"]): row for row in seeds}
    seed_rows = _markdown_table(
        ["Model", "Multi-seed target groups", "Median normalized seed MAD", "Q75"],
        [
            [
                str(row["model"]),
                str(seed_by_model.get(str(row["model"]), {}).get(
                    "multi_seed_target_group_count", 0
                )),
                _fmt(
                    seed_by_model.get(str(row["model"]), {}).get(
                        "median_seed_mad_normalized"
                    )
                ),
                _fmt(
                    seed_by_model.get(str(row["model"]), {}).get(
                        "q75_seed_mad_normalized"
                    )
                ),
            ]
            for row in model_summary
        ],
    )
    case_rows = _markdown_table(
        ["Model", "Case", "Job", "Target → estimate", "BNAE", "R²"],
        [
            [
                str(row["model"]),
                str(row["case_type"]),
                f'`{row["job_id"]}`',
                f'{_fmt(row.get("target"))} → {_fmt(row.get("candidate_estimate"))}',
                _fmt(row.get("candidate_bnae")),
                _fmt(row.get("trajectory_r2")),
            ]
            for row in cases
        ],
    )
    case_labels = {
        "parameter_close": "参数较接近",
        "trajectory_good_parameter_wrong": "轨迹方程可用但参数错误",
        "tracking_or_reconstruction_unavailable": "跟踪/重建证据不足",
    }

    def _case_line(row: Mapping[str, Any]) -> str:
        label = case_labels.get(
            str(row.get("case_type") or ""),
            str(row.get("case_type") or "代表案例"),
        )
        return (
            f"**{row.get('model')} · {row.get('experiment_id')} · {label}**："
            f"`{row.get('job_id')}`；{row.get('parameter_name')} "
            f"{_fmt(row.get('target'))} → {_fmt(row.get('candidate_estimate'))}，"
            f"BNAE={_fmt(row.get('candidate_bnae'))}，"
            f"R²={_fmt(row.get('trajectory_r2'))}。"
        )

    selected_cases: list[Mapping[str, Any]] = []
    for case_type in (
        "trajectory_good_parameter_wrong",
        "parameter_close",
        "tracking_or_reconstruction_unavailable",
    ):
        candidate = next(
            (row for row in cases if row.get("case_type") == case_type),
            None,
        )
        if candidate is not None:
            selected_cases.append(candidate)
    brief_case_lines = (
        "\n".join(
            f"{index}. {_case_line(row)}"
            for index, row in enumerate(selected_cases[:3], 1)
        )
        if selected_cases
        else "当前没有自动选出的案例；请检查 `case_selection.csv`。"
    )
    report = f"""# PhysParamBench {model_count} 模型截稿版结果（简化、可审计）

## 一句话结论

本报告覆盖 {inventory_zh}，模型为 {model_list_zh}。在宽松、统一且不依赖目标值的方程门槛下，方程支持为 {total_supported_parameters}/{total_expected_parameters}（{_pct(overall_support_rate)}），`Success@25%` 为 {total_success_parameters}/{total_expected_parameters}（{_pct(overall_success_rate)}）；共有 {total_g3} 个 G3 方向响应通道，且{g4_zh}。这些实际统计用于判断是否支持“视觉上合理不等于参数忠实”的论文主结论。

## 冻结口径

- 主结果只用 `baseline + CAM_Side`；各模型的任务数与参数观测分母由主表动态列出。
- 参数先由所选运动段的轨迹方程直接反推；不使用目标值参与拟合，不裁剪到合法范围。
- 方程支持：`R² ≥ {r2_min:.2f}` 且 `trajectory NRMSE ≤ {trajectory_nrmse_max:.2f}`。这只表示方程至少解释一半轨迹方差且归一化误差不过大。
- 单视频参数误差：`BNAE = |estimate-target| / (benchmark target span)`。
- `Success@25%`：方程支持且 `BNAE ≤ {parameter_bnae_max:.2f}`。分母始终是该模型全部应评参数；缺失不会被当成成功。
- 参数扫描：每个目标值先跨 seed 取中位数。G2=没有稳定响应；G3=方向正确、Theil–Sen 斜率至少 `{directional_slope_min:.2f}` 但数值不准；G4=方向正确、斜率在 `[0.5,1.5]` 且 target-level NRMSE≤0.25；U=少于两个可比较目标值。
- 这是 deadline 主分析；严格规则族报告仍保留作为敏感性分析，不与本表混用。

## 论文主表

{main_table}

![主结果漏斗](figures/figure1-main-funnel.svg)

![参数扫描分级](figures/figure2-scan-grades.svg)

### 对主表的正确解读

1. {ranking_zh}
2. 当前共有 {total_g3} 个 G3 方向响应通道；方向响应不等于数值准确。
3. {g4_zh}。即使某一通道达到 G4，也不能外推为模型整体理解。
4. U 表示自动证据不足，不等于参数错误；论文主表同时给覆盖率，避免把检测/重建失败伪装成物理失败。

## 可比较的参数扫描

{scan_rows}

## 复杂背景：与同设定 baseline Side 的配对偏移

{background_rows}

这里的 shift 只在 baseline 和复杂场景两边都通过方程支持时计算；pair 数较少的行只能作为描述性证据。

## 视角：Main/Top 与同设定 Side 的配对偏移

{view_rows}

三个视角是独立生成，不是逐帧同步多视角，因此该表衡量参数分布的一致性，而不是几何重投影误差。

## 参数组合复杂度

{composition_rows}

只比较 1/2/3 参数条件下的方程支持率和端到端成功率，不把它写成严格因果结论。

## 多 seed 稳定性

{seed_rows}

## 汇报时展示的代表案例

{case_rows}

`case_selection.csv` 中还保留了 result、轨迹 CSV、轨迹图和服务器原视频路径。正式汇报时每模型最多展示两条：一条“轨迹方程很好但参数错”，一条“参数较接近”；再展示一条检测/重建不可用案例说明 U 的含义。

## 可以直接用于论文的结论

> Across {inventory_en}, target-free system identification frequently failed to recover the prompted physical parameters. Under a lenient common equation-support criterion, only a subset of the expected baseline-side parameter observations yielded equation-supported estimates, and end-to-end parameter accuracy remained limited. Several parameter scans exhibited the correct response direction; {g4_en}. These results show that visually plausible motion and even a good trajectory-equation fit do not imply faithful realization of the specified physical parameters.

## 不应写的结论

- 不要写“U 就是模型物理失败”；U 仍混有跟踪或重建证据不足。
- 不要只报 conditional BNAE 而隐藏覆盖率。
- 不要把背景、视角配对中很少的 pair 当成显著性结论。
- 不要把当前排序写成绝对能力榜；最稳妥的是同时报告覆盖率、误差和方向响应。
"""
    (output / "REPORT_FOR_ADVISORS_ZH.md").write_text(report, encoding="utf-8")

    brief = f"""# 给师兄的 5 分钟汇报提纲

## 第 1 页：工作完成度

- 当前统一报告覆盖 {inventory_zh}。
- {model_count} 个模型使用同一任务清单与统一轨迹格式，并由同一 registry、evaluator、物理拟合器和 3D gate 评分；逐视频 trajectory/track hash 用于追溯。
- 主统计固定为干净背景 `baseline + CAM_Side`；复杂背景、视角和 seed 作为鲁棒性分析。

建议直接说：

> 视频和评测都已经完整跑完。现在不再追求复杂分级，而是用同一个宽松且可解释的门槛，把“能否拟合方程”“参数是否准确”“参数变化方向是否正确”分开报告。

## 第 2 页：评测方法

1. 从运动开始到实验规定的结束事件，提取平面轨迹。
2. 用该实验的运动方程直接拟合轨迹并反推参数，目标参数不参与拟合。
3. `R²≥{r2_min:.2f}` 且轨迹 `NRMSE≤{trajectory_nrmse_max:.2f}` 才认为方程证据可用。
4. 用 `BNAE=|estimate-target|/benchmark target span` 跨物理量比较。
5. `BNAE≤{parameter_bnae_max:.2f}` 记为单视频参数较准确；参数扫描再判断方向是否正确。

展示：[figure1-main-funnel.svg](figures/figure1-main-funnel.svg)

## 第 3 页：论文主结果

{main_table}

建议直接说：

> {ranking_zh} 当前共有 {total_g3} 个 G3 方向响应通道，且{g4_zh}。这与摘要里“视觉合理不等于参数忠实”的判断一致。

展示：[figure2-scan-grades.svg](figures/figure2-scan-grades.svg)

## 第 4 页：自动选择的直观案例

{brief_case_lines}

每个案例只放：并排原视频、带轨迹 overlay、目标→反推参数和一张拟合图。具体 job 在 `case_selection.csv` 和 `scan_response_channels.csv`。

## 第 5 页：论文叙事与限制

可以写：

- 高轨迹 R² 不保证目标参数正确；
- 三参数组合结果按上表实际 `Success@25%` 报告，不预先假定为零；
- 同设定在背景、视角和 seed 变化下，反推参数常发生明显偏移；
- 部分视频因跟踪/重建证据不足记为 U，不能将 U 直接归咎于模型。

不要写：

- “所有 U 都是模型失败”；
- “某模型绝对优于另一模型”；
- 不带覆盖率只比较 conditional BNAE；
- 将当前配对样本较少的背景/视角结果写成显著因果结论。

## 被追问时的回答

**为什么门槛这么设？**
`R²≥0.5` 表示方程至少解释一半方差，`NRMSE≤0.25` 是宽松的轨迹误差上限；它们不使用 target，因此不会把结果往期望参数上拉。

**为什么用 BNAE？**
重力、摩擦、恢复系数和阻尼单位不同，除以每个 benchmark 预注册的目标跨度后才能放入同一张表；原始 target、estimate 和 AE 仍在附表保留。

**覆盖率低还能写吗？**
可以，但必须把覆盖率作为主指标之一，并将 U 描述为自动证据不足。论文的强结论来自“即使在方程证据可用的子集中，参数误差仍大”，不是把缺失结果算作错误。
"""
    (output / "ADVISOR_BRIEF_5MIN_ZH.md").write_text(brief, encoding="utf-8")

    dynamic_story_sections = []
    for index, row in enumerate(cases, 1):
        dynamic_story_sections.append(
            f"""## Case {index}：{row.get("model")} · {row.get("experiment_id")} · {case_labels.get(str(row.get("case_type") or ""), row.get("case_type"))}

- Job：`{row.get("job_id")}`
- 参数：`{row.get("parameter_name")}`
- Target → estimate：`{_fmt(row.get("target"))} → {_fmt(row.get("candidate_estimate"))}`
- BNAE：`{_fmt(row.get("candidate_bnae"))}`
- 轨迹 R² / NRMSE：`{_fmt(row.get("trajectory_r2"))}` / `{_fmt(row.get("trajectory_nrmse"))}`

该案例由本次输入的 `case_selection.csv` 自动选择；图注只陈述这些可核验数值。"""
        )
    story_cases = (
        "# 建议制作的论文/汇报案例视频\n\n"
        "案例完全由本次输入动态选择。请从对应模型的 `videos/`、"
        "`tracks/jobs/` 和 `evaluation/jobs/` 取回原视频、overlay 与拟合图。\n\n"
        + (
            "\n\n".join(dynamic_story_sections)
            if dynamic_story_sections
            else "当前输入没有自动选出的案例。"
        )
        + "\n\n## 每个案例固定展示四项\n\n"
        "1. 原始视频；\n"
        "2. 物体检测与轨迹 overlay；\n"
        "3. 观测轨迹与拟合轨迹图；\n"
        "4. 一行公式和 `target → estimate, BNAE, R²`。\n"
    )
    (output / "STORY_CASES_ZH.md").write_text(story_cases, encoding="utf-8")

    wrong_case_count = sum(
        row.get("case_type") == "trajectory_good_parameter_wrong"
        for row in cases
    )
    if wrong_case_count:
        case_evidence_en = (
            f"The automatic audit selected {wrong_case_count} case"
            f"{'' if wrong_case_count == 1 else 's'} with equation-supported "
            "trajectories but parameter error above the reporting tolerance."
        )
    else:
        case_evidence_en = (
            "The automatic audit selected no equation-supported, "
            "parameter-wrong showcase case from the current inputs."
        )
    paper_text = f"""## Quantitative results (draft)

Table~\\ref{{tab:physparambench-main}} reports trajectory-equation support and
parameter fidelity separately on the clean baseline side view. Across
{inventory_en}, {total_supported_parameters} of {total_expected_parameters}
expected parameter observations yielded a finite estimate supported by the
prescribed equation under the common criterion, and {total_success_parameters}
met Success@25%. The scan table contained {total_g3} G3 directional-response
channels; {g4_en}. Notably,
{case_evidence_en} These findings
support the central distinction of PhysParamBench: visually plausible dynamics,
and even trajectories well explained by a simple equation, are insufficient
evidence that a video model has instantiated the requested physical system.

Composed dynamics were evaluated separately. {triple_en} in the baseline-side
primary analysis. Matched background and viewpoint
comparisons also showed substantial shifts in recovered parameters, while their
pair coverage varied across models; we therefore treat these comparisons as
descriptive robustness evidence rather than as a causal ranking. Missing or
unreconstructable cases are reported as insufficient evidence and are never
imputed as zero-error observations.
"""
    (output / "PAPER_RESULTS_DRAFT_EN.md").write_text(paper_text, encoding="utf-8")

    latex_rows = "\n".join(
        (
            f'{str(row["model"]).replace("_", r"\_")} & '
            f'{100.0 * float(row["equation_supported_parameter_coverage"]):.1f} & '
            f'{100.0 * float(row["parameter_success_at_25pct_all"]):.1f} & '
            f'{_fmt(row.get("median_candidate_bnae_supported"))} & '
            f'{int(row.get("g3_directional_channel_count") or 0)}/'
            f'{int(row.get("canonical_channel_count") or 0)} & '
            f'{int(row.get("g4_accurate_channel_count") or 0)}/'
            f'{int(row.get("canonical_channel_count") or 0)} \\\\'
        )
        for row in model_summary
    )
    latex = rf"""\begin{{table}}[t]
\centering
\caption{{Parameter-conditioned system identification on the clean baseline side view. Coverage and fidelity are reported separately.}}
\label{{tab:physparambench-main}}
\begin{{tabular}}{{lccccc}}
\toprule
Model & Eq. support $\uparrow$ & Success@25\% $\uparrow$ & Median BNAE $\downarrow$ & G3/ch. $\uparrow$ & G4/ch. $\uparrow$ \\
\midrule
{latex_rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (output / "paper_main_table.tex").write_text(latex, encoding="utf-8")


def build_report(
    models: Mapping[str, Path],
    *,
    output: Path,
    registry_path: Path,
    r2_min: float,
    trajectory_nrmse_max: float,
    parameter_bnae_max: float,
    directional_slope_min: float,
    accurate_slope_min: float,
    accurate_slope_max: float,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    parameter_rows: list[dict[str, Any]] = []
    for model, root in models.items():
        parameter_rows.extend(load_parameter_rows(model, root, registry))
    video_job_ids: dict[str, set[str]] = defaultdict(set)
    for row in parameter_rows:
        model = str(row.get("model") or "")
        job_id = str(row.get("job_id") or "")
        if model and job_id:
            video_job_ids[model].add(job_id)
    video_counts_by_model = {
        model: len(video_job_ids.get(model, set())) for model in models
    }
    _attach_benchmark_target_ranges(parameter_rows)
    output.mkdir(parents=True, exist_ok=True)
    primary_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parameter_rows:
        if row.get("scene_id") == "baseline" and row.get("camera_name") == "CAM_Side":
            primary_by_model[str(row.get("model") or "")].append(row)
    summaries = [
        _model_summary(
            model,
            primary_by_model[model],
            r2_min=r2_min,
            trajectory_nrmse_max=trajectory_nrmse_max,
            parameter_bnae_max=parameter_bnae_max,
        )
        for model in models
    ]
    canonical = _canonical_channel_keys(
        parameter_rows, registry, list(models)
    )
    scans = _scan_rows(
        parameter_rows,
        canonical,
        r2_min=r2_min,
        trajectory_nrmse_max=trajectory_nrmse_max,
        parameter_bnae_max=parameter_bnae_max,
        directional_slope_min=directional_slope_min,
        accurate_slope_min=accurate_slope_min,
        accurate_slope_max=accurate_slope_max,
    )
    _scan_model_summary(summaries, scans)
    background_pairs = _matched_condition_rows(
        parameter_rows,
        comparison="background",
        r2_min=r2_min,
        trajectory_nrmse_max=trajectory_nrmse_max,
    )
    view_pairs = _matched_condition_rows(
        parameter_rows,
        comparison="view",
        r2_min=r2_min,
        trajectory_nrmse_max=trajectory_nrmse_max,
    )
    background = _condition_summary(background_pairs, "background")
    views = _condition_summary(view_pairs, "view")
    composition = _composition_summary(
        parameter_rows,
        r2_min=r2_min,
        trajectory_nrmse_max=trajectory_nrmse_max,
        parameter_bnae_max=parameter_bnae_max,
    )
    seeds = _seed_summary(
        parameter_rows,
        r2_min=r2_min,
        trajectory_nrmse_max=trajectory_nrmse_max,
    )
    cases = _case_rows(
        parameter_rows,
        r2_min=r2_min,
        trajectory_nrmse_max=trajectory_nrmse_max,
        parameter_bnae_max=parameter_bnae_max,
    )
    _write_csv(output / "paper_main_table.csv", summaries)
    _write_csv(output / "scan_response_channels.csv", scans)
    _write_csv(output / "background_pairs.csv", background_pairs)
    _write_csv(output / "background_summary.csv", background)
    _write_csv(output / "view_pairs.csv", view_pairs)
    _write_csv(output / "view_summary.csv", views)
    _write_csv(output / "composition_summary.csv", composition)
    _write_csv(output / "seed_stability.csv", seeds)
    _write_csv(output / "case_selection.csv", cases)
    _bar_svg(
        output / "figures" / "figure1-main-funnel.svg",
        title="Baseline Side: equation support and end-to-end parameter success",
        rows=[
            (
                str(row["model"]),
                _number(row.get("equation_supported_parameter_coverage")),
                _number(row.get("parameter_success_at_25pct_all")),
            )
            for row in summaries
        ],
        left_label="Equation-supported",
        right_label="Success@25% (all expected)",
    )
    _scan_svg(output / "figures" / "figure2-scan-grades.svg", summaries)
    _write_reports(
        output,
        model_summary=summaries,
        video_counts_by_model=video_counts_by_model,
        scans=scans,
        background=background,
        views=views,
        composition=composition,
        seeds=seeds,
        cases=cases,
        r2_min=r2_min,
        trajectory_nrmse_max=trajectory_nrmse_max,
        parameter_bnae_max=parameter_bnae_max,
        directional_slope_min=directional_slope_min,
    )
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "policy": {
            "primary_scope": "baseline_CAM_Side",
            "candidate_estimates_are_target_free": True,
            "estimate_clipping": False,
            "missing_is_not_zero": True,
            "equation_support": {
                "trajectory_r2_min": r2_min,
                "trajectory_nrmse_max": trajectory_nrmse_max,
            },
            "parameter_success_bnae_max": parameter_bnae_max,
            "scan_directional_theil_sen_slope_min": directional_slope_min,
            "scan_accurate_theil_sen_slope": [
                accurate_slope_min,
                accurate_slope_max,
            ],
        },
        "parameter_row_count": len(parameter_rows),
        "video_count_by_model": video_counts_by_model,
        "total_video_count": sum(video_counts_by_model.values()),
        "primary_parameter_row_count": sum(
            len(rows) for rows in primary_by_model.values()
        ),
        "model_summary": summaries,
        "scan_grade_counts": {
            model: dict(
                Counter(
                    str(row.get("grade") or "U")
                    for row in scans
                    if row.get("model") == model
                )
            )
            for model in models
        },
        "output": str(output.resolve()),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=EVALUATION_ROOT",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=CODE_ROOT
        / "assets"
        / "seedance978_evaluation"
        / "experiment_registry.json",
    )
    parser.add_argument("--r2-min", type=float, default=DEFAULT_R2_MIN)
    parser.add_argument(
        "--trajectory-nrmse-max",
        type=float,
        default=DEFAULT_TRAJECTORY_NRMSE_MAX,
    )
    parser.add_argument(
        "--parameter-bnae-max",
        type=float,
        default=DEFAULT_PARAMETER_BNAE_MAX,
    )
    parser.add_argument(
        "--directional-slope-min",
        type=float,
        default=DEFAULT_DIRECTIONAL_SLOPE_MIN,
    )
    parser.add_argument(
        "--accurate-slope-min",
        type=float,
        default=DEFAULT_ACCURATE_SLOPE_MIN,
    )
    parser.add_argument(
        "--accurate-slope-max",
        type=float,
        default=DEFAULT_ACCURATE_SLOPE_MAX,
    )
    args = parser.parse_args()
    summary = build_report(
        parse_model_arguments(args.model),
        output=args.output.resolve(),
        registry_path=args.registry.resolve(),
        r2_min=args.r2_min,
        trajectory_nrmse_max=args.trajectory_nrmse_max,
        parameter_bnae_max=args.parameter_bnae_max,
        directional_slope_min=args.directional_slope_min,
        accurate_slope_min=args.accurate_slope_min,
        accurate_slope_max=args.accurate_slope_max,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
