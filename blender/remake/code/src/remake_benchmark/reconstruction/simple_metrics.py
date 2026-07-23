"""Simple, model-agnostic parameter-fidelity report.

The report intentionally keeps the paper-facing contract small:

``target, estimate, absolute error, trajectory R², target-range status``.

The benchmark-span normalized absolute error (BNAE) is retained only as an
auxiliary dimensionless quantity for tables that compare different physical
units.  It is unbounded and never clips an estimated parameter.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .deadline_report import trajectory_metrics


REPORT_SCHEMA_VERSION = "1.1.0"
PRIMARY_SCENE = "baseline"
PRIMARY_CAMERA = "CAM_Side"
LINEAGE_SIGNATURE_KEYS = (
    "registry_sha256",
    "evaluator_source_sha256",
    "physics_fitter_source_sha256",
    "dynamic_3d_gate_source_sha256",
    "evaluator_version",
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        output.append(value)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
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


def _median(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _number(value)) is not None]
    return float(statistics.median(finite)) if finite else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _number(value)) is not None]
    return float(statistics.fmean(finite)) if finite else None


def _stdev(values: Iterable[Any]) -> float | None:
    finite = [number for value in values if (number := _number(value)) is not None]
    if len(finite) < 2:
        return None
    return float(statistics.stdev(finite))


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = max(0.0, min(1.0, fraction)) * (len(finite) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    weight = position - left
    return float(finite[left] * (1.0 - weight) + finite[right] * weight)


def _format(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _format_percent(value: Any, digits: int = 1) -> str:
    number = _number(value)
    return "—" if number is None else f"{100.0 * number:.{digits}f}%"


def _registry_index(
    registry: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    experiment_ids: list[str] = []
    experiments: dict[str, dict[str, Any]] = {}
    anchors: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in registry.get("experiments", []):
        if not isinstance(raw, Mapping):
            continue
        spec = dict(raw)
        experiment_id = str(spec["id"])
        experiment_ids.append(experiment_id)
        experiments[experiment_id] = spec
        for anchor in spec.get("anchor_tuples", []):
            if isinstance(anchor, Mapping):
                anchors[(experiment_id, str(anchor["id"]))] = dict(anchor)
    return experiment_ids, experiments, anchors


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
        nuisance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_nuisance_signatures(
    registry: Mapping[str, Any],
) -> dict[tuple[str, str], str]:
    """Freeze one OAT response branch per parameter from registry order."""

    output: dict[tuple[str, str], str] = {}
    for spec in registry.get("experiments", []):
        if not isinstance(spec, Mapping) or not spec.get("id"):
            continue
        experiment_id = str(spec["id"])
        anchors = [
            anchor
            for anchor in spec.get("anchor_tuples", [])
            if isinstance(anchor, Mapping)
        ]
        for parameter in spec.get("hidden_parameters", []):
            if not isinstance(parameter, Mapping) or not parameter.get("name"):
                continue
            parameter_name = str(parameter["name"])
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
            if signatures_in_order:
                output[(experiment_id, parameter_name)] = max(
                    signatures_in_order,
                    key=lambda signature: len(targets_by_signature[signature]),
                )
    return output


def _target_range(spec: Mapping[str, Any], parameter: Mapping[str, Any]) -> tuple[float, float, str]:
    name = str(parameter["name"])
    values = sorted(
        {
            number
            for anchor in spec.get("anchor_tuples", [])
            if isinstance(anchor, Mapping)
            and (number := _number(anchor.get(name))) is not None
        }
    )
    if len(values) >= 2 and values[-1] - values[0] > 1e-12:
        return values[0], values[-1], "anchor_target_sweep"
    valid_range = parameter.get("valid_range", [])
    if isinstance(valid_range, Sequence) and len(valid_range) == 2:
        lower, upper = (_number(value) for value in valid_range)
        if lower is not None and upper is not None and upper > lower:
            return lower, upper, "valid_range_fallback"
    raise ValueError(f"no non-degenerate reporting range for {name}")


def _result_estimate(result: Mapping[str, Any], parameter_name: str) -> float | None:
    if str(result.get("status") or "").strip().lower() not in {
        "succeeded",
        "success",
        "complete",
        "completed",
        "ok",
    }:
        return None
    scored = result.get("metrics")
    if isinstance(scored, Mapping):
        parameter_metrics = scored.get("parameters")
        if isinstance(parameter_metrics, Mapping):
            item = parameter_metrics.get(parameter_name)
            if isinstance(item, Mapping):
                if item.get("observed") is False:
                    return None
                estimate = _number(
                    item.get("estimate", item.get("estimate_raw"))
                )
                if estimate is not None:
                    return estimate
    fit = result.get("fit")
    if not isinstance(fit, Mapping):
        return None
    observed = fit.get("parameter_observed")
    estimates = fit.get("parameter_estimates")
    if not isinstance(estimates, Mapping):
        return None
    estimate = _number(estimates.get(parameter_name))
    if estimate is None:
        return None
    if isinstance(observed, Mapping) and observed.get(parameter_name) is False:
        return None
    return estimate


def _job_trajectory_r2(result: Mapping[str, Any]) -> float | None:
    fit = result.get("fit")
    if not isinstance(fit, Mapping):
        return None
    try:
        return _number(trajectory_metrics(fit).get("trajectory_r2"))
    except (TypeError, ValueError):
        return None


def _lineage_signature(result: Mapping[str, Any]) -> dict[str, str] | None:
    lineage = result.get("evaluation_lineage")
    if not isinstance(lineage, Mapping):
        return None
    signature = {
        key: str(lineage[key])
        for key in LINEAGE_SIGNATURE_KEYS
        if lineage.get(key) is not None
    }
    return signature if len(signature) == len(LINEAGE_SIGNATURE_KEYS) else None


def _validate_result_identity(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    result_path: Path,
) -> None:
    """Fail closed when a result file belongs to a different frozen task."""

    job = result.get("job")
    if not isinstance(job, Mapping):
        raise ValueError(f"result is missing frozen job metadata: {result_path}")
    factors = manifest.get("factors")
    if not isinstance(factors, Mapping):
        factors = {}
    expected = {
        "experiment_id": manifest.get("experiment_id"),
        "parameter_tuple_id": factors.get("parameter_tuple_id"),
        "scene_id": factors.get("scene_id"),
        "object_id": factors.get("object_id"),
        "camera_name": factors.get("camera"),
        "seed": manifest.get("seed"),
    }
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        if job.get(key) is None:
            raise ValueError(
                f"result is missing frozen identity field {key!r}: {result_path}"
            )
        if str(job.get(key)) != str(expected_value):
            raise ValueError(
                f"result/manifest mismatch at {result_path}: "
                f"{key}={job.get(key)!r}, expected {expected_value!r}"
            )
    expected_video_name = f"{manifest['job_id']}.mp4"
    if job.get("video_name") is None:
        raise ValueError(
            f"result is missing frozen identity field 'video_name': {result_path}"
        )
    if str(job["video_name"]) != expected_video_name:
        raise ValueError(
            f"result/manifest mismatch at {result_path}: "
            f"video_name={job['video_name']!r}, expected {expected_video_name!r}"
        )


def load_simple_parameter_rows(
    model_name: str,
    evaluation_root: Path,
    *,
    registry: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one complete model into the frozen per-video/per-parameter schema."""

    _, experiments, anchors = _registry_index(registry)
    canonical_signatures = _canonical_nuisance_signatures(registry)
    rows: list[dict[str, Any]] = []
    results_found = 0
    missing_lineage_job_ids: list[str] = []
    lineage_signatures: set[str] = set()
    for manifest in manifest_rows:
        job_id = str(manifest["job_id"])
        experiment_id = str(manifest["experiment_id"])
        factors = manifest.get("factors", {})
        if not isinstance(factors, Mapping):
            factors = {}
        tuple_id = str(factors.get("parameter_tuple_id"))
        spec = experiments[experiment_id]
        anchor = anchors[(experiment_id, tuple_id)]
        result_path = evaluation_root / "jobs" / job_id / "result.json"
        if result_path.is_file():
            result = _read_json(result_path)
            _validate_result_identity(result, manifest, result_path)
            signature = _lineage_signature(result)
            if signature is None:
                missing_lineage_job_ids.append(job_id)
            else:
                lineage_signatures.add(
                    json.dumps(signature, ensure_ascii=False, sort_keys=True)
                )
            results_found += 1
        else:
            result = {}
        fit = result.get("fit")
        fit_status = str(fit.get("status")) if isinstance(fit, Mapping) and fit.get("status") else None
        trajectory_r2 = _job_trajectory_r2(result)
        for parameter in spec.get("hidden_parameters", []):
            if not isinstance(parameter, Mapping):
                continue
            name = str(parameter["name"])
            target = float(anchor[name])
            nuisance_signature = _nuisance_signature(spec, name, anchor)
            canonical_parameter_scan = bool(
                canonical_signatures.get((experiment_id, name))
                == nuisance_signature
            )
            estimate = _result_estimate(result, name)
            lower, upper, range_source = _target_range(spec, parameter)
            span = upper - lower
            absolute_error = None if estimate is None else abs(estimate - target)
            bnae = None if absolute_error is None else absolute_error / span
            in_range = None if estimate is None else bool(lower <= estimate <= upper)
            rows.append(
                {
                    "model": str(model_name),
                    "job_id": job_id,
                    "experiment_id": experiment_id,
                    "parameter_tuple_id": tuple_id,
                    "parameter": name,
                    "nuisance_signature": nuisance_signature,
                    "canonical_parameter_scan": canonical_parameter_scan,
                    "unit": parameter.get("unit"),
                    "target": target,
                    "estimate": estimate,
                    "absolute_error": absolute_error,
                    "trajectory_r2": trajectory_r2,
                    "target_range_min": lower,
                    "target_range_max": upper,
                    "target_range_span": span,
                    "target_range_source": range_source,
                    "target_range_status": (
                        "not_estimated"
                        if estimate is None
                        else "in_range"
                        if in_range
                        else "out_of_range"
                    ),
                    "in_target_range": in_range,
                    "bnae_aux": bnae,
                    "scene_id": str(factors.get("scene_id")),
                    "object_id": str(factors.get("object_id")),
                    "camera": str(factors.get("camera")),
                    "seed": manifest.get("seed"),
                    "primary_baseline_side": bool(
                        str(factors.get("scene_id")) == PRIMARY_SCENE
                        and str(factors.get("camera")) == PRIMARY_CAMERA
                    ),
                    "evaluation_status": result.get("status"),
                    "fit_status": fit_status,
                    "reconstruction_route": result.get("reconstruction_route"),
                    "result_json": str(result_path) if result_path.is_file() else None,
                    "trajectory_csv": result.get("trajectory_csv"),
                }
            )
    return rows, {
        "model": str(model_name),
        "manifest_jobs": len(manifest_rows),
        "results_found": results_found,
        "missing_results": len(manifest_rows) - results_found,
        "missing_lineage_results": len(missing_lineage_job_ids),
        "missing_lineage_job_ids": missing_lineage_job_ids,
        "evaluation_lineage_signatures": [
            json.loads(value) for value in sorted(lineage_signatures)
        ],
        "evaluation_root": str(evaluation_root),
    }


def _summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    experiment_id: str | None = None,
    scene_group: str | None = None,
    camera: str | None = None,
) -> dict[str, Any]:
    estimated = [row for row in rows if _number(row.get("estimate")) is not None]
    in_range = [row for row in estimated if row.get("target_range_status") == "in_range"]
    out_range = [row for row in estimated if row.get("target_range_status") == "out_of_range"]
    job_ids = {str(row["job_id"]) for row in rows}
    r2_by_job = {
        str(row["job_id"]): float(r2)
        for row in rows
        if (r2 := _number(row.get("trajectory_r2"))) is not None
    }
    accepted_r2_by_job = {
        str(row["job_id"]): float(r2)
        for row in estimated
        if (r2 := _number(row.get("trajectory_r2"))) is not None
    }
    output: dict[str, Any] = {
        "model": model,
        "parameter_records_expected": len(rows),
        "parameter_records_estimated": len(estimated),
        "estimate_coverage": len(estimated) / len(rows) if rows else None,
        "in_target_range_count": len(in_range),
        "out_of_target_range_count": len(out_range),
        "in_target_range_rate": len(in_range) / len(estimated) if estimated else None,
        "out_of_target_range_rate": len(out_range) / len(estimated) if estimated else None,
        "median_bnae_aux": _median(row.get("bnae_aux") for row in estimated),
        "median_trajectory_r2": _median(r2_by_job.values()),
        "trajectory_r2_job_count": len(r2_by_job),
        "trajectory_r2_coverage": len(r2_by_job) / len(job_ids) if job_ids else None,
        "median_accepted_trajectory_r2": _median(accepted_r2_by_job.values()),
        "accepted_trajectory_r2_job_count": len(accepted_r2_by_job),
        "job_count_expected": len(job_ids),
    }
    if experiment_id is not None:
        output["experiment_id"] = experiment_id
    if scene_group is not None:
        output["scene_group"] = scene_group
    if camera is not None:
        output["camera"] = camera
    return output


def _scene_group(scene_id: str) -> str:
    if scene_id == "baseline":
        return "baseline"
    if scene_id.startswith("indoor"):
        return "indoor"
    if scene_id.startswith("outdoor"):
        return "outdoor"
    return "other"


def _paired_condition_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair background/view conditions to the same baseline-Side task."""

    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            str(row["model"]),
            str(row["experiment_id"]),
            str(row["parameter_tuple_id"]),
            str(row["parameter"]),
            str(row["object_id"]),
            str(row["seed"]),
        )

    baseline = {
        key(row): row
        for row in rows
        if row.get("primary_baseline_side")
    }
    paired: list[dict[str, Any]] = []
    for row in rows:
        if row.get("primary_baseline_side"):
            continue
        reference = baseline.get(key(row))
        scene_id = str(row["scene_id"])
        camera = str(row["camera"])
        comparison_type = (
            "background"
            if scene_id != PRIMARY_SCENE and camera == PRIMARY_CAMERA
            else "view"
            if scene_id == PRIMARY_SCENE and camera != PRIMARY_CAMERA
            else "background_and_view"
        )
        reference_estimate = (
            None if reference is None else _number(reference.get("estimate"))
        )
        condition_estimate = _number(row.get("estimate"))
        difference = (
            None
            if reference_estimate is None or condition_estimate is None
            else condition_estimate - reference_estimate
        )
        span = _number(row.get("target_range_span"))
        paired.append(
            {
                "model": row["model"],
                "experiment_id": row["experiment_id"],
                "parameter_tuple_id": row["parameter_tuple_id"],
                "parameter": row["parameter"],
                "scene_id": scene_id,
                "scene_group": _scene_group(scene_id),
                "camera": camera,
                "seed": row["seed"],
                "comparison_type": comparison_type,
                "baseline_job_id": None if reference is None else reference["job_id"],
                "condition_job_id": row["job_id"],
                "pair_found": reference is not None,
                "target": row["target"],
                "baseline_estimate": reference_estimate,
                "condition_estimate": condition_estimate,
                "estimate_difference": difference,
                "absolute_estimate_difference": (
                    None if difference is None else abs(difference)
                ),
                "span_normalized_condition_difference": (
                    None
                    if difference is None or span is None or span <= 1e-12
                    else abs(difference) / span
                ),
                "baseline_trajectory_r2": (
                    None if reference is None else reference.get("trajectory_r2")
                ),
                "condition_trajectory_r2": row.get("trajectory_r2"),
            }
        )
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[
            (
                str(row["model"]),
                str(row["experiment_id"]),
                str(row["comparison_type"]),
                str(row["scene_group"]),
                str(row["camera"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, experiment, comparison, scene_group, camera), group in sorted(
        grouped.items()
    ):
        found = [row for row in group if row["pair_found"]]
        estimated = [
            row for row in found if _number(row["absolute_estimate_difference"]) is not None
        ]
        summaries.append(
            {
                "model": model,
                "experiment_id": experiment,
                "comparison_type": comparison,
                "scene_group": scene_group,
                "camera": camera,
                "parameter_pairs_expected": len(group),
                "baseline_pairs_found": len(found),
                "paired_estimates_found": len(estimated),
                "paired_estimate_coverage": (
                    len(estimated) / len(group) if group else None
                ),
                "median_absolute_estimate_difference": _median(
                    row["absolute_estimate_difference"] for row in estimated
                ),
                "median_span_normalized_condition_difference": _median(
                    row["span_normalized_condition_difference"] for row in estimated
                ),
            }
        )
    return paired, summaries


def _target_level_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("primary_baseline_side") or not row.get(
            "canonical_parameter_scan"
        ):
            continue
        grouped[
            (
                str(row["model"]),
                str(row["experiment_id"]),
                str(row["parameter"]),
                float(row["target"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (model, experiment_id, parameter, target), group in sorted(grouped.items()):
        estimated = [
            float(value)
            for row in group
            if (value := _number(row.get("estimate"))) is not None
        ]
        absolute_errors = [
            float(value)
            for row in group
            if (value := _number(row.get("absolute_error"))) is not None
        ]
        bnaes = [
            float(value)
            for row in group
            if (value := _number(row.get("bnae_aux"))) is not None
        ]
        r2_by_job = {
            str(row["job_id"]): float(value)
            for row in group
            if (value := _number(row.get("trajectory_r2"))) is not None
        }
        group_job_ids = {str(row["job_id"]) for row in group}
        unit = next((row.get("unit") for row in group if row.get("unit") is not None), None)
        output.append(
            {
                "model": model,
                "experiment_id": experiment_id,
                "parameter": parameter,
                "unit": unit,
                "target": target,
                "rollout_count": len(group),
                "estimate_count": len(estimated),
                "estimate_coverage": len(estimated) / len(group) if group else None,
                "estimate_mean": _mean(estimated),
                "estimate_std": _stdev(estimated),
                "estimate_median": _median(estimated),
                "estimate_q25": _quantile(estimated, 0.25),
                "estimate_q75": _quantile(estimated, 0.75),
                "median_absolute_error": _median(absolute_errors),
                "median_bnae_aux": _median(bnaes),
                "median_trajectory_r2": _median(r2_by_job.values()),
                "trajectory_r2_coverage": (
                    len(r2_by_job) / len(group_job_ids) if group_job_ids else None
                ),
                "out_of_target_range_rate": (
                    sum(row.get("target_range_status") == "out_of_range" for row in group)
                    / len(estimated)
                    if estimated
                    else None
                ),
            }
        )
    return output


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(safe(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(safe(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _latex_escape(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ):
        text = text.replace(source, replacement)
    return text


def _overview_markdown(
    model_summaries: Sequence[Mapping[str, Any]],
    experiment_summaries: Sequence[Mapping[str, Any]],
    experiment_ids: Sequence[str],
) -> str:
    by_key = {
        (str(row["model"]), str(row["experiment_id"])): row
        for row in experiment_summaries
    }
    summary_rows: list[list[str]] = []
    experiment_rows: list[list[str]] = []
    for model_row in model_summaries:
        model = str(model_row["model"])
        summary_rows.append(
            [
                model,
                _format_percent(
                    model_row.get("macro_experiment_estimate_coverage")
                ),
                _format_percent(
                    model_row.get("macro_experiment_out_of_target_range_rate")
                ),
                _format(model_row.get("macro_experiment_mean_trajectory_r2")),
                _format_percent(
                    model_row.get("macro_experiment_trajectory_r2_coverage")
                ),
                _format(model_row.get("macro_experiment_median_bnae_aux")),
            ]
        )
        values: list[str] = [model]
        for experiment_id in experiment_ids:
            summary = by_key.get((model, experiment_id), {})
            values.append(
                f"{_format_percent(summary.get('out_of_target_range_rate'), digits=0)}"
                " / "
                f"{_format_percent(summary.get('estimate_coverage'), digits=0)}"
            )
        experiment_rows.append(values)
    return (
        "### Equal-weight summary across 13 experiments\n\n"
        + _markdown_table(
            [
                "Model",
                "Estimate coverage↑",
                "Out-of-range↓",
                "Trajectory R²↑",
                "R² coverage↑",
                "BNAE (aux; median)",
            ],
            summary_rows,
        )
        + "\n\n### Per-experiment diagnostic (OOR / estimate coverage)\n\n"
        + _markdown_table(
            ["Model", *experiment_ids],
            experiment_rows,
        )
    )


def _overview_latex(
    model_summaries: Sequence[Mapping[str, Any]],
    experiment_summaries: Sequence[Mapping[str, Any]],
    experiment_ids: Sequence[str],
) -> str:
    by_key = {
        (str(row["model"]), str(row["experiment_id"])): row
        for row in experiment_summaries
    }
    columns = "l" + "c" * (len(experiment_ids) + 5)
    grouped_ids = {
        version: [value for value in experiment_ids if value.startswith(f"{version}_")]
        for version in ("v1", "v2", "v3")
    }
    group_headers: list[str] = []
    group_rules: list[str] = []
    start_column = 2
    for version in ("v1", "v2", "v3"):
        count = len(grouped_ids[version])
        if not count:
            continue
        end_column = start_column + count - 1
        group_headers.append(rf"\multicolumn{{{count}}}{{c}}{{{version.upper()}}}")
        group_rules.append(rf"\cmidrule(lr){{{start_column}-{end_column}}}")
        start_column = end_column + 1
    group_headers.append(r"\multicolumn{5}{c}{Summary}")
    group_rules.append(rf"\cmidrule(lr){{{start_column}-{start_column + 4}}}")
    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        "Model & " + " & ".join(group_headers) + r" \\",
        "".join(group_rules),
        " & "
        + " & ".join(value.replace("_", r"\_") for value in experiment_ids)
        + r" & Coverage$\uparrow$ & OOR$\downarrow$ & $R^2\uparrow$"
        + r" & $R^2$ coverage$\uparrow$ & BNAE (aux.) \\",
        r"\midrule",
    ]
    for model_row in model_summaries:
        model = str(model_row["model"])
        cells: list[str] = []
        for experiment_id in experiment_ids:
            summary = by_key.get((model, experiment_id), {})
            cells.append(
                f"{_format_percent(summary.get('out_of_target_range_rate'), digits=0).replace('%', '')}"
                "/"
                f"{_format_percent(summary.get('estimate_coverage'), digits=0).replace('%', '')}"
            )
        cells.extend(
            [
                _format_percent(
                    model_row.get("macro_experiment_estimate_coverage")
                ).replace("%", r"\%"),
                _format_percent(
                    model_row.get("macro_experiment_out_of_target_range_rate")
                ).replace("%", r"\%"),
                _format(model_row.get("macro_experiment_mean_trajectory_r2")),
                _format_percent(
                    model_row.get("macro_experiment_trajectory_r2_coverage")
                ).replace("%", r"\%"),
                _format(model_row.get("macro_experiment_median_bnae_aux")),
            ]
        )
        lines.append(model.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _estimate_tables_markdown(
    target_rows: Sequence[Mapping[str, Any]],
    model_names: Sequence[str],
    experiment_ids: Sequence[str],
    output: Path,
) -> str:
    sections: list[str] = []
    latex_sections: list[str] = []
    table_root = output / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    parameters = sorted(
        {
            (str(row["experiment_id"]), str(row["parameter"]))
            for row in target_rows
        },
        key=lambda item: (experiment_ids.index(item[0]), item[1]),
    )
    for experiment_id, parameter in parameters:
        selected = [
            row
            for row in target_rows
            if row["experiment_id"] == experiment_id and row["parameter"] == parameter
        ]
        targets = sorted({float(row["target"]) for row in selected})
        indexed = {
            (str(row["model"]), float(row["target"])): row
            for row in selected
        }
        eligible_for_closest = {
            (model, target): bool(
                _number(indexed.get((model, target), {}).get("estimate_mean"))
                is not None
                and int(indexed.get((model, target), {}).get("estimate_count") or 0)
                >= 2
                and (
                    _number(
                        indexed.get((model, target), {}).get("estimate_coverage")
                    )
                    or 0.0
                )
                >= 0.50
            )
            for model in model_names
            for target in targets
        }
        best_error = {
            target: min(
                (
                    abs(float(mean) - target)
                    for model in model_names
                    if eligible_for_closest[(model, target)]
                    if (mean := _number(indexed.get((model, target), {}).get("estimate_mean")))
                    is not None
                ),
                default=None,
            )
            for target in targets
        }
        unit = next((str(row["unit"]) for row in selected if row.get("unit")), "1")
        display_rows: list[list[str]] = [["Target", *[_format(value) for value in targets]]]
        latex_rows: list[str] = [
            "Target & " + " & ".join(_format(value) for value in targets) + r" \\"
        ]
        for model in model_names:
            cells = [model]
            latex_cells: list[str] = []
            for target in targets:
                row = indexed.get((model, target), {})
                mean = _number(row.get("estimate_mean"))
                std = _number(row.get("estimate_std"))
                if mean is None:
                    cells.append("—")
                    latex_cells.append("--")
                    continue
                count = int(row.get("estimate_count") or 0)
                spread = "N/A" if std is None else f"{std:.3f}"
                cell = f"{mean:.3f} ± {spread} (n={count})"
                latex_cell = f"{mean:.3f} $\\pm$ {spread} (n={count})"
                error = abs(mean - target)
                is_best = (
                    eligible_for_closest[(model, target)]
                    and best_error[target] is not None
                    and math.isclose(
                    error,
                    float(best_error[target]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                    )
                )
                if is_best:
                    cell = f"**{cell}**"
                    latex_cell = (
                        rf"\textbf{{{mean:.3f}}} $\pm$ {spread} (n={count})"
                    )
                cells.append(cell)
                latex_cells.append(latex_cell)
            display_rows.append(cells)
            latex_rows.append(
                _latex_escape(model) + " & " + " & ".join(latex_cells) + r" \\"
            )
        title = f"{experiment_id} — {parameter} ({unit})"
        table = _markdown_table(["Model", *[f"Target {_format(value)}" for value in targets]], display_rows)
        content = (
            f"# {title}\n\n"
            "Cells are recovered parameter mean ± standard deviation across "
            "the registry-frozen baseline Side one-factor-at-a-time branch: the "
            "reported parameter changes while the other hidden parameters remain "
            "fixed. Bold is closest to the assigned target among cells "
            "with at least two estimates and 50% coverage; single-sample spread is "
            "reported as N/A.\n\n"
            f"{table}\n"
        )
        path = table_root / f"{experiment_id}__{parameter}.md"
        path.write_text(content, encoding="utf-8")
        sections.extend([f"### {title}", "", table, ""])
        latex_sections.append(
            "\n".join(
                [
                    r"\begin{table}[t]",
                    r"\centering",
                    (
                        rf"\caption{{Recovered {_latex_escape(parameter)} for "
                        rf"{_latex_escape(experiment_id)} ({_latex_escape(unit)}). "
                        r"Bold is closest to the assigned target among cells with "
                        r"$n\geq2$ and at least 50\% coverage; N/A means the "
                        r"single-sample standard deviation is not estimable.}"
                    ),
                    r"\resizebox{\linewidth}{!}{%",
                    r"\begin{tabular}{l" + "c" * len(targets) + "}",
                    r"\toprule",
                    "Model & "
                    + " & ".join(
                        rf"Target {_format(value)}" for value in targets
                    )
                    + r" \\",
                    r"\midrule",
                    *latex_rows,
                    r"\bottomrule",
                    r"\end{tabular}}",
                    r"\end{table}",
                ]
            )
        )
    combined = "\n".join(sections)
    (output / "paper_parameter_tables.md").write_text(
        "# WorldBench-style recovered parameter tables\n\n"
        "Target values form the reference row; each model cell reports recovered "
        "mean ± sample standard deviation and evidence count on the registry-frozen "
        "one-factor-at-a-time branch. See each table note for the bolding rule.\n\n"
        + combined,
        encoding="utf-8",
    )
    (output / "paper_parameter_tables.tex").write_text(
        "\n\n".join(latex_sections) + "\n",
        encoding="utf-8",
    )
    return combined


def _html_report(markdown_overview: str, model_summaries: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Model",
        "Experiment-macro coverage",
        "Experiment-macro out of range",
        "Experiment-macro trajectory R²",
        "Experiment-macro R² coverage",
        "Median BNAE (auxiliary)",
    ]
    body_rows = []
    for row in model_summaries:
        values = [
            row["model"],
            _format_percent(row.get("macro_experiment_estimate_coverage")),
            _format_percent(row.get("macro_experiment_out_of_target_range_rate")),
            _format(row.get("macro_experiment_mean_trajectory_r2")),
            _format_percent(row.get("macro_experiment_trajectory_r2_coverage")),
            _format(row.get("macro_experiment_median_bnae_aux")),
        ]
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>PhysParamBench simple report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#20242a}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #ccd2da;padding:7px;text-align:center}}
th{{background:#eef2f7}}code,pre{{background:#f5f7fa;padding:2px 5px}}.note{{background:#fff7d6;padding:12px}}
</style></head><body>
<h1>PhysParamBench simple model comparison</h1>
<p class="note">Primary fields are Target, Estimate, AE, trajectory R², and target-range status.
BNAE is auxiliary, unbounded, and used only to compare heterogeneous parameter units.</p>
<table><thead><tr>{''.join(f'<th>{html.escape(value)}</th>' for value in headers)}</tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
  <p>The WorldBench-style recovered-parameter tables are in
  <code>paper_parameter_tables.md/.tex</code>. The overview is a coverage/QC table;
  per-video evidence is in <code>parameter_results.csv</code>, and individual tables
  are under <code>tables/</code>.</p>
<h2>Metric contract</h2>
<ul><li>AE = |estimate − target| in the native physical unit.</li>
<li>Trajectory R² evaluates the fitted motion equation, not parameter accuracy.</li>
<li>Out-of-range estimates are retained without clipping.</li>
<li>3-D trajectories are fitted in the frozen X-Z experiment plane; Y depth remains diagnostic.</li></ul>
<details><summary>Markdown experiment table source</summary><pre>{html.escape(markdown_overview)}</pre></details>
</body></html>
"""


def parse_model_arguments(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=EVALUATION_ROOT, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty model name in {value!r}")
        if name in output:
            raise ValueError(f"duplicate model name: {name}")
        output[name] = Path(path).expanduser().resolve()
    return output


def build_simple_physics_report(
    model_roots: Mapping[str, Path],
    *,
    output: Path,
    registry_path: Path,
    manifest_path: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Build per-video metrics and WorldBench-style model comparison tables."""

    if not model_roots:
        raise ValueError("at least one model is required")
    output = output.resolve()
    registry_path = registry_path.resolve()
    registry = _read_json(registry_path)
    registry_sha256 = _sha256(registry_path)
    manifest_rows = _read_jsonl(manifest_path.resolve())
    experiment_ids, _, _ = _registry_index(registry)

    all_rows: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    for model, root in model_roots.items():
        rows, state = load_simple_parameter_rows(
            str(model),
            Path(root).resolve(),
            registry=registry,
            manifest_rows=manifest_rows,
        )
        all_rows.extend(rows)
        completeness.append(state)
        if state["missing_results"] and not allow_partial:
            raise FileNotFoundError(
                f"{model}: missing {state['missing_results']} evaluation result(s) "
                f"under {root}; use --allow-partial only for progress reports"
            )
        if state["missing_lineage_results"] and not allow_partial:
            raise ValueError(
                f"{model}: {state['missing_lineage_results']} result(s) lack the "
                "frozen evaluator/fitter lineage; rerun the fit stage or use "
                "--allow-partial only for a non-final progress report"
            )

    lineage_signatures = {
        json.dumps(signature, ensure_ascii=False, sort_keys=True)
        for state in completeness
        for signature in state["evaluation_lineage_signatures"]
    }
    if len(lineage_signatures) > 1:
        raise ValueError(
            "evaluation lineage mismatch across jobs/models; rerun every model "
            "with the same evaluator, fitter, dynamic gate, and experiment registry"
        )
    for encoded in lineage_signatures:
        signature = json.loads(encoded)
        if signature["registry_sha256"] != registry_sha256:
            raise ValueError(
                "evaluation results were scored with a different experiment registry; "
                "rerun the fit stage before rebuilding this report"
            )

    primary_rows = [row for row in all_rows if row["primary_baseline_side"]]
    model_names = list(model_roots)
    model_summaries = [
        _summarize_group(
            [row for row in primary_rows if row["model"] == model],
            model=model,
        )
        for model in model_names
    ]
    experiment_summaries = [
        _summarize_group(
            [
                row
                for row in primary_rows
                if row["model"] == model and row["experiment_id"] == experiment_id
            ],
            model=model,
            experiment_id=experiment_id,
        )
        for model in model_names
        for experiment_id in experiment_ids
    ]
    for model_summary in model_summaries:
        selected = [
            row
            for row in experiment_summaries
            if row["model"] == model_summary["model"]
        ]
        model_summary.update(
            {
                # Each of the 13 designed experiments contributes once,
                # independent of its number of parameters or rollouts.
                "macro_experiment_estimate_coverage": _mean(
                    row.get("estimate_coverage") for row in selected
                ),
                "macro_experiment_out_of_target_range_rate": _mean(
                    row.get("out_of_target_range_rate") for row in selected
                ),
                "macro_experiment_mean_trajectory_r2": _mean(
                    row.get("median_trajectory_r2") for row in selected
                ),
                "macro_experiment_trajectory_r2_coverage": _mean(
                    row.get("trajectory_r2_coverage") for row in selected
                ),
                "macro_experiment_median_bnae_aux": _median(
                    row.get("median_bnae_aux") for row in selected
                ),
                "macro_experiment_count": len(selected),
            }
        )
    condition_summaries: list[dict[str, Any]] = []
    condition_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        condition_groups[
            (
                str(row["model"]),
                str(row["experiment_id"]),
                _scene_group(str(row["scene_id"])),
                str(row["camera"]),
            )
        ].append(row)
    for (model, experiment_id, scene_group, camera), rows in sorted(
        condition_groups.items()
    ):
        condition_summaries.append(
            _summarize_group(
                rows,
                model=model,
                experiment_id=experiment_id,
                scene_group=scene_group,
                camera=camera,
            )
        )
    paired_condition_rows, paired_condition_summaries = _paired_condition_rows(
        all_rows
    )
    target_summaries = _target_level_summary(all_rows)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "parameter_results.csv", all_rows)
    _write_csv(output / "primary_baseline_side_results.csv", primary_rows)
    _write_csv(output / "model_summary.csv", model_summaries)
    _write_csv(output / "experiment_summary.csv", experiment_summaries)
    _write_csv(output / "condition_summary.csv", condition_summaries)
    _write_csv(output / "paired_condition_results.csv", paired_condition_rows)
    _write_csv(output / "paired_condition_summary.csv", paired_condition_summaries)
    _write_csv(output / "target_level_estimates.csv", target_summaries)
    _write_csv(output / "completeness.csv", completeness)

    overview = _overview_markdown(model_summaries, experiment_summaries, experiment_ids)
    (output / "paper_table_overview.md").write_text(
        "# PhysParamBench coverage and range QC overview\n\n"
        "The summary macro-averages the 13 designed experiments so experiments with "
        "more parameters or rollouts do not receive extra weight. Per-experiment cells "
        "show out-of-range rate / estimate coverage on baseline Side videos. "
        "Out-of-range is a coarse failure flag, not parameter accuracy; use the raw-unit "
        "Target/Estimate/AE tables below for fidelity. BNAE is shown only as an "
        "auxiliary cross-unit scalar and is not used to rank or bold models. "
        "Out-of-range estimates are never clipped.\n\n"
        + overview
        + "\n",
        encoding="utf-8",
    )
    (output / "paper_table_overview.tex").write_text(
        _overview_latex(model_summaries, experiment_summaries, experiment_ids),
        encoding="utf-8",
    )
    estimate_sections = _estimate_tables_markdown(
        target_summaries,
        model_names,
        experiment_ids,
        output,
    )
    report_md = f"""# PhysParamBench simple evaluation report

## Frozen metric contract

- **Target**: physical parameter assigned in the prompt.
- **Estimate**: effective parameter recovered directly from the fitted trajectory equation.
- **AE**: `abs(estimate-target)` in the native unit.
- **Trajectory R²**: how well the expected equation explains the observed motion segment.
- **Target-range status**: whether the unmodified estimate lies inside the pre-registered target sweep.
- **Target-range status is a coarse failure flag, not an accuracy claim**: an in-range estimate can still be far from its assigned target; AE remains the per-video fidelity result.
- **BNAE (auxiliary)**: `AE/(max target-min target)`; unbounded and used only across units.
- Dynamic 3-D tracks are projected to the frozen Blender **X-Z motion plane**. World Y depth is retained only as a diagnostic warning.
- The model overview gives each of the 13 designed experiments equal weight. It does not let multi-parameter experiments contribute extra rows.
- Per-target mean ± standard deviation tables use a registry-frozen baseline Side one-factor-at-a-time branch, so the shown parameter changes while the other hidden parameters remain fixed.
- Background/view consistency is reported only after matching the same experiment, tuple, object, seed, and parameter to its baseline Side counterpart (`paired_condition_*.csv`).

## Coverage and range QC overview

{overview}

## WorldBench-style recovered parameter tables

{estimate_sections}
"""
    (output / "report.md").write_text(report_md, encoding="utf-8")
    (output / "report.html").write_text(
        _html_report(overview, model_summaries),
        encoding="utf-8",
    )
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "models": model_names,
        "manifest_jobs": len(manifest_rows),
        "registry_sha256": registry_sha256,
        "parameter_records": len(all_rows),
        "primary_parameter_records": len(primary_rows),
        "metric_contract": {
            "primary": [
                "target",
                "estimate",
                "absolute_error",
                "trajectory_r2",
                "target_range_status",
            ],
            "auxiliary": ["bnae_aux"],
            "estimate_clipping": False,
            "bnae_is_unbounded": True,
            "in_range_is_not_parameter_accuracy": True,
            "trajectory_r2_is_not_parameter_accuracy": True,
            "dynamic_3d_projection": "blender_world_xz_ignore_depth_y",
        },
        "completeness": completeness,
        "evaluation_lineage_signature": (
            json.loads(next(iter(lineage_signatures))) if lineage_signatures else None
        ),
        "outputs": {
            "report_markdown": str(output / "report.md"),
            "report_html": str(output / "report.html"),
            "paper_table_markdown": str(output / "paper_table_overview.md"),
            "paper_table_latex": str(output / "paper_table_overview.tex"),
            "paper_parameter_tables_markdown": str(
                output / "paper_parameter_tables.md"
            ),
            "paper_parameter_tables_latex": str(
                output / "paper_parameter_tables.tex"
            ),
            "parameter_results_csv": str(output / "parameter_results.csv"),
            "model_summary_csv": str(output / "model_summary.csv"),
            "experiment_summary_csv": str(output / "experiment_summary.csv"),
            "condition_summary_csv": str(output / "condition_summary.csv"),
            "paired_condition_results_csv": str(
                output / "paired_condition_results.csv"
            ),
            "paired_condition_summary_csv": str(
                output / "paired_condition_summary.csv"
            ),
            "target_level_estimates_csv": str(output / "target_level_estimates.csv"),
        },
    }
    _write_json(output / "summary.json", summary)
    return summary


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_simple_physics_report",
    "load_simple_parameter_rows",
    "parse_model_arguments",
]
