"""Transparent four-level grading for the paper's primary benchmark claim.

This module intentionally implements a smaller contract than
``hierarchical_grading``.  It evaluates only the frozen primary slice:

* ``scene_id == "baseline"``;
* ``camera_name == "CAM_Side"``;
* one caller-declared primary seed (341867882 by default); and
* one canonical, honest one-at-a-time (OAT) scan per experiment parameter.

The preferred measurement route is calibrated 2-D.  A baseline Side sample
that was routed to dynamic 3-D remains eligible only when the caller attaches
an ``include`` decision from the target-independent metric motion-manifold
gate.  The route is retained in every evidence row.

Tracking or inverse-measurement insufficiency is reported as ``X`` and is not
counted as a model failure.  This layer does not recompute trajectory NRMSE or
R2; it consumes the target-independent, experiment-specific rule-family and
per-parameter gates produced by the inverse fitter.  All functions use only
the Python standard library and accept already-loaded CSV rows and
experiment-registry JSON data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from statistics import median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.2.0"
DEFAULT_PRIMARY_SEED = 341867882
PRIMARY_SCENE_ID = "baseline"
PRIMARY_CAMERA_NAME = "CAM_Side"

GRADE_LABELS = {
    "L1": "clear_motion_generation_failure",
    "L2": "motion_valid_parameter_response_failed",
    "L3": "positive_direction_numeric_inaccurate",
    "L4": "positive_direction_numeric_accurate",
    "X": "measurement_or_tracking_insufficient",
}

DEFAULT_THRESHOLDS = {
    "minimum_usable_levels": 2,
    "minimum_pairwise_direction_concordance_exclusive": 0.5,
    "minimum_slope_exclusive": 0.0,
    "l4_min_slope_inclusive": 0.2,
    "l4_max_median_valid_range_nae_inclusive": 0.25,
}

_PASS = {"pass", "passed", "ok", "success", "succeeded", "valid", "true", "1"}
_FAIL = {"fail", "failed", "invalid", "g0", "g1"}
_REVIEW = {"review", "needs_review", "provisional_review", "soft_review"}
_INDETERMINATE = {
    "",
    "unknown",
    "indeterminate",
    "not_run",
    "not-run",
    "missing",
    "unavailable",
    "u",
    "x",
}
_FIT_SUCCESS = {"ok", "pass", "passed", "success", "succeeded", "complete", "completed"}
_FIT_EVIDENCE_AVAILABLE = _FIT_SUCCESS | {"partial", "model_mismatch"}
_DYNAMIC_ROUTE = "spatialtrackerv2_dynamic"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _status(value: Any) -> str:
    return _text(value).lower()


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _status(value)
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _seed(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _codes(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _text(value)
    if not text:
        return []
    for separator in (";", "|"):
        text = text.replace(separator, ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def _row_id(row: Mapping[str, Any], ordinal: int) -> str:
    return (
        _text(row.get("job_id"))
        or _text(row.get("video_name"))
        or f"row-{ordinal:06d}"
    )


def _dynamic_3d_decision(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    if _text(row.get("reconstruction_route")) != _DYNAMIC_ROUTE:
        return "not_applicable", []
    decision = _status(
        row.get("simple_dynamic_3d_decision")
        or row.get("dynamic_3d_inclusion_decision")
    )
    reasons = _codes(
        row.get("simple_dynamic_3d_reason_codes")
        or row.get("dynamic_3d_inclusion_reason_codes")
    )
    if decision == "include":
        return "include", reasons
    return "X", reasons or ["dynamic_3d_evidence_gate_not_passed"]


def _generation_status(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return status and evidence source.

    Automatic failure gates in the current benchmark are deliberately not
    promoted to a paper-facing model failure.  Several such failures are known
    to be caused by tracking/background leakage.  Only a frozen manual decision
    may produce L1 in this compact report; otherwise the row remains X until it
    is reviewed.
    """

    manual = _status(row.get("manual_generation_validity_status"))
    if manual in _PASS | _FAIL | _REVIEW:
        return manual, "manual"
    automatic_values = [
        (_status(row.get(key)), key)
        for key in (
            "generation_validity_status",
            "automatic_generation_validity_status",
            "g0_generation_status",
        )
    ]
    for value, _key in automatic_values:
        if value in _FAIL:
            return value, "automatic"
    if _status(row.get("video_stage")) == "g0":
        return "fail", "automatic_hierarchical_gate"
    if manual:
        return manual, "manual"
    for value, _key in automatic_values:
        if value:
            return value, "automatic"
    stage = _status(row.get("video_stage"))
    if stage == "g0":
        return "fail", "automatic_hierarchical_gate"
    if stage in {"pass_to_scan", "g1"}:
        return "pass", "automatic_hierarchical_gate"
    return "", "missing"


def _generation_review_state(
    row: Mapping[str, Any],
    generation: str,
    generation_source: str,
) -> dict[str, Any]:
    """Keep soft REVIEW usable while retaining hard-failure fail-closed logic."""

    failure_codes = _codes(row.get("generation_failure_codes"))
    warning_codes = _codes(row.get("generation_warning_codes"))
    review_codes = [value for value in warning_codes if "review" in value.lower()]
    manual_pass = generation_source == "manual" and generation in _PASS
    hard_failure = bool(
        not manual_pass
        and (generation in _FAIL or failure_codes)
    )
    provisional_review = bool(
        not hard_failure
        and not manual_pass
        and (generation in _REVIEW or review_codes)
    )
    return {
        "hard_failure": hard_failure,
        "provisional_review": provisional_review,
        "failure_codes": failure_codes,
        "review_codes": list(dict.fromkeys(review_codes)),
    }


def _motion_status(row: Mapping[str, Any]) -> tuple[str, str]:
    manual = _status(row.get("manual_motion_validity_status"))
    if manual:
        return manual, "manual"
    for key in ("motion_type_status", "motion_validity_status", "g1_motion_status"):
        value = _status(row.get(key))
        if value:
            return value, "automatic"
    stage = _status(row.get("video_stage"))
    if stage == "g1":
        return "fail", "automatic_hierarchical_gate"
    if stage == "pass_to_scan":
        return "pass", "automatic_hierarchical_gate"
    return "", "missing"


def _parameter_estimate(row: Mapping[str, Any], parameter_name: str) -> float | None:
    direct = _number(row.get(f"{parameter_name}__estimate"))
    if direct is not None:
        return direct
    estimates = row.get("parameter_estimates")
    if isinstance(estimates, Mapping):
        return _number(estimates.get(parameter_name))
    fit = row.get("fit")
    if isinstance(fit, Mapping):
        nested = fit.get("parameter_estimates")
        if isinstance(nested, Mapping):
            return _number(nested.get(parameter_name))
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _parameter_fit_evidence(
    row: Mapping[str, Any],
    parameter_name: str,
    estimate: float | None,
) -> dict[str, Any]:
    """Normalize target-parameter evidence from new and legacy result rows."""

    status = _status(row.get(f"{parameter_name}__attribution_status"))
    reasons = _codes(row.get(f"{parameter_name}__attribution_reason_codes"))
    attribution = _mapping(row.get("parameter_attribution"))
    if not attribution:
        attribution = _mapping(row.get("parameter_attribution_json"))
    fit = _mapping(row.get("fit"))
    if not attribution:
        attribution = _mapping(fit.get("parameter_attribution"))
    item = _mapping(attribution.get(parameter_name))
    if not status:
        status = _status(item.get("status"))
    if not reasons:
        reasons = _codes(item.get("reason_codes"))

    family_status = _status(
        row.get("rule_family_status")
        or _mapping(fit.get("rule_family_evaluation")).get("status")
    )
    family_reasons = _codes(row.get("rule_family_reason_codes"))
    if not family_reasons:
        family_reasons = _codes(
            _mapping(fit.get("rule_family_evaluation")).get("reason_codes")
        )
    fit_status = _status(row.get("fit_status") or fit.get("status"))

    if family_status in _FAIL | {"rejected", "model_mismatch"}:
        normalized = "model_mismatch"
        reasons = reasons or family_reasons
    elif status in _PASS | {"accepted"}:
        normalized = "accepted" if estimate is not None else "indeterminate"
    elif status in _FAIL | {"rejected", "model_mismatch"}:
        normalized = "model_mismatch"
    elif estimate is not None:
        normalized = "accepted"
    elif fit_status == "model_mismatch":
        normalized = "model_mismatch"
    else:
        normalized = "indeterminate"
    return {
        "status": normalized,
        "attribution_status": status or None,
        "reason_codes": list(dict.fromkeys(reasons or family_reasons)),
        "rule_family_status": family_status or None,
        "fit_status": fit_status or None,
    }


def _classify_row(
    row: Mapping[str, Any],
    parameter_name: str,
    *,
    row_id: str,
) -> dict[str, Any]:
    generation, generation_source = _generation_status(row)
    generation_review = _generation_review_state(row, generation, generation_source)
    motion, motion_source = _motion_status(row)
    video_stage = _status(row.get("video_stage"))
    generation_codes = _codes(row.get("generation_failure_codes"))
    dynamic_decision, dynamic_reasons = _dynamic_3d_decision(row)
    dynamic_measurement_substitute = bool(
        dynamic_decision == "include" and not generation_review["hard_failure"]
    )

    failure_reasons: list[str] = []
    if generation_source == "manual" and generation in _FAIL:
        failure_reasons.append("clear_generation_validity_failure")
    if motion_source == "manual" and motion in _FAIL:
        failure_reasons.append("clear_motion_type_failure")
    if failure_reasons:
        return {
            "row_id": row_id,
            "state": "clear_model_failure",
            "reason_codes": failure_reasons + [f"generation_failure:{code}" for code in generation_codes],
            "generation_status": generation or None,
            "generation_status_source": generation_source,
            "generation_review_provisional": False,
            "generation_hard_failure": True,
            "motion_status": motion or None,
            "motion_status_source": motion_source,
            "estimate": None,
            "measurement_route": _text(row.get("simple_measurement_route")) or None,
        }

    insufficiency: list[str] = []
    if dynamic_decision == "X":
        insufficiency.extend(dynamic_reasons)
    if generation_review["hard_failure"] and generation_source != "manual":
        insufficiency.append("automatic_failure_pending_manual_confirmation")
    if motion in _FAIL and motion_source != "manual":
        insufficiency.append("automatic_motion_failure_pending_manual_confirmation")
    generation_accepted = bool(
        generation in _PASS or generation_review["provisional_review"]
    )
    if not generation_accepted and not dynamic_measurement_substitute:
        insufficiency.append(
            "generation_validity_evidence_indeterminate"
            if generation in _INDETERMINATE
            else "generation_validity_evidence_unrecognized"
        )
    if motion in _INDETERMINATE - {""}:
        insufficiency.append("motion_type_evidence_indeterminate")

    trajectory_eligible = _bool(row.get("trajectory_fit_eligible"))
    if trajectory_eligible is False:
        insufficiency.append("trajectory_not_fit_eligible")

    estimate = _parameter_estimate(row, parameter_name)
    parameter_evidence = _parameter_fit_evidence(row, parameter_name, estimate)
    fit_complete = _bool(row.get("fit_complete"))
    fit_status = _status(row.get("fit_status"))
    if insufficiency:
        return {
            "row_id": row_id,
            "state": "measurement_or_tracking_insufficient",
            "reason_codes": list(dict.fromkeys(insufficiency)),
            "generation_status": generation or None,
            "generation_status_source": generation_source,
            "generation_review_provisional": generation_review["provisional_review"],
            "generation_hard_failure": generation_review["hard_failure"],
            "motion_status": motion or None,
            "motion_status_source": motion_source,
            "fit_status": fit_status or None,
            "parameter_fit_evidence": parameter_evidence,
            "estimate": estimate,
            "measurement_route": _text(row.get("simple_measurement_route")) or None,
        }

    if parameter_evidence["status"] == "model_mismatch":
        return {
            "row_id": row_id,
            "state": "clear_parameter_rule_failure",
            "reason_codes": parameter_evidence["reason_codes"] or [
                "target_parameter_rule_family_mismatch"
            ],
            "generation_status": generation or None,
            "generation_status_source": generation_source,
            "generation_review_provisional": generation_review["provisional_review"],
            "generation_hard_failure": generation_review["hard_failure"],
            "motion_status": motion or None,
            "motion_status_source": motion_source,
            "fit_status": fit_status or None,
            "parameter_fit_evidence": parameter_evidence,
            "estimate": None,
            "measurement_route": _text(row.get("simple_measurement_route")) or None,
        }

    if parameter_evidence["status"] != "accepted" or estimate is None:
        reasons = ["target_parameter_evidence_indeterminate"]
        if fit_complete is False:
            reasons.append("inverse_measurement_incomplete_for_target_parameter")
        elif fit_complete is None and fit_status not in _FIT_EVIDENCE_AVAILABLE:
            reasons.append("inverse_measurement_completion_unproven")
        return {
            "row_id": row_id,
            "state": "measurement_or_tracking_insufficient",
            "reason_codes": reasons,
            "generation_status": generation or None,
            "generation_status_source": generation_source,
            "generation_review_provisional": generation_review["provisional_review"],
            "generation_hard_failure": generation_review["hard_failure"],
            "motion_status": motion or None,
            "motion_status_source": motion_source,
            "fit_status": fit_status or None,
            "parameter_fit_evidence": parameter_evidence,
            "estimate": estimate,
            "measurement_route": _text(row.get("simple_measurement_route")) or None,
        }

    return {
        "row_id": row_id,
        "state": "usable",
        "reason_codes": (
            ["soft_review_provisionally_accepted"]
            if generation_review["provisional_review"]
            else []
        ),
        "generation_status": generation,
        "generation_status_source": generation_source,
        "generation_review_provisional": generation_review["provisional_review"],
        "generation_hard_failure": generation_review["hard_failure"],
        "motion_status": motion or "not_explicitly_reported",
        "motion_status_source": motion_source,
        "fit_status": fit_status or None,
        "parameter_fit_evidence": parameter_evidence,
        "estimate": estimate,
        "measurement_route": _text(row.get("simple_measurement_route")) or "calibrated_2d",
    }


def _classify_video_row(row: Mapping[str, Any], *, row_id: str) -> dict[str, Any]:
    """Paper-facing video gate independent of any target parameter column."""

    generation, generation_source = _generation_status(row)
    generation_review = _generation_review_state(row, generation, generation_source)
    motion, motion_source = _motion_status(row)
    dynamic_decision, dynamic_reasons = _dynamic_3d_decision(row)
    dynamic_measurement_substitute = bool(
        dynamic_decision == "include" and not generation_review["hard_failure"]
    )
    reasons: list[str] = []
    if generation_source == "manual" and generation in _FAIL:
        grade = "L1"
        reasons.append("clear_generation_validity_failure")
    elif motion_source == "manual" and motion in _FAIL:
        grade = "L1"
        reasons.append("clear_motion_type_failure")
    elif dynamic_decision == "X":
        grade = "X"
        reasons.extend(dynamic_reasons)
    elif generation_review["hard_failure"]:
        grade = "X"
        reasons.append("automatic_failure_pending_manual_confirmation")
    elif motion in _FAIL:
        grade = "X"
        reasons.append("automatic_motion_failure_pending_manual_confirmation")
    elif (
        generation not in _PASS
        and not generation_review["provisional_review"]
        and not dynamic_measurement_substitute
    ):
        grade = "X"
        reasons.append("generation_validity_evidence_insufficient")
    else:
        fit_complete = _bool(row.get("fit_complete"))
        fit_status = _status(row.get("fit_status"))
        trajectory_eligible = _bool(row.get("trajectory_fit_eligible"))
        if trajectory_eligible is False:
            grade = "X"
            reasons.append("trajectory_or_inverse_measurement_unavailable")
        elif fit_complete is False and fit_status not in _FIT_EVIDENCE_AVAILABLE:
            grade = "X"
            reasons.append("inverse_measurement_unavailable")
        elif fit_complete is None and fit_status not in _FIT_EVIDENCE_AVAILABLE:
            grade = "X"
            reasons.append("inverse_measurement_completion_unproven")
        else:
            grade = "PASS_TO_SCAN"
            reasons.append(
                "soft_review_provisionally_accepted"
                if generation_review["provisional_review"]
                else (
                    "motion_generated_and_rule_failure_evidence_available"
                    if fit_status == "model_mismatch"
                    else "motion_generated_and_inverse_measurement_available"
                )
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "experiment_id": _text(row.get("experiment_id")),
        "parameter_tuple_id": _text(row.get("parameter_tuple_id")),
        "scene_id": _text(row.get("scene_id")),
        "camera_name": _text(row.get("camera_name")),
        "seed": _seed(row.get("seed")),
        "grade": grade,
        "grade_label": GRADE_LABELS.get(grade, "motion_generated_and_measurable"),
        "reason_codes": reasons,
        "generation_status": generation or None,
        "generation_status_source": generation_source,
        "generation_review_provisional": generation_review["provisional_review"],
        "generation_hard_failure": generation_review["hard_failure"],
        "motion_status": motion or None,
        "motion_status_source": motion_source,
        "measurement_route": _text(row.get("simple_measurement_route")) or (
            "qualified_dynamic_3d" if dynamic_decision == "include" else "calibrated_2d"
        ),
    }


def _registry_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for experiment in registry.get("experiments", []):
        experiment_id = _text(experiment.get("id"))
        if not experiment_id:
            continue
        parameters: dict[str, dict[str, Any]] = {}
        for parameter in experiment.get("hidden_parameters", []):
            name = _text(parameter.get("name"))
            valid_range = parameter.get("valid_range")
            if not name or not isinstance(valid_range, Sequence) or len(valid_range) != 2:
                continue
            low, high = _number(valid_range[0]), _number(valid_range[1])
            if low is None or high is None:
                continue
            parameters[name] = {
                **dict(parameter),
                "name": name,
                "valid_range": [low, high],
            }
        anchors: dict[str, dict[str, float]] = {}
        for anchor in experiment.get("anchor_tuples", []):
            anchor_id = _text(anchor.get("id"))
            values = {
                name: value
                for name in parameters
                if (value := _number(anchor.get(name))) is not None
            }
            if anchor_id and len(values) == len(parameters):
                anchors[anchor_id] = values
        output[experiment_id] = {
            "id": experiment_id,
            "parameters": parameters,
            "anchors": anchors,
        }
    return output


def _reference_anchor(spec: Mapping[str, Any]) -> dict[str, float]:
    anchors = spec["anchors"]
    if "default" in anchors:
        return dict(anchors["default"])
    parameters = spec["parameters"]

    def distance(anchor: Mapping[str, float]) -> float:
        total = 0.0
        for name, parameter in parameters.items():
            low, high = parameter["valid_range"]
            span = max(high - low, 1e-12)
            total += abs(float(anchor[name]) - 0.5 * (low + high)) / span
        return total

    return dict(min(anchors.values(), key=distance)) if anchors else {}


def _signature(values: Mapping[str, float]) -> str:
    payload = json.dumps({key: values[key] for key in sorted(values)}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _canonical_oat_family(
    spec: Mapping[str, Any],
    parameter_name: str,
) -> tuple[dict[str, float], dict[float, list[str]]] | None:
    parameters = spec["parameters"]
    anchors = spec["anchors"]
    other_names = sorted(name for name in parameters if name != parameter_name)
    families: dict[tuple[float, ...], dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))
    for anchor_id, values in anchors.items():
        if parameter_name not in values or any(name not in values for name in other_names):
            continue
        key = tuple(float(values[name]) for name in other_names)
        families[key][float(values[parameter_name])].append(anchor_id)
    candidates = [
        (key, levels)
        for key, levels in families.items()
        if len(levels) >= int(DEFAULT_THRESHOLDS["minimum_usable_levels"])
    ]
    if not candidates:
        return None
    reference = _reference_anchor(spec)

    def rank(candidate: tuple[tuple[float, ...], dict[float, list[str]]]) -> tuple[Any, ...]:
        key, levels = candidate
        fixed = dict(zip(other_names, key))
        reference_distance = 0.0
        for name, value in fixed.items():
            low, high = parameters[name]["valid_range"]
            reference_distance += abs(value - float(reference.get(name, value))) / max(high - low, 1e-12)
        return (-len(levels), reference_distance, key)

    fixed_key, level_map = min(candidates, key=rank)
    return dict(zip(other_names, fixed_key)), {target: sorted(ids) for target, ids in level_map.items()}


def _median_pairwise_slope(targets: Sequence[float], estimates: Sequence[float]) -> float | None:
    slopes: list[float] = []
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            delta = targets[right] - targets[left]
            if delta != 0.0:
                slopes.append((estimates[right] - estimates[left]) / delta)
    return float(median(slopes)) if slopes else None


def _response_metrics(levels: Sequence[Mapping[str, Any]], valid_span: float) -> dict[str, Any]:
    usable = [level for level in levels if level.get("median_estimate") is not None]
    usable = sorted(usable, key=lambda level: float(level["target_value"]))
    targets = [float(level["target_value"]) for level in usable]
    estimates = [float(level["median_estimate"]) for level in usable]
    concordant = 0
    pair_count = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            pair_count += 1
            if (targets[right] - targets[left]) * (estimates[right] - estimates[left]) > 0.0:
                concordant += 1
    concordance = concordant / pair_count if pair_count else None
    slope = _median_pairwise_slope(targets, estimates)
    level_naes = [abs(estimate - target) / valid_span for target, estimate in zip(targets, estimates)]
    return {
        "pairwise_direction_concordance": concordance,
        "concordant_pair_count": concordant,
        "pair_count": pair_count,
        "theil_sen_slope": slope,
        "per_level_valid_range_nae": level_naes,
        "median_valid_range_nae": float(median(level_naes)) if level_naes else None,
    }


def _grade_scan(
    *,
    experiment_id: str,
    parameter_name: str,
    parameter_spec: Mapping[str, Any],
    fixed: Mapping[str, float],
    level_map: Mapping[float, Sequence[str]],
    rows_by_tuple: Mapping[tuple[str, str], Sequence[tuple[int, Mapping[str, Any]]]],
    primary_seed: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    levels: list[dict[str, Any]] = []
    clear_failures: list[dict[str, Any]] = []
    parameter_rule_failures: list[dict[str, Any]] = []
    insufficient_rows: list[dict[str, Any]] = []
    for target, anchor_ids in sorted(level_map.items()):
        row_evidence: list[dict[str, Any]] = []
        for anchor_id in anchor_ids:
            matches = list(rows_by_tuple.get((experiment_id, anchor_id), []))
            if not matches:
                evidence = {
                    "row_id": f"missing:{experiment_id}:{anchor_id}",
                    "parameter_tuple_id": anchor_id,
                    "state": "measurement_or_tracking_insufficient",
                    "reason_codes": ["primary_slice_row_missing"],
                    "estimate": None,
                }
                row_evidence.append(evidence)
                insufficient_rows.append(evidence)
                continue
            for ordinal, row in matches:
                evidence = _classify_row(row, parameter_name, row_id=_row_id(row, ordinal))
                evidence["parameter_tuple_id"] = anchor_id
                row_evidence.append(evidence)
                if evidence["state"] == "clear_model_failure":
                    clear_failures.append(evidence)
                elif evidence["state"] == "clear_parameter_rule_failure":
                    parameter_rule_failures.append(evidence)
                elif evidence["state"] == "measurement_or_tracking_insufficient":
                    insufficient_rows.append(evidence)
        estimates = [
            float(item["estimate"])
            for item in row_evidence
            if item["state"] == "usable" and item.get("estimate") is not None
        ]
        levels.append(
            {
                "target_value": float(target),
                "parameter_tuple_ids": list(anchor_ids),
                "planned_row_count": len(anchor_ids),
                "observed_row_count": sum(not item["row_id"].startswith("missing:") for item in row_evidence),
                "usable_row_count": len(estimates),
                "parameter_rule_failure_row_count": sum(
                    item["state"] == "clear_parameter_rule_failure"
                    for item in row_evidence
                ),
                "estimates": estimates,
                "median_estimate": float(median(estimates)) if estimates else None,
                "rows": row_evidence,
            }
        )

    low, high = (float(value) for value in parameter_spec["valid_range"])
    valid_span = high - low
    usable_levels = [level for level in levels if level["median_estimate"] is not None]
    evidenced_levels = [
        level for level in levels
        if level["median_estimate"] is not None
        or level["parameter_rule_failure_row_count"] > 0
    ]
    metrics = _response_metrics(levels, valid_span) if valid_span > 0.0 else {}
    reasons: list[str]
    if valid_span <= 0.0:
        grade = "X"
        reasons = ["parameter_valid_range_is_not_positive"]
    elif len(evidenced_levels) < int(thresholds["minimum_usable_levels"]):
        grade = "X"
        reasons = ["fewer_than_two_levels_with_measurement_or_rule_failure_evidence"]
        reasons.extend(
            reason
            for item in insufficient_rows
            for reason in item.get("reason_codes", [])
        )
        reasons = list(dict.fromkeys(reasons))
    elif parameter_rule_failures:
        grade = "L2"
        reasons = ["one_or_more_parameter_levels_fail_the_target_rule_family"]
        reasons.extend(
            reason
            for item in parameter_rule_failures
            for reason in item.get("reason_codes", [])
        )
        reasons = list(dict.fromkeys(reasons))
    elif len(usable_levels) < int(thresholds["minimum_usable_levels"]):
        grade = "X"
        reasons = ["fewer_than_two_numerically_usable_parameter_levels"]
    else:
        concordance = metrics["pairwise_direction_concordance"]
        slope = metrics["theil_sen_slope"]
        direction_pass = bool(
            concordance is not None
            and concordance > float(thresholds["minimum_pairwise_direction_concordance_exclusive"])
            and slope is not None
            and slope > float(thresholds["minimum_slope_exclusive"])
        )
        if not direction_pass:
            grade = "L2"
            reasons = []
            if concordance is None or concordance <= float(thresholds["minimum_pairwise_direction_concordance_exclusive"]):
                reasons.append("pairwise_direction_concordance_not_above_half")
            if slope is None or slope <= float(thresholds["minimum_slope_exclusive"]):
                reasons.append("non_positive_theil_sen_slope")
        elif (
            float(metrics["theil_sen_slope"]) >= float(thresholds["l4_min_slope_inclusive"])
            and metrics["median_valid_range_nae"]
            <= float(thresholds["l4_max_median_valid_range_nae_inclusive"])
        ):
            grade = "L4"
            reasons = ["positive_direction_response_magnitude_and_median_nae_within_threshold"]
        else:
            grade = "L3"
            reasons = ["positive_direction_but_response_magnitude_or_numeric_error_outside_l4_threshold"]

    fixed_signature = _signature(fixed)
    return {
        "schema_version": SCHEMA_VERSION,
        "scan_id": f"{experiment_id}__{parameter_name}__baseline__CAM_Side__seed-{primary_seed}__fixed-{fixed_signature}",
        "experiment_id": experiment_id,
        "target_parameter": parameter_name,
        "parameter_unit": parameter_spec.get("unit"),
        "valid_range": [low, high],
        "fixed_other_parameters": dict(fixed),
        "scene_id": PRIMARY_SCENE_ID,
        "camera_name": PRIMARY_CAMERA_NAME,
        "seed": primary_seed,
        "grade": grade,
        "grade_label": GRADE_LABELS[grade],
        "reason_codes": reasons,
        "evidence": {
            "planned_level_count": len(levels),
            "usable_level_count": len(usable_levels),
            "clear_failure_row_count": len(clear_failures),
            "parameter_rule_failure_row_count": len(parameter_rule_failures),
            "measurement_or_tracking_insufficient_row_count": len(insufficient_rows),
            "levels": levels,
            **metrics,
        },
        "thresholds": dict(thresholds),
        "scope_note": "Primary baseline/CAM_Side/primary-seed OAT scan only; X is excluded from model failure.",
    }


def grade_simple_paper_benchmark(
    all_jobs_rows: Sequence[Mapping[str, Any]],
    experiment_registry: Mapping[str, Any],
    *,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Grade the paper's primary baseline parameter-response benchmark.

    Args:
        all_jobs_rows: Rows loaded from ``all_jobs.csv`` (for example with
            ``csv.DictReader``).  Extra columns are ignored.
        experiment_registry: Parsed experiment-registry JSON object.
        primary_seed: Frozen main seed.  Seed variation is intentionally not
            evaluated by this module.

    Returns:
        A JSON-serializable mapping with ``per_scan_rows`` and ``summary``.
    """

    effective_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for key in effective_thresholds:
            value = _number(thresholds.get(key))
            if value is not None:
                effective_thresholds[key] = value

    primary_rows: list[tuple[int, Mapping[str, Any]]] = []
    excluded_counts: Counter[str] = Counter()
    for ordinal, row in enumerate(all_jobs_rows):
        if _text(row.get("scene_id")) != PRIMARY_SCENE_ID:
            excluded_counts["non_baseline_scene"] += 1
            continue
        if _text(row.get("camera_name")) != PRIMARY_CAMERA_NAME:
            excluded_counts["non_side_camera"] += 1
            continue
        if _seed(row.get("seed")) != int(primary_seed):
            excluded_counts["non_primary_seed"] += 1
            continue
        primary_rows.append((ordinal, row))

    rows_by_tuple: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for ordinal, row in primary_rows:
        experiment_id = _text(row.get("experiment_id"))
        tuple_id = _text(row.get("parameter_tuple_id"))
        if experiment_id and tuple_id:
            rows_by_tuple[(experiment_id, tuple_id)].append((ordinal, row))

    per_video_rows = [
        _classify_video_row(row, row_id=_row_id(row, ordinal))
        for ordinal, row in primary_rows
    ]
    scans: list[dict[str, Any]] = []
    registry = _registry_index(experiment_registry)
    for experiment_id, spec in sorted(registry.items()):
        for parameter_name, parameter_spec in sorted(spec["parameters"].items()):
            family = _canonical_oat_family(spec, parameter_name)
            if family is None:
                scans.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scan_id": f"{experiment_id}__{parameter_name}__baseline__CAM_Side__seed-{primary_seed}__no-oat",
                        "experiment_id": experiment_id,
                        "target_parameter": parameter_name,
                        "parameter_unit": parameter_spec.get("unit"),
                        "valid_range": list(parameter_spec["valid_range"]),
                        "fixed_other_parameters": {},
                        "scene_id": PRIMARY_SCENE_ID,
                        "camera_name": PRIMARY_CAMERA_NAME,
                        "seed": primary_seed,
                        "grade": "X",
                        "grade_label": GRADE_LABELS["X"],
                        "reason_codes": ["honest_oat_scan_not_defined_by_registry"],
                        "evidence": {
                            "planned_level_count": 0,
                            "usable_level_count": 0,
                            "levels": [],
                        },
                        "thresholds": dict(effective_thresholds),
                        "scope_note": "X denotes insufficient benchmark design/evidence, not model failure.",
                    }
                )
                continue
            fixed, level_map = family
            scans.append(
                _grade_scan(
                    experiment_id=experiment_id,
                    parameter_name=parameter_name,
                    parameter_spec=parameter_spec,
                    fixed=fixed,
                    level_map=level_map,
                    rows_by_tuple=rows_by_tuple,
                    primary_seed=int(primary_seed),
                    thresholds=effective_thresholds,
                )
            )

    scans.sort(key=lambda row: (row["experiment_id"], row["target_parameter"], row["scan_id"]))
    grade_counts = {grade: 0 for grade in ("L2", "L3", "L4", "X")}
    grade_counts.update(Counter(str(scan["grade"]) for scan in scans))
    non_x = len(scans) - grade_counts["X"]
    video_grade_counts = {grade: 0 for grade in ("L1", "PASS_TO_SCAN", "X")}
    video_grade_counts.update(Counter(str(row["grade"]) for row in per_video_rows))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "grading_scheme": "simple_paper_four_level_v2_segmented_parameters",
        "scope": {
            "scene_id": PRIMARY_SCENE_ID,
            "camera_name": PRIMARY_CAMERA_NAME,
            "primary_seed": int(primary_seed),
            "scan_design": "one canonical honest OAT scan per experiment and target parameter",
        },
        "thresholds": dict(effective_thresholds),
        "input_row_count": len(all_jobs_rows),
        "selected_primary_slice_row_count": len(primary_rows),
        "excluded_input_row_counts": dict(sorted(excluded_counts.items())),
        "experiment_count": len(registry),
        "total_scan_count": len(scans),
        "grade_counts": grade_counts,
        "video_grade_counts": video_grade_counts,
        "non_x_scan_count": non_x,
        "x_scan_count": grade_counts["X"],
        "x_excluded_from_model_failure_denominator": True,
        "positive_parameter_response_scan_count": grade_counts["L3"] + grade_counts["L4"],
        "numeric_accurate_scan_count": grade_counts["L4"],
        "grade_definitions": dict(GRADE_LABELS),
    }
    return {"per_video_rows": per_video_rows, "per_scan_rows": scans, "summary": summary}


__all__ = [
    "DEFAULT_PRIMARY_SEED",
    "DEFAULT_THRESHOLDS",
    "GRADE_LABELS",
    "grade_simple_paper_benchmark",
]
