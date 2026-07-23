"""Trajectory-to-parameter inverse models for all 13 frozen experiments.

This module intentionally has no video, image, or ground-truth-anchor input.
It consumes only a metric trajectory and experiment-known constants.  Target
parameters are looked up later by :func:`score_parameter_fit`, which prevents
ground-truth leakage into the inverse fit.

The estimators use robust linear/integral forms where possible.  They are
deliberately NumPy-only so the benchmark does not require SciPy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .motion_segmentation import (
    combine_rule_checks,
    complete_cycle_segments,
    first_contact_index,
    impact_dwell_diagnostics,
    linear_motion_diagnostics,
    motion_start_index,
    motion_stop_index,
    smooth_series,
    turning_indices,
)


EXPERIMENT_CONSTANTS: dict[str, dict[str, Any]] = {
    "v1_A": {"contact_z": 0.44},
    "v1_B": {"wall_contact_x": 2.96},
    "v1_C": {"gravity_g": 9.81},
    "v1_D": {"pivot": [0.0, 0.0, 3.8], "length": 2.3, "period": 3.0},
    "v2_A": {"cycloid_a": 0.65},
    "v2_B": {"left_wall_x": -2.75, "right_wall_x": 2.75},
    "v2_C": {"transition_x": 0.0, "normal_over_mass": 1.0},
    "v2_D": {"pivot": [0.0, 0.0, 3.28], "length": 1.85},
    "v2_E": {"contact_z": 0.82},
    "v3_A": {"contact_z": 0.82},
    "v3_B": {"left_wall_x": -1.72, "right_wall_x": 1.72, "gravity_g": 9.8},
    "v3_C": {
        "ramp_angle_deg": 7.0,
        "floor_transition_x": -0.5,
        "wall_contact_x": 4.61,
    },
    "v3_D": {
        "pivot": [0.0, 0.0, 4.05],
        "length": 1.35,
        "magnet_center_deg": 38.0,
        "magnet_width_deg": 18.0,
    },
}


def _finite_trajectory(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values: list[tuple[float, float, float, int]] = []
    for ordinal, row in enumerate(rows):
        # Reconstructors may retain diagnostic xyz values for frames rejected
        # by the known-sphere size or geometry gate.  Those values are useful
        # in overlays but must never leak into the inverse physics fit.
        if row.get("measurement_valid") is False or row.get("physics_fit_used") is False:
            continue
        if str(row.get("interpolated", "")).strip().lower() in {"1", "true", "yes"}:
            continue
        try:
            time_s = float(row["time_s"])
            x_m = float(row["x_m"])
            z_m = float(row["z_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(time_s) and math.isfinite(x_m) and math.isfinite(z_m):
            try:
                frame = int(row.get("source_frame_index", row.get("frame_index", ordinal)))
            except (TypeError, ValueError):
                frame = int(ordinal)
            values.append((time_s, x_m, z_m, frame))
    if not values:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=int)
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    unique = np.r_[True, np.diff(array[:, 0]) > 1e-9]
    return (
        array[unique, 0],
        array[unique, 1],
        array[unique, 2],
        array[unique, 3].astype(int),
    )


def _mad_scale(residual: np.ndarray) -> float:
    if not len(residual):
        return 1.0
    center = float(np.median(residual))
    scale = 1.4826 * float(np.median(np.abs(residual - center)))
    return max(scale, 1e-8)


def _robust_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 12,
    huber_k: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if design.ndim != 2 or design.shape[0] != len(target):
        raise ValueError("design and target shapes do not match")
    if design.shape[0] < design.shape[1]:
        raise ValueError("underdetermined robust least-squares system")
    weights = np.ones(len(target), dtype=np.float64)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(iterations):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
        residual = target - design @ coefficients
        scale = _mad_scale(residual)
        normalized = np.abs(residual) / (huber_k * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-12))
    return coefficients, weights


def _fit_diagnostics(
    observed: np.ndarray,
    predicted: np.ndarray,
    parameter_count: int,
    *,
    time_s: np.ndarray | None = None,
    series_name: str = "q",
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
    centered = np.asarray(observed) - float(np.mean(observed)) if len(observed) else np.empty(0)
    denominator = float(np.sum(centered**2))
    r2 = None if denominator <= 1e-12 else float(1.0 - np.sum(residual**2) / denominator)
    output: dict[str, Any] = {
        "fit_points": int(len(observed)),
        "fit_parameter_count": int(parameter_count),
        "fit_rmse": rmse,
        "fit_r2": r2,
        "observed_range": float(np.ptp(observed)) if len(observed) else None,
        "fit_nrmse": (
            None
            if rmse is None or not len(observed) or float(np.ptp(observed)) <= 1e-12
            else float(rmse / float(np.ptp(observed)))
        ),
    }
    if time_s is not None:
        sample_time = np.asarray(time_s, dtype=np.float64).reshape(-1)
        if len(sample_time) != len(observed):
            raise ValueError("fit diagnostic time and observation lengths do not match")
    else:
        sample_time = np.arange(len(observed), dtype=np.float64)
    output["fit_series"] = {
        "series_name": str(series_name),
        "time_s": [float(value) for value in sample_time],
        "observed": [float(value) for value in observed],
        "predicted": [float(value) for value in predicted],
        "residual": [float(value) for value in residual],
        "time_basis": "video_time_s" if time_s is not None else "fit_sample_index",
    }
    return output


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    return smooth_series(values, window=window)


def _turning_indices(values: np.ndarray, min_separation: int = 3) -> list[int]:
    return turning_indices(values, min_separation=min_separation)


def _segments_from_turns(length: int, turns: Sequence[int], guard: int = 1) -> list[np.ndarray]:
    boundaries = [0] + [int(value) for value in turns] + [length - 1]
    segments: list[np.ndarray] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = left + (guard if left > 0 else 0)
        stop = right - (guard if right < length - 1 else 0)
        if stop - start + 1 >= 4:
            segments.append(np.arange(start, stop + 1, dtype=int))
    return segments


def _pendulum_angle(x: np.ndarray, z: np.ndarray, pivot: Sequence[float], sign: float = 1.0) -> np.ndarray:
    px, _, pz = (float(value) for value in pivot)
    return np.unwrap(sign * np.arctan2(x - px, pz - z))


def _cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float64)
    if len(values) > 1:
        output[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(times))
    return output


def _failure(experiment_id: str, reason: str, parameters: Sequence[str]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": "insufficient_evidence",
        "reason": reason,
        "fit_validity": {
            "status": "insufficient_evidence",
            "category": "measurement_or_segmentation",
            "primary_reason_code": reason,
            "secondary_reason_codes": [],
            "target_free": True,
        },
        "raw_parameter_estimates": {name: None for name in parameters},
        "parameter_estimates": {name: None for name in parameters},
        "parameter_observed": {name: False for name in parameters},
        "parameter_attribution": {
            name: {"status": "indeterminate", "reason_codes": [reason]}
            for name in parameters
        },
        "rule_family_evaluation": {
            "status": "indeterminate",
            "reason_codes": [reason],
            "target_parameters_used": False,
        },
        "diagnostics": {},
        "target_not_used_for_fit": True,
    }


def _result(
    experiment_id: str,
    estimates: Mapping[str, float | None],
    diagnostics: Mapping[str, Any],
    *,
    observed: Mapping[str, bool] | None = None,
    method: str,
    parameter_quality: Mapping[str, Mapping[str, Any]] | None = None,
    rule_family: Mapping[str, Any] | None = None,
    segmentation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw: dict[str, float | None] = {}
    for name, value in estimates.items():
        raw[name] = float(value) if value is not None and math.isfinite(float(value)) else None
    observed_map = {
        name: bool(raw[name] is not None) if observed is None else bool(
            observed.get(name, False) and raw[name] is not None
        )
        for name in raw
    }
    attribution: dict[str, dict[str, Any]] = {}
    for name in raw:
        quality = dict((parameter_quality or {}).get(name, {}))
        quality_status = str(quality.get("status", "pass" if observed_map[name] else "indeterminate"))
        quality.setdefault("status", quality_status)
        quality.setdefault("reason_codes", [])
        attribution[name] = quality
        if quality_status != "pass":
            observed_map[name] = False
    family = dict(rule_family or {})
    if not family:
        family = {
            "status": "pass" if all(observed_map.values()) else "indeterminate",
            "rule_family": experiment_id,
            "reason_codes": [],
            "target_parameters_used": False,
        }
    family_status = str(family.get("status", "indeterminate"))
    if family_status != "pass":
        observed_map = {name: False for name in observed_map}
    clean = {
        name: raw[name] if observed_map[name] else None
        for name in raw
    }
    diagnostic_payload = dict(diagnostics)
    if any(raw[name] is not None and clean[name] is None for name in raw):
        diagnostic_payload["candidate_parameter_estimates"] = dict(raw)
    parameter_mismatch = any(
        value.get("status") == "fail" for value in attribution.values()
    )
    if family_status == "fail":
        status = "model_mismatch"
        validity_status = "rejected"
        validity_category = "rule_family_mismatch"
    elif all(observed_map.values()):
        status = "ok"
        validity_status = "accepted"
        validity_category = "ok"
    elif any(observed_map.values()):
        status = "partial"
        validity_status = "partially_accepted"
        validity_category = (
            "parameter_specific_rule_mismatch"
            if parameter_mismatch
            else "parameter_specific_identifiability"
        )
    elif parameter_mismatch:
        status = "model_mismatch"
        validity_status = "rejected"
        validity_category = "parameter_specific_rule_mismatch"
    else:
        status = "insufficient_evidence"
        validity_status = "insufficient_evidence"
        validity_category = "unidentifiable"
    family_reasons = [str(value) for value in family.get("reason_codes", [])]
    parameter_reasons = [
        f"{name}:{reason}"
        for name, value in attribution.items()
        if value.get("status") == "fail"
        for reason in value.get("reason_codes", [])
    ]
    validity_reasons = family_reasons or parameter_reasons
    return {
        "experiment_id": experiment_id,
        "status": status,
        "method": method,
        "fit_validity": {
            "status": validity_status,
            "category": validity_category,
            "primary_reason_code": validity_reasons[0] if validity_reasons else None,
            "secondary_reason_codes": validity_reasons[1:],
            "target_free": True,
        },
        "raw_parameter_estimates": raw,
        "parameter_estimates": clean,
        "parameter_observed": observed_map,
        "parameter_attribution": attribution,
        "rule_family_evaluation": family,
        "segmentation": dict(segmentation or {}),
        "diagnostics": diagnostic_payload,
        "target_not_used_for_fit": True,
    }


def _fit_quadratic(
    times: np.ndarray,
    values: np.ndarray,
    *,
    series_name: str = "q",
) -> tuple[np.ndarray, dict[str, Any]]:
    tau = times - times[0]
    design = np.column_stack((np.ones(len(tau)), tau, tau**2))
    coefficients, _ = _robust_lstsq(design, values)
    return coefficients, _fit_diagnostics(
        values,
        design @ coefficients,
        3,
        time_s=times,
        series_name=series_name,
    )


def _model_fit_check(
    diagnostics: Mapping[str, Any],
    *,
    maximum_nrmse: float = 0.12,
    minimum_r2: float = 0.75,
    label: str = "trajectory_model",
) -> dict[str, Any]:
    nrmse = diagnostics.get("fit_nrmse")
    r2 = diagnostics.get("fit_r2")
    if nrmse is None or r2 is None:
        return {
            "status": "indeterminate",
            "reason_codes": [f"{label}_fit_quality_unavailable"],
            "fit_nrmse": nrmse,
            "fit_r2": r2,
        }
    reasons: list[str] = []
    if float(nrmse) > maximum_nrmse:
        reasons.append(f"{label}_nrmse_exceeds_limit")
    if float(r2) < minimum_r2:
        reasons.append(f"{label}_r2_below_limit")
    return {
        "status": "fail" if reasons else "pass",
        "reason_codes": reasons,
        "fit_nrmse": float(nrmse),
        "fit_r2": float(r2),
        "maximum_nrmse": float(maximum_nrmse),
        "minimum_r2": float(minimum_r2),
    }


def _quadratic_acceleration_stability(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    maximum_relative_difference: float = 0.40,
) -> dict[str, Any]:
    if len(time_s) < 12:
        return {
            "status": "indeterminate",
            "reason_codes": ["acceleration_stability_support_insufficient"],
        }
    middle = len(time_s) // 2
    blocks = (np.arange(0, middle + 1), np.arange(middle, len(time_s)))
    accelerations: list[float] = []
    for block in blocks:
        tau = time_s[block] - time_s[block][0]
        coefficients = np.polyfit(tau, values[block], 2)
        accelerations.append(float(2.0 * coefficients[0]))
    reference = max(float(np.mean(np.abs(accelerations))), 1e-8)
    difference = abs(accelerations[1] - accelerations[0]) / reference
    failed = difference > maximum_relative_difference
    return {
        "status": "fail" if failed else "pass",
        "reason_codes": ["acceleration_nonstationary"] if failed else [],
        "first_half_acceleration": accelerations[0],
        "second_half_acceleration": accelerations[1],
        "relative_difference": difference,
        "maximum_relative_difference": float(maximum_relative_difference),
    }


def _domain_check(
    value: float | None,
    *,
    lower: float | None = None,
    upper: float | None = None,
    label: str,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if value is None or not math.isfinite(float(value)):
        return {
            "status": "indeterminate",
            "reason_codes": [f"{label}_unidentifiable"],
            "value": value,
        }
    reasons: list[str] = []
    if lower is not None and float(value) < lower - tolerance:
        reasons.append(f"{label}_below_physical_domain")
    if upper is not None and float(value) > upper + tolerance:
        reasons.append(f"{label}_above_physical_domain")
    return {
        "status": "fail" if reasons else "pass",
        "reason_codes": reasons,
        "value": float(value),
        "physical_domain": [lower, upper],
    }


def _fit_v1a(t: np.ndarray, _x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    contact = float(EXPERIMENT_CONSTANTS["v1_A"]["contact_z"])
    start = motion_start_index(t, z, direction=-1)
    contact_index = first_contact_index(
        z,
        contact_position=contact,
        start_index=start,
        direction=-1,
    )
    if contact_index is None:
        return _failure("v1_A", "first_floor_contact_not_observed", ["gravity_g"])
    indices = np.arange(start, contact_index + 1, dtype=int)
    if len(indices) < 8:
        return _failure(
            "v1_A",
            "fewer_than_8_points_from_motion_start_to_first_contact",
            ["gravity_g"],
        )
    coefficients, diagnostics = _fit_quadratic(t[indices], z[indices], series_name="z_m")
    gravity = -2.0 * float(coefficients[2])
    velocity = np.gradient(_smooth(z[indices], window=3), t[indices])
    direction_check = {
        "status": "pass" if float(np.mean(velocity <= 0.02)) >= 0.90 else "fail",
        "reason_codes": (
            []
            if float(np.mean(velocity <= 0.02)) >= 0.90
            else ["free_fall_direction_not_sustained"]
        ),
        "non_upward_fraction": float(np.mean(velocity <= 0.02)),
    }
    fit_check = _model_fit_check(
        diagnostics,
        maximum_nrmse=0.10,
        minimum_r2=0.85,
        label="first_free_fall_arc",
    )
    stationarity = _quadratic_acceleration_stability(t[indices], z[indices])
    gravity_domain = _domain_check(gravity, lower=0.0, label="gravity")
    rule = combine_rule_checks(
        {
            "downward_motion": direction_check,
            "quadratic_trajectory": fit_check,
            "constant_acceleration": stationarity,
            "physical_gravity": gravity_domain,
        },
        rule_family="first_free_fall_to_first_contact",
    )
    diagnostics.update(
        {
            "motion_start_index": int(start),
            "first_contact_index": int(contact_index),
            "discarded_static_prefix_points": int(start),
            "discarded_post_contact_points": int(len(t) - contact_index - 1),
            "constant_acceleration_check": stationarity,
        }
    )
    return _result(
        "v1_A",
        {"gravity_g": gravity},
        diagnostics,
        method="first_motion_to_first_contact_quadratic",
        parameter_quality={"gravity_g": rule},
        rule_family=rule,
        segmentation={
            "algorithm": "first_sustained_downward_motion_to_first_floor_contact",
            "status": "ok",
            "events": [
                {"name": "motion_start", "index": int(start), "time_s": float(t[start])},
                {
                    "name": "first_floor_contact",
                    "index": int(contact_index),
                    "time_s": float(t[contact_index]),
                },
            ],
            "segments": [
                {
                    "name": "first_free_fall",
                    "start_index": int(start),
                    "end_index": int(contact_index),
                    "point_count": int(len(indices)),
                }
            ],
        },
    )


def _best_piecewise_linear_impact(
    t: np.ndarray,
    q: np.ndarray,
    *,
    expected_position: float | None = None,
    minimum_side: int = 5,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    scale = max(float(np.ptp(q)), 0.1)
    for split in range(minimum_side, len(q) - minimum_side):
        left = np.arange(0, split)
        right = np.arange(split, len(q))
        left_coef = np.polyfit(t[left], q[left], 1)
        right_coef = np.polyfit(t[right], q[right], 1)
        v_pre, v_post = float(left_coef[0]), float(right_coef[0])
        if v_pre * v_post >= 0:
            continue
        pred_left = np.polyval(left_coef, t[left])
        pred_right = np.polyval(right_coef, t[right])
        predicted = np.empty_like(q, dtype=np.float64)
        predicted[left] = pred_left
        predicted[right] = pred_right
        loss = float(np.mean((q[left] - pred_left) ** 2) + np.mean((q[right] - pred_right) ** 2))
        impact_position = 0.5 * (float(np.polyval(left_coef, t[split])) + float(np.polyval(right_coef, t[split])))
        if expected_position is not None:
            loss += 0.15 * ((impact_position - expected_position) / scale) ** 2
        candidate = {
            "split_index": split,
            "impact_time_s": float(t[split]),
            "impact_position_m": impact_position,
            "v_pre": v_pre,
            "v_post": v_post,
            "loss": loss,
            "rmse": math.sqrt(max(loss, 0.0) / 2.0),
            **_fit_diagnostics(q, predicted, 4, time_s=t, series_name="x_m"),
        }
        if best is None or candidate["loss"] < best["loss"]:
            best = candidate
    return best


def _local_impact_velocity_ratio(
    t: np.ndarray,
    q: np.ndarray,
    expected_position: float,
    *,
    window: int = 10,
) -> dict[str, Any] | None:
    turns = _turning_indices(q, min_separation=3)
    if not turns:
        return None
    turn = min(turns, key=lambda index: abs(float(q[index]) - expected_position))
    local_start = max(0, turn - 4)
    local_stop = min(len(q), turn + 5)
    impact = local_start + int(np.argmin(np.abs(q[local_start:local_stop] - expected_position)))
    guard = 2
    left = np.arange(max(0, impact - window), max(0, impact - guard + 1))
    right = np.arange(min(len(q), impact + guard), min(len(q), impact + guard + window))
    if len(left) < 4 or len(right) < 4:
        return None
    degree_left = min(2, len(left) - 1)
    degree_right = min(2, len(right) - 1)
    left_coef = np.polyfit(t[left], q[left], degree_left)
    right_coef = np.polyfit(t[right], q[right], degree_right)
    event_time = float(t[impact])
    v_pre = float(np.polyval(np.polyder(left_coef), event_time))
    v_post = float(np.polyval(np.polyder(right_coef), event_time))
    if v_pre * v_post >= 0 or abs(v_pre) <= 1e-8:
        return None
    return {
        "split_index": int(impact),
        "impact_time_s": event_time,
        "impact_position_m": float(q[impact]),
        "v_pre": v_pre,
        "v_post": v_post,
        "velocity_ratio": abs(v_post / v_pre),
        "pre_fit_points": int(len(left)),
        "post_fit_points": int(len(right)),
    }


def _fit_v1b(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    if len(t) < 12:
        return _failure("v1_B", "fewer_than_12_points", ["restitution_e"])
    wall = float(EXPERIMENT_CONSTANTS["v1_B"]["wall_contact_x"])
    start = motion_start_index(t, x, direction=1)
    turns = _turning_indices(x[start:], min_separation=4)
    turns = [start + value for value in turns]
    significant_turns: list[int] = []
    minimum_excursion = max(0.05, 0.04 * float(np.ptp(x)))
    for raw_turn in turns:
        local = np.arange(max(start, raw_turn - 4), min(len(x), raw_turn + 5), dtype=int)
        turn = int(local[np.argmax(x[local])])
        left = max(start, turn - 5)
        right = min(len(x) - 1, turn + 5)
        if (
            abs(float(x[turn] - x[left])) >= minimum_excursion
            and abs(float(x[right] - x[turn])) >= minimum_excursion
        ):
            significant_turns.append(turn)
    if not significant_turns:
        return _failure("v1_B", "wall_velocity_reversal_not_found", ["restitution_e"])
    impact = min(significant_turns, key=lambda index: abs(float(x[index]) - wall))
    local_start = max(start, impact - 3)
    local_stop = min(len(x), impact + 4)
    impact = local_start + int(np.argmax(x[local_start:local_stop]))
    guard = 2
    pre_indices = np.arange(start, impact - guard + 1, dtype=int)
    post_indices = np.arange(impact + guard, len(x), dtype=int)
    if len(pre_indices) < 6 or len(post_indices) < 6:
        return _failure("v1_B", "impact_at_clip_boundary", ["restitution_e"])
    pre = linear_motion_diagnostics(t, x, pre_indices, series_name="preimpact_x_m")
    post = linear_motion_diagnostics(t, x, post_indices, series_name="postimpact_x_m")
    v_pre = pre.get("velocity")
    v_post = post.get("velocity")
    if v_pre is None or v_post is None or float(v_pre) <= 0 or float(v_post) >= 0:
        return _failure("v1_B", "wall_velocity_reversal_not_found", ["restitution_e"])
    estimate = abs(float(v_post) / float(v_pre))
    wall_tolerance = max(0.15, 0.08 * max(float(np.ptp(x)), 1.0))
    contact_check = {
        "status": "pass" if abs(float(x[impact]) - wall) <= wall_tolerance else "fail",
        "reason_codes": (
            []
            if abs(float(x[impact]) - wall) <= wall_tolerance
            else ["velocity_reversal_not_at_known_wall"]
        ),
        "impact_position_m": float(x[impact]),
        "known_wall_position_m": wall,
        "absolute_contact_error_m": abs(float(x[impact]) - wall),
        "maximum_contact_error_m": wall_tolerance,
    }
    topology_check = {
        "status": "pass" if len(significant_turns) == 1 else "fail",
        "reason_codes": [] if len(significant_turns) == 1 else ["multiple_significant_impacts"],
        "significant_turning_point_indices": significant_turns,
    }
    dwell = impact_dwell_diagnostics(t, x, impact)
    domain = _domain_check(
        estimate,
        lower=0.0,
        upper=1.0,
        tolerance=0.05,
        label="restitution_energy_ratio",
    )
    local_event = _local_impact_velocity_ratio(t, x, wall)
    if local_event is None:
        estimator_check = {
            "status": "indeterminate",
            "reason_codes": ["local_restitution_estimator_unavailable"],
        }
    else:
        local_estimate = float(local_event["velocity_ratio"])
        disagreement = abs(local_estimate - estimate) / max(estimate, local_estimate, 1e-9)
        estimator_check = {
            "status": "fail" if disagreement > 0.30 else "pass",
            "reason_codes": (
                ["global_and_local_restitution_estimators_disagree"]
                if disagreement > 0.30
                else []
            ),
            "piecewise_estimate": estimate,
            "local_estimate": local_estimate,
            "relative_disagreement": disagreement,
        }
    rule = combine_rule_checks(
        {
            "single_impact_topology": topology_check,
            "known_wall_contact": contact_check,
            "preimpact_constant_speed": pre,
            "postimpact_constant_speed": post,
            "impact_dwell": dwell,
            "restitution_physical_domain": domain,
            "restitution_estimator_consistency": estimator_check,
        },
        rule_family="single_wall_constant_speed_restitution",
    )
    event = {
        "split_index": int(impact),
        "impact_time_s": float(t[impact]),
        "impact_position_m": float(x[impact]),
        "v_pre": float(v_pre),
        "v_post": float(v_post),
        "velocity_ratio": estimate,
        "preimpact": pre,
        "postimpact": post,
        "dwell": dwell,
        "local_estimator": local_event,
    }
    return _result(
        "v1_B",
        {"restitution_e": estimate},
        event,
        method="single_impact_segmented_velocity_ratio_with_consistency_gate",
        parameter_quality={"restitution_e": rule},
        rule_family=rule,
        segmentation={
            "algorithm": "known_wall_single_reversal_with_impact_guard",
            "status": "ok",
            "events": [
                {"name": "motion_start", "index": int(start), "time_s": float(t[start])},
                {"name": "wall_impact", "index": int(impact), "time_s": float(t[impact])},
            ],
            "segments": [
                {
                    "name": "preimpact_constant_velocity",
                    "start_index": int(pre_indices[0]),
                    "end_index": int(pre_indices[-1]),
                    "point_count": int(len(pre_indices)),
                },
                {
                    "name": "postimpact_constant_velocity",
                    "start_index": int(post_indices[0]),
                    "end_index": int(post_indices[-1]),
                    "point_count": int(len(post_indices)),
                },
            ],
        },
    )


def _fit_v1c(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    if len(t) < 8:
        return _failure("v1_C", "fewer_than_8_points", ["kinetic_friction_mu"])
    detected_start = motion_start_index(t, x, direction=1)
    # The detector returns the frame immediately before the first sustained
    # moving interval.  If a static prefix exists, that interval can straddle
    # the release; advance once to obtain the first fully moving frame.
    start = min(detected_start + (1 if detected_start > 0 else 0), len(t) - 2)
    stop = motion_stop_index(t, x, start_index=start, direction=1)
    indices = np.arange(start, stop + 1, dtype=int)
    if len(indices) < 8:
        return _failure(
            "v1_C", "fewer_than_8_points_in_first_moving_run", ["kinetic_friction_mu"]
        )

    # The first moving interval is the experiment's observed initial speed.
    # Holding it fixed prevents a delayed/static prefix from being absorbed by
    # the intercept and acceleration terms.
    dt0 = float(t[start + 1] - t[start])
    if dt0 <= 1e-9:
        return _failure("v1_C", "initial_velocity_interval_invalid", ["kinetic_friction_mu"])
    initial_velocity = float((x[start + 1] - x[start]) / dt0)
    tau = t[indices] - t[start]
    target = x[indices] - float(x[start]) - initial_velocity * tau
    design = (0.5 * tau**2)[:, None]
    coefficient, _ = _robust_lstsq(design, target)
    acceleration = float(coefficient[0])
    predicted = float(x[start]) + initial_velocity * tau + 0.5 * acceleration * tau**2
    diagnostics = _fit_diagnostics(
        x[indices], predicted, 1, time_s=t[indices], series_name="x_m"
    )
    friction = -acceleration / float(EXPERIMENT_CONSTANTS["v1_C"]["gravity_g"])
    fit_check = _model_fit_check(
        diagnostics, maximum_nrmse=0.12, minimum_r2=0.82, label="first_friction_run"
    )
    stationarity = _quadratic_acceleration_stability(t[indices], x[indices])
    direction = {
        "status": "pass" if initial_velocity > 0.0 and acceleration <= 0.03 else "fail",
        "reason_codes": [] if initial_velocity > 0.0 and acceleration <= 0.03 else [
            "friction_run_does_not_start_forward_and_decelerate"
        ],
        "initial_velocity_m_s": initial_velocity,
        "acceleration_m_s2": acceleration,
    }
    domain = _domain_check(friction, lower=0.0, label="kinetic_friction")
    rule = combine_rule_checks(
        {
            "forward_decelerating_motion": direction,
            "constant_deceleration": stationarity,
            "trajectory_model": fit_check,
            "physical_friction": domain,
        },
        rule_family="single_surface_constant_kinetic_friction",
    )
    diagnostics.update(
        {
            "estimated_acceleration_m_s2": acceleration,
            "observed_initial_velocity_m_s": initial_velocity,
            "initial_velocity_assumption": "equal_to_first_moving_frame_interval",
            "motion_onset_preceding_frame_index": int(detected_start),
            "moving_fit_points": int(len(indices)),
        }
    )
    return _result(
        "v1_C",
        {"kinetic_friction_mu": friction},
        diagnostics,
        method="first_motion_fixed_initial_velocity_pre_stop",
        parameter_quality={"kinetic_friction_mu": rule},
        rule_family=rule,
        segmentation={
            "algorithm": "first_sustained_forward_motion_to_first_stationary_tail",
            "status": "ok",
            "events": [
                {"name": "motion_start", "index": int(start), "time_s": float(t[start])},
                {"name": "motion_stop", "index": int(stop), "time_s": float(t[stop])},
            ],
            "segments": [{
                "name": "first_friction_run",
                "start_index": int(start),
                "end_index": int(stop),
                "point_count": int(len(indices)),
            }],
        },
    )


def _fit_fixed_frequency_decay(
    t: np.ndarray,
    theta: np.ndarray,
    period: float,
    beta_range: tuple[float, float],
) -> tuple[float, dict[str, Any]]:
    omega = 2.0 * math.pi / period
    grid = np.linspace(beta_range[0], beta_range[1], 501)
    best_beta = float(grid[0])
    best_pred = np.zeros_like(theta)
    best_loss = float("inf")
    tau = t - t[0]
    for beta in grid:
        envelope = np.exp(-beta * tau)
        design = np.column_stack((envelope * np.cos(omega * tau), envelope * np.sin(omega * tau)))
        coefficients = np.linalg.lstsq(design, theta, rcond=None)[0]
        predicted = design @ coefficients
        loss = float(np.mean((theta - predicted) ** 2))
        if loss < best_loss:
            best_loss, best_beta, best_pred = loss, float(beta), predicted
    diagnostics = _fit_diagnostics(
        theta,
        best_pred,
        3,
        time_s=t,
        series_name="theta_rad",
    )
    turns = _turning_indices(theta)
    diagnostics.update({"turning_point_indices": turns, "turning_point_count": len(turns), "fixed_period_s": period})
    return best_beta, diagnostics


def _fit_v1d(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v1_D"]
    theta = _pendulum_angle(x, z, constants["pivot"])
    if len(theta) < 12:
        return _failure("v1_D", "fewer_than_12_points", ["amplitude_decay_beta"])
    start = motion_start_index(t, theta, direction=0)
    cycles, extrema = complete_cycle_segments(t, theta, start_index=start)
    if not cycles:
        return _failure("v1_D", "first_complete_period_not_observed", ["amplitude_decay_beta"])
    first = cycles[0]
    indices = np.arange(first["start_index"], first["end_index"] + 1, dtype=int)
    if len(indices) < 12:
        return _failure("v1_D", "fewer_than_12_points_in_first_period", ["amplitude_decay_beta"])
    beta, diagnostics = _fit_fixed_frequency_decay(
        t[indices], theta[indices], float(constants["period"]), (0.0, 1.0)
    )
    fit_check = _model_fit_check(
        diagnostics, maximum_nrmse=0.16, minimum_r2=0.72, label="first_period_decay"
    )
    period_error = abs(float(first["duration_s"]) - float(constants["period"])) / float(constants["period"])
    period_check = {
        "status": "fail" if period_error > 0.30 else "pass",
        "reason_codes": ["first_period_duration_inconsistent_with_known_period"] if period_error > 0.30 else [],
        "observed_period_s": float(first["duration_s"]),
        "known_period_s": float(constants["period"]),
        "relative_error": period_error,
    }
    domain = _domain_check(beta, lower=0.0, label="amplitude_decay")
    rule = combine_rule_checks(
        {"first_period_fit": fit_check, "period_geometry": period_check, "physical_decay": domain},
        rule_family="first_period_fixed_frequency_exponential_decay",
    )
    diagnostics.update({"detected_cycle_count": len(cycles), "detected_extrema": extrema})
    return _result(
        "v1_D",
        {"amplitude_decay_beta": beta},
        diagnostics,
        method="first_complete_cycle_fixed_period_decay_search",
        parameter_quality={"amplitude_decay_beta": rule},
        rule_family=rule,
        segmentation={
            "algorithm": "same_phase_extrema_complete_periods",
            "status": "ok",
            "events": [{"name": "motion_start", "index": int(start), "time_s": float(t[start])}],
            "cycles": cycles,
            "selected_cycle_index": 0,
            "segments": [{
                "name": "first_complete_period",
                "start_index": int(indices[0]),
                "end_index": int(indices[-1]),
                "point_count": int(len(indices)),
            }],
        },
    )


def _fit_periodic_frequency(
    t: np.ndarray,
    q: np.ndarray,
    omega_range: tuple[float, float],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    grid = np.linspace(omega_range[0], omega_range[1], 801)
    tau = t - t[0]
    best_omega = float(grid[0])
    best_pred = np.zeros_like(q)
    best_loss = float("inf")
    for omega in grid:
        columns = [np.ones(len(tau))]
        for harmonic in (1, 2, 3):
            columns.extend((np.cos(harmonic * omega * tau), np.sin(harmonic * omega * tau)))
        design = np.column_stack(columns)
        coefficients = np.linalg.lstsq(design, q, rcond=None)[0]
        predicted = design @ coefficients
        loss = float(np.mean((q - predicted) ** 2))
        if loss < best_loss:
            best_loss, best_omega, best_pred = loss, float(omega), predicted
    return best_omega, best_pred, _fit_diagnostics(
        q,
        best_pred,
        8,
        time_s=t,
        series_name="x_m",
    )


def _fit_v2a(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    if len(t) < 24:
        return _failure("v2_A", "fewer_than_24_points", ["gravity_g"])
    a = float(EXPERIMENT_CONSTANTS["v2_A"]["cycloid_a"])
    omega_min = math.sqrt(2.0 / (4.0 * a))
    omega_max = math.sqrt(14.7 / (4.0 * a))
    start = motion_start_index(t, x, direction=0)
    cycles, extrema = complete_cycle_segments(t, x, start_index=start)
    if not cycles:
        return _failure("v2_A", "no_complete_cycloid_period", ["gravity_g"])
    per_cycle: list[dict[str, Any]] = []
    estimates: list[float] = []
    predictions: list[dict[str, Any]] = []
    for cycle in cycles:
        indices = np.arange(cycle["start_index"], cycle["end_index"] + 1, dtype=int)
        if len(indices) < 12:
            continue
        omega, predicted, cycle_diag = _fit_periodic_frequency(t[indices], x[indices], (omega_min, omega_max))
        estimate = 4.0 * a * omega**2
        estimates.append(float(estimate))
        per_cycle.append({
            **cycle,
            "estimated_omega_rad_s": float(omega),
            "estimated_period_s": float(2.0 * math.pi / omega),
            "estimated_gravity_g": float(estimate),
            "fit": cycle_diag,
        })
        predictions.append(cycle_diag["fit_series"])
    if not estimates:
        return _failure("v2_A", "no_cycle_with_sufficient_fit_points", ["gravity_g"])
    gravity = float(np.median(estimates))
    relative_spread = (
        float(np.ptp(estimates) / max(abs(gravity), 1e-9)) if len(estimates) > 1 else 0.0
    )
    fit_checks = [
        _model_fit_check(item["fit"], maximum_nrmse=0.13, minimum_r2=0.80, label=f"cycle_{index}")
        for index, item in enumerate(per_cycle)
    ]
    cycle_consistency = {
        "status": "fail" if relative_spread > 0.25 else "pass",
        "reason_codes": ["cycle_gravity_estimates_inconsistent"] if relative_spread > 0.25 else [],
        "relative_range": relative_spread,
        "per_cycle_gravity_g": estimates,
    }
    rule = combine_rule_checks(
        {
            **{f"cycle_{index}_trajectory": check for index, check in enumerate(fit_checks)},
            "cycle_parameter_consistency": cycle_consistency,
            "physical_gravity": _domain_check(gravity, lower=0.0, label="gravity"),
        },
        rule_family="cycloid_periodic_motion_per_complete_cycle",
    )
    diagnostics = {
        "detected_cycle_count": len(cycles),
        "usable_cycle_count": len(per_cycle),
        "turning_point_indices": extrema,
        "per_cycle": per_cycle,
        "fit_series_collection": predictions,
        "cycle_gravity_relative_range": relative_spread,
    }
    return _result(
        "v2_A",
        {"gravity_g": gravity},
        diagnostics,
        method="per_complete_cycle_three_harmonic_period_search",
        parameter_quality={"gravity_g": rule},
        rule_family=rule,
        segmentation={
            "algorithm": "same_phase_extrema_complete_periods",
            "status": "ok",
            "cycles": cycles,
            "segments": [{
                "name": f"period_{index + 1}",
                "start_index": int(item["start_index"]),
                "end_index": int(item["end_index"]),
            } for index, item in enumerate(cycles)],
        },
    )


def _linear_segment_velocities(t: np.ndarray, q: np.ndarray, turns: Sequence[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for indices in _segments_from_turns(len(q), turns, guard=1):
        diagnostic = linear_motion_diagnostics(t, q, indices, series_name="x_m")
        output.append({"start": int(indices[0]), "end": int(indices[-1]), **diagnostic})
    return output


def _fit_v2b(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v2_B"]
    left_wall = float(constants["left_wall_x"])
    right_wall = float(constants["right_wall_x"])
    raw_turns = _turning_indices(x, min_separation=4)
    tolerance = max(0.18, 0.07 * abs(right_wall - left_wall))
    turns: list[int] = []
    labels: list[str] = []
    for raw in raw_turns:
        local = np.arange(max(1, raw - 3), min(len(x) - 1, raw + 4), dtype=int)
        if not len(local):
            continue
        index = int(local[np.argmin(np.minimum(np.abs(x[local] - left_wall), np.abs(x[local] - right_wall)))])
        left_distance = abs(float(x[index]) - left_wall)
        right_distance = abs(float(x[index]) - right_wall)
        if min(left_distance, right_distance) > tolerance:
            continue
        if turns and index - turns[-1] < 4:
            continue
        turns.append(index)
        labels.append("left" if left_distance <= right_distance else "right")
    if not turns:
        return _failure("v2_B", "no_valid_frozen_wall_impact", ["restitution_e"])
    segments = _linear_segment_velocities(t, x, turns)
    ratios: list[float] = []
    impact_records: list[dict[str, Any]] = []
    for index, impact in enumerate(turns):
        if index + 1 >= len(segments):
            break
        before, after = segments[index], segments[index + 1]
        v_pre, v_post = before.get("velocity"), after.get("velocity")
        if v_pre is None or v_post is None or abs(float(v_pre)) <= 1e-6 or float(v_pre) * float(v_post) >= 0:
            impact_records.append({"index": impact, "wall": labels[index], "status": "fail", "reason": "impact_velocity_reversal_missing"})
            continue
        ratio = abs(float(v_post) / float(v_pre))
        dwell = impact_dwell_diagnostics(t, x, impact)
        domain = _domain_check(ratio, lower=0.0, upper=1.0, tolerance=0.08, label="restitution")
        status = "pass" if dwell["status"] == "pass" and domain["status"] == "pass" else "fail"
        impact_records.append({
            "index": int(impact), "time_s": float(t[impact]), "wall": labels[index],
            "v_pre": float(v_pre), "v_post": float(v_post), "restitution_e": ratio,
            "status": status, "dwell": dwell, "physical_domain": domain,
        })
        if status == "pass":
            ratios.append(ratio)
    if not ratios:
        estimate = None
    else:
        estimate = float(np.median(ratios))
    alternates = all(a != b for a, b in zip(labels[:-1], labels[1:]))
    geometry_check = {
        "status": "pass" if alternates else "fail",
        "reason_codes": [] if alternates else ["wall_impacts_do_not_alternate"],
        "wall_labels": labels,
        "impact_indices": turns,
    }
    segment_check = combine_rule_checks(
        {f"free_flight_{index}": segment for index, segment in enumerate(segments)},
        rule_family="constant_velocity_between_wall_impacts",
    )
    impact_check = {
        "status": "pass" if ratios and all(item.get("status") == "pass" for item in impact_records) else "fail",
        "reason_codes": [] if ratios and all(item.get("status") == "pass" for item in impact_records) else [
            "one_or_more_wall_impacts_show_sticking_or_invalid_velocity_change"
        ],
        "events": impact_records,
    }
    relative_spread = float(np.ptp(ratios) / max(abs(float(np.median(ratios))), 1e-9)) if len(ratios) > 1 else 0.0
    consistency = {
        "status": "fail" if relative_spread > 0.30 else "pass",
        "reason_codes": ["restitution_not_consistent_across_impacts"] if relative_spread > 0.30 else [],
        "relative_range": relative_spread,
    }
    rule = combine_rule_checks(
        {
            "frozen_wall_topology": geometry_check,
            "constant_speed_between_impacts": segment_check,
            "instantaneous_impacts": impact_check,
            "restitution_consistency": consistency,
        },
        rule_family="alternating_wall_impacts_with_constant_restitution",
    )
    diagnostics = {
        "impact_indices": turns, "impact_count": len(ratios), "per_impact_e": ratios,
        "impact_events": impact_records, "segments": segments,
    }
    return _result(
        "v2_B", {"restitution_e": estimate}, diagnostics,
        method="per_wall_impact_guarded_velocity_ratios",
        parameter_quality={"restitution_e": rule}, rule_family=rule,
        segmentation={
            "algorithm": "frozen_wall_contact_reversal_state_machine", "status": "ok",
            "events": [{"name": f"{label}_wall_impact", "index": int(index), "time_s": float(t[index])} for index, label in zip(turns, labels)],
            "segments": [{"name": f"between_impacts_{index + 1}", "start_index": item["start"], "end_index": item["end"]} for index, item in enumerate(segments)],
        },
    )


def _fit_v2c(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    transition = float(EXPERIMENT_CONSTANTS["v2_C"]["transition_x"])
    start = motion_start_index(t, x, direction=1)
    stop = motion_stop_index(t, x, start_index=start, direction=1)
    crossings = np.flatnonzero((x[:-1] < transition) & (x[1:] >= transition)) + 1
    if len(crossings) != 1:
        return _failure("v2_C", "expected_exactly_one_surface_transition", ["kinetic_friction_mu_A", "kinetic_friction_mu_B"])
    crossing = int(crossings[0])
    guard = 1
    left = np.arange(start, max(start, crossing - guard) + 1, dtype=int)
    right = np.arange(min(stop, crossing + guard), stop + 1, dtype=int)
    if len(left) < 6 or len(right) < 6:
        return _failure("v2_C", "both_friction_surfaces_not_observed", ["kinetic_friction_mu_A", "kinetic_friction_mu_B"])
    coef_a, diag_a = _fit_quadratic(t[left], x[left], series_name="surface_A_x_m")
    coef_b, diag_b = _fit_quadratic(t[right], x[right], series_name="surface_B_x_m")
    normal_over_mass = float(EXPERIMENT_CONSTANTS["v2_C"]["normal_over_mass"])
    estimates = {
        "kinetic_friction_mu_A": -2.0 * coef_a[2] / normal_over_mass,
        "kinetic_friction_mu_B": -2.0 * coef_b[2] / normal_over_mass,
    }
    transition_time = float(t[crossing])
    tau_a = transition_time - float(t[left[0]])
    tau_b = transition_time - float(t[right[0]])
    velocity_a = float(coef_a[1] + 2.0 * coef_a[2] * tau_a)
    velocity_b = float(coef_b[1] + 2.0 * coef_b[2] * tau_b)
    velocity_jump = abs(velocity_b - velocity_a) / max(abs(velocity_a), abs(velocity_b), 1e-9)
    continuity = {
        "status": "fail" if velocity_jump > 0.30 else "pass",
        "reason_codes": ["velocity_reset_at_surface_transition"] if velocity_jump > 0.30 else [],
        "surface_A_extrapolated_velocity_m_s": velocity_a,
        "surface_B_extrapolated_velocity_m_s": velocity_b,
        "relative_velocity_jump": velocity_jump,
    }
    topology = {
        "status": "pass", "reason_codes": [], "transition_count": 1,
        "transition_index": crossing,
    }
    family = combine_rule_checks(
        {"single_surface_transition": topology, "velocity_continuity": continuity},
        rule_family="two_consecutive_constant_friction_surfaces",
    )
    quality_a = combine_rule_checks(
        {
            "quadratic_fit": _model_fit_check(diag_a, maximum_nrmse=0.12, minimum_r2=0.82, label="surface_A"),
            "constant_acceleration": _quadratic_acceleration_stability(t[left], x[left]),
            "physical_mu": _domain_check(float(estimates["kinetic_friction_mu_A"]), lower=0.0, label="surface_A_mu"),
        }, rule_family="surface_A_constant_friction",
    )
    quality_b = combine_rule_checks(
        {
            "quadratic_fit": _model_fit_check(diag_b, maximum_nrmse=0.12, minimum_r2=0.82, label="surface_B"),
            "constant_acceleration": _quadratic_acceleration_stability(t[right], x[right]),
            "physical_mu": _domain_check(float(estimates["kinetic_friction_mu_B"]), lower=0.0, label="surface_B_mu"),
        }, rule_family="surface_B_constant_friction",
    )
    diagnostics = {
        "transition_nearest_index": crossing,
        "surface_A": diag_a,
        "surface_B": diag_b,
        "acceleration_A_m_s2": float(2.0 * coef_a[2]),
        "acceleration_B_m_s2": float(2.0 * coef_b[2]),
        "transition_velocity_continuity": continuity,
    }
    return _result(
        "v2_C", estimates, diagnostics, method="two_surface_segmented_quadratic_with_velocity_continuity",
        parameter_quality={"kinetic_friction_mu_A": quality_a, "kinetic_friction_mu_B": quality_b},
        rule_family=family,
        segmentation={
            "algorithm": "single_hysteretic_surface_crossing_and_motion_tail_trim", "status": "ok",
            "events": [{"name": "surface_transition", "index": crossing, "time_s": transition_time}],
            "segments": [
                {"name": "surface_A_friction", "start_index": int(left[0]), "end_index": int(left[-1])},
                {"name": "surface_B_friction", "start_index": int(right[0]), "end_index": int(right[-1])},
            ],
        },
    )


def _fit_pendulum_integral(
    t: np.ndarray,
    theta: np.ndarray,
    *,
    length: float,
    magnetic_center: float | None = None,
    magnetic_width: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    tau = t - t[0]
    theta0 = float(theta[0])
    sin_theta = np.sin(theta)
    i_sin = _cumulative_trapezoid(sin_theta, tau)
    i_t_sin = _cumulative_trapezoid(tau * sin_theta, tau)
    gravity_basis = -(tau * i_sin - i_t_sin) / length
    damping_basis = -2.0 * _cumulative_trapezoid(theta - theta0, tau)
    columns = [gravity_basis, damping_basis]
    if magnetic_center is not None and magnetic_width is not None:
        u = (theta - magnetic_center) / magnetic_width
        profile = -u * np.exp(-0.5 * u**2)
        i_h = _cumulative_trapezoid(profile, tau)
        i_t_h = _cumulative_trapezoid(tau * profile, tau)
        columns.append(tau * i_h - i_t_h)
    design = np.column_stack(columns)
    target = theta - theta0
    keep = (tau >= min(0.5, 0.15 * float(tau[-1]))) & np.all(np.isfinite(design), axis=1)
    if int(np.sum(keep)) < design.shape[1] + 4:
        raise ValueError("insufficient integral-regression support")
    coefficients, _ = _robust_lstsq(design[keep], target[keep])
    predicted = design[keep] @ coefficients
    normalized = design[keep] / np.maximum(np.linalg.norm(design[keep], axis=0), 1e-12)
    singular = np.linalg.svd(normalized, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else float("inf")
    diagnostics = _fit_diagnostics(
        target[keep],
        predicted,
        len(coefficients),
        time_s=t[keep],
        series_name="theta_delta_rad",
    )
    diagnostics.update({
        "design_matrix_rank": int(np.linalg.matrix_rank(design[keep])),
        "normalized_condition_number": condition,
        "turning_point_indices": _turning_indices(theta),
    })
    return coefficients, diagnostics


def _simulate_pendulum(
    time_s: np.ndarray,
    *,
    theta0: float,
    omega0: float,
    gravity_g: float,
    damping_beta: float,
    length: float,
    magnetic_kappa: float = 0.0,
    magnetic_center: float | None = None,
    magnetic_width: float | None = None,
) -> np.ndarray:
    """Forward integrate the fitted equation for an observed-vs-model gate."""

    output = np.empty(len(time_s), dtype=np.float64)
    output[0] = float(theta0)
    state = np.asarray([float(theta0), float(omega0)], dtype=np.float64)

    def derivative(current: np.ndarray) -> np.ndarray:
        angle, speed = (float(value) for value in current)
        acceleration = -(gravity_g / length) * math.sin(angle) - 2.0 * damping_beta * speed
        if magnetic_center is not None and magnetic_width is not None:
            u = (angle - magnetic_center) / magnetic_width
            acceleration += magnetic_kappa * (-u * math.exp(-0.5 * u * u))
        return np.asarray([speed, acceleration], dtype=np.float64)

    for index in range(1, len(time_s)):
        dt = float(time_s[index] - time_s[index - 1])
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        output[index] = float(state[0])
    return output


def _fit_v2d(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v2_D"]
    # V2D's Blender positive angle places the ball at negative world X.
    theta = _pendulum_angle(x, z, constants["pivot"], sign=-1.0)
    if len(theta) < 16:
        return _failure("v2_D", "fewer_than_16_points", ["gravity_g", "linear_damping_beta"])
    start = motion_start_index(t, theta, direction=0)
    cycles, extrema = complete_cycle_segments(t, theta, start_index=start)
    if not cycles:
        return _failure("v2_D", "no_complete_damped_pendulum_period", ["gravity_g", "linear_damping_beta"])
    indices = np.arange(cycles[0]["start_index"], cycles[-1]["end_index"] + 1, dtype=int)
    try:
        coefficients, diagnostics = _fit_pendulum_integral(
            t[indices], theta[indices], length=float(constants["length"])
        )
    except ValueError as error:
        return _failure("v2_D", str(error), ["gravity_g", "linear_damping_beta"])
    gravity, beta = (float(value) for value in coefficients[:2])
    omega0 = float(np.gradient(_smooth(theta[indices], window=3), t[indices])[0])
    predicted = _simulate_pendulum(
        t[indices], theta0=float(theta[indices[0]]), omega0=omega0,
        gravity_g=gravity, damping_beta=beta, length=float(constants["length"]),
    )
    forward = _fit_diagnostics(
        theta[indices], predicted, 2, time_s=t[indices], series_name="theta_forward_rad"
    )
    forward_check = _model_fit_check(
        forward, maximum_nrmse=0.12, minimum_r2=0.76, label="forward_pendulum_trajectory"
    )
    identifiability = {
        "status": "pass" if diagnostics.get("design_matrix_rank") == 2 and float(diagnostics.get("normalized_condition_number", float("inf"))) <= 30.0 else "fail",
        "reason_codes": [] if diagnostics.get("design_matrix_rank") == 2 and float(diagnostics.get("normalized_condition_number", float("inf"))) <= 30.0 else ["pendulum_parameter_design_not_identifiable"],
        "rank": diagnostics.get("design_matrix_rank"),
        "condition_number": diagnostics.get("normalized_condition_number"),
    }
    durations = [float(item["duration_s"]) for item in cycles]
    duration_cv = float(np.std(durations) / max(np.mean(durations), 1e-9)) if len(durations) > 1 else 0.0
    duration_range = float(np.ptp(durations) / max(np.mean(durations), 1e-9)) if len(durations) > 1 else 0.0
    period_failed = duration_cv > 0.06 or duration_range > 0.15
    periodicity = {
        "status": "fail" if period_failed else "pass",
        "reason_codes": ["pendulum_period_drift_exceeds_limit"] if period_failed else [],
        "cycle_durations_s": durations, "duration_cv": duration_cv,
        "duration_relative_range": duration_range,
        "maximum_duration_cv": 0.06,
        "maximum_duration_relative_range": 0.15,
    }
    family = combine_rule_checks(
        {"forward_equation_match": forward_check, "parameter_identifiability": identifiability, "period_stability": periodicity},
        rule_family="damped_pendulum_constant_gravity_and_damping",
    )
    gravity_quality = combine_rule_checks(
        {"physical_gravity": _domain_check(gravity, lower=0.0, label="gravity")},
        rule_family="pendulum_gravity",
    )
    amplitude_change = abs(float(theta[indices[0]]) - float(theta[indices[-1]]))
    beta_support = {
        "status": "pass" if amplitude_change >= 0.015 else "indeterminate",
        "reason_codes": [] if amplitude_change >= 0.015 else ["damping_signal_below_trajectory_resolution"],
        "endpoint_amplitude_change_rad": amplitude_change,
    }
    beta_quality = combine_rule_checks(
        {"physical_damping": _domain_check(beta, lower=0.0, label="linear_damping"), "damping_support": beta_support},
        rule_family="pendulum_linear_damping",
    )
    diagnostics.update({
        "forward_validation": forward, "detected_cycles": cycles,
        "detected_extrema": extrema, "period_stability": periodicity,
    })
    return _result(
        "v2_D",
        {"gravity_g": gravity, "linear_damping_beta": beta},
        diagnostics,
        method="segmented_integral_regression_with_forward_trajectory_validation",
        parameter_quality={"gravity_g": gravity_quality, "linear_damping_beta": beta_quality},
        rule_family=family,
        segmentation={
            "algorithm": "complete_pendulum_periods", "status": "ok", "cycles": cycles,
            "segments": [{"name": "complete_period_support", "start_index": int(indices[0]), "end_index": int(indices[-1])}],
        },
    )


def _fit_shared_flight_model(
    t: np.ndarray,
    z: np.ndarray,
    turns: Sequence[int],
    *,
    drag_beta: float | None = None,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    segments = _segments_from_turns(len(z), turns, guard=1)
    if not segments:
        segments = [np.arange(len(z), dtype=int)]
    minimum_excursion = max(0.02, 0.015 * float(np.ptp(z)))
    segments = [
        indices
        for indices in segments
        if len(indices) >= 4 and float(np.ptp(z[indices])) >= minimum_excursion
    ]
    if not segments:
        raise ValueError("no dynamic flight segments")
    # A flight can include an apex; impact minima define segment boundaries.
    row_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    segment_meta: list[dict[str, Any]] = []
    nuisance_columns = 2 * len(segments)
    for segment_id, indices in enumerate(segments):
        tau = t[indices] - t[indices[0]]
        block = np.zeros((len(indices), nuisance_columns + 1), dtype=np.float64)
        block[:, 2 * segment_id] = 1.0
        if drag_beta is None:
            block[:, 2 * segment_id + 1] = tau
            block[:, -1] = -0.5 * tau**2
        else:
            block[:, 2 * segment_id + 1] = 1.0 - np.exp(-drag_beta * tau)
            block[:, -1] = tau
        row_blocks.append(block)
        targets.append(z[indices])
        segment_meta.append({"indices": indices, "segment_id": segment_id})
    if not row_blocks:
        raise ValueError("no usable flight segments")
    design = np.vstack(row_blocks)
    target = np.concatenate(targets)
    coefficients, _ = _robust_lstsq(design, target)
    shared = float(coefficients[-1])
    velocity_models: list[dict[str, Any]] = []
    for item in segment_meta:
        segment_id = item["segment_id"]
        indices = item["indices"]
        duration = float(t[indices[-1]] - t[indices[0]])
        if drag_beta is None:
            initial_v = float(coefficients[2 * segment_id + 1])
            final_v = initial_v - shared * duration
        else:
            amplitude = float(coefficients[2 * segment_id + 1])
            initial_v = amplitude * drag_beta + shared
            final_v = amplitude * drag_beta * math.exp(-drag_beta * duration) + shared
        velocity_models.append({
            "start": int(indices[0]),
            "end": int(indices[-1]),
            "initial_velocity": initial_v,
            "final_velocity": final_v,
        })
    diagnostics = _fit_diagnostics(
        target,
        design @ coefficients,
        len(coefficients),
        time_s=np.concatenate([t[item["indices"]] for item in segment_meta]),
        series_name="z_m",
    )
    diagnostics["flight_segments"] = velocity_models
    return shared, velocity_models, diagnostics


def _impact_minima(values: np.ndarray, contact: float | None = None) -> list[int]:
    smoothed = _smooth(values)
    turns = _turning_indices(smoothed, min_separation=3)
    minima: list[int] = []
    # Turning-point indices from a centered derivative can lie one or two
    # samples after a sharp bounce.  Snap each candidate to the raw local
    # minimum so contact-position filtering and flight segmentation are stable.
    for turn in turns:
        start = max(1, turn - 3)
        stop = min(len(values) - 1, turn + 4)
        if stop <= start:
            continue
        index = start + int(np.argmin(values[start:stop]))
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]:
            if not minima or index - minima[-1] >= 3:
                minima.append(index)
    if contact is not None:
        tolerance = max(0.12, 0.08 * max(float(np.ptp(values)), 1.0))
        minima = [index for index in minima if values[index] <= contact + tolerance]
    return minima


def _restitution_from_flights(flights: Sequence[Mapping[str, Any]]) -> tuple[float | None, list[float]]:
    ratios: list[float] = []
    for before, after in zip(flights[:-1], flights[1:]):
        v_pre = float(before["final_velocity"])
        v_post = float(after["initial_velocity"])
        if v_pre < -1e-6 and v_post > 1e-6:
            ratios.append(-v_post / v_pre)
    return (float(np.median(ratios)) if ratios else None), ratios


def _fit_v2e(t: np.ndarray, _x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    contact = float(EXPERIMENT_CONSTANTS["v2_E"]["contact_z"])
    impacts = _impact_minima(z, contact)
    try:
        shared, flights, diagnostics = _fit_shared_flight_model(t, z, impacts)
    except ValueError as error:
        return _failure("v2_E", str(error), ["gravity_g", "restitution_e"])
    restitution, ratios = _restitution_from_flights(flights)
    flight_checks: dict[str, dict[str, Any]] = {}
    for index, flight in enumerate(flights):
        segment = np.arange(int(flight["start"]), int(flight["end"]) + 1, dtype=int)
        if len(segment) < 6:
            continue
        _, local_diag = _fit_quadratic(t[segment], z[segment], series_name=f"flight_{index + 1}_z_m")
        check = combine_rule_checks(
            {
                "quadratic_fit": _model_fit_check(local_diag, maximum_nrmse=0.14, minimum_r2=0.70, label=f"flight_{index + 1}"),
                "constant_acceleration": _quadratic_acceleration_stability(t[segment], z[segment]),
            }, rule_family=f"ballistic_flight_{index + 1}",
        )
        # Very short terminal micro-bounces are below the resolution needed
        # for an acceleration-stability split.  They are reported, but do not
        # invalidate earlier well-resolved ballistic flights.
        flight_checks[f"flight_{index + 1}_ballistic"] = check
    resolved_flight_checks = {
        name: value for name, value in flight_checks.items()
        if value.get("status") != "indeterminate"
    }
    family = combine_rule_checks(
        {
            "shared_ballistic_fit": _model_fit_check(diagnostics, maximum_nrmse=0.15, minimum_r2=0.68, label="shared_ballistic_flights"),
            **resolved_flight_checks,
        }, rule_family="repeated_ballistic_flights",
    )
    gravity_quality = combine_rule_checks(
        {"physical_gravity": _domain_check(shared, lower=0.0, label="gravity")},
        rule_family="shared_ballistic_gravity",
    )
    impact_checks: dict[str, dict[str, Any]] = {}
    for index, impact in enumerate(impacts[: len(ratios)]):
        impact_checks[f"impact_{index + 1}_dwell"] = impact_dwell_diagnostics(t, z, impact)
    ratio_spread = float(np.ptp(ratios) / max(abs(float(np.median(ratios))), 1e-9)) if len(ratios) > 1 else 0.0
    restitution_quality = combine_rule_checks(
        {
            "impact_support": {
                "status": "pass" if ratios else "indeterminate",
                "reason_codes": [] if ratios else ["no_connected_pre_and_post_impact_flights"],
            },
            "physical_restitution": _domain_check(restitution, lower=0.0, upper=1.0, tolerance=0.08, label="restitution"),
            "impact_consistency": {
                "status": "fail" if ratio_spread > 0.35 else "pass",
                "reason_codes": ["restitution_varies_across_impacts"] if ratio_spread > 0.35 else [],
                "relative_range": ratio_spread,
            },
            **impact_checks,
        }, rule_family="instantaneous_repeated_floor_restitution",
    )
    diagnostics.update({"impact_indices": impacts, "per_impact_e": ratios, "flight_rule_checks": flight_checks})
    return _result(
        "v2_E",
        {"gravity_g": shared, "restitution_e": restitution},
        diagnostics,
        observed={"gravity_g": True, "restitution_e": restitution is not None},
        method="per_flight_ballistic_curvature_and_per_impact_velocity_ratio",
        parameter_quality={"gravity_g": gravity_quality, "restitution_e": restitution_quality},
        rule_family=family,
        segmentation={
            "algorithm": "floor_contact_state_machine_with_ballistic_flights", "status": "ok",
            "events": [{"name": "floor_impact", "index": int(index), "time_s": float(t[index])} for index in impacts],
            "segments": [{"name": f"flight_{index + 1}", "start_index": int(item["start"]), "end_index": int(item["end"])} for index, item in enumerate(flights)],
        },
    )


def _fit_v3a(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    if len(t) < 16:
        return _failure("v3_A", "fewer_than_16_points", ["gravity_g", "linear_drag_beta", "restitution_e"])
    horizontal_start = motion_start_index(t, x, direction=1)
    horizontal_indices = np.arange(horizontal_start, len(t), dtype=int)
    tau = t[horizontal_indices] - t[horizontal_start]
    best_beta = 0.1
    best_loss = float("inf")
    best_pred = np.zeros_like(x[horizontal_indices])
    for beta in np.linspace(0.05, 0.35, 601):
        design = np.column_stack((np.ones(len(tau)), 1.0 - np.exp(-beta * tau)))
        coefficients = np.linalg.lstsq(design, x[horizontal_indices], rcond=None)[0]
        predicted = design @ coefficients
        loss = float(np.mean((x[horizontal_indices] - predicted) ** 2))
        if loss < best_loss:
            best_beta, best_loss, best_pred = float(beta), loss, predicted
    horizontal_diag = _fit_diagnostics(
        x[horizontal_indices], best_pred, 3, time_s=t[horizontal_indices], series_name="horizontal_x_m"
    )
    horizontal_fit = _model_fit_check(
        horizontal_diag, maximum_nrmse=0.12, minimum_r2=0.82, label="horizontal_linear_drag"
    )
    beta_boundary = {
        "status": "fail" if best_beta <= 0.051 or best_beta >= 0.349 else "pass",
        "reason_codes": ["linear_drag_optimum_on_search_boundary"] if best_beta <= 0.051 or best_beta >= 0.349 else [],
        "search_range": [0.05, 0.35], "estimate": best_beta,
    }
    beta_quality = combine_rule_checks(
        {"horizontal_trajectory": horizontal_fit, "interior_optimum": beta_boundary, "physical_drag": _domain_check(best_beta, lower=0.0, label="linear_drag")},
        rule_family="horizontal_linear_drag_component",
    )
    impacts = _impact_minima(z, float(EXPERIMENT_CONSTANTS["v3_A"]["contact_z"]))
    try:
        shared, flights, diagnostics = _fit_shared_flight_model(t, z, impacts, drag_beta=best_beta)
        gravity = -best_beta * shared
    except ValueError:
        flights, diagnostics, gravity = [], {}, None
    restitution, ratios = _restitution_from_flights(flights)
    vertical_quality = combine_rule_checks(
        {
            "horizontal_drag_required_by_vertical_model": beta_quality,
            "vertical_drag_flight_fit": _model_fit_check(diagnostics, maximum_nrmse=0.16, minimum_r2=0.65, label="vertical_drag_flights") if diagnostics else {"status": "indeterminate", "reason_codes": ["vertical_flight_fit_unavailable"]},
            "physical_gravity": _domain_check(gravity, lower=0.0, label="gravity"),
        }, rule_family="vertical_gravity_with_linear_drag",
    )
    impact_checks = {
        f"impact_{index + 1}_dwell": impact_dwell_diagnostics(t, z, impact)
        for index, impact in enumerate(impacts[: len(ratios)])
    }
    ratio_spread = float(np.ptp(ratios) / max(abs(float(np.median(ratios))), 1e-9)) if len(ratios) > 1 else 0.0
    restitution_quality = combine_rule_checks(
        {
            "impact_support": {"status": "pass" if ratios else "indeterminate", "reason_codes": [] if ratios else ["no_connected_floor_impact_flights"]},
            "physical_restitution": _domain_check(restitution, lower=0.0, upper=1.0, tolerance=0.08, label="restitution"),
            "impact_consistency": {"status": "fail" if ratio_spread > 0.35 else "pass", "reason_codes": ["restitution_varies_across_impacts"] if ratio_spread > 0.35 else [], "relative_range": ratio_spread},
            **impact_checks,
        }, rule_family="vertical_floor_restitution_component",
    )
    family = {
        "status": "pass", "rule_family": "separable_horizontal_drag_and_vertical_bounce",
        "reason_codes": [], "target_parameters_used": False,
        "component_policy": "parameters_are_accepted_independently",
    }
    diagnostics.update({
        "horizontal_fit": horizontal_diag,
        "impact_indices": impacts,
        "per_impact_e": ratios,
        "component_split": {"linear_drag_beta": "horizontal_x", "gravity_g": "vertical_z", "restitution_e": "vertical_floor_impacts"},
    })
    return _result(
        "v3_A",
        {"gravity_g": gravity, "linear_drag_beta": best_beta, "restitution_e": restitution},
        diagnostics,
        observed={"gravity_g": gravity is not None, "linear_drag_beta": True, "restitution_e": restitution is not None},
        method="separate_horizontal_drag_and_vertical_gravity_restitution_components",
        parameter_quality={"gravity_g": vertical_quality, "linear_drag_beta": beta_quality, "restitution_e": restitution_quality},
        rule_family=family,
        segmentation={
            "algorithm": "axis_specific_motion_and_floor_contact_segmentation", "status": "ok",
            "events": [{"name": "horizontal_motion_start", "index": int(horizontal_start), "time_s": float(t[horizontal_start])}] + [{"name": "floor_impact", "index": int(index), "time_s": float(t[index])} for index in impacts],
            "segments": [{"name": "horizontal_drag_support", "start_index": int(horizontal_indices[0]), "end_index": int(horizontal_indices[-1])}] + [{"name": f"vertical_flight_{index + 1}", "start_index": int(item["start"]), "end_index": int(item["end"])} for index, item in enumerate(flights)],
        },
    )


def _fit_v3b(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v3_B"]
    raw_turns = _turning_indices(x, min_separation=4)
    tolerance = max(0.12, 0.07 * abs(float(constants["right_wall_x"]) - float(constants["left_wall_x"])))
    turns: list[int] = []
    labels: list[str] = []
    for raw in raw_turns:
        local = np.arange(max(1, raw - 3), min(len(x) - 1, raw + 4), dtype=int)
        if not len(local):
            continue
        distances = np.minimum(np.abs(x[local] - float(constants["left_wall_x"])), np.abs(x[local] - float(constants["right_wall_x"])))
        event = int(local[int(np.argmin(distances))])
        if float(np.min(distances)) > tolerance or (turns and event - turns[-1] < 4):
            continue
        turns.append(event)
        labels.append("left" if abs(float(x[event]) - float(constants["left_wall_x"])) <= abs(float(x[event]) - float(constants["right_wall_x"])) else "right")
    segments = _segments_from_turns(len(x), turns, guard=1)
    if not segments:
        return _failure("v3_B", "no_wall_segments", ["kinetic_friction_mu_k", "left_restitution_e_L", "right_restitution_e_R"])
    gravity = float(constants["gravity_g"])
    segment_records: list[dict[str, Any]] = []
    mu_values: list[float] = []
    for segment_id, indices in enumerate(segments):
        direction = float(np.sign(x[indices[-1]] - x[indices[0]]))
        if direction == 0.0 or len(indices) < 6:
            continue
        coefficients, fit_diag = _fit_quadratic(t[indices], x[indices], series_name=f"wall_run_{segment_id + 1}_x_m")
        acceleration = float(2.0 * coefficients[2])
        mu_value = -acceleration * direction / gravity
        quality = combine_rule_checks(
            {
                "quadratic_fit": _model_fit_check(fit_diag, maximum_nrmse=0.12, minimum_r2=0.80, label=f"wall_run_{segment_id + 1}"),
                "constant_acceleration": _quadratic_acceleration_stability(t[indices], x[indices]),
                "opposes_motion": {"status": "pass" if acceleration * direction <= 0.02 else "fail", "reason_codes": [] if acceleration * direction <= 0.02 else ["acceleration_does_not_oppose_motion"]},
                "physical_mu": _domain_check(mu_value, lower=0.0, label="kinetic_friction"),
            }, rule_family=f"friction_run_{segment_id + 1}",
        )
        tau_end = float(t[indices[-1]] - t[indices[0]])
        record = {
            "start": int(indices[0]), "end": int(indices[-1]), "direction": direction,
            "initial_velocity": float(coefficients[1]),
            "final_velocity": float(coefficients[1] + acceleration * tau_end),
            "acceleration_m_s2": acceleration, "kinetic_friction_mu": mu_value,
            "fit": fit_diag, "quality": quality,
        }
        segment_records.append(record)
        if quality["status"] == "pass":
            mu_values.append(mu_value)
    if not segment_records:
        return _failure("v3_B", "no_moving_wall_segments", ["kinetic_friction_mu_k", "left_restitution_e_L", "right_restitution_e_R"])
    mu = float(np.median(mu_values)) if mu_values else None
    left_ratios: list[float] = []
    right_ratios: list[float] = []
    impact_records: list[dict[str, Any]] = []
    for index, (before, after) in enumerate(zip(segment_records[:-1], segment_records[1:])):
        if abs(float(before["final_velocity"])) <= 1e-6:
            continue
        ratio = abs(float(after["initial_velocity"]) / float(before["final_velocity"]))
        event_index = turns[min(index, len(turns) - 1)]
        label = labels[min(index, len(labels) - 1)]
        dwell = impact_dwell_diagnostics(t, x, event_index)
        domain = _domain_check(ratio, lower=0.0, upper=1.0, tolerance=0.08, label=f"{label}_restitution")
        impact_records.append({"index": event_index, "wall": label, "restitution_e": ratio, "dwell": dwell, "physical_domain": domain})
        if dwell["status"] != "pass" or domain["status"] != "pass":
            continue
        if label == "left":
            left_ratios.append(ratio)
        else:
            right_ratios.append(ratio)
    e_left = float(np.median(left_ratios)) if left_ratios else None
    e_right = float(np.median(right_ratios)) if right_ratios else None
    mu_spread = float(np.ptp(mu_values) / max(abs(float(np.median(mu_values))), 1e-9)) if len(mu_values) > 1 else 0.0
    mu_quality = combine_rule_checks(
        {
            **{f"run_{index + 1}": item["quality"] for index, item in enumerate(segment_records)},
            "coefficient_consistency": {"status": "fail" if mu_spread > 0.35 else "pass", "reason_codes": ["friction_coefficient_varies_between_runs"] if mu_spread > 0.35 else [], "relative_range": mu_spread},
        }, rule_family="shared_friction_across_wall_runs",
    )
    def restitution_quality(values: list[float], wall_name: str) -> dict[str, Any]:
        spread = float(np.ptp(values) / max(abs(float(np.median(values))), 1e-9)) if len(values) > 1 else 0.0
        relevant = [item for item in impact_records if item["wall"] == wall_name]
        return combine_rule_checks(
            {
                "impact_support": {"status": "pass" if values else "indeterminate", "reason_codes": [] if values else [f"no_valid_{wall_name}_wall_impact"]},
                "instantaneous_impacts": {"status": "pass" if relevant and all(item["dwell"]["status"] == "pass" for item in relevant) else "fail", "reason_codes": [] if relevant and all(item["dwell"]["status"] == "pass" for item in relevant) else [f"{wall_name}_wall_sticking_or_invalid_collision"]},
                "coefficient_consistency": {"status": "fail" if spread > 0.35 else "pass", "reason_codes": [f"{wall_name}_wall_restitution_inconsistent"] if spread > 0.35 else [], "relative_range": spread},
            }, rule_family=f"{wall_name}_wall_restitution",
        )
    family = {"status": "pass", "rule_family": "friction_between_wall_specific_collisions", "reason_codes": [], "target_parameters_used": False, "component_policy": "parameters_are_accepted_independently"}
    diagnostics = {"impact_indices": turns, "left_impact_e": left_ratios, "right_impact_e": right_ratios, "impact_events": impact_records, "segments": segment_records}
    return _result(
        "v3_B",
        {"kinetic_friction_mu_k": mu, "left_restitution_e_L": e_left, "right_restitution_e_R": e_right},
        diagnostics,
        observed={"kinetic_friction_mu_k": True, "left_restitution_e_L": e_left is not None, "right_restitution_e_R": e_right is not None},
        method="per_run_friction_and_per_wall_impact_restitution",
        parameter_quality={"kinetic_friction_mu_k": mu_quality, "left_restitution_e_L": restitution_quality(left_ratios, "left"), "right_restitution_e_R": restitution_quality(right_ratios, "right")},
        rule_family=family,
        segmentation={
            "algorithm": "frozen_wall_impacts_and_between_wall_friction_runs", "status": "ok",
            "events": [{"name": f"{label}_wall_impact", "index": int(index), "time_s": float(t[index])} for index, label in zip(turns, labels)],
            "segments": [{"name": f"friction_run_{index + 1}", "start_index": int(item["start"]), "end_index": int(item["end"])} for index, item in enumerate(segment_records)],
        },
    )


def _fit_v3c(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v3_C"]
    transition = float(constants["floor_transition_x"])
    wall = float(constants["wall_contact_x"])
    alpha = math.radians(float(constants["ramp_angle_deg"]))
    start = motion_start_index(t, x, direction=1)
    crossings = np.flatnonzero((x[:-1] < transition) & (x[1:] >= transition)) + 1
    if not len(crossings):
        return _failure("v3_C", "ramp_to_floor_transition_not_observed", ["gravity_g", "kinetic_friction_mu", "restitution_e"])
    floor_start = int(crossings[0])
    event = _local_impact_velocity_ratio(t, x, wall)
    if event is None:
        return _failure("v3_C", "right_wall_impact_not_observed", ["gravity_g", "kinetic_friction_mu", "restitution_e"])
    impact = int(event["split_index"])
    ramp = np.arange(start, max(start, floor_start - 1) + 1, dtype=int)
    pre_floor = np.arange(floor_start + 1, max(floor_start + 1, impact - 1) + 1, dtype=int)
    post_stop = motion_stop_index(t, x, start_index=min(impact + 2, len(x) - 1), direction=-1)
    post_floor = np.arange(min(len(x) - 1, impact + 2), post_stop + 1, dtype=int)
    if len(ramp) < 6 or len(pre_floor) < 6 or len(post_floor) < 6:
        return _failure("v3_C", "ramp_pre_wall_and_post_wall_segments_required", ["gravity_g", "kinetic_friction_mu", "restitution_e"])
    s = (x - x[0]) * math.cos(alpha) - (z - z[0]) * math.sin(alpha)
    coef_ramp, diag_ramp = _fit_quadratic(t[ramp], s[ramp], series_name="ramp_s_m")
    coef_pre, diag_pre = _fit_quadratic(t[pre_floor], x[pre_floor], series_name="prewall_floor_x_m")
    coef_post, diag_post = _fit_quadratic(t[post_floor], x[post_floor], series_name="postwall_floor_x_m")
    a_ramp = float(2.0 * coef_ramp[2])
    a_pre = float(2.0 * coef_pre[2])
    a_post = float(2.0 * coef_post[2])
    floor_decelerations = [-a_pre, a_post]
    a_floor = float(np.median(floor_decelerations))
    gravity = (a_ramp + a_floor * math.cos(alpha)) / math.sin(alpha)
    friction = a_floor / gravity if abs(gravity) > 1e-9 else None
    restitution = None
    if event is not None and event["v_pre"] > 0 and event["v_post"] < 0:
        restitution = float(event["velocity_ratio"])
    floor_consistency_value = abs(floor_decelerations[1] - floor_decelerations[0]) / max(abs(float(np.mean(floor_decelerations))), 1e-9)
    floor_consistency = {
        "status": "fail" if floor_consistency_value > 0.35 else "pass",
        "reason_codes": ["friction_deceleration_differs_before_and_after_wall"] if floor_consistency_value > 0.35 else [],
        "prewall_deceleration_m_s2": floor_decelerations[0],
        "postwall_deceleration_m_s2": floor_decelerations[1],
        "relative_difference": floor_consistency_value,
    }
    ramp_quality = combine_rule_checks(
        {
            "trajectory_fit": _model_fit_check(diag_ramp, maximum_nrmse=0.12, minimum_r2=0.82, label="ramp_slide"),
            "constant_acceleration": _quadratic_acceleration_stability(t[ramp], s[ramp]),
        }, rule_family="inclined_ramp_slide",
    )
    pre_quality = combine_rule_checks(
        {
            "trajectory_fit": _model_fit_check(diag_pre, maximum_nrmse=0.12, minimum_r2=0.82, label="prewall_floor"),
            "constant_acceleration": _quadratic_acceleration_stability(t[pre_floor], x[pre_floor]),
        }, rule_family="prewall_floor_friction",
    )
    post_quality = combine_rule_checks(
        {
            "trajectory_fit": _model_fit_check(diag_post, maximum_nrmse=0.12, minimum_r2=0.82, label="postwall_floor"),
            "constant_acceleration": _quadratic_acceleration_stability(t[post_floor], x[post_floor]),
        }, rule_family="postwall_floor_friction",
    )
    gravity_quality = combine_rule_checks(
        {
            "ramp_component": ramp_quality, "prewall_floor_component": pre_quality,
            "postwall_floor_component": post_quality, "floor_friction_consistency": floor_consistency,
            "physical_gravity": _domain_check(gravity, lower=0.0, label="gravity"),
        }, rule_family="ramp_plus_floor_gravity_decomposition",
    )
    friction_quality = combine_rule_checks(
        {
            "prewall_floor_component": pre_quality, "postwall_floor_component": post_quality,
            "shared_floor_friction": floor_consistency,
            "physical_friction": _domain_check(friction, lower=0.0, label="kinetic_friction"),
        }, rule_family="shared_ground_friction_before_and_after_wall",
    )
    dwell = impact_dwell_diagnostics(t, x, impact)
    restitution_quality = combine_rule_checks(
        {
            "instantaneous_wall_contact": dwell,
            "physical_restitution": _domain_check(restitution, lower=0.0, upper=1.0, tolerance=0.08, label="restitution"),
        }, rule_family="right_wall_restitution",
    )
    family = {
        "status": "pass", "rule_family": "ramp_then_floor_then_wall_then_floor",
        "reason_codes": [], "target_parameters_used": False,
        "component_policy": "gravity_uses_ramp_plus_floor; friction_uses_both_floor_runs; restitution_uses_wall_event",
    }
    diagnostics = {
        "ramp": diag_ramp,
        "prewall_floor": diag_pre,
        "postwall_floor": diag_post,
        "ramp_acceleration_m_s2": a_ramp,
        "prewall_floor_acceleration_m_s2": a_pre,
        "postwall_floor_acceleration_m_s2": a_post,
        "shared_floor_deceleration_m_s2": a_floor,
        "floor_friction_consistency": floor_consistency,
        "wall_event": event,
        "identifiability_note": "gravity is not identifiable from the ramp alone because a_ramp=g*(sin(alpha)-mu*cos(alpha)); the two floor runs provide mu*g",
    }
    return _result(
        "v3_C",
        {"gravity_g": gravity, "kinetic_friction_mu": friction, "restitution_e": restitution},
        diagnostics,
        observed={"gravity_g": True, "kinetic_friction_mu": friction is not None, "restitution_e": restitution is not None},
        method="separate_ramp_prewall_floor_wall_impact_and_postwall_floor",
        parameter_quality={"gravity_g": gravity_quality, "kinetic_friction_mu": friction_quality, "restitution_e": restitution_quality},
        rule_family=family,
        segmentation={
            "algorithm": "apparatus_geometry_phase_state_machine", "status": "ok",
            "events": [
                {"name": "motion_start", "index": int(start), "time_s": float(t[start])},
                {"name": "ramp_floor_transition", "index": floor_start, "time_s": float(t[floor_start])},
                {"name": "right_wall_impact", "index": impact, "time_s": float(t[impact])},
                {"name": "postwall_motion_stop", "index": int(post_stop), "time_s": float(t[post_stop])},
            ],
            "segments": [
                {"name": "ramp_slide", "start_index": int(ramp[0]), "end_index": int(ramp[-1])},
                {"name": "prewall_floor_friction", "start_index": int(pre_floor[0]), "end_index": int(pre_floor[-1])},
                {"name": "postwall_floor_friction", "start_index": int(post_floor[0]), "end_index": int(post_floor[-1])},
            ],
        },
    )


def _fit_v3d(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v3_D"]
    # V3D uses negative Blender Y rotation for a ball at positive world X.
    theta = _pendulum_angle(x, z, constants["pivot"], sign=-1.0)
    if len(theta) < 20:
        return _failure("v3_D", "fewer_than_20_points", ["gravity_g", "linear_damping_beta", "magnetic_kappa"])
    start = motion_start_index(t, theta, direction=0)
    cycles, extrema = complete_cycle_segments(t, theta, start_index=start)
    if not cycles:
        return _failure("v3_D", "no_complete_magnetic_pendulum_period", ["gravity_g", "linear_damping_beta", "magnetic_kappa"])
    indices = np.arange(cycles[0]["start_index"], cycles[-1]["end_index"] + 1, dtype=int)
    try:
        coefficients, diagnostics = _fit_pendulum_integral(
            t[indices],
            theta[indices],
            length=float(constants["length"]),
            magnetic_center=math.radians(float(constants["magnet_center_deg"])),
            magnetic_width=math.radians(float(constants["magnet_width_deg"])),
        )
    except ValueError as error:
        return _failure("v3_D", str(error), ["gravity_g", "linear_damping_beta", "magnetic_kappa"])
    magnet_center = math.radians(float(constants["magnet_center_deg"]))
    magnet_width = math.radians(float(constants["magnet_width_deg"]))
    inside = np.abs(theta[indices] - magnet_center) <= magnet_width
    diagnostics["magnet_zone_transition_count"] = int(np.sum(inside[1:] != inside[:-1]))
    gravity, beta, kappa = (float(value) for value in coefficients[:3])
    omega0 = float(np.gradient(_smooth(theta[indices], window=3), t[indices])[0])
    predicted = _simulate_pendulum(
        t[indices], theta0=float(theta[indices[0]]), omega0=omega0,
        gravity_g=gravity, damping_beta=beta, length=float(constants["length"]),
        magnetic_kappa=kappa, magnetic_center=magnet_center, magnetic_width=magnet_width,
    )
    forward = _fit_diagnostics(
        theta[indices], predicted, 3, time_s=t[indices], series_name="magnetic_pendulum_forward_theta_rad"
    )
    forward_check = _model_fit_check(
        forward, maximum_nrmse=0.16, minimum_r2=0.76, label="magnetic_pendulum_forward_trajectory"
    )
    rank = int(diagnostics.get("design_matrix_rank", 0))
    condition = float(diagnostics.get("normalized_condition_number", float("inf")))
    identifiability = {
        "status": "pass" if rank == 3 and condition <= 25.0 else "fail",
        "reason_codes": [] if rank == 3 and condition <= 25.0 else ["three_parameter_magnetic_pendulum_not_identifiable"],
        "rank": rank, "required_rank": 3, "condition_number": condition, "maximum_condition_number": 25.0,
    }
    family = combine_rule_checks(
        {"forward_equation_match": forward_check, "three_parameter_identifiability": identifiability},
        rule_family="gravity_damping_and_local_magnetic_pendulum",
    )
    zone_support = {
        "status": "pass" if diagnostics["magnet_zone_transition_count"] >= 2 else "indeterminate",
        "reason_codes": [] if diagnostics["magnet_zone_transition_count"] >= 2 else ["magnetic_zone_not_crossed_enough_for_kappa"],
        "transition_count": diagnostics["magnet_zone_transition_count"],
    }
    gravity_quality = combine_rule_checks(
        {"physical_gravity": _domain_check(gravity, lower=0.0, label="gravity")}, rule_family="magnetic_pendulum_gravity"
    )
    beta_quality = combine_rule_checks(
        {"physical_damping": _domain_check(beta, lower=0.0, label="linear_damping")}, rule_family="magnetic_pendulum_damping"
    )
    kappa_quality = combine_rule_checks(
        {"magnet_zone_support": zone_support, "physical_magnetic_strength": _domain_check(kappa, lower=0.0, label="magnetic_kappa")},
        rule_family="localized_magnetic_force",
    )
    diagnostics.update({
        "forward_validation": forward, "detected_cycles": cycles, "detected_extrema": extrema,
        "interpretation_warning": "coupled three-parameter result is publishable only when forward fit, rank, conditioning, physical domains, and magnet-zone support pass",
    })
    return _result(
        "v3_D",
        {"gravity_g": gravity, "linear_damping_beta": beta, "magnetic_kappa": kappa},
        diagnostics,
        method="complete_cycles_three_basis_regression_with_forward_identifiability_gate",
        parameter_quality={"gravity_g": gravity_quality, "linear_damping_beta": beta_quality, "magnetic_kappa": kappa_quality},
        rule_family=family,
        segmentation={
            "algorithm": "complete_magnetic_pendulum_periods", "status": "ok", "cycles": cycles,
            "segments": [{"name": "complete_cycle_support", "start_index": int(indices[0]), "end_index": int(indices[-1])}],
        },
    )


FITTERS = {
    "v1_A": _fit_v1a,
    "v1_B": _fit_v1b,
    "v1_C": _fit_v1c,
    "v1_D": _fit_v1d,
    "v2_A": _fit_v2a,
    "v2_B": _fit_v2b,
    "v2_C": _fit_v2c,
    "v2_D": _fit_v2d,
    "v2_E": _fit_v2e,
    "v3_A": _fit_v3a,
    "v3_B": _fit_v3b,
    "v3_C": _fit_v3c,
    "v3_D": _fit_v3d,
}


def fit_physics_parameters(experiment_id: str, trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit hidden parameters from a metric trajectory without target access."""

    if experiment_id not in FITTERS:
        raise KeyError(f"unsupported experiment: {experiment_id}")
    t, x, z, source_frames = _finite_trajectory(trajectory)
    if len(t) < 4:
        return _failure(experiment_id, "fewer_than_4_metric_points", [])
    result = FITTERS[experiment_id](t, x, z)
    result["fit_input"] = {
        "metric_point_count": int(len(t)),
        "source_frame_indices": [int(value) for value in source_frames],
        "interpolated_points_excluded": True,
        "coordinate_axes_used": ["x_m", "z_m"],
        "motion_plane": "blender_world_xz",
        "ignored_diagnostic_axis": "y_m",
    }
    segmentation = result.get("segmentation")
    if isinstance(segmentation, dict):
        for collection_name in ("events", "segments", "cycles"):
            collection = segmentation.get(collection_name)
            if not isinstance(collection, list):
                continue
            for item in collection:
                if not isinstance(item, dict):
                    continue
                for index_name, frame_name in (
                    ("index", "source_frame"),
                    ("start_index", "start_source_frame"),
                    ("middle_index", "middle_source_frame"),
                    ("end_index", "end_source_frame"),
                ):
                    index = item.get(index_name)
                    if isinstance(index, (int, np.integer)) and 0 <= int(index) < len(source_frames):
                        item[frame_name] = int(source_frames[int(index)])
    return result


def _benchmark_target_range(
    experiment_spec: Mapping[str, Any],
    parameter: Mapping[str, Any],
) -> tuple[float, float, str]:
    """Return the pre-registered target sweep used only as a reporting scale.

    The generated video is never constrained to this interval.  Estimates
    outside the interval are retained and explicitly reported as out of range.
    ``valid_range`` is a compatibility fallback for synthetic/unit-test specs
    that do not contain the frozen anchor tuples.
    """

    name = str(parameter["name"])
    anchor_values: list[float] = []
    for anchor in experiment_spec.get("anchor_tuples", []):
        if not isinstance(anchor, Mapping) or name not in anchor:
            continue
        try:
            value = float(anchor[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            anchor_values.append(value)
    unique = sorted(set(anchor_values))
    if len(unique) >= 2 and unique[-1] - unique[0] > 1e-12:
        return unique[0], unique[-1], "anchor_target_sweep"

    lower, upper = (float(value) for value in parameter["valid_range"])
    return lower, upper, "valid_range_fallback"


def score_parameter_fit(
    fit: Mapping[str, Any],
    experiment_spec: Mapping[str, Any],
    target_parameters: Mapping[str, float],
) -> dict[str, Any]:
    """Score a completed fit against its registry tuple.

    The legacy valid-range NAE fields are retained for compatibility.  The
    simple paper-facing contract adds target, estimate, absolute error,
    pre-registered target-sweep range status, and an auxiliary benchmark-span
    NAE.  Raw estimates are never clipped to either interval.
    """

    estimates = fit.get("parameter_estimates", {})
    observed = fit.get("parameter_observed", {})
    metrics: dict[str, dict[str, Any]] = {}
    scoring_naes: list[float] = []
    raw_naes: list[float] = []
    benchmark_span_naes: list[float] = []
    out_of_target_range_count = 0
    for parameter in experiment_spec.get("hidden_parameters", []):
        name = str(parameter["name"])
        lower, upper = (float(value) for value in parameter["valid_range"])
        target_lower, target_upper, target_range_source = _benchmark_target_range(
            experiment_spec,
            parameter,
        )
        target_span = target_upper - target_lower
        target = float(target_parameters[name])
        estimate_value = estimates.get(name)
        is_observed = bool(observed.get(name, estimate_value is not None)) and estimate_value is not None
        if is_observed and math.isfinite(float(estimate_value)):
            estimate = float(estimate_value)
            absolute_error = abs(estimate - target)
            relative_error = absolute_error / max(abs(target), 1e-12)
            nae = absolute_error / (upper - lower)
            benchmark_span_nae = (
                absolute_error / target_span if target_span > 1e-12 else None
            )
            in_target_range = target_lower <= estimate <= target_upper
            target_range_status = "in_range" if in_target_range else "out_of_range"
            if not in_target_range:
                out_of_target_range_count += 1
            scoring_nae = nae
            score = 100.0 * max(0.0, 1.0 - nae)
            raw_naes.append(nae)
            if benchmark_span_nae is not None:
                benchmark_span_naes.append(benchmark_span_nae)
        else:
            estimate = None
            absolute_error = relative_error = nae = None
            benchmark_span_nae = None
            in_target_range = None
            target_range_status = "not_estimated"
            scoring_nae = 1.0
            score = 0.0
        scoring_naes.append(scoring_nae)
        metrics[name] = {
            "unit": parameter.get("unit"),
            "valid_range": [lower, upper],
            "benchmark_target_range": [target_lower, target_upper],
            "benchmark_target_span": target_span,
            "benchmark_target_range_source": target_range_source,
            "gt": target,
            "target": target,
            "estimate_raw": estimate,
            "estimate": estimate,
            "observed": is_observed,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "in_target_range": in_target_range,
            "target_range_status": target_range_status,
            "benchmark_span_normalized_absolute_error": benchmark_span_nae,
            "bnae_aux": benchmark_span_nae,
            "normalized_absolute_error": nae,
            "scoring_normalized_absolute_error": scoring_nae,
            "score_0_100": score,
        }
    experiment_nmae = float(np.mean(scoring_naes)) if scoring_naes else None
    return {
        "parameters": metrics,
        "experiment_nmae": experiment_nmae,
        "experiment_score_0_100": None if experiment_nmae is None else 100.0 * max(0.0, 1.0 - experiment_nmae),
        "valid_only_nmae": float(np.mean(raw_naes)) if raw_naes else None,
        "valid_only_benchmark_span_nmae": (
            float(np.mean(benchmark_span_naes)) if benchmark_span_naes else None
        ),
        "out_of_target_range_count": int(out_of_target_range_count),
        "fit_complete": bool(metrics) and all(item["observed"] for item in metrics.values()),
        "missing_parameter_count": int(sum(not item["observed"] for item in metrics.values())),
        "metric_definition": "NAE=abs(estimate-gt)/(valid_max-valid_min); score=100*max(0,1-NAE)",
        "simple_metric_definition": (
            "primary fields: target, estimate, absolute_error, trajectory_R2, "
            "target_range_status; auxiliary BNAE=absolute_error/"
            "(max_registered_target-min_registered_target)"
        ),
    }


def lookup_target_tuple(experiment_spec: Mapping[str, Any], tuple_id: str) -> dict[str, float]:
    for anchor in experiment_spec.get("anchor_tuples", []):
        if str(anchor.get("id")) == tuple_id:
            return {
                str(parameter["name"]): float(anchor[str(parameter["name"])])
                for parameter in experiment_spec.get("hidden_parameters", [])
            }
    raise KeyError(f"unknown tuple id {tuple_id!r} for experiment {experiment_spec.get('id')!r}")
