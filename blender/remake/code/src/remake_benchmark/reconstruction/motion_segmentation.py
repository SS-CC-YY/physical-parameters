"""Target-independent temporal segmentation and motion-law diagnostics.

The benchmark must decide *which frames describe a physical phase* before it
looks up the requested parameter value.  This module therefore consumes only
time/trajectory arrays and experiment-known geometry.  Its outputs are plain
JSON-compatible dictionaries so the exact events and segments can be reused by
fitters, graders, plots, and human evidence reports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def smooth_series(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    window = min(int(window), len(values) if len(values) % 2 else len(values) - 1)
    window = max(3, window | 1)
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + window]) for index in range(len(values))],
        dtype=np.float64,
    )


def turning_indices(values: np.ndarray, min_separation: int = 3) -> list[int]:
    values = smooth_series(values)
    if len(values) < 5:
        return []
    velocity = np.gradient(values)
    signs = np.sign(velocity)
    for index in range(1, len(signs)):
        if signs[index] == 0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0:
            signs[index] = signs[index + 1]
    raw = np.flatnonzero(signs[:-1] * signs[1:] < 0) + 1
    output: list[int] = []
    for index in raw.tolist():
        if not output or index - output[-1] >= min_separation:
            output.append(index)
        elif abs(velocity[index]) < abs(velocity[output[-1]]):
            output[-1] = index
    return output


def motion_start_index(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    direction: int = 0,
    consecutive_intervals: int = 2,
) -> int:
    """Return the first frame of sustained motion, excluding a static prefix.

    ``direction`` is ``-1`` for decreasing motion, ``+1`` for increasing
    motion, and ``0`` for either direction.  The threshold is based only on the
    observed trajectory scale and includes a small metric noise floor.
    """

    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return 0
    smoothed = smooth_series(values, window=3)
    dt = np.diff(time_s)
    safe_dt = np.where(dt > 1e-9, dt, np.nan)
    interval_velocity = np.diff(smoothed) / safe_dt
    duration = max(float(time_s[-1] - time_s[0]), 1e-6)
    scale_velocity = float(np.ptp(smoothed)) / duration
    threshold = max(0.005, 0.025 * scale_velocity)
    signed = interval_velocity if direction == 0 else direction * interval_velocity
    moving = np.abs(signed) > threshold if direction == 0 else signed > threshold
    run = max(1, int(consecutive_intervals))
    for index in range(0, len(moving) - run + 1):
        if bool(np.all(moving[index : index + run])):
            return int(index)
    return 0


def motion_stop_index(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    start_index: int,
    direction: int = 0,
    consecutive_intervals: int = 4,
) -> int:
    """Return the first frame of a sustained stationary tail, or the last frame."""

    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return max(0, len(values) - 1)
    smoothed = smooth_series(values, window=5)
    dt = np.diff(time_s)
    safe_dt = np.where(dt > 1e-9, dt, np.nan)
    interval_velocity = np.diff(smoothed) / safe_dt
    moving_scale = float(np.nanpercentile(np.abs(interval_velocity), 80))
    threshold = max(0.01, 0.08 * moving_scale)
    signed = interval_velocity if direction == 0 else direction * interval_velocity
    stationary = np.abs(signed) <= threshold if direction == 0 else signed <= threshold
    run = max(2, int(consecutive_intervals))
    search_start = min(len(stationary), max(int(start_index) + 3, 0))
    for index in range(search_start, len(stationary) - run + 1):
        if bool(np.all(stationary[index : index + run])):
            return int(index)
    return len(values) - 1


def first_contact_index(
    values: np.ndarray,
    *,
    contact_position: float,
    start_index: int,
    tolerance: float = 0.025,
    direction: int = -1,
) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    for index in range(max(0, int(start_index) + 1), len(values)):
        reached = (
            values[index] <= contact_position + tolerance
            if direction < 0
            else values[index] >= contact_position - tolerance
        )
        if reached:
            return int(index)
    return None


def _endpoint_is_extremum(values: np.ndarray, index: int) -> bool:
    if len(values) < 5 or float(np.ptp(values)) <= 1e-9:
        return False
    center = float(np.median(values))
    excursion = abs(float(values[index]) - center)
    return excursion >= 0.28 * float(np.ptp(values))


def complete_cycle_segments(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    start_index: int = 0,
    min_half_cycle_points: int = 4,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Find non-overlapping complete cycles between equal-phase extrema."""

    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    start_index = max(0, min(int(start_index), max(len(values) - 1, 0)))
    local = values[start_index:]
    extrema = [start_index + value for value in turning_indices(local, min_separation=3)]
    if _endpoint_is_extremum(local, 0):
        extrema.insert(0, start_index)
    # A five-second clip often ends very close to the same-phase extremum that
    # completes its first (and only) observable period.  The centered-gradient
    # turning detector cannot emit the last sample, so admit the endpoint only
    # when it is both a large excursion and locally flat compared with the
    # typical inter-frame motion.  This avoids inventing a cycle from a
    # truncated monotonic arc.
    if len(local) >= 5 and _endpoint_is_extremum(local, len(local) - 1):
        local_step = abs(float(local[-1] - local[-2]))
        typical_step = float(np.median(np.abs(np.diff(local))))
        if local_step <= max(1e-8, 0.45 * typical_step):
            extrema.append(len(values) - 1)
    extrema = sorted(set(extrema))
    filtered: list[int] = []
    for value in extrema:
        if not filtered or value - filtered[-1] >= min_half_cycle_points:
            filtered.append(value)
    cycles: list[dict[str, Any]] = []
    cursor = 0
    while cursor + 2 < len(filtered):
        start, middle, end = filtered[cursor : cursor + 3]
        if (
            middle - start >= min_half_cycle_points
            and end - middle >= min_half_cycle_points
        ):
            cycles.append(
                {
                    "cycle_index": len(cycles),
                    "start_index": int(start),
                    "middle_index": int(middle),
                    "end_index": int(end),
                    "start_time_s": float(time_s[start]),
                    "end_time_s": float(time_s[end]),
                    "duration_s": float(time_s[end] - time_s[start]),
                }
            )
            cursor += 2
        else:
            cursor += 1
    return cycles, filtered


def linear_motion_diagnostics(
    time_s: np.ndarray,
    values: np.ndarray,
    indices: Sequence[int],
    *,
    series_name: str,
    maximum_speed_drift_fraction: float = 0.30,
    maximum_speed_relative_mad: float = 0.20,
    maximum_nrmse: float = 0.08,
    minimum_r2: float = 0.94,
) -> dict[str, Any]:
    """Fit one constant-velocity segment and test persistent speed drift."""

    indices = np.asarray(indices, dtype=int)
    if len(indices) < 6:
        return {
            "status": "indeterminate",
            "reason_codes": ["segment_has_fewer_than_6_points"],
            "fit_points": int(len(indices)),
        }
    segment_time = np.asarray(time_s, dtype=np.float64)[indices]
    segment_values = np.asarray(values, dtype=np.float64)[indices]
    coefficients = np.polyfit(segment_time, segment_values, 1)
    predicted = np.polyval(coefficients, segment_time)
    residual = segment_values - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    observed_range = float(np.ptp(segment_values))
    nrmse = None if observed_range <= 1e-12 else rmse / observed_range
    centered = segment_values - float(np.mean(segment_values))
    denominator = float(np.sum(centered**2))
    r2 = None if denominator <= 1e-12 else float(
        1.0 - np.sum(residual**2) / denominator
    )

    local_velocities: list[float] = []
    half_window = min(4, max(2, len(indices) // 8))
    for local_index in range(half_window, len(indices) - half_window):
        window = np.arange(local_index - half_window, local_index + half_window + 1)
        local_velocities.append(
            float(np.polyfit(segment_time[window], segment_values[window], 1)[0])
        )
    velocity = np.asarray(local_velocities, dtype=np.float64)
    reasons: list[str] = []
    if not len(velocity):
        reasons.append("local_velocity_support_insufficient")
        early_speed = late_speed = relative_mad = speed_drift = None
    else:
        speed = np.abs(velocity)
        third = max(1, len(speed) // 3)
        early_speed = float(np.median(speed[:third]))
        late_speed = float(np.median(speed[-third:]))
        median_speed = float(np.median(speed))
        relative_mad = float(
            np.median(np.abs(speed - median_speed)) / max(median_speed, 1e-9)
        )
        speed_drift = abs(late_speed - early_speed) / max(median_speed, 1e-9)
        if speed_drift > maximum_speed_drift_fraction:
            reasons.append("persistent_nonuniform_speed")
        if relative_mad > maximum_speed_relative_mad:
            reasons.append("local_speed_variation_exceeds_limit")
        main_sign = math.copysign(1.0, float(coefficients[0])) if abs(coefficients[0]) > 1e-9 else 0.0
        wrong_fraction = (
            float(np.mean(velocity * main_sign <= 0)) if main_sign else 1.0
        )
        if wrong_fraction > 0.10:
            reasons.append("within_segment_direction_reversal")
    if nrmse is None or r2 is None:
        reasons.append("linear_fit_identifiability_insufficient")
    else:
        if nrmse > maximum_nrmse:
            reasons.append("linear_motion_nrmse_exceeds_limit")
        if r2 < minimum_r2:
            reasons.append("linear_motion_r2_below_limit")

    substantive = [
        reason for reason in reasons if reason != "local_velocity_support_insufficient"
    ]
    return {
        "status": "fail" if substantive else (
            "indeterminate" if reasons else "pass"
        ),
        "reason_codes": reasons,
        "start_index": int(indices[0]),
        "end_index": int(indices[-1]),
        "start_time_s": float(segment_time[0]),
        "end_time_s": float(segment_time[-1]),
        "fit_points": int(len(indices)),
        "velocity": float(coefficients[0]),
        "fit_rmse": rmse,
        "fit_nrmse": nrmse,
        "fit_r2": r2,
        "early_speed": early_speed,
        "late_speed": late_speed,
        "speed_drift_fraction": speed_drift,
        "speed_relative_mad": relative_mad,
        "fit_series": {
            "series_name": str(series_name),
            "time_s": [float(value) for value in segment_time],
            "observed": [float(value) for value in segment_values],
            "predicted": [float(value) for value in predicted],
            "residual": [float(value) for value in residual],
            "time_basis": "video_time_s",
        },
    }


def impact_dwell_diagnostics(
    time_s: np.ndarray,
    values: np.ndarray,
    impact_index: int,
    *,
    maximum_dwell_s: float = 0.20,
) -> dict[str, Any]:
    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    impact_index = int(impact_index)
    tolerance = max(0.025, 0.015 * max(float(np.ptp(values)), 1.0))
    anchor = float(values[impact_index])
    left = impact_index
    right = impact_index
    while left > 0 and abs(float(values[left - 1]) - anchor) <= tolerance:
        left -= 1
    while right + 1 < len(values) and abs(float(values[right + 1]) - anchor) <= tolerance:
        right += 1
    dwell = float(time_s[right] - time_s[left])
    return {
        "status": "fail" if dwell > maximum_dwell_s else "pass",
        "reason_codes": ["impact_sticking_exceeds_limit"] if dwell > maximum_dwell_s else [],
        "impact_index": impact_index,
        "dwell_start_index": int(left),
        "dwell_end_index": int(right),
        "dwell_time_s": dwell,
        "maximum_dwell_s": float(maximum_dwell_s),
        "position_tolerance_m": tolerance,
    }


def combine_rule_checks(
    checks: Mapping[str, Mapping[str, Any]],
    *,
    rule_family: str,
) -> dict[str, Any]:
    statuses = [str(value.get("status", "indeterminate")) for value in checks.values()]
    if "fail" in statuses:
        status = "fail"
    elif "indeterminate" in statuses or not statuses:
        status = "indeterminate"
    else:
        status = "pass"
    reasons: list[str] = []
    for name, value in checks.items():
        reasons.extend(
            f"{name}:{reason}" for reason in value.get("reason_codes", [])
        )
    return {
        "status": status,
        "rule_family": str(rule_family),
        "reason_codes": reasons,
        "checks": {str(name): dict(value) for name, value in checks.items()},
        "target_parameters_used": False,
    }


__all__ = [
    "combine_rule_checks",
    "complete_cycle_segments",
    "first_contact_index",
    "impact_dwell_diagnostics",
    "linear_motion_diagnostics",
    "motion_start_index",
    "motion_stop_index",
    "smooth_series",
    "turning_indices",
]
