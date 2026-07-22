"""Target-free G1 motion-topology checks for the frozen experiments.

This module deliberately does not import the parameter registry or inspect a
parameter tuple.  It asks only whether a sufficiently observed metric
trajectory exhibits the experiment's qualitative motion and event order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _trajectory(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values: list[tuple[int, float, float, float, float]] = []
    for ordinal, row in enumerate(rows):
        # G1 is a metric-trajectory statement.  Display coordinates from a
        # frame rejected by the sphere/geometry fit must not create either a
        # pass or a failure.  Require an explicit fit-use flag and finite
        # fit-space coordinates; never fall back to diagnostic x/y/z values.
        if "physics_fit_used" in row:
            physics_fit_used = _truth(row.get("physics_fit_used"))
            if not physics_fit_used:
                continue
            extraction_fit_eligible = False
        else:
            physics_fit_used = False
            extraction_fit_eligible = _truth(row.get("fit_eligible"))
            if not extraction_fit_eligible:
                continue
        if _truth(row.get("interpolated")):
            continue
        time = _finite(row.get("time_s"))
        if physics_fit_used:
            x = _finite(row.get("x_m"))
            y = _finite(row.get("y_m"))
            z = _finite(row.get("z_m"))
        else:
            x = _finite(row.get("fit_x_m"))
            y = _finite(row.get("fit_y_m"))
            z = _finite(row.get("fit_z_m"))
        if time is None or x is None or z is None:
            continue
        frame = int(row.get("source_frame_index", row.get("frame_index", ordinal)))
        values.append((frame, time, x, 0.0 if y is None else y, z))
    if not values:
        empty = np.empty(0, dtype=float)
        return empty.astype(int), empty, empty, empty, empty
    array = np.asarray(values, dtype=float)
    order = np.argsort(array[:, 1], kind="stable")
    array = array[order]
    unique = np.r_[True, np.diff(array[:, 1]) > 1e-9]
    array = array[unique]
    return array[:, 0].astype(int), array[:, 1], array[:, 2], array[:, 3], array[:, 4]


def _complete_metric_coverage(
    rows: Sequence[Mapping[str, Any]],
    frames: np.ndarray,
    time: np.ndarray,
) -> bool:
    """Whether trusted metric evidence densely covers the decoded clip.

    A missing expected event is a G1 failure only when the event window was
    actually observed.  The full-frame contract retained in trajectory CSVs
    lets us distinguish that case from a truncated or sparse metric track.
    """

    if len(frames) < 2 or len(time) < 2:
        return False
    all_frames: list[int] = []
    for ordinal, row in enumerate(rows):
        try:
            all_frames.append(
                int(row.get("source_frame_index", row.get("frame_index", ordinal)))
            )
        except (TypeError, ValueError):
            continue
    if not all_frames:
        return False
    expected = sorted(set(all_frames))
    observed = sorted(set(int(value) for value in frames))
    expected_start, expected_end = expected[0], expected[-1]
    endpoint_covered = (
        observed[0] <= expected_start + 1
        and observed[-1] >= expected_end - 1
    )
    coverage = len(set(observed).intersection(expected)) / max(len(expected), 1)
    frame_gaps = np.diff(np.asarray(observed, dtype=int))
    frame_continuity = not len(frame_gaps) or int(np.max(frame_gaps)) <= 2
    positive_dt = np.diff(time)
    positive_dt = positive_dt[positive_dt > 1e-9]
    if not len(positive_dt):
        time_continuity = False
    else:
        nominal_dt = float(np.percentile(positive_dt, 25))
        time_continuity = float(np.max(positive_dt)) <= 2.75 * nominal_dt
    return bool(
        endpoint_covered
        and coverage >= 0.90
        and frame_continuity
        and time_continuity
    )


def _smooth(values: np.ndarray, time: np.ndarray, window_s: float) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    output = np.empty_like(values, dtype=float)
    half = max(window_s / 2.0, 1e-6)
    for index, center in enumerate(time):
        mask = np.abs(time - center) <= half
        output[index] = float(np.median(values[mask]))
    return output


def _velocity(values: np.ndarray, time: np.ndarray, window_s: float) -> np.ndarray:
    return np.gradient(_smooth(values, time, window_s), time)


def _turns(values: np.ndarray, time: np.ndarray, window_s: float) -> list[int]:
    velocity = _velocity(values, time, window_s)
    signs = np.sign(velocity)
    for index in range(1, len(signs)):
        if signs[index] == 0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0:
            signs[index] = signs[index + 1]
    raw = (np.flatnonzero(signs[:-1] * signs[1:] < 0) + 1).tolist()
    minimum_gap = max(2, int(round(0.20 / max(float(np.median(np.diff(time))), 1e-6))))
    output: list[int] = []
    for index in raw:
        if not output or index - output[-1] >= minimum_gap:
            output.append(int(index))
    return output


def _significant_turns(
    values: np.ndarray,
    time: np.ndarray,
    turns: Sequence[int],
    *,
    minimum_prominence: float,
    neighborhood_s: float,
) -> tuple[list[int], list[dict[str, float | int]]]:
    """Reject derivative sign flips caused by near-static subpixel jitter."""

    significant: list[int] = []
    evidence: list[dict[str, float | int]] = []
    for value in turns:
        index = int(value)
        left = values[
            (time >= time[index] - neighborhood_s) & (time < time[index])
        ]
        right = values[
            (time > time[index]) & (time <= time[index] + neighborhood_s)
        ]
        if not len(left) or not len(right):
            continue
        prominence = min(
            float(np.max(np.abs(values[index] - left))),
            float(np.max(np.abs(values[index] - right))),
        )
        evidence.append({"index": index, "prominence": prominence})
        if prominence >= minimum_prominence:
            significant.append(index)
    return significant, evidence


def _check(
    name: str,
    status: str,
    *,
    observed: Any = None,
    expected: Any = None,
    evidence_frames: Sequence[int] = (),
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "observed": observed,
        "expected": expected,
        "evidence_frames": [int(value) for value in evidence_frames],
        "reason": reason,
    }


def _direction_check(
    q: np.ndarray,
    time: np.ndarray,
    frames: np.ndarray,
    *,
    direction: float,
    window_s: float,
    minimum_consistency: float,
    minimum_speed: float,
) -> dict[str, Any]:
    velocity = direction * _velocity(q, time, window_s)
    significant = np.abs(velocity) >= minimum_speed
    if int(np.sum(significant)) < 4:
        return _check(
            "directed_motion",
            "fail",
            observed={"significant_velocity_points": int(np.sum(significant))},
            expected=f"at least 4 points moving in direction {direction:+g}",
            reason="no_sustained_directed_motion",
        )
    consistency = float(np.mean(velocity[significant] > 0))
    return _check(
        "directed_motion",
        "pass" if consistency >= minimum_consistency else "fail",
        observed={"direction_consistency": consistency},
        expected={"minimum_direction_consistency": minimum_consistency},
        evidence_frames=frames[significant],
        reason=None if consistency >= minimum_consistency else "frequent_direction_reversal",
    )


def _deceleration_check(
    q: np.ndarray,
    time: np.ndarray,
    frames: np.ndarray,
    *,
    direction: float,
    window_s: float,
    minimum_speed: float,
    name: str = "deceleration",
) -> dict[str, Any]:
    velocity = direction * _velocity(q, time, window_s)
    moving = np.flatnonzero(velocity >= minimum_speed)
    if len(moving) < 8:
        return _check(
            name,
            "indeterminate",
            observed={"moving_points": int(len(moving))},
            expected={"minimum_moving_points": 8},
            reason="insufficient_moving_support",
        )
    quartile = max(3, len(moving) // 4)
    early = float(np.median(velocity[moving[:quartile]]))
    late = float(np.median(velocity[moving[-quartile:]]))
    tolerance = max(0.02, 0.05 * abs(early))
    passed = late <= early - tolerance
    return _check(
        name,
        "pass" if passed else "fail",
        observed={"early_speed_m_s": early, "late_speed_m_s": late},
        expected={"minimum_speed_drop_m_s": tolerance},
        evidence_frames=[int(frames[moving[0]]), int(frames[moving[-1]])],
        reason=None if passed else "speed_does_not_decrease",
    )


def _wall_bounce_check(
    x: np.ndarray,
    time: np.ndarray,
    frames: np.ndarray,
    *,
    wall: float,
    window_s: float,
    tolerance: float,
    minimum_speed: float,
    name: str = "wall_bounce_event",
) -> dict[str, Any]:
    contact = int(np.argmin(np.abs(x - wall)))
    velocity = _velocity(x, time, window_s)
    side = max(2, int(round(0.20 / max(float(np.median(np.diff(time))), 1e-6))))
    if contact < side or contact + side >= len(x):
        return _check(
            name,
            "indeterminate",
            observed={"nearest_wall_distance_m": abs(float(x[contact] - wall))},
            expected={"event_side_points": side},
            evidence_frames=[int(frames[contact])],
            reason="contact_too_close_to_clip_boundary",
        )
    pre = float(np.median(velocity[max(0, contact - side):contact]))
    post = float(np.median(velocity[contact + 1:contact + 1 + side]))
    near = abs(float(x[contact] - wall)) <= tolerance
    passed = near and pre >= minimum_speed and post <= -minimum_speed
    return _check(
        name,
        "pass" if passed else "fail",
        observed={
            "contact_frame": int(frames[contact]),
            "nearest_wall_distance_m": abs(float(x[contact] - wall)),
            "pre_velocity_m_s": pre,
            "post_velocity_m_s": post,
        },
        expected={
            "contact_tolerance_m": tolerance,
            "pre_velocity_min_m_s": minimum_speed,
            "post_velocity_max_m_s": -minimum_speed,
        },
        evidence_frames=frames[max(0, contact - side):min(len(frames), contact + side + 1)],
        reason=None if passed else "wall_contact_and_velocity_reversal_not_observed",
    )


def _vertical_contact_check(
    z: np.ndarray,
    time: np.ndarray,
    frames: np.ndarray,
    *,
    contact_z: float,
    window_s: float,
    tolerance: float,
    minimum_speed: float,
    require_rebound: bool,
    complete_metric_coverage: bool,
) -> dict[str, Any]:
    velocity = _velocity(z, time, window_s)
    side = max(2, int(round(0.20 / max(float(np.median(np.diff(time))), 1e-6))))
    smoothed = _smooth(z, time, window_s)
    candidates = [
        index
        for index in range(1, len(z) - 1)
        if smoothed[index] <= smoothed[index - 1]
        and smoothed[index] <= smoothed[index + 1]
        and abs(float(z[index] - contact_z)) <= tolerance
    ]
    # Collapse flat contact plateaus to their first sample so a resting tail
    # does not create dozens of duplicate impacts.
    collapsed: list[int] = []
    for index in candidates:
        if not collapsed or index - collapsed[-1] > side:
            collapsed.append(index)
    interior = [index for index in collapsed if index >= side and index + side < len(z)]
    measurements: list[tuple[int, float, float, bool]] = []
    for contact in interior:
        pre = float(np.median(velocity[max(0, contact - side):contact]))
        post = float(np.median(velocity[contact + 1:contact + 1 + side]))
        passed = pre <= -minimum_speed and (not require_rebound or post >= minimum_speed)
        measurements.append((contact, pre, post, passed))
    passed_measurements = [item for item in measurements if item[3]]
    if passed_measurements:
        contact, pre, post, passed = passed_measurements[0]
    elif measurements:
        contact, pre, post, passed = min(
            measurements,
            key=lambda item: abs(float(z[item[0]] - contact_z)),
        )
    else:
        contact = int(np.argmin(z))
        distance = abs(float(z[contact] - contact_z))
        interior_minimum = contact >= side and contact + side < len(z)
        if complete_metric_coverage and interior_minimum and distance > tolerance:
            return _check(
                "vertical_bounce_event" if require_rebound else "vertical_contact_event",
                "fail",
                observed={
                    "minimum_z_m": float(z[contact]),
                    "contact_error_m": distance,
                    "metric_coverage_complete": True,
                },
                expected={"contact_tolerance_m": tolerance},
                evidence_frames=[int(frames[contact])],
                reason="expected_vertical_contact_not_observed",
            )
        return _check(
            "vertical_bounce_event" if require_rebound else "vertical_contact_event",
            "indeterminate",
            observed={
                "minimum_z_m": float(z[contact]),
                "contact_error_m": distance,
                "metric_coverage_complete": complete_metric_coverage,
            },
            expected={"event_side_points": side},
            evidence_frames=[int(frames[contact])],
            reason=(
                "contact_too_close_to_clip_boundary"
                if not interior_minimum
                else "insufficient_metric_coverage_for_contact_event"
            ),
        )
    near = abs(float(z[contact] - contact_z)) <= tolerance
    kinematics = pre <= -minimum_speed and (not require_rebound or post >= minimum_speed)
    passed = near and kinematics
    return _check(
        "vertical_bounce_event" if require_rebound else "vertical_contact_event",
        "pass" if passed else "fail",
        observed={
            "contact_frame": int(frames[contact]),
            "minimum_z_m": float(z[contact]),
            "contact_error_m": abs(float(z[contact] - contact_z)),
            "pre_velocity_m_s": pre,
            "post_velocity_m_s": post,
        },
        expected={
            "contact_tolerance_m": tolerance,
            "descending_speed_m_s": minimum_speed,
            "rebound_required": require_rebound,
        },
        evidence_frames=frames[max(0, contact - side):min(len(frames), contact + side + 1)],
        reason=None if passed else "expected_vertical_contact_topology_not_observed",
    )


def _pendulum_checks(
    x: np.ndarray,
    z: np.ndarray,
    time: np.ndarray,
    frames: np.ndarray,
    *,
    pivot: Sequence[float],
    length: float,
    minimum_turns: int,
    window_s: float,
    radius_p95_limit: float,
    minimum_angular_span: float,
    minimum_turn_prominence: float,
) -> tuple[list[dict[str, Any]], np.ndarray, list[int]]:
    px, pz = (float(value) for value in pivot)
    radius = np.hypot(x - px, z - pz)
    radius_error = np.abs(radius - length) / max(length, 1e-9)
    p95 = float(np.percentile(radius_error, 95))
    theta = np.unwrap(np.arctan2(x - px, pz - z))
    raw_turns = _turns(theta, time, window_s)
    turns, turn_evidence = _significant_turns(
        theta,
        time,
        raw_turns,
        minimum_prominence=minimum_turn_prominence,
        neighborhood_s=max(0.50, 2.0 * window_s),
    )
    angular_span = float(np.percentile(theta, 95) - np.percentile(theta, 5))
    motion_significant = angular_span >= minimum_angular_span
    return (
        [
            _check(
                "pendulum_length_constraint",
                "pass" if p95 <= radius_p95_limit else "fail",
                observed={"radius_relative_error_p95": p95},
                expected={"maximum_relative_error_p95": radius_p95_limit},
                reason=None if p95 <= radius_p95_limit else "pendulum_length_not_preserved",
            ),
            _check(
                "pendulum_motion_significance",
                "pass" if motion_significant else "fail",
                observed={
                    "angular_span_rad_p95_p05": angular_span,
                    "angular_span_deg_p95_p05": math.degrees(angular_span),
                },
                expected={
                    "minimum_angular_span_rad": minimum_angular_span,
                    "minimum_angular_span_deg": math.degrees(minimum_angular_span),
                },
                reason=None if motion_significant else "pendulum_motion_is_nearly_static",
            ),
            _check(
                "pendulum_oscillation",
                "pass" if len(turns) >= minimum_turns else "fail",
                observed={
                    "raw_turn_count": len(raw_turns),
                    "significant_turn_count": len(turns),
                    "turn_prominence_rad": turn_evidence,
                },
                expected={
                    "minimum_turn_count": minimum_turns,
                    "minimum_turn_prominence_rad": minimum_turn_prominence,
                    "minimum_turn_prominence_deg": math.degrees(minimum_turn_prominence),
                },
                evidence_frames=[int(frames[index]) for index in turns],
                reason=None if len(turns) >= minimum_turns else "insufficient_pendulum_reversals",
            ),
        ],
        theta,
        turns,
    )


def evaluate_motion_type(
    experiment_id: str,
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate G1 motion type without looking at the requested parameters."""

    frames, time, x, _y, z = _trajectory(rows)
    thresholds = dict(profile.get("global_thresholds", {}))
    thresholds.setdefault("minimum_pendulum_angular_span_deg", 8.0)
    thresholds.setdefault("minimum_pendulum_turn_prominence_deg", 3.0)
    thresholds.setdefault("minimum_ramp_vertical_drop_m", 0.05)
    thresholds.setdefault("maximum_floor_vertical_slope", 0.05)
    thresholds.setdefault("maximum_floor_vertical_residual_m", 0.10)
    experiment = profile.get("experiments", {}).get(experiment_id)
    base: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "target_parameters_used": False,
        "status": "indeterminate",
        "motion_type": None if experiment is None else experiment.get("type"),
        "direct_metric_point_count": int(len(time)),
        "checks": [],
        "failure_codes": [],
        "indeterminate_codes": [],
        "event_sequence": [],
        "thresholds": thresholds,
    }
    if experiment is None:
        base["indeterminate_codes"] = ["missing_motion_type_profile"]
        return base
    minimum_points = int(thresholds.get("minimum_direct_points", 12))
    minimum_time = float(thresholds.get("minimum_time_span_s", 0.5))
    if len(time) < minimum_points:
        base["indeterminate_codes"] = ["insufficient_direct_metric_points"]
        return base
    if float(time[-1] - time[0]) < minimum_time:
        base["indeterminate_codes"] = ["insufficient_observed_time_span"]
        return base

    complete_metric_coverage = _complete_metric_coverage(rows, frames, time)

    window_s = float(thresholds.get("derivative_window_s", 0.25))
    minimum_span = float(thresholds.get("minimum_translation_span_m", 0.25))
    minimum_consistency = float(thresholds.get("minimum_direction_consistency", 0.8))
    minimum_speed = float(thresholds.get("minimum_significant_speed_m_s", 0.05))
    tolerance = float(thresholds.get("contact_tolerance_m", 0.17))
    motion_type = str(experiment["type"])
    checks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    span_x = float(np.ptp(x))
    span_z = float(np.ptp(z))
    if motion_type not in {"damped_pendulum", "magnetic_damped_pendulum"}:
        span = max(span_x, span_z)
        checks.append(
            _check(
                "nontrivial_motion_span",
                "pass" if span >= minimum_span else "fail",
                observed={"x_span_m": span_x, "z_span_m": span_z, "maximum_span_m": span},
                expected={"minimum_span_m": minimum_span},
                reason=None if span >= minimum_span else "trajectory_is_nearly_static",
            )
        )

    if motion_type == "free_fall":
        checks.append(_direction_check(z, time, frames, direction=-1, window_s=window_s, minimum_consistency=minimum_consistency, minimum_speed=minimum_speed))
        contact_z = float(experiment["contact_z_m"])
        airborne = np.flatnonzero(z > contact_z + 0.025)
        if len(airborne) >= 8:
            gaps = np.flatnonzero(np.diff(airborne) > 1)
            if len(gaps):
                airborne = airborne[: gaps[0] + 1]
            tau = time[airborne] - time[airborne[0]]
            acceleration = float(2.0 * np.polyfit(tau, z[airborne], 2)[0])
            checks.append(_check("downward_curvature", "pass" if acceleration < -0.05 else "fail", observed={"quadratic_acceleration_m_s2": acceleration, "airborne_points": len(airborne)}, expected="negative vertical acceleration on the first airborne arc", evidence_frames=frames[airborne], reason=None if acceleration < -0.05 else "free_fall_curvature_not_negative"))
        else:
            checks.append(_check("downward_curvature", "indeterminate", observed={"airborne_points": len(airborne)}, expected={"minimum_airborne_points": 8}, reason="insufficient_airborne_curvature_support"))
        contact = _vertical_contact_check(z, time, frames, contact_z=contact_z, window_s=window_s, tolerance=tolerance, minimum_speed=minimum_speed, require_rebound=False, complete_metric_coverage=complete_metric_coverage)
        checks.append(contact)
        events.append({"name": "contact", "frame": contact.get("observed", {}).get("contact_frame")})
    elif motion_type == "single_wall_bounce":
        # A successful bounce necessarily contains both positive approach and
        # negative departure.  A whole-clip direction-consistency test would
        # therefore reject the very topology we want; the event-local velocity
        # signs below are the correct predicate.
        wall = _wall_bounce_check(x, time, frames, wall=float(experiment["wall_x_m"]), window_s=window_s, tolerance=tolerance, minimum_speed=minimum_speed)
        checks.append(wall)
        events.append({"name": "right_wall_impact", "frame": wall.get("observed", {}).get("contact_frame")})
    elif motion_type == "one_way_friction_slide":
        direction = float(experiment.get("direction", 1))
        checks.append(_direction_check(x, time, frames, direction=direction, window_s=window_s, minimum_consistency=minimum_consistency, minimum_speed=minimum_speed))
        checks.append(_deceleration_check(x, time, frames, direction=direction, window_s=window_s, minimum_speed=minimum_speed))
    elif motion_type in {"damped_pendulum", "magnetic_damped_pendulum"}:
        pendulum, theta, turns = _pendulum_checks(
            x,
            z,
            time,
            frames,
            pivot=experiment["pivot_xz_m"],
            length=float(experiment["length_m"]),
            minimum_turns=int(experiment.get("minimum_turns", 2)),
            window_s=window_s,
            radius_p95_limit=float(thresholds.get("maximum_pendulum_radius_relative_p95", 0.15)),
            minimum_angular_span=math.radians(float(thresholds["minimum_pendulum_angular_span_deg"])),
            minimum_turn_prominence=math.radians(float(thresholds["minimum_pendulum_turn_prominence_deg"])),
        )
        checks.extend(pendulum)
        events.extend({"name": "turn", "frame": int(frames[index])} for index in turns)
        if motion_type == "magnetic_damped_pendulum":
            center = math.radians(float(experiment["magnet_center_deg"]))
            width = math.radians(float(experiment["magnet_width_deg"]))
            inside = np.abs(theta - center) <= width
            transitions = (np.flatnonzero(inside[1:] != inside[:-1]) + 1).tolist()
            required = int(experiment.get("minimum_magnet_zone_transitions", 1))
            checks.append(_check("magnet_zone_traversal", "pass" if len(transitions) >= required else "fail", observed={"transition_count": len(transitions)}, expected={"minimum_transition_count": required}, evidence_frames=[int(frames[index]) for index in transitions], reason=None if len(transitions) >= required else "magnet_zone_not_traversed"))
            events.extend({"name": "magnet_zone_transition", "frame": int(frames[index])} for index in transitions)
    elif motion_type == "bounded_periodic_track":
        turns = _turns(x, time, window_s)
        required = int(experiment.get("minimum_turns", 2))
        checks.append(_check("bounded_periodic_reversals", "pass" if len(turns) >= required else "fail", observed={"turn_count": len(turns)}, expected={"minimum_turn_count": required}, evidence_frames=[int(frames[index]) for index in turns], reason=None if len(turns) >= required else "periodic_track_reversals_not_observed"))
        events.extend({"name": "turn", "frame": int(frames[index])} for index in turns)
    elif motion_type in {"alternating_wall_bounces", "frictional_alternating_wall_bounces"}:
        turns = _turns(x, time, window_s)
        required = int(experiment.get("minimum_turns", 2))
        wall_geometry = str(experiment.get("wall_geometry", "frozen"))
        if motion_type == "alternating_wall_bounces" and wall_geometry != "frozen":
            # Reversals alone do not prove wall contact.  Until V2_B's wall
            # locations are frozen, a periodic free-space trajectory is
            # observationally indistinguishable from the requested topology.
            checks.append(
                _check(
                    "alternating_reversals",
                    "indeterminate",
                    observed={
                        "turn_count": len(turns),
                        "wall_geometry": wall_geometry,
                    },
                    expected={
                        "minimum_turn_count": required,
                        "frozen_wall_geometry": True,
                    },
                    evidence_frames=[int(frames[index]) for index in turns],
                    reason="wall_geometry_unavailable_for_bounce_validation",
                )
            )
        else:
            checks.append(_check("alternating_reversals", "pass" if len(turns) >= required else "fail", observed={"turn_count": len(turns), "wall_geometry": wall_geometry}, expected={"minimum_turn_count": required}, evidence_frames=[int(frames[index]) for index in turns], reason=None if len(turns) >= required else "alternating_wall_reversals_not_observed"))
        events.extend({"name": "wall_turn", "frame": int(frames[index])} for index in turns)
        if motion_type == "frictional_alternating_wall_bounces" and turns:
            left_wall = float(experiment["left_wall_x_m"])
            right_wall = float(experiment["right_wall_x_m"])
            snap_radius = max(2, int(round(window_s / max(float(np.median(np.diff(time))), 1e-6))))
            snapped_turns: list[int] = []
            for index in turns:
                start = max(0, index - snap_radius)
                stop = min(len(x), index + snap_radius + 1)
                local = np.arange(start, stop)
                snapped_turns.append(int(local[np.argmin(np.minimum(np.abs(x[local] - left_wall), np.abs(x[local] - right_wall)))]))
            labels: list[str | None] = []
            for index in snapped_turns:
                left_distance = abs(float(x[index] - left_wall))
                right_distance = abs(float(x[index] - right_wall))
                distance = min(left_distance, right_distance)
                labels.append(("left" if left_distance <= right_distance else "right") if distance <= tolerance else None)
            geometry_pass = all(label is not None for label in labels) and all(
                left != right for left, right in zip(labels, labels[1:])
            ) and {label for label in labels if label is not None} == {"left", "right"}
            checks.append(_check("frozen_wall_geometry_and_alternation", "pass" if geometry_pass else "fail", observed={"turn_wall_labels": labels, "turn_x_m": [float(x[index]) for index in snapped_turns], "derivative_turn_frames": [int(frames[index]) for index in turns]}, expected={"left_wall_x_m": left_wall, "right_wall_x_m": right_wall, "contact_tolerance_m": tolerance, "alternating": True}, evidence_frames=[int(frames[index]) for index in snapped_turns], reason=None if geometry_pass else "turns_do_not_match_alternating_frozen_walls"))
        if motion_type == "frictional_alternating_wall_bounces" and len(turns) >= required:
            speed = np.abs(_velocity(x, time, window_s))
            segment_edges = [0] + turns + [len(x) - 1]
            decreases: list[bool] = []
            for left, right in zip(segment_edges[:-1], segment_edges[1:]):
                if right - left < 6:
                    continue
                width = max(2, (right - left) // 4)
                decreases.append(float(np.median(speed[right - width:right])) <= float(np.median(speed[left:left + width])) + 0.03)
            checks.append(_check("between_wall_frictional_deceleration", "pass" if decreases and float(np.mean(decreases)) >= 0.5 else "fail", observed={"segment_pass_fraction": float(np.mean(decreases)) if decreases else None}, expected={"minimum_segment_pass_fraction": 0.5}, reason=None if decreases and float(np.mean(decreases)) >= 0.5 else "between_wall_speed_does_not_decrease"))
    elif motion_type == "two_surface_friction_slide":
        direction = float(experiment.get("direction", 1))
        transition = float(experiment["transition_x_m"])
        checks.append(_direction_check(x, time, frames, direction=direction, window_s=window_s, minimum_consistency=minimum_consistency, minimum_speed=minimum_speed))
        left = np.flatnonzero(x < transition - 0.03)
        right = np.flatnonzero(x > transition + 0.03)
        both = len(left) >= 6 and len(right) >= 6
        checks.append(_check("both_surfaces_observed", "pass" if both else "fail", observed={"surface_A_points": len(left), "surface_B_points": len(right)}, expected={"minimum_points_per_surface": 6}, reason=None if both else "surface_transition_not_completed"))
        if both:
            checks.append(_deceleration_check(x[left], time[left], frames[left], direction=direction, window_s=window_s, minimum_speed=minimum_speed, name="surface_A_deceleration"))
            checks.append(_deceleration_check(x[right], time[right], frames[right], direction=direction, window_s=window_s, minimum_speed=minimum_speed, name="surface_B_deceleration"))
            events.append({"name": "surface_transition", "frame": int(frames[int(np.argmin(np.abs(x - transition)))])})
    elif motion_type == "vertical_bounce":
        contact = _vertical_contact_check(z, time, frames, contact_z=float(experiment["contact_z_m"]), window_s=window_s, tolerance=tolerance, minimum_speed=minimum_speed, require_rebound=True, complete_metric_coverage=complete_metric_coverage)
        checks.append(contact)
        events.append({"name": "ground_impact", "frame": contact.get("observed", {}).get("contact_frame")})
    elif motion_type == "dragged_projectile_bounce":
        checks.append(_direction_check(x, time, frames, direction=float(experiment.get("horizontal_direction", 1)), window_s=window_s, minimum_consistency=0.70, minimum_speed=minimum_speed))
        contact = _vertical_contact_check(z, time, frames, contact_z=float(experiment["contact_z_m"]), window_s=window_s, tolerance=tolerance, minimum_speed=minimum_speed, require_rebound=True, complete_metric_coverage=complete_metric_coverage)
        checks.append(contact)
        events.append({"name": "ground_impact", "frame": contact.get("observed", {}).get("contact_frame")})
    elif motion_type == "ramp_floor_wall_bounce":
        transition = float(experiment["floor_transition_x_m"])
        wall_x = float(experiment["wall_x_m"])
        transition_index = int(np.argmin(np.abs(x - transition)))
        wall_index = int(np.argmin(np.abs(x - wall_x)))
        order_ok = transition_index < wall_index and abs(float(x[transition_index] - transition)) <= tolerance
        checks.append(_check("ramp_to_floor_order", "pass" if order_ok else "fail", observed={"transition_frame": int(frames[transition_index]), "wall_nearest_frame": int(frames[wall_index])}, expected="floor transition before wall impact", evidence_frames=[int(frames[transition_index]), int(frames[wall_index])], reason=None if order_ok else "ramp_floor_event_order_not_observed"))

        indices = np.arange(len(x), dtype=int)
        ramp = indices[
            (indices <= transition_index)
            & (x < transition - 0.03)
        ]
        floor = indices[
            (indices >= transition_index)
            & (indices <= wall_index)
            & (x > transition + 0.03)
            & (x < wall_x - tolerance)
        ]
        stage_support = len(ramp) >= 6 and len(floor) >= 6
        support_status = (
            "pass"
            if stage_support
            else "fail"
            if complete_metric_coverage
            else "indeterminate"
        )
        checks.append(
            _check(
                "ramp_and_floor_stage_support",
                support_status,
                observed={
                    "ramp_points": int(len(ramp)),
                    "floor_points": int(len(floor)),
                    "metric_coverage_complete": complete_metric_coverage,
                },
                expected={"minimum_points_per_stage": 6},
                reason=(
                    None
                    if stage_support
                    else "ramp_or_floor_stage_not_observed"
                    if complete_metric_coverage
                    else "insufficient_metric_coverage_for_ramp_floor_stages"
                ),
            )
        )
        if stage_support:
            ramp_coefficient = np.polyfit(x[ramp], z[ramp], 1)
            ramp_slope = float(ramp_coefficient[0])
            ramp_drop = -ramp_slope * float(np.ptp(x[ramp]))
            minimum_drop = float(thresholds["minimum_ramp_vertical_drop_m"])
            ramp_pass = ramp_slope < 0.0 and ramp_drop >= minimum_drop
            checks.append(
                _check(
                    "ramp_descent_geometry",
                    "pass" if ramp_pass else "fail",
                    observed={
                        "dz_dx": ramp_slope,
                        "fitted_vertical_drop_m": ramp_drop,
                    },
                    expected={
                        "dz_dx_sign": "negative",
                        "minimum_vertical_drop_m": minimum_drop,
                    },
                    evidence_frames=[int(frames[ramp[0]]), int(frames[ramp[-1]])],
                    reason=None if ramp_pass else "ramp_descent_not_observed",
                )
            )

            floor_coefficient = np.polyfit(x[floor], z[floor], 1)
            floor_slope = float(floor_coefficient[0])
            floor_residual = z[floor] - np.polyval(floor_coefficient, x[floor])
            floor_residual_p95 = float(np.percentile(np.abs(floor_residual), 95))
            maximum_floor_slope = float(thresholds["maximum_floor_vertical_slope"])
            maximum_floor_residual = float(thresholds["maximum_floor_vertical_residual_m"])
            floor_pass = (
                abs(floor_slope) <= maximum_floor_slope
                and floor_residual_p95 <= maximum_floor_residual
            )
            checks.append(
                _check(
                    "floor_planarity_geometry",
                    "pass" if floor_pass else "fail",
                    observed={
                        "dz_dx": floor_slope,
                        "vertical_residual_p95_m": floor_residual_p95,
                    },
                    expected={
                        "maximum_abs_dz_dx": maximum_floor_slope,
                        "maximum_vertical_residual_p95_m": maximum_floor_residual,
                    },
                    evidence_frames=[int(frames[floor[0]]), int(frames[floor[-1]])],
                    reason=None if floor_pass else "floor_segment_not_planar",
                )
            )
        wall = _wall_bounce_check(x, time, frames, wall=wall_x, window_s=window_s, tolerance=tolerance, minimum_speed=minimum_speed)
        checks.append(wall)
        events.extend([{"name": "floor_transition", "frame": int(frames[transition_index])}, {"name": "right_wall_impact", "frame": wall.get("observed", {}).get("contact_frame")}])
    else:
        base["indeterminate_codes"] = ["unsupported_motion_type"]
        return base

    failures = [check for check in checks if check["status"] == "fail"]
    indeterminate = [check for check in checks if check["status"] == "indeterminate"]
    if failures:
        status = "fail"
    elif indeterminate:
        status = "indeterminate"
    else:
        status = "pass"
    base.update(
        status=status,
        checks=checks,
        failure_codes=[str(check.get("reason") or check["name"]) for check in failures],
        indeterminate_codes=[str(check.get("reason") or check["name"]) for check in indeterminate],
        event_sequence=[event for event in events if event.get("frame") is not None],
        trajectory_summary={
            "time_span_s": float(time[-1] - time[0]),
            "x_span_m": span_x,
            "z_span_m": span_z,
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "complete_metric_coverage": complete_metric_coverage,
        },
    )
    return base


__all__ = ["evaluate_motion_type"]
