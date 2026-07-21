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


EXPERIMENT_CONSTANTS: dict[str, dict[str, Any]] = {
    "v1_A": {"contact_z": 0.44},
    "v1_B": {"wall_contact_x": 2.96},
    "v1_C": {"gravity_g": 9.81},
    "v1_D": {"pivot": [0.0, 0.0, 3.8], "length": 2.3, "period": 3.0},
    "v2_A": {"cycloid_a": 0.65},
    "v2_B": {},
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


def _finite_trajectory(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: list[tuple[float, float, float]] = []
    for row in rows:
        # Reconstructors may retain diagnostic xyz values for frames rejected
        # by the known-sphere size or geometry gate.  Those values are useful
        # in overlays but must never leak into the inverse physics fit.
        if row.get("measurement_valid") is False or row.get("physics_fit_used") is False:
            continue
        try:
            time_s = float(row["time_s"])
            x_m = float(row["x_m"])
            z_m = float(row["z_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(time_s) and math.isfinite(x_m) and math.isfinite(z_m):
            values.append((time_s, x_m, z_m))
    if not values:
        return np.empty(0), np.empty(0), np.empty(0)
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    unique = np.r_[True, np.diff(array[:, 0]) > 1e-9]
    return array[unique, 0], array[unique, 1], array[unique, 2]


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


def _fit_diagnostics(observed: np.ndarray, predicted: np.ndarray, parameter_count: int) -> dict[str, Any]:
    residual = np.asarray(observed) - np.asarray(predicted)
    rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
    centered = np.asarray(observed) - float(np.mean(observed)) if len(observed) else np.empty(0)
    denominator = float(np.sum(centered**2))
    r2 = None if denominator <= 1e-12 else float(1.0 - np.sum(residual**2) / denominator)
    return {
        "fit_points": int(len(observed)),
        "fit_parameter_count": int(parameter_count),
        "fit_rmse": rmse,
        "fit_r2": r2,
    }


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    window = min(int(window), len(values) if len(values) % 2 else len(values) - 1)
    window = max(3, window | 1)
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.asarray([np.median(padded[i : i + window]) for i in range(len(values))])


def _turning_indices(values: np.ndarray, min_separation: int = 3) -> list[int]:
    values = _smooth(values)
    if len(values) < 5:
        return []
    velocity = np.gradient(values)
    signs = np.sign(velocity)
    # A sampled extremum commonly has an exactly zero centered difference.
    # Carry neighbouring signs through zero plateaus before testing a reversal.
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
        "parameter_estimates": {name: None for name in parameters},
        "parameter_observed": {name: False for name in parameters},
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
) -> dict[str, Any]:
    clean: dict[str, float | None] = {}
    for name, value in estimates.items():
        clean[name] = float(value) if value is not None and math.isfinite(float(value)) else None
    observed_map = {
        name: bool(clean[name] is not None) if observed is None else bool(observed.get(name, False))
        for name in clean
    }
    status = "ok" if all(observed_map.values()) else "partial"
    return {
        "experiment_id": experiment_id,
        "status": status,
        "method": method,
        "parameter_estimates": clean,
        "parameter_observed": observed_map,
        "diagnostics": dict(diagnostics),
        "target_not_used_for_fit": True,
    }


def _fit_quadratic(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    tau = times - times[0]
    design = np.column_stack((np.ones(len(tau)), tau, tau**2))
    coefficients, _ = _robust_lstsq(design, values)
    return coefficients, _fit_diagnostics(values, design @ coefficients, 3)


def _fit_v1a(t: np.ndarray, _x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    contact = float(EXPERIMENT_CONSTANTS["v1_A"]["contact_z"])
    airborne = z > contact + 0.025
    indices = np.flatnonzero(airborne)
    if len(indices) < 8:
        return _failure("v1_A", "fewer_than_8_airborne_points", ["gravity_g"])
    # Use the first contiguous airborne run, avoiding the static contact tail.
    gaps = np.flatnonzero(np.diff(indices) > 1)
    if len(gaps):
        indices = indices[: gaps[0] + 1]
    coefficients, diagnostics = _fit_quadratic(t[indices], z[indices])
    diagnostics.update({"airborne_start_index": int(indices[0]), "airborne_end_index": int(indices[-1])})
    return _result(
        "v1_A",
        {"gravity_g": -2.0 * coefficients[2]},
        diagnostics,
        method="robust_airborne_quadratic",
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
    event = _best_piecewise_linear_impact(
        t,
        x,
        expected_position=float(EXPERIMENT_CONSTANTS["v1_B"]["wall_contact_x"]),
    )
    if event is None or event["v_pre"] <= 0 or event["v_post"] >= 0:
        return _failure("v1_B", "wall_velocity_reversal_not_found", ["restitution_e"])
    estimate = abs(float(event["v_post"]) / float(event["v_pre"]))
    return _result(
        "v1_B",
        {"restitution_e": estimate},
        event,
        method="robust_piecewise_linear_velocity_ratio",
    )


def _fit_v1c(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    if len(t) < 8:
        return _failure("v1_C", "fewer_than_8_points", ["kinetic_friction_mu"])
    smoothed = _smooth(x)
    velocity = np.gradient(smoothed, t)
    moving = np.flatnonzero(velocity > max(0.03, 0.03 * float(np.nanmax(np.abs(velocity)))))
    if len(moving) >= 8:
        indices = np.arange(0, moving[-1] + 1)
    else:
        indices = np.arange(len(t))
    coefficients, diagnostics = _fit_quadratic(t[indices], x[indices])
    acceleration = 2.0 * float(coefficients[2])
    diagnostics.update({"estimated_acceleration_m_s2": acceleration, "moving_fit_points": int(len(indices))})
    return _result(
        "v1_C",
        {"kinetic_friction_mu": -acceleration / float(EXPERIMENT_CONSTANTS["v1_C"]["gravity_g"])},
        diagnostics,
        method="robust_pre_stop_quadratic",
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
    diagnostics = _fit_diagnostics(theta, best_pred, 3)
    turns = _turning_indices(theta)
    diagnostics.update({"turning_point_indices": turns, "turning_point_count": len(turns), "fixed_period_s": period})
    return best_beta, diagnostics


def _fit_v1d(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v1_D"]
    theta = _pendulum_angle(x, z, constants["pivot"])
    if len(theta) < 12:
        return _failure("v1_D", "fewer_than_12_points", ["amplitude_decay_beta"])
    beta, diagnostics = _fit_fixed_frequency_decay(t, theta, float(constants["period"]), (0.0, 1.0))
    return _result(
        "v1_D",
        {"amplitude_decay_beta": beta},
        diagnostics,
        method="fixed_period_bounded_decay_search",
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
    return best_omega, best_pred, _fit_diagnostics(q, best_pred, 8)


def _fit_v2a(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    if len(t) < 24:
        return _failure("v2_A", "fewer_than_24_points", ["gravity_g"])
    a = float(EXPERIMENT_CONSTANTS["v2_A"]["cycloid_a"])
    omega_min = math.sqrt(2.0 / (4.0 * a))
    omega_max = math.sqrt(14.7 / (4.0 * a))
    omega, predicted, diagnostics = _fit_periodic_frequency(t, x, (omega_min, omega_max))
    period = 2.0 * math.pi / omega
    diagnostics.update({
        "estimated_period_s": period,
        "turning_point_indices": _turning_indices(predicted),
    })
    return _result(
        "v2_A",
        {"gravity_g": 4.0 * a * omega**2},
        diagnostics,
        method="three_harmonic_bounded_period_search",
    )


def _linear_segment_velocities(t: np.ndarray, q: np.ndarray, turns: Sequence[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for indices in _segments_from_turns(len(q), turns, guard=1):
        coefficients = np.polyfit(t[indices], q[indices], 1)
        predicted = np.polyval(coefficients, t[indices])
        output.append({
            "start": int(indices[0]),
            "end": int(indices[-1]),
            "velocity": float(coefficients[0]),
            "rmse": float(np.sqrt(np.mean((q[indices] - predicted) ** 2))),
        })
    return output


def _fit_v2b(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    turns = _turning_indices(x, min_separation=3)
    segments = _linear_segment_velocities(t, x, turns)
    ratios = [
        abs(segments[index + 1]["velocity"] / segments[index]["velocity"])
        for index in range(len(segments) - 1)
        if abs(segments[index]["velocity"]) > 1e-6
        and segments[index]["velocity"] * segments[index + 1]["velocity"] < 0
    ]
    if not ratios:
        return _failure("v2_B", "no_two_wall_velocity_reversal", ["restitution_e"])
    estimate = float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-9)))))
    diagnostics = {"impact_indices": turns, "impact_count": len(ratios), "per_impact_e": ratios, "segments": segments}
    return _result("v2_B", {"restitution_e": estimate}, diagnostics, method="piecewise_line_geometric_mean_ratio")


def _fit_v2c(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    transition = float(EXPERIMENT_CONSTANTS["v2_C"]["transition_x"])
    left = np.flatnonzero(x < transition - 0.03)
    right = np.flatnonzero(x > transition + 0.03)
    if len(left) < 6 or len(right) < 6:
        return _failure("v2_C", "both_friction_surfaces_not_observed", ["kinetic_friction_mu_A", "kinetic_friction_mu_B"])
    coef_a, diag_a = _fit_quadratic(t[left], x[left])
    # Exclude a static tail on B using positive velocity support.
    right_values = _smooth(x[right])
    right_velocity = np.gradient(right_values, t[right])
    moving_right = right[right_velocity > 0.02]
    if len(moving_right) >= 6:
        right = moving_right
    coef_b, diag_b = _fit_quadratic(t[right], x[right])
    normal_over_mass = float(EXPERIMENT_CONSTANTS["v2_C"]["normal_over_mass"])
    estimates = {
        "kinetic_friction_mu_A": -2.0 * coef_a[2] / normal_over_mass,
        "kinetic_friction_mu_B": -2.0 * coef_b[2] / normal_over_mass,
    }
    diagnostics = {
        "transition_nearest_index": int(np.argmin(np.abs(x - transition))),
        "surface_A": diag_a,
        "surface_B": diag_b,
        "acceleration_A_m_s2": float(2.0 * coef_a[2]),
        "acceleration_B_m_s2": float(2.0 * coef_b[2]),
    }
    return _result("v2_C", estimates, diagnostics, method="two_segment_robust_quadratic")


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
    diagnostics = _fit_diagnostics(target[keep], predicted, len(coefficients))
    diagnostics.update({
        "design_matrix_rank": int(np.linalg.matrix_rank(design[keep])),
        "normalized_condition_number": condition,
        "turning_point_indices": _turning_indices(theta),
    })
    return coefficients, diagnostics


def _fit_v2d(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v2_D"]
    # V2D's Blender positive angle places the ball at negative world X.
    theta = _pendulum_angle(x, z, constants["pivot"], sign=-1.0)
    if len(theta) < 16:
        return _failure("v2_D", "fewer_than_16_points", ["gravity_g", "linear_damping_beta"])
    try:
        coefficients, diagnostics = _fit_pendulum_integral(t, theta, length=float(constants["length"]))
    except ValueError as error:
        return _failure("v2_D", str(error), ["gravity_g", "linear_damping_beta"])
    return _result(
        "v2_D",
        {"gravity_g": coefficients[0], "linear_damping_beta": coefficients[1]},
        diagnostics,
        method="double_integral_robust_regression",
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
    diagnostics = _fit_diagnostics(target, design @ coefficients, len(coefficients))
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
    diagnostics.update({"impact_indices": impacts, "per_impact_e": ratios})
    return _result(
        "v2_E",
        {"gravity_g": shared, "restitution_e": restitution},
        diagnostics,
        observed={"gravity_g": True, "restitution_e": restitution is not None},
        method="shared_ballistic_curvature_and_impact_velocity_ratio",
    )


def _fit_v3a(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    if len(t) < 16:
        return _failure("v3_A", "fewer_than_16_points", ["gravity_g", "linear_drag_beta", "restitution_e"])
    tau = t - t[0]
    best_beta = 0.1
    best_loss = float("inf")
    best_pred = np.zeros_like(x)
    for beta in np.linspace(0.05, 0.35, 601):
        design = np.column_stack((np.ones(len(tau)), 1.0 - np.exp(-beta * tau)))
        coefficients = np.linalg.lstsq(design, x, rcond=None)[0]
        predicted = design @ coefficients
        loss = float(np.mean((x - predicted) ** 2))
        if loss < best_loss:
            best_beta, best_loss, best_pred = float(beta), loss, predicted
    impacts = _impact_minima(z, float(EXPERIMENT_CONSTANTS["v3_A"]["contact_z"]))
    try:
        shared, flights, diagnostics = _fit_shared_flight_model(t, z, impacts, drag_beta=best_beta)
        gravity = -best_beta * shared
    except ValueError:
        flights, diagnostics, gravity = [], _fit_diagnostics(x, best_pred, 3), None
    restitution, ratios = _restitution_from_flights(flights)
    diagnostics.update({
        "horizontal_fit": _fit_diagnostics(x, best_pred, 3),
        "impact_indices": impacts,
        "per_impact_e": ratios,
    })
    return _result(
        "v3_A",
        {"gravity_g": gravity, "linear_drag_beta": best_beta, "restitution_e": restitution},
        diagnostics,
        observed={"gravity_g": gravity is not None, "linear_drag_beta": True, "restitution_e": restitution is not None},
        method="horizontal_exponential_plus_shared_drag_flights",
    )


def _fit_v3b(t: np.ndarray, x: np.ndarray, _z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v3_B"]
    turns = _turning_indices(x, min_separation=3)
    segments = _segments_from_turns(len(x), turns, guard=1)
    if not segments:
        return _failure("v3_B", "no_wall_segments", ["kinetic_friction_mu_k", "left_restitution_e_L", "right_restitution_e_R"])
    column_count = 2 * len(segments) + 1
    blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    gravity = float(constants["gravity_g"])
    for segment_id, indices in enumerate(segments):
        tau = t[indices] - t[indices[0]]
        direction = float(np.sign(x[indices[-1]] - x[indices[0]]))
        if direction == 0:
            continue
        block = np.zeros((len(indices), column_count))
        block[:, 2 * segment_id] = 1.0
        block[:, 2 * segment_id + 1] = tau
        block[:, -1] = -0.5 * gravity * direction * tau**2
        blocks.append(block)
        targets.append(x[indices])
        meta.append({"segment_id": segment_id, "indices": indices, "direction": direction})
    if not blocks:
        return _failure("v3_B", "no_moving_wall_segments", ["kinetic_friction_mu_k", "left_restitution_e_L", "right_restitution_e_R"])
    design = np.vstack(blocks)
    target = np.concatenate(targets)
    coefficients, _ = _robust_lstsq(design, target)
    mu = float(coefficients[-1])
    velocities: list[dict[str, Any]] = []
    for item in meta:
        segment_id, indices, direction = item["segment_id"], item["indices"], item["direction"]
        duration = float(t[indices[-1]] - t[indices[0]])
        v0 = float(coefficients[2 * segment_id + 1])
        vend = v0 - mu * gravity * direction * duration
        velocities.append({"start": int(indices[0]), "end": int(indices[-1]), "v0": v0, "vend": vend})
    left_ratios: list[float] = []
    right_ratios: list[float] = []
    for index, (before, after) in enumerate(zip(velocities[:-1], velocities[1:])):
        if abs(before["vend"]) <= 1e-6:
            continue
        ratio = abs(after["v0"] / before["vend"])
        event_index = turns[min(index, len(turns) - 1)] if turns else before["end"]
        if abs(x[event_index] - float(constants["left_wall_x"])) <= abs(x[event_index] - float(constants["right_wall_x"])):
            left_ratios.append(ratio)
        else:
            right_ratios.append(ratio)
    e_left = float(np.median(left_ratios)) if left_ratios else None
    e_right = float(np.median(right_ratios)) if right_ratios else None
    diagnostics = _fit_diagnostics(target, design @ coefficients, len(coefficients))
    diagnostics.update({"impact_indices": turns, "left_impact_e": left_ratios, "right_impact_e": right_ratios, "segments": velocities})
    return _result(
        "v3_B",
        {"kinetic_friction_mu_k": mu, "left_restitution_e_L": e_left, "right_restitution_e_R": e_right},
        diagnostics,
        observed={"kinetic_friction_mu_k": True, "left_restitution_e_L": e_left is not None, "right_restitution_e_R": e_right is not None},
        method="shared_segment_deceleration_and_wall_specific_ratios",
    )


def _fit_v3c(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v3_C"]
    transition = float(constants["floor_transition_x"])
    wall = float(constants["wall_contact_x"])
    alpha = math.radians(float(constants["ramp_angle_deg"]))
    ramp = np.flatnonzero(x < transition - 0.04)
    floor = np.flatnonzero((x > transition + 0.04) & (x < wall - 0.08))
    if len(ramp) < 6 or len(floor) < 6:
        return _failure("v3_C", "ramp_and_floor_not_both_observed", ["gravity_g", "kinetic_friction_mu", "restitution_e"])
    s = (x - x[0]) * math.cos(alpha) - (z - z[0]) * math.sin(alpha)
    coef_ramp, diag_ramp = _fit_quadratic(t[ramp], s[ramp])
    # Keep the positive pre-wall floor branch only.
    floor_velocity = np.gradient(_smooth(x[floor]), t[floor])
    positive_floor = floor[floor_velocity > 0.03]
    if len(positive_floor) >= 6:
        floor = positive_floor
    coef_floor, diag_floor = _fit_quadratic(t[floor], x[floor])
    a_ramp = float(2.0 * coef_ramp[2])
    a_floor = float(-2.0 * coef_floor[2])
    gravity = (a_ramp + a_floor * math.cos(alpha)) / math.sin(alpha)
    friction = a_floor / gravity if abs(gravity) > 1e-9 else None
    event = _local_impact_velocity_ratio(t, x, wall)
    restitution = None
    if event is not None and event["v_pre"] > 0 and event["v_post"] < 0:
        restitution = float(event["velocity_ratio"])
    diagnostics = {
        "ramp": diag_ramp,
        "floor": diag_floor,
        "ramp_acceleration_m_s2": a_ramp,
        "floor_deceleration_m_s2": a_floor,
        "wall_event": event,
    }
    return _result(
        "v3_C",
        {"gravity_g": gravity, "kinetic_friction_mu": friction, "restitution_e": restitution},
        diagnostics,
        observed={"gravity_g": True, "kinetic_friction_mu": friction is not None, "restitution_e": restitution is not None},
        method="ramp_floor_acceleration_decomposition_and_wall_ratio",
    )


def _fit_v3d(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    constants = EXPERIMENT_CONSTANTS["v3_D"]
    # V3D uses negative Blender Y rotation for a ball at positive world X.
    theta = _pendulum_angle(x, z, constants["pivot"], sign=-1.0)
    if len(theta) < 20:
        return _failure("v3_D", "fewer_than_20_points", ["gravity_g", "linear_damping_beta", "magnetic_kappa"])
    try:
        coefficients, diagnostics = _fit_pendulum_integral(
            t,
            theta,
            length=float(constants["length"]),
            magnetic_center=math.radians(float(constants["magnet_center_deg"])),
            magnetic_width=math.radians(float(constants["magnet_width_deg"])),
        )
    except ValueError as error:
        return _failure("v3_D", str(error), ["gravity_g", "linear_damping_beta", "magnetic_kappa"])
    magnet_center = math.radians(float(constants["magnet_center_deg"]))
    magnet_width = math.radians(float(constants["magnet_width_deg"]))
    inside = np.abs(theta - magnet_center) <= magnet_width
    diagnostics["magnet_zone_transition_count"] = int(np.sum(inside[1:] != inside[:-1]))
    return _result(
        "v3_D",
        {"gravity_g": coefficients[0], "linear_damping_beta": coefficients[1], "magnetic_kappa": coefficients[2]},
        diagnostics,
        method="three_basis_double_integral_robust_regression",
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
    t, x, z = _finite_trajectory(trajectory)
    if len(t) < 4:
        return _failure(experiment_id, "fewer_than_4_metric_points", [])
    return FITTERS[experiment_id](t, x, z)


def score_parameter_fit(
    fit: Mapping[str, Any],
    experiment_spec: Mapping[str, Any],
    target_parameters: Mapping[str, float],
) -> dict[str, Any]:
    """Score a completed fit against its registry tuple.

    The primary error is the frozen range-normalized absolute error (NAE).
    Missing estimates receive scoring NAE=1 and score=0, while their raw NAE
    remains null.  Raw estimates are never clipped to the valid range.
    """

    estimates = fit.get("parameter_estimates", {})
    observed = fit.get("parameter_observed", {})
    metrics: dict[str, dict[str, Any]] = {}
    scoring_naes: list[float] = []
    raw_naes: list[float] = []
    for parameter in experiment_spec.get("hidden_parameters", []):
        name = str(parameter["name"])
        lower, upper = (float(value) for value in parameter["valid_range"])
        target = float(target_parameters[name])
        estimate_value = estimates.get(name)
        is_observed = bool(observed.get(name, estimate_value is not None)) and estimate_value is not None
        if is_observed and math.isfinite(float(estimate_value)):
            estimate = float(estimate_value)
            absolute_error = abs(estimate - target)
            relative_error = absolute_error / max(abs(target), 1e-12)
            nae = absolute_error / (upper - lower)
            scoring_nae = nae
            score = 100.0 * max(0.0, 1.0 - nae)
            raw_naes.append(nae)
        else:
            estimate = None
            absolute_error = relative_error = nae = None
            scoring_nae = 1.0
            score = 0.0
        scoring_naes.append(scoring_nae)
        metrics[name] = {
            "unit": parameter.get("unit"),
            "valid_range": [lower, upper],
            "gt": target,
            "estimate_raw": estimate,
            "observed": is_observed,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
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
        "fit_complete": bool(metrics) and all(item["observed"] for item in metrics.values()),
        "missing_parameter_count": int(sum(not item["observed"] for item in metrics.values())),
        "metric_definition": "NAE=abs(estimate-gt)/(valid_max-valid_min); score=100*max(0,1-NAE)",
    }


def lookup_target_tuple(experiment_spec: Mapping[str, Any], tuple_id: str) -> dict[str, float]:
    for anchor in experiment_spec.get("anchor_tuples", []):
        if str(anchor.get("id")) == tuple_id:
            return {
                str(parameter["name"]): float(anchor[str(parameter["name"])])
                for parameter in experiment_spec.get("hidden_parameters", [])
            }
    raise KeyError(f"unknown tuple id {tuple_id!r} for experiment {experiment_spec.get('id')!r}")
