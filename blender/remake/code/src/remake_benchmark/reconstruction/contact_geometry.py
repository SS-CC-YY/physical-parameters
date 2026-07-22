"""Target-free contact and apparatus geometry checks for the G0 gate.

The checks consume only the frozen metric trajectory and experiment apparatus
geometry.  They never read the parameter tuple.  Small contact errors are left
for trajectory fitting; only sustained, large violations are a G0 failure.
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


def _trajectory_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values: list[tuple[int, float, float, float]] = []
    for ordinal, row in enumerate(rows):
        # Contact geometry is a metric-space hard gate.  A detector can expose
        # a useful display coordinate while the known-sphere lift explicitly
        # rejects that frame.  Never turn such a diagnostic point into a G0
        # accusation: require the independent fit/geometry eligibility flag
        # and the corresponding fit-space coordinates.
        if "physics_fit_used" in row:
            # The evaluation contract is authoritative when present.  An
            # extraction-era fit_eligible=True must not revive a point the
            # independent evaluator explicitly rejected.
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
        frame = int(row.get("source_frame_index", row.get("frame_index", ordinal)))
        if physics_fit_used:
            # Evaluation fit CSVs intentionally normalize the trusted metric
            # coordinates back to x_m/y_m/z_m.
            x = _finite(row.get("x_m"))
            z = _finite(row.get("z_m"))
        else:
            # Extraction CSVs retain display and fit coordinates side by side.
            x = _finite(row.get("fit_x_m"))
            z = _finite(row.get("fit_z_m"))
        time = _finite(row.get("time_s"))
        if x is not None and z is not None and time is not None:
            values.append((frame, time, x, z))
    if not values:
        empty = np.empty(0, dtype=float)
        return empty.astype(int), empty, empty, empty
    array = np.asarray(values, dtype=float)
    order = np.argsort(array[:, 1], kind="stable")
    array = array[order]
    return array[:, 0].astype(int), array[:, 1], array[:, 2], array[:, 3]


def _runs(
    mask: np.ndarray,
    frames: np.ndarray,
    time: np.ndarray,
) -> list[np.ndarray]:
    """Return genuinely consecutive violation runs.

    The input arrays contain only trusted metric observations.  Consecutive
    array positions are not necessarily consecutive decoded frames, so a
    detector gap must split the run instead of manufacturing persistent G0
    evidence.
    """

    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    positive_dt = np.diff(time)
    positive_dt = positive_dt[positive_dt > 1e-9]
    nominal_dt = float(np.percentile(positive_dt, 25)) if len(positive_dt) else None
    adjacent_positions = np.diff(indices) == 1
    adjacent_frames = np.diff(frames[indices]) == 1
    if nominal_dt is None:
        adjacent_times = np.ones(len(indices) - 1, dtype=bool)
    else:
        adjacent_times = np.diff(time[indices]) <= 1.75 * nominal_dt
    linked = adjacent_positions & adjacent_frames & adjacent_times
    split = np.flatnonzero(~linked) + 1
    return [part for part in np.split(indices, split) if len(part)]


def _surface_check(
    *,
    name: str,
    values: np.ndarray,
    frames: np.ndarray,
    time: np.ndarray,
    limit: float,
    direction: str,
    minimum_consecutive: int,
) -> dict[str, Any]:
    if direction == "below":
        mask = values < limit
        signed_excess = limit - values
    elif direction == "above":
        mask = values > limit
        signed_excess = values - limit
    else:
        raise ValueError(direction)
    runs = _runs(mask, frames, time)
    longest = max((len(run) for run in runs), default=0)
    offending = [
        int(frames[index])
        for run in runs
        if len(run) >= minimum_consecutive
        for index in run
    ]
    return {
        "name": name,
        "status": "fail" if offending else "pass",
        "limit": float(limit),
        "direction": direction,
        "minimum_consecutive_frames": int(minimum_consecutive),
        "longest_violation_run": int(longest),
        "maximum_excess_m": float(np.max(signed_excess[mask])) if np.any(mask) else 0.0,
        "offending_frames": offending,
    }


def evaluate_contact_geometry(
    experiment_id: str,
    rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return pass/fail/indeterminate evidence for severe geometry violations."""

    frames, time, x, z = _trajectory_arrays(rows)
    global_thresholds = profile.get("global_thresholds", {})
    experiment = profile.get("experiments", {}).get(experiment_id)
    minimum_points = int(global_thresholds.get("minimum_direct_points", 12))
    severe = float(global_thresholds.get("severe_penetration_m", 0.24))
    minimum_consecutive = 3
    base = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "target_parameters_used": False,
        "direct_metric_point_count": int(len(time)),
        "thresholds": {
            "minimum_direct_points": minimum_points,
            "severe_penetration_m": severe,
            "minimum_consecutive_frames": minimum_consecutive,
        },
        "checks": [],
        "failure_codes": [],
        "indeterminate_codes": [],
        "offending_frames": [],
    }
    if experiment is None:
        base.update(
            status="indeterminate",
            indeterminate_codes=["missing_motion_type_profile"],
        )
        return base
    if len(time) < minimum_points:
        base.update(
            status="indeterminate",
            indeterminate_codes=["insufficient_direct_metric_points_for_contact_geometry"],
        )
        return base

    checks: list[dict[str, Any]] = []
    if "contact_z_m" in experiment:
        contact = float(experiment["contact_z_m"])
        checks.append(
            _surface_check(
                name="ground_penetration",
                values=z,
                frames=frames,
                time=time,
                limit=contact - severe,
                direction="below",
                minimum_consecutive=minimum_consecutive,
            )
        )
    if "wall_x_m" in experiment:
        wall = float(experiment["wall_x_m"])
        checks.append(
            _surface_check(
                name="right_wall_penetration",
                values=x,
                frames=frames,
                time=time,
                limit=wall + severe,
                direction="above",
                minimum_consecutive=minimum_consecutive,
            )
        )
    if "right_wall_x_m" in experiment:
        wall = float(experiment["right_wall_x_m"])
        checks.append(
            _surface_check(
                name="right_wall_penetration",
                values=x,
                frames=frames,
                time=time,
                limit=wall + severe,
                direction="above",
                minimum_consecutive=minimum_consecutive,
            )
        )
    if "left_wall_x_m" in experiment:
        wall = float(experiment["left_wall_x_m"])
        checks.append(
            _surface_check(
                name="left_wall_penetration",
                values=x,
                frames=frames,
                time=time,
                limit=wall - severe,
                direction="below",
                minimum_consecutive=minimum_consecutive,
            )
        )
    if "pivot_xz_m" in experiment and "length_m" in experiment:
        pivot_x, pivot_z = (float(value) for value in experiment["pivot_xz_m"])
        length = float(experiment["length_m"])
        relative_error = np.abs(np.hypot(x - pivot_x, z - pivot_z) - length) / max(length, 1e-9)
        severe_mask = relative_error > 0.30
        runs = _runs(severe_mask, frames, time)
        offending = [
            int(frames[index])
            for run in runs
            if len(run) >= minimum_consecutive
            for index in run
        ]
        checks.append(
            {
                "name": "pendulum_length_constraint",
                "status": "fail" if offending else "pass",
                "expected_length_m": length,
                "severe_relative_error": 0.30,
                "relative_error_p95": float(np.percentile(relative_error, 95)),
                "maximum_relative_error": float(np.max(relative_error)),
                "minimum_consecutive_frames": minimum_consecutive,
                "longest_violation_run": max((len(run) for run in runs), default=0),
                "offending_frames": offending,
            }
        )

    failures = [check for check in checks if check["status"] == "fail"]
    offending_frames = sorted(
        {
            int(frame)
            for check in failures
            for frame in check.get("offending_frames", [])
        }
    )
    base.update(
        status="fail" if failures else "pass",
        checks=checks,
        failure_codes=[f"severe_{check['name']}" for check in failures],
        offending_frames=offending_frames,
    )
    return base


__all__ = ["evaluate_contact_geometry"]
