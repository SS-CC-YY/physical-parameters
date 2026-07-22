"""Target-independent usability gate for dynamically reconstructed 3-D tracks.

The gate answers one narrow question: is a SpaTrackerV2 trajectory reliable
enough to be consumed by the same inverse-physics code as a calibrated 2-D
track?  It never compares against the requested parameter and therefore never
assigns a video-model failure grade.  A rejected track is measurement ``X``.

The frozen Blender experiments move either along one world axis or in the
world X-Z plane.  After camera compensation and metric alignment, motion away
from that expected manifold is treated as reconstruction error unless the
benchmark definition explicitly allows it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0.0"
DYNAMIC_ROUTE = "spatialtrackerv2_dynamic"

# This mapping follows the frozen Blender apparatus and the coordinate columns
# consumed by physics_parameters.py.  It is deliberately independent of the
# requested parameter values.
EXPERIMENT_MOTION_MANIFOLDS: dict[str, dict[str, Any]] = {
    "v1_A": {"kind": "axis", "axes": ("z",)},
    "v1_B": {"kind": "axis", "axes": ("x",)},
    "v1_C": {"kind": "axis", "axes": ("x",)},
    "v1_D": {"kind": "plane", "axes": ("x", "z")},
    "v2_A": {"kind": "plane", "axes": ("x", "z")},
    "v2_B": {"kind": "axis", "axes": ("x",)},
    "v2_C": {"kind": "axis", "axes": ("x",)},
    "v2_D": {"kind": "plane", "axes": ("x", "z")},
    "v2_E": {"kind": "axis", "axes": ("z",)},
    "v3_A": {"kind": "plane", "axes": ("x", "z")},
    "v3_B": {"kind": "axis", "axes": ("x",)},
    "v3_C": {"kind": "plane", "axes": ("x", "z")},
    "v3_D": {"kind": "plane", "axes": ("x", "z")},
}


# These are intentionally permissive evidence thresholds.  Their purpose is to
# exclude plainly unstable depth reconstruction, not to demand a perfect
# trajectory before the physical model is allowed to explain it.
DEFAULT_THRESHOLDS: dict[str, float | bool] = {
    "minimum_direct_points": 12,
    "minimum_valid_fraction": 0.50,
    "minimum_time_span_s": 0.50,
    "maximum_missing_gap_s": 0.75,
    "minimum_main_motion_range_m": 0.05,
    "maximum_axis_off_manifold_range_ratio": 0.35,
    "maximum_plane_off_manifold_range_ratio": 0.30,
    "minimum_axis_expected_variance_fraction": 0.75,
    "minimum_plane_expected_variance_fraction": 0.80,
    "maximum_jump_q95_ratio": 0.25,
    "maximum_jump_ratio": 0.75,
    "maximum_motion_law_median_nrmse": 0.35,
    "require_motion_law_residual_evidence": True,
    "require_native_quality_pass": True,
}

WORLD_AXIS_COORDINATE_FRAMES = {
    "blender_world_m",
    "spatialtrackerv2_frame0_metric_aligned_m",
}
WORLD_AXIS_COORDINATE_POLICIES = {
    "frame0 metric sphere sim(3), then per-frame static-background sim(3) camera compensation",
    "per-frame sim(3) to calibrated blender static anchors",
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed", "ok"}


def _quantile_span(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    low, high = np.quantile(values, (0.05, 0.95))
    return float(max(0.0, high - low))


def _dynamic_native(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    pipeline = metadata.get("pipeline")
    if isinstance(pipeline, Mapping):
        native = pipeline.get("dynamic_reconstruction")
        if isinstance(native, Mapping):
            return native
    native = metadata.get("dynamic_reconstruction")
    if isinstance(native, Mapping):
        return native
    return metadata


def _direct_xyz(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, int, bool]:
    points: list[tuple[float, float, float, float]] = []
    used_rows_world_axis_aligned: list[bool] = []
    for row in rows:
        # Only direct measurements that the reconstruction stage explicitly
        # admitted to physics fitting are evidence for this gate.  Missing is
        # not equivalent to True.
        if not _truth(row.get("fit_eligible")):
            continue
        coordinate = str(row.get("coordinate_frame_3d") or "").strip().lower()
        if _truth(row.get("interpolated")):
            continue
        values = []
        for axis in ("x", "y", "z"):
            value = _number(row.get(f"fit_{axis}_m"))
            if value is None:
                value = _number(row.get(f"{axis}_m"))
            values.append(value)
        time_s = _number(row.get("time_s"))
        if time_s is None or any(value is None for value in values):
            continue
        points.append((time_s, float(values[0]), float(values[1]), float(values[2])))
        used_rows_world_axis_aligned.append(coordinate in WORLD_AXIS_COORDINATE_FRAMES)
    all_used_rows_world_axis_aligned = bool(
        used_rows_world_axis_aligned and all(used_rows_world_axis_aligned)
    )
    if not points:
        return np.empty((0, 4), dtype=np.float64), len(rows), all_used_rows_world_axis_aligned
    array = np.asarray(points, dtype=np.float64)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    unique = np.r_[True, np.diff(array[:, 0]) > 1e-9]
    return array[unique], len(rows), all_used_rows_world_axis_aligned


def _effective_thresholds(overrides: Mapping[str, Any] | None) -> dict[str, float | bool]:
    output = dict(DEFAULT_THRESHOLDS)
    if not overrides:
        return output
    for key, default in DEFAULT_THRESHOLDS.items():
        if key not in overrides:
            continue
        if isinstance(default, bool):
            output[key] = _truth(overrides[key])
        else:
            value = _number(overrides[key])
            if value is not None:
                output[key] = value
    return output


def assess_dynamic_3d_trajectory(
    experiment_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    reconstruction_metadata: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``include`` or measurement ``X`` for a dynamic 3-D trajectory."""

    limits = _effective_thresholds(thresholds)
    manifold = EXPERIMENT_MOTION_MANIFOLDS.get(str(experiment_id))
    reasons: list[str] = []
    native = _dynamic_native(reconstruction_metadata)
    native_status = str(native.get("status") or "").strip().lower()
    native_quality = native.get("quality_pass")
    if bool(limits["require_native_quality_pass"]):
        if native_status not in {"success", "succeeded", "ok", "passed"}:
            reasons.append("X_DYNAMIC_RECONSTRUCTION_NOT_SUCCEEDED")
        if native_quality is not True:
            reasons.append("X_DYNAMIC_RECONSTRUCTION_QUALITY_NOT_PASSED")
    if manifold is None:
        reasons.append("X_EXPECTED_MOTION_MANIFOLD_UNDEFINED")

    samples, frame_count, all_used_rows_world_axis_aligned = _direct_xyz(rows)
    native_rigidity = native.get("rigidity_evidence")
    nested_coordinate_policy = (
        native_rigidity.get("coordinate_policy")
        if isinstance(native_rigidity, Mapping)
        else None
    )
    native_coordinate_policy = str(
        native.get("coordinate_policy") or nested_coordinate_policy or ""
    ).lower()
    native_world_axis_alignment_proven = bool(
        native.get("world_axis_aligned") is True
        or native_coordinate_policy in WORLD_AXIS_COORDINATE_POLICIES
    )
    world_axis_alignment_seen = bool(
        all_used_rows_world_axis_aligned or native_world_axis_alignment_proven
    )
    direct_count = int(len(samples))
    valid_fraction = direct_count / max(frame_count, 1)
    if direct_count < int(float(limits["minimum_direct_points"])):
        reasons.append("X_TOO_FEW_DIRECT_POINTS")
    if valid_fraction < float(limits["minimum_valid_fraction"]):
        reasons.append("X_LOW_VALID_COVERAGE")
    if not world_axis_alignment_seen:
        reasons.append("X_WORLD_AXIS_ALIGNMENT_UNPROVEN")

    metrics: dict[str, Any] = {
        "frame_count": frame_count,
        "direct_point_count": direct_count,
        "valid_fraction": valid_fraction,
        "native_status": native_status or None,
        "native_quality_pass": native_quality if isinstance(native_quality, bool) else None,
        "all_used_rows_world_axis_aligned": all_used_rows_world_axis_aligned,
        "native_world_axis_alignment_proven": native_world_axis_alignment_proven,
        "world_axis_alignment_proven": world_axis_alignment_seen,
    }
    if direct_count >= 2:
        times = samples[:, 0]
        xyz = samples[:, 1:]
        time_span = float(times[-1] - times[0])
        deltas_t = np.diff(times)
        positive_dt = deltas_t[deltas_t > 1e-9]
        median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.0
        maximum_missing_gap = (
            float(max(0.0, float(np.max(positive_dt)) - median_dt)) if len(positive_dt) else 0.0
        )
        spans = {
            axis: _quantile_span(xyz[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        }
        variances = {
            axis: float(np.var(xyz[:, index]))
            for index, axis in enumerate(("x", "y", "z"))
        }
        metrics.update(
            {
                "time_span_s": time_span,
                "median_sample_interval_s": median_dt,
                "maximum_missing_gap_s": maximum_missing_gap,
                "robust_axis_ranges_m": spans,
                "axis_variances_m2": variances,
            }
        )
        if time_span < float(limits["minimum_time_span_s"]):
            reasons.append("X_SHORT_TIME_SPAN")
        if maximum_missing_gap > float(limits["maximum_missing_gap_s"]):
            reasons.append("X_LONG_MISSING_GAP")

        if manifold is not None:
            expected_axes = tuple(str(axis) for axis in manifold["axes"])
            off_axes = tuple(axis for axis in ("x", "y", "z") if axis not in expected_axes)
            main_range = math.sqrt(sum(spans[axis] ** 2 for axis in expected_axes))
            off_range = math.sqrt(sum(spans[axis] ** 2 for axis in off_axes))
            off_ratio = off_range / max(main_range, 1e-12)
            total_variance = sum(variances.values())
            expected_variance_fraction = (
                sum(variances[axis] for axis in expected_axes) / total_variance
                if total_variance > 1e-12
                else 0.0
            )
            metrics.update(
                {
                    "expected_manifold_kind": manifold["kind"],
                    "expected_motion_axes": list(expected_axes),
                    "main_motion_range_m": main_range,
                    "off_manifold_range_m": off_range,
                    "off_manifold_range_ratio": off_ratio,
                    "expected_variance_fraction": expected_variance_fraction,
                }
            )
            if main_range < float(limits["minimum_main_motion_range_m"]):
                reasons.append("X_INSUFFICIENT_MOTION_RANGE")
            ratio_key = (
                "maximum_axis_off_manifold_range_ratio"
                if manifold["kind"] == "axis"
                else "maximum_plane_off_manifold_range_ratio"
            )
            energy_key = (
                "minimum_axis_expected_variance_fraction"
                if manifold["kind"] == "axis"
                else "minimum_plane_expected_variance_fraction"
            )
            if off_ratio > float(limits[ratio_key]):
                reasons.append("X_EXCESS_OFF_MANIFOLD_DRIFT")
            if expected_variance_fraction < float(limits[energy_key]):
                reasons.append("X_EXPECTED_MOTION_NOT_DOMINANT")

            if len(positive_dt) and main_range > 1e-12:
                step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
                normalized = (step / main_range) * (median_dt / np.maximum(deltas_t, 1e-12))
                jump_q95 = float(np.quantile(normalized, 0.95))
                jump_max = float(np.max(normalized))
                metrics.update({"jump_q95_ratio": jump_q95, "jump_max_ratio": jump_max})
                if (
                    jump_q95 > float(limits["maximum_jump_q95_ratio"])
                    or jump_max > float(limits["maximum_jump_ratio"])
                ):
                    reasons.append("X_DISCONTINUOUS_3D_TRACK")
    else:
        metrics.update(
            {
                "time_span_s": 0.0,
                "maximum_missing_gap_s": None,
                "main_motion_range_m": None,
                "off_manifold_range_ratio": None,
                "expected_variance_fraction": None,
            }
        )
        reasons.append("X_SHORT_TIME_SPAN")

    reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "include" if not reasons else "X",
        "measurement_route": "qualified_dynamic_3d" if not reasons else "dynamic_3d_measurement_x",
        "experiment_id": str(experiment_id),
        "expected_geometry": None if manifold is None else dict(manifold),
        "reason_codes": reasons,
        "metrics": metrics,
        "thresholds": limits,
        "target_parameters_used": False,
        "scope_note": "This is a target-independent evidence-usability gate, not a video-model failure grade.",
    }


def _diagnostic_nrmse_values(value: Any) -> list[float]:
    output: list[float] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == "fit_nrmse":
                number = _number(item)
                if number is not None:
                    output.append(number)
            else:
                output.extend(_diagnostic_nrmse_values(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            output.extend(_diagnostic_nrmse_values(item))
    return output


def finalize_dynamic_3d_inclusion(
    geometry_gate: Mapping[str, Any],
    fit: Mapping[str, Any] | None,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add the target-free equation-fit check to a geometry-gate result."""

    result = dict(geometry_gate)
    result["metrics"] = dict(geometry_gate.get("metrics", {}))
    reasons = [str(value) for value in geometry_gate.get("reason_codes", [])]
    limits = _effective_thresholds(thresholds or geometry_gate.get("thresholds"))
    if geometry_gate.get("decision") != "include":
        result.update(
            {
                "decision": "X",
                "measurement_route": "dynamic_3d_measurement_x",
                "fit_state": "not_attempted_geometry_x",
                "reason_codes": list(dict.fromkeys(reasons)),
                "thresholds": limits,
            }
        )
        return result

    if not isinstance(fit, Mapping):
        reasons.append("X_TARGET_FREE_FIT_RESULT_MISSING")
        fit_status = None
        observed: Mapping[str, Any] = {}
        target_free = False
        nrmse_values: list[float] = []
    else:
        fit_status = str(fit.get("status") or "").strip().lower()
        observed_value = fit.get("parameter_observed")
        observed = observed_value if isinstance(observed_value, Mapping) else {}
        target_free = fit.get("target_not_used_for_fit") is True
        nrmse_values = _diagnostic_nrmse_values(fit.get("diagnostics", {}))
        if fit_status not in {"ok", "complete", "completed", "success", "succeeded"}:
            reasons.append("X_TARGET_FREE_FIT_INCOMPLETE")
        if not observed or not all(_truth(value) for value in observed.values()):
            reasons.append("X_TARGET_FREE_PARAMETERS_NOT_IDENTIFIABLE")
        if not target_free:
            reasons.append("X_TARGET_FREE_FIT_CONTRACT_NOT_PROVEN")

    median_nrmse = float(np.median(nrmse_values)) if nrmse_values else None
    max_nrmse = float(np.max(nrmse_values)) if nrmse_values else None
    result["metrics"].update(
        {
            "motion_law_fit_nrmse_count": len(nrmse_values),
            "motion_law_median_nrmse": median_nrmse,
            "motion_law_max_nrmse": max_nrmse,
        }
    )
    if bool(limits["require_motion_law_residual_evidence"]) and not nrmse_values:
        reasons.append("X_MOTION_LAW_RESIDUAL_EVIDENCE_MISSING")
    if (
        median_nrmse is not None
        and median_nrmse > float(limits["maximum_motion_law_median_nrmse"])
    ):
        reasons.append("X_MOTION_LAW_FIT_RESIDUAL_HIGH")

    reasons = list(dict.fromkeys(reasons))
    result.update(
        {
            "decision": "include" if not reasons else "X",
            "measurement_route": "qualified_dynamic_3d" if not reasons else "dynamic_3d_measurement_x",
            "fit_state": "completed" if not reasons else "unusable",
            "fit_status": fit_status,
            "fit_target_not_used": target_free,
            "reason_codes": reasons,
            "thresholds": limits,
            "target_parameters_used": False,
        }
    )
    return result


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DYNAMIC_ROUTE",
    "EXPERIMENT_MOTION_MANIFOLDS",
    "WORLD_AXIS_COORDINATE_FRAMES",
    "assess_dynamic_3d_trajectory",
    "finalize_dynamic_3d_inclusion",
]
