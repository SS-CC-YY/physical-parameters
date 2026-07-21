"""Conservative, model-agnostic validity gate for generated experiment videos.

The gate deliberately consumes *measurements*, not video frames.  It can
therefore be run after any detector/tracker and before any physics fitter.
Only persistent, strong evidence is called a generation failure.  Missing or
ambiguous evidence is ``indeterminate`` rather than silently accepted.

The public result has three states:

``pass``
    No hard failure was found and the evidence needed by the selected camera
    route is sufficient.  Only this state is eligible for physics fitting.
``fail``
    There is strong evidence of a scene cut, persistent ball deformation, or
    persistent 3-D non-rigidity.
``indeterminate``
    Tracking/evidence is insufficient.  In particular, a moving camera needs
    3-D object *and* background rigidity evidence; 2-D scale changes alone are
    never used to accuse a moving-camera video of deformation.

No OpenCV dependency is required.  Inputs and outputs use JSON-safe Python
types so the result can be written directly into per-video metadata.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "1.0.0"

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "tracking": {
        "min_tracked_fraction": 0.70,
        "min_reliable_fraction": 0.60,
        "min_reliable_frames": 8,
        "min_confidence": 0.30,
        "ambiguous_candidate_count": 2,
        "persistent_frames": 3,
        "max_consecutive_unresolved_frames": 12,
    },
    "object_shape_2d": {
        "max_axis_ratio": 1.55,
        "severe_axis_ratio": 1.85,
        "max_circularity_with_elongation": 0.72,
        "severe_min_circularity": 0.28,
        "min_edge_support_fraction": 0.55,
        "max_edge_radial_residual_ratio": 0.18,
        "persistent_frames": 3,
        "min_evaluable_frames": 8,
        "min_evaluable_fraction": 0.20,
        "max_consecutive_unresolved_frames": 12,
    },
    "object_scale_2d": {
        "max_relative_expected_radius_error": 0.50,
        "persistent_frames": 3,
    },
    "object_rigidity_3d": {
        "max_normalized_rmse": 0.22,
        "min_inlier_fraction": 0.60,
        "persistent_frames": 3,
        "min_evaluable_frames": 3,
    },
    "scene_rigidity_3d": {
        "max_normalized_rmse": 0.20,
        "min_inlier_fraction": 0.55,
        "min_failure_spatial_coverage": 0.45,
        "persistent_frames": 3,
        "min_evaluable_frames": 3,
    },
}

_STATIC_CAMERA_CATEGORIES = {
    "fixed",
    "static",
    "no_significant_camera_change",
    "side_2d_motion_within_tolerance",
    "unchanged",
}
_CHANGED_CAMERA_CATEGORIES = {
    "changed",
    "camera_changed",
    "moving",
    "dynamic",
    "borderline_below_threshold",
    "review",
}


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _frame_index(row: Mapping[str, Any], fallback: int) -> int:
    value = row.get("frame_index", row.get("frame", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _first_number(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = _finite_float(row.get(name))
        if value is not None:
            return value
    return None


def _axis_ratio(row: Mapping[str, Any]) -> float | None:
    direct = _first_number(
        row,
        (
            "ellipse_axis_ratio",
            "silhouette_axis_ratio",
            "bbox_axis_ratio",
            "axis_ratio",
            "aspect_ratio",
        ),
    )
    if direct is None:
        width = _first_number(row, ("bbox_width_px", "width_px", "ellipse_width_px"))
        height = _first_number(row, ("bbox_height_px", "height_px", "ellipse_height_px"))
        if width is not None and height is not None and width > 0.0 and height > 0.0:
            direct = width / height
    if direct is None or direct <= 0.0:
        return None
    return float(max(direct, 1.0 / direct))


def _runs(frames: Sequence[int], minimum: int) -> list[list[int]]:
    """Return strictly consecutive runs meeting ``minimum`` length."""

    ordered = sorted(set(int(item) for item in frames))
    if not ordered:
        return []
    result: list[list[int]] = []
    current = [ordered[0]]
    for frame in ordered[1:]:
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            if len(current) >= minimum:
                result.append(current)
            current = [frame]
    if len(current) >= minimum:
        result.append(current)
    return result


def _flatten(runs: Sequence[Sequence[int]]) -> list[int]:
    return sorted({int(frame) for run in runs for frame in run})


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _camera_category(evidence: Mapping[str, Any] | None) -> str | None:
    if not evidence:
        return None
    for name in (
        "effective_camera_motion_category",
        "final_category",
        "camera_motion_category",
        "category",
        "decision",
    ):
        value = evidence.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    changed = evidence.get("camera_changed")
    if changed is True:
        return "camera_changed"
    if changed is False:
        return "no_significant_camera_change"
    return None


def _scene_cut_frames(evidence: Mapping[str, Any] | None) -> list[int]:
    if not evidence:
        return []
    pairs = evidence.get("cut_pairs")
    frames: list[int] = []
    if isinstance(pairs, Sequence) and not isinstance(pairs, (str, bytes)):
        for fallback, pair in enumerate(pairs):
            if isinstance(pair, Mapping):
                for name in ("frame_index", "to_frame", "frame_b", "end_frame"):
                    if name in pair:
                        frames.append(_frame_index(pair, fallback))
                        break
            elif isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)):
                numeric = [_finite_float(value) for value in pair]
                numeric = [value for value in numeric if value is not None]
                if numeric:
                    frames.append(int(max(numeric)))
            else:
                number = _finite_float(pair)
                if number is not None:
                    frames.append(int(number))
    count = _finite_float(evidence.get("cut_pair_count"))
    if count is not None and count > 0 and not frames:
        # The audit established a cut but did not expose the exact pair.
        frames.append(-1)
    return sorted(set(frames))


def _trajectory_rows(evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if evidence is None:
        return []
    if isinstance(evidence, Mapping):
        for name in ("trajectory_rows", "rows", "frames"):
            value = evidence.get(name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [item for item in value if isinstance(item, Mapping)]
        return []
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        return [item for item in evidence if isinstance(item, Mapping)]
    return []


def _evidence_block(
    evidence: Mapping[str, Any] | None,
    names: Sequence[str],
) -> Mapping[str, Any] | Sequence[Mapping[str, Any]] | None:
    if not evidence:
        return None
    for name in names:
        value = evidence.get(name)
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
            return value
    return None


def _rigidity_records(
    block: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None]:
    if block is None:
        return [], None
    if isinstance(block, Mapping):
        for name in ("frames", "rows", "per_frame"):
            value = block.get(name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [item for item in value if isinstance(item, Mapping)], block
        return [], block
    return [item for item in block if isinstance(item, Mapping)], None


def _rigidity_check(
    block: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    *,
    require_spatial_coverage_for_failure: bool,
) -> dict[str, Any]:
    records, summary = _rigidity_records(block)
    persistent = int(config["persistent_frames"])
    max_rmse = float(config["max_normalized_rmse"])
    min_inliers = float(config["min_inlier_fraction"])
    min_coverage = float(config.get("min_failure_spatial_coverage", 0.0))

    bad_frames: list[int] = []
    unresolved_frames: list[int] = []
    good_frames: list[int] = []
    metrics: list[dict[str, Any]] = []
    for fallback, row in enumerate(records):
        frame = _frame_index(row, fallback)
        rmse = _first_number(
            row,
            ("normalized_rmse", "rmse_over_radius", "rigid_rmse_normalized", "relative_rmse"),
        )
        inliers = _first_number(row, ("inlier_fraction", "rigid_inlier_fraction"))
        coverage = _first_number(
            row,
            ("spatial_coverage_fraction", "outlier_spatial_coverage", "affected_spatial_coverage"),
        )
        metrics.append(
            {
                "frame_index": frame,
                "normalized_rmse": rmse,
                "inlier_fraction": inliers,
                "spatial_coverage_fraction": coverage,
            }
        )
        if rmse is None or inliers is None:
            unresolved_frames.append(frame)
            continue
        geometrically_bad = rmse > max_rmse and inliers < min_inliers
        severe = rmse > 2.0 * max_rmse
        coverage_ok = (
            not require_spatial_coverage_for_failure
            or (coverage is not None and coverage >= min_coverage)
        )
        if coverage_ok and (geometrically_bad or severe):
            bad_frames.append(frame)
        elif geometrically_bad or severe:
            unresolved_frames.append(frame)
        else:
            good_frames.append(frame)

    bad_runs = _runs(bad_frames, persistent)
    explicit_failure = bool(summary and summary.get("persistent_failure") is True)
    explicit_status = str(summary.get("status", "")).lower() if summary else ""
    summary_failed_frames = []
    if summary:
        raw = summary.get("failed_frame_indices", summary.get("offending_frames", []))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            summary_failed_frames = [int(value) for value in raw if _finite_float(value) is not None]
    if explicit_status in {"fail", "failed", "deformed", "nonrigid"}:
        explicit_failure = explicit_failure or len(summary_failed_frames) >= persistent
    failed = bool(bad_runs or explicit_failure)

    explicit_pass = bool(summary and str(summary.get("status", "")).lower() == "pass")
    enough = len(good_frames) >= int(config["min_evaluable_frames"])
    passed = not failed and (explicit_pass or enough)
    status = "fail" if failed else "pass" if passed else "indeterminate"
    return {
        "status": status,
        "evaluable_frame_count": len(good_frames) + len(bad_frames),
        "good_frame_count": len(good_frames),
        "bad_frame_count": len(bad_frames),
        "unresolved_frame_count": len(unresolved_frames),
        "offending_frames": sorted(set(_flatten(bad_runs) + summary_failed_frames)),
        "unresolved_frames": sorted(set(unresolved_frames)),
        "per_frame_metrics": metrics,
    }


def evaluate_generation_validity(
    track_rows: Sequence[Mapping[str, Any]],
    *,
    camera_motion_evidence: Mapping[str, Any] | None = None,
    static_scene_rigidity_evidence: Mapping[str, Any] | None = None,
    trajectory_evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    dynamic_rigidity_evidence: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a generated experiment video is safe to fit.

    ``track_rows`` should contain one row per decoded frame.  The gate accepts
    the existing fixed-camera tracker fields (``found``, ``track_confidence``,
    ``measurement_radius_px``, ``circularity``) and optional ellipse/bounding
    box shape fields.  ``trajectory_evidence`` may add projected-radius rows.

    For a moving/borderline camera, pass ``dynamic_rigidity_evidence`` with
    ``object_frames`` and ``background_frames``.  Each row should provide
    ``frame_index``, ``normalized_rmse`` and ``inlier_fraction``.  Background
    failure additionally requires broad ``spatial_coverage_fraction`` so an
    unstable 4-D fit is not mislabeled as scene deformation.
    """

    config = copy.deepcopy(DEFAULT_THRESHOLDS)
    if thresholds:
        _deep_update(config, thresholds)

    rows = [row for row in track_rows if isinstance(row, Mapping)]
    rows = sorted(enumerate(rows), key=lambda item: _frame_index(item[1], item[0]))
    failures: list[str] = []
    warnings: list[str] = []
    blocking_indeterminate: list[str] = []
    offending: dict[str, list[int]] = {}

    camera_category = _camera_category(camera_motion_evidence)
    camera_static = camera_category in _STATIC_CAMERA_CATEGORIES
    camera_changed = camera_category in _CHANGED_CAMERA_CATEGORIES
    cut_frames = _scene_cut_frames(camera_motion_evidence)
    if cut_frames:
        _append_unique(failures, "scene_cut")
        offending["scene_cut"] = cut_frames

    tracking_config = config["tracking"]
    total = len(rows)
    found_rows: list[tuple[int, Mapping[str, Any]]] = []
    reliable_rows: list[tuple[int, Mapping[str, Any]]] = []
    missing_frames: list[int] = []
    unverified_frames: list[int] = []
    unreliable_frames: list[int] = []
    ambiguous_frames: list[int] = []
    for fallback, row in rows:
        frame = _frame_index(row, fallback)
        observation_status = str(
            row.get(
                "observation_status",
                "measured" if row.get("found") is True else "missing",
            )
        ).lower()
        identity_verified = row.get("identity_verified") is not False
        measurement_valid = row.get("measurement_valid") is not False
        is_measured = (
            row.get("found") is True
            and observation_status == "measured"
            and measurement_valid
        )
        if not is_measured:
            missing_frames.append(frame)
            continue
        found_rows.append((fallback, row))
        if not identity_verified:
            unverified_frames.append(frame)
            continue
        confidence = _finite_float(row.get("track_confidence"))
        boundary = row.get("touches_frame_boundary") is True
        if (confidence is None or confidence >= float(tracking_config["min_confidence"])) and not boundary:
            reliable_rows.append((fallback, row))
        else:
            unreliable_frames.append(frame)
        # ``candidate_count`` historically counted every colour blob in the
        # full image.  Indoor backgrounds can contain dozens of irrelevant
        # blobs, so only locally plausible identity candidates are meaningful
        # when the tracker supplies that diagnostic.
        explicit_ambiguity = row.get("identity_ambiguous")
        count = _finite_float(row.get("plausible_candidate_count", row.get("candidate_count")))
        margin = _finite_float(row.get("association_margin"))
        ambiguous = (
            explicit_ambiguity is True
            or (
                explicit_ambiguity is None
                and count is not None
                and count >= float(tracking_config["ambiguous_candidate_count"])
                # When the tracker reports an association margin, multiple
                # candidates are only ambiguous if the top two are close.  A
                # frozen first-frame identity with a clear winner must not be
                # blocked merely because an indoor scene contains orange props.
                and (margin is None or margin < 0.30)
            )
        )
        if ambiguous:
            ambiguous_frames.append(frame)

    tracked_fraction = len(found_rows) / total if total else 0.0
    reliable_fraction = len(reliable_rows) / total if total else 0.0
    required_reliable_frames = min(
        int(tracking_config["min_reliable_frames"]),
        max(1, math.ceil(total * float(tracking_config["min_reliable_fraction"]))) if total else 1,
    )
    tracking_unresolved = sorted(
        set(missing_frames + unverified_frames + unreliable_frames)
    )
    unresolved_tracking_runs = _runs(
        tracking_unresolved,
        int(tracking_config["max_consecutive_unresolved_frames"]) + 1,
    )
    tracking_ok = (
        total > 0
        and tracked_fraction >= float(tracking_config["min_tracked_fraction"])
        and reliable_fraction >= float(tracking_config["min_reliable_fraction"])
        and len(reliable_rows) >= required_reliable_frames
        and not unresolved_tracking_runs
    )
    trusted_first_frame = any(
        _frame_index(row, fallback) == 0
        for fallback, row in reliable_rows
    )
    if not tracking_ok:
        _append_unique(warnings, "insufficient_reliable_object_tracking")
        _append_unique(blocking_indeterminate, "insufficient_reliable_object_tracking")
        offending["object_identity_unresolved"] = tracking_unresolved
    if not trusted_first_frame:
        # Metric sphere-size calibration is defined relative to the generated
        # frame-zero observation.  Later good detections cannot reconstruct
        # that missing scale anchor, so stop before geometry instead of
        # surfacing an internal pipeline exception.
        _append_unique(warnings, "missing_trusted_first_frame_observation")
        _append_unique(blocking_indeterminate, "missing_trusted_first_frame_observation")
    ambiguous_runs = _runs(ambiguous_frames, int(tracking_config["persistent_frames"]))
    if ambiguous_runs:
        _append_unique(warnings, "persistent_object_identity_ambiguity")
        _append_unique(blocking_indeterminate, "persistent_object_identity_ambiguity")
        offending["object_identity_ambiguity"] = _flatten(ambiguous_runs)

    shape_config = config["object_shape_2d"]
    shape_bad: list[int] = []
    shape_unresolved: list[int] = []
    shape_metrics: list[dict[str, Any]] = []
    for fallback, row in reliable_rows:
        # A centre/radius recovered from a circle detector can be a useful
        # trajectory measurement, but it is not an observed silhouette and
        # must not be used to accuse the generated object of deformation.
        if row.get("shape_evidence_available") is False:
            shape_unresolved.append(_frame_index(row, fallback))
            continue
        frame = _frame_index(row, fallback)
        ratio = _axis_ratio(row)
        circularity = _finite_float(row.get("circularity"))
        edge_support = _finite_float(row.get("edge_support_fraction"))
        edge_radial_residual = _finite_float(row.get("edge_radial_residual_ratio"))
        evidence_source = str(row.get("shape_evidence_source", "segmentation_contour"))
        if evidence_source == "hough_edge_ring" and (
            edge_support is None
            or edge_support < float(shape_config["min_edge_support_fraction"])
            or edge_radial_residual is None
            or edge_radial_residual
            > float(shape_config["max_edge_radial_residual_ratio"])
        ):
            shape_unresolved.append(frame)
            continue
        shape_metrics.append(
            {
                "frame_index": frame,
                "evidence_source": evidence_source,
                "axis_ratio": ratio,
                "circularity": circularity,
                "edge_support_fraction": edge_support,
                "edge_radial_residual_ratio": edge_radial_residual,
            }
        )
        elongated_and_noncircular = (
            ratio is not None
            and ratio > float(shape_config["max_axis_ratio"])
            and circularity is not None
            and circularity < float(shape_config["max_circularity_with_elongation"])
        )
        severe_elongation = ratio is not None and ratio > float(shape_config["severe_axis_ratio"])
        severe_noncircularity = (
            circularity is not None
            and circularity < float(shape_config["severe_min_circularity"])
        )
        if (
            elongated_and_noncircular
            or severe_elongation
            or severe_noncircularity
        ):
            shape_bad.append(frame)
    shape_runs = _runs(shape_bad, int(shape_config["persistent_frames"]))
    shape_evaluable_fraction = len(shape_metrics) / max(len(reliable_rows), 1)
    unresolved_shape_runs = _runs(
        shape_unresolved,
        int(shape_config["max_consecutive_unresolved_frames"]) + 1,
    )
    shape_evidence_sufficient = bool(
        len(shape_metrics) >= int(shape_config["min_evaluable_frames"])
        and shape_evaluable_fraction >= float(shape_config["min_evaluable_fraction"])
        and not unresolved_shape_runs
    )
    if shape_runs:
        _append_unique(failures, "persistent_experiment_object_deformation_2d")
        offending["object_shape_2d"] = _flatten(shape_runs)
    elif shape_bad:
        _append_unique(warnings, "transient_object_shape_outlier")
        offending["transient_object_shape_2d"] = sorted(set(shape_bad))
    if not shape_runs and not shape_evidence_sufficient:
        _append_unique(warnings, "insufficient_object_shape_evidence")
        _append_unique(blocking_indeterminate, "insufficient_object_shape_evidence")
        offending["object_shape_evidence_unresolved"] = sorted(set(shape_unresolved))

    # Join optional projected-radius evidence by frame without mutating inputs.
    trajectory_by_frame = {
        _frame_index(row, fallback): row
        for fallback, row in enumerate(_trajectory_rows(trajectory_evidence))
    }
    scale_config = config["object_scale_2d"]
    scale_bad: list[int] = []
    scale_metrics: list[dict[str, Any]] = []
    for fallback, base_row in reliable_rows:
        frame = _frame_index(base_row, fallback)
        row = dict(base_row)
        row.update(trajectory_by_frame.get(frame, {}))
        observed = _first_number(row, ("measurement_radius_px", "observed_radius_px", "radius_px"))
        expected = _first_number(
            row,
            ("expected_radius_px", "projected_radius_px", "predicted_radius_px"),
        )
        ratio = _first_number(
            row,
            ("sphere_radius_ratio", "radius_ratio_to_expected", "sphere_scale_ratio"),
        )
        if ratio is None and observed is not None and expected is not None and expected > 1e-9:
            ratio = observed / expected
        relative_error = abs(ratio - 1.0) if ratio is not None else None
        if relative_error is None:
            residual = _finite_float(row.get("radius_residual_px"))
            denominator = expected if expected is not None else observed
            if residual is not None and denominator is not None and denominator > 1e-9:
                relative_error = abs(residual) / denominator
        scale_metrics.append(
            {
                "frame_index": frame,
                "observed_radius_px": observed,
                "expected_radius_px": expected,
                "relative_error": relative_error,
            }
        )
        if relative_error is not None and relative_error > float(
            scale_config["max_relative_expected_radius_error"]
        ):
            scale_bad.append(frame)
    scale_runs = _runs(scale_bad, int(scale_config["persistent_frames"]))
    if scale_runs and camera_static:
        _append_unique(failures, "persistent_experiment_object_scale_violation_2d")
        offending["object_scale_2d"] = _flatten(scale_runs)
    elif scale_runs:
        # A moving camera changes apparent size.  Without metric 3-D evidence
        # this is deliberately unresolved, never a deformation accusation.
        _append_unique(warnings, "object_scale_change_requires_3d_camera_compensation")
        offending["unresolved_object_scale_2d"] = _flatten(scale_runs)
    elif scale_bad:
        _append_unique(warnings, "transient_object_scale_outlier")
        offending["transient_object_scale_2d"] = sorted(set(scale_bad))

    # Some reconstructors naturally attach rigidity diagnostics to their
    # trajectory payload.  Accept those fields there as well as through the
    # explicit dynamic argument; explicit values win when both are present.
    dynamic: dict[str, Any] = {}
    if isinstance(trajectory_evidence, Mapping):
        for key in (
            "object_frames",
            "object_rigidity_frames",
            "object_rigidity",
            "experiment_object",
            "background_frames",
            "scene_frames",
            "background_rigidity",
            "scene_rigidity",
        ):
            if key in trajectory_evidence:
                dynamic[key] = trajectory_evidence[key]
    if dynamic_rigidity_evidence:
        dynamic.update(dynamic_rigidity_evidence)
    object_block = _evidence_block(
        dynamic,
        ("object_frames", "object_rigidity_frames", "object_rigidity", "experiment_object"),
    )
    scene_block = _evidence_block(
        dynamic,
        ("background_frames", "scene_frames", "background_rigidity", "scene_rigidity"),
    )
    object_3d = _rigidity_check(
        object_block,
        config["object_rigidity_3d"],
        require_spatial_coverage_for_failure=False,
    )
    scene_3d = _rigidity_check(
        scene_block,
        config["scene_rigidity_3d"],
        require_spatial_coverage_for_failure=True,
    )
    if object_3d["status"] == "fail":
        _append_unique(failures, "persistent_experiment_object_deformation_3d")
        offending["object_rigidity_3d"] = object_3d["offending_frames"]
    if scene_3d["status"] == "fail":
        _append_unique(failures, "persistent_non_experimental_scene_deformation_3d")
        offending["scene_rigidity_3d"] = scene_3d["offending_frames"]

    static_scene_status = "not_run"
    if static_scene_rigidity_evidence:
        static_scene_status = str(
            static_scene_rigidity_evidence.get("status", "indeterminate")
        ).lower()
        static_failure_code = static_scene_rigidity_evidence.get("failure_code")
        if static_scene_status == "fail":
            _append_unique(
                failures,
                str(static_failure_code or "background_or_prop_nonrigid_deformation_2d"),
            )
            sample_rows = static_scene_rigidity_evidence.get("sample_rows", [])
            offending["scene_rigidity_2d"] = [
                _frame_index(row, index)
                for index, row in enumerate(sample_rows)
                if isinstance(row, Mapping)
                and (row.get("local_event") is True or row.get("global_event") is True)
            ]
        elif static_scene_status == "indeterminate" and camera_static:
            _append_unique(warnings, "static_scene_rigidity_unresolved")
            _append_unique(blocking_indeterminate, "static_scene_rigidity_unresolved")

    if camera_changed:
        if object_3d["status"] != "pass":
            _append_unique(warnings, "moving_camera_object_rigidity_unresolved")
            _append_unique(blocking_indeterminate, "moving_camera_object_rigidity_unresolved")
        if scene_3d["status"] != "pass":
            _append_unique(warnings, "moving_camera_scene_rigidity_unresolved")
            _append_unique(blocking_indeterminate, "moving_camera_scene_rigidity_unresolved")
    elif not camera_static:
        _append_unique(warnings, "camera_motion_evidence_missing_or_unresolved")
        _append_unique(blocking_indeterminate, "camera_motion_evidence_missing_or_unresolved")

    if failures:
        status = "fail"
    elif blocking_indeterminate:
        status = "indeterminate"
    else:
        status = "pass"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "fit_eligible": status == "pass",
        "failure_codes": failures,
        "warning_codes": warnings,
        "indeterminate_codes": blocking_indeterminate,
        "offending_frames": offending,
        "checks": {
            "camera_motion": {
                "status": (
                    "fail"
                    if cut_frames
                    else "pass"
                    if camera_static
                    else "requires_3d_evidence"
                    if camera_changed
                    else "indeterminate"
                ),
                "category": camera_category,
                "scene_cut_frames": cut_frames,
            },
            "object_identity": {
                "status": (
                    "pass"
                    if tracking_ok and trusted_first_frame and not ambiguous_runs
                    else "indeterminate"
                ),
                "frame_count": total,
                "found_count": len(found_rows),
                "reliable_count": len(reliable_rows),
                "tracked_fraction": tracked_fraction,
                "reliable_fraction": reliable_fraction,
                "required_reliable_frames": required_reliable_frames,
                "trusted_first_frame": trusted_first_frame,
                "missing_frames": missing_frames,
                "unverified_frames": unverified_frames,
                "unreliable_frames": unreliable_frames,
                "unresolved_runs_exceeding_limit": unresolved_tracking_runs,
                "ambiguous_frames": _flatten(ambiguous_runs),
            },
            "object_shape_2d": {
                "status": (
                    "fail"
                    if shape_runs
                    else "pass"
                    if shape_evidence_sufficient
                    else "indeterminate"
                ),
                "evaluable_frame_count": len(shape_metrics),
                "evaluable_fraction": shape_evaluable_fraction,
                "evidence_sufficient": shape_evidence_sufficient,
                "unresolved_frames": sorted(set(shape_unresolved)),
                "unresolved_runs_exceeding_limit": unresolved_shape_runs,
                "offending_frames": _flatten(shape_runs),
                "transient_outlier_frames": [] if shape_runs else sorted(set(shape_bad)),
                "per_frame_metrics": shape_metrics,
            },
            "object_scale_2d": {
                "status": (
                    "fail"
                    if scale_runs and camera_static
                    else "indeterminate"
                    if scale_runs
                    else "pass"
                ),
                "offending_frames": _flatten(scale_runs),
                "transient_outlier_frames": [] if scale_runs else sorted(set(scale_bad)),
                "per_frame_metrics": scale_metrics,
                "interpretation": "2-D scale is a hard gate only for an audited-static camera.",
            },
            "object_rigidity_3d": object_3d,
            "scene_rigidity_3d": scene_3d,
            "scene_rigidity_2d": (
                dict(static_scene_rigidity_evidence)
                if static_scene_rigidity_evidence
                else {"status": static_scene_status}
            ),
        },
        "thresholds": config,
        "decision_rule": (
            "Physics fitting is allowed only when status=pass. Persistent strong deformation or "
            "a scene cut fails; insufficient evidence is indeterminate."
        ),
    }


__all__ = ["DEFAULT_THRESHOLDS", "SCHEMA_VERSION", "evaluate_generation_validity"]
