"""Two-stage trajectory extraction and trajectory-only physics evaluation.

The extraction stage never reads target physical parameters.  It writes one
row for every decoded source frame and one overlay for every input video.  The
evaluation stage consumes only those frozen artifacts plus the experiment
registry; it never decodes a video or invokes either tracker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from remake_benchmark.hybrid import choose_reconstruction_route

from .fixed_camera_ball import (
    calibration_path,
    load_calibration,
    parse_video_job,
    reconstruct_metric_trajectory,
    run_ball_tracking,
    write_trajectory_csv,
)
from .generation_validity import evaluate_generation_validity
from .physics_evaluation import (
    benchmark_split,
    load_experiment_registry,
    write_trajectory_plot,
    write_validity_visuals,
)
from .physics_parameters import fit_physics_parameters, lookup_target_tuple, score_parameter_fit
from .scene_rigidity import assess_static_scene_rigidity
from .trajectory_3d_inclusion import (
    assess_dynamic_3d_trajectory,
    finalize_dynamic_3d_inclusion,
)
from .seedance978 import (
    _audit_index,
    _ordered_manifest,
    _phase_rows,
    _read_jsonl,
    _route_adjusted_camera_evidence,
    _runtime_spatracker_manifest,
    _write_jsonl,
    validate_978_inputs,
    write_seedance978_reports,
)


TRACK_SCHEMA_VERSION = "1.0.0"
EVALUATOR_VERSION = "1.1.0"
STATIC_ROUTE = "calibrated_static_sphere"
DYNAMIC_ROUTE = "spatialtrackerv2_dynamic"
DYNAMIC_MIN_ANCHOR_INLIER_FRACTION = 0.50
DYNAMIC_MAX_ANCHOR_RMSE_M = 0.30

COMMON_FIELDS = [
    "frame_index",
    "source_frame_index",
    "time_s",
    "route",
    "position_source",
    "center_u_px",
    "center_v_px",
    "radius_px",
    "x_m",
    "y_m",
    "z_m",
    "x_raw_m",
    "y_raw_m",
    "z_raw_m",
    "fit_x_m",
    "fit_y_m",
    "fit_z_m",
    "q_value",
    "theta_rad",
    "coordinate_frame_3d",
    "valid_2d",
    "valid_3d",
    "measurement_valid",
    "interpolated",
    "fit_eligible",
    "track_confidence",
    "uncertainty_px",
    "uncertainty_m",
    "identity_verified",
    "object_visible_fraction",
    "object_inlier_fraction",
    "object_rigid_rmse_over_radius",
    "anchor_inlier_fraction",
    "anchor_rmse_m",
    "alignment_fallback",
    "invalid_reason_codes",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    fieldnames = list(fields or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _finite_xyz(row: Mapping[str, Any]) -> bool:
    return all(_float(row.get(name)) is not None for name in ("x_m", "y_m", "z_m"))


def _finite_fit_xyz(row: Mapping[str, Any]) -> bool:
    return all(_float(row.get(name)) is not None for name in ("fit_x_m", "fit_y_m", "fit_z_m"))


def _csv_scalar(value: Any) -> Any:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return value
    if number.is_integer() and not any(char in text.lower() for char in (".", "e")):
        return int(number)
    return number


def _read_typed_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: _csv_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_checkout_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not (path / ".git").exists():
        return {"root": str(path), "git_commit": None, "git_dirty": None}
    try:
        git_prefix = ["git", "-c", f"safe.directory={path}", "-C", str(path)]
        commit = subprocess.check_output(
            [*git_prefix, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [*git_prefix, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"root": str(path), "git_commit": commit, "git_dirty": dirty}


def normalize_static_positions(
    track_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    fps: float,
) -> list[dict[str, Any]]:
    """Map fixed-camera native rows to the common one-row-per-frame contract."""

    metric_by_frame = {int(row.get("frame_index", index)): row for index, row in enumerate(trajectory_rows)}
    output: list[dict[str, Any]] = []
    for index, source in enumerate(track_rows):
        frame_index = int(source.get("frame_index", index))
        metric = metric_by_frame.get(frame_index, {})
        continuous_source = str(source.get("continuous_source") or "")
        measured = _truth(source.get("measurement_valid")) and source.get("center_u_px") is not None
        if source.get("continuous_center_u_px") is not None and source.get("continuous_center_v_px") is not None:
            u = _float(source.get("continuous_center_u_px"))
            v = _float(source.get("continuous_center_v_px"))
            radius = _float(source.get("continuous_radius_px"))
            position_source = continuous_source or "fixed_camera_continuous"
        elif source.get("center_u_px") is not None and source.get("center_v_px") is not None:
            u = _float(source.get("center_u_px"))
            v = _float(source.get("center_v_px"))
            radius = _float(source.get("measurement_radius_px") or source.get("display_radius_px"))
            position_source = "fixed_camera_measurement"
        elif source.get("predicted_center_u_px") is not None and source.get("predicted_center_v_px") is not None:
            u = _float(source.get("predicted_center_u_px"))
            v = _float(source.get("predicted_center_v_px"))
            radius = _float(source.get("display_radius_px"))
            position_source = "motion_prediction_unverified"
        else:
            u = v = radius = None
            position_source = "unavailable"

        interpolated = position_source == "bracketed_linear_interpolation"
        raw_x = _float(metric.get("x_m"))
        raw_y = _float(metric.get("y_m"))
        raw_z = _float(metric.get("z_m"))
        raw_q = _float(metric.get("q_value"))
        raw_metric_available = all(value is not None for value in (raw_x, raw_y, raw_z))
        if _truth(metric.get("continuous_metric_available")):
            x = _float(metric.get("continuous_x_m"))
            y = _float(metric.get("continuous_y_m"))
            z = _float(metric.get("continuous_z_m"))
            q_value = _float(metric.get("continuous_q_value"))
        else:
            x, y, z, q_value = raw_x, raw_y, raw_z, raw_q
        valid_3d = all(value is not None for value in (x, y, z))
        fit_eligible = bool(
            _truth(metric.get("physics_fit_used"))
            and measured
            and not interpolated
            and raw_metric_available
        )
        fit_x, fit_y, fit_z = (
            (raw_x, raw_y, raw_z) if fit_eligible else (None, None, None)
        )
        invalid_reasons: list[str] = []
        if u is None or v is None:
            invalid_reasons.append("position_2d_unavailable")
        if not measured:
            invalid_reasons.append("not_direct_verified_measurement")
        if not valid_3d:
            invalid_reasons.append("metric_position_unavailable")
        if interpolated:
            invalid_reasons.append("interpolated_not_fit_evidence")
        output.append(
            {
                "frame_index": frame_index,
                "source_frame_index": frame_index,
                "time_s": frame_index / max(float(fps), 1e-9),
                "route": STATIC_ROUTE,
                "position_source": position_source,
                "center_u_px": u,
                "center_v_px": v,
                "radius_px": radius,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "x_raw_m": raw_x,
                "y_raw_m": raw_y,
                "z_raw_m": raw_z,
                "fit_x_m": fit_x,
                "fit_y_m": fit_y,
                "fit_z_m": fit_z,
                "q_value": q_value,
                "theta_rad": _float(metric.get("theta_rad")),
                "coordinate_frame_3d": "blender_world_m" if valid_3d else None,
                "valid_2d": u is not None and v is not None,
                "valid_3d": valid_3d,
                "measurement_valid": measured,
                "interpolated": interpolated,
                "fit_eligible": fit_eligible,
                "track_confidence": _float(source.get("track_confidence")),
                "uncertainty_px": _float(source.get("continuous_uncertainty_px")),
                "uncertainty_m": None,
                "identity_verified": _truth(source.get("identity_verified", source.get("found"))),
                "object_visible_fraction": 1.0 if measured else 0.0,
                "object_inlier_fraction": None,
                "object_rigid_rmse_over_radius": None,
                "anchor_inlier_fraction": None,
                "anchor_rmse_m": None,
                "alignment_fallback": False,
                "invalid_reason_codes": ";".join(invalid_reasons),
            }
        )
    return output


def _normalise_tn(array: np.ndarray, frames: int, points: int) -> np.ndarray:
    value = np.squeeze(np.asarray(array))
    if value.shape == (frames, points):
        return value
    if value.shape == (points, frames):
        return value.T
    if value.ndim == 1 and len(value) == points:
        return np.repeat(value[None], frames, axis=0)
    raise ValueError(f"cannot normalize {value.shape} to ({frames}, {points})")


def normalize_dynamic_positions(dynamic_dir: Path) -> list[dict[str, Any]]:
    """Combine SpaTrackerV2 raw 2D queries and postprocessed metric centres."""

    raw_path = dynamic_dir / "raw_spatialtrackerv2.npz"
    trajectory_path = dynamic_dir / "trajectory_world.csv"
    if not raw_path.is_file() or not trajectory_path.is_file():
        return []
    raw = dict(np.load(raw_path, allow_pickle=True))
    trajectory = _read_typed_csv(trajectory_path)
    tracks = np.squeeze(np.asarray(raw["track2d_input"], dtype=np.float64))[..., :2]
    if tracks.ndim != 3:
        raise ValueError(f"unexpected track2d_input shape: {tracks.shape}")
    frames, points = tracks.shape[:2]
    visible = _normalise_tn(np.asarray(raw["visibs"]), frames, points) > 0.5
    confidence = _normalise_tn(np.asarray(raw.get("track_confidence", np.ones((frames, points)))), frames, points)
    kinds = np.asarray(raw["query_meta_query_kind"]).astype(str).reshape(-1)
    object_mask = kinds == "object"
    if object_mask.sum() < 3:
        raise ValueError("fewer than three object queries in SpaTrackerV2 result")
    scale_x, scale_y = np.asarray(raw["preprocess_scale_xy"], dtype=np.float64).reshape(2)
    crop_top = float(np.asarray(raw["preprocess_crop_top"]).reshape(()))
    tracks[..., 0] /= scale_x
    tracks[..., 1] = (tracks[..., 1] + crop_top) / scale_y
    source_indices = np.asarray(raw["source_frame_indices"], dtype=int).reshape(-1)
    source_fps = float(np.asarray(raw["source_fps"]).reshape(()))
    trajectory_by_source = {
        int(row.get("source_frame", row.get("frame_index", index))): row
        for index, row in enumerate(trajectory)
    }
    alignment_path = dynamic_dir / "alignment.json"
    alignment_rows = json.loads(alignment_path.read_text(encoding="utf-8")) if alignment_path.is_file() else []
    alignment_by_index = {
        int(row.get("frame_index", index)): row for index, row in enumerate(alignment_rows)
    }
    output: list[dict[str, Any]] = []
    for time_index, source_frame in enumerate(source_indices):
        object_tracks = tracks[time_index, object_mask]
        object_visible = visible[time_index, object_mask]
        object_confidence = confidence[time_index, object_mask]
        finite = np.isfinite(object_tracks).all(axis=1)
        direct = finite & object_visible
        chosen = direct if direct.any() else finite
        if chosen.any():
            centre = np.median(object_tracks[chosen], axis=0)
            u, v = float(centre[0]), float(centre[1])
            confidence_value = float(np.clip(np.median(object_confidence[chosen]), 0.0, 1.0))
        else:
            u = v = confidence_value = None
        metric = trajectory_by_source.get(int(source_frame), {})
        alignment = alignment_by_index.get(time_index, {})
        x, y, z = (_float(metric.get(name)) for name in ("x_m", "y_m", "z_m"))
        x_raw, y_raw, z_raw = (
            _float(metric.get(name)) for name in ("x_raw_m", "y_raw_m", "z_raw_m")
        )
        valid_3d = all(value is not None for value in (x, y, z))
        raw_metric_available = all(value is not None for value in (x_raw, y_raw, z_raw))
        visible_fraction = float(direct.sum() / max(int(object_mask.sum()), 1))
        inlier_fraction = _float(metric.get("object_inlier_fraction"))
        alignment_fallback = _truth(alignment.get("fallback"))
        anchor_inlier_fraction = _float(alignment.get("inlier_fraction"))
        anchor_rmse_m = _float(alignment.get("rmse_m"))
        anchor_quality_pass = bool(
            anchor_inlier_fraction is not None
            and anchor_inlier_fraction >= DYNAMIC_MIN_ANCHOR_INLIER_FRACTION
            and anchor_rmse_m is not None
            and anchor_rmse_m <= DYNAMIC_MAX_ANCHOR_RMSE_M
        )
        direct_measurement = bool(
            direct.sum() >= 3
            and raw_metric_available
            and not alignment_fallback
            and anchor_quality_pass
        )
        temporally_imputed = bool(valid_3d and not raw_metric_available)
        fit_eligible = bool(
            direct_measurement
            and (inlier_fraction is None or inlier_fraction >= 0.50)
        )
        invalid_reasons: list[str] = []
        if u is None or v is None:
            invalid_reasons.append("position_2d_unavailable")
        if not direct.any():
            invalid_reasons.append("no_visible_object_query")
        if not valid_3d:
            invalid_reasons.append("metric_position_unavailable")
        if temporally_imputed:
            invalid_reasons.append("temporal_imputation_not_fit_evidence")
        if alignment_fallback:
            invalid_reasons.append("background_alignment_fallback")
        if not anchor_quality_pass:
            invalid_reasons.append("background_alignment_quality_low")
        if inlier_fraction is not None and inlier_fraction < 0.50:
            invalid_reasons.append("object_rigid_inlier_fraction_low")
        output.append(
            {
                "frame_index": time_index,
                "source_frame_index": int(source_frame),
                "time_s": int(source_frame) / max(source_fps, 1e-9),
                "route": DYNAMIC_ROUTE,
                "position_source": (
                    "spatialtrackerv2_direct_temporal_median"
                    if direct_measurement
                    else "temporal_filter_imputation"
                    if temporally_imputed
                    else "spatialtrackerv2_prediction_unverified"
                    if chosen.any()
                    else "unavailable"
                ),
                "center_u_px": u,
                "center_v_px": v,
                "radius_px": None,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "x_raw_m": x_raw,
                "y_raw_m": y_raw,
                "z_raw_m": z_raw,
                "fit_x_m": x_raw if fit_eligible else None,
                "fit_y_m": y_raw if fit_eligible else None,
                "fit_z_m": z_raw if fit_eligible else None,
                "q_value": None,
                "theta_rad": None,
                "coordinate_frame_3d": (
                    "spatialtrackerv2_frame0_metric_aligned_m" if valid_3d else None
                ),
                "valid_2d": u is not None and v is not None,
                "valid_3d": valid_3d,
                "measurement_valid": direct_measurement,
                "interpolated": temporally_imputed,
                "fit_eligible": fit_eligible,
                "track_confidence": confidence_value,
                "uncertainty_px": None,
                "uncertainty_m": _float(metric.get("object_rigid_rmse_m")),
                "identity_verified": bool(direct.sum() >= 3),
                "object_visible_fraction": visible_fraction,
                "object_inlier_fraction": inlier_fraction,
                "object_rigid_rmse_over_radius": _float(metric.get("object_all_point_rmse_over_radius")),
                "anchor_inlier_fraction": anchor_inlier_fraction,
                "anchor_rmse_m": anchor_rmse_m,
                "alignment_fallback": alignment_fallback,
                "invalid_reason_codes": ";".join(invalid_reasons),
            }
        )
    return output


def _video_properties(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0}
    properties: dict[str, float | int] = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 24.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return properties


def _decoded_video_properties(video_path: Path) -> dict[str, float | int | bool]:
    """Count decodable frames instead of trusting container header metadata."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "opened": False,
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
        }
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return {
        "opened": True,
        "frame_count": count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _reusable_track_result(
    result_path: Path,
    video_path: Path,
    *,
    expected_route: str | None = None,
) -> dict[str, Any] | None:
    """Return a complete cached extraction, otherwise force an automatic retry."""

    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        result.get("schema_version") != TRACK_SCHEMA_VERSION
        or result.get("status") not in {"succeeded", "partial"}
        or result.get("target_parameters_used") is not False
        or (
            expected_route is not None
            and result.get("reconstruction_route") != expected_route
        )
    ):
        return None
    job_dir = result_path.parent
    trajectory_path = job_dir / "trajectory_frames.csv"
    overlay_path = job_dir / "object_track_overlay.mp4"
    if not trajectory_path.is_file() or not overlay_path.is_file():
        return None
    try:
        rows = _read_typed_csv(trajectory_path)
    except (OSError, csv.Error, ValueError):
        return None
    required_fields = set(COMMON_FIELDS)
    if not rows or not required_fields.issubset(rows[0]):
        return None
    source = _decoded_video_properties(video_path)
    overlay = _decoded_video_properties(overlay_path)
    frame_count = int(source["frame_count"])
    if (
        not source["opened"]
        or not overlay["opened"]
        or frame_count <= 0
        or len(rows) != frame_count
        or int(overlay["frame_count"]) != frame_count
        or (source["width"], source["height"]) != (overlay["width"], overlay["height"])
        or abs(float(source["fps"]) - float(overlay["fps"])) > 0.05
    ):
        return None
    indices = [int(row.get("source_frame_index", -1)) for row in rows]
    if indices != list(range(frame_count)):
        return None
    return result


def _complete_frame_contract(
    video_path: Path,
    rows: Sequence[Mapping[str, Any]],
    route: str,
    *,
    expected_frame_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Guarantee exactly one explicit row for every source video frame."""

    properties = _video_properties(video_path)
    fps = float(properties["fps"])
    by_source: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_index = int(row.get("source_frame_index", row.get("frame_index", 0)))
        if source_index in by_source:
            raise ValueError(f"duplicate source_frame_index={source_index}")
        by_source[source_index] = dict(row)
    decoded_from_rows = max(by_source, default=-1) + 1
    frame_count = (
        int(expected_frame_count)
        if expected_frame_count is not None
        else decoded_from_rows
        if decoded_from_rows > 0
        else int(properties["frame_count"])
    )
    if decoded_from_rows > frame_count:
        raise ValueError(
            f"trajectory contains source frame {decoded_from_rows - 1}, "
            f"but decoded frame count is {frame_count}"
        )
    output: list[dict[str, Any]] = []
    for source_index in range(frame_count):
        if source_index in by_source:
            row = by_source[source_index]
            row["frame_index"] = source_index
            row["source_frame_index"] = source_index
            row["time_s"] = source_index / max(fps, 1e-9)
        else:
            row = {
                **{field: None for field in COMMON_FIELDS},
                "frame_index": source_index,
                "source_frame_index": source_index,
                "time_s": source_index / max(fps, 1e-9),
                "route": route,
                "position_source": "unavailable",
                "valid_2d": False,
                "valid_3d": False,
                "measurement_valid": False,
                "interpolated": False,
                "fit_eligible": False,
                "identity_verified": False,
                "alignment_fallback": route == DYNAMIC_ROUTE,
                "invalid_reason_codes": "missing_source_frame_result",
            }
        output.append(row)
    properties["frame_count"] = frame_count
    return output, properties


def _write_common_overlay(
    video_path: Path,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, float | int]:
    """Render the same auditable position overlay for either reconstruction route."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"cannot open overlay VideoWriter: {output_path}")
    by_source = {int(row["source_frame_index"]): row for row in rows}
    history_segments: list[list[tuple[int, int]]] = []
    active_history: list[tuple[int, int]] | None = None
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = by_source.get(frame_index)
        valid_2d = bool(row and _truth(row.get("valid_2d")))
        measured = bool(row and _truth(row.get("measurement_valid")))
        interpolated = bool(row and _truth(row.get("interpolated")))
        if valid_2d and row is not None:
            u = int(round(float(row["center_u_px"])))
            v = int(round(float(row["center_v_px"])))
            if measured:
                colour = (0, 210, 0)
            elif interpolated:
                colour = (255, 210, 0)
            else:
                colour = (0, 165, 255)
            if measured:
                if active_history is None:
                    active_history = []
                    history_segments.append(active_history)
                active_history.append((u, v))
            else:
                active_history = None
            radius = _float(row.get("radius_px"))
            if radius is not None and radius > 0:
                cv2.circle(frame, (u, v), max(2, int(round(radius))), colour, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (u, v), colour, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        else:
            active_history = None
        for segment in history_segments:
            if len(segment) >= 2:
                cv2.polylines(frame, [np.asarray(segment, dtype=np.int32)], False, (40, 220, 40), 2, cv2.LINE_AA)

        if row is None:
            lines = [f"frame={frame_index}  TRACK UNAVAILABLE", "reason: no trajectory row"]
        else:
            u_value = _float(row.get("center_u_px"))
            v_value = _float(row.get("center_v_px"))
            xyz = [_float(row.get(name)) for name in ("x_m", "y_m", "z_m")]
            uv_text = "unavailable" if u_value is None or v_value is None else f"({u_value:.1f}, {v_value:.1f}) px"
            xyz_text = (
                "unavailable"
                if any(value is None for value in xyz)
                else f"({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) m"
            )
            confidence = _float(row.get("track_confidence"))
            confidence_text = "n/a" if confidence is None else f"{confidence:.3f}"
            lines = [
                f"frame: {frame_index}    time: {float(row.get('time_s') or 0.0):.3f} s",
                f"route: {row.get('route')}",
                f"source: {row.get('position_source')}    confidence: {confidence_text}",
                f"2D: {uv_text}    fit eligible: {_truth(row.get('fit_eligible'))}",
                f"3D: {xyz_text}",
            ]
            if row.get("invalid_reason_codes"):
                lines.append(f"reason: {str(row['invalid_reason_codes'])[:96]}")
        panel_height = 12 + len(lines) * 22
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (width, min(height, panel_height)), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.58, frame, 0.42, 0.0, frame)
        for line_index, line in enumerate(lines):
            y = 22 + line_index * 22
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()
    os.replace(temporary, output_path)
    return {"frame_count": frame_index, "fps": fps, "width": width, "height": height}


def _write_unavailable_video(
    video_path: Path,
    output_path: Path,
    message: str,
) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"cannot open unavailable-overlay VideoWriter: {output_path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, "TRACK UNAVAILABLE", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, message[:90], (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        count += 1
    writer.release()
    cap.release()
    os.replace(temporary, output_path)
    return {"frame_count": count, "fps": fps, "width": width, "height": height}


def _position_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "frame_count": count,
        "valid_2d_count": sum(_truth(row.get("valid_2d")) for row in rows),
        "valid_2d_fraction": sum(_truth(row.get("valid_2d")) for row in rows) / max(count, 1),
        "valid_3d_count": sum(_truth(row.get("valid_3d")) for row in rows),
        "valid_3d_fraction": sum(_truth(row.get("valid_3d")) for row in rows) / max(count, 1),
        "direct_measurement_count": sum(_truth(row.get("measurement_valid")) for row in rows),
        "direct_measurement_fraction": sum(_truth(row.get("measurement_valid")) for row in rows) / max(count, 1),
        "fit_eligible_count": sum(_truth(row.get("fit_eligible")) for row in rows),
        "fit_eligible_fraction": sum(_truth(row.get("fit_eligible")) for row in rows) / max(count, 1),
        "interpolated_count": sum(_truth(row.get("interpolated")) for row in rows),
    }


def _extract_static_job(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    job_id = str(payload["job_id"])
    video_path = Path(payload["video_path"])
    job_dir = Path(payload["output_root"]) / "jobs" / job_id
    result_path = job_dir / "track_result.json"
    if not payload.get("overwrite"):
        cached = _reusable_track_result(
            result_path,
            video_path,
            expected_route=STATIC_ROUTE,
        )
        if cached is not None:
            return job_id, cached
    job_dir.mkdir(parents=True, exist_ok=True)
    job = parse_video_job(video_path)
    calibration_sidecar = calibration_path(Path(payload["calibration_root"]), job)
    error: dict[str, Any] | None = None
    frames: Sequence[np.ndarray] = []
    track_rows: Sequence[Mapping[str, Any]] = []
    trajectory_rows: Sequence[Mapping[str, Any]] = []
    pipeline: dict[str, Any] = {}
    validity: dict[str, Any] = {
        "status": "indeterminate",
        "fit_eligible": False,
        "warning_codes": ["extraction_not_started"],
        "failure_codes": [],
    }
    try:
        calibration = load_calibration(calibration_sidecar)
        frames, track_rows, pipeline = run_ball_tracking(video_path, calibration)
        write_trajectory_csv(job_dir / "native_tracking.csv", track_rows)
        video = pipeline["video"]
        reconstruction_error = None
        try:
            trajectory_rows, geometry = reconstruct_metric_trajectory(
                track_rows,
                calibration,
                experiment_id=str(job["experiment_id"]),
                video_size=(int(video["width"]), int(video["height"])),
                fps=float(video["fps"]),
            )
            pipeline["geometry"] = geometry
            write_trajectory_csv(job_dir / "native_trajectory.csv", trajectory_rows)
        except Exception as exc:
            reconstruction_error = f"{type(exc).__name__}: {exc}"
            pipeline["geometry_error"] = reconstruction_error
        static_evidence = None
        if payload.get("assess_background_rigidity", True):
            static_evidence = assess_static_scene_rigidity(
                frames,
                track_rows,
                camera_motion_evidence=payload.get("camera_motion_evidence"),
                calibration=calibration,
            )
            pipeline["static_scene_rigidity"] = static_evidence
        validity = evaluate_generation_validity(
            track_rows,
            camera_motion_evidence=payload.get("camera_motion_evidence"),
            static_scene_rigidity_evidence=static_evidence,
            trajectory_evidence=trajectory_rows,
        )
        positions = normalize_static_positions(track_rows, trajectory_rows, float(video["fps"]))
        positions, source_video_properties = _complete_frame_contract(
            video_path,
            positions,
            STATIC_ROUTE,
            expected_frame_count=len(frames),
        )
        _write_rows(job_dir / "trajectory_frames.csv", positions, COMMON_FIELDS)
        visuals = write_validity_visuals(
            job_dir,
            frames,
            track_rows,
            validity,
            float(video["fps"]),
        )
        overlay_source = Path(visuals["overlay_video"]) if visuals.get("overlay_video") else None
        native_overlay = str(overlay_source) if overlay_source and overlay_source.is_file() else None
        try:
            overlay_properties = _write_common_overlay(
                video_path,
                positions,
                job_dir / "object_track_overlay.mp4",
            )
            overlay = (
                str(job_dir / "object_track_overlay.mp4")
                if overlay_properties["frame_count"]
                else None
            )
        except Exception as overlay_exc:
            overlay_properties = {
                "frame_count": 0,
                "fps": source_video_properties["fps"],
                "width": source_video_properties["width"],
                "height": source_video_properties["height"],
            }
            overlay = None
            pipeline["overlay_error"] = f"{type(overlay_exc).__name__}: {overlay_exc}"
        summary = _position_summary(positions)
        status = "succeeded" if summary["valid_2d_count"] else "failed"
        if (reconstruction_error or overlay is None) and status == "succeeded":
            status = "partial"
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            overlay_properties = _write_unavailable_video(
                video_path,
                job_dir / "object_track_overlay.mp4",
                str(exc),
            )
        except Exception as overlay_exc:
            decoded = _decoded_video_properties(video_path)
            overlay_properties = {
                "frame_count": int(decoded["frame_count"]),
                "fps": float(decoded["fps"]),
                "width": int(decoded["width"]),
                "height": int(decoded["height"]),
            }
            error["overlay_error"] = f"{type(overlay_exc).__name__}: {overlay_exc}"
        count = int(overlay_properties["frame_count"])
        positions = [
            {
                **{field: None for field in COMMON_FIELDS},
                "frame_index": index,
                "source_frame_index": index,
                "time_s": index / max(float(overlay_properties["fps"]), 1e-9),
                "route": STATIC_ROUTE,
                "position_source": "unavailable",
                "valid_2d": False,
                "valid_3d": False,
                "measurement_valid": False,
                "interpolated": False,
                "fit_eligible": False,
                "identity_verified": False,
                "alignment_fallback": False,
                "invalid_reason_codes": "extraction_error",
            }
            for index in range(count)
        ]
        _write_rows(job_dir / "trajectory_frames.csv", positions, COMMON_FIELDS)
        summary = _position_summary(positions)
        overlay = str(job_dir / "object_track_overlay.mp4") if count else None
        native_overlay = None
        status = "failed"
    result = {
        "schema_version": TRACK_SCHEMA_VERSION,
        "status": status,
        "job": job,
        "benchmark_split": benchmark_split(job),
        "reconstruction_route": STATIC_ROUTE,
        "source_video": str(video_path),
        "calibration_path": str(calibration_sidecar),
        "calibration_sha256": (
            _sha256(calibration_sidecar) if calibration_sidecar.is_file() else None
        ),
        "target_parameters_used": False,
        "trajectory_frames_csv": str(job_dir / "trajectory_frames.csv"),
        "native_tracking_csv": str(job_dir / "native_tracking.csv") if track_rows else None,
        "native_trajectory_csv": str(job_dir / "native_trajectory.csv") if trajectory_rows else None,
        "native_object_track_overlay": native_overlay,
        "object_track_overlay": overlay,
        "source_video_properties": (
            source_video_properties if error is None else overlay_properties
        ),
        "overlay_properties": overlay_properties,
        "position_summary": summary,
        "video_generation_validity": validity,
        # This is a trajectory-quality statement only.  Video-generation
        # validity is applied later by the independent evaluation stage.
        "trajectory_fit_eligible": summary["fit_eligible_count"] >= 3,
        "pipeline": pipeline,
        "camera_motion_evidence": dict(payload.get("camera_motion_evidence") or {}),
        "error": error,
    }
    _write_json(result_path, result)
    return job_id, result


def _extract_dynamic_result(
    *,
    job_id: str,
    video_path: Path,
    dynamic_dir: Path,
    output_root: Path,
    camera_motion_evidence: Mapping[str, Any],
    calibration_sidecar: Path,
) -> dict[str, Any]:
    job_dir = output_root / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = parse_video_job(video_path)
    native_result_path = dynamic_dir / "result.json"
    native_result = (
        json.loads(native_result_path.read_text(encoding="utf-8"))
        if native_result_path.is_file()
        else {"status": "failed", "error": "SpaTrackerV2 result.json missing"}
    )
    error = None
    try:
        positions = normalize_dynamic_positions(dynamic_dir)
        if positions:
            positions, source_video_properties = _complete_frame_contract(
                video_path,
                positions,
                DYNAMIC_ROUTE,
                expected_frame_count=max(
                    int(row.get("source_frame_index", -1)) for row in positions
                )
                + 1,
            )
        else:
            source_video_properties = _video_properties(video_path)
    except Exception as exc:
        positions = []
        source_video_properties = _video_properties(video_path)
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    dynamic_native_overlay = dynamic_dir / "object_track_overlay.mp4"
    native_overlay = str(dynamic_native_overlay) if dynamic_native_overlay.is_file() else None
    if not positions:
        try:
            overlay_properties = _write_unavailable_video(
                video_path,
                job_dir / "object_track_overlay.mp4",
                str(error or native_result.get("error") or "SpaTrackerV2 output unavailable"),
            )
        except Exception as overlay_exc:
            decoded = _decoded_video_properties(video_path)
            overlay_properties = {
                "frame_count": int(decoded["frame_count"]),
                "fps": float(decoded["fps"]),
                "width": int(decoded["width"]),
                "height": int(decoded["height"]),
            }
            if error is None:
                error = {"type": "DynamicExtractionUnavailable", "message": str(native_result.get("error"))}
            error["overlay_error"] = f"{type(overlay_exc).__name__}: {overlay_exc}"
        count = int(overlay_properties["frame_count"])
        positions = [
            {
                **{field: None for field in COMMON_FIELDS},
                "frame_index": index,
                "source_frame_index": index,
                "time_s": index / max(float(overlay_properties["fps"]), 1e-9),
                "route": DYNAMIC_ROUTE,
                "position_source": "unavailable",
                "valid_2d": False,
                "valid_3d": False,
                "measurement_valid": False,
                "interpolated": False,
                "fit_eligible": False,
                "identity_verified": False,
                "alignment_fallback": True,
                "invalid_reason_codes": "dynamic_extraction_unavailable",
            }
            for index in range(count)
        ]
        overlay = str(job_dir / "object_track_overlay.mp4") if count else None
    else:
        try:
            overlay_properties = _write_common_overlay(
                video_path,
                positions,
                job_dir / "object_track_overlay.mp4",
            )
            overlay = (
                str(job_dir / "object_track_overlay.mp4")
                if overlay_properties["frame_count"]
                else None
            )
        except Exception as overlay_exc:
            overlay_properties = {
                "frame_count": 0,
                "fps": source_video_properties["fps"],
                "width": source_video_properties["width"],
                "height": source_video_properties["height"],
            }
            overlay = None
            if error is None:
                error = {"type": type(overlay_exc).__name__, "message": str(overlay_exc)}
            error["overlay_error"] = f"{type(overlay_exc).__name__}: {overlay_exc}"
    _write_rows(job_dir / "trajectory_frames.csv", positions, COMMON_FIELDS)
    summary = _position_summary(positions)
    native_validity = dict(native_result.get("generation_validity") or {})
    native_status = str(native_validity.get("status", "indeterminate")).lower()
    if native_status == "failed":
        native_validity["status"] = "fail"
    trajectory_fit_eligible = bool(
        native_result.get("status") == "succeeded"
        and native_result.get("quality_pass") is True
        and summary["fit_eligible_count"] >= 3
    )
    status = (
        "succeeded"
        if native_result.get("status") == "succeeded"
        and summary["valid_2d_count"] > 0
        and overlay is not None
        else "partial"
        if summary["valid_2d_count"] > 0
        else "failed"
    )
    result = {
        "schema_version": TRACK_SCHEMA_VERSION,
        "status": status,
        "job": job,
        "benchmark_split": benchmark_split(job),
        "reconstruction_route": DYNAMIC_ROUTE,
        "source_video": str(video_path),
        "calibration_path": str(calibration_sidecar),
        "calibration_sha256": (
            _sha256(calibration_sidecar) if calibration_sidecar.is_file() else None
        ),
        "target_parameters_used": False,
        "trajectory_frames_csv": str(job_dir / "trajectory_frames.csv"),
        "native_tracking_csv": None,
        "native_trajectory_csv": (
            str(dynamic_dir / "trajectory_world.csv")
            if (dynamic_dir / "trajectory_world.csv").is_file()
            else None
        ),
        "raw_spatialtrackerv2_npz": (
            str(dynamic_dir / "raw_spatialtrackerv2.npz")
            if (dynamic_dir / "raw_spatialtrackerv2.npz").is_file()
            else None
        ),
        "native_object_track_overlay": native_overlay,
        "object_track_overlay": overlay,
        "source_video_properties": source_video_properties,
        "overlay_properties": overlay_properties,
        "position_summary": summary,
        "video_generation_validity": native_validity,
        "trajectory_fit_eligible": trajectory_fit_eligible,
        "pipeline": {"dynamic_reconstruction": native_result},
        "camera_motion_evidence": dict(camera_motion_evidence),
        "error": error,
    }
    _write_json(job_dir / "track_result.json", result)
    return result


def _refresh_extraction_summary(output_root: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in manifest_rows:
        job_id = str(source["job_id"])
        result_path = output_root / "jobs" / job_id / "track_result.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            summary = result.get("position_summary", {})
            rows.append(
                {
                    "job_id": job_id,
                    "status": result.get("status"),
                    "route": result.get("reconstruction_route"),
                    "frame_count": summary.get("frame_count"),
                    "valid_2d_fraction": summary.get("valid_2d_fraction"),
                    "valid_3d_fraction": summary.get("valid_3d_fraction"),
                    "direct_measurement_fraction": summary.get("direct_measurement_fraction"),
                    "fit_eligible_fraction": summary.get("fit_eligible_fraction"),
                    "trajectory_fit_eligible": result.get("trajectory_fit_eligible"),
                    "generation_validity_status": result.get("video_generation_validity", {}).get("status"),
                    "trajectory_frames_csv": result.get("trajectory_frames_csv"),
                    "object_track_overlay": result.get("object_track_overlay"),
                    "error": (
                        (result.get("error") or {}).get("message")
                        if isinstance(result.get("error"), Mapping)
                        else result.get("error")
                    ),
                }
            )
    _write_rows(output_root / "summary.csv", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    summary = {
        "schema_version": TRACK_SCHEMA_VERSION,
        "scheduled_jobs": len(manifest_rows),
        "extraction_results_found": len(rows),
        "status_counts": counts,
        "all_results_have_frame_csv": all(Path(str(row["trajectory_frames_csv"])).is_file() for row in rows),
        "all_results_have_overlay": all(
            row.get("object_track_overlay")
            and Path(str(row["object_track_overlay"])).is_file()
            for row in rows
        ),
        "rows": rows,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_seedance978_trajectory_extraction(
    *,
    workspace_root: Path,
    videos_dir: Path,
    manifest_path: Path,
    audit_jsonl: Path,
    calibration_root: Path,
    output_root: Path,
    spatialtracker_script: Path,
    phase: str = "all",
    job_ids: Sequence[str] | None = None,
    max_jobs: int | None = None,
    overwrite: bool = False,
    assess_background_rigidity: bool = True,
    workers: int = 1,
    run_dynamic: bool = True,
    dynamic_frame_stride: int = 1,
    isolated_process: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dynamic_frame_stride != 1:
        raise ValueError("The full-frame trajectory contract requires --dynamic-frame-stride 1")
    manifest_rows = _ordered_manifest(_read_jsonl(manifest_path))
    validate_978_inputs(videos_dir, manifest_rows)
    audit = _audit_index(audit_jsonl)
    runtime_manifest = _runtime_spatracker_manifest(
        manifest_rows,
        videos_dir=videos_dir,
        workspace_root=workspace_root,
        calibration_root=calibration_root,
    )
    runtime_manifest_path = output_root / "runtime_spatialtracker_manifest.jsonl"
    _write_jsonl(runtime_manifest_path, runtime_manifest)
    selected = _phase_rows(manifest_rows, phase)
    requested = list(dict.fromkeys(str(value) for value in (job_ids or [])))
    if requested:
        known = {str(row["job_id"]) for row in selected}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(f"unknown/out-of-phase job ids: {unknown}")
        selected = [row for row in selected if str(row["job_id"]) in set(requested)]
    if max_jobs is not None:
        selected = selected[: max(0, int(max_jobs))]
    output_root.mkdir(parents=True, exist_ok=True)

    routes: list[dict[str, Any]] = []
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for source in selected:
        job_id = str(source["job_id"])
        filename = f"{job_id}.mp4"
        camera = str(source.get("factors", {}).get("camera") or "")
        if filename not in audit:
            raise ValueError(f"camera audit missing: {filename}")
        decision = choose_reconstruction_route(audit[filename], filename, camera_name=camera)
        evidence = _route_adjusted_camera_evidence(audit[filename], decision)
        evidence_by_id[job_id] = evidence
        routes.append({"job_id": job_id, **decision})
    _write_jsonl(output_root / f"routing_{phase}.jsonl", routes)
    route_by_id = {str(row["job_id"]): str(row["route"]) for row in routes}
    route_counts = {
        route: sum(value == route for value in route_by_id.values())
        for route in sorted(set(route_by_id.values()))
    }
    external_spatialtracker_root = (
        Path(os.environ["SPATIALTRACKERV2_ROOT"]).expanduser().resolve()
        if os.environ.get("SPATIALTRACKERV2_ROOT")
        else None
    )
    spatialtracker_checkout = _git_checkout_state(external_spatialtracker_root)
    dynamic_job_ids = [job_id for job_id, route in route_by_id.items() if route == DYNAMIC_ROUTE]
    if dry_run:
        result = {
            "schema_version": TRACK_SCHEMA_VERSION,
            "stage": "trajectory_extraction_dry_run",
            "phase": phase,
            "selected_jobs": len(selected),
            "route_counts": route_counts,
            "dynamic_job_ids": dynamic_job_ids,
            "spatialtrackerv2_checkout": spatialtracker_checkout,
            "runtime_spatialtracker_manifest": str(runtime_manifest_path),
            "routing_jsonl": str(output_root / f"routing_{phase}.jsonl"),
        }
        _write_json(output_root / f"run_metadata_{phase}.json", result)
        return result

    static_payloads = []
    dynamic_ids = []
    for source in selected:
        job_id = str(source["job_id"])
        video_path = videos_dir / f"{job_id}.mp4"
        if route_by_id[job_id] == DYNAMIC_ROUTE:
            dynamic_ids.append(job_id)
        else:
            static_payloads.append(
                {
                    "job_id": job_id,
                    "video_path": video_path,
                    "calibration_root": calibration_root,
                    "output_root": output_root,
                    "camera_motion_evidence": evidence_by_id[job_id],
                    "overwrite": overwrite,
                    "assess_background_rigidity": assess_background_rigidity,
                }
            )

    worker_count = max(1, int(workers))
    if worker_count == 1:
        for ordinal, payload in enumerate(static_payloads, 1):
            job_id, result = _extract_static_job(payload)
            print(
                f"[static {ordinal}/{len(static_payloads)}] {job_id} status={result['status']} "
                f"valid2d={result['position_summary']['valid_2d_fraction']:.3f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_extract_static_job, payload): str(payload["job_id"])
                for payload in static_payloads
            }
            for ordinal, future in enumerate(as_completed(futures), 1):
                job_id, result = future.result()
                print(
                    f"[static {ordinal}/{len(static_payloads)}] {job_id} status={result['status']} "
                    f"valid2d={result['position_summary']['valid_2d_fraction']:.3f}",
                    flush=True,
                )

    if dynamic_ids and run_dynamic:
        command = [
            sys.executable,
            str(spatialtracker_script),
            "--manifest",
            str(runtime_manifest_path),
            "--output-root",
            str(output_root / "dynamic_native"),
            "--work-root",
            str(output_root / "dynamic_work"),
            "--frame-stride",
            "1",
        ]
        for job_id in dynamic_ids:
            command.extend(["--job-id", job_id])
        if overwrite:
            command.append("--rerun")
        if isolated_process:
            command.append("--isolated-process")
        subprocess.run(command, check=True, cwd=spatialtracker_script.parent.parent)
        for ordinal, job_id in enumerate(dynamic_ids, 1):
            result = _extract_dynamic_result(
                job_id=job_id,
                video_path=videos_dir / f"{job_id}.mp4",
                dynamic_dir=output_root / "dynamic_native" / job_id,
                output_root=output_root,
                camera_motion_evidence=evidence_by_id[job_id],
                calibration_sidecar=calibration_path(
                    calibration_root,
                    parse_video_job(videos_dir / f"{job_id}.mp4"),
                ),
            )
            print(
                f"[dynamic {ordinal}/{len(dynamic_ids)}] {job_id} status={result['status']} "
                f"valid3d={result['position_summary']['valid_3d_fraction']:.3f}",
                flush=True,
            )
    elif dynamic_ids:
        for job_id in dynamic_ids:
            job_dir = output_root / "jobs" / job_id
            if not (job_dir / "track_result.json").is_file():
                _write_json(
                    job_dir / "track_result.json",
                    {
                        "schema_version": TRACK_SCHEMA_VERSION,
                        "status": "pending_dynamic",
                        "job": parse_video_job(videos_dir / f"{job_id}.mp4"),
                        "reconstruction_route": DYNAMIC_ROUTE,
                        "target_parameters_used": False,
                        "trajectory_frames_csv": None,
                        "object_track_overlay": None,
                        "position_summary": _position_summary([]),
                        "trajectory_fit_eligible": False,
                        "camera_motion_evidence": evidence_by_id[job_id],
                    },
                )
    summary = _refresh_extraction_summary(output_root, manifest_rows)
    _write_json(
        output_root / f"run_metadata_{phase}.json",
        {
            "schema_version": TRACK_SCHEMA_VERSION,
            "stage": "trajectory_extraction",
            "phase": phase,
            "selected_jobs": len(selected),
            "route_counts": route_counts,
            "dynamic_job_ids": dynamic_job_ids,
            "dynamic_frame_stride": dynamic_frame_stride,
            "target_parameters_used": False,
            "spatialtrackerv2_checkout": spatialtracker_checkout,
            "paths": {
                "videos": str(videos_dir),
                "manifest": str(manifest_path),
                "camera_audit": str(audit_jsonl),
                "calibration_root": str(calibration_root),
                "output": str(output_root),
                "runtime_spatialtracker_manifest": str(runtime_manifest_path),
            },
        },
    )
    return summary


def _skipped_fit(experiment_id: str, spec: Mapping[str, Any], reason: str) -> dict[str, Any]:
    names = [str(item["name"]) for item in spec.get("hidden_parameters", [])]
    return {
        "experiment_id": experiment_id,
        "status": "skipped_trajectory_not_eligible",
        "reason": reason,
        "parameter_estimates": {name: None for name in names},
        "parameter_observed": {name: False for name in names},
        "diagnostics": {},
        "target_not_used_for_fit": True,
    }


def _not_scored(reason: str) -> dict[str, Any]:
    return {
        "status": "not_scored",
        "reason": reason,
        "fit_complete": False,
        "parameters": {},
        "experiment_nmae": None,
        "experiment_score_0_100": None,
    }


def _assess_frozen_trajectory_eligibility(
    validity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the established partial-track policy without touching a video."""

    status = str(validity.get("status", "indeterminate")).lower()
    if status == "pass" and validity.get("fit_eligible") is True:
        return {
            "eligible": True,
            "status": "full_generation_validity",
            "reason": "generation_validity_pass",
        }
    if status in {"fail", "failed"} or validity.get("failure_codes"):
        return {"eligible": False, "status": "blocked", "reason": "generation_validity_fail"}
    indeterminate_codes = set(str(value) for value in validity.get("indeterminate_codes", []))
    if not indeterminate_codes or not indeterminate_codes.issubset(
        {"insufficient_reliable_object_tracking"}
    ):
        return {
            "eligible": False,
            "status": "blocked",
            "reason": "indeterminate_for_more_than_late_tracking_coverage",
            "blocking_codes": sorted(indeterminate_codes),
        }
    checks = validity.get("checks", {})
    identity = checks.get("object_identity", {})
    safety_checks = {
        "trusted_first_frame": identity.get("trusted_first_frame") is True,
        "camera_motion": checks.get("camera_motion", {}).get("status") == "pass",
        "object_shape_2d": checks.get("object_shape_2d", {}).get("status") == "pass",
        "object_scale_2d": checks.get("object_scale_2d", {}).get("status") != "fail",
        "scene_rigidity_2d": checks.get("scene_rigidity_2d", {}).get("status") == "pass",
    }
    if not all(safety_checks.values()):
        return {
            "eligible": False,
            "status": "blocked",
            "reason": "partial_track_safety_checks_failed",
            "safety_checks": safety_checks,
        }
    trusted = sorted(
        int(row.get("source_frame_index", row.get("frame_index", index)))
        for index, row in enumerate(rows)
        if _truth(row.get("fit_eligible"))
        and _truth(row.get("measurement_valid"))
        and not _truth(row.get("interpolated"))
    )
    runs: list[list[int]] = []
    for frame in trusted:
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    anchored = next((run for run in runs if run and run[0] == 0), [])
    required = max(12, int(math.ceil(max(len(rows), 1) * 0.10)))
    if len(anchored) < required:
        return {
            "eligible": False,
            "status": "blocked",
            "reason": "insufficient_contiguous_metric_anchor_segment",
            "trusted_segment_frame_count": len(anchored),
            "required_frame_count": required,
        }
    return {
        "eligible": True,
        "status": "partial_verified_trajectory",
        "reason": "only_late_tracking_coverage_is_unresolved",
        "trusted_segment_start_frame": int(anchored[0]),
        "trusted_segment_end_frame": int(anchored[-1]),
        "trusted_segment_frame_count": len(anchored),
        "required_frame_count": required,
        "safety_checks": safety_checks,
    }


def evaluate_extracted_seedance978(
    *,
    extraction_root: Path,
    manifest_path: Path,
    registry_path: Path,
    output_root: Path,
    phase: str = "all",
    job_ids: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit and score physics using frozen CSV artifacts only."""

    manifest_rows = _ordered_manifest(_read_jsonl(manifest_path))
    selected = _phase_rows(manifest_rows, phase)
    requested = list(dict.fromkeys(str(value) for value in (job_ids or [])))
    if requested:
        known = {str(row["job_id"]) for row in selected}
        unknown = sorted(set(requested) - known)
        if unknown:
            raise ValueError(f"unknown/out-of-phase job ids: {unknown}")
        selected = [row for row in selected if str(row["job_id"]) in set(requested)]
    registry = load_experiment_registry(registry_path)
    evaluator_source_sha256 = _sha256(Path(__file__).resolve())
    physics_source_file = Path(
        str(sys.modules[fit_physics_parameters.__module__].__file__)
    ).resolve()
    physics_fitter_source_sha256 = _sha256(physics_source_file)
    dynamic_gate_source_file = Path(
        str(sys.modules[assess_dynamic_3d_trajectory.__module__].__file__)
    ).resolve()
    dynamic_3d_gate_source_sha256 = _sha256(dynamic_gate_source_file)
    registry_sha256 = _sha256(registry_path)
    for ordinal, source in enumerate(selected, 1):
        job_id = str(source["job_id"])
        extraction_path = extraction_root / "jobs" / job_id / "track_result.json"
        if not extraction_path.is_file():
            raise FileNotFoundError(f"missing extraction artifact: {extraction_path}")
        destination = output_root / "jobs" / job_id / "result.json"
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        trajectory_path = extraction_path.parent / "trajectory_frames.csv"
        if not trajectory_path.is_file() and extraction.get("trajectory_frames_csv"):
            trajectory_path = Path(str(extraction["trajectory_frames_csv"]))
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        lineage = {
            "trajectory_sha256": _sha256(trajectory_path),
            "track_result_sha256": _sha256(extraction_path),
            "registry_sha256": registry_sha256,
            "evaluator_source_sha256": evaluator_source_sha256,
            "physics_fitter_source_sha256": physics_fitter_source_sha256,
            "dynamic_3d_gate_source_sha256": dynamic_3d_gate_source_sha256,
            "evaluator_version": EVALUATOR_VERSION,
        }
        if destination.is_file() and not overwrite:
            try:
                cached = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if cached.get("evaluation_lineage") == lineage:
                portable_trajectory_path = destination.parent / "trajectory_frames.csv"
                portable_valid = bool(
                    portable_trajectory_path.is_file()
                    and _sha256(portable_trajectory_path) == lineage["trajectory_sha256"]
                )
                if not portable_valid:
                    portable_trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                    if trajectory_path.resolve() != portable_trajectory_path.resolve():
                        shutil.copyfile(trajectory_path, portable_trajectory_path)
                    cached["trajectory_csv"] = str(portable_trajectory_path)
                    _write_json(destination, cached)
                print(f"[{ordinal}/{len(selected)}] SKIP {job_id} lineage=unchanged", flush=True)
                continue
            print(f"[{ordinal}/{len(selected)}] REEVALUATE {job_id} lineage=changed", flush=True)
        rows = _read_typed_csv(trajectory_path)
        fit_rows: list[dict[str, Any]] = []
        for row in rows:
            eligible = (
                _truth(row.get("fit_eligible"))
                and _finite_fit_xyz(row)
                and not _truth(row.get("interpolated"))
            )
            converted = dict(row)
            converted["frame_index"] = int(row.get("source_frame_index", row.get("frame_index", 0)))
            converted["display_x_m"] = row.get("x_m")
            converted["display_y_m"] = row.get("y_m")
            converted["display_z_m"] = row.get("z_m")
            if eligible:
                converted["x_m"] = _float(row.get("fit_x_m"))
                converted["y_m"] = _float(row.get("fit_y_m"))
                converted["z_m"] = _float(row.get("fit_z_m"))
            converted["measurement_valid"] = eligible
            converted["physics_fit_used"] = eligible
            fit_rows.append(converted)
        experiment_id = str(extraction["job"]["experiment_id"])
        spec = registry["experiments_by_id"][experiment_id]
        validity = extraction.get("video_generation_validity", {})
        route = str(extraction.get("reconstruction_route"))
        fit_measurement_count = sum(_truth(row.get("physics_fit_used")) for row in fit_rows)
        track_quality_ok = bool(
            extraction.get("trajectory_fit_eligible")
            and fit_measurement_count >= 3
        )
        dynamic_3d_inclusion = None
        if route == STATIC_ROUTE:
            validity_eligibility = _assess_frozen_trajectory_eligibility(validity, fit_rows)
        else:
            validity_status = str(validity.get("status", "indeterminate")).lower()
            dynamic_3d_inclusion = assess_dynamic_3d_trajectory(
                experiment_id,
                rows,
                reconstruction_metadata=extraction,
            )
            hard_validity_failure = bool(
                validity_status in {"fail", "failed"}
                or validity.get("failure_codes")
            )
            dynamic_evidence_usable = dynamic_3d_inclusion.get("decision") == "include"
            validity_eligibility = {
                # Moving-camera rigidity may remain formally indeterminate in
                # the 2-D validity gate.  A target-independent metric 3-D
                # manifold check is the appropriate replacement evidence.
                "eligible": bool(not hard_validity_failure and dynamic_evidence_usable),
                "status": (
                    "qualified_dynamic_3d"
                    if not hard_validity_failure and dynamic_evidence_usable
                    else "blocked"
                ),
                "reason": (
                    "dynamic_3d_motion_manifold_pass"
                    if not hard_validity_failure and dynamic_evidence_usable
                    else "dynamic_generation_validity_fail"
                    if hard_validity_failure
                    else "dynamic_3d_measurement_x"
                ),
                "original_generation_validity_status": validity_status,
                "dynamic_3d_inclusion": dynamic_3d_inclusion,
            }
        eligible = bool(track_quality_ok and validity_eligibility.get("eligible"))
        target_lookup_performed = False
        fit_attempted = False
        if eligible:
            fit_attempted = True
            fit = fit_physics_parameters(experiment_id, fit_rows)
            if route == DYNAMIC_ROUTE:
                dynamic_3d_inclusion = finalize_dynamic_3d_inclusion(
                    dynamic_3d_inclusion or {},
                    fit,
                )
                eligible = dynamic_3d_inclusion.get("decision") == "include"
                validity_eligibility.update(
                    {
                        "eligible": eligible,
                        "status": "qualified_dynamic_3d" if eligible else "blocked",
                        "reason": (
                            "dynamic_3d_motion_manifold_and_target_free_fit_pass"
                            if eligible
                            else "dynamic_3d_target_free_fit_x"
                        ),
                        "dynamic_3d_inclusion": dynamic_3d_inclusion,
                    }
                )
            if eligible:
                target = lookup_target_tuple(spec, str(extraction["job"]["parameter_tuple_id"]))
                target_lookup_performed = True
                metrics = score_parameter_fit(fit, spec, target)
                reason = str(validity_eligibility.get("reason") or "trajectory_fit_completed")
            else:
                reason = str(validity_eligibility.get("reason") or "dynamic_3d_target_free_fit_x")
                metrics = _not_scored(reason)
        else:
            reason = (
                str(validity_eligibility.get("reason"))
                if not validity_eligibility.get("eligible")
                else "extracted_trajectory_not_fit_eligible"
            )
            fit = _skipped_fit(experiment_id, spec, reason)
            metrics = _not_scored(reason)
        job_dir = destination.parent
        job_dir.mkdir(parents=True, exist_ok=True)
        # Keep a portable frozen copy beside result.json.  Reports can then be
        # rebuilt after moving the evaluation directory without depending on
        # the original extraction-root absolute path.
        portable_trajectory_path = job_dir / "trajectory_frames.csv"
        if trajectory_path.resolve() != portable_trajectory_path.resolve():
            shutil.copyfile(trajectory_path, portable_trajectory_path)
        _write_rows(job_dir / "fit_frames.csv", [row for row in fit_rows if row["physics_fit_used"]])
        plot = None
        if any(_finite_xyz(row) for row in fit_rows):
            try:
                plot = write_trajectory_plot(job_dir, fit_rows, title=job_id, fit=fit)
            except Exception:
                plot = None
        result = {
            "schema_version": "2.1.0",
            "status": "succeeded" if eligible else "trajectory_not_eligible",
            "job": extraction["job"],
            "benchmark_split": extraction.get("benchmark_split", benchmark_split(extraction["job"])),
            "reconstruction_route": extraction.get("reconstruction_route"),
            "source_extraction": str(extraction_path),
            "source_trajectory": str(trajectory_path),
            "source_trajectory_sha256": lineage["trajectory_sha256"],
            "evaluation_lineage": lineage,
            "evaluation_reads_video": False,
            "evaluation_invokes_tracker": False,
            "video_generation_validity": validity,
            "trajectory_fit_eligibility": {
                "eligible": eligible,
                "status": validity_eligibility.get("status", "blocked") if track_quality_ok else "blocked",
                "reason": reason,
                "track_quality_ok": track_quality_ok,
                "fit_measurement_count": fit_measurement_count,
                "validity_policy": validity_eligibility,
            },
            "dynamic_3d_inclusion": dynamic_3d_inclusion,
            "fit_attempted": fit_attempted,
            "fit": fit,
            "metrics": metrics,
            "target_lookup_performed": target_lookup_performed,
            "pipeline": extraction.get("pipeline", {}),
            "visual_evidence": {"trajectory_plot": plot},
            "camera_motion_evidence": extraction.get("camera_motion_evidence", {}),
            "tracking_csv": extraction.get("native_tracking_csv"),
            "trajectory_csv": str(portable_trajectory_path),
            "error": extraction.get("error"),
        }
        _write_json(destination, result)
        print(
            f"[{ordinal}/{len(selected)}] {job_id} eligible={eligible} fit={fit.get('status')}",
            flush=True,
        )
    aggregate = write_seedance978_reports(output_root, manifest_rows, registry)
    _write_json(
        output_root / f"evaluation_metadata_{phase}.json",
        {
            "schema_version": "1.0.0",
            "stage": "trajectory_only_physics_evaluation",
            "phase": phase,
            "selected_jobs": len(selected),
            "evaluation_reads_video": False,
            "evaluation_invokes_tracker": False,
            "extraction_root": str(extraction_root),
            "manifest": str(manifest_path),
            "registry": str(registry_path),
            "output": str(output_root),
        },
    )
    return aggregate
