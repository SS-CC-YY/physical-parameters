"""End-to-end fixed-camera trajectory and parameter benchmark runner."""

from __future__ import annotations

import csv
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import cv2

from .fixed_camera_ball import (
    calibration_path,
    load_calibration,
    parse_video_job,
    reconstruct_metric_trajectory,
    run_ball_tracking,
    write_trajectory_csv,
)
from .generation_validity import evaluate_generation_validity
from .scene_rigidity import assess_static_scene_rigidity
from .physics_parameters import (
    fit_physics_parameters,
    lookup_target_tuple,
    score_parameter_fit,
)


DEFAULT_BENCHMARK_SEED = 341867882
STATIC_CAMERA_CATEGORIES = {"fixed", "no_significant_camera_change"}


def load_experiment_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    experiments = registry.get("experiments", [])
    registry["experiments_by_id"] = {str(item["id"]): item for item in experiments}
    return registry


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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _skipped_fit(experiment_id: str, spec: Mapping[str, Any], reason: str) -> dict[str, Any]:
    names = [str(item["name"]) for item in spec.get("hidden_parameters", [])]
    return {
        "experiment_id": experiment_id,
        "status": "skipped_video_generation_validity",
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


def benchmark_split(job: Mapping[str, Any]) -> str:
    camera = str(job["camera_name"])
    seed = int(job["seed"])
    if camera == "CAM_Side" and seed == DEFAULT_BENCHMARK_SEED:
        return "side_primary"
    if camera == "CAM_Side":
        return "side_seed_stability_extra"
    if camera == "CAM_Main":
        return "main_robustness"
    if camera == "CAM_Top":
        return "top_robustness"
    return "other"


def _camera_category(evidence: Mapping[str, Any] | None) -> str | None:
    if not evidence:
        return None
    value = evidence.get("final_category") or evidence.get("decision")
    return None if value is None else str(value)


def _load_dynamic_trajectory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            values: dict[str, Any] = dict(source)
            frame_index = source.get("source_frame") or source.get("frame_index")
            values["frame_index"] = int(frame_index or len(rows))
            for name in ("time_s", "x_m", "y_m", "z_m"):
                text = source.get(name)
                values[name] = None if text in {None, ""} else float(text)
            finite = all(
                values[name] is not None and math.isfinite(float(values[name]))
                for name in ("x_m", "y_m", "z_m")
            )
            values["measurement_valid"] = finite
            values["physics_fit_used"] = finite
            rows.append(values)
    return rows


def _dynamic_payload(dynamic_result_dir: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if dynamic_result_dir is None:
        return [], None
    result_path = dynamic_result_dir / "result.json"
    trajectory_path = dynamic_result_dir / "trajectory_world.csv"
    if not result_path.is_file() or not trajectory_path.is_file():
        return [], None
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "succeeded":
        return [], result
    return _load_dynamic_trajectory(trajectory_path), result


def _draw_track_frame(
    frame: np.ndarray,
    row: Mapping[str, Any] | None,
    history_segments: Sequence[Sequence[tuple[int, int]]],
    status: str,
    failure_codes: Sequence[str],
) -> np.ndarray:
    output = frame.copy()
    colour = (0, 190, 0) if status == "pass" else (0, 0, 230) if status == "fail" else (0, 190, 255)
    for history in history_segments:
        for start, end in zip(history[:-1], history[1:]):
            cv2.line(output, start, end, colour, 2, cv2.LINE_AA)
    trusted_measurement = _is_trusted_measurement(row)
    if trusted_measurement and row is not None:
        center = (int(round(float(row["center_u_px"]))), int(round(float(row["center_v_px"]))))
        raw_radius = max(3, int(round(float(row["measurement_radius_px"]))))
        radius_value = row.get("display_radius_px", row.get("measurement_radius_px"))
        radius = max(3, int(round(float(radius_value))))
        # Solid: temporally stable display/physical-size prior.  Dashed cyan:
        # the raw radius that metric reconstruction actually consumes.  Both
        # are shown so the overlay remains an honest audit artifact.
        cv2.circle(output, center, radius, colour, 2, cv2.LINE_AA)
        for start_angle in range(0, 360, 30):
            cv2.ellipse(
                output,
                center,
                (raw_radius, raw_radius),
                0.0,
                float(start_angle),
                float(start_angle + 15),
                (255, 190, 0),
                1,
                cv2.LINE_AA,
            )
        major = row.get("ellipse_major_axis_px")
        minor = row.get("ellipse_minor_axis_px")
        angle = row.get("ellipse_angle_deg")
        if (
            row.get("shape_evidence_available") is not False
            and major is not None
            and minor is not None
        ):
            cv2.ellipse(
                output,
                center,
                (max(2, int(round(float(major) / 2.0))), max(2, int(round(float(minor) / 2.0)))),
                float(angle or 0.0),
                0.0,
                360.0,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.circle(output, center, 3, colour, -1, cv2.LINE_AA)
    label = f"generation_validity={status}"
    if failure_codes:
        label += " | " + ",".join(str(value) for value in failure_codes[:2])
    cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    if row is not None:
        observation = str(
            row.get(
                "observation_status",
                "measured" if row.get("found") is True else "missing",
            )
        )
        identity = "verified" if trusted_measurement else "unverified"
        confidence = row.get("track_confidence")
        confidence_text = "n/a" if confidence is None else f"{float(confidence):.3f}"
        raw_text = row.get("measurement_radius_px")
        display_text = row.get("display_radius_px", raw_text)
        radius_text = (
            "n/a"
            if raw_text is None or display_text is None
            else f"{float(raw_text):.1f}/{float(display_text):.1f}px"
        )
        source = str(row.get("measurement_source") or "none")
        shape_source = str(row.get("shape_evidence_source") or "unresolved")
        tracking_label = (
            f"track={identity}/{observation} | source={source} | shape={shape_source}"
        )
        measurement_label = (
            f"confidence={confidence_text} | radius raw/reference={radius_text}"
        )
        for text, y in ((tracking_label, 51), (measurement_label, 73)):
            cv2.putText(
                output,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return output


def _is_trusted_measurement(row: Mapping[str, Any] | None) -> bool:
    if not row or row.get("found") is not True:
        return False
    observation = str(row.get("observation_status", "measured")).lower()
    return (
        observation == "measured"
        and row.get("measurement_valid") is not False
        and row.get("identity_verified") is not False
    )


def write_validity_visuals(
    output_dir: Path,
    frames: Sequence[np.ndarray],
    track_rows: Sequence[Mapping[str, Any]],
    validity: Mapping[str, Any],
    fps: float,
) -> dict[str, str | None]:
    if not frames:
        return {"overlay_video": None, "evidence_contact_sheet": None}
    output_dir.mkdir(parents=True, exist_ok=True)
    status = str(validity.get("status", "indeterminate"))
    failures = [str(item) for item in validity.get("failure_codes", [])]
    overlay_path = output_dir / "validity_object_track_overlay.mp4"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (width, height),
    )
    history_segments: list[list[tuple[int, int]]] = []
    active_history: list[tuple[int, int]] | None = None
    rendered: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        row = track_rows[index] if index < len(track_rows) else None
        if _is_trusted_measurement(row) and row is not None:
            if active_history is None:
                active_history = []
                history_segments.append(active_history)
            active_history.append(
                (
                    int(round(float(row["center_u_px"]))),
                    int(round(float(row["center_v_px"]))),
                )
            )
        else:
            # Keep prior verified segments visible, but never draw a line over
            # an unresolved gap: that line would look like measured motion.
            active_history = None
        rendered_frame = _draw_track_frame(
            frame,
            row,
            history_segments,
            status,
            failures,
        )
        if writer.isOpened():
            writer.write(rendered_frame)
        rendered.append(rendered_frame)
    writer.release()
    if not overlay_path.is_file():
        overlay_value: str | None = None
    else:
        overlay_value = str(overlay_path)

    evidence_frames = validity.get("offending_frames", {})
    preferred_keys = {
        "scene_cut",
        "object_shape_2d",
        "object_scale_2d",
        "object_rigidity_3d",
        "scene_rigidity_2d",
        "scene_rigidity_3d",
    }
    preferred_values = [
        values for key, values in evidence_frames.items() if key in preferred_keys
    ]
    selected_values = preferred_values or list(evidence_frames.values())
    offending = sorted(
        {
            int(frame)
            for values in selected_values
            for frame in values
            if isinstance(frame, (int, float)) and int(frame) >= 0
        }
    )
    if not offending:
        offending = [int(round(value)) for value in np.linspace(0, len(frames) - 1, min(6, len(frames)))]
    bounded = sorted(set(max(0, min(len(frames) - 1, value)) for value in offending))
    if len(bounded) > 6:
        positions = np.linspace(0, len(bounded) - 1, 6).round().astype(int)
        selected = [bounded[int(position)] for position in positions]
    else:
        selected = bounded
    thumbs: list[np.ndarray] = []
    thumb_width = 432
    for index in selected:
        image = rendered[index]
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(image.shape[0] * scale)))))
        cv2.putText(thumb, f"frame {index}", (10, thumb.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(thumb, f"frame {index}", (10, thumb.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    if thumbs:
        rows = []
        blank = np.zeros_like(thumbs[0])
        while len(thumbs) % 3:
            thumbs.append(blank.copy())
        for offset in range(0, len(thumbs), 3):
            rows.append(np.concatenate(thumbs[offset : offset + 3], axis=1))
        sheet = np.concatenate(rows, axis=0)
        sheet_path = output_dir / "validity_evidence_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheet_value: str | None = str(sheet_path)
    else:
        sheet_value = None
    return {"overlay_video": overlay_value, "evidence_contact_sheet": sheet_value}


def _plot_polyline(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    x_values: np.ndarray,
    y_values: np.ndarray,
    colour: tuple[int, int, int],
) -> None:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x = x_values[finite]
    y = y_values[finite]
    if len(x) < 1:
        return
    left, top, right, bottom = rect
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    if x_max - x_min < 1e-9:
        x_min -= 0.5
        x_max += 0.5
    if y_max - y_min < 1e-9:
        y_min -= 0.5
        y_max += 0.5
    padding_x = 0.06 * (x_max - x_min)
    padding_y = 0.06 * (y_max - y_min)
    x_min -= padding_x
    x_max += padding_x
    y_min -= padding_y
    y_max += padding_y
    pixels = np.column_stack(
        [
            left + (x - x_min) / (x_max - x_min) * (right - left),
            bottom - (y - y_min) / (y_max - y_min) * (bottom - top),
        ]
    ).round().astype(np.int32)
    for start, end in zip(pixels[:-1], pixels[1:]):
        cv2.line(canvas, tuple(start), tuple(end), colour, 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pixels[0]), 5, (0, 170, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pixels[-1]), 5, (200, 80, 0), -1, cv2.LINE_AA)


def write_trajectory_plot(
    output_dir: Path,
    trajectory: Sequence[Mapping[str, Any]],
    *,
    title: str,
    fit: Mapping[str, Any] | None = None,
) -> str | None:
    usable = [
        row
        for row in trajectory
        if row.get("measurement_valid") is not False
        if all(row.get(name) is not None for name in ("time_s", "x_m", "y_m", "z_m"))
    ]
    if not usable:
        return None
    time = np.asarray([float(row["time_s"]) for row in usable], dtype=float)
    x = np.asarray([float(row["x_m"]) for row in usable], dtype=float)
    y = np.asarray([float(row["y_m"]) for row in usable], dtype=float)
    z = np.asarray([float(row["z_m"]) for row in usable], dtype=float)
    canvas = np.full((650, 1400, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (35, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    if fit:
        estimates = fit.get("parameter_estimates", {})
        text = "  ".join(
            f"{name}={float(value):.5g}" for name, value in estimates.items() if value is not None
        )
        if text:
            cv2.putText(canvas, text, (35, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1, cv2.LINE_AA)
    left_rect = (70, 105, 650, 585)
    right_rect = (760, 105, 1335, 585)
    for rect in (left_rect, right_rect):
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (80, 80, 80), 1)
    cv2.putText(canvas, "World trajectory: x-z plane", (210, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Coordinates over decoded-video time", (875, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
    _plot_polyline(canvas, left_rect, x, z, (30, 50, 210))
    # Share one time/value scale so component magnitudes are directly comparable.
    values = np.concatenate([x, y, z])
    t_stack = np.concatenate([time, time, time])
    finite = np.isfinite(values) & np.isfinite(t_stack)
    if finite.any():
        t_min, t_max = float(t_stack[finite].min()), float(t_stack[finite].max())
        v_min, v_max = float(values[finite].min()), float(values[finite].max())
        if t_max - t_min < 1e-9:
            t_max = t_min + 1.0
        if v_max - v_min < 1e-9:
            v_min -= 0.5
            v_max += 0.5
        for component, colour in ((x, (210, 80, 40)), (y, (40, 160, 40)), (z, (30, 50, 210))):
            px = right_rect[0] + (time - t_min) / (t_max - t_min) * (right_rect[2] - right_rect[0])
            py = right_rect[3] - (component - v_min) / (v_max - v_min) * (right_rect[3] - right_rect[1])
            points = np.column_stack([px, py]).round().astype(np.int32)
            for start, end in zip(points[:-1], points[1:]):
                cv2.line(canvas, tuple(start), tuple(end), colour, 2, cv2.LINE_AA)
    cv2.putText(canvas, "x", (1120, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 80, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "y", (1170, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 160, 40), 2, cv2.LINE_AA)
    cv2.putText(canvas, "z", (1220, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 50, 210), 2, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trajectory_plot.png"
    return str(path) if cv2.imwrite(str(path), canvas) else None


def run_physics_job(
    video_path: Path,
    *,
    calibration_root: Path,
    registry: Mapping[str, Any],
    output_root: Path,
    overwrite: bool = False,
    camera_motion_evidence: Mapping[str, Any] | None = None,
    reconstruction_route: str = "calibrated_static_sphere",
    dynamic_result_dir: Path | None = None,
    assess_background_rigidity: bool = False,
    make_overlay: bool = False,
) -> dict[str, Any]:
    job = parse_video_job(video_path)
    if camera_motion_evidence is None and reconstruction_route == "calibrated_static_sphere":
        camera_motion_evidence = {
            "filename": video_path.name,
            "final_category": "no_significant_camera_change",
            "status": "assumed_by_fixed_camera_runner",
            "measurement_source": "explicit_fixed_camera_runner_assumption",
        }
    experiment_id = str(job["experiment_id"])
    spec = registry["experiments_by_id"].get(experiment_id)
    if spec is None:
        raise KeyError(f"experiment not in registry: {experiment_id}")
    if job["object_id"] != "standard_ball":
        raise ValueError("fixed-camera sphere evaluator currently requires object_id=standard_ball")
    job_dir = output_root / video_path.stem
    result_path = job_dir / "result.json"
    if result_path.is_file() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    calibration_sidecar = calibration_path(calibration_root, job)
    trajectory_path = job_dir / "trajectory.csv"
    tracking_path = job_dir / "tracking.csv"
    pipeline: dict[str, Any] = {}
    trajectory: Sequence[Mapping[str, Any]] = []
    track_rows: Sequence[Mapping[str, Any]] = []
    frames: Sequence[np.ndarray] = []
    validity: dict[str, Any] = {
        "status": "indeterminate",
        "fit_eligible": False,
        "failure_codes": [],
        "warning_codes": ["evaluation_not_started"],
    }
    error: dict[str, Any] | None = None
    target_lookup_performed = False
    visuals: dict[str, Any] = {
        "overlay_video": None,
        "evidence_contact_sheet": None,
        "trajectory_plot": None,
    }
    try:
        calibration = load_calibration(calibration_sidecar)
        frames, track_rows, pipeline = run_ball_tracking(video_path, calibration)
        write_trajectory_csv(tracking_path, track_rows)

        static_scene_evidence: dict[str, Any] | None = None
        if reconstruction_route == "calibrated_static_sphere" and assess_background_rigidity:
            static_scene_evidence = assess_static_scene_rigidity(
                frames,
                track_rows,
                camera_motion_evidence=camera_motion_evidence,
                calibration=calibration,
            )
            pipeline["static_scene_rigidity"] = static_scene_evidence

        if reconstruction_route == "calibrated_static_sphere":
            # The cheap gate runs before metric lifting.  A clear generation
            # failure does not need a trajectory or a physics fit.
            validity = evaluate_generation_validity(
                track_rows,
                camera_motion_evidence=camera_motion_evidence,
                static_scene_rigidity_evidence=static_scene_evidence,
            )
            if validity["fit_eligible"]:
                video = pipeline["video"]
                trajectory, geometry_summary = reconstruct_metric_trajectory(
                    track_rows,
                    calibration,
                    experiment_id=experiment_id,
                    video_size=(int(video["width"]), int(video["height"])),
                    fps=float(video["fps"]),
                )
                pipeline["geometry"] = geometry_summary
                write_trajectory_csv(trajectory_path, trajectory)
                validity = evaluate_generation_validity(
                    track_rows,
                    camera_motion_evidence=camera_motion_evidence,
                    static_scene_rigidity_evidence=static_scene_evidence,
                    trajectory_evidence=trajectory,
                )
        elif reconstruction_route == "spatialtrackerv2_dynamic":
            trajectory, dynamic_result = _dynamic_payload(dynamic_result_dir)
            pipeline["dynamic_reconstruction"] = dynamic_result
            rigidity = None if dynamic_result is None else dynamic_result.get("rigidity_evidence")
            validity = evaluate_generation_validity(
                track_rows,
                camera_motion_evidence=camera_motion_evidence,
                trajectory_evidence=trajectory,
                dynamic_rigidity_evidence=rigidity,
            )
            if dynamic_result is not None:
                native_validity = dynamic_result.get("generation_validity", {})
                native_status = str(native_validity.get("status", "")).lower()
                if native_status in {"failed", "fail"} and validity.get("status") != "fail":
                    validity["status"] = "fail"
                    validity["fit_eligible"] = False
                    validity.setdefault("failure_codes", []).append(
                        "dynamic_reconstruction_confirmed_generation_failure"
                    )
                elif (
                    native_status in {"review", "indeterminate"}
                    or dynamic_result.get("quality_pass") is not True
                ) and validity.get("status") == "pass":
                    validity["status"] = "indeterminate"
                    validity["fit_eligible"] = False
                    validity.setdefault("warning_codes", []).append(
                        "dynamic_reconstruction_quality_insufficient"
                    )
                    validity.setdefault("indeterminate_codes", []).append(
                        "dynamic_reconstruction_quality_insufficient"
                    )
            if trajectory:
                write_trajectory_csv(trajectory_path, trajectory)
        else:
            raise ValueError(f"unknown reconstruction route: {reconstruction_route}")

        if validity.get("fit_eligible") and trajectory:
            fit = fit_physics_parameters(experiment_id, trajectory)
            # Ground truth is intentionally looked up only after reconstruction + fit.
            target = lookup_target_tuple(spec, str(job["parameter_tuple_id"]))
            target_lookup_performed = True
            metrics = score_parameter_fit(fit, spec, target)
        else:
            reason = (
                f"video_generation_validity_{validity.get('status', 'indeterminate')}"
                if validity
                else "video_generation_validity_unavailable"
            )
            fit = _skipped_fit(experiment_id, spec, reason)
            metrics = _not_scored(reason)
    except Exception as exception:  # Per-video failure is part of benchmark coverage.
        reason = f"{type(exception).__name__}: {exception}"
        fit = _skipped_fit(experiment_id, spec, f"evaluation_indeterminate: {reason}")
        metrics = _not_scored(f"evaluation_indeterminate: {reason}")
        validity = {
            "schema_version": "1.0.0",
            "status": "indeterminate",
            "fit_eligible": False,
            "failure_codes": [],
            "warning_codes": ["evaluation_pipeline_error"],
            "indeterminate_codes": ["evaluation_pipeline_error"],
            "offending_frames": {},
        }
        error = {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
        }
    if trajectory:
        try:
            visuals["trajectory_plot"] = write_trajectory_plot(
                job_dir,
                trajectory,
                title=video_path.stem,
                fit=fit,
            )
        except Exception as plot_error:
            visuals["trajectory_plot_error"] = f"{type(plot_error).__name__}: {plot_error}"
    if frames and (make_overlay or validity.get("status") != "pass"):
        try:
            visuals.update(
                write_validity_visuals(
                    job_dir,
                    frames,
                    track_rows,
                    validity,
                    float(pipeline.get("video", {}).get("fps") or 16.0),
                )
            )
        except Exception as visual_error:  # Visual evidence must not change the metric decision.
            visuals["error"] = f"{type(visual_error).__name__}: {visual_error}"
    status = (
        "generation_validity_failed"
        if validity.get("status") == "fail"
        else "generation_validity_indeterminate"
        if validity.get("status") == "indeterminate"
        else "succeeded"
    )
    result = {
        "schema_version": "2.0.0",
        "status": status,
        "job": job,
        "benchmark_split": benchmark_split(job),
        "reconstruction_route": reconstruction_route,
        "assumptions": {
            "camera_pose_fixed": reconstruction_route == "calibrated_static_sphere",
            "uses_first_frame_gt_ball_size": True,
            "uses_camera_intrinsics_extrinsics": True,
            "target_parameters_not_used_for_tracking_or_fit": True,
            "evaluated_object": "standard_ball",
            "independent_generated_views_not_used_as_synchronized_multiview": True,
        },
        "calibration_path": str(calibration_sidecar),
        "tracking_csv": str(tracking_path) if track_rows else None,
        "trajectory_csv": str(trajectory_path) if trajectory else None,
        "pipeline": pipeline,
        "video_generation_validity": validity,
        "fit_attempted": bool(validity.get("fit_eligible") and trajectory),
        "fit": fit,
        "metrics": metrics,
        "target_lookup_performed": target_lookup_performed,
        "visual_evidence": visuals,
        "camera_motion_evidence": dict(camera_motion_evidence or {}),
        "error": error,
    }
    _write_json(result_path, result)
    return result


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    job = result["job"]
    fit = result["fit"]
    metrics = result["metrics"]
    tracking = result.get("pipeline", {}).get("tracking", {})
    validity = result.get("video_generation_validity", {})
    row: dict[str, Any] = {
        "video_name": job["video_name"],
        "experiment_id": job["experiment_id"],
        "parameter_tuple_id": job["parameter_tuple_id"],
        "scene_id": job["scene_id"],
        "camera_name": job["camera_name"],
        "seed": job["seed"],
        "benchmark_split": result.get("benchmark_split") or benchmark_split(job),
        "reconstruction_route": result.get("reconstruction_route"),
        "generation_validity_status": validity.get("status"),
        "generation_failure_codes": ";".join(validity.get("failure_codes", [])),
        "generation_warning_codes": ";".join(validity.get("warning_codes", [])),
        "fit_attempted": result.get("fit_attempted"),
        "fit_status": fit.get("status"),
        "fit_complete": metrics.get("fit_complete"),
        "tracked_fraction": tracking.get("tracked_fraction"),
        "experiment_nmae": metrics.get("experiment_nmae"),
        "experiment_score_0_100": metrics.get("experiment_score_0_100"),
        "error": None if result.get("error") is None else result["error"].get("message"),
    }
    for name, item in metrics.get("parameters", {}).items():
        row[f"{name}__gt"] = item.get("gt")
        row[f"{name}__estimate"] = item.get("estimate_raw")
        row[f"{name}__nae"] = item.get("normalized_absolute_error")
        row[f"{name}__score"] = item.get("score_0_100")
    return row


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result["job"]["experiment_id"])].append(result)
    by_experiment: dict[str, Any] = {}
    experiment_nmaes: list[float] = []
    for experiment_id, items in sorted(groups.items()):
        scored = [
            item
            for item in items
            if item.get("metrics", {}).get("experiment_nmae") is not None
        ]
        nmaes = [float(item["metrics"]["experiment_nmae"]) for item in scored]
        scores = [float(item["metrics"]["experiment_score_0_100"]) for item in scored]
        tracked = [
            float(item.get("pipeline", {}).get("tracking", {}).get("tracked_fraction", 0.0))
            for item in items
        ]
        validity_counts: dict[str, int] = defaultdict(int)
        for item in items:
            validity_counts[
                str(item.get("video_generation_validity", {}).get("status", "missing"))
            ] += 1
        experiment_mean = float(np.mean(nmaes)) if nmaes else None
        if experiment_mean is not None:
            experiment_nmaes.append(experiment_mean)
        by_experiment[experiment_id] = {
            "job_count": len(items),
            "generation_validity_counts": dict(validity_counts),
            "generation_valid_rate": validity_counts.get("pass", 0) / max(len(items), 1),
            "fit_attempted_count": int(sum(bool(item.get("fit_attempted")) for item in items)),
            "complete_fit_count": int(sum(bool(item["metrics"]["fit_complete"]) for item in scored)),
            "complete_fit_rate_conditional_on_generation_valid": (
                None
                if not scored
                else float(np.mean([bool(item["metrics"]["fit_complete"]) for item in scored]))
            ),
            "mean_tracking_fraction": float(np.mean(tracked)),
            "mean_experiment_nmae_conditional_on_generation_valid": experiment_mean,
            "mean_experiment_score_0_100_conditional_on_generation_valid": (
                float(np.mean(scores)) if scores else None
            ),
        }
    macro_nmae = float(np.mean(experiment_nmaes)) if experiment_nmaes else None
    validity_counts: dict[str, int] = defaultdict(int)
    for result in results:
        validity_counts[
            str(result.get("video_generation_validity", {}).get("status", "missing"))
        ] += 1
    return {
        "job_count": len(results),
        "experiment_count": len(by_experiment),
        "scored_experiment_count": len(experiment_nmaes),
        "generation_validity_counts": dict(validity_counts),
        "generation_valid_rate": validity_counts.get("pass", 0) / max(len(results), 1),
        "by_experiment": by_experiment,
        "conditional_macro_nmae_equal_experiment_weight": macro_nmae,
        "conditional_macro_score_0_100_equal_experiment_weight": (
            None if macro_nmae is None else 100.0 * max(0.0, 1.0 - macro_nmae)
        ),
        "aggregation": (
            "Generation validity uses every scheduled video as denominator. Physics accuracy is "
            "reported only conditional on generation-valid, fitted videos: mean within experiment, "
            "then equal-weight macro mean across experiments. Invalid videos are never assigned a "
            "fabricated parameter estimate."
        ),
    }


def run_physics_batch(
    videos_dir: Path,
    *,
    calibration_root: Path,
    registry_path: Path,
    output_root: Path,
    video_glob: str = "*.mp4",
    limit: int | None = None,
    one_per_experiment: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    registry = load_experiment_registry(registry_path)
    videos = sorted(videos_dir.glob(video_glob))
    if one_per_experiment:
        selected: list[Path] = []
        seen: set[str] = set()
        for video in videos:
            experiment_id = str(parse_video_job(video)["experiment_id"])
            if experiment_id not in seen:
                selected.append(video)
                seen.add(experiment_id)
        videos = selected
    if limit is not None:
        videos = videos[: max(0, int(limit))]
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        run_physics_job(
            path,
            calibration_root=calibration_root,
            registry=registry,
            output_root=output_root,
            overwrite=overwrite,
        )
        for path in videos
    ]
    rows = [_summary_row(result) for result in results]
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = aggregate_results(results)
    _write_json(output_root / "aggregate.json", aggregate)
    return aggregate
