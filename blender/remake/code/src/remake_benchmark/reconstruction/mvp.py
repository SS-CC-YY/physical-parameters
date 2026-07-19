from __future__ import annotations

import csv
import html
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from remake_benchmark.core.io import read_yaml, write_json
from remake_benchmark.evaluators.freefall import _track_video


RECONSTRUCTION_MVP_VERSION = "0.1.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "version": RECONSTRUCTION_MVP_VERSION,
    "camera": {
        "sample_stride": 4,
        "max_corners": 600,
        "quality_level": 0.01,
        "min_distance_px": 8,
        "min_features": 20,
        "ransac_threshold_px": 2.0,
    },
    "tracker": {
        "tracker": "auto",
        "min_template_score": 0.18,
        "motion_threshold": 14,
        "size_lock_warmup_frames": 3,
        "size_lock_warmup_max_dimension_ratio": 1.35,
        "min_locked_bbox_dimension_ratio": 0.85,
        "max_locked_bbox_dimension_ratio": 1.10,
        "lock_bbox_size_near_support": True,
        "deformation_min_dimension_change_ratio": 0.22,
        "deformation_max_area_change_ratio": 1.30,
        "deformation_confirmation_frames": 3,
    },
    "identity": {
        "max_horizontal_object_widths": 2.2,
        "max_step_object_diagonals": 3.0,
        "fallback_max_step_object_diagonals": 1.25,
        "min_bbox_dimension_ratio": 0.65,
        "max_bbox_dimension_ratio": 1.40,
    },
    "calibration": {
        "mode": "known_object_size_side_view_1d_approx",
        "object_diameter_m": 0.48,
        "initial_height_z0_m": 4.20,
        "contact_height_zc_m": 0.44,
        "drop_distance_m": 3.76,
        "scale_relative_uncertainty": 0.05,
        "center_uncertainty_px": 1.5,
    },
    "validity": {
        "min_reference_correlation": 0.80,
        "min_camera_success_fraction": 0.75,
        "min_camera_inlier_ratio": 0.35,
        "max_camera_residual_p95_px": 2.0,
        "min_identity_valid_fraction": 0.30,
        "max_bridge_gap_frames": 2,
        "min_primary_component_coverage": 0.85,
        "min_valid_points": 20,
        "min_longest_valid_run": 12,
        "min_vertical_span_object_diameters": 2.0,
        "max_diameter_cv": 0.15,
    },
    "physics": {
        "min_fit_points": 12,
        "min_fit_duration_s": 0.40,
        "smoothing_window_s": 0.20,
        "release_pre_roll_s": 0.17,
        "release_motion_object_diameters": 0.15,
        "reversal_speed_object_diameters_per_s": 1.20,
        "reversal_duration_s": 0.08,
        "contact_drop_fraction": 0.88,
        "fit_min_r2": 0.90,
        "similarity_pass": 0.75,
        "robust_iterations": 4,
        "robust_sigma": 3.0,
    },
}


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_update(output[key], value)
        else:
            output[key] = value
    return output


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    return _deep_update(DEFAULT_CONFIG, read_yaml(path))


def _decode_number(text: str) -> float:
    return float(text.replace("m", "-").replace("p", "."))


def parse_video_job(video_path: Path) -> dict[str, Any]:
    """Recover the small legacy job contract encoded in a generated filename."""
    parts = video_path.stem.split("__")
    if len(parts) != 6:
        raise ValueError(
            "expected '<experiment>__<target>-<value>__<scene>__<object>__<camera>__seed-<n>.mp4': "
            f"{video_path.name}"
        )
    experiment_id, target_part, scene_id, object_id, camera, seed_part = parts
    if "-" not in target_part or not seed_part.startswith("seed-"):
        raise ValueError(f"unrecognised benchmark video filename: {video_path.name}")
    target_id, encoded_value = target_part.rsplit("-", 1)
    return {
        "job_id": video_path.stem,
        "video_path": str(video_path.resolve()),
        "experiment_id": experiment_id,
        "factors": {
            "scene_id": scene_id,
            "object_id": object_id,
            "camera": camera,
            "seed": int(seed_part.removeprefix("seed-")),
        },
        "targets": {target_id: _decode_number(encoded_value)},
    }


def conditioning_image_for_job(workspace_root: Path, job: dict[str, Any]) -> Path:
    factors = job["factors"]
    path = (
        workspace_root
        / "first_frames_v1_0"
        / "images"
        / str(job["experiment_id"])
        / str(factors["scene_id"])
        / str(factors["object_id"])
        / f"{factors['camera']}.png"
    )
    if not path.is_file():
        raise FileNotFoundError(f"conditioning image not found: {path}")
    return path


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    return None if not len(array) else float(np.percentile(array, percentile))


def _longest_true_run(values: Iterable[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _valid_components(values: list[bool], max_bridge_gap: int) -> list[list[int]]:
    """Group valid measurements while allowing only very short rejected gaps."""
    indices = [index for index, value in enumerate(values) if value]
    if not indices:
        return []
    components: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        missing = index - components[-1][-1] - 1
        if missing <= max(0, int(max_bridge_gap)):
            components[-1].append(index)
        else:
            components.append([index])
    return components


def _reference_alignment(conditioning_path: Path, video_path: Path) -> dict[str, Any]:
    source = cv2.imread(str(conditioning_path), cv2.IMREAD_COLOR)
    capture = cv2.VideoCapture(str(video_path))
    ok, frame0 = capture.read()
    capture.release()
    if source is None or not ok:
        raise RuntimeError("cannot read conditioning image or generated first frame")
    target_h, target_w = frame0.shape[:2]
    source_h, source_w = source.shape[:2]
    target_aspect = target_w / float(target_h)
    source_aspect = source_w / float(source_h)
    if source_aspect >= target_aspect:
        crop_w = max(1, int(round(source_h * target_aspect)))
        x0 = max(0, (source_w - crop_w) // 2)
        crop = source[:, x0 : x0 + crop_w]
        crop_rect = [x0, 0, x0 + crop_w, source_h]
    else:
        crop_h = max(1, int(round(source_w / target_aspect)))
        y0 = max(0, (source_h - crop_h) // 2)
        crop = source[y0 : y0 + crop_h, :]
        crop_rect = [0, y0, source_w, y0 + crop_h]
    mapped = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    source_gray = cv2.cvtColor(mapped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mae = float(np.mean(np.abs(source_gray - target_gray)))
    mse = float(np.mean((source_gray - target_gray) ** 2))
    psnr = None if mse <= 1e-12 else float(10.0 * math.log10((255.0**2) / mse))
    correlation = float(np.corrcoef(source_gray.ravel(), target_gray.ravel())[0, 1])
    return {
        "method": "center_crop_then_resize",
        "conditioning_size": [source_w, source_h],
        "video_size": [target_w, target_h],
        "conditioning_crop_xyxy": crop_rect,
        "mean_absolute_error": mae,
        "psnr_db": psnr,
        "pearson_correlation": correlation if math.isfinite(correlation) else None,
    }


def _estimate_camera_motion(
    video_path: Path,
    initial_center_x: float,
    object_width: float,
    config: dict[str, Any],
    fallback_fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else float(fallback_fps)
    reported_frames_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    reported_frames = (
        int(round(reported_frames_raw))
        if math.isfinite(reported_frames_raw) and reported_frames_raw > 0
        else 0
    )
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
    ok, first = capture.read()
    if not ok or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"empty video or invalid FPS: {video_path}")
    reference = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    feature_mask = np.full(reference.shape, 255, dtype=np.uint8)
    border = max(5, int(round(min(width, height) * 0.02)))
    feature_mask[:border, :] = 0
    feature_mask[-border:, :] = 0
    feature_mask[:, :border] = 0
    feature_mask[:, -border:] = 0
    corridor = max(3.0 * object_width, 0.08 * width)
    x0 = max(0, int(round(initial_center_x - corridor)))
    x1 = min(width, int(round(initial_center_x + corridor)))
    feature_mask[:, x0:x1] = 0
    points0 = cv2.goodFeaturesToTrack(
        reference,
        mask=feature_mask,
        maxCorners=int(config["max_corners"]),
        qualityLevel=float(config["quality_level"]),
        minDistance=float(config["min_distance_px"]),
        blockSize=7,
    )
    first_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    observed_timestamps_ms: list[float] = [first_timestamp_ms]
    rows: list[dict[str, Any]] = [
        {
            "frame_index": 0,
            "time_s": 0.0,
            "success": points0 is not None,
            "tracked_features": 0 if points0 is None else int(len(points0)),
            "inliers": 0 if points0 is None else int(len(points0)),
            "inlier_ratio": 1.0 if points0 is not None else None,
            "tx_px": 0.0,
            "ty_px": 0.0,
            "translation_px": 0.0,
            "rotation_deg": 0.0,
            "scale": 1.0,
            "affine_a": 1.0,
            "affine_b": 0.0,
            "affine_c": 0.0,
            "affine_d": 1.0,
            "median_reprojection_residual_px": 0.0,
        }
    ]
    stride = max(1, int(config["sample_stride"]))
    decoded_frames = 1
    frame_index = 1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or float("nan"))
        observed_timestamps_ms.append(timestamp_ms)
        decoded_frames += 1
        is_sample = frame_index % stride == 0 or (reported_frames and frame_index == reported_frames - 1)
        if is_sample:
            row: dict[str, Any] = {
                "frame_index": frame_index,
                "time_s": frame_index / fps,
                "success": False,
                "tracked_features": 0,
                "inliers": 0,
                "inlier_ratio": None,
                "tx_px": None,
                "ty_px": None,
                "translation_px": None,
                "rotation_deg": None,
                "scale": None,
                "affine_a": None,
                "affine_b": None,
                "affine_c": None,
                "affine_d": None,
                "median_reprojection_residual_px": None,
            }
            if points0 is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                points1, status, _ = cv2.calcOpticalFlowPyrLK(
                    reference,
                    gray,
                    points0,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                )
                if points1 is not None and status is not None:
                    good = status.ravel().astype(bool)
                    p0 = points0.reshape(-1, 2)[good]
                    p1 = points1.reshape(-1, 2)[good]
                    row["tracked_features"] = int(len(p0))
                    if len(p0) >= max(6, int(config["min_features"]) // 2):
                        affine, inlier_mask = cv2.estimateAffinePartial2D(
                            p0,
                            p1,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=float(config["ransac_threshold_px"]),
                            maxIters=2000,
                            confidence=0.995,
                            refineIters=10,
                        )
                        if affine is not None and inlier_mask is not None:
                            inliers = inlier_mask.ravel().astype(bool)
                            predicted = cv2.transform(p0.reshape(1, -1, 2), affine).reshape(-1, 2)
                            residuals = np.linalg.norm(p1 - predicted, axis=1)
                            a, b, tx = affine[0]
                            c, d, ty = affine[1]
                            scale = math.sqrt(float(a * a + c * c))
                            row.update(
                                {
                                    "success": True,
                                    "inliers": int(np.sum(inliers)),
                                    "inlier_ratio": float(np.mean(inliers)),
                                    "tx_px": float(tx),
                                    "ty_px": float(ty),
                                    "translation_px": float(math.hypot(tx, ty)),
                                    "rotation_deg": float(math.degrees(math.atan2(c, a))),
                                    "scale": scale,
                                    "affine_a": float(a),
                                    "affine_b": float(b),
                                    "affine_c": float(c),
                                    "affine_d": float(d),
                                    "median_reprojection_residual_px": float(
                                        np.median(residuals[inliers] if np.any(inliers) else residuals)
                                    ),
                                }
                            )
            rows.append(row)
        frame_index += 1
    capture.release()
    timestamp_array = np.asarray(observed_timestamps_ms, dtype=float)
    pts_valid = bool(
        len(timestamp_array) == decoded_frames
        and np.all(np.isfinite(timestamp_array))
        and len(timestamp_array) >= 2
        and np.all(np.diff(timestamp_array) > 0)
    )
    if pts_valid:
        timestamps_s = ((timestamp_array - timestamp_array[0]) / 1000.0).tolist()
        timeline_method = "opencv_pos_msec"
    else:
        timestamps_s = (np.arange(decoded_frames, dtype=float) / fps).tolist()
        timeline_method = "frame_index_over_reported_or_fallback_fps_cfr"
    for row in rows:
        row["time_s"] = float(timestamps_s[int(row["frame_index"])])
    successful = [row for row in rows if row["success"]]
    summary = {
        "method": "reference_frame_lk_affine_ransac",
        "sample_stride": stride,
        "sample_count": len(rows),
        "successful_samples": len(successful),
        "success_fraction": len(successful) / max(len(rows), 1),
        "median_inlier_ratio": _percentile(
            [row["inlier_ratio"] for row in successful if row["inlier_ratio"] is not None], 50
        ),
        "translation_p95_px": _percentile(
            [row["translation_px"] for row in successful if row["translation_px"] is not None], 95
        ),
        "absolute_rotation_p95_deg": _percentile(
            [abs(row["rotation_deg"]) for row in successful if row["rotation_deg"] is not None], 95
        ),
        "absolute_scale_change_p95": _percentile(
            [abs(row["scale"] - 1.0) for row in successful if row["scale"] is not None], 95
        ),
        "residual_p95_px": _percentile(
            [row["median_reprojection_residual_px"] for row in successful], 95
        ),
    }
    media = {
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
        "decoded_frames": decoded_frames,
        "reported_fps": reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else None,
        "timeline_method": timeline_method,
        "native_pts_available": pts_valid,
        "first_to_last_span_s": float(timestamps_s[-1] - timestamps_s[0]),
        "container_frame_duration_s": float(
            timestamps_s[-1] + (np.median(np.diff(timestamps_s)) if len(timestamps_s) > 1 else 1.0 / fps)
        ),
        "complete_decode": decoded_frames > 0
        and (
            reported_frames <= 0
            or abs(decoded_frames - reported_frames) <= max(1, int(math.ceil(0.01 * decoded_frames)))
        ),
    }
    return rows, summary, media, timestamps_s


def _interpolated_camera_affine(camera_rows: list[dict[str, Any]], frame_count: int) -> dict[str, np.ndarray]:
    successful = [row for row in camera_rows if row["success"] and row["tx_px"] is not None]
    if not successful:
        return {
            "a": np.ones(frame_count),
            "b": np.zeros(frame_count),
            "c": np.zeros(frame_count),
            "d": np.ones(frame_count),
            "tx": np.zeros(frame_count),
            "ty": np.zeros(frame_count),
        }
    indices = np.asarray([row["frame_index"] for row in successful], dtype=float)
    all_indices = np.arange(frame_count, dtype=float)
    return {
        key: np.interp(
            all_indices,
            indices,
            np.asarray([row[source] for row in successful], dtype=float),
        )
        for key, source in {
            "a": "affine_a",
            "b": "affine_b",
            "c": "affine_c",
            "d": "affine_d",
            "tx": "tx_px",
            "ty": "ty_px",
        }.items()
    }


def _mark_identity_and_lift(
    tracks: list[dict[str, Any]],
    camera_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    found = [row for row in tracks[:10] if row["found"]]
    if not found:
        raise RuntimeError("the object tracker did not find the object in the opening frames")
    ref_x = float(np.median([row["center_x_px"] for row in found[:5]]))
    ref_y = float(np.median([row["center_y_px"] for row in found[:5]]))
    ref_width = float(np.median([row["bbox_width_px"] for row in found[:5]]))
    ref_height = float(np.median([row["bbox_height_px"] for row in found[:5]]))
    ref_diag = math.hypot(ref_width, ref_height)
    affine = _interpolated_camera_affine(camera_rows, len(tracks))
    identity = config["identity"]
    last_valid: tuple[float, float] | None = None
    output: list[dict[str, Any]] = []
    diameter_samples: list[float] = []
    for index, source in enumerate(tracks):
        row = dict(source)
        reasons: list[str] = []
        if not row["found"]:
            reasons.append("not_found")
        center_x = row["center_x_px"]
        center_y = row["center_y_px"]
        width = row["bbox_width_px"]
        height = row["bbox_height_px"]
        stabilized_x = stabilized_y = None
        determinant = affine["a"][index] * affine["d"][index] - affine["b"][index] * affine["c"][index]
        if center_x is not None and abs(determinant) > 1e-9:
            shifted_x = float(center_x - affine["tx"][index])
            shifted_y = float(center_y - affine["ty"][index])
            # The estimated transform maps reference frame 0 to the current
            # frame. Apply its full inverse, including rotation and scale, to
            # express the measurement back in the reference image coordinates.
            stabilized_x = float(
                (affine["d"][index] * shifted_x - affine["b"][index] * shifted_y) / determinant
            )
            stabilized_y = float(
                (-affine["c"][index] * shifted_x + affine["a"][index] * shifted_y) / determinant
            )
        if center_x is not None and abs(stabilized_x - ref_x) > float(
            identity["max_horizontal_object_widths"]
        ) * ref_width:
            reasons.append("horizontal_identity_gate")
        if width is not None and height is not None:
            ratios = (width / ref_width, height / ref_height)
            if min(ratios) < float(identity["min_bbox_dimension_ratio"]) or max(ratios) > float(
                identity["max_bbox_dimension_ratio"]
            ):
                reasons.append("bbox_size_gate")
        if stabilized_x is not None and last_valid is not None:
            step = math.hypot(stabilized_x - last_valid[0], stabilized_y - last_valid[1])
            limit = float(identity["max_step_object_diagonals"]) * ref_diag
            if "size-fallback" in str(row["tracking_source"]):
                limit = float(identity["fallback_max_step_object_diagonals"]) * ref_diag
            if step > limit:
                reasons.append("center_step_gate")
        valid = not reasons
        if valid and stabilized_x is not None:
            last_valid = (stabilized_x, stabilized_y)
            diameter_samples.append(math.sqrt(float(width) * float(height)))
        row.update(
            {
                "camera_affine_a": float(affine["a"][index]),
                "camera_affine_b": float(affine["b"][index]),
                "camera_affine_c": float(affine["c"][index]),
                "camera_affine_d": float(affine["d"][index]),
                "camera_tx_px": float(affine["tx"][index]),
                "camera_ty_px": float(affine["ty"][index]),
                "center_x_stabilized_px": stabilized_x,
                "center_y_stabilized_px": stabilized_y,
                "measurement_valid": valid,
                "measurement_invalid_reasons": ";".join(reasons),
            }
        )
        output.append(row)
    calibration_cfg = config["calibration"]
    diameter_px = float(np.median(diameter_samples[: max(5, min(20, len(diameter_samples)))]))
    diameter_cv = float(np.std(diameter_samples) / max(np.mean(diameter_samples), 1e-9))
    pixels_per_meter = diameter_px / float(calibration_cfg["object_diameter_m"])
    valid_rows = [row for row in output if row["measurement_valid"]]
    y0_px = float(np.median([row["center_y_stabilized_px"] for row in valid_rows[:5]]))
    x0_px = float(np.median([row["center_x_stabilized_px"] for row in valid_rows[:5]]))
    z0 = float(calibration_cfg["initial_height_z0_m"])
    center_sigma_px = float(calibration_cfg["center_uncertainty_px"])
    scale_uncertainty = float(calibration_cfg["scale_relative_uncertainty"])
    for row in output:
        if row["measurement_valid"]:
            dx_px = float(row["center_x_stabilized_px"] - x0_px)
            dz_px = float(row["center_y_stabilized_px"] - y0_px)
            row["x_m"] = dx_px / pixels_per_meter
            row["y_m"] = 0.0
            row["z_m"] = z0 - dz_px / pixels_per_meter
            displacement_m = abs(dz_px / pixels_per_meter)
            row["sigma_position_m"] = math.sqrt(
                (center_sigma_px / pixels_per_meter) ** 2 + (scale_uncertainty * displacement_m) ** 2
            )
        else:
            row["x_m"] = row["y_m"] = row["z_m"] = row["sigma_position_m"] = None
    validity_cfg = config["validity"]
    components = _valid_components(
        [bool(row["measurement_valid"]) for row in output],
        int(validity_cfg["max_bridge_gap_frames"]),
    )
    opening_components = [component for component in components if component[0] <= 5]
    primary = max(opening_components or components, key=len) if components else []
    primary_set = set(primary)
    for index, row in enumerate(output):
        row["primary_trajectory"] = index in primary_set
    primary_rows = [output[index] for index in primary]
    primary_span_px = (
        max(row["center_y_stabilized_px"] for row in primary_rows)
        - min(row["center_y_stabilized_px"] for row in primary_rows)
        if primary_rows
        else 0.0
    )
    primary_coverage = (
        len(primary) / max(primary[-1] - primary[0] + 1, 1)
        if primary
        else 0.0
    )
    primary_longest_run = (
        _longest_true_run(index in primary_set for index in range(primary[0], primary[-1] + 1))
        if primary
        else 0
    )
    tracking = {
        "raw_detection_fraction": sum(bool(row["found"]) for row in output) / max(len(output), 1),
        "identity_valid_fraction": sum(bool(row["measurement_valid"]) for row in output) / max(len(output), 1),
        "valid_points": sum(bool(row["measurement_valid"]) for row in output),
        "longest_valid_run": _longest_true_run(row["measurement_valid"] for row in output),
        "component_count": len(components),
        "max_bridge_gap_frames": int(validity_cfg["max_bridge_gap_frames"]),
        "primary_component_frame_start": primary[0] if primary else None,
        "primary_component_frame_end": primary[-1] if primary else None,
        "primary_component_valid_points": len(primary),
        "primary_component_coverage": primary_coverage,
        "primary_component_longest_valid_run": primary_longest_run,
        "primary_component_vertical_span_px": primary_span_px,
        "primary_component_vertical_span_object_diameters": primary_span_px / max(diameter_px, 1e-9),
        "reference_center_px": [ref_x, ref_y],
        "reference_bbox_px": [ref_width, ref_height],
        "vertical_span_px": (
            max(row["center_y_stabilized_px"] for row in valid_rows)
            - min(row["center_y_stabilized_px"] for row in valid_rows)
        ),
        "vertical_span_object_diameters": (
            max(row["center_y_stabilized_px"] for row in valid_rows)
            - min(row["center_y_stabilized_px"] for row in valid_rows)
        )
        / max(diameter_px, 1e-9),
    }
    calibration = {
        "mode": str(calibration_cfg["mode"]),
        "strict_metric_3d": False,
        "known_object_diameter_m": float(calibration_cfg["object_diameter_m"]),
        "observed_object_diameter_px": diameter_px,
        "diameter_cv": diameter_cv,
        "pixels_per_meter": pixels_per_meter,
        "initial_height_z0_m": z0,
        "contact_height_zc_m": float(calibration_cfg["contact_height_zc_m"]),
        "drop_distance_m": float(calibration_cfg["drop_distance_m"]),
        "assumptions": [
            "the generated sphere preserves its known 0.48 m diameter",
            "the object remains in the calibrated side-view motion plane",
            "camera drift is well described by a background affine transform",
            "depth is not independently observable from this fixed-view MP4",
        ],
    }
    return output, tracking, calibration


def _fit_quadratic_robust(t: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(t), dtype=bool)
    coefficients = np.polyfit(t, y, 2)
    for _ in range(max(1, int(config["robust_iterations"]))):
        if np.sum(mask) < 3:
            break
        coefficients = np.polyfit(t[mask], y[mask], 2)
        residuals = y - np.polyval(coefficients, t)
        center = float(np.median(residuals[mask]))
        mad = float(np.median(np.abs(residuals[mask] - center)))
        sigma = max(1.0, 1.4826 * mad)
        updated = np.abs(residuals - center) <= float(config["robust_sigma"]) * sigma
        if np.array_equal(updated, mask):
            break
        mask = updated
    if np.sum(mask) >= 3:
        coefficients = np.polyfit(t[mask], y[mask], 2)
    return coefficients, mask


def _similarity(estimate: float | None, target: float | None) -> float | None:
    if estimate is None or target is None or estimate <= 0 or target <= 0:
        return None
    return min(estimate, target) / max(estimate, target)


def fit_freefall_physics(
    trajectory: list[dict[str, Any]],
    target_gravity: float,
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit only the first continuous airborne episode; do not cherry-pick a later arc."""
    physics = config["physics"] if "physics" in config else config
    valid = np.asarray([bool(row.get("primary_trajectory")) for row in trajectory])
    if not np.any(valid):
        return {"status": "insufficient", "reason": "no_valid_measurements"}
    indices = np.flatnonzero(valid)
    y = np.asarray([trajectory[index]["center_y_stabilized_px"] for index in indices], dtype=float)
    t = np.asarray([trajectory[index]["time_s"] for index in indices], dtype=float)
    diameter = float(calibration["observed_object_diameter_px"])
    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 0]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0 / 24.0
    window = max(1, int(round(float(physics["smoothing_window_s"]) / median_dt)))
    if window % 2 == 0:
        window += 1
    if len(y) >= window:
        pad = window // 2
        smooth = np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(window) / window, mode="valid")
    else:
        smooth = y.copy()
    opening = float(np.median(smooth[: min(5, len(smooth))]))
    motion_threshold = float(physics["release_motion_object_diameters"]) * diameter
    moved = np.flatnonzero(smooth - opening >= motion_threshold)
    pre_roll = max(0, int(round(float(physics["release_pre_roll_s"]) / median_dt)))
    release_local = max(0, int(moved[0]) - pre_roll) if len(moved) else 0
    stop_local = len(indices) - 1
    expected_drop_px = float(calibration["drop_distance_m"]) * float(calibration["pixels_per_meter"])
    contact_candidates = np.flatnonzero(
        smooth - opening >= float(physics["contact_drop_fraction"]) * expected_drop_px
    )
    if len(contact_candidates):
        stop_local = min(stop_local, int(contact_candidates[0]) + 1)
    derivative = np.diff(smooth) / np.maximum(np.diff(t), 1e-9)
    reversal_limit = -float(physics["reversal_speed_object_diameters_per_s"]) * diameter
    reversal_run = max(1, int(math.ceil(float(physics["reversal_duration_s"]) / median_dt)))
    meaningful_drop = smooth - opening >= 2.0 * diameter
    for index in range(max(release_local + 2, 1), len(derivative) - reversal_run + 1):
        if meaningful_drop[index] and np.all(derivative[index : index + reversal_run] < reversal_limit):
            stop_local = min(stop_local, index)
            break
    fit_indices = indices[release_local : stop_local + 1]
    fit_duration = (
        float(trajectory[int(fit_indices[-1])]["time_s"] - trajectory[int(fit_indices[0])]["time_s"])
        if len(fit_indices) >= 2
        else 0.0
    )
    if len(fit_indices) < int(physics["min_fit_points"]) or fit_duration < float(physics["min_fit_duration_s"]):
        return {
            "status": "insufficient",
            "reason": "airborne_segment_too_short",
            "candidate_points": int(len(fit_indices)),
            "candidate_duration_s": fit_duration,
            "segment_frame_start": int(fit_indices[0]) if len(fit_indices) else None,
            "segment_frame_end": int(fit_indices[-1]) if len(fit_indices) else None,
            "target_gravity_m_s2": float(target_gravity),
        }
    fit_t = np.asarray([trajectory[index]["time_s"] for index in fit_indices], dtype=float)
    fit_y = np.asarray([trajectory[index]["center_y_stabilized_px"] for index in fit_indices], dtype=float)
    coefficients, inliers = _fit_quadratic_robust(fit_t, fit_y, physics)
    predicted = np.polyval(coefficients, fit_t)
    residuals = fit_y - predicted
    ss_res = float(np.sum(residuals[inliers] ** 2))
    centered = fit_y[inliers] - np.mean(fit_y[inliers])
    ss_tot = float(np.sum(centered**2))
    r2 = None if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    acceleration_px_s2 = float(2.0 * coefficients[0])
    gravity = acceleration_px_s2 / float(calibration["pixels_per_meter"])
    similarity = _similarity(gravity, float(target_gravity))
    fit_valid = (
        r2 is not None
        and r2 >= float(physics["fit_min_r2"])
        and gravity > 0
        and int(np.sum(inliers)) >= int(physics["min_fit_points"])
    )
    adherence = bool(fit_valid and similarity is not None and similarity >= float(physics["similarity_pass"]))
    for index in fit_indices[inliers]:
        trajectory[int(index)]["physics_fit_used"] = True
    return {
        "status": "pass" if adherence else "fail",
        "fit_valid": fit_valid,
        "equation_image": "v(t)=c2*t^2+c1*t+c0; a_px=2*c2",
        "equation_world": "z(t)=z0+v0*t-0.5*g*t^2",
        "segment_frame_start": int(fit_indices[0]),
        "segment_frame_end": int(fit_indices[-1]),
        "segment_time_start_s": float(fit_t[0]),
        "segment_time_end_s": float(fit_t[-1]),
        "fit_points": int(len(fit_t)),
        "fit_inliers": int(np.sum(inliers)),
        "quadratic_c2_px_s2": float(coefficients[0]),
        "linear_c1_px_s": float(coefficients[1]),
        "intercept_c0_px": float(coefficients[2]),
        "vertical_acceleration_px_s2": acceleration_px_s2,
        "estimated_gravity_m_s2": gravity,
        "target_gravity_m_s2": float(target_gravity),
        "parameter_similarity": similarity,
        "fit_r2": r2,
        "fit_rmse_px": rmse,
        "pixels_per_meter": float(calibration["pixels_per_meter"]),
        "adherence_pass": adherence,
    }


def _validate_reconstruction(
    media: dict[str, Any],
    alignment: dict[str, Any],
    camera: dict[str, Any],
    tracking: dict[str, Any],
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["validity"]
    gates = {
        "media_complete_decode": bool(media["complete_decode"]),
        "reference_alignment": alignment["pearson_correlation"] is not None
        and alignment["pearson_correlation"] >= float(thresholds["min_reference_correlation"]),
        "camera_samples": camera["success_fraction"] >= float(thresholds["min_camera_success_fraction"]),
        "camera_inliers": camera["median_inlier_ratio"] is not None
        and camera["median_inlier_ratio"] >= float(thresholds["min_camera_inlier_ratio"]),
        "camera_residual": camera["residual_p95_px"] is not None
        and camera["residual_p95_px"] <= float(thresholds["max_camera_residual_p95_px"]),
        "identity_fraction": tracking["identity_valid_fraction"]
        >= float(thresholds["min_identity_valid_fraction"]),
        "primary_component_coverage": tracking["primary_component_coverage"]
        >= float(thresholds["min_primary_component_coverage"]),
        "valid_points": tracking["primary_component_valid_points"] >= int(thresholds["min_valid_points"]),
        "continuous_track": tracking["primary_component_longest_valid_run"]
        >= int(thresholds["min_longest_valid_run"]),
        "motion_span": tracking["primary_component_vertical_span_object_diameters"]
        >= float(thresholds["min_vertical_span_object_diameters"]),
        "scale_stability": calibration["diameter_cv"] <= float(thresholds["max_diameter_cv"]),
    }
    reasons = [name for name, passed in gates.items() if not passed]
    constrained_valid = not reasons
    return {
        "status": "pass" if constrained_valid else "invalid",
        "constrained_trajectory_valid": constrained_valid,
        "strict_metric_3d_valid": False,
        "strict_metric_3d_reason": "missing full camera K/T and fixed-view MP4 has no independent depth observability",
        "calibration_mode": calibration["mode"],
        "gates": gates,
        "failed_gates": reasons,
        "interpretation": (
            "Pass means the object-centric side-view trajectory is measurable under the declared priors. "
            "It does not certify a complete scene mesh or physical correctness."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_ply(path: Path, rows: list[dict[str, Any]]) -> None:
    points = [row for row in rows if row.get("primary_trajectory") and row["z_m"] is not None]
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment object-centric trajectory; reconstruction_mvp={RECONSTRUCTION_MVP_VERSION}",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(f"{row['x_m']:.8f} {row['y_m']:.8f} {row['z_m']:.8f} 34 139 230" for row in points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_diagnostics(
    video_path: Path,
    trajectory: list[dict[str, Any]],
    physics: dict[str, Any],
    output_path: Path,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame0 = capture.read()
    capture.release()
    if not ok:
        return
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    valid = [row for row in trajectory if row.get("primary_trajectory")]
    secondary = [
        row for row in trajectory if row["measurement_valid"] and not row.get("primary_trajectory")
    ]
    invalid = [row for row in trajectory if not row["measurement_valid"] and row["center_x_stabilized_px"] is not None]
    figure = plt.figure(figsize=(15, 4.6), constrained_layout=True)
    axis_image = figure.add_subplot(1, 3, 1)
    axis_image.imshow(frame0)
    axis_image.plot(
        [row["center_x_stabilized_px"] for row in valid],
        [row["center_y_stabilized_px"] for row in valid],
        color="#00d084",
        linewidth=2,
        label="valid measurement",
    )
    if invalid:
        axis_image.scatter(
            [row["center_x_stabilized_px"] for row in invalid],
            [row["center_y_stabilized_px"] for row in invalid],
            s=10,
            color="#ff4d4f",
            label="rejected",
        )
    if secondary:
        axis_image.scatter(
            [row["center_x_stabilized_px"] for row in secondary],
            [row["center_y_stabilized_px"] for row in secondary],
            s=10,
            color="#ff8c00",
            label="secondary fragment",
        )
    axis_image.set_title("Image-plane trajectory")
    axis_image.set_axis_off()
    axis_image.legend(loc="lower left", fontsize=8)

    axis_height = figure.add_subplot(1, 3, 2)
    axis_height.plot([row["time_s"] for row in valid], [row["z_m"] for row in valid], ".-", ms=3, label="reconstructed")
    if physics.get("fit_valid"):
        start = int(physics["segment_frame_start"])
        end = int(physics["segment_frame_end"])
        fit_rows = trajectory[start : end + 1]
        times = np.asarray([row["time_s"] for row in fit_rows], dtype=float)
        c2 = float(physics["quadratic_c2_px_s2"])
        c1 = float(physics["linear_c1_px_s"])
        c0 = float(physics["intercept_c0_px"])
        y_pred = c2 * times**2 + c1 * times + c0
        reference = fit_rows[0]
        observed_y = float(reference["center_y_stabilized_px"])
        observed_z = float(reference["z_m"])
        ppm = float(physics["pixels_per_meter"])
        if ppm and math.isfinite(ppm):
            z_pred = observed_z - (y_pred - y_pred[0]) / ppm
            axis_height.plot(times, z_pred, "--", color="#ff8c00", label="quadratic fit")
    axis_height.set_xlabel("time (s), native FPS")
    axis_height.set_ylabel("height z (m, approximate)")
    axis_height.grid(alpha=0.25)
    axis_height.legend(fontsize=8)
    axis_height.set_title("Height over time")

    axis_3d = figure.add_subplot(1, 3, 3, projection="3d")
    axis_3d.plot(
        [row["x_m"] for row in valid],
        [row["y_m"] for row in valid],
        [row["z_m"] for row in valid],
        color="#228be6",
        linewidth=2,
    )
    axis_3d.scatter([valid[0]["x_m"]], [0.0], [valid[0]["z_m"]], color="#00a86b", s=28, label="start")
    axis_3d.set_xlabel("x (m)")
    axis_3d.set_ylabel("y (constrained)")
    axis_3d.set_zlabel("z (m)")
    axis_3d.set_title("Constrained 2.5D path")
    axis_3d.legend(fontsize=8)
    figure.suptitle(video_path.stem, fontsize=10)
    figure.savefig(output_path, dpi=145)
    plt.close(figure)


def _write_overlay(video_path: Path, trajectory: list[dict[str, Any]], output_path: Path) -> str | None:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        return None
    tail: list[tuple[int, int]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index < len(trajectory):
            row = trajectory[frame_index]
            if row.get("primary_trajectory"):
                color = (70, 210, 40)
                point = (int(round(row["center_x_px"])), int(round(row["center_y_px"])))
                tail.append(point)
                tail = tail[-24:]
                label = "primary"
            elif row["measurement_valid"]:
                color = (0, 140, 255)
                label = "secondary"
            else:
                color = (40, 40, 230)
                label = "rejected"
            if row["found"]:
                p0 = (int(round(row["bbox_x0"])), int(round(row["bbox_y0"])))
                p1 = (int(round(row["bbox_x1"])), int(round(row["bbox_y1"])))
                cv2.rectangle(frame, p0, p1, color, 2)
            for start, end in zip(tail, tail[1:]):
                cv2.line(frame, start, end, (230, 170, 30), 2, cv2.LINE_AA)
            cv2.putText(frame, f"track: {label}  t={row['time_s']:.3f}s", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    return str(output_path.resolve())


def _relative_artifacts(job_dir: Path, overlay: str | None) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "result_json": str((job_dir / "result.json").resolve()),
        "video_probe_json": str((job_dir / "video_probe.json").resolve()),
        "reference_alignment_json": str((job_dir / "reference_alignment.json").resolve()),
        "camera_motion_csv": str((job_dir / "camera_motion.csv").resolve()),
        "trajectory_world_csv": str((job_dir / "trajectory_world.csv").resolve()),
        "trajectory_ply": str((job_dir / "trajectory.ply").resolve()),
        "diagnostic_plot": str((job_dir / "trajectory_diagnostics.png").resolve()),
        "reconstruction_validity_json": str((job_dir / "reconstruction_validity.json").resolve()),
        "physics_adherence_json": str((job_dir / "physics_adherence.json").resolve()),
    }
    if overlay:
        artifacts["overlay_video"] = overlay
    return artifacts


def run_reconstruction_job(
    job: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
    config: dict[str, Any] | None = None,
    make_overlay: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = deepcopy(config or DEFAULT_CONFIG)
    video_path = Path(job["video_path"])
    conditioning_path = conditioning_image_for_job(workspace_root, job)
    job_dir = output_root / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    alignment = _reference_alignment(conditioning_path, video_path)
    tracker_job = {
        "job_id": job["job_id"],
        "factors": job["factors"],
        "generation": {"fps": 24.0},
    }
    raw_tracks, native_fps, frame_shape = _track_video(
        tracker_job,
        video_path,
        conditioning_path,
        config["tracker"],
    )
    initial = next(row for row in raw_tracks if row["found"])
    camera_rows, camera_summary, media, timestamps_s = _estimate_camera_motion(
        video_path,
        float(initial["center_x_px"]),
        float(initial["bbox_width_px"]),
        config["camera"],
        native_fps,
    )
    if abs(native_fps - media["fps"]) > 1e-6:
        raise RuntimeError("tracker and media probe disagree on native FPS")
    if len(timestamps_s) != len(raw_tracks):
        raise RuntimeError("tracker and media probe disagree on decoded frame count")
    for row, timestamp_s in zip(raw_tracks, timestamps_s):
        row["time_s"] = float(timestamp_s)
    trajectory, tracking, calibration = _mark_identity_and_lift(raw_tracks, camera_rows, config)
    for row in trajectory:
        row.setdefault("physics_fit_used", False)
    target_gravity = float(job["targets"].get("gravity_g"))
    physics = fit_freefall_physics(trajectory, target_gravity, calibration, config)
    validity = _validate_reconstruction(media, alignment, camera_summary, tracking, calibration, config)

    _write_csv(job_dir / "camera_motion.csv", camera_rows)
    _write_csv(job_dir / "trajectory_world.csv", trajectory)
    _write_ply(job_dir / "trajectory.ply", trajectory)
    _plot_diagnostics(video_path, trajectory, physics, job_dir / "trajectory_diagnostics.png")
    overlay = _write_overlay(video_path, trajectory, job_dir / "track_overlay.mp4") if make_overlay else None
    write_json(job_dir / "video_probe.json", media)
    write_json(job_dir / "reference_alignment.json", alignment)
    write_json(job_dir / "reconstruction_validity.json", validity)
    write_json(job_dir / "physics_adherence.json", physics)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "1.0.0",
        "module_version": RECONSTRUCTION_MVP_VERSION,
        "job_id": job["job_id"],
        "source_video": str(video_path.resolve()),
        "conditioning_image": str(conditioning_path.resolve()),
        "experiment_id": job["experiment_id"],
        "factors": job["factors"],
        "targets": job["targets"],
        "media": media,
        "reference_alignment": alignment,
        "camera_motion": camera_summary,
        "tracking": tracking,
        "calibration": calibration,
        "reconstruction_validity": validity,
        "physics_adherence": physics,
        "elapsed_seconds": elapsed,
        "artifacts": _relative_artifacts(job_dir, overlay),
    }
    write_json(job_dir / "result.json", result)
    return result


def _job_worker(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_reconstruction_job(
            payload["job"],
            Path(payload["workspace_root"]),
            Path(payload["output_root"]),
            payload["config"],
            bool(payload["make_overlay"]),
        )
    except Exception as exc:  # batch mode must preserve all other diagnostics
        job_dir = Path(payload["output_root"]) / payload["job"]["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": "1.0.0",
            "module_version": RECONSTRUCTION_MVP_VERSION,
            "job_id": payload["job"]["job_id"],
            "experiment_id": payload["job"].get("experiment_id"),
            "factors": payload["job"].get("factors", {}),
            "targets": payload["job"].get("targets", {}),
            "error": f"{type(exc).__name__}: {exc}",
            "reconstruction_validity": {
                "status": "error",
                "constrained_trajectory_valid": False,
                "strict_metric_3d_valid": False,
            },
            "physics_adherence": {"status": "not_run"},
            "artifacts": {"result_json": str((job_dir / "result.json").resolve())},
        }
        # Always replace a result from an older successful attempt so the new
        # batch report cannot silently point at stale success metadata.
        write_json(job_dir / "result.json", result)
        return result


def _choose_overlay_jobs(jobs: list[dict[str, Any]], count: int) -> set[str]:
    if count <= 0:
        return set()
    selected: list[dict[str, Any]] = []
    scenes = sorted({job["factors"]["scene_id"] for job in jobs})
    for scene in scenes:
        candidates = [job for job in jobs if job["factors"]["scene_id"] == scene]
        selected.append(min(candidates, key=lambda job: abs(float(job["targets"].get("gravity_g", 0)) - 9.81)))
    for job in jobs:
        if job not in selected:
            selected.append(job)
    return {job["job_id"] for job in selected[:count]}


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    camera = result.get("camera_motion", {})
    tracking = result.get("tracking", {})
    calibration = result.get("calibration", {})
    validity = result.get("reconstruction_validity", {})
    physics = result.get("physics_adherence", {})
    factors = result.get("factors", {})
    targets = result.get("targets", {})
    return {
        "job_id": result["job_id"],
        "scene_id": factors.get("scene_id"),
        "camera": factors.get("camera"),
        "target_gravity_m_s2": targets.get("gravity_g"),
        "native_fps": result.get("media", {}).get("fps"),
        "decoded_frames": result.get("media", {}).get("decoded_frames"),
        "reconstruction_status": validity.get("status"),
        "constrained_trajectory_valid": validity.get("constrained_trajectory_valid"),
        "strict_metric_3d_valid": validity.get("strict_metric_3d_valid"),
        "failed_reconstruction_gates": ";".join(validity.get("failed_gates", [])),
        "reference_correlation": result.get("reference_alignment", {}).get("pearson_correlation"),
        "camera_translation_p95_px": camera.get("translation_p95_px"),
        "camera_residual_p95_px": camera.get("residual_p95_px"),
        "identity_valid_fraction": tracking.get("identity_valid_fraction"),
        "primary_component_coverage": tracking.get("primary_component_coverage"),
        "primary_component_valid_points": tracking.get("primary_component_valid_points"),
        "primary_component_span_object_diameters": tracking.get(
            "primary_component_vertical_span_object_diameters"
        ),
        "longest_valid_run": tracking.get("longest_valid_run"),
        "calibration_mode": calibration.get("mode"),
        "estimated_gravity_m_s2": physics.get("estimated_gravity_m_s2"),
        "fit_r2": physics.get("fit_r2"),
        "parameter_similarity": physics.get("parameter_similarity"),
        "physics_status": physics.get("status"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "error": result.get("error"),
    }


def _write_html_report(output_root: Path, results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    rows = []
    for result in results:
        summary = _summary_row(result)
        job_id = str(result["job_id"])
        job_dir = html.escape(job_id)
        status = html.escape(str(summary["reconstruction_status"]))
        physics = html.escape(str(summary["physics_status"]))
        gravity = summary["estimated_gravity_m_s2"]
        similarity = summary["parameter_similarity"]
        camera_drift = summary["camera_translation_p95_px"]
        identity_fraction = summary["primary_component_coverage"]
        gravity_text = "" if gravity is None else f"{gravity:.3f}"
        similarity_text = "" if similarity is None else f"{similarity:.3f}"
        camera_drift_text = "" if camera_drift is None else f"{camera_drift:.3f}"
        identity_text = "" if identity_fraction is None else f"{identity_fraction:.3f}"
        artifacts = result.get("artifacts", {})
        artifact_links: list[str] = []
        if artifacts.get("diagnostic_plot"):
            artifact_links.append(f"<a href='../{job_dir}/trajectory_diagnostics.png'>plot</a>")
        if artifacts.get("overlay_video"):
            artifact_links.append(f"<a href='../{job_dir}/track_overlay.mp4'>overlay</a>")
        rows.append(
            "<tr>"
            f"<td><a href='../{job_dir}/result.json'>{html.escape(job_id)}</a></td>"
            f"<td>{status}</td><td>{physics}</td>"
            f"<td>{summary['target_gravity_m_s2']}</td>"
            f"<td>{gravity_text}</td>"
            f"<td>{similarity_text}</td>"
            f"<td>{camera_drift_text}</td>"
            f"<td>{identity_text}</td>"
            f"<td>{' · '.join(artifact_links)}</td></tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Reconstruction MVP report</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#17202a}}.note{{max-width:78rem;padding:1rem;background:#f1f5f9;border-left:4px solid #228be6}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d9e2ec;padding:.45rem;text-align:left}}th{{background:#eef2f7;position:sticky;top:0}}code{{background:#eef2f7;padding:.12rem .25rem}}</style></head>
<body><h1>Seedance 20 — object-centric reconstruction MVP</h1>
<div class="note"><b>Interpretation boundary:</b> <code>constrained_trajectory_valid</code> checks whether the side-view object trajectory is measurable. <code>physics_status</code> independently checks the target equation/parameter. Fixed-view MP4 without full K/T cannot pass <code>strict_metric_3d_valid</code>.</div>
<p>Total: {aggregate['jobs']} · constrained pass: {aggregate['constrained_trajectory_passed']} · physics pass: {aggregate['physics_passed']} · errors: {aggregate['errors']} · wall time: {aggregate['wall_seconds']:.1f}s</p>
<table><thead><tr><th>job</th><th>reconstruction</th><th>physics</th><th>target g</th><th>estimated g</th><th>similarity</th><th>camera drift p95 px</th><th>primary coverage</th><th>artifacts</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text(document, encoding="utf-8")


def _validate_mvp_jobs(jobs: list[dict[str, Any]]) -> None:
    unsupported = [
        job["job_id"]
        for job in jobs
        if job.get("experiment_id") != "v1_A"
        or "gravity_g" not in job.get("targets", {})
        or job.get("factors", {}).get("object_id") != "standard_ball"
        or job.get("factors", {}).get("camera") != "CAM_Side"
    ]
    if unsupported:
        preview = ", ".join(unsupported[:3])
        suffix = " ..." if len(unsupported) > 3 else ""
        raise ValueError(
            "reconstruction MVP 0.1 supports only v1_A/standard_ball/CAM_Side/gravity_g; "
            f"unsupported jobs: {preview}{suffix}"
        )


def _run_reconstruction_batch_unlocked(
    workspace_root: Path,
    videos_dir: Path,
    output_root: Path,
    config: dict[str, Any] | None = None,
    workers: int = 1,
    overlay_count: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = deepcopy(config or DEFAULT_CONFIG)
    videos = sorted(videos_dir.glob("*.mp4"))
    if limit is not None:
        videos = videos[: max(0, int(limit))]
    if not videos:
        raise FileNotFoundError(f"no MP4 files found in {videos_dir}")
    jobs = [parse_video_job(path) for path in videos]
    _validate_mvp_jobs(jobs)
    overlays = _choose_overlay_jobs(jobs, overlay_count)
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "job": job,
            "workspace_root": str(workspace_root),
            "output_root": str(output_root),
            "config": config,
            "make_overlay": job["job_id"] in overlays,
        }
        for job in jobs
    ]
    results: list[dict[str, Any]] = []
    worker_count = max(1, int(workers))
    if worker_count == 1:
        for index, payload in enumerate(payloads, 1):
            result = _job_worker(payload)
            results.append(result)
            print(f"[{index}/{len(payloads)}] {result['job_id']}: {result['reconstruction_validity']['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_job_worker, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(payloads)}] {result['job_id']}: {result['reconstruction_validity']['status']}", flush=True)
    results.sort(key=lambda item: item["job_id"])
    summary_rows = [_summary_row(result) for result in results]
    _write_csv(output_root / "summary.csv", summary_rows)
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    wall = time.perf_counter() - started
    aggregate = {
        "schema_version": "1.0.0",
        "module_version": RECONSTRUCTION_MVP_VERSION,
        "jobs": len(results),
        "constrained_trajectory_passed": sum(
            result["reconstruction_validity"].get("constrained_trajectory_valid") is True for result in results
        ),
        "strict_metric_3d_passed": sum(
            result["reconstruction_validity"].get("strict_metric_3d_valid") is True for result in results
        ),
        "physics_passed": sum(result.get("physics_adherence", {}).get("status") == "pass" for result in results),
        "physics_failed": sum(result.get("physics_adherence", {}).get("status") == "fail" for result in results),
        "physics_insufficient": sum(
            result.get("physics_adherence", {}).get("status") == "insufficient" for result in results
        ),
        "errors": sum("error" in result for result in results),
        "wall_seconds": wall,
        "workers": worker_count,
        "native_fps_values": sorted(
            {result.get("media", {}).get("fps") for result in results if result.get("media", {}).get("fps") is not None}
        ),
        "calibration_modes": sorted(
            {result.get("calibration", {}).get("mode") for result in results if result.get("calibration", {}).get("mode")}
        ),
        "interpretation": {
            "constrained_trajectory_passed": "measurable object-centric trajectory under declared priors",
            "strict_metric_3d_passed": "full metric 3D with calibrated K/T and observable depth; expected zero for this dataset",
            "physics_passed": "fit quality and target-parameter similarity both pass",
        },
    }
    write_json(output_root / "aggregate.json", aggregate)
    _write_html_report(output_root, results, aggregate)
    return aggregate


def run_reconstruction_batch(
    workspace_root: Path,
    videos_dir: Path,
    output_root: Path,
    config: dict[str, Any] | None = None,
    workers: int = 1,
    overlay_count: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".reconstruction_mvp.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_unix_s": time.time()}) + "\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"output directory is already in use or has a stale lock: {lock_path}; "
            "verify no batch is running, then remove only this lock file"
        ) from exc
    try:
        return _run_reconstruction_batch_unlocked(
            workspace_root=workspace_root,
            videos_dir=videos_dir,
            output_root=output_root,
            config=config,
            workers=workers,
            overlay_count=overlay_count,
            limit=limit,
        )
    finally:
        lock_path.unlink(missing_ok=True)
