"""Conservative 2D rigidity evidence for nominally static generated scenes.

The check is deliberately not a raw frame-difference detector.  It tracks a
spatially distributed set of background features, compensates one global
homography, excludes the complete ball-motion corridor (and the pendulum
apparatus sweep when applicable), and only reports a hard failure when local
geometric and appearance anomalies persist across several sampled frames.

For videos whose camera audit is changed or uncertain, this 2D check returns
``indeterminate``.  Parallax from a moving camera is not evidence that the
generated scene itself deformed; those videos require the dynamic 3D/4D path.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .v1a_geometry import crop_resize_homography, project_points


STATIC_CAMERA_CATEGORIES = {
    "fixed",
    "no_significant_camera_change",
    "calibrated_motion_within_tolerance",
    "side_2d_motion_within_tolerance",
}


def _camera_category(evidence: Mapping[str, Any] | None) -> str | None:
    if not evidence:
        return None
    value = (
        evidence.get("effective_camera_motion_category")
        or evidence.get("final_category")
        or evidence.get("decision")
    )
    return None if value is None else str(value)


def _sample_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 1 or count <= 0:
        return []
    values = np.linspace(1, frame_count - 1, min(count, frame_count - 1))
    return sorted(set(int(round(item)) for item in values))


def _resize_gray(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_width) / max(width, 1))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0), scale


def _pendulum_pivot_uv(
    calibration: Mapping[str, Any] | None,
    video_size: tuple[int, int],
    first_track: Mapping[str, Any] | None,
) -> np.ndarray | None:
    if not calibration or not first_track or not first_track.get("found"):
        return None
    pivot_info = calibration.get("standard_ball", {}).get("pendulum_ancestor_pivot")
    if pivot_info is None:
        pivot_info = calibration.get("object", {}).get("pendulum_ancestor_pivot")
    if not isinstance(pivot_info, Mapping) or "pivot_world_m" not in pivot_info:
        return None
    try:
        camera = calibration["camera"]
        source_image = calibration.get("image") or calibration["source_image"]
        source_size = (int(source_image["width_px"]), int(source_image["height_px"]))
        transform = crop_resize_homography(source_size, video_size)
        K = np.asarray(camera["K"], dtype=np.float64)
        R = np.asarray(camera["R_world_to_camera_opencv"], dtype=np.float64)
        t = np.asarray(camera["t_world_to_camera_opencv_m"], dtype=np.float64)
        pivot_source = project_points(
            np.asarray([pivot_info["pivot_world_m"]], dtype=np.float64), K, R, t
        )[0]
        pivot_h = transform @ np.asarray([pivot_source[0], pivot_source[1], 1.0])
        pivot_video = pivot_h[:2] / pivot_h[2]

        validation = calibration.get("projection_validation", {})
        ball_source = validation.get("initial_center_uv_from_opencv_P")
        if ball_source is None:
            ball_source = calibration.get("frame1_projection", {}).get("center_uv_px")
        if ball_source is not None:
            ball_h = transform @ np.asarray([ball_source[0], ball_source[1], 1.0])
            ball_video = ball_h[:2] / ball_h[2]
            observed = np.asarray(
                [first_track["center_u_px"], first_track["center_v_px"]], dtype=np.float64
            )
            pivot_video += observed - ball_video
        return pivot_video
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _exclusion_mask(
    shape: tuple[int, int],
    track_rows: Sequence[Mapping[str, Any]],
    scale: float,
    pivot_uv: np.ndarray | None,
) -> np.ndarray:
    height, width = shape
    allowed = np.full((height, width), 255, dtype=np.uint8)
    border = max(3, int(round(0.02 * min(width, height))))
    allowed[:border] = 0
    allowed[-border:] = 0
    allowed[:, :border] = 0
    allowed[:, -border:] = 0
    found: list[tuple[tuple[int, int], int]] = []
    for row in track_rows:
        if not row.get("found"):
            continue
        center = (
            int(round(float(row["center_u_px"]) * scale)),
            int(round(float(row["center_v_px"]) * scale)),
        )
        radius = max(4, int(round(float(row["measurement_radius_px"]) * scale * 3.2)))
        found.append((center, radius))
        cv2.circle(allowed, center, radius, 0, -1, cv2.LINE_AA)
    if pivot_uv is not None and found:
        pivot = tuple(np.round(pivot_uv * scale).astype(int))
        thickness = max(radius for _, radius in found)
        for center, radius in found:
            cv2.line(allowed, pivot, center, 0, max(thickness, radius), cv2.LINE_AA)
            cv2.circle(allowed, pivot, max(thickness, radius), 0, -1, cv2.LINE_AA)
    return allowed


def _grid_features(gray: np.ndarray, allowed: np.ndarray, max_corners: int = 480) -> np.ndarray | None:
    height, width = gray.shape
    rows, cols = 4, 6
    per_cell = max(8, int(math.ceil(max_corners / float(rows * cols))))
    points: list[np.ndarray] = []
    for gy in range(rows):
        y0, y1 = int(gy * height / rows), int((gy + 1) * height / rows)
        for gx in range(cols):
            x0, x1 = int(gx * width / cols), int((gx + 1) * width / cols)
            mask = np.zeros_like(allowed)
            mask[y0:y1, x0:x1] = allowed[y0:y1, x0:x1]
            selected = cv2.goodFeaturesToTrack(
                gray,
                mask=mask,
                maxCorners=per_cell,
                qualityLevel=0.012,
                minDistance=6.0,
                blockSize=7,
            )
            if selected is not None:
                points.append(selected.reshape(-1, 2))
    if not points:
        return None
    return np.concatenate(points, axis=0)[:max_corners].astype(np.float32).reshape(-1, 1, 2)


def _patch_ncc(first: np.ndarray, second: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    radius = 4
    x0, y0 = (int(round(value)) for value in p0)
    x1, y1 = (int(round(value)) for value in p1)
    if (
        min(x0, y0, x1, y1) < radius
        or x0 + radius >= first.shape[1]
        or x1 + radius >= second.shape[1]
        or y0 + radius >= first.shape[0]
        or y1 + radius >= second.shape[0]
    ):
        return 1.0
    a = first[y0 - radius : y0 + radius + 1, x0 - radius : x0 + radius + 1].astype(float)
    b = second[y1 - radius : y1 + radius + 1, x1 - radius : x1 + radius + 1].astype(float)
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if denominator <= 1e-9 else float(np.sum(a * b) / denominator)


def _cell_count(points: np.ndarray, width: int, height: int) -> int:
    if not len(points):
        return 0
    gx = np.clip((points[:, 0] / max(width, 1) * 6).astype(int), 0, 5)
    gy = np.clip((points[:, 1] / max(height, 1) * 4).astype(int), 0, 3)
    return len(set(zip(gx.tolist(), gy.tolist())))


def assess_static_scene_rigidity(
    frames: Sequence[np.ndarray],
    track_rows: Sequence[Mapping[str, Any]],
    *,
    camera_motion_evidence: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None = None,
    sample_count: int = 8,
    max_width: int = 432,
) -> dict[str, Any]:
    """Return conservative background/non-experimental-object rigidity evidence."""

    category = _camera_category(camera_motion_evidence)
    base: dict[str, Any] = {
        "method": "global_homography_compensated_lk_plus_local_appearance",
        "camera_motion_category": category,
        "status": "indeterminate",
        "hard_failure": False,
        "failure_code": None,
        "sample_rows": [],
    }
    if camera_motion_evidence and int(camera_motion_evidence.get("cut_pair_count") or 0) > 0:
        base.update(
            {
                "status": "fail",
                "hard_failure": True,
                "failure_code": "scene_cut_or_abrupt_reframe",
            }
        )
        return base
    if category not in STATIC_CAMERA_CATEGORIES:
        base["reason"] = "moving_or_uncertain_camera_requires_3d_scene_compensation"
        return base
    if len(frames) < 2:
        base["reason"] = "fewer_than_two_decoded_frames"
        return base

    reference, scale = _resize_gray(frames[0], max_width)
    first_track = track_rows[0] if track_rows else None
    pivot = _pendulum_pivot_uv(
        calibration,
        (frames[0].shape[1], frames[0].shape[0]),
        first_track,
    )
    allowed = _exclusion_mask(reference.shape, track_rows, scale, pivot)
    points0 = _grid_features(reference, allowed)
    if points0 is None or len(points0) < 24:
        base["reason"] = "insufficient_background_features_after_dynamic_exclusion"
        base["reference_feature_count"] = 0 if points0 is None else int(len(points0))
        return base

    height, width = reference.shape
    diagonal_original = math.hypot(frames[0].shape[1], frames[0].shape[0])
    strong_residual_scaled = max(6.0, 0.008 * diagonal_original) * scale
    global_p95_scaled = max(8.0, 0.010 * diagonal_original) * scale
    persistent_local = 0
    persistent_global = 0
    successful = 0
    for frame_index in _sample_indices(len(frames), sample_count):
        current, current_scale = _resize_gray(frames[frame_index], max_width)
        if current.shape != reference.shape or abs(current_scale - scale) > 1e-9:
            continue
        p1, status1, _ = cv2.calcOpticalFlowPyrLK(
            reference,
            current,
            points0,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
        )
        if p1 is None or status1 is None:
            continue
        back, status2, _ = cv2.calcOpticalFlowPyrLK(
            current,
            reference,
            p1,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
        )
        if back is None or status2 is None:
            continue
        p0_all = points0.reshape(-1, 2)
        p1_all = p1.reshape(-1, 2)
        back_all = back.reshape(-1, 2)
        good = status1.ravel().astype(bool) & status2.ravel().astype(bool)
        good &= np.linalg.norm(p0_all - back_all, axis=1) <= 1.5
        p0 = p0_all[good]
        p1_good = p1_all[good]
        if len(p0) < 18:
            continue
        homography, inlier_mask = cv2.findHomography(
            p0,
            p1_good,
            cv2.RANSAC,
            1.5,
            maxIters=2500,
            confidence=0.997,
        )
        if homography is None or inlier_mask is None:
            continue
        predicted = cv2.perspectiveTransform(
            p0.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        residual = np.linalg.norm(p1_good - predicted, axis=1)
        inlier = inlier_mask.ravel().astype(bool)
        ncc = np.asarray(
            [_patch_ncc(reference, current, a, b) for a, b in zip(p0, p1_good)],
            dtype=float,
        )
        local_bad = (residual >= strong_residual_scaled) & (ncc < 0.60)
        local_count = int(local_bad.sum())
        local_cells = _cell_count(p0[local_bad], width, height)
        local_event = local_count >= 8 and local_cells >= 2
        residual_p95 = float(np.percentile(residual, 95))
        outlier_cells = _cell_count(p0[~inlier], width, height)
        global_event = bool(
            float(inlier.mean()) < 0.50
            and residual_p95 >= global_p95_scaled
            and outlier_cells >= 8
        )
        persistent_local += int(local_event)
        persistent_global += int(global_event)
        successful += 1
        base["sample_rows"].append(
            {
                "frame_index": frame_index,
                "tracked_features": int(len(p0)),
                "homography_inlier_fraction": float(inlier.mean()),
                "residual_p95_px": residual_p95 / scale,
                "local_bad_feature_count": local_count,
                "local_bad_spatial_cells": local_cells,
                "local_event": local_event,
                "global_event": global_event,
            }
        )

    base.update(
        {
            "reference_feature_count": int(len(points0)),
            "successful_sample_count": successful,
            "persistent_local_event_count": persistent_local,
            "persistent_global_event_count": persistent_global,
            "thresholds": {
                "sample_count": sample_count,
                "minimum_confirming_samples": 3,
                "local_residual_px": strong_residual_scaled / scale,
                "local_patch_ncc_max": 0.60,
                "local_min_features": 8,
                "local_min_spatial_cells": 2,
                "global_inlier_fraction_max": 0.50,
                "global_residual_p95_px": global_p95_scaled / scale,
                "global_min_outlier_cells_of_24": 8,
            },
        }
    )
    if persistent_local >= 3 or persistent_global >= 3:
        base.update(
            {
                "status": "fail",
                "hard_failure": True,
                "failure_code": "background_or_prop_nonrigid_deformation",
            }
        )
    elif successful < 3:
        base["reason"] = "insufficient_successful_scene_rigidity_samples"
    else:
        base["status"] = "pass"
    return base


__all__ = ["assess_static_scene_rigidity"]
