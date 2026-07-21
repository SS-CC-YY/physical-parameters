#!/usr/bin/env python3
"""Convert SpatialTrackerV2 point tracks into a metric ball trajectory.

Calibrated Blender anchors remain the preferred absolute reference.  When an
all-experiment sidecar has no such point cloud, frame-0 object correspondences
establish metric scale/world orientation and frame-0 background points become
self-reference anchors.  A per-frame robust Sim(3) then removes global camera
drift before object motion is estimated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def fit_sim3(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    src = source - source_mean
    dst = target - target_mean
    covariance = (dst.T @ src) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(src * src, axis=1)))
    if variance < 1e-12:
        raise ValueError("Degenerate Sim(3) source points")
    scale = float(np.sum(singular * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_sim3(points: np.ndarray, model: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = model
    return scale * (np.asarray(points) @ rotation.T) + translation


def robust_sim3(
    source: np.ndarray,
    target: np.ndarray,
    threshold_m: float,
    rng: np.random.Generator,
    iterations: int = 256,
) -> tuple[tuple[float, np.ndarray, np.ndarray], np.ndarray, float]:
    if len(source) < 4:
        raise ValueError("At least four anchors are required")
    best_inliers = None
    best_rmse = math.inf
    for _ in range(iterations):
        sample = rng.choice(len(source), size=4, replace=False)
        try:
            model = fit_sim3(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = np.linalg.norm(apply_sim3(source, model) - target, axis=1)
        inliers = residual <= threshold_m
        if inliers.sum() < 4:
            continue
        rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
        if best_inliers is None or inliers.sum() > best_inliers.sum() or (
            inliers.sum() == best_inliers.sum() and rmse < best_rmse
        ):
            best_inliers = inliers
            best_rmse = rmse
    if best_inliers is None:
        raise ValueError("RANSAC could not fit Sim(3)")
    model = fit_sim3(source[best_inliers], target[best_inliers])
    residual = np.linalg.norm(apply_sim3(source, model) - target, axis=1)
    inliers = residual <= threshold_m
    if inliers.sum() >= 4:
        model = fit_sim3(source[inliers], target[inliers])
        residual = np.linalg.norm(apply_sim3(source, model) - target, axis=1)
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    return model, inliers, rmse


def fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    keep = np.ones(len(source), dtype=bool)
    rotation = np.eye(3)
    translation = np.zeros(3)
    for _ in range(4):
        src = source[keep]
        dst = target[keep]
        if len(src) < 3:
            raise ValueError("At least three object points are required")
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        u, _, vt = np.linalg.svd((dst - dst_mean).T @ (src - src_mean))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        translation = dst_mean - rotation @ src_mean
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        threshold = max(0.06, median + 3.5 * 1.4826 * mad)
        updated = residual <= threshold
        if updated.sum() < 3 or np.array_equal(updated, keep):
            break
        keep = updated
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    return rotation, translation, rmse, keep


def normalize_tn(array: np.ndarray, frames: int, points: int) -> np.ndarray:
    value = np.asarray(array)
    value = np.squeeze(value)
    if value.shape == (frames, points):
        return value
    if value.shape == (points, frames):
        return value.T
    if value.ndim == 1 and len(value) == points:
        return np.repeat(value[None], frames, axis=0)
    raise ValueError(f"Cannot normalize array shape {value.shape} to ({frames}, {points})")


def rolling_median(values: np.ndarray, radius: int = 2) -> np.ndarray:
    result = values.copy()
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        for axis in range(3):
            finite = values[lo:hi, axis]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                result[index, axis] = np.median(finite)
    return result


def scalar_text(value: object, default: str) -> str:
    if value is None:
        return default
    array = np.asarray(value)
    if array.size == 0:
        return default
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def occupied_grid_cells(points_uv: np.ndarray, frame_size_hw: np.ndarray, rows: int = 4, cols: int = 6) -> int:
    points = np.asarray(points_uv, dtype=np.float64)
    if not len(points):
        return 0
    height, width = np.asarray(frame_size_hw, dtype=np.float64)
    x = np.clip((points[:, 0] / max(width, 1.0) * cols).astype(int), 0, cols - 1)
    y = np.clip((points[:, 1] / max(height, 1.0) * rows).astype(int), 0, rows - 1)
    return len(set(zip(y.tolist(), x.tolist())))


def make_overlay(
    video_path: Path,
    output_path: Path,
    raw: dict,
    object_mask: np.ndarray,
    visible: np.ndarray,
) -> None:
    tracks = np.asarray(raw["track2d_input"], dtype=np.float64)[..., :2]
    scale_x, scale_y = np.asarray(raw["preprocess_scale_xy"], dtype=np.float64)
    crop_top = float(np.asarray(raw["preprocess_crop_top"]).reshape(()))
    tracks[..., 0] /= scale_x
    tracks[..., 1] = (tracks[..., 1] + crop_top) / scale_y
    source_indices = np.asarray(raw["source_frame_indices"], dtype=int)
    source_fps = float(np.asarray(raw["source_fps"]).reshape(()))
    stride = int(np.median(np.diff(source_indices))) if len(source_indices) > 1 else 1

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, source_fps / stride),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create overlay: {output_path}")
    centre_history: list[tuple[int, int]] = []
    object_indices = np.flatnonzero(object_mask)
    for time_index, source_index in enumerate(source_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
        ok, frame = cap.read()
        if not ok:
            break
        current = []
        for point_index in object_indices:
            if not visible[time_index, point_index]:
                continue
            x, y = tracks[time_index, point_index]
            if not np.isfinite([x, y]).all():
                continue
            pixel = (int(round(x)), int(round(y)))
            current.append(pixel)
            cv2.circle(frame, pixel, 3, (0, 190, 255), -1, cv2.LINE_AA)
        if current:
            centre = tuple(np.median(np.asarray(current), axis=0).astype(int))
            centre_history.append(centre)
            cv2.circle(frame, centre, 7, (0, 0, 255), 2, cv2.LINE_AA)
        for start, end in zip(centre_history[:-1], centre_history[1:]):
            cv2.line(frame, start, end, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"frame={source_index} visible_ball_queries={len(current)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    cap.release()


def make_plot(times: np.ndarray, trajectory: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    finite = np.isfinite(trajectory).all(axis=1)
    if finite.any():
        ax3d.plot(trajectory[finite, 0], trajectory[finite, 1], trajectory[finite, 2], color="#d62728")
        ax3d.scatter(*trajectory[finite][0], color="#2ca02c", s=45, label="start")
        ax3d.scatter(*trajectory[finite][-1], color="#1f77b4", s=45, label="end")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("Metric 3D ball-centre trajectory")
    ax3d.legend(loc="best")

    axis = fig.add_subplot(1, 2, 2)
    for index, label in enumerate(["x", "y", "z"]):
        axis.plot(times, trajectory[:, index], label=f"{label} (m)")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("position (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title("World-coordinate components")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def postprocess(
    raw_path: Path,
    video_path: Path,
    output_dir: Path,
    align_threshold_m: float = 0.30,
) -> dict:
    raw = dict(np.load(raw_path, allow_pickle=True))
    coords = np.asarray(raw["coords"], dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")
    frames, points, _ = coords.shape
    query_kind = np.asarray(raw["query_meta_query_kind"]).astype(str)
    anchor_target = np.asarray(raw["query_meta_anchor_xyz_world_m"], dtype=np.float64)
    object_target = np.asarray(raw["query_meta_object_xyz_world_initial_m"], dtype=np.float64)
    initial_center = np.asarray(raw["query_meta_object_initial_center_world_m"], dtype=np.float64)
    known_radius = float(np.asarray(raw["query_meta_known_radius_m"]).reshape(()))
    anchor_reference_mode = scalar_text(
        raw.get("query_meta_anchor_reference_mode"), "calibrated_world"
    )
    object_mask = query_kind == "object"
    anchor_mask = query_kind == "anchor"
    if object_mask.sum() < 3 or anchor_mask.sum() < 4:
        raise ValueError("Insufficient object or anchor queries in raw result")
    visible = normalize_tn(raw["visibs"], frames, points).astype(bool)
    confidence = normalize_tn(raw["track_confidence"], frames, points).astype(np.float64)
    confidence_floor = np.nanquantile(confidence, 0.15) if np.isfinite(confidence).any() else -np.inf

    rng = np.random.default_rng(20260720)
    working_coords = coords.copy()
    initial_alignment = None
    if anchor_reference_mode == "self_frame0":
        # The self-reference point cloud has arbitrary scale/orientation.  The
        # known spherical surface in frame 0 supplies the metric Sim(3).
        initial_usable = (
            object_mask
            & visible[0]
            & np.isfinite(coords[0]).all(axis=1)
            & np.isfinite(object_target).all(axis=1)
            & (confidence[0] >= confidence_floor)
        )
        if initial_usable.sum() < 6:
            raise RuntimeError(
                f"Frame 0 has only {int(initial_usable.sum())} usable metric object correspondences"
            )
        initial_model, initial_inliers, initial_rmse = robust_sim3(
            coords[0, initial_usable],
            object_target[initial_usable],
            max(0.04, 0.35 * known_radius),
            rng,
        )
        if initial_inliers.sum() < 6:
            raise RuntimeError("Frame-0 metric Sim(3) has fewer than six object inliers")
        working_coords = apply_sim3(coords, initial_model)
        anchor_target = working_coords[0].copy()
        anchor_target[~anchor_mask] = np.nan
        anchor_target[~visible[0]] = np.nan
        initial_alignment = {
            "object_correspondences": int(initial_usable.sum()),
            "object_inliers": int(initial_inliers.sum()),
            "rmse_m": float(initial_rmse),
            "scale": float(initial_model[0]),
        }
    elif anchor_reference_mode != "calibrated_world":
        raise ValueError(f"Unsupported anchor_reference_mode: {anchor_reference_mode}")

    query_uv = np.asarray(raw.get("query_meta_query_xy_video", np.full((points, 2), np.nan)), dtype=np.float64)
    frame_size_hw = np.asarray(
        raw.get("query_meta_video_frame_size_hw", raw.get("source_frame_size_hw", [1, 1])),
        dtype=np.int32,
    )
    aligned = np.full_like(coords, np.nan)
    alignment_rows = []
    last_model = None
    for frame in range(frames):
        usable = anchor_mask & visible[frame] & np.isfinite(working_coords[frame]).all(axis=1) & np.isfinite(anchor_target).all(axis=1)
        usable &= confidence[frame] >= confidence_floor
        fallback = False
        inlier_fraction = 0.0
        outlier_grid_cells = 0
        try:
            if usable.sum() < 4:
                raise ValueError("too few anchors")
            model, inliers, rmse = robust_sim3(
                working_coords[frame, usable], anchor_target[usable], align_threshold_m, rng
            )
            if inliers.sum() < 6:
                raise ValueError("too few Sim(3) inliers")
            last_model = model
            inlier_count = int(inliers.sum())
            inlier_fraction = inlier_count / int(usable.sum())
            usable_indices = np.flatnonzero(usable)
            outlier_grid_cells = occupied_grid_cells(
                query_uv[usable_indices[~inliers]], frame_size_hw
            )
        except (ValueError, np.linalg.LinAlgError):
            if last_model is None:
                alignment_rows.append(
                    {
                        "frame": frame,
                        "anchors": int(usable.sum()),
                        "inliers": 0,
                        "inlier_fraction": 0.0,
                        "outlier_grid_cells": 0,
                        "rmse_m": None,
                        "fallback": True,
                    }
                )
                continue
            model = last_model
            rmse = math.nan
            inlier_count = 0
            fallback = True
        aligned[frame] = apply_sim3(working_coords[frame], model)
        alignment_rows.append(
            {
                "frame": frame,
                "anchors": int(usable.sum()),
                "inliers": inlier_count,
                "inlier_fraction": inlier_fraction,
                "outlier_grid_cells": outlier_grid_cells,
                "rmse_m": None if not math.isfinite(rmse) else rmse,
                "scale": float(model[0]),
                "fallback": fallback,
            }
        )

    valid_reference = object_mask & visible[0] & np.isfinite(aligned[0]).all(axis=1)
    if valid_reference.sum() < 3:
        raise RuntimeError("Frame 0 has fewer than three aligned ball queries")
    reference_indices = np.flatnonzero(valid_reference)
    reference_cloud = aligned[0, reference_indices]
    trajectory = np.full((frames, 3), np.nan, dtype=np.float64)
    object_rmse = np.full(frames, np.nan, dtype=np.float64)
    object_all_rmse = np.full(frames, np.nan, dtype=np.float64)
    object_inliers = np.zeros(frames, dtype=np.int32)
    object_visible_count = np.zeros(frames, dtype=np.int32)
    object_inlier_fraction = np.full(frames, np.nan, dtype=np.float64)
    trajectory[0] = initial_center
    object_visible_count[0] = int(valid_reference.sum())
    object_inlier_fraction[0] = 1.0
    for frame in range(1, frames):
        valid = visible[frame, reference_indices] & np.isfinite(aligned[frame, reference_indices]).all(axis=1)
        object_visible_count[frame] = int(valid.sum())
        if valid.sum() < 3:
            continue
        try:
            rotation, translation, rmse, keep = fit_rigid(reference_cloud[valid], aligned[frame, reference_indices[valid]])
        except (ValueError, np.linalg.LinAlgError):
            continue
        trajectory[frame] = rotation @ initial_center + translation
        object_rmse[frame] = rmse
        predicted = reference_cloud[valid] @ rotation.T + translation
        all_residual = np.linalg.norm(predicted - aligned[frame, reference_indices[valid]], axis=1)
        object_all_rmse[frame] = float(np.sqrt(np.mean(all_residual**2)))
        object_inliers[frame] = int(keep.sum())
        object_inlier_fraction[frame] = float(keep.mean())
    object_inliers[0] = int(valid_reference.sum())
    object_rmse[0] = 0.0
    object_all_rmse[0] = 0.0
    smoothed = rolling_median(trajectory, radius=2)
    smoothed[0] = initial_center

    source_indices = np.asarray(raw["source_frame_indices"], dtype=np.int32)
    source_fps = float(np.asarray(raw["source_fps"]).reshape(()))
    times = source_indices.astype(np.float64) / source_fps
    csv_path = output_dir / "trajectory_world.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index", "source_frame", "time_s", "x_m", "y_m", "z_m",
                "x_raw_m", "y_raw_m", "z_raw_m", "object_visible_queries",
                "object_inliers", "object_inlier_fraction", "object_rigid_rmse_m",
                "object_all_point_rmse_m", "object_all_point_rmse_over_radius",
            ]
        )
        for index in range(frames):
            writer.writerow(
                [
                    index,
                    int(source_indices[index]),
                    f"{times[index]:.9g}",
                    *["" if not math.isfinite(v) else f"{v:.9g}" for v in smoothed[index]],
                    *["" if not math.isfinite(v) else f"{v:.9g}" for v in trajectory[index]],
                    int(object_visible_count[index]),
                    int(object_inliers[index]),
                    "" if not math.isfinite(object_inlier_fraction[index]) else f"{object_inlier_fraction[index]:.9g}",
                    "" if not math.isfinite(object_rmse[index]) else f"{object_rmse[index]:.9g}",
                    "" if not math.isfinite(object_all_rmse[index]) else f"{object_all_rmse[index]:.9g}",
                    "" if not math.isfinite(object_all_rmse[index]) else f"{object_all_rmse[index] / known_radius:.9g}",
                ]
            )

    make_plot(times, smoothed, output_dir / "trajectory_3d.png")
    make_overlay(video_path, output_dir / "object_track_overlay.mp4", raw, object_mask, visible)
    (output_dir / "alignment.json").write_text(json.dumps(alignment_rows, indent=2), encoding="utf-8")
    finite = np.isfinite(smoothed).all(axis=1)
    nonfallback_alignment = sum(not row.get("fallback", True) for row in alignment_rows)
    finite_rmse = [row["rmse_m"] for row in alignment_rows if row.get("rmse_m") is not None]
    object_visible_fraction = object_visible_count.astype(np.float64) / max(1, len(reference_indices))
    object_normalized_rmse = object_all_rmse / max(known_radius, 1e-9)
    object_informative = (
        (object_visible_count >= max(6, int(math.ceil(0.25 * len(reference_indices)))))
        & np.isfinite(object_normalized_rmse)
        & np.isfinite(object_inlier_fraction)
    )
    # Require both poor rigid explainability and a large sphere-normalized
    # residual, or an exceptionally large residual.  Persistence prevents a
    # single occlusion/contact frame from becoming a deformation verdict.
    object_suspicious = object_informative & (
        ((object_normalized_rmse > 0.45) & (object_inlier_fraction < 0.70))
        | (object_normalized_rmse > 0.85)
    )
    object_deformation_failed = bool(
        longest_true_run(object_suspicious) >= 3
        and object_suspicious.sum() >= max(3, int(math.ceil(0.20 * max(1, object_informative.sum()))))
    )

    background_suspicious = np.asarray(
        [
            not row.get("fallback", True)
            and row.get("anchors", 0) >= 12
            and row.get("inlier_fraction", 0.0) < 0.45
            and row.get("outlier_grid_cells", 0) >= 8
            for row in alignment_rows
        ],
        dtype=bool,
    )
    background_deformation_failed = bool(
        longest_true_run(background_suspicious) >= 3
        and background_suspicious.sum() >= max(3, int(math.ceil(0.20 * frames)))
    )
    generation_failure_reasons = []
    if object_deformation_failed:
        generation_failure_reasons.append("experimental_object_deformation")
    if background_deformation_failed:
        generation_failure_reasons.append("non_experimental_scene_deformation")
    review_reasons = []
    if object_informative.sum() < max(3, int(math.ceil(0.50 * frames))):
        review_reasons.append("insufficient_visible_object_queries")
    if nonfallback_alignment / frames < 0.75:
        review_reasons.append("insufficient_direct_background_alignment")
    generation_validity_status = (
        "failed"
        if generation_failure_reasons
        else "review"
        if review_reasons
        else "pass"
    )
    generation_validity = {
        "status": generation_validity_status,
        "failure_reasons": generation_failure_reasons,
        "review_reasons": review_reasons,
        "skip_physics_fit": bool(generation_failure_reasons),
        "thresholds": {
            "object_rmse_over_radius": 0.45,
            "object_severe_rmse_over_radius": 0.85,
            "object_inlier_fraction": 0.70,
            "background_inlier_fraction": 0.45,
            "background_outlier_grid_cells": 8,
            "minimum_consecutive_suspicious_frames": 3,
            "minimum_suspicious_fraction": 0.20,
        },
        "object": {
            "reference_queries": int(len(reference_indices)),
            "median_visible_fraction": float(np.median(object_visible_fraction)),
            "median_inlier_fraction": float(np.nanmedian(object_inlier_fraction)),
            "median_rigid_rmse_over_radius": float(np.nanmedian(object_normalized_rmse)),
            "p90_rigid_rmse_over_radius": float(np.nanquantile(object_normalized_rmse, 0.90)),
            "suspicious_frames": int(object_suspicious.sum()),
            "suspicious_source_frames": source_indices[object_suspicious].astype(int).tolist(),
            "longest_suspicious_run": int(longest_true_run(object_suspicious)),
            "deformation_failed": object_deformation_failed,
        },
        "background": {
            "registration_model": "per_frame_robust_sim3_to_static_reference",
            "camera_drift_removed_before_residual_test": True,
            "suspicious_frames": int(background_suspicious.sum()),
            "suspicious_source_frames": source_indices[background_suspicious].astype(int).tolist(),
            "longest_suspicious_run": int(longest_true_run(background_suspicious)),
            "deformation_failed": background_deformation_failed,
        },
        "decision_note": (
            "Only persistent, spatially broad rigidity violations are hard failures; "
            "isolated occlusion/contact frames and global camera drift are not."
        ),
    }
    (output_dir / "generation_validity.json").write_text(
        json.dumps(generation_validity, indent=2), encoding="utf-8"
    )
    rigidity_evidence = {
        "object_frames": [
            {
                "frame_index": index,
                "source_frame": int(source_indices[index]),
                "normalized_rmse": (
                    None
                    if not math.isfinite(object_normalized_rmse[index])
                    else float(object_normalized_rmse[index])
                ),
                "inlier_fraction": (
                    None
                    if not math.isfinite(object_inlier_fraction[index])
                    else float(object_inlier_fraction[index])
                ),
                "visible_fraction": float(object_visible_fraction[index]),
            }
            for index in range(frames)
        ],
        "background_frames": [
            {
                "frame_index": index,
                "source_frame": int(source_indices[index]),
                "normalized_rmse": (
                    None
                    if row.get("rmse_m") is None
                    else float(row["rmse_m"]) / max(known_radius, 1e-9)
                ),
                "inlier_fraction": float(row.get("inlier_fraction", 0.0)),
                "spatial_coverage_fraction": float(row.get("outlier_grid_cells", 0)) / 24.0,
                "fallback": bool(row.get("fallback", True)),
            }
            for index, row in enumerate(alignment_rows)
        ],
        "coordinate_policy": (
            "frame0 metric sphere Sim(3), then per-frame static-background Sim(3) camera compensation"
            if anchor_reference_mode == "self_frame0"
            else "per-frame Sim(3) to calibrated Blender static anchors"
        ),
    }
    summary = {
        "status": "succeeded",
        "frames": frames,
        "source_fps": source_fps,
        "anchor_reference_mode": anchor_reference_mode,
        "initial_metric_alignment": initial_alignment,
        "trajectory_valid_frames": int(finite.sum()),
        "trajectory_valid_fraction": float(finite.mean()),
        "alignment_direct_frames": nonfallback_alignment,
        "alignment_direct_fraction": nonfallback_alignment / frames,
        "median_anchor_rmse_m": float(np.median(finite_rmse)) if finite_rmse else None,
        "median_object_rigid_rmse_m": float(np.nanmedian(object_rmse)),
        "median_object_visible_fraction": float(np.median(object_visible_fraction)),
        "median_object_inlier_fraction": float(np.nanmedian(object_inlier_fraction)),
        "median_object_rigid_rmse_over_radius": float(np.nanmedian(object_normalized_rmse)),
        "generation_validity": generation_validity,
        "rigidity_evidence": rigidity_evidence,
        "quality_pass": bool(
            finite.mean() >= 0.80
            and nonfallback_alignment / frames >= 0.75
            and finite_rmse
            and float(np.median(finite_rmse)) <= align_threshold_m
            and generation_validity_status != "failed"
        ),
        "quality_note": "Self-consistency only; no Blender ground-truth video comparison was performed.",
        "trajectory_csv": str(csv_path),
        "overlay_video": str(output_dir / "object_track_overlay.mp4"),
        "trajectory_plot": str(output_dir / "trajectory_3d.png"),
    }
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--align-threshold-m", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(postprocess(args.raw, args.video, args.output_dir, args.align_threshold_m), indent=2))


if __name__ == "__main__":
    main()
