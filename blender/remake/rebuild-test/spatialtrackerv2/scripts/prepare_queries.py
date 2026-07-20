#!/usr/bin/env python3
"""Prepare ball and static-scene queries for one generated V1A video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def load_first_frame(video_path: Path) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame 0: {video_path}")
    return frame, fps, frames


def estimate_registration(reference_bgr: np.ndarray, frame0_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate a conservative reference-resized -> generated-frame homography."""
    height, width = frame0_bgr.shape[:2]
    resized = cv2.resize(reference_bgr, (width, height), interpolation=cv2.INTER_AREA)
    ref_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=8)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(dst_gray, None)
    identity = np.eye(3, dtype=np.float64)
    stats = {"method": "identity_fallback", "matches": 0, "inliers": 0, "accepted": False}
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return identity, stats

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    stats["matches"] = len(good)
    if len(good) < 12:
        return identity, stats
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
    if matrix is None or not np.isfinite(matrix).all():
        return identity, stats
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    stats["inliers"] = inliers
    stats["inlier_fraction"] = inliers / max(1, len(good))

    corners = np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]])
    warped = cv2.perspectiveTransform(corners, matrix)[0]
    displacement = np.linalg.norm(warped - corners[0], axis=1)
    max_displacement = float(displacement.max())
    stats["max_corner_displacement_px"] = max_displacement
    # Frame 0 is the frozen conditioning image. Reject implausible projective fits.
    accepted = inliers >= 12 and stats["inlier_fraction"] >= 0.35 and max_displacement < 0.12 * math.hypot(width, height)
    if not accepted:
        return identity, stats
    stats["method"] = "orb_homography"
    stats["accepted"] = True
    return matrix.astype(np.float64), stats


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts.astype(np.float32), matrix.astype(np.float64)).reshape(-1, 2)


def disk_points(center: np.ndarray, radii: np.ndarray, count: int) -> np.ndarray:
    """Deterministic sunflower sampling inside 78% of the projected ball."""
    points = [center.astype(np.float64)]
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(1, count):
        radius = 0.78 * math.sqrt(index / max(1, count - 1))
        theta = index * golden_angle
        offset = np.array([math.cos(theta), math.sin(theta)]) * radii * radius
        points.append(center + offset)
    return np.asarray(points, dtype=np.float32)


def farthest_point_subset(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return np.arange(len(points), dtype=np.int64)
    # Cap the candidate pool deterministically to keep the O(NK) selection cheap.
    if len(points) > 6000:
        pool = np.linspace(0, len(points) - 1, 6000, dtype=np.int64)
        candidate = points[pool]
    else:
        pool = np.arange(len(points), dtype=np.int64)
        candidate = points
    chosen = [int(np.argmin(np.linalg.norm(candidate - np.median(candidate, axis=0), axis=1)))]
    min_dist = np.full(len(candidate), np.inf, dtype=np.float64)
    for _ in range(1, count):
        last = candidate[chosen[-1]]
        min_dist = np.minimum(min_dist, np.sum((candidate - last) ** 2, axis=1))
        min_dist[chosen] = -1.0
        chosen.append(int(np.argmax(min_dist)))
    return pool[np.asarray(chosen, dtype=np.int64)]


def ray_sphere_intersections(
    uv_source: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    camera = calibration["camera"]
    obj = calibration["object"]
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    rotation_world_to_camera = np.asarray(camera["R_world_to_camera_opencv"], dtype=np.float64)
    translation = np.asarray(camera["t_world_to_camera_opencv_m"], dtype=np.float64)
    camera_world = -rotation_world_to_camera.T @ translation
    center = np.asarray(obj["initial_center_world_m"], dtype=np.float64)
    radius = float(obj["known_radius_m"])
    inv_intrinsic = np.linalg.inv(intrinsic)
    output = []
    for u, v in uv_source:
        ray_camera = inv_intrinsic @ np.asarray([u, v, 1.0], dtype=np.float64)
        ray_world = rotation_world_to_camera.T @ ray_camera
        ray_world /= np.linalg.norm(ray_world)
        relative = camera_world - center
        b = 2.0 * float(np.dot(ray_world, relative))
        c = float(np.dot(relative, relative) - radius * radius)
        discriminant = b * b - 4.0 * c
        if discriminant < 0:
            # Numerical/projected-bbox edge fallback: closest point on the sphere.
            closest_t = max(0.0, -0.5 * b)
            closest = camera_world + closest_t * ray_world
            direction = closest - center
            direction /= max(np.linalg.norm(direction), 1e-9)
            output.append(center + radius * direction)
            continue
        root = math.sqrt(discriminant)
        candidates = [(-b - root) / 2.0, (-b + root) / 2.0]
        positive = [value for value in candidates if value > 0]
        distance = min(positive) if positive else max(candidates)
        output.append(camera_world + distance * ray_world)
    return np.asarray(output, dtype=np.float32)


def prepare_queries(
    video_path: Path,
    first_frame_path: Path,
    calibration_path: Path,
    output_path: Path,
    diagnostic_path: Path,
    object_points: int = 64,
    anchor_points: int = 128,
) -> dict:
    frame0, detected_fps, frame_count = load_first_frame(video_path)
    reference = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"Cannot read first frame: {first_frame_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    source_width = int(calibration["image"]["width_px"])
    source_height = int(calibration["image"]["height_px"])
    height, width = frame0.shape[:2]

    registration, registration_stats = estimate_registration(reference, frame0)
    resize_source_to_video = np.asarray(
        [[width / source_width, 0.0, 0.0], [0.0, height / source_height, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    source_to_video = registration @ resize_source_to_video

    validation = calibration["projection_validation"]
    center_source = np.asarray(validation["initial_center_uv_from_opencv_P"], dtype=np.float32)
    bbox = np.asarray(validation["initial_mesh_vertex_bbox_xyxy_px"], dtype=np.float32)
    radii_source = np.asarray([(bbox[2] - bbox[0]) / 2.0, (bbox[3] - bbox[1]) / 2.0], dtype=np.float32)
    object_source = disk_points(center_source, radii_source, object_points)
    object_video = transform_points(object_source, source_to_video)
    object_initial_xyz = ray_sphere_intersections(object_source, calibration)

    anchor_relative = Path(calibration["reference_geometry_anchors"]["path"])
    calibration_root = calibration_path.parent.parent
    anchor_path = calibration_root / anchor_relative
    anchor_data = np.load(anchor_path, allow_pickle=True)
    anchor_uv_source = np.asarray(anchor_data["uv_px"], dtype=np.float32)
    anchor_xyz = np.asarray(anchor_data["xyz_world_m"], dtype=np.float32)
    anchor_uv_video = transform_points(anchor_uv_source, source_to_video)
    in_bounds = (
        (anchor_uv_video[:, 0] >= 8)
        & (anchor_uv_video[:, 0] < width - 8)
        & (anchor_uv_video[:, 1] >= 8)
        & (anchor_uv_video[:, 1] < height - 8)
    )
    object_center_video = transform_points(center_source[None], source_to_video)[0]
    object_radius_video = np.maximum(
        np.linalg.norm(transform_points(np.asarray([center_source, center_source + [radii_source[0], 0]], dtype=np.float32), source_to_video)[1] - object_center_video),
        4.0,
    )
    away_from_ball = np.linalg.norm(anchor_uv_video - object_center_video, axis=1) > 2.5 * object_radius_video
    valid_anchor_indices = np.flatnonzero(in_bounds & away_from_ball)
    if len(valid_anchor_indices) < 12:
        raise RuntimeError(f"Only {len(valid_anchor_indices)} usable static anchors for {calibration_path}")
    subset_local = farthest_point_subset(anchor_uv_video[valid_anchor_indices], anchor_points)
    selected = valid_anchor_indices[subset_local]
    selected_anchor_video = anchor_uv_video[selected]
    selected_anchor_xyz = anchor_xyz[selected]

    query_xy = np.concatenate([object_video, selected_anchor_video], axis=0).astype(np.float32)
    query_kind = np.asarray(["object"] * len(object_video) + ["anchor"] * len(selected), dtype="U8")
    anchor_xyz_full = np.full((len(query_xy), 3), np.nan, dtype=np.float32)
    anchor_xyz_full[len(object_video) :] = selected_anchor_xyz
    object_xyz_full = np.full((len(query_xy), 3), np.nan, dtype=np.float32)
    object_xyz_full[: len(object_video)] = object_initial_xyz

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        query_xy_video=query_xy,
        query_kind=query_kind,
        anchor_xyz_world_m=anchor_xyz_full,
        object_xyz_world_initial_m=object_xyz_full,
        object_initial_center_world_m=np.asarray(calibration["object"]["initial_center_world_m"], dtype=np.float32),
        known_radius_m=np.asarray(calibration["object"]["known_radius_m"], dtype=np.float32),
        source_to_video_homography=source_to_video.astype(np.float64),
        video_frame_size_hw=np.asarray([height, width], dtype=np.int32),
        detected_fps=np.asarray(detected_fps, dtype=np.float32),
        detected_frame_count=np.asarray(frame_count, dtype=np.int32),
    )

    diagnostic = frame0.copy()
    for index, (x, y) in enumerate(query_xy):
        colour = (0, 180, 255) if query_kind[index] == "object" else (255, 180, 0)
        cv2.circle(diagnostic, (int(round(x)), int(round(y))), 2, colour, -1, cv2.LINE_AA)
    cv2.putText(
        diagnostic,
        f"object={len(object_video)} anchors={len(selected)} reg={registration_stats['method']}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(diagnostic_path), diagnostic)

    metadata = {
        "video": str(video_path),
        "frame_size_hw": [height, width],
        "detected_fps": detected_fps,
        "frame_count": frame_count,
        "object_queries": len(object_video),
        "anchor_queries": len(selected),
        "registration": registration_stats,
        "anchor_file": str(anchor_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--object-points", type=int, default=64)
    parser.add_argument("--anchor-points", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_queries(
        args.video,
        args.first_frame,
        args.calibration,
        args.output,
        args.diagnostic,
        args.object_points,
        args.anchor_points,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
