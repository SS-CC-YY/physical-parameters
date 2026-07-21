"""Fixed-camera standard-sphere tracking and metric trajectory recovery.

The first-frame Blender sidecar supplies the ground-truth sphere center,
physical radius, projected radius, and camera K/R/t.  Generated frame zero is
used only to estimate a fixed image translation and segmentation colour.  Each
later frame is lifted independently while K/R/t remain fixed.

All frozen experiments move in the Blender world plane y=0 (or an equivalent
constant-y plane).  Center-ray/plane intersection therefore gives the main
metric reconstruction.  A sphere-size residual, normalized to the GT first
frame, refines x/z and makes the requested first-frame size constraint explicit.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .v1a_geometry import crop_resize_homography
from .v1a_tracker import estimate_ball_colour, orange_ball_mask, read_video_frames


@dataclass(frozen=True)
class SphereCandidate:
    center_u_px: float
    center_v_px: float
    radius_px: float
    area_px: float
    circularity: float
    ellipse_major_axis_px: float
    ellipse_minor_axis_px: float
    ellipse_angle_deg: float
    ellipse_axis_ratio: float
    radial_residual_ratio: float
    hue_distance: float
    boundary: bool
    measurement_source: str


LINE_MANIFOLDS = {
    "v1_A": "vertical",
    "v1_B": "horizontal",
    "v1_C": "horizontal",
    "v2_B": "horizontal",
    "v2_C": "horizontal",
    "v2_E": "vertical",
    "v3_B": "horizontal",
}

PENDULUMS = {
    "v1_D": {"pivot": (0.0, 0.0, 3.8), "length": 2.3},
    "v2_D": {"pivot": (0.0, 0.0, 3.28), "length": 1.85},
    "v3_D": {"pivot": (0.0, 0.0, 4.05), "length": 1.35},
}


def parse_video_job(video_path: Path) -> dict[str, Any]:
    """Parse the canonical six-field generated-video filename."""

    fields = video_path.stem.split("__")
    if len(fields) < 6:
        raise ValueError(
            "expected <experiment>__<tuple>__<scene>__<object>__<camera>__<seed>.mp4: "
            f"{video_path.name}"
        )
    experiment_id, tuple_id, scene_id, object_id, camera_name = fields[:5]
    seed_text = "__".join(fields[5:])
    if not seed_text.startswith("seed-"):
        raise ValueError(f"missing seed-* filename field: {video_path.name}")
    try:
        seed = int(seed_text.removeprefix("seed-"))
    except ValueError as error:
        raise ValueError(f"invalid seed field in {video_path.name}") from error
    return {
        "experiment_id": experiment_id,
        "parameter_tuple_id": tuple_id,
        "scene_id": scene_id,
        "object_id": object_id,
        "camera_name": camera_name,
        "seed": seed,
        "video_path": str(video_path),
        "video_name": video_path.name,
    }


def calibration_path(calibration_root: Path, job: Mapping[str, Any]) -> Path:
    return (
        calibration_root
        / str(job["experiment_id"])
        / str(job["scene_id"])
        / f"{job['camera_name']}.json"
    )


def load_calibration(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Accept both the original V1A sidecar schema and the all-experiment
    # exporter schema.  Normalize once so geometry code stays schema-agnostic.
    if "source_image" in data and "image" not in data:
        data["image"] = data["source_image"]
    if "standard_ball" in data and "object" not in data:
        ball = data["standard_ball"]
        data["object"] = {
            "object_id": "standard_ball",
            "initial_center_world_m": ball["center_world_m"],
            "known_radius_m": ball["radius_m"],
            "known_diameter_m": ball["diameter_m"],
            "mesh_objects": [ball["selected_mesh_object"]],
        }
    if "frame1_projection" in data:
        projection = data["frame1_projection"]
        validation = data.setdefault("projection_validation", {})
        validation.setdefault("initial_center_uv_from_opencv_P", projection["center_uv_px"])
        validation.setdefault("initial_mesh_vertex_bbox_xyxy_px", projection["mesh_vertex_bbox_xyxy_px"])
    for key in ("camera", "object", "image", "projection_validation"):
        if key not in data:
            raise ValueError(f"calibration missing {key}: {path}")
    return data


def _hue_distance(values: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(values.astype(np.float32) - float(center))
    return np.minimum(delta, 180.0 - delta)


def _enumerate_candidates(
    frame: np.ndarray,
    colour: Mapping[str, float],
    reference_radius_px: float,
    *,
    hue_tolerance: float = 18.0,
) -> list[SphereCandidate]:
    mask = orange_ball_mask(frame, dict(colour), hue_tolerance=hue_tolerance)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[SphereCandidate] = []
    min_radius = max(2.5, 0.28 * reference_radius_px)
    max_radius = max(min_radius + 1.0, 2.4 * reference_radius_px)
    min_area = max(8.0, math.pi * min_radius**2 * 0.35)
    max_area = math.pi * max_radius**2 * 1.8
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (min_area <= area <= max_area):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 1e-9 else 4.0 * math.pi * area / perimeter**2
        if circularity < 0.18:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) <= 1e-9:
            continue
        center_u = float(moments["m10"] / moments["m00"])
        center_v = float(moments["m01"] / moments["m00"])
        if len(contour) >= 5:
            _, axes, ellipse_angle = cv2.fitEllipse(contour)
            major_axis = float(max(axes))
            minor_axis = float(min(axes))
            radius = 0.25 * float(major_axis + minor_axis)
        else:
            _, radius_value = cv2.minEnclosingCircle(contour)
            radius = float(radius_value)
            major_axis = 2.0 * radius
            minor_axis = 2.0 * radius
            ellipse_angle = 0.0
        if not (min_radius <= radius <= max_radius):
            continue
        component = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        hues = hsv[:, :, 0][component > 0]
        distance = float(np.median(_hue_distance(hues, float(colour["hue"])))) if len(hues) else 180.0
        x0, y0, box_width, box_height = cv2.boundingRect(contour)
        boundary = x0 <= 0 or y0 <= 0 or x0 + box_width >= width or y0 + box_height >= height
        radial = np.linalg.norm(
            contour.reshape(-1, 2).astype(np.float64)
            - np.asarray([center_u, center_v], dtype=np.float64),
            axis=1,
        )
        median_radial = float(np.median(radial)) if len(radial) else 0.0
        radial_residual_ratio = (
            float(np.median(np.abs(radial - median_radial))) / median_radial
            if median_radial > 1e-9
            else 0.0
        )
        candidates.append(
            SphereCandidate(
                center_u,
                center_v,
                radius,
                area,
                float(circularity),
                major_axis,
                minor_axis,
                float(ellipse_angle),
                major_axis / max(minor_axis, 1e-6),
                radial_residual_ratio,
                distance,
                boundary,
                "segmentation_contour",
            )
        )
    return candidates


def _hough_candidates(
    frame: np.ndarray,
    colour: Mapping[str, float],
    reference_radius_px: float,
    predicted: np.ndarray | None,
) -> list[SphereCandidate]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(8.0, 1.2 * reference_radius_px),
        param1=110.0,
        param2=23.0,
        minRadius=max(3, int(round(0.45 * reference_radius_px))),
        maxRadius=max(5, int(round(1.8 * reference_radius_px))),
    )
    if circles is None:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output: list[SphereCandidate] = []
    for u, v, radius in circles[0]:
        center = np.asarray([float(u), float(v)])
        if predicted is not None and np.linalg.norm(center - predicted) > 8.0 * reference_radius_px:
            continue
        region = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(region, (int(round(u)), int(round(v))), max(2, int(round(0.7 * radius))), 255, -1)
        vivid = (region > 0) & (hsv[:, :, 1] >= float(colour["saturation_min"]))
        hues = hsv[:, :, 0][vivid]
        if len(hues) < 8:
            continue
        hue_distance = float(np.median(_hue_distance(hues, float(colour["hue"]))))
        if hue_distance > 22.0:
            continue
        output.append(
            SphereCandidate(
                float(u),
                float(v),
                float(radius),
                float(math.pi * radius**2),
                0.95,
                2.0 * float(radius),
                2.0 * float(radius),
                0.0,
                1.0,
                0.0,
                hue_distance,
                bool(u - radius <= 0 or v - radius <= 0 or u + radius >= frame.shape[1] or v + radius >= frame.shape[0]),
                "hough_circle_fallback",
            )
        )
    return output


def _choose_candidate(
    candidates: Iterable[SphereCandidate],
    predicted: np.ndarray,
    previous_radius: float,
    reference_radius: float,
    *,
    gap: int,
) -> tuple[SphereCandidate | None, float]:
    best: SphereCandidate | None = None
    best_cost = float("inf")
    max_distance = (4.5 + 2.0 * min(gap, 4)) * reference_radius
    for item in candidates:
        center = np.asarray([item.center_u_px, item.center_v_px])
        distance = float(np.linalg.norm(center - predicted))
        radius_ratio = item.radius_px / max(reference_radius, 1e-6)
        if distance > max_distance or not (0.28 <= radius_ratio <= 2.4):
            continue
        radius_change = abs(math.log(max(item.radius_px, 1e-6) / max(previous_radius, 1e-6)))
        shape_penalty = max(0.0, 0.72 - item.circularity)
        boundary_penalty = 0.25 if item.boundary else 0.0
        cost = (
            distance / max(2.0 * reference_radius, 1.0)
            + 1.15 * radius_change
            + 0.8 * shape_penalty
            + 0.18 * item.hue_distance / 18.0
            + boundary_penalty
        )
        if cost < best_cost:
            best, best_cost = item, cost
    return best, best_cost


def track_standard_ball(
    frames: Sequence[np.ndarray],
    expected_center_uv: Sequence[float],
    expected_radius_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not frames:
        raise ValueError("frames cannot be empty")
    expected = np.asarray(expected_center_uv, dtype=np.float64)
    colour = estimate_ball_colour(frames[0], tuple(expected), float(expected_radius_px))
    rows: list[dict[str, Any]] = []
    centers: list[np.ndarray] = []
    center_frames: list[int] = []
    previous_radius = float(expected_radius_px)
    for frame_index, frame in enumerate(frames):
        gap = frame_index - center_frames[-1] if center_frames else frame_index + 1
        if len(centers) >= 2 and gap <= 4:
            dt = max(1, center_frames[-1] - center_frames[-2])
            predicted = centers[-1] + (centers[-1] - centers[-2]) * gap / dt
        elif centers:
            predicted = centers[-1]
        else:
            predicted = expected
        candidates = _enumerate_candidates(frame, colour, expected_radius_px)
        candidate, cost = _choose_candidate(
            candidates,
            predicted,
            previous_radius,
            expected_radius_px,
            gap=gap,
        )
        if candidate is None:
            hough = _hough_candidates(frame, colour, expected_radius_px, predicted)
            candidate, cost = _choose_candidate(
                hough,
                predicted,
                previous_radius,
                expected_radius_px,
                gap=gap,
            )
            candidates.extend(hough)
        if candidate is None:
            rows.append(
                {
                    "frame_index": frame_index,
                    "found": False,
                    "center_u_px": None,
                    "center_v_px": None,
                    "measurement_radius_px": None,
                    "track_confidence": 0.0,
                    "candidate_count": len(candidates),
                    "measurement_source": None,
                    "ellipse_major_axis_px": None,
                    "ellipse_minor_axis_px": None,
                    "ellipse_angle_deg": None,
                    "ellipse_axis_ratio": None,
                    "radial_residual_ratio": None,
                }
            )
            continue
        center = np.asarray([candidate.center_u_px, candidate.center_v_px])
        centers.append(center)
        center_frames.append(frame_index)
        previous_radius = candidate.radius_px
        rows.append(
            {
                "frame_index": frame_index,
                "found": True,
                "center_u_px": candidate.center_u_px,
                "center_v_px": candidate.center_v_px,
                "measurement_radius_px": candidate.radius_px,
                "track_confidence": float(math.exp(-max(cost, 0.0))),
                "candidate_count": len(candidates),
                "contour_area_px": candidate.area_px,
                "circularity": candidate.circularity,
                "ellipse_major_axis_px": candidate.ellipse_major_axis_px,
                "ellipse_minor_axis_px": candidate.ellipse_minor_axis_px,
                "ellipse_angle_deg": candidate.ellipse_angle_deg,
                "ellipse_axis_ratio": candidate.ellipse_axis_ratio,
                "radial_residual_ratio": candidate.radial_residual_ratio,
                "hue_distance": candidate.hue_distance,
                "touches_frame_boundary": candidate.boundary,
                "measurement_source": candidate.measurement_source,
            }
        )
    found = [row for row in rows if row["found"]]
    return rows, {
        "method": "first_frame_gt_initialized_orange_sphere_local_association",
        "colour_model": colour,
        "frame_count": len(rows),
        "found_count": len(found),
        "tracked_fraction": len(found) / max(len(rows), 1),
        "expected_first_center_uv_px": expected.tolist(),
        "expected_first_radius_px": float(expected_radius_px),
    }


def _transform_point(homography: np.ndarray, point: Sequence[float]) -> np.ndarray:
    homogeneous = homography @ np.asarray([float(point[0]), float(point[1]), 1.0])
    return homogeneous[:2] / homogeneous[2]


def _reference_circle(calibration: Mapping[str, Any], video_size: tuple[int, int]) -> dict[str, Any]:
    source_size = (int(calibration["image"]["width_px"]), int(calibration["image"]["height_px"]))
    transform = crop_resize_homography(source_size, video_size)
    validation = calibration["projection_validation"]
    center = _transform_point(transform, validation["initial_center_uv_from_opencv_P"])
    bbox = validation["initial_mesh_vertex_bbox_xyxy_px"]
    source_radius = 0.25 * (abs(float(bbox[2]) - float(bbox[0])) + abs(float(bbox[3]) - float(bbox[1])))
    scale = math.sqrt(abs(float(np.linalg.det(transform[:2, :2]))))
    return {
        "source_to_video_H": transform,
        "center_uv_px": center,
        "radius_px": source_radius * scale,
        "source_radius_px": source_radius,
    }


def _ray_plane_intersection(
    uv: Sequence[float],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    plane_y: float,
) -> np.ndarray | None:
    camera_center = -R.T @ t
    ray_camera = np.linalg.inv(K) @ np.asarray([float(uv[0]), float(uv[1]), 1.0])
    ray_world = R.T @ ray_camera
    if abs(float(ray_world[1])) <= 1e-10:
        return None
    distance = (float(plane_y) - float(camera_center[1])) / float(ray_world[1])
    if not math.isfinite(distance) or distance <= 0:
        return None
    return camera_center + distance * ray_world


def _project_one(world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
    camera = R @ world + t
    if camera[2] <= 1e-9:
        return np.asarray([np.nan, np.nan]), float(camera[2])
    pixel = K @ camera
    return pixel[:2] / pixel[2], float(camera[2])


def _sphere_refine_xz(
    initial_world: np.ndarray,
    uv: np.ndarray,
    radius_px: float,
    sphere_depth_scale: float,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    plane_y: float,
    *,
    radius_weight: float = 0.35,
) -> tuple[np.ndarray, dict[str, float]]:
    state = np.asarray([initial_world[0], initial_world[2]], dtype=np.float64)

    def residual(value: np.ndarray) -> np.ndarray:
        world = np.asarray([value[0], plane_y, value[1]], dtype=np.float64)
        pixel, depth = _project_one(world, K, R, t)
        predicted_radius = sphere_depth_scale / max(depth, 1e-9)
        return np.asarray(
            [pixel[0] - uv[0], pixel[1] - uv[1], radius_weight * (predicted_radius - radius_px)],
            dtype=np.float64,
        )

    initial = state.copy()
    for _ in range(8):
        base = residual(state)
        jacobian = np.zeros((3, 2), dtype=np.float64)
        for column in range(2):
            step = 1e-5 * max(1.0, abs(float(state[column])))
            perturbed = state.copy()
            perturbed[column] += step
            jacobian[:, column] = (residual(perturbed) - base) / step
        lhs = jacobian.T @ jacobian + 1e-6 * np.eye(2)
        delta = np.linalg.solve(lhs, -jacobian.T @ base)
        if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) > 2.0:
            state = initial
            break
        state += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    world = np.asarray([state[0], plane_y, state[1]], dtype=np.float64)
    final = residual(state)
    return world, {
        "center_reprojection_residual_px": float(np.linalg.norm(final[:2])),
        "radius_residual_px": float(final[2] / radius_weight) if radius_weight else 0.0,
    }


def _project_to_manifold(experiment_id: str, world: np.ndarray, initial_world: np.ndarray) -> tuple[np.ndarray, float | None]:
    if experiment_id in LINE_MANIFOLDS:
        if LINE_MANIFOLDS[experiment_id] == "vertical":
            return np.asarray([initial_world[0], initial_world[1], world[2]]), float(world[2])
        return np.asarray([world[0], initial_world[1], initial_world[2]]), float(world[0])
    if experiment_id in PENDULUMS:
        constants = PENDULUMS[experiment_id]
        pivot = np.asarray(constants["pivot"], dtype=np.float64)
        length = float(constants["length"])
        theta = math.atan2(float(world[0] - pivot[0]), float(pivot[2] - world[2]))
        projected = np.asarray(
            [pivot[0] + length * math.sin(theta), pivot[1], pivot[2] - length * math.cos(theta)],
            dtype=np.float64,
        )
        return projected, theta
    return world, None


def reconstruct_metric_trajectory(
    track_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    *,
    experiment_id: str,
    video_size: tuple[int, int],
    fps: float,
    sphere_radius_ratio_range: tuple[float, float] = (0.65, 1.35),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    reference = _reference_circle(calibration, video_size)
    camera = calibration["camera"]
    K_reference = np.asarray(camera["K"], dtype=np.float64)
    K_nominal = reference["source_to_video_H"] @ K_reference
    R = np.asarray(camera["R_world_to_camera_opencv"], dtype=np.float64)
    t_camera = np.asarray(camera["t_world_to_camera_opencv_m"], dtype=np.float64)
    initial_world = np.asarray(calibration["object"]["initial_center_world_m"], dtype=np.float64)
    first_found = track_rows[0] if track_rows and track_rows[0].get("found") else None
    if first_found is None:
        raise ValueError("frame-0 sphere observation is required for GT size calibration")
    observed_first_center = np.asarray(
        [float(first_found["center_u_px"]), float(first_found["center_v_px"])], dtype=np.float64
    )
    # The generated first frame may include a fixed crop/resize rounding offset.
    # Model it as a constant image translation while keeping camera pose fixed.
    image_shift = observed_first_center - np.asarray(reference["center_uv_px"])
    alignment = np.asarray([[1.0, 0.0, image_shift[0]], [0.0, 1.0, image_shift[1]], [0.0, 0.0, 1.0]])
    K = alignment @ K_nominal
    gt_first_radius = float(reference["radius_px"])
    observed_first_radius = float(first_found["measurement_radius_px"])
    radius_measurement_scale = gt_first_radius / max(observed_first_radius, 1e-9)
    initial_depth = float((R @ initial_world + t_camera)[2])
    sphere_depth_scale = initial_depth * gt_first_radius
    plane_y = float(initial_world[1])
    output: list[dict[str, Any]] = []
    for source in track_rows:
        frame_index = int(source["frame_index"])
        base = dict(source)
        base["time_s"] = frame_index / fps
        base.update(
            {
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "q_value": None,
                "theta_rad": None,
                "measurement_valid": False,
                "physics_fit_used": False,
                "radius_px_gt_normalized": None,
                "expected_radius_px": None,
                "sphere_radius_ratio": None,
                "sphere_size_constraint_pass": False,
                "center_reprojection_residual_px": None,
                "radius_residual_px": None,
            }
        )
        if not source.get("found"):
            output.append(base)
            continue
        uv = np.asarray([float(source["center_u_px"]), float(source["center_v_px"])])
        radius = float(source["measurement_radius_px"]) * radius_measurement_scale
        plane_point = _ray_plane_intersection(uv, K, R, t_camera, plane_y)
        if plane_point is None:
            output.append(base)
            continue
        refined, residuals = _sphere_refine_xz(
            plane_point,
            uv,
            radius,
            sphere_depth_scale,
            K,
            R,
            t_camera,
            plane_y,
        )
        world, q_value = _project_to_manifold(experiment_id, refined, initial_world)
        projected_uv, manifold_depth = _project_one(world, K, R, t_camera)
        manifold_residual = float(np.linalg.norm(projected_uv - uv))
        expected_radius = sphere_depth_scale / max(manifold_depth, 1e-9)
        sphere_radius_ratio = radius / max(expected_radius, 1e-9)
        size_constraint_pass = bool(
            sphere_radius_ratio_range[0]
            <= sphere_radius_ratio
            <= sphere_radius_ratio_range[1]
        )
        measurement_valid = bool(
            size_constraint_pass
            and not bool(source.get("touches_frame_boundary", False))
        )
        base.update(
            {
                "x_m": float(world[0]),
                "y_m": float(world[1]),
                "z_m": float(world[2]),
                "q_value": q_value,
                "theta_rad": q_value if experiment_id in PENDULUMS else None,
                "measurement_valid": measurement_valid,
                "physics_fit_used": measurement_valid,
                "radius_px_gt_normalized": radius,
                "expected_radius_px": expected_radius,
                "sphere_radius_ratio": sphere_radius_ratio,
                "sphere_size_constraint_pass": size_constraint_pass,
                "center_reprojection_residual_px": manifold_residual,
                "radius_residual_px": residuals["radius_residual_px"],
            }
        )
        output.append(base)
    valid = [row for row in output if row["measurement_valid"]]
    size_checked = [row for row in output if row.get("sphere_radius_ratio") is not None]
    size_valid = [row for row in size_checked if row.get("sphere_size_constraint_pass")]
    return output, {
        "method": "fixed_KRt_constant_y_plane_with_first_frame_gt_sphere_size_refinement",
        "video_K": K.tolist(),
        "source_to_video_H": reference["source_to_video_H"].tolist(),
        "fixed_first_frame_image_shift_px": image_shift.tolist(),
        "gt_first_frame_radius_video_px": gt_first_radius,
        "observed_first_frame_radius_px": observed_first_radius,
        "radius_measurement_scale": radius_measurement_scale,
        "known_ball_radius_m": float(calibration["object"]["known_radius_m"]),
        "initial_center_world_m": initial_world.tolist(),
        "initial_camera_depth_m": initial_depth,
        "constant_world_plane_y_m": plane_y,
        "metric_frame_count": len(valid),
        "metric_fraction": len(valid) / max(len(output), 1),
        "sphere_radius_ratio_range": list(sphere_radius_ratio_range),
        "sphere_size_checked_frame_count": len(size_checked),
        "sphere_size_valid_frame_count": len(size_valid),
        "sphere_size_valid_fraction": len(size_valid) / max(len(size_checked), 1),
        "camera_pose_fixed_for_all_frames": True,
    }


def run_ball_tracking(
    video_path: Path,
    calibration: Mapping[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Decode a video and track the standard sphere without lifting to 3D."""

    frames, video = read_video_frames(video_path)
    reference = _reference_circle(calibration, (int(video["width"]), int(video["height"])))
    track, tracking_summary = track_standard_ball(frames, reference["center_uv_px"], reference["radius_px"])
    return frames, track, {"video": video, "tracking": tracking_summary}


def run_tracking_and_reconstruction(
    video_path: Path,
    calibration: Mapping[str, Any],
    experiment_id: str,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    frames, track, pipeline = run_ball_tracking(video_path, calibration)
    video = pipeline["video"]
    trajectory, geometry_summary = reconstruct_metric_trajectory(
        track,
        calibration,
        experiment_id=experiment_id,
        video_size=(int(video["width"]), int(video["height"])),
        fps=float(video["fps"]),
    )
    pipeline["geometry"] = geometry_summary
    return frames, trajectory, pipeline


def write_trajectory_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
