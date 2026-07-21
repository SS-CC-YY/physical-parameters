"""Calibrated geometry primitives for the V1A object trajectory pipeline.

The functions in this module deliberately contain no experiment target physics
(for example, the prompted gravity).  They convert image observations into
camera/world geometry only, so a later physics fit remains an independent
measurement.

Conventions
-----------
* image sizes are ``(width, height)``;
* image coordinates are OpenCV pixel coordinates, with ``u`` right and ``v``
  down;
* ``R`` and ``t`` map a world point into an OpenCV camera frame:
  ``X_camera = R @ X_world + t``;
* ``K`` is the intrinsic matrix for the image containing the observations.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


DEFAULT_POSE_QUALITY: dict[str, float | int] = {
    "min_inlier_count": 12,
    "min_inlier_ratio": 0.45,
    "max_reprojection_p95_px": 2.0,
    "max_all_reprojection_median_px": 4.0,
    "min_positive_depth_fraction": 0.90,
}


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}, got {array.shape}")
    return array


def _points(value: Any, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    return array


def crop_resize_homography(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Return the source-pixel to target-pixel center-crop/resize transform.

    This matches :func:`reconstruction.mvp._reference_alignment`: the source
    is center-cropped to the target aspect ratio and then resized.  Applying
    the returned matrix to both pixel coordinates and intrinsics preserves the
    calibrated projection under this deterministic preprocessing.
    """

    source_width, source_height = (int(source_size[0]), int(source_size[1]))
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source_size and target_size dimensions must be positive")

    target_aspect = target_width / float(target_height)
    source_aspect = source_width / float(source_height)
    if source_aspect >= target_aspect:
        crop_width = max(1, int(round(source_height * target_aspect)))
        crop_height = source_height
        crop_x = max(0, (source_width - crop_width) // 2)
        crop_y = 0
    else:
        crop_width = source_width
        crop_height = max(1, int(round(source_width / target_aspect)))
        crop_x = 0
        crop_y = max(0, (source_height - crop_height) // 2)

    scale_x = target_width / float(crop_width)
    scale_y = target_height / float(crop_height)
    return np.asarray(
        [
            [scale_x, 0.0, -scale_x * crop_x],
            [0.0, scale_y, -scale_y * crop_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compose_projection(K: Any, R: Any, t: Any) -> np.ndarray:
    """Compose ``P = K [R | t]`` from OpenCV camera parameters."""

    intrinsics = _matrix(K, (3, 3), "K")
    rotation = _matrix(R, (3, 3), "R")
    translation = np.asarray(t, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("t must contain three finite values")
    return intrinsics @ np.column_stack((rotation, translation))


def project_points(points_world: Any, K: Any, R: Any, t: Any) -> np.ndarray:
    """Project world points into image pixels.

    Points with zero projective depth produce ``nan`` coordinates.  Negative
    depth is intentionally not hidden: callers that estimate camera quality
    must explicitly evaluate cheirality rather than accepting a mirrored pose.
    """

    points = _points(points_world, 3, "points_world")
    projection = compose_projection(K, R, t)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    image_h = (projection @ homogeneous.T).T
    denominator = image_h[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    uv[valid] = image_h[valid, :2] / denominator[valid, None]
    return uv


def recover_vertical_line_z(
    uv: Any,
    P: Any,
    x0: float = 0.0,
    y0: float = 0.0,
    z_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Recover height on the known world line ``(x0, y0, z)``.

    The perspective projection of this line is rational in ``z``.  Cross
    multiplying the two pixel equations yields a two-row, one-unknown linear
    least-squares problem.  The returned residual is a true pixel reprojection
    residual after optionally clamping to physical height bounds.
    """

    observation = np.asarray(uv, dtype=np.float64).reshape(-1)
    projection = _matrix(P, (3, 4), "P")
    if observation.shape != (2,) or not np.all(np.isfinite(observation)):
        raise ValueError("uv must contain two finite pixel coordinates")
    if not math.isfinite(float(x0)) or not math.isfinite(float(y0)):
        raise ValueError("x0 and y0 must be finite")

    # P @ [x0, y0, z, 1] = a*z + b.
    a = projection[:, 2]
    b = projection[:, 0] * float(x0) + projection[:, 1] * float(y0) + projection[:, 3]
    coefficients = np.asarray(
        [
            observation[0] * a[2] - a[0],
            observation[1] * a[2] - a[1],
        ],
        dtype=np.float64,
    )
    right_hand_side = np.asarray(
        [
            b[0] - observation[0] * b[2],
            b[1] - observation[1] * b[2],
        ],
        dtype=np.float64,
    )
    denominator = float(coefficients @ coefficients)
    if not math.isfinite(denominator) or denominator <= 1e-18:
        return {
            "success": False,
            "reason": "vertical_line_is_unobservable",
            "z": None,
            "unclamped_z": None,
            "reprojected_uv": None,
            "residual_px": None,
            "line_observability_px_per_m": 0.0,
            "at_bound": False,
        }

    unconstrained_z = float((coefficients @ right_hand_side) / denominator)
    recovered_z = unconstrained_z
    at_bound = False
    if z_bounds is not None:
        lower, upper = float(z_bounds[0]), float(z_bounds[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("z_bounds must be a finite increasing pair")
        recovered_z = min(upper, max(lower, recovered_z))
        at_bound = recovered_z != unconstrained_z

    world = np.asarray([float(x0), float(y0), recovered_z, 1.0], dtype=np.float64)
    image_h = projection @ world
    if not np.all(np.isfinite(image_h)) or abs(float(image_h[2])) <= 1e-12:
        return {
            "success": False,
            "reason": "recovered_point_has_invalid_projective_depth",
            "z": recovered_z,
            "unclamped_z": unconstrained_z,
            "reprojected_uv": None,
            "residual_px": None,
            "line_observability_px_per_m": 0.0,
            "at_bound": at_bound,
        }
    reprojected = image_h[:2] / image_h[2]
    residual = float(np.linalg.norm(reprojected - observation))

    # Analytic image-space derivative of the projected line at recovered_z.
    projective_depth = float(a[2] * recovered_z + b[2])
    if abs(projective_depth) <= 1e-12:
        observability = 0.0
    else:
        du_dz = float((a[0] * b[2] - b[0] * a[2]) / (projective_depth**2))
        dv_dz = float((a[1] * b[2] - b[1] * a[2]) / (projective_depth**2))
        observability = math.hypot(du_dz, dv_dz)
    return {
        "success": bool(math.isfinite(recovered_z) and math.isfinite(residual)),
        "reason": None,
        "z": recovered_z,
        "unclamped_z": unconstrained_z,
        "reprojected_uv": reprojected.tolist(),
        "residual_px": residual,
        "line_observability_px_per_m": observability,
        "at_bound": at_bound,
    }


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if not len(finite) else float(np.percentile(finite, percentile))


def _pose_quality(
    result: dict[str, Any],
    thresholds: dict[str, float | int] | None,
) -> dict[str, Any]:
    limits = dict(DEFAULT_POSE_QUALITY)
    if thresholds:
        limits.update(thresholds)
    gates = {
        "solver_success": bool(result.get("success")),
        "inlier_count": int(result.get("inlier_count") or 0) >= int(limits["min_inlier_count"]),
        "inlier_ratio": float(result.get("inlier_ratio") or 0.0) >= float(limits["min_inlier_ratio"]),
        "reprojection_p95": result.get("reprojection_p95_px") is not None
        and float(result["reprojection_p95_px"]) <= float(limits["max_reprojection_p95_px"]),
        "all_reprojection_median": result.get("all_reprojection_median_px") is not None
        and float(result["all_reprojection_median_px"])
        <= float(limits["max_all_reprojection_median_px"]),
        "positive_depth": float(result.get("positive_depth_fraction") or 0.0)
        >= float(limits["min_positive_depth_fraction"]),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "status": "pass" if not failed else "invalid",
        "passed": not failed,
        "gates": gates,
        "failed_gates": failed,
        "thresholds": limits,
    }


def _failed_pose(reason: str, valid_points: int, thresholds: dict[str, float | int] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "reason": reason,
        "R": None,
        "t": None,
        "rvec": None,
        "inlier_indices": [],
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "valid_correspondences": int(valid_points),
        "reprojection_rmse_px": None,
        "reprojection_median_px": None,
        "reprojection_p95_px": None,
        "all_reprojection_p95_px": None,
        "all_reprojection_median_px": None,
        "positive_depth_fraction": 0.0,
        "refined_lm": False,
    }
    result["quality"] = _pose_quality(result, thresholds)
    return result


def estimate_pose_pnp(
    xyz: Any,
    uv: Any,
    K: Any,
    *,
    distortion: Any | None = None,
    ransac_reprojection_error_px: float = 2.0,
    confidence: float = 0.999,
    iterations: int = 2000,
    min_points: int = 6,
    quality_thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Estimate world-to-camera pose with PnP RANSAC and LM refinement.

    Non-finite correspondence pairs are discarded while ``inlier_indices``
    always refer to rows in the original input arrays.  Pose quality is kept
    separate from solver success so callers can reject a numerically returned
    pose with weak support or large residuals.
    """

    world = _points(xyz, 3, "xyz")
    image = _points(uv, 2, "uv")
    intrinsics = _matrix(K, (3, 3), "K")
    if len(world) != len(image):
        raise ValueError("xyz and uv must contain the same number of rows")
    if min_points < 4:
        raise ValueError("min_points must be at least 4")
    if ransac_reprojection_error_px <= 0 or not 0 < confidence < 1 or iterations <= 0:
        raise ValueError("invalid PnP RANSAC controls")

    finite = np.all(np.isfinite(world), axis=1) & np.all(np.isfinite(image), axis=1)
    original_indices = np.flatnonzero(finite)
    world_valid = np.ascontiguousarray(world[finite], dtype=np.float64)
    image_valid = np.ascontiguousarray(image[finite], dtype=np.float64)
    if len(world_valid) < int(min_points):
        return _failed_pose("insufficient_correspondences", len(world_valid), quality_thresholds)

    if distortion is None:
        distortion_array = np.zeros((5, 1), dtype=np.float64)
    else:
        distortion_array = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
        if not np.all(np.isfinite(distortion_array)):
            raise ValueError("distortion coefficients must be finite")

    try:
        success, rvec, tvec, local_inliers = cv2.solvePnPRansac(
            world_valid,
            image_valid,
            intrinsics,
            distortion_array,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=int(iterations),
            reprojectionError=float(ransac_reprojection_error_px),
            confidence=float(confidence),
        )
    except cv2.error as exc:
        return _failed_pose(f"opencv_solvepnp_error: {exc}", len(world_valid), quality_thresholds)
    if not success or rvec is None or tvec is None or local_inliers is None:
        return _failed_pose("solvepnp_ransac_failed", len(world_valid), quality_thresholds)

    local_inliers = np.asarray(local_inliers, dtype=np.int64).reshape(-1)
    if len(local_inliers) < 4:
        return _failed_pose("too_few_ransac_inliers", len(world_valid), quality_thresholds)
    refined = False
    try:
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            world_valid[local_inliers],
            image_valid[local_inliers],
            intrinsics,
            distortion_array,
            rvec,
            tvec,
        )
        if refined_rvec is not None and refined_tvec is not None:
            rvec, tvec = refined_rvec, refined_tvec
            refined = True
    except (cv2.error, AttributeError):
        # RANSAC's estimate is still a valid, auditable fallback on OpenCV
        # builds without solvePnPRefineLM.
        refined = False

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    projected, _ = cv2.projectPoints(
        world_valid,
        np.asarray(rvec, dtype=np.float64),
        translation.reshape(3, 1),
        intrinsics,
        distortion_array,
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_valid, axis=1)
    inlier_errors = errors[local_inliers]
    camera_points = (rotation @ world_valid.T).T + translation
    positive_depth_fraction = float(np.mean(camera_points[:, 2] > 0.0))
    mapped_inliers = original_indices[local_inliers]
    result = {
        "success": True,
        "reason": None,
        "R": rotation.tolist(),
        "t": translation.tolist(),
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
        "inlier_indices": mapped_inliers.astype(int).tolist(),
        "inlier_count": int(len(local_inliers)),
        "inlier_ratio": float(len(local_inliers) / len(world_valid)),
        "valid_correspondences": int(len(world_valid)),
        "reprojection_rmse_px": float(np.sqrt(np.mean(inlier_errors**2))),
        "reprojection_median_px": _percentile(inlier_errors, 50),
        "reprojection_p95_px": _percentile(inlier_errors, 95),
        "all_reprojection_median_px": _percentile(errors, 50),
        "all_reprojection_p95_px": _percentile(errors, 95),
        "positive_depth_fraction": positive_depth_fraction,
        "refined_lm": refined,
    }
    result["quality"] = _pose_quality(result, quality_thresholds)
    return result


__all__ = [
    "DEFAULT_POSE_QUALITY",
    "compose_projection",
    "crop_resize_homography",
    "estimate_pose_pnp",
    "project_points",
    "recover_vertical_line_z",
]
