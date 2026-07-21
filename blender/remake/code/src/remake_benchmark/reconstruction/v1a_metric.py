from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from remake_benchmark.core.io import read_yaml, write_json
from remake_benchmark.hybrid import choose_reconstruction_route
from remake_benchmark.reconstruction.v1a_geometry import (
    compose_projection,
    crop_resize_homography,
    estimate_pose_pnp,
    project_points,
    recover_vertical_line_z,
)
from remake_benchmark.reconstruction.v1a_tracker import (
    apply_homography,
    estimate_background_homographies,
    read_video_frames,
    track_ball_sequence,
)


V1A_METRIC_VERSION = "0.3.0"

DEFAULT_V1A_CONFIG: dict[str, Any] = {
    "version": V1A_METRIC_VERSION,
    "runtime": {"opencv_rng_seed": 20260720},
    "camera_pose": {
        "use_pnp_anchors": True,
        "min_registration_features": 24,
        "min_registration_inlier_ratio": 0.30,
        "max_registration_reprojection_p95_px": 3.0,
        "min_registration_image_coverage_fraction": 0.20,
        "min_registration_area_ratio": 0.75,
        "max_registration_area_ratio": 1.30,
        "max_registration_corner_shift_fraction": 0.18,
        "min_anchor_tracks": 16,
        "min_anchor_singular_value_ratio": 0.02,
        "min_anchor_world_extent_m": 1.0,
        "min_inlier_ratio": 0.60,
        "max_reprojection_median_px": 3.0,
        "max_all_reprojection_median_px": 4.0,
        "min_primary_pnp_fraction": 0.95,
        "max_primary_pnp_failure_run": 1,
        "lk_forward_backward_px": 1.75,
    },
    "tracking": {
        "hue_tolerance": 18.0,
        "min_circularity": 0.50,
        "min_radius_ratio": 0.55,
        "max_radius_ratio": 1.50,
        "max_curve_distance_diameters": 0.75,
        "max_prediction_distance_diameters": 1.25,
        "prediction_reset_gap_frames": 3,
        "min_detection_fraction": 0.70,
        "max_line_residual_object_diameters": 1.35,
    },
    "static_camera": {
        # A detected circle that violates the projection of the known Blender
        # sphere is not allowed into the metric fit on the static-camera path.
        "hard_sphere_radius_constraint": True,
        "min_observed_expected_radius_ratio": 0.65,
        "max_observed_expected_radius_ratio": 1.35,
        "min_sphere_size_valid_fraction": 0.70,
    },
    "physics": {
        "min_fit_points": 10,
        "min_fit_duration_s": 0.30,
        "release_drop_m": 0.035,
        "contact_margin_m": 0.10,
        "contact_confirmation_frames": 3,
        "robust_iterations": 6,
        "huber_k": 1.5,
        "min_r2": 0.90,
        "similarity_pass": 0.75,
    },
    "validity": {
        "min_valid_fraction": 0.60,
        "min_airborne_coverage_fraction": 0.80,
        "max_airborne_gap_frames": 3,
        "min_primary_duration_s": 0.45,
        "min_primary_vertical_span_m": 1.0,
        "max_initial_height_error_m": 0.30,
        "max_initial_center_error_diameters": 0.75,
        "min_background_success_fraction": 0.80,
        "min_background_inlier_ratio_p10": 0.60,
        "max_background_failure_run": 3,
        "max_verified_static_corner_motion_p95_px": 2.0,
        "max_line_residual_p95_px": 28.0,
        "max_background_reprojection_p95_px": 4.0,
        "max_metric_sigma_p95_m": 0.20,
        "max_pnp_vs_projective_curve_p95_px": 8.0,
        "max_radius_based_lateral_offset_p95_m": 0.75,
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


def load_v1a_config(path: Path | None = None) -> dict[str, Any]:
    return deepcopy(DEFAULT_V1A_CONFIG) if path is None else _deep_update(DEFAULT_V1A_CONFIG, read_yaml(path))


def _decode_number(text: str) -> float:
    return float(text.replace("m", "-").replace("p", "."))


def parse_v1a_video_job(video_path: Path) -> dict[str, Any]:
    parts = video_path.stem.split("__")
    if len(parts) != 6:
        raise ValueError(f"unrecognised V1A filename: {video_path.name}")
    experiment, target_token, scene, object_id, camera, seed_token = parts
    if experiment != "v1_A" or object_id != "standard_ball" or camera not in {"CAM_Main", "CAM_Side", "CAM_Top"}:
        raise ValueError(f"metric V1A currently requires v1_A/standard_ball/three benchmark cameras: {video_path.name}")
    target_value: float | None = None
    if target_token.startswith("gravity_g-"):
        target_value = _decode_number(target_token.removeprefix("gravity_g-"))
    elif re.fullmatch(r"g[m0-9p.]+", target_token):
        target_value = _decode_number(target_token[1:])
    if target_value is None:
        raise ValueError(f"cannot decode gravity target from {target_token!r}")
    seed = int(seed_token.removeprefix("seed-")) if seed_token.startswith("seed-") else None
    return {
        "job_id": video_path.stem,
        "video_path": str(video_path.resolve()),
        "experiment_id": experiment,
        "factors": {"scene_id": scene, "object_id": object_id, "camera": camera, "seed": seed},
        "targets": {"gravity_g": target_value},
    }


def conditioning_image_path(workspace_root: Path, job: dict[str, Any]) -> Path:
    factors = job["factors"]
    scene_id = "baseline" if str(factors["scene_id"]) == "base" else str(factors["scene_id"])
    path = (
        workspace_root
        / "first_frames_v1_0"
        / "images"
        / "v1_A"
        / scene_id
        / "standard_ball"
        / f"{factors['camera']}.png"
    )
    if not path.is_file():
        raise FileNotFoundError(f"conditioning image not found: {path}")
    return path


def calibration_path(calibration_root: Path, job: dict[str, Any]) -> Path:
    factors = job["factors"]
    scene_id = "base" if str(factors["scene_id"]) == "baseline" else str(factors["scene_id"])
    path = calibration_root / scene_id / f"{factors['camera']}.json"
    if not path.is_file():
        raise FileNotFoundError(f"V1A calibration sidecar not found: {path}")
    return path


def _load_calibration(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    camera = data["camera"]
    required = ["K", "R_world_to_camera_opencv", "t_world_to_camera_opencv_m"]
    missing = [key for key in required if key not in camera]
    if missing:
        raise ValueError(f"calibration camera contract is missing {missing}: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_provenance(opencv_rng_seed: int) -> dict[str, Any]:
    reconstruction_dir = Path(__file__).resolve().parent
    source_files = {
        "v1a_metric": Path(__file__).resolve(),
        "v1a_tracker": reconstruction_dir / "v1a_tracker.py",
        "v1a_geometry": reconstruction_dir / "v1a_geometry.py",
        "core_io": reconstruction_dir.parent / "core" / "io.py",
    }
    hashes = {name: _sha256(path) for name, path in source_files.items()}
    combined = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_sha256": hashes,
        "combined_source_sha256": combined,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "opencv_rng_seed": int(opencv_rng_seed),
    }


def _percentile(values: Iterable[float], q: float) -> float | None:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    return None if not len(finite) else float(np.percentile(finite, q))


def _line_contract(calibration: dict[str, Any]) -> tuple[float, float, float, float, float]:
    obj = calibration["object"]
    line = obj["motion_line"]
    origin = line.get("origin_world_m", [0.0, 0.0, 0.0])
    start = obj.get("initial_center_world_m", [origin[0], origin[1], 4.2])
    contact = calibration["support"].get("sphere_center_at_first_contact_world_m", [origin[0], origin[1], 0.44])
    return float(origin[0]), float(origin[1]), float(start[2]), float(contact[2]), float(obj["known_radius_m"])


def _anchor_path(sidecar_path: Path, calibration: dict[str, Any]) -> Path | None:
    contract = calibration.get("reference_geometry_anchors") or {}
    relative = contract.get("path") or contract.get("npz_path")
    if not relative:
        return None
    path = Path(str(relative))
    # Exporter paths are relative to the calibration root (the parent of the
    # scene directory), not relative to the individual sidecar file.
    return path if path.is_absolute() else (sidecar_path.parent.parent / path).resolve()


def _load_anchors(path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None or not path.is_file():
        return None
    with np.load(path) as payload:
        uv_key = "uv_ref" if "uv_ref" in payload else "uv_px"
        xyz_key = "xyz_world" if "xyz_world" in payload else "xyz_world_m"
        if uv_key not in payload or xyz_key not in payload:
            return None
        uv = np.asarray(payload[uv_key], dtype=np.float64).reshape(-1, 2)
        xyz = np.asarray(payload[xyz_key], dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(uv), axis=1) & np.all(np.isfinite(xyz), axis=1)
    return uv[finite], xyz[finite]


def _anchor_geometry_quality(xyz: np.ndarray | None, config: dict[str, Any]) -> dict[str, Any]:
    if xyz is None or len(xyz) < 4:
        return {
            "passed": False,
            "points": 0 if xyz is None else int(len(xyz)),
            "singular_values": [],
            "smallest_to_largest_singular_ratio": 0.0,
            "world_extent_m": 0.0,
            "failed_gates": ["insufficient_anchor_geometry"],
        }
    centered = np.asarray(xyz, dtype=np.float64) - np.mean(xyz, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    ratio = float(singular[-1] / max(singular[0], 1e-12))
    extent = float(np.linalg.norm(np.ptp(np.asarray(xyz, dtype=np.float64), axis=0)))
    gates = {
        "noncoplanar_world_support": ratio
        >= float(config["min_anchor_singular_value_ratio"]),
        "world_extent": extent >= float(config["min_anchor_world_extent_m"]),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "passed": not failed,
        "points": int(len(xyz)),
        "singular_values": singular.tolist(),
        "smallest_to_largest_singular_ratio": ratio,
        "world_extent_m": extent,
        "gates": gates,
        "failed_gates": failed,
    }


def _transform_uv(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(homography, dtype=np.float64),
    ).reshape(-1, 2)


def _distance_to_polyline(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Return the nearest sampled-curve distance for each image point."""
    query = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    line = np.asarray(curve, dtype=np.float64).reshape(-1, 2)
    line = line[np.all(np.isfinite(line), axis=1)]
    if not len(line):
        return np.full(len(query), np.inf, dtype=np.float64)
    # V1A curves contain only 160 samples; chunking avoids a large temporary
    # array if a denser anchor export is used later.
    output = np.full(len(query), np.inf, dtype=np.float64)
    for start in range(0, len(query), 1024):
        block = query[start : start + 1024]
        output[start : start + len(block)] = np.min(
            np.linalg.norm(block[:, None, :] - line[None, :, :], axis=2), axis=1
        )
    return output


def _registration_feature_mask(
    shape: tuple[int, int],
    curve_uv: np.ndarray,
    object_radius_px: float,
) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    finite = np.asarray(curve_uv, dtype=np.float64)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if len(finite) >= 2:
        points = np.rint(finite).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            mask,
            [points],
            False,
            0,
            max(15, int(round(8.0 * object_radius_px))),
        )
    border = max(6, int(round(min(height, width) * 0.025)))
    mask[:border] = 0
    mask[-border:] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def register_conditioning_to_frame0(
    conditioning_path: Path,
    frame0: np.ndarray,
    nominal_reference_to_video: np.ndarray,
    nominal_curve_video: np.ndarray,
    object_radius_px: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Register the frozen conditioning PNG to decoded video frame 0.

    The object corridor is excluded, so registration is supported only by
    static scene appearance.  The returned transform maps original frozen-PNG
    pixels directly into decoded frame-0 pixels.
    """
    reference = cv2.imread(str(conditioning_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"cannot decode conditioning image: {conditioning_path}")
    height, width = frame0.shape[:2]
    warped = cv2.warpPerspective(
        reference,
        np.asarray(nominal_reference_to_video, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_LINEAR,
    )
    gray_reference = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_video = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    mask = _registration_feature_mask(
        gray_reference.shape,
        nominal_curve_video,
        object_radius_px,
    )
    points0 = cv2.goodFeaturesToTrack(
        gray_reference,
        mask=mask,
        maxCorners=1400,
        qualityLevel=0.006,
        minDistance=6,
        blockSize=7,
    )
    minimum = int(config["min_registration_features"])
    failed: dict[str, Any] = {
        "success": False,
        "method": "frozen_frame0_background_lk_homography",
        "detected_features": 0 if points0 is None else int(len(points0)),
        "tracked_features": 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "reprojection_median_px": None,
        "reprojection_p95_px": None,
        "failed_gates": [],
    }
    if points0 is None or len(points0) < minimum:
        failed["failed_gates"] = ["insufficient_reference_features"]
        return np.asarray(nominal_reference_to_video, dtype=np.float64), failed
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(
        gray_reference,
        gray_video,
        points0,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005),
    )
    back = status2 = None
    if points1 is not None:
        back, status2, _ = cv2.calcOpticalFlowPyrLK(
            gray_video,
            gray_reference,
            points1,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005),
        )
    if points1 is None or status1 is None or back is None or status2 is None:
        failed["failed_gates"] = ["registration_optical_flow_failed"]
        return np.asarray(nominal_reference_to_video, dtype=np.float64), failed
    valid = status1.ravel().astype(bool) & status2.ravel().astype(bool)
    fb_error = np.linalg.norm(points0.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
    valid &= fb_error <= float(config["lk_forward_backward_px"])
    source = points0.reshape(-1, 2)[valid]
    target = points1.reshape(-1, 2)[valid]
    failed["tracked_features"] = int(len(source))
    if len(source) < minimum:
        failed["failed_gates"] = ["insufficient_registered_features"]
        return np.asarray(nominal_reference_to_video, dtype=np.float64), failed
    delta, inlier_mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        2.5,
        maxIters=4000,
        confidence=0.999,
    )
    if delta is None or inlier_mask is None:
        failed["failed_gates"] = ["registration_homography_failed"]
        return np.asarray(nominal_reference_to_video, dtype=np.float64), failed
    inliers = inlier_mask.ravel().astype(bool)
    prediction = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2), delta
    ).reshape(-1, 2)
    residual = np.linalg.norm(prediction - target, axis=1)
    inlier_count = int(np.sum(inliers))
    ratio = inlier_count / max(len(source), 1)
    median = None if not inlier_count else float(np.median(residual[inliers]))
    p95 = None if not inlier_count else float(np.percentile(residual[inliers], 95))
    inlier_source = source[inliers]
    image_coverage = (
        0.0
        if not len(inlier_source)
        else float(
            np.ptp(inlier_source[:, 0])
            * np.ptp(inlier_source[:, 1])
            / max(float(width * height), 1.0)
        )
    )
    corners = np.asarray(
        [[[0.0, 0.0]], [[width - 1.0, 0.0]], [[width - 1.0, height - 1.0]], [[0.0, height - 1.0]]],
        dtype=np.float64,
    )
    warped_corners = cv2.perspectiveTransform(corners, delta).reshape(-1, 2)
    source_corners = corners.reshape(-1, 2)

    def signed_area(points: np.ndarray) -> float:
        return 0.5 * float(
            np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
        )

    source_area = signed_area(source_corners)
    warped_area = signed_area(warped_corners)
    area_ratio = abs(warped_area) / max(abs(source_area), 1e-9)
    corner_shift_fraction = float(
        np.max(np.linalg.norm(warped_corners - source_corners, axis=1))
        / max(math.hypot(width, height), 1.0)
    )
    homography_condition = float(np.linalg.cond(np.asarray(delta, dtype=np.float64)))
    gates = {
        "inlier_count": inlier_count >= minimum,
        "inlier_ratio": ratio >= float(config["min_registration_inlier_ratio"]),
        "reprojection_p95": p95 is not None
        and p95 <= float(config["max_registration_reprojection_p95_px"]),
        "image_coverage": image_coverage
        >= float(config["min_registration_image_coverage_fraction"]),
        "orientation_preserved": bool(source_area * warped_area > 0.0),
        "area_ratio": float(config["min_registration_area_ratio"])
        <= area_ratio
        <= float(config["max_registration_area_ratio"]),
        "corner_shift": corner_shift_fraction
        <= float(config["max_registration_corner_shift_fraction"]),
        "homography_finite": bool(
            np.all(np.isfinite(warped_corners)) and math.isfinite(homography_condition)
        ),
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    if abs(float(delta[2, 2])) > 1e-12:
        delta = delta / delta[2, 2]
    result = {
        **failed,
        "success": not failed_gates,
        "inlier_count": inlier_count,
        "inlier_ratio": float(ratio),
        "reprojection_median_px": median,
        "reprojection_p95_px": p95,
        "inlier_image_bbox_coverage_fraction": image_coverage,
        "warped_image_area_ratio": float(area_ratio),
        "max_corner_shift_fraction_of_diagonal": corner_shift_fraction,
        "homography_condition_number": homography_condition,
        "failed_gates": failed_gates,
        "delta_warp_video_pixels": np.asarray(delta, dtype=float).tolist(),
    }
    composed = np.asarray(delta, dtype=np.float64) @ np.asarray(
        nominal_reference_to_video, dtype=np.float64
    )
    if abs(float(composed[2, 2])) > 1e-12:
        composed /= composed[2, 2]
    return composed, result


def _pose_sequence_from_anchors(
    frames: list[np.ndarray],
    conditioning_path: Path,
    anchor_uv_ref: np.ndarray,
    anchor_xyz_world: np.ndarray,
    h_reference_to_video: np.ndarray,
    k_video: np.ndarray,
    nominal_r: np.ndarray,
    nominal_t: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    uv0_theoretical = _transform_uv(anchor_uv_ref, h_reference_to_video)
    height, width = frames[0].shape[:2]
    inside = (
        (uv0_theoretical[:, 0] >= 5)
        & (uv0_theoretical[:, 0] < width - 5)
        & (uv0_theoretical[:, 1] >= 5)
        & (uv0_theoretical[:, 1] < height - 5)
    )
    uv0_theoretical = uv0_theoretical[inside]
    xyz = anchor_xyz_world[inside]
    conditioning = cv2.imread(str(conditioning_path), cv2.IMREAD_COLOR)
    if conditioning is None:
        raise RuntimeError(f"cannot decode conditioning image: {conditioning_path}")
    registered_reference = cv2.warpPerspective(
        conditioning,
        np.asarray(h_reference_to_video, dtype=np.float64),
        (width, height),
        flags=cv2.INTER_LINEAR,
    )
    gray_reference = cv2.cvtColor(registered_reference, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    observed0, status0, _ = cv2.calcOpticalFlowPyrLK(
        gray_reference,
        gray0,
        uv0_theoretical.astype(np.float32).reshape(-1, 1, 2),
        None,
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
    )
    back0 = back_status0 = None
    if observed0 is not None:
        back0, back_status0, _ = cv2.calcOpticalFlowPyrLK(
            gray0,
            gray_reference,
            observed0,
            None,
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
        )
    valid0 = (
        np.zeros(len(uv0_theoretical), dtype=bool)
        if status0 is None
        else status0.ravel().astype(bool)
    )
    if back0 is not None and back_status0 is not None:
        fb0 = np.linalg.norm(
            back0.reshape(-1, 2) - uv0_theoretical,
            axis=1,
        )
        valid0 &= back_status0.ravel().astype(bool) & (
            fb0 <= float(config["lk_forward_backward_px"])
        )
    if observed0 is None:
        uv0 = np.empty((0, 2), dtype=np.float64)
        xyz = xyz[:0]
    else:
        uv0 = observed0.reshape(-1, 2)[valid0].astype(np.float64)
        xyz = xyz[valid0]
    local_residual = (
        np.empty(0, dtype=np.float64)
        if observed0 is None
        else np.linalg.norm(
            observed0.reshape(-1, 2)[valid0] - uv0_theoretical[valid0],
            axis=1,
        )
    )
    frame0_anchor_observation = {
        "theoretical_anchors_in_frame": int(len(uv0_theoretical)),
        "locally_observed_anchors": int(len(uv0)),
        "local_registration_residual_median_px": _percentile(local_residual, 50),
        "local_registration_residual_p95_px": _percentile(local_residual, 95),
        "observation_method": "registered_conditioning_to_decoded_frame0_anchor_lk_fb",
    }
    rows: list[dict[str, Any]] = []
    projections: list[np.ndarray] = []
    nominal_p = compose_projection(k_video, nominal_r, nominal_t)
    for frame_index, frame in enumerate(frames):
        if frame_index == 0:
            pose = estimate_pose_pnp(
                xyz,
                uv0,
                k_video,
                ransac_reprojection_error_px=float(config["max_reprojection_median_px"]),
                min_points=int(config["min_anchor_tracks"]),
                quality_thresholds={
                    "min_inlier_count": int(config["min_anchor_tracks"]),
                    "min_inlier_ratio": float(config["min_inlier_ratio"]),
                    "max_reprojection_p95_px": 1.5
                    * float(config["max_reprojection_median_px"]),
                    "max_all_reprojection_median_px": float(
                        config["max_all_reprojection_median_px"]
                    ),
                },
            )
            row = {
                "frame_index": 0,
                "method": "registered_frame0_solvepnp",
                "geometry_source": "anchor_pnp" if _pose_quality_passed(pose) else "unusable",
                "frame0_anchor_observation": frame0_anchor_observation,
                **pose,
            }
            rows.append(row)
            if _pose_quality_passed(pose):
                projections.append(
                    compose_projection(
                        k_video,
                        np.asarray(pose["R"], dtype=float),
                        np.asarray(pose["t"], dtype=float),
                    )
                )
            else:
                projections.append(nominal_p.copy())
            continue
        if len(uv0) < int(config["min_anchor_tracks"]):
            pose = {
                "success": False,
                "quality": False,
                "reason": "insufficient_frame0_observed_anchor_tracks",
            }
            rows.append(
                {
                    "frame_index": frame_index,
                    "method": "anchor_lk_solvepnp",
                    "geometry_source": "unusable",
                    **pose,
                }
            )
            projections.append(nominal_p.copy())
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            gray0,
            gray,
            uv0.astype(np.float32).reshape(-1, 1, 2),
            None,
            winSize=(25, 25),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
        )
        back = back_status = None
        if tracked is not None:
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                gray0,
                tracked,
                None,
                winSize=(25, 25),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
            )
        valid = np.zeros(len(uv0), dtype=bool) if status is None else status.ravel().astype(bool)
        if back is not None and back_status is not None:
            fb = np.linalg.norm(back.reshape(-1, 2) - uv0, axis=1)
            valid &= back_status.ravel().astype(bool) & (fb <= float(config["lk_forward_backward_px"]))
        if tracked is None or int(np.sum(valid)) < int(config["min_anchor_tracks"]):
            pose = {"success": False, "quality": False, "reason": "insufficient_anchor_tracks"}
        else:
            pose = estimate_pose_pnp(
                xyz[valid],
                tracked.reshape(-1, 2)[valid],
                k_video,
                ransac_reprojection_error_px=float(config["max_reprojection_median_px"]),
                min_points=int(config["min_anchor_tracks"]),
                quality_thresholds={
                    "min_inlier_count": int(config["min_anchor_tracks"]),
                    "min_inlier_ratio": float(config["min_inlier_ratio"]),
                    "max_reprojection_p95_px": 1.5 * float(config["max_reprojection_median_px"]),
                    "max_all_reprojection_median_px": float(
                        config["max_all_reprojection_median_px"]
                    ),
                },
            )
        row = {
            "frame_index": frame_index,
            "method": "anchor_lk_solvepnp",
            "geometry_source": "anchor_pnp" if _pose_quality_passed(pose) else "unusable",
            **pose,
        }
        rows.append(row)
        if _pose_quality_passed(pose):
            projections.append(
                compose_projection(
                    k_video,
                    np.asarray(pose["R"], dtype=float),
                    np.asarray(pose["t"], dtype=float),
                )
            )
        else:
            projections.append(nominal_p.copy())
    return rows, projections


def _pose_quality_passed(pose: dict[str, Any]) -> bool:
    quality = pose.get("quality")
    if isinstance(quality, dict):
        return bool(pose.get("success") and quality.get("passed"))
    return bool(pose.get("success") and quality)


def _project_curve(projection: np.ndarray, x0: float, y0: float, contact_z: float, start_z: float) -> np.ndarray:
    z_values = np.linspace(contact_z, start_z, 160)
    points = np.column_stack([np.full_like(z_values, x0), np.full_like(z_values, y0), z_values])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    image = (projection @ homogeneous.T).T
    return image[:, :2] / image[:, 2:3]


def _pose_drift_summary(
    pose_rows: list[dict[str, Any]],
    nominal_r: np.ndarray,
    nominal_t: np.ndarray,
) -> dict[str, Any]:
    nominal_center = -nominal_r.T @ nominal_t.reshape(3)
    translations: list[float] = []
    rotations: list[float] = []
    for pose in pose_rows:
        if not _pose_quality_passed(pose) or pose.get("R") is None or pose.get("t") is None:
            continue
        rotation = np.asarray(pose["R"], dtype=float)
        translation = np.asarray(pose["t"], dtype=float).reshape(3)
        center = -rotation.T @ translation
        translations.append(float(np.linalg.norm(center - nominal_center)))
        relative = rotation @ nominal_r.T
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        rotations.append(float(math.degrees(math.acos(cosine))))
    return {
        "translation_from_nominal_p50_m": _percentile(translations, 50),
        "translation_from_nominal_p95_m": _percentile(translations, 95),
        "rotation_from_nominal_p50_deg": _percentile(rotations, 50),
        "rotation_from_nominal_p95_deg": _percentile(rotations, 95),
    }


def _pnp_projective_curve_disagreement(
    pnp_projections: list[np.ndarray],
    pose_rows: list[dict[str, Any]],
    background_h: list[np.ndarray],
    nominal_p: np.ndarray,
    x0: float,
    y0: float,
    contact_z: float,
    start_z: float,
) -> dict[str, Any]:
    per_frame: list[float] = []
    for pnp, pose, homography in zip(pnp_projections, pose_rows, background_h):
        if not _pose_quality_passed(pose):
            continue
        curve_pnp = _project_curve(pnp, x0, y0, contact_z, start_z)[::16]
        curve_projective = _project_curve(homography @ nominal_p, x0, y0, contact_z, start_z)[::16]
        per_frame.append(float(np.median(np.linalg.norm(curve_pnp - curve_projective, axis=1))))
    return {
        "pnp_vs_projective_curve_median_p50_px": _percentile(per_frame, 50),
        "pnp_vs_projective_curve_median_p95_px": _percentile(per_frame, 95),
        "pnp_vs_projective_compared_frames": len(per_frame),
    }


def _initial_projected_radius(
    projection: np.ndarray,
    center_world: tuple[float, float, float],
    radius_m: float,
) -> float:
    x, y, z = center_world
    points = np.asarray(
        [[x, y, z], [x + radius_m, y, z], [x - radius_m, y, z], [x, y + radius_m, z], [x, y - radius_m, z]],
        dtype=float,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (projection @ homogeneous.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    distances = np.linalg.norm(uv[1:] - uv[0], axis=1)
    return float(np.median(distances))


def _calibrated_initial_radius(
    calibration: dict[str, Any],
    reference_to_video: np.ndarray,
    fallback: float,
) -> float:
    bbox = calibration.get("projection_validation", {}).get("initial_mesh_vertex_bbox_xyxy_px")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return fallback
    x0, y0, x1, y1 = (float(value) for value in bbox)
    corners = _transform_uv(np.asarray([[x0, y0], [x1, y1]], dtype=float), reference_to_video)
    width = abs(float(corners[1, 0] - corners[0, 0]))
    height = abs(float(corners[1, 1] - corners[0, 1]))
    radius = 0.25 * (width + height)
    return radius if math.isfinite(radius) and radius > 2.0 else fallback


def _radius_based_world_center(
    uv: tuple[float, float],
    radius_px: float,
    radius_m: float,
    k: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
) -> list[float] | None:
    if radius_px <= 1e-6:
        return None
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    focal = math.sqrt(abs(fx * fy))
    depth = focal * radius_m / radius_px
    x_cam = (float(uv[0]) - cx) / fx * depth
    y_cam = (float(uv[1]) - cy) / fy * depth
    camera_point = np.asarray([x_cam, y_cam, depth], dtype=float)
    world = np.asarray(r, dtype=float).T @ (camera_point - np.asarray(t, dtype=float).reshape(3))
    return world.tolist()


def reconstruct_track(
    tracks: list[dict[str, Any]],
    projections: list[np.ndarray],
    pose_rows: list[dict[str, Any]],
    k_video: np.ndarray,
    nominal_r: np.ndarray,
    nominal_t: np.ndarray,
    x0: float,
    y0: float,
    start_z: float,
    contact_z: float,
    radius_m: float,
    fps: float,
    config: dict[str, Any],
    registration_reprojection_p95_px: float | None = None,
    hard_sphere_constraint: bool = False,
    min_observed_expected_radius_ratio: float = 0.65,
    max_observed_expected_radius_ratio: float = 1.35,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < min_observed_expected_radius_ratio <= max_observed_expected_radius_ratio:
        raise ValueError("invalid observed/expected sphere-radius ratio bounds")
    output: list[dict[str, Any]] = []
    max_line = float(config["tracking"]["max_line_residual_object_diameters"])
    reference_radius = float(np.median([row["display_radius_px"] for row in tracks]))
    for track, projection, pose in zip(tracks, projections, pose_rows):
        row = dict(track)
        row["time_s"] = float(row["frame_index"] / fps)
        geometry_source = str(pose.get("geometry_source") or "unusable")
        metric_geometry_valid = geometry_source in {
            "anchor_pnp",
            "verified_static_calibration",
            "audited_static_calibration",
        }
        row.update(
            {
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "z_sigma_m": None,
                "line_residual_px": None,
                "line_observability_px_per_m": None,
                "geometry_source": geometry_source,
                "metric_geometry_valid": metric_geometry_valid,
                "expected_radius_px": None,
                "sphere_radius_ratio": None,
                "sphere_radius_relative_error": None,
                "sphere_size_valid": None,
                "hard_sphere_constraint_applied": bool(hard_sphere_constraint),
            }
        )
        if not track["found"]:
            row["measurement_valid"] = False
            row["invalid_reason"] = "object_not_found"
            output.append(row)
            continue
        uv = (float(track["center_u_px"]), float(track["center_v_px"]))
        recovered = recover_vertical_line_z(uv, projection, x0=x0, y0=y0)
        if not recovered.get("success") or recovered.get("z") is None:
            row["measurement_valid"] = False
            row["invalid_reason"] = str(recovered.get("reason") or "vertical_line_recovery_failed")
            output.append(row)
            continue
        z = float(recovered["z"])
        residual = float(recovered["residual_px"])
        observability = float(recovered.get("line_observability_px_per_m") or 0.0)
        center_sigma_px = max(0.75, 0.08 * float(track["measurement_radius_px"]))
        pose_residual = (
            pose.get("reprojection_p95_px")
            or pose.get("reprojection_median_px")
            or pose.get("reprojection_median")
            or 0.0
        )
        registration_residual = float(registration_reprojection_p95_px or 0.0)
        combined_sigma_px = math.sqrt(
            center_sigma_px**2
            + float(pose_residual) ** 2
            + registration_residual**2
        )
        sigma_z = (
            None
            if not metric_geometry_valid or observability <= 1e-9
            else float(combined_sigma_px / observability)
        )
        within_height = contact_z - 0.35 <= z <= start_z + 0.65
        within_line = residual <= max_line * 2.0 * reference_radius
        expected_radius = _initial_projected_radius(
            projection,
            (x0, y0, z),
            radius_m,
        )
        observed_radius = float(track["measurement_radius_px"])
        sphere_ratio = (
            None
            if not math.isfinite(expected_radius) or expected_radius <= 1e-6
            else observed_radius / expected_radius
        )
        sphere_size_valid = bool(
            sphere_ratio is not None
            and min_observed_expected_radius_ratio
            <= sphere_ratio
            <= max_observed_expected_radius_ratio
        )
        valid_metric_measurement = bool(
            metric_geometry_valid
            and sigma_z is not None
            and within_height
            and within_line
            and (not hard_sphere_constraint or sphere_size_valid)
        )
        row.update(
            {
                "x_m": x0,
                "y_m": y0,
                "z_m": z,
                "z_sigma_m": sigma_z,
                "line_residual_px": residual,
                "line_observability_px_per_m": observability,
                "expected_radius_px": expected_radius,
                "sphere_radius_ratio": sphere_ratio,
                "sphere_radius_relative_error": (
                    None if sphere_ratio is None else abs(sphere_ratio - 1.0)
                ),
                "sphere_size_valid": sphere_size_valid,
                "measurement_valid": valid_metric_measurement,
                "invalid_reason": (
                    None
                    if valid_metric_measurement
                    else (
                        "nonmetric_projective_geometry"
                        if not metric_geometry_valid
                        else (
                            "metric_uncertainty_unavailable"
                            if sigma_z is None
                            else (
                                "outside_height_range"
                                if not within_height
                                else (
                                    "off_motion_line"
                                    if not within_line
                                    else "known_sphere_size_constraint_failed"
                                )
                            )
                        )
                    )
                ),
            }
        )
        if _pose_quality_passed(pose) and pose.get("R") is not None and pose.get("t") is not None:
            r_use = np.asarray(pose["R"], dtype=float)
            t_use = np.asarray(pose["t"], dtype=float)
        else:
            r_use, t_use = nominal_r, nominal_t
        unconstrained = _radius_based_world_center(
            uv,
            float(track["measurement_radius_px"]),
            radius_m,
            k_video,
            r_use,
            t_use,
        )
        row["radius_based_world_center_m"] = unconstrained
        row["radius_based_lateral_offset_m"] = None if unconstrained is None else float(math.hypot(unconstrained[0] - x0, unconstrained[1] - y0))
        output.append(row)
    valid_indices = [index for index, row in enumerate(output) if row["measurement_valid"]]
    components: list[list[int]] = []
    for index in valid_indices:
        if not components or index - components[-1][-1] > 3:
            components.append([index])
        else:
            components[-1].append(index)
    initial_components = [component for component in components if component[0] == 0]
    # Identity is anchored to the frozen frame-0 sphere.  A later smooth blob
    # is never promoted to the primary object merely because it is long.
    primary = max(initial_components or [[]], key=len)
    primary_set = set(primary)
    for index, row in enumerate(output):
        row["primary_metric_trajectory"] = index in primary_set
        row["physics_fit_used"] = False
    valid = [row for row in output if row["measurement_valid"]]
    sphere_checked = [
        row for row in output if row.get("found") and row.get("sphere_size_valid") is not None
    ]
    sphere_valid = [row for row in sphere_checked if row["sphere_size_valid"]]
    primary_rows = [output[index] for index in primary]
    primary_span = 0 if not primary else primary[-1] - primary[0] + 1
    primary_gaps = (
        []
        if len(primary) < 2
        else [right - left - 1 for left, right in zip(primary[:-1], primary[1:])]
    )
    frame0 = output[0] if output else None
    start_projection = projections[0] if projections else None
    expected_start_uv = (
        None
        if start_projection is None
        else _project_curve(start_projection, x0, y0, start_z, start_z)[0]
    )
    initial_center_error = None
    if frame0 is not None and frame0.get("found") and expected_start_uv is not None:
        initial_center_error = float(
            np.linalg.norm(
                np.asarray([frame0["center_u_px"], frame0["center_v_px"]], dtype=float)
                - expected_start_uv
            )
        )
    return output, {
        "valid_points": len(valid),
        "valid_fraction": len(valid) / max(len(output), 1),
        "line_residual_p50_px": _percentile([row["line_residual_px"] for row in valid], 50),
        "line_residual_p95_px": _percentile([row["line_residual_px"] for row in valid], 95),
        "z_sigma_p95_m": _percentile([row["z_sigma_m"] for row in valid], 95),
        "hard_sphere_constraint_applied": bool(hard_sphere_constraint),
        "sphere_size_checked_frames": len(sphere_checked),
        "sphere_size_valid_frames": len(sphere_valid),
        "sphere_size_valid_fraction": len(sphere_valid) / max(len(sphere_checked), 1),
        "sphere_radius_ratio_p50": _percentile(
            [row["sphere_radius_ratio"] for row in sphere_checked], 50
        ),
        "sphere_radius_relative_error_p95": _percentile(
            [row["sphere_radius_relative_error"] for row in sphere_checked], 95
        ),
        "radius_based_lateral_offset_p95_m": _percentile(
            [row["radius_based_lateral_offset_m"] for row in valid if row["radius_based_lateral_offset_m"] is not None], 95
        ),
        "primary_component_start_frame": None if not primary else primary[0],
        "primary_component_end_frame": None if not primary else primary[-1],
        "primary_component_points": len(primary),
        "primary_component_coverage": len(primary) / max(primary_span, 1),
        "primary_max_gap_frames": max(primary_gaps, default=0),
        "primary_component_duration_s": 0.0 if len(primary) < 2 else float((primary[-1] - primary[0]) / fps),
        "primary_component_vertical_span_m": None
        if not primary_rows
        else float(max(row["z_m"] for row in primary_rows) - min(row["z_m"] for row in primary_rows)),
        "initial_frame_metric_valid": bool(frame0 and frame0.get("measurement_valid")),
        "initial_center_error_px": initial_center_error,
        "initial_height_error_m": None
        if not frame0 or frame0.get("z_m") is None
        else float(abs(float(frame0["z_m"]) - start_z)),
    }


def _robust_quadratic(times: np.ndarray, heights: np.ndarray, iterations: int, huber_k: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(times)), times, times * times])
    weights = np.ones(len(times), dtype=float)
    coefficients = np.linalg.lstsq(design, heights, rcond=None)[0]
    for _ in range(max(1, iterations)):
        residuals = heights - design @ coefficients
        scale = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))) + 1e-9
        normalized = np.abs(residuals) / (huber_k * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-12))
        coefficients = np.linalg.lstsq(design * np.sqrt(weights[:, None]), heights * np.sqrt(weights), rcond=None)[0]
    return coefficients, weights


def fit_freefall_metric(
    trajectory: list[dict[str, Any]],
    contact_z: float,
    target_gravity: float | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    primary_available = any(row.get("primary_metric_trajectory") for row in trajectory)
    valid_indices = [
        index
        for index, row in enumerate(trajectory)
        if row["measurement_valid"] and (not primary_available or row.get("primary_metric_trajectory"))
    ]
    if len(valid_indices) < int(config["min_fit_points"]):
        return {"status": "insufficient", "fit_valid": False, "reason": "too_few_metric_points"}
    initial_indices = valid_indices[: min(5, len(valid_indices))]
    initial_z = float(np.median([trajectory[index]["z_m"] for index in initial_indices]))
    release_index = valid_indices[0]
    release_observed = False
    for index in valid_indices:
        if initial_z - float(trajectory[index]["z_m"]) >= float(config["release_drop_m"]):
            release_index = max(valid_indices[0], index - 1)
            release_observed = True
            break
    contact_index = valid_indices[-1] + 1
    contact_observed = False
    confirmation = int(config["contact_confirmation_frames"])
    for position, index in enumerate(valid_indices):
        if index <= release_index:
            continue
        window = valid_indices[position : position + confirmation]
        if len(window) == confirmation and all(float(trajectory[i]["z_m"]) <= contact_z + float(config["contact_margin_m"]) for i in window):
            contact_index = index
            contact_observed = True
            break
    if (
        not contact_observed
        and valid_indices
        and float(trajectory[valid_indices[-1]]["z_m"])
        <= contact_z + float(config["contact_margin_m"])
    ):
        # A bouncing sphere may occupy the contact band for a single decoded
        # frame.  The final point of the frame-0-connected airborne component
        # is therefore a valid contact event even without three resting frames.
        contact_index = valid_indices[-1]
        contact_observed = True
    fit_indices = [index for index in valid_indices if release_index <= index < contact_index]
    if len(fit_indices) < int(config["min_fit_points"]):
        return {
            "status": "insufficient",
            "fit_valid": False,
            "reason": "airborne_segment_too_short",
            "release_frame": release_index,
            "release_observed": release_observed,
            "contact_frame_exclusive": contact_index,
            "contact_observed": contact_observed,
            "fit_points": len(fit_indices),
        }
    absolute_times = np.asarray([trajectory[index]["time_s"] for index in fit_indices], dtype=float)
    t0 = float(absolute_times[0])
    times = absolute_times - t0
    heights = np.asarray([trajectory[index]["z_m"] for index in fit_indices], dtype=float)
    if float(times[-1] - times[0]) < float(config["min_fit_duration_s"]):
        return {
            "status": "insufficient",
            "fit_valid": False,
            "reason": "airborne_duration_too_short",
            "release_frame": release_index,
            "release_observed": release_observed,
            "contact_frame_exclusive": contact_index,
            "contact_observed": contact_observed,
            "fit_points": len(fit_indices),
        }
    coefficients, weights = _robust_quadratic(
        times,
        heights,
        int(config["robust_iterations"]),
        float(config["huber_k"]),
    )
    predicted = np.column_stack([np.ones(len(times)), times, times * times]) @ coefficients
    residuals = heights - predicted
    ss_res = float(np.sum(residuals * residuals))
    ss_tot = float(np.sum((heights - np.mean(heights)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    gravity = float(-2.0 * coefficients[2])
    similarity = None
    if target_gravity is not None and target_gravity > 0 and gravity > 0:
        similarity = float(math.exp(-abs(math.log(gravity / target_gravity))))
    for index, weight, fit_z, residual in zip(fit_indices, weights, predicted, residuals):
        trajectory[index]["physics_fit_used"] = True
        trajectory[index]["fit_z_m"] = float(fit_z)
        trajectory[index]["fit_residual_m"] = float(residual)
        trajectory[index]["fit_weight"] = float(weight)
    fit_valid = bool(math.isfinite(gravity) and gravity > 0 and r2 >= float(config["min_r2"]))
    status = "pass" if fit_valid and similarity is not None and similarity >= float(config["similarity_pass"]) else ("fail" if fit_valid else "invalid")
    return {
        "status": status,
        "fit_valid": fit_valid,
        "release_frame": release_index,
        "release_observed": release_observed,
        "contact_frame_exclusive": contact_index,
        "contact_observed": contact_observed,
        "fit_points": len(fit_indices),
        "fit_time_origin_s": t0,
        "z0_m": float(coefficients[0]),
        "v0_m_s": float(coefficients[1]),
        "quadratic_coefficient_m_s2": float(coefficients[2]),
        "estimated_gravity_m_s2": gravity,
        "target_gravity_m_s2": target_gravity,
        "parameter_similarity": similarity,
        "fit_r2": float(r2),
        "residual_rmse_m": float(math.sqrt(np.mean(residuals * residuals))),
        "trajectory_equation": "z(t)=z0+v0*(t-t0)-0.5*g*(t-t0)^2",
        "target_parameter_not_used_for_reconstruction_or_fit": True,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row[key], (list, dict)):
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _plot_trajectory(path: Path, trajectory: list[dict[str, Any]], physics: dict[str, Any], job: dict[str, Any]) -> None:
    valid = [row for row in trajectory if row["measurement_valid"]]
    figure = plt.figure(figsize=(12, 5.2))
    ax1 = figure.add_subplot(1, 2, 1)
    if valid:
        ax1.scatter([row["time_s"] for row in valid], [row["z_m"] for row in valid], s=14, label="metric reconstruction")
    fit = [row for row in trajectory if row.get("physics_fit_used")]
    if fit:
        ax1.plot([row["time_s"] for row in fit], [row["fit_z_m"] for row in fit], color="tab:red", label="robust free-fall fit")
    ax1.set(xlabel="time (s)", ylabel="world z (m)", title=f"{job['factors']['camera']} metric height")
    ax1.grid(alpha=0.25)
    if valid or fit:
        ax1.legend(loc="best")
    ax2 = figure.add_subplot(1, 2, 2, projection="3d")
    if valid:
        ax2.plot([row["x_m"] for row in valid], [row["y_m"] for row in valid], [row["z_m"] for row in valid], marker=".")
    ax2.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="V1A constrained 3D route")
    text = (
        f"g_est={physics.get('estimated_gravity_m_s2')} m/s²\n"
        f"R²={physics.get('fit_r2')}\n"
        f"benchmark_status={physics.get('benchmark_status', physics.get('status'))}\n"
        f"raw_status={physics.get('raw_status', physics.get('status'))}"
    )
    figure.text(0.50, 0.01, text, ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_overlay(path: Path, frames: list[np.ndarray], trajectory: list[dict[str, Any]], fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create overlay: {path}")
    trail: list[tuple[int, int]] = []
    for frame, row in zip(frames, trajectory):
        canvas = frame.copy()
        if row["found"]:
            center = (int(round(row["center_u_px"])), int(round(row["center_v_px"])))
            trail.append(center)
            display_radius = int(round(row["display_radius_px"]))
            measurement_radius = int(round(row["measurement_radius_px"]))
            color = (30, 220, 30) if row["measurement_valid"] else (0, 170, 255)
            cv2.circle(canvas, center, measurement_radius, (255, 190, 20), 2)
            cv2.rectangle(canvas, (center[0] - display_radius, center[1] - display_radius), (center[0] + display_radius, center[1] + display_radius), color, 2)
        if len(trail) >= 2:
            cv2.polylines(canvas, [np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)], False, (255, 80, 40), 2)
        z_text = "NA" if row["z_m"] is None else f"{row['z_m']:.3f}m"
        cv2.putText(canvas, f"frame={row['frame_index']} metric_z={z_text} valid={row['measurement_valid']}", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "cyan=circle measurement; green/orange=fixed-size QA box", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def _longest_false_run(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _interval_quality(
    trajectory: list[dict[str, Any]],
    start: int | None,
    end_exclusive: int | None,
) -> dict[str, Any]:
    if start is None or end_exclusive is None or end_exclusive <= start:
        return {
            "start_frame": start,
            "end_frame_exclusive": end_exclusive,
            "frames": 0,
            "valid_points": 0,
            "coverage_fraction": 0.0,
            "max_gap_frames": 0,
        }
    bounded_start = max(0, int(start))
    bounded_end = min(len(trajectory), int(end_exclusive))
    flags = [
        bool(trajectory[index].get("measurement_valid"))
        for index in range(bounded_start, bounded_end)
    ]
    return {
        "start_frame": bounded_start,
        "end_frame_exclusive": bounded_end,
        "frames": len(flags),
        "valid_points": int(sum(flags)),
        "coverage_fraction": float(sum(flags) / max(len(flags), 1)),
        "max_gap_frames": _longest_false_run(flags),
    }


def _geometry_interval_quality(
    trajectory: list[dict[str, Any]],
    start: int | None,
    end_exclusive: int | None,
) -> dict[str, Any]:
    if start is None or end_exclusive is None or end_exclusive <= start:
        rows: list[dict[str, Any]] = []
    else:
        rows = trajectory[max(0, int(start)) : min(len(trajectory), int(end_exclusive))]
    pnp = [row.get("geometry_source") == "anchor_pnp" for row in rows]
    metric = [bool(row.get("metric_geometry_valid")) for row in rows]
    return {
        "frames": len(rows),
        "metric_geometry_fraction": float(sum(metric) / max(len(metric), 1)),
        "pnp_fraction": float(sum(pnp) / max(len(pnp), 1)),
        "max_non_pnp_run_frames": _longest_false_run(pnp),
        "geometry_sources": sorted({str(row.get("geometry_source")) for row in rows}),
        "start_frame": start,
        "end_frame_exclusive": end_exclusive,
    }


def run_v1a_metric_job(
    job: dict[str, Any],
    workspace_root: Path,
    calibration_root: Path,
    output_root: Path,
    config: dict[str, Any] | None = None,
    make_overlay: bool = True,
    camera_motion_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = deepcopy(config or DEFAULT_V1A_CONFIG)
    opencv_rng_seed = int(config.get("runtime", {}).get("opencv_rng_seed", 20260720))
    cv2.setRNGSeed(opencv_rng_seed & 0x7FFFFFFF)
    video_path = Path(job["video_path"])
    sidecar_path = calibration_path(calibration_root, job)
    conditioning = conditioning_image_path(workspace_root, job)
    calibration = _load_calibration(sidecar_path)
    frames, media = read_video_frames(video_path)
    camera = calibration["camera"]
    k_reference = np.asarray(camera["K"], dtype=float)
    nominal_r = np.asarray(camera["R_world_to_camera_opencv"], dtype=float)
    nominal_t = np.asarray(camera["t_world_to_camera_opencv_m"], dtype=float)
    source_size = (int(calibration["image"]["width_px"]), int(calibration["image"]["height_px"]))
    target_size = (int(media["width"]), int(media["height"]))
    h_crop_resize = crop_resize_homography(source_size, target_size)
    # K remains a physical pinhole intrinsic matrix.  The additional frame-0
    # projective registration is applied to reference pixels/anchors, then PnP
    # absorbs the actual camera pose; it must not be folded into K.
    k_video = h_crop_resize @ k_reference
    nominal_p = compose_projection(k_video, nominal_r, nominal_t)
    reference_p = compose_projection(k_reference, nominal_r, nominal_t)
    x0, y0, start_z, contact_z, radius_m = _line_contract(calibration)
    reference_curve = _project_curve(reference_p, x0, y0, contact_z, start_z)
    crop_curve = _project_curve(nominal_p, x0, y0, contact_z, start_z)
    crop_radius = _calibrated_initial_radius(
        calibration,
        h_crop_resize,
        _initial_projected_radius(nominal_p, (x0, y0, start_z), radius_m),
    )
    verified_static = bool(
        config["camera_pose"].get("verified_static_camera", False)
        and config["camera_pose"].get("verified_static_evidence")
        == "legacy_v1a_clean_gt_validator"
    )
    camera_route = choose_reconstruction_route(camera_motion_evidence, video_path.name)
    audited_static = bool(camera_route["static_calibration_allowed"])
    fixed_camera = bool(verified_static or audited_static)
    if verified_static:
        h_ref_video = h_crop_resize
        registration = {
            "success": True,
            "method": "verified_static_calibration_bypass",
            "basis": "caller-declared clean Blender GT with explicit frame-0 projection validation",
            "inlier_count": None,
            "inlier_ratio": None,
            "reprojection_median_px": 0.0,
            "reprojection_p95_px": 0.0,
            "failed_gates": [],
        }
    else:
        h_ref_video, registration = register_conditioning_to_frame0(
            conditioning,
            frames[0],
            h_crop_resize,
            crop_curve,
            crop_radius,
            config["camera_pose"],
        )
    registered_nominal_p = h_ref_video @ reference_p
    initial_radius = _calibrated_initial_radius(
        calibration,
        h_ref_video,
        _initial_projected_radius(registered_nominal_p, (x0, y0, start_z), radius_m),
    )
    anchor_payload = _load_anchors(_anchor_path(sidecar_path, calibration))
    anchor_count_raw = 0 if anchor_payload is None else int(len(anchor_payload[0]))
    if anchor_payload is not None:
        # Never use pixels on the frozen sphere or anywhere along its possible
        # V1A path as static 2D-3D correspondences.  This also sanitises anchor
        # files exported by versions that ray-cast through dynamic objects.
        reference_radius = _calibrated_initial_radius(
            calibration,
            np.eye(3, dtype=np.float64),
            _initial_projected_radius(reference_p, (x0, y0, start_z), radius_m),
        )
        safe = _distance_to_polyline(anchor_payload[0], reference_curve) > max(
            24.0,
            5.0 * reference_radius,
        )
        anchor_payload = (anchor_payload[0][safe], anchor_payload[1][safe])
    anchor_count_safe = 0 if anchor_payload is None else int(len(anchor_payload[0]))
    anchor_geometry = _anchor_geometry_quality(
        None if anchor_payload is None else anchor_payload[1],
        config["camera_pose"],
    )
    pose_rows: list[dict[str, Any]]
    projections: list[np.ndarray]
    pose_mode: str
    if fixed_camera:
        static_projection = nominal_p if verified_static else registered_nominal_p
        static_geometry_source = (
            "verified_static_calibration"
            if verified_static
            else "audited_static_calibration"
        )
        pose_rows = [
            {
                "frame_index": index,
                "success": True,
                "quality": {"status": "pass", "passed": True, "failed_gates": []},
                "method": static_geometry_source,
                "geometry_source": static_geometry_source,
                "R": nominal_r.tolist(),
                "t": nominal_t.reshape(3).tolist(),
                "reprojection_median_px": 0.0,
                "reprojection_p95_px": 0.0,
            }
            for index in range(len(frames))
        ]
        projections = [static_projection.copy() for _ in frames]
        pnp_quality_fraction = 0.0
        pose_mode = static_geometry_source
    elif (
        config["camera_pose"]["use_pnp_anchors"]
        and bool(registration.get("success"))
        and anchor_payload is not None
        and anchor_count_safe >= int(config["camera_pose"]["min_anchor_tracks"])
        and bool(anchor_geometry["passed"])
    ):
        pose_rows, projections = _pose_sequence_from_anchors(
            frames,
            conditioning,
            anchor_payload[0],
            anchor_payload[1],
            h_ref_video,
            k_video,
            nominal_r,
            nominal_t,
            config["camera_pose"],
        )
        pnp_quality_fraction = float(np.mean([_pose_quality_passed(row) for row in pose_rows]))
        pose_mode = (
            "calibrated_6dof_anchor_pnp"
            if pnp_quality_fraction == 1.0
            else "mixed_pnp_projective_diagnostic"
        )
    else:
        pnp_quality_fraction = 0.0
        pose_rows = []
        projections = []
        pose_mode = "projective_background_diagnostic_only"
    raw_pnp_projections = [projection.copy() for projection in projections]
    if pose_rows and _pose_quality_passed(pose_rows[0]):
        frame0_projection = projections[0]
    else:
        frame0_projection = registered_nominal_p
    frame0_curve = _project_curve(frame0_projection, x0, y0, contact_z, start_z)
    background_h, background_rows = estimate_background_homographies(
        frames,
        frame0_curve,
        2.0 * initial_radius,
        min_inlier_ratio=float(config["validity"]["min_background_inlier_ratio_p10"]),
    )
    pose_drift = _pose_drift_summary(pose_rows, nominal_r, nominal_t)
    pnp_projective = (
        _pnp_projective_curve_disagreement(
            raw_pnp_projections,
            pose_rows,
            background_h,
            frame0_projection,
            x0,
            y0,
            contact_z,
            start_z,
        )
        if raw_pnp_projections
        else {
            "pnp_vs_projective_curve_median_p50_px": None,
            "pnp_vs_projective_curve_median_p95_px": None,
            "pnp_vs_projective_compared_frames": 0,
        }
    )
    if not projections:
        projections = [h @ frame0_projection for h in background_h]
        pose_rows = [
            {
                "frame_index": row["frame_index"],
                "success": row["success"],
                "quality": False,
                "method": row["method"],
                "geometry_source": "projective_diagnostic",
                "reprojection_median_px": row["reprojection_median_px"],
            }
            for row in background_rows
        ]
    elif not fixed_camera:
        for index, pose in enumerate(pose_rows):
            if not _pose_quality_passed(pose):
                projections[index] = background_h[index] @ frame0_projection
                pose["fallback_method"] = "reference_lk_homography_diagnostic"
                pose["geometry_source"] = "projective_diagnostic"
    curves = [_project_curve(projection, x0, y0, contact_z, start_z) for projection in projections]
    tracks, tracking_summary = track_ball_sequence(
        frames,
        curves,
        initial_radius,
        hue_tolerance=float(config["tracking"]["hue_tolerance"]),
        min_circularity=float(config["tracking"]["min_circularity"]),
        min_radius_ratio=float(config["tracking"]["min_radius_ratio"]),
        max_radius_ratio=float(config["tracking"]["max_radius_ratio"]),
        max_curve_distance_diameters=float(config["tracking"]["max_curve_distance_diameters"]),
        max_prediction_distance_diameters=float(config["tracking"]["max_prediction_distance_diameters"]),
        prediction_reset_gap_frames=int(config["tracking"]["prediction_reset_gap_frames"]),
        trusted_geometry=[
            row.get("geometry_source")
            in {
                "anchor_pnp",
                "verified_static_calibration",
                "audited_static_calibration",
            }
            for row in pose_rows
        ],
    )
    trajectory, trajectory_summary = reconstruct_track(
        tracks,
        projections,
        pose_rows,
        k_video,
        nominal_r,
        nominal_t,
        x0,
        y0,
        start_z,
        contact_z,
        radius_m,
        float(media["fps"]),
        config,
        registration_reprojection_p95_px=registration.get("reprojection_p95_px"),
        hard_sphere_constraint=bool(
            audited_static
            and config["static_camera"]["hard_sphere_radius_constraint"]
        ),
        min_observed_expected_radius_ratio=float(
            config["static_camera"]["min_observed_expected_radius_ratio"]
        ),
        max_observed_expected_radius_ratio=float(
            config["static_camera"]["max_observed_expected_radius_ratio"]
        ),
    )
    physics = fit_freefall_metric(
        trajectory,
        contact_z,
        float(job["targets"]["gravity_g"]),
        config["physics"],
    )
    background_success = float(np.mean([bool(row["success"]) for row in background_rows]))
    background_failure_run = _longest_false_run(bool(row["success"]) for row in background_rows)
    background_inlier_ratio_p10 = _percentile(
        [float(row.get("inlier_ratio") or 0.0) for row in background_rows],
        10,
    )
    video_corners = np.asarray(
        [
            [0.0, 0.0],
            [float(media["width"] - 1), 0.0],
            [float(media["width"] - 1), float(media["height"] - 1)],
            [0.0, float(media["height"] - 1)],
        ],
        dtype=np.float64,
    )
    background_corner_motion_p95 = _percentile(
        [
            float(
                np.median(
                    np.linalg.norm(
                        _transform_uv(video_corners, homography) - video_corners,
                        axis=1,
                    )
                )
            )
            for homography in background_h
        ],
        95,
    )
    background_p95 = _percentile(
        [
            row["reprojection_median_px"]
            for row in background_rows
            if row.get("success") and row.get("reprojection_median_px") is not None
        ],
        95,
    )
    primary_geometry = _geometry_interval_quality(
        trajectory,
        trajectory_summary["primary_component_start_frame"],
        None
        if trajectory_summary["primary_component_end_frame"] is None
        else int(trajectory_summary["primary_component_end_frame"]) + 1,
    )
    fit_indices = [
        index for index, row in enumerate(trajectory) if row.get("physics_fit_used")
    ]
    fit_geometry = _geometry_interval_quality(
        trajectory,
        None if not fit_indices else fit_indices[0],
        None if not fit_indices else fit_indices[-1] + 1,
    )
    airborne_interval = _interval_quality(
        trajectory,
        physics.get("release_frame"),
        physics.get("contact_frame_exclusive")
        if physics.get("contact_observed")
        else None,
    )
    constrained_failed: list[str] = []
    if not verified_static and not bool(registration.get("success")):
        constrained_failed.append("conditioning_frame0_registration_failed")
    if (
        not fixed_camera
        and (
            anchor_payload is None
            or anchor_count_safe < int(config["camera_pose"]["min_anchor_tracks"])
            or not bool(anchor_geometry["passed"])
        )
    ):
        constrained_failed.append("missing_safe_metric_background_anchors")
    if trajectory_summary["primary_component_points"] < int(config["physics"]["min_fit_points"]):
        constrained_failed.append("insufficient_primary_metric_points")
    if trajectory_summary["primary_component_start_frame"] != 0:
        constrained_failed.append("primary_identity_not_anchored_at_frame0")
    if trajectory_summary["primary_component_coverage"] < float(config["validity"]["min_valid_fraction"]):
        constrained_failed.append("fragmented_primary_metric_trajectory")
    if trajectory_summary["primary_max_gap_frames"] > int(config["validity"]["max_airborne_gap_frames"]):
        constrained_failed.append("primary_metric_gap_too_long")
    if trajectory_summary["primary_component_duration_s"] < float(config["validity"]["min_primary_duration_s"]):
        constrained_failed.append("primary_metric_duration_too_short")
    if (
        trajectory_summary["primary_component_vertical_span_m"] is None
        or trajectory_summary["primary_component_vertical_span_m"]
        < float(config["validity"]["min_primary_vertical_span_m"])
    ):
        constrained_failed.append("insufficient_primary_vertical_span")
    if not trajectory_summary["initial_frame_metric_valid"]:
        constrained_failed.append("initial_metric_sphere_not_valid")
    if (
        trajectory_summary["initial_center_error_px"] is None
        or trajectory_summary["initial_center_error_px"]
        > float(config["validity"]["max_initial_center_error_diameters"])
        * 2.0
        * initial_radius
    ):
        constrained_failed.append("initial_sphere_center_mismatch")
    if (
        trajectory_summary["initial_height_error_m"] is None
        or trajectory_summary["initial_height_error_m"]
        > float(config["validity"]["max_initial_height_error_m"])
    ):
        constrained_failed.append("initial_metric_height_mismatch")
    if trajectory_summary["line_residual_p95_px"] is None or trajectory_summary["line_residual_p95_px"] > float(config["validity"]["max_line_residual_p95_px"]):
        constrained_failed.append("object_deviates_from_v1a_motion_line")
    background_quality_failed = bool(
        background_success < float(config["validity"]["min_background_success_fraction"])
        or background_failure_run > int(config["validity"]["max_background_failure_run"])
        or background_inlier_ratio_p10 is None
        or background_inlier_ratio_p10
        < float(config["validity"]["min_background_inlier_ratio_p10"])
        or background_p95 is None
        or background_p95 > float(config["validity"]["max_background_reprojection_p95_px"])
    )
    # Dynamic-camera metric geometry requires independently observable
    # background motion.  A fixed-camera path has already been selected by an
    # external whole-video motion audit, so this second LK pass is diagnostic.
    if not fixed_camera and background_quality_failed:
        constrained_failed.append("nonrigid_or_untracked_background")
    fixed_static_corner_motion_warning = bool(
        fixed_camera
        and (
            background_corner_motion_p95 is None
            or background_corner_motion_p95
            > float(config["validity"]["max_verified_static_corner_motion_p95_px"])
        )
    )
    if trajectory_summary["z_sigma_p95_m"] is None or trajectory_summary["z_sigma_p95_m"] > float(config["validity"]["max_metric_sigma_p95_m"]):
        constrained_failed.append("metric_uncertainty_too_high")
    object_scale_consistent = bool(
        trajectory_summary["radius_based_lateral_offset_p95_m"] is not None
        and trajectory_summary["radius_based_lateral_offset_p95_m"]
        <= float(config["validity"]["max_radius_based_lateral_offset_p95_m"])
    )
    sphere_constraint_sufficient = bool(
        not audited_static
        or not config["static_camera"]["hard_sphere_radius_constraint"]
        or trajectory_summary["sphere_size_valid_fraction"]
        >= float(config["static_camera"]["min_sphere_size_valid_fraction"])
    )
    if not sphere_constraint_sufficient:
        constrained_failed.append("known_sphere_size_constraint_insufficient")
    if primary_geometry["metric_geometry_fraction"] < 1.0:
        constrained_failed.append("primary_contains_nonmetric_geometry")
    if not fixed_camera and (
        primary_geometry["pnp_fraction"]
        < float(config["camera_pose"]["min_primary_pnp_fraction"])
        or primary_geometry["max_non_pnp_run_frames"]
        > int(config["camera_pose"]["max_primary_pnp_failure_run"])
    ):
        constrained_failed.append("primary_pnp_coverage_insufficient")
    pnp_disagreement = pnp_projective["pnp_vs_projective_curve_median_p95_px"]
    if not fixed_camera and primary_geometry["pnp_fraction"] > 0 and (
        pnp_disagreement is None
        or pnp_disagreement > float(config["validity"]["max_pnp_vs_projective_curve_p95_px"])
    ):
        constrained_failed.append("pnp_pose_inconsistent_with_background_projective_motion")
    constrained_failed = list(dict.fromkeys(constrained_failed))
    constrained_valid = not constrained_failed
    airborne_failed: list[str] = []
    if not physics.get("release_observed"):
        airborne_failed.append("release_not_observed")
    if not physics.get("contact_observed"):
        airborne_failed.append("contact_not_observed")
    if (
        airborne_interval["coverage_fraction"]
        < float(config["validity"]["min_airborne_coverage_fraction"])
    ):
        airborne_failed.append("airborne_interval_coverage_insufficient")
    if airborne_interval["max_gap_frames"] > int(config["validity"]["max_airborne_gap_frames"]):
        airborne_failed.append("airborne_interval_gap_too_long")
    airborne_valid = constrained_valid and not airborne_failed
    failed_gates = constrained_failed + airborne_failed
    strict_metric = bool(
        airborne_valid
        and not fixed_camera
        and primary_geometry["pnp_fraction"]
        >= float(config["camera_pose"]["min_primary_pnp_fraction"])
        and primary_geometry["max_non_pnp_run_frames"]
        <= int(config["camera_pose"]["max_primary_pnp_failure_run"])
        and (
            fit_geometry["frames"] == 0
            or fit_geometry["pnp_fraction"] == 1.0
        )
    )
    physics["raw_status"] = physics.get("status")
    physics["benchmark_status"] = (
        physics.get("status")
        if airborne_valid
        else "not_scorable_invalid_reconstruction"
    )
    job_dir = output_root / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(job_dir / "trajectory_metric.csv", trajectory)
    _write_csv(job_dir / "camera_pose.csv", pose_rows)
    _write_jsonl(job_dir / "camera_pose.jsonl", pose_rows)
    _write_csv(job_dir / "background_motion.csv", background_rows)
    _plot_trajectory(job_dir / "trajectory_metric.png", trajectory, physics, job)
    overlay_error = None
    if make_overlay:
        try:
            _write_overlay(job_dir / "object_track_overlay.mp4", frames, trajectory, float(media["fps"]))
        except Exception as exc:  # diagnostics should not erase numerical results
            overlay_error = f"{type(exc).__name__}: {exc}"
    result = {
        "schema_version": "1.1.0",
        "module_version": V1A_METRIC_VERSION,
        "job_id": job["job_id"],
        "experiment_id": "v1_A",
        "factors": job["factors"],
        "targets": job["targets"],
        "media": media,
        "inputs": {
            "video": str(video_path),
            "video_sha256": _sha256(video_path),
            "conditioning_image": str(conditioning),
            "conditioning_image_sha256": _sha256(conditioning),
            "calibration_sidecar": str(sidecar_path),
            "calibration_sidecar_sha256": _sha256(sidecar_path),
            "source_blend_sha256": calibration.get("source", {}).get("blend_sha256"),
            "anchor_npz": None if _anchor_path(sidecar_path, calibration) is None else str(_anchor_path(sidecar_path, calibration)),
            "anchor_npz_sha256": None
            if _anchor_path(sidecar_path, calibration) is None
            or not _anchor_path(sidecar_path, calibration).is_file()
            else _sha256(_anchor_path(sidecar_path, calibration)),
            "resolved_config": config,
            "resolved_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "code_provenance": _code_provenance(opencv_rng_seed),
        "geometry": {
            "trajectory_mode": "known_v1a_vertical_line_metric_3d",
            "camera_pose_mode": pose_mode,
            "hybrid_camera_route": camera_route,
            "camera_motion_evidence": camera_motion_evidence,
            "reference_to_video_homography": h_ref_video.tolist(),
            "nominal_crop_resize_homography": h_crop_resize.tolist(),
            "frame0_registration": registration,
            "K_video": k_video.tolist(),
            "world_motion_line_xy_m": [x0, y0],
            "start_z_m": start_z,
            "contact_z_m": contact_z,
            "known_sphere_radius_m": radius_m,
            "initial_projected_radius_px": initial_radius,
            "pnp_quality_fraction": pnp_quality_fraction,
            "anchor_count_raw": anchor_count_raw,
            "anchor_count_after_dynamic_corridor_filter": anchor_count_safe,
            "anchor_geometry_quality": anchor_geometry,
            "primary_geometry_quality": primary_geometry,
            "physics_fit_geometry_quality": fit_geometry,
            "background_homography_success_fraction": background_success,
            "background_homography_max_failure_run_frames": background_failure_run,
            "background_homography_inlier_ratio_p10": background_inlier_ratio_p10,
            "background_corner_motion_p95_px": background_corner_motion_p95,
            "background_reprojection_p95_px": background_p95,
            "background_diagnostics_role": (
                "diagnostic_only_in_hash_pinned_verified_static_validation"
                if verified_static
                else (
                    "diagnostic_after_external_fixed_camera_audit"
                    if audited_static
                    else "independent_dynamic_camera_quality_gate"
                )
            ),
            **pose_drift,
            **pnp_projective,
            "uncertainty_model": (
                "first_order_pixel_equivalent_from_center_registration_and_pnp_residuals; "
                "anchor rank is gated, but full K/R/t covariance is not estimated"
            ),
        },
        "tracking": tracking_summary,
        "trajectory": trajectory_summary,
        "reconstruction_validity": {
            "status": "valid" if airborne_valid else "invalid",
            "constrained_metric_3d_valid": constrained_valid,
            "airborne_metric_trajectory_valid": airborne_valid,
            "whole_video_identity_valid": tracking_summary["detection_fraction"] >= float(config["tracking"]["min_detection_fraction"]),
            "rigid_sphere_scale_consistency_valid": (
                sphere_constraint_sufficient if audited_static else object_scale_consistent
            ),
            "hard_sphere_constraint_passed": sphere_constraint_sufficient,
            "strict_dynamic_camera_metric_3d_valid": strict_metric,
            "failed_gates": failed_gates,
            "constrained_failed_gates": constrained_failed,
            "airborne_failed_gates": airborne_failed,
            "airborne_interval": airborne_interval,
            "warnings": (
                []
                if tracking_summary["detection_fraction"] >= float(config["tracking"]["min_detection_fraction"])
                else ["post_release_or_post_contact_object_identity_is_incomplete"]
            )
            + (
                []
                if (sphere_constraint_sufficient if audited_static else object_scale_consistent)
                else ["known_sphere_radius_is_inconsistent_with_apparent_video_size"]
            )
            + (
                []
                if not (fixed_camera and background_quality_failed)
                else ["fixed_camera_background_lk_is_not_fully_observable"]
            )
            + (
                []
                if not fixed_static_corner_motion_warning
                else ["fixed_camera_background_homography_has_apparent_motion"]
            ),
            "interpretation": "Metric xyz is recovered on the experiment-defined V1A vertical line using either an externally audited fixed camera with one frame-0 calibration registration, or quality-passed anchor-PnP. On the audited-static route, the projected known sphere radius is a hard per-frame gate. Projective homography rows are diagnostic and never enter metric validity or physics fitting.",
        },
        "physics_fit": physics,
        "overlay_error": overlay_error,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "trajectory_csv": "trajectory_metric.csv",
            "camera_pose_csv": "camera_pose.csv",
            "camera_pose_jsonl": "camera_pose.jsonl",
            "background_motion_csv": "background_motion.csv",
            "trajectory_plot": "trajectory_metric.png",
            "overlay_video": "object_track_overlay.mp4" if make_overlay and overlay_error is None else None,
        },
    }
    write_json(job_dir / "result.json", result)
    return result


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    validity = result["reconstruction_validity"]
    physics = result["physics_fit"]
    factors = result["factors"]
    return {
        "job_id": result["job_id"],
        "scene_id": factors["scene_id"],
        "camera": factors["camera"],
        "target_gravity_m_s2": result["targets"]["gravity_g"],
        "reconstruction_status": validity["status"],
        "strict_dynamic_camera_metric_3d_valid": validity["strict_dynamic_camera_metric_3d_valid"],
        "failed_gates": ";".join(validity["failed_gates"]),
        "camera_pose_mode": result["geometry"]["camera_pose_mode"],
        "hybrid_route": result["geometry"]["hybrid_camera_route"]["route"],
        "detection_fraction": result["tracking"]["detection_fraction"],
        "metric_valid_fraction": result["trajectory"]["valid_fraction"],
        "sphere_size_valid_fraction": result["trajectory"].get("sphere_size_valid_fraction"),
        "line_residual_p95_px": result["trajectory"]["line_residual_p95_px"],
        "estimated_gravity_m_s2": physics.get("estimated_gravity_m_s2"),
        "parameter_similarity": physics.get("parameter_similarity"),
        "fit_r2": physics.get("fit_r2"),
        "physics_status": physics.get("benchmark_status", physics.get("status")),
        "physics_raw_status": physics.get("raw_status", physics.get("status")),
        "physics_reason": physics.get("reason"),
        "rigid_sphere_scale_consistency_valid": validity.get("rigid_sphere_scale_consistency_valid"),
        "warnings": ";".join(validity.get("warnings") or []),
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _write_report(output_root: Path, results: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    rows = []
    for result in results:
        summary = _summary_row(result)
        job_id = html.escape(result["job_id"])
        gravity = summary["estimated_gravity_m_s2"]
        gravity_text = "" if gravity is None else f"{gravity:.3f}"
        fit_r2 = summary["fit_r2"]
        fit_r2_text = "" if fit_r2 is None else f"{fit_r2:.3f}"
        physics_text = html.escape(str(summary["physics_status"]))
        physics_reason = html.escape(str(summary.get("physics_reason") or ""))
        gate_text = html.escape(str(summary.get("failed_gates") or ""))
        warning_text = html.escape(str(summary.get("warnings") or ""))
        overlay = result.get("artifacts", {}).get("overlay_video")
        overlay_link = (
            f" · <a href='../{job_id}/{html.escape(str(overlay))}'>overlay</a>"
            if overlay
            else " · overlay not generated"
        )
        rows.append(
            "<tr>"
            f"<td><a href='../{job_id}/result.json'>{job_id}</a></td>"
            f"<td>{html.escape(str(summary['reconstruction_status']))}</td>"
            f"<td>{html.escape(str(summary['camera_pose_mode']))}</td>"
            f"<td>{summary['detection_fraction']:.3f}</td>"
            f"<td>{summary['metric_valid_fraction']:.3f}</td>"
            f"<td>{summary['target_gravity_m_s2']}</td>"
            f"<td>{gravity_text}</td>"
            f"<td>{fit_r2_text}</td>"
            f"<td>{physics_text}</td>"
            f"<td>{physics_reason}</td>"
            f"<td>{gate_text}</td>"
            f"<td>{warning_text}</td>"
            f"<td><a href='../{job_id}/trajectory_metric.png'>trajectory</a>{overlay_link}</td>"
            "</tr>"
        )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>V1A metric reconstruction</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d7dee7;padding:.45rem;text-align:left}}th{{background:#edf2f7}}.note{{padding:1rem;background:#fff7d6;border-left:4px solid #e0a800}}</style></head><body>
<h1>V1A metric 3D trajectory validation</h1><div class='note'>Primary xyz uses the declared V1A vertical motion line plus calibrated camera geometry. Target gravity is withheld until the independent trajectory fit has completed. A projective background fallback is reported separately from true 6-DoF PnP.</div>
<p>jobs={aggregate['jobs']} · reconstruction valid={aggregate['reconstruction_valid']} · strict dynamic-camera valid={aggregate['strict_dynamic_camera_valid']} · physics pass={aggregate['physics_pass']}</p>
<table><thead><tr><th>job</th><th>reconstruction</th><th>pose mode</th><th>detection</th><th>metric valid</th><th>target g</th><th>estimated g</th><th>R²</th><th>physics</th><th>physics reason</th><th>reconstruction gates</th><th>warnings</th><th>artifacts</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text(document, encoding="utf-8")


def run_v1a_metric_batch(
    workspace_root: Path,
    videos_dir: Path,
    calibration_root: Path,
    output_root: Path,
    config: dict[str, Any] | None = None,
    *,
    overlay_count: int = 6,
    limit: int | None = None,
    video_glob: str = "*.mp4",
) -> dict[str, Any]:
    videos = sorted(videos_dir.glob(video_glob))
    if limit is not None:
        videos = videos[: max(0, int(limit))]
    if not videos:
        raise FileNotFoundError(f"no MP4 files matching {video_glob!r} found in {videos_dir}")
    jobs = [parse_v1a_video_job(path) for path in videos]
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, 1):
        result = run_v1a_metric_job(
            job,
            workspace_root,
            calibration_root,
            output_root,
            config=config,
            make_overlay=index <= overlay_count,
        )
        results.append(result)
        print(f"[{index}/{len(jobs)}] {job['job_id']}: {result['reconstruction_validity']['status']}", flush=True)
    rows = [_summary_row(result) for result in results]
    _write_csv(output_root / "summary.csv", rows)
    with (output_root / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    aggregate = {
        "schema_version": "1.1.0",
        "module_version": V1A_METRIC_VERSION,
        "jobs": len(results),
        "reconstruction_valid": sum(result["reconstruction_validity"]["airborne_metric_trajectory_valid"] for result in results),
        "constrained_metric_valid": sum(result["reconstruction_validity"]["constrained_metric_3d_valid"] for result in results),
        "strict_dynamic_camera_valid": sum(result["reconstruction_validity"]["strict_dynamic_camera_metric_3d_valid"] for result in results),
        "physics_pass": sum(result["physics_fit"].get("benchmark_status") == "pass" for result in results),
        "physics_fail": sum(result["physics_fit"].get("benchmark_status") == "fail" for result in results),
        "physics_invalid_shape": sum(result["physics_fit"].get("benchmark_status") == "invalid" for result in results),
        "physics_insufficient": sum(result["physics_fit"].get("benchmark_status") == "insufficient" for result in results),
        "physics_not_scorable_invalid_reconstruction": sum(
            result["physics_fit"].get("benchmark_status")
            == "not_scorable_invalid_reconstruction"
            for result in results
        ),
    }
    write_json(output_root / "aggregate.json", aggregate)
    _write_report(output_root, results, aggregate)
    return aggregate
