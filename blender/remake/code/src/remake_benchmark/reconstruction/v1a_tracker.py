from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    center: tuple[float, float]
    radius_px: float
    area_px: float
    circularity: float
    hue_distance: float
    contour: np.ndarray
    # Default preserves compatibility with existing positional construction in
    # tests and any downstream callers created before boundary-aware gating.
    touches_frame_boundary: bool = False


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"empty video: {video_path}")
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    height, width = frames[0].shape[:2]
    return frames, {
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_frames": len(frames),
        "reported_frames": reported_frames,
        "duration_s": len(frames) / fps,
    }


def _hue_distance(values: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(values.astype(np.float32) - float(center))
    return np.minimum(delta, 180.0 - delta)


def estimate_ball_colour(
    frame: np.ndarray,
    expected_center: tuple[float, float],
    expected_radius_px: float,
) -> dict[str, float]:
    """Estimate the orange sphere colour only inside its calibrated frame-0 ROI."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    cx, cy = expected_center
    radius = max(5, int(round(expected_radius_px * 1.35)))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), radius, 255, -1)
    pixels = hsv[mask > 0]
    vivid = pixels[(pixels[:, 1] >= 70) & (pixels[:, 2] >= 70)]
    if len(vivid) < 12:
        vivid = pixels[(pixels[:, 1] >= 35) & (pixels[:, 2] >= 45)]
    if not len(vivid):
        return {"hue": 12.0, "saturation_min": 55.0, "value_min": 45.0}
    # The ROI can include green boards or colourful indoor props.  The standard
    # sphere is deliberately orange, so select the warm subset before taking a
    # robust circular hue median.  This does not inspect any later frames.
    warm = vivid[(vivid[:, 0] <= 35) | (vivid[:, 0] >= 170)]
    if len(warm) >= 12:
        vivid = warm
    angles = vivid[:, 0].astype(np.float64) * (2.0 * np.pi / 180.0)
    hue = (math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))) % (2.0 * np.pi))
    hue *= 180.0 / (2.0 * np.pi)
    return {
        "hue": float(hue),
        "saturation_min": float(max(40.0, np.percentile(vivid[:, 1], 8) * 0.65)),
        "value_min": float(max(35.0, np.percentile(vivid[:, 2], 5) * 0.55)),
    }


def orange_ball_mask(frame: np.ndarray, colour: dict[str, float], hue_tolerance: float = 16.0) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_ok = _hue_distance(hsv[:, :, 0], float(colour["hue"])) <= float(hue_tolerance)
    mask = (
        hue_ok
        & (hsv[:, :, 1] >= float(colour["saturation_min"]))
        & (hsv[:, :, 2] >= float(colour["value_min"]))
    ).astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=2)
    return mask


def _corridor_mask(shape: tuple[int, int], curve_uv: np.ndarray, radius_px: float) -> np.ndarray:
    height, width = shape
    output = np.zeros((height, width), dtype=np.uint8)
    finite = np.all(np.isfinite(curve_uv), axis=1)
    points = np.rint(curve_uv[finite]).astype(np.int32)
    if len(points) >= 2:
        cv2.polylines(output, [points.reshape(-1, 1, 2)], False, 255, max(3, int(round(2.0 * radius_px))))
    elif len(points) == 1:
        cv2.circle(output, tuple(points[0]), max(3, int(round(radius_px))), 255, -1)
    return output


def enumerate_ball_candidates(
    frame: np.ndarray,
    colour: dict[str, float],
    curve_uv: np.ndarray,
    reference_radius_px: float,
    *,
    hue_tolerance: float = 16.0,
    corridor_radius_diameters: float = 2.25,
) -> tuple[list[BallCandidate], np.ndarray]:
    mask = orange_ball_mask(frame, colour, hue_tolerance=hue_tolerance)
    corridor_radius = max(18.0, 2.0 * reference_radius_px * corridor_radius_diameters)
    corridor = _corridor_mask(mask.shape, curve_uv, corridor_radius)
    masked = cv2.bitwise_and(mask, corridor)
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[BallCandidate] = []
    min_area = max(12.0, math.pi * (0.35 * reference_radius_px) ** 2)
    max_area = math.pi * (2.4 * reference_radius_px) ** 2
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 1e-9 else float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < 0.24:
            continue
        contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)
        touches_frame_boundary = bool(
            contour_x <= 0
            or contour_y <= 0
            or contour_x + contour_width >= mask.shape[1]
            or contour_y + contour_height >= mask.shape[0]
        )
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) <= 1e-9:
            continue
        center = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
        if len(contour) >= 5:
            _, axes, _ = cv2.fitEllipse(contour)
            radius = 0.25 * float(axes[0] + axes[1])
        else:
            _, radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
        if not (0.35 * reference_radius_px <= radius <= 2.25 * reference_radius_px):
            continue
        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_values = hsv[:, :, 0][component_mask > 0]
        hue_distance = float(np.median(_hue_distance(hue_values, float(colour["hue"])))) if len(hue_values) else 180.0
        candidates.append(
            BallCandidate(
                center=center,
                radius_px=radius,
                area_px=area,
                circularity=circularity,
                hue_distance=hue_distance,
                contour=contour,
                touches_frame_boundary=touches_frame_boundary,
            )
        )
    # Indoor4/CAM_Side begins in front of a similarly coloured tool board, so
    # an HSV connected component can merge the ball with the background.  Edge
    # circles provide a geometry-only fallback; the calibrated corridor and
    # temporal gates below still decide identity.
    has_plausible_colour_ball = any(
        item.circularity >= 0.45
        and 0.50 <= item.radius_px / max(reference_radius_px, 1e-6) <= 1.60
        and _distance_to_curve(item.center, curve_uv) <= 0.85 * 2.0 * reference_radius_px
        for item in candidates
    )
    hough = None
    if not has_plausible_colour_ball:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
        hough = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8.0, 1.2 * reference_radius_px),
            param1=110.0,
            param2=24.0,
            minRadius=max(3, int(round(0.50 * reference_radius_px))),
            maxRadius=max(5, int(round(1.52 * reference_radius_px))),
        )
    if hough is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for u, v, radius in hough[0]:
            center = (float(u), float(v))
            if _distance_to_curve(center, curve_uv) > 0.85 * 2.0 * reference_radius_px:
                continue
            if any(np.linalg.norm(np.asarray(center) - np.asarray(item.center)) < 0.45 * reference_radius_px for item in candidates):
                continue
            inner = np.zeros(mask.shape, dtype=np.uint8)
            cv2.circle(inner, (int(round(u)), int(round(v))), max(2, int(round(0.72 * radius))), 255, -1)
            vivid = (inner > 0) & (hsv[:, :, 1] >= float(colour["saturation_min"]))
            hues = hsv[:, :, 0][vivid]
            if len(hues) < max(10, int(0.08 * math.pi * radius * radius)):
                continue
            hue_distance = float(np.median(_hue_distance(hues, float(colour["hue"]))))
            if hue_distance > max(18.0, hue_tolerance + 3.0):
                continue
            angles = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
            contour = np.rint(
                np.column_stack([u + radius * np.cos(angles), v + radius * np.sin(angles)])
            ).astype(np.int32).reshape(-1, 1, 2)
            candidates.append(
                BallCandidate(
                    center=center,
                    radius_px=float(radius),
                    area_px=float(math.pi * radius * radius),
                    circularity=0.98,
                    hue_distance=hue_distance,
                    contour=contour,
                    touches_frame_boundary=bool(
                        u - radius <= 0.0
                        or v - radius <= 0.0
                        or u + radius >= mask.shape[1] - 1
                        or v + radius >= mask.shape[0] - 1
                    ),
                )
            )
    return candidates, masked


def _distance_to_curve(point: tuple[float, float], curve: np.ndarray) -> float:
    finite = curve[np.all(np.isfinite(curve), axis=1)]
    if not len(finite):
        return float("inf")
    return float(np.min(np.linalg.norm(finite - np.asarray(point, dtype=float), axis=1)))


def choose_candidate(
    candidates: Iterable[BallCandidate],
    curve_uv: np.ndarray,
    reference_radius_px: float,
    predicted_center: tuple[float, float] | None,
    previous_radius_px: float | None,
    *,
    min_circularity: float = 0.50,
    min_radius_ratio: float = 0.55,
    max_radius_ratio: float = 1.50,
    max_curve_distance_diameters: float = 0.75,
    max_prediction_distance_diameters: float = 1.25,
) -> tuple[BallCandidate | None, dict[str, float]]:
    best: BallCandidate | None = None
    best_terms: dict[str, float] = {}
    best_cost = float("inf")
    diameter = max(2.0 * reference_radius_px, 1.0)
    for candidate in candidates:
        curve_distance = _distance_to_curve(candidate.center, curve_uv) / diameter
        prediction_distance = 0.0
        if predicted_center is not None:
            prediction_distance = float(
                np.linalg.norm(np.asarray(candidate.center) - np.asarray(predicted_center)) / diameter
            )
        radius_reference = previous_radius_px or reference_radius_px
        radius_change = abs(math.log(max(candidate.radius_px, 1e-6) / max(radius_reference, 1e-6)))
        radius_ratio = candidate.radius_px / max(reference_radius_px, 1e-6)
        required_circularity = (
            min(0.28, min_circularity)
            if candidate.touches_frame_boundary
            else min_circularity
        )
        if (
            candidate.circularity < required_circularity
            or radius_ratio < min_radius_ratio
            or radius_ratio > max_radius_ratio
            or curve_distance > max_curve_distance_diameters
            or (predicted_center is not None and prediction_distance > max_prediction_distance_diameters)
        ):
            continue
        shape_penalty = max(0.0, 0.78 - candidate.circularity)
        colour_penalty = candidate.hue_distance / 16.0
        cost = (
            1.65 * prediction_distance
            + 1.10 * curve_distance
            + 1.30 * radius_change
            + 0.70 * shape_penalty
            + 0.25 * colour_penalty
        )
        if cost < best_cost:
            best = candidate
            best_cost = cost
            best_terms = {
                "association_cost": float(cost),
                "curve_distance_px": float(curve_distance * diameter),
                "prediction_distance_px": float(prediction_distance * diameter),
                "radius_log_change": float(radius_change),
            }
    # A strict local association is what prevents an indoor orange book or tool
    # from becoming the object after one missed frame.
    return best, best_terms


def estimate_background_homographies(
    frames: list[np.ndarray],
    frame0_curve_uv: np.ndarray,
    object_diameter_px: float,
    *,
    max_corners: int = 900,
    ransac_threshold_px: float = 2.5,
    min_inlier_ratio: float = 0.60,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Track static background from frame 0 and fit a projective image warp.

    This is an auditable fallback for generated videos without usable 3D anchor
    matches.  It must not be labelled a recovered 6-DoF camera pose.
    """
    reference = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    feature_mask = np.full(reference.shape, 255, dtype=np.uint8)
    corridor = _corridor_mask(reference.shape, frame0_curve_uv, max(30.0, 2.5 * object_diameter_px))
    feature_mask[corridor > 0] = 0
    border = max(5, int(round(min(reference.shape) * 0.02)))
    feature_mask[:border] = feature_mask[-border:] = 0
    feature_mask[:, :border] = feature_mask[:, -border:] = 0
    points0 = cv2.goodFeaturesToTrack(
        reference,
        mask=feature_mask,
        maxCorners=max_corners,
        qualityLevel=0.008,
        minDistance=7,
        blockSize=7,
    )
    identity = np.eye(3, dtype=np.float64)
    homographies = [identity.copy()]
    diagnostics: list[dict[str, Any]] = [
        {
            "frame_index": 0,
            "success": points0 is not None,
            "method": "identity",
            "tracked_features": 0 if points0 is None else int(len(points0)),
            "inliers": 0 if points0 is None else int(len(points0)),
            "inlier_ratio": 1.0 if points0 is not None else 0.0,
            "reprojection_median_px": 0.0,
        }
    ]
    previous_h = identity.copy()
    for frame_index, frame in enumerate(frames[1:], 1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = previous_h.copy()
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "success": False,
            "method": "previous_fallback",
            "tracked_features": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reprojection_median_px": None,
        }
        if points0 is not None:
            points1, status1, _ = cv2.calcOpticalFlowPyrLK(
                reference,
                gray,
                points0,
                None,
                winSize=(25, 25),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
            )
            if points1 is not None and status1 is not None:
                back, status2, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    reference,
                    points1,
                    None,
                    winSize=(25, 25),
                    maxLevel=4,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
                )
                valid = status1.ravel().astype(bool)
                if back is not None and status2 is not None:
                    fb = np.linalg.norm(points0.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
                    valid &= status2.ravel().astype(bool) & (fb <= 1.5)
                p0 = points0.reshape(-1, 2)[valid]
                p1 = points1.reshape(-1, 2)[valid]
                row["tracked_features"] = int(len(p0))
                if len(p0) >= 12:
                    estimate, inlier_mask = cv2.findHomography(
                        p0,
                        p1,
                        cv2.RANSAC,
                        ransac_threshold_px,
                        maxIters=2500,
                        confidence=0.995,
                    )
                    if estimate is not None and inlier_mask is not None:
                        inliers = inlier_mask.ravel().astype(bool)
                        predicted = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), estimate).reshape(-1, 2)
                        residual = np.linalg.norm(predicted - p1, axis=1)
                        if int(np.sum(inliers)) >= 12 and float(np.mean(inliers)) >= float(min_inlier_ratio):
                            h = estimate.astype(np.float64)
                            if abs(float(h[2, 2])) > 1e-12:
                                h /= h[2, 2]
                            row.update(
                                {
                                    "success": True,
                                    "method": "reference_lk_homography_ransac",
                                    "inliers": int(np.sum(inliers)),
                                    "inlier_ratio": float(np.mean(inliers)),
                                    "reprojection_median_px": float(np.median(residual[inliers])),
                                }
                            )
        if row["success"]:
            previous_h = h.copy()
        homographies.append(h)
        diagnostics.append(row)
    return homographies, diagnostics


def apply_homography(points_uv: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points_uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)


def track_ball_sequence(
    frames: list[np.ndarray],
    nominal_curves_uv: list[np.ndarray],
    initial_radius_px: float,
    *,
    hue_tolerance: float = 16.0,
    min_circularity: float = 0.50,
    min_radius_ratio: float = 0.55,
    max_radius_ratio: float = 1.50,
    max_curve_distance_diameters: float = 0.75,
    max_prediction_distance_diameters: float = 1.25,
    prediction_reset_gap_frames: int = 3,
    trusted_geometry: list[bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(frames) != len(nominal_curves_uv):
        raise ValueError("one projected V1A curve is required per frame")
    if trusted_geometry is not None and len(trusted_geometry) != len(frames):
        raise ValueError("trusted_geometry must contain one flag per frame")
    initial_center = tuple(float(v) for v in nominal_curves_uv[0][-1])
    colour = estimate_ball_colour(frames[0], initial_center, initial_radius_px)
    rows: list[dict[str, Any]] = []
    trusted_centers: list[np.ndarray] = []
    trusted_frames: list[int] = []
    trusted_radius: float | None = None
    diagnostic_centers: list[np.ndarray] = []
    diagnostic_frames: list[int] = []
    diagnostic_radius: float | None = None
    for frame_index, (frame, curve) in enumerate(zip(frames, nominal_curves_uv)):
        frame_trusted = True if trusted_geometry is None else bool(trusted_geometry[frame_index])
        candidates, _ = enumerate_ball_candidates(
            frame,
            colour,
            curve,
            initial_radius_px,
            hue_tolerance=hue_tolerance,
        )
        if frame_trusted:
            state_centers = trusted_centers
            state_frames = trusted_frames
            state_radius = trusted_radius
        elif diagnostic_centers:
            state_centers = diagnostic_centers
            state_frames = diagnostic_frames
            state_radius = diagnostic_radius
        else:
            # A diagnostic run may start immediately after a trusted frame,
            # but its updates are kept in a separate state thereafter.
            state_centers = trusted_centers
            state_frames = trusted_frames
            state_radius = trusted_radius
        predicted: tuple[float, float] | None
        if not state_centers:
            predicted = initial_center
        elif frame_index - state_frames[-1] > prediction_reset_gap_frames:
            # After a real occlusion, the calibrated motion corridor is safer
            # than extrapolating stale velocity.  This permits a true sphere to
            # re-enter while the hard corridor gate still rejects indoor props.
            predicted = None
        elif len(state_centers) == 1:
            predicted = tuple(float(v) for v in state_centers[-1])
        else:
            previous_delta = max(1, state_frames[-1] - state_frames[-2])
            forward_delta = max(1, frame_index - state_frames[-1])
            velocity = (state_centers[-1] - state_centers[-2]) / previous_delta
            step = np.clip(velocity * forward_delta, -2.0 * initial_radius_px, 2.0 * initial_radius_px)
            predicted_array = state_centers[-1] + step
            predicted = tuple(float(v) for v in predicted_array)
        candidate, terms = choose_candidate(
            candidates,
            curve,
            initial_radius_px,
            predicted,
            state_radius,
            min_circularity=min_circularity,
            min_radius_ratio=min_radius_ratio,
            max_radius_ratio=max_radius_ratio,
            max_curve_distance_diameters=max_curve_distance_diameters,
            max_prediction_distance_diameters=max_prediction_distance_diameters,
        )
        if candidate is None:
            rows.append(
                {
                    "frame_index": frame_index,
                    "found": False,
                    "candidate_count": len(candidates),
                    "center_u_px": None,
                    "center_v_px": None,
                    "measurement_radius_px": None,
                    "display_radius_px": float(initial_radius_px),
                    "geometry_trusted_for_identity_update": frame_trusted,
                    **terms,
                }
            )
            continue
        center_array = np.asarray(candidate.center, dtype=float)
        if frame_trusted:
            trusted_centers.append(center_array)
            trusted_frames.append(frame_index)
            trusted_radius = candidate.radius_px
        else:
            diagnostic_centers.append(center_array)
            diagnostic_frames.append(frame_index)
            diagnostic_radius = candidate.radius_px
        rows.append(
            {
                "frame_index": frame_index,
                "found": True,
                "candidate_count": len(candidates),
                "center_u_px": float(candidate.center[0]),
                "center_v_px": float(candidate.center[1]),
                "measurement_radius_px": float(candidate.radius_px),
                # The visual QA box is deliberately size-stable.  It cannot
                # feed back into the physical radius/depth measurement above.
                "display_radius_px": float(initial_radius_px),
                "contour_area_px": float(candidate.area_px),
                "circularity": float(candidate.circularity),
                "hue_distance": float(candidate.hue_distance),
                "touches_frame_boundary": bool(candidate.touches_frame_boundary),
                "geometry_trusted_for_identity_update": frame_trusted,
                **terms,
            }
        )
    found = [row for row in rows if row["found"]]
    summary = {
        "method": "calibrated_vertical_corridor_orange_component_local_association",
        "colour_model": colour,
        "frames": len(rows),
        "found": len(found),
        "detection_fraction": len(found) / max(len(rows), 1),
        "measurement_radius_median_px": None
        if not found
        else float(np.median([row["measurement_radius_px"] for row in found])),
        "measurement_radius_cv": None
        if len(found) < 2
        else float(np.std([row["measurement_radius_px"] for row in found]) / max(np.mean([row["measurement_radius_px"] for row in found]), 1e-9)),
        "display_box_locked": True,
        "diagnostic_geometry_cannot_update_trusted_identity": True,
    }
    return rows, summary
