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
from dataclasses import dataclass, replace
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
    circularity: float | None
    ellipse_major_axis_px: float | None
    ellipse_minor_axis_px: float | None
    ellipse_angle_deg: float | None
    ellipse_axis_ratio: float | None
    radial_residual_ratio: float | None
    hue_distance: float
    boundary: bool
    measurement_source: str
    # A Hough circle is a useful centre/radius observation, but its circularity
    # is an assumption made by the detector rather than measured silhouette
    # evidence.  Keeping this flag explicit prevents downstream validity checks
    # from mistaking a detector prior for proof that the generated object is
    # rigid and round.
    shape_evidence_available: bool = True
    colour_support_fraction: float | None = None
    shape_evidence_source: str | None = "segmentation_contour"
    edge_support_fraction: float | None = None
    edge_radial_residual_ratio: float | None = None
    # Fraction of pixels inside the candidate disk that differ materially from
    # frame zero.  For a fixed camera this separates a moving generated ball
    # from same-colour circular props that remain baked into the background.
    reference_change_fraction: float | None = None


_MAX_TRUSTED_ASSOCIATION_COST = 1.20
_MIN_TRUSTED_CONFIDENCE = math.exp(-_MAX_TRUSTED_ASSOCIATION_COST)
_MAX_IMMEDIATE_INNOVATION_RADII = 1.55
_MAX_COLLISION_CONTINUITY_INNOVATION_RADII = 2.25
_HOUGH_SHAPE_MIN_COLOUR_SUPPORT = 0.45
_HOUGH_SHAPE_MIN_EDGE_SUPPORT = 0.55
_HOUGH_SHAPE_MAX_RADIAL_RESIDUAL = 0.10
_REFERENCE_CHANGE_PIXEL_THRESHOLD = 18
_REFERENCE_CHANGE_LOW_SUPPORT = 0.20
_REFERENCE_CHANGE_HIGH_SUPPORT = 0.55


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


def _reference_change_mask(reference_frame: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Return compression-tolerant foreground change relative to frame zero."""

    reference = cv2.GaussianBlur(reference_frame, (3, 3), 0.0)
    current = cv2.GaussianBlur(frame, (3, 3), 0.0)
    delta = cv2.absdiff(reference, current)
    return np.max(delta, axis=2) >= _REFERENCE_CHANGE_PIXEL_THRESHOLD


def _with_reference_change(
    candidates: Sequence[SphereCandidate],
    change_mask: np.ndarray,
) -> list[SphereCandidate]:
    height, width = change_mask.shape[:2]
    output: list[SphereCandidate] = []
    for candidate in candidates:
        # The inner disk is less sensitive to Hough radius over-estimation and
        # floor/prop edges than a full bounding box.
        radius = max(2.0, 0.78 * float(candidate.radius_px))
        x0 = max(0, int(math.floor(float(candidate.center_u_px) - radius)))
        y0 = max(0, int(math.floor(float(candidate.center_v_px) - radius)))
        x1 = min(width, int(math.ceil(float(candidate.center_u_px) + radius)) + 1)
        y1 = min(height, int(math.ceil(float(candidate.center_v_px) + radius)) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (
            (xx - float(candidate.center_u_px)) ** 2
            + (yy - float(candidate.center_v_px)) ** 2
            <= radius**2
        )
        crop = change_mask[y0:y1, x0:x1]
        support = float(np.mean(crop[disk])) if np.any(disk) else 0.0
        output.append(replace(candidate, reference_change_fraction=support))
    return output


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


def _edge_ring_evidence(
    edge_map: np.ndarray,
    center_xy: Sequence[float],
    radius_px: float,
    *,
    angular_bins: int = 72,
    search_band_ratio: float = 0.45,
    support_residual_ratio: float = 0.13,
) -> tuple[float, float | None]:
    """Measure actual edge support around a proposed circle.

    The Hough proposal itself is not evidence: this function goes back to the
    Canny pixels and, independently for each angular sector, records the nearest
    radial edge.  A partial arc therefore has low angular coverage, while an
    ellipse has high residuals in the sectors where its boundary departs from
    the proposed radius.
    """

    radius = float(radius_px)
    if radius <= 1e-6 or angular_bins < 8:
        return 0.0, None
    ys, xs = np.nonzero(np.asarray(edge_map) > 0)
    if not len(xs):
        return 0.0, None
    dx = xs.astype(np.float64) - float(center_xy[0])
    dy = ys.astype(np.float64) - float(center_xy[1])
    radial = np.hypot(dx, dy)
    band = np.abs(radial - radius) <= search_band_ratio * radius
    if not np.any(band):
        return 0.0, None
    dx = dx[band]
    dy = dy[band]
    radial = radial[band]
    angles = np.mod(np.arctan2(dy, dx), 2.0 * math.pi)
    bins = np.minimum(
        angular_bins - 1,
        np.floor(angles * angular_bins / (2.0 * math.pi)).astype(np.int32),
    )
    residual = np.abs(radial - radius) / radius
    nearest = np.full(angular_bins, np.inf, dtype=np.float64)
    np.minimum.at(nearest, bins, residual)
    finite = np.isfinite(nearest)
    supported = finite & (nearest <= support_residual_ratio)
    support_fraction = float(np.mean(supported))
    radial_residual = float(np.median(nearest[finite])) if np.any(finite) else None
    return support_fraction, radial_residual


def _hough_candidates(
    frame: np.ndarray,
    colour: Mapping[str, float],
    reference_radius_px: float,
    predicted: np.ndarray | None,
    *,
    roi_radius_in_reference_radii: float = 4.0,
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> list[SphereCandidate]:
    height, width = frame.shape[:2]
    if roi_bounds is not None:
        x0 = max(0, min(width, int(roi_bounds[0])))
        y0 = max(0, min(height, int(roi_bounds[1])))
        x1 = max(x0, min(width, int(roi_bounds[2])))
        y1 = max(y0, min(height, int(roi_bounds[3])))
    elif predicted is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        half_size = max(24, int(math.ceil(float(roi_radius_in_reference_radii) * reference_radius_px)))
        x0 = max(0, int(math.floor(float(predicted[0]))) - half_size)
        y0 = max(0, int(math.floor(float(predicted[1]))) - half_size)
        x1 = min(width, int(math.ceil(float(predicted[0]))) + half_size + 1)
        y1 = min(height, int(math.ceil(float(predicted[1]))) + half_size + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []

    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    # Generated videos often soften the ball boundary substantially.  A low
    # Canny threshold recovers that edge; the angular-coverage, radial-residual,
    # and colour-support conjunction below rejects unrelated weak clutter.
    canny = cv2.Canny(gray, 25, 80, L2gradient=True)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=max(8.0, 1.2 * reference_radius_px),
        param1=110.0,
        # Use a permissive proposal threshold, then retain circles only after
        # the independent colour/edge/radius checks below.  This improves
        # recall for compressed indoor videos without weakening identity.
        param2=18.0,
        minRadius=max(3, int(round(0.45 * reference_radius_px))),
        maxRadius=max(5, int(round(1.8 * reference_radius_px))),
    )
    if circles is None:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output: list[SphereCandidate] = []
    for crop_u, crop_v, radius in circles[0]:
        u = float(crop_u) + x0
        v = float(crop_v) + y0
        center = np.asarray([u, v])
        if predicted is not None and np.linalg.norm(center - predicted) > float(
            roi_radius_in_reference_radii * reference_radius_px
        ):
            continue
        region = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(region, (int(round(u)), int(round(v))), max(2, int(round(0.7 * radius))), 255, -1)
        region_pixels = region > 0
        colour_support = (
            region_pixels
            & (hsv[:, :, 1] >= float(colour["saturation_min"]))
            & (hsv[:, :, 2] >= float(colour["value_min"]))
            & (_hue_distance(hsv[:, :, 0], float(colour["hue"])) <= 22.0)
        )
        support_fraction = float(np.sum(colour_support)) / max(1, int(np.sum(region_pixels)))
        if support_fraction < 0.30:
            continue
        hues = hsv[:, :, 0][colour_support]
        if len(hues) < 8:
            continue
        hue_distance = float(np.median(_hue_distance(hues, float(colour["hue"]))))
        if hue_distance > 22.0:
            continue
        edge_support, edge_radial_residual = _edge_ring_evidence(
            canny,
            (float(crop_u), float(crop_v)),
            float(radius),
        )
        shape_evidence = bool(
            support_fraction >= _HOUGH_SHAPE_MIN_COLOUR_SUPPORT
            and edge_support >= _HOUGH_SHAPE_MIN_EDGE_SUPPORT
            and edge_radial_residual is not None
            and edge_radial_residual <= _HOUGH_SHAPE_MAX_RADIAL_RESIDUAL
        )
        output.append(
            SphereCandidate(
                u,
                v,
                float(radius),
                float(math.pi * radius**2),
                None,
                None,
                None,
                None,
                None,
                None,
                hue_distance,
                bool(
                    u - radius <= 0
                    or v - radius <= 0
                    or u + radius >= frame.shape[1]
                    or v + radius >= frame.shape[0]
                ),
                "hough_circle_fallback",
                shape_evidence,
                support_fraction,
                "hough_edge_ring" if shape_evidence else None,
                edge_support,
                edge_radial_residual,
            )
        )
    return output


def _manifold_hough_candidates(
    frame: np.ndarray,
    colour: Mapping[str, float],
    reference_radius_px: float,
    centers: Sequence[np.ndarray],
    expected: np.ndarray,
    experiment_id: str | None,
) -> list[SphereCandidate]:
    """Search a frozen line-motion corridor after local prediction is lost."""

    manifold = LINE_MANIFOLDS.get(str(experiment_id))
    if manifold is None:
        return []
    height, width = frame.shape[:2]
    recent = (
        np.stack([np.asarray(center, dtype=np.float64) for center in centers[-9:]])
        if centers
        else np.asarray([expected], dtype=np.float64)
    )
    center = np.median(recent, axis=0)
    # Hough voting needs some surrounding edge context even though the final
    # centre is constrained to the much tighter manifold below.
    half_width = max(28, int(math.ceil(4.00 * reference_radius_px)))
    if manifold == "vertical":
        bounds = (
            int(math.floor(float(center[0]))) - half_width,
            0,
            int(math.ceil(float(center[0]))) + half_width + 1,
            height,
        )
    else:
        bounds = (
            0,
            int(math.floor(float(center[1]))) - half_width,
            width,
            int(math.ceil(float(center[1]))) + half_width + 1,
        )
    return _hough_candidates(
        frame,
        colour,
        reference_radius_px,
        None,
        roi_bounds=bounds,
    )


def _association_terms(
    item: SphereCandidate,
    predicted: np.ndarray,
    previous_radius: float,
    reference_radius: float,
    *,
    gap: int,
) -> dict[str, float | bool]:
    center = np.asarray([item.center_u_px, item.center_v_px], dtype=np.float64)
    distance = float(np.linalg.norm(center - predicted))
    radius_ratio = item.radius_px / max(reference_radius, 1e-6)
    radius_change = abs(math.log(max(item.radius_px, 1e-6) / max(previous_radius, 1e-6)))

    # A good motion prediction makes a wide, full-frame colour search
    # unnecessary.  Uncertainty grows mildly across missed observations, but a
    # distant orange prop must never become reachable simply because the sphere
    # was absent for a few frames.
    gap_growth = min(1.5, 0.75 * math.sqrt(max(0, int(gap) - 1)))
    max_distance = (2.75 + gap_growth) * reference_radius
    position_pass = distance <= max_distance
    # The benchmark sphere is rigid and its first-frame projected size is
    # frozen.  This is deliberately tighter than Hough's proposal range: a
    # large circular wall prop must not become the ball merely because the
    # motion prediction temporarily lags an accelerating trajectory.
    radius_pass = 0.65 <= radius_ratio <= 1.40

    circularity = item.circularity if item.shape_evidence_available else None
    axis_ratio = item.ellipse_axis_ratio if item.shape_evidence_available else None
    radial_residual = item.radial_residual_ratio if item.shape_evidence_available else None
    shape_penalty = 0.0 if circularity is None else max(0.0, 0.72 - circularity)
    elongation_penalty = 0.0 if axis_ratio is None else max(0.0, axis_ratio - 1.45)
    radial_penalty = 0.0 if radial_residual is None else max(0.0, radial_residual - 0.18)
    boundary_penalty = 0.25 if item.boundary else 0.0
    source_penalty = 0.04 if item.measurement_source == "hough_circle_fallback" else 0.0
    colour_support_penalty = (
        0.0
        if item.colour_support_fraction is None
        else max(0.0, 0.55 - item.colour_support_fraction)
    )
    reference_change_term = 0.0
    if item.reference_change_fraction is not None:
        support = float(item.reference_change_fraction)
        if support < _REFERENCE_CHANGE_LOW_SUPPORT:
            # A candidate that looks unchanged from frame zero is likely a
            # static prop.  Keep the penalty small at a precise motion
            # prediction so pendulums/bounces may legitimately revisit their
            # initial image location.
            innovation_scale = min(
                1.0,
                distance / max(0.45 * reference_radius, 1.0),
            )
            deficit = (_REFERENCE_CHANGE_LOW_SUPPORT - support) / _REFERENCE_CHANGE_LOW_SUPPORT
            reference_change_term = 1.20 * deficit * innovation_scale
        elif support >= _REFERENCE_CHANGE_HIGH_SUPPORT:
            # Strong frame-zero foreground change is independent evidence for
            # the generated moving object.  It lets an accelerating ball beat
            # a nearer but static same-colour background circle.
            strength = min(
                1.0,
                (support - _REFERENCE_CHANGE_HIGH_SUPPORT)
                / max(1.0 - _REFERENCE_CHANGE_HIGH_SUPPORT, 1e-6),
            )
            reference_change_term = -0.20 - 0.40 * strength
    cost = (
        distance / max(2.0 * reference_radius, 1.0)
        + 1.15 * radius_change
        + 0.8 * shape_penalty
        + 0.28 * elongation_penalty
        + 0.35 * radial_penalty
        + 0.18 * item.hue_distance / 18.0
        + boundary_penalty
        + source_penalty
        + 0.40 * colour_support_penalty
        + reference_change_term
    )
    return {
        "cost": float(cost),
        "prediction_distance_px": distance,
        "max_prediction_distance_px": float(max_distance),
        "position_pass": position_pass,
        "radius_pass": radius_pass,
        "reference_change_fraction": item.reference_change_fraction,
        "reference_change_term": float(reference_change_term),
    }


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
    for item in candidates:
        terms = _association_terms(
            item,
            predicted,
            previous_radius,
            reference_radius,
            gap=gap,
        )
        if not bool(terms["position_pass"]) or not bool(terms["radius_pass"]):
            continue
        cost = float(terms["cost"])
        if cost < best_cost:
            best, best_cost = item, cost
    return best, best_cost


def _candidate_needs_hough(
    candidate: SphereCandidate | None,
    cost: float,
    predicted: np.ndarray,
    previous_radius: float,
    reference_radius: float,
) -> bool:
    if candidate is not None and candidate.boundary:
        # A clipped segmentation component can bias both centre and radius.
        # Ask the independent circle detector for a second local proposal;
        # the association/ambiguity gates below still decide whether either
        # proposal is trustworthy.
        return True
    if candidate is None or not math.isfinite(cost) or cost > 0.90:
        return True
    distance = float(
        np.linalg.norm(
            np.asarray([candidate.center_u_px, candidate.center_v_px], dtype=np.float64) - predicted
        )
    )
    if distance > 1.65 * reference_radius:
        return True
    if abs(math.log(max(candidate.radius_px, 1e-6) / max(previous_radius, 1e-6))) > math.log(1.28):
        return True
    reference_ratio = candidate.radius_px / max(reference_radius, 1e-6)
    if not (0.82 <= reference_ratio <= 1.18):
        return True
    if candidate.shape_evidence_available:
        if candidate.circularity is not None and candidate.circularity < 0.68:
            return True
        if candidate.ellipse_axis_ratio is not None and candidate.ellipse_axis_ratio > 1.35:
            return True
        if candidate.radial_residual_ratio is not None and candidate.radial_residual_ratio > 0.24:
            return True
    return False


def _calibration_supported_boundary_candidate(
    candidate: SphereCandidate | None,
    predicted: np.ndarray,
    reference_radius: float,
    frame_shape: Sequence[int],
    *,
    plausible_candidate_count: int,
    association_cost: float,
) -> bool:
    """Allow a clipped ball only when frozen calibration predicts the clip.

    Top-view first frames can place the sphere partly outside the image.  A
    blanket boundary rejection then loses the only metric alignment anchor.
    This exception remains deliberately local: the prediction itself must
    intersect the image edge, exactly one plausible identity cluster may be
    present, and the selected warm-colour candidate must pass the normal
    association gate.  It therefore cannot turn an arbitrary boundary blob
    into a full-frame re-detection.
    """

    if candidate is None or not candidate.boundary:
        return False
    if plausible_candidate_count != 1:
        return False
    if not math.isfinite(association_cost) or association_cost > _MAX_TRUSTED_ASSOCIATION_COST:
        return False
    if candidate.hue_distance > 12.0:
        return False
    height, width = int(frame_shape[0]), int(frame_shape[1])
    radius = float(reference_radius)
    prediction_intersects_boundary = bool(
        float(predicted[0]) - radius <= 1.0
        or float(predicted[1]) - radius <= 1.0
        or float(predicted[0]) + radius >= width - 1.0
        or float(predicted[1]) + radius >= height - 1.0
    )
    if not prediction_intersects_boundary:
        return False
    distance = float(
        np.linalg.norm(
            np.asarray([candidate.center_u_px, candidate.center_v_px], dtype=np.float64)
            - predicted
        )
    )
    return distance <= 1.10 * radius


def _plausible_candidate_clusters(
    candidates: Sequence[SphereCandidate],
    predicted: np.ndarray,
    previous_radius: float,
    reference_radius: float,
    *,
    gap: int,
) -> list[tuple[float, np.ndarray]]:
    # Segmentation and Hough can describe the same physical sphere.  Cluster
    # accepted centres before reporting ambiguity so the two detector sources
    # do not count as two competing objects.
    plausible: list[tuple[float, np.ndarray]] = []
    for item in candidates:
        terms = _association_terms(
            item,
            predicted,
            previous_radius,
            reference_radius,
            gap=gap,
        )
        if (
            bool(terms["position_pass"])
            and bool(terms["radius_pass"])
            and float(terms["cost"]) <= _MAX_TRUSTED_ASSOCIATION_COST
        ):
            plausible.append(
                (
                    float(terms["cost"]),
                    np.asarray([item.center_u_px, item.center_v_px], dtype=np.float64),
                )
            )
    clusters: list[tuple[float, np.ndarray]] = []
    for cost, center in sorted(plausible, key=lambda value: value[0]):
        if not any(
            np.linalg.norm(center - existing_center) <= 0.65 * reference_radius
            for _, existing_center in clusters
        ):
            clusters.append((cost, center))
    return clusters


def _plausible_candidate_count(
    candidates: Sequence[SphereCandidate],
    predicted: np.ndarray,
    previous_radius: float,
    reference_radius: float,
    *,
    gap: int,
) -> int:
    return len(
        _plausible_candidate_clusters(
            candidates,
            predicted,
            previous_radius,
            reference_radius,
            gap=gap,
        )
    )


def _persistent_distractor_match(
    candidate: SphereCandidate | None,
    memory: Sequence[Mapping[str, Any]],
    frame_index: int,
    reference_radius: float,
) -> bool:
    if candidate is None:
        return False
    if (
        candidate.reference_change_fraction is not None
        and candidate.reference_change_fraction >= _REFERENCE_CHANGE_HIGH_SUPPORT
    ):
        # A candidate whose interior changed strongly from the frozen first
        # frame is not the unchanged prop stored in distractor memory.  This
        # matters after impact: the real sphere can stop at a location where a
        # previously rejected warm-colour blob was remembered.
        return False
    center = np.asarray([candidate.center_u_px, candidate.center_v_px], dtype=np.float64)
    for item in memory:
        if int(item["count"]) < 3 or int(item["last_frame"]) < frame_index - 1:
            continue
        remembered_radius = float(item["radius_px"])
        radius_ratio = candidate.radius_px / max(remembered_radius, 1e-6)
        if not (0.72 <= radius_ratio <= 1.38):
            continue
        if np.linalg.norm(center - np.asarray(item["center"], dtype=np.float64)) <= 0.42 * reference_radius:
            return True
    return False


def _update_persistent_distractor_memory(
    memory: list[dict[str, Any]],
    candidates: Sequence[SphereCandidate],
    accepted: SphereCandidate | None,
    frame_index: int,
    reference_radius: float,
) -> None:
    # Only unselected candidates enter this memory.  A correctly tracked ball
    # may stop after impact, but it can therefore never label itself as a static
    # distractor.  Conversely, a same-colour prop visible beside the ball for
    # several earlier frames remains identifiable when the real ball is absent.
    accepted_center = (
        None
        if accepted is None
        else np.asarray([accepted.center_u_px, accepted.center_v_px], dtype=np.float64)
    )
    unique: list[SphereCandidate] = []
    for candidate in sorted(candidates, key=lambda item: not item.shape_evidence_available):
        center = np.asarray([candidate.center_u_px, candidate.center_v_px], dtype=np.float64)
        if accepted_center is not None and np.linalg.norm(center - accepted_center) <= 0.75 * reference_radius:
            continue
        if any(
            np.linalg.norm(
                center - np.asarray([other.center_u_px, other.center_v_px], dtype=np.float64)
            )
            <= 0.35 * reference_radius
            for other in unique
        ):
            continue
        unique.append(candidate)

    for candidate in unique:
        center = np.asarray([candidate.center_u_px, candidate.center_v_px], dtype=np.float64)
        best_index = None
        best_distance = float("inf")
        for index, item in enumerate(memory):
            distance = float(
                np.linalg.norm(center - np.asarray(item["center"], dtype=np.float64))
            )
            if distance <= 0.32 * reference_radius and distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is None:
            memory.append(
                {
                    "center": center.tolist(),
                    "radius_px": float(candidate.radius_px),
                    "count": 1,
                    "last_frame": int(frame_index),
                }
            )
            continue
        item = memory[best_index]
        consecutive = int(item["last_frame"]) == frame_index - 1
        count = int(item["count"]) + 1 if consecutive else 1
        blend = 1.0 / min(count, 8)
        old_center = np.asarray(item["center"], dtype=np.float64)
        item.update(
            {
                "center": ((1.0 - blend) * old_center + blend * center).tolist(),
                "radius_px": float(
                    (1.0 - blend) * float(item["radius_px"]) + blend * candidate.radius_px
                ),
                "count": count,
                "last_frame": int(frame_index),
            }
        )
    # Keep a short grace period for a distractor missed by one noisy mask, but
    # do not accumulate a full-video catalogue of stale candidates.
    memory[:] = [item for item in memory if int(item["last_frame"]) >= frame_index - 2]


def _robust_recent_radius(radii: Sequence[float], reference_radius: float) -> float:
    if not radii:
        return float(reference_radius)
    recent = np.asarray(radii[-7:], dtype=np.float64)
    finite = recent[np.isfinite(recent)]
    return float(np.median(finite)) if len(finite) else float(reference_radius)


def _robust_recent_prediction(
    centers: Sequence[np.ndarray],
    center_frames: Sequence[int],
    frame_index: int,
    reference_radius: float,
    expected: np.ndarray,
) -> np.ndarray:
    if not centers:
        return expected.copy()
    last = np.asarray(centers[-1], dtype=np.float64)
    gap = max(1, int(frame_index) - int(center_frames[-1]))
    if len(centers) < 2:
        return last

    recent_centers = centers[-6:]
    recent_frames = center_frames[-6:]
    velocities: list[np.ndarray] = []
    for start, end, start_frame, end_frame in zip(
        recent_centers[:-1],
        recent_centers[1:],
        recent_frames[:-1],
        recent_frames[1:],
    ):
        dt = max(1, int(end_frame) - int(start_frame))
        velocities.append((np.asarray(end) - np.asarray(start)) / dt)
    robust_velocity = np.median(np.stack(velocities), axis=0)
    latest_velocity = velocities[-1]
    # Recent real impacts can reverse velocity.  Honour a trusted latest step
    # when it differs materially from the history; otherwise blend it with the
    # median to suppress pixel jitter.
    if np.linalg.norm(latest_velocity - robust_velocity) > 1.25 * reference_radius:
        velocity = latest_velocity
    else:
        velocity = 0.65 * latest_velocity + 0.35 * robust_velocity
    speed = float(np.linalg.norm(velocity))
    max_speed = 5.0 * reference_radius
    if speed > max_speed:
        velocity = velocity * (max_speed / speed)
    # Extrapolate normally through a short occlusion.  Beyond three missed
    # frames, damp stale velocity so the search ROI stays near the last credible
    # object path rather than sweeping into distant background clutter.
    effective_gap = float(min(gap, 3))
    if gap > 3:
        effective_gap += sum(0.72**offset for offset in range(1, gap - 2))
    return last + velocity * effective_gap


def _gt_manifold_reacquisition_candidate(
    candidates: Sequence[SphereCandidate],
    centers: Sequence[np.ndarray],
    center_frames: Sequence[int],
    frame_index: int,
    expected: np.ndarray,
    reference_radius: float,
    experiment_id: str | None,
) -> tuple[SphereCandidate | None, float, float | None]:
    """Strictly re-acquire a lost sphere using frozen first-frame evidence.

    This is intentionally not an unrestricted nearest-colour search.  A
    proposal must have the calibrated size/colour, differ strongly from frame
    zero, remain compatible with the experiment's known 2-D motion manifold,
    and be a clear winner.  The independent constraints let an accelerating
    falling ball recover after a stale constant-velocity prediction without
    allowing an indoor orange prop to become the tracked object.
    """

    if frame_index <= 0 or not centers:
        return None, float("inf"), None

    last = np.asarray(centers[-1], dtype=np.float64)
    recent = np.stack([np.asarray(center, dtype=np.float64) for center in centers[-9:]])
    reference_line_center = np.median(recent, axis=0)
    manifold = LINE_MANIFOLDS.get(str(experiment_id))
    gap = max(1, int(frame_index) - int(center_frames[-1]))
    # One surprising frame is held for future confirmation.  Reacquisition is
    # only enabled after at least one full missed observation, which prevents a
    # newly appearing same-colour object from hijacking the track immediately.
    if gap < 2:
        return None, float("inf"), None
    maximum_step = (10.0 + min(8.0, 1.5 * max(0, gap - 1))) * reference_radius
    scored: list[tuple[float, SphereCandidate, float]] = []
    for item in candidates:
        if item.boundary or item.hue_distance > 12.0:
            continue
        radius_ratio = float(item.radius_px) / max(reference_radius, 1e-6)
        if not (0.65 <= radius_ratio <= 1.55):
            continue
        change = item.reference_change_fraction
        if change is None or float(change) < _REFERENCE_CHANGE_HIGH_SUPPORT:
            continue
        center = np.asarray([item.center_u_px, item.center_v_px], dtype=np.float64)
        step = float(np.linalg.norm(center - last))
        if step > maximum_step:
            continue
        if manifold == "vertical":
            cross_track = abs(float(center[0] - reference_line_center[0]))
            if cross_track > 1.65 * reference_radius:
                continue
        elif manifold == "horizontal":
            cross_track = abs(float(center[1] - reference_line_center[1]))
            if cross_track > 1.65 * reference_radius:
                continue
        else:
            # For curved/compound experiments the frozen first frame does not
            # define a global line.  Permit only a bounded, unique reappearance;
            # the colour, size and reference-change gates still all apply.
            cross_track = 0.0
            if step > (7.0 + min(5.0, float(gap))) * reference_radius:
                continue
        source_penalty = 0.08 if item.measurement_source == "hough_circle_fallback" else 0.0
        shape_penalty = 0.0
        if item.shape_evidence_available:
            if item.circularity is not None:
                shape_penalty += max(0.0, 0.58 - float(item.circularity))
            if item.ellipse_axis_ratio is not None:
                shape_penalty += 0.12 * max(0.0, float(item.ellipse_axis_ratio) - 1.8)
        score = (
            0.40 * cross_track / max(reference_radius, 1e-6)
            + 0.20 * step / max(maximum_step, 1e-6)
            + 0.75 * abs(math.log(max(radius_ratio, 1e-6)))
            + 0.16 * item.hue_distance / 12.0
            + 0.35 * (1.0 - float(change))
            + source_penalty
            + shape_penalty
        )
        scored.append((float(score), item, step))

    if not scored:
        return None, float("inf"), None
    scored.sort(key=lambda value: value[0])
    best_score, best, best_step = scored[0]
    if best_score > 1.05:
        return None, float("inf"), None
    if len(scored) >= 2:
        second_score, second, _ = scored[1]
        separation = float(
            np.linalg.norm(
                np.asarray([best.center_u_px, best.center_v_px], dtype=np.float64)
                - np.asarray([second.center_u_px, second.center_v_px], dtype=np.float64)
            )
        )
        if separation > 0.65 * reference_radius and second_score - best_score < 0.28:
            return None, float("inf"), None
    return best, float(min(best_score, _MAX_TRUSTED_ASSOCIATION_COST)), best_step


def _add_continuous_image_trajectory(
    rows: list[dict[str, Any]],
    reference_radius: float,
    *,
    max_bracketed_gap: int = 24,
) -> dict[str, int]:
    """Add a dense visualization path without relabelling estimates as data.

    Measured rows remain the only observations eligible for metric fitting.
    Short gaps bracketed by two verified observations receive linear image-
    plane interpolation and an explicit uncertainty.  This dense path is for
    overlays, event localization, and future detector re-search only.
    """

    measured = [
        index
        for index, row in enumerate(rows)
        if row.get("found") is True
        and row.get("identity_verified") is True
        and row.get("center_u_px") is not None
        and row.get("center_v_px") is not None
    ]
    for index in measured:
        row = rows[index]
        row.update(
            {
                "continuous_center_u_px": float(row["center_u_px"]),
                "continuous_center_v_px": float(row["center_v_px"]),
                "continuous_radius_px": float(
                    row.get("measurement_radius_px") or reference_radius
                ),
                "continuous_source": "verified_measurement",
                "continuous_uncertainty_px": 0.0,
                "continuous_physics_fit_used": bool(row.get("measurement_valid", False)),
            }
        )

    interpolated = 0
    for start, end in zip(measured[:-1], measured[1:]):
        missing_count = end - start - 1
        if missing_count <= 0 or missing_count > max_bracketed_gap:
            continue
        start_row, end_row = rows[start], rows[end]
        start_center = np.asarray(
            [start_row["center_u_px"], start_row["center_v_px"]], dtype=np.float64
        )
        end_center = np.asarray(
            [end_row["center_u_px"], end_row["center_v_px"]], dtype=np.float64
        )
        start_radius = float(start_row.get("measurement_radius_px") or reference_radius)
        end_radius = float(end_row.get("measurement_radius_px") or reference_radius)
        for index in range(start + 1, end):
            alpha = (index - start) / float(end - start)
            center = (1.0 - alpha) * start_center + alpha * end_center
            radius = (1.0 - alpha) * start_radius + alpha * end_radius
            rows[index].update(
                {
                    "continuous_center_u_px": float(center[0]),
                    "continuous_center_v_px": float(center[1]),
                    "continuous_radius_px": float(radius),
                    "continuous_source": "bracketed_linear_interpolation",
                    "continuous_uncertainty_px": float(
                        reference_radius * (0.20 + 0.04 * missing_count)
                    ),
                    "continuous_physics_fit_used": False,
                }
            )
            interpolated += 1

    for row in rows:
        if row.get("continuous_source") is not None:
            continue
        predicted_u = row.get("predicted_center_u_px")
        predicted_v = row.get("predicted_center_v_px")
        if predicted_u is None or predicted_v is None:
            row.update(
                {
                    "continuous_center_u_px": None,
                    "continuous_center_v_px": None,
                    "continuous_radius_px": None,
                    "continuous_source": "unavailable",
                    "continuous_uncertainty_px": None,
                    "continuous_physics_fit_used": False,
                }
            )
            continue
        row.update(
            {
                "continuous_center_u_px": float(predicted_u),
                "continuous_center_v_px": float(predicted_v),
                "continuous_radius_px": float(row.get("display_radius_px") or reference_radius),
                "continuous_source": "unverified_motion_prediction",
                "continuous_uncertainty_px": float(3.0 * reference_radius),
                "continuous_physics_fit_used": False,
            }
        )
    return {
        "verified_measurement_count": len(measured),
        "bracketed_interpolation_count": interpolated,
        "unverified_prediction_count": sum(
            row.get("continuous_source") == "unverified_motion_prediction" for row in rows
        ),
    }


def track_standard_ball(
    frames: Sequence[np.ndarray],
    expected_center_uv: Sequence[float],
    expected_radius_px: float,
    *,
    experiment_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not frames:
        raise ValueError("frames cannot be empty")
    expected = np.asarray(expected_center_uv, dtype=np.float64)
    colour = estimate_ball_colour(frames[0], tuple(expected), float(expected_radius_px))
    rows: list[dict[str, Any]] = []
    centers: list[np.ndarray] = []
    center_frames: list[int] = []
    trusted_radii: list[float] = []
    persistent_distractors: list[dict[str, Any]] = []
    display_radius = float(expected_radius_px)
    for frame_index, frame in enumerate(frames):
        gap = frame_index - center_frames[-1] if center_frames else frame_index + 1
        predicted = _robust_recent_prediction(
            centers,
            center_frames,
            frame_index,
            float(expected_radius_px),
            expected,
        )
        # Collision/bounce fallback: position remains continuous even when
        # velocity reverses abruptly.  The last trusted centre is therefore a
        # second, tightly gated hypothesis, not a general full-frame re-search.
        collision_prediction = None
        if centers and np.linalg.norm(predicted - centers[-1]) > 1.50 * expected_radius_px:
            collision_prediction = np.asarray(centers[-1], dtype=np.float64)
        previous_radius = _robust_recent_radius(trusted_radii, float(expected_radius_px))
        candidates = _enumerate_candidates(frame, colour, expected_radius_px)
        reference_change = None
        if frame_index > 0:
            reference_change = _reference_change_mask(frames[0], frame)
            candidates = _with_reference_change(
                candidates,
                reference_change,
            )
        candidate, cost = _choose_candidate(
            candidates,
            predicted,
            previous_radius,
            expected_radius_px,
            gap=gap,
        )
        selected_prediction = predicted
        using_collision_prediction = False
        if collision_prediction is not None:
            collision_candidate, collision_cost = _choose_candidate(
                candidates,
                collision_prediction,
                previous_radius,
                expected_radius_px,
                gap=1,
            )
            collision_cost += 0.12
            if collision_candidate is not None and collision_cost < cost:
                candidate, cost = collision_candidate, collision_cost
                selected_prediction = collision_prediction
                using_collision_prediction = True
        if _candidate_needs_hough(
            candidate,
            cost,
            selected_prediction,
            previous_radius,
            float(expected_radius_px),
        ):
            hough = _hough_candidates(frame, colour, expected_radius_px, predicted)
            if collision_prediction is not None:
                hough.extend(
                    _hough_candidates(
                        frame,
                        colour,
                        expected_radius_px,
                        collision_prediction,
                    )
                )
            if reference_change is not None:
                hough = _with_reference_change(
                    hough,
                    reference_change,
                )
            candidates.extend(hough)
            candidate, cost = _choose_candidate(
                candidates,
                predicted,
                previous_radius,
                expected_radius_px,
                gap=gap,
            )
            selected_prediction = predicted
            using_collision_prediction = False
            if collision_prediction is not None:
                collision_candidate, collision_cost = _choose_candidate(
                    candidates,
                    collision_prediction,
                    previous_radius,
                    expected_radius_px,
                    gap=1,
                )
                collision_cost += 0.12
                if collision_candidate is not None and collision_cost < cost:
                    candidate, cost = collision_candidate, collision_cost
                    selected_prediction = collision_prediction
                    using_collision_prediction = True
        calibration_first_frame_gt_forced = False
        if frame_index == 0:
            # Frame zero is the benchmark conditioning image: its sphere
            # centre and physical/projected radius are supplied by frozen
            # calibration rather than inferred from the generated motion.
            # Use that known observation only at t=0 when segmentation is
            # confused by the indoor background.  No later frame receives a
            # synthetic measurement.
            height, width = frame.shape[:2]
            boundary = bool(
                expected[0] - expected_radius_px <= 0
                or expected[1] - expected_radius_px <= 0
                or expected[0] + expected_radius_px >= width
                or expected[1] + expected_radius_px >= height
            )
            candidate = SphereCandidate(
                center_u_px=float(expected[0]),
                center_v_px=float(expected[1]),
                radius_px=float(expected_radius_px),
                area_px=float(math.pi * expected_radius_px**2),
                circularity=None,
                ellipse_major_axis_px=None,
                ellipse_minor_axis_px=None,
                ellipse_angle_deg=None,
                ellipse_axis_ratio=None,
                radial_residual_ratio=None,
                hue_distance=0.0,
                boundary=boundary,
                measurement_source="calibrated_first_frame_gt_anchor",
                shape_evidence_available=False,
                colour_support_fraction=None,
                shape_evidence_source=None,
            )
            candidates.append(candidate)
            cost = 0.0
            selected_prediction = expected.copy()
            using_collision_prediction = False
            calibration_first_frame_gt_forced = True
        preliminary_terms = (
            None
            if candidate is None
            else _association_terms(
                candidate,
                selected_prediction,
                previous_radius,
                float(expected_radius_px),
                gap=1 if using_collision_prediction else gap,
            )
        )
        if (
            gap >= 2
            and experiment_id in LINE_MANIFOLDS
            and (
                candidate is None
                or not math.isfinite(cost)
                or cost > _MAX_TRUSTED_ASSOCIATION_COST
                or (
                    preliminary_terms is not None
                    and float(preliminary_terms["prediction_distance_px"])
                    > _MAX_IMMEDIATE_INNOVATION_RADII * expected_radius_px
                )
            )
        ):
            manifold_hough = _manifold_hough_candidates(
                frame,
                colour,
                float(expected_radius_px),
                centers,
                expected,
                experiment_id,
            )
            if reference_change is not None:
                manifold_hough = _with_reference_change(
                    manifold_hough,
                    reference_change,
                )
            candidates.extend(manifold_hough)
        normal_terms = (
            None
            if candidate is None
            else _association_terms(
                candidate,
                selected_prediction,
                previous_radius,
                float(expected_radius_px),
                gap=1 if using_collision_prediction else gap,
            )
        )
        normal_persistent_match = _persistent_distractor_match(
            candidate,
            persistent_distractors,
            frame_index,
            float(expected_radius_px),
        )
        needs_gt_reacquisition = bool(
            candidate is None
            or not math.isfinite(cost)
            or cost > _MAX_TRUSTED_ASSOCIATION_COST
            or normal_persistent_match
            or (
                normal_terms is not None
                and float(normal_terms["prediction_distance_px"])
                > _MAX_IMMEDIATE_INNOVATION_RADII * expected_radius_px
            )
        )
        gt_assisted_reacquisition = False
        gt_reacquisition_step_px = None
        if needs_gt_reacquisition:
            reacquired, reacquisition_cost, reacquisition_step = (
                _gt_manifold_reacquisition_candidate(
                    candidates,
                    centers,
                    center_frames,
                    frame_index,
                    expected,
                    float(expected_radius_px),
                    experiment_id,
                )
            )
            if reacquired is not None:
                candidate = reacquired
                cost = reacquisition_cost
                selected_prediction = predicted
                using_collision_prediction = False
                gt_assisted_reacquisition = True
                gt_reacquisition_step_px = reacquisition_step
        plausible_clusters = _plausible_candidate_clusters(
            candidates,
            selected_prediction,
            previous_radius,
            float(expected_radius_px),
            gap=1 if using_collision_prediction else gap,
        )
        plausible_count = len(plausible_clusters)
        if gt_assisted_reacquisition or calibration_first_frame_gt_forced:
            # The strict first-frame/manifold selector already required a
            # unique candidate, although it can lie outside the local motion
            # gate used to form ordinary plausible clusters.
            plausible_count = 1
        association_margin = (
            None
            if calibration_first_frame_gt_forced
            else
            float(plausible_clusters[1][0] - plausible_clusters[0][0])
            if len(plausible_clusters) >= 2
            else None
        )
        persistent_distractor = _persistent_distractor_match(
            candidate,
            persistent_distractors,
            frame_index,
            float(expected_radius_px),
        )
        if gt_assisted_reacquisition:
            persistent_distractor = False
        locally_ambiguous = bool(
            plausible_count >= 2
            and association_margin is not None
            and association_margin < 0.30
        )
        selected_terms = (
            None
            if candidate is None
            else _association_terms(
                candidate,
                selected_prediction,
                previous_radius,
                float(expected_radius_px),
                gap=1 if using_collision_prediction else gap,
            )
        )
        innovation_motion_supported = False
        if candidate is not None and len(centers) >= 2:
            recent_dt = max(1, int(center_frames[-1]) - int(center_frames[-2]))
            recent_speed = float(np.linalg.norm(centers[-1] - centers[-2])) / recent_dt
            candidate_step = float(
                np.linalg.norm(
                    np.asarray([candidate.center_u_px, candidate.center_v_px], dtype=np.float64)
                    - centers[-1]
                )
            )
            motion_gap = max(1, frame_index - int(center_frames[-1]))
            innovation_motion_supported = bool(
                candidate_step
                <= 2.50 * recent_speed * motion_gap + 0.50 * expected_radius_px
            )
        high_innovation = bool(
            selected_terms is not None
            and float(selected_terms["prediction_distance_px"])
            > (
                _MAX_COLLISION_CONTINUITY_INNOVATION_RADII
                if using_collision_prediction
                else _MAX_IMMEDIATE_INNOVATION_RADII
            )
            * expected_radius_px
            and not innovation_motion_supported
            and not gt_assisted_reacquisition
        )
        calibration_boundary_supported = _calibration_supported_boundary_candidate(
            candidate,
            selected_prediction,
            float(expected_radius_px),
            frame.shape,
            plausible_candidate_count=plausible_count,
            association_cost=cost,
        )
        rejection_reason = None
        if candidate is None:
            rejection_reason = "no_candidate_within_gate"
        elif persistent_distractor:
            rejection_reason = "persistent_background_candidate"
        elif not math.isfinite(cost) or cost > _MAX_TRUSTED_ASSOCIATION_COST:
            rejection_reason = "association_cost_exceeds_gate"
        elif candidate.boundary and not calibration_boundary_supported:
            rejection_reason = "candidate_touches_frame_boundary"
        elif locally_ambiguous:
            rejection_reason = "ambiguous_local_candidates"
        elif high_innovation:
            # Keep extrapolating the last trusted state.  A real reappearance
            # will enter the tight innovation gate on a following frame; a
            # one-frame background impostor cannot alter velocity or radius.
            rejection_reason = "innovation_requires_future_confirmation"
        trusted = bool(
            candidate is not None
            and math.isfinite(cost)
            and cost <= _MAX_TRUSTED_ASSOCIATION_COST
            and (not candidate.boundary or calibration_boundary_supported)
            and not persistent_distractor
            and not locally_ambiguous
            and not high_innovation
        )
        if not trusted:
            diagnostic_terms = None
            if candidate is not None:
                diagnostic_terms = _association_terms(
                    candidate,
                    selected_prediction,
                    previous_radius,
                    float(expected_radius_px),
                    gap=1 if using_collision_prediction else gap,
                )
            rows.append(
                {
                    "frame_index": frame_index,
                    "found": False,
                    "observation_status": "missing",
                    "identity_verified": False,
                    "measurement_valid": False,
                    "center_u_px": None,
                    "center_v_px": None,
                    "measurement_radius_px": None,
                    "track_confidence": 0.0,
                    "candidate_count": len(candidates),
                    "plausible_candidate_count": plausible_count,
                    "association_margin": association_margin,
                    "identity_ambiguous": locally_ambiguous,
                    "identity_rejection_reason": rejection_reason,
                    "innovation_motion_supported": innovation_motion_supported,
                    "gt_assisted_reacquisition": False,
                    "gt_reacquisition_step_px": None,
                    "calibration_first_frame_gt_forced": False,
                    "association_cost": (
                        None if diagnostic_terms is None else float(diagnostic_terms["cost"])
                    ),
                    "prediction_distance_px": (
                        None
                        if diagnostic_terms is None
                        else float(diagnostic_terms["prediction_distance_px"])
                    ),
                    "predicted_center_u_px": float(selected_prediction[0]),
                    "predicted_center_v_px": float(selected_prediction[1]),
                    "prediction_mode": (
                        "gt_first_frame_manifold_reacquisition"
                        if gt_assisted_reacquisition
                        else (
                            "collision_continuity"
                            if using_collision_prediction
                            else "recent_motion"
                        )
                    ),
                    "display_radius_px": display_radius,
                    "shape_evidence_available": False,
                    "colour_support_fraction": None,
                    "shape_evidence_source": None,
                    "edge_support_fraction": None,
                    "edge_radial_residual_ratio": None,
                    "reference_change_fraction": (
                        None
                        if candidate is None
                        else candidate.reference_change_fraction
                    ),
                    "measurement_source": None,
                    "calibration_boundary_supported": False,
                    "shape_evidence_out_of_frame": False,
                    "metric_measurement_eligible": False,
                    "ellipse_major_axis_px": None,
                    "ellipse_minor_axis_px": None,
                    "ellipse_angle_deg": None,
                    "ellipse_axis_ratio": None,
                    "radial_residual_ratio": None,
                }
            )
            _update_persistent_distractor_memory(
                persistent_distractors,
                candidates,
                None,
                frame_index,
                float(expected_radius_px),
            )
            continue
        assert candidate is not None
        candidate_center = np.asarray(
            [candidate.center_u_px, candidate.center_v_px], dtype=np.float64
        )
        metric_first_frame_anchor = bool(
            frame_index == 0
            and plausible_count == 1
            and candidate.hue_distance <= 12.0
            and np.linalg.norm(candidate_center - expected) <= 1.25 * expected_radius_px
            and (not candidate.boundary or calibration_boundary_supported)
        )
        metric_boundary_anchor = bool(
            metric_first_frame_anchor and calibration_boundary_supported
        )
        center = (
            expected.copy()
            if metric_first_frame_anchor
            else candidate_center
        )
        centers.append(center)
        center_frames.append(frame_index)
        trusted_radius = (
            float(expected_radius_px)
            if metric_first_frame_anchor
            else float(
                np.clip(
                    candidate.radius_px,
                    0.80 * expected_radius_px,
                    1.20 * expected_radius_px,
                )
            )
            if gt_assisted_reacquisition
            else float(candidate.radius_px)
        )
        trusted_radii.append(trusted_radius)
        terms = _association_terms(
            candidate,
            selected_prediction,
            previous_radius,
            float(expected_radius_px),
            gap=1 if using_collision_prediction else gap,
        )
        shape_evidence = bool(candidate.shape_evidence_available and not candidate.boundary)
        rows.append(
            {
                "frame_index": frame_index,
                "found": True,
                "observation_status": "measured",
                "identity_verified": True,
                "measurement_valid": True,
                "center_u_px": float(center[0]),
                "center_v_px": float(center[1]),
                "measurement_radius_px": trusted_radius,
                "display_radius_px": display_radius,
                "track_confidence": float(math.exp(-max(cost, 0.0))),
                "candidate_count": len(candidates),
                "plausible_candidate_count": plausible_count,
                "association_margin": association_margin,
                "identity_ambiguous": locally_ambiguous,
                "identity_rejection_reason": None,
                "innovation_motion_supported": innovation_motion_supported,
                "gt_assisted_reacquisition": gt_assisted_reacquisition,
                "gt_reacquisition_step_px": gt_reacquisition_step_px,
                "calibration_first_frame_gt_forced": calibration_first_frame_gt_forced,
                "association_cost": float(cost),
                "prediction_distance_px": float(terms["prediction_distance_px"]),
                "predicted_center_u_px": float(selected_prediction[0]),
                "predicted_center_v_px": float(selected_prediction[1]),
                "prediction_mode": (
                    "gt_first_frame_manifold_reacquisition"
                    if gt_assisted_reacquisition
                    else (
                        "collision_continuity"
                        if using_collision_prediction
                        else "recent_motion"
                    )
                ),
                "shape_evidence_available": shape_evidence,
                "shape_evidence_out_of_frame": bool(candidate.boundary),
                "calibration_boundary_supported": calibration_boundary_supported,
                "metric_measurement_eligible": bool(
                    not candidate.boundary or metric_first_frame_anchor
                ),
                "raw_candidate_center_u_px": candidate.center_u_px,
                "raw_candidate_center_v_px": candidate.center_v_px,
                "raw_candidate_radius_px": candidate.radius_px,
                "measurement_radius_gt_regularized": bool(
                    gt_assisted_reacquisition
                    and not math.isclose(
                        trusted_radius,
                        float(candidate.radius_px),
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                ),
                "colour_support_fraction": candidate.colour_support_fraction,
                "shape_evidence_source": candidate.shape_evidence_source,
                "edge_support_fraction": candidate.edge_support_fraction,
                "edge_radial_residual_ratio": candidate.edge_radial_residual_ratio,
                "reference_change_fraction": candidate.reference_change_fraction,
                "contour_area_px": (
                    candidate.area_px
                    if shape_evidence and candidate.measurement_source == "segmentation_contour"
                    else None
                ),
                "circularity": candidate.circularity if shape_evidence else None,
                "ellipse_major_axis_px": candidate.ellipse_major_axis_px if shape_evidence else None,
                "ellipse_minor_axis_px": candidate.ellipse_minor_axis_px if shape_evidence else None,
                "ellipse_angle_deg": candidate.ellipse_angle_deg if shape_evidence else None,
                "ellipse_axis_ratio": candidate.ellipse_axis_ratio if shape_evidence else None,
                "radial_residual_ratio": candidate.radial_residual_ratio if shape_evidence else None,
                "hue_distance": candidate.hue_distance,
                "touches_frame_boundary": candidate.boundary,
                "measurement_source": (
                    "calibrated_boundary_anchor"
                    if metric_boundary_anchor
                    else (
                        "calibrated_first_frame_gt_anchor"
                        if calibration_first_frame_gt_forced
                        else "calibrated_first_frame_anchor"
                        if metric_first_frame_anchor
                        else candidate.measurement_source
                    )
                ),
            }
        )
        if not calibration_first_frame_gt_forced:
            # At t=0 the accepted identity comes from GT.  Detector candidates
            # displaced by compression/crop error must not be memorized as
            # background and then reject the real ball a few frames later.
            _update_persistent_distractor_memory(
                persistent_distractors,
                candidates,
                candidate,
                frame_index,
                float(expected_radius_px),
            )
    continuous_summary = _add_continuous_image_trajectory(
        rows,
        float(expected_radius_px),
    )
    found = [row for row in rows if row["found"]]
    hough_support = [
        float(row["colour_support_fraction"])
        for row in found
        if row.get("measurement_source") == "hough_circle_fallback"
        and row.get("colour_support_fraction") is not None
    ]
    shape_rows = [row for row in found if row.get("shape_evidence_available") is True]
    hough_shape_rows = [
        row
        for row in shape_rows
        if row.get("shape_evidence_source") == "hough_edge_ring"
    ]
    current_shape_gap = 0
    maximum_shape_gap = 0
    for row in rows:
        if row.get("found") and row.get("shape_evidence_available") is True:
            current_shape_gap = 0
        else:
            current_shape_gap += 1
            maximum_shape_gap = max(maximum_shape_gap, current_shape_gap)
    return rows, {
        "method": "first_frame_gt_initialized_orange_sphere_local_association",
        "colour_model": colour,
        "frame_count": len(rows),
        "found_count": len(found),
        "tracked_fraction": len(found) / max(len(rows), 1),
        "identity_verified_count": len(found),
        "identity_verified_fraction": len(found) / max(len(rows), 1),
        "trusted_association_max_cost": _MAX_TRUSTED_ASSOCIATION_COST,
        "trusted_association_min_confidence": _MIN_TRUSTED_CONFIDENCE,
        "immediate_measurement_max_innovation_radii": _MAX_IMMEDIATE_INNOVATION_RADII,
        "collision_continuity_max_innovation_radii": (
            _MAX_COLLISION_CONTINUITY_INNOVATION_RADII
        ),
        "display_radius_temporally_limited": True,
        "display_radius_locked_to_calibrated_first_frame": True,
        "hough_colour_support_minimum": 0.30,
        "hough_colour_support_observed_min": min(hough_support) if hough_support else None,
        "hough_colour_support_observed_median": (
            float(np.median(hough_support)) if hough_support else None
        ),
        "shape_evidence_count": len(shape_rows),
        "shape_evidence_fraction": len(shape_rows) / max(len(found), 1),
        "hough_edge_ring_shape_evidence_count": len(hough_shape_rows),
        "maximum_consecutive_shape_evidence_gap": maximum_shape_gap,
        "calibration_supported_boundary_count": sum(
            row.get("calibration_boundary_supported") is True for row in rows
        ),
        "metric_boundary_anchor_count": sum(
            row.get("measurement_source") == "calibrated_boundary_anchor" for row in rows
        ),
        "metric_first_frame_anchor_count": sum(
            row.get("measurement_source")
            in {
                "calibrated_first_frame_anchor",
                "calibrated_first_frame_gt_anchor",
                "calibrated_boundary_anchor",
            }
            for row in rows
        ),
        "calibration_first_frame_gt_forced_count": sum(
            row.get("calibration_first_frame_gt_forced") is True for row in rows
        ),
        "gt_assisted_reacquisition_count": sum(
            row.get("gt_assisted_reacquisition") is True for row in rows
        ),
        "continuous_image_trajectory": continuous_summary,
        "continuous_interpolation_is_measurement": False,
        "hough_shape_evidence_thresholds": {
            "min_colour_support_fraction": _HOUGH_SHAPE_MIN_COLOUR_SUPPORT,
            "min_edge_support_fraction": _HOUGH_SHAPE_MIN_EDGE_SUPPORT,
            "max_edge_radial_residual_ratio": _HOUGH_SHAPE_MAX_RADIAL_RESIDUAL,
        },
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


def _tracker_measurement_flags(source: Mapping[str, Any]) -> dict[str, bool]:
    found = source.get("found") is True
    observation_status = source.get("observation_status")
    observation_measured = (
        found if observation_status is None else str(observation_status).lower() == "measured"
    )
    identity_value = source.get("identity_verified")
    identity_verified = found if identity_value is None else identity_value is True
    input_valid_value = source.get("measurement_valid")
    input_measurement_valid = found if input_valid_value is None else input_valid_value is True
    try:
        confidence = float(source.get("track_confidence"))
    except (TypeError, ValueError):
        confidence = float("nan")
    confidence_pass = math.isfinite(confidence) and confidence >= _MIN_TRUSTED_CONFIDENCE
    metric_measurement_eligible = source.get("metric_measurement_eligible") is not False
    return {
        "found": found,
        "observation_measured": observation_measured,
        "identity_verified": identity_verified,
        "input_measurement_valid": input_measurement_valid,
        "confidence_pass": confidence_pass,
        "metric_measurement_eligible": metric_measurement_eligible,
        "eligible": bool(
            found
            and observation_measured
            and identity_verified
            and input_measurement_valid
            and confidence_pass
            and metric_measurement_eligible
        ),
    }


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
    first_found = (
        track_rows[0]
        if track_rows and _tracker_measurement_flags(track_rows[0])["eligible"]
        else None
    )
    if first_found is None:
        raise ValueError("trusted measured frame-0 sphere observation is required for GT size calibration")
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
        tracker_flags = _tracker_measurement_flags(source)
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
                "tracker_observation_measured": tracker_flags["observation_measured"],
                "tracker_identity_verified": tracker_flags["identity_verified"],
                "tracker_input_measurement_valid": tracker_flags["input_measurement_valid"],
                "tracker_confidence_pass": tracker_flags["confidence_pass"],
                "tracker_measurement_eligible": tracker_flags["eligible"],
                "radius_px_gt_normalized": None,
                "expected_radius_px": None,
                "sphere_radius_ratio": None,
                "sphere_size_constraint_pass": False,
                "center_reprojection_residual_px": None,
                "radius_residual_px": None,
                "continuous_x_m": None,
                "continuous_y_m": None,
                "continuous_z_m": None,
                "continuous_q_value": None,
                "continuous_metric_available": False,
                # Dense values are estimates for visualization/event timing;
                # they are never additional observations for parameter fits.
                "continuous_metric_physics_fit_used": False,
            }
        )
        continuous_source = str(source.get("continuous_source", ""))
        if continuous_source in {
            "verified_measurement",
            "bracketed_linear_interpolation",
        }:
            continuous_u = source.get("continuous_center_u_px")
            continuous_v = source.get("continuous_center_v_px")
            continuous_radius = source.get("continuous_radius_px")
            if (
                continuous_u is not None
                and continuous_v is not None
                and continuous_radius is not None
            ):
                continuous_uv = np.asarray(
                    [float(continuous_u), float(continuous_v)], dtype=np.float64
                )
                continuous_plane = _ray_plane_intersection(
                    continuous_uv,
                    K,
                    R,
                    t_camera,
                    plane_y,
                )
                if continuous_plane is not None:
                    continuous_radius_normalized = (
                        float(continuous_radius) * radius_measurement_scale
                    )
                    continuous_refined, _ = _sphere_refine_xz(
                        continuous_plane,
                        continuous_uv,
                        continuous_radius_normalized,
                        sphere_depth_scale,
                        K,
                        R,
                        t_camera,
                        plane_y,
                    )
                    continuous_world, continuous_q = _project_to_manifold(
                        experiment_id,
                        continuous_refined,
                        initial_world,
                    )
                    base.update(
                        {
                            "continuous_x_m": float(continuous_world[0]),
                            "continuous_y_m": float(continuous_world[1]),
                            "continuous_z_m": float(continuous_world[2]),
                            "continuous_q_value": continuous_q,
                            "continuous_metric_available": True,
                        }
                    )
        if not tracker_flags["eligible"]:
            output.append(base)
            continue
        uv = np.asarray([float(source["center_u_px"]), float(source["center_v_px"])])
        raw_radius = float(source["measurement_radius_px"]) * radius_measurement_scale
        plane_point = _ray_plane_intersection(uv, K, R, t_camera, plane_y)
        if plane_point is None:
            output.append(base)
            continue
        _, plane_depth = _project_one(plane_point, K, R, t_camera)
        center_geometry_radius = sphere_depth_scale / max(plane_depth, 1e-9)
        radius = float(
            np.clip(
                raw_radius,
                0.75 * center_geometry_radius,
                1.30 * center_geometry_radius,
            )
        )
        radius_gt_constraint_applied = not math.isclose(
            radius,
            raw_radius,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
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
            tracker_flags["eligible"]
            and size_constraint_pass
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
                "raw_radius_px_gt_normalized": raw_radius,
                "center_geometry_expected_radius_px": center_geometry_radius,
                "radius_preconstraint_ratio": (
                    raw_radius / max(center_geometry_radius, 1e-9)
                ),
                "radius_gt_constraint_applied": radius_gt_constraint_applied,
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
    tracker_eligible = [row for row in output if row.get("tracker_measurement_eligible")]
    continuous_metric = [row for row in output if row.get("continuous_metric_available")]
    continuous_interpolated = [
        row
        for row in continuous_metric
        if row.get("continuous_source") == "bracketed_linear_interpolation"
    ]
    radius_constrained = [row for row in output if row.get("radius_gt_constraint_applied")]
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
        "tracker_eligible_frame_count": len(tracker_eligible),
        "tracker_eligible_fraction": len(tracker_eligible) / max(len(output), 1),
        "metric_frame_count": len(valid),
        "metric_fraction": len(valid) / max(len(output), 1),
        "continuous_metric_frame_count": len(continuous_metric),
        "continuous_metric_fraction": len(continuous_metric) / max(len(output), 1),
        "continuous_interpolated_metric_frame_count": len(continuous_interpolated),
        "continuous_metric_is_fit_evidence": False,
        "sphere_radius_ratio_range": list(sphere_radius_ratio_range),
        "sphere_size_checked_frame_count": len(size_checked),
        "sphere_size_valid_frame_count": len(size_valid),
        "sphere_size_valid_fraction": len(size_valid) / max(len(size_checked), 1),
        "radius_gt_constraint_applied_count": len(radius_constrained),
        "radius_gt_constraint_applied_fraction": (
            len(radius_constrained) / max(len(tracker_eligible), 1)
        ),
        "radius_gt_constraint_ratio_range": [0.75, 1.30],
        "camera_pose_fixed_for_all_frames": True,
    }


def run_ball_tracking(
    video_path: Path,
    calibration: Mapping[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Decode a video and track the standard sphere without lifting to 3D."""

    frames, video = read_video_frames(video_path)
    reference = _reference_circle(calibration, (int(video["width"]), int(video["height"])))
    track, tracking_summary = track_standard_ball(
        frames,
        reference["center_uv_px"],
        reference["radius_px"],
        experiment_id=parse_video_job(video_path)["experiment_id"],
    )
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
