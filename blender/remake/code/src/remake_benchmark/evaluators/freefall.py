from __future__ import annotations

import csv
import html
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if len(values) < window or window == 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def fit_vertical_quadratic(t_s: np.ndarray, y_px: np.ndarray) -> dict[str, float]:
    """Fit y=c2*t^2+c1*t+c0; image-plane acceleration is 2*c2."""
    if len(t_s) < 3 or len(t_s) != len(y_px):
        raise ValueError("quadratic fit requires at least three paired points")
    coefficients = np.polyfit(t_s, y_px, 2)
    predicted = np.polyval(coefficients, t_s)
    residuals = y_px - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_px - np.mean(y_px)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "quadratic_c2_px_s2": float(coefficients[0]),
        "linear_c1_px_s": float(coefficients[1]),
        "intercept_c0_px": float(coefficients[2]),
        "vertical_acceleration_px_s2": float(2.0 * coefficients[0]),
        "fit_rmse_px": float(np.sqrt(np.mean(residuals**2))),
        "fit_r2": r2,
    }


def _object_mask(roi: np.ndarray, object_id: str) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if object_id == "standard_ball":
        mask = (hue >= 4) & (hue <= 28) & (saturation >= 60) & (value >= 70)
    elif object_id == "standard_cube":
        mask = (hue >= 85) & (hue <= 135) & (saturation >= 40) & (value >= 50)
    elif object_id == "cardboard_box":
        mask = (hue >= 8) & (hue <= 35) & (saturation >= 25) & (saturation <= 190) & (value >= 55)
    elif object_id == "volleyball":
        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 40, 120)
        mask = ((saturation >= 25) & (value >= 95)) | (edges > 0)
    else:
        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 45, 120)
        mask = (saturation > 35) | (value > np.percentile(value, 82)) | (edges > 0)
    output = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)


def _best_component(mask: np.ndarray, expected: tuple[float, float], min_area: float, max_area: float) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 2 or box_height < 2:
            continue
        aspect = box_width / float(box_height)
        if not 0.25 <= aspect <= 4.0:
            continue
        center_x, center_y = x + box_width / 2.0, y + box_height / 2.0
        distance = math.hypot((center_x - expected[0]) / max(width, 1), (center_y - expected[1]) / max(height, 1))
        fill = area / float(box_width * box_height)
        score = 2.2 * (1.0 - distance) + min(1.0, fill * 2.0) + 0.4 * min(1.0, area / max(min_area * 4, 1))
        bbox = (x, y, x + box_width, y + box_height)
        if best is None or score > best[0]:
            best = (score, bbox)
    return None if best is None else best[1]


def _conditioning_bbox(image_path: Path, object_id: str) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read conditioning image: {image_path}")
    height, width = image.shape[:2]
    x0, x1 = int(0.30 * width), int(0.70 * width)
    # CAM_Top can place the object against the top image boundary, while the
    # side/main views place it around 20% image height.
    y0, y1 = 0, int(0.55 * height)
    roi = image[y0:y1, x0:x1]
    mask = _object_mask(roi, object_id)
    bbox = _best_component(
        mask,
        expected=(0.5 * (x1 - x0), 0.32 * (y1 - y0)),
        min_area=max(60.0, 0.00010 * width * height),
        max_area=max(600.0, 0.030 * width * height),
    )
    if bbox is None:
        center_x, center_y = int(0.5 * width), int(0.22 * height)
        half = int(0.045 * width)
        return (center_x - half, center_y - half, center_x + half, center_y + half), (height, width)
    bx0, by0, bx1, by1 = bbox
    pad = int(0.18 * max(bx1 - bx0, by1 - by0)) + 6
    return (
        max(0, x0 + bx0 - pad),
        max(0, y0 + by0 - pad),
        min(width, x0 + bx1 + pad),
        min(height, y0 + by1 + pad),
    ), (height, width)


def _scale_bbox(bbox: tuple[int, int, int, int], source_shape: tuple[int, int], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    source_height, source_width = source_shape
    frame_height, frame_width = frame_shape
    sx, sy = frame_width / float(source_width), frame_height / float(source_height)
    x0, y0, x1, y1 = bbox
    return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)


def _template_detection(frame: np.ndarray, template_gray: np.ndarray, initial_bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int] | None, float]:
    height, width = frame.shape[:2]
    template_height, template_width = template_gray.shape[:2]
    initial_center_x = 0.5 * (initial_bbox[0] + initial_bbox[2])
    strip_half = max(3 * template_width, int(0.24 * width))
    search_x0 = max(0, int(initial_center_x - strip_half))
    search_x1 = min(width, int(initial_center_x + strip_half))
    search_y0 = max(0, int(initial_bbox[1] - 2 * template_height))
    search_y1 = min(height, int(0.96 * height))
    search = frame[search_y0:search_y1, search_x0:search_x1]
    if search.shape[0] < template_height or search.shape[1] < template_width:
        return None, float("nan")
    result = cv2.matchTemplate(cv2.cvtColor(search, cv2.COLOR_BGR2GRAY), template_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    x0, y0 = search_x0 + location[0], search_y0 + location[1]
    return (x0, y0, x0 + template_width, y0 + template_height), float(score)


def _motion_detection(
    previous: np.ndarray,
    frame: np.ndarray,
    initial_bbox: tuple[int, int, int, int],
    previous_center: tuple[float, float],
    threshold: float,
) -> tuple[tuple[int, int, int, int], float] | None:
    height, width = frame.shape[:2]
    previous_gray = cv2.GaussianBlur(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(cv2.absdiff(gray, previous_gray), threshold, 255, cv2.THRESH_BINARY)
    initial_width = max(4, initial_bbox[2] - initial_bbox[0])
    initial_height = max(4, initial_bbox[3] - initial_bbox[1])
    initial_center_x = 0.5 * (initial_bbox[0] + initial_bbox[2])
    strip_half = max(int(0.26 * width), 4 * initial_width)
    gate = np.zeros_like(mask)
    x0, x1 = max(0, int(initial_center_x - strip_half)), min(width, int(initial_center_x + strip_half))
    y0, y1 = max(0, initial_bbox[1] - 2 * initial_height), min(height, int(0.96 * height))
    gate[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gate = cv2.morphologyEx(cv2.morphologyEx(gate, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(gate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, tuple[int, int, int, int], float] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(10.0, 0.00005 * width * height) or area > 0.08 * width * height:
            continue
        bx, by, box_width, box_height = cv2.boundingRect(contour)
        center_x, center_y = bx + box_width / 2.0, by + box_height / 2.0
        distance = math.hypot(
            (center_x - previous_center[0]) / max(0.20 * width, initial_width),
            (center_y - previous_center[1]) / max(0.22 * height, initial_height),
        )
        down_bonus = 0.25 if center_y >= previous_center[1] - initial_height else 0.0
        score = 1.5 - distance + min(1.0, area / (initial_width * initial_height)) + down_bonus
        confidence = min(1.0, area / max(initial_width * initial_height, 1))
        bbox = (bx, by, bx + box_width, by + box_height)
        if best is None or score > best[0]:
            best = (score, bbox, confidence)
    return None if best is None else (best[1], best[2])


def _refine_bbox(
    frame: np.ndarray,
    coarse_bbox: tuple[int, int, int, int],
    object_id: str,
    initial_area: float,
) -> tuple[int, int, int, int] | None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = coarse_bbox
    pad_x = max(10, int(0.70 * (x1 - x0)))
    pad_y = max(10, int(0.70 * (y1 - y0)))
    rx0, ry0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    rx1, ry1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
    roi = frame[ry0:ry1, rx0:rx1]
    if roi.size == 0:
        return None
    mask = _object_mask(roi, object_id)
    expected = (0.5 * (x0 + x1) - rx0, 0.5 * (y0 + y1) - ry0)
    bbox = _best_component(mask, expected, max(12.0, 0.12 * initial_area), max(80.0, 4.0 * initial_area))
    if bbox is None:
        return None
    return bbox[0] + rx0, bbox[1] + ry0, bbox[2] + rx0, bbox[3] + ry0


def _track_video(job: dict[str, Any], video_path: Path, image_path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], float, tuple[int, int]]:
    object_id = str(job["factors"]["object_id"])
    source_bbox, source_shape = _conditioning_bbox(image_path, object_id)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or job["generation"]["fps"])
    ok, first = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"empty video: {video_path}")
    frame_shape = first.shape[:2]
    scaled_bbox = _scale_bbox(source_bbox, source_shape, frame_shape)
    pad = max(4, int(0.12 * max(scaled_bbox[2] - scaled_bbox[0], scaled_bbox[3] - scaled_bbox[1])))
    x0, y0 = max(0, scaled_bbox[0] - pad), max(0, scaled_bbox[1] - pad)
    x1, y1 = min(frame_shape[1], scaled_bbox[2] + pad), min(frame_shape[0], scaled_bbox[3] + pad)
    template = first[y0:y1, x0:x1]
    if template.size == 0:
        capture.release()
        raise RuntimeError("conditioning bbox produced an empty video template")
    initial_bbox = (x0, y0, x1, y1)
    initial_area = float(max(1, (x1 - x0) * (y1 - y0)))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    previous, frame = first, first
    previous_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    tracks: list[dict[str, Any]] = []
    frame_index = 0
    tracker = str(config.get("tracker", "auto"))
    min_template_score = float(config.get("min_template_score", 0.18))
    while True:
        template_bbox, template_score = _template_detection(frame, template_gray, initial_bbox)
        template_found = template_bbox is not None and math.isfinite(template_score) and template_score >= min_template_score
        motion = None
        if frame_index > 0 and tracker != "template":
            motion = _motion_detection(
                previous,
                frame,
                initial_bbox,
                previous_center,
                float(config.get("motion_threshold", 14)),
            )
        if tracker == "motion":
            chosen = motion
            source = "motion" if motion else "missing"
            score = motion[1] if motion else None
            bbox = motion[0] if motion else None
        elif tracker == "template":
            chosen = (template_bbox, template_score) if template_found else None
            source = "template" if chosen else "missing"
            score = template_score if chosen else None
            bbox = template_bbox if chosen else None
        elif template_found:
            chosen = (template_bbox, template_score) if template_found else None
            source = "template" if chosen else "missing"
            score = template_score if chosen else None
            bbox = template_bbox if chosen else None
        elif motion is not None:
            chosen, source, score, bbox = motion, "motion", motion[1], motion[0]
        else:
            chosen = None
            source = "missing"
            score = None
            bbox = None
        if chosen is not None and bbox is not None:
            refined = _refine_bbox(frame, bbox, object_id, initial_area)
            if refined is not None:
                bbox = refined
                source = f"{source}+mask"
            center_x, center_y = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            previous_center = (center_x, center_y)
        else:
            center_x = center_y = None
        tracks.append(
            {
                "job_id": job["job_id"],
                "frame_index": frame_index,
                "time_s": frame_index / fps,
                "found": bbox is not None,
                "tracking_source": source,
                "score": _finite(score),
                "template_score": _finite(template_score),
                "center_x_px": center_x,
                "center_y_px": center_y,
                "bbox_x0": None if bbox is None else bbox[0],
                "bbox_y0": None if bbox is None else bbox[1],
                "bbox_x1": None if bbox is None else bbox[2],
                "bbox_y1": None if bbox is None else bbox[3],
                "bbox_width_px": None if bbox is None else bbox[2] - bbox[0],
                "bbox_height_px": None if bbox is None else bbox[3] - bbox[1],
                "bbox_area_px2": None if bbox is None else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "bbox_aspect_ratio": None if bbox is None else (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1),
                "fit_used": False,
                "center_y_smoothed_px": None,
            }
        )
        previous = frame
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
    capture.release()
    return tracks, fps, frame_shape


def _max_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _support_surface_profile(image_path: Path, frame_shape: tuple[int, int]) -> np.ndarray | None:
    """Estimate the top of the wide brown support plank in conditioning-image coordinates."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    source_height, source_width = image.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (
        (hue >= 3)
        & (hue <= 38)
        & (saturation >= 24)
        & (value >= 28)
        & (np.indices((source_height, source_width))[0] >= int(0.34 * source_height))
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if width < 0.22 * source_width or area < 0.002 * source_width * source_height:
            continue
        candidates.append((area * width / max(height, 1), contour))
    if not candidates:
        return None
    contour = max(candidates, key=lambda item: item[0])[1]
    component = np.zeros((source_height, source_width), dtype=np.uint8)
    cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
    profile = np.full(source_width, np.nan, dtype=float)
    for x in range(source_width):
        ys = np.flatnonzero(component[:, x])
        if len(ys):
            profile[x] = float(ys[0])
    known = np.flatnonzero(np.isfinite(profile))
    if len(known) < max(8, int(0.08 * source_width)):
        return None
    profile = np.interp(np.arange(source_width), known, profile[known])
    frame_height, frame_width = frame_shape
    source_x = np.linspace(0, source_width - 1, frame_width)
    return np.interp(source_x, np.arange(source_width), profile) * frame_height / float(source_height)


def _physical_gate(
    tracks: list[dict[str, Any]],
    frame_shape: tuple[int, int],
    image_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Run conservative 2-D heuristics before any physical-parameter fit."""
    valid = [row for row in tracks if row["found"]]
    frame_height, frame_width = frame_shape
    detection_rate = len(valid) / max(len(tracks), 1)
    missing_run = _max_true_run([not row["found"] for row in tracks])
    severe: list[str] = []
    if detection_rate < float(config.get("min_detection_rate", 0.70)):
        severe.append("severe_low_detection_rate")
    if missing_run > max(3, int(float(config.get("max_missing_run_fraction", 0.18)) * len(tracks))):
        severe.append("severe_object_disappearance")

    metrics: dict[str, Any] = {
        "detection_rate": detection_rate,
        "max_missing_run_frames": missing_run,
        "support_surface_detected": False,
        "max_penetration_depth_px": None,
        "max_penetration_depth_object_heights": None,
    }
    if len(valid) < 3:
        return metrics, list(dict.fromkeys(severe))

    widths = np.asarray([row["bbox_width_px"] for row in valid], dtype=float)
    heights = np.asarray([row["bbox_height_px"] for row in valid], dtype=float)
    areas = np.asarray([row["bbox_area_px2"] for row in valid], dtype=float)
    aspects = np.asarray([row["bbox_aspect_ratio"] for row in valid], dtype=float)
    centers = np.asarray([[row["center_x_px"], row["center_y_px"]] for row in valid], dtype=float)
    object_height = float(np.median(heights))
    object_diagonal = float(np.median(np.hypot(widths, heights)))
    area_ratio = float(np.percentile(areas, 95) / max(np.percentile(areas, 5), 1.0))
    aspect_ratio_change = float(np.percentile(aspects, 95) / max(np.percentile(aspects, 5), 1e-6))
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    max_jump_normalized = float(np.max(jumps) / max(object_diagonal, 1.0)) if len(jumps) else 0.0
    centered = centers - np.mean(centers, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[-1]
    orthogonal = np.abs(centered @ normal)
    line_residual = float(np.percentile(orthogonal, 95) / max(object_diagonal, 1.0))
    boundary = [
        row["bbox_x0"] <= 1
        or row["bbox_y0"] <= 1
        or row["bbox_x1"] >= frame_width - 2
        or row["bbox_y1"] >= frame_height - 2
        for row in valid
    ]
    boundary_fraction = float(np.mean(boundary))
    metrics.update(
        {
            "median_object_height_px": object_height,
            "median_object_diagonal_px": object_diagonal,
            "bbox_area_p95_p05_ratio": area_ratio,
            "bbox_aspect_p95_p05_ratio": aspect_ratio_change,
            "max_center_jump_object_diagonals": max_jump_normalized,
            "trajectory_line_residual_object_diagonals": line_residual,
            "out_of_frame_fraction": boundary_fraction,
        }
    )
    if area_ratio > float(config.get("max_bbox_area_ratio", 3.0)):
        severe.append("severe_scale_change")
    if aspect_ratio_change > float(config.get("max_bbox_aspect_ratio_change", 2.5)):
        severe.append("severe_object_deformation")
    if max_jump_normalized > float(config.get("max_center_jump_object_diagonals", 2.5)):
        severe.append("severe_teleportation")
    if line_residual > float(config.get("max_trajectory_line_residual_object_diagonals", 1.5)):
        severe.append("severe_erratic_trajectory")
    if boundary_fraction > float(config.get("max_out_of_frame_fraction", 0.10)):
        severe.append("severe_out_of_frame")

    support = _support_surface_profile(image_path, frame_shape)
    if support is not None:
        first_valid = valid[0]
        first_x = min(frame_width - 1, max(0, int(round(first_valid["center_x_px"]))))
        initial_clearance = float(support[first_x] - first_valid["bbox_y1"])
        metrics["support_surface_initial_clearance_px"] = initial_clearance
        if initial_clearance < 0.25 * max(float(first_valid["bbox_height_px"]), 1.0):
            # Every v1_A conditioning frame is airborne. A surface above or
            # touching the initial object is therefore a failed plank split,
            # not evidence of penetration.
            support = None
    if support is not None:
        metrics["support_surface_detected"] = True
        depths_by_frame: list[float | None] = []
        ratios_by_frame: list[float | None] = []
        for row in tracks:
            if not row["found"]:
                depths_by_frame.append(None)
                ratios_by_frame.append(None)
                continue
            x = min(frame_width - 1, max(0, int(round(row["center_x_px"]))))
            depth = float(row["bbox_y1"] - support[x])
            height = max(float(row["bbox_height_px"]), 1.0)
            depths_by_frame.append(depth)
            ratios_by_frame.append(depth / height)
        finite_depths = [value for value in depths_by_frame if value is not None]
        finite_ratios = [value for value in ratios_by_frame if value is not None]
        metrics["max_penetration_depth_px"] = max(finite_depths) if finite_depths else None
        metrics["max_penetration_depth_object_heights"] = max(finite_ratios) if finite_ratios else None
        threshold = float(config.get("max_penetration_object_heights", 0.60))
        penetration_run = _max_true_run([value is not None and value > threshold for value in ratios_by_frame])
        metrics["penetration_run_frames"] = penetration_run
        if penetration_run >= int(config.get("min_penetration_frames", 3)):
            severe.append("severe_support_penetration")

    return metrics, list(dict.fromkeys(severe))


def _fit_tracks(tracks: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    valid_positions = [index for index, row in enumerate(tracks) if row["found"]]
    min_points = int(config.get("min_fit_points", 12))
    if len(valid_positions) < min_points:
        raise RuntimeError(f"too few detected frames: {len(valid_positions)} < {min_points}")
    times = np.asarray([tracks[index]["time_s"] for index in valid_positions], dtype=float)
    ys = np.asarray([tracks[index]["center_y_px"] for index in valid_positions], dtype=float)
    smoothed = _moving_average(ys, int(config.get("smoothing_window", 5)))
    for index, smooth_y in zip(valid_positions, smoothed):
        tracks[index]["center_y_smoothed_px"] = float(smooth_y)
    bbox_heights = [tracks[index]["bbox_y1"] - tracks[index]["bbox_y0"] for index in valid_positions]
    object_height = float(np.median(bbox_heights))
    floor_y = float(np.percentile(smoothed, float(config.get("floor_percentile", 92))))
    airborne_threshold = floor_y - max(6.0, 0.20 * object_height)
    max_count = max(min_points, int(len(times) * float(config.get("fit_max_frame_fraction", 0.72))))
    fit_positions = [index for index in range(min(len(times), max_count)) if smoothed[index] < airborne_threshold]
    fit_selection = "airborne_before_floor"
    if len(fit_positions) < min_points:
        fit_positions = list(range(min(max_count, len(times))))
        fit_selection = "early_window_fallback"
    if len(fit_positions) < min_points:
        raise RuntimeError(f"too few fit points after airborne selection: {len(fit_positions)}")
    fit_times_absolute = times[fit_positions]
    fit_t0 = float(fit_times_absolute[0])
    fit_times = fit_times_absolute - fit_t0
    fit_y = smoothed[fit_positions]
    fit = fit_vertical_quadratic(fit_times, fit_y)
    for position in fit_positions:
        tracks[valid_positions[position]]["fit_used"] = True
    fit.update(
        {
            "fit_t0_s": fit_t0,
            "num_fit_points": len(fit_positions),
            "num_detected_frames": len(valid_positions),
            "num_video_frames": len(tracks),
            "detection_rate": len(valid_positions) / len(tracks),
            "floor_center_y_px": floor_y,
            "airborne_threshold_y_px": airborne_threshold,
            "fit_selection": fit_selection,
        }
    )
    return fit, fit_times_absolute, fit_y


def _write_tracks(path: Path, tracks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tracks[0]))
        writer.writeheader()
        writer.writerows(tracks)


def _trajectory_plot(
    path: Path,
    job: dict[str, Any],
    tracks: list[dict[str, Any]],
    fit: dict[str, Any] | None,
    gate_flags: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detected = [row for row in tracks if row["found"]]
    xs = np.asarray([row["center_x_px"] for row in detected], dtype=float)
    ys = np.asarray([row["center_y_px"] for row in detected], dtype=float)
    times = np.asarray([row["time_s"] for row in detected], dtype=float)
    smooth_values = [row["center_y_smoothed_px"] for row in detected]
    if any(value is None for value in smooth_values):
        smooth = _moving_average(ys, 5)
    else:
        smooth = np.asarray(smooth_values, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    scatter = axes[0].scatter(xs, ys, c=times, cmap="viridis", s=18)
    axes[0].plot(xs, ys, color="#555555", linewidth=0.8, alpha=0.55)
    axes[0].invert_yaxis()
    axes[0].set_title("Detected image-plane route")
    axes[0].set_xlabel("center x (px)")
    axes[0].set_ylabel("center y (px; downward in image)")
    figure.colorbar(scatter, ax=axes[0], label="time (s)")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(times, ys, s=13, color="#888888", label="detected y")
    axes[1].plot(times, smooth, linewidth=1.2, color="#1f77b4", label="smoothed y")
    if fit is not None:
        fit_rows = [row for row in tracks if row["fit_used"]]
        fit_times = np.asarray([row["time_s"] for row in fit_rows], dtype=float)
        relative = fit_times - float(fit["fit_t0_s"])
        predicted = (
            fit["quadratic_c2_px_s2"] * relative**2
            + fit["linear_c1_px_s"] * relative
            + fit["intercept_c0_px"]
        )
        axes[1].plot(fit_times, predicted, linewidth=2.0, color="#d62728", label="quadratic fit")
        axes[1].set_title(
            f"a={fit['vertical_acceleration_px_s2']:.2f} px/s², R²={fit['fit_r2']:.3f}"
        )
    else:
        reason = ", ".join(gate_flags) if gate_flags else "fit unavailable"
        axes[1].set_title(f"Parameter fit skipped: {reason}", fontsize=9)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("center y (px)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle(job["job_id"], fontsize=9)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _overlay_video(
    source: Path,
    output: Path,
    tracks: list[dict[str, Any]],
    fps: float,
    frame_shape: tuple[int, int],
    tail_frames: int,
    gate_flags: list[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.stem}.opencv.mp4")
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(temporary_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_shape[1], frame_shape[0]),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open MP4 overlay writer")
    centers: list[tuple[int, int] | None] = []
    for frame_index, track in enumerate(tracks):
        ok, frame = capture.read()
        if not ok:
            break
        if track["found"]:
            bbox = tuple(int(track[key]) for key in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"))
            center = (int(track["center_x_px"]), int(track["center_y_px"]))
            centers.append(center)
            color = (40, 210, 40) if track["fit_used"] else (0, 190, 255)
            cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
            cv2.circle(frame, center, 4, (255, 255, 0), -1)
            label = f"object {track['tracking_source']} score={track['score'] or 0:.2f}"
            cv2.putText(frame, label, (bbox[0], max(20, bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        else:
            centers.append(None)
            cv2.putText(frame, "OBJECT NOT DETECTED", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)
        start = max(1, len(centers) - tail_frames)
        for index in range(start, len(centers)):
            if centers[index - 1] is not None and centers[index] is not None:
                cv2.line(frame, centers[index - 1], centers[index], (255, 220, 0), 2)
        cv2.putText(frame, f"frame={frame_index} fit={'yes' if track['fit_used'] else 'no'}", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if gate_flags:
            cv2.putText(
                frame,
                "PHYSICS GATE: " + ",".join(gate_flags[:2]),
                (20, frame_shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        writer.write(frame)
    capture.release()
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        process = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(temporary_output),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode == 0 and output.is_file():
            temporary_output.unlink(missing_ok=True)
            return
    temporary_output.replace(output)


def evaluate_freefall_job(
    job: dict[str, Any],
    video_path: Path,
    workspace_root: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    make_overlay: bool,
) -> dict[str, Any]:
    image_path = Path(job["inputs"]["image"])
    if not image_path.is_absolute():
        image_path = workspace_root / image_path
    tracks, fps, frame_shape = _track_video(job, video_path, image_path, config)
    gate_metrics, severe_flags = _physical_gate(tracks, frame_shape, image_path, config)
    fit: dict[str, Any] | None = None
    fit_error = None
    if not severe_flags:
        try:
            fit, _, _ = _fit_tracks(tracks, config)
        except Exception as exc:
            fit_error = str(exc)
    job_id = job["job_id"]
    track_path = output_root / "tracks" / f"{job_id}.csv"
    plot_path = output_root / "plots" / f"{job_id}.png"
    overlay_path = output_root / "overlays" / f"{job_id}.mp4"
    _write_tracks(track_path, tracks)
    _trajectory_plot(plot_path, job, tracks, fit, severe_flags)
    overlay_error = None
    if make_overlay:
        try:
            _overlay_video(
                video_path,
                overlay_path,
                tracks,
                fps,
                frame_shape,
                int(config.get("overlay_tail_frames", 24)),
                severe_flags,
            )
        except Exception as exc:
            overlay_error = str(exc)
    quality_flags = list(severe_flags)
    if fit_error:
        quality_flags.append("parameter_fit_failed")
    if fit is not None and (not math.isfinite(fit["fit_r2"]) or fit["fit_r2"] < float(config.get("min_fit_r2", 0.70))):
        quality_flags.append("low_fit_r2")
    if fit is not None and fit["fit_selection"] == "early_window_fallback":
        quality_flags.append("airborne_interval_fallback")
    if overlay_error:
        quality_flags.append("overlay_write_failed")
    camera = str(job["factors"].get("camera", "unknown"))
    pixels_per_meter = config.get("pixels_per_meter")
    acceleration_m_s2 = None
    if fit is not None and camera == "CAM_Side" and pixels_per_meter is not None and float(pixels_per_meter) > 0:
        acceleration_m_s2 = fit["vertical_acceleration_px_s2"] / float(pixels_per_meter)
    if fit is None:
        fit_status = "skipped_severe_gate" if severe_flags else "failed"
    else:
        fit_status = "fitted_image_plane_proxy"
    if acceleration_m_s2 is not None:
        parameter_identifiability = "metric_side_view_with_scale"
    elif camera == "CAM_Side":
        parameter_identifiability = "image_plane_only_missing_metric_scale"
    else:
        parameter_identifiability = "image_plane_only_projective_view"
    camera_note = {
        "CAM_Side": "Best 2-D gravity-proxy view; metric g still requires a valid pixels-per-meter calibration.",
        "CAM_Main": "Perspective projection is fitted in image space; a scalar pixel scale is insufficient for metric g.",
        "CAM_Top": "Top/oblique projection is fully tracked and gated, but metric g needs camera extrinsics and depth/plane calibration.",
    }.get(camera, "Unknown camera geometry; only image-plane motion is reported.")
    fit_metrics: dict[str, Any] = {} if fit is None else fit
    return {
        "job_id": job_id,
        "evaluator_id": "v1a_freefall",
        "evaluator_version": str(config.get("version", "1.0.0")),
        "status": "ok" if not quality_flags else "invalid",
        "quality_flags": quality_flags,
        "metrics": {
            **fit_metrics,
            **gate_metrics,
            "target_gravity_m_s2": float(job["targets"]["gravity_g"]),
            "estimated_gravity_m_s2": acceleration_m_s2,
            "pixels_per_meter": pixels_per_meter,
            "fps": fps,
            "fit_status": fit_status,
            "parameter_identifiability": parameter_identifiability,
            "camera_interpretation": camera_note,
            "severe_gate_passed": not severe_flags,
        },
        "artifacts": {
            "track_csv": str(track_path),
            "trajectory_plot": str(plot_path),
            "overlay_video": str(overlay_path) if make_overlay and overlay_path.exists() else None,
        },
        "overlay_error": overlay_error,
        "fit_error": fit_error,
        "factors": dict(job["factors"]),
    }


def aggregate_freefall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("metrics") and _finite(row["metrics"].get("vertical_acceleration_px_s2")) is not None]
    tracked = [row for row in rows if _finite(row.get("metrics", {}).get("detection_rate")) is not None]
    targets = np.asarray([row["metrics"]["target_gravity_m_s2"] for row in usable], dtype=float)
    estimates = np.asarray([row["metrics"]["vertical_acceleration_px_s2"] for row in usable], dtype=float)

    def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
        if len(left) < 2 or float(np.std(left)) == 0 or float(np.std(right)) == 0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    rank_target = np.argsort(np.argsort(targets)) if len(targets) else targets
    rank_estimate = np.argsort(np.argsort(estimates)) if len(estimates) else estimates
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_scene[str(row["factors"]["scene_id"])].append(row)
    per_camera: dict[str, Any] = {}
    for camera in sorted({str(row.get("factors", {}).get("camera", "unknown")) for row in rows}):
        camera_rows = [row for row in rows if str(row.get("factors", {}).get("camera", "unknown")) == camera]
        camera_usable = [row for row in camera_rows if _finite(row.get("metrics", {}).get("vertical_acceleration_px_s2")) is not None]
        camera_targets = np.asarray([row["metrics"]["target_gravity_m_s2"] for row in camera_usable], dtype=float)
        camera_estimates = np.asarray([row["metrics"]["vertical_acceleration_px_s2"] for row in camera_usable], dtype=float)
        per_camera[camera] = {
            "n": len(camera_rows),
            "usable_proxy_fits": len(camera_usable),
            "valid": sum(row.get("status") == "ok" for row in camera_rows),
            "severe_gate_failures": sum(not row.get("metrics", {}).get("severe_gate_passed", False) for row in camera_rows),
            "pearson_target_vs_acceleration_proxy": correlation(camera_targets, camera_estimates),
        }
    return {
        "n": len(rows),
        "usable": len(usable),
        "valid": sum(row.get("status") == "ok" for row in rows),
        "severe_gate_failures": sum(not row.get("metrics", {}).get("severe_gate_passed", False) for row in rows),
        "pearson_target_vs_acceleration_proxy": correlation(targets, estimates),
        "spearman_target_vs_acceleration_proxy": correlation(rank_target, rank_estimate),
        "mean_detection_rate": float(np.mean([row["metrics"]["detection_rate"] for row in tracked])) if tracked else None,
        "mean_fit_r2": float(np.mean([row["metrics"]["fit_r2"] for row in usable])) if usable else None,
        "scene_counts": {scene: len(items) for scene, items in by_scene.items()},
        "per_camera": per_camera,
        "unit_warning": "vertical_acceleration_px_s2 is an image-plane proxy; it is not m/s^2 without pixels_per_meter calibration",
        "view_warning": "CAM_Main and CAM_Top are projective views; metric gravity is not identifiable from a scalar pixel scale alone",
        "design_warning": "object_values binds one gravity to each object, so gravity response is confounded with object identity",
    }


def write_freefall_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job_id", "status", "scene_id", "object_id", "camera", "target_gravity_m_s2",
        "vertical_acceleration_px_s2", "estimated_gravity_m_s2", "fit_r2", "fit_rmse_px",
        "detection_rate", "num_fit_points", "fit_selection", "fit_status", "parameter_identifiability",
        "severe_gate_passed", "max_missing_run_frames", "max_center_jump_object_diagonals",
        "bbox_area_p95_p05_ratio", "bbox_aspect_p95_p05_ratio", "out_of_frame_fraction",
        "max_penetration_depth_object_heights", "quality_flags", "trajectory_plot", "overlay_video",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics, factors, artifacts = row.get("metrics", {}), row.get("factors", {}), row.get("artifacts", {})
            writer.writerow(
                {
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "scene_id": factors.get("scene_id"),
                    "object_id": factors.get("object_id"),
                    "camera": factors.get("camera"),
                    "target_gravity_m_s2": metrics.get("target_gravity_m_s2"),
                    "vertical_acceleration_px_s2": metrics.get("vertical_acceleration_px_s2"),
                    "estimated_gravity_m_s2": metrics.get("estimated_gravity_m_s2"),
                    "fit_r2": metrics.get("fit_r2"),
                    "fit_rmse_px": metrics.get("fit_rmse_px"),
                    "detection_rate": metrics.get("detection_rate"),
                    "num_fit_points": metrics.get("num_fit_points"),
                    "fit_selection": metrics.get("fit_selection"),
                    "fit_status": metrics.get("fit_status"),
                    "parameter_identifiability": metrics.get("parameter_identifiability"),
                    "severe_gate_passed": metrics.get("severe_gate_passed"),
                    "max_missing_run_frames": metrics.get("max_missing_run_frames"),
                    "max_center_jump_object_diagonals": metrics.get("max_center_jump_object_diagonals"),
                    "bbox_area_p95_p05_ratio": metrics.get("bbox_area_p95_p05_ratio"),
                    "bbox_aspect_p95_p05_ratio": metrics.get("bbox_aspect_p95_p05_ratio"),
                    "out_of_frame_fraction": metrics.get("out_of_frame_fraction"),
                    "max_penetration_depth_object_heights": metrics.get("max_penetration_depth_object_heights"),
                    "quality_flags": ";".join(row.get("quality_flags", [])),
                    "trajectory_plot": artifacts.get("trajectory_plot"),
                    "overlay_video": artifacts.get("overlay_video"),
                }
            )


def write_freefall_html_report(path: Path, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def relative(target: str | None) -> str | None:
        return None if not target else Path(os.path.relpath(target, path.parent)).as_posix()

    lines = [
        "<!doctype html><html><head><meta charset='utf-8'><title>V1A free-fall evaluation</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.5}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px;font-size:12px}th{background:#f4f4f4}img{max-width:900px;width:100%}video{max-width:900px;width:100%}.warn{color:#a33}code{background:#f4f4f4;padding:2px 4px}</style></head><body>",
        "<h1>V1A three-view physics gate, tracking and fit report</h1>",
        f"<p>Usable proxy fits: {aggregate.get('usable', 0)} / {aggregate.get('n', 0)}; severe-gate failures: {aggregate.get('severe_gate_failures', 0)}; mean detection rate: {aggregate.get('mean_detection_rate')}; mean R²: {aggregate.get('mean_fit_r2')}</p>",
        "<h2>Evaluation order</h2>",
        "<p>Each generated video is processed before the next generation starts: (1) object detection and route extraction; (2) severe 2-D physics/geometry gate for disappearance, teleportation, scale/deformation, frame exit, erratic route and support-plank penetration; (3) only after the gate passes, airborne trajectory fitting; (4) route plot and selected annotated video.</p>",
        "<p class='warn'>The severe gate is an auditable image-space heuristic, not a collision-engine proof. A flagged sample requires visual confirmation in its annotated video.</p>",
        "<h2>Fitting equation and units</h2>",
        "<p>Image coordinates use positive y downward. For selected airborne frames, fit <code>y_px(t)=c0+c1·(t−t0)+c2·(t−t0)²</code>. Therefore <code>v_px(t)=c1+2·c2·(t−t0)</code> and the projected acceleration is <code>a_px=2·c2</code> in px/s². For a calibrated side view only, <code>g_est=a_px/s_px_per_m</code>.</p>",
        "<p class='warn'>Without calibration, px/s² is only a projected motion proxy. CAM_Main and CAM_Top additionally need camera extrinsics/homography or depth calibration; their scalar metric gravity is intentionally reported as unidentifiable. The object-values design also confounds object identity with gravity value.</p>",
        "<h2>Per-video visual verification</h2>",
    ]
    for row in rows:
        metrics, artifacts = row.get("metrics", {}), row.get("artifacts", {})
        lines.append(f"<h3>{html.escape(row['job_id'])}</h3>")
        lines.append(
            f"<p>View={html.escape(str(row.get('factors', {}).get('camera')))}; status={html.escape(row['status'])}; "
            f"gate_passed={metrics.get('severe_gate_passed')}; detection={metrics.get('detection_rate')}; "
            f"fit={metrics.get('fit_status')}; a={metrics.get('vertical_acceleration_px_s2')} px/s²; R²={metrics.get('fit_r2')}; "
            f"flags={html.escape(', '.join(row.get('quality_flags', [])))}</p>"
        )
        lines.append(f"<p>{html.escape(str(metrics.get('camera_interpretation', '')))}</p>")
        plot = relative(artifacts.get("trajectory_plot"))
        overlay = relative(artifacts.get("overlay_video"))
        if plot:
            lines.append(f"<img src='{html.escape(plot)}' loading='lazy'>")
        if overlay:
            lines.append(f"<video controls preload='metadata' src='{html.escape(overlay)}'></video>")
    lines.append("</body></html>")
    path.write_text("\n".join(lines), encoding="utf-8")
