#!/usr/bin/env python3
"""Audit unexpected camera/viewpoint changes in a directory of short videos.

The detector estimates a global background similarity transform with sparse LK
optical flow and RANSAC.  It deliberately reports both a conservative binary
decision and the underlying measurements so borderline cases can be reviewed.

This script is standalone apart from NumPy and OpenCV.  It does not require the
benchmark package or model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


VERSION = "1.0.0"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# All spatial thresholds below refer to the original video resolution.
DEFAULT_THRESHOLDS = {
    "translation_px": 8.0,
    "rotation_deg": 0.75,
    "scale_fraction": 0.020,
    "strong_translation_px": 16.0,
    "strong_rotation_deg": 1.50,
    "strong_scale_fraction": 0.040,
    "min_motion_samples": 2,
    "min_valid_fraction": 0.55,
    "min_inlier_ratio": 0.35,
    "min_spatial_cells": 4,
    "cut_gray_diff": 0.20,
    "cut_gray_correlation": 0.72,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _percentile(values: Iterable[float], q: float) -> float | None:
    vals = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return None if vals.size == 0 else float(np.percentile(vals, q))


def _safe_max(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return None if not vals else float(max(vals))


def _parse_factors(path: Path) -> dict[str, str | None]:
    parts = path.stem.split("__")
    result: dict[str, str | None] = {
        "experiment": None,
        "target": None,
        "scene": None,
        "object": None,
        "camera": None,
        "seed": None,
    }
    if len(parts) >= 6:
        result.update(
            {
                "experiment": parts[0],
                "target": parts[1],
                "scene": parts[2],
                "object": parts[3],
                "camera": parts[4],
                "seed": parts[5],
            }
        )
    return result


def _resize_gray(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_width) / float(width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0), scale


def _grid_features(gray: np.ndarray, max_corners: int = 720) -> np.ndarray | None:
    """Find spatially distributed corners so a moving object cannot dominate."""
    height, width = gray.shape
    rows, cols = 4, 6
    per_cell = max(8, int(math.ceil(max_corners / float(rows * cols))))
    border = max(4, int(round(min(width, height) * 0.025)))
    points: list[np.ndarray] = []
    for gy in range(rows):
        y0 = int(round(gy * height / rows))
        y1 = int(round((gy + 1) * height / rows))
        for gx in range(cols):
            x0 = int(round(gx * width / cols))
            x1 = int(round((gx + 1) * width / cols))
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[max(y0, border) : min(y1, height - border), max(x0, border) : min(x1, width - border)] = 255
            found = cv2.goodFeaturesToTrack(
                gray,
                mask=mask,
                maxCorners=per_cell,
                qualityLevel=0.012,
                minDistance=6.0,
                blockSize=7,
            )
            if found is not None:
                points.append(found.reshape(-1, 2))
    if not points:
        return None
    merged = np.concatenate(points, axis=0)
    return merged[:max_corners].astype(np.float32).reshape(-1, 1, 2)


def _spatial_cell_count(points: np.ndarray, width: int, height: int) -> int:
    if points.size == 0:
        return 0
    gx = np.clip((points[:, 0] / max(width, 1) * 4).astype(int), 0, 3)
    gy = np.clip((points[:, 1] / max(height, 1) * 3).astype(int), 0, 2)
    return len(set((int(x), int(y)) for x, y in zip(gx, gy)))


def _appearance_delta(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    size = (96, 54)
    a = cv2.resize(first, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(second, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    gray_diff = float(np.mean(np.abs(a - b)) / 255.0)
    av = a.ravel() - float(np.mean(a))
    bv = b.ravel() - float(np.mean(b))
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    correlation = 1.0 if denom <= 1e-9 else float(np.dot(av, bv) / denom)
    return gray_diff, correlation


def _estimate_pair(
    first: np.ndarray,
    second: np.ndarray,
    inverse_scale: float,
) -> dict[str, Any]:
    height, width = first.shape
    points0 = _grid_features(first)
    base: dict[str, Any] = {
        "success": False,
        "tracked": 0,
        "inliers": 0,
        "inlier_ratio": None,
        "spatial_cells": 0,
        "residual_median_px": None,
        "center_translation_px": None,
        "rotation_deg": None,
        "scale": None,
        "matrix": None,
    }
    gray_diff, correlation = _appearance_delta(first, second)
    base["gray_diff"] = gray_diff
    base["gray_correlation"] = correlation
    if points0 is None or len(points0) < 12:
        return base

    points1, status1, _ = cv2.calcOpticalFlowPyrLK(
        first,
        second,
        points0,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
    )
    if points1 is None or status1 is None:
        return base
    points0_back, status2, _ = cv2.calcOpticalFlowPyrLK(
        second,
        first,
        points1,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.01),
    )
    if points0_back is None or status2 is None:
        return base
    p0_all = points0.reshape(-1, 2)
    p1_all = points1.reshape(-1, 2)
    p0_back = points0_back.reshape(-1, 2)
    good = status1.ravel().astype(bool) & status2.ravel().astype(bool)
    good &= np.linalg.norm(p0_all - p0_back, axis=1) <= 1.5
    p0 = p0_all[good]
    p1 = p1_all[good]
    base["tracked"] = int(len(p0))
    if len(p0) < 12:
        return base

    affine, inlier_mask = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=3000,
        confidence=0.997,
        refineIters=15,
    )
    if affine is None or inlier_mask is None:
        return base
    inlier = inlier_mask.ravel().astype(bool)
    if not np.any(inlier):
        return base
    predicted = cv2.transform(p0.reshape(1, -1, 2), affine).reshape(-1, 2)
    residuals = np.linalg.norm(p1 - predicted, axis=1)
    a, b, tx = [float(x) for x in affine[0]]
    c, d, ty = [float(x) for x in affine[1]]
    scale = math.sqrt(max(a * a + c * c, 0.0))
    center = np.asarray([width / 2.0, height / 2.0], dtype=float)
    shifted = np.asarray([[a, b], [c, d]], dtype=float) @ center + np.asarray([tx, ty])
    center_translation = float(np.linalg.norm(shifted - center) * inverse_scale)
    matrix = np.asarray([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]], dtype=float)
    base.update(
        {
            "success": True,
            "inliers": int(np.sum(inlier)),
            "inlier_ratio": float(np.mean(inlier)),
            "spatial_cells": _spatial_cell_count(p0[inlier], width, height),
            "residual_median_px": float(np.median(residuals[inlier]) * inverse_scale),
            "center_translation_px": center_translation,
            "rotation_deg": float(math.degrees(math.atan2(c, a))),
            "scale": float(scale),
            "matrix": matrix.tolist(),
        }
    )
    return base


def _decompose_global(matrix: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    a, b, tx = [float(x) for x in matrix[0]]
    c, d, ty = [float(x) for x in matrix[1]]
    center = np.asarray([width / 2.0, height / 2.0], dtype=float)
    shifted = np.asarray([[a, b], [c, d]], dtype=float) @ center + np.asarray([tx, ty])
    return (
        float(np.linalg.norm(shifted - center)),
        float(abs(math.degrees(math.atan2(c, a)))),
        float(abs(math.sqrt(max(a * a + c * c, 0.0)) - 1.0)),
    )


def _longest_temporal_cluster(frame_indices: Iterable[int], max_gap_frames: int) -> int:
    ordered = sorted(set(int(value) for value in frame_indices))
    if not ordered:
        return 0
    longest = 1
    current = 1
    for previous, current_frame in zip(ordered, ordered[1:]):
        if current_frame - previous <= max_gap_frames:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return longest


def _classify(result: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if result.get("cut_pair_count", 0) > 0:
        reasons.append("scene_cut_or_abrupt_reframe")

    translation_hits = int(result.get("direct_translation_hit_count", 0))
    rotation_hits = int(result.get("direct_rotation_hit_count", 0))
    scale_hits = int(result.get("direct_scale_hit_count", 0))
    sustained = int(result.get("direct_motion_cluster_max", 0)) >= int(
        thresholds["min_motion_samples"]
    )
    strong = bool(
        result.get("direct_strong_motion_sample_count", 0) >= 1
        or result.get("strong_adjacent_pair_count", 0) >= 1
    )
    if translation_hits >= int(thresholds["min_motion_samples"]):
        reasons.append("background_translation")
    if rotation_hits >= int(thresholds["min_motion_samples"]):
        reasons.append("background_rotation")
    if scale_hits >= int(thresholds["min_motion_samples"]):
        reasons.append("background_zoom")

    valid_fraction = float(result.get("valid_pair_fraction") or 0.0)
    median_inlier = result.get("median_inlier_ratio")
    reliable = valid_fraction >= thresholds["min_valid_fraction"] and (
        median_inlier is not None and median_inlier >= thresholds["min_inlier_ratio"]
    )
    if reasons and ((sustained and reliable) or strong or result.get("cut_pair_count", 0) > 0):
        return "changed", reasons

    near_threshold = bool(
        (result.get("max_direct_translation_px") or 0.0) >= 0.70 * thresholds["translation_px"]
        or (result.get("max_direct_rotation_deg") or 0.0) >= 0.70 * thresholds["rotation_deg"]
        or (result.get("max_direct_scale_change") or 0.0) >= 0.70 * thresholds["scale_fraction"]
        or max(translation_hits, rotation_hits, scale_hits) >= 1
        or valid_fraction < thresholds["min_valid_fraction"]
    )
    if near_threshold:
        return "review", reasons or ["weak_or_low_confidence_global_motion"]
    return "fixed", []


def analyze_video(
    video_path_text: str,
    stride: int,
    max_width: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(video_path_text)
    factors = _parse_factors(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"video": str(path), **factors, "status": "error", "error": "cannot open video"}
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    frames: list[tuple[int, np.ndarray]] = []
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded == 0 or decoded % stride == 0 or (reported_frames and decoded == reported_frames - 1):
            gray, scale = _resize_gray(frame, max_width)
            frames.append((decoded, gray))
        decoded += 1
    capture.release()
    if decoded > 0 and frames and frames[-1][0] != decoded - 1:
        # The last frame was not sampled when the container frame count was missing/wrong.
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, decoded - 1)
        ok, last = cap.read()
        cap.release()
        if ok:
            gray, scale = _resize_gray(last, max_width)
            frames.append((decoded - 1, gray))
    if len(frames) < 2:
        return {
            "video": str(path),
            **factors,
            "status": "error",
            "error": "fewer than two decoded sample frames",
            "decoded_frames": decoded,
        }

    inverse_scale = 1.0 / scale
    adjacent: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    global_matrix = np.eye(3, dtype=float)
    global_samples: list[dict[str, Any]] = []
    for index in range(1, len(frames)):
        prev_index, prev = frames[index - 1]
        frame_index, current = frames[index]
        pair = _estimate_pair(prev, current, inverse_scale)
        pair.update({"from_frame": prev_index, "to_frame": frame_index})
        adjacent.append(pair)
        if pair["success"]:
            global_matrix = np.asarray(pair["matrix"], dtype=float) @ global_matrix
            translation, rotation, scale_change = _decompose_global(global_matrix, width, height)
            global_samples.append(
                {
                    "frame_index": frame_index,
                    "translation_px": translation,
                    "rotation_deg": rotation,
                    "scale_change": scale_change,
                    "source": "chained_adjacent",
                }
            )

        ref = _estimate_pair(frames[0][1], current, inverse_scale)
        ref.update({"from_frame": frames[0][0], "to_frame": frame_index})
        reference.append(ref)
        if ref["success"]:
            reference_matrix = np.asarray(ref["matrix"], dtype=float)
            translation, rotation, scale_change = _decompose_global(reference_matrix, width, height)
            global_samples.append(
                {
                    "frame_index": frame_index,
                    "translation_px": translation,
                    "rotation_deg": rotation,
                    "scale_change": scale_change,
                    "source": "direct_reference",
                }
            )

    valid_adjacent = [row for row in adjacent if row["success"]]
    valid_reference = [row for row in reference if row["success"]]
    reliable_global = []
    for sample in global_samples:
        frame_index = sample["frame_index"]
        if sample["source"] == "direct_reference":
            matching = next(
                (r for r in reference if r["to_frame"] == frame_index and r["success"]), None
            )
        else:
            matching = next(
                (r for r in adjacent if r["to_frame"] == frame_index and r["success"]), None
            )
        if matching is not None and matching["inlier_ratio"] >= thresholds["min_inlier_ratio"]:
            if matching["spatial_cells"] >= int(thresholds["min_spatial_cells"]):
                reliable_global.append(sample)

    cut_pairs = [
        row
        for row in adjacent
        if row["gray_diff"] >= thresholds["cut_gray_diff"]
        and row["gray_correlation"] <= thresholds["cut_gray_correlation"]
        and (not row["success"] or row["inlier_ratio"] < thresholds["min_inlier_ratio"])
    ]
    reliable_direct = [row for row in reliable_global if row["source"] == "direct_reference"]
    direct_translation_hits = [
        row for row in reliable_direct if row["translation_px"] >= thresholds["translation_px"]
    ]
    direct_rotation_hits = [
        row for row in reliable_direct if row["rotation_deg"] >= thresholds["rotation_deg"]
    ]
    direct_scale_hits = [
        row for row in reliable_direct if row["scale_change"] >= thresholds["scale_fraction"]
    ]
    direct_motion_hit_frames = [
        row["frame_index"]
        for row in reliable_direct
        if row["translation_px"] >= thresholds["translation_px"]
        or row["rotation_deg"] >= thresholds["rotation_deg"]
        or row["scale_change"] >= thresholds["scale_fraction"]
    ]
    direct_strong_hits = [
        row
        for row in reliable_direct
        if row["translation_px"] >= thresholds["strong_translation_px"]
        or row["rotation_deg"] >= thresholds["strong_rotation_deg"]
        or row["scale_change"] >= thresholds["strong_scale_fraction"]
    ]
    strong_adjacent = [
        row
        for row in valid_adjacent
        if row["inlier_ratio"] >= thresholds["min_inlier_ratio"]
        and row["spatial_cells"] >= int(thresholds["min_spatial_cells"])
        and (
            row["center_translation_px"] >= thresholds["strong_translation_px"]
            or abs(row["rotation_deg"]) >= thresholds["strong_rotation_deg"]
            or abs(row["scale"] - 1.0) >= thresholds["strong_scale_fraction"]
        )
    ]

    result: dict[str, Any] = {
        "video": str(path),
        "filename": path.name,
        **factors,
        "status": "ok",
        "width": width,
        "height": height,
        "fps": fps,
        "reported_frames": reported_frames,
        "decoded_frames": decoded,
        "sample_count": len(frames),
        "valid_adjacent_pairs": len(valid_adjacent),
        "valid_reference_pairs": len(valid_reference),
        "valid_pair_fraction": len(valid_adjacent) / max(len(adjacent), 1),
        "median_inlier_ratio": _percentile(
            [row["inlier_ratio"] for row in valid_adjacent], 50
        ),
        "median_residual_px": _percentile(
            [row["residual_median_px"] for row in valid_adjacent], 50
        ),
        "max_global_translation_px": _safe_max(
            row["translation_px"] for row in reliable_global
        ),
        "p95_global_translation_px": _percentile(
            [row["translation_px"] for row in reliable_global], 95
        ),
        "max_global_rotation_deg": _safe_max(row["rotation_deg"] for row in reliable_global),
        "p95_global_rotation_deg": _percentile(
            [row["rotation_deg"] for row in reliable_global], 95
        ),
        "max_global_scale_change": _safe_max(row["scale_change"] for row in reliable_global),
        "p95_global_scale_change": _percentile(
            [row["scale_change"] for row in reliable_global], 95
        ),
        "max_adjacent_translation_px": _safe_max(
            row["center_translation_px"] for row in valid_adjacent
        ),
        "max_adjacent_rotation_deg": _safe_max(
            abs(row["rotation_deg"]) for row in valid_adjacent
        ),
        "max_adjacent_scale_change": _safe_max(
            abs(row["scale"] - 1.0) for row in valid_adjacent
        ),
        "max_direct_translation_px": _safe_max(
            row["translation_px"] for row in reliable_direct
        ),
        "p95_direct_translation_px": _percentile(
            [row["translation_px"] for row in reliable_direct], 95
        ),
        "max_direct_rotation_deg": _safe_max(row["rotation_deg"] for row in reliable_direct),
        "p95_direct_rotation_deg": _percentile(
            [row["rotation_deg"] for row in reliable_direct], 95
        ),
        "max_direct_scale_change": _safe_max(row["scale_change"] for row in reliable_direct),
        "p95_direct_scale_change": _percentile(
            [row["scale_change"] for row in reliable_direct], 95
        ),
        "direct_translation_hit_count": len(direct_translation_hits),
        "direct_rotation_hit_count": len(direct_rotation_hits),
        "direct_scale_hit_count": len(direct_scale_hits),
        "direct_motion_cluster_max": _longest_temporal_cluster(
            direct_motion_hit_frames, max_gap_frames=2 * stride
        ),
        "direct_strong_motion_sample_count": len(direct_strong_hits),
        "strong_adjacent_pair_count": len(strong_adjacent),
        # Backward-compatible aliases now intentionally use direct-reference
        # evidence, not drift-prone chained transforms.
        "translation_hit_count": len(direct_translation_hits),
        "rotation_hit_count": len(direct_rotation_hits),
        "scale_hit_count": len(direct_scale_hits),
        "strong_motion_sample_count": len(direct_strong_hits),
        "cut_pair_count": len(cut_pairs),
        "elapsed_seconds": time.perf_counter() - started,
        "motion_samples": reliable_global,
        "cut_pairs": cut_pairs,
    }
    decision, reasons = _classify(result, thresholds)
    result["decision"] = decision
    result["reasons"] = reasons
    return result


def _contact_sheet(video_path: Path, output_path: Path, decision: str) -> None:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    if frame_count <= 0:
        capture.release()
        return
    indices = np.linspace(0, frame_count - 1, 9).round().astype(int).tolist()
    tiles: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        target_width = 320
        target_height = max(1, int(round(frame.shape[0] * target_width / frame.shape[1])))
        tile = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (106, 24), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"frame {index}",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    capture.release()
    if not tiles:
        return
    while len(tiles) < 9:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i : i + 3], axis=1) for i in range(0, 9, 3)]
    sheet = np.concatenate(rows, axis=0)
    banner = np.full((42, sheet.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        banner,
        f"{decision}: {video_path.name}",
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate([banner, sheet], axis=0))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(_json_ready(value), ensure_ascii=False, separators=(",", ":"))
    return value


def _write_outputs(
    results: list[dict[str, Any]],
    output_dir: Path,
    source_dir: Path,
    thresholds: dict[str, float],
    make_contact_sheets: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda row: row.get("filename") or row.get("video") or "")
    scalar_keys: list[str] = []
    for row in ordered:
        for key, value in row.items():
            if key not in scalar_keys and not isinstance(value, (list, dict)):
                scalar_keys.append(key)
    with (output_dir / "camera_motion_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys + ["reasons"])
        writer.writeheader()
        for row in ordered:
            payload = {key: _csv_value(row.get(key)) for key in scalar_keys}
            payload["reasons"] = ";".join(row.get("reasons", []))
            writer.writerow(payload)
    with (output_dir / "camera_motion_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, separators=(",", ":")) + "\n")

    counts: dict[str, int] = {}
    for row in ordered:
        key = row.get("decision") or row.get("status") or "unknown"
        counts[key] = counts.get(key, 0) + 1
    changed = [row for row in ordered if row.get("decision") == "changed"]
    review = [row for row in ordered if row.get("decision") == "review"]
    summary = {
        "schema_version": "1.0.0",
        "detector_version": VERSION,
        "source_dir": str(source_dir.resolve()),
        "video_count": len(ordered),
        "counts": counts,
        "thresholds": thresholds,
        "changed_videos": [row.get("filename") for row in changed],
        "review_videos": [row.get("filename") for row in review],
        "error_videos": [row.get("filename") or row.get("video") for row in ordered if row.get("status") == "error"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if make_contact_sheets:
        review_dir = output_dir / "contact_sheets"
        for row in changed + review:
            video_path = Path(row["video"])
            _contact_sheet(video_path, review_dir / f"{row['decision']}__{video_path.stem}.jpg", row["decision"])
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Directory containing videos")
    parser.add_argument("--output", type=Path, required=True, help="Audit output directory")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--stride", type=int, default=4, help="Analyze every Nth decoded frame")
    parser.add_argument("--max-width", type=int, default=432, help="Analysis width cap")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-contact-sheets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    videos = sorted(
        path.resolve()
        for path in args.videos.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if args.limit is not None:
        videos = videos[: max(0, args.limit)]
    if not videos:
        print(f"No videos found under {args.videos}", file=sys.stderr)
        return 2
    thresholds = dict(DEFAULT_THRESHOLDS)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for index, video in enumerate(videos, 1):
            results.append(analyze_video(str(video), args.stride, args.max_width, thresholds))
            print(f"[{index}/{len(videos)}] {video.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(analyze_video, str(video), args.stride, args.max_width, thresholds): video
                for video in videos
            }
            for index, future in enumerate(as_completed(futures), 1):
                video = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # keep the batch auditable
                    results.append(
                        {
                            "video": str(video),
                            "filename": video.name,
                            **_parse_factors(video),
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if index == 1 or index % 25 == 0 or index == len(videos):
                    print(f"[{index}/{len(videos)}] analyzed", flush=True)
    summary = _write_outputs(
        results,
        args.output.resolve(),
        args.videos.resolve(),
        thresholds,
        not args.no_contact_sheets,
    )
    summary["elapsed_seconds"] = time.perf_counter() - started
    with (args.output.resolve() / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))
    print(f"elapsed_seconds={summary['elapsed_seconds']:.3f}")
    return 0 if not summary["error_videos"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
