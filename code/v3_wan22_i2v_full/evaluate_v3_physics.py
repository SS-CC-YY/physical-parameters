#!/usr/bin/env python3
"""Track V3 generated videos and write proxy evaluation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HSV_RANGES = {
    "red_orange": [((0, 50, 40), (34, 255, 255)), ((170, 50, 40), (179, 255, 255))],
    "orange": [((5, 50, 40), (34, 255, 255))],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--camera", default="CAM_Side")
    parser.add_argument("--include-all-cameras", action="store_true")
    parser.add_argument("--color", choices=sorted(HSV_RANGES), default="red_orange")
    parser.add_argument("--min-area", type=float, default=35.0)
    parser.add_argument("--fallback-fps", type=float, default=16.0)
    parser.add_argument("--override-fps", type=float, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def read_manifest(path: Path, max_jobs: int | None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
    return jobs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def locate_video(root: Path, job_id: str) -> Path | None:
    for candidate in [root / "videos" / f"{job_id}.mp4", root / f"{job_id}.mp4"]:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(f"{job_id}.mp4"))
    return matches[0] if matches else None


def detect_target(frame: np.ndarray, color: str, min_area: float) -> dict[str, float] | None:
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-8:
        return None
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    (_, _), radius = cv2.minEnclosingCircle(contour)
    return {"cx": cx, "cy": cy, "radius": float(radius), "area": area, "num_contours": float(len(contours))}


def track_video(video_path: Path, job: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = args.override_fps if args.override_fps else cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6 or math.isnan(float(fps)):
        fps = float(job.get("fps") or args.fallback_fps)
    rows: list[dict[str, Any]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detect_target(frame, args.color, args.min_area)
        row: dict[str, Any] = {"frame_idx": idx, "time_s": idx / float(fps), "found": det is not None}
        if det:
            row.update({"cx_px": det["cx"], "cy_px": det["cy"], "radius_px": det["radius"], "area_px": det["area"], "num_contours": det["num_contours"]})
        else:
            row.update({"cx_px": None, "cy_px": None, "radius_px": None, "area_px": None, "num_contours": 0})
        rows.append(row)
        idx += 1
    cap.release()
    return rows, float(fps)


def arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([np.nan if row[key] is None else row[key] for row in rows], dtype=np.float64)


def count_turning_points(values: np.ndarray) -> int:
    valid = values[np.isfinite(values)]
    if len(valid) < 5:
        return 0
    diff = np.diff(valid)
    signs = np.sign(diff)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def trajectory_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = arr(rows, "cx_px")
    y = arr(rows, "cy_px")
    radius = arr(rows, "radius_px")
    valid = np.isfinite(x) & np.isfinite(y)
    found = int(np.sum(valid))
    if found < 2:
        return {
            "found_frames": found,
            "detection_rate": found / max(1, len(rows)),
            "x_range_px": float("nan"),
            "y_range_px": float("nan"),
            "path_length_px": float("nan"),
            "turning_points_x": 0,
            "turning_points_y": 0,
            "motion_score": float("nan"),
            "radius_px_median": float("nan"),
        }
    xv = x[valid]
    yv = y[valid]
    dx = np.diff(xv)
    dy = np.diff(yv)
    path = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    radius_med = float(np.nanmedian(radius[np.isfinite(radius)])) if np.any(np.isfinite(radius)) else float("nan")
    range_x = float(np.nanmax(xv) - np.nanmin(xv))
    range_y = float(np.nanmax(yv) - np.nanmin(yv))
    motion_score = path / max(radius_med, 1.0) if np.isfinite(radius_med) else float("nan")
    return {
        "found_frames": found,
        "detection_rate": found / max(1, len(rows)),
        "x_range_px": range_x,
        "y_range_px": range_y,
        "path_length_px": path,
        "turning_points_x": count_turning_points(xv),
        "turning_points_y": count_turning_points(yv),
        "motion_score": motion_score,
        "radius_px_median": radius_med,
    }


def estimate_proxy(job: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    experiment = job["experiment"]
    proxy = {
        "trajectory_quality": {
            "detection_rate": metrics["detection_rate"],
            "motion_score": metrics["motion_score"],
            "x_range_px": metrics["x_range_px"],
            "y_range_px": metrics["y_range_px"],
        }
    }
    if experiment in {"v3_B", "v3_D", "v3_F"}:
        proxy["oscillation_proxy"] = {
            "turning_points_x": metrics["turning_points_x"],
            "turning_points_y": metrics["turning_points_y"],
        }
    if experiment in {"v3_A", "v3_C", "v3_E", "v3_G"}:
        proxy["transport_proxy"] = {
            "x_range_px": metrics["x_range_px"],
            "y_range_px": metrics["y_range_px"],
            "path_length_px": metrics["path_length_px"],
        }
    return proxy


def evaluate_job(job: dict[str, Any], video_path: Path, tracks_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows, fps = track_video(video_path, job, args)
    write_csv(tracks_dir / f"{job['job_id']}.csv", rows)
    metrics = trajectory_metrics(rows)
    proxy = estimate_proxy(job, metrics)
    return {
        "job_id": job["job_id"],
        "experiment": job.get("experiment"),
        "variant": job.get("variant"),
        "camera": job.get("camera"),
        "video": str(video_path),
        "status": "ok",
        "target_param_name": "multi_hidden",
        "target_param_value": json.dumps(job.get("hidden_params", {}), ensure_ascii=False),
        "estimate_name": "v3_proxy",
        "estimate_value": json.dumps(proxy, ensure_ascii=False),
        "abs_error": "",
        "rel_error": "",
        "fps": fps,
        "num_frames": len(rows),
        "proxy_note": "V3 first-pass tracking/proxy evaluation; use geometry-aware multi-parameter fitting for final metrics.",
        **metrics,
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    jobs = read_manifest(args.manifest, args.max_jobs)
    tracks_dir = args.outdir / "tracks"
    summary: list[dict[str, Any]] = []

    for job in jobs:
        if not args.include_all_cameras and job.get("camera") != args.camera:
            summary.append({"job_id": job.get("job_id"), "experiment": job.get("experiment"), "variant": job.get("variant"), "camera": job.get("camera"), "status": "skipped_camera"})
            continue
        video_path = locate_video(args.generated_root, job["job_id"])
        if video_path is None:
            summary.append({"job_id": job["job_id"], "experiment": job.get("experiment"), "variant": job.get("variant"), "camera": job.get("camera"), "status": "missing_video"})
            continue
        try:
            row = evaluate_job(job, video_path, tracks_dir, args)
            print(f"{job['job_id']}: detection={row['detection_rate']:.3f}, motion_score={row['motion_score']}")
            summary.append(row)
        except Exception as exc:
            print(f"{job['job_id']}: error: {exc}")
            summary.append({"job_id": job["job_id"], "experiment": job.get("experiment"), "variant": job.get("variant"), "camera": job.get("camera"), "status": "error", "error": str(exc)})

    write_csv(args.outdir / "summary.csv", summary)
    write_json(args.outdir / "summary.json", summary)
    print(f"wrote {args.outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()

