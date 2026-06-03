#!/usr/bin/env python3
"""Track an orange benchmark ball and fit a free-fall acceleration proxy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--max-area-ratio", type=float, default=0.20)
    return parser.parse_args()


def load_metadata(metadata_dir: Path | None, job_id: str) -> dict[str, Any]:
    if metadata_dir is None:
        return {}
    path = metadata_dir / f"{job_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def detect_orange_ball(frame: np.ndarray, min_area: float, max_area: float) -> tuple[float, float, float] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 45, 40], dtype=np.uint8)
    upper1 = np.array([28, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 45, 40], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x = float(moments["m10"] / moments["m00"])
        y = float(moments["m01"] / moments["m00"])
        candidates.append((area, x, y))
    if not candidates:
        return None
    area, x, y = max(candidates, key=lambda item: item[0])
    return x, y, area


def track_video(video_path: Path, fps: float, min_area: float, max_area_ratio: float) -> list[dict[str, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
    max_area = float(width * height * max_area_ratio)
    rows: list[dict[str, float]] = []
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detected = detect_orange_ball(frame, min_area=min_area, max_area=max_area)
        row: dict[str, float] = {
            "frame_idx": float(frame_idx),
            "time_s": float(frame_idx / fps),
            "detected": 0.0,
            "x_px": float("nan"),
            "y_px": float("nan"),
            "area_px": float("nan"),
        }
        if detected is not None:
            x, y, area = detected
            row.update({"detected": 1.0, "x_px": x, "y_px": y, "area_px": area})
        rows.append(row)
        frame_idx += 1
    capture.release()
    return rows


def fit_acceleration(rows: list[dict[str, float]]) -> dict[str, float]:
    valid = [row for row in rows if row["detected"] > 0]
    if len(valid) < 8:
        return {
            "num_frames": float(len(rows)),
            "num_detected": float(len(valid)),
            "detection_rate": float(len(valid) / max(1, len(rows))),
            "g_proxy_px_per_s2": float("nan"),
            "fit_r2": float("nan"),
        }
    t = np.array([row["time_s"] for row in valid], dtype=np.float64)
    y = np.array([row["y_px"] for row in valid], dtype=np.float64)
    coeff = np.polyfit(t, y, deg=2)
    y_hat = np.polyval(coeff, t)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "num_frames": float(len(rows)),
        "num_detected": float(len(valid)),
        "detection_rate": float(len(valid) / max(1, len(rows))),
        "quad_a": float(coeff[0]),
        "linear_b": float(coeff[1]),
        "const_c": float(coeff[2]),
        "g_proxy_px_per_s2": float(2.0 * coeff[0]),
        "fit_r2": r2,
    }


def write_tracks(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_idx", "time_s", "detected", "x_px", "y_px", "area_px"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    videos = sorted(args.videos_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no mp4 files found in {args.videos_dir}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    tracks_dir = args.outdir / "tracks"
    summary_path = args.outdir / "summary.csv"
    summary_rows: list[dict[str, Any]] = []

    for video_path in videos:
        job_id = video_path.stem
        metadata = load_metadata(args.metadata_dir, job_id)
        fps = float(metadata.get("fps") or args.fps)
        rows = track_video(video_path, fps=fps, min_area=args.min_area, max_area_ratio=args.max_area_ratio)
        write_tracks(tracks_dir / f"{job_id}.csv", rows)
        fit = fit_acceleration(rows)
        summary_rows.append(
            {
                "job_id": job_id,
                "video": str(video_path),
                "target_param_name": metadata.get("target_param_name", ""),
                "target_param_value": metadata.get("target_param_value", ""),
                "fps": fps,
                **fit,
            }
        )
        print(f"{job_id}: detected={fit['num_detected']:.0f}/{fit['num_frames']:.0f}, g_proxy={fit['g_proxy_px_per_s2']}")

    fields = [
        "job_id",
        "video",
        "target_param_name",
        "target_param_value",
        "fps",
        "num_frames",
        "num_detected",
        "detection_rate",
        "quad_a",
        "linear_b",
        "const_c",
        "g_proxy_px_per_s2",
        "fit_r2",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
