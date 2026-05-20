#!/usr/bin/env python3
"""Track generated free-fall videos and estimate a gravity-related proxy.

The estimator is intentionally lightweight.  It uses color segmentation to find
the ball center in each frame, fits the vertical trajectory with a quadratic,
and reports g_hat in ball-diameter units per second squared.  For phase 1 the
main signal is whether g_hat is monotonic/correlated with the manifest g values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLOR_RANGES = {
    "red": [((0, 70, 50), (12, 255, 255)), ((170, 70, 50), (179, 255, 255))],
    "orange": [((5, 55, 50), (28, 255, 255))],
    "yellow": [((18, 60, 60), (42, 255, 255))],
    "red_orange": [((0, 55, 50), (32, 255, 255)), ((170, 55, 50), (179, 255, 255))],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--eval-outdir", type=Path, required=True)
    parser.add_argument("--color", default="red_orange", choices=sorted(COLOR_RANGES))
    parser.add_argument("--min-area", type=float, default=40.0)
    parser.add_argument("--fit-max-seconds", type=float, default=2.5)
    parser.add_argument("--min-fit-points", type=int, default=8)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path, max_jobs: int | None) -> list[dict[str, Any]]:
    jobs = []
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_mask(frame_bgr: np.ndarray, color_name: str) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color_name]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_ball(frame_bgr: np.ndarray, color_name: str, min_area: float) -> dict[str, float] | None:
    mask = build_mask(frame_bgr, color_name)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[0]
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-8:
        return None
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    (_, _), radius = cv2.minEnclosingCircle(contour)
    significant = sum(1 for item in contours if cv2.contourArea(item) >= max(min_area, 0.15 * area))
    return {"cx": cx, "cy": cy, "radius": float(radius), "area": area, "significant_contours": significant}


def locate_video(generated_root: Path, job_id: str) -> Path | None:
    direct = generated_root / "videos" / f"{job_id}.mp4"
    if direct.exists():
        return direct
    direct = generated_root / f"{job_id}.mp4"
    if direct.exists():
        return direct
    matches = sorted(generated_root.rglob(f"{job_id}.mp4"))
    return matches[0] if matches else None


def interpolate(values: list[float | None], max_gap: int = 4) -> list[float | None]:
    out = values[:]
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        start = i - 1
        j = i
        while j < n and out[j] is None:
            j += 1
        gap = j - i
        if start >= 0 and j < n and out[start] is not None and out[j] is not None and gap <= max_gap:
            for k in range(1, gap + 1):
                alpha = k / (gap + 1)
                out[start + k] = (1.0 - alpha) * out[start] + alpha * out[j]
        i = j
    return out


def fit_quadratic(t: np.ndarray, y: np.ndarray) -> dict[str, float]:
    coeff = np.polyfit(t, y, 2)
    y_hat = np.polyval(coeff, t)
    resid = y - y_hat
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "a": float(coeff[0]),
        "b": float(coeff[1]),
        "c": float(coeff[2]),
        "rmse": float(np.sqrt(np.mean(resid * resid))),
        "r2": r2,
    }


def choose_fit_indices(rows: list[dict[str, Any]], fit_max_seconds: float, min_fit_points: int, diameter_px: float) -> list[int]:
    valid = [i for i, row in enumerate(rows) if row["cx_px"] is not None and row["cy_px"] is not None]
    if not valid:
        return []
    t0 = rows[valid[0]]["time_s"]
    limited = [i for i in valid if rows[i]["time_s"] - t0 <= fit_max_seconds]
    if len(limited) < min_fit_points:
        limited = valid[: max(min_fit_points, len(valid))]

    stop_eps = max(1.0, 0.035 * diameter_px)
    for pos in range(min_fit_points, max(min_fit_points, len(limited) - 3)):
        recent = limited[pos : pos + 4]
        dy = [abs(rows[b]["cy_px"] - rows[a]["cy_px"]) for a, b in zip(recent, recent[1:])]
        total_drop = rows[recent[0]]["cy_px"] - rows[limited[0]]["cy_px"]
        if total_drop > 0.8 * diameter_px and max(dy) < stop_eps:
            return limited[:pos]
    return limited


def make_overlay(video_path: Path, outpath: Path, rows: list[dict[str, Any]], fit: dict[str, float], fit_indices: list[int], diameter_px: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    row_map = {row["frame_idx"]: row for row in rows}
    t0 = rows[fit_indices[0]]["time_s"] if fit_indices else 0.0
    y0 = rows[fit_indices[0]]["cy_px"] if fit_indices else 0.0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = row_map.get(frame_idx)
        if row and row["cx_px"] is not None and row["cy_px"] is not None:
            cx = int(round(row["cx_px"]))
            cy = int(round(row["cy_px"]))
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        if fit_indices and row:
            dt = row["time_s"] - t0
            y_down_d = fit["a"] * dt * dt + fit["b"] * dt + fit["c"]
            cy_fit = int(round(y0 + y_down_d * diameter_px))
            if 0 <= cy_fit < height:
                cv2.circle(frame, (width // 2, cy_fit), 4, (255, 0, 0), -1)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()


def evaluate_video(job: dict[str, Any], video_path: Path, outdir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6 or math.isnan(fps):
        fps = float(job.get("fps", 16))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows: list[dict[str, Any]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det = detect_ball(frame, args.color, args.min_area)
        row: dict[str, Any] = {"frame_idx": frame_idx, "time_s": frame_idx / fps}
        if det is None:
            row.update({"found": False, "cx_px_raw": None, "cy_px_raw": None, "radius_px": None, "area_px": None, "significant_contours": 0})
        else:
            row.update(
                {
                    "found": True,
                    "cx_px_raw": det["cx"],
                    "cy_px_raw": det["cy"],
                    "radius_px": det["radius"],
                    "area_px": det["area"],
                    "significant_contours": det["significant_contours"],
                }
            )
        rows.append(row)
        frame_idx += 1
    cap.release()

    if not rows:
        raise RuntimeError("No frames decoded.")

    cx = interpolate([row["cx_px_raw"] for row in rows])
    cy = interpolate([row["cy_px_raw"] for row in rows])
    for row, cx_val, cy_val in zip(rows, cx, cy):
        row["cx_px"] = cx_val
        row["cy_px"] = cy_val

    radii = [row["radius_px"] for row in rows[:12] if row["radius_px"] is not None]
    if not radii:
        radii = [row["radius_px"] for row in rows if row["radius_px"] is not None]
    if not radii:
        raise RuntimeError("Ball was never detected.")
    diameter_px = 2.0 * float(np.median(np.array(radii, dtype=float)))

    fit_indices = choose_fit_indices(rows, args.fit_max_seconds, args.min_fit_points, diameter_px)
    if len(fit_indices) < args.min_fit_points:
        raise RuntimeError(f"Too few fit points: {len(fit_indices)}")

    t0 = rows[fit_indices[0]]["time_s"]
    cy0 = rows[fit_indices[0]]["cy_px"]
    cx0 = rows[fit_indices[0]]["cx_px"]
    t = np.array([rows[i]["time_s"] - t0 for i in fit_indices], dtype=float)
    y_down_d = np.array([(rows[i]["cy_px"] - cy0) / diameter_px for i in fit_indices], dtype=float)
    x_d = np.array([(rows[i]["cx_px"] - cx0) / diameter_px for i in fit_indices], dtype=float)
    fit_y = fit_quadratic(t, y_down_d)
    fit_x = fit_quadratic(t, x_d)

    g_hat = 2.0 * fit_y["a"]
    y_span = float(np.max(y_down_d) - np.min(y_down_d)) if len(y_down_d) else 0.0
    x_span = float(np.max(x_d) - np.min(x_d)) if len(x_d) else 0.0
    detection_rate = float(np.mean([1.0 if row["found"] else 0.0 for row in rows]))
    single_object_rate = float(np.mean([1.0 if row["significant_contours"] <= 1 else 0.0 for row in rows if row["found"]])) if any(row["found"] for row in rows) else 0.0

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "trajectory.csv", rows)
    if not args.no_overlay:
        make_overlay(video_path, outdir / "tracked_overlay.mp4", rows, fit_y, fit_indices, diameter_px)

    result = {
        "job_id": job["job_id"],
        "video_path": str(video_path),
        "experiment": job.get("experiment"),
        "variant": job.get("variant"),
        "camera": job.get("camera"),
        "target_param_name": job.get("target_param_name", "g_hidden"),
        "target_param_value": job.get("target_param_value"),
        "frame_count": len(rows),
        "fps": fps,
        "width": width,
        "height": height,
        "diameter_px": diameter_px,
        "fit_points": len(fit_indices),
        "g_hat_D_per_s2": float(g_hat),
        "fit_y": fit_y,
        "fit_x": fit_x,
        "detection_rate": detection_rate,
        "single_object_rate": single_object_rate,
        "vertical_motion_D": y_span,
        "horizontal_drift_D": x_span,
        "horizontal_to_vertical_ratio": x_span / max(y_span, 1e-8),
    }
    save_json(outdir / "evaluation.json", result)
    return result


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1)
        for k in range(i, j):
            out[order[k]] = rank
        i = j
    return out


def corrcoef(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    arr_x = np.array(x, dtype=float)
    arr_y = np.array(y, dtype=float)
    if np.std(arr_x) < 1e-12 or np.std(arr_y) < 1e-12:
        return None
    return float(np.corrcoef(arr_x, arr_y)[0, 1])


def summarize(results: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in results if row.get("target_param_value") is not None and math.isfinite(row["g_hat_D_per_s2"])]
    truth = [float(row["target_param_value"]) for row in valid]
    estimate = [float(row["g_hat_D_per_s2"]) for row in valid]
    pearson = corrcoef(truth, estimate)
    spearman = corrcoef(ranks(truth), ranks(estimate)) if len(valid) >= 2 else None
    slope = None
    intercept = None
    r2 = None
    if len(valid) >= 2 and pearson is not None:
        coeff = np.polyfit(np.array(truth), np.array(estimate), 1)
        slope, intercept = float(coeff[0]), float(coeff[1])
        r2 = float(pearson * pearson)
    return {
        "num_results": len(results),
        "num_errors": len(errors),
        "num_valid_param_pairs": len(valid),
        "mean_detection_rate": float(np.mean([row["detection_rate"] for row in results])) if results else 0.0,
        "mean_fit_r2": float(np.mean([row["fit_y"]["r2"] for row in results])) if results else 0.0,
        "pearson_g_true_vs_g_hat": pearson,
        "spearman_g_true_vs_g_hat": spearman,
        "linear_fit_g_hat_vs_g_true": {"slope": slope, "intercept": intercept, "r2": r2},
        "acceptance_hint": {
            "generation_and_tracking": "All jobs should decode and have high detection_rate.",
            "physics_signal": "For this prompt-conditioned phase-1 baseline, g_hat_D_per_s2 should increase with target_param_value.",
        },
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    jobs = read_manifest(args.manifest, args.max_jobs)
    args.eval_outdir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []
    for index, job in enumerate(jobs, 1):
        job_id = job["job_id"]
        try:
            video = locate_video(args.generated_root, job_id)
            if video is None:
                raise FileNotFoundError(f"Missing generated video for job_id={job_id}")
            outdir = args.eval_outdir / "per_video" / job_id
            result = evaluate_video(job, video, outdir, args)
            results.append(result)
            print(f"[{index}/{len(jobs)}] ok {job_id} g_hat={result['g_hat_D_per_s2']:.4f} r2={result['fit_y']['r2']:.3f}")
        except Exception as exc:
            item = {"job_id": job_id, "error": f"{type(exc).__name__}: {exc}"}
            errors.append(item)
            print(f"[{index}/{len(jobs)}] error {json.dumps(item, ensure_ascii=False)}")

    summary_rows = [
        {
            "job_id": row["job_id"],
            "variant": row["variant"],
            "target_param_value": row["target_param_value"],
            "g_hat_D_per_s2": row["g_hat_D_per_s2"],
            "fit_r2": row["fit_y"]["r2"],
            "detection_rate": row["detection_rate"],
            "horizontal_to_vertical_ratio": row["horizontal_to_vertical_ratio"],
            "video_path": row["video_path"],
        }
        for row in results
    ]
    write_csv(args.eval_outdir / "summary.csv", summary_rows)
    save_json(args.eval_outdir / "phase1_report.json", summarize(results, errors))
    print(f"saved: {args.eval_outdir}")


if __name__ == "__main__":
    main()
