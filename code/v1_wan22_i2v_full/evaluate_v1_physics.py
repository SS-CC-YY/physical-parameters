#!/usr/bin/env python3
"""Estimate V1 hidden physics parameters from generated videos."""

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
    "red_orange": [((0, 55, 45), (32, 255, 255)), ((170, 55, 45), (179, 255, 255))],
    "orange": [((5, 55, 45), (32, 255, 255))],
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
    parser.add_argument("--ball-radius-m", type=float, default=0.24)
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
    candidates = [
        root / "videos" / f"{job_id}.mp4",
        root / f"{job_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(f"{job_id}.mp4"))
    return matches[0] if matches else None


def build_mask(frame: np.ndarray, color: str) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_ball(frame: np.ndarray, color: str, min_area: float) -> dict[str, float] | None:
    mask = build_mask(frame, color)
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
    return {"cx": cx, "cy": cy, "radius": float(radius), "area": area}


def interpolate_nan(values: np.ndarray, max_gap: int = 5) -> np.ndarray:
    out = values.copy()
    n = len(out)
    valid = np.where(np.isfinite(out))[0]
    if len(valid) < 2:
        return out
    for i in range(n):
        if np.isfinite(out[i]):
            continue
        left_candidates = valid[valid < i]
        right_candidates = valid[valid > i]
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            continue
        left = int(left_candidates[-1])
        right = int(right_candidates[0])
        if right - left - 1 <= max_gap:
            alpha = (i - left) / (right - left)
            out[i] = (1.0 - alpha) * out[left] + alpha * out[right]
    return out


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


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
        det = detect_ball(frame, args.color, args.min_area)
        row: dict[str, Any] = {"frame_idx": idx, "time_s": idx / float(fps), "found": det is not None}
        if det:
            row.update({"cx_px": det["cx"], "cy_px": det["cy"], "radius_px": det["radius"], "area_px": det["area"]})
        else:
            row.update({"cx_px": None, "cy_px": None, "radius_px": None, "area_px": None})
        rows.append(row)
        idx += 1
    cap.release()
    return rows, float(fps)


def arrays_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.array([r["time_s"] for r in rows], dtype=np.float64)
    x = np.array([np.nan if r["cx_px"] is None else r["cx_px"] for r in rows], dtype=np.float64)
    y = np.array([np.nan if r["cy_px"] is None else r["cy_px"] for r in rows], dtype=np.float64)
    radius = np.array([np.nan if r["radius_px"] is None else r["radius_px"] for r in rows], dtype=np.float64)
    return t, interpolate_nan(x), interpolate_nan(y), radius


def fit_poly(t: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, float, float]:
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid * resid)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(resid * resid))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return coeff, rmse, r2


def estimate_scale(radius_px: np.ndarray, ball_radius_m: float) -> tuple[float, float]:
    valid = radius_px[np.isfinite(radius_px) & (radius_px > 0)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    median_radius = float(np.median(valid))
    return ball_radius_m / median_radius, median_radius


def local_extrema(values: np.ndarray, mode: str, min_sep: int) -> list[int]:
    indices: list[int] = []
    last = -10**9
    for i in range(1, len(values) - 1):
        if not np.isfinite(values[i - 1 : i + 2]).all():
            continue
        is_extreme = values[i] >= values[i - 1] and values[i] >= values[i + 1] if mode == "max" else values[i] <= values[i - 1] and values[i] <= values[i + 1]
        if not is_extreme:
            continue
        if i - last < min_sep:
            if indices:
                better = values[i] > values[indices[-1]] if mode == "max" else values[i] < values[indices[-1]]
                if better:
                    indices[-1] = i
                    last = i
            continue
        indices.append(i)
        last = i
    return indices


def estimate_v1_a(t: np.ndarray, y: np.ndarray, radius_px: float, m_per_px: float) -> dict[str, Any]:
    y_s = moving_average(y, 5)
    valid = np.isfinite(y_s)
    t_v = t[valid]
    y_v = y_s[valid]
    if len(t_v) < 12:
        raise RuntimeError("too few tracked points for free-fall fit")
    floor_y = float(np.nanpercentile(y_v, 92))
    moving = y_v < floor_y - max(4.0, 0.35 * radius_px)
    idx = np.where(moving)[0]
    if len(idx) < 12:
        idx = np.arange(max(12, int(0.65 * len(t_v))))
    t_fit = t_v[idx] - t_v[idx[0]]
    y_fit = y_v[idx]
    coeff, rmse, r2 = fit_poly(t_fit, y_fit, 2)
    accel_px = 2.0 * float(coeff[0])
    g_hat = abs(accel_px) * m_per_px
    return {"estimate": g_hat, "estimate_name": "g_hidden", "fit_r2": r2, "fit_rmse_px": rmse, "num_fit_points": int(len(t_fit)), "accel_px_s2": accel_px}


def estimate_v1_b(t: np.ndarray, y: np.ndarray, radius_px: float, _m_per_px: float) -> dict[str, Any]:
    y_s = moving_average(y, 5)
    valid = np.isfinite(y_s)
    t_v = t[valid]
    y_v = y_s[valid]
    if len(t_v) < 20:
        raise RuntimeError("too few tracked points for restitution fit")
    floor_y = float(np.nanpercentile(y_v, 96))
    min_sep = max(3, int(0.25 / np.median(np.diff(t_v))))
    peak_indices = local_extrema(y_v, "min", min_sep)
    peaks = [(t_v[i], floor_y - y_v[i]) for i in peak_indices if floor_y - y_v[i] > max(0.8 * radius_px, 5.0)]
    if len(peaks) < 2:
        raise RuntimeError("fewer than two bounce peaks detected")
    ratios = []
    for (_, h0), (_, h1) in zip(peaks, peaks[1:]):
        if h0 > 1e-6 and h1 > 1e-6 and h1 <= h0 * 1.15:
            ratios.append(math.sqrt(max(h1 / h0, 0.0)))
    if not ratios:
        raise RuntimeError("no valid peak-height ratios for restitution")
    e_hat = float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-8)))))
    return {
        "estimate": e_hat,
        "estimate_name": "e_hidden",
        "num_peaks": len(peaks),
        "peak_heights_px": [float(h) for _, h in peaks],
        "peak_times_s": [float(tt) for tt, _ in peaks],
        "fit_r2": float("nan"),
        "fit_rmse_px": float("nan"),
    }


def estimate_v1_c(t: np.ndarray, x: np.ndarray, radius_px: float, m_per_px: float) -> dict[str, Any]:
    x_s = moving_average(x, 5)
    valid = np.isfinite(x_s)
    t_v = t[valid]
    x_v = x_s[valid]
    if len(t_v) < 12:
        raise RuntimeError("too few tracked points for friction fit")
    direction = 1.0 if x_v[-1] >= x_v[0] else -1.0
    progress = direction * (x_v - x_v[0])
    vel = np.gradient(progress, t_v)
    threshold = max(2.0, 0.15 * radius_px)
    stop_idx = len(t_v)
    for i in range(8, len(vel) - 6):
        if np.all(np.abs(vel[i : i + 6]) < threshold):
            stop_idx = i
            break
    fit_end = max(12, stop_idx)
    t_fit = t_v[:fit_end] - t_v[0]
    x_fit = progress[:fit_end]
    coeff, rmse, r2 = fit_poly(t_fit, x_fit, 2)
    accel_px = 2.0 * float(coeff[0])
    mu_hat = abs(accel_px) * m_per_px / 9.81
    return {"estimate": mu_hat, "estimate_name": "mu_hidden", "fit_r2": r2, "fit_rmse_px": rmse, "num_fit_points": int(len(t_fit)), "accel_px_s2": accel_px}


def estimate_v1_d(t: np.ndarray, x: np.ndarray, radius_px: float, _m_per_px: float) -> dict[str, Any]:
    x_s = moving_average(x, 5)
    valid = np.isfinite(x_s)
    t_v = t[valid]
    x_v = x_s[valid]
    if len(t_v) < 20:
        raise RuntimeError("too few tracked points for damping fit")
    center = float(np.median(x_v))
    amp = np.abs(x_v - center)
    min_sep = max(4, int(0.55 / np.median(np.diff(t_v))))
    peaks = local_extrema(amp, "max", min_sep)
    amp_threshold = max(0.35 * radius_px, 3.0)
    peak_pairs = [(t_v[i], amp[i]) for i in peaks if amp[i] > amp_threshold]
    if len(peak_pairs) >= 2:
        pt = np.array([p[0] for p in peak_pairs], dtype=np.float64)
        pa = np.array([p[1] for p in peak_pairs], dtype=np.float64)
        coeff, rmse, r2 = fit_poly(pt - pt[0], np.log(np.maximum(pa, 1e-6)), 1)
        gamma_hat = -float(coeff[0])
        return {
            "estimate": gamma_hat,
            "estimate_name": "gamma_hidden",
            "num_peaks": len(peak_pairs),
            "peak_amplitudes_px": [float(a) for _, a in peak_pairs],
            "peak_times_s": [float(tt) for tt, _ in peak_pairs],
            "fit_r2": r2,
            "fit_rmse_log_amp": rmse,
        }

    n = len(amp)
    first = float(np.nanmedian(amp[: max(3, n // 4)]))
    last = float(np.nanmedian(amp[-max(3, n // 4) :]))
    dt = float(t_v[-max(3, n // 4)] - t_v[max(0, n // 4 - 1)])
    if first <= 1e-6 or last <= 1e-6 or dt <= 0:
        raise RuntimeError("not enough amplitude signal for damping fallback")
    gamma_hat = math.log(first / last) / dt
    return {"estimate": gamma_hat, "estimate_name": "gamma_hidden", "num_peaks": len(peak_pairs), "fit_r2": float("nan"), "fit_rmse_log_amp": float("nan")}


def evaluate_job(job: dict[str, Any], video_path: Path, tracks_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows, fps = track_video(video_path, job, args)
    write_csv(tracks_dir / f"{job['job_id']}.csv", rows)
    found_count = sum(1 for r in rows if r["found"])
    detection_rate = found_count / max(1, len(rows))
    t, x, y, radius = arrays_from_rows(rows)
    m_per_px, radius_px = estimate_scale(radius, float(job.get("ball_radius_m") or args.ball_radius_m))
    if not np.isfinite(m_per_px):
        raise RuntimeError("cannot estimate pixel-to-meter scale")

    experiment = job["experiment"]
    if experiment == "v1_A":
        estimate = estimate_v1_a(t, y, radius_px, m_per_px)
    elif experiment == "v1_B":
        estimate = estimate_v1_b(t, y, radius_px, m_per_px)
    elif experiment == "v1_C":
        estimate = estimate_v1_c(t, x, radius_px, m_per_px)
    elif experiment == "v1_D":
        estimate = estimate_v1_d(t, x, radius_px, m_per_px)
    else:
        raise RuntimeError(f"unsupported experiment: {experiment}")

    target = job.get("target_param_value")
    abs_error = abs(float(estimate["estimate"]) - float(target)) if target is not None else float("nan")
    rel_error = abs_error / abs(float(target)) if target not in (None, 0) else float("nan")
    return {
        "job_id": job["job_id"],
        "experiment": experiment,
        "variant": job.get("variant"),
        "camera": job.get("camera"),
        "video": str(video_path),
        "status": "ok",
        "target_param_name": job.get("target_param_name"),
        "target_param_value": target,
        "estimate_name": estimate.pop("estimate_name"),
        "estimate_value": estimate.pop("estimate"),
        "abs_error": abs_error,
        "rel_error": rel_error,
        "fps": fps,
        "num_frames": len(rows),
        "found_frames": found_count,
        "detection_rate": detection_rate,
        "radius_px_median": radius_px,
        "m_per_px": m_per_px,
        **estimate,
    }


def main() -> None:
    args = parse_args()
    jobs = read_manifest(args.manifest, args.max_jobs)
    args.outdir.mkdir(parents=True, exist_ok=True)
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
            print(f"{job['job_id']}: {row['estimate_name']}={row['estimate_value']:.6g}, target={row['target_param_value']}, rel_error={row['rel_error']:.4g}")
            summary.append(row)
        except Exception as exc:
            print(f"{job['job_id']}: error: {exc}")
            summary.append({"job_id": job["job_id"], "experiment": job.get("experiment"), "variant": job.get("variant"), "camera": job.get("camera"), "status": "error", "error": str(exc)})

    write_csv(args.outdir / "summary.csv", summary)
    write_json(args.outdir / "summary.json", summary)
    print(f"wrote {args.outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()

