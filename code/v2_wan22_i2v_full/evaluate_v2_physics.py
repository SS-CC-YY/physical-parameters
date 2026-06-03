#!/usr/bin/env python3
"""Estimate V2 hidden physics parameters from generated videos.

This is a first-pass proxy evaluator. It keeps the same output contract as the
V1 evaluator while using simple image-plane trajectory fits for V2 tasks.
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
    parser.add_argument("--ball-radius-m", type=float, default=0.24)
    parser.add_argument("--block-half-height-m", type=float, default=0.24)
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


def build_mask(frame: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_target(frame: np.ndarray, color: str, min_area: float) -> dict[str, float] | None:
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
    valid = np.where(np.isfinite(out))[0]
    if len(valid) < 2:
        return out
    for i in range(len(out)):
        if np.isfinite(out[i]):
            continue
        left = valid[valid < i]
        right = valid[valid > i]
        if len(left) == 0 or len(right) == 0:
            continue
        l = int(left[-1])
        r = int(right[0])
        if r - l - 1 <= max_gap:
            alpha = (i - l) / (r - l)
            out[i] = (1 - alpha) * out[l] + alpha * out[r]
    return out


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")


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


def estimate_scale(radius_px: np.ndarray, length_m: float) -> tuple[float, float]:
    valid = radius_px[np.isfinite(radius_px) & (radius_px > 0)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    median_radius = float(np.median(valid))
    return length_m / median_radius, median_radius


def extrema(values: np.ndarray, mode: str, min_sep: int) -> list[int]:
    out: list[int] = []
    for i in range(1, len(values) - 1):
        if not np.isfinite(values[i - 1 : i + 2]).all():
            continue
        ok = values[i] <= values[i - 1] and values[i] <= values[i + 1] if mode == "min" else values[i] >= values[i - 1] and values[i] >= values[i + 1]
        if not ok:
            continue
        if out and i - out[-1] < min_sep:
            better = values[i] < values[out[-1]] if mode == "min" else values[i] > values[out[-1]]
            if better:
                out[-1] = i
            continue
        out.append(i)
    return out


def estimate_gravity(t: np.ndarray, y: np.ndarray, radius_px: float, m_per_px: float) -> dict[str, Any]:
    y_s = smooth(y, 5)
    valid = np.isfinite(y_s)
    t_v = t[valid]
    y_v = y_s[valid]
    if len(t_v) < 14:
        raise RuntimeError("too few tracked points for gravity fit")
    floor_y = float(np.nanpercentile(y_v, 93))
    moving = y_v < floor_y - max(4.0, 0.35 * radius_px)
    idx = np.where(moving)[0]
    if len(idx) < 14:
        idx = np.arange(min(len(t_v), max(14, int(0.55 * len(t_v)))))
    t_fit = t_v[idx] - t_v[idx[0]]
    y_fit = y_v[idx]
    coeff, rmse, r2 = fit_poly(t_fit, y_fit, 2)
    accel_px = 2.0 * float(coeff[0])
    return {"estimate": abs(accel_px) * m_per_px, "estimate_name": "g_hidden", "fit_r2": r2, "fit_rmse_px": rmse, "num_fit_points": int(len(t_fit)), "accel_px_s2": accel_px}


def estimate_restitution(t: np.ndarray, y: np.ndarray, radius_px: float) -> dict[str, Any]:
    y_s = smooth(y, 5)
    valid = np.isfinite(y_s)
    t_v = t[valid]
    y_v = y_s[valid]
    if len(t_v) < 20:
        raise RuntimeError("too few tracked points for restitution fit")
    floor_y = float(np.nanpercentile(y_v, 96))
    dt = float(np.median(np.diff(t_v))) if len(t_v) > 2 else 1 / 16
    peak_idx = extrema(y_v, "min", max(3, int(0.22 / max(dt, 1e-6))))
    peaks = [(t_v[i], floor_y - y_v[i]) for i in peak_idx if floor_y - y_v[i] > max(0.7 * radius_px, 4.0)]
    if len(peaks) < 2:
        raise RuntimeError("fewer than two peaks detected")
    ratios = []
    for (_, h0), (_, h1) in zip(peaks, peaks[1:]):
        if h0 > 1e-6 and h1 > 1e-6 and h1 <= 1.2 * h0:
            ratios.append(math.sqrt(max(h1 / h0, 1e-8)))
    if not ratios:
        raise RuntimeError("no valid restitution ratios")
    return {"estimate": float(np.exp(np.mean(np.log(ratios)))), "estimate_name": "e_hidden", "num_peaks": len(peaks), "peak_heights_px": [float(h) for _, h in peaks], "peak_times_s": [float(tt) for tt, _ in peaks], "fit_r2": float("nan"), "fit_rmse_px": float("nan")}


def estimate_friction(t: np.ndarray, x: np.ndarray, radius_px: float, m_per_px: float) -> dict[str, Any]:
    x_s = smooth(x, 5)
    valid = np.isfinite(x_s)
    t_v = t[valid]
    x_v = x_s[valid]
    if len(t_v) < 16:
        raise RuntimeError("too few tracked points for friction fit")
    direction = 1.0 if x_v[-1] >= x_v[0] else -1.0
    progress = direction * (x_v - x_v[0])
    velocity = np.gradient(progress, t_v)
    threshold = max(2.0, 0.12 * radius_px)
    stop_idx = len(t_v)
    for i in range(10, len(velocity) - 8):
        if np.all(np.abs(velocity[i : i + 8]) < threshold):
            stop_idx = i
            break
    start = min(6, max(0, len(t_v) // 10))
    end = max(start + 14, stop_idx)
    t_fit = t_v[start:end] - t_v[start]
    x_fit = progress[start:end]
    coeff, rmse, r2 = fit_poly(t_fit, x_fit, 2)
    accel_px = 2.0 * float(coeff[0])
    return {"estimate": abs(accel_px) * m_per_px / 9.81, "estimate_name": "mu_hidden", "fit_r2": r2, "fit_rmse_px": rmse, "num_fit_points": int(len(t_fit)), "accel_px_s2": accel_px}


def estimate_damping(t: np.ndarray, x: np.ndarray, radius_px: float) -> dict[str, Any]:
    x_s = smooth(x, 5)
    valid = np.isfinite(x_s)
    t_v = t[valid]
    x_v = x_s[valid]
    if len(t_v) < 20:
        raise RuntimeError("too few tracked points for damping fit")
    center = float(np.median(x_v))
    amp = np.abs(x_v - center)
    dt = float(np.median(np.diff(t_v))) if len(t_v) > 2 else 1 / 16
    peaks = extrema(amp, "max", max(4, int(0.45 / max(dt, 1e-6))))
    pairs = [(t_v[i], amp[i]) for i in peaks if amp[i] > max(0.25 * radius_px, 3.0)]
    if len(pairs) >= 2:
        pt = np.array([p[0] for p in pairs], dtype=np.float64)
        pa = np.array([p[1] for p in pairs], dtype=np.float64)
        coeff, rmse, r2 = fit_poly(pt - pt[0], np.log(np.maximum(pa, 1e-6)), 1)
        return {"estimate": -float(coeff[0]), "estimate_name": "gamma_hidden", "num_peaks": len(pairs), "fit_r2": r2, "fit_rmse_log_amp": rmse, "peak_amplitudes_px": [float(a) for _, a in pairs]}
    n = len(amp)
    first = float(np.nanmedian(amp[: max(3, n // 4)]))
    last = float(np.nanmedian(amp[-max(3, n // 4) :]))
    dt_window = float(t_v[-max(3, n // 4)] - t_v[max(0, n // 4 - 1)])
    if first <= 1e-6 or last <= 1e-6 or dt_window <= 0:
        raise RuntimeError("not enough amplitude signal for damping fallback")
    return {"estimate": math.log(first / last) / dt_window, "estimate_name": "gamma_hidden", "num_peaks": len(pairs), "fit_r2": float("nan"), "fit_rmse_log_amp": float("nan")}


def evaluate_job(job: dict[str, Any], video_path: Path, tracks_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows, fps = track_video(video_path, job, args)
    write_csv(tracks_dir / f"{job['job_id']}.csv", rows)
    found_count = sum(1 for row in rows if row["found"])
    t, x, y, radius = arrays_from_rows(rows)
    calibration_m = float(job.get("block_half_height_m") or args.block_half_height_m) if job.get("track_object") == "block" else float(job.get("ball_radius_m") or args.ball_radius_m)
    m_per_px, radius_px = estimate_scale(radius, calibration_m)
    if not np.isfinite(m_per_px):
        raise RuntimeError("cannot estimate pixel-to-meter scale")

    experiment = job["experiment"]
    if experiment == "v2_A":
        estimate = estimate_gravity(t, y, radius_px, m_per_px)
    elif experiment in {"v2_B", "v2_E"}:
        estimate = estimate_restitution(t, y, radius_px)
    elif experiment == "v2_C":
        estimate = estimate_friction(t, x, radius_px, m_per_px)
    elif experiment in {"v2_D", "v2_F"}:
        estimate = estimate_damping(t, x, radius_px)
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
        "detection_rate": found_count / max(1, len(rows)),
        "radius_px_median": radius_px,
        "m_per_px": m_per_px,
        "proxy_note": "V2 first-pass image-plane proxy; upgrade C/E/F with geometry-aware fitting for final metrics.",
        **estimate,
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

