#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自由落体批量评测（基于 config parameters）
============================================================

目标：
1. 只评测指定 model_name 在 free_fall 目录下的所有 mp4
2. 用 config json 里的 parameters(g, h0, v0, color, object_name) 作为“真值”
3. 对每个视频输出：
   - trajectory.csv
   - evaluation.json
   - fit_plot_y.png
   - fit_plot_x.png
   - trajectory_plot.png
   - residual_plot.png
   - fitted_overlay.mp4
4. 聚合输出：
   - summary.csv
   - summary.json
   - style_summary.csv
   - ranking.txt

适用目录结构：
outputs/baseline_v1_all_tasks/free_fall/<prompt_style>/<model_name>/*.mp4

说明：
- 脚本关注的是“物理量与运动约束”评估，不做高层语义场景评估。
- 要真正比较 g/h0/v0，建议提供：
  --object-diameter-m 或 --meters-per-pixel
  否则参数会在像素域拟合，只能做形状/趋势评估，不能和 config 中的米制严格对比。
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR_PRESETS = {
    "red": [
        ((0, 80, 60), (12, 255, 255)),
        ((170, 80, 60), (179, 255, 255)),
    ],
    "blue": [
        ((90, 80, 60), (130, 255, 255)),
    ],
    "green": [
        ((35, 60, 50), (90, 255, 255)),
    ],
    "yellow": [
        ((18, 80, 80), (40, 255, 255)),
    ],
    "white": [
        ((0, 0, 160), (179, 70, 255)),
    ],
}


def ensure_odd(k: int) -> int:
    if k <= 0:
        return 0
    return k if k % 2 == 1 else k + 1


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def normalize_color_name(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = text.lower().strip()
    if "red" in s:
        return "red"
    if "blue" in s:
        return "blue"
    if "green" in s:
        return "green"
    if "yellow" in s:
        return "yellow"
    if "white" in s:
        return "white"
    return None


def build_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    ranges = COLOR_PRESETS[color_name]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def postprocess_mask(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
    if open_kernel > 0:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    if close_kernel > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return mask


def detect_object(
    frame_bgr: np.ndarray,
    color_name: str,
    min_area: float,
    open_kernel: int,
    close_kernel: int,
    blur_ksize: int,
    significant_area_ratio: float = 0.15,
) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[float], int]:
    img = frame_bgr.copy()
    blur_ksize = ensure_odd(blur_ksize)
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color_name)
    mask = postprocess_mask(mask, open_kernel, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, 0

    areas = [float(cv2.contourArea(c)) for c in contours]
    order = np.argsort(areas)[::-1]
    contours = [contours[i] for i in order]
    areas = [areas[i] for i in order]

    best = contours[0]
    best_area = areas[0]
    if best_area < min_area:
        return None, None, None, 0

    num_significant = sum(1 for a in areas if a >= max(min_area, significant_area_ratio * best_area))

    M = cv2.moments(best)
    if abs(M["m00"]) < 1e-8:
        return None, None, None, num_significant

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    (_, _), radius = cv2.minEnclosingCircle(best)
    return (cx, cy), float(radius), best_area, num_significant


def interpolate_small_gaps(values: List[Optional[float]], max_gap: int) -> List[Optional[float]]:
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
        end = j
        gap = end - i
        left_ok = start >= 0 and out[start] is not None
        right_ok = end < n and out[end] is not None
        if left_ok and right_ok and gap <= max_gap:
            v0 = out[start]
            v1 = out[end]
            for k in range(1, gap + 1):
                alpha = k / (gap + 1)
                out[start + k] = (1 - alpha) * v0 + alpha * v1
        i = j
    return out


def fit_quadratic(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    coeff = np.polyfit(t, y, deg=2)
    a, b, c = [float(v) for v in coeff]
    y_hat = np.polyval(coeff, t)
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {
        "a": a, "b": b, "c": c,
        "rmse": rmse,
        "r2": r2,
        "y_hat": y_hat.tolist(),
        "residual": resid.tolist(),
    }


def fit_linear(t: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    coeff = np.polyfit(t, x, deg=1)
    m, q = [float(v) for v in coeff]
    x_hat = np.polyval(coeff, t)
    resid = x - x_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "m": m, "q": q,
        "rmse": rmse,
        "x_hat": x_hat.tolist(),
        "residual": resid.tolist(),
    }


def load_config_entries(config_json: Path) -> Dict[str, Dict]:
    data = json.loads(config_json.read_text(encoding="utf-8"))
    out = {}
    for item in data:
        if item.get("task") == "free_fall":
            out[item["id"]] = item
    return out


def match_config_entry(video_stem: str, cfg_index: Dict[str, Dict]) -> Optional[Dict]:
    # 取“最长前缀匹配”，避免后续可能有相似 id
    matches = []
    for cfg_id, item in cfg_index.items():
        if video_stem.startswith(cfg_id):
            matches.append((len(cfg_id), item))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def parse_prompt_style(video_path: Path, input_root: Path, model_name: str) -> str:
    try:
        rel = video_path.relative_to(input_root)
        parts = rel.parts
        if len(parts) >= 3 and parts[1] == model_name:
            return parts[0]
        for i, p in enumerate(parts):
            if p == model_name and i - 1 >= 0:
                return parts[i - 1]
    except Exception:
        pass
    return "unknown"


def find_videos_for_model(input_root: Path, model_name: str) -> List[Path]:
    videos = []
    for p in input_root.rglob("*.mp4"):
        if model_name in p.parts:
            videos.append(p)
    return sorted(videos)


def build_plots(
    outdir: Path,
    t_fit: np.ndarray,
    y_fit_vals: np.ndarray,
    fit_y: Dict[str, float],
    x_fit_vals: np.ndarray,
    fit_x: Dict[str, float],
    unit_name: str,
):
    # y(t)
    t_dense = np.linspace(t_fit.min(), t_fit.max(), 400)
    y_dense = fit_y["a"] * t_dense**2 + fit_y["b"] * t_dense + fit_y["c"]

    plt.figure(figsize=(8, 5))
    plt.scatter(t_fit, y_fit_vals, s=22, label="tracked y(t)")
    plt.plot(t_dense, y_dense, linewidth=2, label="quadratic fit")
    plt.xlabel("time (s)")
    plt.ylabel(f"y ({unit_name}, upward positive)")
    plt.title("Free-fall fit: y(t) = a t^2 + b t + c")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit_plot_y.png", dpi=180)
    plt.close()

    # x(t)
    x_dense = fit_x["m"] * t_dense + fit_x["q"]
    plt.figure(figsize=(8, 5))
    plt.scatter(t_fit, x_fit_vals, s=22, label="tracked x(t)")
    plt.plot(t_dense, x_dense, linewidth=2, label="linear fit")
    plt.xlabel("time (s)")
    plt.ylabel(f"x ({unit_name})")
    plt.title("Horizontal drift check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit_plot_x.png", dpi=180)
    plt.close()

    # trajectory
    plt.figure(figsize=(6, 6))
    plt.scatter(x_fit_vals, y_fit_vals, s=18)
    plt.plot(x_fit_vals, y_fit_vals, linewidth=1)
    plt.xlabel(f"x ({unit_name})")
    plt.ylabel(f"y ({unit_name}, upward positive)")
    plt.title("Tracked trajectory")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outdir / "trajectory_plot.png", dpi=180)
    plt.close()

    # residual
    resid = np.array(fit_y["residual"], dtype=float)
    plt.figure(figsize=(8, 4))
    plt.axhline(0.0, linewidth=1)
    plt.scatter(t_fit, resid, s=22)
    plt.xlabel("time (s)")
    plt.ylabel(f"residual ({unit_name})")
    plt.title("Vertical fit residuals")
    plt.tight_layout()
    plt.savefig(outdir / "residual_plot.png", dpi=180)
    plt.close()


def make_fit_overlay_video(
    video_path: Path,
    outpath: Path,
    fps: float,
    width: int,
    height: int,
    frame_rows: List[Dict],
    ground_y_px: float,
    meters_per_pixel: float,
    x0_px: float,
    t0_fit: float,
    fit_y: Dict[str, float],
    fit_x: Dict[str, float],
    fit_until_frame_idx: Optional[int],
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(outpath), fourcc, fps, (width, height))

    a, b, c = fit_y["a"], fit_y["b"], fit_y["c"]
    m, q = fit_x["m"], fit_x["q"]

    row_map = {r["frame_idx"]: r for r in frame_rows}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = row_map.get(frame_idx)
        vis = frame.copy()
        cv2.line(vis, (0, int(round(ground_y_px))), (width - 1, int(round(ground_y_px))), (0, 255, 255), 2)

        if r is not None:
            # tracked
            if r["cx_px"] is not None and r["cy_px"] is not None:
                cx = int(round(r["cx_px"]))
                cy = int(round(r["cy_px"]))
                cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(vis, "tracked", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # fitted: only pre-contact or before fit_until_frame_idx
            if fit_until_frame_idx is None or frame_idx < fit_until_frame_idx:
                t_global = float(r["time_s"])
                t_fit = t_global - t0_fit
                if t_fit >= 0:
                    x_pred = m * t_fit + q
                    y_pred = a * t_fit**2 + b * t_fit + c
                    cx_fit = int(round(x0_px + x_pred / meters_per_pixel))
                    cy_fit = int(round(ground_y_px - y_pred / meters_per_pixel))
                    if 0 <= cx_fit < width and 0 <= cy_fit < height:
                        cv2.circle(vis, (cx_fit, cy_fit), 5, (255, 0, 0), -1)
                        cv2.putText(vis, "fit", (cx_fit + 8, cy_fit + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

            score_text = f"frame={frame_idx} t={float(r['time_s']):.3f}s"
            cv2.putText(vis, score_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()


def gravity_deduction(g_rel_err: Optional[float]) -> Tuple[float, str]:
    if g_rel_err is None:
        return 0.0, "未提供物理尺度，未比较 g"
    if g_rel_err <= 0.10:
        return 0.0, "g 相对误差 <= 10%"
    if g_rel_err <= 0.20:
        return 5.0, "10% < g 相对误差 <= 20%"
    if g_rel_err <= 0.30:
        return 10.0, "20% < g 相对误差 <= 30%"
    if g_rel_err <= 0.50:
        return 20.0, "30% < g 相对误差 <= 50%"
    return 35.0, "g 相对误差 > 50%"


def v0_deduction(v0_abs_err: Optional[float]) -> Tuple[float, str]:
    if v0_abs_err is None:
        return 0.0, "未提供物理尺度，未比较 v0"
    if v0_abs_err <= 0.20:
        return 0.0, "|v0误差| <= 0.20 m/s"
    if v0_abs_err <= 0.50:
        return 5.0, "0.20 < |v0误差| <= 0.50 m/s"
    if v0_abs_err <= 1.00:
        return 10.0, "0.50 < |v0误差| <= 1.00 m/s"
    if v0_abs_err <= 2.00:
        return 15.0, "1.00 < |v0误差| <= 2.00 m/s"
    return 20.0, "|v0误差| > 2.00 m/s"


def h0_deduction(h0_rel_err: Optional[float]) -> Tuple[float, str]:
    if h0_rel_err is None:
        return 0.0, "未提供物理尺度或地面参考，未比较 h0"
    if h0_rel_err <= 0.10:
        return 0.0, "h0 相对误差 <= 10%"
    if h0_rel_err <= 0.20:
        return 3.0, "10% < h0 相对误差 <= 20%"
    if h0_rel_err <= 0.30:
        return 6.0, "20% < h0 相对误差 <= 30%"
    if h0_rel_err <= 0.50:
        return 10.0, "30% < h0 相对误差 <= 50%"
    return 15.0, "h0 相对误差 > 50%"


def r2_deduction(r2: float) -> Tuple[float, str]:
    if math.isnan(r2):
        return 8.0, "R² 不可用"
    if r2 >= 0.995:
        return 0.0, "R² >= 0.995"
    if r2 >= 0.990:
        return 2.0, "0.990 <= R² < 0.995"
    if r2 >= 0.980:
        return 4.0, "0.980 <= R² < 0.990"
    if r2 >= 0.950:
        return 6.0, "0.950 <= R² < 0.980"
    return 8.0, "R² < 0.950"


def verticality_deduction(verticality_ratio: float) -> Tuple[float, str]:
    if verticality_ratio <= 0.02:
        return 0.0, "水平漂移很小"
    if verticality_ratio <= 0.05:
        return 3.0, "有轻微水平漂移"
    if verticality_ratio <= 0.10:
        return 6.0, "水平漂移明显"
    return 10.0, "水平漂移很大，不符合纯竖直下落"


def monotonic_deduction(monotonic_down_rate: float) -> Tuple[float, str]:
    if monotonic_down_rate >= 0.98:
        return 0.0, "基本单调下落"
    if monotonic_down_rate >= 0.95:
        return 2.0, "少量非单调波动"
    if monotonic_down_rate >= 0.90:
        return 4.0, "非单调波动较明显"
    return 8.0, "明显不满足单调下落"


def tracking_deduction(detection_rate: float, single_object_rate: float, in_frame_rate: float) -> Tuple[float, List[str]]:
    ded = 0.0
    notes = []

    if detection_rate < 0.95:
        if detection_rate >= 0.90:
            ded += 2.0
            notes.append("检测率略低")
        elif detection_rate >= 0.80:
            ded += 4.0
            notes.append("检测率较低")
        else:
            ded += 8.0
            notes.append("检测率很低")

    if single_object_rate < 0.98:
        if single_object_rate >= 0.95:
            ded += 1.0
            notes.append("偶尔出现多个显著前景")
        elif single_object_rate >= 0.90:
            ded += 3.0
            notes.append("多主体干扰较明显")
        else:
            ded += 6.0
            notes.append("经常出现多主体/杂散前景")

    if in_frame_rate < 0.98:
        if in_frame_rate >= 0.95:
            ded += 1.0
            notes.append("偶尔出框")
        elif in_frame_rate >= 0.90:
            ded += 3.0
            notes.append("部分帧出框")
        else:
            ded += 5.0
            notes.append("较严重出框")

    if not notes:
        notes = ["跟踪质量良好"]
    return ded, notes


def penetration_deduction(penetration_depth_px: float, penetration_depth_norm: float) -> Tuple[float, str]:
    if penetration_depth_px <= 2.0:
        return 0.0, "未检测到穿模"
    if penetration_depth_norm <= 0.10:
        return 2.0, "轻微穿模"
    if penetration_depth_norm <= 0.25:
        return 4.0, "中等穿模"
    return 6.0, "严重穿模"


def bounce_deduction(bounce_height_norm: float) -> Tuple[float, str]:
    if bounce_height_norm <= 0.02:
        return 0.0, "未检测到明显反弹"
    if bounce_height_norm <= 0.05:
        return 2.0, "有轻微反弹"
    if bounce_height_norm <= 0.10:
        return 4.0, "反弹明显"
    return 6.0, "严重反弹"


def evaluate_one_video(
    video_path: Path,
    input_root: Path,
    model_name: str,
    cfg_entry: Dict,
    outdir: Path,
    default_color: str,
    min_area: float,
    open_kernel: int,
    close_kernel: int,
    blur_ksize: int,
    max_interp_gap: int,
    ground_y_px_arg: Optional[float],
    meters_per_pixel_arg: Optional[float],
    object_diameter_m: Optional[float],
    start_frame: int,
    end_frame: int,
):
    params = cfg_entry.get("parameters", {})
    g_expected = params.get("g")
    h0_expected = params.get("h0")
    v0_expected = params.get("v0")
    color_name = normalize_color_name(params.get("color")) or default_color

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6 or np.isnan(fps):
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, start_frame)
    end_frame = frame_count - 1 if end_frame < 0 else min(end_frame, frame_count - 1)
    ground_y_px = float(ground_y_px_arg) if ground_y_px_arg is not None else float(height - 1)

    detections = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_idx / fps
        center, radius_px, area_px, num_sig = detect_object(
            frame, color_name, min_area, open_kernel, close_kernel, blur_ksize
        )
        if center is not None:
            cx, cy = center
            detections.append({
                "frame_idx": frame_idx,
                "time_s": time_s,
                "cx_px_raw": cx,
                "cy_px_raw": cy,
                "radius_px": radius_px,
                "area_px": area_px,
                "num_significant_contours": num_sig,
                "found": True,
            })
        else:
            detections.append({
                "frame_idx": frame_idx,
                "time_s": time_s,
                "cx_px_raw": None,
                "cy_px_raw": None,
                "radius_px": None,
                "area_px": None,
                "num_significant_contours": 0,
                "found": False,
            })
    cap.release()

    if not detections:
        raise RuntimeError("没有读取到任何检测帧。")

    cx_list = [d["cx_px_raw"] for d in detections]
    cy_list = [d["cy_px_raw"] for d in detections]
    cx_interp = interpolate_small_gaps(cx_list, max_interp_gap)
    cy_interp = interpolate_small_gaps(cy_list, max_interp_gap)

    radii = [d["radius_px"] for d in detections if d["radius_px"] is not None and d["radius_px"] > 0]
    meters_per_pixel = meters_per_pixel_arg
    if meters_per_pixel is None and object_diameter_m is not None and len(radii) > 0:
        median_diameter_px = float(np.median(np.array(radii) * 2.0))
        meters_per_pixel = float(object_diameter_m) / median_diameter_px

    if meters_per_pixel is None:
        meters_per_pixel = 1.0
        unit_name = "px"
        metric_scale_available = False
    else:
        unit_name = "m"
        metric_scale_available = True

    x0_px = None
    frame_rows = []
    valid_idx = []
    valid_t = []
    valid_x = []
    valid_y = []

    for i, d in enumerate(detections):
        cx = cx_interp[i]
        cy = cy_interp[i]
        if cx is not None and x0_px is None:
            x0_px = cx

        if cx is not None and cy is not None and x0_px is not None:
            x_phys = (cx - x0_px) * meters_per_pixel
            y_phys = (ground_y_px - cy) * meters_per_pixel
            used = True
            valid_idx.append(i)
            valid_t.append(d["time_s"])
            valid_x.append(x_phys)
            valid_y.append(y_phys)
        else:
            x_phys = None
            y_phys = None
            used = False

        row = dict(d)
        row["cx_px"] = cx
        row["cy_px"] = cy
        row["x"] = x_phys
        row["y"] = y_phys
        row["used_for_fit"] = used
        frame_rows.append(row)

    write_csv(outdir / "trajectory.csv", frame_rows)

    if len(valid_t) < 5:
        raise RuntimeError("有效追踪点太少，无法稳定拟合。")

    valid_t = np.array(valid_t, dtype=float)
    valid_x = np.array(valid_x, dtype=float)
    valid_y = np.array(valid_y, dtype=float)

    # 接触检测：圆底触到 ground_y_px
    contact_tol_px = 2.0
    contact_global_i = None
    for i, d in enumerate(frame_rows):
        if d["cx_px"] is None or d["cy_px"] is None or d["radius_px"] is None:
            continue
        bottom_y = d["cy_px"] + d["radius_px"]
        if bottom_y >= ground_y_px - contact_tol_px:
            contact_global_i = i
            break

    if contact_global_i is None:
        fit_mask = np.array([True] * len(valid_t), dtype=bool)
    else:
        fit_mask = np.array([gi < contact_global_i for gi in valid_idx], dtype=bool)
        if fit_mask.sum() < 5:
            fit_mask[:] = True

    t_fit_all = valid_t[fit_mask]
    x_fit_all = valid_x[fit_mask]
    y_fit_all = valid_y[fit_mask]

    t0_fit = float(t_fit_all[0])
    t_fit = t_fit_all - t0_fit

    fit_y = fit_quadratic(t_fit, y_fit_all)
    fit_x = fit_linear(t_fit, x_fit_all)

    g_est = -2.0 * fit_y["a"]
    v0_est = fit_y["b"]
    h0_est = fit_y["c"]

    # 轨迹图与拟合图
    build_plots(outdir, t_fit, y_fit_all, fit_y, x_fit_all, fit_x, unit_name)

    # fitted overlay
    make_fit_overlay_video(
        video_path=video_path,
        outpath=outdir / "fitted_overlay.mp4",
        fps=fps,
        width=width,
        height=height,
        frame_rows=frame_rows,
        ground_y_px=ground_y_px,
        meters_per_pixel=meters_per_pixel,
        x0_px=float(x0_px),
        t0_fit=t0_fit,
        fit_y=fit_y,
        fit_x=fit_x,
        fit_until_frame_idx=(contact_global_i if contact_global_i is not None else None),
    )

    # 约束统计
    total_frames = len(frame_rows)
    detected_frames = sum(int(bool(r["found"])) for r in frame_rows)
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0

    detected_valid = [
        r for r in frame_rows
        if r["found"] and r["cx_px"] is not None and r["cy_px"] is not None and r["radius_px"] is not None
    ]

    if detected_valid:
        single_object_rate = sum(int(r["num_significant_contours"] <= 1) for r in detected_valid) / len(detected_valid)
    else:
        single_object_rate = 0.0

    in_frame_flags = []
    for r in detected_valid:
        ok = (
            r["cx_px"] - r["radius_px"] >= 1.0 and
            r["cx_px"] + r["radius_px"] <= width - 2.0 and
            r["cy_px"] - r["radius_px"] >= 1.0 and
            r["cy_px"] + r["radius_px"] <= height - 2.0
        )
        in_frame_flags.append(int(ok))
    in_frame_rate = sum(in_frame_flags) / len(in_frame_flags) if in_frame_flags else 0.0

    y_span = float(np.max(y_fit_all) - np.min(y_fit_all)) if len(y_fit_all) > 1 else 0.0
    x_span = float(np.max(x_fit_all) - np.min(x_fit_all)) if len(x_fit_all) > 1 else 0.0
    verticality_ratio = x_span / max(y_span, 1e-8)

    dy = np.diff(y_fit_all)
    downward_tol = 0.01 * max(y_span, 1e-8)
    monotonic_down_rate = float(np.mean(dy <= downward_tol)) if len(dy) > 0 else 0.0

    penetration_depth_px = 0.0
    for r in detected_valid:
        penetration = (r["cy_px"] + r["radius_px"]) - ground_y_px
        if penetration > penetration_depth_px:
            penetration_depth_px = penetration
    median_radius_px = float(np.median([r["radius_px"] for r in detected_valid])) if detected_valid else 10.0
    penetration_depth_norm = penetration_depth_px / max(median_radius_px, 1e-6)

    bounce_height = 0.0
    if contact_global_i is not None:
        post_y = []
        for gi, yv in zip(valid_idx, valid_y):
            if gi >= contact_global_i:
                post_y.append(yv)
        if len(post_y) >= 2:
            y_contact = post_y[0]
            bounce_height = max(post_y) - y_contact
    bounce_height_norm = bounce_height / max(y_span, 1e-8) if y_span > 1e-8 else 0.0

    # 参数误差
    g_rel_err = None
    v0_abs_err = None
    h0_rel_err = None

    if metric_scale_available and g_expected is not None and abs(g_expected) > 1e-8:
        g_rel_err = abs(g_est - float(g_expected)) / abs(float(g_expected))
    if metric_scale_available and v0_expected is not None:
        v0_abs_err = abs(v0_est - float(v0_expected))
    if metric_scale_available and ground_y_px_arg is not None and h0_expected is not None and abs(h0_expected) > 1e-8:
        h0_rel_err = abs(h0_est - float(h0_expected)) / abs(float(h0_expected))

    # 打分：从100分开始扣分，物理量占主导
    deductions = {}

    d, reason = gravity_deduction(g_rel_err)
    deductions["gravity"] = {"points": d, "reason": reason}

    d, reason = v0_deduction(v0_abs_err)
    deductions["initial_velocity"] = {"points": d, "reason": reason}

    d, reason = h0_deduction(h0_rel_err)
    deductions["initial_height"] = {"points": d, "reason": reason}

    d, reason = r2_deduction(fit_y["r2"])
    deductions["quadratic_fit"] = {"points": d, "reason": reason}

    d, reason = verticality_deduction(verticality_ratio)
    deductions["vertical_motion_constraint"] = {"points": d, "reason": reason}

    d, reason = monotonic_deduction(monotonic_down_rate)
    deductions["monotonic_downward_constraint"] = {"points": d, "reason": reason}

    d, reasons = tracking_deduction(detection_rate, single_object_rate, in_frame_rate)
    deductions["tracking_and_single_object_constraint"] = {"points": d, "reason": "；".join(reasons)}

    d, reason = penetration_deduction(penetration_depth_px, penetration_depth_norm)
    deductions["no_penetration_constraint"] = {"points": d, "reason": reason}

    d, reason = bounce_deduction(bounce_height_norm)
    deductions["no_bounce_constraint"] = {"points": d, "reason": reason}

    total_deduction = sum(item["points"] for item in deductions.values())
    final_score = max(0.0, 100.0 - total_deduction)

    # 硬性规范
    hard_constraints = {
        "single_object_rate_ge_0.90": single_object_rate >= 0.90,
        "detection_rate_ge_0.90": detection_rate >= 0.90,
        "in_frame_rate_ge_0.95": in_frame_rate >= 0.95,
        "verticality_ratio_le_0.10": verticality_ratio <= 0.10,
        "monotonic_down_rate_ge_0.90": monotonic_down_rate >= 0.90,
        "quadratic_r2_ge_0.98": fit_y["r2"] >= 0.98 if not math.isnan(fit_y["r2"]) else False,
        "no_penetration": penetration_depth_px <= 2.0,
        "no_large_bounce": bounce_height_norm <= 0.05,
    }
    if g_rel_err is not None:
        hard_constraints["g_relative_error_le_0.30"] = g_rel_err <= 0.30
    if v0_abs_err is not None:
        hard_constraints["v0_abs_error_le_1.0"] = v0_abs_err <= 1.0
    if h0_rel_err is not None:
        hard_constraints["h0_relative_error_le_0.30"] = h0_rel_err <= 0.30

    strict_pass = all(hard_constraints.values())

    prompt_style = parse_prompt_style(video_path, input_root, model_name)

    evaluation = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "stem": video_path.stem,
        "prompt_style": prompt_style,
        "model_name": model_name,
        "matched_config_id": cfg_entry.get("id"),
        "expected_parameters_from_config": {
            "g": g_expected,
            "h0": h0_expected,
            "v0": v0_expected,
            "color": params.get("color"),
            "object_name": params.get("object_name"),
        },
        "metric_scale_available": metric_scale_available,
        "meters_per_pixel": meters_per_pixel if metric_scale_available else None,
        "unit_name": unit_name,
        "fit": {
            "quadratic_y": {
                "a": fit_y["a"],
                "b": fit_y["b"],
                "c": fit_y["c"],
                "rmse": fit_y["rmse"],
                "r2": fit_y["r2"],
            },
            "linear_x": {
                "m": fit_x["m"],
                "q": fit_x["q"],
                "rmse": fit_x["rmse"],
            },
            "recovered_parameters": {
                "g": g_est,
                "v0": v0_est,
                "h0": h0_est,
            },
        },
        "parameter_errors": {
            "g_relative_error": g_rel_err,
            "v0_absolute_error": v0_abs_err,
            "h0_relative_error": h0_rel_err,
        },
        "motion_metrics": {
            "detection_rate": detection_rate,
            "single_object_rate": single_object_rate,
            "in_frame_rate": in_frame_rate,
            "verticality_ratio": verticality_ratio,
            "monotonic_down_rate": monotonic_down_rate,
            "x_span": x_span,
            "y_span": y_span,
            "penetration_depth_px": penetration_depth_px,
            "penetration_depth_norm": penetration_depth_norm,
            "bounce_height": bounce_height,
            "bounce_height_norm": bounce_height_norm,
        },
        "deductions": deductions,
        "total_deduction": total_deduction,
        "final_score_100": final_score,
        "hard_constraints": hard_constraints,
        "strict_pass": strict_pass,
        "scoring_note": "以100分为满分，优先依据 g/h0/v0 的拟合误差扣分，再依据不穿模、无反弹、纯竖直下落、二次拟合优度、单主体与跟踪质量等硬性规范扣分。",
    }

    save_json(outdir / "evaluation.json", evaluation)
    return evaluation


def summarize_by_style(rows: List[Dict]) -> List[Dict]:
    groups = {}
    for r in rows:
        groups.setdefault(r["prompt_style"], []).append(r)

    out = []
    for style, lst in sorted(groups.items()):
        def mean(key):
            vals = [x.get(key) for x in lst]
            vals = [v for v in vals if isinstance(v, (int, float))]
            return sum(vals) / len(vals) if vals else None

        out.append({
            "prompt_style": style,
            "count": len(lst),
            "mean_final_score_100": mean("final_score_100"),
            "mean_detection_rate": mean("detection_rate"),
            "mean_quadratic_r2": mean("quadratic_r2"),
            "mean_g_relative_error": mean("g_relative_error"),
            "strict_pass_rate": sum(int(bool(x["strict_pass"])) for x in lst) / len(lst) if lst else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="按 config parameters 批量评测自由落体视频")
    ap.add_argument("--input-root", type=str, required=True, help="如 outputs/baseline_v1_all_tasks/free_fall")
    ap.add_argument("--model-name", type=str, required=True, help="如 wan2.2_ti2v_5b")
    ap.add_argument("--config-json", type=str, required=True, help="如 configs/baseline_v1_jobs.json")
    ap.add_argument("--eval-outdir", type=str, required=True, help="评测输出目录")
    ap.add_argument("--default-color", type=str, default="red", choices=list(COLOR_PRESETS.keys()))
    ap.add_argument("--min-area", type=float, default=80.0)
    ap.add_argument("--open-kernel", type=int, default=5)
    ap.add_argument("--close-kernel", type=int, default=7)
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--max-interp-gap", type=int, default=5)
    ap.add_argument("--ground-y-px", type=float, default=None, help="若不给，默认图像底边")
    ap.add_argument("--meters-per-pixel", type=float, default=None, help="若给，则直接使用该比例尺")
    ap.add_argument("--object-diameter-m", type=float, default=None, help="若给，则通过检测球直径自动估计尺度")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1)
    args = ap.parse_args()

    input_root = Path(args.input_root)
    eval_outdir = Path(args.eval_outdir)
    eval_outdir.mkdir(parents=True, exist_ok=True)

    cfg_index = load_config_entries(Path(args.config_json))
    videos = find_videos_for_model(input_root, args.model_name)
    if not videos:
        raise RuntimeError(f"在 {input_root} 下没有找到 model={args.model_name} 的 mp4")

    summary_rows = []
    errors = []

    for idx, video in enumerate(videos, 1):
        try:
            cfg_entry = match_config_entry(video.stem, cfg_index)
            if cfg_entry is None:
                raise RuntimeError(f"未在 config 中找到与视频名匹配的 free_fall 条目: {video.stem}")

            rel = video.relative_to(input_root)
            per_video_outdir = eval_outdir / rel.with_suffix("")
            per_video_outdir.mkdir(parents=True, exist_ok=True)

            result = evaluate_one_video(
                video_path=video,
                input_root=input_root,
                model_name=args.model_name,
                cfg_entry=cfg_entry,
                outdir=per_video_outdir,
                default_color=args.default_color,
                min_area=args.min_area,
                open_kernel=args.open_kernel,
                close_kernel=args.close_kernel,
                blur_ksize=args.blur_ksize,
                max_interp_gap=args.max_interp_gap,
                ground_y_px_arg=args.ground_y_px,
                meters_per_pixel_arg=args.meters_per_pixel,
                object_diameter_m=args.object_diameter_m,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )

            row = {
                "video_path": result["video_path"],
                "video_name": result["video_name"],
                "prompt_style": result["prompt_style"],
                "matched_config_id": result["matched_config_id"],
                "model_name": result["model_name"],
                "final_score_100": round(result["final_score_100"], 4),
                "strict_pass": result["strict_pass"],
                "g_expected": result["expected_parameters_from_config"]["g"],
                "h0_expected": result["expected_parameters_from_config"]["h0"],
                "v0_expected": result["expected_parameters_from_config"]["v0"],
                "g_est": round(result["fit"]["recovered_parameters"]["g"], 6),
                "h0_est": round(result["fit"]["recovered_parameters"]["h0"], 6),
                "v0_est": round(result["fit"]["recovered_parameters"]["v0"], 6),
                "g_relative_error": result["parameter_errors"]["g_relative_error"],
                "h0_relative_error": result["parameter_errors"]["h0_relative_error"],
                "v0_absolute_error": result["parameter_errors"]["v0_absolute_error"],
                "quadratic_r2": result["fit"]["quadratic_y"]["r2"],
                "detection_rate": result["motion_metrics"]["detection_rate"],
                "single_object_rate": result["motion_metrics"]["single_object_rate"],
                "in_frame_rate": result["motion_metrics"]["in_frame_rate"],
                "verticality_ratio": result["motion_metrics"]["verticality_ratio"],
                "monotonic_down_rate": result["motion_metrics"]["monotonic_down_rate"],
                "penetration_depth_px": result["motion_metrics"]["penetration_depth_px"],
                "bounce_height_norm": result["motion_metrics"]["bounce_height_norm"],
                "total_deduction": result["total_deduction"],
            }
            summary_rows.append(row)
            print(f"[{idx}/{len(videos)}] OK  score={row['final_score_100']:.2f} strict={row['strict_pass']}  {video.name}")

        except Exception as e:
            errors.append({"video_path": str(video), "error": f"{type(e).__name__}: {e}"})
            print(f"[{idx}/{len(videos)}] ERR {video.name} -> {type(e).__name__}: {e}")

    if summary_rows:
        summary_rows = sorted(summary_rows, key=lambda x: x["final_score_100"], reverse=True)
        write_csv(eval_outdir / "summary.csv", summary_rows)
        save_json(eval_outdir / "summary.json", {"results": summary_rows, "errors": errors})

        style_rows = summarize_by_style(summary_rows)
        write_csv(eval_outdir / "style_summary.csv", style_rows)

        with (eval_outdir / "ranking.txt").open("w", encoding="utf-8") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Input root: {input_root}\n")
            f.write(f"Config: {args.config_json}\n")
            f.write(f"Total ok: {len(summary_rows)}\n")
            f.write(f"Total err: {len(errors)}\n\n")
            for i, r in enumerate(summary_rows, 1):
                f.write(
                    f"{i:03d}. score={r['final_score_100']:.2f} strict={r['strict_pass']} "
                    f"style={r['prompt_style']} id={r['matched_config_id']} file={Path(r['video_name']).name}\n"
                )

        mean_score = sum(r["final_score_100"] for r in summary_rows) / len(summary_rows)
        strict_rate = sum(int(bool(r["strict_pass"])) for r in summary_rows) / len(summary_rows)
        print("=" * 72)
        print("批量评测完成")
        print(f"模型: {args.model_name}")
        print(f"成功评测视频数: {len(summary_rows)}")
        print(f"失败视频数: {len(errors)}")
        print(f"平均得分: {mean_score:.2f}")
        print(f"严格合规率: {strict_rate:.2%}")
        print(f"结果目录: {eval_outdir}")
        print("=" * 72)
    else:
        save_json(eval_outdir / "summary.json", {"results": [], "errors": errors})
        print("没有成功评测任何视频。")


if __name__ == "__main__":
    main()
