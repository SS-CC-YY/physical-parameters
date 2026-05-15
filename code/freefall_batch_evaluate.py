#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评测：自由落体视频 -> 目标追踪 -> 物理拟合 -> 约束检查 -> 评分

适用目录结构：
    outputs/baseline_v1_all_tasks/free_fall/<prompt_style>/<model_name>/*.mp4

核心能力：
1. 只扫描指定 model_name 的视频
2. 逐视频执行：
   - HSV 颜色分割 + 最大轮廓质心追踪
   - 小缺口线性插值
   - 预接触阶段自由落体拟合 y(t)=a t^2+b t+c
   - 恢复参数 g=-2a, v0=b, h0=c
   - 强约束检查：
       * 单主体
       * 全程在画面内
       * 主要竖直下落
       * 单调下落
       * 不穿模
       * 不反弹
       * 若有物理尺度，则和 prompt 里的 g/h0/v0 做一致性比较
3. 输出：
   - 每个视频一个 evaluation.json
   - summary.csv / summary.json
   - style_summary.csv
   - ranking.txt

限制：
- “是否严格按照 prompt 生成”这里做的是物理/几何层面的自动检查，
  不判断高层语义，比如“是不是实验室背景”。
- 若不给 ground_y_px，则默认图像底边为参考地面。
- 若不给 meters_per_pixel 或 object_diameter_m，则单位为像素域，
  g/v0/h0 只做相对拟合，不适合和 prompt 的米制数值严格比较。
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


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


@dataclass
class Detection:
    frame_idx: int
    time_s: float
    cx: Optional[float]
    cy: Optional[float]
    radius_px: Optional[float]
    area_px: Optional[float]
    num_significant_contours: int
    found: bool


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def ensure_odd(k: int) -> int:
    if k <= 0:
        return 0
    return k if k % 2 == 1 else k + 1


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


def parse_float_token(token: str) -> Optional[float]:
    """
    解析文件名中的数字 token:
    9p8 -> 9.8
    5 -> 5.0
    m1p5 -> -1.5
    neg1p5 -> -1.5
    """
    if token is None:
        return None
    s = token.strip()
    if not s:
        return None
    neg = False
    if s.startswith("neg"):
        neg = True
        s = s[3:]
    elif s.startswith("m"):
        neg = True
        s = s[1:]
    s = s.replace("p", ".")
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None


def parse_expected_from_stem(stem: str) -> Dict[str, Optional[float]]:
    """
    解析:
    freefall_g9p8_h5_v0_0_short_raw_wan2.2_ti2v_5b_s42
    """
    result = {"g_expected": None, "h0_expected": None, "v0_expected": None}
    m = re.search(r"freefall_g([A-Za-z0-9p]+)_h([A-Za-z0-9p]+)_v0_([A-Za-z0-9p]+)", stem)
    if m:
        result["g_expected"] = parse_float_token(m.group(1))
        result["h0_expected"] = parse_float_token(m.group(2))
        result["v0_expected"] = parse_float_token(m.group(3))
    return result


def parse_style_from_path(video_path: Path, input_root: Path, model_name: str) -> str:
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
    a, b, c = [float(x) for x in coeff]
    y_hat = np.polyval(coeff, t)
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"a": a, "b": b, "c": c, "rmse": rmse, "r2": r2}


def fit_linear(t: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    coeff = np.polyfit(t, x, deg=1)
    m, q = [float(v) for v in coeff]
    x_hat = np.polyval(coeff, t)
    resid = x - x_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {"m": m, "q": q, "rmse": rmse}


def evaluate_one_video(
    video_path: Path,
    input_root: Path,
    model_name: str,
    color_name: str,
    outdir: Path,
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
    save_annotated_video: bool,
) -> Dict:
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

    writer = None
    if save_annotated_video:
        ann_path = outdir / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(ann_path), fourcc, fps, (width, height))

    detections: List[Detection] = []

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
            detections.append(Detection(frame_idx, time_s, cx, cy, radius_px, area_px, num_sig, True))
        else:
            detections.append(Detection(frame_idx, time_s, None, None, None, None, 0, False))

        if writer is not None:
            vis = frame.copy()
            cv2.line(vis, (0, int(round(ground_y_px))), (width - 1, int(round(ground_y_px))), (0, 255, 255), 2)
            if center is not None:
                cx_i, cy_i = int(round(center[0])), int(round(center[1]))
                rr = int(round(radius_px or 5))
                cv2.circle(vis, (cx_i, cy_i), max(rr, 3), (0, 255, 0), 2)
                cv2.circle(vis, (cx_i, cy_i), 3, (0, 0, 255), -1)
                cv2.putText(vis, f"{frame_idx} {time_s:.3f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                cv2.putText(vis, f"{frame_idx} {time_s:.3f}s LOST", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()

    if not detections:
        raise RuntimeError("没有有效帧。")

    cx_list = [d.cx for d in detections]
    cy_list = [d.cy for d in detections]
    cx_interp = interpolate_small_gaps(cx_list, max_interp_gap)
    cy_interp = interpolate_small_gaps(cy_list, max_interp_gap)

    radii = [d.radius_px for d in detections if d.radius_px is not None and d.radius_px > 0]
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
            valid_idx.append(i)
            valid_t.append(d.time_s)
            valid_x.append(x_phys)
            valid_y.append(y_phys)
            used = True
        else:
            x_phys = None
            y_phys = None
            used = False

        frame_rows.append({
            "frame_idx": d.frame_idx,
            "time_s": round(d.time_s, 6),
            "cx_px_raw": None if d.cx is None else round(d.cx, 4),
            "cy_px_raw": None if d.cy is None else round(d.cy, 4),
            "cx_px": None if cx is None else round(cx, 4),
            "cy_px": None if cy is None else round(cy, 4),
            "radius_px": None if d.radius_px is None else round(d.radius_px, 4),
            "area_px": None if d.area_px is None else round(d.area_px, 4),
            "num_significant_contours": d.num_significant_contours,
            "x": None if x_phys is None else round(x_phys, 8),
            "y": None if y_phys is None else round(y_phys, 8),
            "used_for_fit": used,
        })

    write_csv(outdir / "trajectory.csv", frame_rows)

    if len(valid_t) < 5:
        raise RuntimeError("有效追踪点太少，无法稳定拟合。")

    valid_t = np.array(valid_t, dtype=float)
    valid_x = np.array(valid_x, dtype=float)
    valid_y = np.array(valid_y, dtype=float)

    # 接触检测：圆底部触到 ground_y_px
    contact_tol_px = 2.0
    contact_global_i = None
    for i, d in enumerate(detections):
        cx = cx_interp[i]
        cy = cy_interp[i]
        if cx is None or cy is None or d.radius_px is None:
            continue
        bottom_y = cy + d.radius_px
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

    t0 = float(t_fit_all[0])
    t_fit = t_fit_all - t0

    fit_y = fit_quadratic(t_fit, y_fit_all)
    fit_x = fit_linear(t_fit, x_fit_all)

    g_est = -2.0 * fit_y["a"]
    v0_est = fit_y["b"]
    h0_est = fit_y["c"]

    total_frames = len(detections)
    detected_frames = sum(int(d.found) for d in detections)
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0

    detected_valid = [
        d for d in detections
        if d.found and d.cx is not None and d.cy is not None and d.radius_px is not None
    ]

    if len(detected_valid) > 0:
        single_object_rate = sum(int(d.num_significant_contours <= 1) for d in detected_valid) / len(detected_valid)
    else:
        single_object_rate = 0.0

    in_frame_flags = []
    for d in detected_valid:
        ok = (
            d.cx - d.radius_px >= 1.0 and
            d.cx + d.radius_px <= width - 2.0 and
            d.cy - d.radius_px >= 1.0 and
            d.cy + d.radius_px <= height - 2.0
        )
        in_frame_flags.append(int(ok))
    in_frame_rate = sum(in_frame_flags) / len(in_frame_flags) if len(in_frame_flags) > 0 else 0.0

    y_span = float(np.max(y_fit_all) - np.min(y_fit_all)) if len(y_fit_all) > 1 else 0.0
    x_span = float(np.max(x_fit_all) - np.min(x_fit_all)) if len(x_fit_all) > 1 else 0.0
    verticality_ratio = x_span / max(y_span, 1e-8)

    dy = np.diff(y_fit_all)
    downward_tol = 0.01 * max(y_span, 1e-8)
    monotonic_down_rate = float(np.mean(dy <= downward_tol)) if len(dy) > 0 else 0.0

    penetration_depth_px = 0.0
    for d in detected_valid:
        penetration = (d.cy + d.radius_px) - ground_y_px
        if penetration > penetration_depth_px:
            penetration_depth_px = penetration
    median_radius_px = float(np.median([d.radius_px for d in detected_valid])) if detected_valid else 10.0
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

    expected = parse_expected_from_stem(video_path.stem)
    g_expected = expected["g_expected"]
    h0_expected = expected["h0_expected"]
    v0_expected = expected["v0_expected"]

    g_rel_err = None
    v0_abs_err = None
    h0_rel_err = None
    parameter_components = []

    if metric_scale_available and g_expected is not None and g_expected > 1e-8:
        g_rel_err = abs(g_est - g_expected) / abs(g_expected)
        g_score = clip01(1.0 - g_rel_err / 0.5)
        parameter_components.append(g_score)

    if metric_scale_available and v0_expected is not None:
        v0_abs_err = abs(v0_est - v0_expected)
        v0_tol = max(0.5, 0.3 * abs(v0_expected) + 0.2)
        v0_score = clip01(1.0 - v0_abs_err / v0_tol)
        parameter_components.append(v0_score)

    if metric_scale_available and ground_y_px_arg is not None and h0_expected is not None and abs(h0_expected) > 1e-8:
        h0_rel_err = abs(h0_est - h0_expected) / abs(h0_expected)
        h0_score = clip01(1.0 - h0_rel_err / 0.5)
        parameter_components.append(h0_score)

    parameter_consistency_score = float(np.mean(parameter_components)) if parameter_components else None

    tracking_score = 0.5 * detection_rate + 0.25 * single_object_rate + 0.25 * in_frame_rate
    vertical_motion_score = 0.6 * clip01(1.0 - verticality_ratio / 0.15) + 0.4 * clip01(1.0 - fit_x["rmse"] / max(0.05 * max(y_span, 1e-8), 1e-8))
    downward_motion_score = 0.5 * monotonic_down_rate + 0.5 * (1.0 if g_est > 0 else 0.0)
    quadratic_fit_score = clip01((fit_y["r2"] - 0.90) / 0.09)
    no_penetration_score = 1.0 if penetration_depth_px <= 2.0 else clip01(1.0 - penetration_depth_norm / 0.5)
    no_bounce_score = 1.0 if bounce_height_norm <= 0.02 else clip01(1.0 - bounce_height_norm / 0.20)

    component_scores = {
        "tracking_score": tracking_score,
        "vertical_motion_score": vertical_motion_score,
        "downward_motion_score": downward_motion_score,
        "quadratic_fit_score": quadratic_fit_score,
        "no_penetration_score": no_penetration_score,
        "no_bounce_score": no_bounce_score,
    }
    component_weights = {
        "tracking_score": 0.20,
        "vertical_motion_score": 0.20,
        "downward_motion_score": 0.10,
        "quadratic_fit_score": 0.20,
        "no_penetration_score": 0.15,
        "no_bounce_score": 0.10,
    }
    if parameter_consistency_score is not None:
        component_scores["parameter_consistency_score"] = parameter_consistency_score
        component_weights["parameter_consistency_score"] = 0.05

    total_weight = sum(component_weights.values())
    overall_score_0_1 = sum(component_scores[k] * component_weights[k] for k in component_scores) / total_weight
    overall_score = 100.0 * overall_score_0_1

    strict_checks = {
        "detection_rate_ge_0.90": detection_rate >= 0.90,
        "single_object_rate_ge_0.90": single_object_rate >= 0.90,
        "in_frame_rate_ge_0.95": in_frame_rate >= 0.95,
        "verticality_ratio_le_0.10": verticality_ratio <= 0.10,
        "monotonic_down_rate_ge_0.90": monotonic_down_rate >= 0.90,
        "quadratic_r2_ge_0.98": fit_y["r2"] >= 0.98 if not math.isnan(fit_y["r2"]) else False,
        "no_penetration": penetration_depth_px <= 2.0,
        "no_bounce_large": bounce_height_norm <= 0.05,
    }
    if g_rel_err is not None:
        strict_checks["g_relative_error_le_0.30"] = g_rel_err <= 0.30
    if v0_abs_err is not None:
        strict_checks["v0_abs_error_reasonable"] = v0_abs_err <= max(0.5, 0.3 * abs(v0_expected) + 0.2)
    if h0_rel_err is not None:
        strict_checks["h0_relative_error_le_0.30"] = h0_rel_err <= 0.30

    strict_pass = all(strict_checks.values())

    prompt_style = parse_style_from_path(video_path, input_root, model_name)

    meta_path = video_path.with_suffix(".json")
    result = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "stem": video_path.stem,
        "prompt_style": prompt_style,
        "model_name": model_name,
        "metadata_json_path": str(meta_path) if meta_path.exists() else None,
        "fps": fps,
        "frame_count": frame_count,
        "frame_width": width,
        "frame_height": height,
        "ground_y_px": ground_y_px,
        "metric_scale_available": metric_scale_available,
        "meters_per_pixel": meters_per_pixel if metric_scale_available else None,
        "object_diameter_m": object_diameter_m,
        "expected_parameters": expected,
        "fit": {
            "quadratic_y": fit_y,
            "linear_x": fit_x,
            "recovered": {
                "g": g_est,
                "v0": v0_est,
                "h0": h0_est,
                "unit": unit_name,
            },
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
        "parameter_errors": {
            "g_relative_error": g_rel_err,
            "v0_absolute_error": v0_abs_err,
            "h0_relative_error": h0_rel_err,
        },
        "component_scores_0_1": component_scores,
        "overall_score_100": overall_score,
        "strict_checks": strict_checks,
        "strict_pass": strict_pass,
        "limitation_note": "该脚本评测的是自由落体物理/几何合规性，不评测高层语义场景是否完全符合 prompt。",
    }

    save_json(outdir / "evaluation.json", result)
    return result


def find_model_videos(input_root: Path, model_name: str) -> List[Path]:
    videos = []
    for p in input_root.rglob("*.mp4"):
        parts = p.parts
        if model_name in parts or model_name in p.stem:
            videos.append(p)
    return sorted(videos)


def summarize_by_style(rows: List[Dict]) -> List[Dict]:
    groups = {}
    for r in rows:
        style = r["prompt_style"]
        groups.setdefault(style, []).append(r)

    out = []
    for style, lst in sorted(groups.items()):
        def avg(key):
            vals = [safe_float(x.get(key)) for x in lst]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        out.append({
            "prompt_style": style,
            "count": len(lst),
            "mean_overall_score_100": avg("overall_score_100"),
            "mean_detection_rate": avg("detection_rate"),
            "mean_verticality_ratio": avg("verticality_ratio"),
            "mean_monotonic_down_rate": avg("monotonic_down_rate"),
            "mean_quadratic_r2": avg("quadratic_r2"),
            "strict_pass_rate": sum(int(bool(x.get("strict_pass"))) for x in lst) / len(lst) if lst else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="批量评测指定模型在自由落体任务上的所有生成视频")
    ap.add_argument("--input-root", type=str, required=True, help="自由落体视频根目录，例如 outputs/baseline_v1_all_tasks/free_fall")
    ap.add_argument("--model-name", type=str, required=True, help="模型目录名，例如 wan2.2_ti2v_5b")
    ap.add_argument("--eval-outdir", type=str, required=True, help="评测输出目录")
    ap.add_argument("--color", type=str, default="red", choices=list(COLOR_PRESETS.keys()), help="主体颜色")
    ap.add_argument("--min-area", type=float, default=80.0)
    ap.add_argument("--open-kernel", type=int, default=5)
    ap.add_argument("--close-kernel", type=int, default=7)
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--max-interp-gap", type=int, default=5)
    ap.add_argument("--ground-y-px", type=float, default=None, help="若不给，默认图像底边")
    ap.add_argument("--meters-per-pixel", type=float, default=None, help="米/像素")
    ap.add_argument("--object-diameter-m", type=float, default=None, help="已知目标球直径，用于自动求尺度")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1)
    ap.add_argument("--save-annotated-video", action="store_true", help="是否保存每视频 annotated.mp4")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    eval_outdir = Path(args.eval_outdir)
    eval_outdir.mkdir(parents=True, exist_ok=True)

    videos = find_model_videos(input_root, args.model_name)
    if not videos:
        raise RuntimeError(f"在 {input_root} 下没有找到 model={args.model_name} 的 mp4")

    summary_rows = []
    errors = []

    for idx, video in enumerate(videos, 1):
        try:
            rel = video.relative_to(input_root)
            per_video_outdir = eval_outdir / rel.with_suffix("")
            per_video_outdir.mkdir(parents=True, exist_ok=True)

            result = evaluate_one_video(
                video_path=video,
                input_root=input_root,
                model_name=args.model_name,
                color_name=args.color,
                outdir=per_video_outdir,
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
                save_annotated_video=args.save_annotated_video,
            )

            row = {
                "video_path": result["video_path"],
                "video_name": result["video_name"],
                "prompt_style": result["prompt_style"],
                "model_name": result["model_name"],
                "overall_score_100": round(result["overall_score_100"], 4),
                "strict_pass": result["strict_pass"],
                "detection_rate": round(result["motion_metrics"]["detection_rate"], 6),
                "single_object_rate": round(result["motion_metrics"]["single_object_rate"], 6),
                "in_frame_rate": round(result["motion_metrics"]["in_frame_rate"], 6),
                "verticality_ratio": round(result["motion_metrics"]["verticality_ratio"], 6),
                "monotonic_down_rate": round(result["motion_metrics"]["monotonic_down_rate"], 6),
                "penetration_depth_px": round(result["motion_metrics"]["penetration_depth_px"], 6),
                "bounce_height_norm": round(result["motion_metrics"]["bounce_height_norm"], 6),
                "quadratic_r2": round(result["fit"]["quadratic_y"]["r2"], 6) if not math.isnan(result["fit"]["quadratic_y"]["r2"]) else None,
                "g_est": round(result["fit"]["recovered"]["g"], 6),
                "v0_est": round(result["fit"]["recovered"]["v0"], 6),
                "h0_est": round(result["fit"]["recovered"]["h0"], 6),
                "g_expected": result["expected_parameters"]["g_expected"],
                "v0_expected": result["expected_parameters"]["v0_expected"],
                "h0_expected": result["expected_parameters"]["h0_expected"],
                "g_relative_error": result["parameter_errors"]["g_relative_error"],
                "v0_absolute_error": result["parameter_errors"]["v0_absolute_error"],
                "h0_relative_error": result["parameter_errors"]["h0_relative_error"],
            }
            summary_rows.append(row)
            print(f"[{idx}/{len(videos)}] OK  score={row['overall_score_100']:.2f}  strict={row['strict_pass']}  {video.name}")

        except Exception as e:
            errors.append({"video_path": str(video), "error": f"{type(e).__name__}: {e}"})
            print(f"[{idx}/{len(videos)}] ERR {video.name} -> {type(e).__name__}: {e}")

    if summary_rows:
        summary_rows_sorted = sorted(summary_rows, key=lambda x: x["overall_score_100"], reverse=True)
        write_csv(eval_outdir / "summary.csv", summary_rows_sorted)
        save_json(eval_outdir / "summary.json", {"results": summary_rows_sorted, "errors": errors})

        style_rows = summarize_by_style(summary_rows_sorted)
        write_csv(eval_outdir / "style_summary.csv", style_rows)

        with (eval_outdir / "ranking.txt").open("w", encoding="utf-8") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Input root: {input_root}\n")
            f.write(f"Total ok: {len(summary_rows_sorted)}\n")
            f.write(f"Total err: {len(errors)}\n\n")
            for i, r in enumerate(summary_rows_sorted, 1):
                f.write(
                    f"{i:03d}. score={r['overall_score_100']:.2f}  strict={r['strict_pass']}  "
                    f"style={r['prompt_style']}  file={Path(r['video_name']).name}\n"
                )

        mean_score = sum(r["overall_score_100"] for r in summary_rows_sorted) / len(summary_rows_sorted)
        strict_pass_rate = sum(int(bool(r["strict_pass"])) for r in summary_rows_sorted) / len(summary_rows_sorted)
        print("=" * 72)
        print("批量评测完成")
        print(f"模型: {args.model_name}")
        print(f"成功评测视频数: {len(summary_rows_sorted)}")
        print(f"失败视频数: {len(errors)}")
        print(f"平均总分: {mean_score:.2f}")
        print(f"严格合规率: {strict_pass_rate:.2%}")
        print(f"结果目录: {eval_outdir}")
        print("=" * 72)
    else:
        save_json(eval_outdir / "summary.json", {"results": [], "errors": errors})
        print("没有成功评测任何视频。")


if __name__ == "__main__":
    main()
