#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自由落体视频追踪与拟合脚本
------------------------------------------------
功能：
1. 读取视频
2. 用 HSV 颜色分割 + 最大轮廓质心 追踪单个物体
3. 可选：根据已知物体直径自动估计 meters_per_pixel
4. 拟合自由落体模型 y(t) = a t^2 + b t + c
5. 恢复参数 g = -2a, v0 = b, h0 = c
6. 输出：
   - trajectory.csv
   - fit_results.json
   - fit_plot.png
   - trajectory_plot.png
   - x_fit_plot.png
   - annotated.mp4

坐标约定：
- 图像坐标：u 向右为正，v 向下为正
- 物理坐标：x 向右为正，y 向上为正
- 若给定 --ground-y-px，则 y = (ground_y_px - cy) * scale
- 否则默认 ground_y_px = frame_height - 1，即以图像底部为“地面/零高度参考”
  此时 h0 是“相对图像底边”的高度，不一定等于真实地面高度

依赖：
pip install opencv-python numpy matplotlib
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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


@dataclass
class Detection:
    frame_idx: int
    time_s: float
    cx: Optional[float]
    cy: Optional[float]
    radius_px: Optional[float]
    area_px: Optional[float]
    found: bool


def parse_args():
    parser = argparse.ArgumentParser(description="自由落体视频追踪与拟合")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--outdir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--color",
        type=str,
        default="red",
        choices=list(COLOR_PRESETS.keys()),
        help="目标主体颜色预设",
    )
    parser.add_argument("--min-area", type=float, default=80.0, help="最小轮廓面积阈值")
    parser.add_argument("--open-kernel", type=int, default=5, help="开运算核大小")
    parser.add_argument("--close-kernel", type=int, default=7, help="闭运算核大小")
    parser.add_argument("--blur-ksize", type=int, default=5, help="高斯模糊核大小，0 表示关闭")
    parser.add_argument(
        "--ground-y-px",
        type=float,
        default=None,
        help="地面对应的图像 y 像素坐标；若不提供，则使用图像底边",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help="像素到米的换算比例（米/像素）；若不给，默认输出单位为像素",
    )
    parser.add_argument(
        "--object-diameter-m",
        type=float,
        default=None,
        help="已知物体真实直径（米）。若提供且未提供 meters-per-pixel，则用检测到的半径自动估算比例尺",
    )
    parser.add_argument(
        "--max-interp-gap",
        type=int,
        default=5,
        help="允许线性插值补点的最大连续丢失帧数",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="从第几帧开始处理")
    parser.add_argument("--end-frame", type=int, default=-1, help="处理到第几帧结束（含），-1 表示到视频末尾")
    parser.add_argument("--save-debug-mask", action="store_true", help="保存前 5 帧的 mask 调试图")
    parser.add_argument("--no-annotated-video", action="store_true", help="不保存带标注视频")
    return parser.parse_args()


def ensure_odd(k: int) -> int:
    if k <= 0:
        return 0
    return k if k % 2 == 1 else k + 1


def build_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    ranges = COLOR_PRESETS[color_name]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))
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
) -> Tuple[Optional[Tuple[float, float]], Optional[float], Optional[float], np.ndarray]:
    img = frame_bgr.copy()
    blur_ksize = ensure_odd(blur_ksize)
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color_name)
    mask = postprocess_mask(mask, open_kernel, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, mask

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best = contours[0]
    area = float(cv2.contourArea(best))
    if area < min_area:
        return None, None, None, mask

    M = cv2.moments(best)
    if abs(M["m00"]) < 1e-8:
        return None, None, None, mask

    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    (_, _), radius = cv2.minEnclosingCircle(best)
    radius = float(radius)

    return (cx, cy), radius, area, mask


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


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: Dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_results(
    outdir: Path,
    times: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    fit_y: Dict[str, float],
    fit_x: Dict[str, float],
    unit_name: str,
):
    t_dense = np.linspace(times.min(), times.max(), 400)
    y_dense = fit_y["a"] * t_dense**2 + fit_y["b"] * t_dense + fit_y["c"]

    plt.figure(figsize=(8, 5))
    plt.scatter(times, y_vals, s=20, label="tracked y(t)")
    plt.plot(t_dense, y_dense, linewidth=2, label="quadratic fit")
    plt.xlabel("time (s)")
    plt.ylabel(f"y ({unit_name}, upward positive)")
    plt.title("Free-fall fit: y(t) = a t^2 + b t + c")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, s=20)
    plt.plot(x_vals, y_vals, linewidth=1)
    plt.xlabel(f"x ({unit_name})")
    plt.ylabel(f"y ({unit_name}, upward positive)")
    plt.title("Tracked trajectory")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outdir / "trajectory_plot.png", dpi=180)
    plt.close()

    x_dense = fit_x["m"] * t_dense + fit_x["q"]
    plt.figure(figsize=(8, 5))
    plt.scatter(times, x_vals, s=20, label="tracked x(t)")
    plt.plot(t_dense, x_dense, linewidth=2, label="linear fit")
    plt.xlabel("time (s)")
    plt.ylabel(f"x ({unit_name})")
    plt.title("Horizontal drift check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "x_fit_plot.png", dpi=180)
    plt.close()


def main():
    args = parse_args()
    video_path = Path(args.video)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6 or np.isnan(fps):
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, args.start_frame)
    end_frame = frame_count - 1 if args.end_frame < 0 else min(args.end_frame, frame_count - 1)

    ground_y_px = float(args.ground_y_px) if args.ground_y_px is not None else float(height - 1)

    writer = None
    if not args.no_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(outdir / "annotated.mp4"), fourcc, fps, (width, height))

    detections: List[Detection] = []
    debug_saved = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break

        time_s = frame_idx / fps
        center, radius_px, area_px, mask = detect_object(
            frame_bgr=frame,
            color_name=args.color,
            min_area=args.min_area,
            open_kernel=args.open_kernel,
            close_kernel=args.close_kernel,
            blur_ksize=args.blur_ksize,
        )

        if center is not None:
            cx, cy = center
            detections.append(Detection(frame_idx, time_s, cx, cy, radius_px, area_px, True))
        else:
            detections.append(Detection(frame_idx, time_s, None, None, None, None, False))

        if args.save_debug_mask and debug_saved < 5:
            cv2.imwrite(str(outdir / f"debug_mask_{debug_saved:02d}.png"), mask)
            debug_saved += 1

        if writer is not None:
            vis = frame.copy()
            cv2.line(vis, (0, int(round(ground_y_px))), (width - 1, int(round(ground_y_px))), (0, 255, 255), 2)
            if center is not None:
                cx_i = int(round(center[0]))
                cy_i = int(round(center[1]))
                rr = int(round(radius_px)) if radius_px is not None else 8
                cv2.circle(vis, (cx_i, cy_i), max(rr, 3), (0, 255, 0), 2)
                cv2.circle(vis, (cx_i, cy_i), 3, (0, 0, 255), -1)
                label = f"frame={frame_idx} t={time_s:.3f}s"
                cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                label = f"frame={frame_idx} t={time_s:.3f}s LOST"
                cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()

    if not detections:
        raise RuntimeError("没有读取到任何帧。")

    times_all = [d.time_s for d in detections]
    cx_all = [d.cx for d in detections]
    cy_all = [d.cy for d in detections]
    r_all = [d.radius_px for d in detections]

    cx_interp = interpolate_small_gaps(cx_all, args.max_interp_gap)
    cy_interp = interpolate_small_gaps(cy_all, args.max_interp_gap)

    meters_per_pixel = args.meters_per_pixel
    diameter_px_list = [2.0 * r for r in r_all if r is not None and r > 1e-6]
    if meters_per_pixel is None and args.object_diameter_m is not None and diameter_px_list:
        median_diameter_px = float(np.median(np.array(diameter_px_list)))
        meters_per_pixel = float(args.object_diameter_m) / median_diameter_px

    if meters_per_pixel is None:
        meters_per_pixel = 1.0
        unit_name = "px"
        scale_mode = "pixel_only"
    else:
        unit_name = "m"
        scale_mode = "metric"

    rows = []
    valid_t = []
    valid_x = []
    valid_y = []

    x0_px = None
    for i, det in enumerate(detections):
        cx = cx_interp[i]
        cy = cy_interp[i]

        if cx is not None and x0_px is None:
            x0_px = cx

        if cx is not None and cy is not None and x0_px is not None:
            x_phys = (cx - x0_px) * meters_per_pixel
            y_phys = (ground_y_px - cy) * meters_per_pixel
            valid_t.append(det.time_s)
            valid_x.append(x_phys)
            valid_y.append(y_phys)
            valid_now = True
        else:
            x_phys = None
            y_phys = None
            valid_now = False

        rows.append({
            "frame_idx": det.frame_idx,
            "time_s": det.time_s,
            "cx_px": None if det.cx is None else round(det.cx, 4),
            "cy_px": None if det.cy is None else round(det.cy, 4),
            "radius_px": None if det.radius_px is None else round(det.radius_px, 4),
            "area_px": None if det.area_px is None else round(det.area_px, 4),
            "cx_interp_px": None if cx is None else round(cx, 4),
            "cy_interp_px": None if cy is None else round(cy, 4),
            "x": None if x_phys is None else round(x_phys, 8),
            "y": None if y_phys is None else round(y_phys, 8),
            "used_for_fit": valid_now,
        })

    write_csv(outdir / "trajectory.csv", rows)

    valid_t = np.array(valid_t, dtype=float)
    valid_x = np.array(valid_x, dtype=float)
    valid_y = np.array(valid_y, dtype=float)

    if len(valid_t) < 5:
        raise RuntimeError("有效追踪点太少，无法做稳定拟合。请先调整颜色阈值或视频质量。")

    t0 = float(valid_t[0])
    t_fit = valid_t - t0

    fit_y = fit_quadratic(t_fit, valid_y)
    fit_x = fit_linear(t_fit, valid_x)

    g_est = -2.0 * fit_y["a"]
    v0_est = fit_y["b"]
    h0_est = fit_y["c"]

    detected_count = sum(int(d.found) for d in detections)
    interp_count = sum(
        int((not d.found) and (cx_interp[i] is not None) and (cy_interp[i] is not None))
        for i, d in enumerate(detections)
    )
    total_count = len(detections)

    results = {
        "video": str(video_path),
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "total_processed_frames": total_count,
        "detected_frames": detected_count,
        "interpolated_frames": interp_count,
        "detection_rate": detected_count / total_count if total_count > 0 else None,
        "ground_y_px": ground_y_px,
        "scale_mode": scale_mode,
        "meters_per_pixel": meters_per_pixel,
        "unit_name": unit_name,
        "estimated_from_object_diameter_m": args.object_diameter_m,
        "fit_model_y": "y(t)=a t^2 + b t + c",
        "fit_model_x": "x(t)=m t + q",
        "time_origin_note": "t=0 is reset to the first valid tracked frame",
        "height_origin_note": "y=0 is the given ground_y_px or image bottom edge",
        "y_fit": fit_y,
        "x_fit": fit_x,
        "recovered_parameters": {
            "g": g_est,
            "v0_vertical": v0_est,
            "h0": h0_est,
            "g_unit": f"{unit_name}/s^2",
            "v0_unit": f"{unit_name}/s",
            "h0_unit": unit_name,
        },
        "quality_checks": {
            "horizontal_drift_speed": fit_x["m"],
            "horizontal_drift_unit": f"{unit_name}/s",
            "horizontal_drift_rmse": fit_x["rmse"],
            "vertical_fit_rmse": fit_y["rmse"],
            "vertical_fit_r2": fit_y["r2"],
        },
    }

    save_json(outdir / "fit_results.json", results)

    plot_results(outdir, t_fit, valid_x, valid_y, fit_y, fit_x, unit_name)

    print("=" * 60)
    print("追踪与拟合完成")
    print(f"视频: {video_path}")
    print(f"输出目录: {outdir}")
    print(f"检测帧数/总帧数: {detected_count}/{total_count} ({detected_count/total_count:.2%})")
    print(f"插值补点帧数: {interp_count}")
    print(f"单位: {unit_name}")
    print("-" * 60)
    print("拟合结果: y(t) = a t^2 + b t + c")
    print(f"a = {fit_y['a']:.8f}")
    print(f"b = {fit_y['b']:.8f}")
    print(f"c = {fit_y['c']:.8f}")
    print("-" * 60)
    print("恢复参数:")
    print(f"g  = {g_est:.8f} {unit_name}/s^2")
    print(f"v0 = {v0_est:.8f} {unit_name}/s")
    print(f"h0 = {h0_est:.8f} {unit_name}")
    print("-" * 60)
    print("质量检查:")
    print(f"x drift speed = {fit_x['m']:.8f} {unit_name}/s")
    print(f"y-fit RMSE    = {fit_y['rmse']:.8f} {unit_name}")
    print(f"y-fit R^2     = {fit_y['r2']:.8f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
