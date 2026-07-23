#!/usr/bin/env python3
"""Render the matched Seedance parameter-response casebook and demo videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.evidence_formulas import (  # noqa: E402
    EXPERIMENT_ORDER,
    FORMULA_REGISTRY,
    benchmark_scope_summary,
    get_formula_spec,
    parameter_channel_manifest,
)

EXPERIMENT_LABELS = {
    experiment_id: str(FORMULA_REGISTRY[experiment_id]["experiment_title_en"])
    for experiment_id in EXPERIMENT_ORDER
}

EVIDENCE_LABELS = {
    "strong_directional_success": "strong",
    "limited_directional_success": "limited",
    "direction_only_uncalibrated": "direction only",
    "failure_wrong_direction": "wrong direction",
    "failure_flat_or_saturated": "flat / saturated",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _values(text: Any) -> list[float]:
    return [float(value) for value in str(text).split(";") if str(value).strip()]


def _job_ids(text: Any) -> list[str]:
    return [value for value in str(text).split(";") if value]


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _slug(parameter: str) -> str:
    return {
        "gravity_g": "gravity g",
        "restitution_e": "restitution e",
        "kinetic_friction_mu": "friction mu",
        "amplitude_decay_beta": "decay beta",
        "kinetic_friction_mu_A": "friction mu_A",
        "kinetic_friction_mu_B": "friction mu_B",
        "linear_damping_beta": "damping beta",
        "linear_drag_beta": "drag beta",
        "kinetic_friction_mu_k": "friction mu_k",
        "left_restitution_e_L": "left restitution e_L",
        "right_restitution_e_R": "right restitution e_R",
        "magnetic_kappa": "magnetic kappa",
    }.get(parameter, parameter)


def _fit_text(row: Mapping[str, Any]) -> str:
    gt = _values(row["fitted_gt_levels"])
    estimates = _values(row["fitted_estimates"])
    return " / ".join(f"{_fmt(x)}->{_fmt(y)}" for x, y in zip(gt, estimates))


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (25, 25, 25),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _chart_ranges(gt: Sequence[float], estimates: Sequence[float]) -> tuple[float, float, float, float]:
    x_low, x_high = min(gt), max(gt)
    y_low, y_high = min(estimates), max(estimates)
    x_span = max(x_high - x_low, max(abs(x_low), abs(x_high), 1.0) * 0.12)
    y_span = max(y_high - y_low, max(abs(y_low), abs(y_high), 1.0) * 0.12)
    return (
        x_low - 0.15 * x_span,
        x_high + 0.15 * x_span,
        y_low - 0.18 * y_span,
        y_high + 0.18 * y_span,
    )


def _draw_response_plot(row: Mapping[str, Any], destination: Path) -> np.ndarray:
    width, height = 720, 520
    image = np.full((height, width, 3), 248, np.uint8)
    left, right, top, bottom = 95, 675, 86, 430
    gt = _values(row["fitted_gt_levels"])
    estimates = _values(row["fitted_estimates"])
    x_low, x_high, y_low, y_high = _chart_ranges(gt, estimates)

    def point(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - x_low) / (x_high - x_low) * (right - left))
        py = bottom - int((y - y_low) / (y_high - y_low) * (bottom - top))
        return px, py

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + int(fraction * (right - left))
        y = top + int(fraction * (bottom - top))
        cv2.line(image, (x, top), (x, bottom), (225, 225, 225), 1)
        cv2.line(image, (left, y), (right, y), (225, 225, 225), 1)
        _put_text(image, _fmt(x_low + fraction * (x_high - x_low)), (x - 18, bottom + 24), 0.42, (80, 80, 80))
        _put_text(image, _fmt(y_high - fraction * (y_high - y_low)), (15, y + 5), 0.42, (80, 80, 80))
    cv2.rectangle(image, (left, top), (right, bottom), (105, 105, 105), 1)

    overlap_low = max(x_low, y_low)
    overlap_high = min(x_high, y_high)
    if overlap_low < overlap_high:
        cv2.line(image, point(overlap_low, overlap_low), point(overlap_high, overlap_high), (155, 155, 155), 2, cv2.LINE_AA)

    points = [point(x, y) for x, y in zip(gt, estimates)]
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (205, 95, 32), 4, cv2.LINE_AA)
    for index, ((px, py), x, y) in enumerate(zip(points, gt, estimates)):
        cv2.circle(image, (px, py), 7, (205, 95, 32), -1, cv2.LINE_AA)
        label_y = py - 11 if index % 2 == 0 else py + 23
        _put_text(image, f"{_fmt(x)}->{_fmt(y)}", (min(px + 9, right - 105), label_y), 0.42, (25, 25, 25))

    case = str(row["case_type"]).upper()
    title = f"{row['experiment_id']} {case}: {_slug(str(row['target_parameter']))}"
    _put_text(image, title, (28, 34), 0.78, (20, 20, 20), 2)
    _put_text(image, f"scene={row['scene_id']}  evidence={EVIDENCE_LABELS.get(str(row['evidence']), row['evidence'])}", (28, 62), 0.52, (70, 70, 70))
    _put_text(image, "requested parameter", (285, 492), 0.54, (45, 45, 45))
    _put_text(image, "fitted", (24, 79), 0.48, (45, 45, 45))
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)
    return image


def _overlay_path(eval_root: Path, job_id: str) -> Path:
    return eval_root / "jobs" / job_id / "validity_object_track_overlay.mp4"


def _write_montage(row: Mapping[str, Any], eval_root: Path, destination: Path) -> dict[str, Any]:
    jobs = _job_ids(row["job_ids"])
    gt = _values(row["fitted_gt_levels"])
    estimates = _values(row["fitted_estimates"])
    sources = [_overlay_path(eval_root, job) for job in jobs]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        return {"written": False, "missing": missing}

    captures = [cv2.VideoCapture(str(path)) for path in sources]
    try:
        fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
        fps = min(value for value in fps_values if value > 0) if any(value > 0 for value in fps_values) else 24.0
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
        frame_count = max(frame_counts)
        tile_width, tile_height = (480, 270) if len(captures) >= 3 else (620, 349)
        title_height = 74
        canvas = np.full((title_height + tile_height, tile_width * len(captures), 3), 245, np.uint8)
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
        last_frames: list[np.ndarray | None] = [None] * len(captures)
        for _ in range(frame_count):
            canvas[:] = 245
            _put_text(canvas, f"{row['experiment_id']} {str(row['case_type']).upper()} | {_slug(str(row['target_parameter']))} | {row['scene_id']}", (16, 30), 0.72, (20, 20, 20), 2)
            _put_text(canvas, "orange=candidate, green=accepted track, trail=measured trajectory", (16, 58), 0.48, (70, 70, 70))
            for index, cap in enumerate(captures):
                ok, frame = cap.read()
                if ok:
                    last_frames[index] = frame
                frame = last_frames[index]
                if frame is None:
                    frame = np.zeros((tile_height, tile_width, 3), np.uint8)
                frame = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                x0 = index * tile_width
                canvas[title_height:, x0:x0 + tile_width] = frame
                cv2.rectangle(canvas, (x0, title_height), (x0 + tile_width - 1, title_height + 33), (15, 15, 15), -1)
                _put_text(canvas, f"GT {_fmt(gt[index])} | fit {_fmt(estimates[index])}", (x0 + 12, title_height + 24), 0.58, (245, 245, 245), 2)
            writer.write(canvas)
        writer.release()
        preview = destination.with_suffix(".jpg")
        cv2.imwrite(str(preview), canvas)
        return {"written": True, "frames": frame_count, "fps": fps, "preview": str(preview)}
    finally:
        for cap in captures:
            cap.release()


def _make_grid(images: Sequence[np.ndarray], title: str, destination: Path) -> None:
    cell_w, cell_h = 480, 347
    columns = 3
    rows = math.ceil(len(images) / columns)
    header = 64
    canvas = np.full((header + rows * cell_h, columns * cell_w, 3), 248, np.uint8)
    _put_text(canvas, title, (22, 39), 0.85, (20, 20, 20), 2)
    for index, image in enumerate(images):
        resized = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        y = header + (index // columns) * cell_h
        x = (index % columns) * cell_w
        canvas[y:y + cell_h, x:x + cell_w] = resized
        cv2.rectangle(canvas, (x, y), (x + cell_w - 1, y + cell_h - 1), (205, 205, 205), 1)
    cv2.imwrite(str(destination), canvas)


def _make_matrix(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    width, row_height, header = 1900, 68, 92
    canvas = np.full((header + len(EXPERIMENT_ORDER) * row_height + 24, width, 3), 248, np.uint8)
    _put_text(canvas, "Matched parameter-intervention evidence: success and counterexample per experiment", (24, 38), 0.82, (20, 20, 20), 2)
    _put_text(canvas, "Success means fitted direction follows requested intervention; it does not imply accurate calibration.", (24, 69), 0.54, (70, 70, 70))
    by_key = {(row["experiment_id"], row["case_type"]): row for row in rows}
    for index, experiment in enumerate(EXPERIMENT_ORDER):
        y0 = header + index * row_height
        if index % 2:
            cv2.rectangle(canvas, (0, y0), (width, y0 + row_height), (239, 239, 239), -1)
        _put_text(canvas, experiment, (22, y0 + 29), 0.68, (20, 20, 20), 2)
        _put_text(canvas, EXPERIMENT_LABELS[experiment], (22, y0 + 54), 0.46, (80, 80, 80))
        for case, x0, color in (("success", 360, (54, 139, 76)), ("failure", 1130, (65, 73, 194))):
            row = by_key[(experiment, case)]
            cv2.rectangle(canvas, (x0, y0 + 8), (x0 + 735, y0 + row_height - 8), tuple(int(0.88 * 248 + 0.12 * c) for c in color), -1)
            tag = EVIDENCE_LABELS.get(str(row["evidence"]), str(row["evidence"]))
            _put_text(canvas, f"{case.upper()} | {row['scene_id']} | {_slug(str(row['target_parameter']))} | rho={_fmt(row['spearman_rho'], 2)}", (x0 + 14, y0 + 29), 0.53, (25, 25, 25), 2)
            _put_text(canvas, f"GT->fit: {_fit_text(row)} | {tag}", (x0 + 14, y0 + 54), 0.43, (60, 60, 60))
    cv2.imwrite(str(destination), canvas)


def _scope_table_lines() -> list[str]:
    scope = benchmark_scope_summary()
    channels = parameter_channel_manifest()
    by_experiment: dict[str, list[dict[str, Any]]] = {
        experiment_id: [] for experiment_id in EXPERIMENT_ORDER
    }
    for channel in channels:
        by_experiment[str(channel["experiment_id"])].append(channel)

    lines = [
        "## 统计单位：13 个实验系统，不是 24 个实验",
        "",
        str(scope["counting_rule_zh"]),
        "本证据册仍只为每个实验系统挑选一组正向案例和一组反例；这是 13×2 个展示组合，"
        "不能与完整的 24 个参数响应通道混为一谈。",
        "",
        "|实验系统|真实运动系统|独立扫描的参数通道|通道数|",
        "|---|---|---|---:|",
    ]
    for experiment_id in EXPERIMENT_ORDER:
        experiment = FORMULA_REGISTRY[experiment_id]
        experiment_channels = by_experiment[experiment_id]
        channel_text = "；".join(
            f"`{row['parameter_name']}`（{row['parameter_title_zh']}）"
            for row in experiment_channels
        )
        lines.append(
            f"|{experiment_id}|{experiment['experiment_title_zh']}|{channel_text}|{len(experiment_channels)}|"
        )
    lines.extend(
        [
            "",
            f"合计：**{scope['experiment_system_count']} 个实验系统，"
            f"{scope['parameter_response_channel_count']} 个 OAT 参数响应通道**。",
        ]
    )
    return lines


def _formula_appendix_lines() -> list[str]:
    lines = [
        "## 13 个实验系统、24 个参数通道的反推公式",
        "",
        "以下方法名与数值 fitter 的 `method` 字段一致。每个过程先从已经通过测量门控的米制轨迹"
        "选择实验所需片段，再独立反演参数；目标参数只在拟合完成后用于 GT→拟合比较。",
    ]
    for experiment_id in EXPERIMENT_ORDER:
        experiment = FORMULA_REGISTRY[experiment_id]
        lines.extend(
            [
                "",
                f"### {experiment_id} · {experiment['experiment_title_zh']}",
                "",
                f"- 观测量：`{', '.join(experiment['observables'])}`",
                f"- 轨迹模型：`{experiment['model_plain']}`",
                f"- 数值方法：`{experiment['fitter_method']}`。{experiment['fitter_summary_zh']}",
            ]
        )
        for parameter_name in experiment["parameters"]:
            spec = get_formula_spec(experiment_id, parameter_name)
            steps = " → ".join(str(step) for step in spec["estimate_steps_zh"])
            lines.extend(
                [
                    f"- **通道 `{experiment_id}/{parameter_name}` · {spec['parameter_title_zh']}**："
                    f"`{spec['estimate_plain']}`",
                    f"  - 简化过程：{steps}",
                ]
            )
    return lines


def _report(rows: Sequence[Mapping[str, Any]], output: Path, montage_status: Mapping[tuple[str, str], Mapping[str, Any]]) -> str:
    by_key = {(row["experiment_id"], row["case_type"]): row for row in rows}
    success_counts = Counter(row["evidence"] for row in rows if row["case_type"] == "success")
    lines = [
        "# Seedance 2.0 参数干预证据册（Side 主评测）",
        "",
        "## 结论先行",
        "",
        f"在 13 个冻结实验中，都能找到至少一组“目标参数提高时，独立拟合值总体同向变化”的组合；其中强方向证据 {success_counts['strong_directional_success']} 组、有限方向证据 {success_counts['limited_directional_success']} 组、仅方向正确但标定失真 {success_counts['direction_only_uncalibrated']} 组。与此同时，每个实验也都能找到一组反向或近乎不响应的反例。",
        "",
        "这里的“成功”只表示参数干预方向与拟合响应一致，不等于参数数值恢复准确。论文中必须把方向遵循、参数标定精度和视频生成有效性分开报告。",
        "",
        "![13 个实验的成功与反例矩阵](fig1_success_failure_matrix.png)",
    ]
    lines.extend(_scope_table_lines())
    lines.extend(
        [
            "",
            "## 选择规则",
            "",
            "- 只使用 `CAM_Side`、主种子 `341867882`、生成有效性为 pass、轨迹覆盖率不低于 0.85 且物理拟合完整的视频。",
            "- 同一组内固定场景、相机、seed 和所有非目标物理参数，只改变一个目标参数（matched one-at-a-time intervention）。",
            "- 检测器和拟合器不读取 prompt 中的目标参数；参数只在拟合完成后用于比较，避免标签泄漏。",
            "- 强证据：至少 3 档、严格递增、Spearman rho>=0.8、响应幅度充分且估计值大致处于合理范围。有限证据：总体同向但档数少、存在饱和或局部非单调。仅方向证据：方向正确，但绝对值明显越界或校准误差过大。",
            "- 反例是同样满足可追踪与可拟合条件、但响应反向或近乎不变的组合；它是模型物理控制失败，不是检测失败或视频形变失败。",
            "",
            "## 逐实验案例",
            "",
            "|实验|正向组合（GT→拟合）|证据等级|失败组合（GT→拟合）|失败类型|",
            "|---|---|---|---|---|",
        ]
    )
    for experiment in EXPERIMENT_ORDER:
        success = by_key[(experiment, "success")]
        failure = by_key[(experiment, "failure")]
        success_path = f"cases/{experiment}/success_overlay_montage.mp4"
        failure_path = f"cases/{experiment}/failure_overlay_montage.mp4"
        success_text = f"[{success['scene_id']} · {_slug(str(success['target_parameter']))} · {_fit_text(success)}]({success_path})"
        failure_text = f"[{failure['scene_id']} · {_slug(str(failure['target_parameter']))} · {_fit_text(failure)}]({failure_path})"
        lines.append(
            f"|{experiment} {EXPERIMENT_LABELS[experiment]}|{success_text}|{EVIDENCE_LABELS.get(str(success['evidence']), success['evidence'])}|{failure_text}|{EVIDENCE_LABELS.get(str(failure['evidence']), failure['evidence'])}|"
        )
    lines.extend([
        "",
        "![正向响应小多图](fig2_success_response_small_multiples.png)",
        "",
        "![反向或饱和反例小多图](fig3_failure_response_small_multiples.png)",
        "",
        "### 如何解读这些案例",
        "",
        "- V1_B、V2_A、V2_D 是当前最适合放正文的强方向案例：都有三档以上或严格的同向变化，并且拟合值没有严重越界。",
        "- V1_A、V1_C、V1_D、V2_B、V3_A、V3_B、V3_C 可作为有限支持：它们说明模型在某些场景能响应参数，但存在压缩、饱和或局部非单调，不能写成稳定理解。",
        "- V2_C、V2_E、V3_D 只能支持“方向偶尔正确”，不能支持“参数被准确恢复”；这些例子的绝对值校准失真，应放补充材料或作为局限性。",
        "- 每个实验的 failure montage 是 matched counterexample：轨迹能检测、拟合能完成，但参数响应方向错误，因此不能把问题归因于检测失败。",
    ])
    lines.extend(_formula_appendix_lines())
    lines.extend([
        "",
        "## 2D 检测与轨迹恢复使用的技术",
        "",
        "当前方法不是单纯的颜色阈值，也不是通用目标检测器，而是面向冻结 Blender 首帧和标准刚性球的“首帧先验 + 多证据候选检测 + 物理约束时序关联 + 相机标定回投影”混合系统。",
        "",
        "1. **首帧几何与外观锚点。** 从冻结的 Blender sidecar 读取球心、真实半径、投影半径、相机内参 K 与外参 R/t；在生成视频首帧中只估计必要的图像平移和球体颜色分布。这样不会把目标物理参数带入检测。",
        "2. **多证据候选生成。** 在 HSV 空间产生颜色候选，并对轮廓面积、圆度、椭圆轴比、最小外接圆、径向边缘残差和边界截断进行筛选；颜色不稳定时再使用 Canny + Hough 圆作为保守回退。",
        "3. **背景干扰抑制。** 把当前帧与首帧背景做差，计算候选区域的变化支持率；静止的同色背景通常没有足够变化，因此不会仅凭“颜色像球”被接受。系统还保留持续同色干扰物的记忆并提高其代价。",
        "4. **受约束的时序关联。** 使用近期鲁棒速度预测下一帧中心，在局部 ROI 内搜索；代价联合位置创新、半径变化、颜色距离、形状证据和首帧变化证据。球半径由首帧锁定并正则化，只有出现明确形变证据时才允许显著改变，避免落地时检测框突然放大。",
        "5. **轨迹连续化但不伪造证据。** 只对两端都有可靠检测的短缺口做插值，用于展示和后续连续优化；插值点明确标记为非测量点，不计入检测覆盖率，也不能单独让拟合通过。",
        "6. **从 2D 到物理平面。** Side 视角近似正交且冻结实验的球心位于已知运动平面。对每个接受的像素中心，用固定 K/R/t 形成相机射线并与该平面求交；再以已知球半径和首帧投影尺度约束做小范围几何细化，得到米制轨迹。",
        "7. **有效性、检测和物理拟合分层。** 先检查球体身份、尺度/形变和背景刚性；生成明显失败就停止物理拟合。可用轨迹再按实验公式独立拟合重力、恢复系数、摩擦或阻尼。检测阶段不使用参数 GT。",
        "",
        "### 为什么采用这种技术",
        "",
        "- Side 相机接近正交、相机和首帧已冻结、标准球真实尺寸已知，因此运动平面上的 2D→3D 映射在几何上可辨识；直接做通用 4D 重建会增加尺度漂移和计算成本。",
        "- 规则由可审计的颜色、形状、尺度、运动和相机几何组成，能明确解释为什么接受或拒绝一个检测；这比黑盒检测器更适合物理 benchmark 的误差审计。",
        "- 不需要额外标注训练和 GPU，可在 978 条视频上快速运行。对于 Main/Top 的显著进深或相机漂移，才切换到 SpaTrackerV2/3D 路线，避免把 Side 的简单可辨识问题过度复杂化。",
        "- 局限是严重遮挡、球体外观显著变化、相似颜色动态干扰或相机大幅运动；这些情况应进入不确定/3D 重建分支，而不是强行给出 2D 参数。",
        "",
        "## 论文正文还需要的图片",
        "",
        "1. **总流程图（正文必需）**：首帧与 prompt → 视频生成 → 生成有效性门控 → Side 2D 检测/相机漂移路由 → 米制轨迹 → 按实验拟合 → 方向、标定与跨视角指标。",
        "2. **检测方法拆解图（正文必需）**：同一帧并排展示首帧先验球、HSV/边缘候选、背景变化支持、最终接受中心与锁定半径；再给 6–8 帧时间序列轨迹。",
        "3. **强正向案例图（正文必需）**：优先 V1_B、V2_A、V2_D。每个案例同时放 3 个参数档位的关键帧、轨迹曲线、GT→拟合响应图。",
        "4. **匹配反例图（正文必需）**：使用相同版式放 1–2 个轨迹清晰但响应反向的案例，证明失败来自模型控制而不是检测器。",
        "5. **碰撞短事件图（恢复系数实验必需）**：接触前后速度、接触帧区间和拟合 e；对粘滞时间过长标为 prolonged-contact / temporal smearing，不应作为正常瞬时碰撞拟合。",
        "6. **汇总图（正文或补充）**：13 个实验的方向遵循率、归一化参数误差、可拟合率，按 baseline、in/outdoor、V1/V2/V3、view 和 seed 分层；不要只给总 pass 数。",
        "7. **失败分类图（补充材料）**：生成形变、非实验物体刚性破坏、相机漂移、检测失败、轨迹可用但参数反向、数值越界分别统计并各给一个视频/关键帧。",
        "8. **跨视角一致性图（补充材料）**：同一参数设置的 Side/Main/Top 轨迹和拟合并排；Side 作为主物理定量，Main/Top 用于鲁棒性与一致性，不把深度不可辨识误写成模型失败。",
        "",
        "## 文件索引",
        "",
        "- `selected_success_failure_cases.csv`：26 个 matched 组合及完整数值。",
        "- `fig1_success_failure_matrix.png`：13×2 证据矩阵。",
        "- `fig2_success_response_small_multiples.png` / `fig3_failure_response_small_multiples.png`：逐实验响应曲线。",
        "- `cases/<experiment>/success_overlay_montage.mp4` / `failure_overlay_montage.mp4`：检测轨迹演示。",
        "- `cases/<experiment>/*_response.png`：单实验论文作图底稿。",
    ])
    missing = [key for key, status in montage_status.items() if not status.get("written")]
    if missing:
        lines.extend(["", "## 未生成的视频拼图", "", *[f"- {experiment} {case}" for experiment, case in missing]])
    return "\n".join(lines) + "\n"


def render(selected_csv: Path, eval_root: Path, output: Path) -> dict[str, Any]:
    rows = _read_csv(selected_csv)
    order = {experiment: index for index, experiment in enumerate(EXPERIMENT_ORDER)}
    rows.sort(key=lambda row: (order[row["experiment_id"]], 0 if row["case_type"] == "success" else 1))
    output.mkdir(parents=True, exist_ok=True)

    plot_images: dict[str, list[np.ndarray]] = {"success": [], "failure": []}
    montage_status: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        case_dir = output / "cases" / row["experiment_id"]
        image = _draw_response_plot(row, case_dir / f"{row['case_type']}_response.png")
        plot_images[row["case_type"]].append(image)
        montage_status[(row["experiment_id"], row["case_type"])] = _write_montage(
            row,
            eval_root,
            case_dir / f"{row['case_type']}_overlay_montage.mp4",
        )

    _make_matrix(rows, output / "fig1_success_failure_matrix.png")
    _make_grid(plot_images["success"], "Positive-direction matched interventions", output / "fig2_success_response_small_multiples.png")
    _make_grid(plot_images["failure"], "Matched wrong-direction / flat counterexamples", output / "fig3_failure_response_small_multiples.png")
    (output / "PARAMETER_CASEBOOK_ZH.md").write_text(_report(rows, output, montage_status), encoding="utf-8")
    scope = benchmark_scope_summary()
    summary = {
        "experiment_system_count": scope["experiment_system_count"],
        "parameter_response_channel_count": scope["parameter_response_channel_count"],
        "counting_rule_zh": scope["counting_rule_zh"],
        "selected_case_count": len(rows),
        "montage_written_count": sum(bool(status.get("written")) for status in montage_status.values()),
        "montage_missing_count": sum(not bool(status.get("written")) for status in montage_status.values()),
        "success_evidence_counts": dict(Counter(row["evidence"] for row in rows if row["case_type"] == "success")),
        "failure_evidence_counts": dict(Counter(row["evidence"] for row in rows if row["case_type"] == "failure")),
        "report": str(output / "PARAMETER_CASEBOOK_ZH.md"),
    }
    (output / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.selected_csv.resolve(), args.eval_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
