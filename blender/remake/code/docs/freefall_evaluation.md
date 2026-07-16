# V1_A 三视角物理门控、轨迹可视化与参数拟合

## 1. 严格执行顺序

每个 canonical job 独立执行以下流程，完成评估后才生成下一个 job：

```text
Wan2.2 I2V 生成
  → 视频完整性检查
  → 实验物体逐帧检测与跟踪
  → 严重物理/几何异常门控
  → 门控通过后选择 airborne 区间并拟合
  → 轨迹 CSV + 路线图 + 抽样标注视频 + 累计 HTML 报告
  → 下一个 job
```

严重门控失败时仍保存轨迹与可视化，但跳过参数拟合，避免用已经穿模、瞬移或识别错误的轨迹反推物理量。

## 2. 三个视角的可辨识边界

- `CAM_Side`：最适合二维竖直运动拟合。没有尺度标定时仍只报告 `px/s²`；提供可靠 `pixels_per_meter` 后才可换算 `m/s²`。
- `CAM_Main`：完整检测、门控和图像平面拟合，但存在透视。单个全局像素尺度不足以恢复真实重力。
- `CAM_Top`：完整检测、门控和图像平面拟合。其投影同时受相机姿态、透视/深度变化影响，必须有相机内外参、支撑平面和深度或单应性标定后才能恢复真实重力。

因此非侧视结果不是“缺少流程”，而是完整执行后把 `parameter_identifiability` 标为 `image_plane_only_projective_view`，不会伪造 `estimated_gravity_m_s2`。

## 3. 检测与轨迹记录

conditioning image 用物体类别相关的颜色、亮度、边缘与连通域规则定位首帧模板。逐帧跟踪结合：

1. 第一帧模板的归一化相关匹配；
2. 相邻帧灰度差分的运动连通域；
3. 上一位置、搜索带和预期向下运动的空间约束。

每帧记录：

```text
frame_index, time_s, found, tracking_source, score, template_score,
center_x_px, center_y_px,
bbox_x0, bbox_y0, bbox_x1, bbox_y1,
bbox_width_px, bbox_height_px, bbox_area_px2, bbox_aspect_ratio,
center_y_smoothed_px, fit_used
```

时间与图像坐标为：

```math
t_i = i / fps
```

OpenCV 原点在左上角，x 向右、y 向下，所以正常自由落体的投影通常表现为 y 增加。

## 4. 参数拟合前的严重异常门控

门控是可审计的二维启发式检查，不等价于三维碰撞引擎证明。所有阈值写在 `configs/evaluations/basic_video_v1.yaml`，每个样本的实测量写入 JSONL/CSV。

### 4.1 目标消失

```math
detection\_rate = N_{found}/N_{frames}
```

低于 0.70 标记 `severe_low_detection_rate`。最长连续丢失超过总帧数的 18% 标记 `severe_object_disappearance`。

### 4.2 瞬移

相邻检测中心位移：

```math
\Delta p_i = \lVert (x_i,y_i)-(x_{i-1},y_{i-1}) \rVert_2
```

用检测框中位对角线 `d_obj` 归一化。若：

```math
max_i(\Delta p_i/d_{obj}) > 2.5
```

标记 `severe_teleportation`。

### 4.3 尺度突变与严重形变

使用检测框面积和长宽比的稳健分位数比：

```math
r_A = P_{95}(A)/P_{5}(A)
r_R = P_{95}(w/h)/P_{5}(w/h)
```

`r_A > 3.0` 标记 `severe_scale_change`；`r_R > 2.5` 标记 `severe_object_deformation`。

### 4.4 出画与紊乱路线

检测框接触画面边缘的帧占比超过 10% 标记 `severe_out_of_frame`。

对所有中心点做 PCA，计算到主运动直线的正交距离 `e_i`。若：

```math
P_{95}(|e_i|)/d_{obj} > 1.5
```

标记 `severe_erratic_trajectory`。

### 4.5 支撑板穿透

从 conditioning image 下半部的大型棕色连通域提取支撑板上边界 `y_surface(x)`。每帧穿透深度为：

```math
d_i = y_{bbox,bottom,i} - y_{surface}(x_i)
```

再用该帧物体框高度归一化：

```math
q_i = d_i/h_{obj,i}
```

连续至少 3 帧满足 `q_i > 0.60` 时标记 `severe_support_penetration`。若图像中无法可靠分割支撑板，`support_surface_detected=false`，不会凭空判定穿模；此时需结合 overlay 人工核验。

## 5. Airborne 区间

门控通过后才进行拟合：

1. 删除未检测帧；
2. 对 `center_y_px` 做默认 5 帧移动平均；
3. 用平滑 y 的第 92 百分位估计落地后中心 `y_floor`；
4. 物体框中位高度为 `h_obj`；
5. airborne 候选满足：

```math
y_{smooth}(t) < y_{floor} - max(6 px, 0.2 h_{obj})
```

6. 仅使用检测序列前 72%，降低接触/静止段污染；
7. 至少需要 12 点。若候选不足则使用早期窗口并标记 `airborne_interval_fallback`。

## 6. 轨迹公式和物理量

以首个拟合点时间 `t0` 为零点，最小二乘拟合：

```math
y_{px}(t)=c_0+c_1(t-t_0)+c_2(t-t_0)^2
```

一阶、二阶导数：

```math
v_{px}(t)=c_1+2c_2(t-t_0)
a_{px}(t)=2c_2
```

输出对应关系：

```text
intercept_c0_px               = c0
linear_c1_px_s                = c1
quadratic_c2_px_s2            = c2
vertical_acceleration_px_s2   = 2*c2
```

拟合优度与误差：

```math
R^2 = 1 - \frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
```

```math
RMSE_{px}=\sqrt{\frac{1}{N}\sum_i(y_i-\hat y_i)^2}
```

若且仅若 `CAM_Side` 有可靠尺度 `s=pixels_per_meter`：

```math
g_{est}=\frac{vertical\_acceleration\_px\_s2}{s}
```

透视视角的严格恢复需要先用相机内参 K、外参 `[R|t]` 与已知运动/支撑平面把像素点反投影到世界坐标 `Y_world(t)`，再拟合：

```math
Y_{world}(t)=Y_0+V_0(t-t_0)+\frac{1}{2}g(t-t_0)^2
```

当前数据未提供足够标定，因此 Main/Top 的 metric g 明确不可辨识。

## 7. 状态解释

- `severe_gate_passed=false`：严重异常或检测证据不足；保存可视化，跳过拟合，样本为 `invalid`。
- `fit_status=fitted_image_plane_proxy`：门控通过并得到二维二次拟合。
- `fit_status=skipped_severe_gate`：门控失败，没有用异常轨迹拟合。
- `low_fit_r2`：门控通过，但二次模型解释度不足。
- `airborne_interval_fallback`：落地前区间不够明确，需要人工检查。
- `overlay_write_failed`：仅可视化编码失败；轨迹和数值仍保留，但样本标为需要检查。

`invalid` 是“当前证据不满足自动接受标准”，不是自动宣称模型一定违反物理。应查看相应轨迹图与 overlay。

## 8. 输出与人工核验

```text
outputs/<run_id>/eval/
├── sample_metrics.jsonl
├── aggregate.json
├── freefall/
│   ├── summary.csv
│   ├── aggregate.json
│   ├── tracks/<job_id>.csv
│   ├── plots/<job_id>.png
│   └── overlays/<selected_job_id>.mp4
└── report/index.html
```

每个 job 都有轨迹 CSV 和路线图。9 条 overlay 按 camera × scene 分层抽取，用检测框、中心点、轨迹尾迹和门控警告直接确认是否识别到真实实验物体。

人工检查顺序：

1. 框是否始终包围同一实验物体，是否误跟踪地板/阴影；
2. 轨迹是否连续，是否出现复制、瞬移、消失或出画；
3. 物体框尺度与形状是否突然改变；
4. 落地后是否穿过支撑板；
5. 门控通过时，红色拟合曲线是否只覆盖 airborne 段；
6. Main/Top 是否只报告投影代理，未输出伪精确 m/s²。

跨样本的 prompt 重力与 `vertical_acceleration_px_s2` 相关性按视角分别报告。由于当前 `object_values` 把四种物体固定绑定四个重力值，这一相关性仍与物体身份混杂，只能作为 demo/smoke 证据，不是严格因果结论。
