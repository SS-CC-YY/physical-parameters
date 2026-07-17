# V1_A 三视角物理门控、轨迹可视化与参数拟合

## 1. 严格执行顺序

本评估流程当前暂停，不由正式生成脚本自动调用。所有目标模型的完整视频生成结束后，再对已经固化的 canonical outputs 执行以下流程：

```text
Wan2.2 I2V 生成
  → 视频完整性检查
  → 实验物体逐帧检测与跟踪
  → 跟踪完整性门控
  → 独立输出刚性穿模判定
  → 跟踪可用时独立拟合 airborne 参数（穿模不阻断拟合）
  → 轨迹 CSV + 路线图 + 抽样标注视频 + 累计 HTML 报告
  → 下一个 job
```

消失、瞬移或严重误跟踪会阻断参数拟合；落地后的刚性穿模不会阻断此前 airborne 区间的参数拟合。每个样本分别输出 `rigid_body_evaluation` 和 `parameter_evaluation`，避免把“是否穿模”与“参数是否接近 prompt 给定值”混成一个结论。

## 2. 三个视角的可辨识边界

- `CAM_Side`：使用已知世界落差和检测到的图像端点位移换算 `m/s²`，是三者中最可靠的视角。
- `CAM_Main`：同样输出参数估计和相似度，但 endpoint scale 受透视影响，属于近似值。
- `CAM_Top`：同样完整输出参数估计和相似度；在完整相机内外参可用前，也标记为 projective approximation。

V1_A 已知 `z0=4.2 m`、`zc=0.44 m`，因此 `drop_distance=3.76 m` 已进入 canonical `known_params`。非侧视结果不会缺失，但会用 `endpoint_calibrated_projective_approximation` 明确区分于侧视标定。

## 3. 检测与轨迹记录

conditioning image 首先读取同一场景、同一视角下的四种物体首帧。由于这些图像的背景完全相同，跨物体像素变化区域可作为强前景先验，将目标初始化限制在真实实验物体附近；缺少同组首帧时才回退到中央空间先验。随后在该小区域内使用物体类别相关的颜色与连通域规则确定精确目标框。排球只使用其高饱和蓝色/黄色纹理，不再使用容易把黑板、粉笔线、海报和植被纳入的通用边缘/饱和度掩膜。逐帧跟踪结合：

1. 第一帧目标前景 mask 约束的归一化相关匹配，避免静态背景主导模板分数；
2. 相邻帧灰度差分的运动连通域；
3. 上一位置、搜索带和预期向下运动的空间约束。

前 3 个远离支撑面的可信颜色连通域以宽、高中位数建立物体尺寸模板，此后锁定模板，不再用后续候选框更新参考尺寸。普通帧的宽、高最多扩大到模板的 1.10 倍，最多缩小到 0.85 倍；接近支撑面时直接使用模板宽、高，仅跟随检测中心。因此落地瞬间即使物体掩膜与木板合并，检测框也不会随连通域突然扩大。

只有满足下面全部条件才把尺寸锁定临时释放为“明显形变”：

1. 宽、高相对模板至少有一边变化 22%；
2. 宽、高沿相反方向变化（压扁或拉伸，而不是两边一起扩大）；
3. 候选框面积保持在模板面积的 `1/1.30` 到 `1.30` 倍；
4. 上述证据连续至少 3 帧出现。

这样可将单帧扩框、木板粘连与真实的持续压扁/拉伸区分开。标准橙色球还使用高饱和度核心掩膜，从颜色上排除低饱和度木板。

每帧记录：

```text
frame_index, time_s, found, tracking_source, score, template_score,
center_x_px, center_y_px,
bbox_x0, bbox_y0, bbox_x1, bbox_y1,
bbox_width_px, bbox_height_px, bbox_area_px2, bbox_aspect_ratio,
refinement_rejected, refinement_area_ratio, near_support,
bbox_size_locked, bbox_size_constrained,
bbox_reference_width_px, bbox_reference_height_px,
deformation_candidate, deformation_candidate_run, deformation_confirmed,
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

透视图中，物体框底部与木板上表面发生投影重叠是正常现象，因此不再用“bbox bottom 超过木板上边界”直接宣布穿模。该量仅保留为 `max_penetration_depth_object_heights` 诊断值。

刚性穿模采用两个高置信证据。第一，提取支撑板整体的下边界 `y_support_bottom(x)`，检查物体中心是否已经越过整个支撑板：

```math
r_{bottom,i}=\frac{y_{center,i}-y_{support,bottom}(x_i)}{h_{obj,i}}
```

第二，用末尾 20% 稳定轨迹的中位中心 `y_settled` 作为最终接触位置，检查接触过程中是否先向下穿入、随后回到终态：

```math
r_{overshoot,i}=\frac{y_{center,i}-y_{settled}}{h_{obj,i}}
```

连续至少 3 帧满足 `r_bottom > 0.25` 或 `r_overshoot > 0.45` 时，`rigid_body_evaluation.status=violation` 且 `penetration_detected=true`。无法可靠分割支撑板或跟踪本身失败时输出 `indeterminate`，不会把证据不足当成“未穿模”。

## 5. Airborne 区间

跟踪完整性门控通过后进行拟合；`rigid_body_evaluation=violation` 不会阻断 airborne 拟合：

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

V1_A 已知世界落差：

```math
D=z_0-z_c=4.2-0.44=3.76\;m
```

检测首帧中心与落地稳定中心的图像位移 `Δy_px`，得到 endpoint effective scale：

```math
s_{eff}=\frac{\Delta y_{px}}{D}
```

再反推：

```math
g_{est}=\frac{vertical\_acceleration\_px\_s2}{s_{eff}}
```

给定参数与反推参数的对称相似度为：

```math
similarity=\frac{min(g_{est},g_{target})}{max(g_{est},g_{target})}\in[0,1]
```

完全一致为 1；非正估计为 0。同时报告 `relative_error=|g_est-g_target|/g_target`。CAM_Main/CAM_Top 的 endpoint scale 受透视影响，所以是明确标注的近似。严格恢复仍需要用相机内参 K、外参 `[R|t]` 与运动平面反投影到世界坐标：

```math
Y_{world}(t)=Y_0+V_0(t-t_0)+\frac{1}{2}g(t-t_0)^2
```

## 7. 状态解释

- `rigid_body_evaluation.status=pass/violation/indeterminate`：独立的刚性穿模结论。
- `parameter_evaluation.status=ok/invalid/unavailable`：独立的参数反推结论，包含 target、estimate、similarity、relative error 和 calibration method。
- `fit_status=fitted_image_plane_proxy`：跟踪可用并完成二维二次拟合，即使落地后检测到穿模也可以成立。
- `fit_status=skipped_tracking_gate`：消失、瞬移或严重误跟踪导致拟合不可用。
- `low_fit_r2` / `nonpositive_parameter_estimate`：参数估计不满足自动接受标准。
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
4. `rigid_body_evaluation` 的 pass/violation/indeterminate 是否与视频一致；
5. 红色拟合曲线是否只覆盖 airborne 段，且穿模结论没有错误阻断拟合；
6. target、estimated gravity、similarity 与 relative error 是否完整；
7. Main/Top 是否标记为 endpoint projective approximation。

跨样本的 prompt 重力与 `vertical_acceleration_px_s2` 相关性按视角分别报告。由于当前 `object_values` 把四种物体固定绑定四个重力值，这一相关性仍与物体身份混杂，只能作为 demo/smoke 证据，不是严格因果结论。
