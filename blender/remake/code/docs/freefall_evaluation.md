# V1_A 自由落体检测、可视化与物理拟合

## 1. 评估目标与边界

当前 `v1a_freefall` evaluator 使用 `CAM_Side` 生成视频，检测实验物体的图像中心并拟合竖直运动。它输出两类结果：

1. 可核验的检测结果：逐帧轨迹 CSV、轨迹路线图、带检测框/中心点/轨迹尾迹的抽样视频；
2. 物理运动代理：图像平面的竖直加速度 `vertical_acceleration_px_s2`。

默认没有相机尺度标定，因此拟合结果单位是 `px/s²`，不是 `m/s²`。只有提供可靠的 `pixels_per_meter` 后才会输出 `estimated_gravity_m_s2`。

## 2. 输出文件

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

每个任务都有轨迹 CSV 和路线图。默认从五个场景各选择一个代表样本，共生成五个 overlay 视频；选择由 evaluation profile 中的 `selection_seed` 固定。

## 3. 逐帧物体检测

### 3.1 初始区域

根据 conditioning image 和 `object_id` 在图像上方中央区域寻找目标：

- standard ball：橙色 HSV 范围；
- standard cube：蓝色 HSV 范围；
- cardboard box：棕色 HSV 范围；
- volleyball：饱和度、亮度与边缘组合。

候选连通域根据面积、长宽比、填充率以及与预期起始位置的距离评分。检测失败时使用保守的中心上方区域作为模板框，并在可视化中通过后续低置信度/低检测率暴露问题。

### 3.2 Hybrid tracking

每一帧同时考虑：

- Template matching：将第一帧目标模板在中央竖直搜索带中进行归一化相关匹配；
- Motion detection：对相邻帧灰度差分、阈值化和形态学处理，选择最接近上一位置且符合向下运动的连通域。

`tracker: auto` 优先使用 motion detection；没有合格运动区域时回退到 template matching。轨迹 CSV 保存：

```text
frame_index, time_s, found, tracking_source, score,
center_x_px, center_y_px, bbox_x0, bbox_y0, bbox_x1, bbox_y1,
center_y_smoothed_px, fit_used
```

## 4. 坐标与时间

视频帧率为 `fps`，第 `i` 帧时间为：

```math
t_i = i / fps
```

OpenCV 图像坐标的原点在左上角：

- x 向右为正；
- y 向下为正。

因此自由落体对应 `y_px` 随时间增加，拟合得到的向下加速度通常为正。

## 5. 拟合区间选择

不应把物体接触地面后的静止帧放入自由落体拟合。当前流程为：

1. 删除没有检测到物体的帧；
2. 对检测中心 `y_px` 做奇数窗口移动平均，默认窗口 5；
3. 用平滑轨迹的第 92 百分位估计地面静止中心位置 `y_floor`；
4. 设物体检测框中位高度为 `h_obj`；
5. 将满足下式的早期帧作为 airborne 候选：

```math
y_{smooth}(t) < y_{floor} - max(6 px, 0.2 h_{obj})
```

6. 只在检测序列前 72% 中选取，降低接触地面和模板漂移进入拟合的概率；
7. 至少需要 12 个拟合点。若 airborne 选择不足，则使用早期窗口回退，并标记 `airborne_interval_fallback`，该样本默认视为需要人工检查。

## 6. 二次曲线与加速度

理想匀加速运动为：

```math
y(t) = y_0 + v_{0,y}t + 1/2 a_y t^2
```

对选中的图像轨迹最小二乘拟合：

```math
y_{px}(t) = c_0 + c_1 t + c_2 t^2
```

系数对应关系：

```math
y_{0,px} = c_0
v_{0,px/s} = c_1
a_{px/s^2} = 2c_2
```

因此 evaluator 输出：

```text
quadratic_c2_px_s2
linear_c1_px_s
intercept_c0_px
vertical_acceleration_px_s2 = 2 * quadratic_c2_px_s2
```

拟合优度：

```math
R^2 = 1 - sum_i (y_i - y_hat_i)^2 / sum_i (y_i - mean(y))^2
```

像素 RMSE：

```math
RMSE_px = sqrt(mean_i((y_i - y_hat_i)^2))
```

## 7. 从像素量换算到真实物理量

如果侧视相机近似正交，运动平面与成像平面平行，并且已知稳定尺度：

```math
s = pixels_per_meter
```

则：

```math
g_est_m_s2 = vertical_acceleration_px_s2 / s
```

如果使用透视相机、物体存在明显景深变化，单个全局 `pixels_per_meter` 不足以完成严格换算，需要利用相机内外参、运动平面和深度做逐点反投影。当前配置的 `pixels_per_meter` 为 `null`，所以不会输出伪精确的 `m/s²`。

## 8. 自动质量标志

- `low_detection_rate`：检测成功率低于 0.70；
- `low_fit_r2`：二次拟合 R² 低于 0.70；
- `airborne_interval_fallback`：未可靠识别落地前区间；
- `overlay_write_failed`：服务器 OpenCV/codec 无法写 overlay MP4；
- `missing_video` / `evaluator_error`：输入或评估运行失败。

`invalid` 不代表模型物理性一定差，它表示检测或拟合证据不足，必须查看轨迹图和 overlay 视频。

## 9. 汇总指标与实验设计限制

跨样本报告目标 prompt 重力与 `vertical_acceleration_px_s2` 的 Pearson/Spearman 相关，用于判断模型是否随 prompt 参数产生有序运动响应。

当前轻量 demo 使用 `object_values`：四个物体分别固定绑定四个重力值。因此物体身份与重力值混杂，相关性只能作为 smoke/demo 证据，不能作为严格的因果结论。若正式论文需要比较重力响应，建议增加 `object × gravity` 全组合或采用平衡拉丁方分配。

## 10. 如何人工核验

打开：

```text
outputs/<run_id>/eval/report/index.html
```

依次检查：

1. 绿色/黄色检测框是否始终包围真实实验物体；
2. 青色中心点和轨迹尾迹是否沿物体真实运动；
3. 轨迹路线图是否无明显跳点或背景漂移；
4. 红色拟合曲线是否覆盖真实 airborne 区间；
5. detection rate、R²、RMSE 和 quality flags 是否与视觉判断一致。
