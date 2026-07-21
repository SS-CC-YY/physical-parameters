# 固定视角：小球轨迹 → 物理参数 → 参数分数

这条 evaluator 只实现当前限定目标：

1. 假设生成视频中的相机内参和外参在所有帧保持不变；
2. 只评估 `standard_ball`；
3. 使用 Blender 第 1 帧 GT 的球心、物理半径、投影半径和相机 `K/R/t`；
4. 从生成视频逐帧跟踪球心与可见半径；
5. 恢复 metric world trajectory；
6. 仅从轨迹反演各实验参数；
7. 反演结束后才读取 registry 中的 tuple GT 并评分。

本文件重点说明固定相机的几何恢复和 13 个逆问题。正式 978 流程会在它之前增加相机路由和视频生成有效性 gate；相机漂移走 SpaTrackerV2，明显形变或证据不足时不调用本文件所述的参数拟合器。完整调度见 `code/docs/seedance978_physics_evaluation.md`。

## 1. 标定数据

通用导出器：

```text
code/scripts/export_fixed_camera_ball_calibration.py
```

它以只读方式打开 registry 中 13 个实验、9 个场景的 `.blend`，导出三台相机，共 `13×9×3=351` 个 sidecar：

```text
code/assets/fixed_camera_ball_calibration/
  manifest.json
  v1_A/baseline/CAM_Main.json
  ...
  v3_D/outdoor4/CAM_Top.json
```

每个 sidecar 包含：

- 源图尺寸；
- `K`；
- OpenCV 约定的 world-to-camera `R/t`；
- 第 1 帧标准球世界坐标和物理半径；
- 第 1 帧球体 mesh 投影 bbox、中心和像素半径；
- 摆锤实验的 pivot ancestor；
- 来源 `.blend` 的 SHA-256。

导出命令：

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe' `
  --background `
  --python .\code\scripts\export_fixed_camera_ball_calibration.py `
  -- `
  --workspace-root . `
  --output-dir .\code\assets\fixed_camera_ball_calibration
```

导出器不会调用 Blender save operator，并在写出结果前核对所有源 `.blend` 的前后哈希。

## 2. 轨迹恢复

若源 GT 图像与生成视频分辨率不同，先使用中心裁剪 + resize 矩阵：

```text
K_video = H_crop_resize K_source
```

第 0 个生成帧的球心与 GT 投影中心之差被建模为一个固定二维平移；它只修正一次，后续所有帧都使用同一个 `K_video/R/t`。

首帧 GT 像素半径记为 `r0_gt`，首帧检测半径记为 `r0_obs`。后续检测半径先做：

```text
r_t = r_t_obs * r0_gt / r0_obs
```

因此首帧分割轮廓的系统偏差不会改变 GT 尺度。再令首帧 GT 球心的 camera depth 为 `Z0`：

```text
sphere_depth_scale = Z0 * r0_gt
r_pred(X) = sphere_depth_scale / Z_camera(X)
```

冻结的 13 个实验都在固定世界平面内运动。主初始化由像素中心射线与首帧运动平面相交获得；随后在固定平面上优化 `x/z`，残差同时包含：

```text
project(K,R,t,X) - observed_center_uv
r_pred(X) - observed_radius
```

V1A/V1B/V1C/V2B/V2C/V2E/V3B 进一步投影到已知直线；V1D/V2D/V3D 投影到已知摆锤圆弧；V2A/V3A/V3C 保留运动平面中的二维轨迹。

逐帧 CSV 至少包含：

```text
frame_index,time_s,
center_u_px,center_v_px,measurement_radius_px,track_confidence,
x_m,y_m,z_m,q_value,theta_rad,
center_reprojection_residual_px,radius_residual_px,
measurement_valid,physics_fit_used
```

时间一律来自解码后 MP4 的真实 fps，不使用 Blender source fps 替代。

## 3. 13 个逆问题

| 实验 | 轨迹反演参数 | 主方法 |
|---|---|---|
| V1A | `gravity_g` | 空中段 robust quadratic |
| V1B | `restitution_e` | 单次撞墙前后速度比 |
| V1C | `kinetic_friction_mu` | 停止前二次减速 |
| V1D | `amplitude_decay_beta` | 固定周期的 bounded decay search |
| V2A | `gravity_g` | 三阶谐波周期搜索，`g=16π²a/T²` |
| V2B | `restitution_e` | 多段直线速度比的几何平均 |
| V2C | `mu_A, mu_B` | 以 `x=0` 分开的两段二次减速 |
| V2D | `gravity_g, linear_damping_beta` | 双重积分 robust regression |
| V2E | `gravity_g, restitution_e` | 多飞行段共享曲率 + 撞击速度比 |
| V3A | `g, drag_beta, e` | 水平指数衰减 + 带 drag 的共享飞行段 |
| V3B | `mu_k, e_L, e_R` | 共享段减速度 + 左右墙速度比 |
| V3C | `g, mu, e` | ramp/floor 加速度分解 + 墙撞速度比 |
| V3D | `g, damping_beta, magnetic_kappa` | 三基函数双重积分 robust regression |

拟合器的输入只有 metric trajectory、registry 参数范围和公开已知常量。tuple target 不会传给 tracker、事件检测或 fitter。

## 4. Metrics

对每个隐藏参数，使用 frozen registry 的合法范围：

```text
NAE_p = abs(p_hat - p_gt) / (p_max - p_min)
score_p = 100 * max(0, 1 - NAE_p)
```

原始估计值不裁剪到合法范围。一个视频的实验分数为其所有隐藏参数的 macro mean：

```text
experiment_nmae = mean(scoring_NAE_p)
experiment_score = 100 * max(0, 1 - experiment_nmae)
```

参数无法观测时：

```text
estimate_raw = null
raw_NAE = null
scoring_NAE = 1
score = 0
```

全数据集先在实验内平均，再对 13 个实验等权 macro average，避免视频数量或参数数量较多的实验主导总分。

## 5. 运行

13 实验各取一条视频做 smoke test：

```powershell
python .\code\scripts\run_fixed_camera_physics_evaluation.py `
  --videos .\seedanceVideos.tar\videos `
  --output .\analysis\fixed_camera_physics_smoke13 `
  --glob 'v?_*__*__baseline__standard_ball__CAM_Side__*.mp4' `
  --one-per-experiment `
  --overwrite
```

运行所有标准球视频：

```powershell
python .\code\scripts\run_fixed_camera_physics_evaluation.py `
  --videos .\seedanceVideos.tar\videos `
  --output .\analysis\fixed_camera_physics_all `
  --glob '*.mp4'
```

已有 `result.json` 默认复用；需要重算时传 `--overwrite`。

输出：

```text
<output>/
  aggregate.json
  summary.csv
  <video_stem>/
    trajectory.csv
    result.json
```

`result.json` 中分别保存视频时间轴、追踪覆盖率、实际使用的 `K_video`、首帧尺度归一化、参数估计、诊断量、参数 GT 与 NAE/score。
