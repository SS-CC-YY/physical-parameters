# 快速物体轨迹重建 MVP

## 这套模块解决什么问题

13 个实验不需要 13 套完全独立的重建系统。推荐结构是：

```text
共享观测前端
  MP4 解码/真实 FPS → 首帧核对 → 相机漂移估计 → 物体跟踪 → 坐标与不确定度
        ↓
实验几何 profile
  运动直线 / 平面 / 已知曲线 / 转轴 / 接触面
        ↓
物理 family 插件
  ballistic-impact / piecewise-translation / angular-oscillator / known-curve
```

当前 MVP 只实现这 20 条 Seedance 2.0 测试视频覆盖到的
`v1_A + standard_ball + CAM_Side` 快路径。它会输出可审计的侧视轨迹、PLY、轨迹图、部分标注视频和自由落体参数拟合。

## 为什么不是“从单个 MP4 重建完整场景”

这 20 条视频的背景相机基本固定。固定机位没有 SfM/NeRF 所需的真实视差，运动球还违反了“场景静态”假设；模型生成的画面漂移也不能当成可信的相机运动。因此，直接对每条视频运行完整 NeRF、3DGS 或 COLMAP，既慢又不能保证公制正确。

当前输出模式是：

```text
known_object_size_side_view_1d_approx
```

标准球已知直径为 0.48 m。在“球尺寸保持不变、球位于侧视运动平面、相机漂移可由背景仿射变换描述”的约束下，把图像轨迹提升为带尺度的 `(x, 0, z)` 轨迹。背景 RANSAC 得到的旋转、缩放和平移会以完整逆仿射作用到球心，而不只是减去平移。因为缺少完整相机 `K/T` 和独立深度观测，结果明确标为近似 2.5D，而不是严格完整 3D。

后续从准确 `.blend` 导出每个场景的 `K`、`T_world_from_camera`、运动平面/直线、球半径、地面和接触几何后，可把同一接口升级为 `metric_planar_2_5d`；同步、标定过的多视角才能升级为 `metric_3d`。

## 两个结论必须分开

- `reconstruction_validity.constrained_trajectory_valid`：球是否被可靠观测、相机补偿是否可靠、轨迹是否有足够跨度与连续性。
- `physics_adherence`：对可测轨迹拟合后，方程质量和反推参数是否接近目标。
- `reconstruction_validity.strict_metric_3d_valid`：当前数据固定为 `false`，直到完整相机标定与深度可观测性成立。

模型可能生成一条明显不符合自由落体的轨迹，但检测框仍然完全正确。这种情况下应当是“重建通过、物理失败”，不能把物理失败包装成重建失败。

## 室内复杂背景的跟踪规则

`indoor3/indoor4` 中的橙色工具、木板和地砖可能与标准球得到几乎相同的模板相关分数，因此不能把“最高模板分数”直接当作球。当前标准球前端采用以下统一规则：

- 优先关联尺寸一致、且符合最近位置/稳健速度预测的高色度连通域；
- 首帧误分到球附近的墙面或工具板支撑区域会通过初始净空检查被禁用；
- 模板候选若与地面合并，仍使用已锁定的球大小；只有低置信、尺寸受限的候选同时突然横跳时才会被拒绝，从而保留真实竖直接触并滤掉背景误锁；
- 被拒绝或掩膜形状可疑的中心不得更新下一帧的运动历史；
- 诊断图在缺失/拒绝帧处断线，后续真实反弹作为独立片段显示，不跨缺口连成假轨迹；
- 全视频有效测量比例低于 `0.65` 时，重建状态为 `invalid`，即使其中存在一小段干净轨迹。

HTML/CSV 同时报告 `full-track valid fraction`、主分量覆盖率和主轨迹横向跨度。横向跨度用于审计模型真实漂移或残余误检，但不会自动把真实水平运动隐藏成跟踪失败。

## 当前 20 条数据的实际时间轴

所有文件均为：

```text
864×496, 24 fps, 121 frames
t_i = i / 24, i=0..120
首末帧跨度 = 5.000 s
```

模块优先读取 OpenCV/FFmpeg 后端给出的逐帧 `POS_MSEC`；只有后端不提供严格递增时间戳时，才回退到容器 FPS。它不使用生成 build 中原先的 16 fps。16 fps 与 24 fps 本身不会改变物理拟合；把 24 fps 视频误按 16 fps 解释才会造成严重参数误差。分段窗口和反转速度阈值都用秒制而不是固定帧数定义。

## 本地或服务器运行

安装项目依赖后，在 `blender/remake` 根目录执行：

```bash
python code/scripts/run_reconstruction_mvp.py \
  --workspace-root "$PWD" \
  --videos rebuild-test/test-videos/videos \
  --output rebuild-test/outputs/seedance20_reconstruction_mvp \
  --workers 4 \
  --overlay-count 5
```

先做两条 smoke：

```bash
python code/scripts/run_reconstruction_mvp.py \
  --workspace-root "$PWD" \
  --videos rebuild-test/test-videos/videos \
  --output rebuild-test/outputs/seedance20_reconstruction_smoke \
  --workers 2 \
  --overlay-count 2 \
  --limit 2
```

默认配置在 `code/configs/reconstruction/object_centric_mvp_v1.yaml`。

## 输出

批次级：

```text
aggregate.json
summary.csv
results.jsonl
report/index.html
```

每条视频：

```text
<job_id>/result.json
<job_id>/video_probe.json
<job_id>/reference_alignment.json
<job_id>/camera_motion.csv
<job_id>/trajectory_world.csv
<job_id>/trajectory.ply
<job_id>/trajectory_diagnostics.png
<job_id>/reconstruction_validity.json
<job_id>/physics_adherence.json
<job_id>/track_overlay.mp4       # 仅抽样视频，默认每个场景一个
```

`trajectory_world.csv` 同时保留原始框、相机补偿坐标、测量拒绝原因、近似世界坐标与拟合使用标记，便于逐帧审计。
`result.json` 的机器可读合同位于 `code/schemas/reconstruction_result.schema.json`。

有效性门只使用同一条主轨迹分量计算有效点、覆盖率和运动跨度，同时检查整段视频的有效测量比例。最多允许配置数量的短缺帧；较远的二次误检不会被拼入 PLY 或用来凑出一次通过。物理拟合在第一个主轨迹缺口前结束，不会把落地后的重现或反弹吸收到首次下落拟合中。同一输出目录同时只允许一个批次写入；若进程异常退出留下 `.reconstruction_mvp.lock`，确认没有任务运行后只删除该锁文件即可。失败重跑会覆盖旧的 `result.json`，HTML 也不会链接旧图或旧 overlay。

## 速度与下一步验收

默认不运行 SAM2、单目深度、NeRF 或 3DGS。20 条视频建议用 4 个 CPU worker；标注视频只抽样生成，以减少编码时间。

当前 Seedance 输出只能作为真实生成视频上的 diagnostic。下一步应先用 Blender GT 视频验收：轨迹 median 误差不超过 0.03 m、p95 不超过 0.08 m、`g` 相对误差不超过 5%、接触时刻误差不超过 1 帧；然后再把生成模型的 `physics_adherence` 作为 benchmark 结果。
