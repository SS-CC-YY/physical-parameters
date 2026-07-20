# SpatialTrackerV2：V1A 三视角轨迹重建可行性测试

这个目录用于测试 **模型生成视频本身** 的三维物体轨迹，不与 6 条 Blender 真值视频比较。首轮固定为：

- 实验：`v1_A`，标准球，`g=-9.81 m/s²`，seed `341867882`
- 场景：baseline、indoor1–4、outdoor1–4（9 个）
- 视角：CAM_Main、CAM_Side、CAM_Top（3 个）
- 总计：27 条 Seedance 2.0 MP4

## 目录

```text
spatialtrackerv2/
├── upstream/SpaTrackerV2/   # 固定的上游源码快照
├── models/                  # 运行时下载模型（Git 忽略）
├── config/                  # 27 条测试的固定配置
├── manifests/               # 输入清单
├── scripts/                 # 查询点、批处理、重建后处理
├── outputs/                 # 每条视频的轨迹与可视化（Git 忽略）
└── work/                    # 中间查询文件（Git 忽略）
```

上游源码固定于提交 `7e12274c52077860cebfe007a6290777db43b63c`。模型权重体积较大，不进入 Git；`setup_server.sh` 会把 Hugging Face 缓存放到本目录的 `models/huggingface`。

## 重建方法

每条视频使用两类查询点：

1. 球体内部查询点：从冻结首帧的标定投影框生成，ORB 将冻结首帧配准到生成视频的第 0 帧。
2. 静态场景锚点：复用 Blender 冻结首帧导出的 `uv_px ↔ xyz_world_m` 对应关系。

SpatialTrackerV2 输出所有查询点的 3D 时序。每一帧用静态锚点做鲁棒 Sim(3) 米制对齐，从而同时校正单目尺度、坐标轴和相机漂移；球体中心则用“首帧相机射线与已知半径球面求交 + 逐帧刚体配准”恢复。输出是视频自身的轨迹和质量指标，不计算 GT 误差。

## 本地/服务器运行

推荐在带 CUDA 的 Linux 服务器上建立独立环境：

```bash
cd /path/to/physical-parameters/blender/remake/rebuild-test/spatialtrackerv2
bash scripts/setup_server.sh
conda activate "$PWD/env"

python scripts/build_manifest.py --strict
python scripts/preflight.py
```

如果服务器上的 Seedance 视频不在默认的 `remake/seedanceVideos.tar/videos`，构建清单时显式指定现有输出目录：

```bash
python scripts/build_manifest.py \
  --videos-root /path/to/seedance_run/videos \
  --strict
```

先跑一条 canary，并保留完整 24 fps 时序：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_batch.py --max-jobs 1
```

确认 `outputs/v1a_seedance27/<job_id>/trajectory_3d.png` 和 `object_track_overlay.mp4` 后跑完 27 条：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_batch.py
```

也可以让包装脚本同时设置视频目录、GPU 和任务数：

```bash
VIDEOS_ROOT=/path/to/seedance_run/videos GPU_ID=0 MAX_JOBS=1 bash scripts/run_27_server.sh
```

默认批处理只加载一次模型并连续处理视频。如果某一视频导致持久进程异常，可给该任务加 `--job-id <完整任务名> --isolated-process --rerun` 单独排查。若只想快速检查模型是否可用，可加 `--frame-stride 2`；正式轨迹测试请使用默认 `1`。批处理可断点续跑，已经存在 `result.json` 且状态为 `succeeded` 的任务会跳过。

## 每条视频输出

- `raw_spatialtrackerv2.npz`：SpatialTrackerV2 原始 2D/3D tracks、可见度、相机和深度
- `trajectory_world.csv`：`frame, time_s, x_m, y_m, z_m` 与有效点/锚点统计
- `trajectory_3d.png`：世界坐标三维轨迹图及三个分量随时间曲线
- `object_track_overlay.mp4`：生成视频上叠加球体查询点与中心轨迹
- `alignment.json`：逐帧 Sim(3) 内点数和残差
- `result.json`：运行状态、耗时和质量摘要

批次级结果位于 `outputs/v1a_seedance27/summary.csv`、`summary_by_scene.csv` 和 `summary.json`；其中 `summary_by_scene.csv` 会把每个场景的三个视角放在同一行。这里先验证“能否稳定恢复三维轨迹”；自由落体参数与轨迹公式拟合应在轨迹质量通过后作为独立评估层接入。
