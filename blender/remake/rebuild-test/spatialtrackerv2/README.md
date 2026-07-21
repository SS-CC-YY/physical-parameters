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

上游源码固定于官方 `main` 当前提交 `7e12274c52077860cebfe007a6290777db43b63c`，并保持原文件不修改。模型权重体积较大，不进入 Git；`setup_server.sh` 会把 Hugging Face 缓存放到本目录的 `models/huggingface`。本 benchmark 的显式球体查询、批处理和米制对齐全部位于外层 `scripts/` 中。

服务器安装使用 `requirements-inference.txt`，只包含 RGB 推理、轨迹对齐和可视化依赖。上游完整依赖中的 Gradio、SAM 和 Ray 不参与本实验，因此不会安装。上游锁定的 `utils3d` 已按原提交放入 `third_party/utils3d`，安装过程不再需要从 GitHub 克隆依赖。

## 混合重建方法

默认采用“先审计相机，再选择重建器”，而不是让 27 条视频全部运行重模型：

1. `fixed` / `no_significant_camera_change`：冻结 PNG 只在第 0 帧配准一次；随后全视频固定使用该视角的 Blender 内外参。V1A 球心被限制在实验定义的竖直运动线上，并且每一帧检测半径必须落在已知 Blender 球半径投影的 `0.65–1.35` 倍内，否则该帧不能进入米制轨迹和物理拟合。
2. `changed` / `camera_changed`：使用 SpatialTrackerV2 的时序 3D tracks、静态锚点和逐帧 Sim(3) 对齐补偿视角漂移。
3. `review`、`borderline_below_threshold`、审计错误或缺少审计证据：保守地进入 SpatialTrackerV2，绝不在不能确认固定时套用静态相机参数。

因此，球尺寸是静态分支的硬门槛，不是把检测框强行拉成固定大小。可视化框仍保持固定大小，但物理测量使用真实检测圆与已知球半径的投影一致性。

SpatialTrackerV2 动态分支的具体方法如下。

每条视频使用两类查询点：

1. 球体内部查询点：从冻结首帧的标定投影框生成，ORB 将冻结首帧配准到生成视频的第 0 帧。
2. 静态场景锚点：复用 Blender 冻结首帧导出的 `uv_px ↔ xyz_world_m` 对应关系。

SpatialTrackerV2 输出所有查询点的 3D 时序。每一帧用静态锚点做鲁棒 Sim(3) 米制对齐，从而同时校正单目尺度、坐标轴和相机漂移；球体中心则用“首帧相机射线与已知半径球面求交 + 逐帧刚体配准”恢复。输出是视频自身的轨迹和质量指标，不计算 GT 误差。

两条分支不会把 Main/Side/Top 当作同步多机位：三条视频是独立生成样本，各自重建后再比较结果。

## 本地/服务器运行

推荐在带 CUDA 的 Linux 服务器上建立独立环境：

```bash
cd /path/to/physical-parameters/blender/remake/rebuild-test/spatialtrackerv2
bash scripts/setup_server.sh
conda activate "$PWD/env"

python scripts/build_manifest.py --strict
python scripts/preflight.py
```

安装脚本可以安全重复执行：若此前在下载依赖时中断，它会复用 `env/` 中已经成功安装的 PyTorch，仅补齐剩余推理依赖和模型文件。

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

推荐使用自动分流脚本。它会先审计 27 条视频；静态视频走快速标定分支，只有漂移或不确定视频才加载 SpatialTrackerV2：

```bash
VIDEOS_ROOT=/path/to/seedance_run/videos \
GPU_ID=0 \
MAX_JOBS=1 \
bash scripts/run_hybrid_27_server.sh
```

确认 canary 后去掉 `MAX_JOBS=1`。如果已有完整的 978 条最终相机审计结果，可直接复用，避免重复审计：

```bash
VIDEOS_ROOT=/path/to/seedance_run/videos \
AUDIT_JSONL=/path/to/final_camera_motion_results.jsonl \
GPU_ID=0 \
bash scripts/run_hybrid_27_server.sh
```

先只查看分流、不执行重建：

```bash
python scripts/run_hybrid_v1a.py \
  --videos /path/to/seedance_run/videos \
  --manifest work/v1a_hybrid_manifest.jsonl \
  --audit-jsonl /path/to/final_camera_motion_results.jsonl \
  --output-root outputs/v1a_hybrid \
  --dry-run
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

混合流程额外输出：

- `outputs/v1a_hybrid/routing_manifest.jsonl`：逐视频审计证据、选择分支和原因；
- `outputs/v1a_hybrid/static/`：标定 + 球尺寸硬约束的原生结果；
- `outputs/v1a_hybrid/dynamic/`：SpatialTrackerV2 原生结果；
- `outputs/v1a_hybrid/normalized/<job_id>/trajectory_world.csv`：两条分支统一后的轨迹字段；
- `outputs/v1a_hybrid/hybrid_summary.csv`：整批状态和质量摘要。
