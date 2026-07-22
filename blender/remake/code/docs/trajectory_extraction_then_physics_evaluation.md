# Seedance 978：先逐帧提取轨迹，再独立评估物理参数

本流程把两个科研阶段彻底分开：

```text
原始 MP4 + 冻结首帧/相机标定 + 冻结相机运动审计
    -> 阶段 A：逐帧轨迹提取
       -> trajectory_frames.csv + object_track_overlay.mp4
    -> 阶段 B：只读取冻结轨迹
       -> 轨迹公式/物理参数拟合 + 与给定参数评分
```

阶段 A 不读取目标物理参数。阶段 B 不读取原始 MP4、首帧 PNG，也不会调用 2D 检测器或 SpaTrackerV2。这样可以单独复核“物体有没有跟对”和“物理拟合是否合理”，避免把检测错误误报成模型不懂物理。

冻结的 978 条相机审计会自动分流为：

| 视角 | 标定约束 2D/米制回投影 | SpaTrackerV2 动态 3D | 总数 |
|---|---:|---:|---:|
| CAM_Side | 509 | 1 | 510 |
| CAM_Main | 233 | 1 | 234 |
| CAM_Top | 233 | 1 | 234 |
| 合计 | 975 | 3 | 978 |

只有相机位移证据达到冻结阈值且具有持续性或显著累计运动的 3 条视频才使用重模型。baseline Top 边界样本的直接和累计 P95 都支持标定 2D；indoor3 Top 的直接配准虽弱，但链式累计位移和接触表均支持动态路线。SpaTrackerV2 正式提取固定使用 `frame_stride=1`；这里的 stride 是抽帧步长，不是视频 FPS。

## 1. 每条视频的统一轨迹契约

轨迹目录固定为：

```text
<TRACK_ROOT>/
  runtime_spatialtracker_manifest.jsonl
  routing_side.jsonl / routing_main.jsonl / routing_top.jsonl
  summary.csv
  jobs/<job_id>/
    track_result.json
    trajectory_frames.csv
    object_track_overlay.mp4
    native_tracking.csv                 # 标定分支
    native_trajectory.csv               # 标定分支
    validity_object_track_overlay.mp4    # 标定分支原生诊断
  dynamic_native/<job_id>/               # 动态分支原生 NPZ/CSV/overlay
```

`trajectory_frames.csv` 对原视频每个解码帧恰好有一行。公共核心字段为：

```text
frame_index, source_frame_index, time_s
route, position_source
center_u_px, center_v_px, radius_px
x_m, y_m, z_m, fit_x_m, fit_y_m, fit_z_m, coordinate_frame_3d
valid_2d, valid_3d, measurement_valid, interpolated, fit_eligible
track_confidence, uncertainty_px, uncertainty_m, identity_verified
invalid_reason_codes
```

- 标定分支同时给出逐帧图像坐标 `(u,v)` 和由冻结相机参数、标准球大小及实验运动平面恢复的米制 `(x,y,z)`。
- 动态分支同时给出 SpaTrackerV2 物体查询点的鲁棒 2D 中心和完成背景对齐/尺度标定后的 3D 球心。
- 没有可靠位置的帧仍保留一行，坐标为空并写明 `position_source=unavailable`，不会伪造数值。
- 插值或未验证运动预测可以画在视频上，但 `measurement_valid=false`、`fit_eligible=false`，默认不能进入参数拟合。
- `x_m/y_m/z_m` 是逐帧展示轨迹，可以包含明确标注的平滑/插值估计；`fit_x_m/fit_y_m/fit_z_m` 只在当前帧存在直接且通过硬约束的观测时有值。阶段 B 只读取 `fit_*`，不会把展示用平滑轨迹偷偷送入拟合。
- 动态 3D 帧还必须同时通过物体刚体内点、背景 anchor 内点率 `>=0.50`、背景对齐 RMSE `<=0.30 m` 且不是 fallback，才允许写入 `fit_*`。
- `object_track_overlay.mp4` 与输入保持相同帧数、分辨率、FPS 和顺序；逐帧显示 2D/3D 坐标、来源、置信度及是否可用于拟合。

## 2. 在服务器更新代码

共享账户不要写全局 `safe.directory`。使用仅对本次命令生效的临时系统配置：

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data
REMAKE_ROOT="$REPO/blender/remake"
SAFE_CFG=$(mktemp /tmp/remake-safe-gitconfig.XXXXXX)

git config --file "$SAFE_CFG" --add safe.directory "$REPO"
GIT_CONFIG_SYSTEM="$SAFE_CFG" GIT_LFS_SKIP_SMUDGE=1 \
git -C "$REPO" pull --ff-only origin codex/remake-wan22-lfs
GIT_CONFIG_SYSTEM="$SAFE_CFG" git -C "$REPO" log -1 --oneline
rm -f "$SAFE_CFG"
```

`outputs/`、轨迹输出和已经生成的视频只要没有被 Git 跟踪，`pull --ff-only` 不会删除它们。若 `git status --short` 显示本地修改与远端修改重叠，应先停止 pull 并保存本地修改。

## 3. 复用已经跑通 example0 的官方仓库和环境

不需要重新 clone。把下面两个值改成服务器真实路径/环境名：

```bash
SPATRACK_ROOT=/绝对路径/SpaTrackerV2
conda activate 你跑通_example0_时使用的环境

test -f "$SPATRACK_ROOT/inference.py"
test -d "$SPATRACK_ROOT/models"
which python

export SPATIALTRACKERV2_ROOT="$SPATRACK_ROOT"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$(python -c 'from huggingface_hub.constants import HF_HOME; print(HF_HOME)')"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
```

最后两行让适配器复用官方 example0 已下载的 Hugging Face 权重缓存，避免再次下载。如果此前明确设置过 `HF_HOME`，保持原值即可。

先定义输入和两个互相独立的输出目录：

```bash
cd "$REMAKE_ROOT"

SEED_RUN_ID=seedance20_factorized978_480p_20260719_140052
VIDEOS="$REMAKE_ROOT/outputs/seedance978/$SEED_RUN_ID/videos"
TRACK_ROOT="$REMAKE_ROOT/outputs/seedance978_tracks_v1"
EVAL_ROOT="$REMAKE_ROOT/outputs/seedance978_physics_from_tracks_v1"

find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' | wc -l
```

最后一条必须输出 `978`。

## 4. 先只预检，不运行检测或占用 GPU

```bash
python code/scripts/extract_seedance978_trajectories.py \
  --videos "$VIDEOS" \
  --output "$TRACK_ROOT" \
  --phase all \
  --spatialtracker-root "$SPATRACK_ROOT" \
  --dry-run

python rebuild-test/spatialtrackerv2/scripts/preflight.py \
  --manifest "$TRACK_ROOT/runtime_spatialtracker_manifest.jsonl" \
  --expected-jobs 978

cat "$TRACK_ROOT/run_metadata_all.json"
```

预期 `route_counts` 为 `974` 条 `calibrated_static_sphere` 和 `4` 条 `spatialtrackerv2_dynamic`。若 preflight 只报告缺少适配器依赖，可在当前已跑通 example0 的环境中补装：

```bash
python -m pip install -r rebuild-test/spatialtrackerv2/requirements-inference.txt
```

注意：本适配器为了注入球体查询点和冻结标定，不是简单调用官方 `inference.py`，而是复用其内部 `Predictor`/`VGGT4Track` 接口。仓库内快照验证的官方提交是 `7e12274c52077860cebfe007a6290777db43b63c`。preflight 会检查这些私有接口并记录你外部 clone 的 Git SHA，但 example0 成功仍不能替代下一节的真实动态 canary；外部 SHA 不同时尤其不要跳过动态 canary。

## 5. 两条 canary：先 2D，再 3D

标定分支不占 GPU：

```bash
STATIC_JOB=v1_A__g2p00__baseline__standard_ball__CAM_Side__seed-341867882

python code/scripts/extract_seedance978_trajectories.py \
  --videos "$VIDEOS" --output "$TRACK_ROOT" --phase side \
  --job-id "$STATIC_JOB" --workers 1 \
  --spatialtracker-root "$SPATRACK_ROOT"
```

然后用 GPU 7 测一条冻结审计认定需要动态 3D 的视频：

```bash
DYNAMIC_JOB=v1_C__mu0p18__indoor1__standard_ball__CAM_Side__seed-341867882

CUDA_VISIBLE_DEVICES=7 python code/scripts/extract_seedance978_trajectories.py \
  --videos "$VIDEOS" --output "$TRACK_ROOT" --phase side \
  --job-id "$DYNAMIC_JOB" --workers 1 \
  --spatialtracker-root "$SPATRACK_ROOT"
```

逐条验证帧数和产物：

```bash
python code/scripts/validate_seedance978_trajectory_outputs.py \
  --videos "$VIDEOS" --tracks "$TRACK_ROOT" --phase side \
  --job-id "$STATIC_JOB" --job-id "$DYNAMIC_JOB"

ls -lh "$TRACK_ROOT/jobs/$STATIC_JOB/trajectory_frames.csv" \
       "$TRACK_ROOT/jobs/$STATIC_JOB/object_track_overlay.mp4"
ls -lh "$TRACK_ROOT/jobs/$DYNAMIC_JOB/trajectory_frames.csv" \
       "$TRACK_ROOT/jobs/$DYNAMIC_JOB/object_track_overlay.mp4"
```

验证器只有在以下条件全部满足时返回 0：CSV 行数等于输入帧数；overlay 帧数、FPS、分辨率与输入相同；帧号连续；有效 2D/3D 坐标有限；插值行没有混入拟合。

## 6. 完整轨迹提取（可断点续跑）

建议按 Side、Main、Top 顺序运行，同一个 `TRACK_ROOT` 可以续跑。不要在正常续跑时加 `--overwrite`。

```bash
mkdir -p "$TRACK_ROOT"

CUDA_VISIBLE_DEVICES=7 nohup python code/scripts/extract_seedance978_trajectories.py \
  --videos "$VIDEOS" --output "$TRACK_ROOT" --phase side \
  --workers 4 --spatialtracker-root "$SPATRACK_ROOT" \
  > "$TRACK_ROOT/extract_side.log" 2>&1 &
echo $!
```

查看进度：

```bash
tail -f "$TRACK_ROOT/extract_side.log"
find "$TRACK_ROOT/jobs" -name track_result.json | wc -l
```

Side 完成并通过验证后依次运行：

```bash
python code/scripts/validate_seedance978_trajectory_outputs.py \
  --videos "$VIDEOS" --tracks "$TRACK_ROOT" --phase side

CUDA_VISIBLE_DEVICES=7 nohup python code/scripts/extract_seedance978_trajectories.py \
  --videos "$VIDEOS" --output "$TRACK_ROOT" --phase main \
  --workers 4 --spatialtracker-root "$SPATRACK_ROOT" \
  > "$TRACK_ROOT/extract_main.log" 2>&1 &

# 等 Main 完成并验证通过后，再启动 Top：
python code/scripts/validate_seedance978_trajectory_outputs.py \
  --videos "$VIDEOS" --tracks "$TRACK_ROOT" --phase main

CUDA_VISIBLE_DEVICES=7 nohup python code/scripts/extract_seedance978_trajectories.py \
  --videos "$VIDEOS" --output "$TRACK_ROOT" --phase top \
  --workers 4 --spatialtracker-root "$SPATRACK_ROOT" \
  > "$TRACK_ROOT/extract_top.log" 2>&1 &
```

Main 和 Top 不要同时启动：二者可能同时占用 GPU 7，而且不能让两个进程同时写同一个 `TRACK_ROOT`。每一阶段完成后分别运行验证器。服务器重启或进程中断时，重新执行相同命令即可；只有状态及逐帧 CSV/overlay 完整性均通过的缓存才会复用。`failed`、截断 JSON、缺文件或帧数不一致的任务会自动重试。

如果只想一次跑完，也可以：

```bash
CUDA_VISIBLE_DEVICES=7 python code/scripts/extract_seedance978_trajectories.py \
  --videos "$VIDEOS" --output "$TRACK_ROOT" --phase all \
  --workers 4 --spatialtracker-root "$SPATRACK_ROOT"
```

## 7. 独立执行物理拟合与评分

轨迹验收后再运行评估。这个命令不需要 GPU：

```bash
python code/scripts/evaluate_seedance978_trajectories.py \
  --tracks "$TRACK_ROOT" --output "$EVAL_ROOT" --phase side

python code/scripts/evaluate_seedance978_trajectories.py \
  --tracks "$TRACK_ROOT" --output "$EVAL_ROOT" --phase main

python code/scripts/evaluate_seedance978_trajectories.py \
  --tracks "$TRACK_ROOT" --output "$EVAL_ROOT" --phase top
```

评估输出为：

```text
<EVAL_ROOT>/
  aggregate.json
  all_jobs.csv
  side_primary_summary.csv
  viewpoint_robustness.csv
  seed_stability.csv
  jobs/<job_id>/
    result.json
    fit_frames.csv
    trajectory_plot.png
```

`fit_frames.csv` 只包含直接测量且通过质量门的帧。`result.json` 会记录输入轨迹文件 SHA-256，并明确写出 `evaluation_reads_video=false`、`evaluation_invokes_tracker=false`。因此后续修改拟合公式时可以反复重跑阶段 B，不必重新检测 978 条视频。

评估缓存同时绑定轨迹 CSV、`track_result.json`、实验 registry、评估器源码和物理拟合器源码的 SHA-256；任一项变化都会自动重算，不会因为旧 `result.json` 存在而错误跳过。

## 8. 失败与人工复核

轨迹阶段的 `failed/partial` 不等于模型物理生成失败：

- `track_result.json` 的 `status` 描述轨迹产物是否完整；
- `video_generation_validity.status` 描述是否有明确形变、场景重构等生成失败证据；
- `trajectory_fit_eligible` 描述轨迹自身是否有足够直接观测；
- 最终是否拟合由阶段 B 综合决定。

人工复核优先打开每条任务的 `object_track_overlay.mp4`。若坐标为空，查看同一帧的 `invalid_reason_codes`；不要把空坐标补成平滑曲线后当作测量证据。
