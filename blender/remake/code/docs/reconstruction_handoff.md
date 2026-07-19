# 物体轨迹重建 MVP 交接套件

这份文档用于把当前视频生成或重建工作交给另一位同学运行。若对方只负责重建，则不需要安装视频生成模型，也不需要 GPU；只要拿到仓库代码、对应首帧和待检查 MP4 即可。

## 零、先区分两种交接角色

### A. 对方负责生成另一种模型的视频

主负责人必须先冻结一份统一任务包，不能让每位模型负责人自行重新生成 prompt 或抽取任务：

```text
frozen_task_pack/
├── manifest.jsonl
├── manifest.selection.json
├── resolved_build.yaml
├── first_frames_v1_0/
├── configs/prompts/
├── schemas/
├── HANDOFF.md
└── sha256sums.txt
```

`manifest.jsonl` 每行应直接包含 `job_id`、首帧相对路径、canonical prompt、negative prompt、目标参数、已知参数、场景、物体、视角、seed 和请求生成参数。其他人只负责逐行调用自己的模型，不能修改实验、场景、首帧、prompt、target 或 job id。

如果对方把新模型接入当前 adapter 框架，只替换 model profile，并创建新的 `build_id`；experiment、prompt 和任务选择保持不变：

```bash
python code/scripts/remake_benchmark.py prepare \
  --build code/builds/<new_model_build>.yaml \
  --run-dir outputs/<model>/<run_id> \
  --workspace-root "$PWD"

python code/scripts/remake_benchmark.py generate \
  --run-dir outputs/<model>/<run_id> \
  --dry-run \
  --max-jobs 1

python code/scripts/remake_benchmark.py generate \
  --run-dir outputs/<model>/<run_id> \
  --concurrency <provider_limit>
```

如果暂时不写 adapter，对方直接读取 frozen `manifest.jsonl` 中的 `inputs.image + prompt + negative_prompt`，然后把原始输出保存为：

```text
videos/<job_id>.mp4
```

模型不支持单独 negative prompt 时，可以发送 `<canonical prompt>\nAvoid: <canonical negative prompt>`，但必须同时保存 canonical prompt 和 provider 最终实际发送的 prompt。不要保存 API Key 本身，只记录使用的环境变量名。

生成负责人需要原样返回整个 run 目录，至少包括 `manifest.jsonl`、`resolved_build.yaml`、`videos/`、`metadata/`、`logs/`、`run_state.jsonl` 和 `run_summary.json`。每条任务还应记录模型/checkpoint、provider task id、seed 支持情况、请求参数、实际媒体参数、耗时、费用、重试及原始 provider JSON。

### B. 对方只负责运行重建

使用下面的重建套件即可。输入是生成负责人返回的 `videos/`，不需要模型权重、API Key、CUDA 或 GPU。

## 一、先明确套件边界

当前套件只支持：

```text
实验：v1_A 自由落体
物体：standard_ball
视角：CAM_Side
参数：gravity_g
结果：已知球尺寸约束下的 1-D/2.5-D 近似轨迹
```

它不能从固定视角 MP4 恢复完整场景网格，也不能给出严格独立深度。其他实验、物体和视角会在 preflight 阶段直接拒绝，不会静默套用错误公式。

## 二、需要交给对方什么

### 1. Git 仓库代码

使用最新分支头；`558bcbb` 是重建核心的基线提交：

```text
branch: codex/remake-wan22-lfs
base reconstruction commit: 558bcbb
```

核心代码、配置和机器可读清单分别是：

```text
code/src/remake_benchmark/reconstruction/
code/configs/reconstruction/object_centric_mvp_v1.yaml
code/scripts/check_reconstruction_handoff.py
code/scripts/run_reconstruction_handoff.sh
code/scripts/run_reconstruction_mvp.py
code/handoffs/reconstruction_mvp_v1.yaml
```

### 2. 首帧数据

至少需要对应场景的：

```text
first_frames_v1_0/images/v1_A/<scene>/<object>/CAM_Side.png
```

建议保留同场景的全部四种物体首帧，因为跟踪器会利用“相同背景、不同物体”的变化区域定位实验球。如果图片仍是 Git LFS pointer，OpenCV 无法读取，preflight 会明确报错。

从 Git LFS 获取首帧：

```bash
git lfs install --local
git lfs pull --include="blender/remake/first_frames_v1_0/**/*.png"
```

当前 20 条 MP4 约 7.85 MiB；绝对最小的 5 张标准球首帧约 3.45 MiB。推荐同时交付 5 个场景×4 种物体的 20 张侧视首帧，约 13.82 MiB，以保留 sibling-variation 定位先验。

### 3. 待处理视频

MP4 放在一个平铺目录中，不要再嵌套子文件夹。文件名必须保留 benchmark job id：

```text
v1_A__gravity_g-9p81__indoor3__standard_ball__CAM_Side__seed-341867882.mp4
```

通用格式：

```text
v1_A__gravity_g-<值>__<场景>__standard_ball__CAM_Side__seed-<整数>.mp4
```

小数点使用 `p`，例如 `4p9`、`9p81`、`14p7`。不要为了统一帧率而重新编码；程序优先读取 MP4 的逐帧时间戳。

正式交接时还应一起保存原始生成任务的：

```text
manifest.jsonl
resolved_build.yaml
metadata/*.json
API 或模型运行日志
```

当前 MVP 从文件名读取 job 信息，但这些 metadata 是后续论文复核和成本统计所必需的。

## 三、服务器环境

要求：

```text
Python >= 3.10
PyYAML
NumPy
opencv-python-headless
matplotlib
```

不要求 CUDA、GPU、ffmpeg 命令行或任何生成模型权重。优先在对方自己的虚拟环境中安装，避免影响共享账户上的其他任务：

```bash
python -m venv .venv-reconstruction
source .venv-reconstruction/bin/activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
```

若已有环境，先运行 preflight，不要直接重复安装依赖。

共享账户上不要写入全局 `safe.directory`。更新代码时使用单条命令作用域：

```bash
git -c safe.directory="$REPO" pull --ff-only origin codex/remake-wan22-lfs
```

## 四、一键运行

在 `blender/remake` 根目录执行：

```bash
bash code/scripts/run_reconstruction_handoff.sh \
  /absolute/path/to/videos \
  /absolute/path/to/output_run \
  4 \
  5
```

四个位置参数依次是：

```text
1. MP4 文件夹
2. 本次输出文件夹
3. CPU worker 数，建议 4
4. 生成标注视频的样本数，建议 5
```

脚本先执行环境与数据检查；有错误时不会启动正式批次。检查也可以单独运行：

```bash
python code/scripts/check_reconstruction_handoff.py \
  --workspace-root "$PWD" \
  --videos /absolute/path/to/videos \
  --output /absolute/path/to/output_run \
  --expect-count 20
```

建议先运行两条 smoke：

```bash
python code/scripts/run_reconstruction_mvp.py \
  --workspace-root "$PWD" \
  --videos /absolute/path/to/videos \
  --output /absolute/path/to/smoke_output \
  --workers 2 \
  --overlay-count 2 \
  --limit 2
```

直接调用底层脚本的等价命令：

```bash
python code/scripts/run_reconstruction_mvp.py \
  --workspace-root "$PWD" \
  --videos /absolute/path/to/videos \
  --output /absolute/path/to/output_run \
  --workers 4 \
  --overlay-count 5
```

每一次正式运行都建议使用新的输出目录，例如：

```text
outputs/reconstruction/<model_id>_<YYYYMMDD_HHMMSS>/
```

同一输出目录同时只允许一个进程写入。发现 `.reconstruction_mvp.lock` 时，应先确认没有任务仍在运行，不能直接删除正在使用的锁。

## 五、对方需要回传什么

最稳妥的做法是原样打包并回传整个 `output_dir`，不要人工挑选文件。至少应包含：

```text
aggregate.json
summary.csv
results.jsonl
report/
每个 job 的 result.json
每个 job 的 reconstruction_validity.json
每个 job 的 physics_adherence.json
每个 job 的 video_probe.json
每个 job 的 reference_alignment.json
每个 job 的 camera_motion.csv
每个 job 的 trajectory_world.csv
每个 job 的 trajectory.ply
每个 job 的 trajectory_diagnostics.png
抽样 track_overlay.mp4
```

当前 20 条标准批次的完整输出约 12 MiB，通常可以直接压缩后传回，无需放入 Git LFS。

不要只回传 `summary.csv`。逐帧轨迹、诊断图和标注视频是确认检测没有认错物体的必要证据。

新版 `artifacts.*` 使用相对于每个 `result.json` 所在目录的路径，因此整体搬迁输出目录后仍然有效；`source_video` 和 `conditioning_image` 是运行机器上的输入 provenance 绝对路径，搬迁后失效是正常的，不应据此判断输出文件丢失。

## 六、如何读结果

首先看：

```text
report/index.html
```

三个字段必须分开理解：

- `constrained_trajectory_valid=true`：在当前先验下，主轨迹可以被可靠测量。
- `physics_adherence.status=pass/fail`：轨迹拟合及反推参数是否符合该 job 的目标重力。
- `strict_metric_3d_valid=false`：当前固定侧视 MP4 没有完整相机 K/T 与独立深度，这不是运行错误。

如果 reconstruction 失败，应先看 `failed_reconstruction_gates`、轨迹诊断图和 overlay；如果 reconstruction 通过但 physics 失败，通常意味着视频中的实际运动不符合目标，而不是检测失败。

## 七、20 条标准交接验收

交接者收到结果后至少检查：

```text
[ ] aggregate.json 中 jobs=20、errors=0
[ ] native FPS/PTS 均有记录
[ ] 每条都有 result、validity、physics、trajectory CSV、PLY 和诊断图
[ ] 5 条抽样 overlay 均可播放且框住实验球
[ ] report 中 reconstruction 与 physics 是两个独立状态
[ ] strict_metric_3d 没有被错误报告为通过
[ ] 输出目录中没有残留 .reconstruction_mvp.lock
```

这套结果目前应称为 `diagnostic reconstruction MVP`。在把它作为正式 benchmark 分数之前，还需要用 Blender GT 做轨迹误差、重力误差和接触时刻误差验收，并从 `.blend` 导出完整相机与场景几何 sidecar。
