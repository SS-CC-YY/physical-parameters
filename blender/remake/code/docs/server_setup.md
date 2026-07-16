# Wan2.2 H20 服务器运行说明

## 1. 当前仓库与路径

VS Code Tunnel 服务器上的 Git 仓库：

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data
REMAKE_ROOT="$REPO/blender/remake"
```

默认模型路径可由环境变量覆盖，不需要改 YAML：

```bash
export WAN_REPO=/root/data/heyuanyu/yefei/chenyu/data/Wan2.2
export WAN_CKPT_DIR=/root/data/heyuanyu/yefei/chenyu/models/Wan2.2-I2V-A14B
export WAN_PYTHON=python
export GPU_ID=0
export REMAKE_ROOT=/root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake
```

共享 root 账户上不要继续写全局 Git 配置。以下所有命令只用临时 `-c safe.directory=...`，不会改变其他用户的 Git 配置。

### outputs 与 git pull

`blender/remake/.gitignore` 使用 `/outputs/` 忽略运行结果，且仓库没有跟踪任何 outputs 文件。因此正常的 `fetch`、`pull --ff-only`、`checkout` 不会删除或覆盖已有视频、日志和评估报告。

更新前可检查：

```bash
git -c safe.directory="$REPO" status --short
git -c safe.directory="$REPO" check-ignore -v \
  blender/remake/outputs/<run_id>/sequential_summary.json
```

若未来远端真的新增了与本地 ignored 文件完全同名的 tracked 文件，Git 会停止并提示冲突，而不是静默覆盖。不要在该仓库运行 `git clean -fdx`；其中 `-x` 会把 ignored 的 outputs 一并删除。

## 2. 从 GitHub 更新代码与完整首帧

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data

git -c safe.directory="$REPO" fetch origin codex/remake-wan22-lfs
git -c safe.directory="$REPO" checkout codex/remake-wan22-lfs
git -c safe.directory="$REPO" pull --ff-only origin codex/remake-wan22-lfs

git -c safe.directory="$REPO" sparse-checkout init --no-cone
git -c safe.directory="$REPO" sparse-checkout set --no-cone \
  /blender/remake/code/ \
  /blender/remake/first_frames_v1_0/ \
  /blender/remake/.gitattributes \
  /blender/remake/.gitignore

git -c safe.directory="$REPO" lfs pull \
  --include="blender/remake/first_frames_v1_0/**"
```

检查 LFS 文件不是 pointer：

```bash
find "$REPO/blender/remake/first_frames_v1_0" -type f | wc -l
find "$REPO/blender/remake/first_frames_v1_0" -name '*.png' -type f | wc -l
file "$REPO/blender/remake/first_frames_v1_0/images/v1_A/baseline/standard_ball/CAM_Side.png"
du -sh "$REPO/blender/remake/first_frames_v1_0"
```

完整目录应有 1437 个文件，其中 1404 个 PNG，体积约 1.1 GB。`file` 应识别为 PNG image，不能显示 Git LFS pointer 文本。

## 3. H20 环境检查

```bash
source /root/data/heyuanyu/yefei/chenyu/wan22_env/bin/activate
cd /root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake
python -m pip install -r code/requirements.txt

export WAN_REPO=/root/data/heyuanyu/yefei/chenyu/data/Wan2.2
export WAN_CKPT_DIR=/root/data/heyuanyu/yefei/chenyu/models/Wan2.2-I2V-A14B
export WAN_PYTHON=python
export GPU_ID=0
export WAN_OFFLOAD_MODEL=false

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
test -f "$WAN_REPO/generate.py"
test -d "$WAN_CKPT_DIR"
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory/1024**3)"
```

H20 97871 MiB 默认使用 `WAN_OFFLOAD_MODEL=false`，并由一个常驻进程只加载一次 Wan2.2。分辨率、81 帧和 40 个采样步不降低，因此该优化不改变生成质量。若第一条真实生成报告 CUDA OOM，用 `WAN_OFFLOAD_MODEL=true` 新建一个 run；这会更慢，但生成参数和质量口径不变。

## 4. 只检查配置与三视角命令

`MAX_JOBS=3` 恰好检查同一基线物体的 CAM_Main、CAM_Side、CAM_Top，不加载模型：

```bash
RUN_ID=wan22_three_view_dryrun \
MAX_JOBS=3 \
DRY_RUN=1 \
GPU_ID=0 \
WAN_OFFLOAD_MODEL=false \
bash code/scripts/run_wan22_demo.sh
```

## 5. 三视角真实 smoke（生成一个、立即评估一个）

```bash
RUN_ID=wan22_three_view_smoke \
MAX_JOBS=3 \
GPU_ID=0 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
bash code/scripts/run_wan22_demo.sh
```

真实执行顺序是：

```text
CAM_Main 生成 → 严重异常门控 → 拟合 → 图/标注视频
CAM_Side 生成 → 严重异常门控 → 拟合 → 图/标注视频
CAM_Top  生成 → 严重异常门控 → 拟合 → 图/标注视频
```

不会先生成三条再统一评估。报告在每条视频评估完成后重写，因此运行中就可以查看累计结果。

## 6. 正式抽样实验（60 个作业）

正式 build 固定包含 baseline，以 seed 36 随机抽 2 个 indoor 和 2 个 outdoor；每个场景包含 4 个物体和 3 个视角：

```text
5 scenes × 4 objects × 3 cameras × 1 seed = 60 jobs
```

后台运行：

```bash
RUN_ID=v1a_wan22_h20_full_$(date +%Y%m%d_%H%M%S) \
DETACHED=1 \
GPU_ID=0 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
STOP_ON_INVALID=0 \
bash code/scripts/run_wan22_demo.sh
```

`FAIL_FAST=1` 会在模型进程或 evaluator 程序错误时停止；物理门控的 `invalid` 样本仍会记录并继续下一条。若希望任何质量异常都立即停止人工检查，设置 `STOP_ON_INVALID=1`。

查看进度：

```bash
tail -f "outputs/$RUN_ID/master.log"
cat "outputs/$RUN_ID/sequential_summary.json"
```

## 7. 断点续跑

同一 run 被中断后，不要重新 prepare：

```bash
cd /root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake
export PYTHONPATH="$PWD/code/src${PYTHONPATH:+:$PYTHONPATH}"

python -m remake_benchmark sequence \
  --run-dir "outputs/<existing_run_id>" \
  --start-index 0 \
  --fail-fast
```

非空视频会跳过生成，但仍会执行/更新相应评估，然后从缺失视频处继续。不要传 `--overwrite`，否则会重新生成已有视频。

如果只是 pull 了新的 evaluator，希望用已有 raw videos 重做评估，也使用同一条 `sequence` 命令；当 manifest 中所有视频都存在时，它不会启动 Wan2.2 worker。V1_A 旧 manifest 即使还没有 `drop_distance_m` 字段，也会使用 release 中冻结的 `4.2-0.44=3.76 m` 兼容值。必须对 `outputs/<run_id>/videos/*.mp4` 原始生成视频重评，不要把已经画过框的 `eval/freefall/overlays/*.mp4` 再作为 evaluator 输入。

## 8. 查看可视化

```bash
python -m http.server 8000 --directory "outputs/<run_id>/eval"
```

在 VS Code Ports 面板转发 8000，打开 `/report/index.html`。主要文件：

```text
eval/report/index.html
eval/freefall/summary.csv
eval/freefall/tracks/<job_id>.csv
eval/freefall/plots/<job_id>.png
eval/freefall/overlays/<selected_job_id>.mp4
eval/sample_metrics.jsonl
```

每个作业都有轨迹 CSV 和轨迹图；按场景与视角分层抽取 9 条视频制作检测框、中心点和轨迹尾迹 overlay。具体公式和门控定义见 `docs/freefall_evaluation.md`。
