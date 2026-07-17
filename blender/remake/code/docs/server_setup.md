# H20 服务器：全量视频纯生成

当前正式流程是“先完成所有模型的全部视频生成，再统一评估”。本页只说明生成；不要运行 `sequence` 或 `evaluate`。

## 0. 安全终止旧生成-评估流水线

共享 root 账户上不要按进程名批量 `pkill`，也不要直接杀整个进程组。先确认 PID 仍属于本项目：

```bash
OLD_PID=332251

ps -o user,pid,ppid,pgid,sid,etime,cmd -p "$OLD_PID"
readlink -f "/proc/$OLD_PID/cwd"
tr '\0' ' ' <"/proc/$OLD_PID/cmdline"; echo
pstree -ap "$OLD_PID" || true
```

只有当工作目录或命令明确包含当前 `chenyu/remake` 项目时，才在当前 shell 定义并执行下面的递归函数。它先终止最深层子进程，再终止父进程，不会按用户名或模糊进程名影响其他用户任务：

```bash
terminate_tree() {
  local parent="$1"
  local child
  for child in $(pgrep -P "$parent" 2>/dev/null); do
    terminate_tree "$child"
  done
  kill -TERM "$parent" 2>/dev/null || true
}

terminate_tree "$OLD_PID"
sleep 5

ps -o user,pid,ppid,etime,cmd -p "$OLD_PID" || true
pstree -ap "$OLD_PID" 2>/dev/null || true
```

若 5 秒后同一个已核对的 PID 仍存在，再只对该树使用 `KILL`，不要使用负 PGID：

```bash
kill_tree_now() {
  local parent="$1"
  local child
  for child in $(pgrep -P "$parent" 2>/dev/null); do
    kill_tree_now "$child"
  done
  kill -KILL "$parent" 2>/dev/null || true
}

kill_tree_now "$OLD_PID"
```

## 1. 更新仓库且不影响共享账户

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data
REMAKE_ROOT="$REPO/blender/remake"

cd "$REPO"
git -c safe.directory="$REPO" fetch origin codex/remake-wan22-lfs
git -c safe.directory="$REPO" checkout codex/remake-wan22-lfs
GIT_LFS_SKIP_SMUDGE=1 git -c safe.directory="$REPO" pull --ff-only \
  origin codex/remake-wan22-lfs
```

命令行中的 `-c safe.directory=...` 只对该次 Git 命令生效，不会继续修改共享 root 账户的全局配置。`blender/remake/outputs/` 已被 Git 忽略，正常 `pull --ff-only` 不会删除生成结果；不要使用 `git clean -fdx`。

若使用 sparse-checkout：

```bash
git -c safe.directory="$REPO" sparse-checkout init --no-cone
GIT_LFS_SKIP_SMUDGE=1 git -c safe.directory="$REPO" sparse-checkout set --no-cone \
  /blender/remake/code/ \
  /blender/remake/first_frames_v1_0/ \
  /blender/remake/.gitattributes \
  /blender/remake/.gitignore

git -c safe.directory="$REPO" lfs pull \
  --include="blender/remake/first_frames_v1_0/**"
```

数据检查：

```bash
find "$REMAKE_ROOT/first_frames_v1_0/images" -name '*.png' -type f | wc -l
file "$REMAKE_ROOT/first_frames_v1_0/images/v1_A/baseline/standard_ball/CAM_Side.png"
du -sh "$REMAKE_ROOT/first_frames_v1_0"
```

PNG 数量必须是 1404，`file` 不能显示 Git LFS pointer 文本。

## 2. Wan2.2 环境与路径

```bash
source /root/data/heyuanyu/yefei/chenyu/wan22_env/bin/activate
cd /root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake
python -m pip install -r code/requirements.txt

export REMAKE_ROOT="$PWD"
export WAN_REPO=/root/data/heyuanyu/yefei/chenyu/data/Wan2.2
export WAN_CKPT_DIR=/root/data/heyuanyu/yefei/chenyu/models/Wan2.2-I2V-A14B
export WAN_PYTHON=python
export GPU_ID=7
export WAN_OFFLOAD_MODEL=false

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
test -f "$WAN_REPO/generate.py"
test -d "$WAN_CKPT_DIR"
```

H20 97871 MiB 默认使用 GPU 7 和一个常驻 Wan2.2 worker。只有首次真实任务报告 CUDA OOM 时，才结束该 run 并用 `WAN_OFFLOAD_MODEL=true` 新建 run。

## 3. 标准球首轮清单

正式 build：

```text
code/builds/standard_ball_all_experiments_wan22_generation.yaml
```

清单覆盖：

```text
13 experiments
69 frozen parameter tuples
9 scenes
1 object: standard_ball
3 cameras
1 seed
= 1863 videos per model
```

默认 `NUM_FRAMES=81`、16 fps、40 sampling steps。相对原四物体、161 帧计划，任务数量减少为四分之一，帧数约减半；同时继续使用常驻模型、GPU 7 和 `WAN_OFFLOAD_MODEL=false`。不降低 sampling steps，以避免进一步损失画质。

81 帧是 speed-first 条件。部分长周期 V2/V3 实验将来恢复物理拟合时，可能需要用新的 `RUN_ID` 和 `NUM_FRAMES=161` 补跑。`NUM_FRAMES` 必须满足 Wan2.2 的 `4n+1`，不能改变已 prepared run 的帧数。

### 3.1 固定输入的四次采样测试

以下 build 固定 `v1_A / g=9.81 / baseline / standard_ball / CAM_Side`，只改变 seed 36、37、38、39，因此总共只有 4 条视频：

```bash
RUN_ID=v1a_seed_variation_wan22_$(date +%Y%m%d_%H%M%S)

RUN_ID="$RUN_ID" \
GPU_ID=7 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
bash code/scripts/run_seed_variation.sh

bash code/scripts/make_seed_variation_grid.sh "outputs/$RUN_ID"
```

输出为：

```text
outputs/<run_id>/seed_variation_grid.mp4
```

布局顺序是左上 seed36、右上 seed37、左下 seed38、右下 seed39。确认四条视频的运动方向、落地行为、物体形状和背景稳定性可接受后，再使用新的正式 `RUN_ID` 跑 1863 条标准球结果。不要把正式 build 的 `seeds` 直接扩成四个。

## 4. 先做三条 dry-run

dry-run 会 prepare 1863-job 标准球 manifest，但只打印前三条模型命令，不加载模型：

```bash
RUN_ID=standard_ball_all13_wan22_dryrun \
MAX_JOBS=3 \
DRY_RUN=1 \
GPU_ID=7 \
bash code/scripts/run_standard_ball_generation.sh

python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/standard_ball_all13_wan22_dryrun/manifest.jsonl')
rows = [json.loads(line) for line in p.open(encoding='utf-8')]
print('jobs =', len(rows))
print('experiments =', len({row['experiment_id'] for row in rows}))
print('parameter tuples =', len({(row['experiment_id'], row['factors']['parameter_tuple_id']) for row in rows}))
print('input PNGs =', len({row['inputs']['image'] for row in rows}))
PY
```

预期依次输出 `1863`、`13`、`69`、`351`。

## 5. 正式后台生成

```bash
RUN_ID=standard_ball_all13_wan22_$(date +%Y%m%d_%H%M%S)

RUN_ID="$RUN_ID" \
DETACHED=1 \
GPU_ID=7 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
bash code/scripts/run_standard_ball_generation.sh

echo "$RUN_ID"
```

查看进度：

```bash
tail -f "outputs/$RUN_ID/master.log"
cat "outputs/$RUN_ID/run_summary.json" 2>/dev/null || true
find "outputs/$RUN_ID/videos" -name '*.mp4' -type f | wc -l
tail -n 20 "outputs/$RUN_ID/run_state.jsonl"
```

`eval/` 不应在本流程中出现。任务次序按“实验 → 参数 tuple → 场景 → 标准球 → 视角”展开，同一个参数 tuple 的 27 个视频连续生成。

## 6. 断点续跑

Tunnel 断开不影响 `nohup` 进程。若进程确实停止，使用完全相同的 `RUN_ID`：

```bash
RUN_ID=<existing_run_id> \
GPU_ID=7 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
bash code/scripts/run_standard_ball_generation.sh
```

脚本检测到 `resolved_build.yaml` 和 `manifest.jsonl` 后不会重新 prepare；已有非空 MP4 会被跳过。不要设置 `OVERWRITE=1`，否则已有视频会重新生成。

也可直接运行：

```bash
export PYTHONPATH="$PWD/code/src${PYTHONPATH:+:$PYTHONPATH}"
python -m remake_benchmark generate \
  --run-dir "outputs/<existing_run_id>" \
  --start-index 0 \
  --fail-fast
```

## 7. 其他模型

冻结任务清单、输入 PNG、prompt 规则和 job ID 语义对所有模型保持一致；每个模型使用独立 build 和独立 `outputs/<run_id>`。当前新框架已可直接运行 Wan2.2 reference adapter。CogVideoX1.5、HunyuanVideo-1.5 和 Cosmos-Predict2.5 的旧运行代码仍在仓库上层 `code/` 中，但在迁入本框架并完成服务器 smoke 前，不应把它们标记为同一正式矩阵的已完成结果。

后续增加模型时只替换 build 中的 model profile，experiment 与 prompt profile必须继续引用：

```text
configs/experiments/all_experiments_full.yaml
configs/prompts/all_experiments_explicit_v1.yaml
```

其他 I2V 模型也应先使用相同的标准球 override，每模型 1863 个 job；四个模型合计 7452 个视频。Cosmos 若同时比较 image2world 与 video2world，应作为两个独立模型条件，各自生成 1863 个输出，不能混在同一个 run 中。
