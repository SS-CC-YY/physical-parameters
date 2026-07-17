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
export GPU_IDS=4,5,6,7
export WAN_OFFLOAD_MODEL=false

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
test -f "$WAN_REPO/generate.py"
test -d "$WAN_CKPT_DIR"
```

四卡流程在 GPU 4、5、6、7 上各启动一个常驻 Wan2.2 worker。每张卡独立加载模型并生成不同视频；不要把 `GPU_ID` 直接写成 `4,5,6,7` 后运行单卡脚本。只有某个 shard 的首次真实任务报告 CUDA OOM 时，才结束四个 shard，并用 `WAN_OFFLOAD_MODEL=true` 和新 `RUN_ID` 重跑。

## 3. 标准球首轮清单

当前推荐的 factorized 正式 build：

```text
code/builds/standard_ball_factorized900_wan22_generation.yaml
```

清单覆盖：

```text
13 experiments
48 selected frozen parameter tuples in the side-view physics track
2 reference/stress tuples per experiment in the main/top viewpoint track
9 scenes
1 object: standard_ball
3 cameras
1 seed
= 900 videos per model
```

48 组参数的筛选规则如下：

- V1 及 `v2_A/v2_B` 单参数实验各保留低值、代表性中间值和高值，共 18 组；
- `v2_C` 的 4 个双摩擦组合全部保留，避免破坏两参数区分能力；
- `v2_D/v2_E` 各保留 5 个主效应组合，删除角点或冗余中间点；
- 四个 V3 实验各保留 default 加每个隐藏参数的一次最大可辨识单因素变化，共 16 组。

因此物理 tuple 总数仍为 `18 + 4 + 5 + 5 + 16 = 48`。生成矩阵进一步拆成两条可独立报告的轨道：

1. 物理辨识主轨：48 tuples × 9 scenes × `CAM_Side` = 432 条，用于参数变化、背景变化与物理运动的一致性；
2. 视角鲁棒性轨：每实验 2 个代表/压力 tuples × 13 experiments × 9 scenes × `CAM_Main/CAM_Top` = 468 条，用于证明非侧视角流程与视角泛化能力。

总数为 `432 + 468 = 900`。每个实验、场景和视角组合仍至少有代表点和压力点；所有 48 个参数 tuple 均在全部 9 个场景的侧视角出现。该设计不再声称完整的 parameter × scene × view 三因素交互，而是明确报告“物理辨识”和“视角鲁棒性”两条 benchmark 结果。相对 1296 条减少 30.6%，相对原 1863 条减少 51.7%。

48-tuple 全三视角 build `code/builds/standard_ball_reduced_parameters_wan22_generation.yaml` 和原始 69-tuple build `code/builds/standard_ball_all_experiments_wan22_generation.yaml` 都继续保留，可在后续补跑。

默认 `NUM_FRAMES=81`、16 fps、40 sampling steps；GPU 4、5、6、7 各使用一个常驻模型，并保持 `WAN_OFFLOAD_MODEL=false`。不降低 sampling steps，以避免进一步损失画质。

81 帧是 speed-first 条件。部分长周期 V2/V3 实验将来恢复物理拟合时，可能需要用新的 `RUN_ID` 和 `NUM_FRAMES=161` 补跑。`NUM_FRAMES` 必须满足 Wan2.2 的 `4n+1`，不能改变已 prepared run 的帧数。

### 3.1 固定输入的四次采样测试

以下 build 固定 `v1_A / g=9.81 / baseline / standard_ball / CAM_Side`，只改变 seed 36、37、38、39，因此总共只有 4 条视频。四卡入口会让 GPU 4、5、6、7 各生成一个 seed：

```bash
RUN_ID=v1a_seed_variation_wan22_4gpu_$(date +%Y%m%d_%H%M%S)

RUN_ID="$RUN_ID" \
GPU_IDS=4,5,6,7 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
bash code/scripts/run_seed_variation_4gpu.sh

bash code/scripts/check_four_gpu_generation.sh "outputs/$RUN_ID"
python code/scripts/collect_sharded_run.py "outputs/$RUN_ID"

bash code/scripts/make_seed_variation_grid.sh "outputs/$RUN_ID"
```

输出为：

```text
outputs/<run_id>/seed_variation_grid.mp4
```

布局顺序是左上 seed36、右上 seed37、左下 seed38、右下 seed39。`collect_sharded_run.py` 只有在四个视频和四份 metadata 均完整时才会建立父目录的统一 `videos/`。确认四条视频的运动方向、落地行为、物体形状和背景稳定性可接受后，再使用新的正式 `RUN_ID` 跑 900 条 factorized 标准球结果。不要把正式 build 的 `seeds` 直接扩成四个。

## 4. 先做三条 dry-run

dry-run 会 prepare 900-job factorized 标准球 manifest，但只打印前三条模型命令，不加载模型：

```bash
RUN_ID=standard_ball_all13_factorized900_wan22_dryrun \
BUILD_FILE=code/builds/standard_ball_factorized900_wan22_generation.yaml \
MAX_JOBS=3 \
DRY_RUN=1 \
GPU_ID=7 \
bash code/scripts/run_full_generation.sh

python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/standard_ball_all13_factorized900_wan22_dryrun/manifest.jsonl')
rows = [json.loads(line) for line in p.open(encoding='utf-8')]
print('jobs =', len(rows))
print('experiments =', len({row['experiment_id'] for row in rows}))
print('parameter tuples =', len({(row['experiment_id'], row['factors']['parameter_tuple_id']) for row in rows}))
print('input PNGs =', len({row['inputs']['image'] for row in rows}))
PY
```

预期依次输出 `900`、`13`、`48`、`351`。

## 5. 四卡正式后台生成

```bash
RUN_ID=standard_ball_all13_factorized900_wan22_4gpu_$(date +%Y%m%d_%H%M%S)

RUN_ID="$RUN_ID" \
GPU_IDS=4,5,6,7 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
bash code/scripts/run_factorized_standard_ball_generation_4gpu.sh

echo "$RUN_ID"
```

启动器会立即返回四个 PID，并将 900 条均匀分为：

```text
GPU 4: start=0,   jobs=225
GPU 5: start=225, jobs=225
GPU 6: start=450, jobs=225
GPU 7: start=675, jobs=225
```

每张卡分别写入 `outputs/<run_id>/shards/gpu-<id>/`，不会并发覆盖状态文件。查看统一进度：

```bash
bash code/scripts/check_four_gpu_generation.sh "outputs/$RUN_ID"

tail -f "outputs/$RUN_ID/shards/gpu-4/master.log"
tail -f "outputs/$RUN_ID/shards/gpu-5/master.log"
tail -f "outputs/$RUN_ID/shards/gpu-6/master.log"
tail -f "outputs/$RUN_ID/shards/gpu-7/master.log"
```

状态显示 `generated=900/900` 后统一汇总：

```bash
python code/scripts/collect_sharded_run.py "outputs/$RUN_ID"
cat "outputs/$RUN_ID/run_summary.json"
find "outputs/$RUN_ID/videos" -maxdepth 1 -type f -name '*.mp4' -size +0c | wc -l
```

汇总器会严格检查缺失、空文件、重复 job ID 和 metadata。检查通过后使用硬链接建立统一 `videos/` 与 `metadata/`，因此不会额外复制 900 个 MP4。`eval/` 不应在本流程中出现。

## 6. 断点续跑

Tunnel 断开不影响四个 `nohup` 进程。若一个或多个 shard 停止，使用完全相同的 `RUN_ID` 重新执行四卡入口：

```bash
RUN_ID=<existing_run_id> \
GPU_IDS=4,5,6,7 \
WAN_OFFLOAD_MODEL=false \
FAIL_FAST=1 \
bash code/scripts/run_factorized_standard_ball_generation_4gpu.sh
```

入口不会重复启动仍存活的本项目 shard；停止的 shard 会从自己的固定范围恢复，已有非空 MP4 会被跳过。不要设置 `OVERWRITE=1`，否则已有视频会重新生成。不要同时对同一个 `RUN_ID` 运行单卡脚本。

## 7. 其他模型

冻结任务清单、输入 PNG、prompt 规则和 job ID 语义对所有模型保持一致；每个模型使用独立 build 和独立 `outputs/<run_id>`。当前新框架已可直接运行 Wan2.2 reference adapter。CogVideoX1.5、HunyuanVideo-1.5 和 Cosmos-Predict2.5 的旧运行代码仍在仓库上层 `code/` 中，但在迁入本框架并完成服务器 smoke 前，不应把它们标记为同一正式矩阵的已完成结果。

后续增加模型时只替换 build 中的 model profile，experiment 与 prompt profile必须继续引用：

```text
configs/experiments/all_experiments_full.yaml
configs/prompts/all_experiments_explicit_v1.yaml
```

其他 I2V 模型也应先使用相同的 factorized900 标准球矩阵，每模型 900 个 job；四个模型合计 3600 个视频。Cosmos 若同时比较 image2world 与 video2world，应作为两个独立模型条件，各自生成 900 个输出，不能混在同一个 run 中。若后续升级为 1296 或 1863 条清单，再对所有模型一致补跑，不能只补某一个模型。
