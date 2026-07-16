# 服务器运行说明

## 1. 默认服务器路径

Wan2.2 model profile 当前采用旧实验已经使用过的服务器布局：

```text
项目：       /root/data/heyuanyu/yefei/chenyu/data/physical-parameters
remake：     /root/data/heyuanyu/yefei/chenyu/data/physical-parameters/blender/remake
Wan2.2：     /root/data/heyuanyu/yefei/chenyu/data/Wan2.2
checkpoint： /root/data/heyuanyu/yefei/chenyu/models/Wan2.2-I2V-A14B
环境：       /root/data/heyuanyu/yefei/chenyu/wan22_env
GPU：        7
```

这些路径只存在于 `configs/models/wan22_i2v_a14b_server.yaml` 和 build 默认值中，不进入 canonical job schema。若服务器实际位置不同，可以用环境变量覆盖，无需改代码：

```bash
export WAN_REPO=/actual/path/to/Wan2.2
export WAN_CKPT_DIR=/actual/path/to/Wan2.2-I2V-A14B
export WAN_PYTHON=python
export GPU_ID=7
export REMAKE_ROOT=/actual/path/to/physical-parameters/blender/remake
```

## 2. 上传完整首帧数据和代码

服务器应保存完整的 `first_frames_v1_0`，而不是只上传本次抽中的场景。当前完整首帧目录包含 1437 个文件，约 1.1 GB。实验抽样仍由 build/manifest 完成，因此后续更换 selection seed 或新增实验时不需要重新传数据。

本地 PowerShell：

```powershell
cd C:\Users\30561\Desktop\physical-parameter-benchmark\blender\remake
tar -czf remake_code_and_all_first_frames.tar.gz code first_frames_v1_0
scp .\remake_code_and_all_first_frames.tar.gz USER@SERVER:/remote/staging/path/
```

服务器：

```bash
mkdir -p /root/data/heyuanyu/yefei/chenyu/data/physical-parameters/blender/remake
cd /root/data/heyuanyu/yefei/chenyu/data/physical-parameters/blender/remake
tar -xzf /remote/staging/path/remake_code_and_all_first_frames.tar.gz
```

解压后必须保持：

```text
blender/remake/
├── code/
└── first_frames_v1_0/
    ├── images/
    ├── render_results.json
    └── ...
```

不需要上传本地 Wan2.2 模型、checkpoint、Blender `.blend`、renders、scenario、v1/v2/v3；模型和后续其他模型都在服务器侧安装。

## 3. 环境准备

```bash
cd /root/data/heyuanyu/yefei/chenyu/data/physical-parameters/blender/remake
source /root/data/heyuanyu/yefei/chenyu/wan22_env/bin/activate
pip install -r code/requirements.txt
```

框架自身只增加 `PyYAML` 和 `jsonschema`。Wan2.2、PyTorch、CUDA 等仍由 Wan2.2 原有环境提供。

## 4. Smoke dry-run

```bash
MAX_JOBS=2 \
DRY_RUN=1 \
GPU_ID=7 \
bash code/scripts/run_wan22_demo.sh
```

检查生成的：

```text
outputs/<run_id>/resolved_build.yaml
outputs/<run_id>/manifest.jsonl
outputs/<run_id>/manifest.selection.json
outputs/<run_id>/logs/*.command.json
outputs/<run_id>/run_state.jsonl
```

dry-run 不要求 Wan repo/checkpoint 实际存在，但会检查首帧输入、schema、模型能力、Wan 帧数约束和分辨率一致性。

## 5. 两任务基础设施 smoke test（不作为实验结果）

```bash
MAX_JOBS=2 \
GPU_ID=7 \
FAIL_FAST=1 \
bash code/scripts/run_wan22_demo.sh
```

## 6. 后台运行测试或正式抽样实验的全部 20 个任务

```bash
DETACHED=1 \
GPU_ID=7 \
FAIL_FAST=0 \
bash code/scripts/run_wan22_demo.sh
```

脚本会打印 PID、run directory 和 `master.log` 路径。查看进度：

```bash
tail -f outputs/v1a_wan22_demo_*/master.log
```

测试和正式实验均固定包含 baseline，并用 scene selection seed 36 从四个 indoor 候选场景中随机选 2 个、从四个 outdoor 候选场景中随机选 2 个。四个物体分别绑定四个重力值，所以只运行 20 个任务，而不是 80 个 object × gravity 全组合任务。实际抽样结果保存在 `manifest.selection.json`。测试和正式运行时均不要设置 `MAX_JOBS`；该选项仅用于 smoke test。

若需要从已有 run 继续，直接运行 generate，不重新 prepare：

```bash
python code/scripts/remake_benchmark.py generate \
  --run-dir outputs/<existing_run_id> \
  --start-index 0
```

已有且非空的视频默认记为 `skip`。只有显式传入 `--overwrite` 才会重新生成。

## 7. 当前评估与可视化

demo 完成生成后自动运行 `basic_video` 和 `v1a_freefall`：

- 视频是否存在；
- 文件大小是否合理；
- 若系统存在 `ffprobe`，提取 codec、宽高、FPS、帧数和时长；
- 输出 `eval/sample_metrics.jsonl` 与 `eval/aggregate.json`。

自由落体 evaluator 另外输出：

- 每个视频的 `eval/freefall/tracks/<job_id>.csv`；
- 每个视频的二维路线和二次拟合图 `eval/freefall/plots/<job_id>.png`；
- 从五个场景各抽一个代表样本生成检测叠加视频 `eval/freefall/overlays/*.mp4`；
- 可直接查看图和视频的 `eval/report/index.html`；
- 汇总表 `eval/freefall/summary.csv`。

默认拟合量为图像平面竖直加速度 `px/s²`。没有 `pixels_per_meter` 或完整相机标定时，不会把它误报为真实 `m/s²`。具体检测、拟合区间、公式、质量标志和人工核验流程见 `docs/freefall_evaluation.md`。

通过 VS Code Tunnel 查看完整 HTML 报告时，可在服务器终端启动只读静态文件服务：

```bash
python -m http.server 8000 --directory outputs/<run_id>/eval
```

在 VS Code 的 Ports 面板转发 8000 端口，然后打开 `/report/index.html`。报告中的轨迹图和检测叠加视频使用相对路径，可以直接浏览。

## 8. 安装其他模型后的接入方式

每个新模型只需要：

1. 在服务器安装模型仓库、checkpoint 和独立环境；
2. 在 `configs/models/` 新建 model profile；
3. 在 `src/remake_benchmark/models/` 实现 adapter；
4. 在 adapter registry 注册；
5. 复制一个 build，只修改 `build_id` 和 `profiles.model`；
6. 对同一 manifest/prompt/evaluator 先 dry-run，再运行跨模型实验。

模型原生输入、命令和输出文件名均由 adapter 处理；experiment builder 和 evaluator 不应包含新模型的专属判断。
