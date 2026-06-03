# 物理参数视频生成 Benchmark 当前进度技术说明

汇报日期：2026-05-29  
项目仓库：`https://github.com/SS-CC-YY/physical-parameters`  
本地工作区：`C:\Users\30561\Desktop\physical-parameter-benchmark`  
服务器工作区建议路径：`~/data/heyuanyu/yefei/chenyu/data/physical-parameters`

## 1. 项目目标

本项目的目标是构建一个面向视频生成模型的物理参数可控 benchmark。核心问题不是只判断视频“看起来是否合理”，而是判断模型生成的视频是否隐式遵守给定或隐藏的物理参数。

当前阶段的主要任务是建立从物理仿真数据到视频生成模型测试，再到物理量反推评估的完整闭环：

1. 用 Blender 生成物理仿真基准视频、首帧/前十帧条件数据、参数真值和轨迹真值。
2. 将 benchmark 数据通过 Git LFS 管理并同步到服务器。
3. 在服务器上部署开源视频生成模型，优先测试 `Wan2.2-I2V-A14B`。
4. 使用模型根据给定条件图像生成后续视频。
5. 从生成视频中检测目标物体轨迹，并反推对应的隐藏物理参数。
6. 对生成视频、轨迹、估计结果进行可视化，形成可解释的实验报告。

当前汇报重点是：数据集构建已经完成第一批可用版本，Wan2.2 的 I2V 实验代码闭环已经建立，HunyuanVideo-1.5 也已经准备了测试脚本，评估与可视化工具已经补齐，正在进入模型批量实验和结果分析阶段。

## 2. 当前 Benchmark 数据资产

### 2.1 数据总体规模

当前仓库中已经组织了 `v1 / v2 / v3` 三个层级的物理实验，共 68 个 canonical 参数变体，每个变体对应 3 个相机视角。

已整理的 seed 条件数据：

| 类型 | 数量 | 说明 |
|---|---:|---|
| `frame_*.png` | 2040 | 每个 seed 视频抽取前 10 帧，204 组 × 10 帧 |
| `seed_10frames.mp4` | 204 | 每个变体/视角一个前 10 帧视频 |
| `prompt.md` | 204 | 每个变体/视角对应一个实验提示词 |

按版本统计的 `frame_10.png` 条件图像数量：

| 版本 | 数量 | 解释 |
|---|---:|---|
| `v1` | 48 | 16 个变体 × 3 视角 |
| `v2` | 72 | 24 个变体 × 3 视角 |
| `v3` | 84 | 28 个变体 × 3 视角 |
| 合计 | 204 | 68 个变体 × 3 视角 |

渲染视频目前纳入 Git LFS 管理的 `.mp4` 共 208 个。其中 `v1` 目录存在少量历史额外产物，因此渲染视频数略高于 canonical seed 数；正式 seed 条件数据以 `blender/seeds` 下的 204 组为准。

### 2.2 V1 实验设计

V1 是当前优先用于模型可行性实验的最小物理参数集合。每个实验只改变一个隐藏参数，便于判断模型是否对物理量有单调响应。

| 实验 | 任务 | 隐藏参数 | 变体 |
|---|---|---|---|
| `v1_A` | 自由落体 | `g_hidden` | `g2`, `g4p9`, `g9p81`, `g14p7` |
| `v1_B` | 竖直弹跳 | `e_hidden` | `e0p45`, `e0p65`, `e0p82`, `e0p95` |
| `v1_C` | 水平滑动 | `mu_hidden` | `mu0p04`, `mu0p10`, `mu0p18`, `mu0p30` |
| `v1_D` | 阻尼摆 | `gamma_hidden` | `gamma0p04`, `gamma0p10`, `gamma0p18`, `gamma0p30` |

V1 的评估目标：

- `v1_A`：从生成视频中的竖直轨迹拟合重力加速度。
- `v1_B`：从连续反弹峰值高度估计恢复系数。
- `v1_C`：从水平位移曲线估计摩擦导致的减速度，再换算摩擦系数。
- `v1_D`：从摆动幅度包络估计阻尼系数。

### 2.3 V2 / V3 数据状态

V2 和 V3 作为后续更复杂评测层级已经完成 seed 组织：

- `v2`：扩展单参数动力学，共 24 个变体，覆盖抛射、连续弹跳、斜面摩擦、长摆阻尼等。
- `v3`：多物理量耦合任务，共 28 个变体，覆盖抛射+阻力+反弹、弹簧阻尼、斜面碰墙、多参数耦合等。

当前模型实验优先从 V1 做起，因为 V1 的单参数变化更容易定位模型是否具备基本物理参数敏感性。V2/V3 后续用于检验更复杂组合场景下的泛化能力。

## 3. 数据管理与 Git LFS

### 3.1 已启用 LFS 的文件类型

仓库 `.gitattributes` 已配置：

```text
blender/renders/**/*.mp4 filter=lfs diff=lfs merge=lfs -text
blender/seeds/**/*.mp4 filter=lfs diff=lfs merge=lfs -text
blender/seeds/**/*.png filter=lfs diff=lfs merge=lfs -text
```

因此大文件数据走 Git LFS，普通代码和配置走 Git。服务器只需要执行：

```bash
git pull origin main
git lfs pull
```

即可拉取最新代码和 LFS 数据。

### 3.2 服务器 Git 权限问题处理

由于服务器登录账号可能属于他人，之前遇到过 Git dubious ownership 问题。解决方式：

```bash
git config --global --add safe.directory /root/data/heyuanyu/yefei/chenyu/data/physical-parameters
```

如果只想在当前仓库使用个人 SSH key，而不影响服务器上别人的 GitHub 配置，可在仓库局部设置：

```bash
git config core.sshCommand "ssh -i ~/.ssh/id_ed25519_ssccyy -o IdentitiesOnly=yes"
```

这样 GitHub 身份只对当前仓库生效，不污染服务器全局账号设置。

## 4. 已完成的代码模块

### 4.1 Blender 数据生成和 seed 提取

相关目录：

```text
blender/
  v1/
  v2/
  v3/
  renders/
  seeds/
```

已完成：

- 参数化 Blender 场景脚本。
- 多版本、多实验、多变体的渲染产物整理。
- 每个视频前 10 帧 seed video 提取。
- 每个任务对应的 `frame_01.png` 到 `frame_10.png`。
- 每个任务对应的 prompt 文档。
- `extract_seed.py` 已调整为优先选择最新修改的视频，避免历史重复产物覆盖新版 seed。

### 4.2 Phase 1 Wan2.2 单任务基线

目录：

```text
code/phase1_wan22_i2v_baseline/
```

功能：

- 针对 `v1_A` 自由落体构建最小可行性实验。
- 支持 Diffusers 格式权重和 ModelScope/官方原始权重两条路线。
- 提供官方 Wan2.2 `generate.py` 的 wrapper。
- 提供自由落体视频的球心检测、二次曲线拟合和重力 proxy 输出。
- 提供 `flash-attn` 不可用时的 PyTorch SDPA fallback patch。

关键脚本：

| 文件 | 功能 |
|---|---|
| `build_manifest.py` | 构建 `v1_A` 测试 manifest |
| `run_wan22_official_i2v.py` | 调用官方 Wan2.2 `generate.py` |
| `run_wan22_i2v.py` | 调用 Diffusers 权重格式 |
| `evaluate_freefall.py` | 追踪球心并拟合自由落体轨迹 |
| `patch_wan22_attention_fallback.py` | 无 flash-attn 时启用 fallback |

### 4.3 V1 Wan2.2 全量实验工程

目录：

```text
code/v1_wan22_i2v_full/
```

这是当前最重要的实验工程，目标是覆盖 V1 全部 16 个单参数变体，并形成生成、后台运行、评估、可视化闭环。

关键文件：

| 文件 | 功能 |
|---|---|
| `build_v1_manifest.py` | 从 `blender/seeds/v1` 生成 V1 全量 manifest |
| `run_v1_wan22_official.py` | 逐任务调用官方 Wan2.2 I2V 生成视频 |
| `launch_v1_background.sh` | 用 `nohup` 后台运行，防止 VS Code tunnel 断开导致任务终止 |
| `evaluate_v1_physics.py` | 对 `v1_A/B/C/D` 分别反推 `g/e/mu/gamma` |
| `visualize_v1_eval.py` | 生成轨迹图、summary 图、HTML 报告和可选 overlay 视频 |
| `requirements.txt` | 评估和可视化依赖 |
| `README.md` | 服务器运行说明 |

当前已验证：

- `build_v1_manifest.py` 可生成 16 个 V1/CAM_Side 任务。
- `run_v1_wan22_official.py --dry-run` 可正确拼接官方 Wan2.2 运行命令。
- `evaluate_v1_physics.py` 和 `visualize_v1_eval.py` 已完成语法编译检查。
- 可视化脚本可读取 `eval/summary.csv` 和 `eval/tracks/*.csv`，输出 `index.html` 汇总报告。

### 4.4 HunyuanVideo-1.5 测试工程

目录：

```text
hunyuantest/
```

功能：

- 针对 `Tencent-Hunyuan/HunyuanVideo-1.5` 准备 I2V 测试 wrapper。
- 支持 ModelScope 下载的权重目录。
- 提供 `build_i2v_manifest.py`、`run_hunyuan15_official.py`、`evaluate_freefall.py`。
- 默认使用 `v1_A/CAM_Side` 做快速可行性测试。

当前状态：

- 本地代码和 README 已完成。
- dry-run 级别命令构建已验证。
- 尚未作为主线批量评测，当前主线仍是 Wan2.2 I2V。

## 5. 服务器模型与环境进展

### 5.1 已下载或计划下载的模型

当前计划测试三个开源模型：

| 模型 | 当前状态 | 用途 |
|---|---|---|
| `Wan2.2-I2V-A14B` | 已在服务器 `models` 目录下载 | 当前主线 baseline |
| `HunyuanVideo-1.5` | 已通过 ModelScope 下载 | 第二个 I2V/T2V 测试模型 |
| `CogVideoX1.5` | 已列入计划 | 后续对比模型 |

Wan2.2 的主要路径假设：

```text
~/data/heyuanyu/yefei/chenyu/data/
  physical-parameters/
  Wan2.2/
  models/Wan2.2-I2V-A14B/
```

### 5.2 Python 环境问题与处理策略

服务器当前环境中，`base` 环境已有可用 PyTorch：

```text
torch 2.5.1+cu124
torchvision 0.20.1+cu124
torchaudio 2.5.1+cu124
```

问题：

- 自建 `phy` venv 中没有 torch。
- 在空 venv 中安装 torch 会反复下载大量 `nvidia-*` CUDA wheel，速度慢且容易失败。
- 官方 Wan2.2 依赖中包含 `flash_attn`，直接 pip 安装时可能从源码编译失败或 GitHub 下载中断。

当前推荐策略：

1. 优先复用已有 `base` 环境中的 torch。
2. 如果需要隔离环境，使用：

```bash
python -m venv --system-site-packages wan22_env
source ~/data/heyuanyu/yefei/chenyu/data/wan22_env/bin/activate
```

3. 对 Wan2.2 安装非 torch 依赖，避免重复装 CUDA。
4. `flash-attn` 优先下载匹配 wheel 后上传服务器安装。
5. 如果短期无法安装 `flash-attn`，使用项目中的 fallback patch 先跑 smoke test。

### 5.3 flash-attn 当前处理方案

服务器环境大概率对应：

```text
Python: cp311
Torch: 2.5.1+cu124
CUDA wheel family: cu12
CXX11 ABI: FALSE
```

匹配 wheel 文件：

```text
flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

安装方式：

```bash
python -m pip install --no-deps ./flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

验证：

```bash
python - <<'PY'
import torch, flash_attn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
print("torch:", torch.__version__, torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
print("flash_attn OK")
PY
```

如果不用 flash-attn，可执行：

```bash
python code/phase1_wan22_i2v_baseline/patch_wan22_attention_fallback.py \
  --wan-repo ../Wan2.2
```

该 fallback 适合 smoke test，但速度和显存效率不如原生 flash-attn。

## 6. Wan2.2 V1 全量实验运行流程

### 6.1 拉取最新仓库与 LFS 数据

服务器上执行：

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters
git pull origin main
git lfs pull
```

如遇 dubious ownership：

```bash
git config --global --add safe.directory /root/data/heyuanyu/yefei/chenyu/data/physical-parameters
```

### 6.2 安装评估依赖

```bash
python -m pip install -r code/v1_wan22_i2v_full/requirements.txt
```

当前 `requirements.txt` 包含：

```text
numpy
opencv-python
matplotlib
```

### 6.3 构建 V1 manifest

```bash
python code/v1_wan22_i2v_full/build_v1_manifest.py \
  --seeds-root blender/seeds \
  --output outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --prompt-mode explicit \
  --cameras CAM_Side
```

输出：16 个任务，覆盖 V1 所有实验和变体。

当前默认使用 `CAM_Side`，原因是反推物理量需要稳定的侧视轨迹。其他视角可以用于后续跨视角一致性分析，但不是第一阶段反推物理量的最佳视角。

### 6.4 后台运行生成任务

为了避免 VS Code tunnel 或 SSH 连接断开导致任务终止，使用 `nohup` 启动：

```bash
bash code/v1_wan22_i2v_full/launch_v1_background.sh
```

默认参数：

```text
WAN_REPO=../Wan2.2
CKPT_DIR=../models/Wan2.2-I2V-A14B
OUTDIR=outputs/v1_wan22_i2v_a14b
GPU_ID=7
FRAME_NUM=121
SIZE=832*480
PROMPT_MODE=explicit
CAMERAS=CAM_Side
```

监控：

```bash
tail -f outputs/v1_wan22_i2v_a14b/run.log
cat outputs/v1_wan22_i2v_a14b/run.pid
ps -fp "$(cat outputs/v1_wan22_i2v_a14b/run.pid)"
```

停止：

```bash
kill "$(cat outputs/v1_wan22_i2v_a14b/run.pid)"
```

恢复：

```bash
bash code/v1_wan22_i2v_full/launch_v1_background.sh
```

脚本默认跳过已存在视频，因此支持断点续跑。

### 6.5 评估生成视频

```bash
python code/v1_wan22_i2v_full/evaluate_v1_physics.py \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --outdir outputs/v1_wan22_i2v_a14b/eval
```

输出：

```text
outputs/v1_wan22_i2v_a14b/eval/summary.csv
outputs/v1_wan22_i2v_a14b/eval/summary.json
outputs/v1_wan22_i2v_a14b/eval/tracks/*.csv
```

### 6.6 可视化评估结果

```bash
python code/v1_wan22_i2v_full/visualize_v1_eval.py \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --eval-dir outputs/v1_wan22_i2v_a14b/eval \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl
```

输出：

```text
outputs/v1_wan22_i2v_a14b/eval/visualization/index.html
outputs/v1_wan22_i2v_a14b/eval/visualization/summary/*.png
outputs/v1_wan22_i2v_a14b/eval/visualization/jobs/*_track.png
```

如需视频 overlay：

```bash
python code/v1_wan22_i2v_full/visualize_v1_eval.py \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --eval-dir outputs/v1_wan22_i2v_a14b/eval \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --make-overlays
```

## 7. 评估算法说明

### 7.1 目标检测

当前评估脚本使用 OpenCV HSV 颜色分割检测橙色球体：

1. 读取视频帧。
2. 转换到 HSV 色彩空间。
3. 根据橙色/红橙色阈值提取 mask。
4. 做形态学开闭操作去噪。
5. 找最大连通区域作为球。
6. 用 contour moment 得到球心 `(cx_px, cy_px)`。
7. 用最小外接圆估计球半径像素值。

这一方法的优点是依赖少、速度快、可解释。缺点是当模型生成的视频中球体颜色变化、阴影严重、球体形变或出现多个橙色区域时，轨迹会出现噪声。

### 7.2 像素到物理量的近似标定

当前脚本使用球半径进行局部比例估计：

```text
m_per_px = ball_radius_m / radius_px
```

默认球半径：

```text
ball_radius_m = 0.24
```

该标定是近似方法，适用于第一阶段方向性验证；如果要做严格物理量绝对误差，需要后续引入相机内外参和透视投影反解。

### 7.3 各实验反推方式

#### v1_A：自由落体重力

从 `y_px(t)` 拟合二次曲线：

```text
y(t) = a t^2 + b t + c
```

像素加速度：

```text
a_px = 2a
```

物理重力估计：

```text
g_hat = |a_px| * m_per_px
```

#### v1_B：弹跳恢复系数

检测弹跳峰值高度：

```text
h_1, h_2, h_3, ...
```

恢复系数估计：

```text
e_hat = geometric_mean(sqrt(h_{n+1} / h_n))
```

#### v1_C：滑动摩擦系数

从水平位移拟合：

```text
x(t) = a t^2 + b t + c
```

水平减速度：

```text
a_x = |2a| * m_per_px
```

摩擦系数：

```text
mu_hat = a_x / 9.81
```

#### v1_D：阻尼摆系数

从水平摆动幅度提取包络：

```text
A(t) = |x(t) - x_center|
```

对峰值幅度拟合：

```text
ln A(t) = ln A0 - gamma * t
```

阻尼估计：

```text
gamma_hat = -slope
```

## 8. 当前技术判断

### 8.1 为什么先用 explicit prompt

当前 Wan2.2 I2V 只接收单张 `frame_10.png` 作为条件。对于自由落体、滑动、弹跳和阻尼摆，真实参数往往需要从前 10 帧的速度、加速度或峰值趋势中推断。

单图 I2V 模型看不到完整前 10 帧历史，因此它很难严格“隐式推断”物理参数。为了先验证完整 pipeline 是否可运行，当前默认采用 `prompt-mode explicit`：

```text
在 prompt 中明确写入目标隐藏参数值。
```

这样可以先回答第一阶段问题：

1. 模型是否能稳定生成可追踪的视频？
2. 生成结果是否会随参数值产生方向性变化？
3. 评估脚本能否从生成视频中提取轨迹并输出物理量估计？

后续再切换到 `prompt-mode hidden` 或视频续写模型，评估模型真正从前 10 帧推断隐藏参数的能力。

### 8.2 当前观察到的可视化问题

在某些生成结果的 `image-plane track` 图中，轨迹呈现明显锯齿，`x_px` 只在很小范围内抖动。这通常说明两类问题之一：

1. 模型生成的视频中目标物体几乎没有产生符合预期的物理运动。
2. OpenCV 颜色检测没有稳定锁定球心，检测点在球表面、阴影或噪声之间跳动。

因此可视化脚本已经支持 overlay 视频，便于判断是模型生成失败，还是检测算法需要调整。

## 9. 当前风险与待解决问题

### 9.1 依赖和环境风险

- `flash-attn` 安装仍是 Wan2.2 官方路线的主要风险点。
- 服务器网络访问 GitHub 不稳定，建议本地下载 wheel 后上传服务器。
- `base` 环境可用，但混用系统环境和 venv 时容易出现 `transformers/tokenizers` 版本冲突。

### 9.2 模型能力风险

- 单图 I2V 模型缺少速度历史，天然不适合严格 hidden-parameter inference。
- 模型可能保持画面外观但不遵守运动规律。
- 对低重力、高弹性、强阻尼等极端参数，模型可能回归常见物理先验。

### 9.3 评估算法风险

- 当前反推使用颜色追踪和近似像素标定，适合第一阶段 proxy，不等同于严格物理测量。
- 如果模型改变球体颜色、形状或引入多目标，轨迹检测会不稳定。
- 透视效应、相机角度和视频分辨率变化会影响物理量绝对值估计。

### 9.4 数据版本风险

- 当前 render 视频中 `v1` 有少量历史额外产物，因此正式实验应以 `blender/seeds` 和 manifest 为准。
- 新增的 `code/v1_wan22_i2v_full` 和 `hunyuantest` 目录目前属于本地新增工程，正式服务器使用前需要 commit/push 或手动同步。

## 10. 下一步计划

### 10.1 短期：下午汇报后立即推进

1. 将 `code/v1_wan22_i2v_full` 和 `hunyuantest` 提交并推送到 GitHub。
2. 服务器执行 `git pull && git lfs pull`。
3. 安装或修复 `flash-attn`。
4. 用 Wan2.2 跑 `MAX_JOBS=1` smoke test。
5. 确认能生成视频后运行 V1 全量 16 任务。
6. 执行 `evaluate_v1_physics.py` 输出 summary。
7. 执行 `visualize_v1_eval.py` 生成 HTML 报告和轨迹图。

### 10.2 中期：模型对比

1. 跑 HunyuanVideo-1.5 的 `v1_A` smoke test。
2. 将 HunyuanVideo-1.5 扩展到 V1 全量任务。
3. 下载并接入 CogVideoX1.5。
4. 对比不同模型在同一物理参数集上的趋势一致性和误差。

### 10.3 长期：Benchmark 强化

1. 将评估从像素 proxy 升级为相机几何反投影。
2. 引入视频续写模型，以前 10 帧视频作为条件，而不是单张 `frame_10.png`。
3. 加入多视角一致性指标。
4. 扩展到 V2/V3 的复杂动力学和多参数耦合。
5. 建立自动化结果表格和论文级可视化。

## 11. 汇报时可强调的阶段性成果

1. 已完成 68 个 canonical 物理变体、204 组 seed 条件数据、2040 张前十帧图像的组织。
2. 已使用 Git LFS 建立大文件管理和服务器同步流程。
3. 已解决服务器 Git 身份、安全目录、LFS 拉取等工程问题。
4. 已完成 Wan2.2 I2V 的官方 ModelScope 权重运行 wrapper。
5. 已完成 V1 全量 16 任务的自动 manifest、后台生成、断点续跑方案。
6. 已完成 V1 四类物理参数的反推评估代码。
7. 已完成可视化工具，包括轨迹图、target-vs-estimate 图、HTML 报告和 overlay 视频。
8. 已为 HunyuanVideo-1.5 准备独立测试工程，后续可快速扩展到模型对比。
9. 当前主要工作从“数据和代码搭建”进入“服务器批量实验与结果解释”阶段。

## 12. 可用于现场展示的命令清单

### 查看数据规模

```bash
find blender/seeds -name frame_10.png | wc -l
find blender/seeds -name seed_10frames.mp4 | wc -l
find blender/seeds -name prompt.md | wc -l
```

### 生成 V1 manifest

```bash
python code/v1_wan22_i2v_full/build_v1_manifest.py \
  --seeds-root blender/seeds \
  --output outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --prompt-mode explicit \
  --cameras CAM_Side
```

### 后台运行 Wan2.2 V1 实验

```bash
bash code/v1_wan22_i2v_full/launch_v1_background.sh
```

### 查看任务进度

```bash
tail -f outputs/v1_wan22_i2v_a14b/run.log
```

### 评估

```bash
python code/v1_wan22_i2v_full/evaluate_v1_physics.py \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --outdir outputs/v1_wan22_i2v_a14b/eval
```

### 可视化

```bash
python code/v1_wan22_i2v_full/visualize_v1_eval.py \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --eval-dir outputs/v1_wan22_i2v_a14b/eval \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --make-overlays
```

## 13. 当前结论

当前项目已经完成从物理仿真数据、条件输入、Git LFS 同步、Wan2.2 生成、物理参数反推、可视化报告的完整工程闭环。现阶段最关键的下一步不是继续扩展更多任务，而是在服务器上稳定跑完 V1 全量实验，拿到第一批真实模型输出结果，并用可视化报告判断：

1. 模型是否能生成可追踪的物理视频；
2. 模型输出是否随物理参数变化产生可测趋势；
3. 评估算法的追踪稳定性是否足够支撑后续严格 benchmark；
4. 单图 I2V 模型在 hidden-parameter 推断上的能力边界在哪里。

这批结果将决定下一阶段是优先优化评估/追踪，还是转向更适合前 10 帧条件的视频续写模型。
