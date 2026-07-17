# Remake 多模型视频实验框架

本目录是 `blender/remake` 后续视频生成实验的统一代码入口。当前已经实现第一阶段可运行版本：build/schema、canonical manifest、通用 runner、Wan2.2 reference adapter 和基础视频完整性评估。

设计参考了旧 `physical-parameter-benchmark/code` 中已有的 Wan2.2、Cosmos-Predict2.5 和 remake demo 流程，保留以下有效做法：

- 用 JSONL manifest 固化每一个生成任务；
- 生成、评估、汇总分阶段执行；
- 每个任务保存 prompt、输入、生成参数、命令、日志和输出路径；
- 支持 `dry-run`、`max-jobs`、断点续跑和跳过已有结果；
- 生成结果采用统一目录布局，便于跨模型比较。

同时把旧代码中分散在不同脚本里的模型专属逻辑移到统一的 model adapter 中，避免为每个模型复制一套 prompt、运行和评估脚本。

所有实验 prompt 强制遵循 `ppb_common_physics_video_v1`：固定按 `task → scene → camera → dynamics → parameter → terminal → quality` 组织。实验 profile 只能补充动力学、参数表达和终止行为，公共的场景保持、静态镜头、物体一致性、时序质量与 negative prompt 由同一基础 profile 提供并经过 schema 校验。

## 目标流程

```text
benchmark/scenario 数据
        ↓
实验任务构建（数据、变量、种子、输入模态）
        ↓
结构化 prompt 生成
        ↓
canonical manifest.jsonl
        ↓
build profile 选择模型与运行参数
        ↓
model adapter（Wan / Cosmos / 其他模型）
        ↓
完整生成所有 jobs（模型进程保持常驻，支持断点续跑）
        ↓
videos + metadata + logs
        ↓
生成阶段全部结束后，再单独启动评估与跨模型汇总
```

## 当前目录

```text
code/
├── README.md                    # 本说明
├── schemas/                     # build、job 与统一 prompt 的 JSON Schema
├── docs/
│   ├── architecture.md          # 模块边界、运行流程和扩展方式
│   └── contracts.md             # manifest、模型和评估的接口契约
├── builds/                      # 一次实验的组装配置；切换 build 即切换模型/参数
├── configs/
│   ├── experiments/             # 实验任务、变量、场景与采样范围
│   ├── models/                  # checkpoint、分辨率、帧数、推理参数
│   ├── prompts/                 # 公共 prompt 框架与实验差异 profile
│   └── evaluations/             # 评估器和阈值组合
├── src/remake_benchmark/
│   ├── core/                    # schema、路径、日志、状态和通用 I/O
│   ├── prompts/                 # 模型无关的结构化提示生成
│   ├── models/                  # 统一接口及各模型 adapter
│   ├── evaluators/              # 通用质量与实验专属物理评估
│   └── orchestration/           # build、generate、evaluate、summarize 流程
├── scripts/                     # 薄命令行入口；不承载核心业务逻辑
└── tests/                       # schema、prompt、adapter 和 evaluator 测试
```

当前已实现 Wan2.2 所需的 core、prompts、models、evaluators 和 orchestration 模块；未实现的其他模型与实验评估器继续按相同接口扩展。

## Build 的定位

建议以后用一个 build profile 表示“一次可复现实验的完整组装”，而不是为每个模型复制一份完整代码。例如：

```text
builds/v1a_wan22_demo.yaml
builds/v1a_cosmos25_explicit.yaml
builds/v1a_new_model_explicit.yaml
```

三个 build 可以引用完全相同的 experiment、prompt 和 evaluation 配置，仅替换 `model` 配置。这样得到的 manifest 语义和评估口径保持一致，模型间差异也更容易解释。

需要注意：旧代码里的 `build_*_manifest.py` 是“构建任务清单”，而新框架中的 build profile 是“组装整条实验流水线”。实现时建议把前者统一命名为 `prepare` 或 `make-manifest`，避免概念混淆。

## 已实现命令界面

```text
prepare   --build <build.yaml> --run-dir <dir>
generate  --run-dir <dir> [--max-jobs N] [--dry-run]
evaluate  --run-dir <dir>
sequence  --run-dir <dir> [--max-jobs N] [--dry-run]
run       --build <build.yaml> --run-dir <dir>
```

`generate` 和 `run` 当前都只负责生成，不会调用 evaluator。`sequence` 作为旧的显式诊断入口保留，但正式批量生成不得使用它。评估必须等所有目标模型的视频生成完成后，再由人工明确执行 `evaluate`。生成阶段不得修改已经固化的 manifest，只把实际运行信息写入 metadata 和 run state。

无需安装 package 也可通过薄入口运行：

```bash
python code/scripts/remake_benchmark.py --help
```

## 标准球优先生成矩阵

首轮正式 build 为 `builds/standard_ball_all_experiments_wan22_generation.yaml`。它通过 build override 复用完整实验配置，只选择 `standard_ball` 并将默认输出设为 81 帧：

- 13 个实验：`v1_A–v1_D`、`v2_A–v2_E`、`v3_A–v3_D`；
- 69 个冻结 parameter anchor tuples；
- 9 个场景：baseline、4 个 indoor、4 个 outdoor；
- 1 个物体：`standard_ball`；
- 3 个视角；
- seed 36；
- 每模型共 `69 × 9 × 1 × 3 = 1863` 个视频；
- 使用 351 张标准球首帧 PNG，每个 PNG 对应多个显式物理参数 continuation；
- 默认 81 帧、16 fps、40 sampling steps，Wan2.2 模型常驻且 H20 默认不 offload。

V3 多参数实验只运行 registry 中有物理意义的联合参数锚点，不把各参数独立做笛卡尔积。尤其 `v3_B` 的冻结实验是“带摩擦的左右墙非对称重复碰撞”，不是旧文件名所暗示的弹簧阻尼实验。

执行：

```bash
DETACHED=1 GPU_ID=7 \
bash code/scripts/run_standard_ball_generation.sh
```

脚本只调用 `prepare` 和 `generate`；不会产生 `eval/`，也不会因检测或拟合结果中断生成。相同 `RUN_DIR` 再次执行会复用不可变 manifest，并跳过已有非空视频。

完整的四物体、161 帧 build `builds/all_experiments_wan22_generation.yaml` 仍然保留，但当前不运行。81 帧是 speed-first 条件，计算量显著降低；部分长周期 V2/V3 实验在未来做完整拟合时可能需要单独补跑 161 帧条件。

在正式生成前，可用 `builds/v1a_seed_variation_wan22.yaml` 对完全相同的首帧和 prompt 运行 seeds 36–39，仅生成 4 条视频：

```bash
RUN_ID=v1a_seed_variation_wan22 \
GPU_ID=7 WAN_OFFLOAD_MODEL=false \
bash code/scripts/run_seed_variation.sh

bash code/scripts/make_seed_variation_grid.sh \
  outputs/v1a_seed_variation_wan22
```

四宫格布局为左上 seed36、右上 seed37、左下 seed38、右下 seed39。该 smoke build 与正式 1863-job manifest 完全分开。

服务器有四张可用卡时，推荐让 GPU 4、5、6、7 各生成一个 seed：

```bash
RUN_ID=v1a_seed_variation_wan22_4gpu_$(date +%Y%m%d_%H%M%S)
RUN_ID="$RUN_ID" GPU_IDS=4,5,6,7 WAN_OFFLOAD_MODEL=false \
  bash code/scripts/run_seed_variation_4gpu.sh

bash code/scripts/check_four_gpu_generation.sh "outputs/$RUN_ID"
python code/scripts/collect_sharded_run.py "outputs/$RUN_ID"
bash code/scripts/make_seed_variation_grid.sh "outputs/$RUN_ID"
```

正式标准球清单可用相同机制分为 `466 + 466 + 466 + 465 = 1863` 条：

```bash
RUN_ID=standard_ball_all13_wan22_4gpu_$(date +%Y%m%d_%H%M%S)
RUN_ID="$RUN_ID" GPU_IDS=4,5,6,7 WAN_OFFLOAD_MODEL=false \
  bash code/scripts/run_standard_ball_generation_4gpu.sh
```

每张卡拥有独立的 `shards/gpu-<id>/`、常驻 worker、日志和状态文件。全部完成后运行 `collect_sharded_run.py`；它先验证 1863 个视频及 metadata 完整且无重复，再用硬链接汇总到父 run 的标准 `videos/` 和 `metadata/`，不会复制 MP4 数据。

## 旧 Wan2.2 60-job demo

demo build 为 `builds/v1a_wan22_demo.yaml`，使用：

- `v1_A` 自由落体；
- 固定包含 `baseline`；
- 使用 scene selection seed 36，从 `indoor1–4` 随机选 2 个、从 `outdoor1–4` 随机选 2 个；
- 4 个物体分别绑定 4 个重力值，不运行 object × gravity 全组合；
- `CAM_Main`、`CAM_Side`、`CAM_Top` 三视角；
- 4 个显式重力值 `2.0, 4.9, 9.81, 14.7 m/s^2`；
- seed 36；
- 共 `5 场景 × 4 物体 × 3 视角 = 60` 个 I2V 任务。

该 build 只保留用于兼容已经生成的 smoke run，不再作为正式实验清单。`run_wan22_demo.sh` 也已改为纯生成。

服务器配置和运行命令见 [docs/server_setup.md](docs/server_setup.md)。建议第一次先运行两个任务的 dry-run：

```bash
MAX_JOBS=3 DRY_RUN=1 bash code/scripts/run_wan22_demo.sh
```

dry-run 会完成 schema 校验、build 展开、输入检查、manifest 写入和完整模型命令生成，但不会加载模型。

## 预期输出目录

```text
outputs/<run_id>/
├── resolved_build.yaml          # 展开后的完整 build 快照
├── manifest.jsonl               # 不可变的 canonical jobs
├── run_state.jsonl              # pending/running/ok/error/skip
├── run_summary.json             # 纯生成累计计数
├── videos/<job_id>.mp4
├── metadata/<job_id>.json
├── logs/<job_id>.*.log
└── eval/                        # 仅在未来显式运行 evaluate 后出现
    ├── sample_metrics.jsonl
    ├── aggregate.json
    ├── freefall/
    │   ├── summary.csv
    │   ├── tracks/
    │   ├── plots/
    │   └── overlays/
    └── report/index.html
```

模型原生输出文件名可以不同，但 adapter 必须将最终视频规范化为 `videos/<job_id>.mp4`，并保留原始输出路径和转换记录。

## 实现顺序建议

1. 已完成 canonical manifest schema、build 解析和校验；
2. 已完成显式参数 prompt renderer 和 Wan2.2 reference adapter；
3. 已完成三视角逐样本跟踪门控、独立刚性穿模判定、独立参数反推/相似度及可视化；
4. 下一步用服务器真实输出复核检测阈值，并补充相机内外参与尺度标定；
5. 在服务器安装第二个模型并增加对应 adapter/model profile；
6. 最后补齐并行 GPU、跨模型汇总报告和打包。

具体模块责任及接口见 [docs/architecture.md](docs/architecture.md)、[docs/contracts.md](docs/contracts.md) 和 [docs/freefall_evaluation.md](docs/freefall_evaluation.md)。
