# Remake 多模型视频实验框架（设计稿）

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
统一 videos + metadata + logs
        ↓
通用质量检查 + 实验专属物理评估
        ↓
逐样本结果 + 跨模型汇总报告
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
run       --build <build.yaml> --run-dir <dir>
```

`run` 只是依次调用 prepare、generate 和 evaluate；每一步都可以单独重跑。生成阶段不得修改已经固化的 manifest，只把实际运行信息写入 metadata 和 run state。跨模型 summarize/report 仍属于下一阶段。

无需安装 package 也可通过薄入口运行：

```bash
python code/scripts/remake_benchmark.py --help
```

## Wan2.2 完整 demo

demo build 为 `builds/v1a_wan22_demo.yaml`，使用：

- `v1_A` 自由落体；
- 固定包含 `baseline`；
- 使用 scene selection seed 36，从 `indoor1–4` 随机选 2 个、从 `outdoor1–4` 随机选 2 个；
- 4 个物体分别绑定 4 个重力值，不运行 object × gravity 全组合；
- `CAM_Side`；
- 4 个显式重力值 `2.0, 4.9, 9.81, 14.7 m/s^2`；
- seed 36；
- 共 `5 场景 × 4 物体 = 20` 个 I2V 任务。

测试和正式实验使用相同的五场景抽样规则，均运行这 20 个任务。`MAX_JOBS=2` 只用于检查环境和命令的基础 smoke test，不作为实验结果。每次 prepare 都会把计划抽取的场景和实际写入 manifest 的场景记录到 `manifest.selection.json`，因此随机选择可检查、可复现。

服务器配置和运行命令见 [docs/server_setup.md](docs/server_setup.md)。建议第一次先运行两个任务的 dry-run：

```bash
MAX_JOBS=2 DRY_RUN=1 bash code/scripts/run_wan22_demo.sh
```

dry-run 会完成 schema 校验、build 展开、输入检查、manifest 写入和完整模型命令生成，但不会加载模型。

## 预期输出目录

```text
outputs/<run_id>/
├── resolved_build.yaml          # 展开后的完整 build 快照
├── manifest.jsonl               # 不可变的 canonical jobs
├── run_state.jsonl              # pending/running/ok/error/skip
├── videos/<job_id>.mp4
├── metadata/<job_id>.json
├── logs/<job_id>.*.log
├── eval/
│   ├── sample_metrics.jsonl
│   ├── aggregate.json
│   ├── freefall/
│   │   ├── summary.csv
│   │   ├── tracks/
│   │   ├── plots/
│   │   └── overlays/
│   └── report/index.html
```

模型原生输出文件名可以不同，但 adapter 必须将最终视频规范化为 `videos/<job_id>.mp4`，并保留原始输出路径和转换记录。

## 实现顺序建议

1. 已完成 canonical manifest schema、build 解析和校验；
2. 已完成显式参数 prompt renderer 和 Wan2.2 reference adapter；
3. 已完成基础输出质量检查和自由落体轨迹/加速度代理拟合；
4. 下一步用服务器生成结果校准检测阈值，并决定是否加入相机尺度标定；
5. 在服务器安装第二个模型并增加对应 adapter/model profile；
6. 最后补齐并行 GPU、跨模型汇总报告和打包。

具体模块责任及接口见 [docs/architecture.md](docs/architecture.md)、[docs/contracts.md](docs/contracts.md) 和 [docs/freefall_evaluation.md](docs/freefall_evaluation.md)。
