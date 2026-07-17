# 架构说明

## 1. 设计原则

### 实验语义与模型执行解耦

实验层只描述“要测试什么”：场景、物理参数、输入资产、prompt mode、随机种子和评估目标。模型层只描述“如何调用某个模型”。实验配置中不应出现 `generate.py`、模型仓库路径或模型专属 CLI 参数。

### Prompt 内容与模型格式解耦

prompt 模块先产生结构化内容，例如：

- 场景保持要求；
- 动作/物理过程；
- 已知参数；
- 隐藏或显式目标参数；
- 摄像机约束；
- 禁止项；
- prompt mode。

所有实验使用 `ppb_common_physics_video_v1` 公共框架，并按 `task → scene → camera → dynamics → parameter → terminal → quality` 的固定顺序渲染。实验配置只覆盖动力学、参数表达和终止行为，场景保持、静态镜头、物体一致性、时序质量和公共禁止项不能各自重写。

model adapter 再决定负面提示是独立参数、拼接进正面提示，还是需要转换成模型自己的输入 JSON。

### 生成与评估只通过规范化产物连接

评估器不读取 Wan、Cosmos 等模型内部目录，也不解析模型运行命令。它只接收 canonical job、规范化视频路径和 metadata。因此同一个评估器可以直接比较不同模型。

当前正式调度采用 generation-first：先为每个目标模型完成并核对完整 manifest 的全部规范化视频，再单独启动 evaluator。`workflow: generation_only` 的 build 禁止 `sequence` 入口，避免生成阶段因检测阈值或拟合问题改变任务覆盖率。

### 配置可复现，运行状态可恢复

每次运行先冻结 `resolved_build.yaml` 和 `manifest.jsonl`。运行状态另写 `run_state.jsonl`，不原地修改 manifest。失败任务可按 job id 重跑，已有成功视频默认跳过。

## 2. 六个核心层

### A. Experiment / Task Builder

负责扫描 benchmark release 或 scenario 资产，生成模型无关的任务。它决定：

- 哪些实验、场景、物体、相机和参数组合参与；
- 输入是文字、首帧、种子视频还是其他条件；
- job id、随机种子和重复次数；
- ground truth 与评估分组字段。

它不负责调用模型。

### B. Prompt Renderer

根据实验定义生成稳定、可审计的 prompt。建议将 prompt 拆成结构化字段，并同时在 manifest 中保存：

- `prompt_spec`：语义结构，便于后续分析；
- `prompt`：最终正向文本；
- `negative_prompt`：最终负向文本；
- `prompt_template_id` 和版本。

这样 prompt 模板变更不会悄悄污染不同批次的比较。

### C. Build Resolver

build profile 引用四类配置：

```text
experiment + prompt + model + evaluation
```

resolver 校验兼容性并生成完整快照。例如 I2V 实验必须提供 conditioning image，所选 model adapter 也必须声明支持 I2V。

### D. Model Adapter

每个模型 adapter 只实现模型差异：

- 声明支持的输入模态与能力；
- 校验任务是否可运行；
- 将 canonical job 转成模型原生输入；
- 构建命令或调用 Python API；
- 设置模型需要的环境变量；
- 定位并规范化生成视频；
- 记录 checkpoint、代码版本、参数、耗时和错误。

调度、重试、日志、状态管理和输出布局属于通用 runner，不应在每个 adapter 中重复。

### E. Evaluator

评估分两级：

1. 通用质量门控：视频可读性、帧数、FPS、尺寸、静帧/黑屏、输入物体保持、镜头变化等；
2. 实验专属评估：轨迹、碰撞、反弹、摩擦、摆动或其他物理参数指标。

每个 evaluator 输出逐样本结构化指标和可选 artifact，不直接负责跨模型画总报告。

### F. Summarizer / Reporter

读取统一的 sample metrics，根据 `model_id`、experiment、scene、object、camera、target value、seed 等维度聚合，输出：

- 成功率和无效样本率；
- 相关性、排序一致性、误差等物理指标；
- bootstrap 置信区间或重复实验统计；
- 跨模型对比表和可视化；
- 配置、代码版本和缺失样本说明。

## 3. Build 切换模型的方式

推荐 build 只保存引用关系：

```text
build_id: v1a_wan22_explicit
experiment: experiments/v1a_all_values
prompt: prompts/explicit_v1
model: models/wan22_i2v_a14b
evaluation: evaluations/v1a_freefall
```

换模型时复制一个很小的 build profile，只修改 `build_id` 和 `model`。如果某模型需要特殊 prompt，不应直接改公共模板；应明确创建新的 prompt profile，使这一差异进入实验记录。

## 4. 一次运行的状态流转

```text
prepared → pending → running → ok
                         └────→ error
             └───────────────→ skip（已有有效结果）
```

状态记录至少包含时间、尝试次数、返回码、错误摘要、输出路径和耗时。任务是否成功不能只看进程返回码，还应确认视频存在并通过最基本的可读性检查。

## 5. 与旧代码的对应关系

| 旧代码职责 | 新框架位置 |
|---|---|
| `build_*_manifest.py` | experiment builder + prompt renderer |
| `run_*_official.py` 中通用循环 | orchestration runner |
| `run_*_official.py` 中模型命令 | model adapter |
| `evaluate_*` | evaluator plugin |
| `visualize_*` / `summarize_*` | summarizer + reporter |
| `run_*_background.sh` | 薄 launcher 或集群提交配置 |

旧 manifest 中的 `job_id`、conditioning input、prompt、hidden/known params、seed 等可映射到 canonical schema；`cosmos_inference_type`、Wan task 名和 checkpoint 参数则应移到 model config/metadata。

## 6. 暂不放进核心框架的内容

- Slurm、nohup、具体服务器路径和 GPU 编号：属于部署配置；
- 某模型仓库内的源码补丁：属于该 adapter 的安装说明；
- 针对单批结果的临时可视化脚本：成熟后再沉淀为 reporter；
- Blender 场景生成逻辑：仍由 benchmark/scenario 侧负责，框架只消费其 release/manifest。
