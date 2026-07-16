# 接口契约（设计稿）

本文只定义应稳定的边界，不给出具体类或函数实现。

## 1. Canonical Job Manifest

manifest 使用 JSONL，每行一个 job。字段建议分成以下几组。

### 身份与追踪

- `schema_version`：schema 版本；
- `job_id`：稳定且文件名安全的唯一标识；
- `experiment_id`、`case_id`、`repeat_id`；
- `build_id`：由哪个 resolved build 产生；
- `seed`。

### 输入

- `task_type`：`t2v`、`i2v`、`v2v` 等；
- `inputs.text`；
- `inputs.image` 或 `inputs.video`；
- `inputs.source_fps`、conditioning frame/time；
- 所有资产路径相对 benchmark/repo root，运行时统一解析。

### Prompt

- `prompt_spec`：结构化语义；
- `prompt`、`negative_prompt`：渲染后的最终文本；
- `prompt_mode`；
- `prompt_template_id`、`prompt_template_version`。

### 实验真值与分组

- `targets`：待检验的隐藏/显式物理参数；
- `known_params`；
- `factors`：scene、object、camera、lighting 等分组因子；
- `units`：与参数名对应的单位；
- `evaluation_tags`：应运行哪些评估器。

### 生成请求（模型无关）

- 期望时长或帧数；
- 期望 FPS；
- 输出宽高或 aspect ratio；
- 重复采样数等公共参数。

manifest 中不应包含模型仓库绝对路径、CUDA 设备号或模型专属命令参数。这些属于 resolved build 和运行 metadata。

## 2. Model Adapter Contract

每个 adapter 应提供以下能力：

1. `capabilities`：声明支持 T2V/I2V/V2V、负面提示、分辨率限制、帧数约束等；
2. `validate`：在占用 GPU 前发现缺失输入和不兼容参数；
3. `prepare`：生成模型原生 JSON/临时输入，但不修改 canonical manifest；
4. `invoke`：返回可记录的调用描述并执行；
5. `collect`：找到模型原生输出并规范化到统一视频路径；
6. `provenance`：记录模型名、checkpoint、revision、代码 revision 和实际参数。

adapter 的成功结果至少包含：

- `job_id`；
- canonical output video；
- 模型原生输出路径（若不同）；
- 开始/结束时间与耗时；
- 实际 generation parameters；
- stdout/stderr/command 或 API 调用摘要；
- 状态与错误信息。

## 3. Evaluator Contract

评估器输入：

- 一条 canonical job；
- canonical video；
- generation metadata；
- evaluator config。

评估器输出一条 sample result，至少包含：

- `job_id`、`evaluator_id`、`evaluator_version`；
- `status`：`ok`、`invalid`、`error`；
- `quality_flags` 和无效原因；
- `metrics`：数值指标；
- `artifacts`：轨迹 CSV、调试帧、可视化等相对路径；
- 运行耗时和错误摘要。

指标必须带清楚的名字和单位。例如旧 demo 的图像平面加速度应明确命名为 `vertical_acceleration_px_s2`，不能记成容易被误解为真实重力的 `g`。

## 4. Prompt Contract

prompt renderer 对相同的 experiment config、input metadata、prompt profile 和 seed 必须产生确定性结果。任何会影响文本的字段都要写入 manifest。

所有实验 prompt 必须继承 `configs/prompts/common_physics_video_v1.yaml`，并通过 `schemas/prompt.schema.json`。公共框架固定使用以下顺序：

```text
task → scene → camera → dynamics → parameter → terminal → quality
```

其中 task、scene、camera、quality、公共 negative prompt 和公共约束由基础框架提供；每个实验只提供 dynamics、parameter、terminal、实验约束和额外 negative prompt。不得为某个模型或实验绕过该结构直接拼接一整段自由文本。

推荐至少支持：

- `explicit`：在文字中提供目标物理参数；
- `hidden`：不提供数值，只要求物理一致续写；
- `visual_trace`：要求仅依赖输入中的运动线索；
- 后续可增加无物理描述的 control prompt。

模型 adapter 不得自行改变实验语义；如果模型需要重写 prompt，必须记录原始 prompt、变换后 prompt 和变换版本。

## 5. Build Validation Contract

prepare 阶段应在生成任务前完成以下检查：

- 配置引用均存在且版本明确；
- `task_type` 与 model capabilities 匹配；
- 所有输入资产存在；
- job id 无重复；
- 参数单位和必需分组字段齐全；
- 目标帧数/分辨率可被 adapter 接受或有明确转换规则；
- evaluator 所需 ground truth/输入元数据齐全；
- resolved build 和 manifest 写入后可被重新读取并得到相同含义。

## 6. 跨模型公平性约束

- 同一比较组使用相同任务集合、prompt 语义、seed 列表和评估器版本；
- 若模型约束导致帧数/FPS/分辨率不同，必须在 metadata 和报告中显式记录；
- 不能因某模型失败而悄悄从其他模型中删除对应样本；
- 汇总时同时报告生成失败、质量门控失败和有效样本指标；
- 所有模型输出先规范化再进入评估，规范化步骤本身也需留痕。
