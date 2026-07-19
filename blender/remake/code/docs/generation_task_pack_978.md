# Factorized978 跨模型视频生成任务包

本文面向负责 Helios、Cosmos-Predict2.5-14B、LongLive2.0 以及后续 I2V 模型的执行人员。目标是让所有模型运行**完全相同的 978 个任务**，并把原始 MP4 和可审计 metadata 交回主仓库。

机器可读合同位于 `code/handoffs/generation_factorized978_v1.yaml`。

## 1. 不能改变的 benchmark 快照

| 项目 | 冻结值 |
|---|---:|
| job 总数 | 978 |
| 独立 case 数 | 900 |
| 独立结构化 `prompt_spec` 数 | 900 |
| 独立渲染后正向 prompt 字符串数 | 100 |
| 实验数 | 13 |
| 物理辨识参数 tuple 数 | 48 |
| seed 审计条件数 | 26 |
| 实际引用的首帧 PNG 数 | 351 |
| 场景数 | 9 |
| 物体 | `standard_ball` |
| 相机 | `CAM_Side`、`CAM_Main`、`CAM_Top` |

978 条由三条轨道构成：

1. `physics-identification-side`：48 个 experiment/parameter tuple × 9 个场景 × `CAM_Side` = 432 条；
2. `viewpoint-robustness-main-top`：26 个代表/压力条件 × 9 个场景 × 2 个视角 = 468 条；
3. `random-seed-stability-audit`：26 个基线侧视条件 × 3 个额外 seed = 78 条。

主 seed `341867882` 覆盖前两条轨道的 900 个 case。额外 seed `1750912582`、`265635392`、`135883006` 各覆盖 26 个审计条件。主 seed 对应的 26 条已经包含在 432 条侧视物理轨中，因此总数是 `900 + 78 = 978`，不是 1004，也不是把全部 900 条重复四次。

## 2. 谁负责冻结任务

主负责人应当只生成一次 canonical manifest，并把它连同首帧一起交给所有模型负责人。模型负责人不得自行随机场景、重新抽 seed、重建参数笛卡尔积或手工拼 prompt。900 个 case 各有确定的结构化 `prompt_spec`；由于公共 scene 文本不直接写入场景名字，最终渲染后的正向文本去重后是 100 条。不得把“100 个唯一字符串”误解成只有 100 个任务，scene、camera、input、target 和 seed 仍以每一行 manifest 为准。

canonical 来源文件为：

```text
code/builds/standard_ball_factorized978_wan22_generation.yaml
code/configs/experiments/all_experiments_full.yaml
code/configs/prompts/common_physics_video_v1.yaml
code/configs/prompts/all_experiments_explicit_v1.yaml
code/schemas/build.schema.json
code/schemas/job.schema.json
code/schemas/prompt.schema.json
first_frames_v1_0/images/
```

虽然源 build 名称中带有 `wan22`，其中 experiment override 定义的是本轮冻结的 978 矩阵。其他模型只能替换 model profile、adapter、checkpoint 和模型原生运行参数，不能改变 experiment、selection、prompt 或 manifest 的任务语义。

主负责人准备冻结任务包：

```bash
cd /path/to/blender/remake

export REMAKE_ROOT="$PWD"
export NUM_FRAMES=81

python code/scripts/remake_benchmark.py prepare \
  --build code/builds/standard_ball_factorized978_wan22_generation.yaml \
  --run-dir handoff/factorized978_canonical \
  --workspace-root "$PWD"
```

随后必须确认：

```text
manifest.jsonl            978 行
manifest.selection.json   job_count = 978
首帧引用去重后            351 张 PNG
experiment_id 去重后      13 个
```

冻结时记录：

- 仓库 Git commit；
- `manifest.jsonl` 的 SHA256；
- 351 张被引用 PNG 的 SHA256；
- prompt 配置和 schema 的 SHA256。

本合同对应的正式冻结文件为：

```text
manifest.jsonl
  SHA256 0239b849a8d2b2922b460d6586e3898199907fed9602dcaf84882049091f565a
manifest.selection.json
  SHA256 fc9735fa42c86b45c26e7ed70525d925b81850f852a7b77508b80ebe71a6d6ed
provenance/source_reference_resolved_build.yaml
  SHA256 32e49d12a25299663555669882580f35b997b2d9953b2d9f59cdc79d0b71eb61
```

取得任务包后先执行 `python tools/validate_bundle.py .`；也可以再执行
`sha256sum -c sha256sums.txt`。任何一项不匹配都应停止生成并联系主负责人，
不能在本地“修复”或重新 prepare 后继续。

所有模型返还的 `manifest.jsonl` 都必须与任务包中的文件**字节级一致**，包括原有
`build_id`；不要在 manifest 内写入模型名。模型条件、checkpoint 和 adapter 版本应
单独记录在 `model_run.json`、每条 metadata 或模型自己的 resolved build 中。以下字段
也必须逐 job 与冻结 manifest 相同：

```text
job_id, experiment_id, case_id, repeat_id, task_type, seed,
inputs, prompt_spec, prompt, negative_prompt, prompt_mode,
prompt_template_id, prompt_template_version, targets,
known_params, factors, units, generation, evaluation_tags
```

`resolved_build.yaml` 只作为任务生成来源的 provenance 快照，其中可能含原始机器路径和
Wan reference profile；其他模型负责人不能把它当成可迁移的运行配置。实际运行应直接消费
冻结 manifest，并把模型专属配置另存到自己的 run 目录。

## 3. 模型负责人需要取得的文件

最小任务包：

```text
factorized978_task_pack/
├── HANDOFF.md
├── TASK_CONTRACT.yaml
├── PACKAGE_MANIFEST.json
├── manifest.jsonl
├── manifest.selection.json
├── sha256sums.txt
├── git_commit.txt
├── provenance/
│   └── source_reference_resolved_build.yaml
├── first_frames_v1_0/
│   └── images/...                         # manifest 实际引用的 351 张 PNG
├── model_runner_template/
│   ├── run_model.py
│   └── model_config.example.json
├── tools/
│   ├── validate_bundle.py
│   └── validate_generation_outputs.py
├── code/
│   ├── configs/prompts/
│   ├── schemas/
│   ├── scripts/validate_generation_outputs.py
│   └── handoffs/generation_factorized978_v1.yaml
```

如果通过 Git 获取数据：

```bash
git lfs install
git lfs pull --include="blender/remake/first_frames_v1_0/**"
```

必须检查 PNG 已经被 LFS 展开，而不是只有几十或几百字节的 LFS pointer。

## 4. 每条 job 如何调用模型

逐行读取 `manifest.jsonl`，只使用该行给出的任务内容：

- I2V 图片：`inputs.image`；
- 正向文本：`prompt`；
- 负向文本：`negative_prompt`；
- canonical seed：`seed`；
- 期望媒体参数：`generation`；
- 唯一输出名：`job_id`。

所有 prompt 已经由公共框架确定性渲染，固定顺序为：

```text
task → scene → camera → dynamics → parameter → terminal → quality
```

禁止：

- 根据模型“优化”或概括物理描述；
- 删除显式物理参数；
- 给某个模型添加额外运动提示而不给其他模型添加；
- 根据生成结果重新写 prompt；
- 从文件名反推 prompt 并忽略 manifest；
- 因某模型失败而从其他模型任务中删除同一 job。

如果 provider 有独立 negative-prompt 字段，分别发送 `prompt` 和 `negative_prompt`。如果没有，可以使用已经声明版本的转换，例如：

```text
<canonical prompt>\nAvoid: <canonical negative prompt>
```

metadata 必须同时保存 canonical 文本、实际 provider 文本和转换版本。adapter 不得静默改变语义。

模型不支持 seed 时，不要编造“已使用 seed”。仍保留 canonical seed 作为 job 身份，并记录：

```json
{
  "requested_seed": 341867882,
  "seed_supported": false,
  "seed_sent": false,
  "effective_seed": null
}
```

这类模型仍可生成，但后续 seed 稳定性结论必须标记为 provider-uncontrolled stochastic repeat。

## 5. MP4 合同

唯一合法的 canonical 视频路径是：

```text
videos/<job_id>.mp4
```

例如：

```text
videos/v1_A__g9p81__baseline__standard_ball__CAM_Side__seed-341867882.mp4
```

实际 `job_id` 必须直接取自 manifest；不要根据示例手写，不得增加模型名、日期、`final`、`retry` 等后缀。

本轮 nominal 请求为：

```text
I2V
standard_ball
832 × 480
81 frames
16 fps
首帧到末帧观测跨度 5.0 s
```

不同模型可以使用最接近的原生 5 秒、约 480p 输出。公平性规则是保存 provider 的原始视频，而不是事后强行制造相同 FPS：

- 不重新编码视频；
- 不插帧、不抽帧；
- 不改变播放速度；
- 不裁剪轨迹或首尾事件；
- 不把 24 fps 文件写成 16 fps；
- provider 不是 MP4 时，可以无损 remux 到 MP4，但必须保留原文件并记录 remux；
- 不添加水印、字幕、边框或调试标注；
- 音频不是 benchmark 输入，优先关闭；provider 强制生成音频时如实记录。

实际评估必须从媒体容器或逐帧 PTS 读取真实时间。metadata 中同时记录 requested 和 actual：

```text
codec, width, height, fps, frame_count, duration_seconds, time_base
```

不能用 manifest 中的 `16 fps / 81 frames` 冒充模型实际输出。16 fps 和 24 fps 都可以形成有效时间轴；错误解释 FPS 才会直接破坏速度、加速度和物理参数拟合。

## 6. 每条 job 的 metadata

每个成功或失败的 job 都必须有：

```text
metadata/<job_id>.json
```

最低字段合同：

```json
{
  "schema_version": "1.0.0",
  "job_id": "exact manifest job_id",
  "status": "succeeded | failed | incomplete",
  "manifest_sha256": "...",
  "input": {
    "image": "first_frames_v1_0/images/.../CAM_Side.png",
    "image_sha256": "..."
  },
  "prompt": {
    "canonical": "...",
    "canonical_negative": "...",
    "provider_text": "...",
    "transform_id": "identity-or-versioned-transform"
  },
  "seed": {
    "requested": 341867882,
    "supported": true,
    "sent": true,
    "effective": 341867882
  },
  "model": {
    "owner": "...",
    "model_id": "...",
    "checkpoint_or_revision": "...",
    "adapter": "...",
    "code_commit": "..."
  },
  "requested_generation": {},
  "actual_generation": {},
  "media_probe": {
    "codec": "...",
    "width": 0,
    "height": 0,
    "fps": 0.0,
    "frame_count": 0,
    "duration_seconds": 0.0,
    "time_base": "..."
  },
  "output_video": "videos/<job_id>.mp4",
  "output_sha256": "...",
  "timing": {
    "started_at": "...",
    "completed_at": "...",
    "queue_seconds": null,
    "processing_seconds": null,
    "wall_seconds": 0.0
  },
  "provider": {
    "task_id": null,
    "usage": {},
    "cost": null
  },
  "postprocess": [],
  "error": null
}
```

可以增加 provider 原始字段，但不能删除上述可审计信息。API 运行还应保留 task id、状态轮询、usage/credits/cost 和失败响应。失败 job 也要返回 metadata，不能只返回成功样本。

## 7. 不得泄露凭据

API Key、访问令牌和云端凭据只能通过环境变量或服务器 secret manager 提供：

```bash
export PROVIDER_API_KEY='...'
```

禁止把 key 写入：

- build YAML；
- manifest；
- command JSON；
- stdout/stderr；
- metadata；
- Git commit；
- 交付压缩包。

metadata 只能记录环境变量名，例如 `PROVIDER_API_KEY`。如果 provider 原始响应包含带签名的临时下载 URL，对外返还前应移除 query token，但保留 task id 和本地 MP4。

## 8. 运行目录与返还内容

推荐每个模型使用独立 run：

```text
outputs/<model_condition>/<run_id>/
├── resolved_build.yaml
├── manifest.jsonl
├── manifest.selection.json
├── videos/
│   └── <job_id>.mp4                    # 978 个
├── metadata/
│   └── <job_id>.json                   # 978 个，包括失败记录
├── logs/
│   └── ...
├── run_state.jsonl
├── run_summary.json
├── generation_validation.json
└── CHECKSUMS.sha256
```

如果使用本框架的新 adapter：

1. 从 factorized978 build 复制一个新 build；
2. 只改 `build_id`、`profiles.model` 及模型专属 runtime/generation；
3. prepare 新 run；
4. 与冻结 manifest 逐行比较上述 identity fields；
5. 先 dry-run 和 1 条 canary；
6. canary 人工确认首帧、时长、视角和对象身份；
7. 再启动完整批次；
8. 中断后复用同一 run，跳过已有且 metadata 完整的 job。

如果不接入本框架，则编写一个 manifest-driven runner；输入和返还合同保持不变。

## 9. 提交前验收

在 run 根目录执行：

```bash
python code/scripts/validate_generation_outputs.py \
  --manifest manifest.jsonl \
  --videos videos \
  --metadata metadata \
  --expected-jobs 978 \
  --ffprobe required \
  --report generation_validation.json
```

必须满足：

- manifest 恰好 978 个唯一 job；
- `videos/` 中恰好存在同名的 978 个非空 MP4；
- 无缺失、额外或重复 job id；
- ffprobe 能读取每个视频流；
- 978 份 metadata 均存在，失败记录不被隐藏；
- seed 计数为 `900 + 26 + 26 + 26`；
- 所有输入图片和 canonical prompt 与 manifest 一致；
- 每条视频 actual FPS、帧数、尺寸和时长已记录；
- 日志和 metadata 中不存在 key/token；
- `eval/` 不属于生成任务，不应因评估失败中断生成。

当前 `validate_generation_outputs.py` 负责 manifest/MP4 集合、ffprobe 可读性，以及
metadata 的数量、JSON、`job_id` 和成功状态检查；metadata 的全部字段完整性、prompt
逐字一致性、secret 扫描和返还文件 SHA256 仍需在最终收件检查中执行。

## 10. 与后续重建的边界

本任务包只负责生成和证据留存。不要在生成负责人处先筛掉“物理看起来错误”的视频，也不要覆盖原 MP4 加检测框。

当前 reconstruction MVP 仅实现 `v1_A + standard_ball + CAM_Side` 快路径，不能把“978 条已经生成”描述成“13 个实验已经完成 3D 重建”。完整评估将在所有模型视频收齐后，基于同一原始 MP4、真实时间轴、相机/场景几何 sidecar 和实验 family 插件统一执行。
