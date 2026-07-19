# Seedance 2.0 / Kling VIDEO 3.0 北京 API：配对 20 条费用与耗时试跑

本流程在服务器上运行，本地只提交代码。Seedance 与 Kling 使用完全相同的 20 个 canonical I2V
任务；模型配置和 runner 都把每个 provider 的上限锁为 20。默认真实运行只做每家 1 条 canary，人工
检查视频和账户扣费后，显式确认才会把同一批次补齐到每家 20 条。

这是纯 REST 流程，不加载本地模型，也不使用服务器 GPU。入口会把子进程的
`CUDA_VISIBLE_DEVICES` 清空，因此不会占用正在给 Wan2.2 使用的 4/5/6/7 号卡。

## 固定样本

- 场景：`baseline`、`indoor3`、`indoor4`、`outdoor1`、`outdoor3`；
- 场景抽样：从 4 个 indoor 和 4 个 outdoor 中以 `20260719` 为随机种子冻结抽取；
- 物体：仅 `standard_ball`；
- 实验：`v1_A`；视角：`CAM_Side`；
- gravity：`2.0 / 4.9 / 9.81 / 14.7 m/s²`；
- canonical seed：`341867882`；
- 组合：`5 scenes × 1 object × 4 gravity values = 20`。

两份 manifest 会逐条校验首帧、prompt、negative prompt、目标参数、canonical seed 和生成语义，
并导出 `selected_20_jobs.csv/jsonl`。API 不支持的控制不会伪装成已应用：Seedance 2.0 和 Kling
VIDEO 3.0 均将 canonical seed 保存到 metadata，同时标记 `seed_applied=false`。

## Provider 设置

### Seedance 2.0

- model：`doubao-seedance-2-0-260128`；
- create：`POST /api/v3/contents/generations/tasks`；
- query：`GET /api/v3/contents/generations/tasks/{task_id}`；
- 严格首帧 I2V：图片 role 使用 `first_frame`，不是广告示例里的 `reference_image`；
- 480p、16:9、5 秒、无音频、无水印；
- Seedance 2.0 当前不支持 API `seed` 和 `camera_fixed`，所以两字段不发送；固定相机要求仍在统一
  prompt 和 negative prompt 中；
- 查询结果保存 `usage.completion_tokens` 与 `usage.total_tokens`。接口不返回实际人民币扣款，
  需与控制台账单或资源包余额对账。

官方文档：

- https://api.volcengine.com/api-docs/view?action=CreateContentsGenerationsTasks&serviceCode=ark&version=2024-01-01
- https://api.volcengine.com/api-docs/view?action=GetContentsGenerationsTask&serviceCode=ark&version=2024-01-01

### Kling VIDEO 3.0

- model：`kling-v3`；
- create：`POST /v1/videos/image2video`；
- query：`GET /v1/videos/image2video/{task_id}`；
- `mode=std`、`sound=off`、5 秒；首帧原始 Base64；
- I2V 画幅继承首帧，std 模式按 provider 原生 720p 记录；
- Kling 不接收 canonical seed，metadata 标记 `seed_applied=false`；
- 保存完成响应中的 `final_unit_deduction`，并与控制台 Credits 余额前后值对账。

请求字段和 endpoint 依据 Kling 团队官方 skill：
https://github.com/KlingAIResearch/kling-skills/tree/main/klingai-video

## 密钥安全

密钥只在服务器交互式 shell 中读入环境变量。不要把它们放进 YAML、命令历史、日志或 Git：

```bash
read -rsp 'ARK_API_KEY: ' ARK_API_KEY
export ARK_API_KEY
echo
read -rsp 'KLING_API_KEY: ' KLING_API_KEY
export KLING_API_KEY
echo
```

`KLING_API_KEY` 是控制台只显示一次的新 API Key，脚本会直接组装为 `Authorization: Bearer ...`。
密钥不会进入命令记录，但同一个 Unix 账户下的其他进程理论上能够读取进程环境；共享 `root`
账户不具备密钥的强隔离保证。测试结束后在启动任务的 shell 中执行：

```bash
unset ARK_API_KEY KLING_API_KEY
```

## 服务器无计费校验

```bash
REMAKE_ROOT=/root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake
cd "$REMAKE_ROOT"

python3 -m venv .venv-api
.venv-api/bin/pip install -r code/requirements-api.txt

PYTHON="$PWD/.venv-api/bin/python" \
DRY_RUN_ONLY=1 DETACHED=0 RUN_TAG=closed_api_cost20_preflight \
bash code/scripts/run_closed_api_cost20_background.sh
```

这一步只完成 schema、manifest、配对和命令检查，不访问 provider，也不产生费用。

## 第一步：每家 1 条真实 canary

先在两个控制台记录测试前余额/资源包剩余额度，然后执行：

```bash
cd /root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake

PYTHON="$PWD/.venv-api/bin/python" \
RUN_TAG=closed_api_cost20_20260719 \
bash code/scripts/run_closed_api_cost20_background.sh
```

默认会让 Seedance 和 Kling 并行运行，但 canary 每家固定只有 1 个在途任务；即使设置了完整阶段的
并发环境变量，canary 也不会扩成 3/5 条。完成后检查：

```bash
RUN_ROOT=outputs/closed_api_cost20/closed_api_cost20_20260719
tail -f "$RUN_ROOT/master.log"
# 看到 canary 完成后按 Ctrl-C 退出 tail，再执行下面的检查；也可以另开终端。
cat "$RUN_ROOT/seedance_canary.log"
cat "$RUN_ROOT/kling_canary.log"
cat "$RUN_ROOT/seedance20/api_cost_time_summary.json"
cat "$RUN_ROOT/kling3_std/api_cost_time_summary.json"
cat "$RUN_ROOT/seedance20/api_cost_time.csv"
cat "$RUN_ROOT/kling3_std/api_cost_time.csv"
find "$RUN_ROOT" -path '*/videos/*.mp4' -o -name '*.api.json'
```

同时在两个控制台记录测试后余额。确认模型、视频、时长、usage/Credits 和扣费都合理后再继续。
两家并行时，如果一家的鉴权失败，另一家的 canary 仍可能已经产生费用。希望完全逐家确认时，可以
使用同一个 `RUN_TAG` 依次执行 `PROVIDERS=seedance` 和 `PROVIDERS=kling`；脚本只要求所选 provider
的密钥：

```bash
PYTHON="$PWD/.venv-api/bin/python" RUN_TAG=closed_api_cost20_20260719 \
PROVIDERS=seedance bash code/scripts/run_closed_api_cost20_background.sh

PYTHON="$PWD/.venv-api/bin/python" RUN_TAG=closed_api_cost20_20260719 \
PROVIDERS=kling bash code/scripts/run_closed_api_cost20_background.sh
```

## 第二步：使用同一 RUN_TAG 补齐到每家 20 条

```bash
cd /root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake

PYTHON="$PWD/.venv-api/bin/python" \
RUN_TAG=closed_api_cost20_20260719 \
CONFIRM_BILLABLE_20=YES \
bash code/scripts/run_closed_api_cost20_background.sh
```

runner 会跳过已经成功的 canary，所以总数是每家 20 条，不是 21 条。两家 provider 彼此并行，
默认仍保持每家内部串行。确认当前账户的试用资源包分别允许 Seedance 3 条、Kling 5 条在途任务后，
可以只在完整阶段显式提速：

```bash
# 两家同时跑：总并发上限为 3 + 5
PYTHON="$PWD/.venv-api/bin/python" RUN_TAG=closed_api_cost20_20260719 \
PROVIDERS=both SEEDANCE_CONCURRENCY=3 KLING_CONCURRENCY=5 \
CONFIRM_BILLABLE_20=YES bash code/scripts/run_closed_api_cost20_background.sh

# 只补 Kling：最多 5 条在途任务
PYTHON="$PWD/.venv-api/bin/python" RUN_TAG=closed_api_cost20_20260719 \
PROVIDERS=kling KLING_CONCURRENCY=5 \
CONFIRM_BILLABLE_20=YES bash code/scripts/run_closed_api_cost20_background.sh
```

runner 在单个进程内做有界调度，不会一次性排队全部 20 条：初始只启动并发上限数量，每完成一条才
补入下一条；若 `--fail-fast` 检测到失败，会停止补入新任务并等待已经在途的任务收尾。不要通过多开
launcher 或手工启动多个 `generate` 进程来增加并发。中断后继续使用相同 `RUN_TAG`；已保存 task id
的任务会续查，已有视频会跳过。3/5 是 API 账户的全局在途额度；如果同一 API 账户还有其他任务，
应把并发变量设为扣除其他在途任务后的剩余空槽。

## 防重复计费与异常恢复

每个任务在 POST 前都会以原子方式创建永久的 `metadata/*.api.json.submit.json` 占位记录。正常收到
task id 后，后续重启只会查询原任务；即使视频下载失败，也会先保存 provider 终态和 token/Credits。
如果服务端可能已接收 POST、但客户端没有拿到 task id，脚本会停止并提示
`submission marker exists`，不会自动再次 POST。此时先到控制台按时间、模型和任务列表核对：

- 如果找到对应任务，保留全部文件，并根据控制台 task id/产物人工处理；
- 只有明确确认供应商没有创建任务时，才可以删除该任务对应的 `.api.json` 和
  `.api.json.submit.json` 后重试；
- 无法确认时不要删除占位记录，改用新的 `RUN_TAG`，并把这次不确定提交计入额度审计。

Kling 的 `external_task_id` 对同一 run/job 稳定、不同 `RUN_TAG` 不同，但脚本不假定该字段一定提供
服务端幂等保证；本地永久占位记录才是自动重提的硬阻断。

## 输出

每个 provider run 目录包含：

- `videos/*.mp4`；
- `metadata/*.api.json`：任务 ID、状态时间线、请求控制是否实际应用、usage/Credits；
- `metadata/*.api.json.submit.json`：提交占位与不确定 POST 的防重计费依据；
- `api_cost_time.csv`：逐条 queue / processing / download / wall time 与额度；
- `api_cost_time_summary.json`：成功率、P50/P90、吞吐、总 token 或总 Credits。

batch 根目录还包含配对任务清单、provider 日志、`master.pid` 和 `master.log`。实际现金花费最终以
控制台测试前后账单差值为准；接口返回额度是可审计依据，但账户套餐和折扣不会被客户端猜测。
`queue_seconds` 和 `processing_seconds` 是客户端按轮询状态观察到的估计值，并非供应商内部的精确
执行计时；误差上限约为对应 provider 的轮询间隔，若第一次观察就已成功则可能为空。
