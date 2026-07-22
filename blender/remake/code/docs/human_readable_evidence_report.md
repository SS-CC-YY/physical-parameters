# PhysParamBench 人可读视频证据报告

`build_physparambench_evidence_report.py` 是冻结评测结果的只读整理器。它不会重新检测、重新拟合或修改 L1/L2/L3/L4/X，只把已经存在的原视频、检测视频、逐帧 2D/3D 轨迹、公式和拟合结果整理为可在浏览器中查看的证据包。

## 统计单位

- 13：实验系统数量；
- 24：`实验 × 待扫描参数` 的 OAT 参数响应轴数量；
- 一次多参数实验会产生 2–3 个响应轴，但并没有被重复计为新实验。

## 服务器运行

先设置实际目录。以下路径沿用当前 Seedance 978 流程；如果报告目录名不同，只需修改变量，不要移动原结果。

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data
REMAKE_ROOT="$REPO/blender/remake"

EVAL_ROOT="$REMAKE_ROOT/analysis/seedance978_physics_v2"
SIMPLE_ROOT="$EVAL_ROOT/simple_paper_v1"
TRACK_ROOT="$REMAKE_ROOT/outputs/seedance978_tracks_v1"
VIDEOS="$REMAKE_ROOT/outputs/seedance978/seedance20_factorized978_480p_20260719_140052/videos"
ALL_JOBS="$EVAL_ROOT/all_jobs.csv"
REPORT_ROOT="$EVAL_ROOT/human_evidence_v1"

cd "$REMAKE_ROOT"
export PYTHONPATH="$REMAKE_ROOT/code/src${PYTHONPATH:+:$PYTHONPATH}"
```

如果 `SIMPLE_ROOT` 或 `ALL_JOBS` 不确定，先查找，不要凭目录名猜：

```bash
find "$REMAKE_ROOT/analysis" -path '*/parameter_scans.jsonl' -print
find "$REMAKE_ROOT/analysis" -name 'all_jobs.csv' -o -name 'adjudication_aware_all_jobs.csv'
```

先做不编码对比视频的全量烟雾测试：

```bash
python code/scripts/build_physparambench_evidence_report.py \
  --all-jobs "$ALL_JOBS" \
  --simple-report "$SIMPLE_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  --tracks-root "$TRACK_ROOT" \
  --videos-root "$VIDEOS" \
  --output "$REPORT_ROOT" \
  --media-mode hardlink \
  --skip-montages
```

确认 `summary.json` 中 `experiment_count=13`、`parameter_response_axis_count=24` 且命令无异常后，使用同一个输出目录生成对比 MP4：

```bash
python code/scripts/build_physparambench_evidence_report.py \
  --all-jobs "$ALL_JOBS" \
  --simple-report "$SIMPLE_ROOT" \
  --evaluation-root "$EVAL_ROOT" \
  --tracks-root "$TRACK_ROOT" \
  --videos-root "$VIDEOS" \
  --output "$REPORT_ROOT" \
  --media-mode hardlink
```

静态轨迹图不需要 OpenCV；视频拼接需要当前环境可导入 `cv2`。若 `cv2` 或 MP4 writer 不可用，报告不会中断，而会保留各个原视频/overlay 和明确的 unavailable 占位图。

这一步不需要 GPU，可以在生成模型占用 GPU 时运行。`hardlink` 在同一文件系统内不会复制视频内容；若不支持硬链接，工具会明确降级为 copy。

## 输出入口

主入口：

```text
$REPORT_ROOT/index.html
```

主要目录：

```text
01_unusable/          L1、X 和自动异常待复核案例
02_parameter_scans/   全部 24 个响应轴；当前 23 个 L2–L4 与 1 个扫描级 X 都有证据页
03_background/        同参数 baseline / indoor / outdoor 对照
04_views/             同设定 Side / Main / Top 对照
05_camera_drift/      Side/非 Side 相机漂移与动态 3D 门控
_assets/jobs/         每个视频只物化一次的原视频、overlay 和轨迹图
```

每个不可用案例包含原视频、轨迹检测视频、2D/3D 轨迹展示及中文原因。每个 L2–L4 扫描包含不同参数档位的原视频拼接、overlay 拼接、请求值与反演值图、各档位轨迹图，以及简洁的公式反推过程。

复杂背景使用

```text
D_scene = |theta_scene - theta_baseline| / (theta_max - theta_min)
```

视角使用

```text
D_view = |theta_view - theta_side| / (theta_max - theta_min)
```

只有两边都有有效反演值时才计算；否则显示 `N/A`，不会填成零或模型失败。

## 在 VS Code Tunnel 中查看

在报告目录启动只监听服务器本机的静态服务：

```bash
cd "$REPORT_ROOT"
python -m http.server 8787 --bind 127.0.0.1
```

然后使用 VS Code 的 Ports 面板转发 `8787`，打开转发后的地址。若要下载，需下载整个 `human_evidence_v1` 目录，不能只拿 `index.html`，因为视频和图片使用相对链接。

## 解释限制

- X 是轨迹、重建或拟合证据不足，不等于模型生成失败；
- REVIEW 是自动有效性门提示异常，必须看原视频后才能人工定为 L1；
- Side 发生明显漂移时，报告首先明确标注严重相机漂移，再展示 3D 门控和拟合；
- Main/Top 漂移报告比较整条轨迹的有效参数与 Side，不声称参数在每个时刻都连续恒定；逐时刻结论需要另做滑窗拟合。
