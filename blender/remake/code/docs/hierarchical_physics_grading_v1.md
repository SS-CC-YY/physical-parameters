# G0–G4 分级评测实现（v1）

## 1. 作用域必须分开

这套分级不是把每条视频直接塞进 `G0 < G1 < G2 < G3 < G4` 的单列分类。

| 作用域 | 允许结果 | 回答的问题 |
|---|---|---|
| 单视频 | `G0`、`G1`、`PASS_TO_SCAN`、`U` | 视频是否有效、运动类型是否成立、证据是否足够 |
| 匹配参数扫描组 | `G2`、`G3`、`G4`、`U` | 改变给定参数时，反演参数是否按正确方向和正确数值响应 |

`U` 是“当前证据不足”，不是比 G0 更差的模型失败。单条视频永远不标 G2/G3/G4。

严格执行以下门控：

```text
G0 明确失败 ──> G0，停止
G0 证据不足 ──> U，停止
G0 通过 ──> G1
G1 明确失败 ──> G1，停止
G1 证据不足 ──> U，停止
G1 通过且参数可反演 ──> 进入 matched parameter scan
scan 证据不足 ──> U
方向不正确/无响应 ──> G2
方向正确、数值不准 ──> G3
方向与数值均达到冻结阈值 ──> G4
```

## 2. G0：生成有效性

G0 先复用现有 `video_generation_validity`：对象身份、二维刚性、尺度、背景刚性、scene cut，以及动态路线的三维刚性证据。

对 `spatialtrackerv2_dynamic` 路线，分级器会再次用与静态流水线同一份 `generation_validity.py` 3D object/background rigidity 阈值检查 native evidence；这样不会因为 SpaTracker 后处理脚本使用另一套阈值而让 G0 口径分裂。缺少动态刚性证据时为 U。

新增 `contact_geometry.py`，只读取冻结米制轨迹与装置几何，不读取目标参数：

- 自由落体/弹跳：持续严重穿过地面；
- 墙碰撞：持续严重越过冻结墙面；
- 摆：摆长连续大幅破坏。

只有“大幅且连续”的几何反证才判 G0；接触附近的小误差留给轨迹拟合。连续性同时要求帧号与时间连续，稀疏的离群点不能拼成“持续穿模”。几何与 G1 只读取 `physics_fit_used=true` 的 evaluation 米制坐标，或 `fit_eligible=true` 的 extraction fit 坐标；已被拟合门拒绝的点和纯展示坐标不得制造 G0/G1。点数不足时输出 U。

## 3. G1：运动类型

`motion_type_validity.py` 在参数拟合评分之前进行 target-free 检查。每个实验使用独立的冻结运动 profile：

| 实验 | 最小运动拓扑 |
|---|---|
| V1A | 下降、首段负曲率、到达地面 |
| V1B | 接近右墙、接触、速度反向 |
| V1C | 单向滑动、速度下降、无频繁反向 |
| V1D | 近似定长摆、至少两次转向 |
| V2A | 有界往复轨迹、至少两次转向 |
| V2B | 冻结左右墙接触面（x=-2.75/+2.75 m）、至少两次转向且左右墙交替接触 |
| V2C | 穿越材料边界，两区均被观测并减速 |
| V2D | 近似定长阻尼摆、至少两次转向 |
| V2E | 飞行、地面接触、反弹 |
| V3A | 水平推进与竖直落地反弹 |
| V3B | 左右交替碰撞、段内速度不增加 |
| V3C | 斜面、平地、右墙碰撞的严格事件顺序 |
| V3D | 近似定长摆、转向并穿越磁区 |

导数窗口按秒而非按帧，因此 16 fps 与 24 fps 使用同一时间尺度。摆还要求可辨识的角跨度与显著转向，微小静止抖动不能通过；V3C 同时检查斜坡下降、平地近似等高以及墙碰撞顺序。完整时间覆盖下始终没有预期接触属于 G1；事件可能落在片尾、遮挡或直接观测不足时才是 U。

## 4. G2–G4：参数扫描组

扫描组固定：

```text
experiment + scene + camera + object + seed + 其余物理参数
```

只改变一个目标参数。组建扫描使用 registry 的 GT，但 GT 不进入跟踪、G0/G1 或反演器。参数扫描入口会再次要求 `target_not_used_for_fit=true`、目标参数已观测且新版拟合证据门通过，防止旧缓存的 `PASS_TO_SCAN` 绕过审计。冻结 978 设计自动得到：

- Side 主结果：`9 scenes × 24 parameter tests = 216` 个原子扫描；
- Main/Top：作为视角鲁棒性扫描，不能当同步多目；
- 多参数实验先对每个参数分别评分，实验总级别取必需参数中的最弱级；任一必需参数为 U，则总级别为 U。

### 4.1 方向门（G2/G3 分界）

目标值按升序为 \(p_i\)，反演中位数为 \(\hat p_i\)。成对方向一致率为：

\[
C = \frac{\#\{(i,j): i<j,\ \hat p_j-\hat p_i>\epsilon\}}{\binom{n}{2}},
\qquad
\epsilon=0.01(p_{\max}-p_{\min}).
\]

同时报告 Spearman \(\rho\) 和 Theil–Sen 斜率：

\[
s=\operatorname{median}_{i<j}\frac{\hat p_j-\hat p_i}{p_j-p_i}.
\]

候选 v1 阈值：`C >= 0.75`、`s > 0.05`；三档以上还要求 `rho >= 0.8`。响应跨度小于 valid range 的 1% 为 flat。跨场景统计只使用同时拥有全部档位的 matched scene block，并按场景块 bootstrap 斜率；其 95% CI 必须整体为正。不能把不同档位各自可用场景的独立中位数拼在一起。方向门失败为 G2。

### 4.2 数值门（G3/G4 分界）

每个可用任务的 range-normalized absolute error：

\[
\mathrm{NAE}_i = \frac{|\hat p_i-p_i|}{p_{\max}-p_{\min}}.
\]

候选 G4 同时要求：

- median NAE `<= 0.10`；
- p90 NAE `<= 0.20`；
- `|s - 1| <= 0.25`；
- range-normalized median bias `<= 0.10`；
- 多参数实验中固定参数的反演漂移（off-target drift / valid range）`<= 0.10`；
- 所有计划档位均有可用证据；
- 进入方向/数值门前，计划任务与档位覆盖均至少为 `2/3`（按整数边界判断，恰好 `2/3` 合格）；G4 仍要求上一条所述的全部计划档位；
- 固定参数（off-target）的估计也达到相同覆盖率；
- 每条任务先取必需拟合序列中的最差 NRMSE，再要求这些任务的 p90 NRMSE `<= 0.20`。

方向通过但任一数值门失败为 G3。若其它数值门本可达到 G4，但旧结果没有标准化拟合残差，则为 U，而不是凭空给 G4。

这些阈值在配置中标记为 `candidate_thresholds_pending_blender_synthetic_calibration`。论文前必须使用 Blender GT 与受控跟踪退化样本校准并冻结；禁止在 Seedance/Wan/其它待测模型结果上调阈值。

## 5. 可审计证据

每个单视频目录：

```text
grading_v1/jobs/<job_id>/
  grade.json              # G0/G1/U/PASS_TO_SCAN、逐项规则、阈值、原因
  grade_card.png/.svg
  motion_type_plot.png/.svg
  trajectory_fit.png/.svg # 空间轨迹、观测 vs 真正拟合、残差、参数表
  fit_series.csv           # 标准化观测/预测/残差序列
  fit_events.json          # contact/impact/turn/transition
```

每个扫描组目录：

```text
grading_v1/groups/<scan_id>/
  group_grade.json
  parameter_response.png/.svg  # y=x、G4 容差带、方向/斜率/NAE、覆盖
  trajectory_comparison.png/.svg # 各参数档位共享坐标轴；不拼 autoscale 图
```

此外，`groups_aggregate/` 会在 camera 与 seed 固定的前提下跨 9 个场景汇总相同扫描，以场景作为重复 benchmark condition，为每个参数档位 bootstrap 中位数 95% CI。这个 CI **只表示跨场景中位数的不确定性**：单视频拟合没有参数 CI，单场景/单 seed 的原子扫描也没有可识别的重复样本 CI。不得把跨场景 CI 写成单视频参数置信区间或生成模型的采样置信区间。场景级原子 grade 仍保留，不会被汇总图替代。

总表：

```text
grading_v1/
  grading_config.json
  video_grades.csv/jsonl
  parameter_scan_grades.csv/jsonl
  cross_scene_parameter_grades.csv/jsonl
  experiment_response_grades.jsonl
  cross_scene_experiment_response_grades.jsonl
  evidence_index.csv
  summary.json
  figures/grade_funnel.png/.svg
```

`fit_series.csv` 来自反演器实际拟合预测，不是为了好看另做的平滑曲线。`trajectory_fit` 会逐条画出 fitter 暴露的全部 `fit_series`，每条都有观测、预测和残差，不能只挑最长的一条。若 fitter 没有显式声明 `supports_parameters`，图中必须标为“series-to-parameter support mapping not declared”：良好的某条残差曲线不能独立证明所有列出的参数都可辨识。

GT 只在反演结束后加入参数表与响应图。图和 `fit_events.json` 从真实 `fit.target_not_used_for_fit` 字段读取审计状态；该字段为 false 或缺失时显示红色警告，禁止硬编码成 true。

### 5.1 可选的人工 G0 裁决

自动 G0 gate 与人工裁决必须同时保留，不能用人工结果静默覆盖原始检测。CLI 可选读取一个冻结 JSONL：

```json
{"schema_version":"1.0.0","job_id":"v1_D__beta0p04__indoor1__standard_ball__CAM_Side__seed-341867882","decision_scope":"g0_generation_validity","g0_status":"pass","reason_codes":["manual_review_tracking_gate_false_positive"],"reviewer":"reviewer_id","timestamp":"2026-07-22T12:00:00+08:00","note":"Object remains rigid; automatic track left the ball.","evidence_paths":{"overlay":"validity_object_track_overlay.mp4"}}
```

约束：

- `decision_scope` 只能是 `g0_generation_validity`；
- `g0_status` 只能是 `pass/fail/indeterminate`；
- 一条 job 只能出现一次，未知 job 直接报错；
- override 只能裁决 G0，不能直接指定 G1/G2/G3/G4；
- `grade.json` 必须同时记录 automatic decision、manual record、是否应用和最终 G0；
- CLI 会校验记录并把它显式传给 `grade_video_result(..., adjudication_override=...)`；自动、人工和最终生效状态同时保留。

仓库不会附带 Seedance 的预填裁决结果，避免把尚未复核的判断伪装成真值。完成冻结人工复核后，可把 JSONL 保存为 `$EVAL_ROOT/g0_adjudication_v1.jsonl`。

运行时附加：

```bash
--adjudication-jsonl "$EVAL_ROOT/g0_adjudication_v1.jsonl"
```

## 6. 服务器运行顺序

先更新并进入仓库：

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data
REMAKE_ROOT="$REPO/blender/remake"

SAFE_CFG=$(mktemp /tmp/remake-safe-gitconfig.XXXXXX)
cleanup_safe_cfg() {
  rm -f -- "$SAFE_CFG"
}
trap cleanup_safe_cfg EXIT HUP INT TERM

git config --file "$SAFE_CFG" --add safe.directory "$REPO"
GIT_CONFIG_SYSTEM="$SAFE_CFG" GIT_LFS_SKIP_SMUDGE=1 \
git -C "$REPO" pull --ff-only origin codex/remake-wan22-lfs
GIT_CONFIG_SYSTEM="$SAFE_CFG" \
git -C "$REPO" log -1 --oneline

cleanup_safe_cfg
trap - EXIT HUP INT TERM
cd "$REMAKE_ROOT"
```

这里没有修改共享账户的全局 Git 配置；临时 protected config 在成功、报错、Ctrl+C 和终端退出时都会清理。

轨迹提取已经完成时，只重跑 CPU 物理拟合。物理 fitter 源码 hash 已改变，缓存会自动显示 `REEVALUATE ... lineage=changed`；不会重新解码视频或调用 tracker：

```bash
TRACK_ROOT="$REMAKE_ROOT/outputs/seedance978_tracks_v1"
EVAL_ROOT="$REMAKE_ROOT/analysis/seedance978_physics_v2"

python code/scripts/evaluate_seedance978_trajectories.py \
  --tracks "$TRACK_ROOT" --output "$EVAL_ROOT" --phase side
python code/scripts/evaluate_seedance978_trajectories.py \
  --tracks "$TRACK_ROOT" --output "$EVAL_ROOT" --phase main
python code/scripts/evaluate_seedance978_trajectories.py \
  --tracks "$TRACK_ROOT" --output "$EVAL_ROOT" --phase top
```

先快速生成全量 machine-readable 等级（不画 978 份图）：

```bash
python code/scripts/grade_seedance978_results.py \
  --evaluation-root "$EVAL_ROOT" \
  --output "$EVAL_ROOT/grading_v1" \
  --stage all \
  --no-visuals \
  > "$EVAL_ROOT/grading_v1.log" 2>&1
```

确认 `summary.json` 后生成全视频证据图，以及默认的跨场景参数响应图。冻结 978 清单会显式写出 720 个 canonical 参数槽位（其中包含非 OAT、混杂或证据不足而必须记为 U 的槽位）；它们默认只写 JSON/CSV，避免无必要地生成几百套重复图片。固定主 seed 的 Side 主结果仍是 216 个场景级扫描，跨场景汇总为每视角 24 个、三视角共 72 个：

```bash
python code/scripts/grade_seedance978_results.py \
  --evaluation-root "$EVAL_ROOT" \
  --output "$EVAL_ROOT/grading_v1" \
  --stage all \
  > "$EVAL_ROOT/grading_v1_visuals.log" 2>&1
```

若论文审计需要把所有场景级原子扫描也画出来，额外加 `--all-scan-visuals`。

默认跨场景 `trajectory_comparison` 会把每个参数档位下所有可用场景逐条画出，并在图例中显示 scene/job；它不会用 `usable_job_ids[0]` 冒充跨场景中位轨迹，也不会合成一条并不存在的“median trajectory”。

若只想先核验一条视频，图只为该 job 生成，其余 job 仍保留轻量 grade JSON：

```bash
JOB=v1_B__e0p90__indoor1__standard_ball__CAM_Side__seed-341867882
python code/scripts/grade_seedance978_results.py \
  --evaluation-root "$EVAL_ROOT" \
  --output "$EVAL_ROOT/grading_v1" \
  --stage video \
  --job-id "$JOB"
```

评测只使用 CPU；SpaTrackerV2/GPU 仅在之前轨迹抽取阶段需要。

## 7. 论文报告原则

- 分母始终保留 G0/G1/U，不能只报告成功拟合子集；
- G0/G1 按视频数报告，G2–G4 按扫描组数报告，两者不可相加成一个“总体 grade accuracy”；
- response 图必须写 planned/usable levels；
- 只有跨场景汇总 response 图可以报告 bootstrap 中位数 95% CI；原子扫描和单视频不得虚构 CI；
- 两档扫描的 Spearman 必然接近 ±1，只能算 descriptive，主要看成对方向、斜率和数值误差；
- Main/Top 是独立生成的视角鲁棒性，不宣称同步多目一致性；
- 图像是可审计证据，不是形式化证明。结论还需阈值冻结、人工盲审子集和跨模型复现。
