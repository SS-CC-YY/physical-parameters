# PhysParamBench 论文收口评测口径

本文档定义与论文 abstract 对齐的最小评测口径。核心问题只有一个：给定同一首帧和不同物理参数，视频模型生成的运动是否产生方向正确、数值可测的变化。

## 1. 主结论与辅助实验

### 主定量：baseline + CAM_Side

主表只使用 baseline 场景的侧视视频。侧视近似正交，优先采用冻结相机标定、标准球尺寸约束和二维轨迹恢复世界坐标；若该视频存在足够大的相机漂移，则改用通过运动流形门的公制 3D 轨迹。两条路线之后使用完全相同的 target-free 物理拟合：

1. 参数反演：从轨迹估计重力、摩擦、恢复系数、阻尼等参数；
2. trajectory-dynamics fit：报告观测轨迹、物理模型拟合轨迹和残差；
3. 参数扫描：比较请求参数与轨迹反演参数的方向一致性和数值误差。

参数拟合不得读取请求值；请求值只能在反演完成后用于评分。

### 辅助鲁棒性

- **Indoor / outdoor**：只回答复杂视觉输入下，实验物体能否被稳定测量、运动规律能否保持。它不是主参数精度排名。
- **CAM_Main / CAM_Top**：只回答跨视角时运动规律是否保持。除非深度证据充分，不把透视视角的数值拟合与侧视主结果直接混合。
- **多 seed**：只回答相同条件理解是否稳定，报告等级一致率、参数估计离散度和响应方向稳定性，不把不同 seed 当作新的参数水平。
- **3D / 4D 轨迹**：只用于相机明显漂移、冻结内外参不再可靠的视频。通过 target-independent 公制运动流形门后，样本回到它原本的作用域：baseline Side 可以进入主扫描，复杂场景或 Main/Top 仍只进入相应鲁棒性统计。所有表必须保留 measurement route，不能把 2D/3D 协议隐藏起来。

### 动态 3D 纳入门

门控先于参数真值比较，且不读取 prompt 中给定的目标物理量。冻结世界坐标约定为 X=装置主要水平运动方向、Y=出平面进深、Z=竖直方向。13 个实验按装置分为：

- Z 主轴：`v1_A`、`v2_E`；
- X 主轴：`v1_B`、`v1_C`、`v2_B`、`v2_C`、`v3_B`；
- X-Z 主平面：`v1_D`、`v2_A`、`v2_D`、`v3_A`、`v3_C`、`v3_D`。

以每轴的 5%–95% 稳健范围为 (R_x,R_y,R_z)。例如 X 主轴实验定义

\[
S=R_x,\qquad r_\perp=\sqrt{R_y^2+R_z^2}/S,
\]

而 X-Z 平面实验定义

\[
S=\sqrt{R_x^2+R_z^2},\qquad r_\perp=R_y/S.
\]

当前宽松工作阈值要求：至少 12 个直接公制点、有效覆盖率至少 50%、时间跨度至少 0.5 s、主运动范围至少 0.05 m；主轴/主平面的离流形范围比分别不超过 0.35/0.30，预期轴方差占比分别至少 0.75/0.80，同时排除明显的 3D 位置跳变。坐标必须明确标记为 `blender_world_m` 或由本项目 SpaTrackerV2 后处理得到的 `spatialtrackerv2_frame0_metric_aligned_m`，仅有“公制尺度”但未证明与 Blender 世界轴对齐的相机坐标不会被用于 X/Y/Z 流形判断。SpaTrackerV2 native `quality_pass` 也必须通过。几何通过后先运行 target-free 方程拟合；参数必须可辨识，必须存在运动方程 NRMSE 证据，且其中位数不得超过宽松阈值 0.35。阈值的目标只是排除明显错误深度和明显不服从当前运动规律的轨迹，并不证明单目 3D 的绝对真值精度。

门控输出只有 `include` 与测量 `X`。`include` 标记为 `qualified_dynamic_3d`，随后照常拟合运动方程；`X` 不计为模型失败。它不会覆盖已经确认的生成有效性失败；未人工确认的自动失败仍保留为 X。一个沿 X 轴运动的实验，只要 Y/Z 漂移较小、X 方向占主导且后续 target-free 方程拟合可完成，就不会仅因使用了 3D 重建而被排除出主统计。

## 2. 简化等级

等级必须区分单视频门控与参数扫描结论：单条视频只能得到 `L1`、`PASS_TO_SCAN` 或 `X`；`L2–L4` 必须由同实验、同场景、同视角、同 seed、仅目标参数变化的匹配扫描得到。

| 等级 | 论文含义 | 判定范围 |
|---|---|---|
| L1 | 生成本身不成立：明确的身份/刚性/接触几何失败，或运动类型和事件顺序明确错误 | 单视频 |
| L2 | 运动已可靠观测，但目标规则族不成立，或反演参数没有随请求参数产生正确响应 | 参数扫描 |
| L3 | 参数响应方向正确，但数值误差仍大 | 参数扫描 |
| L4 | 响应方向正确，且参数误差不超过参数有效范围的 25% | 参数扫描 |
| X | 测量不可用或证据不足，不能判断模型成败 | 单视频或参数扫描 |

L1 只接受人工确认或等价的冻结人工裁决。未人工确认的自动 G0/G1 不直接计作模型失败，而是先记为 X。软 `REVIEW` 在没有明确 `failure_codes` 且轨迹质量合格时暂按有效进入拟合，同时保留复核来源；明确硬失败仍阻断。严格版 G0–G4 继续保存在附录。

多参数组合实验按参数分别准入。例如同一条 V2_E 轨迹可以提供有效重力估计，但因碰撞粘滞而拒绝恢复系数；前者仍进入重力扫描，后者以规则族失败进入恢复系数的 L2 证据，不能用全局 `fit_complete=False` 把两者一起丢弃。可靠轨迹与规定方程明显不符时保留 raw estimate 供诊断，但 accepted estimate 置空，禁止强行拟合。

简化扫描层只重新计算五个透明阈值：至少两个有数值估计或明确规则失败证据的参数档位；成对方向一致率大于 0.5；Theil–Sen 斜率大于 0；L4 额外要求斜率至少为 0.2 且 median valid-range NAE 不超过 0.25。扫描层不再次设置统一 NRMSE/R² 阈值，而是读取各实验 target-free fitter 已记录的分段、物理域、方程残差和规则族门控。这样既不会把轻微检测抖动直接写成失败，也不会把明显不符合方程的轨迹硬拟合成参数。

## 3. X 不是模型失败

下列情况必须记为 `X`，不得计入 L1：

- 物体检测或身份关联不可靠；
- 抖动、离散误检或轨迹覆盖不足使参数不可辨识；
- 遮挡覆盖关键碰撞/转向窗口；
- 相机运动使冻结标定失效，但 3D 重建的公制尺度、覆盖率、连续性或运动流形证据不足；
- 轨迹点数量、时间跨度或设计矩阵可辨识性不足。

这里必须区分“测量证据不足”和“模型规则失配”：若轨迹本身可靠，但观测曲线与规定方程显著不符、发生碰撞粘滞或异常加减速，则是可观测的参数/规则理解失败，进入 L2 证据；只有无法可靠获得轨迹或无法辨识参数时才记 X。

只有视频中存在可审计的强证据，例如物体消失/明显形变、持续穿模、完整观测下没有预期碰撞、事件顺序相反，才可判 L1。论文必须同时报告 `X rate` 和各类 X 原因；不能通过把 X 并入失败率来夸大模型缺陷，也不能删除 X 来夸大成功率。

## 4. 主报告与附录

主报告保留：

- baseline + CAM_Side 的有效测量覆盖率与 X rate；
- 每个实验的请求参数—反演参数响应图；
- L2/L3/L4 扫描等级、方向一致率、归一化参数误差；
- 代表性 observed-vs-fitted 轨迹图和检测 overlay；
- 所有数字明确分母，并区分“全计划样本”与“条件于可测样本”。

附录提供：

- indoor/outdoor、Main/Top、多 seed 的独立鲁棒性表；
- 全部 L1 和 X 的 reason codes、证据帧及可视化索引；
- 每个实验的完整参数扫描和拟合残差；
- 3D/4D measurement route、运动流形门指标、运行成本和失败案例。

## 5. 结论边界

- 不把轨迹外观相似写成“模型理解了精确物理参数”；只有 L3/L4 扫描证据支持参数响应结论。
- 不把检测、标定或重建失败写成生成模型失败。
- 不把单条视频写成 L2/L3/L4，也不把多个视角或场景误当独立参数水平。
- 不把未通过运动流形门的 3D 结果混入统计；通过者可以进入其原有作用域，但必须显式标记 measurement route，并单独报告 2D/3D 数量。
- 所用宽松阈值必须随报告和 Git commit 一起冻结；若日后进行严格 Blender 校准，应作为新的严格分析版本，不能静默覆盖本结果。
- 报告“所测试模型、所测试参数范围和所测试视频长度内”的结果，不外推到所有世界模型或所有物理现象。

## 6. 服务器 CLI 的预期输入输出

主报告入口：

```bash
EVAL_ROOT=/root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake/analysis/seedance978_physics_v2
SIMPLE_ROOT="$EVAL_ROOT/simple_paper_report_v2"

python code/scripts/build_physparambench_simple_report.py \
  --all-jobs "$EVAL_ROOT/all_jobs.csv" \
  --evaluation-root "$EVAL_ROOT" \
  --output "$SIMPLE_ROOT"
```

只要 `all_jobs.csv` 中存在 `spatialtrackerv2_dynamic` 行，`--evaluation-root` 就是必填项；报告会优先从 `jobs/<job_id>/` 读取同目录冻结轨迹，校验它与 `result.json` 所记录的 SHA-256 一致，再重新计算当前版本的几何门和 target-free 方程门。这样既不会静默复用未知旧阈值，也不会把后来更新的轨迹和旧拟合结果拼在一起。新版轨迹评估会精确复制 `trajectory_frames.csv` 到对应 evaluation job 目录，并在缓存命中但副本丢失时自动修复，便于整个评估目录移动后重建报告。

若 CSV 已包含 `manual_generation_validity_status`，脚本自动采用人工状态；否则自动 fail 不会直接成为 L1，而先进入 X 等待复核。

输出：

- `REPORT_ZH.md` 与 `ABSTRACT_RESULTS_EN.md`：可直接汇报/写作的收口结论；
- `fig1_video_gate.svg`：L1、可进入扫描与 X；
- `fig2_baseline_response_grades.svg`：L2/L3/L4/X 主统计；
- `fig3_requested_vs_recovered.svg`：24 个 baseline Side 参数扫描；
- `fig4_evaluation_scopes.svg`：baseline、背景、视角、seed 四个作用域；
- `fig5_dynamic_3d_gate.svg`：动态 3D 通过与测量 X 的计数；
- `dynamic_3d_gate.csv`：逐视频的主运动范围、离流形比例、覆盖率和原因码；
- `measurement_cohorts.csv`：2D/3D measurement route 的显式分组；
- `parameter_scans.*`、`video_gate.csv`、`evidence_index.csv`、`representative_scans.csv`：可审计数据和展示案例索引。

严格附录入口仍为：

```bash
python code/scripts/grade_seedance978_results.py \
  --evaluation-root "$EVAL_ROOT" \
  --manifest code/assets/seedance978_evaluation/manifest.jsonl \
  --registry code/assets/seedance978_evaluation/experiment_registry.json \
  --policy code/configs/evaluations/hierarchical_physics_grading_v1.json \
  --motion-profile code/configs/evaluations/motion_type_profiles_v1.json \
  --output "$GRADE_ROOT" \
  --stage all
```

输入合同：

- `jobs/<job_id>/result.json`：生成有效性、拟合状态、参数估计和证据路径；
- 每条结果引用的逐帧轨迹 CSV：时间、可信测量标记和米制坐标；
- manifest：计划任务及实验/场景/视角/seed/参数 tuple；
- registry：冻结参数范围和目标 tuple；
- policy/profile：冻结等级阈值和运动拓扑。

主要输出：

- `video_grades.csv/jsonl`：单视频 G0/G1/U；论文层映射为 L1/PASS_TO_SCAN/X；
- `parameter_scan_grades.csv/jsonl`：扫描级 G2/G3/G4/U；论文层映射为 L2/L3/L4/X；
- `cross_scene_parameter_grades.*`：复杂场景汇总，仅作鲁棒性；
- `jobs/<job_id>/`：grade card、轨迹/拟合图和逐帧证据；
- `groups*/<scan_id>/`：参数响应图和同组轨迹对比；
- `summary.json` 与 `evidence_index.csv`：分母、等级计数和证据索引。

论文表格应在这些可审计输出上做 L1–L4/X 报告层映射，不修改底层证据和原始判定。
