# PhysParamBench 论文收口评测口径

本文档定义与论文 abstract 对齐的最小评测口径。核心问题只有一个：给定同一首帧和不同物理参数，视频模型生成的运动是否产生方向正确、数值可测的变化。

## 1. 主结论与辅助实验

### 主定量：baseline + CAM_Side

主表只使用 baseline 场景的侧视视频。侧视近似正交，优先采用冻结相机标定、标准球尺寸约束和二维轨迹恢复世界坐标，然后完成：

1. 参数反演：从轨迹估计重力、摩擦、恢复系数、阻尼等参数；
2. trajectory-dynamics fit：报告观测轨迹、物理模型拟合轨迹和残差；
3. 参数扫描：比较请求参数与轨迹反演参数的方向一致性和数值误差。

参数拟合不得读取请求值；请求值只能在反演完成后用于评分。

### 辅助鲁棒性

- **Indoor / outdoor**：只回答复杂视觉输入下，实验物体能否被稳定测量、运动规律能否保持。它不是主参数精度排名。
- **CAM_Main / CAM_Top**：只回答跨视角时运动规律是否保持。除非深度证据充分，不把透视视角的数值拟合与侧视主结果直接混合。
- **多 seed**：只回答相同条件理解是否稳定，报告等级一致率、参数估计离散度和响应方向稳定性，不把不同 seed 当作新的参数水平。
- **3D / 4D 轨迹**：仅为相机明显漂移样本的探索性分析，单列结果和失败率，不进入主定量结论。

## 2. 简化等级

等级必须区分单视频门控与参数扫描结论：单条视频只能得到 `L1`、`PASS_TO_SCAN` 或 `X`；`L2–L4` 必须由同实验、同场景、同视角、同 seed、仅目标参数变化的匹配扫描得到。

| 等级 | 论文含义 | 判定范围 |
|---|---|---|
| L1 | 生成本身不成立：明确的身份/刚性/接触几何失败，或运动类型和事件顺序明确错误 | 单视频 |
| L2 | 运动类型成立，但反演参数没有随请求参数产生正确响应 | 参数扫描 |
| L3 | 参数响应方向正确，但数值误差仍大 | 参数扫描 |
| L4 | 响应方向正确，且参数误差不超过参数有效范围的 25% | 参数扫描 |
| X | 测量不可用或证据不足，不能判断模型成败 | 单视频或参数扫描 |

L1 只接受人工确认或等价的冻结人工裁决。未人工确认的自动 G0/G1 不直接计作模型失败，而是先记为 X。严格版 G0–G4 仍保存在附录，不再直接控制主报告。

简化主评测只使用五个透明阈值：至少两个可用参数档位；成对方向一致率大于 0.5；Theil–Sen 斜率大于 0；L4 额外要求斜率至少为 0.2 且 median valid-range NAE 不超过 0.25。这样不会把“方向略有变化但响应几乎为零”写成完全理解。NRMSE、R² 和小幅轨迹抖动仍作描述性诊断，但不再作为主等级的硬门槛。

## 3. X 不是模型失败

下列情况必须记为 `X`，不得计入 L1：

- 物体检测或身份关联不可靠；
- 抖动、离散误检或轨迹拟合失败使参数不可辨识；
- 遮挡覆盖关键碰撞/转向窗口；
- 相机运动使冻结标定失效，但 3D 重建证据又不足；
- 轨迹点数量、时间跨度或物理拟合质量不足。

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
- 3D/4D 探索性结果、运行成本和失败案例。

## 5. 结论边界

- 不把轨迹外观相似写成“模型理解了精确物理参数”；只有 L3/L4 扫描证据支持参数响应结论。
- 不把检测、标定或重建失败写成生成模型失败。
- 不把单条视频写成 L2/L3/L4，也不把多个视角或场景误当独立参数水平。
- 不把探索性 3D 结果混入侧视主定量。
- 所用宽松阈值必须随报告和 Git commit 一起冻结；若日后进行严格 Blender 校准，应作为新的严格分析版本，不能静默覆盖本结果。
- 报告“所测试模型、所测试参数范围和所测试视频长度内”的结果，不外推到所有世界模型或所有物理现象。

## 6. 服务器 CLI 的预期输入输出

主报告入口：

```bash
EVAL_ROOT=/root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake/analysis/seedance978_physics_v2
SIMPLE_ROOT="$EVAL_ROOT/simple_paper_report_v1"

python code/scripts/build_physparambench_simple_report.py \
  --all-jobs "$EVAL_ROOT/all_jobs.csv" \
  --evaluation-root "$EVAL_ROOT" \
  --output "$SIMPLE_ROOT"
```

若 CSV 已包含 `manual_generation_validity_status`，脚本自动采用人工状态；否则自动 fail 不会直接成为 L1，而先进入 X 等待复核。

输出：

- `REPORT_ZH.md` 与 `ABSTRACT_RESULTS_EN.md`：可直接汇报/写作的收口结论；
- `fig1_video_gate.svg`：L1、可进入扫描与 X；
- `fig2_baseline_response_grades.svg`：L2/L3/L4/X 主统计；
- `fig3_requested_vs_recovered.svg`：24 个 baseline Side 参数扫描；
- `fig4_evaluation_scopes.svg`：baseline、背景、视角、seed 四个作用域；
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
