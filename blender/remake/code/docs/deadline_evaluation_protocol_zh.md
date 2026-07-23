# PhysParamBench 截稿版评测协议

这套协议只回答四个问题，不再把所有诊断状态混成一套复杂等级：

1. 干净背景 Side 视角能否反推出目标物理参数；
2. 目标运动方程能否解释观测轨迹；
3. 复杂背景和视角变化是否改变恢复参数；
4. 多参数组合是否削弱模型对单个参数的理解。

## 1. 运动区间与 13 个实验的拟合

所有实验都先删除物体静止的前缀和运动结束后的尾帧。所谓“从运动开始到运动结束”并不等于用一个多项式跨过所有碰撞：有明确事件的实验在完整运动区间内分段拟合，再联合恢复共享参数。

| 实验 | 使用的运动证据 | 直接恢复量 |
|---|---|---|
| V1-A | 第一次自由下落，`z=z0+v0*t-0.5*g*t^2` | `g` |
| V1-B | 第一次墙碰撞前后速度 | `e=abs(v_after/v_before)` |
| V1-C | 开始滑动后的速度衰减，保留非零初速度 | `mu=-a/g` |
| V1-D | 自动识别第一个完整摆动周期与振幅包络 | `beta` |
| V2-A | 识别完整摆线周期；每周期用三谐波频率搜索恢复 `omega` | `g=4*a*omega^2` |
| V2-B | 逐次识别左右墙碰撞，比较每次碰撞前后速度 | 共享 `e` |
| V2-C | 分别拟合两段材料表面的减速度 | `mu_A, mu_B` |
| V2-D | 完整周期上的阻尼摆积分方程与前向轨迹 | `g, beta` |
| V2-E | 多段竖直飞行共享加速度，逐次碰撞共享速度比 | `g, e` |
| V3-A | 横向阻力与纵向飞行/反弹分开拟合 | `drag beta, g, e` |
| V3-B | 地面减速与左右墙碰撞分开拟合 | `mu, e_L, e_R` |
| V3-C | 斜坡、地面摩擦、墙面反弹三阶段拟合 | `g, mu, e` |
| V3-D | 完整周期上的重力、阻尼和局部磁力项 | `g, beta, kappa` |

反推过程不读取目标参数。完成反推后，才由冻结 registry 查找 target 并计算误差。

拟合器同时保留 **raw candidate** 和 **accepted estimate**。raw candidate 只用于解释“为什么方程门控拒绝了这个候选值”；论文主指标、背景/视角比较和参数等级只使用通过实验方程与参数归因门控的 accepted estimate。不能用数值看起来接近 target 的 rejected candidate 改善主结果。

## 2. 主指标

### 参数保真度

- **Range-normalized absolute error（主指标）**

  `NAE = abs(p_hat - p_target) / (p_max - p_min)`

  它可以在重力、恢复系数、摩擦、阻尼之间比较，也不会在 target 接近 0 时爆炸。

- **Normalized parameter RMSE**

  `NRMSE_param = sqrt(mean(NAE_i^2))`

  与平均 NAE 相比，它更惩罚少数很大的参数错误。

- **Relative error**

  `RE = abs(p_hat - p_target) / abs(p_target)`

  便于解释单个非零参数，但 target 为 0 时不作为主指标。

### 动力学/轨迹保真度

- **Trajectory RMSE**：原坐标单位下的均方根误差，保留绝对量纲。
- **Trajectory NRMSE**：每条拟合序列先除以观测轨迹的 P95-P5 范围，再合并；可跨实验比较。
- **Trajectory NMAE**：对少数跟踪抖动比 RMSE 更稳健。
- **R²**：目标方程解释观测轨迹方差的比例；可能为负，负值意味着连均值基线都不如。
- **Pearson r**：只描述形状/时间关联，不能单独证明数值正确。
- **Fit coverage**：拟合使用的独立时刻数除以有效轨迹点数，防止只挑一小段好看的轨迹。

参数保真度和轨迹保真度必须分别报告，不合成单一分数。

单视频辅助标签中，`TRAJECTORY_MISMATCH` 表示 accepted estimate 之外的轨迹检查发现 `trajectory NRMSE > 0.35` 或 `R² < 0.20`；它用于暴露明显的运动方程失配，不替代各实验自身更具体的碰撞、周期和几何门控。

## 3. 简单参数响应评价

13 个实验系统一共包含 24 个目标参数通道。评价单位是“同实验、同场景、同视角、固定其他参数的一条 OAT 参数通道”，而不是把 24 个通道称作 24 个实验。每个 target level 先在不同 seed 间取恢复参数中位数，再比较不同 target level：

- **Accurate**：参数方向一致率大于 50%，Theil-Sen response slope 至少为 0.2，median NAE 不超过有效范围的 25%，且至少一半反推证据通过对应实验方程门控；
- **Directional only**：方向一致率大于 50%，Theil-Sen slope 为正，但响应幅度、方程门控或数值误差未达到 Accurate；
- **No parameter response**：至少两个参数档位可测，但恢复参数没有正确响应；
- **Insufficient**：不足两个可测参数档位。

模型总览先在每个实验内部汇总参数通道，再对 13 个实验等权宏平均；不会把 978 条视频或多参数实验当成更多独立实验而获得额外权重。OAT nuisance 切片只由冻结 registry 的设计结构预先选择，不根据任何模型的可用率或结果好坏选择。

同时输出：

- 成对方向一致率；
- Spearman rank correlation；
- Theil-Sen response slope（所有成对斜率的中位数，抗离群值）。

## 4. 三项稳健性问题

- **背景影响**：相同模型、实验、参数、Side、seed，将 indoor/outdoor 与 baseline 配对，计算
  `D_background = abs(p_hat_scene-p_hat_baseline)/(p_max-p_min)`。报告 median、IQR 和落在 25% 参数范围内的稳定率。
- **视角影响**：相同模型、实验、参数、场景、seed，将 Main/Top 与 Side 配对，计算同样的 normalized shift。三个视频是独立生成结果，因此只讨论恢复参数一致性，不宣称逐帧多视角一致。
- **组合影响**：在 baseline Side 中，对同一参数族比较 1/2/3 个目标参数实验的 median NAE，并报告相对单参数实验的 `delta NAE`。这是描述性比较，不强行写成因果结论。

背景和视角先在每个 `(experiment, parameter)` 通道内汇总，再对通道等权；不会因为某个多参数实验或某个 seed 的视频更多而得到更大权重。

## 5. 服务器运行

每个模型必须先单独计算自己的相机漂移 audit，不能复用 Seedance 或另一模型的 audit：

```bash
python code/scripts/audit_camera_motion.py \
  --videos "$VIDEOS" \
  --output "$CAMERA_AUDIT" \
  --workers 8 \
  --no-contact-sheets
```

然后统一提取每帧实验物体位置。赶时间时先跑 `--phase side`，完成主参数与背景结论；随后在同一个输出根继续跑 `--phase robustness` 补 Main/Top：

```bash
CUDA_VISIBLE_DEVICES=4 \
python code/scripts/extract_physparambench_trajectories.py \
  --videos "$VIDEOS" \
  --camera-audit "$CAMERA_AUDIT/camera_motion_audit.jsonl" \
  --output "$TRACK_ROOT" \
  --phase side \
  --workers 1 \
  --spatialtracker-root "$SPATRACK_ROOT" \
  --isolated-process

CUDA_VISIBLE_DEVICES=4 \
python code/scripts/extract_physparambench_trajectories.py \
  --videos "$VIDEOS" \
  --camera-audit "$CAMERA_AUDIT/camera_motion_audit.jsonl" \
  --output "$TRACK_ROOT" \
  --phase robustness \
  --workers 1 \
  --spatialtracker-root "$SPATRACK_ROOT" \
  --isolated-process
```

每个模型再从冻结的逐帧轨迹反推参数：

```bash
python code/scripts/evaluate_physparambench_trajectories.py \
  --tracks "$TRACK_ROOT" \
  --output "$EVAL_ROOT" \
  --phase all
```

四个模型全部完成后，一次生成统一报告：

```bash
python code/scripts/build_deadline_physics_report.py \
  --model wan22="$WAN_EVAL" \
  --model seedance="$SEEDANCE_EVAL" \
  --model cosmos="$COSMOS_EVAL" \
  --model helios="$HELIOS_EVAL" \
  --output "$REMAKE_ROOT/analysis/four_models_deadline_report"
```

最终报告默认要求每个模型都完整覆盖冻结的 978 个唯一 job，且四个模型使用同一 registry/evaluator/fitter lineage。仅做中途进度检查时可以加 `--allow-partial`；带该选项的报告不能直接用于最终论文表格。

首先打开：

```bash
ls "$REMAKE_ROOT/analysis/four_models_deadline_report"
```

给人阅读的入口是 `index.html` 和 `REPORT_ZH.md`。CSV 用于核验和论文制表，不要求人工逐行阅读。
