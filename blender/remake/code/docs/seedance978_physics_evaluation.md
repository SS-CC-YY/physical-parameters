# Seedance 978：生成有效性 → 3D 轨迹 → 物理参数

这条流程覆盖冻结的全部 `978` 条标准球视频，并固定采用以下执行和汇报顺序：

1. `CAM_Side` 共 `510` 条先运行；
2. 其中默认 seed `341867882` 的 `432` 条构成 headline 物理主评估；
3. 其余 `78` 条 Side 视频只用于随机 seed 稳定性，与主集中对应的 26 条组成 `26×4=104` 条比较，不重复加权到 headline；
4. `CAM_Main` `234` 条随后运行；
5. `CAM_Top` `234` 条最后运行；
6. Main/Top 只与同一 `(experiment, tuple, scene, object, seed)` 的 Side 结果组成 `234` 个鲁棒性三元组。它们是三次独立生成，不是假设同步的多目视频。

精简后的冻结任务清单与相机审计位于：

```text
code/assets/seedance978_evaluation/
  manifest.jsonl          # 978 行
  camera_motion.jsonl     # 978 行
  experiment_registry.json # 13 个实验、69 个冻结参数组合和评分范围
```

## 1. 每条视频的严格顺序

```text
相机审计
  → 标准球追踪
  → 视频生成有效性 gate
  → 选择静态标定或动态 3D/4D 重建
  → 再次做球尺寸/3D 刚性 gate
  → 仅 pass 样本拟合轨迹公式和物理参数
  → 最后读取给定物理参数并评分
```

相机漂移不是视频生成失败。它只决定重建路线：

- `no_significant_camera_change`：使用冻结首帧的相机 `K/R/t`、标准球真实半径和实验运动流形；
- `camera_changed`、`borderline_below_threshold`、审计不确定：使用 SpaTrackerV2；
- 缺失相机审计证据不会擅自套用静态相机参数。

## 2. 视频生成有效性 gate

状态只有三种：

- `pass`：证据完整，允许物理拟合；
- `fail`：有持续且明显的生成失败证据，直接写 `skipped_video_generation_validity`；
- `indeterminate`：检测、场景刚性或动态 3D 证据不足，不冒充明确失败，但也不发布物理参数。

明确失败包括：

- 场景切换或突然重构；
- 标准球连续多帧明显拉伸、非圆或尺度崩坏；
- 3D 物体点云无法由一个刚体运动持续解释；
- 去除全局相机运动后，背景/非实验物体仍在多个区域持续出现几何与外观异常。

单帧运动模糊、落地接触、遮挡、全局平移/旋转/缩放不会单独触发明确失败。摆锤实验会从背景特征中排除球轨迹和 pivot-to-ball 装置扫掠区域。

固定相机分支还使用已知球大小作为逐帧米制重建硬约束：

```text
0.65 <= observed_projected_radius / expected_projected_radius <= 1.35
```

超出范围的帧不能进入物理拟合；只有更严重且持续的尺度异常才上升为视频生成失败。

SpaTrackerV2 动态分支在没有 Blender 背景 3D anchor 时使用：

1. 首帧球面查询点与已知球面 3D 点做 Sim(3)，建立米制世界坐标；
2. 首帧背景查询点冻结为静态参考；
3. 后续逐帧用背景 robust Sim(3) 消除全局相机漂移；
4. 球查询点做刚体配准，恢复球心轨迹；
5. 用球刚体残差、内点率和背景空间覆盖率判断持续非刚性。

## 3. 物理量和评分

13 个实验的拟合公式与具体逆问题见 `code/docs/fixed_camera_physics_evaluation.md`。统一参数误差为：

```text
NAE_p = abs(p_est - p_gt) / (p_max - p_min)
score_p = 100 * max(0, 1 - NAE_p)
```

生成失败/证据不确定的样本没有 `p_est`，不会被伪造成一个数值误差。最终报告并列给出：

- 所有计划视频为分母的生成有效率；
- 只在生成有效且成功拟合的视频上计算的条件参数精度；
- Side headline；
- Main/Top 相对 Side 的参数差异；
- 4 个 seed 的参数标准差。

## 4. 服务器更新

共享账户不要修改全局 `safe.directory`。使用临时 Git 配置：

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data
REMAKE_ROOT="$REPO/blender/remake"
SAFE_CFG=$(mktemp /tmp/remake-safe-gitconfig.XXXXXX)

git config --file "$SAFE_CFG" --add safe.directory "$REPO"
GIT_CONFIG_SYSTEM="$SAFE_CFG" GIT_LFS_SKIP_SMUDGE=1 \
git -C "$REPO" pull --ff-only origin codex/remake-wan22-lfs
rm -f "$SAFE_CFG"

cd "$REMAKE_ROOT"
git -c safe.directory="$REPO" log -1 --oneline
```

首次使用动态分支时安装 SpaTrackerV2 环境；已经安装过则只需 activate：

```bash
cd "$REMAKE_ROOT/rebuild-test/spatialtrackerv2"
bash scripts/setup_server.sh
conda activate "$PWD/env"
```

## 5. 先做一条 canary

把 `VIDEOS` 指向服务器上 978 条 MP4 的实际目录。现有 Seedance run 通常是：

```bash
REMAKE_ROOT=/root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake
SEED_RUN_ID=seedance20_factorized978_480p_20260719_140052
VIDEOS="$REMAKE_ROOT/outputs/seedance978/$SEED_RUN_ID/videos"
EVAL_OUT="$REMAKE_ROOT/outputs/seedance978_physics_v2"

find "$VIDEOS" -maxdepth 1 -type f -name '*.mp4' | wc -l
```

应输出 `978`。然后：

```bash
cd "$REMAKE_ROOT"
CUDA_VISIBLE_DEVICES=7 python code/scripts/run_seedance978_physics_evaluation.py \
  --videos "$VIDEOS" \
  --output "$EVAL_OUT" \
  --phase side \
  --max-jobs 1 \
  --run-dynamic \
  --overlay-count 1
```

检查：

```bash
find "$EVAL_OUT/jobs" -name result.json | wc -l
find "$EVAL_OUT/jobs" -name 'validity_*' -type f
```

## 6. 正式运行并断点续跑

先完成 Side：

```bash
cd "$REMAKE_ROOT"
CUDA_VISIBLE_DEVICES=7 nohup python code/scripts/run_seedance978_physics_evaluation.py \
  --videos "$VIDEOS" \
  --output "$EVAL_OUT" \
  --phase side \
  --run-dynamic \
  --overlay-count 24 \
  > "$EVAL_OUT.side.log" 2>&1 &
echo $!
```

Side 完成后依次运行 Main、Top：

```bash
CUDA_VISIBLE_DEVICES=7 nohup python code/scripts/run_seedance978_physics_evaluation.py \
  --videos "$VIDEOS" --output "$EVAL_OUT" --phase main --run-dynamic --overlay-count 12 \
  > "$EVAL_OUT.main.log" 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python code/scripts/run_seedance978_physics_evaluation.py \
  --videos "$VIDEOS" --output "$EVAL_OUT" --phase top --run-dynamic --overlay-count 12 \
  > "$EVAL_OUT.top.log" 2>&1 &
```

同一命令、同一 `--output` 可安全断点续跑；已有成功结果会复用。不要在正常续跑时加 `--overwrite`，它会重新计算已有任务，并让动态任务也重跑。

也可以前台一次运行完整顺序：

```bash
CUDA_VISIBLE_DEVICES=7 python code/scripts/run_seedance978_physics_evaluation.py \
  --videos "$VIDEOS" --output "$EVAL_OUT" --phase all --run-dynamic
```

## 7. 输出

批次级：

```text
aggregate.json                 # 总结；Side 是 headline
all_jobs.csv                   # 所有已完成任务
side_primary_summary.csv       # 432 条主评估
viewpoint_robustness.csv       # 234 个三视角组
seed_stability.csv             # 26 个四 seed 组
routing_side/main/top.jsonl    # 静态/动态路由证据
```

每条视频：

```text
jobs/<job_id>/result.json
jobs/<job_id>/tracking.csv
jobs/<job_id>/trajectory.csv                  # 仅重建成功时
jobs/<job_id>/trajectory_plot.png             # x-z 轨迹与 x/y/z-time 曲线
jobs/<job_id>/validity_object_track_overlay.mp4
jobs/<job_id>/validity_evidence_contact_sheet.jpg
```

动态候选另外保存在 `dynamic/<job_id>/`，包含 SpaTrackerV2 原始轨迹、3D 轨迹图、查询点 overlay、刚性证据和 `generation_validity.json`。
