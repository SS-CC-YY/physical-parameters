# PhysParamBench 简化多模型评测

这套入口对任意模型使用相同的冻结任务清单、相机标定、轨迹方程和参数指标：

```text
模型视频
  -> 模型专属相机运动审计
  -> 固定相机 2D/公制平面轨迹 或 SpaTrackerV2 动态 3D 轨迹
  -> 统一 X-Z 实验平面轨迹
  -> 13 个实验对应方程拟合
  -> Target / Estimate / AE / trajectory R² / In-range 或 Out-of-range
  -> 单模型报告和跨模型论文表格
```

每个模型输入目录中的视频必须使用冻结 manifest 的文件名：
`<job_id>.mp4`。完整模型应有 978 条；不要在不同模型之间改名或重新排列任务。

## 冻结指标

- `Target`：prompt 中指定的参数。
- `Estimate`：由轨迹方程系数直接反推的有效参数。
- `AE = abs(Estimate-Target)`：保留物理单位。
- `trajectory R²`：方程对观测运动区间的解释程度，不是参数准确率。
- `target_range_status`：未截断的 Estimate 是否落在该实验预注册的目标扫描区间。
- `BNAE(aux) = AE/(max target-min target)`：仅用于跨单位汇总，可以大于 1。

`in_range` 只是用于发现极端参数失真的粗标记，不等于参数准确；单视频是否接近
指定值仍然看带物理单位的 AE。

模型汇总时先在每个实验内部统计，再对 13 个实验等权宏平均，避免组合实验因为
参数更多、记录更多而被重复加权。轨迹 R² 使用所有能形成拟合曲线的结果计算，
并单独报告 R² 覆盖率；不会只保留已经成功反推出参数的样本。

所有 3D 原始坐标仍会保存。物理拟合和动态 3D 纳入判断只使用冻结 Blender
世界坐标的 `X-Z` 实验平面；`Y` 是深度诊断轴。Y 漂移会记录
`W_DEPTH_AXIS_UNSTABLE_IGNORED`，但不会单独阻断 X-Z 参数拟合。

## 每个模型的完整入口

```bash
REMAKE_ROOT=/root/data/heyuanyu/yefei/chenyu/remake/data/blender/remake
SPATRACK_ROOT=/root/data/heyuanyu/yefei/chenyu/SpaTrackerV2
MODEL=seedance2
VIDEOS=/path/to/$MODEL/videos
OUT="$REMAKE_ROOT/analysis/models/$MODEL"

cd "$REMAKE_ROOT"
export PYTHONPATH="$REMAKE_ROOT/code/src${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=4 \
python code/scripts/run_physparambench_model_pipeline.py \
  --model-name "$MODEL" \
  --videos "$VIDEOS" \
  --output-root "$OUT" \
  --spatialtracker-root "$SPATRACK_ROOT" \
  --workers 1
```

脚本默认可续跑，输出结构固定为：

```text
analysis/models/<model>/
  camera_audit/
  tracks/
    jobs/<job_id>/trajectory_frames.csv
    jobs/<job_id>/object_track_overlay.mp4
  evaluation/
    jobs/<job_id>/result.json
    jobs/<job_id>/trajectory_plot.png
  report/
    report.html
    report.md
    parameter_results.csv
    model_summary.csv
    experiment_summary.csv
    condition_summary.csv
    paired_condition_results.csv
    paired_condition_summary.csv
    paper_parameter_tables.md
    paper_parameter_tables.tex
    paper_table_overview.md       # 覆盖率/越界率 QC
    paper_table_overview.tex      # 覆盖率/越界率 QC
    tables/
```

如果中断，可按阶段继续：

```bash
python code/scripts/run_physparambench_model_pipeline.py ... --stage audit
python code/scripts/run_physparambench_model_pipeline.py ... --stage extract
python code/scripts/run_physparambench_model_pipeline.py ... --stage fit
python code/scripts/run_physparambench_model_pipeline.py ... --stage report
```

`--skip-dynamic` 只用于 `--stage extract` 的快速预览；完整的 `all/fit/report`
必须补齐移动相机视频的 SpaTrackerV2 轨迹。

最终实验不要给不同模型复用同一个 `camera_motion_audit.jsonl`；相机漂移是各模型输出的属性。
最终跨模型报告还会检查任务身份以及 registry、evaluator、轨迹拟合器和动态
3D 门控的版本签名；不同版本的结果不会被静默混在同一张表中。

## 多模型论文表

所有模型完成参数评测后：

```bash
python code/scripts/build_simple_physics_report.py \
  --model "Wan2.2=$REMAKE_ROOT/analysis/models/wan22/evaluation" \
  --model "Seedance2.0=$REMAKE_ROOT/analysis/models/seedance2/evaluation" \
  --model "Helios=$REMAKE_ROOT/analysis/models/helios/evaluation" \
  --model "Cosmos-Predict2.5=$REMAKE_ROOT/analysis/models/cosmos25/evaluation" \
  --model "LongLive2.0=$REMAKE_ROOT/analysis/models/longlive20/evaluation" \
  --output "$REMAKE_ROOT/analysis/model_comparison_simple_v1"
```

论文主表使用 baseline + CAM_Side。`condition_summary.csv` 另外保留
各实验的 baseline/indoor/outdoor 和 Side/Main/Top 描述统计。
`paired_condition_results.csv` 和 `paired_condition_summary.csv` 会把相同实验、
参数组合、物体、seed 和参数与 baseline Side 一一配对，用于背景与视角一致性分析。
每个目标值的 `mean ± std` 主表使用 registry 顺序冻结的一次只改变一个参数分支，
避免把组合实验中不同 nuisance 参数混在同一条响应曲线里。
