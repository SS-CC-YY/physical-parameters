# 五模型统一轨迹评测

该入口只读取已经冻结的逐帧轨迹：

```text
tracks/jobs/<job_id>/track_result.json
tracks/jobs/<job_id>/trajectory_frames.csv
```

它不会重新读取 MP4，不会重新运行 2D 检测或 SpaTrackerV2。五个模型依次使用
同一份 978-job manifest、experiment registry、物理拟合器和 evaluator 1.3.0，
随后才允许生成统一表格和实验级证据报告。

## 服务器运行

先停止误启动的旧评测进程，再更新代码。不要停止视频生成或轨迹提取进程：

```bash
pgrep -af 'evaluate_physparambench_trajectories.py|run_unified_five_model_evaluation.py'
kill <确认属于旧评测的PID>
```

设置仓库和五套轨迹路径。下面的 Wan2.2 与 Seedance 路径来自当前服务器；
其他三项如果目录名不同，只改右侧路径：

```bash
REPO=/root/data/heyuanyu/yefei/chenyu/remake/data
REMAKE_ROOT="$REPO/blender/remake"

WAN_TRACKS="$REMAKE_ROOT/analysis/models/wan22/tracks"
SEEDANCE_TRACKS="$REMAKE_ROOT/outputs/seedance978_tracks_v1"
COSMOS_TRACKS="$REMAKE_ROOT/analysis/models/cosmos3_nano/tracks"
HELIOS_TRACKS="$REMAKE_ROOT/analysis/models/helios/tracks"
LONGLIVE_TRACKS="$REMAKE_ROOT/analysis/models/longlive20/tracks"

for root in \
  "$WAN_TRACKS" "$SEEDANCE_TRACKS" "$COSMOS_TRACKS" \
  "$HELIOS_TRACKS" "$LONGLIVE_TRACKS"
do
  printf '%s: ' "$root"
  find "$root/jobs" -mindepth 2 -maxdepth 2 \
    -name trajectory_frames.csv | wc -l
done
```

五行都必须输出 `978`。然后执行：

```bash
conda activate SpaTrack2
cd "$REMAKE_ROOT"
export PYTHONPATH="$REMAKE_ROOT/code/src${PYTHONPATH:+:$PYTHONPATH}"

OUT="$REMAKE_ROOT/analysis/five_models_unified_v13"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="" \
python code/scripts/run_unified_five_model_evaluation.py \
  --model "Wan2.2=$WAN_TRACKS" \
  --model "Seedance2.0=$SEEDANCE_TRACKS" \
  --model "Cosmos-Predict2.5-14B=$COSMOS_TRACKS" \
  --model "Helios=$HELIOS_TRACKS" \
  --model "LongLive2.0=$LONGLIVE_TRACKS" \
  --output-root "$OUT" \
  2>&1 | tee "$OUT.run.log"
```

命令可安全续跑。未传 `--overwrite` 时，逐视频 lineage 未变化的结果会跳过；
代码、registry 或 fitter 发生变化的结果会自动重新计算。

## 验收

成功后：

```bash
python - "$OUT/lineage_audit.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
print("status:", d["status"])
print("models:", d["model_count"])
print("jobs/model:", d["manifest_job_count"])
print(json.dumps(d["common_lineage"], indent=2))
assert d["status"] == "passed"
assert d["model_count"] == 5
assert d["manifest_job_count"] == 978
assert d["phase"] == "all"
assert d["common_lineage"]["evaluator_version"] == "1.3.0"
PY
```

只有下列公共字段完全一致时，统一报告才会生成：

- `registry_sha256`
- `evaluator_source_sha256`
- `physics_fitter_source_sha256`
- `dynamic_3d_gate_source_sha256`
- `evaluator_version`

`trajectory_sha256` 和 `track_result_sha256` 必然随模型/视频变化，仅用于逐条追溯。

主要输出：

```text
analysis/five_models_unified_v13/
├── lineage_audit.json
├── unified_run_summary.json
├── models/<model>/evaluation/
├── report/
│   ├── paper_main_table.csv
│   ├── paper_main_table.tex
│   ├── scan_response_channels.csv
│   └── REPORT_FOR_ADVISORS_ZH.md
└── experiment_evidence/
    ├── experiment_evidence_matrix.csv
    ├── paper_evidence_main_table.csv
    ├── PAPER_MAIN_TABLE_AUDITABLE_ZH.md
    └── EXPERIMENT_CONCLUSIONS_ZH.md
```

如果任一模型缺任务、混入旧 evaluator、registry/fitter 哈希不一致，程序会在
报告生成前失败，不会产生一张混合口径的五模型主表。
