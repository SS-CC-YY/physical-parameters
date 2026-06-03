# V2 Wan2.2 I2V A14B Full Experiment

This folder is the V2 counterpart of `code/v1_wan22_i2v_full`. It runs all V2
I2V tasks with the official `Wan2.2-I2V-A14B` model, evaluates hidden physics
parameters from generated videos, and produces visual diagnostics.

Default scope:

- Experiments: `v2_A`, `v2_B`, `v2_C`, `v2_D`, `v2_E`, `v2_F`
- Variants: 4 variants per experiment, 24 jobs total
- Camera: `CAM_Side`
- Model: official / ModelScope `Wan2.2-I2V-A14B`
- GPU: `7`

`CAM_Side` remains the default because V2 evaluation relies on visible image
trajectory. The manifest builder supports other cameras, but inverse-physics
metrics are intended for side view first.

## Files

- `build_v2_manifest.py`: builds V2 I2V jobs from `blender/seeds/v2`.
- `run_v2_wan22_official.py`: official Wan2.2 `generate.py` batch runner with
  resume support and per-job logs.
- `launch_v2_background.sh`: starts a background `nohup` run and writes a PID
  file so VS Code tunnel disconnects do not stop generation.
- `evaluate_v2_physics.py`: tracks the target object and estimates V2 hidden
  parameters with first-pass proxy estimators.
- `visualize_v2_eval.py`: creates track plots, summary plots, HTML report, and
  optional overlay videos.
- `requirements.txt`: lightweight evaluation and visualization dependencies.

## Evaluation Scope

The V2 evaluator is intentionally a first-pass proxy evaluator:

- `v2_A`: projectile + bounce, estimates `g_hidden` from early vertical
  acceleration before/around the first bounce.
- `v2_B`: repeated vertical bounce, estimates `e_hidden` from peak-height ratios.
- `v2_C`: incline + floor slide block, estimates `mu_hidden` mainly from
  horizontal deceleration after the ramp/floor transition.
- `v2_D`: long damped pendulum, estimates `gamma_hidden` from amplitude decay.
- `v2_E`: tilted-floor bounce, estimates `e_hidden` from peak-height ratios as a
  proxy; full normal/tangent decomposition can be added later.
- `v2_F`: forced damped pendulum, estimates `gamma_hidden` from amplitude
  envelope proxy; full forced ODE fitting can be added later.

The purpose is to keep the V2 pipeline operational and comparable to V1. For
paper-level metrics, C/E/F should later be upgraded with geometry-aware fitting.

## Server Setup

Assumed layout:

```text
~/data/heyuanyu/yefei/chenyu/data/
  physical-parameters/
  Wan2.2/
  models/Wan2.2-I2V-A14B/
```

Install lightweight dependencies:

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters
python -m pip install -r code/v2_wan22_i2v_full/requirements.txt
```

If official Wan2.2 still lacks `flash-attn`, either install the matching wheel
or temporarily apply the fallback patch:

```bash
python code/phase1_wan22_i2v_baseline/patch_wan22_attention_fallback.py \
  --wan-repo ../Wan2.2
```

## 1. Build Manifest

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

python code/v2_wan22_i2v_full/build_v2_manifest.py \
  --seeds-root blender/seeds \
  --output outputs/v2_wan22_i2v_a14b/manifest.jsonl \
  --prompt-mode explicit \
  --cameras CAM_Side
```

Expected output: 24 jobs.

`--prompt-mode explicit` writes the hidden value into the prompt. This is the
recommended feasibility baseline for single-image I2V. Use `--prompt-mode
hidden` only after the explicit pipeline is stable.

## 2. Smoke Test

```bash
python code/v2_wan22_i2v_full/run_v2_wan22_official.py \
  --manifest outputs/v2_wan22_i2v_a14b/manifest.jsonl \
  --wan-repo ../Wan2.2 \
  --ckpt-dir ../models/Wan2.2-I2V-A14B \
  --outdir outputs/v2_wan22_i2v_a14b \
  --gpu-id 7 \
  --size '832*480' \
  --frame-num 121 \
  --max-jobs 1
```

If memory is tight, use `--frame-num 81` for smoke tests.

## 3. Run In Background

```bash
bash code/v2_wan22_i2v_full/launch_v2_background.sh
```

Default environment variables:

```text
WAN_REPO=../Wan2.2
CKPT_DIR=../models/Wan2.2-I2V-A14B
OUTDIR=outputs/v2_wan22_i2v_a14b
GPU_ID=7
FRAME_NUM=121
SIZE=832*480
PROMPT_MODE=explicit
CAMERAS=CAM_Side
```

Monitor:

```bash
tail -f outputs/v2_wan22_i2v_a14b/run.log
cat outputs/v2_wan22_i2v_a14b/run.pid
ps -fp "$(cat outputs/v2_wan22_i2v_a14b/run.pid)"
```

Resume simply by launching again. Existing videos are skipped unless
`OVERWRITE=1` is set.

## 4. Evaluate

```bash
python code/v2_wan22_i2v_full/evaluate_v2_physics.py \
  --manifest outputs/v2_wan22_i2v_a14b/manifest.jsonl \
  --generated-root outputs/v2_wan22_i2v_a14b \
  --outdir outputs/v2_wan22_i2v_a14b/eval
```

Outputs:

```text
outputs/v2_wan22_i2v_a14b/eval/summary.csv
outputs/v2_wan22_i2v_a14b/eval/summary.json
outputs/v2_wan22_i2v_a14b/eval/tracks/*.csv
```

## 5. Visualize

```bash
python code/v2_wan22_i2v_full/visualize_v2_eval.py \
  --generated-root outputs/v2_wan22_i2v_a14b \
  --eval-dir outputs/v2_wan22_i2v_a14b/eval \
  --manifest outputs/v2_wan22_i2v_a14b/manifest.jsonl
```

Optional overlay videos:

```bash
python code/v2_wan22_i2v_full/visualize_v2_eval.py \
  --generated-root outputs/v2_wan22_i2v_a14b \
  --eval-dir outputs/v2_wan22_i2v_a14b/eval \
  --manifest outputs/v2_wan22_i2v_a14b/manifest.jsonl \
  --make-overlays
```

