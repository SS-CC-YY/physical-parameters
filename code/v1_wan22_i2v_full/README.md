# V1 Wan2.2 I2V A14B Full Experiment

This folder runs all V1 variants with the official `Wan2.2-I2V-A14B` model and
then estimates the hidden physics parameter from generated videos.

Default scope:

- Experiments: `v1_A`, `v1_B`, `v1_C`, `v1_D`
- Variants: all 4 variants per experiment, 16 jobs total
- Camera: `CAM_Side`
- Model: official / ModelScope `Wan2.2-I2V-A14B`
- GPU: `7`

`CAM_Side` is the default because the inverse-physics validation relies on a
side-view image trajectory. The manifest builder supports more cameras, but the
validator skips non-side cameras unless explicitly allowed.

## Files

- `build_v1_manifest.py`: builds all V1 I2V jobs from `blender/seeds/v1`.
- `run_v1_wan22_official.py`: runs official Wan2.2 `generate.py` job by job,
  with resume support and per-job logs.
- `evaluate_v1_physics.py`: tracks the orange ball and estimates `g`, `e`,
  `mu`, or `gamma` depending on the experiment.
- `visualize_v1_eval.py`: creates trajectory plots, target-vs-estimate summary
  plots, an HTML report, and optional overlay videos.
- `launch_v1_background.sh`: starts the full run with `nohup`, writes a PID
  file, and survives VS Code tunnel disconnects.
- `requirements.txt`: minimal dependencies for the manifest and evaluator.

## Server Setup

Assumed layout:

```text
~/data/heyuanyu/yefei/chenyu/data/
  physical-parameters/
  Wan2.2/
  models/Wan2.2-I2V-A14B/
```

Install only the lightweight evaluation dependencies in this repo:

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters
python -m pip install -r code/v1_wan22_i2v_full/requirements.txt
```

The official Wan2.2 dependencies are still installed from `../Wan2.2`. If
official Wan2.2 raises `FLASH_ATTN_2_AVAILABLE`, either install the matching
`flash-attn` wheel or apply the fallback patch already provided in:

```bash
python code/phase1_wan22_i2v_baseline/patch_wan22_attention_fallback.py \
  --wan-repo ../Wan2.2
```

## 1. Build The Full V1 Manifest

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

python code/v1_wan22_i2v_full/build_v1_manifest.py \
  --seeds-root blender/seeds \
  --output outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --prompt-mode explicit \
  --cameras CAM_Side
```

This writes 16 jobs.

`--prompt-mode explicit` puts the target hidden value into the prompt. This is
the practical feasibility baseline for a one-image I2V model. `--prompt-mode
hidden` is available, but Wan2.2 I2V only receives `frame_10.png`, so it cannot
reliably infer velocity/history-dependent parameters from the first 10 frames.

## 2. Smoke Test One Job

```bash
python code/v1_wan22_i2v_full/run_v1_wan22_official.py \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --wan-repo ../Wan2.2 \
  --ckpt-dir ../models/Wan2.2-I2V-A14B \
  --outdir outputs/v1_wan22_i2v_a14b \
  --gpu-id 7 \
  --size '832*480' \
  --frame-num 121 \
  --max-jobs 1
```

`frame-num 121` is close to 7.6 seconds at 16 fps and gives enough motion for
bounce/friction/pendulum validation. If memory is tight, use `--frame-num 81`
for a cheaper smoke test.

## 3. Run In Background

Use the launcher so a VS Code tunnel reset does not stop the job:

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

bash code/v1_wan22_i2v_full/launch_v1_background.sh
```

Default paths are:

```text
WAN_REPO=../Wan2.2
CKPT_DIR=../models/Wan2.2-I2V-A14B
OUTDIR=outputs/v1_wan22_i2v_a14b
GPU_ID=7
FRAME_NUM=121
SIZE=832*480
PROMPT_MODE=explicit
CAMERAS=CAM_Side
```

Override them as needed:

```bash
OUTDIR=outputs/v1_wan22_i2v_a14b_smoke MAX_JOBS=1 \
bash code/v1_wan22_i2v_full/launch_v1_background.sh
```

Monitor:

```bash
tail -f outputs/v1_wan22_i2v_a14b/run.log
cat outputs/v1_wan22_i2v_a14b/run.pid
ps -fp "$(cat outputs/v1_wan22_i2v_a14b/run.pid)"
```

Stop:

```bash
kill "$(cat outputs/v1_wan22_i2v_a14b/run.pid)"
```

Resume:

```bash
bash code/v1_wan22_i2v_full/launch_v1_background.sh
```

Existing videos are skipped unless `OVERWRITE=1` is set.

## 4. Evaluate Hidden Physics

After generation:

```bash
python code/v1_wan22_i2v_full/evaluate_v1_physics.py \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --outdir outputs/v1_wan22_i2v_a14b/eval
```

Outputs:

```text
outputs/v1_wan22_i2v_a14b/eval/summary.csv
outputs/v1_wan22_i2v_a14b/eval/summary.json
outputs/v1_wan22_i2v_a14b/eval/tracks/*.csv
```

Estimator mapping:

- `v1_A`: fit vertical quadratic motion and estimate `g_hidden`.
- `v1_B`: detect bounce peak heights and estimate restitution `e_hidden`.
- `v1_C`: fit horizontal deceleration and estimate friction `mu_hidden`.
- `v1_D`: fit pendulum amplitude decay and estimate damping `gamma_hidden`.

The estimates use image tracking and radius-based pixel-to-meter calibration, so
they are validation proxies. The main first-stage question is whether estimates
are directionally correct and rank the variants properly.

## 5. Visualize Evaluation

After `evaluate_v1_physics.py` has written `summary.csv` and `tracks/*.csv`:

```bash
python code/v1_wan22_i2v_full/visualize_v1_eval.py \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --eval-dir outputs/v1_wan22_i2v_a14b/eval \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl
```

Outputs:

```text
outputs/v1_wan22_i2v_a14b/eval/visualization/index.html
outputs/v1_wan22_i2v_a14b/eval/visualization/summary/*.png
outputs/v1_wan22_i2v_a14b/eval/visualization/jobs/*_track.png
```

If you also want tracked-center overlay videos, add `--make-overlays`:

```bash
python code/v1_wan22_i2v_full/visualize_v1_eval.py \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --eval-dir outputs/v1_wan22_i2v_a14b/eval \
  --manifest outputs/v1_wan22_i2v_a14b/manifest.jsonl \
  --make-overlays
```

For a quick check, only render one overlay:

```bash
python code/v1_wan22_i2v_full/visualize_v1_eval.py \
  --generated-root outputs/v1_wan22_i2v_a14b \
  --eval-dir outputs/v1_wan22_i2v_a14b/eval \
  --overlay-max-jobs 1 \
  --make-overlays
```
