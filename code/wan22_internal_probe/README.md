# Wan2.2 internal-state probe experiment

This folder implements a practical first internal-probing experiment for the
physical-parameter benchmark.

The experiment focuses on V1-A free-fall gravity:

- Generate Wan2.2 I2V videos from the same benchmark seed frames. By default,
  the probe uses `center_10f` conditioning images: the current frame with small
  center markers for the first ten seed frames.
- Capture compact per-block DiT activations during the official `generate.py`
  forward denoising process by injecting a `sitecustomize.py` hook.
- Train ridge-regression linear probes to predict `g_hidden` from the captured
  internal states.
- Compare internal-state predictability with output-video parameter recovery.

Important interpretation note:

The default `visual_trace` prompt mode does **not** include the numeric target
gravity or the `g_hidden` parameter name in the text prompt. The intended signal
source is the visual motion history encoded by the center markers. A successful
probe in this setting is therefore a stronger check than the old `explicit`
baseline, but it still proves decodability from the generation computation rather
than a full mechanistic explanation of physical reasoning.

`PROMPT_MODE=explicit` remains available only as a leakage/control baseline: it
puts the target value in text and should not be used as the main result.

If no activation records are captured, rerun with a broader hook filter:

```bash
CLASS_REGEX="(Block|Layer)" bash code/wan22_internal_probe/run_probe_background.sh
```

## Quick server smoke test

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

WAN_REPO=../Wan2.2 \
CKPT_DIR=../../models/Wan2.2-I2V-A14B \
GPU_ID=7 \
SEEDS="11 22" \
MAX_JOBS=4 \
FRAME_NUM=81 \
SAMPLE_STEPS=12 \
bash code/wan22_internal_probe/run_probe_background.sh
```

Monitor:

```bash
tail -f outputs/wan22_internal_probe_*/master.log
```

## Full first-pass run

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

WAN_REPO=../Wan2.2 \
CKPT_DIR=../../models/Wan2.2-I2V-A14B \
GPU_ID=7 \
SEEDS="11 22 33 44 55" \
FRAME_NUM=121 \
SAMPLE_STEPS=40 \
bash code/wan22_internal_probe/run_probe_background.sh
```

Expected outputs:

```text
outputs/wan22_internal_probe_YYYYMMDD_HHMMSS/
  manifest.jsonl
  videos/
  features/*.npz
  probe/
    probe_summary.csv
    best_probe.json
  eval_video/
    summary.csv
  comparison/
    comparison_summary.json
```

## Recommended controls

The default run creates conditioning images under:

```text
outputs/wan22_internal_probe_YYYYMMDD_HHMMSS/conditioning_images/center_10f/
```

If your server already has updated `frame_10.png` files with center markers
drawn in place, and you do not want this script to build separate conditioning
images, run:

```bash
USE_CONDITIONING_ROOT=0 \
PROMPT_MODE=visual_trace \
bash code/wan22_internal_probe/run_probe_background.sh
```

If you already generated conditioning images elsewhere, point to them directly:

```bash
WAN_REPO=../Wan2.2 \
CKPT_DIR=../../models/Wan2.2-I2V-A14B \
GPU_ID=7 \
PROMPT_MODE=visual_trace \
MAKE_CONDITIONING_IMAGES=0 \
CONDITIONING_ROOT=/path/to/conditioning_images \
CONDITIONING_MODE=center_10f \
SEEDS="11 22 33 44 55" \
FRAME_NUM=121 \
SAMPLE_STEPS=40 \
bash code/wan22_internal_probe/run_probe_background.sh
```

Interpretation:

- `visual_trace` high probe score + poor output recovery: the internal denoising
  states carry the gravity signal from visual conditioning, but the rendered
  continuation does not obey it.
- `visual_trace` low probe score: the single-image visual trace is not enough for
  Wan2.2 internal states to expose the target parameter linearly.
- `explicit` high score should be treated as a text-conditioning leakage/control
  result, not as evidence of visual physical inference.
