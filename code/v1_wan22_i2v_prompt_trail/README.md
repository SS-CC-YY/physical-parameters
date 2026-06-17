# V1 Wan2.2 I2V prompt + center-trajectory ablation

This folder tests whether Wan2.2-I2V-A14B improves on V1 physical-parameter tasks when the single conditioning image is made more informative and the prompt is stricter.

It does **not** change the Wan2.2 model. It still uses the official single-image I2V interface:

```text
--image conditioning.png
--prompt strict_prompt
```

The main ablation is:

```text
raw_frame10 + strict prompt
center_10f  + strict prompt
optional center_24f / center_48f + strict prompt
optional trail_10f as legacy silhouette comparison
```

`center_10f` extracts the orange ball center in each previous frame and draws small center markers on the final frame. This is less visually invasive than `trail_10f`, which overlays faint translucent object silhouettes and can interfere with the generated object. The output prompt explicitly tells the model that the blue/red dots are motion-history annotations and must disappear in the generated video.

## Smoke test

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

WAN_REPO=../Wan2.2 \
CKPT_DIR=../../models/Wan2.2-I2V-A14B \
GPU_ID=7 \
EXPERIMENTS="v1_A" \
CONDITIONING_MODES="raw_frame10 center_10f" \
MAX_JOBS=2 \
FRAME_NUM=81 \
SAMPLE_STEPS=20 \
bash code/v1_wan22_i2v_prompt_trail/run_v1_prompt_trail_background.sh
```

Monitor:

```bash
tail -f outputs/v1_wan22_prompt_trail_*/master.log
```

## Full V1 first pass

```bash
WAN_REPO=../Wan2.2 \
CKPT_DIR=../../models/Wan2.2-I2V-A14B \
GPU_ID=7 \
EXPERIMENTS="v1_A v1_B v1_C v1_D" \
CONDITIONING_MODES="raw_frame10 center_10f" \
FRAME_NUM=121 \
SAMPLE_STEPS=40 \
bash code/v1_wan22_i2v_prompt_trail/run_v1_prompt_trail_background.sh
```

## Optional longer center history

`center_24f` and `center_48f` use the full render mp4 under `blender/renders/v1`. They change the current conditioning frame to frame 24 or 48 respectively. Use them only after `center_10f` works.

```bash
CONDITIONING_MODES="center_24f center_48f" \
bash code/v1_wan22_i2v_prompt_trail/run_v1_prompt_trail_background.sh
```

## Expected interpretation

- `raw_frame10 + strict prompt` better than old prompt: text constraints help.
- `center_10f` better than `raw_frame10`: single image lacked motion history.
- `center_10f` better than `trail_10f`: silhouette trails were interfering; center annotations are cleaner.
- Both still collapse: Wan2.2-I2V is not reliably controllable by numeric physical prompts under single-image conditioning.
