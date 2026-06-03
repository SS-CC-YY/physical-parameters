# V3 Wan2.2 I2V A14B Full Experiment

This folder runs all V3 tasks with official `Wan2.2-I2V-A14B`.

V3 is multi-parameter and multi-physics. The evaluator here is a first-pass
proxy evaluator for overnight screening: it tracks the visible orange target,
computes image-plane trajectory quality, and estimates simple proxy parameters
where possible. Use it to decide whether the generated videos are trustworthy
enough before upgrading to geometry-aware multi-parameter fitting.

Default scope:

- Experiments: `v3_A` ... `v3_G`
- Variants: 4 variants per experiment, 28 jobs total
- Camera: `CAM_Side`
- GPU: `7`

## Files

- `build_v3_manifest.py`: builds V3 I2V jobs from `blender/seeds/v3`.
- `run_v3_wan22_official.py`: V3 wrapper that reuses the V2 official runner.
- `launch_v3_background.sh`: background `nohup` launcher.
- `evaluate_v3_physics.py`: V3 tracking and proxy evaluation.
- `visualize_v3_eval.py`: V3 wrapper that reuses the V2 visualizer.
- `requirements.txt`: lightweight dependencies.

## Run

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

bash code/v3_wan22_i2v_full/launch_v3_background.sh
```

Evaluate:

```bash
python code/v3_wan22_i2v_full/evaluate_v3_physics.py \
  --manifest outputs/v3_wan22_i2v_a14b/manifest.jsonl \
  --generated-root outputs/v3_wan22_i2v_a14b \
  --outdir outputs/v3_wan22_i2v_a14b/eval
```

Visualize:

```bash
python code/v3_wan22_i2v_full/visualize_v3_eval.py \
  --generated-root outputs/v3_wan22_i2v_a14b \
  --eval-dir outputs/v3_wan22_i2v_a14b/eval \
  --manifest outputs/v3_wan22_i2v_a14b/manifest.jsonl
```

