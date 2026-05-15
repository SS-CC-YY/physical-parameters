# Baseline V1 T2V/TI2V code (v2)

This folder contains an updated generation framework for Baseline V1 experiments.

## What changed

1. `generate_videos.py`
   - fixed local model path resolution
   - uses `local_files_only=True` when the resolved path exists locally
   - supports `wan2.2_ti2v_5b`
   - supports `wan2.2_t2v_a14b`
   - supports optional `init_image` for TI2V
   - supports `guidance_scale_2`
   - stores `prompt_style` and resolved model path in metadata

2. `build_manifest_v1.py`
   - expands combinations over `model x prompt_style x seed`
   - supports `init_image`
   - supports `guidance_scale_2`

3. `baseline_v1_prompts.py`
   - adds `short_raw`, `structured_raw`, `structured_detailed`

4. `launch_wan_official_dist.py`
   - launches a single JSON job through the official Wan repo using `torchrun`
   - useful when you want one task to span 4 GPUs

## Example: build manifest

```bash
python build_manifest_v1.py \
  --config configs/baseline_v1_prompt_ablation.example.json \
  --output manifests/freefall_prompt_ablation.jsonl \
  --models wan2.2_ti2v_5b \
  --prompt-styles short_raw structured_raw structured_detailed \
  --num-seeds 3 \
  --seed-base 42
```

## Example: run local diffusers batch generation

```bash
python generate_videos.py \
  --manifest manifests/freefall_prompt_ablation.jsonl \
  --outdir outputs/freefall_prompt_ablation \
  --gpus 0,1,2,3 \
  --local-model-root /mnt/data/scy \
  --allow-tf32 \
  --cpu-offload \
  --force-local-only
```

## Example: run one single job across 4 GPUs with the official Wan repo

1. Save one job as a JSON file, for example `job_one.json`.
2. Run:

```bash
python launch_wan_official_dist.py \
  --job-json job_one.json \
  --wan-repo-dir /mnt/data/scy/Wan2.2 \
  --gpus 0,1,2,3
```

## Notes

- The official Wan2.2 repo states that TI2V-5B supports 720P at 24 FPS and can run on consumer-grade 24GB GPUs; its TI2V size is `1280*704`. It also states that if the `image` argument is configured, the task becomes image-to-video; otherwise it defaults to text-to-video. The same repo provides multi-GPU commands using FSDP + DeepSpeed Ulysses. citeturn449945view0
- The official Wan2.2 diffusers model card says TI2V-5B supports both text-to-video and image-to-video at 720P/24fps, and highlights larger training data than Wan2.1. citeturn449945view1
- The official Wan2.2 T2V-A14B model card says A14B supports both 480P and 720P, and the repo provides single-GPU and multi-GPU inference examples. citeturn449945view2turn449945view0
