# Phase 1 Wan2.2 I2V Feasibility Baseline

目标：先做一个最小闭环，确认 Wan2.2 I2V A14B 可以在服务器上正常生成视频，并且我们可以从生成视频中跟踪球、拟合轨迹、恢复一个和隐藏物理量 `g_hidden` 相关的 proxy。

这个阶段不是最终 benchmark。因为 Wan2.2 I2V 是单图输入模型，看不到前 10 帧的速度历史，所以默认使用 `explicit_g` prompt：把目标重力写进 prompt，然后检查生成视频里的运动是否随 `g` 单调变化。后续正式实验再切换到更严格的 hidden / video-extension 设置。

## Files

- `build_manifest.py`：从 `blender/seeds` 生成阶段一 manifest。
- `run_wan22_i2v.py`：调用本地 Wan2.2 I2V A14B Diffusers 模型生成视频。
- `evaluate_freefall.py`：颜色分割跟踪球心，拟合二次轨迹，输出 `g_hat_D_per_s2`。
- `requirements.txt`：服务器环境依赖。

## Server Setup

在服务器 repo 目录下执行：

```bash
cd ~/data/heyuanyu/yefei/chenyu/data/physical-parameters

git pull origin main
git lfs pull

pip install -r code/phase1_wan22_i2v_baseline/requirements.txt
```

如果服务器环境没有合适的 PyTorch，请先按 CUDA 版本安装 PyTorch。Wan2.2 官方 Diffusers 示例使用 `WanImageToVideoPipeline`，并提示该模型可能需要安装 diffusers 源码版。

## 1. Build Manifest

默认使用 `v1_A` 的四个重力变体、`CAM_Side`、`frame_10.png`：

```bash
python code/phase1_wan22_i2v_baseline/build_manifest.py \
  --output code/phase1_wan22_i2v_baseline/manifests/v1a_cam_side_explicit_g.jsonl \
  --prompt-mode explicit_g \
  --num-seeds 1
```

预期生成 4 个任务：`g2`, `g4p9`, `g9p81`, `g14p7`。

## 2. Generate Videos On GPU 7

假设你的模型在 repo 同级目录：

```text
~/data/heyuanyu/yefei/chenyu/data/models/Wan2.2-I2V-A14B-Diffusers
```

注意：这套脚本针对 Hugging Face Diffusers 格式的 `Wan-AI/Wan2.2-I2V-A14B-Diffusers`。如果你下载的是官方原始权重目录 `Wan2.2-I2V-A14B`，需要先改用 Diffusers 版本，或者走 Wan2.2 官方 repo 的 `generate.py`。

执行：

```bash
python code/phase1_wan22_i2v_baseline/run_wan22_i2v.py \
  --manifest code/phase1_wan22_i2v_baseline/manifests/v1a_cam_side_explicit_g.jsonl \
  --model-path ../models/Wan2.2-I2V-A14B-Diffusers \
  --outdir outputs/phase1_wan22_i2v_v1a \
  --gpu-id 7 \
  --force-local-only \
  --cpu-offload \
  --allow-tf32
```

输出位置：

```text
outputs/phase1_wan22_i2v_v1a/videos/*.mp4
outputs/phase1_wan22_i2v_v1a/metadata/*.json
outputs/phase1_wan22_i2v_v1a/generation_log.jsonl
```

如果显存足够，可以去掉 `--cpu-offload`。如果想先只跑一个视频：

```bash
python code/phase1_wan22_i2v_baseline/run_wan22_i2v.py \
  --manifest code/phase1_wan22_i2v_baseline/manifests/v1a_cam_side_explicit_g.jsonl \
  --model-path ../models/Wan2.2-I2V-A14B-Diffusers \
  --outdir outputs/phase1_wan22_i2v_v1a_smoke \
  --gpu-id 7 \
  --force-local-only \
  --cpu-offload \
  --allow-tf32 \
  --max-jobs 1
```

## 3. Evaluate Generated Videos

```bash
python code/phase1_wan22_i2v_baseline/evaluate_freefall.py \
  --manifest code/phase1_wan22_i2v_baseline/manifests/v1a_cam_side_explicit_g.jsonl \
  --generated-root outputs/phase1_wan22_i2v_v1a \
  --eval-outdir outputs/phase1_wan22_i2v_v1a_eval
```

主要输出：

```text
outputs/phase1_wan22_i2v_v1a_eval/summary.csv
outputs/phase1_wan22_i2v_v1a_eval/phase1_report.json
outputs/phase1_wan22_i2v_v1a_eval/per_video/<job_id>/trajectory.csv
outputs/phase1_wan22_i2v_v1a_eval/per_video/<job_id>/tracked_overlay.mp4
```

重点看 `phase1_report.json`：

- `num_errors` 是否为 0。
- `mean_detection_rate` 是否较高。
- `pearson_g_true_vs_g_hat` / `spearman_g_true_vs_g_hat` 是否为正且较高。
- `summary.csv` 中 `target_param_value` 从小到大时，`g_hat_D_per_s2` 是否整体增大。

## Practical Notes

- `g_hat_D_per_s2` 是球直径单位下的加速度 proxy，不是严格米制重力值。阶段一先看相关性和可跟踪性。
- 如果视频生成正常但 tracker 失败，先打开 `tracked_overlay.mp4` 看颜色分割是否跟住球；可以调 `--color red_orange`、`--min-area`。
- 如果生成视频太慢，可以临时降低 `--num-frames` 或 `--num-inference-steps`。正式记录实验时要把参数写进 metadata。
- 如果模型路径不是 `../models/Wan2.2-I2V-A14B-Diffusers`，改 `--model-path` 即可。

## References

- Wan2.2 I2V Diffusers model card: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
- Wan2.2 official repo: https://github.com/Wan-Video/Wan2.2
