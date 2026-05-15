# I2V Freefall Suite

文件说明：
- `prepare_i2v_anchor.py`：轻量增强初始图，并输出球的初始半径/位置元数据
- `i2v_freefall_prompts.py`：I2V 自由落体 prompt 模板
- `build_i2v_freefall_sweep_manifest.py`：从 sweep config 生成 manifest.jsonl
- `evaluate_i2v_freefall_sweep.py`：批量评测 sweep，重点检查 prompt 物理量与估计 proxy 的线性关联
- `configs/i2v_freefall_sweeps.example.json`：示例 sweep 配置

推荐流程：
1. 准备 anchor 图
2. 生成 manifest
3. 用现有 `generate_videos.py` 跑视频
4. 批量评测 sweep

示例：

```bash
python prepare_i2v_anchor.py   --image /mnt/data/i2v_v1_freefall.png   --outdir anchor_prep   --color red

python build_i2v_freefall_sweep_manifest.py   --config configs/i2v_freefall_sweeps.example.json   --output manifests/i2v_freefall_sweep.jsonl   --model wan2.2_ti2v_5b   --prompt-styles structured_raw structured_detailed   --seed-base 42   --num-seeds 3

python generate_videos.py   --manifest manifests/i2v_freefall_sweep.jsonl   --outdir outputs/i2v_freefall_sweep   --gpus 0,1,2,3   --local-model-root /mnt/data/scy   --allow-tf32   --cpu-offload   --force-local-only

python evaluate_i2v_freefall_sweep.py   --manifest manifests/i2v_freefall_sweep.jsonl   --generated-root outputs/i2v_freefall_sweep   --eval-outdir outputs/i2v_freefall_eval
```

说明：
- 对同一初始图，`g` 和 `v0` 的 sweep 最适合做线性关联测试
- `h0` 也可以测，但同一 anchor image 会让它和图像几何初始位置耦合，解释要更谨慎
- 评测默认用“球直径 D”为长度单位，得到 `g_hat_D`, `v0_hat_D`, `h0_hat_D`
