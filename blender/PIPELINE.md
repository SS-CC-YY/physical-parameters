# Benchmark Pipeline

## Scene Generation + Ground Truth

Each `blender/v1`, `blender/v2`, and `blender/v3` experiment now exports:

- `baseline...blend`
- `baseline..._trajectory.csv`
- `baseline..._params.json`

The CSV contains per-frame metric-space transforms. The JSON records hidden parameters, known parameters, variant id, frame settings, and all cameras.

Parameter sweeps are defined in `benchmark_variants.py`. To generate commands without running Blender:

```bash
python blender/build_variant_blends.py
```

To actually build variant `.blend` files and GT files:

```bash
python blender/build_variant_blends.py --execute
```

## L4 Cross-View Rendering

Base worlds include three cameras:

- `CAM_Main`
- `CAM_Side`
- `CAM_Top`

`render_all.py` is dry-run by default and prints all render commands:

```bash
python blender/render_all.py --versions v3
```

Render only when needed:

```bash
python blender/render_all.py --versions v3 --execute
```

## Evaluation

Use `code/benchmark_evaluation.py` to compare a predicted trajectory CSV against GT:

```bash
python code/benchmark_evaluation.py \
  --gt-csv blender/v3/baselinev3_G_block_spatial_friction_trajectory.csv \
  --pred-csv outputs/model_G_CAM_Side_trajectory.csv \
  --params-json blender/v3/baselinev3_G_block_spatial_friction_params.json \
  --out-json outputs/model_G_CAM_Side_eval.json
```

Use `code/cross_view_consistency.py` to aggregate per-camera evaluation files:

```bash
python code/cross_view_consistency.py \
  --eval-json outputs/model_G_CAM_Main_eval.json outputs/model_G_CAM_Side_eval.json outputs/model_G_CAM_Top_eval.json \
  --out-json outputs/model_G_l4_consistency.json
```
