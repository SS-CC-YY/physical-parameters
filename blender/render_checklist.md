# Benchmark Render Checklist

## 快速启动（三条命令跑完全部流程）

> 当前工作区的统一进度口径见 `阶段文档/总览/当前进度对齐与汇报摘要.md`。
> 说明：`benchmark_variants.py` 当前定义的是 **68 个 canonical 变体**；本工作区现已按该主集完成收束，`variants/` 目录按 **68 个变体目录** 维护。

```bash
# 1. 生成当前 benchmark_variants.py 中定义的全部 canonical 变体 .blend 文件 + GT CSV/JSON（当前为 68 个）
python build_variant_blends.py --blender "blender" --execute

# 2. 渲染视频
#   当前主线以 variant 目录下的 CAM_Side 为主；
#   CAM_Top 已调整为“保留名称，但使用类似 CAM_Main 的 y 轴镜像斜视角”
#   历史严格俯视版 CAM_Top 产物视为旧资产，不再用于正式 L4 对照
python render_all.py --blender "blender" --root variants --execute

# 3. 查看输出
#   .blend / CSV / JSON  →  variants/{version}/{exp_id}/{variant_id}/
#   渲染视频             →  renders/{version}/{exp_id}/{variant_id}/{camera}/

# 4. 批量评测
#   将模型输出的视频先转换为标准 trajectory.csv，再批量汇总
python ..\code\batch_benchmark_eval.py --pred-root <predictions> --out-root ..\results\benchmark_eval
```

> **注意**：`CAM_Top` 并非完全缺失。当前工作区中：
> 1. `V3` 的历史 baseline 目录已存在 `CAM_Main / CAM_Side / CAM_Top`
> 2. `V2` 已有部分实验补齐了三视角
> 3. 新版 variant 主线目前最完整的是 `CAM_Side`
>
> 但从 `2026-05-08` 起，`CAM_Top` 的推荐定义不再是严格俯视，而是作为 `CAM_Main` 的镜像斜视角使用，以避免“只有尺度变化、缺少有效轨迹”的退化视图。

---

## 目录结构说明

```
blender/
├── build_variant_blends.py   ← 生成 .blend + GT 文件
├── render_all.py             ← 渲染视频
├── benchmark_io.py           ← GT 导出工具（被 exp_*.py 调用）
├── benchmark_variants.py     ← 所有参数变体定义
├── render_checklist.md       ← 本文件
├── prompts/                  ← 提示词文件（一实验一文件）
│   ├── README.md
│   ├── v1_A_freefall_gravity.md
│   ├── ...
├── variants/                 ← build 后生成
│   ├── v1/v1_A/g2/
│   │   ├── baselinev1_A_ball_freefall_gravity_g2.blend
│   │   ├── baselinev1_A_ball_freefall_gravity_g2_trajectory.csv
│   │   └── baselinev1_A_ball_freefall_gravity_g2_params.json
│   └── ...
└── renders/                  ← render 后生成
    └── v1/v1_A/g2/CAM_Side/
        └── baselinev1_A_ball_freefall_gravity_g2_CAM_Side_0001-0240.mp4
```

---

## 完整变体清单（当前工作区按 canonical 主线维护 68 个变体目录）

### V1 — Minimal Parameter Probes（Level 1）

| 渲染 ID | 实验 | 变体 ID | 隐藏参数 | 已知参数 | 提示词文件 |
|---|---|---|---|---|---|
| `v1_A_g2` | V1-A 自由落体 | `g2` | g=2.0 m/s² | z₀=4.2, radius=0.24, z_floor=0.44 | `prompts/v1_A_freefall_gravity.md` |
| `v1_A_g4p9` | V1-A 自由落体 | `g4p9` | g=4.9 m/s² | 同上 | `prompts/v1_A_freefall_gravity.md` |
| `v1_A_g9p81` | V1-A 自由落体 | `g9p81` | g=9.81 m/s² | 同上 | `prompts/v1_A_freefall_gravity.md` |
| `v1_A_g14p7` | V1-A 自由落体 | `g14p7` | g=14.7 m/s² | 同上 | `prompts/v1_A_freefall_gravity.md` |
| `v1_B_e0p45` | V1-B 弹跳 | `e0p45` | e=0.45 | g=9.81, z₀=4.2, radius=0.24 | `prompts/v1_B_bounce_restitution.md` |
| `v1_B_e0p65` | V1-B 弹跳 | `e0p65` | e=0.65 | 同上 | `prompts/v1_B_bounce_restitution.md` |
| `v1_B_e0p82` | V1-B 弹跳 | `e0p82` | e=0.82 | 同上 | `prompts/v1_B_bounce_restitution.md` |
| `v1_B_e0p95` | V1-B 弹跳 | `e0p95` | e=0.95 | 同上 | `prompts/v1_B_bounce_restitution.md` |
| `v1_C_mu0p04` | V1-C 滑行 | `mu0p04` | μ=0.04 | g=9.81 | `prompts/v1_C_slide_friction.md` |
| `v1_C_mu0p10` | V1-C 滑行 | `mu0p10` | μ=0.10 | 同上 | `prompts/v1_C_slide_friction.md` |
| `v1_C_mu0p18` | V1-C 滑行 | `mu0p18` | μ=0.18 | 同上 | `prompts/v1_C_slide_friction.md` |
| `v1_C_mu0p30` | V1-C 滑行 | `mu0p30` | μ=0.30 | 同上 | `prompts/v1_C_slide_friction.md` |
| `v1_D_gamma0p04` | V1-D 阻尼摆 | `gamma0p04` | γ=0.04 s⁻¹ | g=9.81, L≈2.3m, θ₀≈30° | `prompts/v1_D_pendulum_damping.md` |
| `v1_D_gamma0p10` | V1-D 阻尼摆 | `gamma0p10` | γ=0.10 s⁻¹ | 同上 | `prompts/v1_D_pendulum_damping.md` |
| `v1_D_gamma0p18` | V1-D 阻尼摆 | `gamma0p18` | γ=0.18 s⁻¹ | 同上 | `prompts/v1_D_pendulum_damping.md` |
| `v1_D_gamma0p30` | V1-D 阻尼摆 | `gamma0p30` | γ=0.30 s⁻¹ | 同上 | `prompts/v1_D_pendulum_damping.md` |

### V2 — Extended Single-Parameter Dynamics（Level 2）

| 渲染 ID | 实验 | 变体 ID | 隐藏参数 | 已知参数 | 提示词文件 |
|---|---|---|---|---|---|
| `v2_A_g4p9` | V2-A 抛射+反弹 | `g4p9` | g=4.9 m/s² | e_known=0.74, vx₀=4.8, vz₀=7.8, x₀=-6, z₀=1.2 | `prompts/v2_A_projectile_gravity.md` |
| `v2_A_g7p5` | V2-A 抛射+反弹 | `g7p5` | g=7.5 m/s² | 同上 | `prompts/v2_A_projectile_gravity.md` |
| `v2_A_g9p81` | V2-A 抛射+反弹 | `g9p81` | g=9.81 m/s² | 同上 | `prompts/v2_A_projectile_gravity.md` |
| `v2_A_g14p7` | V2-A 抛射+反弹 | `g14p7` | g=14.7 m/s² | 同上 | `prompts/v2_A_projectile_gravity.md` |
| `v2_B_e0p35` | V2-B 连续弹跳 | `e0p35` | e=0.35 | g=9.81 | `prompts/v2_B_repeated_bounce.md` |
| `v2_B_e0p55` | V2-B 连续弹跳 | `e0p55` | e=0.55 | 同上 | `prompts/v2_B_repeated_bounce.md` |
| `v2_B_e0p72` | V2-B 连续弹跳 | `e0p72` | e=0.72 | 同上 | `prompts/v2_B_repeated_bounce.md` |
| `v2_B_e0p88` | V2-B 连续弹跳 | `e0p88` | e=0.88 | 同上 | `prompts/v2_B_repeated_bounce.md` |
| `v2_C_mu0p12` | V2-C 斜面+地面 | `mu0p12` | μ=0.12 | 同上 | `prompts/v2_C_incline_friction.md` |
| `v2_C_mu0p18` | V2-C 斜面+地面 | `mu0p18` | μ=0.18 | 同上 | `prompts/v2_C_incline_friction.md` |
| `v2_C_mu0p24` | V2-C 斜面+地面 | `mu0p24` | μ=0.24 | 同上 | `prompts/v2_C_incline_friction.md` |
| `v2_C_mu0p32` | V2-C 斜面+地面 | `mu0p32` | μ=0.32 | 同上 | `prompts/v2_C_incline_friction.md` |
| `v2_D_gamma0p04` | V2-D 长阻尼摆 | `gamma0p04` | γ=0.04 s⁻¹ | g=9.81, L≈2.3m | `prompts/v2_D_long_pendulum.md` |
| `v2_D_gamma0p10` | V2-D 长阻尼摆 | `gamma0p10` | γ=0.10 s⁻¹ | 同上 | `prompts/v2_D_long_pendulum.md` |
| `v2_D_gamma0p18` | V2-D 长阻尼摆 | `gamma0p18` | γ=0.18 s⁻¹ | 同上 | `prompts/v2_D_long_pendulum.md` |
| `v2_D_gamma0p28` | V2-D 长阻尼摆 | `gamma0p28` | γ=0.28 s⁻¹ | 同上 | `prompts/v2_D_long_pendulum.md` |

### V3 — Coupled Multi-Physics Probes（Level 3）

| 渲染 ID | 实验 | 变体 ID | 隐藏参数 | 已知参数 | 提示词文件 |
|---|---|---|---|---|---|
| `v3_A_earth_low_drag` | V3-A 抛射+阻力+反弹 | `earth_low_drag` | g=9.81, c=0.18, e=0.84 | vx₀=5.2, vz₀=7.8, x₀=-6, z₀=1.1 | `prompts/v3_A_projectile_drag_bounce.md` |
| `v3_A_earth_mid_drag` | V3-A | `earth_mid_drag` | g=9.81, c=0.42, e=0.78 | 同上 | `prompts/v3_A_projectile_drag_bounce.md` |
| `v3_A_low_g_bouncy` | V3-A | `low_g_bouncy` | g=4.9, c=0.28, e=0.88 | 同上 | `prompts/v3_A_projectile_drag_bounce.md` |
| `v3_A_high_g_lossy` | V3-A | `high_g_lossy` | g=14.7, c=0.55, e=0.62 | 同上 | `prompts/v3_A_projectile_drag_bounce.md` |
| `v3_B_slow_low_damp` | V3-B 弹簧-阻尼 | `slow_low_damp` | ω₀=1.75, γ=0.08 | x_eq=-1.8, A₀=1.65 | `prompts/v3_B_spring_damper.md` |
| `v3_B_base` | V3-B | `base` | ω₀=2.35, γ=0.20 | 同上 | `prompts/v3_B_spring_damper.md` |
| `v3_B_fast_low_damp` | V3-B | `fast_low_damp` | ω₀=3.05, γ=0.12 | 同上 | `prompts/v3_B_spring_damper.md` |
| `v3_B_fast_high_damp` | V3-B | `fast_high_damp` | ω₀=3.05, γ=0.36 | 同上 | `prompts/v3_B_spring_damper.md` |
| `v3_C_low_mu_bouncy` | V3-C 斜面+地面+碰墙 | `low_mu_bouncy` | g=9.81, μ=0.06, e=0.86 | θ=20° | `prompts/v3_C_incline_wall.md` |
| `v3_C_base` | V3-C | `base` | g=9.81, μ=0.16, e=0.72 | 同上 | `prompts/v3_C_incline_wall.md` |
| `v3_C_high_mu_lossy` | V3-C | `high_mu_lossy` | g=9.81, μ=0.28, e=0.54 | 同上 | `prompts/v3_C_incline_wall.md` |
| `v3_C_low_g_mid_mu` | V3-C | `low_g_mid_mu` | g=4.9, μ=0.16, e=0.72 | 同上 | `prompts/v3_C_incline_wall.md` |
| `v3_D_low_g_low_zone` | V3-D 局部阻尼摆 | `low_g_low_zone` | g=4.9, γ_zone=0.28 | L=2.06m, θ₀=24°, 阻尼区±7.5° | `prompts/v3_D_em_pendulum.md` |
| `v3_D_earth_low_zone` | V3-D | `earth_low_zone` | g=9.81, γ_zone=0.28 | 同上 | `prompts/v3_D_em_pendulum.md` |
| `v3_D_base` | V3-D | `base` | g=9.81, γ_zone=0.58 | 同上 | `prompts/v3_D_em_pendulum.md` |
| `v3_D_high_g_high_zone` | V3-D | `high_g_high_zone` | g=14.7, γ_zone=0.82 | 同上 | `prompts/v3_D_em_pendulum.md` |
| `v3_E_low_roll_bouncy` | V3-E 滚动+碰墙 | `low_roll_bouncy` | μ_r=0.018, e=0.86 | g=9.81, θ=18°, 实心球 | `prompts/v3_E_rolling_collision.md` |
| `v3_E_base` | V3-E | `base` | μ_r=0.035, e=0.74 | 同上 | `prompts/v3_E_rolling_collision.md` |
| `v3_E_high_roll_lossy` | V3-E | `high_roll_lossy` | μ_r=0.070, e=0.56 | 同上 | `prompts/v3_E_rolling_collision.md` |
| `v3_E_high_roll_bouncy` | V3-E | `high_roll_bouncy` | μ_r=0.070, e=0.86 | 同上 | `prompts/v3_E_rolling_collision.md` |
| `v3_F_elastic_low_drag` | V3-F 双球碰撞 | `elastic_low_drag` | e=0.92, c=0.16 | m₁=m₂=1, u₁=4.2 m/s, u₂=0 | `prompts/v3_F_two_ball_collision.md` |
| `v3_F_base` | V3-F | `base` | e=0.74, c=0.34 | 同上 | `prompts/v3_F_two_ball_collision.md` |
| `v3_F_lossy_low_drag` | V3-F | `lossy_low_drag` | e=0.52, c=0.16 | 同上 | `prompts/v3_F_two_ball_collision.md` |
| `v3_F_lossy_high_drag` | V3-F | `lossy_high_drag` | e=0.52, c=0.58 | 同上 | `prompts/v3_F_two_ball_collision.md` |
| `v3_G_low_to_high` | V3-G 非均匀摩擦 | `low_to_high` | μ₁=0.05, μ₂=0.30 | g=9.81, v₀=4.6 m/s, 边界@x=-0.2m | `prompts/v3_G_spatial_friction.md` |
| `v3_G_base` | V3-G | `base` | μ₁=0.09, μ₂=0.26 | 同上 | `prompts/v3_G_spatial_friction.md` |
| `v3_G_high_to_low` | V3-G | `high_to_low` | μ₁=0.24, μ₂=0.08 | 同上 | `prompts/v3_G_spatial_friction.md` |
| `v3_G_uniform_control` | V3-G（控制组） | `uniform_control` | μ₁=0.16, μ₂=0.16 | 同上 | `prompts/v3_G_spatial_friction.md` |

---

## 渲染输出路径对照

若按当前 `benchmark_variants.py` 的 canonical 主集补齐三视角，则总视频数为 `68 × 3 = 204`。
历史 baseline 目录与旧版严格俯视 `CAM_Top` 已从主线资产中清理。路径规律如下：

```
variants/{version}/{exp_id}/{variant_id}/{blend_stem}_{variant_id}.blend
renders/{version}/{exp_id}/{variant_id}/CAM_Side/{blend_stem}_{variant_id}_CAM_Side_0001-0240.mp4
renders/{version}/{exp_id}/{variant_id}/CAM_Main/{blend_stem}_{variant_id}_CAM_Main_0001-0240.mp4
```

示例（V1-A g2 变体）：
```
variants/v1/v1_A/g2/baselinev1_A_ball_freefall_gravity_g2.blend
variants/v1/v1_A/g2/baselinev1_A_ball_freefall_gravity_g2_trajectory.csv
variants/v1/v1_A/g2/baselinev1_A_ball_freefall_gravity_g2_params.json
renders/v1/v1_A/g2/CAM_Side/baselinev1_A_ball_freefall_gravity_g2_CAM_Side_0001-0240.mp4
renders/v1/v1_A/g2/CAM_Main/baselinev1_A_ball_freefall_gravity_g2_CAM_Main_0001-0240.mp4
```

---

## 建议渲染顺序

| 阶段 | 内容 | 视频数 | 用途 |
|---|---|---|---|
| Pilot | V1 全部 × CAM_Side | 16 | 验证完整流程 |
| Phase 1 | V1+V2 全部 × CAM_Side | 42 | L1+L2 基准测试 |
| Phase 2 | V3 全部 × CAM_Side | 28 | L3 多参数测试 |
| Phase 3 | 全部 × CAM_Main | 68 | L4 跨视角测试 |
| Phase 4 | 全部 × CAM_Top（镜像斜视角） | 68 | L4 跨视角测试 |
| **Total** | | **204** | 以当前 canonical `68` 个变体目录计的全量三视角渲染上限 |
