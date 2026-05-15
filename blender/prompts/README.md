# Prompts — Index

## 评测范式（新）

> 当前工作区的统一进度口径见 `../阶段文档/总览/当前进度对齐与汇报摘要.md`。

**视频续写 → 轨迹拟合 → 参数反推** 而非"让模型直接算参数值"。

1. 用 `extract_seed.py` 从每段 8 秒渲染视频抽出前 10 帧（0.417 s 种子）。
2. 把 `seed_10frames.mp4`（或 `frame_10.png`）+ 对应提示词送给视频生成/I2V 模型，请它续写到 8 秒。
3. 对生成的视频做物体追踪拿到轨迹。
4. 用该实验对应的物理模型拟合轨迹，反推隐藏参数。
5. 反推值 vs `_params.json` 里的真值 → 得到模型的"物理理解分数"。

每个提示词文件包含：
- **Hard 版本**（无公式，只给"物理规则约束"）
- **Easy 版本**（给出公式，隐藏参数值必须由种子推断）
- **Scene Description**（视频生成模型的一致性条件）
- **Evaluation Loop**（如何从生成视频反推参数）

## 对照索引

| 提示词文件 | 对应实验 | Level | 隐藏参数 | 对应渲染 ID |
|---|---|---|---|---|
| `v1_A_freefall_gravity.md` | V1-A 自由落体 | L1 | g | v1_A_g2 / g4p9 / g9p81 / g14p7 |
| `v1_B_bounce_restitution.md` | V1-B 弹跳 | L1 | e | v1_B_e0p45 / e0p65 / e0p82 / e0p95 |
| `v1_C_slide_friction.md` | V1-C 水平滑行 | L1 | μ | v1_C_mu0p04 / mu0p10 / mu0p18 / mu0p30 |
| `v1_D_pendulum_damping.md` | V1-D 阻尼摆 | L1 | γ | v1_D_gamma0p04 / gamma0p10 / gamma0p18 / gamma0p30 |
| `v2_A_projectile_gravity.md` | V2-A 抛射+反弹 | L2 | g | v2_A_g4p9 / g7p5 / g9p81 / g14p7 |
| `v2_B_repeated_bounce.md` | V2-B 连续弹跳 | L2 | e | v2_B_e0p35 / e0p55 / e0p72 / e0p88 |
| `v2_C_incline_friction.md` | V2-C 斜面+地面 | L2 | μ | v2_C_mu0p12 / mu0p18 / mu0p24 / mu0p32 |
| `v2_D_long_pendulum.md` | V2-D 大角度非线性阻尼摆 | L2 | γ | v2_D_gamma0p04 / gamma0p10 / gamma0p18 / gamma0p28 |
| `v2_E_tilted_bounce.md` | V2-E 倾斜地面连续弹跳 | L2 | e | v2_E_e0p35 / e0p55 / e0p72 / e0p88 |
| `v2_F_forced_pendulum.md` | V2-F 受迫阻尼摆 | L2 | γ | v2_F_gamma0p04 / gamma0p10 / gamma0p18 / gamma0p28 |
| `v3_A_projectile_drag_bounce.md` | V3-A 抛射+阻力+反弹 | L3 | g, c, e | v3_A_earth_low_drag / earth_mid_drag / low_g_bouncy / high_g_lossy |
| `v3_B_spring_damper.md` | V3-B 弹簧-阻尼振动 | L3 | ω₀, γ | v3_B_slow_low_damp / base / fast_low_damp / fast_high_damp |
| `v3_C_incline_wall.md` | V3-C 斜面+地面+碰墙 | L3 | g, μ, e | v3_C_low_mu_bouncy / base / high_mu_lossy / low_g_mid_mu |
| `v3_D_em_pendulum.md` | V3-D 局部阻尼摆 | L3 | g, γ_zone | v3_D_low_g_low_zone / earth_low_zone / base / high_g_high_zone |
| `v3_E_rolling_collision.md` | V3-E 滚动+碰墙 | L3 | μ_r, e | v3_E_low_roll_bouncy / base / high_roll_lossy / high_roll_bouncy |
| `v3_F_two_ball_collision.md` | V3-F 双球碰撞 | L3 | e, c | v3_F_elastic_low_drag / base / lossy_low_drag / lossy_high_drag |
| `v3_G_spatial_friction.md` | V3-G 非均匀摩擦 | L3 | μ₁, μ₂ | v3_G_low_to_high / base / high_to_low / uniform_control |

## 使用规则

- 同一实验的所有变体使用**同一份**提示词；当前 canonical 主线统一按 `benchmark_variants.py` 中定义的变体集组织
- 变体之间物理参数不同，但模型不被告知隐藏参数值，提示词内容不变
- 每次测试时，向模型提供对应 **seed（种子视频或末帧图）** + 提示词里的 Hard 或 Easy 版本
- 模型输出**生成的视频**；对该视频做追踪→拟合→反推参数
- 记录结果时用渲染 ID（如 `v1_A_g9p81_CAM_Side`）唯一标识每条测试记录
- V1（4）+ V2（6）+ V3（7）= **全部 17 份提示词**均已按"续写→拟合→反推"范式重写
- 每份提示词文件都包含：Task / Scene Description / Hard / Easy / Model Usage / Evaluation Loop 六段
