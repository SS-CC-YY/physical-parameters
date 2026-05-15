# V3-A — Projectile + Drag + Bounce: Video Continuation (Three Hidden Parameters)

**Level**: L3 | **Experiment**: v3_A | **Script**: `v3/exp_A_ball_projectile_drag_bounce.py`
**Hidden**: `g_hidden` (m/s²), `c_hidden` (linear drag, s⁻¹), `e_hidden` (restitution)
**Known**: x₀ = −6.0, z₀ = 1.1, v_x0 = 5.2, v_z0 = 7.8, ball radius = 0.24, floor at z_center = 0.44
**Variants**: earth_low_drag / earth_mid_drag / low_g_bouncy / high_g_lossy
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A ball is launched diagonally under **three unknown quantities**:
gravitational acceleration g, linear air-drag coefficient c, and floor
restitution e. Air drag is linear (F_drag = −c·v) and applies independently to
horizontal and vertical velocity components.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose full flight +
multi-bounce trajectory is consistent with all three hidden parameters as
implied by the seed.

We will later track (x, z) in your generated video, fit the three parameters
jointly, and compute per-parameter relative errors.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Orange rubber-matte ball, radius 0.24 m.
- Launch: x₀ = −6.0 m, z₀ = 1.1 m, v_x0 = 5.2 m/s, v_z0 = 7.8 m/s.
- Floor top at z = 0.20 m; ball-center at floor contact is z = 0.44 m.
- Air drag is LINEAR — not quadratic — and is always present (non-zero c).
- Lighting, walls, camera all static.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video of a ball launched diagonally, subject to air
drag, landing on the floor, and bouncing multiple times.
First 10 frames given (0–0.417 s); generate frames 11–240 at 24 fps.

Rules you must obey:
- Gravity is constant throughout; its value must match what the seed implies.
- Horizontal velocity decays gradually due to air drag (ball slows horizontally
  — it does NOT travel as far as a no-drag parabola would).
- Vertical motion is asymmetric: ascent slower than descent due to drag.
- At each floor contact, vertical speed is scaled by a restitution factor
  (direction flipped upward); horizontal speed is preserved across the bounce
  except for the drag acting between bounces.
- After enough bounces, vertical energy dies out; ball settles (z_center = 0.44 m)
  and continues rolling forward slowed by the drag.
- All three hidden values (g, c, e) must match the opening frames — do not guess.
- Preserve camera, lighting, ball, floor, resolution.
```

## Easy Version (physics formula given, values still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (g, c, e all unknown):
  Between bounces (using ẋ = v_x, ż = v_z):
    v_x(t) = v_x0 · exp(−c·t)                  x(t) = x₀ + (v_x0/c) · (1 − exp(−c·t))
    v_z(t) = (v_z0 + g/c) · exp(−c·t) − g/c    z(t) integrates as usual
  At each floor contact (z_center = 0.44 m):
    v_z_after = e · |v_z_before|  (upward)
    v_x unchanged  (still decaying as exp(−c·t) between bounces, restarting
    the exp clock at each bounce event)
  Launch: x₀ = −6.0, z₀ = 1.1, v_x0 = 5.2, v_z0 = 7.8.

Match the (g, c, e) implied by the seed. Preserve camera, lighting, materials.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the ball is mid-flight with non-zero
horizontal + vertical velocity components.

---

## Evaluation Loop

1. Track (x, z) of ball-center in generated video.
2. Fit the three-parameter projectile-plus-drag model (with bounce events) jointly:
   - Use pre-first-bounce arc to constrain g and c.
   - Use consecutive bounce peak ratio to constrain e (then re-refine g, c with
     the full trajectory).
3. Report (g_fit, c_fit, e_fit) and per-parameter relative errors vs `_params.json`.
