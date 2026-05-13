# V2-A — Projectile + Bounce: Video Continuation (Hidden Gravity)

**Level**: L2 | **Experiment**: v2_A | **Script**: `v2/exp_A_ball_projectile_hidden_gravity.py`
**Hidden**: `g_hidden` (gravitational acceleration, m/s²)
**Known**: `e_known = 0.74` (restitution — given to the model)
**Variants**: g4p9 / g7p5 / g9p81 / g14p7
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A ball is launched diagonally and will follow a parabolic arc, land
on the floor, and bounce. The **gravitational acceleration g is unknown** and
constant throughout this world.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose full parabolic +
bounce trajectory is consistent with the gravity implied by the opening frames.

We will later track the ball's (x, z) trajectory and fit against projectile +
restitution motion to back-compute g_fit. Your score = |g_fit − g_true| / g_true.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera, no camera motion.
- Orange rubber-matte ball, radius 0.24 m.
- Ball launch position: x₀ = −6.0 m, z₀ = 1.2 m (ball-center).
- Launch velocity: v_x0 = 4.8 m/s (horizontal, rightward), v_z0 = 7.8 m/s (upward).
- Floor top surface at z = 0.20 m; ball-center at floor contact is z = 0.44 m.
- Restitution at every floor bounce: e = 0.74 (26% of vertical speed lost per bounce).
- Gravity is constant over the full 10 s; no drag, no wind, no rotation physics
  on the ball beyond what visually appears consistent with the arc.
- Lighting and background static.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video of a ball following a parabolic arc and bouncing.
The first 10 frames (0–0.417 s) are given; generate frames 11–240 (to t = 10 s).

Rules you must obey:
- Constant gravity throughout; value must match what the opening frames imply.
- Launch was: x₀ = -6.0 m, z₀ = 1.2 m, v_x0 = 4.8 m/s, v_z0 = 7.8 m/s.
- Horizontal velocity is preserved in flight (no air drag).
- At each floor contact, vertical speed is multiplied by 0.74 (direction flipped
  upward); horizontal speed is unchanged.
- Ball eventually settles with its center at z = 0.44 m when bounces die out.
- Preserve camera, lighting, ball appearance, floor geometry, resolution.
```

## Easy Version (physics formula given, value still hidden)

```
Continue this 10-second video at 24 fps.
First 10 frames given (0–0.417 s); generate frames 11–240.

Physics to obey (g is unknown — infer from the opening frames):
  Between bounces:
    x(t) = x₀ + v_x0·t
    z(t) = z₀ + v_z0·t − ½·g·t²          (z reset at each bounce)
  At each floor contact (z_center = 0.44 m):
    v_z_after = 0.74 · |v_z_before|  (upward)
    v_x unchanged
  Known: x₀ = −6.0, z₀ = 1.2, v_x0 = 4.8, v_z0 = 7.8, e = 0.74.

Match the g implied by the seed. Preserve camera, lighting, materials.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the ball is already in mid-flight
(non-zero horizontal + upward velocity) at that instant.

---

## Evaluation Loop

1. Track ball-center (x, z) in the generated video.
2. Fit the pre-first-bounce segment with z(t) = z₀ + v_z0·t − ½·g·t² to get g_fit_1.
3. Cross-check using the time between first and second bounce: for a bouncing
   projectile, t_gap = 2 · v_z_after / g → another g estimate g_fit_2.
4. Report g_fit (average) and relative error vs `g_hidden` in `_params.json`.
