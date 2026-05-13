# V2-B — Repeated Bounce: Video Continuation (Hidden Restitution)

**Level**: L2 | **Experiment**: v2_B | **Script**: `v2/exp_B_ball_repeated_bounce_hidden_restitution.py`
**Hidden**: `e_hidden` (coefficient of restitution, 0 < e < 1, constant every bounce)
**Known**: g = 9.81 m/s²
**Variants**: e0p35 / e0p55 / e0p72 / e0p88
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A ball is dropped from rest at z₀ = 4.2 m and will bounce repeatedly
on a flat floor. Gravity is Earth-normal 9.81 m/s². The **coefficient of
restitution e is unknown** but constant at every bounce.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose full bounce series
is physically consistent with the e implied by the seed.

We will later track z(t) in the generated video, extract peak heights, and
compute e_fit = √(h_{n+1}/h_n) averaged across visible peaks. Your score =
|e_fit − e_true|.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Orange rubber-matte ball, radius 0.24 m, bouncing vertically only.
- Ball-center starts at z₀ = 4.2 m, at rest.
- At floor contact the ball-center is at z = 0.44 m (floor top at z = 0.20 m).
- Gravity constant = 9.81 m/s² (NOT hidden).
- No horizontal drift, no spin, no air drag.
- Lighting, walls, floor geometry static.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video of a ball bouncing vertically.
First 10 frames (0–0.417 s) given; generate frames 11–240 (to t = 10 s) at 24 fps.

Rules you must obey:
- Gravity is 9.81 m/s², constant.
- Every bounce uses the SAME restitution ratio; this ratio must match what
  the opening frames imply (do not guess a popular value).
- Consecutive peak heights form a strict geometric sequence (each peak is
  strictly lower than the one before).
- Motion is purely vertical; no lateral drift, no rotation on the ball.
- As the bounce energy dies out, the ball settles at rest with its center
  at z = 0.44 m and remains still.
- Preserve camera, lighting, ball appearance, floor geometry, resolution.
```

## Easy Version (physics formula given, value still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (e is unknown, same for every bounce):
- Between bounces: free fall, z̈ = −9.81 m/s², purely vertical.
- At each contact (ball-center z = 0.44 m):
    v_after = e · |v_before|,  direction flipped upward.
- Peak height recurrence: h_{n+1} = e² · h_n.
- Starting drop from z₀ = 4.2 m at rest.

Match the e implied by the seed. Preserve camera, lighting, materials.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the ball is already partway through
its first fall and has non-zero downward velocity at that instant.

---

## Evaluation Loop

1. Track ball-center z(t) in generated video.
2. Locate successive local maxima → peak heights h_1, h_2, h_3, ...
3. Compute e_fit = geometric mean of √(h_{n+1}/h_n) across the first ≥3 ratios
   (or fewer if fewer bounces visible).
4. Compare e_fit against `e_hidden` in the variant's `_params.json`.
