# V1-B — Vertical Bounce: Video Continuation (Hidden Restitution)

**Level**: L1 | **Experiment**: v1_B | **Script**: `v1/exp_B_ball_bounce_restitution.py`
**Hidden**: `e_hidden` (coefficient of restitution, 0 < e < 1, constant every bounce)
**Variants**: e0p45 / e0p65 / e0p82 / e0p95
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of an 8 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of an 8-second physics
simulation. A ball is released from rest and bounces repeatedly on a flat floor.
Gravity is Earth-normal (9.81 m/s²). The **coefficient of restitution e is
unknown** but is the same at every bounce.

**Do not output a number.** Your job is to **generate the remaining 7.583 s
(frames 11–192)**, producing a complete 8 s video whose bounce dynamics are
physically consistent with the restitution implied by the opening frames
(if the opening frames haven't yet shown a bounce, infer e from the pre-bounce
motion plus the fact that bounces are about to begin).

We will later track peak heights in your generated video and compute
e_fit = √(h_{n+1}/h_n), averaged over the visible peaks. Your score =
|e_fit − e_true|.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Single orange rubber-matte ball, radius ≈ 0.24 m, bouncing vertically only.
- Ball-center starts at z₀ = 3.5 m; at rest on the floor the ball-center is at z = 0.44 m.
- Floor slab top at z = 0.20 m. Gray matte floor, white walls, soft static lighting.
- Gravity is constant = 9.81 m/s² (this is NOT the hidden variable).
- The ball only moves in z — no horizontal drift, no spin.

---

## Hard Version (no physics formula given)

```
Continue this 8-second video of a single ball bouncing vertically.
The first 10 frames (0–0.417 s) are given; generate frames 11–192 (up to t = 8 s).

Rules you must obey:
- Gravity is 9.81 m/s² throughout (Earth-normal, constant).
- Every bounce has the SAME restitution ratio (hidden value — match what the
  opening frames imply).
- Consecutive peak heights must form a strict geometric sequence (each peak
  lower than the previous).
- Purely vertical motion: no sideways drift, no rotation on the ball.
- Eventually the bounce energy goes to zero and the ball rests at z_center = 0.44 m
  for the remainder of the video.
- Preserve camera, lighting, ball appearance, floor geometry, resolution.
```

## Easy Version (physics formula given, value still hidden)

```
Continue this 8-second video of a single ball bouncing vertically at 24 fps.
The first 10 frames (0–0.417 s) are given; generate frames 11–192.

Physics to obey (e is unknown, same for every bounce):
- Between bounces, free fall under g = 9.81 m/s².
- At each floor contact (ball-center reaches z = 0.44 m):
     v_after = e · |v_before|,  direction flipped upward.
- Peak height recurrence: h_{n+1} = e² · h_n.
- Motion is purely vertical.

Match the e implied by the opening frames. Preserve camera, lighting, materials.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 8 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the ball is already partway through
its first fall and bounces are imminent.

---

## Evaluation Loop

1. Track the ball-center z(t) in the generated video.
2. Find local maxima → peak heights h_1, h_2, h_3, ...
3. Compute e_fit = geometric mean of √(h_{n+1}/h_n) across the first ≥2 ratios.
4. Compare e_fit against `e_hidden` in the variant's `_params.json`.
