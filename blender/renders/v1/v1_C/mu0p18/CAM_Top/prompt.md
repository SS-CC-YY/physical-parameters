# V1-C — Horizontal Slide: Video Continuation (Hidden Friction)

**Level**: L1 | **Experiment**: v1_C | **Script**: `v1/exp_C_ball_slide_friction.py`
**Hidden**: `mu_hidden` (kinetic friction coefficient, dimensionless)
**Variants**: mu0p04 / mu0p10 / mu0p18 / mu0p30
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of an 8 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of an 8-second
physics simulation. A ball enters the scene already moving horizontally
to the right on a flat uniform floor, and decelerates because of
**an unknown kinetic friction μ between the ball and the floor**. Gravity is
9.81 m/s². The ball may or may not come to a stop within the 8-second window,
depending on μ.

**Do not output a number.** Your job is to **generate the remaining 7.583 s
(frames 11–192)**, producing an 8 s video whose deceleration is consistent with
the μ implied by the opening frames.

We will later track the ball-center x(t) in your generated video, fit it
against uniformly-decelerated motion, and back-compute μ_fit = a_fit / g.
Your score = |μ_fit − μ_true| / μ_true.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Single orange rubber-matte ball, radius ≈ 0.24 m, sliding along a flat gray floor.
- Ball-center height stays constant at z = 0.44 m (ball is in rolling contact with the floor).
- Motion is purely horizontal (+x direction); no bouncing, no vertical component.
- Gravity is 9.81 m/s² (NOT the hidden variable).
- Floor is uniform throughout — same μ everywhere.

---

## Hard Version (no physics formula given)

```
Continue this 8-second video of a ball sliding horizontally on a flat floor.
The first 10 frames (0–0.417 s) are given; generate frames 11–192 (to t = 8 s).

Rules you must obey:
- Ball decelerates uniformly at a constant rate dictated by the hidden friction.
  The deceleration you use MUST match what the opening frames imply.
- If the ball stops before t = 8 s, it remains perfectly still at its stop
  position for the remainder of the video. It does NOT start moving again.
- If the deceleration implied by the seed is so mild that the ball is still
  moving at t = 8 s, continue with the linearly decreasing speed — do not
  force it to stop.
- Purely horizontal motion; ball-center z stays at 0.44 m.
- Preserve camera, lighting, ball appearance, floor, resolution.
```

## Easy Version (physics formula given, value still hidden)

```
Continue this 8-second video of a ball sliding on a flat floor at 24 fps.
The first 10 frames (0–0.417 s) are given; generate frames 11–192.

Physics to obey (μ is unknown, constant, uniform across the floor):
- Constant deceleration a = μ · g, with g = 9.81 m/s².
- x(t) = x(t₀) + v(t₀)·(t−t₀) − ½·a·(t−t₀)²   while v > 0.
- Ball stops when velocity hits zero and stays at rest afterward.
- Motion is purely along +x; z stays at 0.44 m.

Infer the initial velocity and the deceleration from the opening frames; both
are implicit. Preserve camera, lighting, materials.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 8 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the ball is already in motion
(it should not start from rest).

---

## Evaluation Loop

1. Track ball-center x(t) in the generated video.
2. Case A — ball stops in-video: locate t_stop (first frame where velocity ≈ 0).
   Fit v₀ from the pre-stop slope; μ_fit = v₀ / (g · t_stop).
3. Case B — ball still moving at t = 8 s: fit a straight line to v(t)
   (numerical derivative of x); μ_fit = −(dv/dt) / g.
4. Compare μ_fit against `mu_hidden` in the variant's `_params.json`.
