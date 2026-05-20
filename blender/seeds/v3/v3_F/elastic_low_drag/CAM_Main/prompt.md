# V3-F — Two-Ball Collision + Drag: Video Continuation (Hidden e + c)

**Level**: L3 | **Experiment**: v3_F | **Script**: `v3/exp_F_two_ball_collision_drag.py`
**Hidden**: `e_hidden` (collision restitution), `c_hidden` (post-collision linear drag, s⁻¹)
**Known**: m₁ = m₂ = 1.0 kg, u₁ = 4.2 m/s (orange ball rightward), u₂ = 0 (blue ball stationary), ball radius 0.25 m
**Variants**: elastic_low_drag / base / lossy_low_drag / lossy_high_drag
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. Two equal-mass balls are on a horizontal track: Ball 1 (orange)
moves rightward with u₁ = 4.2 m/s; Ball 2 (blue) is stationary. Ball 1 will
collide head-on with Ball 2 exactly once. After the collision, both balls move
subject to linear air drag F = −c·v. The **collision restitution e and the
drag coefficient c are both unknown**.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose collision outcome
and post-collision deceleration are consistent with the (e, c) implied by the
seed.

We will later track both balls, extract the collision event, compute velocity
split → e, and exponential decay of each ball's velocity → c.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Two balls, each of radius 0.25 m, rolling on a horizontal track at z_center ≈ 0.41 m.
- Ball 1 (orange, MAT_Rubber color) starts at x₁ = −6.6 m with u₁ = 4.2 m/s rightward.
- Ball 2 (blue, MAT_Accent color) starts at x₂ = −0.8 m, at rest.
- Masses equal: m₁ = m₂ = 1.0 kg.
- Collision is a 1-D head-on event; balls separate only along x.
- Both balls are subject to the SAME linear drag (F = −c·v) throughout the
  video — the drag is present from t = 0, not only post-collision. (Note:
  before the collision, Ball 2 is at rest so drag does nothing; Ball 1's
  speed decays slowly from 4.2 m/s during its approach.)
- Motion strictly along x.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video of two-ball collision with post-collision drag.
First 10 frames given (0–0.417 s); generate frames 11–240 at 24 fps.

Rules you must obey:
- Ball 1 (orange) continues rightward from the seed frame, decelerating slowly
  (due to drag) until it collides with Ball 2 (blue).
- Collision is 1-D head-on. Equal masses, so after collision Ball 1's speed
  splits some fraction to Ball 2; the split ratio is governed by the hidden e.
- A perfectly elastic collision (e = 1) would transfer all of Ball 1's speed
  to Ball 2 (Ball 1 stops). Lower e means Ball 1 retains some rightward speed
  and Ball 2 gets less. Match the split implied by the seed.
- After collision, BOTH balls experience the same linear drag: their speeds
  decay exponentially toward zero. The rate of decay must match what the
  seed implies.
- No ball bounces off any wall; they just slow and stop (or nearly so) before
  reaching any boundary of the visible scene.
- Motion strictly along x; no vertical bouncing, no rotation artifacts.
- Preserve camera, lighting, both ball colors and sizes, track geometry.
```

## Easy Version (physics formulae given, values still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (e and c both unknown; equal masses):
  Pre-collision:
    Ball 1: v_1(t) = u₁ · exp(−c·t) = 4.2 · exp(−c·t)
    Ball 2: at rest (drag does nothing on zero velocity).
  Collision (equal mass, 1-D):
    v_1_after = v_1_before · (1 − e) / 2
    v_2_after = v_1_before · (1 + e) / 2
  Post-collision (each ball independent, same drag c):
    v_i(t) = v_i_after · exp(−c · (t − t_collision)),  i = 1, 2

Match (e, c) implied by the seed. Preserve camera, lighting, materials.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model Ball 1 is moving rightward with
non-zero velocity and Ball 2 is stationary.

---

## Evaluation Loop

1. Track both ball centers x_1(t), x_2(t) in generated video.
2. Identify the collision frame t_c (the frame where ball separation stops
   decreasing and Ball 2 begins to move).
3. Compute v_1_before (just before t_c), v_1_after, v_2_after (just after t_c).
4. e_fit = (v_2_after − v_1_after) / v_1_before.
5. Fit each post-collision velocity curve v_i(t) to an exponential:
   v_i(t) = v_i_after · exp(−c · (t − t_c)) → two independent estimates of c;
   average → c_fit.
6. Compare (e_fit, c_fit) against `e_hidden`, `c_hidden` in `_params.json`.
