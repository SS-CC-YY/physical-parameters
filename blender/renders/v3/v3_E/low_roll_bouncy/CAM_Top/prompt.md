# V3-E — Rolling Ball + Wall: Video Continuation (Hidden μ_r + e)

**Level**: L3 | **Experiment**: v3_E | **Script**: `v3/exp_E_ball_rolling_collision.py`
**Hidden**: `mu_r_hidden` (rolling resistance on floor), `e_hidden` (wall restitution)
**Known**: g = 9.81, θ = 18°, ramp_length = 6.4 m, floor_run = 8.2 m, wall at x = 6.4 m, ball radius 0.25 m, SOLID SPHERE (I = 2/5·mR²)
**Variants**: low_roll_bouncy / base / high_roll_lossy / high_roll_bouncy
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A **solid sphere** rolls without slipping down an 18° ramp,
continues rolling on a flat floor with rolling resistance, strikes a vertical
wall, and bounces back. Two quantities are hidden: the rolling-resistance
coefficient μ_r on the floor, and the wall restitution e.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose floor-phase
deceleration and wall rebound are consistent with (μ_r, e) implied by the seed.

Note: because the ball rolls without slipping on the ramp and g is known, the
ramp acceleration is fixed at a_ramp = (5/7)·g·sin18° ≈ 2.14 m/s² and the
speed at ramp base is v_floor = √(2·a_ramp·6.4) ≈ 5.23 m/s. These are NOT
hidden — they are determined by known quantities.

We will later track the ball, measure floor deceleration → μ_r, and wall
rebound ratio → e.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Ramp inclined at θ = 18° from horizontal. Ramp top at (x = −7.0, z = 3.2 m)
  and extends 6.4 m along the slope.
- Flat floor continues for 8.2 m past the ramp base.
- Vertical wall at x = 6.4 m.
- Orange rubber-matte ball, radius 0.25 m (slightly larger than V1/V2 experiments).
- Ball ROLLS without slipping on both ramp and floor (visible ball rotation
  consistent with translation: angular velocity ω = v/R, surface of the ball
  rotates backward as the ball moves forward).
- Gravity = 9.81 m/s² (NOT hidden).
- Motion strictly in the x–z plane.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video: solid sphere rolls down an 18° ramp, rolls on
the floor, hits a wall, bounces back.
First 10 frames given (0–0.417 s); generate frames 11–240 at 24 fps.

Rules you must obey:
- Ball is a solid sphere rolling without slipping on both ramp and floor.
  Ball rotation MUST be consistent with translation (ω·R = v, visible spin).
- Ramp acceleration is fixed by geometry and gravity (known, not hidden):
  the ball reaches about 5.2 m/s at the ramp base.
- On the floor the ball decelerates uniformly due to rolling resistance; the
  rate of deceleration MUST match what the opening frames imply.
- At the wall (x = 6.4 m), the ball rebounds; horizontal speed is scaled by
  the hidden wall restitution (direction flipped).
- After rebound, the ball decelerates again on the floor with the same
  rolling resistance; eventually comes to rest if time allows.
- Preserve camera, lighting, ramp, floor, wall, ball appearance.
```

## Easy Version (physics formulae given, values still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (μ_r and e both unknown; g = 9.81, θ = 18°, radius R = 0.25 m):
  Ramp (rolling without slip, solid sphere):
    a_ramp = (5/7) · g · sin18° ≈ 2.14 m/s²       (known, not hidden)
    v at ramp base: v_floor = √(2 · a_ramp · 6.4) ≈ 5.23 m/s   (known)
  Floor:
    a_floor = μ_r · g       (deceleration)
    Angular velocity always satisfies ω = v/R (pure rolling).
  Wall (x = 6.4 m):
    v_after = e · |v_before|,  direction flipped.
  Post-wall: same μ_r on floor until stop.

Match (μ_r, e) implied by seed. Preserve camera, lighting, geometry.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the ball is rolling downhill with
non-zero translational + angular velocity.

---

## Evaluation Loop

1. Track ball-center (x, z) in generated video.
2. Identify the ramp → floor transition; measure v_floor from x-velocity at the
   first few floor frames (cross-check against expected 5.23 m/s).
3. On pre-wall floor segment, fit uniform deceleration a_floor → μ_r_fit = a_floor / 9.81.
4. At wall bounce, measure v_before and v_after → e_fit = |v_after|/|v_before|.
5. Compare (μ_r_fit, e_fit) against `mu_r_hidden`, `e_hidden` in `_params.json`.
