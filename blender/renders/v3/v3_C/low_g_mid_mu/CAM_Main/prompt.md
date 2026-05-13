# V3-C — Incline + Floor + Wall: Video Continuation (Hidden g + μ + e)

**Level**: L3 | **Experiment**: v3_C | **Script**: `v3/exp_C_block_incline_floor_friction.py`
**Hidden**: `g_hidden` (m/s²), `mu_hidden` (friction), `e_hidden` (wall restitution)
**Known**: θ = 20°, ramp_length = 8.6 m, floor_run = 7.0 m, wall at x = 5.9 m
**Variants**: low_mu_bouncy / base / high_mu_lossy / low_g_mid_mu
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A rectangular block is released from rest at the top of a 20° ramp,
slides down, crosses a flat floor, hits a vertical wall, and rebounds backward.
Three quantities are hidden: gravitational acceleration g, kinetic friction μ
(same on ramp and floor), and wall restitution e.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose ramp acceleration,
floor deceleration, and wall rebound speed are all consistent with the
(g, μ, e) implied by the seed.

We will later track the block centroid, segment by phase (ramp / floor / post-wall),
fit accelerations and the rebound velocity ratio, and jointly recover (g, μ, e).

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Ramp inclined at θ = 20° from horizontal. Ramp top surface starts at
  (x = −6.0, z = 2.9 m) and extends 8.6 m along the slope toward +x, meeting
  the floor at the ramp base.
- Flat floor continues for 7.0 m beyond the ramp base.
- Vertical wall at x = 5.9 m (visible as a colored block).
- Sliding body: rectangular block with half-extents (0.34, —, 0.24).
- Block slides along ramp surface, transitions smoothly to the floor, then
  collides with the wall once (head-on).
- Block does not leave the ramp surface in mid-air; motion is strictly in x–z.
- Gravity constant, same μ on ramp and floor, e applies only to the wall collision.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video. Block released at top of 20° ramp, slides down,
crosses floor, hits wall, rebounds. First 10 frames given (0–0.417 s);
generate frames 11–240 at 24 fps.

Rules you must obey:
- Gravity constant across the full video. Its value must match what the seed
  implies (do NOT assume 9.81 — some variants use low_g).
- On the ramp: block accelerates with a constant acceleration dictated by
  (g, μ, 20°). Rate MUST match what the seed implies.
- At ramp base → floor: motion continues smoothly (no hop, no jolt).
- On the floor: block decelerates uniformly (deceleration = μ·g).
- At the wall (x = 5.9): block bounces once, with horizontal speed scaled by
  the hidden e (direction flipped); no vertical launch.
- After bouncing back, block decelerates on the floor again with the same μ,
  and eventually stops.
- If it stops before t = 10 s it stays perfectly still for the rest of the video.
- Preserve ramp, floor, wall geometry, block size, camera, lighting.
```

## Easy Version (physics formulae given, values still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (g, μ, e all unknown):
  On ramp:  a_ramp  = g · (sin20° − μ·cos20°) = g · (0.342 − 0.940·μ)
  On floor: a_floor = μ · g     (deceleration)
  At wall (x = 5.9 m):  v_after = e · |v_before|,  direction flipped
  Then decelerate again on floor with the same μ·g until the block stops.

Match (g, μ, e) implied by the seed. Preserve camera, lighting, geometry.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the block already has non-zero
downhill velocity and has not yet reached the ramp base.

---

## Evaluation Loop

1. Track block centroid (x, z) in generated video.
2. Split trajectory into ramp phase, pre-wall floor phase, wall-bounce event,
   post-wall floor phase.
3. Fit a_ramp from the ramp segment; fit a_floor from the pre-wall floor segment
   and confirm on the post-wall segment.
4. Compute v_before and v_after at the wall from the slope of x(t) immediately
   before and after the bounce event.
5. Solve jointly:
      μ_fit · g_fit = a_floor
      g_fit · (0.342 − 0.940·μ_fit) = a_ramp
      e_fit = |v_after| / |v_before|
   → two equations / two unknowns gives (g_fit, μ_fit); e_fit directly from ratio.
6. Compare against `g_hidden`, `mu_hidden`, `e_hidden` in `_params.json`.
