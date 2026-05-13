# V2-C — Incline + Floor Slide: Video Continuation (Hidden Friction)

**Level**: L2 | **Experiment**: v2_C | **Script**: `v2/exp_C_block_incline_floor_hidden_friction.py`
**Hidden**: `mu_hidden` (kinetic friction coefficient, dimensionless — SAME on ramp and floor)
**Known**: g = 9.81 m/s², θ = 20° incline
**Variants**: mu0p05 / mu0p12 / mu0p20 / mu0p32
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A rectangular block is released from rest at the top of a 20° ramp,
slides down, crosses a flat floor, and eventually stops. The **kinetic friction
coefficient μ is unknown**, and is the **same on the ramp and the floor**.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose two-phase deceleration
is consistent with the μ implied by the seed.

We will later track the block centroid trajectory, segment it into ramp-phase
and floor-phase, fit accelerations for each, and back-compute μ_fit. Your
score = |μ_fit − μ_true| / μ_true.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Ramp inclined at **θ = 20°** from horizontal. Ramp top surface starts at
  (x = −6.0 m, z = 2.9 m) and extends 8 m down along the slope toward positive x.
  Ramp base connects smoothly to a flat floor of length 7 m.
- Sliding body: a rectangular block with half-extents (0.34, —, 0.24) in (x, z);
  block centroid slides along the ramp surface and then along the floor.
- g = 9.81 m/s² (NOT hidden).
- Same μ applies on the ramp and on the floor — no discontinuity at transition.
- Motion strictly in the x–z plane; no tumbling or tipping.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video: block sliding down a 20° ramp, crossing a floor,
eventually stopping. First 10 frames given (0–0.417 s); generate frames 11–240.

Rules you must obey:
- Block accelerates uniformly down the ramp at a rate dictated by the hidden μ
  (the rate MUST match what the opening frames imply — do not guess).
- At the ramp base, velocity continues smoothly along the horizontal floor;
  no hop, no bounce.
- On the floor, block decelerates uniformly and eventually comes to rest.
- Once stopped, block remains perfectly still for the rest of the video.
- For low μ, the block may still be moving at t = 10 s; in that case continue
  the linear deceleration to the end of the video without forcing a stop.
- Preserve ramp geometry, floor length, block size, camera, lighting, colors.
```

## Easy Version (physics formula given, value still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (μ is unknown, same on ramp and floor):
  On ramp:  a_ramp  = g·(sin20° − μ·cos20°) = 9.81·(0.342 − 0.940·μ)
  On floor: a_floor = μ · g = 9.81 · μ
  Block released from rest at the top of the ramp; transitions smoothly to
  floor at the base; decelerates to rest on floor.

Match the μ implied by the seed. Preserve camera, lighting, geometry.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the block has non-zero downhill
velocity at that instant and has not yet reached the floor.

---

## Evaluation Loop

1. Track block-centroid trajectory in the generated video.
2. Split into ramp phase (while z is decreasing along the 20° slope) and floor
   phase (while z ≈ constant at the block's floor-riding height).
3. Fit a_ramp and a_floor (each a uniform acceleration) from the two segments.
4. Solve μ from a_floor: μ_fit = a_floor / 9.81.
   Cross-check: plug μ_fit into the ramp formula and compare predicted a_ramp
   against observed; quality of fit = consistency score.
5. Compare μ_fit against `mu_hidden` in the variant's `_params.json`.
