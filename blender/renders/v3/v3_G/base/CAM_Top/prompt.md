# V3-G — Spatially Non-Uniform Friction: Video Continuation (Hidden μ₁ + μ₂)

**Level**: L3 | **Experiment**: v3_G | **Script**: `v3/exp_G_block_spatial_friction.py`
**Hidden**: `mu_1_hidden` (friction in Zone 1, left of boundary), `mu_2_hidden` (friction in Zone 2, right of boundary)
**Known**: g = 9.81 m/s², v₀ = 4.6 m/s (block moving rightward), boundary at x = −0.2 m (yellow marker)
**Variants**: low_to_high / base / high_to_low / uniform_control
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A rectangular block slides rightward on a floor split into two
friction zones by a thin yellow vertical marker at x = −0.2 m. Zone 1 (left of
the marker) has friction μ₁; Zone 2 (right of the marker) has friction μ₂.
**Both coefficients are hidden, and they may or may not be equal.**

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose two-phase deceleration
(possibly with a visible change rate at the zone boundary) is consistent with
the (μ₁, μ₂) implied by the seed.

We will later track the block, segment by zone, fit uniform decelerations in
each, and back-compute (μ₁, μ₂).

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Block half-extent 0.36 m in x, centroid height z ≈ 0.25 m, sliding along the floor.
- Block starts at x_start = −6.6 m, with v₀ = 4.6 m/s rightward.
- Zone boundary at x = −0.2 m, marked by a thin yellow vertical line/strip on the floor.
  Floor surface visual texture is the same in both zones — you cannot visually
  tell from the floor alone which zone has higher friction. (In some variants
  μ₁ equals μ₂, so the boundary may be physically meaningless; in others the
  deceleration changes at the boundary.)
- g = 9.81 m/s² (NOT hidden).
- Motion strictly along +x; the block may or may not come to rest inside the
  visible scene, depending on the friction values.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video: block slides rightward across a two-zone floor,
possibly decelerating at different rates in each zone.
First 10 frames given (0–0.417 s); generate frames 11–240 at 24 fps.

Rules you must obey:
- Block decelerates uniformly within each zone.
- Before crossing the yellow boundary at x = −0.2, the deceleration rate
  is dictated by μ₁.
- After crossing, the deceleration rate changes (or doesn't, if μ₁ = μ₂)
  according to μ₂.
- The transition at the boundary is INSTANTANEOUS (no smoothing, no ramp).
- If the block stops before t = 10 s, it stays still for the rest of the video.
- CRITICAL: do not hallucinate a deceleration change if the seed doesn't imply
  one. For the uniform_control variant, μ₁ = μ₂ and the block's deceleration
  is the same on both sides of the marker — no change at the boundary.
- Motion strictly along +x; no lateral drift.
- Preserve camera, lighting, block, floor, yellow boundary marker.
```

## Easy Version (physics formulae given, values still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (μ₁ and μ₂ both unknown; g = 9.81 m/s², boundary at x = −0.2):
  While x < −0.2:  a = −μ₁ · g
  While x > −0.2:  a = −μ₂ · g
  The transition between the two decelerations is instantaneous at the boundary.
  Block enters with v₀ = 4.6 m/s rightward.

Match (μ₁, μ₂) implied by the seed. Preserve camera, lighting, geometry.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the block is moving rightward with
non-zero velocity; it is currently still in Zone 1 (x < −0.2).

---

## Evaluation Loop

1. Track block centroid x(t) in generated video; compute v(t) by finite difference.
2. Locate the boundary crossing frame t_cross at which x(t) = −0.2.
3. Zone 1 phase (t < t_cross): fit v(t) = v₀ − a₁ · t → a₁ → μ₁_fit = a₁ / 9.81.
4. Zone 2 phase (t > t_cross, up to stop): fit v(t) linearly → a₂ → μ₂_fit = a₂ / 9.81.
5. Compare (μ₁_fit, μ₂_fit) against `mu_1_hidden`, `mu_2_hidden` in `_params.json`.
6. For the `uniform_control` variant, a successful model produces μ₁_fit ≈ μ₂_fit;
   a model that hallucinates a change will show a significant gap.
