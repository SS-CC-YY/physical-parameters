# V3-D — Localized-Damping-Zone Pendulum: Video Continuation (Hidden g + γ_zone)

**Level**: L3 | **Experiment**: v3_D | **Script**: `v3/exp_D_ball_em_damping_pendulum.py`
**Hidden**: `g_hidden` (m/s²), `gamma_zone_hidden` (in-zone damping, s⁻¹)
**Known**: L = 2.06 m + ball radius 0.24 m, θ₀ = 24°, damping zone spans ±7.5° around vertical
**Variants**: low_g_low_zone / earth_low_zone / base / high_g_high_zone
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A pendulum swings through a **localized electromagnetic damping zone**
at the bottom of its arc — the zone is visually marked by two blue rectangular
magnets flanking the swing path. Damping is zero outside this zone; inside the
zone an extra linear damping torque acts on the ball. The **gravity g and the
in-zone damping γ_zone are both unknown**.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose oscillation period
and per-pass energy loss are consistent with the (g, γ_zone) implied by the seed.

We will later track the ball, extract θ(t), measure oscillation period (→ g)
and per-cycle amplitude drop (→ γ_zone), and report the fit.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Horizontal pivot bar at z = 4.8 m; thin rigid rod hanging.
- Rod length = 2.06 m, ball radius = 0.24 m → effective L = 2.30 m (pivot to ball-center).
- Orange rubber-matte ball at the end of the rod.
- Two BLUE rectangular magnets near the bottom of the arc, on either side of
  the vertical — these mark the damping zone boundaries at θ = ±7.5°.
- OUTSIDE the zone: no damping (pure pendulum motion).
- INSIDE the zone: extra linear damping γ_zone acts on angular velocity.
- Initial release at θ₀ = 24° from vertical, from rest.
- Motion strictly in x–z plane; no rod flex, no ball spin.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video of a pendulum with localized bottom-zone damping.
First 10 frames given (0–0.417 s); generate frames 11–240 at 24 fps.

Rules you must obey:
- Pendulum geometry fixed: pivot, rod length, ball radius, magnets' positions.
- Gravity constant; its value must match what the seed implies (some variants
  use non-Earth gravity, so do NOT assume 9.81).
- Swing amplitude drops noticeably during each bottom-of-arc crossing (the
  "magnet zone" region, θ ∈ [−7.5°, +7.5°]); OUTSIDE this zone the swing
  is essentially lossless.
- Between consecutive peaks, the peak amplitude decreases by a fixed ratio
  (because each half-period crosses the zone exactly once); the decrement
  must match the γ_zone implied by the seed.
- Period of oscillation is governed by g and L, and is roughly constant across
  the video (approximately 2π·√(L/g) for small angles).
- Motion stays strictly in the x–z plane; no twisting.
- Preserve camera, lighting, pivot, rod, ball, and the two blue magnet boxes.
```

## Easy Version (physics formulae given, values still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (g and γ_zone both unknown; L = 2.30 m, θ₀ = 24°):
  Pendulum equation of motion:
    θ̈ = −(g/L) · sin θ − 2·γ_effective(θ) · θ̇
  Where γ_effective(θ) = γ_zone if |θ| ≤ 7.5°, else 0.
  Small-angle period: T ≈ 2π·√(L/g)  →  g = 4π²·L/T²
  Ball position: x = L·sin θ,  z = pivot_z − L·cos θ,  pivot_z = 4.8 m.

Match the (g, γ_zone) implied by the seed. Preserve camera, lighting, geometry.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the pendulum is in mid-swing with
non-zero angular velocity.

---

## Evaluation Loop

1. Track ball-center (x, z) in generated video; convert to angle
   θ(t) = atan2(x − pivot_x, pivot_z − z), pivot_x = 0, pivot_z = 4.8.
2. Measure period T from zero-crossings of θ(t); g_fit = 4π²·L/T² with L = 2.30 m.
3. Locate successive amplitude peaks A_1, A_2, ... (one per half-period).
4. For this zone-damping model, the per-half-period amplitude ratio r = A_{n+1}/A_n
   is approximately constant (each half-period includes exactly one zone crossing).
   Energy lost per zone crossing corresponds to γ_zone; numerically integrate
   the equation of motion to find γ_zone_fit that best reproduces r.
5. Compare (g_fit, γ_zone_fit) against `g_hidden`, `gamma_zone_hidden` in `_params.json`.
