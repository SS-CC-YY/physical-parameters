# V3-B — Spring-Damper Oscillation: Video Continuation (Hidden ω₀ + γ)

**Level**: L3 | **Experiment**: v3_B | **Script**: `v3/exp_B_block_spring_damper.py`
**Hidden**: `omega0_hidden` (undamped natural frequency, rad/s), `gamma_hidden` (linear damping, s⁻¹)
**Known**: x_eq = −1.8 m, initial displacement A₀ = 1.65 m (block starts at x = −0.15 m), released from rest
**Variants**: slow_low_damp / base / fast_low_damp / fast_high_damp
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of a 10 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of a 10-second physics
simulation. A block slides on a horizontal guide rail, connected to a wall on
the left by a coil spring with a linear damper. The block starts displaced to
the right (at x = −0.15 m) and is released from rest. The **spring's natural
frequency ω₀ and the damping γ are both unknown**.

**Do not output a number.** Your job is to **generate the remaining 9.583 s
(frames 11–240 at 24 fps)**, producing a 10 s video whose oscillation period
and amplitude envelope are both consistent with the (ω₀, γ) implied by the seed.

We will later track block x(t), measure period T → damped frequency ω_d, and
fit amplitude envelope A(t) = A₀·exp(−γ·t) to recover both parameters via
ω₀ = √(ω_d² + γ²).

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- Rectangular block (half-extent 0.34 m in x), sliding on a horizontal guide
  rail at z_guide = 0.5 m. Block centroid translates only along x.
- A visible coil spring connects the block's left edge to an anchor at
  x_anchor = −6.3 m on the left wall. The spring visually compresses and
  stretches as the block moves.
- Guide rail spans x ∈ [−7.0, 3.8] m (the block should stay well within this range).
- Equilibrium position: x_eq = −1.8 m.
- Initial release: block at rest at x = −0.15 m (i.e. +1.65 m to the right of x_eq).
- Motion is 1-D along x — block height and y-position fixed.
- Lighting and background static.

---

## Hard Version (no physics formula given)

```
Continue this 10-second video of a spring-damper oscillation.
First 10 frames given (0–0.417 s); generate frames 11–240 at 24 fps.

Rules you must obey:
- Block oscillates about x_eq = −1.8 m, with initial displacement +1.65 m,
  released from rest. Motion is 1-D along x.
- Successive swing amplitudes decrease monotonically; the rate of amplitude
  decay must match what the opening frames imply.
- Oscillation period (time between successive left-turn points, or right-turn
  points) is constant throughout the video and must match what the seed implies.
- Spring coils visually compress when block moves left of equilibrium and
  stretch when block moves right; keep this animation plausible.
- Block cannot leave the guide rail (stays within x ∈ [−7.0, 3.8]).
- Over 10 s, expect roughly 1 to 5 full oscillations depending on the
  implicit frequency.
- Preserve camera, lighting, block, spring, guide rail geometry.
```

## Easy Version (physics formulae given, values still hidden)

```
Continue this 10-second video at 24 fps. First 10 frames given; generate frames 11–240.

Physics to obey (ω₀ and γ are both unknown — infer from the opening frames):
  Define displacement from equilibrium: u(t) = x(t) − x_eq,  x_eq = −1.8 m.
  u(t) = A₀ · exp(−γ·t) · cos(ω_d · t)
  ω_d = √(ω₀² − γ²)     (the observable, damped, angular frequency)
  A₀ = 1.65 m, u(0) = +1.65 m, u̇(0) = 0 (released from rest).
  Amplitude envelope: A(t) = A₀ · exp(−γ·t).

Match the (ω₀, γ) implied by the seed. Preserve camera, lighting, materials.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 10 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the block is nearly stationary at
that instant but has just begun accelerating back toward equilibrium.

---

## Evaluation Loop

1. Track block centroid x(t) in generated video; compute u(t) = x(t) − (−1.8).
2. Extract oscillation period T from successive zero crossings or peak times
   of u(t); ω_d_fit = 2π / T.
3. Extract amplitude envelope samples A(t_n) at each peak; fit
   ln A(t) = ln A₀ − γ·t → γ_fit from linear regression.
4. Combine: ω₀_fit = √(ω_d_fit² + γ_fit²).
5. Compare (ω₀_fit, γ_fit) against `omega0_hidden` and `gamma_hidden` in
   `_params.json`.
