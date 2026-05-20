# V1-D — Damped Pendulum: Video Continuation (Hidden Damping)

**Level**: L1 | **Experiment**: v1_D | **Script**: `v1/exp_D_ball_pendulum_damping.py`
**Hidden**: `gamma_hidden` (linear damping coefficient, s⁻¹)
**Variants**: gamma0p04 / gamma0p10 / gamma0p18 / gamma0p30
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of an 8 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of an 8-second physics
simulation. A rigid-rod pendulum is released from rest at an angle of about
24° from vertical and swings back and forth. The amplitude decays over time
because of **an unknown linear damping coefficient γ**.

**Do not output a number.** Your job is to **generate the remaining 7.583 s
(frames 11–192)**, producing an 8 s video whose amplitude envelope matches the
damping γ implied by the opening frames.

We will later track the ball-center in the generated video, extract the
amplitude envelope of the swing, and fit A(t) = A₀·exp(−γ·t) to back-compute
γ_fit. Your score = |γ_fit − γ_true| / γ_true.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera.
- A horizontal pivot bar at z = 4.8 m; a thin rigid rod hangs from it.
- Rod length (pivot to ball-center) L ≈ 2.3 m.
- A single orange rubber-matte ball (radius 0.24 m) at the end of the rod.
- Swing is in the x–z plane only — no out-of-plane motion, no rod flex.
- Gravity is 9.81 m/s² (NOT the hidden variable).
- Initial angle ≈ 24° from vertical, initially at rest.
- Damping is uniform throughout the entire swing (not localized to a region).

---

## Hard Version (no physics formula given)

```
Continue this 8-second video of a pendulum swinging with gradually decreasing
amplitude. The first 10 frames (0–0.417 s) are given; generate frames 11–192
up to t = 8 s.

Rules you must obey:
- Pendulum geometry is fixed: straight rigid rod, pivot position, rod length,
  ball radius — all must remain constant.
- Gravity is 9.81 m/s² (constant).
- Each successive peak amplitude must be slightly smaller than the previous one;
  the rate of amplitude decay must match the damping implied by the opening
  frames — do not guess.
- Undamped period ≈ 2π·√(L/g) ≈ 3.04 s, so roughly 2–3 full oscillations fit
  in the 8-second window.
- Motion stays in the x–z plane; no twisting, no rod flex.
- Eventually the swing amplitude is very small and the pendulum approaches
  vertical rest.
- Preserve camera, lighting, rod, pivot, ball appearance.
```

## Easy Version (physics formula given, value still hidden)

```
Continue this 8-second video of a damped pendulum at 24 fps.
The first 10 frames (0–0.417 s) are given; generate frames 11–192.

Physics to obey (γ is unknown — infer from the opening frames):
  θ(t)  = θ₀ · exp(−γ·t) · cos(ω_d · t)
  A(t)  = θ₀ · exp(−γ·t)          (amplitude envelope)
  ω_d   = sqrt(g/L − γ²),  with L = 2.3 m, g = 9.81 m/s²
  θ₀ ≈ 24° (initial release angle)

The ball's position: x(t) = L·sin(θ(t)),  z(t) = pivot_z − L·cos(θ(t)).
Preserve camera, lighting, geometry. Motion strictly in the x–z plane.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Request 8 s total output at 24 fps.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png`. Tell the model the pendulum has non-zero angular
velocity at that instant (it has just begun the outward swing from rest).

---

## Evaluation Loop

1. Track ball-center (x, z) in the generated video; convert to angle:
   θ(t) = atan2(x − pivot_x, pivot_z − z).
2. Locate successive peaks of |θ(t)| → amplitude samples A₁, A₂, A₃, …
3. Fit ln(A_n) = ln(A₀) − γ·t_n to get γ_fit (linear regression of log amplitude).
   If < 3 peaks are visible, use envelope at two non-peak instants:
   γ_fit = ln(A(t_a) / A(t_b)) / (t_b − t_a).
4. Compare γ_fit against `gamma_hidden` in the variant's `_params.json`.
