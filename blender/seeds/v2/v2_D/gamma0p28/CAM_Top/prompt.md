# V2-D — Large-Angle Nonlinear Pendulum: Video Continuation (Hidden Damping)

**Level**: L2 | **Experiment**: v2_D | **Script**: `v2/exp_D_ball_long_pendulum_hidden_damping.py`
**Hidden**: `gamma_hidden` (linear damping coefficient, s^-1)
**Known**: g = 9.81 m/s^2, rod length, initial release angle about 68 deg
**Variants**: gamma0p04 / gamma0p10 / gamma0p18 / gamma0p28
**Seed**: `seed_10frames.mp4` + `frame_01..10.png`

---

## Task — Continuation, NOT Calculation

You are given the opening 10 frames of a 10-second physics simulation. A rigid
pendulum is released from a large angle, so the small-angle cosine formula is
not the correct motion model. The hidden parameter is a constant damping
coefficient gamma.

Generate frames 11-240 so that the nonlinear swing period and the amplitude
decay are both consistent with the seed.

---

## Scene Description

- Static camera; rigid rod and orange ball swing in the x-z plane.
- Pivot is fixed near z = 5.8 m.
- Initial angle is large, about 68 deg from vertical.
- Gravity is 9.81 m/s^2 and is known.
- Damping is uniform over the whole swing.
- No twisting, rod flex, out-of-plane motion, or extra forces.

---

## Hard Version

```
Continue this 10-second video of a large-angle damped pendulum.
First 10 frames are given; generate frames 11-240 at 24 fps.

Rules:
- The pendulum starts from a large angle, so the swing is visibly nonlinear.
- Gravity is fixed at 9.81 m/s^2.
- Damping is constant and must match the opening frames.
- Successive amplitudes decay smoothly; no localized damping zones.
- Preserve camera, lighting, rod, pivot, ball appearance, and resolution.
```

## Easy Version

```
Continue this 10-second video at 24 fps.

Physics:
  theta_ddot = -(g/L) sin(theta) - 2 gamma theta_dot

The value gamma is hidden and must be inferred from the seed. Use the full
sin(theta) equation, not the small-angle approximation. Preserve the scene.
```

---

## Evaluation Loop

1. Track the ball-center and convert position to angle theta(t).
2. Fit the full nonlinear damped pendulum model to theta(t).
3. Compare fitted gamma against `gamma_hidden` in `_params.json`.
