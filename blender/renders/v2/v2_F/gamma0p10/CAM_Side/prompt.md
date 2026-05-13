# V2-F — Forced Damped Pendulum: Video Continuation (Hidden Damping)

**Level**: L2 | **Experiment**: v2_F | **Script**: `v2/exp_F_forced_pendulum_hidden_damping.py`
**Hidden**: `gamma_hidden` (linear damping coefficient, s^-1)
**Known**: g = 9.81 m/s^2, rod length, drive amplitude, drive frequency
**Variants**: gamma0p04 / gamma0p10 / gamma0p18 / gamma0p28
**Seed**: `seed_10frames.mp4` + `frame_01..10.png`

---

## Task — Continuation, NOT Calculation

You are given the opening 10 frames of a driven pendulum. The pendulum is not
just freely decaying: it is continuously forced by a known periodic drive. The
hidden parameter is the damping coefficient gamma, which controls both the
transient decay and the near-steady response amplitude.

Generate frames 11-240 so that the driven oscillation remains phase-consistent
and amplitude-consistent with the seed.

---

## Scene Description

- Rigid rod and orange ball swing in the x-z plane.
- Pivot, rod length, drive frequency, and drive strength are known.
- Gravity is 9.81 m/s^2.
- Damping is uniform and constant.
- No out-of-plane motion, rod flex, or sudden energy jumps.

---

## Hard Version

```
Continue this 10-second forced pendulum video at 24 fps.

Rules:
- The pendulum is continuously driven by the same periodic force.
- Damping is constant and must match the seed.
- Keep phase, amplitude envelope, and late-time response physically smooth.
- Preserve camera, lighting, rod, pivot, ball appearance, and resolution.
```

## Easy Version

```
Continue this video using the known forced-pendulum model:

  theta_ddot = -(g/L) sin(theta) - 2 gamma theta_dot
              + (drive_accel/L) cos(drive_omega t)

The value gamma is hidden. Infer it from the seed and continue to frame 240.
```

---

## Evaluation Loop

1. Track ball-center and convert to theta(t).
2. Fit the forced damped pendulum model with gamma as the unknown.
3. Evaluate both transient and late-window amplitude/phase error.
4. Compare fitted gamma against `gamma_hidden` in `_params.json`.
