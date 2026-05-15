# V2-E — Tilted-Floor Bounce: Video Continuation (Hidden Restitution)

**Level**: L2 | **Experiment**: v2_E | **Script**: `v2/exp_E_ball_tilted_floor_bounce_hidden_restitution.py`
**Hidden**: `e_hidden` (normal restitution coefficient)
**Known**: g = 9.81 m/s^2, floor tilt angle, tangential impact retention
**Variants**: e0p35 / e0p55 / e0p72 / e0p88
**Seed**: `seed_10frames.mp4` + `frame_01..10.png`

---

## Task — Continuation, NOT Calculation

You are given the opening 10 frames of a 10-second video. A ball falls onto a
tilted floor and then repeatedly bounces while drifting along the slope. The
hidden parameter is the same normal restitution coefficient at every impact.

Generate frames 11-240. The horizontal drift and the bounce heights must both
be consistent with one constant restitution value.

---

## Scene Description

- Orange rubber ball, radius 0.24 m.
- Floor is tilted by a known small angle.
- Gravity is 9.81 m/s^2 and is known.
- Impact response is applied along the floor normal.
- Tangential speed loses a known fixed fraction at contact; the hidden
  quantity is still only the normal restitution coefficient.
- Tangential motion continues along the slope; do not turn this into pure
  vertical bouncing.
- Preserve camera, lighting, materials, and floor geometry.

---

## Hard Version

```
Continue this tilted-floor bouncing-ball video to 10 seconds at 24 fps.

Rules:
- The ball repeatedly impacts the same tilted plane.
- Each impact uses the same normal restitution ratio.
- The ball should drift along the slope while it bounces.
- Gravity is 9.81 m/s^2.
- Preserve the camera, floor tilt, ball appearance, and lighting.
```

## Easy Version

```
Continue this 10-second video.

Physics:
- Between impacts: ballistic motion under gravity.
- At contact with the tilted floor, split velocity into normal and tangent
  components.
- Normal component after impact: v_after_n = -e * v_before_n.
- Tangent component is reduced by the known tangential retention factor.

The value e is hidden and must be inferred from the seed.
```

---

## Evaluation Loop

1. Track ball-center x(t), z(t).
2. Detect contacts with the tilted plane and estimate pre/post normal velocity.
3. Fit one constant restitution value across all visible impacts.
4. Compare fitted e against `e_hidden` in `_params.json`.
