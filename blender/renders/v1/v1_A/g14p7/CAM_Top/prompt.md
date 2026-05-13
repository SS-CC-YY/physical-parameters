# V1-A — Free Fall: Video Continuation (Hidden Gravity)

**Level**: L1 | **Experiment**: v1_A | **Script**: `v1/exp_A_ball_freefall_gravity.py`
**Hidden**: `g_hidden` (gravitational acceleration, m/s²)
**Variants**: g2 / g4p9 / g9p81 / g14p7
**Seed**: `seed_10frames.mp4` + `frame_01..10.png` (first 0.417 s of an 8 s video at 24 fps)

---

## Task — Continuation, NOT Calculation

You are given the opening **10 frames (t = 0 → 0.417 s)** of an 8-second physics
simulation. A ball is released from rest and falls under a **constant but unknown
gravitational acceleration g** specific to this world.

**Do not output a number.** Your job is to **generate the remaining 7.583 s
(frames 11–192 at 24 fps)**, producing a complete 8 s video that is physically
consistent with the gravity implied by the opening frames.

We will later track the ball's trajectory in your generated video and fit it
against `z(t) = z₀ − ½·g·t²` to back-compute the implicit g in your output.
Your score = |g_fit − g_true| / g_true.

---

## Scene Description (must be preserved)

- Static 1080p side-view camera (no pan / no zoom / no cuts).
- Static scene: gray matte floor, white back-wall and left-wall, dark world background.
- Three soft area lights (top + two fills). Lighting is constant throughout.
- Single orange rubber-matte ball, radius ≈ 0.24 m, falling vertically.
- Ball-center starts at z₀ = 4.2 m and must come to rest with its center at
  z = 0.44 m (the floor is a thin slab whose top is at z = 0.20 m).
- No other objects, no text overlays appearing mid-video.

---

## Hard Version (no physics formula given)

```
Continue this 8-second video of a single ball falling straight down.
The first 10 frames (0–0.417 s) are given; generate frames 11–192 (up to t = 8 s).

Rules you must obey:
- Constant gravity throughout the full 8 s; value must match what the opening
  frames imply (do not guess a standard value).
- Purely vertical motion (no horizontal drift, no rotation changes).
- No air drag, no bouncing.
- The ball decelerates to rest the instant its center reaches z = 0.44 m
  and then stays perfectly still for the remainder of the video.
- Preserve camera angle, lighting, ball appearance, floor geometry, and resolution.
```

## Easy Version (physics formula given, value still hidden)

```
Continue this 8-second video of a single ball falling straight down.
The first 10 frames (0–0.417 s) are given; generate frames 11–192.

Physics to obey (g is unknown — infer it from the opening frames):
  z_center(t) = 4.2 − ½ · g · t²    for t ≤ t_contact
  z_center(t) = 0.44                for t > t_contact
  where t_contact = sqrt(2·(4.2 − 0.44) / g) = sqrt(7.52 / g)

Preserve camera, lighting, materials, and frame rate (24 fps).
No bouncing, no drag, no rotation.
```

---

## Model Usage by Input Type

### A. Video-extension models (Sora 2, Veo 3, Runway Gen-4, Luma Ray 3, Kling 2.0)
Condition on `seed_10frames.mp4`. Paste Hard or Easy as the extension prompt.
Request an 8-second total output at 24 fps; the first 0.417 s is the seed, the
remaining 7.583 s is the generation.

### B. Image-to-video models (Kling I2V, Wan 2.2 I2V, HunyuanVideo I2V)
Condition on `frame_10.png` (the last seed frame). Paste Hard or Easy, adjusted
so the start-of-generation corresponds to t = 0.417 s rather than t = 0.
Request ≈7.6 s of generation.

### C. Keyframe / first-frame-only I2V
If the model only accepts one image, use `frame_10.png`. Tell the model the ball
is already in free fall (non-zero downward velocity) at that instant, so it
should not re-accelerate from rest.

---

## Evaluation Loop

1. Download the generated video.
2. Run a ball tracker (e.g. CoTracker / SAM2 + centroid) to extract z_center(t).
3. Fit `z(t) = 4.2 − ½·g·t²` to the pre-contact segment to get `g_fit`.
4. Report the relative error vs ground truth (`g_true` is in the `_params.json`).
