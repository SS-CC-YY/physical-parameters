#!/usr/bin/env python3
"""Build V1 Wan2.2 I2V manifests with strict prompts and optional trail images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPERIMENTS: dict[str, dict[str, str]] = {
    "v1_A": {"prefix": "g", "target": "g_hidden", "unit": "m/s^2", "title": "free-fall gravity"},
    "v1_B": {"prefix": "e", "target": "e_hidden", "unit": "", "title": "bounce restitution"},
    "v1_C": {"prefix": "mu", "target": "mu_hidden", "unit": "", "title": "kinetic friction"},
    "v1_D": {"prefix": "gamma", "target": "gamma_hidden", "unit": "s^-1", "title": "pendulum damping"},
}

STRICT_NEGATIVE_PROMPT = (
    "No camera movement, no zoom, no pan, no scene cut, no new object, no duplicated ball, "
    "no duplicated rod, no disappearing object, no changing floor, no changing wall, no changing "
    "lighting, no changing object size, no text, no subtitles, no watermark, no persistent trail, "
    "no center markers, no blue dots, no red dots, no arrows in the output, no deformation, "
    "no cartoon motion, no impossible teleportation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-root", type=Path, default=Path("blender/seeds"))
    parser.add_argument("--conditioning-root", type=Path, required=True)
    parser.add_argument("--conditioning-mode", default="trail_10f")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=sorted(EXPERIMENTS))
    parser.add_argument("--cameras", nargs="+", default=["CAM_Side"])
    parser.add_argument("--prompt-mode", choices=["explicit", "hidden"], default="explicit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def parse_variant_value(experiment: str, variant: str) -> float:
    prefix = EXPERIMENTS[experiment]["prefix"]
    match = re.match(rf"^{re.escape(prefix)}(\d+)(?:p(\d+))?$", variant)
    if not match:
        raise ValueError(f"cannot parse {experiment}/{variant}")
    whole, frac = match.groups()
    return float(whole if frac is None else f"{whole}.{frac}")


def variant_sort_key(experiment: str, path: Path) -> tuple[int, float | str]:
    try:
        return (0, parse_variant_value(experiment, path.name))
    except ValueError:
        return (1, path.name)


def trail_sentence(conditioning_mode: str) -> str:
    if conditioning_mode.startswith("trail_"):
        return (
            "The faint translucent orange ghost trail in the conditioning image shows the object's previous "
            "positions over the last frames. Treat this trail only as motion-history guidance. It is not an "
            "extra object, and it must disappear in the generated video. Generate exactly one solid current "
            "orange object. "
        )
    if conditioning_mode.startswith("center_"):
        return (
            "The small blue and red dots in the conditioning image are motion-history annotations. Each dot marks "
            "the center position of the orange ball in one previous frame, and the red dot marks the current center. "
            "Use the spacing and direction of these dots only to infer the ball's recent velocity and motion trend. "
            "These dots are not physical objects, not decorations, and not part of the scene. They must completely "
            "disappear in the generated video. Generate exactly one solid orange ball with no dots or markers. "
        )
    return ""


def scene_lock(experiment: str, camera: str) -> str:
    base = (
        "Continue this exact physics benchmark scene from the current solid object state. "
        f"Keep the {camera} camera completely static. Preserve the same gray floor, white walls, "
        "lighting, object size, object color, material, scale, and background. Do not move the camera. "
        "Do not change the floor height. Do not add, remove, resize, recolor, or duplicate any object. "
    )
    if experiment == "v1_D":
        return (
            base
            + "This scene contains a rigid pendulum: one orange ball is permanently attached to one thin "
            "dark rigid rod. The pivot point is fixed at the same image location for the entire video. "
            "The rod length never changes and the ball never detaches from the rod. "
        )
    if experiment == "v1_C":
        return base + "The scene contains one orange ball sliding on a perfectly flat horizontal floor. "
    return base + "The scene contains one orange rubber-matte ball and one flat gray floor. "


def hidden_or_explicit(target: str, value: float, unit: str, prompt_mode: str) -> str:
    if prompt_mode == "hidden":
        return (
            f"The hidden physical parameter is {target}. Infer the motion from the conditioning image "
            "and its motion-history annotation, then keep the continuation physically consistent. "
        )
    unit_text = f" {unit}" if unit else ""
    return f"The target physical parameter is {target} = {value:g}{unit_text}. "


def gravity_description(value: float) -> str:
    if value < 5.0:
        qualitative = "This is much weaker than Earth gravity, so the ball should fall slowly and its downward speed increases only gently."
    elif value > 12.0:
        qualitative = "This is stronger than Earth gravity, so the ball should accelerate downward quickly and reach the floor much sooner."
    else:
        qualitative = "This is close to Earth gravity, so the ball should show a normal smooth free-fall acceleration."
    return (
        f"Physics rule: gravity is exactly {value:g} meters per second squared downward. "
        f"{qualitative} The ball continues vertical free fall with constant downward acceleration. "
        "Its downward speed must increase smoothly every frame. There is no horizontal drift, no spin, "
        "no air drag, and no bounce before floor contact. After contacting the floor, the ball stays still on the floor. "
    )


def restitution_description(value: float) -> str:
    height_ratio = value * value
    if value < 0.6:
        qualitative = "This is a lossy bounce. The ball should rapidly lose bounce height and settle after a few impacts."
    elif value > 0.85:
        qualitative = "This is a very elastic bounce. The bounce peak heights should remain almost the same, with only a small loss each bounce."
    else:
        qualitative = "This is a moderately elastic bounce. The bounce heights should decrease visibly but not immediately vanish."
    return (
        f"Physics rule: the coefficient of restitution is exactly {value:g}. "
        f"At every floor impact, the upward speed after the bounce is {value:g} times the downward impact speed. "
        f"The next bounce peak height is about {height_ratio:.3g} times the previous drop height. "
        f"{qualitative} The ball moves only vertically; do not introduce horizontal drift. "
    )


def friction_description(value: float) -> str:
    decel = value * 9.81
    if value < 0.08:
        qualitative = "This is weak friction, so the ball should keep sliding for a long distance."
    elif value > 0.22:
        qualitative = "This is strong friction, so the ball should slow down quickly and stop sooner."
    else:
        qualitative = "This is moderate friction, so the ball should slow down steadily."
    return (
        f"Physics rule: the kinetic friction coefficient is exactly {value:g} on the flat floor. "
        f"The horizontal deceleration magnitude is about {decel:.3g} meters per second squared. "
        f"{qualitative} The ball slides horizontally to the right while staying at the same height. "
        "Its horizontal speed decreases smoothly and uniformly. There is no bounce, no vertical motion, "
        "no hop, and no camera movement. If the speed reaches zero, the ball remains still. "
    )


def damping_description(value: float) -> str:
    if value < 0.08:
        qualitative = "Damping is weak; the amplitude should decrease slowly and the pendulum should keep swinging for many cycles."
    elif value > 0.22:
        qualitative = "Damping is strong; the amplitude should shrink quickly and the pendulum should settle much sooner."
    else:
        qualitative = "Damping is moderate; the amplitude should decay smoothly over time."
    return (
        f"Physics rule: the damping coefficient is exactly {value:g} per second. "
        f"{qualitative} The pendulum swings only in the image plane. The pivot stays fixed, the rod is rigid, "
        "the rod length is constant, and the orange ball remains attached to the rod. There is no out-of-plane motion. "
    )


def make_prompt(experiment: str, value: float, camera: str, prompt_mode: str, conditioning_mode: str) -> str:
    meta = EXPERIMENTS[experiment]
    prompt = scene_lock(experiment, camera)
    prompt += trail_sentence(conditioning_mode)
    prompt += hidden_or_explicit(meta["target"], value, meta["unit"], prompt_mode)
    if experiment == "v1_A":
        prompt += gravity_description(value)
    elif experiment == "v1_B":
        prompt += "Known gravity is 9.81 meters per second squared downward. " + restitution_description(value)
    elif experiment == "v1_C":
        prompt += "Known gravity is 9.81 meters per second squared downward. " + friction_description(value)
    elif experiment == "v1_D":
        prompt += "Known gravity is 9.81 meters per second squared downward. " + damping_description(value)
    else:
        raise KeyError(experiment)
    prompt += "Keep the output photorealistic, clean, and easy to track by orange color segmentation."
    return prompt


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for experiment in args.experiments:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"unknown experiment: {experiment}")
        exp_dir = args.seeds_root / "v1" / experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"seed experiment directory not found: {exp_dir}")
        variants = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: variant_sort_key(experiment, p))
        for variant_dir in variants:
            value = parse_variant_value(experiment, variant_dir.name)
            target_name = EXPERIMENTS[experiment]["target"]
            for camera in args.cameras:
                conditioning_image = args.conditioning_root / args.conditioning_mode / experiment / variant_dir.name / camera / "conditioning.png"
                if not conditioning_image.exists():
                    continue
                job_id = f"{experiment}_{variant_dir.name}_{camera}_{args.prompt_mode}_{args.conditioning_mode}_strict_wan22_i2v_a14b_s{args.seed}"
                job = {
                    "job_id": job_id,
                    "phase": "v1_wan22_i2v_prompt_trail",
                    "version": "v1",
                    "experiment": experiment,
                    "variant": variant_dir.name,
                    "camera": camera,
                    "conditioning_mode": args.conditioning_mode,
                    "conditioning_image": str(conditioning_image.as_posix()),
                    "conditioning_seed_video": str((variant_dir / camera / "seed_10frames.mp4").as_posix()),
                    "conditioning_frame_index": 10 if args.conditioning_mode == "raw_frame10" else int(re.search(r"(\d+)", args.conditioning_mode).group(1)),
                    "prompt_mode": args.prompt_mode,
                    "prompt_style": "strict_scene_physics_observable",
                    "prompt": make_prompt(experiment, value, camera, args.prompt_mode, args.conditioning_mode),
                    "negative_prompt": STRICT_NEGATIVE_PROMPT,
                    "hidden_params": {target_name: value},
                    "target_param_name": target_name,
                    "target_param_value": value,
                    "known_params": {"g_known": 9.81} if experiment != "v1_A" else {},
                    "ball_radius_m": 0.24,
                    "model_name": "wan2.2_i2v_a14b",
                    "model_id": "Wan2.2-I2V-A14B",
                    "seed": args.seed,
                    "source_fps": 24,
                }
                jobs.append(job)
                if args.max_jobs is not None and len(jobs) >= args.max_jobs:
                    return jobs
    return jobs


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
