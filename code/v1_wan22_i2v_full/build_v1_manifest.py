#!/usr/bin/env python3
"""Build a complete V1 manifest for Wan2.2 I2V A14B."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPERIMENTS = {
    "v1_A": {
        "variant_prefix": "g",
        "target": "g_hidden",
        "unit": "m/s^2",
        "title": "free fall under hidden gravity",
    },
    "v1_B": {
        "variant_prefix": "e",
        "target": "e_hidden",
        "unit": "",
        "title": "vertical bounce with hidden restitution",
    },
    "v1_C": {
        "variant_prefix": "mu",
        "target": "mu_hidden",
        "unit": "",
        "title": "horizontal slide with hidden kinetic friction",
    },
    "v1_D": {
        "variant_prefix": "gamma",
        "target": "gamma_hidden",
        "unit": "s^-1",
        "title": "damped pendulum with hidden damping",
    },
}

NEGATIVE_PROMPT = (
    "No extra objects, no people, no text, no subtitles, no watermark, no camera movement, "
    "no pan, no zoom, no scene cut, no duplicated ball, no severe deformation, no large "
    "lighting change, no background change."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-root", type=Path, default=Path("blender/seeds"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=sorted(EXPERIMENTS))
    parser.add_argument("--cameras", nargs="+", default=["CAM_Side"])
    parser.add_argument("--frame-name", default="frame_10.png")
    parser.add_argument("--prompt-mode", choices=["explicit", "hidden"], default="explicit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def parse_variant_value(experiment: str, variant: str) -> float:
    prefix = EXPERIMENTS[experiment]["variant_prefix"]
    pattern = rf"^{re.escape(prefix)}(\d+)(?:p(\d+))?$"
    match = re.match(pattern, variant)
    if not match:
        raise ValueError(f"Cannot parse variant value for {experiment}/{variant}")
    whole, frac = match.groups()
    return float(whole if frac is None else f"{whole}.{frac}")


def variant_sort_key(experiment: str, path: Path) -> tuple[int, float | str]:
    try:
        return (0, parse_variant_value(experiment, path.name))
    except ValueError:
        return (1, path.name)


def explicit_sentence(meta: dict[str, str], value: float, prompt_mode: str) -> str:
    if prompt_mode == "hidden":
        return (
            f"The hidden parameter is {meta['target']}; infer the motion from the conditioning frame "
            "and keep the continuation physically consistent."
        )
    unit = f" {meta['unit']}" if meta["unit"] else ""
    return f"For this feasibility baseline, use {meta['target']} = {value:g}{unit}."


def make_prompt(experiment: str, value: float, camera: str, prompt_mode: str) -> str:
    meta = EXPERIMENTS[experiment]
    target_line = explicit_sentence(meta, value, prompt_mode)
    common = (
        "Continue the provided frame as a clean physics benchmark video. Preserve the same static "
        f"{camera} camera, orange rubber-matte ball, gray floor, white walls, lighting, object scale, "
        "and background. The input image is frame 10 of the original seed sequence, so the object is "
        "already in motion at the start of this generated clip. Keep the object compact and easy to "
        "track by color segmentation. "
    )

    if experiment == "v1_A":
        return (
            common
            + target_line
            + " The ball continues falling vertically under constant gravity. It has no horizontal "
            "drift, no spin, no drag, and no bounce. Once the ball reaches the floor, it remains "
            "perfectly still on the floor for the rest of the video."
        )
    if experiment == "v1_B":
        return (
            common
            + target_line
            + " The ball continues a purely vertical bounce sequence under known gravity 9.81 m/s^2. "
            "Every floor contact uses the same restitution. Successive bounce peak heights should "
            "decrease geometrically, and the ball should eventually settle on the floor."
        )
    if experiment == "v1_C":
        return (
            common
            + target_line
            + " The ball slides horizontally to the right on a flat floor under known gravity "
            "9.81 m/s^2. It decelerates uniformly from kinetic friction. The height stays constant, "
            "there is no bounce, and if the velocity reaches zero the ball stays at rest."
        )
    if experiment == "v1_D":
        return (
            common
            + target_line
            + " The scene is a rigid-rod pendulum with an orange ball at the end. Gravity is known "
            "9.81 m/s^2. The pendulum swings in the x-z plane only, with a fixed pivot and rigid rod. "
            "The amplitude decays smoothly over time according to the damping value."
        )
    raise KeyError(experiment)


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    v1_root = args.seeds_root / "v1"
    for experiment in args.experiments:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment}")
        exp_dir = v1_root / experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"Seed directory not found: {exp_dir}")
        variants = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: variant_sort_key(experiment, p))
        for variant_dir in variants:
            value = parse_variant_value(experiment, variant_dir.name)
            target_name = EXPERIMENTS[experiment]["target"]
            for camera in args.cameras:
                camera_dir = variant_dir / camera
                image_path = camera_dir / args.frame_name
                if not image_path.exists():
                    continue
                job_id = f"{experiment}_{variant_dir.name}_{camera}_{args.prompt_mode}_wan22_i2v_a14b_s{args.seed}"
                job = {
                    "job_id": job_id,
                    "phase": "v1_wan22_i2v_a14b_full",
                    "version": "v1",
                    "experiment": experiment,
                    "variant": variant_dir.name,
                    "camera": camera,
                    "conditioning_image": str(image_path.as_posix()),
                    "conditioning_seed_video": str((camera_dir / "seed_10frames.mp4").as_posix()),
                    "source_prompt_md": str((camera_dir / "prompt.md").as_posix()),
                    "conditioning_frame_index": 10,
                    "generation_start_time_s": 10.0 / 24.0,
                    "prompt_mode": args.prompt_mode,
                    "prompt": make_prompt(experiment, value, camera, args.prompt_mode),
                    "negative_prompt": NEGATIVE_PROMPT,
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

