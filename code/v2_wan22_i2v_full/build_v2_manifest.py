#!/usr/bin/env python3
"""Build a complete V2 manifest for Wan2.2 I2V A14B."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "v2_A": {
        "variant_prefix": "g",
        "target": "g_hidden",
        "unit": "m/s^2",
        "object": "orange rubber-matte ball",
        "known_params": {"e_known": 0.74, "vx0": 4.8, "vz0": 7.8},
    },
    "v2_B": {
        "variant_prefix": "e",
        "target": "e_hidden",
        "unit": "",
        "object": "orange rubber-matte ball",
        "known_params": {"g_known": 9.81},
    },
    "v2_C": {
        "variant_prefix": "mu",
        "target": "mu_hidden",
        "unit": "",
        "object": "orange rectangular block",
        "known_params": {"g_known": 9.81, "theta_deg": 20.0},
    },
    "v2_D": {
        "variant_prefix": "gamma",
        "target": "gamma_hidden",
        "unit": "s^-1",
        "object": "orange ball at the end of a rigid pendulum rod",
        "known_params": {"g_known": 9.81, "rod_length": 3.0},
    },
    "v2_E": {
        "variant_prefix": "e",
        "target": "e_hidden",
        "unit": "",
        "object": "orange rubber-matte ball",
        "known_params": {"g_known": 9.81, "floor_angle_deg": 4.0, "tangent_retention_known": 0.68},
    },
    "v2_F": {
        "variant_prefix": "gamma",
        "target": "gamma_hidden",
        "unit": "s^-1",
        "object": "orange ball at the end of a forced pendulum rod",
        "known_params": {"g_known": 9.81, "rod_length": 2.45, "drive_accel": 1.05, "drive_omega": 1.82},
    },
}

NEGATIVE_PROMPT = (
    "No extra objects, no people, no text, no subtitles, no watermark, no camera movement, "
    "no pan, no zoom, no scene cut, no duplicated object, no severe deformation, no large "
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
    match = re.match(rf"^{re.escape(prefix)}(\d+)(?:p(\d+))?$", variant)
    if not match:
        raise ValueError(f"Cannot parse variant value for {experiment}/{variant}")
    whole, frac = match.groups()
    return float(whole if frac is None else f"{whole}.{frac}")


def variant_sort_key(experiment: str, path: Path) -> tuple[int, float | str]:
    try:
        return (0, parse_variant_value(experiment, path.name))
    except ValueError:
        return (1, path.name)


def parameter_sentence(meta: dict[str, Any], value: float, prompt_mode: str) -> str:
    target = meta["target"]
    if prompt_mode == "hidden":
        return f"The hidden parameter is {target}; infer its value from the conditioning frame and keep the continuation physically consistent."
    unit = f" {meta['unit']}" if meta["unit"] else ""
    return f"For this feasibility baseline, use {target} = {value:g}{unit}."


def known_sentence(known: dict[str, Any]) -> str:
    if not known:
        return ""
    pairs = ", ".join(f"{key} = {value:g}" if isinstance(value, float) else f"{key} = {value}" for key, value in known.items())
    return f" Known physical constants and settings: {pairs}."


def make_prompt(experiment: str, value: float, camera: str, prompt_mode: str) -> str:
    meta = EXPERIMENTS[experiment]
    common = (
        "Continue the provided frame as a clean physics benchmark video. Preserve the same static "
        f"{camera} camera, {meta['object']}, scene geometry, gray floor or ramp, white walls, lighting, "
        "object scale, and background. The input image is frame 10 of the original seed sequence, so "
        "the object is already in motion at the start of this generated clip. Keep the target object "
        "compact and easy to track by color segmentation. "
        + parameter_sentence(meta, value, prompt_mode)
        + known_sentence(meta["known_params"])
        + " "
    )
    if experiment == "v2_A":
        return common + (
            "The ball continues a diagonal projectile trajectory under constant gravity, lands on the floor, "
            "and bounces with the known restitution. Horizontal speed should be preserved in flight, there is "
            "no air drag, and the parabolic arcs should visibly depend on gravity."
        )
    if experiment == "v2_B":
        return common + (
            "The ball continues repeated vertical bouncing. Each bounce uses the same restitution value. "
            "Successive peak heights should decrease consistently and the ball should eventually settle."
        )
    if experiment == "v2_C":
        return common + (
            "The rectangular block slides down a 20 degree ramp, transitions smoothly to a flat floor, and "
            "then decelerates due to the same kinetic friction. It must not tumble, bounce, or hop."
        )
    if experiment == "v2_D":
        return common + (
            "The long pendulum swings in the x-z plane only. The rod is rigid, the pivot is fixed, and the "
            "amplitude decays smoothly according to the damping coefficient."
        )
    if experiment == "v2_E":
        return common + (
            "The ball falls onto a slightly tilted floor and repeatedly bounces while drifting along the slope. "
            "At every contact, the same normal restitution is used and tangent speed is reduced by the known retention."
        )
    if experiment == "v2_F":
        return common + (
            "The forced pendulum continues with the same periodic drive. Keep phase, amplitude, and damping "
            "smooth; no out-of-plane motion, rod flex, or sudden energy jumps."
        )
    raise KeyError(experiment)


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    v2_root = args.seeds_root / "v2"
    for experiment in args.experiments:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment}")
        exp_dir = v2_root / experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"Seed directory not found: {exp_dir}")
        variants = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: variant_sort_key(experiment, p))
        for variant_dir in variants:
            value = parse_variant_value(experiment, variant_dir.name)
            meta = EXPERIMENTS[experiment]
            for camera in args.cameras:
                camera_dir = variant_dir / camera
                image_path = camera_dir / args.frame_name
                if not image_path.exists():
                    continue
                job_id = f"{experiment}_{variant_dir.name}_{camera}_{args.prompt_mode}_wan22_i2v_a14b_s{args.seed}"
                job = {
                    "job_id": job_id,
                    "phase": "v2_wan22_i2v_a14b_full",
                    "version": "v2",
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
                    "hidden_params": {meta["target"]: value},
                    "target_param_name": meta["target"],
                    "target_param_value": value,
                    "known_params": meta["known_params"],
                    "track_object": "block" if experiment == "v2_C" else "ball",
                    "ball_radius_m": 0.24,
                    "block_half_height_m": 0.24,
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

