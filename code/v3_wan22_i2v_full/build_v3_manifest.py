#!/usr/bin/env python3
"""Build a complete V3 manifest for Wan2.2 I2V A14B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PARAM_VARIANTS: dict[str, dict[str, dict[str, float]]] = {
    "v3_A": {
        "earth_low_drag": {"g_hidden": 9.81, "c_hidden": 0.18, "e_hidden": 0.84},
        "earth_mid_drag": {"g_hidden": 9.81, "c_hidden": 0.42, "e_hidden": 0.78},
        "low_g_bouncy": {"g_hidden": 4.9, "c_hidden": 0.28, "e_hidden": 0.88},
        "high_g_lossy": {"g_hidden": 14.7, "c_hidden": 0.55, "e_hidden": 0.62},
    },
    "v3_B": {
        "slow_low_damp": {"omega0_hidden": 1.75, "gamma_hidden": 0.08},
        "base": {"omega0_hidden": 2.35, "gamma_hidden": 0.20},
        "fast_low_damp": {"omega0_hidden": 3.05, "gamma_hidden": 0.12},
        "fast_high_damp": {"omega0_hidden": 3.05, "gamma_hidden": 0.36},
    },
    "v3_C": {
        "low_mu_bouncy": {"g_hidden": 9.81, "mu_hidden": 0.06, "e_hidden": 0.86},
        "base": {"g_hidden": 9.81, "mu_hidden": 0.16, "e_hidden": 0.72},
        "high_mu_lossy": {"g_hidden": 9.81, "mu_hidden": 0.28, "e_hidden": 0.54},
        "low_g_mid_mu": {"g_hidden": 4.9, "mu_hidden": 0.16, "e_hidden": 0.72},
    },
    "v3_D": {
        "low_g_low_zone": {"g_hidden": 4.9, "gamma_zone_hidden": 0.28},
        "earth_low_zone": {"g_hidden": 9.81, "gamma_zone_hidden": 0.28},
        "base": {"g_hidden": 9.81, "gamma_zone_hidden": 0.58},
        "high_g_high_zone": {"g_hidden": 14.7, "gamma_zone_hidden": 0.82},
    },
    "v3_E": {
        "low_roll_bouncy": {"mu_r_hidden": 0.018, "e_hidden": 0.86},
        "base": {"mu_r_hidden": 0.035, "e_hidden": 0.74},
        "high_roll_lossy": {"mu_r_hidden": 0.070, "e_hidden": 0.56},
        "high_roll_bouncy": {"mu_r_hidden": 0.070, "e_hidden": 0.86},
    },
    "v3_F": {
        "elastic_low_drag": {"e_hidden": 0.92, "c_hidden": 0.16},
        "base": {"e_hidden": 0.74, "c_hidden": 0.34},
        "lossy_low_drag": {"e_hidden": 0.52, "c_hidden": 0.16},
        "lossy_high_drag": {"e_hidden": 0.52, "c_hidden": 0.58},
    },
    "v3_G": {
        "low_to_high": {"mu_1_hidden": 0.05, "mu_2_hidden": 0.30},
        "base": {"mu_1_hidden": 0.09, "mu_2_hidden": 0.26},
        "high_to_low": {"mu_1_hidden": 0.24, "mu_2_hidden": 0.08},
        "uniform_control": {"mu_1_hidden": 0.16, "mu_2_hidden": 0.16},
    },
}

DESCRIPTIONS = {
    "v3_A": "projectile motion with drag and bounce",
    "v3_B": "block spring-damper oscillation",
    "v3_C": "incline, floor friction, and wall collision",
    "v3_D": "pendulum with localized damping zone",
    "v3_E": "rolling ball with collision",
    "v3_F": "two-ball collision with drag",
    "v3_G": "block moving through two spatial friction zones",
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
    parser.add_argument("--experiments", nargs="+", default=sorted(PARAM_VARIANTS))
    parser.add_argument("--cameras", nargs="+", default=["CAM_Side"])
    parser.add_argument("--frame-name", default="frame_10.png")
    parser.add_argument("--prompt-mode", choices=["explicit", "hidden"], default="explicit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def hidden_sentence(params: dict[str, float], mode: str) -> str:
    if mode == "hidden":
        names = ", ".join(params)
        return f"The hidden physical parameters are {names}; infer them from the conditioning frame and keep the continuation physically consistent."
    values = ", ".join(f"{key} = {value:g}" for key, value in params.items())
    return f"For this feasibility baseline, use these hidden physical parameters explicitly: {values}."


def make_prompt(experiment: str, params: dict[str, float], camera: str, prompt_mode: str) -> str:
    description = DESCRIPTIONS[experiment]
    common = (
        "Continue the provided frame as a clean V3 multi-physics benchmark video. Preserve the same static "
        f"{camera} camera, object identities, scene geometry, lighting, object scale, floor/ramp/track layout, "
        "and background. The input image is frame 10 of the original seed sequence, so the scene is already "
        "in motion at the start of this generated clip. Keep the moving target objects compact and trackable. "
        + hidden_sentence(params, prompt_mode)
        + " "
    )
    rules = {
        "v3_A": "The ball should follow projectile motion with drag, contact the floor, and bounce with consistent restitution.",
        "v3_B": "The block should oscillate on the guide as a spring-damper system; frequency and amplitude decay must remain smooth.",
        "v3_C": "The block should slide down the incline, move across the floor, collide with the wall, and continue with consistent friction and restitution.",
        "v3_D": "The pendulum should swing in-plane with gravity and lose energy primarily through the localized damping zone.",
        "v3_E": "The rolling ball should move along the slope/track, experience rolling resistance, and collide with consistent restitution.",
        "v3_F": "The moving ball should collide with the second ball; after collision both balls should separate and slow under drag.",
        "v3_G": "The block should slide through two spatial friction zones with a visible change in deceleration after crossing the boundary.",
    }
    return common + f"The experiment is {description}. " + rules[experiment]


def sort_key(experiment: str, path: Path) -> tuple[int, int | str]:
    order = list(PARAM_VARIANTS[experiment])
    if path.name in order:
        return (0, order.index(path.name))
    return (1, path.name)


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    v3_root = args.seeds_root / "v3"
    for experiment in args.experiments:
        if experiment not in PARAM_VARIANTS:
            raise ValueError(f"Unknown experiment: {experiment}")
        exp_dir = v3_root / experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"Seed directory not found: {exp_dir}")
        variants = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: sort_key(experiment, p))
        for variant_dir in variants:
            params = PARAM_VARIANTS[experiment][variant_dir.name]
            for camera in args.cameras:
                camera_dir = variant_dir / camera
                image_path = camera_dir / args.frame_name
                if not image_path.exists():
                    continue
                job_id = f"{experiment}_{variant_dir.name}_{camera}_{args.prompt_mode}_wan22_i2v_a14b_s{args.seed}"
                job = {
                    "job_id": job_id,
                    "phase": "v3_wan22_i2v_a14b_full",
                    "version": "v3",
                    "experiment": experiment,
                    "variant": variant_dir.name,
                    "camera": camera,
                    "conditioning_image": str(image_path.as_posix()),
                    "conditioning_seed_video": str((camera_dir / "seed_10frames.mp4").as_posix()),
                    "source_prompt_md": str((camera_dir / "prompt.md").as_posix()),
                    "conditioning_frame_index": 10,
                    "generation_start_time_s": 10.0 / 24.0,
                    "prompt_mode": args.prompt_mode,
                    "prompt": make_prompt(experiment, params, camera, args.prompt_mode),
                    "negative_prompt": NEGATIVE_PROMPT,
                    "hidden_params": params,
                    "target_param_name": "multi_hidden",
                    "target_param_value": "",
                    "model_name": "wan2.2_i2v_a14b",
                    "model_id": "Wan2.2-I2V-A14B",
                    "seed": args.seed,
                    "source_fps": 24,
                    "ball_radius_m": 0.24,
                    "block_half_height_m": 0.24,
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

