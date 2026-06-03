#!/usr/bin/env python3
"""Build HunyuanVideo-1.5 I2V manifests from benchmark seed frames."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_NEGATIVE_PROMPT = (
    "no extra objects, no people, no hands, no text, no subtitles, no watermark, "
    "no camera movement, no pan, no zoom, no scene cut, no duplicated ball, "
    "no deformed ball, no strong motion blur"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-root", type=Path, default=Path("blender/seeds"))
    parser.add_argument("--version", default="v1")
    parser.add_argument("--experiments", nargs="+", default=["v1_A"])
    parser.add_argument("--cameras", nargs="+", default=["CAM_Side"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-name", default="frame_10.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--model-name", default="hunyuanvideo_1.5")
    parser.add_argument("--model-id", default="Tencent-Hunyuan/HunyuanVideo-1.5")
    return parser.parse_args()


def variant_to_float(name: str) -> float | None:
    match = re.fullmatch(r"g(\d+)(?:p(\d+))?", name)
    if not match:
        return None
    integer, decimal = match.groups()
    return float(integer if decimal is None else f"{integer}.{decimal}")


def variant_sort_key(path: Path) -> tuple[int, float | str]:
    value = variant_to_float(path.name)
    if value is None:
        return (1, path.name)
    return (0, value)


def make_prompt(experiment: str, variant: str, camera: str) -> tuple[str, dict[str, Any], str | None, float | None]:
    if experiment.endswith("_A"):
        g_value = variant_to_float(variant)
        if g_value is not None:
            prompt = (
                "Continue the provided image as a short physics benchmark video. "
                "The scene contains one orange rubber-matte ball in a static side-view camera setup, "
                "with a gray floor and fixed lighting. Preserve the exact object identity, camera, "
                "background, lighting, and ball size. The ball is already in free fall at the start "
                "of this generated clip. Generate a clean continuation with a single smooth vertical "
                "trajectory, no horizontal drift, no bounce, and no camera motion. "
                f"For this feasibility baseline, the target hidden physical parameter is gravity g = {g_value:g} m/s^2. "
                "The generated motion should visibly reflect this gravity: larger g should fall faster "
                "across otherwise identical jobs. Keep the ball compact and trackable for later "
                "centroid-based measurement."
            )
            return prompt, {"g_hidden": g_value}, "g_hidden", g_value

    prompt = (
        "Continue the provided physics benchmark image as a short, clean video. Preserve the scene, "
        f"object identity, fixed {camera} camera, lighting, and background. Generate physically coherent "
        "motion with no scene cut, no camera motion, and a compact trackable object."
    )
    return prompt, {}, None, None


def iter_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.seeds_root / args.version
    jobs: list[dict[str, Any]] = []

    for experiment in args.experiments:
        exp_dir = root / experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"experiment seed directory not found: {exp_dir}")

        for variant_dir in sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=variant_sort_key):
            variant = variant_dir.name
            for camera in args.cameras:
                cam_dir = variant_dir / camera
                image_path = cam_dir / args.frame_name
                if not image_path.exists():
                    continue

                prompt, hidden_params, target_name, target_value = make_prompt(experiment, variant, camera)
                job_id = f"{experiment}_{variant}_{camera}_explicit_g_hunyuan15_s{args.seed}"
                job: dict[str, Any] = {
                    "job_id": job_id,
                    "phase": "phase1_hunyuan15_i2v_feasibility",
                    "version": args.version,
                    "experiment": experiment,
                    "variant": variant,
                    "camera": camera,
                    "conditioning_image": str(image_path.as_posix()),
                    "conditioning_seed_video": str((cam_dir / "seed_10frames.mp4").as_posix()),
                    "source_prompt_md": str((cam_dir / "prompt.md").as_posix()),
                    "conditioning_frame_index": 10,
                    "prompt_mode": "explicit_g",
                    "prompt": prompt,
                    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                    "hidden_params": hidden_params,
                    "target_param_name": target_name,
                    "target_param_value": target_value,
                    "model_name": args.model_name,
                    "model_id": args.model_id,
                    "seed": args.seed,
                    "num_frames": 121,
                    "fps": 24,
                }
                jobs.append(job)
                if args.max_jobs is not None and len(jobs) >= args.max_jobs:
                    return jobs
    return jobs


def main() -> None:
    args = parse_args()
    jobs = iter_jobs(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()
