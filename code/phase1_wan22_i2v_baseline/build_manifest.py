#!/usr/bin/env python3
"""Build a small Wan2.2-I2V phase-1 feasibility manifest.

The default manifest uses V1-A free-fall side-view seeds.  This is intentionally
small: the first feasibility question is whether the model can generate clean
continuations and whether a simple tracker can recover a gravity-related proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDER_ROOT = REPO_ROOT / "blender"
sys.path.insert(0, str(BLENDER_ROOT))

from benchmark_variants import PARAM_VARIANTS  # noqa: E402


DEFAULT_NEGATIVE_PROMPT = (
    "no extra objects, no people, no hands, no text, no subtitles, no watermark, "
    "no camera movement, no pan, no zoom, no scene cut, no duplicated ball, "
    "no deformed ball, no strong motion blur, no bounce, no rolling"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--experiment", default="v1_A")
    parser.add_argument("--camera", default="CAM_Side")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--image-name", default="frame_10.png")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--model-name", default="wan2.2_i2v_a14b")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument(
        "--prompt-mode",
        choices=["explicit_g", "hidden_g"],
        default="explicit_g",
        help="explicit_g is the phase-1 feasibility mode. hidden_g is harder and not recommended for single-image I2V.",
    )
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-area", default="480*832")
    return parser.parse_args()


def parse_max_area(text: str) -> int:
    text = str(text).strip().lower().replace(" ", "")
    if "*" in text:
        a, b = text.split("*", 1)
        return int(a) * int(b)
    return int(text)


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ordered_variants(experiment: str, requested: Iterable[str] | None) -> list[str]:
    all_variants = list(PARAM_VARIANTS[experiment])
    if not requested:
        return all_variants
    missing = [item for item in requested if item not in PARAM_VARIANTS[experiment]]
    if missing:
        raise ValueError(f"Unknown variants for {experiment}: {missing}")
    return list(requested)


def make_prompt(experiment: str, variant: str, hidden: dict[str, float], prompt_mode: str) -> str:
    g_true = hidden.get("g_hidden")
    if experiment != "v1_A" or g_true is None:
        raise ValueError("Phase-1 manifest currently supports free-fall gravity jobs with g_hidden.")

    prefix = (
        "Continue the provided image as a short physics benchmark video. "
        "The scene contains one orange rubber-matte ball in a static side-view camera setup, "
        "with a gray floor and fixed lighting. Preserve the exact object identity, camera, "
        "background, lighting, and ball size. The ball is already in free fall at the start "
        "of this generated clip. Generate a clean continuation with a single smooth vertical "
        "trajectory, no horizontal drift, no bounce, and no camera motion. "
    )
    if prompt_mode == "explicit_g":
        return (
            prefix +
            f"For this feasibility baseline, the target hidden physical parameter is gravity "
            f"g = {g_true:g} m/s^2. The generated motion should visibly reflect this gravity: "
            "larger g should fall faster across otherwise identical jobs. "
            "Keep the ball compact and trackable for later centroid-based measurement."
        )
    return (
        prefix +
        "The gravity is hidden. Infer a plausible continuation from the provided frame and keep "
        "the motion physically consistent and easy to track. This mode is mainly for stress tests; "
        "single-image I2V cannot observe the first ten-frame velocity history."
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    variants = ordered_variants(args.experiment, args.variants)
    max_area = parse_max_area(args.max_area)

    jobs = []
    for variant in variants:
        hidden = PARAM_VARIANTS[args.experiment][variant]
        seed_dir = repo_root / "blender" / "seeds" / args.version / args.experiment / variant / args.camera
        image_path = seed_dir / args.image_name
        seed_video = seed_dir / "seed_10frames.mp4"
        prompt_md = seed_dir / "prompt.md"
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        for seed_offset in range(args.num_seeds):
            seed = args.seed_base + seed_offset
            job_id = (
                f"{args.experiment}_{variant}_{args.camera}_"
                f"{args.prompt_mode}_{args.model_name}_s{seed}"
            )
            jobs.append(
                {
                    "job_id": job_id,
                    "phase": "phase1_wan22_i2v_feasibility",
                    "version": args.version,
                    "experiment": args.experiment,
                    "variant": variant,
                    "camera": args.camera,
                    "conditioning_image": relpath(image_path, repo_root),
                    "conditioning_seed_video": relpath(seed_video, repo_root) if seed_video.exists() else None,
                    "source_prompt_md": relpath(prompt_md, repo_root) if prompt_md.exists() else None,
                    "conditioning_frame_index": 10,
                    "prompt_mode": args.prompt_mode,
                    "prompt": make_prompt(args.experiment, variant, hidden, args.prompt_mode),
                    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                    "hidden_params": hidden,
                    "target_param_name": "g_hidden",
                    "target_param_value": hidden["g_hidden"],
                    "model_name": args.model_name,
                    "model_id": args.model_id,
                    "seed": seed,
                    "num_frames": args.num_frames,
                    "fps": args.fps,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "max_area": max_area,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    print(f"saved: {args.output}")
    print(f"jobs: {len(jobs)}")


if __name__ == "__main__":
    main()
