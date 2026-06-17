#!/usr/bin/env python3
"""Build a multi-seed V1-A gravity manifest for Wan2.2 internal probing."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-root", type=Path, default=Path("blender/seeds"))
    parser.add_argument("--conditioning-root", type=Path, default=None)
    parser.add_argument("--conditioning-mode", default="center_10f")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="CAM_Side")
    parser.add_argument("--prompt-mode", choices=["explicit", "hidden", "visual_trace"], default="visual_trace")
    parser.add_argument("--base-seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--frame-name", default="frame_10.png")
    parser.add_argument("--max-jobs", type=int, default=None)
    return parser.parse_args()


def load_v1_builder() -> ModuleType:
    script = REPO_ROOT / "code" / "v1_wan22_i2v_full" / "build_v1_manifest.py"
    spec = importlib.util.spec_from_file_location("v1_wan22_manifest_builder", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import manifest builder: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_strict_builder() -> ModuleType:
    script = REPO_ROOT / "code" / "v1_wan22_i2v_prompt_trail" / "build_v1_strict_manifest.py"
    spec = importlib.util.spec_from_file_location("v1_wan22_strict_manifest_builder", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import strict manifest builder: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    use_conditioning_root = args.conditioning_root is not None
    builder = load_strict_builder() if use_conditioning_root else load_v1_builder()
    jobs: list[dict[str, Any]] = []
    for seed in args.base_seeds:
        if use_conditioning_root:
            builder_args = argparse.Namespace(
                seeds_root=args.seeds_root,
                conditioning_root=args.conditioning_root,
                conditioning_mode=args.conditioning_mode,
                output=args.output,
                experiments=["v1_A"],
                cameras=[args.camera],
                prompt_mode=args.prompt_mode,
                seed=seed,
                max_jobs=None,
            )
        else:
            builder_args = argparse.Namespace(
                seeds_root=args.seeds_root,
                output=args.output,
                experiments=["v1_A"],
                cameras=[args.camera],
                frame_name=args.frame_name,
                prompt_mode=args.prompt_mode,
                seed=seed,
                max_jobs=None,
            )
        for job in builder.build_jobs(builder_args):
            job = dict(job)
            job["phase"] = "wan22_internal_probe_v1_A"
            job["probe_task"] = "v1_A_gravity_regression"
            job["probe_label_name"] = "g_hidden"
            job["probe_label_value"] = float(job["target_param_value"])
            job["probe_base_seed"] = seed
            job["probe_prompt_contains_numeric_target"] = args.prompt_mode == "explicit"
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
    print(f"wrote {len(jobs)} probe jobs to {args.output}")


if __name__ == "__main__":
    main()
