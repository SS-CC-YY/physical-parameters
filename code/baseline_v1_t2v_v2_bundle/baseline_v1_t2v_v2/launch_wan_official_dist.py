from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List

WAN_MODELS: Dict[str, Dict[str, Any]] = {
    "wan2.1_t2v_1.3b": {
        "task": "t2v-1.3B",
        "size": "832*480",
        "extra": ["--sample_shift", "8", "--sample_guide_scale", "6", "--ring_size", "4", "--ulysses_size", "1"],
    },
    "wan2.2_ti2v_5b": {
        "task": "ti2v-5B",
        "size": "1280*704",
        "extra": ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", "4", "--offload_model", "True", "--convert_model_dtype", "--t5_cpu"],
    },
    "wan2.2_t2v_a14b": {
        "task": "t2v-A14B",
        "size": "1280*720",
        "extra": ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", "4", "--offload_model", "True", "--convert_model_dtype", "--t5_cpu"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a single manifest entry through the official Wan repo with torchrun.")
    parser.add_argument("--job-json", type=Path, required=True, help="Path to a single job JSON file.")
    parser.add_argument("--wan-repo-dir", type=Path, required=True, help="Path to the cloned official Wan repo containing generate.py")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU ids to use for one job.")
    parser.add_argument("--dry-run", action="store_true", help="Print the torchrun command without executing it.")
    return parser.parse_args()


def build_cmd(job: Dict[str, Any], nproc: int) -> List[str]:
    model = job["model"]
    if model not in WAN_MODELS:
        raise ValueError(f"Unsupported official Wan launcher model: {model}")

    spec = WAN_MODELS[model]
    size = job.get("size_str", spec["size"])
    ckpt_dir = job["ckpt_dir"]

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "generate.py",
        "--task", spec["task"],
        "--size", size,
        "--ckpt_dir", ckpt_dir,
        "--prompt", job["prompt"],
    ]
    cmd += spec["extra"]

    if job.get("init_image") and model == "wan2.2_ti2v_5b":
        cmd += ["--image", job["init_image"]]

    if job.get("use_prompt_extend"):
        cmd += ["--use_prompt_extend"]
        if job.get("prompt_extend_method"):
            cmd += ["--prompt_extend_method", job["prompt_extend_method"]]
        if job.get("prompt_extend_target_lang"):
            cmd += ["--prompt_extend_target_lang", job["prompt_extend_target_lang"]]
        if job.get("prompt_extend_model"):
            cmd += ["--prompt_extend_model", job["prompt_extend_model"]]

    return cmd


def main() -> None:
    args = parse_args()
    job = json.loads(args.job_json.read_text(encoding="utf-8"))
    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = build_cmd(job, nproc=len(gpu_ids))
    print("RUN:", " ".join(shlex.quote(x) for x in cmd))

    if not args.dry_run:
        subprocess.run(cmd, cwd=str(args.wan_repo_dir), env=env, check=True)


if __name__ == "__main__":
    main()
