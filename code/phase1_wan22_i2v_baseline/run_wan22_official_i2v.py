#!/usr/bin/env python3
"""Run phase-1 I2V jobs with the official Wan2.2 codebase.

Use this runner when the checkpoint was downloaded from ModelScope as
`Wan2.2-I2V-A14B`, not as the Hugging Face Diffusers-format repository.
It shells out to:

    python <wan_repo>/generate.py --task i2v-A14B --ckpt_dir <ckpt_dir> ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--wan-repo",
        type=Path,
        required=True,
        help="Path to the cloned Wan-Video/Wan2.2 code repository containing generate.py.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Path to the ModelScope/HuggingFace official Wan2.2-I2V-A14B checkpoint directory.",
    )
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--size", default="832*480", help="Use 832*480 for cheaper 480P smoke tests, or 1280*720 for 720P.")
    parser.add_argument("--frame-num", type=int, default=81, help="Wan expects frame_num = 4n + 1.")
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--sample-guide-scale", type=float, default=None)
    parser.add_argument("--offload-model", action="store_true", default=True)
    parser.add_argument("--no-offload-model", action="store_false", dest="offload_model")
    parser.add_argument("--convert-model-dtype", action="store_true", default=True)
    parser.add_argument("--no-convert-model-dtype", action="store_false", dest="convert_model_dtype")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path, max_jobs: int | None) -> list[dict[str, Any]]:
    jobs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
    return jobs


def resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def build_command(args: argparse.Namespace, job: dict[str, Any], image_path: Path, output_mp4: Path) -> list[str]:
    cmd = [
        sys.executable,
        str((args.wan_repo / "generate.py").resolve()),
        "--task",
        "i2v-A14B",
        "--size",
        args.size,
        "--ckpt_dir",
        str(args.ckpt_dir.resolve()),
        "--image",
        str(image_path),
        "--prompt",
        job["prompt"],
        "--save_file",
        str(output_mp4),
        "--base_seed",
        str(int(job.get("seed", 42))),
        "--frame_num",
        str(int(args.frame_num)),
    ]
    if args.offload_model:
        cmd.extend(["--offload_model", "True"])
    if args.convert_model_dtype:
        cmd.append("--convert_model_dtype")
    if args.t5_cpu:
        cmd.append("--t5_cpu")
    if args.sample_steps is not None:
        cmd.extend(["--sample_steps", str(args.sample_steps)])
    if args.sample_shift is not None:
        cmd.extend(["--sample_shift", str(args.sample_shift)])
    if args.sample_guide_scale is not None:
        cmd.extend(["--sample_guide_scale", str(args.sample_guide_scale)])
    return cmd


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    args.wan_repo = resolve_path(args.wan_repo, Path.cwd())
    args.ckpt_dir = resolve_path(args.ckpt_dir, Path.cwd())

    generate_py = args.wan_repo / "generate.py"
    if not args.dry_run:
        if not generate_py.exists():
            raise FileNotFoundError(f"generate.py not found under --wan-repo: {generate_py}")
        if not args.ckpt_dir.exists():
            raise FileNotFoundError(f"checkpoint dir not found: {args.ckpt_dir}")

    jobs = read_manifest(args.manifest, args.max_jobs)
    videos_dir = args.outdir / "videos"
    metadata_dir = args.outdir / "metadata"
    logs_dir = args.outdir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = args.outdir / "generation_log.jsonl"

    print(f"jobs: {len(jobs)}")
    print(f"wan_repo: {args.wan_repo}")
    print(f"ckpt_dir: {args.ckpt_dir}")
    print(f"gpu_id: {args.gpu_id}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for index, job in enumerate(jobs, 1):
        job_id = job["job_id"]
        image_path = resolve_path(Path(job["conditioning_image"]), repo_root)
        output_mp4 = videos_dir / f"{job_id}.mp4"
        output_meta = metadata_dir / f"{job_id}.json"
        stdout_log = logs_dir / f"{job_id}.stdout.log"
        stderr_log = logs_dir / f"{job_id}.stderr.log"

        cmd = build_command(args, job, image_path, output_mp4)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        if output_mp4.exists() and not args.overwrite:
            print(f"[{index}/{len(jobs)}] skip {job_id}")
            append_jsonl(run_log, {"status": "skip", "job_id": job_id, "output": str(output_mp4)})
            continue

        start = time.time()
        print(f"[{index}/{len(jobs)}] run {job_id}")
        with stdout_log.open("w", encoding="utf-8") as stdout_f, stderr_log.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.run(cmd, cwd=str(args.wan_repo), env=env, stdout=stdout_f, stderr=stderr_f, text=True)
        elapsed = time.time() - start

        record = {
            "job_id": job_id,
            "returncode": proc.returncode,
            "output": str(output_mp4),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "elapsed_seconds": elapsed,
        }
        if proc.returncode == 0 and output_mp4.exists():
            record["status"] = "ok"
            write_json(
                output_meta,
                {
                    **job,
                    "runner": "official_wan2.2_generate.py",
                    "wan_repo": str(args.wan_repo),
                    "ckpt_dir": str(args.ckpt_dir),
                    "output_video": str(output_mp4),
                    "command": cmd,
                    "elapsed_seconds": elapsed,
                },
            )
            print(f"[{index}/{len(jobs)}] ok {job_id} {elapsed:.1f}s")
        else:
            record["status"] = "error"
            print(f"[{index}/{len(jobs)}] error {job_id}; see {stderr_log}")
            append_jsonl(run_log, record)
            raise SystemExit(proc.returncode or 1)
        append_jsonl(run_log, record)


if __name__ == "__main__":
    main()
