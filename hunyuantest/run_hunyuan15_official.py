#!/usr/bin/env python3
"""Run HunyuanVideo-1.5 jobs through the official generate.py entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean string, got: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--hunyuan-repo",
        type=Path,
        required=True,
        help="Official HunyuanVideo-1.5 source directory containing generate.py.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="ModelScope/HuggingFace checkpoint directory passed to official --model_path.",
    )
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--video-length", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--rewrite", type=str_to_bool, default=False)
    parser.add_argument("--sr", type=str_to_bool, default=False)
    parser.add_argument("--save-pre-sr-video", type=str_to_bool, default=False)
    parser.add_argument("--cfg-distilled", type=str_to_bool, default=False)
    parser.add_argument("--enable-step-distill", type=str_to_bool, default=False)
    parser.add_argument("--sparse-attn", type=str_to_bool, default=False)
    parser.add_argument("--use-sageattn", type=str_to_bool, default=False)
    parser.add_argument("--enable-cache", type=str_to_bool, default=False)
    parser.add_argument("--cache-type", default="deepcache")
    parser.add_argument("--offloading", type=str_to_bool, default=True)
    parser.add_argument("--group-offloading", type=str_to_bool, default=None)
    parser.add_argument("--overlap-group-offloading", type=str_to_bool, default=False)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path, max_jobs: int | None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_no}: {exc}") from exc
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
    return jobs


def resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bool_arg(value: bool) -> str:
    return "true" if value else "false"


def torchrun_command(nproc_per_node: int) -> list[str]:
    torchrun = "torchrun"
    return [torchrun, f"--nproc_per_node={nproc_per_node}"]


def python_distributed_command(nproc_per_node: int) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc_per_node}"]


def command_available(command: list[str], cwd: Path, env: dict[str, str]) -> bool:
    try:
        result = subprocess.run(
            [*command, "--help"],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def launcher_command(nproc_per_node: int, cwd: Path, env: dict[str, str]) -> list[str]:
    direct = torchrun_command(nproc_per_node)
    if command_available(direct, cwd, env):
        return direct
    fallback = python_distributed_command(nproc_per_node)
    if command_available(fallback, cwd, env):
        return fallback
    raise RuntimeError("Neither torchrun nor python -m torch.distributed.run is available in this Python environment.")


def build_command(args: argparse.Namespace, job: dict[str, Any], output_mp4: Path, launcher: list[str]) -> list[str]:
    image_value = job.get("conditioning_image") or job.get("image_path") or "none"
    if image_value not in {"none", "None", ""}:
        image_path = resolve_path(Path(image_value), args.repo_root)
        if not args.dry_run and not image_path.exists():
            raise FileNotFoundError(f"conditioning image not found: {image_path}")
        image_arg = str(image_path)
    else:
        image_arg = "none"

    cmd = [
        *launcher,
        str((args.hunyuan_repo / "generate.py").resolve()),
        "--prompt",
        str(job["prompt"]),
        "--negative_prompt",
        str(job.get("negative_prompt", "")),
        "--image_path",
        image_arg,
        "--resolution",
        args.resolution,
        "--aspect_ratio",
        args.aspect_ratio,
        "--seed",
        str(int(job.get("seed", 123))),
        "--rewrite",
        bool_arg(args.rewrite),
        "--cfg_distilled",
        bool_arg(args.cfg_distilled),
        "--enable_step_distill",
        bool_arg(args.enable_step_distill),
        "--sparse_attn",
        bool_arg(args.sparse_attn),
        "--use_sageattn",
        bool_arg(args.use_sageattn),
        "--enable_cache",
        bool_arg(args.enable_cache),
        "--cache_type",
        args.cache_type,
        "--offloading",
        bool_arg(args.offloading),
        "--overlap_group_offloading",
        bool_arg(args.overlap_group_offloading),
        "--dtype",
        args.dtype,
        "--sr",
        bool_arg(args.sr),
        "--save_pre_sr_video",
        bool_arg(args.save_pre_sr_video),
        "--output_path",
        str(output_mp4),
        "--model_path",
        str(args.model_path.resolve()),
    ]
    if args.group_offloading is not None:
        cmd.extend(["--group_offloading", bool_arg(args.group_offloading)])
    if args.num_inference_steps is not None:
        cmd.extend(["--num_inference_steps", str(args.num_inference_steps)])
        cmd.extend(["--total_steps", str(args.num_inference_steps)])
    if args.video_length is not None:
        cmd.extend(["--video_length", str(args.video_length)])
    return cmd


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.hunyuan_repo = resolve_path(args.hunyuan_repo, Path.cwd())
    args.model_path = resolve_path(args.model_path, Path.cwd())
    args.outdir = resolve_path(args.outdir, Path.cwd())

    generate_py = args.hunyuan_repo / "generate.py"
    if not args.dry_run:
        if not generate_py.exists():
            raise FileNotFoundError(f"generate.py not found under --hunyuan-repo: {generate_py}")
        if not args.model_path.exists():
            raise FileNotFoundError(f"model path not found: {args.model_path}")

    jobs = read_manifest(args.manifest, args.max_jobs)
    videos_dir = args.outdir / "videos"
    metadata_dir = args.outdir / "metadata"
    logs_dir = args.outdir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = args.outdir / "generation_log.jsonl"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    launcher = launcher_command(args.nproc_per_node, args.hunyuan_repo, env) if not args.dry_run else torchrun_command(args.nproc_per_node)

    print(f"jobs: {len(jobs)}")
    print(f"hunyuan_repo: {args.hunyuan_repo}")
    print(f"model_path: {args.model_path}")
    print(f"gpu_id: {args.gpu_id}")
    print(f"launcher: {' '.join(launcher)}")

    for index, job in enumerate(jobs, 1):
        job_id = job["job_id"]
        output_mp4 = videos_dir / f"{job_id}.mp4"
        output_meta = metadata_dir / f"{job_id}.json"
        stdout_log = logs_dir / f"{job_id}.stdout.log"
        stderr_log = logs_dir / f"{job_id}.stderr.log"
        cmd = build_command(args, job, output_mp4, launcher)

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
            proc = subprocess.run(
                cmd,
                cwd=str(args.hunyuan_repo),
                env=env,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
            )
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
                    "runner": "official_hunyuanvideo_1.5_generate.py",
                    "hunyuan_repo": str(args.hunyuan_repo),
                    "model_path": str(args.model_path),
                    "output_video": str(output_mp4),
                    "command": cmd,
                    "elapsed_seconds": elapsed,
                    "resolution": args.resolution,
                    "aspect_ratio": args.aspect_ratio,
                    "num_inference_steps": args.num_inference_steps,
                    "video_length": args.video_length,
                    "cfg_distilled": args.cfg_distilled,
                    "enable_step_distill": args.enable_step_distill,
                    "sr": args.sr,
                    "offloading": args.offloading,
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
