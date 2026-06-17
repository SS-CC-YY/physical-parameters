#!/usr/bin/env python3
"""Run Wan2.2 generation while capturing compact DiT hidden-state features."""

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
PROBE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame-num", type=int, default=121)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--sample-guide-scale", type=float, default=None)
    parser.add_argument("--offload-model", action="store_true", default=True)
    parser.add_argument("--no-offload-model", action="store_false", dest="offload_model")
    parser.add_argument("--convert-model-dtype", action="store_true", default=True)
    parser.add_argument("--no-convert-model-dtype", action="store_false", dest="convert_model_dtype")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--class-regex", default=r"(Block)")
    parser.add_argument("--module-regex", default=r"^wan\.modules\.model")
    parser.add_argument("--max-records", type=int, default=4096)
    parser.add_argument("--max-feature-dim", type=int, default=8192)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def read_manifest(path: Path, max_jobs: int | None, start_index: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < start_index:
                continue
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
    return jobs


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prompt_for_wan(job: dict[str, Any]) -> str:
    negative = job.get("negative_prompt")
    if not negative:
        return str(job["prompt"])
    return f"{job['prompt']} Avoid: {negative}"


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
        prompt_for_wan(job),
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


def build_env(args: argparse.Namespace, job_id: str, feature_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["WAN22_PROBE_CAPTURE"] = "1"
    env["WAN22_PROBE_JOB_ID"] = job_id
    env["WAN22_PROBE_FEATURE_FILE"] = str(feature_file)
    env["WAN22_PROBE_CLASS_REGEX"] = args.class_regex
    env["WAN22_PROBE_MODULE_REGEX"] = args.module_regex
    env["WAN22_PROBE_MAX_RECORDS"] = str(args.max_records)
    env["WAN22_PROBE_MAX_FEATURE_DIM"] = str(args.max_feature_dim)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROBE_DIR) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    return env


def main() -> None:
    args = parse_args()
    args.repo_root = resolve_path(args.repo_root, Path.cwd())
    args.manifest = resolve_path(args.manifest, Path.cwd())
    args.wan_repo = resolve_path(args.wan_repo, Path.cwd())
    args.ckpt_dir = resolve_path(args.ckpt_dir, Path.cwd())
    args.outdir = resolve_path(args.outdir, Path.cwd())

    if not args.dry_run:
        if not (args.wan_repo / "generate.py").exists():
            raise FileNotFoundError(f"generate.py not found under --wan-repo: {args.wan_repo}")
        if not args.ckpt_dir.exists():
            raise FileNotFoundError(f"checkpoint dir not found: {args.ckpt_dir}")

    jobs = read_manifest(args.manifest, args.max_jobs, args.start_index)
    videos_dir = args.outdir / "videos"
    features_dir = args.outdir / "features"
    metadata_dir = args.outdir / "metadata"
    logs_dir = args.outdir / "logs"
    for directory in [videos_dir, features_dir, metadata_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    run_log = args.outdir / "generation_log.jsonl"

    ok_count = 0
    skip_count = 0
    error_count = 0
    print(f"jobs: {len(jobs)}")
    print(f"wan_repo: {args.wan_repo}")
    print(f"ckpt_dir: {args.ckpt_dir}")
    print(f"outdir: {args.outdir}")
    print(f"class_regex: {args.class_regex}")
    print(f"module_regex: {args.module_regex}")

    for index, job in enumerate(jobs, 1):
        job_id = job["job_id"]
        image_path = resolve_path(Path(job["conditioning_image"]), args.repo_root)
        output_mp4 = videos_dir / f"{job_id}.mp4"
        feature_file = features_dir / f"{job_id}.npz"
        output_meta = metadata_dir / f"{job_id}.json"
        stdout_log = logs_dir / f"{job_id}.stdout.log"
        stderr_log = logs_dir / f"{job_id}.stderr.log"
        command_log = logs_dir / f"{job_id}.command.json"
        cmd = build_command(args, job, image_path, output_mp4)
        write_json(command_log, {"job_id": job_id, "command": cmd, "feature_file": str(feature_file)})

        if args.dry_run:
            print(" ".join(cmd))
            continue
        if not image_path.exists():
            raise FileNotFoundError(f"conditioning image not found: {image_path}")
        if output_mp4.exists() and feature_file.exists() and not args.overwrite:
            skip_count += 1
            print(f"[{index}/{len(jobs)}] skip {job_id}")
            append_jsonl(run_log, {"status": "skip", "job_id": job_id, "output": str(output_mp4), "features": str(feature_file)})
            continue

        env = build_env(args, job_id, feature_file)
        start = time.time()
        print(f"[{index}/{len(jobs)}] run {job_id}", flush=True)
        with stdout_log.open("w", encoding="utf-8") as stdout_f, stderr_log.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.run(cmd, cwd=str(args.wan_repo), env=env, stdout=stdout_f, stderr=stderr_f, text=True)
        elapsed = time.time() - start
        record = {
            "job_id": job_id,
            "returncode": proc.returncode,
            "output": str(output_mp4),
            "features": str(feature_file),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "elapsed_seconds": elapsed,
        }
        if proc.returncode == 0 and output_mp4.exists() and feature_file.exists():
            ok_count += 1
            record["status"] = "ok"
            write_json(
                output_meta,
                {
                    **job,
                    "runner": "wan22_internal_probe_forward_capture",
                    "wan_repo": str(args.wan_repo),
                    "ckpt_dir": str(args.ckpt_dir),
                    "output_video": str(output_mp4),
                    "feature_file": str(feature_file),
                    "size": args.size,
                    "frame_num": args.frame_num,
                    "sample_steps": args.sample_steps,
                    "class_regex": args.class_regex,
                    "module_regex": args.module_regex,
                    "elapsed_seconds": elapsed,
                },
            )
            print(f"[{index}/{len(jobs)}] ok {job_id} {elapsed:.1f}s", flush=True)
        else:
            error_count += 1
            record["status"] = "error"
            print(f"[{index}/{len(jobs)}] error {job_id}; see {stderr_log}", flush=True)
            if args.fail_fast:
                append_jsonl(run_log, record)
                raise SystemExit(proc.returncode or 1)
        append_jsonl(run_log, record)

    summary = {"ok": ok_count, "skip": skip_count, "error": error_count, "total": len(jobs)}
    write_json(args.outdir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if error_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
