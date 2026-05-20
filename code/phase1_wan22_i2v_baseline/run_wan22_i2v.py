#!/usr/bin/env python3
"""Run Wan2.2 I2V jobs from a JSONL manifest.

This script is designed for the server workflow where the model has already
been downloaded into a local `models/` directory.  It sets CUDA_VISIBLE_DEVICES
before importing torch/diffusers so `--gpu-id 7` maps to `cuda:0` inside Python.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default="../models/Wan2.2-I2V-A14B-Diffusers",
        help="Local model directory or Hugging Face model id.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--gpu-id", default="7")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-local-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true", default=True)
    parser.add_argument("--no-vae-tiling", action="store_false", dest="vae_tiling")
    parser.add_argument("--max-area", default=None, help="Override manifest max_area, e.g. 480*832.")
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_max_area(value: Any) -> int:
    if value is None:
        return 480 * 832
    text = str(value).strip().lower().replace(" ", "")
    if "*" in text:
        a, b = text.split("*", 1)
        return int(a) * int(b)
    return int(text)


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


def resolve_repo_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_model_source(model_path: str, repo_root: Path) -> str:
    path = Path(model_path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((Path.cwd() / path).resolve())
        candidates.append((repo_root / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return model_path


def resize_to_max_area(image: Any, pipe: Any, max_area: int) -> Any:
    import numpy as np

    aspect_ratio = image.height / image.width
    mod_value = 16
    try:
        mod_value = int(pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1])
    except Exception:
        pass
    height = max(mod_value, round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value)
    width = max(mod_value, round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value)
    return image.resize((int(width), int(height)))


def from_pretrained_compat(pipe_cls: Any, model_source: str, dtype: Any, local_files_only: bool) -> Any:
    kwargs = {"local_files_only": local_files_only}
    try:
        return pipe_cls.from_pretrained(model_source, torch_dtype=dtype, **kwargs)
    except TypeError:
        return pipe_cls.from_pretrained(model_source, dtype=dtype, **kwargs)


def load_pipeline(model_source: str, cpu_offload: bool, force_local_only: bool, vae_tiling: bool) -> Any:
    import torch

    try:
        from diffusers import WanImageToVideoPipeline

        pipe_cls = WanImageToVideoPipeline
    except ImportError:
        from diffusers import DiffusionPipeline

        pipe_cls = DiffusionPipeline

    local_files_only = force_local_only or Path(model_source).exists()
    pipe = from_pretrained_compat(pipe_cls, model_source, torch.bfloat16, local_files_only)

    if vae_tiling and hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def export_job(pipe: Any, job: dict[str, Any], image_path: Path, output_mp4: Path, args: argparse.Namespace) -> None:
    import torch
    from diffusers.utils import export_to_video, load_image

    image = load_image(str(image_path)).convert("RGB")
    max_area = parse_max_area(args.max_area if args.max_area is not None else job.get("max_area"))
    image = resize_to_max_area(image, pipe, max_area)

    generator = torch.Generator(device="cuda").manual_seed(int(job.get("seed", 42)))
    kwargs: dict[str, Any] = {
        "image": image,
        "prompt": job["prompt"],
        "negative_prompt": job.get("negative_prompt"),
        "height": image.height,
        "width": image.width,
        "num_frames": int(args.num_frames or job.get("num_frames", 81)),
        "num_inference_steps": int(args.num_inference_steps or job.get("num_inference_steps", 40)),
        "guidance_scale": float(args.guidance_scale or job.get("guidance_scale", 3.5)),
        "generator": generator,
    }
    if job.get("guidance_scale_2") is not None:
        kwargs["guidance_scale_2"] = float(job["guidance_scale_2"])

    result = pipe(**kwargs)
    frames = result.frames[0]
    fps = int(args.fps or job.get("fps", 16))
    export_to_video(frames, output_video_path=str(output_mp4), fps=fps)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = args.repo_root.resolve()
    jobs = read_manifest(args.manifest, args.max_jobs)
    args.outdir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.outdir / "videos"
    metadata_dir = args.outdir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / "generation_log.jsonl"

    model_source = resolve_model_source(args.model_path, repo_root)
    print(f"jobs: {len(jobs)}")
    print(f"model_source: {model_source}")
    print(f"gpu_id: {args.gpu_id}")

    if args.dry_run:
        for job in jobs:
            print(job["job_id"], job["conditioning_image"])
        return

    import torch

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe = load_pipeline(model_source, args.cpu_offload, args.force_local_only, args.vae_tiling)

    for index, job in enumerate(jobs, 1):
        job_id = job["job_id"]
        image_path = resolve_repo_path(job["conditioning_image"], repo_root)
        output_mp4 = videos_dir / f"{job_id}.mp4"
        output_meta = metadata_dir / f"{job_id}.json"
        if output_mp4.exists() and not args.overwrite:
            record = {"status": "skip", "job_id": job_id, "output": str(output_mp4)}
            append_jsonl(log_path, record)
            print(f"[{index}/{len(jobs)}] skip {job_id}")
            continue

        start = time.time()
        try:
            export_job(pipe, job, image_path, output_mp4, args)
            elapsed = time.time() - start
            meta = {
                **job,
                "resolved_model_source": model_source,
                "resolved_conditioning_image": str(image_path),
                "output_video": str(output_mp4),
                "elapsed_seconds": elapsed,
            }
            output_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            append_jsonl(log_path, {"status": "ok", "job_id": job_id, "output": str(output_mp4), "elapsed_seconds": elapsed})
            print(f"[{index}/{len(jobs)}] ok {job_id} {elapsed:.1f}s")
        except Exception as exc:
            record = {"status": "error", "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"}
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False, indent=2))
            raise


if __name__ == "__main__":
    main()
