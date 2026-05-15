from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "wan2.1_t2v_1.3b": {
        "hf_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "pipeline_type": "wan21",
        "dtype": "bfloat16",
        "flow_shift": 5.0,
        "width": 832,
        "height": 480,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "guidance_scale": 6.0,
    },
    "wan2.2_ti2v_5b": {
        "hf_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "pipeline_type": "wan22_ti2v",
        "dtype": "bfloat16",
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "guidance_scale_2": 5.0,
    },
    "wan2.2_t2v_a14b": {
        "hf_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "pipeline_type": "wan22_a14b",
        "dtype": "bfloat16",
        "width": 1280,
        "height": 720,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
    },
    "cogvideox_2b": {
        "hf_id": "THUDM/CogVideoX-2b",
        "pipeline_type": "cogvideox",
        "dtype": "float16",
        "width": 720,
        "height": 480,
        "num_frames": 49,
        "fps": 8,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
    },
}


@dataclass
class WorkerArgs:
    outdir: str
    cpu_offload: bool
    allow_tf32: bool
    local_model_root: Optional[str]
    overwrite: bool
    log_jsonl: str
    force_local_only: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate Baseline V1 videos on multiple GPUs using diffusers-compatible pipelines."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL manifest path.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for mp4 outputs and metadata.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated CUDA GPU ids, e.g. 0,1,2,3")
    parser.add_argument(
        "--local-model-root",
        type=Path,
        default=None,
        help="Optional local directory containing downloaded model folders, or a direct model folder path.",
    )
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload for extra memory safety.")
    parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 matmul/cudnn where supported.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if target mp4 already exists.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument(
        "--force-local-only",
        action="store_true",
        help="When set, refuse to download from the Hub and only load from local files.",
    )
    return parser.parse_args()


def load_manifest(path: Path, max_jobs: Optional[int] = None) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
    return jobs


def resolve_model_source(job: Dict[str, Any], local_model_root: Optional[str]) -> str:
    model_id = job.get("model_id") or MODEL_SPECS[job["model"]]["hf_id"]
    if local_model_root is None:
        return model_id

    root = Path(local_model_root).expanduser().resolve()
    leaf = model_id.split("/")[-1]

    candidate = root / leaf
    if candidate.exists():
        return str(candidate)

    if root.exists() and root.name == leaf:
        return str(root)

    return model_id


def write_log(log_path: str, record: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def unload_pipe(pipe: Any) -> None:
    if pipe is None:
        return
    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def load_pipe(model_name: str, model_source: str, cpu_offload: bool, force_local_only: bool) -> Any:
    import torch
    from diffusers import CogVideoXPipeline, WanPipeline, AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    spec = MODEL_SPECS[model_name]
    dtype = torch.bfloat16 if spec.get("dtype") == "bfloat16" else torch.float16
    is_local = Path(model_source).exists()
    local_files_only = bool(force_local_only or is_local)

    if spec["pipeline_type"] == "wan21":
        pipe = WanPipeline.from_pretrained(
            model_source,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=float(spec.get("flow_shift", 5.0)),
        )
    elif spec["pipeline_type"] in {"wan22_ti2v", "wan22_a14b"}:
        vae = AutoencoderKLWan.from_pretrained(
            model_source,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )
        pipe = WanPipeline.from_pretrained(
            model_source,
            vae=vae,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    elif spec["pipeline_type"] == "cogvideox":
        pipe = CogVideoXPipeline.from_pretrained(
            model_source,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    else:
        raise ValueError(f"Unknown pipeline_type for model {model_name}")

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def export_one_video(pipe: Any, job: Dict[str, Any], output_mp4: Path) -> None:
    import torch
    from diffusers.utils import export_to_video, load_image

    generator = torch.Generator(device="cuda").manual_seed(int(job["seed"]))

    common_kwargs = dict(
        prompt=job["prompt"],
        negative_prompt=job.get("negative_prompt"),
        width=int(job["width"]),
        height=int(job["height"]),
        num_frames=int(job["num_frames"]),
        num_inference_steps=int(job["num_inference_steps"]),
        guidance_scale=float(job["guidance_scale"]),
        generator=generator,
    )

    if job.get("guidance_scale_2") is not None:
        common_kwargs["guidance_scale_2"] = float(job["guidance_scale_2"])

    init_image = job.get("init_image")
    if init_image:
        common_kwargs["image"] = load_image(init_image)

    result = pipe(**common_kwargs)
    frames = result.frames[0]
    export_to_video(frames, output_video_path=str(output_mp4), fps=int(job["fps"]))


def worker_main(global_gpu_id: int, q: mp.Queue, worker_args: WorkerArgs) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(global_gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    if worker_args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    current_model_name: Optional[str] = None
    current_model_source: Optional[str] = None
    pipe: Any = None

    while True:
        try:
            job = q.get(timeout=3)
        except queue.Empty:
            continue

        if job is None:
            break

        model_name = job["model"]
        model_source = resolve_model_source(job, worker_args.local_model_root)

        output_dir = (
            Path(worker_args.outdir)
            / job.get("output_subdir", job["task"])
            / job.get("prompt_style", "default")
            / model_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_mp4 = output_dir / f"{job['id']}.mp4"
        output_meta = output_dir / f"{job['id']}.json"

        if output_mp4.exists() and not worker_args.overwrite:
            write_log(
                worker_args.log_jsonl,
                {
                    "status": "skip",
                    "gpu": global_gpu_id,
                    "job_id": job["id"],
                    "output": str(output_mp4),
                    "model_source": model_source,
                },
            )
            continue

        try:
            if current_model_name != model_name or current_model_source != model_source or pipe is None:
                unload_pipe(pipe)
                pipe = load_pipe(
                    model_name=model_name,
                    model_source=model_source,
                    cpu_offload=worker_args.cpu_offload,
                    force_local_only=worker_args.force_local_only,
                )
                current_model_name = model_name
                current_model_source = model_source

            export_one_video(pipe, job, output_mp4)
            output_meta.write_text(
                json.dumps(
                    {
                        **job,
                        "resolved_model_source": model_source,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_log(
                worker_args.log_jsonl,
                {
                    "status": "ok",
                    "gpu": global_gpu_id,
                    "job_id": job["id"],
                    "output": str(output_mp4),
                    "model_source": model_source,
                    "prompt_style": job.get("prompt_style"),
                },
            )
        except Exception as e:
            err = {
                "status": "error",
                "gpu": global_gpu_id,
                "job_id": job["id"],
                "error": f"{type(e).__name__}: {e}",
                "model_source": model_source,
                "traceback": traceback.format_exc(),
            }
            write_log(worker_args.log_jsonl, err)
            print(json.dumps(err, ensure_ascii=False, indent=2))
            unload_pipe(pipe)
            pipe = None
            current_model_name = None
            current_model_source = None

    unload_pipe(pipe)


def main() -> None:
    args = parse_args()
    jobs = load_manifest(args.manifest, max_jobs=args.max_jobs)
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    log_jsonl = str(args.outdir / "generation_log.jsonl")

    q: mp.Queue = mp.Queue()
    for job in jobs:
        q.put(job)
    for _ in gpu_ids:
        q.put(None)

    worker_args = WorkerArgs(
        outdir=str(args.outdir),
        cpu_offload=args.cpu_offload,
        allow_tf32=args.allow_tf32,
        local_model_root=str(args.local_model_root) if args.local_model_root else None,
        overwrite=args.overwrite,
        log_jsonl=log_jsonl,
        force_local_only=args.force_local_only,
    )

    procs: List[mp.Process] = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=worker_main, args=(gpu_id, q, worker_args), daemon=False)
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode not in (0, None):
            exit_code = p.exitcode

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
