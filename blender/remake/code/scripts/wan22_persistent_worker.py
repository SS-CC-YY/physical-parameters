#!/usr/bin/env python3
"""Keep the official Wan2.2 I2V pipeline alive while processing JSONL requests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


RESULT_PREFIX = "__REMAKE_WAN22_RESULT__"


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--task", default="i2v-A14B")
    parser.add_argument("--offload-model", type=_bool, default=False)
    parser.add_argument("--convert-model-dtype", type=_bool, default=True)
    parser.add_argument("--t5-cpu", type=_bool, default=False)
    return parser


def _emit(payload: dict[str, Any]) -> None:
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    args = _parser().parse_args()
    repo = args.wan_repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stderr)],
        force=True,
    )

    import torch
    from PIL import Image

    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    if args.task != "i2v-A14B":
        raise RuntimeError(f"persistent reference worker only supports i2v-A14B, got {args.task}")
    cfg = WAN_CONFIGS[args.task]
    logging.info("Loading one persistent Wan2.2 I2V pipeline from %s", args.ckpt_dir)
    pipeline = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info("Persistent Wan2.2 worker is ready")

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request: dict[str, Any] = {}
        started = time.time()
        try:
            request = json.loads(raw_line)
            job_id = str(request["job_id"])
            output = Path(request["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.unlink(missing_ok=True)
            size = str(request["size"])
            if size not in MAX_AREA_CONFIGS:
                raise ValueError(f"unsupported Wan2.2 size: {size}")
            with Image.open(request["image"]) as opened_image:
                image = opened_image.convert("RGB")
            shift = request.get("sample_shift")
            guide_scale = request.get("sample_guide_scale")
            logging.info("Generating %s -> %s", job_id, output)
            video = pipeline.generate(
                str(request["prompt"]),
                image,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=int(request["frame_num"]),
                shift=cfg.sample_shift if shift is None else float(shift),
                sample_solver=str(request.get("sample_solver", "unipc")),
                sampling_steps=int(request.get("sample_steps") or cfg.sample_steps),
                guide_scale=cfg.sample_guide_scale if guide_scale is None else float(guide_scale),
                seed=int(request["seed"]),
                offload_model=args.offload_model,
            )
            save_video(
                tensor=video[None],
                save_file=str(output),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            del video
            torch.cuda.synchronize()
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("Wan2.2 returned without producing a non-empty video")
            _emit(
                {
                    "ok": True,
                    "job_id": job_id,
                    "output": str(output),
                    "elapsed_seconds": time.time() - started,
                    "offload_model": args.offload_model,
                }
            )
        except Exception as exc:
            output_value = request.get("output")
            if output_value:
                Path(output_value).unlink(missing_ok=True)
            traceback.print_exc(file=sys.stderr)
            _emit(
                {
                    "ok": False,
                    "job_id": request.get("job_id"),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": time.time() - started,
                }
            )


if __name__ == "__main__":
    main()
