#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from i2v_freefall_prompts import freefall_i2v_prompt

MODEL_DEFAULTS = {
    "wan2.2_ti2v_5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "guidance_scale_2": None,
    }
}

def make_job_id(base_id, axis, value, style, model_name, seed):
    token = str(value).replace("-", "m").replace(".", "p")
    return f"{base_id}__axis_{axis}_{token}__{style}__{model_name}__s{seed}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="wan2.2_ti2v_5b")
    ap.add_argument("--prompt-styles", nargs="+", default=["structured_detailed"])
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = MODEL_DEFAULTS[args.model]

    base_id = cfg["id"]
    init_image = cfg["init_image"]
    base_params = cfg["base_parameters"]
    sweeps = cfg["sweeps"]

    jobs = []
    for axis_name, axis_values in sweeps.items():
        if axis_name not in ("g", "v0", "h0"):
            continue
        for val in axis_values:
            params = dict(base_params)
            params[axis_name] = val
            for style in args.prompt_styles:
                prompt = freefall_i2v_prompt(params, style)
                for s in range(args.num_seeds):
                    seed = args.seed_base + s
                    job_id = make_job_id(base_id, axis_name, val, style, args.model, seed)
                    jobs.append({
                        "id": job_id,
                        "task": "free_fall_i2v",
                        "axis_name": axis_name,
                        "axis_value": val,
                        "base_id": base_id,
                        "prompt_style": style,
                        "model": args.model,
                        "model_id": spec["model_id"],
                        "prompt": prompt,
                        "negative_prompt": None,
                        "init_image": init_image,
                        "parameters": params,
                        "width": spec["width"],
                        "height": spec["height"],
                        "num_frames": spec["num_frames"],
                        "fps": spec["fps"],
                        "num_inference_steps": spec["num_inference_steps"],
                        "guidance_scale": spec["guidance_scale"],
                        "guidance_scale_2": spec["guidance_scale_2"],
                        "seed": seed,
                    })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    print(f"saved: {out_path}")
    print(f"jobs: {len(jobs)}")

if __name__ == "__main__":
    main()
