from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from baseline_v1_prompts import COMMON_NEGATIVE_PROMPT, PROMPT_STYLES, build_prompt

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "wan2.1_t2v_1.3b": {
        "hf_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "width": 832,
        "height": 480,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "guidance_scale": 6.0,
        "guidance_scale_2": None,
    },
    "wan2.2_ti2v_5b": {
        "hf_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
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
        "width": 720,
        "height": 480,
        "num_frames": 49,
        "fps": 8,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "guidance_scale_2": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest for Baseline V1 generation with multiple models, prompt styles, and seeds."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest path.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["wan2.2_ti2v_5b"],
        choices=sorted(MODEL_SPECS.keys()),
        help="Model aliases to include in the manifest.",
    )
    parser.add_argument(
        "--prompt-styles",
        nargs="+",
        default=["structured_raw"],
        choices=PROMPT_STYLES,
        help="Prompt styles to include in the manifest.",
    )
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1, help="Replicate each model/style combination across N seeds.")
    return parser.parse_args()


def load_config(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Config must be a JSON list of experiment objects.")
    return data


def main() -> None:
    args = parse_args()
    experiments = load_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seed = args.seed_base
    with args.output.open("w", encoding="utf-8") as f:
        for exp in experiments:
            exp_id = exp["id"]
            task = exp["task"]
            params = exp["parameters"]
            negative_prompt = exp.get("negative_prompt", COMMON_NEGATIVE_PROMPT)
            chosen_models = exp.get("models", args.models)
            chosen_styles = exp.get("prompt_styles", args.prompt_styles)

            for model_name in chosen_models:
                if model_name not in args.models:
                    continue

                spec = MODEL_SPECS[model_name]

                for prompt_style in chosen_styles:
                    if prompt_style not in args.prompt_styles:
                        continue

                    prompt = build_prompt(task, params, prompt_style=prompt_style)

                    for rep in range(args.num_seeds):
                        this_seed = exp.get("seed", seed)
                        init_image = None
                        if model_name == "wan2.2_ti2v_5b":
                            init_image = exp.get("init_image")

                        merged = {
                            "id": f"{exp_id}__{prompt_style}__{model_name}__s{this_seed}",
                            "base_id": exp_id,
                            "task": task,
                            "parameters": params,
                            "model": model_name,
                            "model_id": spec["hf_id"],
                            "prompt_style": prompt_style,
                            "prompt": prompt,
                            "negative_prompt": negative_prompt,
                            "width": exp.get("width", spec["width"]),
                            "height": exp.get("height", spec["height"]),
                            "num_frames": exp.get("num_frames", spec["num_frames"]),
                            "fps": exp.get("fps", spec["fps"]),
                            "num_inference_steps": exp.get("num_inference_steps", spec["num_inference_steps"]),
                            "guidance_scale": exp.get("guidance_scale", spec["guidance_scale"]),
                            "guidance_scale_2": exp.get("guidance_scale_2", spec["guidance_scale_2"]),
                            "seed": this_seed,
                            "output_subdir": exp.get("output_subdir", task),
                            "metadata": exp.get("metadata", {}),
                            "init_image": init_image,
                        }
                        f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                        seed = max(seed + 1, this_seed + 1)

    print(f"Wrote manifest to {args.output}")


if __name__ == "__main__":
    main()
