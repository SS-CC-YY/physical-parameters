from __future__ import annotations

import itertools
import random
import re
from pathlib import Path
from typing import Any

from remake_benchmark.prompts import render_prompt

from .errors import ConfigError
from .schema import validate_job


def _slug_number(value: float) -> str:
    rendered = f"{value:.8f}".rstrip("0").rstrip(".")
    return rendered.replace("-", "m").replace(".", "p")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")


def _as_nonempty_list(selection: dict[str, Any], key: str) -> list[Any]:
    value = selection.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"experiment selection.{key} must be a non-empty list")
    return value


def _resolve_scenes(selection: dict[str, Any]) -> tuple[list[str], int | None]:
    direct = selection.get("scenes")
    if direct is not None:
        return [str(value) for value in _as_nonempty_list(selection, "scenes")], None
    sampling = selection.get("random_scene_sampling")
    if not isinstance(sampling, dict):
        raise ConfigError("selection requires either 'scenes' or 'random_scene_sampling'")
    baseline = [str(value) for value in sampling.get("always_include", ["baseline"])]
    indoor_candidates = sampling.get("indoor_candidates")
    outdoor_candidates = sampling.get("outdoor_candidates")
    if not isinstance(indoor_candidates, list) or not indoor_candidates:
        raise ConfigError("random_scene_sampling.indoor_candidates must be a non-empty list")
    if not isinstance(outdoor_candidates, list) or not outdoor_candidates:
        raise ConfigError("random_scene_sampling.outdoor_candidates must be a non-empty list")
    indoor_count = int(sampling.get("indoor_count", 2))
    outdoor_count = int(sampling.get("outdoor_count", 2))
    if not 0 < indoor_count <= len(indoor_candidates):
        raise ConfigError("random_scene_sampling.indoor_count is out of range")
    if not 0 < outdoor_count <= len(outdoor_candidates):
        raise ConfigError("random_scene_sampling.outdoor_count is out of range")
    selection_seed = int(sampling.get("seed", 36))
    rng = random.Random(selection_seed)
    indoors = rng.sample([str(value) for value in indoor_candidates], indoor_count)
    outdoors = rng.sample([str(value) for value in outdoor_candidates], outdoor_count)
    scenes = baseline + indoors + outdoors
    if len(scenes) != len(set(scenes)):
        raise ConfigError(f"scene sampling produced duplicate scene ids: {scenes}")
    return scenes, selection_seed


def build_jobs(resolved: dict[str, Any], *, check_inputs: bool = True, max_jobs: int | None = None) -> list[dict[str, Any]]:
    experiment = resolved["experiment"]
    prompt_profile = resolved["prompt"]
    workspace_root = Path(resolved["workspace_root"])
    input_root = Path(experiment["input_root"])
    if input_root.is_absolute():
        raise ConfigError("experiment.input_root must be relative to workspace_root")

    selection = experiment["selection"]
    scenes, scene_selection_seed = _resolve_scenes(selection)
    objects = _as_nonempty_list(selection, "objects")
    cameras = _as_nonempty_list(selection, "cameras")
    seeds = _as_nonempty_list(selection, "seeds")
    targets_config = experiment["targets"]
    if not isinstance(targets_config, dict) or not targets_config:
        raise ConfigError("experiment.targets must be a non-empty object")
    target_names = list(targets_config)
    target_values: list[list[float]] = []
    units: dict[str, str] = {}
    for name, config in targets_config.items():
        if not isinstance(config, dict) or not isinstance(config.get("values"), list) or not config["values"]:
            raise ConfigError(f"experiment.targets.{name}.values must be a non-empty list")
        target_values.append([float(value) for value in config["values"]])
        units[name] = str(config.get("unit", ""))

    generation = experiment["generation"]
    required_generation = ("width", "height", "num_frames", "fps")
    if any(key not in generation for key in required_generation):
        raise ConfigError(f"experiment.generation requires {required_generation}")

    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    assignment = str(selection.get("target_assignment", "all_values"))
    if assignment == "object_values":
        if len(target_names) != 1:
            raise ConfigError("target_assignment=object_values currently requires exactly one target parameter")
        if len(target_values[0]) != len(objects):
            raise ConfigError("target_assignment=object_values requires one target value per object")
        object_targets = [(object_id, (target_values[0][index],)) for index, object_id in enumerate(objects)]
    elif assignment == "all_values":
        object_targets = list(itertools.product(objects, itertools.product(*target_values)))
    else:
        raise ConfigError(f"unsupported selection.target_assignment: {assignment}")

    combinations = itertools.product(scenes, object_targets, cameras, seeds)
    for scene_id, object_target, camera, seed in combinations:
        object_id, target_tuple = object_target
        targets = dict(zip(target_names, target_tuple))
        target_id = "_".join(f"{name}-{_slug_number(value)}" for name, value in targets.items())
        case_id = _safe_id(f"{experiment['experiment_id']}__{target_id}__{scene_id}__{object_id}__{camera}")
        repeat_id = f"seed-{int(seed)}"
        job_id = _safe_id(f"{case_id}__{repeat_id}")
        if job_id in seen_ids:
            raise ConfigError(f"duplicate job_id generated: {job_id}")
        seen_ids.add(job_id)

        image_relative = input_root / str(experiment["experiment_id"]) / str(scene_id) / str(object_id) / f"{camera}.png"
        image_path = workspace_root / image_relative
        if check_inputs and not image_path.is_file():
            raise ConfigError(f"conditioning image not found: {image_path}")
        prompt_spec, prompt, negative_prompt = render_prompt(
            prompt_profile,
            experiment_id=str(experiment["experiment_id"]),
            scene_id=str(scene_id),
            object_id=str(object_id),
            camera=str(camera),
            targets=targets,
        )
        job = {
            "schema_version": "1.0.0",
            "job_id": job_id,
            "build_id": resolved["build_id"],
            "release_id": str(experiment["release_id"]),
            "experiment_id": str(experiment["experiment_id"]),
            "case_id": case_id,
            "repeat_id": repeat_id,
            "task_type": str(experiment["task_type"]),
            "seed": int(seed),
            "inputs": {
                "image": image_relative.as_posix(),
                "source_fps": float(experiment.get("source_fps", 24)),
                "conditioning_frame_index": int(experiment.get("conditioning_frame_index", 1)),
            },
            "prompt_spec": prompt_spec,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "prompt_mode": str(prompt_profile["mode"]),
            "prompt_template_id": str(prompt_profile["prompt_template_id"]),
            "prompt_template_version": str(prompt_profile["version"]),
            "targets": targets,
            "known_params": dict(experiment.get("known_params", {})),
            "factors": {
                "scene_id": scene_id,
                "object_id": object_id,
                "camera": camera,
                "scene_selection_seed": scene_selection_seed,
                "target_assignment": assignment,
            },
            "units": units,
            "generation": {
                "width": int(generation["width"]),
                "height": int(generation["height"]),
                "num_frames": int(generation["num_frames"]),
                "fps": float(generation["fps"]),
            },
            "evaluation_tags": list(experiment.get("evaluation_tags", [])),
        }
        validate_job(job)
        jobs.append(job)
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
    if not jobs:
        raise ConfigError("build generated zero jobs")
    return jobs
