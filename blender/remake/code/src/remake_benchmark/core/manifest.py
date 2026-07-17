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


def _generation_config(experiment: dict[str, Any]) -> dict[str, Any]:
    generation = experiment["generation"]
    required = ("width", "height", "num_frames", "fps")
    if not isinstance(generation, dict) or any(key not in generation for key in required):
        raise ConfigError(f"experiment.generation requires {required}")
    return {
        "width": int(generation["width"]),
        "height": int(generation["height"]),
        "num_frames": int(generation["num_frames"]),
        "fps": float(generation["fps"]),
    }


def _input_image(
    workspace_root: Path,
    input_root: Path,
    experiment_id: str,
    scene_id: str,
    object_id: str,
    camera: str,
    *,
    check_inputs: bool,
) -> tuple[Path, Path]:
    relative = input_root / experiment_id / scene_id / object_id / f"{camera}.png"
    absolute = workspace_root / relative
    if check_inputs and not absolute.is_file():
        raise ConfigError(f"conditioning image not found: {absolute}")
    return relative, absolute


def _build_directory_grid_jobs(
    resolved: dict[str, Any], *, check_inputs: bool, max_jobs: int | None
) -> list[dict[str, Any]]:
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

    generation = _generation_config(experiment)

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

        image_relative, _image_path = _input_image(
            workspace_root,
            input_root,
            str(experiment["experiment_id"]),
            str(scene_id),
            str(object_id),
            str(camera),
            check_inputs=check_inputs,
        )
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
            "generation": dict(generation),
            "evaluation_tags": list(experiment.get("evaluation_tags", [])),
        }
        validate_job(job)
        jobs.append(job)
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
    if not jobs:
        raise ConfigError("build generated zero jobs")
    return jobs


def _registry_experiments(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    configured = experiment.get("experiments")
    if not isinstance(configured, list) or not configured:
        raise ConfigError("registry_exhaustive_v1 requires a non-empty experiment.experiments list")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, item in enumerate(configured):
        if not isinstance(item, dict):
            raise ConfigError(f"experiment.experiments[{index}] must be an object")
        experiment_id = str(item.get("id", ""))
        if not experiment_id:
            raise ConfigError(f"experiment.experiments[{index}].id must be non-empty")
        if experiment_id in seen:
            raise ConfigError(f"duplicate registry experiment id: {experiment_id}")
        anchors = item.get("anchor_tuples")
        parameters = item.get("hidden_parameters")
        if not isinstance(anchors, list) or not anchors:
            raise ConfigError(f"registry experiment {experiment_id} has no anchor_tuples")
        if not isinstance(parameters, list) or not parameters:
            raise ConfigError(f"registry experiment {experiment_id} has no hidden_parameters")
        seen.add(experiment_id)
        result.append(item)
    return result


def _anchor_targets(
    experiment_id: str, anchor: dict[str, Any], parameter_names: list[str]
) -> tuple[str, dict[str, float]]:
    variant_id = str(anchor.get("id", ""))
    if not variant_id:
        raise ConfigError(f"registry experiment {experiment_id} has an anchor without id")
    missing = [name for name in parameter_names if name not in anchor]
    extra = [name for name in anchor if name != "id" and name not in parameter_names]
    if missing or extra:
        raise ConfigError(
            f"registry anchor {experiment_id}/{variant_id} parameter mismatch; missing={missing}, extra={extra}"
        )
    try:
        targets = {name: float(anchor[name]) for name in parameter_names}
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"registry anchor {experiment_id}/{variant_id} contains a non-numeric target") from exc
    return variant_id, targets


def _selected_anchor_ids(
    selection: dict[str, Any], registry: list[dict[str, Any]]
) -> dict[str, set[str]] | None:
    configured = selection.get("parameter_tuple_ids")
    if configured is None:
        return None
    if not isinstance(configured, dict) or not configured:
        raise ConfigError("selection.parameter_tuple_ids must be a non-empty object")

    registry_by_id = {str(item["id"]): item for item in registry}
    configured_by_id = {str(key): value for key, value in configured.items()}
    missing_experiments = sorted(set(registry_by_id) - set(configured_by_id))
    unknown_experiments = sorted(set(configured_by_id) - set(registry_by_id))
    if missing_experiments or unknown_experiments:
        raise ConfigError(
            "selection.parameter_tuple_ids must cover every registry experiment; "
            f"missing={missing_experiments}, unknown={unknown_experiments}"
        )

    result: dict[str, set[str]] = {}
    for experiment_id, registry_experiment in registry_by_id.items():
        raw_ids = configured_by_id.get(experiment_id)
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ConfigError(
                f"selection.parameter_tuple_ids.{experiment_id} must be a non-empty list"
            )
        selected = [str(value) for value in raw_ids]
        if len(selected) != len(set(selected)):
            raise ConfigError(
                f"selection.parameter_tuple_ids.{experiment_id} contains duplicate ids"
            )
        available = {str(anchor.get("id", "")) for anchor in registry_experiment["anchor_tuples"]}
        unknown = sorted(set(selected) - available)
        if unknown:
            raise ConfigError(
                f"selection.parameter_tuple_ids.{experiment_id} contains unknown ids: {unknown}"
            )
        result[experiment_id] = set(selected)
    return result


def _registry_coverage_blocks(
    selection: dict[str, Any],
    registry: list[dict[str, Any]],
    cameras: list[str],
    scenes: list[str],
    seeds: list[int],
) -> list[tuple[list[str], dict[str, set[str]] | None, list[str], list[int]]]:
    configured = selection.get("coverage_blocks")
    if configured is None:
        return [(cameras, _selected_anchor_ids(selection, registry), scenes, seeds)]
    if selection.get("parameter_tuple_ids") is not None:
        raise ConfigError(
            "selection cannot define both parameter_tuple_ids and coverage_blocks"
        )
    if not isinstance(configured, list) or not configured:
        raise ConfigError("selection.coverage_blocks must be a non-empty list")

    selected_cameras = set(cameras)
    selected_scenes = set(scenes)
    covered_cameras: set[str] = set()
    covered_cases: set[tuple[str, str, str, str, int]] = set()
    blocks: list[tuple[list[str], dict[str, set[str]] | None, list[str], list[int]]] = []
    for index, block in enumerate(configured):
        if not isinstance(block, dict):
            raise ConfigError(f"selection.coverage_blocks[{index}] must be an object")
        raw_cameras = block.get("cameras")
        if not isinstance(raw_cameras, list) or not raw_cameras:
            raise ConfigError(
                f"selection.coverage_blocks[{index}].cameras must be a non-empty list"
            )
        block_cameras = [str(value) for value in raw_cameras]
        if len(block_cameras) != len(set(block_cameras)):
            raise ConfigError(f"selection.coverage_blocks[{index}].cameras contains duplicates")
        unknown_cameras = sorted(set(block_cameras) - selected_cameras)
        if unknown_cameras:
            raise ConfigError(
                f"selection.coverage_blocks[{index}] contains cameras outside selection.cameras: "
                f"{unknown_cameras}"
            )
        raw_scenes = block.get("scenes", scenes)
        if not isinstance(raw_scenes, list) or not raw_scenes:
            raise ConfigError(
                f"selection.coverage_blocks[{index}].scenes must be a non-empty list"
            )
        block_scenes = [str(value) for value in raw_scenes]
        if len(block_scenes) != len(set(block_scenes)):
            raise ConfigError(f"selection.coverage_blocks[{index}].scenes contains duplicates")
        unknown_scenes = sorted(set(block_scenes) - selected_scenes)
        if unknown_scenes:
            raise ConfigError(
                f"selection.coverage_blocks[{index}] contains scenes outside selection.scenes: "
                f"{unknown_scenes}"
            )
        raw_seeds = block.get("seeds", seeds)
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ConfigError(
                f"selection.coverage_blocks[{index}].seeds must be a non-empty list"
            )
        try:
            block_seeds = [int(value) for value in raw_seeds]
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"selection.coverage_blocks[{index}].seeds contains a non-integer value"
            ) from exc
        if len(block_seeds) != len(set(block_seeds)):
            raise ConfigError(f"selection.coverage_blocks[{index}].seeds contains duplicates")
        block_anchor_ids = _selected_anchor_ids(block, registry)
        if block_anchor_ids is None:
            raise ConfigError(
                f"selection.coverage_blocks[{index}] requires parameter_tuple_ids"
            )
        for camera in block_cameras:
            for experiment_id, anchor_ids in block_anchor_ids.items():
                for anchor_id in anchor_ids:
                    for scene_id in block_scenes:
                        for seed in block_seeds:
                            case = (camera, experiment_id, anchor_id, scene_id, seed)
                            if case in covered_cases:
                                raise ConfigError(
                                    "selection.coverage_blocks overlap at "
                                    f"camera={camera}, experiment={experiment_id}, "
                                    f"tuple={anchor_id}, scene={scene_id}, seed={seed}"
                                )
                            covered_cases.add(case)
        covered_cameras.update(block_cameras)
        blocks.append((block_cameras, block_anchor_ids, block_scenes, block_seeds))

    missing_cameras = sorted(selected_cameras - covered_cameras)
    if missing_cameras:
        raise ConfigError(
            f"selection.coverage_blocks leave selected cameras uncovered: {missing_cameras}"
        )
    return blocks


def _build_registry_exhaustive_jobs(
    resolved: dict[str, Any], *, check_inputs: bool, max_jobs: int | None
) -> list[dict[str, Any]]:
    experiment = resolved["experiment"]
    prompt_profile = resolved["prompt"]
    workspace_root = Path(resolved["workspace_root"])
    input_root = Path(experiment["input_root"])
    if input_root.is_absolute():
        raise ConfigError("experiment.input_root must be relative to workspace_root")

    selection = experiment["selection"]
    scenes, scene_selection_seed = _resolve_scenes(selection)
    objects = [str(value) for value in _as_nonempty_list(selection, "objects")]
    cameras = [str(value) for value in _as_nonempty_list(selection, "cameras")]
    seeds = [int(value) for value in _as_nonempty_list(selection, "seeds")]
    generation = _generation_config(experiment)
    registry = _registry_experiments(experiment)
    coverage_blocks = _registry_coverage_blocks(
        selection, registry, cameras, scenes, seeds
    )

    jobs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for registry_experiment in registry:
        experiment_id = str(registry_experiment["id"])
        parameters = registry_experiment["hidden_parameters"]
        parameter_names: list[str] = []
        units: dict[str, str] = {}
        for parameter in parameters:
            if not isinstance(parameter, dict) or not parameter.get("name"):
                raise ConfigError(f"registry experiment {experiment_id} has an invalid hidden parameter")
            name = str(parameter["name"])
            parameter_names.append(name)
            units[name] = str(parameter.get("unit", ""))
        if len(parameter_names) != len(set(parameter_names)):
            raise ConfigError(f"registry experiment {experiment_id} has duplicate hidden parameter names")

        known_params = {
            "descriptions": [str(value) for value in registry_experiment.get("known_parameters", [])]
        }
        level = int(registry_experiment.get("level", 0))
        for raw_anchor in registry_experiment["anchor_tuples"]:
            if not isinstance(raw_anchor, dict):
                raise ConfigError(f"registry experiment {experiment_id} contains a non-object anchor")
            variant_id, targets = _anchor_targets(experiment_id, raw_anchor, parameter_names)
            for block_cameras, block_anchor_ids, block_scenes, block_seeds in coverage_blocks:
                if block_anchor_ids is not None and variant_id not in block_anchor_ids[experiment_id]:
                    continue
                combinations = itertools.product(
                    block_scenes, objects, block_cameras, block_seeds
                )
                for scene_id, object_id, camera, seed in combinations:
                    case_id = _safe_id(
                        f"{experiment_id}__{variant_id}__{scene_id}__{object_id}__{camera}"
                    )
                    repeat_id = f"seed-{seed}"
                    job_id = _safe_id(f"{case_id}__{repeat_id}")
                    if job_id in seen_ids:
                        raise ConfigError(f"duplicate job_id generated: {job_id}")
                    seen_ids.add(job_id)
                    image_relative, _image_path = _input_image(
                        workspace_root,
                        input_root,
                        experiment_id,
                        scene_id,
                        object_id,
                        camera,
                        check_inputs=check_inputs,
                    )
                    prompt_spec, prompt, negative_prompt = render_prompt(
                        prompt_profile,
                        experiment_id=experiment_id,
                        scene_id=scene_id,
                        object_id=object_id,
                        camera=camera,
                        targets=targets,
                    )
                    job = {
                        "schema_version": "1.0.0",
                        "job_id": job_id,
                        "build_id": resolved["build_id"],
                        "release_id": str(experiment["release_id"]),
                        "experiment_id": experiment_id,
                        "case_id": case_id,
                        "repeat_id": repeat_id,
                        "task_type": str(experiment["task_type"]),
                        "seed": seed,
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
                        "known_params": known_params,
                        "factors": {
                            "level": level,
                            "parameter_tuple_id": variant_id,
                            "scene_id": scene_id,
                            "object_id": object_id,
                            "camera": camera,
                            "scene_selection_seed": scene_selection_seed,
                            "target_assignment": "registry_anchor_tuples",
                        },
                        "units": units,
                        "generation": dict(generation),
                        "evaluation_tags": list(experiment.get("evaluation_tags", [])),
                    }
                    validate_job(job)
                    jobs.append(job)
                    if max_jobs is not None and len(jobs) >= max_jobs:
                        return jobs
    if not jobs:
        raise ConfigError("build generated zero jobs")
    return jobs


def build_jobs(
    resolved: dict[str, Any], *, check_inputs: bool = True, max_jobs: int | None = None
) -> list[dict[str, Any]]:
    builder = str(resolved["experiment"].get("builder", ""))
    if builder == "directory_grid_v1":
        return _build_directory_grid_jobs(resolved, check_inputs=check_inputs, max_jobs=max_jobs)
    if builder == "registry_exhaustive_v1":
        return _build_registry_exhaustive_jobs(resolved, check_inputs=check_inputs, max_jobs=max_jobs)
    raise ConfigError(f"unsupported experiment builder: {builder!r}")
