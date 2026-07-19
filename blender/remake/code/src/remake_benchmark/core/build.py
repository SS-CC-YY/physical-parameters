from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .io import read_yaml
from .schema import CODE_ROOT, validate_build, validate_prompt


PROFILE_NAMES = ("experiment", "prompt", "model", "evaluation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_path(reference: str) -> Path:
    path = (CODE_ROOT / reference).resolve()
    try:
        path.relative_to(CODE_ROOT.resolve())
    except ValueError as exc:
        raise ConfigError(f"profile reference leaves code root: {reference}") from exc
    return path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_prompt_profile(path: Path, seen: set[Path] | None = None) -> tuple[dict[str, Any], list[Path]]:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"cyclic prompt profile inheritance: {path}")
    seen.add(path)
    data = read_yaml(path)
    parent_reference = data.pop("extends", None)
    if parent_reference is None:
        return data, [path]
    if not isinstance(parent_reference, str):
        raise ConfigError(f"{path}: prompt 'extends' must be a code-root-relative path")
    parent_path = _profile_path(parent_reference)
    parent, chain = _read_prompt_profile(parent_path, seen)
    return _deep_merge(parent, data), chain + [path]


def _require_mapping(data: dict[str, Any], key: str, origin: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{origin}: '{key}' must be an object")
    return value


def _validate_components(resolved: dict[str, Any], origins: dict[str, Path]) -> None:
    experiment = resolved["experiment"]
    for key in ("release_id", "task_type", "builder", "input_root", "selection", "generation"):
        if key not in experiment:
            raise ConfigError(f"{origins['experiment']}: missing '{key}'")
    if experiment["task_type"] not in {"t2v", "i2v", "v2v"}:
        raise ConfigError(f"{origins['experiment']}: unsupported task_type {experiment['task_type']!r}")
    builder = str(experiment["builder"])
    if builder == "directory_grid_v1":
        for key in ("experiment_id", "targets"):
            if key not in experiment:
                raise ConfigError(f"{origins['experiment']}: missing '{key}' for {builder}")
    elif builder == "registry_exhaustive_v1":
        if not isinstance(experiment.get("experiments"), list) or not experiment["experiments"]:
            raise ConfigError(f"{origins['experiment']}: 'experiments' must be a non-empty list for {builder}")
    else:
        raise ConfigError(f"{origins['experiment']}: unsupported builder {experiment['builder']!r}")

    validate_prompt(resolved["prompt"])

    model = resolved["model"]
    for key in ("model_id", "adapter", "capabilities", "runtime", "generation"):
        if key not in model:
            raise ConfigError(f"{origins['model']}: missing '{key}'")
    if experiment["task_type"] not in model["capabilities"]:
        raise ConfigError(
            f"model {model['model_id']} does not support experiment task_type {experiment['task_type']}"
        )
    _require_mapping(model, "runtime", origins["model"])
    _require_mapping(model, "generation", origins["model"])
    safety = model.get("safety")
    if safety is not None:
        if not isinstance(safety, dict):
            raise ConfigError(f"{origins['model']}: 'safety' must be an object")
        for key in ("max_billable_jobs_per_run", "max_concurrent_jobs"):
            maximum = safety.get(key)
            if maximum is not None and (
                not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0
            ):
                raise ConfigError(f"{origins['model']}: safety.{key} must be a positive integer")

    evaluation = resolved["evaluation"]
    if not isinstance(evaluation.get("evaluators"), list) or not evaluation["evaluators"]:
        raise ConfigError(f"{origins['evaluation']}: 'evaluators' must be a non-empty list")


def resolve_build(build_path: Path, workspace_root_override: Path | None = None) -> dict[str, Any]:
    build_path = build_path.resolve()
    source = read_yaml(build_path)
    validate_build(source)

    resolved: dict[str, Any] = {
        "schema_version": source["schema_version"],
        "build_id": source["build_id"],
        "description": source.get("description", ""),
        "workflow": source.get("workflow", "generation_only"),
        "workspace_root": str(
            workspace_root_override.resolve() if workspace_root_override else Path(source["workspace_root"]).expanduser()
        ),
        "output_root": source["output_root"],
    }
    origins: dict[str, Path] = {}
    profile_hashes: dict[str, str] = {}
    profile_files: dict[str, str] = {}
    prompt_dependencies: list[dict[str, str]] = []
    for name in PROFILE_NAMES:
        path = _profile_path(source["profiles"][name])
        origins[name] = path
        if name == "prompt":
            resolved[name], dependency_paths = _read_prompt_profile(path)
            required_prompt_base = _profile_path("configs/prompts/common_physics_video_v1.yaml")
            if required_prompt_base not in dependency_paths:
                raise ConfigError(
                    f"prompt profile must inherit the common framework: {required_prompt_base}"
                )
            prompt_dependencies = [{"file": str(item), "sha256": _sha256(item)} for item in dependency_paths]
        else:
            resolved[name] = read_yaml(path)
        profile_hashes[name] = _sha256(path)
        profile_files[name] = str(path)

    overrides = source.get("overrides", {})
    for name, override in overrides.items():
        resolved[name] = _deep_merge(resolved[name], override)
    resolved["applied_overrides"] = overrides

    _validate_components(resolved, origins)
    resolved["provenance"] = {
        "build_file": str(build_path),
        "build_sha256": _sha256(build_path),
        "profile_files": profile_files,
        "profile_sha256": profile_hashes,
        "prompt_dependencies": prompt_dependencies,
    }
    return resolved
