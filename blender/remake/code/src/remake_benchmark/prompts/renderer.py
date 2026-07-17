from __future__ import annotations

from typing import Any

from remake_benchmark.core.errors import ConfigError


class _StrictFormat(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise ConfigError(f"prompt template references an unknown field: {key}")


def render_prompt(
    profile: dict[str, Any],
    *,
    experiment_id: str,
    scene_id: str,
    object_id: str,
    camera: str,
    targets: dict[str, float],
) -> tuple[dict[str, Any], str, str]:
    object_descriptions = profile.get("object_descriptions", {})
    object_description = object_descriptions.get(object_id, object_id.replace("_", " "))
    values: dict[str, Any] = {
        "experiment_id": experiment_id,
        "scene_id": scene_id,
        "object_id": object_id,
        "object_description": object_description,
        "camera": camera,
        **targets,
    }
    configured_templates = profile["experiment_templates"]
    if all(key in configured_templates for key in ("dynamics", "parameter", "terminal")):
        experiment_templates = configured_templates
    else:
        experiment_templates = configured_templates.get(experiment_id)
        if not isinstance(experiment_templates, dict):
            raise ConfigError(f"prompt profile has no experiment templates for {experiment_id}")
    configured_constraints = profile["experiment_constraints"]
    if isinstance(configured_constraints, list):
        experiment_constraints = configured_constraints
    else:
        experiment_constraints = configured_constraints.get(experiment_id)
        if not isinstance(experiment_constraints, list):
            raise ConfigError(f"prompt profile has no experiment constraints for {experiment_id}")

    section_templates = {
        "task": profile["shared_templates"]["task"],
        "scene": profile["shared_templates"]["scene"],
        "camera": profile["shared_templates"]["camera"],
        "dynamics": experiment_templates["dynamics"],
        "parameter": experiment_templates["parameter"],
        "terminal": experiment_templates["terminal"],
        "quality": profile["shared_templates"]["quality"],
    }
    try:
        rendered_sections = {
            name: str(section_templates[name]).format_map(_StrictFormat(values)).strip()
            for name in profile["section_order"]
        }
        prompt = " ".join(rendered_sections[name] for name in profile["section_order"] if rendered_sections[name])
        negative_parts = [str(profile["negative_prompt"]).format_map(_StrictFormat(values)).strip()]
        additional_negative = str(profile.get("additional_negative_prompt", "")).format_map(_StrictFormat(values)).strip()
        if additional_negative:
            negative_parts.append(additional_negative)
        negative = ", ".join(part for part in negative_parts if part)
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"cannot render prompt profile {profile.get('prompt_template_id')}: {exc}") from exc
    prompt_spec = {
        "prompt_framework_id": profile["prompt_framework_id"],
        "framework_version": profile["framework_version"],
        "section_order": list(profile["section_order"]),
        "sections": rendered_sections,
        "experiment_id": experiment_id,
        "scene_id": scene_id,
        "object_id": object_id,
        "object_description": object_description,
        "camera": camera,
        "targets": targets,
        "constraints": list(profile["shared_constraints"]) + list(experiment_constraints),
    }
    return prompt_spec, prompt, negative
