from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .errors import ConfigError, JobValidationError
from .io import read_json


CODE_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_ROOT = CODE_ROOT / "schemas"


def _validate(instance: Any, schema_name: str, error_type: type[Exception]) -> None:
    schema = read_json(SCHEMA_ROOT / schema_name)
    errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda item: list(item.path))
    if not errors:
        return
    messages = []
    for error in errors[:12]:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")
    raise error_type("schema validation failed:\n- " + "\n- ".join(messages))


def validate_build(data: dict[str, Any]) -> None:
    _validate(data, "build.schema.json", ConfigError)


def validate_job(data: dict[str, Any]) -> None:
    _validate(data, "job.schema.json", JobValidationError)


def validate_prompt(data: dict[str, Any]) -> None:
    _validate(data, "prompt.schema.json", ConfigError)
