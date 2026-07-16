from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Invocation:
    command: list[str]
    cwd: Path
    environment: dict[str, str] = field(default_factory=dict)
    output_video: Path | None = None


class ModelAdapter(ABC):
    adapter_id: str

    def __init__(self, model_config: dict[str, Any], workspace_root: Path):
        self.model_config = model_config
        self.workspace_root = workspace_root

    @abstractmethod
    def validate(self, job: dict[str, Any], *, dry_run: bool) -> None:
        """Validate configuration and inputs before launching a model."""

    @abstractmethod
    def build_invocation(self, job: dict[str, Any], output_video: Path) -> Invocation:
        """Translate a canonical job into one model-native invocation."""

    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return model/checkpoint/runtime information for metadata."""
