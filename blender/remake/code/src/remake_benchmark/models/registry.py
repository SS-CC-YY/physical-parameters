from __future__ import annotations

from pathlib import Path
from typing import Any

from remake_benchmark.core.errors import ConfigError

from .base import ModelAdapter
from .closed_api import KlingApiAdapter, SeedanceArkApiAdapter
from .wan22 import Wan22Adapter


ADAPTERS: dict[str, type[ModelAdapter]] = {
    Wan22Adapter.adapter_id: Wan22Adapter,
    SeedanceArkApiAdapter.adapter_id: SeedanceArkApiAdapter,
    KlingApiAdapter.adapter_id: KlingApiAdapter,
}


def get_adapter(model_config: dict[str, Any], workspace_root: Path) -> ModelAdapter:
    adapter_id = str(model_config.get("adapter", ""))
    adapter_type = ADAPTERS.get(adapter_id)
    if adapter_type is None:
        available = ", ".join(sorted(ADAPTERS))
        raise ConfigError(f"unknown model adapter {adapter_id!r}; available: {available}")
    return adapter_type(model_config, workspace_root)
