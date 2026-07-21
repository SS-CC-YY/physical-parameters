"""Fast, auditable object-centric reconstruction primitives.

Public compatibility names are loaded lazily so a trajectory evaluator that
only needs NumPy/OpenCV does not import the legacy plotting stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MVP_EXPORTS = {
    "DEFAULT_CONFIG",
    "fit_freefall_physics",
    "parse_video_job",
    "run_reconstruction_batch",
    "run_reconstruction_job",
}
_V1A_EXPORTS = {
    "DEFAULT_V1A_CONFIG",
    "fit_freefall_metric",
    "parse_v1a_video_job",
    "run_v1a_metric_batch",
    "run_v1a_metric_job",
}


def __getattr__(name: str) -> Any:
    if name in _MVP_EXPORTS:
        return getattr(import_module(".mvp", __name__), name)
    if name == "choose_reconstruction_route":
        return getattr(import_module("remake_benchmark.hybrid"), name)
    if name in _V1A_EXPORTS:
        return getattr(import_module(".v1a_metric", __name__), name)
    raise AttributeError(name)


__all__ = [
    "DEFAULT_CONFIG",
    "fit_freefall_physics",
    "parse_video_job",
    "run_reconstruction_batch",
    "run_reconstruction_job",
    "choose_reconstruction_route",
    "DEFAULT_V1A_CONFIG",
    "fit_freefall_metric",
    "parse_v1a_video_job",
    "run_v1a_metric_batch",
    "run_v1a_metric_job",
]
