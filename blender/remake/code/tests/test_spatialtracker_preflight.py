from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


REMAKE_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = REMAKE_ROOT / "rebuild-test" / "spatialtrackerv2" / "scripts" / "preflight.py"
CUDA_BOOTSTRAP = PREFLIGHT.parent / "cuda_bootstrap.py"
SPATIALTRACKER_SESSION = PREFLIGHT.parent / "spatialtracker_session.py"


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("spatialtracker_preflight_for_test", PREFLIGHT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    scripts_dir = str(PREFLIGHT.parent)
    sys.path.insert(0, scripts_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts_dir)
    return module


def load_cuda_bootstrap_module():
    spec = importlib.util.spec_from_file_location("spatialtracker_cuda_bootstrap_for_test", CUDA_BOOTSTRAP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DynamicAttributeModule(types.ModuleType):
    def __getattr__(self, name: str):
        raise ModuleNotFoundError(f"No module named '{self.__name__}.{name}'")


class SpatialTrackerPreflightTests(unittest.TestCase):
    def test_version_probe_does_not_invoke_dynamic_module_getattr(self) -> None:
        preflight = load_preflight_module()
        dependency = DynamicAttributeModule("utils3d")

        self.assertEqual(preflight.module_version(dependency), "installed")

    def test_eager_module_version_is_reported(self) -> None:
        preflight = load_preflight_module()
        dependency = types.ModuleType("dependency")
        dependency.__version__ = "1.2.3"

        self.assertEqual(preflight.module_version(dependency), "1.2.3")

    def test_cuda_is_initialized_before_properties_are_queried(self) -> None:
        bootstrap = load_cuda_bootstrap_module()
        calls: list[str] = []

        class Properties:
            name = "NVIDIA H20"

        class Cuda:
            @staticmethod
            def is_available() -> bool:
                calls.append("is_available")
                return True

            @staticmethod
            def init() -> None:
                calls.append("init")

            @staticmethod
            def current_device() -> int:
                calls.append("current_device")
                return 0

            @staticmethod
            def get_device_properties(index: int) -> Properties:
                calls.append(f"get_device_properties:{index}")
                return Properties()

        fake_torch = types.SimpleNamespace(cuda=Cuda())
        result = bootstrap.initialize_cuda_before_xformers(fake_torch)

        self.assertEqual(
            calls,
            ["is_available", "init", "current_device", "get_device_properties:0"],
        )
        self.assertEqual(result, {"logical_device_index": 0, "device_name": "NVIDIA H20"})

    def test_session_bootstraps_cuda_before_predictor_import(self) -> None:
        source = SPATIALTRACKER_SESSION.read_text(encoding="utf-8")

        self.assertLess(
            source.index("cuda_bootstrap = initialize_cuda_before_xformers(torch)"),
            source.index("from models.SpaTrackV2.models.predictor import Predictor"),
        )
        self.assertLess(
            source.index("class SpatialTrackerSession:"),
            source.index("cuda_bootstrap = initialize_cuda_before_xformers(torch)"),
        )


if __name__ == "__main__":
    unittest.main()
