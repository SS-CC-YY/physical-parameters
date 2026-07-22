from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path


REMAKE_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = REMAKE_ROOT / "rebuild-test" / "spatialtrackerv2" / "scripts" / "preflight.py"


def load_preflight_module():
    spec = importlib.util.spec_from_file_location("spatialtracker_preflight_for_test", PREFLIGHT)
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


if __name__ == "__main__":
    unittest.main()
