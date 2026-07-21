from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CODE_ROOT / "src" / "remake_benchmark" / "hybrid.py"
SPEC = importlib.util.spec_from_file_location("hybrid_reconstruction_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
choose_reconstruction_route = MODULE.choose_reconstruction_route


class HybridRoutingTests(unittest.TestCase):
    def test_only_positive_matching_fixed_evidence_uses_calibration(self) -> None:
        route = choose_reconstruction_route(
            {
                "filename": "sample.mp4",
                "status": "ok",
                "final_category": "no_significant_camera_change",
            },
            "sample.mp4",
        )
        self.assertEqual(route["route"], "calibrated_static_sphere")
        self.assertTrue(route["static_calibration_allowed"])

    def test_borderline_changed_missing_and_mismatch_use_dynamic_route(self) -> None:
        cases = [
            {"filename": "sample.mp4", "status": "ok", "decision": "review"},
            {"filename": "sample.mp4", "status": "ok", "decision": "changed"},
            {"filename": "other.mp4", "status": "ok", "decision": "fixed"},
            None,
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence):
                route = choose_reconstruction_route(evidence, "sample.mp4")
                self.assertEqual(route["route"], "spatialtrackerv2_dynamic")
                self.assertFalse(route["static_calibration_allowed"])

    def test_audit_error_never_allows_static_calibration(self) -> None:
        route = choose_reconstruction_route(
            {
                "filename": "sample.mp4",
                "status": "error",
                "decision": "fixed",
            },
            "sample.mp4",
        )
        self.assertEqual(route["reason"], "camera_motion_audit_error")
        self.assertFalse(route["static_calibration_allowed"])


if __name__ == "__main__":
    unittest.main()
