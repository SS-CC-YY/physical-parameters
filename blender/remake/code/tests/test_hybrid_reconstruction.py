from __future__ import annotations

import importlib.util
import json
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

    @staticmethod
    def _side_motion_evidence(
        *,
        translation_px: float,
        cluster: int,
        category: str = "camera_changed",
        inlier_ratio: float = 0.95,
    ) -> dict[str, object]:
        return {
            "filename": "sample__CAM_Side__seed-1.mp4",
            "status": "ok",
            "final_category": category,
            "width": 864,
            "height": 496,
            "max_direct_translation_px": translation_px,
            "direct_motion_cluster_max": cluster,
            "valid_pair_fraction": 1.0,
            "median_inlier_ratio": inlier_ratio,
            "cut_pair_count": 0,
        }

    def test_side_borderline_motion_stays_on_calibrated_2d_route(self) -> None:
        route = choose_reconstruction_route(
            self._side_motion_evidence(
                translation_px=6.4,
                cluster=3,
                category="borderline_below_threshold",
            ),
            "sample__CAM_Side__seed-1.mp4",
        )
        self.assertEqual(route["route"], "calibrated_static_sphere")
        self.assertEqual(route["reason"], "side_motion_below_reliable_3d_threshold")
        self.assertEqual(
            route["effective_camera_motion_category"],
            "side_2d_motion_within_tolerance",
        )
        self.assertTrue(route["calibrated_2d_approximation"])

    def test_side_requires_persistent_one_percent_translation_for_3d(self) -> None:
        route = choose_reconstruction_route(
            self._side_motion_evidence(translation_px=12.2, cluster=8),
            "sample__CAM_Side__seed-1.mp4",
        )
        self.assertEqual(route["route"], "spatialtrackerv2_dynamic")
        self.assertEqual(route["reason"], "side_persistent_translation_supports_3d")
        self.assertGreaterEqual(
            route["side_3d_evidence"]["translation_diagonal_fraction"],
            0.01,
        )

    def test_side_large_but_nonpersistent_audit_does_not_trigger_3d(self) -> None:
        route = choose_reconstruction_route(
            self._side_motion_evidence(translation_px=15.0, cluster=2),
            "sample__CAM_Side__seed-1.mp4",
        )
        self.assertEqual(route["route"], "calibrated_static_sphere")
        self.assertFalse(route["side_3d_evidence"]["threshold_checks"]["persistence"])

    def test_non_side_borderline_policy_remains_conservative(self) -> None:
        route = choose_reconstruction_route(
            {
                "filename": "sample__CAM_Main__seed-1.mp4",
                "status": "ok",
                "final_category": "borderline_below_threshold",
            },
            "sample__CAM_Main__seed-1.mp4",
        )
        self.assertEqual(route["route"], "spatialtrackerv2_dynamic")

    def test_frozen_seedance_side_audit_routes_508_to_2d_and_2_to_3d(self) -> None:
        audit_path = CODE_ROOT / "assets" / "seedance978_evaluation" / "camera_motion.jsonl"
        audit_rows = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and "__CAM_Side__" in line
        ]
        decisions = [
            (row, choose_reconstruction_route(row, row["filename"]))
            for row in audit_rows
        ]
        calibrated = [item for item in decisions if item[1]["route"] == "calibrated_static_sphere"]
        dynamic = [item for item in decisions if item[1]["route"] == "spatialtrackerv2_dynamic"]
        self.assertEqual(len(audit_rows), 510)
        self.assertEqual(len(calibrated), 508)
        self.assertEqual(
            [row["filename"] for row, _ in dynamic],
            [
                "v1_C__mu0p18__indoor1__standard_ball__CAM_Side__seed-341867882.mp4",
                "v3_A__default__baseline__standard_ball__CAM_Side__seed-265635392.mp4",
            ],
        )


if __name__ == "__main__":
    unittest.main()
