from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from remake_benchmark.reconstruction.contact_geometry import evaluate_contact_geometry
from remake_benchmark.reconstruction.hierarchical_grading import (
    build_cross_scene_parameter_scan_grades,
    build_parameter_scan_grades,
    grade_video_result,
    score_parameter_response,
)
from remake_benchmark.reconstruction.motion_type_validity import evaluate_motion_type


CODE_ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    return json.loads((CODE_ROOT / "configs" / "evaluations" / name).read_text(encoding="utf-8"))


def _rows(time: np.ndarray, x: np.ndarray, z: np.ndarray) -> list[dict]:
    return [
        {
            "frame_index": index,
            "source_frame_index": index,
            "time_s": float(t),
            "x_m": float(x_value),
            "y_m": 0.0,
            "z_m": float(z_value),
            "fit_x_m": float(x_value),
            "fit_y_m": 0.0,
            "fit_z_m": float(z_value),
            "measurement_valid": True,
            "physics_fit_used": True,
            "fit_eligible": True,
            "interpolated": False,
        }
        for index, (t, x_value, z_value) in enumerate(zip(time, x, z))
    ]


class ContactGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _config("motion_type_profiles_v1.json")

    def test_sustained_ground_penetration_is_g0_evidence(self) -> None:
        time = np.linspace(0.0, 2.0, 41)
        z = np.full_like(time, 0.44)
        z[20:25] = 0.10
        result = evaluate_contact_geometry("v1_A", _rows(time, np.zeros_like(time), z), self.profile)
        self.assertEqual(result["status"], "fail")
        self.assertIn("severe_ground_penetration", result["failure_codes"])

    def test_sparse_contact_evidence_is_u(self) -> None:
        time = np.linspace(0.0, 0.4, 6)
        result = evaluate_contact_geometry("v1_A", _rows(time, np.zeros_like(time), np.ones_like(time)), self.profile)
        self.assertEqual(result["status"], "indeterminate")


class MotionTypeValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _config("motion_type_profiles_v1.json")

    def test_free_fall_topology_passes(self) -> None:
        time = np.linspace(0.0, 3.0, 73)
        ballistic = 3.2 - 0.5 * 2.0 * time**2
        z = np.maximum(0.44, ballistic)
        result = evaluate_motion_type("v1_A", _rows(time, np.zeros_like(time), z), self.profile)
        self.assertEqual(result["status"], "pass", result)

    def test_wall_motion_without_reversal_is_g1_evidence(self) -> None:
        time = np.linspace(0.0, 3.0, 73)
        moving = np.linspace(-2.0, 2.96, 49)
        x = np.r_[moving, np.full(len(time) - len(moving), 2.96)]
        result = evaluate_motion_type("v1_B", _rows(time, x, np.full_like(time, 0.5)), self.profile)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("reversal" in code for code in result["failure_codes"]))

    def test_sparse_motion_is_u(self) -> None:
        time = np.linspace(0.0, 0.3, 5)
        result = evaluate_motion_type("v1_C", _rows(time, time, np.zeros_like(time)), self.profile)
        self.assertEqual(result["status"], "indeterminate")

    def test_damped_pendulum_topology_passes(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        theta = 0.65 * np.exp(-0.08 * time) * np.cos(2.0 * math.pi * time / 2.5)
        length = 2.3
        x = length * np.sin(theta)
        z = 3.8 - length * np.cos(theta)
        result = evaluate_motion_type("v1_D", _rows(time, x, z), self.profile)
        self.assertEqual(result["status"], "pass", result)

    def test_remaining_profiles_accept_canonical_synthetic_motion(self) -> None:
        time = np.linspace(0.0, 5.0, 121)

        def pendulum(pivot_z: float, length: float, amplitude: float = 0.65) -> tuple[np.ndarray, np.ndarray]:
            theta = amplitude * np.exp(-0.04 * time) * np.cos(2.0 * math.pi * time / 2.5)
            return length * np.sin(theta), pivot_z - length * np.cos(theta)

        triangle = (2.0 / math.pi) * np.arcsin(np.sin(2.0 * math.pi * time / 2.0))
        v2d_x, v2d_z = pendulum(3.28, 1.85)
        theta_v3d = 0.95 * np.exp(-0.025 * time) * np.cos(2.0 * math.pi * time / 2.5)
        v3d_x = 1.35 * np.sin(theta_v3d)
        v3d_z = 4.05 - 1.35 * np.cos(theta_v3d)
        v3c_x = np.where(time <= 2.5, -2.0 + (6.61 / 2.5) * time, 4.61 - 1.2 * (time - 2.5))
        cases = {
            "v1_B": (np.where(time <= 2.5, -2.0 + (4.96 / 2.5) * time, 2.96 - 1.2 * (time - 2.5)), np.full_like(time, 0.44)),
            "v1_C": (2.0 * time - 0.16 * time**2, np.full_like(time, 0.44)),
            "v2_A": (2.0 * np.sin(2.0 * math.pi * time / 2.0), np.zeros_like(time)),
            "v2_B": (2.75 * triangle, np.zeros_like(time)),
            "v2_C": (-2.0 + 2.0 * time - 0.15 * time**2, np.zeros_like(time)),
            "v2_D": (v2d_x, v2d_z),
            "v2_E": (np.zeros_like(time), 0.82 + np.abs(time - 2.5)),
            "v3_A": (0.8 * time, 0.82 + np.abs(time - 2.5)),
            "v3_B": (1.72 * triangle, np.zeros_like(time)),
            "v3_C": (v3c_x, np.where(v3c_x < -0.5, 0.8 - 0.08 * (v3c_x + 0.5), 0.8)),
            "v3_D": (v3d_x, v3d_z),
        }
        for experiment_id, (x, z) in cases.items():
            with self.subTest(experiment_id=experiment_id):
                result = evaluate_motion_type(experiment_id, _rows(time, x, z), self.profile)
                self.assertEqual(result["status"], "pass", result)

class ResponseGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _config("hierarchical_physics_grading_v1.json")
        cls.spec = {"name": "gravity_g", "unit": "m/s^2", "valid_range": [0.0, 10.0]}

    @staticmethod
    def levels(estimates: list[float | None]) -> list[dict]:
        return [
            {
                "target_value": target,
                "planned_job_count": 1,
                "usable_job_count": 0 if estimate is None else 1,
                "median_estimate": estimate,
                "fit_nrmse_values": [] if estimate is None else [0.05],
            }
            for target, estimate in zip([2.0, 5.0, 8.0], estimates)
        ]

    def test_flat_response_is_g2(self) -> None:
        result = score_parameter_response(self.levels([5.0, 5.0, 5.0]), self.spec, self.policy)
        self.assertEqual(result["grade"], "G2")

    def test_correct_direction_but_biased_is_g3(self) -> None:
        result = score_parameter_response(self.levels([5.0, 8.0, 11.0]), self.spec, self.policy)
        self.assertEqual(result["grade"], "G3")

    def test_direction_and_numeric_accuracy_is_g4(self) -> None:
        result = score_parameter_response(self.levels([2.1, 5.0, 7.9]), self.spec, self.policy)
        self.assertEqual(result["grade"], "G4", result)

    def test_missing_levels_is_u(self) -> None:
        result = score_parameter_response(self.levels([2.0, None, None]), self.spec, self.policy)
        self.assertEqual(result["grade"], "U")

    def test_missing_standardized_residual_is_u_when_otherwise_g4(self) -> None:
        levels = self.levels([2.0, 5.0, 8.0])
        for level in levels:
            level["fit_nrmse_values"] = []
        result = score_parameter_response(levels, self.spec, self.policy)
        self.assertEqual(result["grade"], "U")

    def test_off_target_crosstalk_prevents_g4(self) -> None:
        levels = self.levels([2.0, 5.0, 8.0])
        for level, nuisance in zip(levels, [0.1, 0.3, 0.5]):
            level["off_target_median_estimates"] = {"nuisance": nuisance}
            level["off_target_valid_ranges"] = {"nuisance": [0.0, 1.0]}
        result = score_parameter_response(levels, self.spec, self.policy)
        self.assertEqual(result["grade"], "G3")
        self.assertFalse(result["numeric_accuracy"]["checks"]["off_target_parameter_stability"])

    def test_missing_off_target_fit_is_u_when_otherwise_g4(self) -> None:
        levels = self.levels([2.0, 5.0, 8.0])
        for index, level in enumerate(levels):
            level["off_target_median_estimates"] = {} if index == 1 else {"nuisance": 0.2}
            level["off_target_valid_ranges"] = {"nuisance": [0.0, 1.0]}
        result = score_parameter_response(levels, self.spec, self.policy)
        self.assertEqual(result["grade"], "U")


class FrozenManifestScanTests(unittest.TestCase):
    def test_side_primary_has_216_canonical_atomic_scans(self) -> None:
        policy = _config("hierarchical_physics_grading_v1.json")
        registry = json.loads(
            (CODE_ROOT / "assets" / "seedance978_evaluation" / "experiment_registry.json").read_text(encoding="utf-8")
        )
        manifest = [
            json.loads(line)
            for line in (CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        anchors = {
            (experiment["id"], anchor["id"]): {key: value for key, value in anchor.items() if key != "id"}
            for experiment in registry["experiments"]
            for anchor in experiment["anchor_tuples"]
        }
        results = {}
        grades = {}
        for row in manifest:
            job_id = row["job_id"]
            target = anchors[(row["experiment_id"], row["factors"]["parameter_tuple_id"])]
            results[job_id] = {
                "fit": {
                    "parameter_estimates": target,
                    "parameter_observed": {name: True for name in target},
                    "target_not_used_for_fit": True,
                    "diagnostics": {"fit_nrmse": 0.01},
                },
                "metrics": {"fit_complete": True},
            }
            grades[job_id] = {
                "video_stage": "PASS_TO_SCAN",
                "inverse_fit_evidence": {"status": "pass", "per_job_worst_fit_nrmse": 0.01},
            }
        scans = build_parameter_scan_grades(
            manifest_rows=manifest,
            results_by_id=results,
            grades_by_id=grades,
            registry=registry,
            policy=policy,
        )
        side_primary = [
            scan for scan in scans
            if scan["canonical_for_target"]
            and scan["camera_name"] == "CAM_Side"
            and scan["seed"] == 341867882
        ]
        self.assertEqual(len(side_primary), 216)
        self.assertTrue(all(scan["response_grade"] == "G4" for scan in side_primary))
        cross_scene = build_cross_scene_parameter_scan_grades(scans, policy=policy)
        side_cross_scene = [scan for scan in cross_scene if scan["camera_name"] == "CAM_Side"]
        self.assertEqual(len(side_cross_scene), 24)
        self.assertTrue(all(scan["response_grade"] == "G4" for scan in side_cross_scene))
        self.assertTrue(
            all(level["median_estimate_ci95"] is not None for scan in side_cross_scene for level in scan["levels"])
        )

class VideoGateOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _config("motion_type_profiles_v1.json")
        cls.policy = _config("hierarchical_physics_grading_v1.json")

    def base_result(self, generation_status: str = "pass") -> dict:
        return {
            "job": {
                "experiment_id": "v1_A",
                "parameter_tuple_id": "g2p00",
                "scene_id": "baseline",
                "object_id": "standard_ball",
                "camera_name": "CAM_Side",
                "seed": 1,
                "video_name": "example.mp4",
            },
            "video_generation_validity": {
                "status": generation_status,
                "failure_codes": ["rigid_shape_failure"] if generation_status == "fail" else [],
            },
            "trajectory_fit_eligibility": {"eligible": True, "status": "full"},
            "fit": {"status": "ok", "parameter_observed": {"gravity_g": True}, "target_not_used_for_fit": True},
            "metrics": {"fit_complete": True},
        }

    def test_g0_stops_before_g1(self) -> None:
        grade = grade_video_result(self.base_result("fail"), [], policy=self.policy, motion_profile=self.profile)
        self.assertEqual(grade["video_stage"], "G0")
        self.assertEqual(grade["g1_motion_type_validity"]["status"], "not_run")

    def test_valid_video_passes_to_scan(self) -> None:
        time = np.linspace(0.0, 3.0, 73)
        z = np.maximum(0.44, 3.2 - time**2)
        grade = grade_video_result(self.base_result("pass"), _rows(time, np.zeros_like(time), z), policy=self.policy, motion_profile=self.profile)
        self.assertEqual(grade["video_stage"], "PASS_TO_SCAN", grade)

    def test_dynamic_3d_rigidity_uses_unified_g0_thresholds(self) -> None:
        time = np.linspace(0.0, 3.0, 73)
        z = np.maximum(0.44, 3.2 - time**2)
        result = self.base_result("pass")
        result["reconstruction_route"] = "spatialtrackerv2_dynamic"
        result["camera_motion_evidence"] = {"final_category": "camera_changed"}
        result["pipeline"] = {
            "dynamic_reconstruction": {
                "rigidity_evidence": {
                    "object_frames": [
                        {"frame_index": index, "normalized_rmse": 0.50, "inlier_fraction": 0.20}
                        for index in range(3)
                    ],
                    "background_frames": [
                        {"frame_index": index, "normalized_rmse": 0.05, "inlier_fraction": 0.95, "spatial_coverage_fraction": 0.8}
                        for index in range(3)
                    ],
                }
            }
        }
        grade = grade_video_result(result, _rows(time, np.zeros_like(time), z), policy=self.policy, motion_profile=self.profile)
        self.assertEqual(grade["video_stage"], "G0", grade)
        self.assertEqual(grade["g0_dynamic_rigidity"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
