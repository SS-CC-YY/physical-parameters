from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from remake_benchmark.reconstruction.hierarchical_grading import (
    aggregate_experiment_scan_grades,
    build_cross_scene_parameter_scan_grades,
    build_parameter_scan_grades,
    grade_video_result,
    score_parameter_response,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    return json.loads((CODE_ROOT / "configs" / "evaluations" / name).read_text(encoding="utf-8"))


def _freefall_rows() -> list[dict]:
    time = np.linspace(0.0, 3.0, 73)
    z = np.maximum(0.44, 3.2 - time**2)
    return [
        {
            "frame_index": index,
            "source_frame_index": index,
            "time_s": float(t),
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": float(height),
            "fit_x_m": 0.0,
            "fit_y_m": 0.0,
            "fit_z_m": float(height),
            "measurement_valid": True,
            "physics_fit_used": True,
            "fit_eligible": True,
            "interpolated": False,
        }
        for index, (t, height) in enumerate(zip(time, z))
    ]


def _bounce_rows() -> list[dict]:
    time = np.linspace(0.0, 5.0, 121)
    z = 0.82 + np.abs(time - 2.5)
    return [
        {
            "frame_index": index,
            "source_frame_index": index,
            "time_s": float(t),
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": float(height),
            "fit_x_m": 0.0,
            "fit_y_m": 0.0,
            "fit_z_m": float(height),
            "measurement_valid": True,
            "physics_fit_used": True,
            "fit_eligible": True,
            "interpolated": False,
        }
        for index, (t, height) in enumerate(zip(time, z))
    ]


def _fit_series(*, nrmse: float = 0.01, r2: float = 0.99, points: int = 73) -> dict:
    return {
        "fit_points": points,
        "fit_nrmse": nrmse,
        "fit_r2": r2,
        "fit_series": {
            "series_name": "z_m",
            "time_s": [float(index) for index in range(points)],
            "observed": [float(index) for index in range(points)],
            "predicted": [float(index) for index in range(points)],
            "residual": [0.0] * points,
        },
    }


def _video_result(*, target_free: bool = True, nrmse: float = 0.01, r2: float = 0.99, generation: str = "pass") -> dict:
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
            "status": generation,
            "failure_codes": ["automatic_shape_failure"] if generation == "fail" else [],
        },
        "trajectory_fit_eligibility": {"eligible": True, "status": "full"},
        "fit": {
            "status": "ok",
            "parameter_estimates": {"gravity_g": 2.0},
            "parameter_observed": {"gravity_g": True},
            "target_not_used_for_fit": target_free,
            "diagnostics": _fit_series(nrmse=nrmse, r2=r2),
        },
        "metrics": {"fit_complete": True},
    }


def _atomic_level(target: float, scene: str, estimate: float | None, *, nuisance: float | None = None) -> dict:
    job_id = f"{scene}-{target:g}"
    usable = estimate is not None
    level = {
        "target_value": target,
        "planned_job_count": 1,
        "usable_job_count": int(usable),
        "planned_job_ids": [job_id],
        "usable_job_ids": [job_id] if usable else [],
        "g0_job_ids": [],
        "g1_job_ids": [],
        "u_job_ids": [] if usable else [job_id],
        "estimates": [] if estimate is None else [estimate],
        "median_estimate": estimate,
        "fit_nrmse_values": [] if estimate is None else [0.01],
        "per_job_worst_fit_nrmse": [] if estimate is None else [0.01],
        "off_target_estimates": {},
        "off_target_median_estimates": {},
        "off_target_valid_ranges": {},
    }
    if nuisance is not None:
        level["off_target_estimates"] = {"q": [nuisance]}
        level["off_target_median_estimates"] = {"q": nuisance}
        level["off_target_valid_ranges"] = {"q": [0.0, 1.0]}
    elif nuisance is None:
        level["off_target_valid_ranges"] = {"q": [0.0, 1.0]}
    return level


def _atomic_scan(scene: str, low: float | None, high: float | None, *, low_q: float | None = None, high_q: float | None = None) -> dict:
    return {
        "schema_version": "1.0.0",
        "entity_scope": "parameter_scan",
        "scan_id": scene,
        "experiment_id": "x",
        "scene_id": scene,
        "camera_name": "CAM_Side",
        "object_id": "ball",
        "seed": 341867882,
        "target_parameter": "p",
        "required_parameters": ["p", "q"],
        "parameter_unit": "1",
        "valid_range": [0.0, 10.0],
        "fixed_other_parameters": {"q": 0.2},
        "confounded_intervention": False,
        "canonical_for_target": True,
        "levels": [
            _atomic_level(2.0, scene, low, nuisance=low_q),
            _atomic_level(8.0, scene, high, nuisance=high_q),
        ],
        "response_grade": "G4",
        "decision": {"grade": "G4"},
    }


class VideoAdmissionAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _config("hierarchical_physics_grading_v1.json")
        cls.motion = _config("motion_type_profiles_v1.json")
        cls.rows = _freefall_rows()

    def test_target_independence_must_be_explicit_true(self) -> None:
        grade = grade_video_result(
            _video_result(target_free=False),
            self.rows,
            policy=self.policy,
            motion_profile=self.motion,
        )
        self.assertEqual(grade["video_stage"], "U", grade)
        self.assertIn("inverse_fit_target_independence_not_proven", grade["reason_codes"])

    def test_bad_parameter_fit_series_cannot_enter_scan(self) -> None:
        grade = grade_video_result(
            _video_result(nrmse=0.6, r2=0.0),
            self.rows,
            policy=self.policy,
            motion_profile=self.motion,
        )
        self.assertEqual(grade["video_stage"], "U", grade)
        self.assertIn("inverse_fit_nrmse_exceeds_limit", grade["reason_codes"])
        self.assertIn("inverse_fit_r2_below_limit", grade["reason_codes"])

    def test_adjudication_overrides_only_automatic_generation_status(self) -> None:
        grade = grade_video_result(
            _video_result(generation="fail"),
            self.rows,
            policy=self.policy,
            motion_profile=self.motion,
            adjudication_override={
                "g0_status": "pass",
                "reason_codes": ["blind_review_pass"],
                "reviewer": "reviewer-1",
                "note": "automatic detector false positive",
                "evidence_paths": {"review_clip": "review.mp4"},
            },
        )
        self.assertEqual(grade["video_stage"], "PASS_TO_SCAN", grade)
        self.assertEqual(grade["g0_generation_validity_automatic"]["status"], "fail")
        self.assertEqual(grade["g0_generation_validity_adjudication"]["g0_status"], "pass")
        self.assertEqual(grade["g0_generation_validity"]["status"], "pass")
        self.assertEqual(grade["g0_contact_geometry"]["status"], "pass")

    def test_partial_multi_parameter_fit_admits_identified_parameter(self) -> None:
        result = _video_result()
        result["job"].update(
            {
                "experiment_id": "v2_E",
                "parameter_tuple_id": "g9p80_e0p76",
                "video_name": "partial-v2e.mp4",
            }
        )
        result["fit"] = {
            "status": "partial",
            "parameter_estimates": {"gravity_g": 9.8, "restitution_e": None},
            "raw_parameter_estimates": {"gravity_g": 9.8, "restitution_e": None},
            "parameter_observed": {"gravity_g": True, "restitution_e": False},
            "parameter_attribution": {
                "gravity_g": {"status": "pass", "reason_codes": []},
                "restitution_e": {
                    "status": "indeterminate",
                    "reason_codes": ["too_few_reliable_impacts"],
                },
            },
            "rule_family_evaluation": {
                "status": "pass",
                "reason_codes": [],
                "target_parameters_used": False,
            },
            "fit_validity": {
                "status": "partially_accepted",
                "category": "parameter_specific_identifiability",
                "target_free": True,
            },
            "target_not_used_for_fit": True,
            "diagnostics": _fit_series(points=121),
        }
        result["metrics"] = {"fit_complete": False}
        grade = grade_video_result(
            result,
            _bounce_rows(),
            policy=self.policy,
            motion_profile=self.motion,
        )
        self.assertEqual(grade["video_stage"], "PASS_TO_SCAN", grade)
        evidence = grade["inverse_fit_evidence"]
        self.assertEqual(evidence["fit_completeness_scope"], "partial_parameters")
        self.assertEqual(evidence["admissible_parameters"], ["gravity_g"])
        self.assertTrue(evidence["parameter_evidence"]["gravity_g"]["scan_ready"])
        self.assertFalse(evidence["parameter_evidence"]["restitution_e"]["admissible"])

    def test_reliable_rule_family_mismatch_is_model_failure_not_u(self) -> None:
        result = _video_result()
        result["fit"].update(
            {
                "status": "model_mismatch",
                "parameter_estimates": {"gravity_g": None},
                "parameter_observed": {"gravity_g": False},
                "rule_family_evaluation": {
                    "status": "fail",
                    "reason_codes": ["ballistic_curvature_not_constant"],
                    "target_parameters_used": False,
                },
                "fit_validity": {
                    "status": "rejected",
                    "category": "rule_family_mismatch",
                    "primary_reason_code": "ballistic_curvature_not_constant",
                    "target_free": True,
                },
            }
        )
        result["metrics"] = {"fit_complete": False}
        grade = grade_video_result(
            result,
            self.rows,
            policy=self.policy,
            motion_profile=self.motion,
        )
        self.assertEqual(grade["video_stage"], "G1", grade)
        self.assertEqual(grade["inverse_fit_evidence"]["status"], "model_mismatch")
        self.assertEqual(
            grade["inverse_fit_evidence"]["classification"],
            "model_physics_failure",
        )
        self.assertIn("inverse_rule_family_model_mismatch", grade["reason_codes"])
        self.assertIn("ballistic_curvature_not_constant", grade["reason_codes"])


class ScanConstructionAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _config("hierarchical_physics_grading_v1.json")

    def test_confounded_design_is_explicit_u_for_every_required_parameter(self) -> None:
        registry = {
            "experiments": [
                {
                    "id": "x",
                    "hidden_parameters": [
                        {"name": "p", "unit": "1", "valid_range": [0.0, 1.0]},
                        {"name": "q", "unit": "1", "valid_range": [0.0, 1.0]},
                    ],
                    "anchor_tuples": [
                        {"id": "low", "p": 0.2, "q": 0.8},
                        {"id": "high", "p": 0.8, "q": 0.2},
                    ],
                }
            ]
        }
        manifest = [
            {
                "job_id": tuple_id,
                "experiment_id": "x",
                "seed": 1,
                "factors": {
                    "parameter_tuple_id": tuple_id,
                    "scene_id": "baseline",
                    "object_id": "standard_ball",
                    "camera": "CAM_Main",
                },
            }
            for tuple_id in ("low", "high")
        ]
        results = {
            "low": {"fit": {"parameter_estimates": {"p": 0.2, "q": 0.8}, "diagnostics": {"fit_nrmse": 0.01}}},
            "high": {"fit": {"parameter_estimates": {"p": 0.8, "q": 0.2}, "diagnostics": {"fit_nrmse": 0.01}}},
        }
        grades = {job_id: {"video_stage": "PASS_TO_SCAN"} for job_id in results}
        scans = build_parameter_scan_grades(
            manifest_rows=manifest,
            results_by_id=results,
            grades_by_id=grades,
            registry=registry,
            policy=self.policy,
        )
        self.assertEqual({scan["target_parameter"] for scan in scans}, {"p", "q"})
        self.assertTrue(all(scan["response_grade"] == "U" for scan in scans))
        self.assertTrue(all(scan["confounded_intervention"] for scan in scans))
        aggregate = aggregate_experiment_scan_grades(scans, registry=registry)
        self.assertEqual(aggregate[0]["overall_response_grade"], "U")

    def test_experiment_aggregate_marks_missing_required_parameter_u(self) -> None:
        scans = [
            {
                "canonical_for_target": True,
                "experiment_id": "x",
                "scene_id": "baseline",
                "camera_name": "CAM_Main",
                "object_id": "ball",
                "seed": 1,
                "target_parameter": "p",
                "required_parameters": ["p", "q"],
                "response_grade": "G4",
                "scan_id": "only-p",
            }
        ]
        result = aggregate_experiment_scan_grades(scans)[0]
        self.assertEqual(result["overall_response_grade"], "U")
        self.assertEqual(result["parameter_grades"]["q"], "U")
        self.assertEqual(result["missing_required_parameters"], ["q"])

    def test_response_stage_rechecks_target_independence_after_stale_pass_grade(self) -> None:
        registry = {
            "experiments": [
                {
                    "id": "x",
                    "hidden_parameters": [
                        {"name": "p", "unit": "1", "valid_range": [0.0, 1.0]},
                    ],
                    "anchor_tuples": [
                        {"id": "low", "p": 0.2},
                        {"id": "high", "p": 0.8},
                    ],
                }
            ]
        }
        manifest = [
            {
                "job_id": tuple_id,
                "experiment_id": "x",
                "seed": 1,
                "factors": {
                    "parameter_tuple_id": tuple_id,
                    "scene_id": "baseline",
                    "object_id": "standard_ball",
                    "camera": "CAM_Side",
                },
            }
            for tuple_id in ("low", "high")
        ]
        results = {
            tuple_id: {
                "fit": {
                    "parameter_estimates": {"p": target},
                    "parameter_observed": {"p": True},
                    "target_not_used_for_fit": False,
                    "diagnostics": {"fit_nrmse": 0.01},
                },
                "metrics": {"fit_complete": True},
            }
            for tuple_id, target in (("low", 0.2), ("high", 0.8))
        }
        # Simulate cached grades produced before the target-independence gate.
        grades = {
            tuple_id: {
                "video_stage": "PASS_TO_SCAN",
                "inverse_fit_evidence": {"status": "pass"},
            }
            for tuple_id in results
        }
        scans = build_parameter_scan_grades(
            manifest_rows=manifest,
            results_by_id=results,
            grades_by_id=grades,
            registry=registry,
            policy=self.policy,
        )
        self.assertEqual(len(scans), 1)
        self.assertEqual(scans[0]["response_grade"], "U", scans[0])
        for level in scans[0]["levels"]:
            self.assertEqual(level["usable_job_count"], 0)
            self.assertTrue(level["scan_admission_rejected_job_ids"])

    def test_partial_fit_does_not_discard_identified_target_parameter(self) -> None:
        registry = {
            "experiments": [
                {
                    "id": "x",
                    "hidden_parameters": [
                        {"name": "p", "unit": "1", "valid_range": [0.0, 1.0]},
                        {"name": "q", "unit": "1", "valid_range": [0.0, 1.0]},
                    ],
                    "anchor_tuples": [
                        {"id": "low", "p": 0.2, "q": 0.5},
                        {"id": "high", "p": 0.8, "q": 0.5},
                    ],
                }
            ]
        }
        manifest = [
            {
                "job_id": tuple_id,
                "experiment_id": "x",
                "seed": 1,
                "factors": {
                    "parameter_tuple_id": tuple_id,
                    "scene_id": "baseline",
                    "object_id": "standard_ball",
                    "camera": "CAM_Side",
                },
            }
            for tuple_id in ("low", "high")
        ]
        results = {}
        grades = {}
        for tuple_id, estimate in (("low", 0.35), ("high", 0.95)):
            results[tuple_id] = {
                "fit": {
                    "status": "partial",
                    "parameter_estimates": {"p": estimate, "q": None},
                    "parameter_observed": {"p": True, "q": False},
                    "parameter_attribution": {
                        "p": {"status": "pass", "reason_codes": []},
                        "q": {
                            "status": "indeterminate",
                            "reason_codes": ["q_not_identifiable"],
                        },
                    },
                    "rule_family_evaluation": {"status": "pass", "reason_codes": []},
                    "target_not_used_for_fit": True,
                    "diagnostics": {"fit_nrmse": 0.01},
                },
                "metrics": {"fit_complete": False},
            }
            grades[tuple_id] = {
                "video_stage": "PASS_TO_SCAN",
                "inverse_fit_evidence": {
                    "status": "pass",
                    "parameter_evidence": {
                        "p": {"admissible": True},
                        "q": {"admissible": False},
                    },
                    "per_job_worst_fit_nrmse": 0.01,
                },
            }
        scans = build_parameter_scan_grades(
            manifest_rows=manifest,
            results_by_id=results,
            grades_by_id=grades,
            registry=registry,
            policy=self.policy,
        )
        p_scan = next(scan for scan in scans if scan["target_parameter"] == "p")
        self.assertEqual(p_scan["response_grade"], "G3", p_scan)
        self.assertEqual([level["usable_job_count"] for level in p_scan["levels"]], [1, 1])
        self.assertEqual(
            [level["median_estimate"] for level in p_scan["levels"]],
            [0.35, 0.95],
        )
        self.assertTrue(
            all(not level["scan_admission_rejected_job_ids"] for level in p_scan["levels"])
        )


class CrossSceneAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _config("hierarchical_physics_grading_v1.json")

    def test_cross_scene_direction_uses_only_matched_blocks(self) -> None:
        # Every complete scene has the wrong direction (high=low-1).  The
        # unmatched low-only/high-only extremes would reverse pooled medians.
        values = [
            (100.0, 99.0),
            (100.0, 99.0),
            (100.0, 99.0),
            (100.0, 99.0),
            (0.0, -1.0),
            (0.0, -1.0),
            (0.0, -1.0),
            (-100.0, None),
            (None, 1000.0),
        ]
        scans = [_atomic_scan(f"scene-{index}", low, high, low_q=0.2, high_q=0.2) for index, (low, high) in enumerate(values)]
        scans[0]["valid_range"] = [-200.0, 1200.0]
        for scan in scans[1:]:
            scan["valid_range"] = [-200.0, 1200.0]
        result = build_cross_scene_parameter_scan_grades(scans, policy=self.policy)[0]
        self.assertEqual(result["response_grade"], "G2", result)
        self.assertEqual(result["matched_scene_count"], 7)
        self.assertLess(result["decision"]["direction"]["theil_sen_slope_ci95"][1], 0.0)

    def test_per_job_p90_prevents_median_only_false_g4(self) -> None:
        scans = []
        for index in range(9):
            bias = 0.0 if index < 5 else 8.0
            scans.append(_atomic_scan(f"scene-{index}", 2.0 + bias, 8.0 + bias, low_q=0.2, high_q=0.2))
        result = build_cross_scene_parameter_scan_grades(scans, policy=self.policy)[0]
        self.assertEqual(result["response_grade"], "G3", result)
        self.assertGreater(result["decision"]["numeric_accuracy"]["p90_valid_range_nae"], 0.2)

    def test_sparse_off_target_evidence_cannot_receive_g4(self) -> None:
        scans = [
            _atomic_scan(
                f"scene-{index}",
                2.0,
                8.0,
                low_q=0.2 if index == 0 else None,
                high_q=0.2 if index == 0 else None,
            )
            for index in range(9)
        ]
        result = build_cross_scene_parameter_scan_grades(scans, policy=self.policy)[0]
        self.assertEqual(result["response_grade"], "U", result)
        coverage = result["decision"]["numeric_accuracy"]["off_target_coverage"]["q"]
        self.assertLess(coverage["job_coverage_fraction"], 2.0 / 3.0)

    def test_p90_of_per_job_worst_series_controls_residual_gate(self) -> None:
        levels = []
        for target in (2.0, 5.0, 8.0):
            levels.append(
                {
                    "target_value": target,
                    "planned_job_count": 10,
                    "usable_job_count": 10,
                    "median_estimate": target,
                    "per_job_normalized_errors": [0.0] * 10,
                    "per_job_worst_fit_nrmse": [0.01] * 8 + [0.6] * 2,
                }
            )
        result = score_parameter_response(
            levels,
            {"name": "p", "unit": "1", "valid_range": [0.0, 10.0]},
            self.policy,
        )
        self.assertEqual(result["grade"], "G3", result)
        self.assertGreater(
            result["numeric_accuracy"]["p90_per_job_worst_trajectory_fit_nrmse"],
            0.2,
        )

    def test_exact_two_thirds_coverage_is_not_rounded_down_to_u(self) -> None:
        levels = [
            {
                "target_value": target,
                "planned_job_count": 1,
                "usable_job_count": int(estimate is not None),
                "median_estimate": estimate,
                "fit_nrmse_values": [] if estimate is None else [0.01],
            }
            for target, estimate in zip((2.0, 5.0, 8.0), (2.0, 5.0, None))
        ]
        result = score_parameter_response(
            levels,
            {"name": "p", "unit": "1", "valid_range": [0.0, 10.0]},
            self.policy,
        )
        self.assertNotEqual(result["grade"], "U", result)
        self.assertEqual(result["grade"], "G3", result)


if __name__ == "__main__":
    unittest.main()
