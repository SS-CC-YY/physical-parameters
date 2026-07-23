from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.simple_paper_grading import (  # noqa: E402
    grade_simple_paper_benchmark,
)


def _experiment(experiment_id: str, estimates: list[float]) -> tuple[dict, list[dict]]:
    anchors = []
    rows = []
    for index, (target, estimate) in enumerate(zip((0.0, 0.5, 1.0), estimates)):
        tuple_id = f"p{index}"
        anchors.append({"id": tuple_id, "p": target})
        rows.append(
            {
                "video_name": f"{experiment_id}__{tuple_id}.mp4",
                "experiment_id": experiment_id,
                "parameter_tuple_id": tuple_id,
                "scene_id": "baseline",
                "camera_name": "CAM_Side",
                "seed": "341867882",
                "manual_generation_validity_status": "pass",
                "generation_validity_status": "pass",
                "fit_complete": "True",
                "trajectory_fit_eligible": "True",
                "fit_status": "ok",
                "p__estimate": str(estimate),
            }
        )
    return (
        {
            "id": experiment_id,
            "hidden_parameters": [{"name": "p", "unit": "1", "valid_range": [0.0, 1.0]}],
            "anchor_tuples": anchors,
        },
        rows,
    )


class SimplePaperGradingTests(unittest.TestCase):
    def test_response_grades_are_lenient_and_do_not_require_fit_diagnostics(self) -> None:
        specs_and_rows = [
            _experiment("l2", [0.8, 0.5, 0.2]),
            _experiment("l3", [0.50, 0.80, 1.30]),
            _experiment("l4", [0.05, 0.55, 0.95]),
        ]
        registry = {"experiments": [item[0] for item in specs_and_rows]}
        rows = [row for item in specs_and_rows for row in item[1]]

        result = grade_simple_paper_benchmark(rows, registry)
        grades = {row["experiment_id"]: row["grade"] for row in result["per_scan_rows"]}

        self.assertEqual(grades, {"l2": "L2", "l3": "L3", "l4": "L4"})
        self.assertEqual(result["summary"]["grade_counts"]["L2"], 1)
        self.assertEqual(result["summary"]["grade_counts"]["L3"], 1)
        self.assertEqual(result["summary"]["grade_counts"]["L4"], 1)

    def test_unconfirmed_automatic_failure_is_x_not_model_failure(self) -> None:
        spec, rows = _experiment("auto", [0.0, 0.5, 1.0])
        rows[0].pop("manual_generation_validity_status")
        rows[0]["generation_validity_status"] = "fail"
        result = grade_simple_paper_benchmark(rows, {"experiments": [spec]})

        video = next(row for row in result["per_video_rows"] if row["row_id"].endswith("p0.mp4"))
        self.assertEqual(video["grade"], "X")
        self.assertIn("automatic_failure_pending_manual_confirmation", video["reason_codes"])
        # The two remaining levels still support a response scan.
        self.assertEqual(result["per_scan_rows"][0]["grade"], "L4")

    def test_soft_review_is_provisionally_usable_but_failure_codes_still_block(self) -> None:
        spec, rows = _experiment("review", [0.0, 0.5, 1.0])
        rows[0].pop("manual_generation_validity_status")
        rows[0]["generation_validity_status"] = "review"
        rows[0]["generation_warning_codes"] = "persistent_shape_outlier_requires_review"
        rows[1].pop("manual_generation_validity_status")
        rows[1]["generation_validity_status"] = "review"
        rows[1]["generation_failure_codes"] = "confirmed_object_disappearance"

        result = grade_simple_paper_benchmark(rows, {"experiments": [spec]})
        videos = {row["row_id"]: row for row in result["per_video_rows"]}

        soft = next(value for key, value in videos.items() if key.endswith("p0.mp4"))
        hard = next(value for key, value in videos.items() if key.endswith("p1.mp4"))
        self.assertEqual(soft["grade"], "PASS_TO_SCAN")
        self.assertTrue(soft["generation_review_provisional"])
        self.assertIn("soft_review_provisionally_accepted", soft["reason_codes"])
        self.assertEqual(hard["grade"], "X")
        self.assertTrue(hard["generation_hard_failure"])
        self.assertIn("automatic_failure_pending_manual_confirmation", hard["reason_codes"])

    def test_manual_failure_is_l1_but_is_not_mixed_into_scan_grade_counts(self) -> None:
        spec, rows = _experiment("manual", [0.0, 0.5, 1.0])
        rows[0]["manual_generation_validity_status"] = "fail"
        rows[0]["reconstruction_route"] = "spatialtrackerv2_dynamic"
        rows[0]["simple_dynamic_3d_decision"] = "include"
        rows[0]["simple_measurement_route"] = "qualified_dynamic_3d"
        result = grade_simple_paper_benchmark(rows, {"experiments": [spec]})

        video = next(row for row in result["per_video_rows"] if row["row_id"].endswith("p0.mp4"))
        self.assertEqual(video["grade"], "L1")
        self.assertEqual(result["summary"]["video_grade_counts"]["L1"], 1)
        self.assertNotIn("L1", result["summary"]["grade_counts"])
        self.assertEqual(result["per_scan_rows"][0]["grade"], "L4")

    def test_only_frozen_baseline_side_primary_seed_is_selected(self) -> None:
        spec, rows = _experiment("scope", [0.0, 0.5, 1.0])
        rows.extend(
            [
                {**rows[0], "video_name": "indoor.mp4", "scene_id": "indoor1"},
                {**rows[0], "video_name": "main.mp4", "camera_name": "CAM_Main"},
                {**rows[0], "video_name": "seed.mp4", "seed": "7"},
            ]
        )
        result = grade_simple_paper_benchmark(rows, {"experiments": [spec]})
        self.assertEqual(result["summary"]["selected_primary_slice_row_count"], 3)

    def test_dynamic_route_is_fail_closed_without_qualified_3d_gate(self) -> None:
        spec, rows = _experiment("dynamic", [0.0, 0.5, 1.0])
        for row in rows:
            row["reconstruction_route"] = "spatialtrackerv2_dynamic"
        result = grade_simple_paper_benchmark(rows, {"experiments": [spec]})

        self.assertEqual(result["summary"]["video_grade_counts"]["X"], 3)
        self.assertEqual(result["per_scan_rows"][0]["grade"], "X")
        self.assertIn(
            "dynamic_3d_evidence_gate_not_passed",
            result["per_scan_rows"][0]["reason_codes"],
        )

    def test_partial_fit_uses_target_parameter_attribution_not_global_completeness(self) -> None:
        spec, rows = _experiment("partial", [0.05, 0.55, 0.95])
        for row in rows:
            row["fit_complete"] = "False"
            row["fit_status"] = "partial"
            row["parameter_attribution_json"] = json.dumps(
                {"p": {"status": "pass", "reason_codes": []}}
            )

        result = grade_simple_paper_benchmark(rows, {"experiments": [spec]})

        self.assertEqual(result["summary"]["video_grade_counts"]["PASS_TO_SCAN"], 3)
        self.assertEqual(result["per_scan_rows"][0]["grade"], "L4")

    def test_reliable_rule_family_failure_is_l2_evidence_not_x(self) -> None:
        spec, rows = _experiment("rulefail", [0.0, 0.5, 1.0])
        for row in rows[:2]:
            row.pop("p__estimate")
            row["fit_complete"] = "False"
            row["fit_status"] = "model_mismatch"
            row["rule_family_status"] = "fail"
            row["rule_family_reason_codes"] = "persistent_nonuniform_speed"
            row["parameter_attribution_json"] = json.dumps(
                {
                    "p": {
                        "status": "fail",
                        "reason_codes": ["persistent_nonuniform_speed"],
                    }
                }
            )

        result = grade_simple_paper_benchmark(rows, {"experiments": [spec]})
        scan = result["per_scan_rows"][0]

        self.assertEqual(result["summary"]["video_grade_counts"]["PASS_TO_SCAN"], 3)
        self.assertEqual(scan["grade"], "L2")
        self.assertEqual(scan["evidence"]["parameter_rule_failure_row_count"], 2)
        self.assertIn(
            "one_or_more_parameter_levels_fail_the_target_rule_family",
            scan["reason_codes"],
        )


if __name__ == "__main__":
    unittest.main()
