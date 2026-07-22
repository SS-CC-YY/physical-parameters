from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grade_seedance978_results.py"
SPEC = importlib.util.spec_from_file_location("grade_seedance978_results_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GradeSeedanceCliHelpersTest(unittest.TestCase):
    def test_evidence_merge_preserves_old_visual_path(self) -> None:
        existing = [{
            "entity_scope": "single_video",
            "entity_id": "job_a",
            "grade": "G1",
            "grade_card": "/old/grade_card.png",
        }]
        current = [{
            "entity_scope": "single_video",
            "entity_id": "job_a",
            "grade": "PASS_TO_SCAN",
            "grade_card": None,
        }, {
            "entity_scope": "parameter_scan",
            "entity_id": "scan_a",
            "grade": "G3",
        }]
        merged = MODULE._merge_evidence_rows(existing, current)
        by_id = {(row["entity_scope"], row["entity_id"]): row for row in merged}
        video = by_id[("single_video", "job_a")]
        self.assertEqual(video["grade"], "PASS_TO_SCAN")
        self.assertEqual(video["grade_card"], "/old/grade_card.png")
        self.assertIn(("parameter_scan", "scan_a"), by_id)

    def test_adjudication_loader_enforces_g0_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adjudication.jsonl"
            path.write_text(json.dumps({
                "schema_version": "1.0.0",
                "job_id": "job_a",
                "decision_scope": "g0_generation_validity",
                "g0_status": "pass",
                "reason_codes": ["manual_review_false_positive"],
                "reviewer": "reviewer",
                "note": "reviewed",
            }) + "\n", encoding="utf-8")
            loaded = MODULE._load_adjudications(path, {"job_a"})
            self.assertEqual(loaded["job_a"]["g0_status"], "pass")

            path.write_text(json.dumps({
                "job_id": "job_a",
                "decision_scope": "g1_motion_type",
                "g0_status": "pass",
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decision_scope"):
                MODULE._load_adjudications(path, {"job_a"})

    def test_adjudication_is_explicitly_forwarded_to_video_grader(self) -> None:
        received = {}

        def fake_grader(
            result,
            rows,
            *,
            policy,
            motion_profile,
            adjudication_override=None,
        ):
            received["override"] = adjudication_override
            return {"video_stage": "PASS_TO_SCAN"}

        original = MODULE.grade_video_result
        MODULE.grade_video_result = fake_grader
        try:
            record = {
                "job_id": "job_a",
                "decision_scope": "g0_generation_validity",
                "g0_status": "pass",
            }
            result = MODULE._grade_video_with_optional_adjudication(
                {},
                [],
                policy={},
                motion_profile={},
                adjudication_override=record,
            )
        finally:
            MODULE.grade_video_result = original
        self.assertEqual(result["video_stage"], "PASS_TO_SCAN")
        self.assertEqual(received["override"], record)


if __name__ == "__main__":
    unittest.main()
