from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_experiment_evidence_report.py"
)
SPEC = importlib.util.spec_from_file_location("experiment_evidence_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CANDIDATE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_candidate_physics_report.py"
)
CANDIDATE_SPEC = importlib.util.spec_from_file_location(
    "candidate_physics_report", CANDIDATE_SCRIPT
)
assert CANDIDATE_SPEC is not None and CANDIDATE_SPEC.loader is not None
CANDIDATE = importlib.util.module_from_spec(CANDIDATE_SPEC)
CANDIDATE_SPEC.loader.exec_module(CANDIDATE)


class ExperimentEvidenceReportTest(unittest.TestCase):
    def test_model_mismatch_is_rule_failure(self) -> None:
        row = MODULE._classify_job(
            {
                "fit_status": "model_mismatch",
                "generation_validity_status": "pass",
                "tracked_fraction": "0.98",
            }
        )
        self.assertEqual(row["job_evidence_class"], "RULE_FAIL")
        self.assertEqual(row["job_state"], "M")

    def test_observed_missing_event_is_event_failure(self) -> None:
        row = MODULE._classify_job(
            {
                "fit_status": "insufficient_evidence",
                "generation_validity_status": "pass",
                "tracked_fraction": "0.95",
                "fit_validity_primary_reason": "wall_velocity_reversal_not_found",
            }
        )
        self.assertEqual(row["job_evidence_class"], "EVENT_FAIL")
        self.assertEqual(row["job_state"], "M")

    def test_missing_required_surface_transition_is_event_failure(self) -> None:
        row = MODULE._classify_job(
            {
                "fit_status": "insufficient_evidence",
                "generation_validity_status": "pass",
                "tracked_fraction": "0.96",
                "fit_validity_primary_reason": (
                    "expected_exactly_one_surface_transition"
                ),
            }
        )
        self.assertEqual(row["job_evidence_class"], "EVENT_FAIL")
        self.assertEqual(row["job_state"], "M")

    def test_low_tracking_missing_event_remains_unknown(self) -> None:
        row = MODULE._classify_job(
            {
                "fit_status": "insufficient_evidence",
                "generation_validity_status": "indeterminate",
                "tracked_fraction": "0.52",
                "fit_validity_primary_reason": "wall_velocity_reversal_not_found",
            }
        )
        self.assertEqual(row["job_evidence_class"], "TRUE_U")
        self.assertEqual(row["job_state"], "U")

    def test_scene_only_gate_becomes_refit_candidate(self) -> None:
        row = MODULE._classify_job(
            {
                "fit_status": "skipped_trajectory_not_eligible",
                "generation_validity_status": "indeterminate",
                "tracked_fraction": "1.0",
                "generation_warning_codes": "static_scene_rigidity_unresolved",
            }
        )
        self.assertEqual(row["job_evidence_class"], "REFIT_CANDIDATE")
        self.assertEqual(row["job_state"], "U")

    def test_event_fit_can_be_supported_without_global_r2(self) -> None:
        self.assertTrue(
            CANDIDATE._equation_supported(
                {
                    "candidate_estimate": 0.75,
                    "fit_status": "partial",
                    "parameter_attribution_status": "pass",
                    "trajectory_r2": None,
                    "trajectory_nrmse": None,
                },
                r2_min=0.5,
                trajectory_nrmse_max=0.25,
            )
        )

    def test_three_unreplicated_levels_are_e2_not_e3(self) -> None:
        tier = MODULE._evidence_tier(
            state="D",
            scan_rows=[
                {
                    "grade": "G3_DIRECTIONAL",
                    "target_level_count_numeric": 3,
                    "minimum_seed_count_per_level": 1,
                }
            ],
            conclusive_tuple_count=3,
            conclusive_tuple_fraction=1.0,
            conclusive_job_count=3,
        )
        self.assertEqual(tier, "E2")

    def test_replicated_three_level_scan_is_e3(self) -> None:
        tier = MODULE._evidence_tier(
            state="D",
            scan_rows=[
                {
                    "grade": "G3_DIRECTIONAL",
                    "target_level_count_numeric": 3,
                    "minimum_seed_count_per_level": 2,
                }
            ],
            conclusive_tuple_count=3,
            conclusive_tuple_fraction=1.0,
            conclusive_job_count=6,
        )
        self.assertEqual(tier, "E3")


if __name__ == "__main__":
    unittest.main()
