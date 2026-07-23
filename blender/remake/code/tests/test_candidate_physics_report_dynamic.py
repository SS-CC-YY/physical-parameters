from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_candidate_physics_report.py"
)
SPEC = importlib.util.spec_from_file_location("candidate_physics_report_dynamic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidatePhysicsReportDynamicTest(unittest.TestCase):
    def test_latex_model_name_escaping_is_python311_compatible(self) -> None:
        self.assertEqual(
            MODULE._latex_escape("model_name&variant"),
            r"model\_name\&variant",
        )
        source = Path(MODULE.__file__).read_text(encoding="utf-8")
        self.assertNotIn('.replace("_", r"\\_")', source)

    def test_five_model_inventory_and_cases_are_data_driven(self) -> None:
        models = [f"Model-{letter}" for letter in "ABCDE"]
        summaries = []
        for index, model in enumerate(models):
            summaries.append(
                {
                    "model": model,
                    "primary_job_count": 126,
                    "fit_attempt_rate": 0.5,
                    "expected_parameter_observation_count": 238,
                    "equation_supported_parameter_count": 40 + index,
                    "equation_supported_parameter_coverage": (40 + index) / 238,
                    "parameter_success_count": index,
                    "parameter_success_at_25pct_all": index / 238,
                    "median_candidate_bnae_supported": 0.5 + index,
                    "canonical_channel_count": 24,
                    "g3_directional_channel_count": index,
                    "g4_accurate_channel_count": 1 if index == 4 else 0,
                    "u_channel_count": 24 - index - (1 if index == 4 else 0),
                }
            )
        composition = [
            {
                "model": model,
                "parameter_count": 3,
                "expected_parameter_observation_count": 10,
                "equation_supported_parameter_count": 2,
                "parameter_success_count": 1 if model == "Model-E" else 0,
                "parameter_success_at_25pct_all": 0.1 if model == "Model-E" else 0.0,
                "median_candidate_bnae_supported": 0.8,
            }
            for model in models
        ]
        cases = [
            {
                "model": "Model-E",
                "case_type": "trajectory_good_parameter_wrong",
                "job_id": "dynamic_case_from_current_input",
                "experiment_id": "v2_E",
                "parameter_name": "g",
                "target": 9.8,
                "candidate_estimate": 15.2,
                "candidate_bnae": 0.42,
                "trajectory_r2": 0.91,
                "trajectory_nrmse": 0.12,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE._write_reports(
                output,
                model_summary=summaries,
                video_counts_by_model={model: 978 for model in models},
                scans=[],
                background=[],
                views=[],
                composition=composition,
                seeds=[],
                cases=cases,
                r2_min=0.5,
                trajectory_nrmse_max=0.25,
                parameter_bnae_max=0.25,
                directional_slope_min=0.1,
            )
            report = (output / "REPORT_FOR_ADVISORS_ZH.md").read_text(
                encoding="utf-8"
            )
            brief = (output / "ADVISOR_BRIEF_5MIN_ZH.md").read_text(
                encoding="utf-8"
            )
            paper = (output / "PAPER_RESULTS_DRAFT_EN.md").read_text(
                encoding="utf-8"
            )
            story = (output / "STORY_CASES_ZH.md").read_text(encoding="utf-8")

        self.assertIn("5 个模型 × 978 条视频/模型 = 4890 条", report)
        self.assertIn("5 个模型 × 978 条视频/模型 = 4890 条", brief)
        self.assertIn("5 video world models and 4,890 generated videos", paper)
        self.assertIn("1 of 50 three-parameter observations", paper)
        self.assertIn("dynamic_case_from_current_input", story)
        for model in models:
            self.assertIn(model, report)

        combined = "\n".join((report, brief, paper, story))
        for stale in (
            "四模型",
            "四个模型",
            "3,912",
            "Across all four",
            "Across four",
            "No model produced",
            "LongLive V1-A",
            "Wan2.2 V2-A",
            "Helios V2-E",
            "Cosmos V1-A",
        ):
            self.assertNotIn(stale, combined)


if __name__ == "__main__":
    unittest.main()
