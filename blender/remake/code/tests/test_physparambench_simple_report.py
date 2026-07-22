from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from build_physparambench_simple_report import build  # noqa: E402


class PhysParamBenchSimpleReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_writes_compact_tables_figures_and_wording(self) -> None:
        registry = {
            "experiments": [
                {
                    "id": "v1_A",
                    "hidden_parameters": [{"name": "gravity_g", "unit": "m/s^2", "valid_range": [0.0, 10.0]}],
                    "anchor_tuples": [
                        {"id": "g2", "gravity_g": 2.0},
                        {"id": "g6", "gravity_g": 6.0},
                        {"id": "g10", "gravity_g": 10.0},
                    ],
                }
            ]
        }
        policy = {
            "policy_id": "test-simple",
            "primary_scope": {"seed": 341867882},
            "response_grading": {
                "minimum_usable_levels": 2,
                "minimum_pairwise_direction_concordance_exclusive": 0.5,
                "minimum_slope_exclusive": 0.0,
                "l4_min_slope_inclusive": 0.2,
                "l4_max_median_valid_range_nae_inclusive": 0.25,
            },
        }
        rows = []
        for tuple_id, estimate in (("g2", 2.2), ("g6", 6.4), ("g10", 9.7)):
            base = {
                "video_name": f"v1_A__{tuple_id}__baseline__standard_ball__CAM_Side__seed-341867882.mp4",
                "experiment_id": "v1_A",
                "parameter_tuple_id": tuple_id,
                "scene_id": "baseline",
                "camera_name": "CAM_Side",
                "seed": "341867882",
                "generation_validity_status": "pass",
                "fit_complete": "True",
                "trajectory_fit_eligible": "True",
                "fit_status": "ok",
                "gravity_g__estimate": str(estimate),
            }
            rows.append(base)
            rows.append({**base, "video_name": base["video_name"].replace("baseline", "indoor1"), "scene_id": "indoor1"})
            rows.append({**base, "video_name": base["video_name"].replace("CAM_Side", "CAM_Main"), "camera_name": "CAM_Main"})
            rows.append({**base, "video_name": base["video_name"].replace("CAM_Side", "CAM_Top"), "camera_name": "CAM_Top"})
        # A four-seed Side group for the stability appendix.
        for seed, estimate in ((1, 5.8), (2, 6.0), (3, 6.2), (4, 6.1)):
            rows.append(
                {
                    **rows[0],
                    "video_name": f"seed-{seed}.mp4",
                    "parameter_tuple_id": "g6",
                    "seed": str(seed),
                    "gravity_g__estimate": str(estimate),
                }
            )

        all_jobs = self.root / "all_jobs.csv"
        fields = list(rows[0])
        with all_jobs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        registry_path = self.root / "registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        policy_path = self.root / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        output = self.root / "report"

        summary = build(all_jobs, registry_path, policy_path, output)

        self.assertEqual(summary["primary_baseline_side"]["grade_counts"]["L4"], 1)
        self.assertEqual(summary["view_robustness"]["matched_triplet_count"], 3)
        self.assertEqual(summary["seed_stability"]["group_count"], 1)
        for name in (
            "summary.json",
            "REPORT_ZH.md",
            "ABSTRACT_RESULTS_EN.md",
            "parameter_scans.csv",
            "parameter_scans.jsonl",
            "video_gate.csv",
            "representative_scans.csv",
            "evidence_index.csv",
            "fig1_video_gate.svg",
            "fig2_baseline_response_grades.svg",
            "fig3_requested_vs_recovered.svg",
            "fig4_evaluation_scopes.svg",
        ):
            self.assertTrue((output / name).is_file(), name)
        ET.parse(output / "fig3_requested_vs_recovered.svg")
        report = (output / "REPORT_ZH.md").read_text(encoding="utf-8")
        self.assertIn("Baseline + CAM_Side", report)
        self.assertIn("测量不可用 X", report)
        english = (output / "ABSTRACT_RESULTS_EN.md").read_text(encoding="utf-8")
        self.assertIn("partial parameter sensitivity", english)


if __name__ == "__main__":
    unittest.main()
