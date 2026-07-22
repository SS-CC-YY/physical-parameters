from __future__ import annotations

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.simple_report_visuals import (  # noqa: E402
    write_experiment_scope_summary_svg,
    write_grade_counts_svg,
    write_parameter_scan_small_multiples_svg,
)


class SimpleReportVisualsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _assert_svg(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("<title", text)
        self.assertIn("<desc", text)
        return text

    def test_horizontal_grade_counts_accept_mapping_and_escape_text(self) -> None:
        output = self.root / "grade_counts.svg"
        result = write_grade_counts_svg(
            output,
            {"G0": 2, "G1": 3, "G2": 5, "G3": 7, "G4": 11, "X": 1},
            title="Grades <audited>",
        )
        text = self._assert_svg(output)
        self.assertEqual(result["total"], 29)
        self.assertEqual(list(result["counts"]), ["G0", "G1", "G2", "G3", "G4", "X"])
        self.assertIn("Grades &lt;audited&gt;", text)
        self.assertIn('data-grade="G4"', text)
        self.assertIn("37.9%", text)

    def test_parameter_small_multiples_read_direct_and_evidence_json(self) -> None:
        group_json = self.root / "group_grade.json"
        group_json.write_text(
            json.dumps(
                {
                    "scan_id": "scan-gravity",
                    "experiment_id": "v1_A",
                    "target_parameter": "gravity_g",
                    "parameter_unit": "m/s^2",
                    "response_grade": "G3",
                    "valid_range": [2.0, 12.0],
                    "levels": [
                        {
                            "target_value": 4.0,
                            "estimates": [4.4, 4.8],
                            "median_estimate": 4.6,
                            "median_estimate_ci95": [4.3, 4.9],
                        },
                        {"target_value": 9.8, "estimates": [], "median_estimate": None},
                    ],
                }
            ),
            encoding="utf-8",
        )
        direct = {
            "scan_id": "scan-e",
            "experiment_id": "v1_B",
            "target_parameter": "restitution_e",
            "parameter_unit": "1",
            "response_grade": "G4",
            "levels": [
                {"target_value": 0.3, "median_estimate": 0.31},
                {"target_value": 0.8, "median_estimate": 0.78},
            ],
        }
        output = self.root / "parameter_scans.svg"
        result = write_parameter_scan_small_multiples_svg(
            output,
            [{"entity_scope": "parameter_scan", "grade_json": str(group_json)}, direct],
            columns=2,
        )
        text = self._assert_svg(output)
        self.assertEqual(result["panel_count"], 2)
        self.assertEqual(result["panels"][0]["missing_level_count"], 1)
        self.assertIn('data-scan-id="scan-gravity"', text)
        self.assertIn('data-scan-id="scan-e"', text)
        self.assertIn("requested gravity_g", text)
        self.assertIn("ideal y = x", text)

    def test_scope_summary_accepts_scan_rows_and_keeps_four_scopes(self) -> None:
        rows = [
            {"scene_id": "baseline", "camera_name": "CAM_Side", "seed": 1, "response_grade": "G3"},
            {"scene_id": "baseline", "camera_name": "CAM_Side", "seed": 2, "response_grade": "G4"},
            {"scene_id": "indoor1", "camera_name": "CAM_Side", "seed": 1, "response_grade": "G2"},
            {"scene_id": "baseline", "camera_name": "CAM_Main", "seed": 1, "response_grade": "U"},
        ]
        output = self.root / "scopes.svg"
        result = write_experiment_scope_summary_svg(output, rows)
        text = self._assert_svg(output)
        self.assertEqual(result["scope_count"], 4)
        self.assertIn('data-scope-id="baseline_side_quantitative"', text)
        self.assertIn('data-scope-id="background_robustness"', text)
        self.assertIn('data-scope-id="view_robustness"', text)
        self.assertIn('data-scope-id="seed_stability"', text)
        self.assertIn("2 seeds · 2 scans", text)
        self.assertIn("G3 1 · G4 1", text)


if __name__ == "__main__":
    unittest.main()
