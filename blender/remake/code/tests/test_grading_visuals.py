from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from remake_benchmark.reconstruction.grading_visuals import (
    write_trajectory_comparison,
    write_trajectory_fit_evidence,
)


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is required")
class GradingVisualsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _rows(offset: float = 0.0) -> list[dict[str, object]]:
        return [
            {
                "frame_index": index,
                "time_s": index * 0.1,
                "x_m": offset + index * 0.2,
                "y_m": 0.0,
                "z_m": 1.0 - index * 0.05,
                "physics_fit_used": True,
                "measurement_source": "direct_measurement",
            }
            for index in range(6)
        ]

    def test_fit_evidence_renders_every_series_and_real_target_contract(self) -> None:
        rows = self._rows()
        result = {
            "fit": {
                "method": "synthetic_two_series",
                "target_not_used_for_fit": False,
                "parameter_estimates": {"alpha": 1.2, "beta": 0.3},
                "diagnostics": {
                    "horizontal": {
                        "fit_series": {
                            "series_name": "x_m",
                            "time_s": [0.0, 0.1, 0.2],
                            "observed": [0.0, 0.2, 0.4],
                            "predicted": [0.0, 0.19, 0.41],
                            "residual": [0.0, 0.01, -0.01],
                            "time_basis": "video_time_s",
                            "supports_parameters": ["alpha"],
                        }
                    },
                    "vertical": {
                        "fit_series": {
                            "series_name": "z_m",
                            "time_s": [0.0, 0.1],
                            "observed": [1.0, 0.95],
                            "predicted": [0.99, 0.96],
                            "residual": [0.01, -0.01],
                            "time_basis": "video_time_s",
                        }
                    },
                },
            },
            "metrics": {
                "fit_complete": True,
                "experiment_nmae": 0.1,
                "parameters": {
                    "alpha": {"gt": 1.0, "normalized_absolute_error": 0.2},
                    "beta": {"gt": 0.4, "normalized_absolute_error": 0.1},
                },
            },
        }
        grade = {
            "job_id": "synthetic_job",
            "video_stage": "PASS_TO_SCAN",
            "g1_motion_type_validity": {
                "event_sequence": [{"name": "impact", "frame": 2}],
            },
        }

        artifacts = write_trajectory_fit_evidence(self.root, rows, result, grade)
        self.assertIsNotNone(artifacts)
        assert artifacts is not None
        self.assertEqual(artifacts["standardized_fit_series_count"], 2)
        self.assertFalse(artifacts["target_not_used_for_fit"])
        self.assertFalse(artifacts["series_support_mapping_complete"])
        self.assertTrue(Path(artifacts["trajectory_fit_plot"]["png"]).is_file())
        self.assertTrue(Path(artifacts["trajectory_fit_plot"]["svg"]).is_file())

        with (self.root / "fit_series.csv").open(encoding="utf-8-sig", newline="") as handle:
            fit_rows = list(csv.DictReader(handle))
        self.assertEqual({row["series_name"] for row in fit_rows}, {"x_m", "z_m"})
        self.assertEqual({row["series_id"] for row in fit_rows}, {"0", "1"})
        self.assertIn("alpha", {row["supports_parameters"] for row in fit_rows})
        events = json.loads((self.root / "fit_events.json").read_text(encoding="utf-8"))
        self.assertFalse(events["target_not_used_for_fit"])
        self.assertTrue(events["target_parameters_used"])
        self.assertIsNotNone(events["target_contract_warning"])

    def test_cross_scene_comparison_draws_all_usable_jobs(self) -> None:
        results: dict[str, dict[str, object]] = {}
        levels = []
        for level_index, target in enumerate((1.0, 2.0)):
            usable = []
            for scene_index, scene in enumerate(("indoor1", "outdoor1")):
                job_id = f"job_{level_index}_{scene_index}"
                usable.append(job_id)
                trajectory = self.root / f"{job_id}.csv"
                rows = self._rows(offset=0.1 * scene_index + target)
                with trajectory.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                results[job_id] = {
                    "job": {"scene_id": scene},
                    "trajectory_csv": str(trajectory),
                }
            levels.append({"target_value": target, "usable_job_ids": usable})
        scan = {
            "scan_id": "aggregate_synthetic",
            "entity_scope": "cross_scene_parameter_scan",
            "levels": levels,
        }

        artifacts = write_trajectory_comparison(
            self.root / "trajectory_comparison.png",
            scan,
            results,
        )
        self.assertIsNotNone(artifacts)
        assert artifacts is not None
        self.assertEqual(set(artifacts["plotted_job_ids"]), set(results))
        self.assertEqual(set(artifacts["plotted_scenes"]), {"indoor1", "outdoor1"})
        self.assertEqual(
            artifacts["selection_rule"],
            "all_usable_jobs_grouped_by_target_and_labelled_by_scene",
        )
        self.assertTrue(Path(artifacts["png"]).is_file())
        self.assertTrue(Path(artifacts["svg"]).is_file())


if __name__ == "__main__":
    unittest.main()
