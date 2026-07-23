from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.simple_metrics import (  # noqa: E402
    _canonical_nuisance_signatures,
    _nuisance_signature,
    _paired_condition_rows,
    build_simple_physics_report,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


LINEAGE = {
    "evaluator_source_sha256": "evaluator",
    "physics_fitter_source_sha256": "fitter",
    "dynamic_3d_gate_source_sha256": "dynamic-gate",
    "evaluator_version": "test-v1",
}


def _registry_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage(registry_path: Path) -> dict:
    return {**LINEAGE, "registry_sha256": _registry_sha256(registry_path)}


def _result(estimate: float, manifest: dict, registry_path: Path) -> dict:
    factors = manifest["factors"]
    return {
        "status": "succeeded",
        "job": {
            "experiment_id": manifest["experiment_id"],
            "parameter_tuple_id": factors["parameter_tuple_id"],
            "scene_id": factors["scene_id"],
            "object_id": factors["object_id"],
            "camera_name": factors["camera"],
            "seed": manifest.get("seed"),
            "video_name": f"{manifest['job_id']}.mp4",
        },
        "evaluation_lineage": _lineage(registry_path),
        "reconstruction_route": "calibrated_static_sphere",
        "fit": {
            "status": "ok",
            "parameter_observed": {"gravity_g": True},
            "parameter_estimates": {"gravity_g": estimate},
            "diagnostics": {
                "position": {
                    "fit_series": {
                        "series_name": "z",
                        "time_s": [0.0, 1.0, 2.0],
                        "observed": [4.0, 3.0, 0.0],
                        "predicted": [4.0, 3.0, 0.0],
                    }
                }
            },
        },
        "trajectory_csv": "trajectory_frames.csv",
    }


class SimpleMetricsTests(unittest.TestCase):
    def test_canonical_oat_branch_and_matched_condition_pairs_are_frozen(self) -> None:
        spec = {
            "id": "v2_X",
            "hidden_parameters": [{"name": "p"}, {"name": "q"}],
            "anchor_tuples": [
                {"id": "p0", "p": 0.0, "q": 10.0},
                {"id": "p1", "p": 1.0, "q": 10.0},
                {"id": "q0", "p": 0.5, "q": 5.0},
                {"id": "q1", "p": 0.5, "q": 15.0},
            ],
        }
        canonical = _canonical_nuisance_signatures({"experiments": [spec]})
        self.assertEqual(
            canonical[("v2_X", "p")],
            _nuisance_signature(spec, "p", spec["anchor_tuples"][0]),
        )
        self.assertEqual(
            canonical[("v2_X", "q")],
            _nuisance_signature(spec, "q", spec["anchor_tuples"][2]),
        )

        common = {
            "model": "model",
            "experiment_id": "v2_X",
            "parameter_tuple_id": "p0",
            "parameter": "p",
            "object_id": "ball",
            "seed": 7,
            "target": 0.0,
            "target_range_span": 1.0,
            "trajectory_r2": 0.9,
        }
        baseline = {
            **common,
            "job_id": "baseline",
            "scene_id": "baseline",
            "camera": "CAM_Side",
            "primary_baseline_side": True,
            "estimate": 0.2,
        }
        main = {
            **common,
            "job_id": "main",
            "scene_id": "baseline",
            "camera": "CAM_Main",
            "primary_baseline_side": False,
            "estimate": 0.5,
        }
        indoor = {
            **common,
            "job_id": "indoor",
            "scene_id": "indoor1",
            "camera": "CAM_Side",
            "primary_baseline_side": False,
            "estimate": 0.1,
        }
        paired, summaries = _paired_condition_rows([baseline, main, indoor])
        by_job = {row["condition_job_id"]: row for row in paired}
        self.assertAlmostEqual(by_job["main"]["estimate_difference"], 0.3)
        self.assertEqual(by_job["main"]["comparison_type"], "view")
        self.assertAlmostEqual(by_job["indoor"]["estimate_difference"], -0.1)
        self.assertEqual(by_job["indoor"]["comparison_type"], "background")
        self.assertEqual(len(summaries), 2)

    def test_builds_unclipped_simple_metrics_and_worldbench_style_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = {
                "experiments": [
                    {
                        "id": "v1_A",
                        "hidden_parameters": [
                            {
                                "name": "gravity_g",
                                "unit": "m/s^2",
                                "valid_range": [2.0, 14.7],
                            }
                        ],
                        "anchor_tuples": [
                            {"id": "g2", "gravity_g": 2.0},
                            {"id": "g9", "gravity_g": 9.81},
                            {"id": "g14", "gravity_g": 14.7},
                        ],
                    }
                ]
            }
            manifest = [
                {
                    "job_id": "primary",
                    "experiment_id": "v1_A",
                    "seed": 1,
                    "factors": {
                        "parameter_tuple_id": "g9",
                        "scene_id": "baseline",
                        "object_id": "standard_ball",
                        "camera": "CAM_Side",
                    },
                },
                {
                    "job_id": "robust",
                    "experiment_id": "v1_A",
                    "seed": 1,
                    "factors": {
                        "parameter_tuple_id": "g2",
                        "scene_id": "indoor1",
                        "object_id": "standard_ball",
                        "camera": "CAM_Main",
                    },
                },
            ]
            registry_path = root / "registry.json"
            manifest_path = root / "manifest.jsonl"
            _write_json(registry_path, registry)
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in manifest) + "\n",
                encoding="utf-8",
            )
            bad_root = root / "bad"
            good_root = root / "good"
            for evaluation_root, estimate in ((bad_root, 100.0), (good_root, 10.0)):
                _write_json(
                    evaluation_root / "jobs" / "primary" / "result.json",
                    _result(estimate, manifest[0], registry_path),
                )
                _write_json(
                    evaluation_root / "jobs" / "robust" / "result.json",
                    _result(2.0, manifest[1], registry_path),
                )

            output = root / "report"
            summary = build_simple_physics_report(
                {"bad": bad_root, "good": good_root},
                output=output,
                registry_path=registry_path,
                manifest_path=manifest_path,
            )

            self.assertEqual(summary["models"], ["bad", "good"])
            with (output / "parameter_results.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            bad = next(
                row
                for row in rows
                if row["model"] == "bad" and row["job_id"] == "primary"
            )
            self.assertEqual(float(bad["estimate"]), 100.0)
            self.assertAlmostEqual(float(bad["absolute_error"]), 90.19)
            self.assertAlmostEqual(float(bad["bnae_aux"]), 90.19 / 12.7)
            self.assertEqual(bad["target_range_status"], "out_of_range")
            self.assertAlmostEqual(float(bad["trajectory_r2"]), 1.0)

            overview = (output / "paper_table_overview.md").read_text(encoding="utf-8")
            self.assertIn("v1_A", overview)
            self.assertIn("Out-of-range", overview)
            self.assertIn("13 experiments", overview)
            self.assertNotIn("BNAE↓/Cov", overview)
            self.assertTrue((output / "paper_table_overview.tex").is_file())
            parameter_table = output / "tables" / "v1_A__gravity_g.md"
            self.assertTrue(parameter_table.is_file())
            parameter_text = parameter_table.read_text(encoding="utf-8")
            self.assertIn("N/A", parameter_text)
            self.assertNotIn("**", parameter_text)
            self.assertTrue((output / "paper_parameter_tables.tex").is_file())
            self.assertTrue((output / "report.html").is_file())

            mismatched = _result(10.0, manifest[0], registry_path)
            mismatched["evaluation_lineage"]["evaluator_version"] = "different"
            _write_json(
                good_root / "jobs" / "primary" / "result.json",
                mismatched,
            )
            with self.assertRaises(ValueError):
                build_simple_physics_report(
                    {"bad": bad_root, "good": good_root},
                    output=root / "mixed-lineage",
                    registry_path=registry_path,
                    manifest_path=manifest_path,
                )

            missing_identity = _result(10.0, manifest[0], registry_path)
            del missing_identity["job"]["camera_name"]
            _write_json(
                good_root / "jobs" / "primary" / "result.json",
                missing_identity,
            )
            with self.assertRaises(ValueError):
                build_simple_physics_report(
                    {"good": good_root},
                    output=root / "missing-identity",
                    registry_path=registry_path,
                    manifest_path=manifest_path,
                )

            _write_json(
                good_root / "jobs" / "primary" / "result.json",
                _result(10.0, manifest[0], registry_path),
            )
            reformatted_registry = root / "same-content-different-file.json"
            reformatted_registry.write_text(
                json.dumps(registry, indent=4),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                build_simple_physics_report(
                    {"good": good_root},
                    output=root / "registry-mismatch",
                    registry_path=reformatted_registry,
                    manifest_path=manifest_path,
                )

    def test_missing_result_requires_explicit_partial_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            manifest_path = root / "manifest.jsonl"
            _write_json(
                registry_path,
                {
                    "experiments": [
                        {
                            "id": "v1_A",
                            "hidden_parameters": [
                                {"name": "p", "unit": "1", "valid_range": [0.0, 1.0]}
                            ],
                            "anchor_tuples": [
                                {"id": "low", "p": 0.0},
                                {"id": "high", "p": 1.0},
                            ],
                        }
                    ]
                },
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "job_id": "missing",
                        "experiment_id": "v1_A",
                        "factors": {
                            "parameter_tuple_id": "low",
                            "scene_id": "baseline",
                            "object_id": "ball",
                            "camera": "CAM_Side",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                build_simple_physics_report(
                    {"model": root / "evaluation"},
                    output=root / "strict",
                    registry_path=registry_path,
                    manifest_path=manifest_path,
                )
            summary = build_simple_physics_report(
                {"model": root / "evaluation"},
                output=root / "partial",
                registry_path=registry_path,
                manifest_path=manifest_path,
                allow_partial=True,
            )
            self.assertEqual(summary["completeness"][0]["missing_results"], 1)

    def test_rejected_fit_candidate_is_not_reported_as_an_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            manifest_path = root / "manifest.jsonl"
            _write_json(
                registry_path,
                {
                    "experiments": [
                        {
                            "id": "v1_A",
                            "hidden_parameters": [
                                {"name": "p", "unit": "1", "valid_range": [0.0, 1.0]}
                            ],
                            "anchor_tuples": [
                                {"id": "low", "p": 0.0},
                                {"id": "high", "p": 1.0},
                            ],
                        }
                    ]
                },
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "job_id": "rejected",
                        "experiment_id": "v1_A",
                        "factors": {
                            "parameter_tuple_id": "low",
                            "scene_id": "baseline",
                            "object_id": "ball",
                            "camera": "CAM_Side",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            evaluation = root / "evaluation"
            result = {
                "status": "trajectory_not_eligible",
                "job": {
                    "experiment_id": "v1_A",
                    "parameter_tuple_id": "low",
                    "scene_id": "baseline",
                    "object_id": "ball",
                    "camera_name": "CAM_Side",
                    "video_name": "rejected.mp4",
                },
                "evaluation_lineage": _lineage(registry_path),
                "fit": {
                    "status": "ok",
                    "parameter_observed": {"p": True},
                    "parameter_estimates": {"p": 100.0},
                    "diagnostics": {
                        "position": {
                            "fit_series": {
                                "series_name": "x",
                                "time_s": [0.0, 1.0, 2.0],
                                "observed": [0.0, 1.0, 2.0],
                                "predicted": [0.0, 1.0, 2.0],
                            }
                        }
                    },
                },
            }
            _write_json(evaluation / "jobs" / "rejected" / "result.json", result)
            output = root / "report"
            build_simple_physics_report(
                {"model": evaluation},
                output=output,
                registry_path=registry_path,
                manifest_path=manifest_path,
            )
            with (output / "parameter_results.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["estimate"], "")
            self.assertEqual(row["target_range_status"], "not_estimated")
            with (output / "model_summary.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as handle:
                model_summary = next(csv.DictReader(handle))
            self.assertEqual(float(model_summary["estimate_coverage"]), 0.0)
            self.assertEqual(float(model_summary["median_trajectory_r2"]), 1.0)
            self.assertEqual(float(model_summary["trajectory_r2_coverage"]), 1.0)


if __name__ == "__main__":
    unittest.main()
