from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from build_physparambench_simple_report import _evidence_path, build  # noqa: E402


class PhysParamBenchSimpleReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_stale_windows_evidence_path_falls_back_to_job_directory(self) -> None:
        job_root = self.root / "jobs" / "case"
        job_root.mkdir(parents=True)
        local = job_root / "trajectory_frames.csv"
        local.write_text("frame_index,time_s\n", encoding="utf-8")

        resolved = _evidence_path(
            r"C:\stale\server\trajectory_frames.csv",
            job_root,
        )

        self.assertEqual(resolved, local)

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
            "fig5_dynamic_3d_gate.svg",
            "dynamic_3d_gate.csv",
            "measurement_cohorts.csv",
        ):
            self.assertTrue((output / name).is_file(), name)
        ET.parse(output / "fig3_requested_vs_recovered.svg")
        ET.parse(output / "fig5_dynamic_3d_gate.svg")
        report = (output / "REPORT_ZH.md").read_text(encoding="utf-8")
        self.assertIn("Baseline + CAM_Side", report)
        self.assertIn("测量不可用 X", report)
        english = (output / "ABSTRACT_RESULTS_EN.md").read_text(encoding="utf-8")
        self.assertIn("partial parameter sensitivity", english)

    def test_qualified_dynamic_3d_baseline_side_can_enter_primary_scan(self) -> None:
        registry = {
            "experiments": [
                {
                    "id": "v1_C",
                    "hidden_parameters": [
                        {"name": "kinetic_friction_mu", "unit": "1", "valid_range": [0.0, 1.0]}
                    ],
                    "anchor_tuples": [
                        {"id": "mu0", "kinetic_friction_mu": 0.0},
                        {"id": "mu05", "kinetic_friction_mu": 0.5},
                        {"id": "mu1", "kinetic_friction_mu": 1.0},
                    ],
                }
            ]
        }
        policy = {
            "policy_id": "test-dynamic",
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
        evaluation_root = self.root / "evaluation"
        for tuple_id, estimate in (("mu0", 0.02), ("mu05", 0.52), ("mu1", 0.96)):
            job_id = f"v1_C__{tuple_id}__baseline__standard_ball__CAM_Side__seed-341867882"
            rows.append(
                {
                    "video_name": f"{job_id}.mp4",
                    "experiment_id": "v1_C",
                    "parameter_tuple_id": tuple_id,
                    "scene_id": "baseline",
                    "camera_name": "CAM_Side",
                    "seed": "341867882",
                    "reconstruction_route": "spatialtrackerv2_dynamic",
                    "generation_validity_status": "indeterminate",
                    "fit_complete": "True",
                    "trajectory_fit_eligible": "True",
                    "fit_status": "ok",
                    "kinetic_friction_mu__estimate": str(estimate),
                }
            )
            job_root = evaluation_root / "jobs" / job_id
            job_root.mkdir(parents=True)
            (job_root / "result.json").write_text(
                json.dumps(
                    {
                        "pipeline": {
                            "dynamic_reconstruction": {
                                "status": "succeeded",
                                "quality_pass": True,
                            }
                        },
                        "fit": {
                            "status": "ok",
                            "parameter_estimates": {"kinetic_friction_mu": estimate},
                            "parameter_observed": {"kinetic_friction_mu": True},
                            "target_not_used_for_fit": True,
                            "diagnostics": {"fit_nrmse": 0.02},
                        },
                    }
                ),
                encoding="utf-8",
            )
            trajectory_path = job_root / "trajectory_frames.csv"
            with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "frame_index", "time_s", "fit_x_m", "fit_y_m", "fit_z_m",
                        "fit_eligible", "interpolated", "coordinate_frame_3d",
                    ],
                )
                writer.writeheader()
                for index in range(25):
                    writer.writerow(
                        {
                            "frame_index": index,
                            "time_s": index / 24.0,
                            "fit_x_m": index / 24.0,
                            "fit_y_m": 0.005 * math.sin(index),
                            "fit_z_m": 0.005 * math.cos(index),
                            "fit_eligible": True,
                            "interpolated": False,
                            "coordinate_frame_3d": "spatialtrackerv2_frame0_metric_aligned_m",
                        }
                    )
            result_path = job_root / "result.json"
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_payload["source_trajectory_sha256"] = hashlib.sha256(
                trajectory_path.read_bytes()
            ).hexdigest()
            result_path.write_text(json.dumps(result_payload), encoding="utf-8")

        all_jobs = self.root / "dynamic_all_jobs.csv"
        with all_jobs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        registry_path = self.root / "dynamic_registry.json"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        policy_path = self.root / "dynamic_policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "--evaluation-root is required"):
            build(
                all_jobs,
                registry_path,
                policy_path,
                self.root / "dynamic_report_without_evidence",
            )

        summary = build(
            all_jobs,
            registry_path,
            policy_path,
            self.root / "dynamic_report",
            evaluation_root=evaluation_root,
        )

        self.assertEqual(summary["dynamic_3d_inclusion"]["qualified_dynamic_3d_count"], 3)
        self.assertEqual(
            summary["dynamic_3d_inclusion"]["primary_baseline_side_qualified_dynamic_3d_count"],
            3,
        )
        self.assertEqual(summary["primary_baseline_side"]["grade_counts"]["L4"], 1)
        with (self.root / "dynamic_report" / "video_gate.csv").open(encoding="utf-8-sig") as handle:
            video_rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["generation_status"] == "indeterminate" for row in video_rows))
        self.assertTrue(all(row["measurement_route"] == "qualified_dynamic_3d" for row in video_rows))

        tampered = (
            evaluation_root
            / "jobs"
            / "v1_C__mu0__baseline__standard_ball__CAM_Side__seed-341867882"
            / "trajectory_frames.csv"
        )
        tampered.write_text(tampered.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        mismatch_summary = build(
            all_jobs,
            registry_path,
            policy_path,
            self.root / "dynamic_report_hash_mismatch",
            evaluation_root=evaluation_root,
        )
        self.assertEqual(
            mismatch_summary["dynamic_3d_inclusion"]["qualified_dynamic_3d_count"],
            2,
        )
        with (
            self.root / "dynamic_report_hash_mismatch" / "dynamic_3d_gate.csv"
        ).open(encoding="utf-8-sig") as handle:
            mismatch_rows = list(csv.DictReader(handle))
        mismatch = next(row for row in mismatch_rows if row["parameter_tuple_id"] == "mu0")
        self.assertIn("X_DYNAMIC_TRAJECTORY_HASH_MISMATCH", mismatch["reason_codes"])


if __name__ == "__main__":
    unittest.main()
