from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.evidence_report import (  # noqa: E402
    ArtifactLocator,
    EvidenceReportBuilder,
    _prepare_owned_output,
    build_evidence_report,
)


REGISTRY = CODE_ROOT / "assets" / "seedance978_evaluation" / "experiment_registry.json"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ArtifactLocatorTests(unittest.TestCase):
    def test_prefers_track_overlay_and_portable_evaluation_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = "v1_A__g2p00__baseline__standard_ball__CAM_Side__seed-341867882"
            videos = root / "videos"
            evaluation = root / "evaluation"
            tracks = root / "tracks"
            (videos / f"{job}.mp4").parent.mkdir(parents=True)
            (videos / f"{job}.mp4").write_bytes(b"video")
            (evaluation / "jobs" / job).mkdir(parents=True)
            (evaluation / "jobs" / job / "trajectory.csv").write_text("frame,u_px,v_px\n0,1,2\n", encoding="utf-8")
            (evaluation / "jobs" / job / "validity_object_track_overlay.mp4").write_bytes(b"old")
            (tracks / "jobs" / job).mkdir(parents=True)
            (tracks / "jobs" / job / "object_track_overlay.mp4").write_bytes(b"new")

            found = ArtifactLocator(videos, evaluation, tracks).locate(job)

            self.assertEqual(found.original, videos / f"{job}.mp4")
            self.assertEqual(found.overlay, tracks / "jobs" / job / "object_track_overlay.mp4")
            self.assertEqual(found.trajectory_csv, evaluation / "jobs" / job / "trajectory.csv")


class EvidenceReportBuildTests(unittest.TestCase):
    def test_output_cleanup_requires_report_ownership_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            foreign = root / "foreign"
            foreign.mkdir()
            (foreign / "user.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                _prepare_owned_output(foreign)
            self.assertTrue((foreign / "user.txt").is_file())

            owned = root / "owned"
            _prepare_owned_output(owned)
            stale = owned / "01_unusable" / "stale"
            stale.mkdir(parents=True)
            (stale / "old.html").write_text("old", encoding="utf-8")
            _prepare_owned_output(owned)
            self.assertFalse((owned / "01_unusable").exists())
            self.assertTrue((owned / ".physparambench_evidence_report").is_file())

    def test_manual_pass_overrides_automatic_generation_failure(self) -> None:
        builder = object.__new__(EvidenceReportBuilder)
        builder.video_gate = {}
        builder.dynamic_gate_by_job = {}
        row = {
            "video_name": "manual_pass.mp4",
            "manual_generation_validity_status": "pass",
            "generation_validity_status": "fail",
            "trajectory_fit_eligible": True,
            "fit_complete": True,
            "fit_status": "ok",
            "reconstruction_route": "calibrated_static_sphere",
        }
        self.assertIsNone(builder._unusable_decision(row))
        self.assertTrue(builder._measurement_evaluable(row))
        self.assertEqual(builder._row_status_label(row), "生成可用且已有拟合")

    def test_representative_view_tuple_maximizes_coverage_with_registry_tie_break(self) -> None:
        builder = object.__new__(EvidenceReportBuilder)
        builder.experiments = {
            "vX": {
                "anchor_tuples": [
                    {"id": "first"},
                    {"id": "second"},
                    {"id": "third"},
                ]
            }
        }
        grouped = {
            ("vX", "first", "baseline"): {"CAM_Side": {}},
            ("vX", "second", "baseline"): {"CAM_Side": {}, "CAM_Main": {}, "CAM_Top": {}},
            ("vX", "third", "baseline"): {"CAM_Side": {}, "CAM_Main": {}, "CAM_Top": {}},
        }
        self.assertEqual(builder._representative_view_tuple("vX", grouped), "second")

    def test_builds_human_entrypoints_without_regrading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            evaluation = root / "evaluation"
            tracks = root / "tracks"
            simple = root / "simple"
            output = root / "evidence"
            videos.mkdir()
            simple.mkdir()
            good = "v1_A__g2p00__baseline__standard_ball__CAM_Side__seed-341867882"
            bad = "v1_A__g4p90__baseline__standard_ball__CAM_Side__seed-341867882"
            for job in (good, bad):
                (videos / f"{job}.mp4").write_bytes(b"not-a-real-video")
                job_root = evaluation / "jobs" / job
                job_root.mkdir(parents=True)
                (job_root / "trajectory.csv").write_text(
                    "frame,time_s,u_px,v_px,x_m,y_m,z_m,valid\n0,0,10,20,0,0,1,true\n1,0.04,11,22,0.1,0,0.9,true\n",
                    encoding="utf-8",
                )
                (job_root / "result.json").write_text(
                    json.dumps(
                        {
                            "job": {"video_path": str(videos / f"{job}.mp4")},
                            "fit": {
                                "status": "ok",
                                "method": "robust_airborne_quadratic",
                                "parameter_estimates": {"gravity_g": 3.0},
                                "diagnostics": {},
                                "target_not_used_for_fit": True,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                overlay = tracks / "jobs" / job / "object_track_overlay.mp4"
                overlay.parent.mkdir(parents=True)
                overlay.write_bytes(b"not-a-real-overlay")
            all_jobs = root / "all_jobs.csv"
            _write_csv(
                all_jobs,
                [
                    {
                        "video_name": f"{good}.mp4",
                        "experiment_id": "v1_A",
                        "parameter_tuple_id": "g2p00",
                        "scene_id": "baseline",
                        "camera_name": "CAM_Side",
                        "seed": 341867882,
                        "reconstruction_route": "calibrated_static_sphere",
                        "generation_validity_status": "pass",
                        "trajectory_fit_eligible": True,
                        "fit_complete": True,
                        "fit_status": "ok",
                        "gravity_g__gt": 2.0,
                        "gravity_g__estimate": 3.0,
                    },
                    {
                        "video_name": f"{bad}.mp4",
                        "experiment_id": "v1_A",
                        "parameter_tuple_id": "g4p90",
                        "scene_id": "baseline",
                        "camera_name": "CAM_Side",
                        "seed": 341867882,
                        "reconstruction_route": "calibrated_static_sphere",
                        "generation_validity_status": "pass",
                        "trajectory_fit_eligible": False,
                        "fit_complete": False,
                        "fit_status": "not_attempted",
                        "gravity_g__gt": 4.9,
                    },
                ],
            )
            scan = {
                "experiment_id": "v1_A",
                "target_parameter": "gravity_g",
                "parameter_unit": "m/s^2",
                "valid_range": [2.0, 14.7],
                "seed": 341867882,
                "grade": "L3",
                "reason_codes": ["positive_direction_but_numeric_inaccurate"],
                "evidence": {
                    "pairwise_direction_concordance": 1.0,
                    "theil_sen_slope": 0.5,
                    "median_valid_range_nae": 0.4,
                    "levels": [
                        {
                            "target_value": 2.0,
                            "parameter_tuple_ids": ["g2p00"],
                            "rows": [{"row_id": f"{good}.mp4", "parameter_tuple_id": "g2p00", "state": "usable", "estimate": 3.0}],
                        },
                        {
                            "target_value": 4.9,
                            "parameter_tuple_ids": ["g4p90"],
                            "rows": [{"row_id": f"{bad}.mp4", "parameter_tuple_id": "g4p90", "state": "measurement_or_tracking_insufficient", "estimate": None}],
                        },
                    ],
                },
            }
            (simple / "parameter_scans.jsonl").write_text(json.dumps(scan) + "\n", encoding="utf-8")
            _write_csv(
                simple / "video_gate.csv",
                [
                    {"row_id": f"{good}.mp4", "grade": "PASS_TO_SCAN", "reason_codes": ""},
                    {"row_id": f"{bad}.mp4", "grade": "X", "reason_codes": "trajectory_fit_ineligible"},
                ],
            )

            def fake_plot(_source: Path, destination: Path, **_kwargs: object) -> dict[str, object]:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"png")
                return {"available": True, "status": "rendered"}

            def fake_card(_result: Path, destination: Path, _spec: dict[str, object], **_kwargs: object) -> dict[str, object]:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"png")
                return {"available": True, "status": "rendered"}

            with patch("remake_benchmark.reconstruction.evidence_report.render_trajectory_diagnostics", side_effect=fake_plot), patch(
                "remake_benchmark.reconstruction.evidence_report.render_fit_process_card", side_effect=fake_card
            ):
                summary = build_evidence_report(
                    all_jobs_csv=all_jobs,
                    simple_report_root=simple,
                    evaluation_root=evaluation,
                    tracks_root=tracks,
                    videos_root=videos,
                    output=output,
                    registry_path=REGISTRY,
                    build_montages=False,
                )

            self.assertTrue(summary["read_only_renderer"])
            self.assertEqual(summary["experiment_count"], 13)
            self.assertEqual(summary["parameter_response_axis_count"], 24)
            self.assertTrue((output / "index.html").is_file())
            self.assertTrue((output / "01_unusable" / "X" / bad / "index.html").is_file())
            self.assertTrue((output / "02_parameter_scans" / "v1_A" / "gravity_g" / "FIT_PROCESS_ZH.md").is_file())
            self.assertIn("13 个实验系统", (output / "index.html").read_text(encoding="utf-8"))
            self.assertIn("target_not_used_for_fit=true", (output / "02_parameter_scans" / "v1_A" / "gravity_g" / "FIT_PROCESS_ZH.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
