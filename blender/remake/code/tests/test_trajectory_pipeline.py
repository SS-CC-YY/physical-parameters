from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from remake_benchmark.reconstruction import trajectory_pipeline as pipeline
from remake_benchmark.reconstruction.seedance978 import (  # noqa: E402
    _seed_stability_rows,
    _viewpoint_rows,
    write_seedance978_reports,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_video(path: Path, frame_count: int, fps: float = 12.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("test VideoWriter unavailable")
    for index in range(frame_count):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TrajectoryPipelineTests(unittest.TestCase):
    @staticmethod
    def _summary_result(
        *,
        camera: str = "CAM_Side",
        seed: int = 341867882,
        estimate: float = 0.2,
        generation_status: str = "pass",
        measurement_evaluable: bool = True,
        dynamic_decision: str | None = None,
    ) -> dict:
        return {
            "job": {
                "video_name": f"case-{camera}-{seed}.mp4",
                "experiment_id": "v1_C",
                "parameter_tuple_id": "mu0p18",
                "scene_id": "baseline",
                "object_id": "standard_ball",
                "camera_name": camera,
                "seed": seed,
            },
            "benchmark_split": "side_primary" if camera == "CAM_Side" else "view_robustness",
            "reconstruction_route": (
                "spatialtrackerv2_dynamic" if dynamic_decision is not None else "calibrated_static_sphere"
            ),
            "dynamic_3d_inclusion": (
                None if dynamic_decision is None else {"decision": dynamic_decision, "reason_codes": []}
            ),
            "video_generation_validity": {"status": generation_status},
            "trajectory_fit_eligibility": {
                "eligible": measurement_evaluable,
                "status": "qualified_dynamic_3d" if measurement_evaluable else "blocked",
            },
            "fit": {
                "status": "ok",
                "parameter_estimates": {"kinetic_friction_mu": estimate},
            },
            "metrics": {
                "fit_complete": measurement_evaluable,
                "parameters": {},
                "experiment_score_0_100": 80.0 if measurement_evaluable else None,
            },
            "pipeline": {"tracking": {}},
            "error": None,
        }

    def test_view_and_seed_reports_separate_generation_validity_from_measurement_evidence(self) -> None:
        side = self._summary_result(estimate=0.2)
        main = self._summary_result(
            camera="CAM_Main",
            estimate=0.9,
            generation_status="indeterminate",
            measurement_evaluable=False,
            dynamic_decision="X",
        )
        top = self._summary_result(camera="CAM_Top", estimate=0.3)
        ranges = {"v1_C": {"kinetic_friction_mu": 1.0}}

        view = _viewpoint_rows([side, main, top], ranges)[0]

        self.assertFalse(view["all_three_generation_valid"])
        self.assertFalse(view["all_three_measurement_evaluable"])
        self.assertIsNone(view["main_vs_side_parameter_range_normalized_mae"])
        self.assertAlmostEqual(view["top_vs_side_parameter_range_normalized_mae"], 0.1)
        self.assertIsNone(view["main_vs_side_score_abs_delta"])
        self.assertEqual(view["top_vs_side_score_abs_delta"], 0.0)

        dynamic_evaluable = self._summary_result(
            seed=2,
            estimate=0.4,
            generation_status="indeterminate",
            measurement_evaluable=True,
            dynamic_decision="include",
        )
        dynamic_x = self._summary_result(
            seed=3,
            estimate=0.95,
            generation_status="pass",
            measurement_evaluable=False,
            dynamic_decision="X",
        )
        stability = _seed_stability_rows([side, dynamic_evaluable, dynamic_x], ranges)[0]

        self.assertEqual(stability["generation_valid_count"], 2)
        self.assertEqual(stability["measurement_evaluable_count"], 2)
        # The rejected 0.95 estimate must not inflate dispersion.
        self.assertAlmostEqual(stability["mean_parameter_range_normalized_std"], 2 ** -0.5 * 0.2)

    def test_frozen_soft_review_is_eligible_while_failure_code_blocks(self) -> None:
        soft = pipeline._assess_frozen_trajectory_eligibility(
            {
                "status": "review",
                "fit_eligible": True,
                "failure_codes": [],
                "warning_codes": ["shape_requires_review"],
            },
            [],
        )
        self.assertTrue(soft["eligible"])
        self.assertTrue(soft["generation_review_provisional"])

        hard = pipeline._assess_frozen_trajectory_eligibility(
            {
                "status": "review",
                "fit_eligible": True,
                "failure_codes": ["confirmed_object_disappearance"],
            },
            [],
        )
        self.assertFalse(hard["eligible"])
        self.assertTrue(hard["generation_hard_failure"])

    def test_standard_headline_is_baseline_side_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            results = []
            manifest = []
            for scene, score in (("baseline", 90.0), ("indoor1", 10.0)):
                job_id = f"v1_C__mu0p18__{scene}__standard_ball__CAM_Side__seed-341867882"
                result = self._summary_result(estimate=0.2)
                result["job"]["video_name"] = f"{job_id}.mp4"
                result["job"]["scene_id"] = scene
                result["metrics"].update(
                    {
                        "experiment_nmae": (100.0 - score) / 100.0,
                        "experiment_score_0_100": score,
                    }
                )
                result["benchmark_split"] = "side_primary"
                result_path = output / "jobs" / job_id / "result.json"
                result_path.parent.mkdir(parents=True)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                results.append(result)
                manifest.append(
                    {
                        "job_id": job_id,
                        "seed": 341867882,
                        "factors": {
                            "camera": "CAM_Side",
                            "scene_id": scene,
                        },
                    }
                )

            aggregate = write_seedance978_reports(
                output,
                manifest,
                {
                    "experiments": [
                        {
                            "id": "v1_C",
                            "hidden_parameters": [
                                {"name": "kinetic_friction_mu", "valid_range": [0.0, 1.0]}
                            ],
                        }
                    ]
                },
            )

            self.assertEqual(aggregate["headline"]["name"], "Baseline CAM_Side primary physics identification")
            self.assertEqual(aggregate["headline"]["scheduled_count"], 1)
            self.assertEqual(aggregate["headline"]["result_count"], 1)
            self.assertEqual(aggregate["side_primary_all_scenes_auxiliary"]["result_count"], 2)
            with (output / "side_primary_summary.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            with (output / "side_primary_all_scenes_summary.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_failed_or_truncated_track_result_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "jobs" / "case" / "track_result.json"
            result_path.parent.mkdir(parents=True)
            video_path = root / "case.mp4"
            result_path.write_text(
                json.dumps(
                    {
                        "schema_version": pipeline.TRACK_SCHEMA_VERSION,
                        "status": "failed",
                        "target_parameters_used": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(pipeline._reusable_track_result(result_path, video_path))
            result_path.write_text('{"status":', encoding="utf-8")
            self.assertIsNone(pipeline._reusable_track_result(result_path, video_path))

    def test_complete_result_reuse_checks_decoded_overlay_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "jobs" / "case"
            video_path = root / "case.mp4"
            overlay_path = job_dir / "object_track_overlay.mp4"
            _write_video(video_path, 4)
            _write_video(overlay_path, 4)
            rows = []
            for index in range(4):
                row = {field: None for field in pipeline.COMMON_FIELDS}
                row.update(
                    {
                        "frame_index": index,
                        "source_frame_index": index,
                        "time_s": index / 12.0,
                        "route": pipeline.STATIC_ROUTE,
                        "position_source": "unavailable",
                        "valid_2d": False,
                        "valid_3d": False,
                        "measurement_valid": False,
                        "interpolated": False,
                        "fit_eligible": False,
                    }
                )
                rows.append(row)
            pipeline._write_rows(job_dir / "trajectory_frames.csv", rows, pipeline.COMMON_FIELDS)
            result_path = job_dir / "track_result.json"
            pipeline._write_json(
                result_path,
                {
                    "schema_version": pipeline.TRACK_SCHEMA_VERSION,
                    "status": "partial",
                    "target_parameters_used": False,
                    "reconstruction_route": pipeline.STATIC_ROUTE,
                },
            )
            self.assertIsNotNone(pipeline._reusable_track_result(result_path, video_path))
            self.assertIsNotNone(
                pipeline._reusable_track_result(
                    result_path,
                    video_path,
                    expected_route=pipeline.STATIC_ROUTE,
                )
            )
            self.assertIsNone(
                pipeline._reusable_track_result(
                    result_path,
                    video_path,
                    expected_route=pipeline.DYNAMIC_ROUTE,
                )
            )
            _write_video(overlay_path, 3)
            self.assertIsNone(pipeline._reusable_track_result(result_path, video_path))

    def test_static_contract_keeps_every_frame_and_excludes_predictions_from_fit(self) -> None:
        track = [
            {
                "frame_index": 0,
                "found": True,
                "identity_verified": True,
                "measurement_valid": True,
                "center_u_px": 10.0,
                "center_v_px": 20.0,
                "measurement_radius_px": 5.0,
                "track_confidence": 0.9,
            },
            {
                "frame_index": 1,
                "found": False,
                "measurement_valid": False,
                "predicted_center_u_px": 11.0,
                "predicted_center_v_px": 21.0,
                "display_radius_px": 5.0,
            },
            {
                "frame_index": 2,
                "found": False,
                "measurement_valid": False,
                "continuous_center_u_px": 12.0,
                "continuous_center_v_px": 22.0,
                "continuous_radius_px": 5.0,
                "continuous_source": "bracketed_linear_interpolation",
            },
        ]
        metric = [
            {
                "frame_index": 0,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "physics_fit_used": True,
                "continuous_metric_available": True,
                "continuous_x_m": 99.0,
                "continuous_y_m": 98.0,
                "continuous_z_m": 97.0,
            },
            {"frame_index": 1, "x_m": None, "y_m": None, "z_m": None, "physics_fit_used": False},
            {
                "frame_index": 2,
                "continuous_metric_available": True,
                "continuous_x_m": 0.0,
                "continuous_y_m": 0.0,
                "continuous_z_m": 0.8,
                "physics_fit_used": False,
            },
        ]

        rows = pipeline.normalize_static_positions(track, metric, fps=24.0)

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["source_frame_index"] for row in rows], [0, 1, 2])
        self.assertTrue(rows[0]["fit_eligible"])
        self.assertEqual(rows[0]["x_m"], 99.0)
        self.assertEqual(rows[0]["fit_x_m"], 0.0)
        self.assertEqual(rows[0]["fit_z_m"], 1.0)
        self.assertEqual(rows[1]["position_source"], "motion_prediction_unverified")
        self.assertFalse(rows[1]["fit_eligible"])
        self.assertTrue(rows[2]["interpolated"])
        self.assertTrue(rows[2]["valid_3d"])
        self.assertFalse(rows[2]["fit_eligible"])

    def test_dynamic_contract_writes_2d_centres_and_rejects_alignment_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dynamic = Path(temporary) / "dynamic"
            dynamic.mkdir()
            tracks = np.asarray(
                [
                    [[10, 20], [12, 20], [10, 22], [12, 22]],
                    [[11, 21], [13, 21], [11, 23], [13, 23]],
                    [[12, 22], [14, 22], [12, 24], [14, 24]],
                ],
                dtype=np.float32,
            )
            visible = np.asarray(
                [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.float32
            )
            np.savez(
                dynamic / "raw_spatialtrackerv2.npz",
                track2d_input=tracks,
                visibs=visible,
                track_confidence=np.full((3, 4), 0.8, dtype=np.float32),
                query_meta_query_kind=np.asarray(["object"] * 4),
                preprocess_scale_xy=np.asarray([1.0, 1.0]),
                preprocess_crop_top=np.asarray(0.0),
                source_frame_indices=np.asarray([0, 1, 2]),
                source_fps=np.asarray(24.0),
            )
            _write_csv(
                dynamic / "trajectory_world.csv",
                [
                    {
                        "source_frame": index,
                        "time_s": index / 24,
                        "x_m": 10.0 + index * 0.1,
                        "y_m": 0.0,
                        "z_m": 1.0 - index * 0.1,
                        "x_raw_m": index * 0.1 if index < 2 else None,
                        "y_raw_m": 0.0 if index < 2 else None,
                        "z_raw_m": 1.0 - index * 0.1 if index < 2 else None,
                        "object_inlier_fraction": 0.8,
                        "object_rigid_rmse_m": 0.01,
                        "object_all_point_rmse_over_radius": 0.1,
                    }
                    for index in range(3)
                ],
            )
            (dynamic / "alignment.json").write_text(
                json.dumps(
                    [
                        {"frame_index": 0, "fallback": False, "inlier_fraction": 0.9, "rmse_m": 0.01},
                        {"frame_index": 1, "fallback": False, "inlier_fraction": 0.3, "rmse_m": 0.4},
                        {"frame_index": 2, "fallback": True, "inlier_fraction": 0.9, "rmse_m": None},
                    ]
                ),
                encoding="utf-8",
            )

            rows = pipeline.normalize_dynamic_positions(dynamic)

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["center_u_px"], 11.0)
            self.assertEqual(rows[0]["center_v_px"], 21.0)
            self.assertTrue(rows[0]["fit_eligible"])
            self.assertEqual(rows[0]["x_m"], 10.0)
            self.assertEqual(rows[0]["fit_x_m"], 0.0)
            self.assertFalse(rows[1]["alignment_fallback"])
            self.assertFalse(rows[1]["fit_eligible"])
            self.assertIn("background_alignment_quality_low", rows[1]["invalid_reason_codes"])
            self.assertTrue(rows[2]["valid_2d"])
            self.assertFalse(rows[2]["measurement_valid"])
            self.assertTrue(rows[2]["interpolated"])
            self.assertEqual(rows[2]["position_source"], "temporal_filter_imputation")

    def test_evaluation_consumes_frozen_csv_without_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "v1_A__g9p81__baseline__standard_ball__CAM_Side__seed-341867882"
            extraction_root = root / "tracks"
            job_dir = extraction_root / "jobs" / job_id
            trajectory = job_dir / "trajectory_frames.csv"
            _write_csv(
                trajectory,
                [
                    {
                        "frame_index": index,
                        "source_frame_index": index,
                        "time_s": index / 24,
                        "x_m": 0.0,
                        "y_m": 0.0,
                        "z_m": 1.0 - 0.1 * index,
                        "fit_x_m": 0.0,
                        "fit_y_m": 0.0,
                        "fit_z_m": 1.0 - 0.1 * index,
                        "fit_eligible": True,
                        "interpolated": False,
                    }
                    for index in range(4)
                ],
            )
            job = {
                "experiment_id": "v1_A",
                "parameter_tuple_id": "g9p81",
                "scene_id": "baseline",
                "object_id": "standard_ball",
                "camera_name": "CAM_Side",
                "seed": 341867882,
            }
            (job_dir / "track_result.json").write_text(
                json.dumps(
                    {
                        "job": job,
                        "benchmark_split": "side_primary",
                        "reconstruction_route": "calibrated_static_sphere",
                        "trajectory_frames_csv": str(trajectory),
                        "trajectory_fit_eligible": True,
                        "video_generation_validity": {"status": "pass", "fit_eligible": True},
                        "pipeline": {},
                        "camera_motion_evidence": {},
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "experiment_id": "v1_A",
                        "seed": 341867882,
                        "inputs": {},
                        "factors": {
                            "parameter_tuple_id": "g9p81",
                            "scene_id": "baseline",
                            "object_id": "standard_ball",
                            "camera": "CAM_Side",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "id": "v1_A",
                                "hidden_parameters": [
                                    {"name": "gravity_g", "unit": "m/s^2", "valid_range": [2.0, 14.7]}
                                ],
                                "anchor_tuples": [{"id": "g9p81", "gravity_g": 9.81}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            fake_fit = {
                "experiment_id": "v1_A",
                "status": "complete",
                "parameter_estimates": {"gravity_g": 9.5},
                "parameter_observed": {"gravity_g": True},
                "diagnostics": {},
                "target_not_used_for_fit": True,
            }
            with (
                patch.object(pipeline, "fit_physics_parameters", return_value=fake_fit) as fit_mock,
                patch.object(
                    pipeline,
                    "score_parameter_fit",
                    return_value={"status": "scored", "fit_complete": True, "parameters": {}},
                ),
                patch.object(pipeline, "write_trajectory_plot", return_value=None),
                patch.object(pipeline, "write_seedance978_reports", return_value={"ok": True}),
            ):
                output = root / "evaluation"
                result = pipeline.evaluate_extracted_seedance978(
                    extraction_root=extraction_root,
                    manifest_path=manifest,
                    registry_path=registry,
                    output_root=output,
                    phase="side",
                )

                portable = output / "jobs" / job_id / "trajectory_frames.csv"
                self.assertEqual(portable.read_bytes(), trajectory.read_bytes())
                portable.unlink()

                # An unchanged lineage is reused without invoking the fitter,
                # while a missing portable evidence copy is repaired.
                pipeline.evaluate_extracted_seedance978(
                    extraction_root=extraction_root,
                    manifest_path=manifest,
                    registry_path=registry,
                    output_root=output,
                    phase="side",
                )
                self.assertEqual(fit_mock.call_count, 1)
                self.assertEqual(portable.read_bytes(), trajectory.read_bytes())

                # Changing only the frozen trajectory invalidates the cache.
                with trajectory.open(encoding="utf-8") as handle:
                    changed_rows = list(csv.DictReader(handle))
                changed_rows[0]["fit_z_m"] = "0.75"
                _write_csv(trajectory, changed_rows)
                pipeline.evaluate_extracted_seedance978(
                    extraction_root=extraction_root,
                    manifest_path=manifest,
                    registry_path=registry,
                    output_root=output,
                    phase="side",
                )
                self.assertEqual(fit_mock.call_count, 2)

            self.assertEqual(result, {"ok": True})
            evaluated = json.loads((output / "jobs" / job_id / "result.json").read_text(encoding="utf-8"))
            self.assertFalse(evaluated["evaluation_reads_video"])
            self.assertFalse(evaluated["evaluation_invokes_tracker"])
            self.assertTrue(evaluated["fit_attempted"])
            self.assertTrue(evaluated["target_lookup_performed"])
            fitted_rows = fit_mock.call_args.args[1]
            self.assertEqual([row["x_m"] for row in fitted_rows], [0.0] * 4)

    def test_qualified_dynamic_3d_track_runs_target_free_fitter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "v1_C__mu0p18__baseline__standard_ball__CAM_Side__seed-341867882"
            extraction_root = root / "tracks"
            job_dir = extraction_root / "jobs" / job_id
            trajectory = job_dir / "trajectory_frames.csv"
            _write_csv(
                trajectory,
                [
                    {
                        "frame_index": index,
                        "source_frame_index": index,
                        "time_s": index / 24.0,
                        "x_m": index / 24.0,
                        "y_m": 0.004 * np.sin(index),
                        "z_m": 0.004 * np.cos(index),
                        "fit_x_m": index / 24.0,
                        "fit_y_m": 0.004 * np.sin(index),
                        "fit_z_m": 0.004 * np.cos(index),
                        "coordinate_frame_3d": "spatialtrackerv2_frame0_metric_aligned_m",
                        "fit_eligible": True,
                        "measurement_valid": True,
                        "interpolated": False,
                    }
                    for index in range(25)
                ],
            )
            job = {
                "experiment_id": "v1_C",
                "parameter_tuple_id": "mu0p18",
                "scene_id": "baseline",
                "object_id": "standard_ball",
                "camera_name": "CAM_Side",
                "seed": 341867882,
                "video_name": f"{job_id}.mp4",
            }
            (job_dir / "track_result.json").write_text(
                json.dumps(
                    {
                        "job": job,
                        "benchmark_split": "side_primary",
                        "reconstruction_route": "spatialtrackerv2_dynamic",
                        "trajectory_frames_csv": str(trajectory),
                        "trajectory_fit_eligible": True,
                        "video_generation_validity": {
                            "status": "indeterminate",
                            "fit_eligible": False,
                            "failure_codes": [],
                            "indeterminate_codes": ["moving_camera_scene_rigidity_unresolved"],
                        },
                        "pipeline": {
                            "dynamic_reconstruction": {
                                "status": "succeeded",
                                "quality_pass": True,
                                "anchor_reference_mode": "self_frame0",
                            }
                        },
                        "camera_motion_evidence": {"final_category": "camera_changed"},
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "experiment_id": "v1_C",
                        "seed": 341867882,
                        "inputs": {},
                        "factors": {
                            "parameter_tuple_id": "mu0p18",
                            "scene_id": "baseline",
                            "object_id": "standard_ball",
                            "camera": "CAM_Side",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "id": "v1_C",
                                "hidden_parameters": [
                                    {
                                        "name": "kinetic_friction_mu",
                                        "unit": "1",
                                        "valid_range": [0.0, 0.5],
                                    }
                                ],
                                "anchor_tuples": [
                                    {"id": "mu0p18", "kinetic_friction_mu": 0.18}
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            fake_fit = {
                "experiment_id": "v1_C",
                "status": "ok",
                "parameter_estimates": {"kinetic_friction_mu": 0.2},
                "parameter_observed": {"kinetic_friction_mu": True},
                "diagnostics": {"fit_nrmse": 0.05},
                "target_not_used_for_fit": True,
            }
            with (
                patch.object(pipeline, "fit_physics_parameters", return_value=fake_fit) as fit_mock,
                patch.object(
                    pipeline,
                    "score_parameter_fit",
                    return_value={"status": "scored", "fit_complete": True, "parameters": {}},
                ),
                patch.object(pipeline, "write_trajectory_plot", return_value=None),
                patch.object(pipeline, "write_seedance978_reports", return_value={"ok": True}),
            ):
                output = root / "evaluation"
                pipeline.evaluate_extracted_seedance978(
                    extraction_root=extraction_root,
                    manifest_path=manifest,
                    registry_path=registry,
                    output_root=output,
                    phase="side",
                )

            self.assertEqual(fit_mock.call_count, 1)
            evaluated = json.loads((output / "jobs" / job_id / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(evaluated["dynamic_3d_inclusion"]["decision"], "include")
            self.assertEqual(evaluated["trajectory_fit_eligibility"]["status"], "qualified_dynamic_3d")
            self.assertTrue(evaluated["fit_attempted"])
            self.assertTrue(evaluated["target_lookup_performed"])


if __name__ == "__main__":
    unittest.main()
