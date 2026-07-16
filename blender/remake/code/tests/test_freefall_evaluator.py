from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

try:
    import cv2
    import numpy as np

    from remake_benchmark.evaluators.freefall import (
        _deformation_evidence,
        _physical_gate,
        _track_video,
        evaluate_freefall_job,
        fit_vertical_quadratic,
        parameter_similarity_score,
    )

    HAS_VISION_DEPS = True
except ImportError:
    HAS_VISION_DEPS = False


@unittest.skipUnless(HAS_VISION_DEPS, "vision evaluation dependencies are not installed")
class FreefallEvaluatorTests(unittest.TestCase):
    def test_quadratic_fit_recovers_known_acceleration(self) -> None:
        times = np.linspace(0.0, 2.0, 41)
        positions = 30.0 + 4.0 * times + 0.5 * 48.0 * times**2
        fit = fit_vertical_quadratic(times, positions)
        self.assertAlmostEqual(fit["vertical_acceleration_px_s2"], 48.0, places=8)
        self.assertAlmostEqual(fit["fit_r2"], 1.0, places=10)

    def test_parameter_similarity_is_symmetric_ratio(self) -> None:
        self.assertAlmostEqual(parameter_similarity_score(2.0, 4.0), 0.5)
        self.assertAlmostEqual(parameter_similarity_score(6.0, 4.0), 2.0 / 3.0)
        self.assertEqual(parameter_similarity_score(-1.0, 4.0), 0.0)

    def test_deformation_requires_reciprocal_area_preserving_shape_change(self) -> None:
        config = {
            "deformation_min_dimension_change_ratio": 0.22,
            "deformation_max_area_change_ratio": 1.30,
        }
        self.assertTrue(_deformation_evidence((130.0, 77.0), (100.0, 100.0), config))
        self.assertFalse(_deformation_evidence((150.0, 100.0), (100.0, 100.0), config))
        self.assertFalse(_deformation_evidence((120.0, 84.0), (100.0, 100.0), config))

    def test_synthetic_video_produces_track_plot_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory(dir=CODE_ROOT / "tests") as temporary:
            root = Path(temporary)
            image_path = root / "conditioning.png"
            video_path = root / "generated.mp4"
            output_root = root / "eval"
            width, height, fps, frames = 320, 240, 20.0, 72
            acceleration = 40.0
            background = np.full((height, width, 3), 45, dtype=np.uint8)
            cv2.rectangle(background, (24, 180), (296, 208), (40, 95, 150), -1)
            conditioning = background.copy()
            cv2.circle(conditioning, (width // 2, 42), 13, (0, 140, 255), -1)
            self.assertTrue(cv2.imwrite(str(image_path), conditioning))
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            self.assertTrue(writer.isOpened())
            for frame_index in range(frames):
                time_s = frame_index / fps
                center_y = min(167, int(round(42 + 0.5 * acceleration * time_s**2)))
                frame = background.copy()
                cv2.circle(frame, (width // 2, center_y), 13, (0, 140, 255), -1)
                writer.write(frame)
            writer.release()
            job = {
                "job_id": "synthetic_freefall",
                "inputs": {"image": image_path.name},
                "factors": {"scene_id": "synthetic", "object_id": "standard_ball", "camera": "CAM_Side"},
                "targets": {"gravity_g": 0.8},
                "known_params": {
                    "initial_height_z0_m": 2.5,
                    "contact_height_zc_m": 0.0,
                    "drop_distance_m": 2.5,
                },
                "generation": {"fps": fps},
            }
            config = {
                "version": "1.0.0",
                "tracker": "template",
                "min_template_score": 0.10,
                "min_detection_rate": 0.90,
                "min_fit_points": 12,
                "min_fit_r2": 0.95,
                "fit_max_frame_fraction": 0.72,
                "floor_percentile": 92,
                "smoothing_window": 5,
                "overlay_tail_frames": 16,
                "pixels_per_meter": None,
            }
            result = evaluate_freefall_job(
                job,
                video_path,
                root,
                output_root,
                config,
                make_overlay=True,
            )
            self.assertGreater(result["metrics"]["detection_rate"], 0.95)
            self.assertGreater(result["metrics"]["fit_r2"], 0.95)
            self.assertAlmostEqual(result["metrics"]["vertical_acceleration_px_s2"], acceleration, delta=5.0)
            self.assertEqual(result["rigid_body_evaluation"]["status"], "pass")
            self.assertFalse(result["rigid_body_evaluation"]["penetration_detected"])
            self.assertEqual(result["parameter_evaluation"]["status"], "ok")
            self.assertGreater(result["parameter_evaluation"]["similarity_score"], 0.90)
            self.assertTrue(Path(result["artifacts"]["track_csv"]).is_file())
            self.assertTrue(Path(result["artifacts"]["trajectory_plot"]).is_file())
            self.assertTrue(Path(result["artifacts"]["overlay_video"]).is_file())

            top_job = {
                **job,
                "job_id": "synthetic_freefall_top",
                "factors": {**job["factors"], "camera": "CAM_Top"},
            }
            top_result = evaluate_freefall_job(
                top_job,
                video_path,
                root,
                output_root,
                config,
                make_overlay=False,
            )
            self.assertEqual(
                top_result["metrics"]["parameter_identifiability"],
                "endpoint_calibrated_projective_approximation",
            )
            self.assertIsNotNone(top_result["metrics"]["estimated_gravity_m_s2"])
            self.assertGreater(top_result["metrics"]["fit_r2"], 0.95)

            penetrating_video = root / "penetrating.mp4"
            writer = cv2.VideoWriter(str(penetrating_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            self.assertTrue(writer.isOpened())
            for frame_index in range(frames):
                time_s = frame_index / fps
                if 50 <= frame_index < 58:
                    center_y = 195
                elif frame_index >= 58:
                    center_y = 167
                else:
                    center_y = min(167, int(round(42 + 0.5 * acceleration * time_s**2)))
                frame = background.copy()
                cv2.circle(frame, (width // 2, center_y), 13, (0, 140, 255), -1)
                writer.write(frame)
            writer.release()
            penetration_job = {**job, "job_id": "synthetic_penetration"}
            penetration_result = evaluate_freefall_job(
                penetration_job,
                penetrating_video,
                root,
                output_root,
                config,
                make_overlay=False,
            )
            self.assertEqual(penetration_result["rigid_body_evaluation"]["status"], "violation")
            self.assertTrue(penetration_result["rigid_body_evaluation"]["penetration_detected"])
            self.assertEqual(penetration_result["parameter_evaluation"]["status"], "ok")
            self.assertIsNotNone(penetration_result["parameter_evaluation"]["estimated_value"])

    def test_contact_mask_merge_does_not_expand_locked_bbox(self) -> None:
        with tempfile.TemporaryDirectory(dir=CODE_ROOT / "tests") as temporary:
            root = Path(temporary)
            image_path = root / "conditioning.png"
            video_path = root / "contact_merge.mp4"
            width, height, fps, frames = 320, 240, 20.0, 64
            background = np.full((height, width, 3), 45, dtype=np.uint8)
            cv2.rectangle(background, (24, 180), (296, 208), (40, 95, 150), -1)
            conditioning = background.copy()
            cv2.circle(conditioning, (width // 2, 42), 13, (0, 140, 255), -1)
            self.assertTrue(cv2.imwrite(str(image_path), conditioning))
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            self.assertTrue(writer.isOpened())
            for frame_index in range(frames):
                center_y = min(167, 42 + 3 * frame_index)
                frame = background.copy()
                cv2.circle(frame, (width // 2, center_y), 13, (0, 140, 255), -1)
                if frame_index >= 44:
                    # High-chroma pixels touching the ball emulate a mask that
                    # merges with a support/object artifact at landing.
                    cv2.rectangle(frame, (104, 166), (216, 181), (0, 140, 255), -1)
                writer.write(frame)
            writer.release()
            job = {
                "job_id": "synthetic_contact_merge",
                "factors": {"object_id": "standard_ball"},
                "generation": {"fps": fps},
            }
            tracks, _, _ = _track_video(
                job,
                video_path,
                image_path,
                {
                    "tracker": "template",
                    "min_template_score": 0.10,
                    "size_lock_warmup_frames": 3,
                    "max_locked_bbox_dimension_ratio": 1.10,
                    "lock_bbox_size_near_support": True,
                    "deformation_confirmation_frames": 3,
                },
            )
            constrained = [row for row in tracks if row["bbox_size_constrained"]]
            self.assertTrue(constrained)
            self.assertTrue(any("contact-size-locked" in row["tracking_source"] for row in constrained))
            self.assertFalse(any(row["deformation_confirmed"] for row in tracks))
            for row in constrained:
                self.assertLessEqual(
                    float(row["bbox_width_px"]),
                    float(row["bbox_reference_width_px"]) + 1.0,
                )

    def test_support_penetration_is_reported_as_rigid_violation(self) -> None:
        with tempfile.TemporaryDirectory(dir=CODE_ROOT / "tests") as temporary:
            root = Path(temporary)
            image_path = root / "conditioning_with_plank.png"
            width, height = 320, 240
            image = np.full((height, width, 3), 45, dtype=np.uint8)
            cv2.rectangle(image, (24, 180), (296, 208), (40, 95, 150), -1)
            cv2.circle(image, (160, 42), 13, (0, 140, 255), -1)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            tracks = []
            for frame_index in range(28):
                center_y = 42 + 7 * frame_index
                tracks.append(
                    {
                        "found": True,
                        "center_x_px": 160.0,
                        "center_y_px": float(center_y),
                        "bbox_x0": 147,
                        "bbox_y0": center_y - 13,
                        "bbox_x1": 173,
                        "bbox_y1": center_y + 13,
                        "bbox_width_px": 26,
                        "bbox_height_px": 26,
                        "bbox_area_px2": 676,
                        "bbox_aspect_ratio": 1.0,
                    }
                )
            metrics, flags = _physical_gate(
                tracks,
                (height, width),
                image_path,
                {"min_detection_rate": 0.70, "max_penetration_object_heights": 0.60, "min_penetration_frames": 3},
            )
            self.assertTrue(metrics["support_surface_detected"])
            self.assertIn("severe_support_penetration", flags)


if __name__ == "__main__":
    unittest.main()
