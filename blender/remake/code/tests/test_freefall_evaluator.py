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

    from remake_benchmark.evaluators.freefall import _physical_gate, evaluate_freefall_job, fit_vertical_quadratic

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

    def test_synthetic_video_produces_track_plot_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory(dir=CODE_ROOT / "tests") as temporary:
            root = Path(temporary)
            image_path = root / "conditioning.png"
            video_path = root / "generated.mp4"
            output_root = root / "eval"
            width, height, fps, frames = 320, 240, 20.0, 48
            acceleration = 40.0
            background = np.full((height, width, 3), 45, dtype=np.uint8)
            conditioning = background.copy()
            cv2.circle(conditioning, (width // 2, 42), 13, (0, 140, 255), -1)
            self.assertTrue(cv2.imwrite(str(image_path), conditioning))
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            self.assertTrue(writer.isOpened())
            for frame_index in range(frames):
                time_s = frame_index / fps
                center_y = int(round(42 + 0.5 * acceleration * time_s**2))
                frame = background.copy()
                cv2.circle(frame, (width // 2, center_y), 13, (0, 140, 255), -1)
                writer.write(frame)
            writer.release()
            job = {
                "job_id": "synthetic_freefall",
                "inputs": {"image": image_path.name},
                "factors": {"scene_id": "synthetic", "object_id": "standard_ball", "camera": "CAM_Side"},
                "targets": {"gravity_g": 9.81},
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
                "image_plane_only_projective_view",
            )
            self.assertIsNone(top_result["metrics"]["estimated_gravity_m_s2"])
            self.assertGreater(top_result["metrics"]["fit_r2"], 0.95)

    def test_support_penetration_is_a_severe_pre_fit_gate(self) -> None:
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
