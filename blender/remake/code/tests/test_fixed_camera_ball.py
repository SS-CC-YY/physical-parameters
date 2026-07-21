from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

try:
    import numpy as np

    from remake_benchmark.reconstruction.fixed_camera_ball import (
        parse_video_job,
        reconstruct_metric_trajectory,
    )
    from remake_benchmark.reconstruction.v1a_geometry import project_points

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def _look_at(camera_center: "np.ndarray", target: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    forward = target - camera_center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    return rotation, -rotation @ camera_center


@unittest.skipUnless(HAS_DEPS, "numpy/opencv reconstruction dependencies unavailable")
class FixedCameraBallTests(unittest.TestCase):
    def test_filename_parser(self) -> None:
        job = parse_video_job(
            Path("v3_A__drag_high__indoor2__standard_ball__CAM_Top__seed-36.mp4")
        )
        self.assertEqual(job["experiment_id"], "v3_A")
        self.assertEqual(job["parameter_tuple_id"], "drag_high")
        self.assertEqual(job["camera_name"], "CAM_Top")
        self.assertEqual(job["seed"], 36)

    def test_plane_lift_recovers_synthetic_world_trajectory(self) -> None:
        width, height = 640, 360
        K = np.asarray([[800.0, 0.0, 320.0], [0.0, 800.0, 180.0], [0.0, 0.0, 1.0]])
        R, t_camera = _look_at(np.asarray([4.0, -10.0, 4.5]), np.asarray([0.0, 0.0, 1.5]))
        world = np.asarray(
            [[-2.0, 0.0, 2.8], [-1.2, 0.0, 2.3], [-0.3, 0.0, 1.4], [0.8, 0.0, 0.82]],
            dtype=float,
        )
        pixels = project_points(world, K, R, t_camera)
        depths = ((R @ world.T).T + t_camera)[:, 2]
        radius0 = 18.0
        radii = radius0 * depths[0] / depths
        calibration = {
            "image": {"width_px": width, "height_px": height},
            "camera": {
                "K": K.tolist(),
                "R_world_to_camera_opencv": R.tolist(),
                "t_world_to_camera_opencv_m": t_camera.tolist(),
            },
            "object": {
                "initial_center_world_m": world[0].tolist(),
                "known_radius_m": 0.25,
            },
            "projection_validation": {
                "initial_center_uv_from_opencv_P": pixels[0].tolist(),
                "initial_mesh_vertex_bbox_xyxy_px": [
                    pixels[0, 0] - radius0,
                    pixels[0, 1] - radius0,
                    pixels[0, 0] + radius0,
                    pixels[0, 1] + radius0,
                ],
            },
        }
        track = [
            {
                "frame_index": index,
                "found": True,
                "center_u_px": float(pixel[0]),
                "center_v_px": float(pixel[1]),
                "measurement_radius_px": float(radius),
                "track_confidence": 1.0,
            }
            for index, (pixel, radius) in enumerate(zip(pixels, radii))
        ]
        trajectory, summary = reconstruct_metric_trajectory(
            track,
            calibration,
            experiment_id="v3_A",
            video_size=(width, height),
            fps=16.0,
        )
        recovered = np.asarray([[row["x_m"], row["y_m"], row["z_m"]] for row in trajectory])
        np.testing.assert_allclose(recovered, world, atol=1e-7)
        self.assertTrue(summary["camera_pose_fixed_for_all_frames"])
        self.assertAlmostEqual(summary["gt_first_frame_radius_video_px"], radius0)


if __name__ == "__main__":
    unittest.main()
