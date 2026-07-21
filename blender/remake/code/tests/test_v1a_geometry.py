from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

try:
    import cv2
    import numpy as np

    from remake_benchmark.reconstruction.v1a_geometry import (
        compose_projection,
        crop_resize_homography,
        estimate_pose_pnp,
        project_points,
        recover_vertical_line_z,
    )

    HAS_GEOMETRY_DEPS = True
except ImportError:
    HAS_GEOMETRY_DEPS = False


def _look_at_world_to_cv(camera_center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct an OpenCV camera pose with world +Z as the visual up axis."""

    forward = target - camera_center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    translation = -rotation @ camera_center
    return rotation, translation


@unittest.skipUnless(HAS_GEOMETRY_DEPS, "numpy/opencv reconstruction dependencies are unavailable")
class V1AGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 1280
        self.height = 720

    def _intrinsics(self, lens_mm: float) -> np.ndarray:
        focal = self.width * lens_mm / 36.0
        return np.asarray(
            [[focal, 0.0, self.width / 2.0], [0.0, focal, self.height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def test_three_views_project_and_recover_vertical_height(self) -> None:
        cameras = [
            (np.asarray([5.8, -12.8, 4.8]), 36.0),
            (np.asarray([0.0, -13.8, 3.6]), 42.0),
            (np.asarray([-6.0, -13.5, 4.8]), 48.0),
        ]
        heights = (0.44, 1.25, 2.70, 4.20)
        for center, lens in cameras:
            with self.subTest(camera=center.tolist()):
                rotation, translation = _look_at_world_to_cv(center, np.asarray([0.0, 0.0, 2.35]))
                intrinsics = self._intrinsics(lens)
                projection = compose_projection(intrinsics, rotation, translation)
                points = np.asarray([[0.0, 0.0, height] for height in heights])
                pixels = project_points(points, intrinsics, rotation, translation)
                for expected_z, pixel in zip(heights, pixels):
                    recovered = recover_vertical_line_z(pixel, projection, z_bounds=(0.44, 4.20))
                    self.assertTrue(recovered["success"])
                    self.assertAlmostEqual(recovered["z"], expected_z, places=8)
                    self.assertLess(recovered["residual_px"], 1e-8)
                    self.assertGreater(recovered["line_observability_px_per_m"], 10.0)

    def test_crop_resize_homography_transforms_pixels_and_intrinsics(self) -> None:
        source_size = (1280, 720)
        target_size = (864, 496)
        transform = crop_resize_homography(source_size, target_size)
        crop_width = round(720 * (864 / 496))
        crop_x = (1280 - crop_width) // 2
        self.assertAlmostEqual(transform[0, 0], 864 / crop_width)
        self.assertAlmostEqual(transform[1, 1], 496 / 720)
        self.assertAlmostEqual(transform[0, 2], -crop_x * 864 / crop_width)

        intrinsics = self._intrinsics(36.0)
        rotation, translation = _look_at_world_to_cv(
            np.asarray([5.8, -12.8, 4.8]),
            np.asarray([0.0, 0.0, 2.35]),
        )
        points = np.asarray([[0.0, 0.0, 4.2], [0.0, 0.0, 0.44], [1.0, 0.5, 2.0]])
        source_pixels = project_points(points, intrinsics, rotation, translation)
        homogeneous = np.column_stack((source_pixels, np.ones(len(source_pixels))))
        expected = (transform @ homogeneous.T).T
        expected = expected[:, :2] / expected[:, 2, None]
        transformed_intrinsics = transform @ intrinsics
        actual = project_points(points, transformed_intrinsics, rotation, translation)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_pnp_recovers_synthetic_six_dof_drift_and_rejects_outliers(self) -> None:
        rng = np.random.default_rng(20260720)
        intrinsics = self._intrinsics(42.0)
        reference_center = np.asarray([0.0, -13.8, 3.6])
        reference_rotation, _ = _look_at_world_to_cv(reference_center, np.asarray([0.0, 0.0, 2.35]))

        # Apply a genuine 6DoF change, not a 2D affine image warp.
        drift_rvec = np.deg2rad(np.asarray([1.7, -2.1, 1.2], dtype=np.float64))
        drift_rotation, _ = cv2.Rodrigues(drift_rvec)
        true_rotation = drift_rotation @ reference_rotation
        true_center = reference_center + np.asarray([0.28, -0.16, 0.19])
        true_translation = -true_rotation @ true_center

        candidates = np.column_stack(
            (
                rng.uniform(-6.0, 6.0, 1200),
                rng.uniform(-0.5, 9.0, 1200),
                rng.uniform(0.1, 6.8, 1200),
            )
        )
        pixels = project_points(candidates, intrinsics, true_rotation, true_translation)
        camera_points = (true_rotation @ candidates.T).T + true_translation
        visible = (
            (camera_points[:, 2] > 0)
            & (pixels[:, 0] >= 20)
            & (pixels[:, 0] < self.width - 20)
            & (pixels[:, 1] >= 20)
            & (pixels[:, 1] < self.height - 20)
        )
        world = candidates[visible][:240]
        observations = pixels[visible][:240]
        self.assertGreaterEqual(len(world), 180)
        observations = observations + rng.normal(0.0, 0.15, observations.shape)
        outlier_rows = rng.choice(len(world), size=48, replace=False)
        observations[outlier_rows] += rng.uniform(-140.0, 140.0, (len(outlier_rows), 2))

        estimate = estimate_pose_pnp(
            world,
            observations,
            intrinsics,
            ransac_reprojection_error_px=1.5,
            quality_thresholds={"max_reprojection_p95_px": 0.8},
        )
        self.assertTrue(estimate["success"])
        self.assertTrue(estimate["quality"]["passed"])
        self.assertGreaterEqual(estimate["inlier_count"], len(world) - len(outlier_rows) - 5)
        self.assertLess(estimate["inlier_ratio"], 0.90)
        self.assertLess(estimate["reprojection_p95_px"], 0.5)
        self.assertTrue(set(estimate["inlier_indices"]).isdisjoint(set(outlier_rows.tolist())))

        estimated_rotation = np.asarray(estimate["R"])
        estimated_translation = np.asarray(estimate["t"])
        delta_rotation = estimated_rotation @ true_rotation.T
        cosine = np.clip((np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_error_deg = np.rad2deg(np.arccos(cosine))
        self.assertLess(rotation_error_deg, 0.05)
        self.assertLess(np.linalg.norm(estimated_translation - true_translation), 0.02)

    def test_pnp_reports_insufficient_correspondences(self) -> None:
        estimate = estimate_pose_pnp(
            np.zeros((5, 3)),
            np.zeros((5, 2)),
            self._intrinsics(42.0),
        )
        self.assertFalse(estimate["success"])
        self.assertEqual(estimate["reason"], "insufficient_correspondences")
        self.assertFalse(estimate["quality"]["passed"])


if __name__ == "__main__":
    unittest.main()
