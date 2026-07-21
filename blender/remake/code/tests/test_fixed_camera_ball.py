from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

try:
    import cv2
    import numpy as np

    from remake_benchmark.reconstruction.fixed_camera_ball import (
        SphereCandidate,
        _choose_candidate,
        _edge_ring_evidence,
        _hough_candidates,
        _reference_change_mask,
        _with_reference_change,
        parse_video_job,
        reconstruct_metric_trajectory,
        track_standard_ball,
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

        # Reconstruction must never revive a row that the tracker marked as a
        # prediction, unverified identity, invalid measurement, or low
        # confidence observation.  Missing new fields remains backwards
        # compatible with the legacy measured rows above.
        invalid_updates = [
            {"observation_status": "predicted"},
            {"identity_verified": False},
            {"measurement_valid": False},
            {"track_confidence": 0.05},
        ]
        for updates in invalid_updates:
            gated_track = [dict(row) for row in track]
            gated_track[1].update(updates)
            gated, gated_summary = reconstruct_metric_trajectory(
                gated_track,
                calibration,
                experiment_id="v3_A",
                video_size=(width, height),
                fps=16.0,
            )
            self.assertFalse(gated[1]["measurement_valid"])
            self.assertFalse(gated[1]["physics_fit_used"])
            self.assertFalse(gated[1]["tracker_measurement_eligible"])
            self.assertIsNone(gated[1]["x_m"])
            self.assertEqual(gated_summary["tracker_eligible_frame_count"], len(track) - 1)

    @staticmethod
    def _draw_synthetic_sphere(
        frame: "np.ndarray",
        center: tuple[int, int],
        radius: int,
    ) -> None:
        # Two warm tones remain one HSV component while providing a strong,
        # deterministic circular edge for the local Hough fallback.
        cv2.circle(frame, center, radius, (0, 100, 200), -1, cv2.LINE_AA)
        cv2.circle(frame, center, max(2, radius - 3), (0, 140, 255), -1, cv2.LINE_AA)

    def test_landing_component_merge_uses_local_hough_not_background_prop(self) -> None:
        radius = 16
        frames = []
        heights = [50, 60, 72, 86, 102, 120, 140, 158, 164] + [164] * 7
        for height in heights:
            frame = np.full((240, 320, 3), 180, dtype=np.uint8)
            # The floor deliberately has the same hue as the ball.  At contact,
            # morphology merges both into an over-sized segmentation component.
            cv2.rectangle(frame, (0, 180), (319, 205), (0, 90, 180), -1)
            # A persistent same-colour distractor tests that fallback remains in
            # the prediction ROI instead of scanning arbitrary background.
            self._draw_synthetic_sphere(frame, (270, 100), radius)
            self._draw_synthetic_sphere(frame, (80, height), radius)
            frames.append(frame)

        rows, summary = track_standard_ball(frames, (80.0, 50.0), float(radius))

        self.assertEqual(summary["found_count"], len(frames))
        for row in rows:
            self.assertTrue(row["identity_verified"])
            self.assertEqual(row["observation_status"], "measured")
            self.assertTrue(row["measurement_valid"])
            self.assertLess(abs(float(row["center_u_px"]) - 80.0), 2.0)
            self.assertAlmostEqual(float(row["display_radius_px"]), float(radius))
        self.assertTrue(summary["display_radius_locked_to_calibrated_first_frame"])
        contact_rows = rows[8:]
        self.assertTrue(
            any(row["measurement_source"] == "hough_circle_fallback" for row in contact_rows)
        )
        for row in contact_rows:
            if row["measurement_source"] == "hough_circle_fallback":
                self.assertGreaterEqual(float(row["colour_support_fraction"]), 0.30)
                if row["shape_evidence_available"]:
                    self.assertEqual(row["shape_evidence_source"], "hough_edge_ring")
                    self.assertGreaterEqual(float(row["edge_support_fraction"]), 0.55)
                    self.assertLessEqual(float(row["edge_radial_residual_ratio"]), 0.10)
                else:
                    self.assertIsNone(row["shape_evidence_source"])
                self.assertIsNone(row["circularity"])
                self.assertIsNone(row["ellipse_axis_ratio"])
                self.assertIsNone(row["radial_residual_ratio"])

    def test_large_jump_is_missing_and_does_not_poison_reacquisition(self) -> None:
        radius = 14
        frames = []
        true_centers: list[tuple[int, int] | None] = [
            (40, 120),
            (50, 120),
            (60, 120),
            (70, 120),
            (80, 120),
            None,
            None,
            None,
            (120, 120),
            (130, 120),
            (140, 120),
        ]
        for center in true_centers:
            frame = np.full((220, 300, 3), 170, dtype=np.uint8)
            # This larger orange prop exists in every frame.  It is deliberately
            # within the legacy tracker's gap-expanded full-frame search range.
            self._draw_synthetic_sphere(frame, (225, 115), 20)
            if center is not None:
                self._draw_synthetic_sphere(frame, center, radius)
            frames.append(frame)

        rows, _ = track_standard_ball(frames, (40.0, 120.0), float(radius))

        for frame_index in (5, 6, 7):
            row = rows[frame_index]
            self.assertFalse(row["found"])
            self.assertEqual(row["observation_status"], "missing")
            self.assertFalse(row["identity_verified"])
            self.assertFalse(row["measurement_valid"])
            self.assertIsNotNone(row["display_radius_px"])
        for frame_index in (8, 9, 10):
            row = rows[frame_index]
            self.assertTrue(row["found"])
            self.assertTrue(row["identity_verified"])
            self.assertLess(abs(float(row["center_u_px"]) - true_centers[frame_index][0]), 2.5)
            self.assertLess(float(row["center_u_px"]), 170.0)
            self.assertLessEqual(float(row["association_cost"]), 1.20)
            self.assertGreaterEqual(int(row["plausible_candidate_count"]), 1)

    def test_one_frame_dropout_does_not_switch_to_persistent_same_size_circle(self) -> None:
        radius = 14
        frames = []
        true_centers: list[tuple[int, int] | None] = [
            (40, 110),
            (45, 110),
            (50, 110),
            (55, 110),
            (60, 110),
            None,
            (70, 110),
            (75, 110),
            (80, 110),
        ]
        for center in true_centers:
            frame = np.full((220, 180, 3), 170, dtype=np.uint8)
            # Same colour and same radius as the real sphere, visible from f0.
            self._draw_synthetic_sphere(frame, (93, 110), radius)
            if center is not None:
                self._draw_synthetic_sphere(frame, center, radius)
            frames.append(frame)

        rows, _ = track_standard_ball(frames, (40.0, 110.0), float(radius))

        dropout = rows[5]
        self.assertFalse(dropout["found"])
        self.assertFalse(dropout["identity_verified"])
        self.assertEqual(dropout["observation_status"], "missing")
        self.assertEqual(dropout["identity_rejection_reason"], "persistent_background_candidate")
        self.assertFalse(dropout["shape_evidence_available"])
        self.assertIsNone(dropout["shape_evidence_source"])
        for frame_index in (6, 7, 8):
            row = rows[frame_index]
            self.assertTrue(row["found"])
            self.assertTrue(row["identity_verified"])
            self.assertLess(
                abs(float(row["center_u_px"]) - float(true_centers[frame_index][0])),
                3.0,
            )
            self.assertLess(float(row["center_u_px"]), 90.0)

    def test_one_frame_dropout_does_not_confirm_newly_appearing_distractor(self) -> None:
        radius = 14
        frames = []
        true_centers: list[tuple[int, int] | None] = [
            (40, 110),
            (45, 110),
            (50, 110),
            (55, 110),
            (60, 110),
            None,
            (70, 110),
            (75, 110),
        ]
        for frame_index, center in enumerate(true_centers):
            frame = np.full((220, 180, 3), 170, dtype=np.uint8)
            # Unlike the previous regression, the distractor first appears on
            # the exact dropout frame, so no historical background memory can
            # identify it.  The immediate innovation gate must hold it pending.
            if frame_index >= 5:
                self._draw_synthetic_sphere(frame, (93, 110), radius)
            if center is not None:
                self._draw_synthetic_sphere(frame, center, radius)
            frames.append(frame)

        rows, _ = track_standard_ball(frames, (40.0, 110.0), float(radius))

        dropout = rows[5]
        self.assertFalse(dropout["found"])
        self.assertFalse(dropout["identity_verified"])
        self.assertEqual(
            dropout["identity_rejection_reason"],
            "innovation_requires_future_confirmation",
        )
        self.assertFalse(dropout["shape_evidence_available"])
        self.assertIsNone(dropout["shape_evidence_source"])
        for frame_index in (6, 7):
            row = rows[frame_index]
            self.assertTrue(row["found"])
            self.assertTrue(row["identity_verified"])
            self.assertLess(
                abs(float(row["center_u_px"]) - float(true_centers[frame_index][0])),
                3.0,
            )
            self.assertLess(float(row["center_u_px"]), 90.0)

    def test_clean_hough_circle_has_independent_edge_ring_shape_evidence(self) -> None:
        radius = 18
        frame = np.full((200, 200, 3), 170, dtype=np.uint8)
        self._draw_synthetic_sphere(frame, (100, 100), radius)
        candidates = _hough_candidates(
            frame,
            {"hue": 15.0, "saturation_min": 80.0, "value_min": 80.0},
            float(radius),
            np.asarray([100.0, 100.0]),
        )
        candidate = min(
            candidates,
            key=lambda item: np.linalg.norm(
                np.asarray([item.center_u_px, item.center_v_px]) - np.asarray([100.0, 100.0])
            ),
        )
        self.assertTrue(candidate.shape_evidence_available)
        self.assertEqual(candidate.shape_evidence_source, "hough_edge_ring")
        self.assertGreaterEqual(float(candidate.edge_support_fraction), 0.55)
        self.assertLessEqual(float(candidate.edge_radial_residual_ratio), 0.10)
        self.assertGreaterEqual(float(candidate.colour_support_fraction), 0.45)
        # These remain unavailable because no segmentation silhouette was used.
        self.assertIsNone(candidate.circularity)
        self.assertIsNone(candidate.ellipse_axis_ratio)

    def test_ellipse_and_partial_arc_fail_strict_edge_ring_shape_gate(self) -> None:
        center = (100, 100)
        proposed_radius = 30.0
        ellipse = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(ellipse, center, (42, 22), 0.0, 0.0, 360.0, 255, 1, cv2.LINE_AA)
        ellipse_support, ellipse_residual = _edge_ring_evidence(
            ellipse,
            center,
            proposed_radius,
        )
        self.assertTrue(
            ellipse_support < 0.55
            or ellipse_residual is None
            or ellipse_residual > 0.10
        )

        arc = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(arc, center, (30, 30), 0.0, 0.0, 100.0, 255, 1, cv2.LINE_AA)
        arc_support, _ = _edge_ring_evidence(arc, center, proposed_radius)
        self.assertLess(arc_support, 0.55)

    def test_frame0_change_prefers_moving_ball_over_nearer_static_orange_prop(self) -> None:
        radius = 14.0
        reference = np.full((180, 180, 3), 170, dtype=np.uint8)
        current = reference.copy()
        self._draw_synthetic_sphere(reference, (50, 50), int(radius))
        self._draw_synthetic_sphere(reference, (100, 100), int(radius))
        # The prop remains unchanged while the benchmark ball moves far enough
        # that a pure nearest-neighbour association would prefer the prop.
        self._draw_synthetic_sphere(current, (50, 50), int(radius))
        self._draw_synthetic_sphere(current, (100, 100), int(radius))
        cv2.circle(current, (50, 50), int(radius) + 2, (170, 170, 170), -1, cv2.LINE_AA)
        self._draw_synthetic_sphere(current, (50, 130), int(radius))

        def candidate(center: tuple[float, float]) -> SphereCandidate:
            return SphereCandidate(
                center_u_px=center[0],
                center_v_px=center[1],
                radius_px=radius,
                area_px=float(np.pi * radius**2),
                circularity=0.92,
                ellipse_major_axis_px=2.0 * radius,
                ellipse_minor_axis_px=2.0 * radius,
                ellipse_angle_deg=0.0,
                ellipse_axis_ratio=1.0,
                radial_residual_ratio=0.02,
                hue_distance=0.0,
                boundary=False,
                measurement_source="segmentation_contour",
            )

        annotated = _with_reference_change(
            [candidate((50.0, 130.0)), candidate((100.0, 100.0))],
            _reference_change_mask(reference, current),
        )
        moving, static = annotated
        self.assertGreater(float(moving.reference_change_fraction), 0.85)
        self.assertLess(float(static.reference_change_fraction), 0.10)
        selected, _ = _choose_candidate(
            annotated,
            np.asarray([80.0, 110.0]),
            radius,
            radius,
            gap=1,
        )
        self.assertIsNotNone(selected)
        self.assertLess(abs(float(selected.center_u_px) - 50.0), 1e-6)
        self.assertLess(abs(float(selected.center_v_px) - 130.0), 1e-6)


if __name__ == "__main__":
    unittest.main()
