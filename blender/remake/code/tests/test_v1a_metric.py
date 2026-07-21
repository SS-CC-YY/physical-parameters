from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

try:
    import cv2
    import numpy as np

    from remake_benchmark.reconstruction.v1a_metric import (
        DEFAULT_V1A_CONFIG,
        _geometry_interval_quality,
        _initial_projected_radius,
        fit_freefall_metric,
        parse_v1a_video_job,
        reconstruct_track,
        register_conditioning_to_frame0,
    )
    from remake_benchmark.reconstruction.v1a_geometry import compose_projection
    import remake_benchmark.reconstruction.v1a_tracker as v1a_tracker_module
    from remake_benchmark.reconstruction.v1a_tracker import (
        BallCandidate,
        choose_candidate,
        track_ball_sequence,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "V1A reconstruction dependencies are unavailable")
class V1AMetricTests(unittest.TestCase):
    def test_frozen_frame0_registration_recovers_small_video_warp(self) -> None:
        reference = np.full((140, 220, 3), 35, dtype=np.uint8)
        for y in range(12, 132, 18):
            for x in range(12, 212, 20):
                colour = (80 + (x % 90), 90 + (y % 100), 120 + ((x + y) % 100))
                cv2.rectangle(reference, (x - 3, y - 3), (x + 3, y + 3), colour, -1)
        delta = np.asarray([[1.0, 0.002, 3.0], [-0.001, 1.0, -2.0], [0.0, 0.0, 1.0]])
        frame0 = cv2.warpPerspective(reference, delta, (220, 140))
        curve = np.column_stack([np.full(80, 110.0), np.linspace(20.0, 120.0, 80)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditioning.png"
            self.assertTrue(cv2.imwrite(str(path), reference))
            recovered, quality = register_conditioning_to_frame0(
                path,
                frame0,
                np.eye(3),
                curve,
                5.0,
                deepcopy(DEFAULT_V1A_CONFIG["camera_pose"]),
            )
        self.assertTrue(quality["success"], quality)
        probes = np.asarray([[[25.0, 25.0]], [[190.0, 110.0]]], dtype=np.float64)
        expected = cv2.perspectiveTransform(probes, delta)
        actual = cv2.perspectiveTransform(probes, recovered)
        self.assertLess(float(np.max(np.linalg.norm(expected - actual, axis=2))), 0.8)

    def test_projective_diagnostic_rows_never_become_metric(self) -> None:
        k = np.asarray([[300.0, 0.0, 160.0], [0.0, 300.0, 90.0], [0.0, 0.0, 1.0]])
        rotation = np.eye(3)
        translation = np.asarray([0.0, 1.0, 8.0])
        projection = compose_projection(k, rotation, translation)
        tracks = []
        poses = []
        for frame in range(12):
            z = 4.2 - 0.12 * frame
            image = projection @ np.asarray([0.0, 0.0, z, 1.0])
            uv = image[:2] / image[2]
            tracks.append(
                {
                    "frame_index": frame,
                    "found": True,
                    "center_u_px": float(uv[0]),
                    "center_v_px": float(uv[1]),
                    "measurement_radius_px": 10.0,
                    "display_radius_px": 10.0,
                }
            )
            poses.append(
                {
                    "frame_index": frame,
                    "success": True,
                    "geometry_source": "projective_diagnostic",
                }
            )
        trajectory, summary = reconstruct_track(
            tracks,
            [projection] * len(tracks),
            poses,
            k,
            rotation,
            translation,
            0.0,
            0.0,
            4.2,
            0.44,
            0.24,
            24.0,
            deepcopy(DEFAULT_V1A_CONFIG),
        )
        self.assertEqual(summary["valid_points"], 0)
        self.assertEqual(summary["primary_component_points"], 0)
        self.assertTrue(all(not row["measurement_valid"] for row in trajectory))
        self.assertTrue(all(row["invalid_reason"] == "nonmetric_projective_geometry" for row in trajectory))

    def test_audited_static_path_hard_rejects_wrong_sphere_size(self) -> None:
        k = np.asarray([[360.0, 0.0, 160.0], [0.0, 360.0, 90.0], [0.0, 0.0, 1.0]])
        rotation = np.eye(3)
        translation = np.asarray([0.0, 0.0, 8.0])
        projection = compose_projection(k, rotation, translation)
        radius_m = 0.24
        tracks = []
        poses = []
        for frame in range(12):
            z = 4.2 - 0.12 * frame
            image = projection @ np.asarray([0.0, 0.0, z, 1.0])
            uv = image[:2] / image[2]
            expected = _initial_projected_radius(projection, (0.0, 0.0, z), radius_m)
            tracks.append(
                {
                    "frame_index": frame,
                    "found": True,
                    "center_u_px": float(uv[0]),
                    "center_v_px": float(uv[1]),
                    "measurement_radius_px": expected * (1.8 if frame == 5 else 1.0),
                    "display_radius_px": expected,
                }
            )
            poses.append(
                {
                    "frame_index": frame,
                    "success": True,
                    "quality": {"passed": True},
                    "geometry_source": "audited_static_calibration",
                    "R": rotation.tolist(),
                    "t": translation.tolist(),
                    "reprojection_p95_px": 0.0,
                }
            )
        trajectory, summary = reconstruct_track(
            tracks,
            [projection] * len(tracks),
            poses,
            k,
            rotation,
            translation,
            0.0,
            0.0,
            4.2,
            0.44,
            radius_m,
            24.0,
            deepcopy(DEFAULT_V1A_CONFIG),
            hard_sphere_constraint=True,
            min_observed_expected_radius_ratio=0.65,
            max_observed_expected_radius_ratio=1.35,
        )
        self.assertFalse(trajectory[5]["measurement_valid"])
        self.assertEqual(
            trajectory[5]["invalid_reason"],
            "known_sphere_size_constraint_failed",
        )
        self.assertEqual(summary["sphere_size_checked_frames"], 12)
        self.assertEqual(summary["sphere_size_valid_frames"], 11)
        self.assertTrue(trajectory[4]["measurement_valid"])

    def test_pnp_interval_quality_counts_projective_gap_frames(self) -> None:
        rows = [
            {"geometry_source": "anchor_pnp", "metric_geometry_valid": True},
            {"geometry_source": "projective_diagnostic", "metric_geometry_valid": False},
            {"geometry_source": "projective_diagnostic", "metric_geometry_valid": False},
            {"geometry_source": "anchor_pnp", "metric_geometry_valid": True},
        ]
        quality = _geometry_interval_quality(rows, 0, 4)
        self.assertEqual(quality["pnp_fraction"], 0.5)
        self.assertEqual(quality["metric_geometry_fraction"], 0.5)
        self.assertEqual(quality["max_non_pnp_run_frames"], 2)

    def test_projective_candidate_cannot_pollute_trusted_identity_state(self) -> None:
        contour = np.asarray([[[45, 45]], [[55, 45]], [[55, 55]], [[45, 55]]], dtype=np.int32)
        candidates = [
            [BallCandidate((50.0, 90.0), 10.0, 300.0, 0.9, 1.0, contour)],
            [BallCandidate((50.0, 70.0), 10.0, 300.0, 0.9, 1.0, contour)],
            [
                BallCandidate((50.0, 89.0), 10.0, 300.0, 0.9, 1.0, contour),
                BallCandidate((50.0, 50.0), 10.0, 300.0, 0.9, 1.0, contour),
            ],
        ]
        call_index = 0

        def fake_enumerate(*args, **kwargs):
            nonlocal call_index
            result = candidates[call_index]
            call_index += 1
            return result, np.zeros((120, 120), dtype=np.uint8)

        frames = [np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(3)]
        curve = np.column_stack([np.full(80, 50.0), np.linspace(10.0, 90.0, 80)])
        with patch.object(v1a_tracker_module, "enumerate_ball_candidates", side_effect=fake_enumerate):
            rows, summary = track_ball_sequence(
                frames,
                [curve, curve, curve],
                10.0,
                trusted_geometry=[True, False, True],
            )
        self.assertAlmostEqual(rows[2]["center_v_px"], 89.0)
        self.assertFalse(rows[1]["geometry_trusted_for_identity_update"])
        self.assertTrue(summary["diagnostic_geometry_cannot_update_trusted_identity"])

    def test_parses_both_generation_filename_conventions(self) -> None:
        short = parse_v1a_video_job(
            Path("v1_A__g9p81__indoor3__standard_ball__CAM_Main__seed-7.mp4")
        )
        explicit = parse_v1a_video_job(
            Path("v1_A__gravity_g-2__baseline__standard_ball__CAM_Side__seed-8.mp4")
        )
        self.assertAlmostEqual(short["targets"]["gravity_g"], 9.81)
        self.assertAlmostEqual(explicit["targets"]["gravity_g"], 2.0)
        self.assertEqual(short["factors"]["camera"], "CAM_Main")

    def test_metric_fit_is_consistent_at_16_and_24_fps(self) -> None:
        config = deepcopy(DEFAULT_V1A_CONFIG["physics"])
        estimates = []
        for fps in (16.0, 24.0):
            rows = []
            for frame in range(int(round(0.82 * fps)) + 1):
                time_s = frame / fps
                rows.append(
                    {
                        "time_s": time_s,
                        "z_m": 4.2 - 0.5 * 9.81 * time_s**2,
                        "measurement_valid": True,
                        "primary_metric_trajectory": True,
                        "physics_fit_used": False,
                    }
                )
            result = fit_freefall_metric(rows, 0.44, 2.0, config)
            self.assertTrue(result["fit_valid"])
            # The deliberately wrong target is compared only after fitting.
            self.assertLess(result["parameter_similarity"], 0.3)
            estimates.append(result["estimated_gravity_m_s2"])
        self.assertAlmostEqual(estimates[0], 9.81, places=6)
        self.assertAlmostEqual(estimates[1], 9.81, places=6)

    def test_hard_association_rejects_indoor_prop_and_merged_blob(self) -> None:
        contour = np.asarray([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.int32)
        true_ball = BallCandidate((50.5, 55.0), 10.0, 300.0, 0.86, 2.0, contour)
        orange_book = BallCandidate((92.0, 55.0), 9.5, 280.0, 0.82, 1.0, contour)
        merged_blob = BallCandidate((50.0, 55.0), 22.0, 900.0, 0.27, 1.0, contour)
        curve = np.column_stack([np.full(80, 50.0), np.linspace(10.0, 100.0, 80)])
        selected, _ = choose_candidate(
            [orange_book, merged_blob, true_ball],
            curve,
            10.0,
            predicted_center=(50.0, 54.0),
            previous_radius_px=10.0,
        )
        self.assertIs(selected, true_ball)

    def test_curve_endpoint_near_boundary_does_not_relax_interior_candidate_shape(self) -> None:
        contour = np.asarray([[[45, 45]], [[55, 45]], [[55, 55]], [[45, 55]]], dtype=np.int32)
        curve = np.column_stack([np.full(80, 50.0), np.linspace(100.0, -4.0, 80)])
        interior_blob = BallCandidate(
            (50.0, 50.0),
            10.0,
            280.0,
            0.31,
            1.0,
            contour,
        )
        selected, _ = choose_candidate(
            [interior_blob],
            curve,
            10.0,
            predicted_center=(50.0, 50.0),
            previous_radius_px=10.0,
            min_circularity=0.50,
        )
        self.assertIsNone(selected)

        clipped_ball = BallCandidate(
            (50.0, 2.0),
            10.0,
            280.0,
            0.31,
            1.0,
            contour,
            touches_frame_boundary=True,
        )
        selected, _ = choose_candidate(
            [clipped_ball],
            curve,
            10.0,
            predicted_center=(50.0, 2.0),
            previous_radius_px=10.0,
            min_circularity=0.50,
        )
        self.assertIs(selected, clipped_ball)

    def test_top_edge_clipping_does_not_unlock_display_box(self) -> None:
        frames = []
        curves = []
        for index in range(8):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            center = (80, -4 + 4 * index)
            cv2.circle(frame, center, 14, (0, 120, 245), -1, cv2.LINE_AA)
            frames.append(frame)
            curves.append(np.column_stack([np.full(80, 80.0), np.linspace(110.0, float(center[1]), 80)]))
        rows, summary = track_ball_sequence(frames, curves, 14.0, min_circularity=0.50)
        self.assertGreaterEqual(sum(row["found"] for row in rows), 6)
        self.assertTrue(all(row.get("touches_frame_boundary") for row in rows[:4] if row["found"]))
        self.assertFalse(rows[-1].get("touches_frame_boundary"))
        self.assertTrue(summary["display_box_locked"])
        self.assertTrue(all(row["display_radius_px"] == 14.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
