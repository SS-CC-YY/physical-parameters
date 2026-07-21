from __future__ import annotations

import unittest

from remake_benchmark.reconstruction.generation_validity import evaluate_generation_validity


def _tracks(count: int = 12, **updates):
    rows = []
    for frame in range(count):
        row = {
            "frame_index": frame,
            "found": True,
            "track_confidence": 0.95,
            "touches_frame_boundary": False,
            "candidate_count": 1,
            "measurement_radius_px": 14.0,
            "expected_radius_px": 14.0,
            "circularity": 0.92,
            "ellipse_axis_ratio": 1.04,
        }
        row.update(updates)
        rows.append(row)
    return rows


def _rigid_frames(count: int = 8, *, bad: bool = False, background: bool = False):
    rows = []
    for frame in range(count):
        row = {
            "frame_index": frame,
            "normalized_rmse": 0.42 if bad else 0.04,
            "inlier_fraction": 0.30 if bad else 0.92,
        }
        if background:
            row["spatial_coverage_fraction"] = 0.72
        rows.append(row)
    return rows


class GenerationValidityTest(unittest.TestCase):
    def test_persistent_ellipse_is_a_hard_failure(self) -> None:
        rows = _tracks()
        for frame in (4, 5, 6, 7):
            rows[frame]["ellipse_axis_ratio"] = 1.95
            rows[frame]["circularity"] = 0.48
        result = evaluate_generation_validity(
            rows,
            camera_motion_evidence={"final_category": "no_significant_camera_change"},
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["fit_eligible"])
        self.assertIn("persistent_experiment_object_deformation_2d", result["failure_codes"])
        self.assertEqual(result["offending_frames"]["object_shape_2d"], [4, 5, 6, 7])

    def test_one_frame_blur_is_warning_but_still_passes(self) -> None:
        rows = _tracks()
        rows[5]["ellipse_axis_ratio"] = 2.0
        rows[5]["circularity"] = 0.20
        result = evaluate_generation_validity(
            rows,
            camera_motion_evidence={"final_category": "no_significant_camera_change"},
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["fit_eligible"])
        self.assertIn("transient_object_shape_outlier", result["warning_codes"])

    def test_scene_cut_is_a_hard_failure(self) -> None:
        result = evaluate_generation_validity(
            _tracks(),
            camera_motion_evidence={
                "final_category": "no_significant_camera_change",
                "cut_pair_count": 1,
                "cut_pairs": [[6, 7]],
            },
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn("scene_cut", result["failure_codes"])
        self.assertEqual(result["offending_frames"]["scene_cut"], [7])

    def test_low_tracking_is_indeterminate_not_generation_failure(self) -> None:
        rows = _tracks()
        for row in rows[3:]:
            row["found"] = False
            row["track_confidence"] = 0.0
        result = evaluate_generation_validity(
            rows,
            camera_motion_evidence={"final_category": "no_significant_camera_change"},
        )
        self.assertEqual(result["status"], "indeterminate")
        self.assertFalse(result["fit_eligible"])
        self.assertEqual(result["failure_codes"], [])
        self.assertIn("insufficient_reliable_object_tracking", result["warning_codes"])

    def test_changed_camera_without_3d_evidence_is_indeterminate(self) -> None:
        result = evaluate_generation_validity(
            _tracks(),
            camera_motion_evidence={"final_category": "camera_changed"},
        )
        self.assertEqual(result["status"], "indeterminate")
        self.assertEqual(result["failure_codes"], [])
        self.assertIn("moving_camera_scene_rigidity_unresolved", result["warning_codes"])

    def test_changed_camera_with_rigid_3d_evidence_passes(self) -> None:
        result = evaluate_generation_validity(
            _tracks(),
            camera_motion_evidence={"final_category": "camera_changed"},
            dynamic_rigidity_evidence={
                "object_frames": _rigid_frames(),
                "background_frames": _rigid_frames(background=True),
            },
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["fit_eligible"])
        self.assertEqual(result["checks"]["object_rigidity_3d"]["status"], "pass")
        self.assertEqual(result["checks"]["scene_rigidity_3d"]["status"], "pass")

    def test_persistent_dynamic_object_nonrigidity_fails(self) -> None:
        result = evaluate_generation_validity(
            _tracks(),
            camera_motion_evidence={"final_category": "camera_changed"},
            dynamic_rigidity_evidence={
                "object_frames": _rigid_frames(bad=True),
                "background_frames": _rigid_frames(background=True),
            },
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["fit_eligible"])
        self.assertIn("persistent_experiment_object_deformation_3d", result["failure_codes"])

    def test_persistent_dynamic_background_nonrigidity_fails(self) -> None:
        result = evaluate_generation_validity(
            _tracks(),
            camera_motion_evidence={"final_category": "camera_changed"},
            trajectory_evidence={
                "object_rigidity": {"frames": _rigid_frames()},
                "scene_rigidity": {"frames": _rigid_frames(bad=True, background=True)},
            },
        )
        self.assertEqual(result["status"], "fail")
        self.assertIn(
            "persistent_non_experimental_scene_deformation_3d",
            result["failure_codes"],
        )
        self.assertEqual(result["checks"]["scene_rigidity_3d"]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
