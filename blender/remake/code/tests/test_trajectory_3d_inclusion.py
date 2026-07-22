from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.trajectory_3d_inclusion import (  # noqa: E402
    EXPERIMENT_MOTION_MANIFOLDS,
    assess_dynamic_3d_trajectory,
    finalize_dynamic_3d_inclusion,
)


def _rows(path, count: int = 25) -> list[dict]:
    output = []
    for index in range(count):
        t = index / 24.0
        x, y, z = path(index, t)
        output.append(
            {
                "frame_index": index,
                "time_s": t,
                "fit_x_m": x,
                "fit_y_m": y,
                "fit_z_m": z,
                "fit_eligible": True,
                "interpolated": False,
                "coordinate_frame_3d": "spatialtrackerv2_frame0_metric_aligned_m",
            }
        )
    return output


def _native(quality: bool = True) -> dict:
    return {
        "pipeline": {
            "dynamic_reconstruction": {
                "status": "succeeded",
                "quality_pass": quality,
            }
        }
    }


class Trajectory3DInclusionTests(unittest.TestCase):
    def test_expected_x_axis_with_small_depth_and_height_noise_is_included(self) -> None:
        rows = _rows(
            lambda index, _t: (
                index / 24.0,
                0.012 * math.sin(index / 5.0),
                0.008 * math.cos(index / 6.0),
            )
        )
        result = assess_dynamic_3d_trajectory("v1_C", rows, reconstruction_metadata=_native())

        self.assertEqual(result["decision"], "include")
        self.assertEqual(result["measurement_route"], "qualified_dynamic_3d")
        self.assertLess(result["metrics"]["off_manifold_range_ratio"], 0.1)
        self.assertGreater(result["metrics"]["expected_variance_fraction"], 0.95)
        self.assertFalse(result["target_parameters_used"])

    def test_large_out_of_plane_or_vertical_motion_is_measurement_x(self) -> None:
        rows = _rows(
            lambda index, _t: (
                index / 24.0,
                0.75 * index / 24.0,
                0.60 * index / 24.0,
            )
        )
        result = assess_dynamic_3d_trajectory("v1_C", rows, reconstruction_metadata=_native())

        self.assertEqual(result["decision"], "X")
        self.assertIn("X_EXCESS_OFF_MANIFOLD_DRIFT", result["reason_codes"])

    def test_xz_plane_motion_with_small_y_drift_is_included(self) -> None:
        rows = _rows(
            lambda index, t: (
                1.2 * t,
                0.01 * math.sin(index / 4.0),
                1.5 - 0.5 * t * t,
            )
        )
        result = assess_dynamic_3d_trajectory("v3_A", rows, reconstruction_metadata=_native())

        self.assertEqual(result["decision"], "include")
        self.assertEqual(result["expected_geometry"]["axes"], ("x", "z"))

    def test_native_quality_and_world_axis_alignment_are_required(self) -> None:
        rows = _rows(lambda index, _t: (index / 24.0, 0.0, 0.0))
        for row in rows:
            row["coordinate_frame_3d"] = "arbitrary_camera_coordinates"
        result = assess_dynamic_3d_trajectory("v1_C", rows, reconstruction_metadata=_native(False))

        self.assertEqual(result["decision"], "X")
        self.assertIn("X_DYNAMIC_RECONSTRUCTION_QUALITY_NOT_PASSED", result["reason_codes"])
        self.assertIn("X_WORLD_AXIS_ALIGNMENT_UNPROVEN", result["reason_codes"])

    def test_metric_camera_coordinates_without_world_axis_proof_fail_closed(self) -> None:
        rows = _rows(lambda index, _t: (index / 24.0, 0.0, 0.0))
        for row in rows:
            row["coordinate_frame_3d"] = "camera_metric_m"

        result = assess_dynamic_3d_trajectory("v1_C", rows, reconstruction_metadata=_native())

        self.assertEqual(result["decision"], "X")
        self.assertIn("X_WORLD_AXIS_ALIGNMENT_UNPROVEN", result["reason_codes"])

        proven = _native()
        proven["pipeline"]["dynamic_reconstruction"]["world_axis_aligned"] = True
        proven_result = assess_dynamic_3d_trajectory(
            "v1_C",
            rows,
            reconstruction_metadata=proven,
        )
        self.assertEqual(proven_result["decision"], "include")

    def test_alignment_and_fit_eligibility_must_cover_every_used_point(self) -> None:
        mixed = _rows(lambda index, _t: (index / 24.0, 0.0, 0.0))
        for row in mixed[1:]:
            row["coordinate_frame_3d"] = "camera_metric_m"
        mixed_result = assess_dynamic_3d_trajectory(
            "v1_C",
            mixed,
            reconstruction_metadata=_native(),
        )
        self.assertEqual(mixed_result["decision"], "X")
        self.assertIn("X_WORLD_AXIS_ALIGNMENT_UNPROVEN", mixed_result["reason_codes"])

        unqualified = _rows(lambda index, _t: (index / 24.0, 0.0, 0.0))
        for row in unqualified:
            row.pop("fit_eligible")
        unqualified_result = assess_dynamic_3d_trajectory(
            "v1_C",
            unqualified,
            reconstruction_metadata=_native(),
        )
        self.assertEqual(unqualified_result["decision"], "X")
        self.assertIn("X_TOO_FEW_DIRECT_POINTS", unqualified_result["reason_codes"])

    def test_target_free_motion_law_fit_must_be_complete_and_stable(self) -> None:
        geometry = assess_dynamic_3d_trajectory(
            "v1_C",
            _rows(lambda index, _t: (index / 24.0, 0.0, 0.0)),
            reconstruction_metadata=_native(),
        )
        good = finalize_dynamic_3d_inclusion(
            geometry,
            {
                "status": "ok",
                "parameter_observed": {"kinetic_friction_mu": True},
                "target_not_used_for_fit": True,
                "diagnostics": {"fit_nrmse": 0.08},
            },
        )
        bad = finalize_dynamic_3d_inclusion(
            geometry,
            {
                "status": "ok",
                "parameter_observed": {"kinetic_friction_mu": True},
                "target_not_used_for_fit": True,
                "diagnostics": {"fit_nrmse": 0.8},
            },
        )
        missing = finalize_dynamic_3d_inclusion(
            geometry,
            {
                "status": "ok",
                "parameter_observed": {"kinetic_friction_mu": True},
                "target_not_used_for_fit": True,
                "diagnostics": {},
            },
        )

        self.assertEqual(good["decision"], "include")
        self.assertEqual(good["fit_state"], "completed")
        self.assertEqual(bad["decision"], "X")
        self.assertIn("X_MOTION_LAW_FIT_RESIDUAL_HIGH", bad["reason_codes"])
        self.assertEqual(missing["decision"], "X")
        self.assertIn("X_MOTION_LAW_RESIDUAL_EVIDENCE_MISSING", missing["reason_codes"])

    def test_all_frozen_experiments_have_a_manifold_and_unknown_fails_closed(self) -> None:
        self.assertEqual(
            set(EXPERIMENT_MOTION_MANIFOLDS),
            {
                "v1_A", "v1_B", "v1_C", "v1_D",
                "v2_A", "v2_B", "v2_C", "v2_D", "v2_E",
                "v3_A", "v3_B", "v3_C", "v3_D",
            },
        )
        self.assertEqual(
            {key: tuple(value["axes"]) for key, value in EXPERIMENT_MOTION_MANIFOLDS.items()},
            {
                "v1_A": ("z",), "v1_B": ("x",), "v1_C": ("x",), "v1_D": ("x", "z"),
                "v2_A": ("x", "z"), "v2_B": ("x",), "v2_C": ("x",),
                "v2_D": ("x", "z"), "v2_E": ("z",),
                "v3_A": ("x", "z"), "v3_B": ("x",),
                "v3_C": ("x", "z"), "v3_D": ("x", "z"),
            },
        )
        vertical = assess_dynamic_3d_trajectory(
            "v1_A",
            _rows(lambda index, _t: (0.004 * math.sin(index), 0.0, 2.0 - index / 24.0)),
            reconstruction_metadata=_native(),
        )
        unknown = assess_dynamic_3d_trajectory(
            "v9_Z",
            _rows(lambda index, _t: (index / 24.0, 0.0, 0.0)),
            reconstruction_metadata=_native(),
        )

        self.assertEqual(vertical["decision"], "include")
        self.assertEqual(vertical["expected_geometry"]["axes"], ("z",))
        self.assertEqual(unknown["decision"], "X")
        self.assertIn("X_EXPECTED_MOTION_MANIFOLD_UNDEFINED", unknown["reason_codes"])


if __name__ == "__main__":
    unittest.main()
