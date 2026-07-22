from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from remake_benchmark.reconstruction.contact_geometry import evaluate_contact_geometry
from remake_benchmark.reconstruction.motion_type_validity import evaluate_motion_type


CODE_ROOT = Path(__file__).resolve().parents[1]


def _profile() -> dict:
    return json.loads(
        (CODE_ROOT / "configs" / "evaluations" / "motion_type_profiles_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _extraction_rows(
    time: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    *,
    frames: np.ndarray | None = None,
) -> list[dict]:
    source = np.arange(len(time)) if frames is None else np.asarray(frames)
    return [
        {
            "frame_index": ordinal,
            "source_frame_index": int(frame),
            "time_s": float(t),
            # Deliberately different display coordinates make accidental
            # fallback visible to the tests.
            "x_m": float(x_value + 100.0),
            "y_m": 100.0,
            "z_m": float(z_value - 100.0),
            "fit_x_m": float(x_value),
            "fit_y_m": 0.0,
            "fit_z_m": float(z_value),
            "measurement_valid": True,
            "fit_eligible": True,
            "interpolated": False,
        }
        for ordinal, (frame, t, x_value, z_value) in enumerate(
            zip(source, time, x, z)
        )
    ]


def _evaluation_rows(time: np.ndarray, x: np.ndarray, z: np.ndarray) -> list[dict]:
    return [
        {
            "frame_index": index,
            "source_frame_index": index,
            "time_s": float(t),
            "x_m": float(x_value),
            "y_m": 0.0,
            "z_m": float(z_value),
            "measurement_valid": True,
            "physics_fit_used": True,
            "interpolated": False,
        }
        for index, (t, x_value, z_value) in enumerate(zip(time, x, z))
    ]


class AdversarialG0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _profile()

    def test_display_coordinates_never_override_trusted_fit_coordinates(self) -> None:
        time = np.arange(12, dtype=float) / 16.0
        rows = _extraction_rows(time, np.zeros(12), np.full(12, 0.44))
        result = evaluate_contact_geometry("v1_A", rows, self.profile)
        self.assertEqual(result["status"], "pass", result)

    def test_rejected_display_points_cannot_create_g0(self) -> None:
        time = np.arange(15, dtype=float) / 16.0
        rows = _extraction_rows(time, np.zeros(15), np.full(15, 0.44))
        for row in rows[12:]:
            row["fit_eligible"] = False
            row["z_m"] = -5.0
            row["fit_z_m"] = -5.0
        result = evaluate_contact_geometry("v1_A", rows, self.profile)
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["direct_metric_point_count"], 12)

    def test_evaluation_physics_fit_contract_is_supported(self) -> None:
        time = np.arange(12, dtype=float) / 16.0
        rows = _evaluation_rows(time, np.zeros(12), np.full(12, 0.44))
        result = evaluate_contact_geometry("v1_A", rows, self.profile)
        self.assertEqual(result["status"], "pass", result)

    def test_evaluation_rejection_overrides_stale_extraction_eligibility(self) -> None:
        time = np.arange(12, dtype=float) / 16.0
        rows = _extraction_rows(time, np.zeros(12), np.full(12, 0.10))
        for row in rows:
            row["physics_fit_used"] = False
        result = evaluate_contact_geometry("v1_A", rows, self.profile)
        self.assertEqual(result["status"], "indeterminate", result)
        self.assertEqual(result["direct_metric_point_count"], 0)

    def test_gapped_penetration_samples_are_not_a_persistent_run(self) -> None:
        time = np.arange(12, dtype=float) / 16.0
        source_frames = np.arange(12, dtype=int) * 10
        z = np.r_[np.full(3, 0.10), np.full(9, 0.44)]
        rows = _extraction_rows(time, np.zeros(12), z, frames=source_frames)
        result = evaluate_contact_geometry("v1_A", rows, self.profile)
        self.assertEqual(result["status"], "pass", result)
        check = result["checks"][0]
        self.assertEqual(check["longest_violation_run"], 1)


class AdversarialG1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = _profile()

    def test_full_coverage_without_expected_ground_contact_is_g1(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        z = 2.0 + 0.30 * np.sin(2.0 * math.pi * time / 1.5)
        result = evaluate_motion_type(
            "v2_E",
            _extraction_rows(time, np.zeros_like(time), z),
            self.profile,
        )
        self.assertEqual(result["status"], "fail", result)
        self.assertIn("expected_vertical_contact_not_observed", result["failure_codes"])

    def test_missing_contact_window_remains_u(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        z = 2.0 + 0.30 * np.sin(2.0 * math.pi * time / 1.5)
        rows = _extraction_rows(time, np.zeros_like(time), z)
        for row in rows[71:]:
            row["fit_eligible"] = False
        result = evaluate_motion_type("v2_E", rows, self.profile)
        self.assertEqual(result["status"], "indeterminate", result)

    def test_near_static_pendulum_jitter_is_g1(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        theta = 0.008 * np.sin(2.0 * math.pi * time)
        length = 2.3
        x = length * np.sin(theta)
        z = 3.8 - length * np.cos(theta)
        result = evaluate_motion_type(
            "v1_D", _extraction_rows(time, x, z), self.profile
        )
        self.assertEqual(result["status"], "fail", result)
        self.assertIn("pendulum_motion_is_nearly_static", result["failure_codes"])

    def test_real_pendulum_amplitude_still_passes(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        theta = 0.65 * np.exp(-0.08 * time) * np.cos(2.0 * math.pi * time / 2.5)
        length = 2.3
        x = length * np.sin(theta)
        z = 3.8 - length * np.cos(theta)
        result = evaluate_motion_type(
            "v1_D", _extraction_rows(time, x, z), self.profile
        )
        self.assertEqual(result["status"], "pass", result)

    def test_flat_horizontal_path_cannot_masquerade_as_v3c_ramp(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        x = np.where(
            time <= 2.5,
            -2.0 + (6.61 / 2.5) * time,
            4.61 - 1.2 * (time - 2.5),
        )
        z = np.full_like(time, 0.82)
        result = evaluate_motion_type(
            "v3_C", _extraction_rows(time, x, z), self.profile
        )
        self.assertEqual(result["status"], "fail", result)
        self.assertIn("ramp_descent_not_observed", result["failure_codes"])

    def test_canonical_v3c_ramp_floor_wall_topology_passes(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        x = np.where(
            time <= 2.5,
            -2.0 + (6.61 / 2.5) * time,
            4.61 - 1.2 * (time - 2.5),
        )
        z = np.where(x < -0.5, 0.8 - 0.08 * (x + 0.5), 0.8)
        result = evaluate_motion_type(
            "v3_C", _extraction_rows(time, x, z), self.profile
        )
        self.assertEqual(result["status"], "pass", result)

    def test_v2b_without_frozen_wall_geometry_is_u(self) -> None:
        time = np.linspace(0.0, 5.0, 121)
        triangle = (2.0 / math.pi) * np.arcsin(
            np.sin(2.0 * math.pi * time / 2.0)
        )
        result = evaluate_motion_type(
            "v2_B",
            _extraction_rows(time, 3.0 * triangle, np.zeros_like(time)),
            self.profile,
        )
        self.assertEqual(result["status"], "indeterminate", result)
        self.assertIn(
            "wall_geometry_unavailable_for_bounce_validation",
            result["indeterminate_codes"],
        )

    def test_measurement_valid_alone_is_not_metric_g1_evidence(self) -> None:
        time = np.linspace(0.0, 3.0, 73)
        rows = _extraction_rows(time, time, np.zeros_like(time))
        for row in rows:
            row["fit_eligible"] = False
        result = evaluate_motion_type("v1_C", rows, self.profile)
        self.assertEqual(result["status"], "indeterminate", result)
        self.assertEqual(result["direct_metric_point_count"], 0)


if __name__ == "__main__":
    unittest.main()
