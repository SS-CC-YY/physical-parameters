from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.physics_parameters import fit_physics_parameters  # noqa: E402


def _rows(time: np.ndarray, x: np.ndarray, z: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {
            "time_s": float(t),
            "x_m": float(x_value),
            "z_m": float(z_value),
            "source_frame_index": int(frame + 100),
        }
        for frame, (t, x_value, z_value) in enumerate(zip(time, x, z))
    ]


class SegmentedPhysicsParameterTests(unittest.TestCase):
    def test_v1a_uses_release_to_first_contact_only(self) -> None:
        time = np.arange(0.0, 3.0 + 1e-9, 1.0 / 24.0)
        release = 0.5
        tau = np.maximum(time - release, 0.0)
        free_fall = 3.8 - 0.5 * 9.81 * tau**2
        contact = int(np.flatnonzero(free_fall <= 0.44)[0])
        z = free_fall.copy()
        z[contact:] = 0.44 + 0.35 * np.abs(np.sin(7.0 * (time[contact:] - time[contact])))
        fit = fit_physics_parameters("v1_A", _rows(time, np.zeros_like(time), z))

        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], 9.81, delta=0.15)
        segment = fit["segmentation"]["segments"][0]
        self.assertGreater(segment["start_index"], 0)
        self.assertEqual(segment["end_index"], fit["segmentation"]["events"][1]["index"])
        self.assertEqual(segment["start_source_frame"], segment["start_index"] + 100)

    def test_v1c_static_prefix_and_tail_do_not_force_zero_initial_speed(self) -> None:
        time = np.arange(0.0, 5.0 + 1e-9, 1.0 / 24.0)
        release = 0.4
        mu = 0.18
        initial_velocity = 3.6
        move_time = initial_velocity / (mu * 9.81)
        tau = np.clip(time - release, 0.0, move_time)
        x = -4.35 + initial_velocity * tau - 0.5 * mu * 9.81 * tau**2
        fit = fit_physics_parameters("v1_C", _rows(time, x, np.full_like(time, 0.44)))

        self.assertAlmostEqual(fit["parameter_estimates"]["kinetic_friction_mu"], mu, delta=0.025)
        self.assertGreater(fit["diagnostics"]["observed_initial_velocity_m_s"], 3.0)
        self.assertEqual(
            fit["diagnostics"]["initial_velocity_assumption"],
            "equal_to_first_moving_frame_interval",
        )

    def test_v1b_nonuniform_speed_is_rule_family_failure(self) -> None:
        time = np.arange(0.0, 5.0 + 1e-9, 1.0 / 24.0)
        impact_time = 2.5
        # Both branches reverse at the correct wall, but their curved paths
        # contain persistent acceleration that a restitution-only law cannot
        # explain.
        pre_tau = np.minimum(time, impact_time)
        pre = -2.04 + 1.2 * pre_tau + 0.32 * pre_tau**2
        wall = float(pre[np.argmin(np.abs(time - impact_time))])
        post_tau = np.maximum(time - impact_time, 0.0)
        post = wall - 1.1 * post_tau - 0.25 * post_tau**2
        x = np.where(time <= impact_time, pre, post)
        # Align the observed impact with the frozen wall exactly.
        x += 2.96 - wall
        fit = fit_physics_parameters("v1_B", _rows(time, x, np.full_like(time, 0.44)))

        self.assertEqual(fit["status"], "model_mismatch")
        self.assertIsNone(fit["parameter_estimates"]["restitution_e"])
        reasons = " ".join(fit["rule_family_evaluation"]["reason_codes"])
        self.assertIn("nonuniform", reasons)

    def test_v2c_velocity_reset_is_not_force_fitted(self) -> None:
        time = np.arange(0.0, 6.0 + 1e-9, 1.0 / 32.0)
        dt = 1.0 / 32.0
        x_values: list[float] = []
        x, velocity = -2.0, 0.9
        reset = False
        for _ in time:
            x_values.append(x)
            acceleration = -0.08 if x < 0.0 else -0.14
            velocity += acceleration * dt
            x += velocity * dt
            if not reset and x >= 0.0:
                velocity = 1.6
                reset = True
        fit = fit_physics_parameters(
            "v2_C", _rows(time, np.asarray(x_values), np.full_like(time, 1.005))
        )

        self.assertEqual(fit["status"], "model_mismatch")
        self.assertIsNone(fit["parameter_estimates"]["kinetic_friction_mu_A"])
        self.assertIn(
            "velocity_continuity:velocity_reset_at_surface_transition",
            fit["rule_family_evaluation"]["reason_codes"],
        )

    def test_v2_rule_family_adversaries_are_not_published(self) -> None:
        time = np.arange(0.0, 10.0 + 1e-9, 1.0 / 32.0)

        chirp_phase = 1.9 * time + 0.04 * time**2
        fit = fit_physics_parameters(
            "v2_A",
            _rows(time, 1.4 * np.cos(chirp_phase), np.ones_like(time)),
        )
        self.assertEqual(fit["status"], "model_mismatch")
        self.assertIsNone(fit["parameter_estimates"]["gravity_g"])

        # A sinusoid reaches both wall coordinates smoothly, but it contains
        # no impulsive wall collisions and is non-uniform between extrema.
        fit = fit_physics_parameters(
            "v2_B",
            _rows(time, 2.75 * np.sin(2.0 * math.pi * time / 3.0), np.ones_like(time)),
        )
        self.assertEqual(fit["status"], "model_mismatch")
        self.assertIsNone(fit["parameter_estimates"]["restitution_e"])

        length = 1.85
        theta = 0.7 * np.exp(-0.05 * time) * np.cos(1.9 * time + 0.035 * time**2)
        fit = fit_physics_parameters(
            "v2_D",
            _rows(time, -length * np.sin(theta), 3.28 - length * np.cos(theta)),
        )
        self.assertEqual(fit["status"], "model_mismatch")
        self.assertIsNone(fit["parameter_estimates"]["gravity_g"])

        # Smooth half-cosine minima are not instantaneous restitution events
        # and their within-flight acceleration is not constant gravity.
        z = 0.82 + 1.3 * (0.5 + 0.5 * np.cos(2.0 * math.pi * time / 1.8))
        fit = fit_physics_parameters(
            "v2_E", _rows(time, np.zeros_like(time), z)
        )
        self.assertEqual(fit["status"], "model_mismatch")
        self.assertIsNone(fit["parameter_estimates"]["gravity_g"])
        self.assertIsNone(fit["parameter_estimates"]["restitution_e"])

    def test_v3d_unphysical_coupled_solution_is_not_published(self) -> None:
        time = np.arange(0.0, 8.0 + 1e-9, 1.0 / 48.0)
        # Growing-amplitude oscillation violates non-negative damping.  The
        # raw regression may still return numbers, but accepted estimates must
        # not publish a negative damping coefficient.
        theta = math.radians(18.0) * np.exp(0.18 * time) * np.cos(2.3 * time)
        x = -1.35 * np.sin(theta)
        z = 4.05 - 1.35 * np.cos(theta)
        fit = fit_physics_parameters("v3_D", _rows(time, x, z))

        self.assertIsNone(fit["parameter_estimates"]["linear_damping_beta"])
        self.assertNotEqual(
            fit["parameter_attribution"]["linear_damping_beta"]["status"], "pass"
        )


if __name__ == "__main__":
    unittest.main()
