from __future__ import annotations

import math
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = CODE_ROOT / "src" / "remake_benchmark" / "reconstruction" / "physics_parameters.py"
SPEC = importlib.util.spec_from_file_location("physics_parameters_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
fit_physics_parameters = MODULE.fit_physics_parameters
score_parameter_fit = MODULE.score_parameter_fit


def _rows(t: np.ndarray, x: np.ndarray, z: np.ndarray) -> list[dict[str, float]]:
    return [
        {"time_s": float(time), "x_m": float(x_value), "z_m": float(z_value)}
        for time, x_value, z_value in zip(t, x, z)
    ]


def _rk4_pendulum(
    t: np.ndarray,
    theta0: float,
    acceleration,
) -> np.ndarray:
    theta = float(theta0)
    omega = 0.0
    output = [theta]
    for index in range(1, len(t)):
        dt = float(t[index] - t[index - 1])

        def derivative(state):
            angle, speed = state
            return np.asarray([speed, acceleration(angle, speed)], dtype=float)

        state = np.asarray([theta, omega])
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt * k1)
        k3 = derivative(state + 0.5 * dt * k2)
        k4 = derivative(state + dt * k3)
        theta, omega = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        output.append(float(theta))
    return np.asarray(output)


class PhysicsParameterTests(unittest.TestCase):
    def test_v1_models(self) -> None:
        t = np.arange(0.0, 5.0 + 1e-9, 1.0 / 16.0)

        gravity = 9.81
        z = np.maximum(0.44, 4.2 - 0.5 * gravity * t**2)
        fit = fit_physics_parameters("v1_A", _rows(t, np.zeros_like(t), z))
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], gravity, delta=0.08)

        # Later bounce samples can make the total airborne count exceed eight,
        # but a four-point first fall is not an identifiable robust quadratic.
        short_t = np.arange(12, dtype=float) / 16.0
        short_z = np.asarray([1.2, 1.0, 0.75, 0.50, 0.44, 0.60, 0.80, 0.95, 0.44, 0.65, 0.85, 1.0])
        short_fit = fit_physics_parameters(
            "v1_A",
            _rows(short_t, np.zeros_like(short_t), short_z),
        )
        self.assertEqual(short_fit["status"], "insufficient_evidence")
        self.assertEqual(
            short_fit["reason"],
            "fewer_than_8_points_in_first_contiguous_airborne_run",
        )

        impact_time = 3.0
        incoming = 1.8
        restitution = 0.75
        x = np.where(t <= impact_time, -2.44 + incoming * t, 2.96 - restitution * incoming * (t - impact_time))
        fit = fit_physics_parameters("v1_B", _rows(t, x, np.full_like(t, 0.44)))
        self.assertAlmostEqual(fit["parameter_estimates"]["restitution_e"], restitution, delta=0.04)

        mu = 0.18
        stop = 3.6 / (mu * 9.81)
        effective = np.minimum(t, stop)
        x = -4.35 + 3.6 * effective - 0.5 * mu * 9.81 * effective**2
        fit = fit_physics_parameters("v1_C", _rows(t, x, np.full_like(t, 0.44)))
        self.assertAlmostEqual(fit["parameter_estimates"]["kinetic_friction_mu"], mu, delta=0.02)

        beta = 0.18
        theta = math.radians(24.0) * np.exp(-beta * t) * np.cos(2.0 * math.pi * t / 3.0)
        x = 2.3 * np.sin(theta)
        z = 3.8 - 2.3 * np.cos(theta)
        fit = fit_physics_parameters("v1_D", _rows(t, x, z))
        self.assertAlmostEqual(fit["parameter_estimates"]["amplitude_decay_beta"], beta, delta=0.01)

    def test_v2_models(self) -> None:
        t = np.arange(0.0, 10.0 + 1e-9, 1.0 / 32.0)

        gravity = 9.81
        omega = math.sqrt(gravity / (4.0 * 0.65))
        x = 1.7 * np.cos(omega * t) + 0.12 * np.cos(2.0 * omega * t)
        fit = fit_physics_parameters("v2_A", _rows(t, x, np.ones_like(t)))
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], gravity, delta=0.12)

        restitution = 0.72
        x_values = []
        x_value, velocity = 0.0, 3.0
        for _ in t:
            x_values.append(x_value)
            x_value += velocity / 32.0
            if x_value >= 3.0:
                x_value = 3.0 - (x_value - 3.0)
                velocity = -restitution * abs(velocity)
            elif x_value <= -3.0:
                x_value = -3.0 + (-3.0 - x_value)
                velocity = restitution * abs(velocity)
        fit = fit_physics_parameters("v2_B", _rows(t, np.asarray(x_values), np.full_like(t, 0.75)))
        self.assertAlmostEqual(fit["parameter_estimates"]["restitution_e"], restitution, delta=0.08)

        mu_a, mu_b = 0.08, 0.14
        dt = 1.0 / 32.0
        x_values, x_value, velocity = [], -3.9, 1.5
        for _ in t:
            x_values.append(x_value)
            mu = mu_a if x_value < 0.0 else mu_b
            next_velocity = max(0.0, velocity - mu * dt)
            x_value += 0.5 * (velocity + next_velocity) * dt
            velocity = next_velocity
        fit = fit_physics_parameters("v2_C", _rows(t, np.asarray(x_values), np.full_like(t, 1.005)))
        self.assertAlmostEqual(fit["parameter_estimates"]["kinetic_friction_mu_A"], mu_a, delta=0.02)
        self.assertAlmostEqual(fit["parameter_estimates"]["kinetic_friction_mu_B"], mu_b, delta=0.025)

        g, beta, length = 9.8, 0.12, 1.85
        theta = _rk4_pendulum(
            t,
            math.radians(42.0),
            lambda angle, speed: -(g / length) * math.sin(angle) - 2.0 * beta * speed,
        )
        x = -length * np.sin(theta)
        z = 3.28 - length * np.cos(theta)
        fit = fit_physics_parameters("v2_D", _rows(t, x, z))
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], g, delta=0.2)
        self.assertAlmostEqual(fit["parameter_estimates"]["linear_damping_beta"], beta, delta=0.025)

        bounce_g, bounce_e = 9.8, 0.68
        z_values, z_value, velocity = [], 3.0, -6.0
        for _ in t:
            z_values.append(z_value)
            velocity -= bounce_g * dt
            z_value += velocity * dt
            if z_value < 0.82:
                z_value = 0.82
                velocity = bounce_e * abs(velocity)
        fit = fit_physics_parameters("v2_E", _rows(t, np.zeros_like(t), np.asarray(z_values)))
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], bounce_g, delta=0.7)
        self.assertAlmostEqual(fit["parameter_estimates"]["restitution_e"], bounce_e, delta=0.12)

    def test_v3d_integral_fit(self) -> None:
        t = np.arange(0.0, 10.0 + 1e-9, 1.0 / 48.0)
        g, beta, kappa = 9.8, 0.045, 0.55
        length = 1.35
        center = math.radians(38.0)
        width = math.radians(18.0)

        def acceleration(angle, speed):
            u = (angle - center) / width
            magnetic = -u * math.exp(-0.5 * u * u)
            return -(g / length) * math.sin(angle) - 2.0 * beta * speed + kappa * magnetic

        theta = _rk4_pendulum(t, math.radians(-55.0), acceleration)
        x = -length * np.sin(theta)
        z = 4.05 - length * np.cos(theta)
        fit = fit_physics_parameters("v3_D", _rows(t, x, z))
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], g, delta=0.25)
        self.assertAlmostEqual(fit["parameter_estimates"]["linear_damping_beta"], beta, delta=0.02)
        self.assertAlmostEqual(fit["parameter_estimates"]["magnetic_kappa"], kappa, delta=0.10)

    def test_v3_a_b_c_models(self) -> None:
        dt = 1.0 / 64.0
        t = np.arange(0.0, 10.0 + 1e-9, dt)

        g, drag, restitution = 9.8, 0.14, 0.88
        xs, zs = [], []
        x_value, z_value, vx, vz = -3.2, 2.8, 1.25, -5.5
        for _ in t:
            xs.append(x_value)
            zs.append(z_value)
            vx += -drag * vx * dt
            vz += (-g - drag * vz) * dt
            x_value += vx * dt
            z_value += vz * dt
            if z_value < 0.82:
                z_value = 0.82
                vz = restitution * abs(vz)
        fit = fit_physics_parameters("v3_A", _rows(t, np.asarray(xs), np.asarray(zs)))
        self.assertAlmostEqual(fit["parameter_estimates"]["linear_drag_beta"], drag, delta=0.025)
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], g, delta=0.9)
        self.assertAlmostEqual(fit["parameter_estimates"]["restitution_e"], restitution, delta=0.14)

        mu, e_left, e_right = 0.008, 0.88, 0.80
        xs = []
        x_value, velocity = -0.85, -3.8
        for _ in t:
            xs.append(x_value)
            acceleration = -mu * 9.8 * math.copysign(1.0, velocity) if abs(velocity) > 1e-9 else 0.0
            velocity += acceleration * dt
            x_value += velocity * dt
            if x_value < -1.72:
                x_value = -1.72
                velocity = e_left * abs(velocity)
            elif x_value > 1.72:
                x_value = 1.72
                velocity = -e_right * abs(velocity)
        fit = fit_physics_parameters("v3_B", _rows(t, np.asarray(xs), np.full_like(t, 0.82)))
        self.assertAlmostEqual(fit["parameter_estimates"]["kinetic_friction_mu_k"], mu, delta=0.006)
        self.assertAlmostEqual(fit["parameter_estimates"]["left_restitution_e_L"], e_left, delta=0.12)
        self.assertAlmostEqual(fit["parameter_estimates"]["right_restitution_e_R"], e_right, delta=0.12)

        g, mu, restitution = 9.8, 0.035, 0.8
        alpha = math.radians(7.0)
        x0, z0 = -4.47018, 1.30748
        ramp_distance = (-0.5 - x0) / math.cos(alpha)
        ramp_acceleration = g * (math.sin(alpha) - mu * math.cos(alpha))
        t_ramp = math.sqrt(2.0 * ramp_distance / ramp_acceleration)
        v_floor0 = ramp_acceleration * t_ramp
        floor_acceleration = mu * g
        wall_distance = 4.61 - (-0.5)
        t_floor = (v_floor0 - math.sqrt(v_floor0**2 - 2.0 * floor_acceleration * wall_distance)) / floor_acceleration
        v_wall = v_floor0 - floor_acceleration * t_floor
        x_values, z_values = [], []
        for time in t:
            if time <= t_ramp:
                distance = 0.5 * ramp_acceleration * time**2
                x_value = x0 + distance * math.cos(alpha)
                z_value = z0 - distance * math.sin(alpha)
            elif time <= t_ramp + t_floor:
                tau = time - t_ramp
                x_value = -0.5 + v_floor0 * tau - 0.5 * floor_acceleration * tau**2
                z_value = 0.82
            else:
                tau = time - t_ramp - t_floor
                post_velocity = -restitution * v_wall
                stop_time = abs(post_velocity) / floor_acceleration
                effective = min(tau, stop_time)
                x_value = 4.61 + post_velocity * effective + 0.5 * floor_acceleration * effective**2
                z_value = 0.82
            x_values.append(x_value)
            z_values.append(z_value)
        fit = fit_physics_parameters("v3_C", _rows(t, np.asarray(x_values), np.asarray(z_values)))
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], g, delta=0.35)
        self.assertAlmostEqual(fit["parameter_estimates"]["kinetic_friction_mu"], mu, delta=0.006)
        self.assertAlmostEqual(fit["parameter_estimates"]["restitution_e"], restitution, delta=0.15)

    def test_scoring_uses_frozen_range_and_penalizes_missing(self) -> None:
        spec = {
            "hidden_parameters": [
                {"name": "a", "unit": "1", "valid_range": [0.0, 2.0]},
                {"name": "b", "unit": "1", "valid_range": [10.0, 20.0]},
            ]
        }
        fit = {
            "parameter_estimates": {"a": 1.5, "b": None},
            "parameter_observed": {"a": True, "b": False},
        }
        metrics = score_parameter_fit(fit, spec, {"a": 1.0, "b": 15.0})
        self.assertAlmostEqual(metrics["parameters"]["a"]["normalized_absolute_error"], 0.25)
        self.assertEqual(metrics["parameters"]["b"]["scoring_normalized_absolute_error"], 1.0)
        self.assertAlmostEqual(metrics["experiment_nmae"], 0.625)
        self.assertAlmostEqual(metrics["experiment_score_0_100"], 37.5)

    def test_reconstruction_rejected_frames_do_not_enter_fit(self) -> None:
        t = np.arange(0.0, 2.0 + 1e-9, 1.0 / 16.0)
        gravity = 9.81
        z = 4.2 - 0.5 * gravity * t**2
        rows = _rows(t, np.zeros_like(t), z)
        for row in rows[-8:]:
            row["z_m"] = 1000.0
            row["measurement_valid"] = False
            row["physics_fit_used"] = False
        fit = fit_physics_parameters("v1_A", rows)
        self.assertAlmostEqual(fit["parameter_estimates"]["gravity_g"], gravity, delta=0.08)


if __name__ == "__main__":
    unittest.main()
