"""
Experiment D: large-angle nonlinear damped pendulum.

Hidden parameter:
    gamma_hidden : damping coefficient

Known parameters:
    g_known, rod length, initial release angle

This is the first V2-D branch: it differs from V1-D by requiring the full
sin(theta) nonlinear pendulum model at large amplitude, not a small-angle
closed-form approximation.
"""

from pathlib import Path
import math
import sys

import bpy
from mathutils import Euler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import export_ground_truth, parse_variant_args
from benchmark_variants import PARAM_VARIANTS
from blender_scene_utils_v2 import FRAME_END, add_cylinder, add_plain_ball, animate_transforms, build_base_world, cleanup_empty_collections, cleanup_orphan_data, look_at, print_summary, t


OUTPUT_FILE = Path(__file__).with_name("baselinev2_D_ball_long_pendulum_hidden_damping.blend")

PHYSICS = {
    "pivot_x": 0.0,
    "pivot_y": 0.0,
    "pivot_z": 5.8,
    "rod_length": 3.0,
    "radius": 0.24,
    "theta0_deg": 68.0,
    "g_known": 9.81,
    "gamma_hidden": 0.08,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v2_D"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V2-D  Large-angle nonlinear pendulum: hidden damping")
    add_cylinder("D_PivotBar", (PHYSICS["pivot_x"], 0.0, PHYSICS["pivot_z"]), 0.08, 3.4, rotation=(0.0, math.radians(90), 0.0), collection=probes, material=mats["metal"])
    rod = add_cylinder("D_Rod", (0.0, 0.0, PHYSICS["pivot_z"] - 0.5 * PHYSICS["rod_length"]), 0.02, PHYSICS["rod_length"], rotation=(0.0, math.radians(180), 0.0), collection=probes, material=mats["metal"])
    ball = add_plain_ball("D_Ball", (0.0, 0.0, PHYSICS["pivot_z"] - PHYSICS["rod_length"] - PHYSICS["radius"]), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.0, -16.2, 3.6)
    cam_side.data.lens = 34
    cam_main.location = (7.2, -15.8, 5.1)
    cam_main.data.lens = 32
    look_at(cam_main, (0.0, 0.0, 3.1))
    scene.camera = cam_main
    return rod, ball


def integrate_theta():
    p = PHYSICS
    theta0 = math.radians(p["theta0_deg"])
    length_eff = p["rod_length"] + p["radius"]
    dt = 1.0 / (24.0 * 12.0)
    next_sample = 1
    theta = theta0
    omega = 0.0
    time_sec = 0.0
    samples = {}

    def accel(th, om):
        return -(p["g_known"] / length_eff) * math.sin(th) - 2.0 * p["gamma_hidden"] * om

    while next_sample <= FRAME_END:
        target_t = t(next_sample)
        while time_sec + 1e-10 < target_t:
            h = min(dt, target_t - time_sec)
            k1_th = omega
            k1_om = accel(theta, omega)
            k2_th = omega + 0.5 * h * k1_om
            k2_om = accel(theta + 0.5 * h * k1_th, omega + 0.5 * h * k1_om)
            k3_th = omega + 0.5 * h * k2_om
            k3_om = accel(theta + 0.5 * h * k2_th, omega + 0.5 * h * k2_om)
            k4_th = omega + h * k3_om
            k4_om = accel(theta + h * k3_th, omega + h * k3_om)
            theta += (h / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)
            omega += (h / 6.0) * (k1_om + 2.0 * k2_om + 2.0 * k3_om + k4_om)
            time_sec += h
        samples[next_sample] = theta
        next_sample += 1
    return samples


def animate_system(rod, ball):
    p = PHYSICS
    theta_by_frame = integrate_theta()
    rod_tf = []
    ball_tf = []
    for frame in range(1, FRAME_END + 1):
        theta = theta_by_frame[frame]
        dir_x = math.sin(theta)
        dir_z = -math.cos(theta)
        attach_x = p["pivot_x"] + p["rod_length"] * dir_x
        attach_z = p["pivot_z"] + p["rod_length"] * dir_z
        ball_x = attach_x + p["radius"] * dir_x
        ball_z = attach_z + p["radius"] * dir_z
        rod_x = 0.5 * (p["pivot_x"] + attach_x)
        rod_z = 0.5 * (p["pivot_z"] + attach_z)
        rod_tf.append((frame, (rod_x, p["pivot_y"], rod_z), Euler((0.0, math.pi - theta, 0.0), "XYZ")))
        ball_tf.append((frame, (ball_x, p["pivot_y"], ball_z), Euler((0.0, 0.0, 0.0), "XYZ")))
    animate_transforms(rod, rod_tf)
    animate_transforms(ball, ball_tf)
    return rod_tf, ball_tf


rod, ball = build()
rod_tf, ball_tf = animate_system(rod, ball)
export_ground_truth(
    OUTPUT_FILE,
    "v2_D",
    "L2",
    PHYSICS,
    {"D_Rod": rod_tf, "D_Ball": ball_tf},
    hidden_params=["gamma_hidden"],
    known_params=["g_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["D_Rod", "D_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
