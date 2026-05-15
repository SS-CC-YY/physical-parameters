"""
Experiment D: single ball damped pendulum with fixed gravity and length.

Hidden parameter:
    gamma_hidden : damping coefficient

Model:
    theta(t) = theta0 exp(-gamma_hidden t) cos(omega_d t)
    omega_d = sqrt(g_known / L - gamma_hidden^2)
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
from blender_scene_utils_v1 import FRAME_END, add_cylinder, add_plain_ball, animate_transforms, build_base_world, cleanup_empty_collections, cleanup_orphan_data, look_at, print_summary, t


OUTPUT_FILE = Path(__file__).with_name("baselinev1_D_ball_pendulum_damping.blend")

PHYSICS = {
    "pivot_x": 0.0,
    "pivot_y": 0.0,
    "pivot_z": 4.8,
    "rod_length": 2.06,
    "radius": 0.24,
    "theta0_deg": 24.0,
    "g_known": 9.81,
    "gamma_hidden": 0.18,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v1_D"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V1-D  Ball pendulum: damping")
    add_cylinder("D_PivotBar", (0.0, 0.0, 4.8), 0.08, 3.0, rotation=(0.0, math.radians(90), 0.0), collection=probes, material=mats["metal"])
    rod = add_cylinder("D_Rod", (0.0, 0.0, 3.77), 0.02, PHYSICS["rod_length"], rotation=(0.0, math.radians(180), 0.0), collection=probes, material=mats["metal"])
    ball = add_plain_ball("D_Ball", (0.0, 0.0, 2.5), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.0, -13.8, 3.3)
    cam_side.data.lens = 42
    cam_main.location = (5.8, -12.8, 4.4)
    cam_main.data.lens = 38
    look_at(cam_main, (0.0, 0.0, 3.15))
    scene.camera = cam_main
    return rod, ball


def sample_theta(time_sec):
    p = PHYSICS
    theta0 = math.radians(p["theta0_deg"])
    length_eff = p["rod_length"] + p["radius"]
    omega_sq = p["g_known"] / length_eff - p["gamma_hidden"] ** 2
    omega_d = math.sqrt(max(0.05, omega_sq))
    return theta0 * math.exp(-p["gamma_hidden"] * time_sec) * math.cos(omega_d * time_sec)


def animate_system(rod, ball):
    p = PHYSICS
    rod_tf = []
    ball_tf = []
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        theta = sample_theta(time_sec)
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
    "v1_D",
    "L1",
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
