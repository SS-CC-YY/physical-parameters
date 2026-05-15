"""
Experiment D: pendulum with localized damping zone near the bottom of the swing.

Hidden parameters:
    g_hidden           : gravity
    gamma_zone_hidden  : damping inside the local damping zone

The pivot is fixed in space. The bob swings in the x-z plane on a rigid rod.
Damping is not uniform over the whole trajectory: it becomes active only when
the pendulum passes through a narrow angular zone around the bottom.
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
from blender_scene_utils_v3 import FRAME_END, add_cylinder, add_plain_ball, add_cube, animate_transforms, build_base_world, cleanup_empty_collections, cleanup_orphan_data, look_at, print_summary, t


OUTPUT_FILE = Path(__file__).with_name("baselinev3_D_ball_em_damping_pendulum.blend")

PHYSICS = {
    "pivot_x": 0.0,
    "pivot_y": 0.0,
    "pivot_z": 4.8,
    "rod_length": 2.06,
    "theta0_deg": 24.0,
    "radius": 0.24,
    "g_hidden": 9.81,
    "gamma_zone_hidden": 0.58,
    "theta_zone_deg": 7.5,
    "integration_substeps": 10,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v3_D"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V3-Pure D  Ball pendulum: gravity + localized damping zone")
    add_cylinder("D_PivotBar", (0.0, 0.0, 4.8), 0.08, 3.0, rotation=(0.0, math.radians(90), 0.0), collection=probes, material=mats["metal"])
    rod = add_cylinder("D_Rod", (0.0, 0.0, 3.77), 0.02, PHYSICS["rod_length"], rotation=(0.0, math.radians(180), 0.0), collection=probes, material=mats["metal"])
    ball = add_plain_ball("D_Ball", (0.0, 0.0, 2.5), PHYSICS["radius"], probes, mats["rubber"])
    add_cube("D_Magnet_L", (-0.58, 0.0, 1.86), (0.22, 0.30, 0.18), collection=probes, material=mats["accent"])
    add_cube("D_Magnet_R", (0.58, 0.0, 1.86), (0.22, 0.30, 0.18), collection=probes, material=mats["accent"])
    add_cube("D_DampingZone", (0.0, 0.0, 1.82), (0.56, 0.18, 0.06), collection=probes, material=mats["metal"])
    cam_side.location = (0.0, -13.8, 3.3)
    cam_side.data.lens = 42
    cam_main.location = (5.8, -12.8, 4.4)
    cam_main.data.lens = 38
    look_at(cam_main, (0.0, 0.0, 3.15))
    scene.camera = cam_main
    return rod, ball


def sample_thetas():
    p = PHYSICS
    sub_dt = 1.0 / (24.0 * p["integration_substeps"])
    length_eff = p["rod_length"] + p["radius"]
    theta = math.radians(p["theta0_deg"])
    omega = 0.0
    theta_zone = math.radians(p["theta_zone_deg"])
    values = [theta]
    for _frame in range(2, FRAME_END + 1):
        for _sub in range(p["integration_substeps"]):
            gamma_local = p["gamma_zone_hidden"] if abs(theta) <= theta_zone else 0.0
            alpha = -(p["g_hidden"] / length_eff) * math.sin(theta) - 2.0 * gamma_local * omega
            omega += alpha * sub_dt
            theta += omega * sub_dt
        values.append(theta)
    return values


def animate_system(rod, ball):
    p = PHYSICS
    rod_tf = []
    ball_tf = []
    theta_values = sample_thetas()
    for frame in range(1, FRAME_END + 1):
        theta = theta_values[frame - 1]
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
    "v3_D",
    "L3/L4",
    PHYSICS,
    {"D_Rod": rod_tf, "D_Ball": ball_tf},
    hidden_params=["g_hidden", "gamma_zone_hidden"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["D_Rod", "D_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
