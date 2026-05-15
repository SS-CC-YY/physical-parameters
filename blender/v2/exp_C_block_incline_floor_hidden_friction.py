"""
Experiment C: block slides down an incline and continues on the floor.

Hidden parameter:
    mu_hidden : kinetic friction coefficient
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
from blender_scene_utils_v2 import FRAME_END, add_cube, animate_transforms, build_base_world, cleanup_empty_collections, cleanup_orphan_data, look_at, print_summary, t


OUTPUT_FILE = Path(__file__).with_name("baselinev2_C_block_incline_floor_hidden_friction.blend")

PHYSICS = {
    "theta_deg": 20.0,
    "ramp_length": 8.0,
    "x_top": -6.0,
    "z_top_surface": 2.9,
    "floor_run": 12.0,
    "block_half_x": 0.34,
    "block_half_z": 0.24,
    "y": 0.0,
    "g_known": 9.81,
    "mu_hidden": 0.22,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v2_C"])


def required_floor_run():
    p = PHYSICS
    theta = math.radians(p["theta_deg"])
    a_ramp = max(0.08, p["g_known"] * (math.sin(theta) - p["mu_hidden"] * math.cos(theta)))
    v_end = math.sqrt(2.0 * a_ramp * p["ramp_length"])
    v_floor0 = v_end * math.cos(theta)
    a_floor = max(1e-6, p["mu_hidden"] * p["g_known"])
    stop_distance = v_floor0 * v_floor0 / (2.0 * a_floor)
    return max(p["floor_run"], stop_distance + 1.5)


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V2-C  Block incline-to-floor slide: hidden friction")
    theta = math.radians(PHYSICS["theta_deg"])
    ramp_center_x = PHYSICS["x_top"] + 0.5 * PHYSICS["ramp_length"] * math.cos(theta)
    ramp_center_z = PHYSICS["z_top_surface"] - 0.5 * PHYSICS["ramp_length"] * math.sin(theta) - 0.12
    add_cube("C_Ramp", (ramp_center_x, 0.0, ramp_center_z), (PHYSICS["ramp_length"] * 0.53, 0.82, 0.12), rotation=(0.0, theta, 0.0), collection=probes, material=mats["wall"])
    z_surface_end = PHYSICS["z_top_surface"] - PHYSICS["ramp_length"] * math.sin(theta)
    z_floor_surface = z_surface_end - PHYSICS["block_half_z"] * (1.0 - math.cos(theta))
    floor_run = required_floor_run()
    x_floor_center = PHYSICS["x_top"] + PHYSICS["ramp_length"] * math.cos(theta) + 0.5 * floor_run
    add_cube("C_Floor", (x_floor_center, 0.0, z_floor_surface - 0.08), (floor_run * 0.52, 0.82, 0.08), collection=probes, material=mats["floor"])
    block = add_cube("C_Block", (-5.6, 0.0, 2.0), (PHYSICS["block_half_x"], 0.28, PHYSICS["block_half_z"]), rotation=(0.0, theta, 0.0), collection=probes, material=mats["crate"])
    cam_side.location = (-0.2, -19.0, 2.9)
    cam_side.data.lens = 31
    cam_main.location = (6.5, -22.0, 5.6)
    cam_main.data.lens = 26
    look_at(cam_main, (0.8, 0.0, 1.65))
    scene.camera = cam_main
    return block


def sample_motion():
    p = PHYSICS
    theta = math.radians(p["theta_deg"])
    a_ramp = p["g_known"] * (math.sin(theta) - p["mu_hidden"] * math.cos(theta))
    a_ramp = max(0.08, a_ramp)
    t_ramp = math.sqrt(2.0 * p["ramp_length"] / a_ramp)
    v_end = a_ramp * t_ramp
    v_floor0 = v_end * math.cos(theta)
    a_floor = p["mu_hidden"] * p["g_known"]
    t_floor = v_floor0 / a_floor
    x_surface_end = p["x_top"] + p["ramp_length"] * math.cos(theta)
    z_surface_end = p["z_top_surface"] - p["ramp_length"] * math.sin(theta)
    x_end = x_surface_end - p["block_half_z"] * math.sin(theta)
    z_floor_center = z_surface_end + p["block_half_z"] * (2.0 * math.cos(theta) - 1.0)
    transforms = []
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        if time_sec <= t_ramp:
            s = min(0.5 * a_ramp * time_sec * time_sec, p["ramp_length"])
            x_surface = p["x_top"] + s * math.cos(theta)
            z_surface = p["z_top_surface"] - s * math.sin(theta)
            x_now = x_surface - p["block_half_z"] * math.sin(theta)
            z_now = z_surface + p["block_half_z"] * math.cos(theta)
            rot = Euler((0.0, theta, 0.0), "XYZ")
        else:
            tau = min(time_sec - t_ramp, t_floor)
            x_now = x_end + v_floor0 * tau - 0.5 * a_floor * tau * tau
            z_now = z_floor_center
            rot = Euler((0.0, 0.0, 0.0), "XYZ")
        transforms.append((frame, (x_now, p["y"], z_now), rot))
    return transforms


block = build()
transforms = sample_motion()
animate_transforms(block, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v2_C",
    "L2",
    PHYSICS,
    {"C_Block": transforms},
    hidden_params=["mu_hidden"],
    known_params=["g_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["C_Block"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
