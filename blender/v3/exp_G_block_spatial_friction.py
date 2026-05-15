"""
Experiment G: a block slides over two friction zones with different hidden coefficients.

Hidden parameters:
    mu_1_hidden : friction coefficient in zone 1
    mu_2_hidden : friction coefficient in zone 2

This probe introduces a spatially varying physics field. The object experiences a clear
change in deceleration when crossing the boundary between the two surface regions.
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
from blender_scene_utils_v3 import (
    FRAME_END,
    add_cube,
    animate_transforms,
    build_base_world,
    cleanup_empty_collections,
    cleanup_orphan_data,
    look_at,
    print_summary,
    t,
)


OUTPUT_FILE = Path(__file__).with_name("baselinev3_G_block_spatial_friction.blend")

PHYSICS = {
    "x_start": -6.6,
    "y": 0.0,
    "z_center": 0.25,
    "half_x": 0.36,
    "v0_known": 4.6,
    "g_known": 9.81,
    "mu_1_hidden": 0.08,
    "mu_2_hidden": 0.26,
    "x_boundary": -0.2,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v3_G"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V3-Pure G  Block spatial friction field: mu1 + mu2")
    add_cube("G_Zone_1", (-3.7, 0.0, 0.08), (6.3, 1.1, 0.08), collection=probes, material=mats["floor"])
    add_cube("G_Zone_2", (4.0, 0.0, 0.08), (5.6, 1.1, 0.08), collection=probes, material=mats["rough"])
    add_cube("G_Boundary", (PHYSICS["x_boundary"], 0.0, 0.10), (0.04, 1.05, 0.10), collection=probes, material=mats["accent"])
    block = add_cube("G_Block", (PHYSICS["x_start"], 0.0, PHYSICS["z_center"]), (PHYSICS["half_x"], 0.28, PHYSICS["z_center"]), collection=probes, material=mats["crate"])
    cam_side.location = (0.0, -17.2, 2.9)
    cam_side.data.lens = 38
    cam_main.location = (3.8, -16.5, 3.4)
    cam_main.data.lens = 34
    look_at(cam_main, (-2.4, 0.0, 0.55))
    scene.camera = cam_main
    return block


def sample_motion():
    p = PHYSICS
    a1 = p["mu_1_hidden"] * p["g_known"]
    a2 = p["mu_2_hidden"] * p["g_known"]
    x_boundary_center = p["x_boundary"] - p["half_x"]
    disc = p["v0_known"] * p["v0_known"] - 2.0 * a1 * (x_boundary_center - p["x_start"])
    cross_zone = disc > 0.0
    if cross_zone:
        t_cross = (p["v0_known"] - math.sqrt(disc)) / a1
        v_cross = max(0.0, p["v0_known"] - a1 * t_cross)
        t_stop_2 = v_cross / a2
    else:
        t_stop_1 = p["v0_known"] / a1

    transforms = []
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        if cross_zone and time_sec > t_cross:
            tau = min(time_sec - t_cross, t_stop_2)
            x_now = x_boundary_center + v_cross * tau - 0.5 * a2 * tau * tau
        else:
            t_local = min(time_sec, t_stop_1) if not cross_zone else min(time_sec, t_cross)
            x_now = p["x_start"] + p["v0_known"] * t_local - 0.5 * a1 * t_local * t_local
        transforms.append((frame, (x_now, p["y"], p["z_center"]), Euler((0.0, 0.0, 0.0), "XYZ")))
    return transforms


block = build()
transforms = sample_motion()
animate_transforms(block, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v3_G",
    "L3",
    PHYSICS,
    {"G_Block": transforms},
    hidden_params=["mu_1_hidden", "mu_2_hidden"],
    known_params=["v0_known", "g_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["G_Block"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
