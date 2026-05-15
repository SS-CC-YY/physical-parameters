"""
Experiment C: block slides down an incline, crosses the floor, and collides with a wall.

Hidden parameters:
    g_hidden  : gravity acceleration
    mu_hidden : kinetic friction coefficient
    e_hidden  : wall restitution coefficient

Model along incline coordinate s:
    s'' = g_hidden (sin(theta) - mu_hidden cos(theta))
    s(t) = 1/2 a_ramp t^2

Floor phase before impact:
    x(t) = x_end + v_floor0 tau - 1/2 mu_hidden g_hidden tau^2

At wall impact:
    v_after = -e_hidden v_before

Floor rebound phase:
    x(t) = x_contact - v_rebound tau + 1/2 mu_hidden g_hidden tau^2

This makes C a true multi-rule coupled probe rather than only a two-stage slide.
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


OUTPUT_FILE = Path(__file__).with_name("baselinev3_C_block_incline_floor_friction.blend")

PHYSICS = {
    "s0": 0.0,
    "theta_deg": 20.0,
    "ramp_length": 8.6,
    "x_top": -6.0,
    "z_top_surface": 2.9,
    "floor_run": 7.0,
    "block_half_x": 0.34,
    "block_half_z": 0.24,
    "y": 0.0,
    "g_hidden": 9.81,
    "mu_hidden": 0.22,
    "e_hidden": 0.62,
    "x_wall": 5.9,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v3_C"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V3-Pure C  Block slide + wall collision: gravity + friction + restitution")
    theta = math.radians(PHYSICS["theta_deg"])
    ramp_center_x = PHYSICS["x_top"] + 0.5 * PHYSICS["ramp_length"] * math.cos(theta)
    ramp_center_z = PHYSICS["z_top_surface"] - 0.5 * PHYSICS["ramp_length"] * math.sin(theta) - 0.12
    add_cube("C_Ramp", (ramp_center_x, 0.0, ramp_center_z), (PHYSICS["ramp_length"] * 0.53, 0.82, 0.12), rotation=(0.0, math.radians(PHYSICS["theta_deg"]), 0.0), collection=probes, material=mats["wall"])
    z_surface_end = PHYSICS["z_top_surface"] - PHYSICS["ramp_length"] * math.sin(theta)
    z_floor_surface = z_surface_end - PHYSICS["block_half_z"] * (1.0 - math.cos(theta))
    x_floor_center = PHYSICS["x_top"] + PHYSICS["ramp_length"] * math.cos(theta) + 0.5 * PHYSICS["floor_run"]
    add_cube("C_Floor", (x_floor_center, 0.0, z_floor_surface - 0.08), (PHYSICS["floor_run"] * 0.52, 0.82, 0.08), collection=probes, material=mats["floor"])
    add_cube("C_Wall", (PHYSICS["x_wall"] + 0.10, 0.0, z_floor_surface + 0.62), (0.10, 0.82, 0.70), collection=probes, material=mats["accent"])
    block = add_cube("C_Block", (-5.6, 0.0, 2.0), (PHYSICS["block_half_x"], 0.28, PHYSICS["block_half_z"]), rotation=(0.0, math.radians(PHYSICS["theta_deg"]), 0.0), collection=probes, material=mats["crate"])
    cam_side.location = (-0.2, -19.0, 2.9)
    cam_side.data.lens = 31
    cam_main.location = (6.5, -20.5, 5.4)
    cam_main.data.lens = 28
    look_at(cam_main, (-0.5, 0.0, 1.55))
    scene.camera = cam_main
    return block


def sample_motion():
    p = PHYSICS
    theta = math.radians(p["theta_deg"])
    a_ramp = p["g_hidden"] * (math.sin(theta) - p["mu_hidden"] * math.cos(theta))
    a_ramp = max(0.08, a_ramp)
    t_ramp = math.sqrt(2.0 * p["ramp_length"] / a_ramp)
    v_end = a_ramp * t_ramp
    v_floor0 = v_end * math.cos(theta)
    a_floor = p["mu_hidden"] * p["g_hidden"]
    x_surface_end = p["x_top"] + p["ramp_length"] * math.cos(theta)
    z_surface_end = p["z_top_surface"] - p["ramp_length"] * math.sin(theta)
    x_end = x_surface_end - p["block_half_z"] * math.sin(theta)
    z_floor_center = z_surface_end + p["block_half_z"] * (2.0 * math.cos(theta) - 1.0)
    x_contact = p["x_wall"] - p["block_half_x"]
    stop_distance = v_floor0 * v_floor0 / (2.0 * a_floor)
    will_hit = x_end + stop_distance >= x_contact
    t_hit_wall = None
    v_rebound = 0.0
    if will_hit:
        disc = v_floor0 * v_floor0 - 2.0 * a_floor * (x_contact - x_end)
        t_hit_wall = (v_floor0 - math.sqrt(max(0.0, disc))) / a_floor
        v_before = max(0.0, v_floor0 - a_floor * t_hit_wall)
        v_rebound = p["e_hidden"] * v_before
        t_rebound_stop = v_rebound / a_floor if v_rebound > 0.0 else 0.0
    else:
        t_floor = v_floor0 / a_floor
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
            tau_total = time_sec - t_ramp
            if will_hit and tau_total <= t_hit_wall:
                tau = tau_total
                x_now = x_end + v_floor0 * tau - 0.5 * a_floor * tau * tau
                z_now = z_floor_center
                rot = Euler((0.0, 0.0, 0.0), "XYZ")
            elif will_hit:
                tau = min(tau_total - t_hit_wall, t_rebound_stop)
                x_now = x_contact - v_rebound * tau + 0.5 * a_floor * tau * tau
                z_now = z_floor_center
                rot = Euler((0.0, 0.0, 0.0), "XYZ")
            else:
                tau = min(tau_total, t_floor)
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
    "v3_C",
    "L3/L4",
    PHYSICS,
    {"C_Block": transforms},
    hidden_params=["g_hidden", "mu_hidden", "e_hidden"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["C_Block"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
