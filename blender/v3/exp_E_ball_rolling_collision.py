"""
Experiment E: rolling ball down a ramp, across the floor, then colliding with a wall.

Hidden parameters:
    mu_r_hidden : effective rolling resistance on the floor
    e_hidden    : wall restitution coefficient

On the incline, the ball is modeled as a solid sphere rolling without slipping:
    a_ramp = g sin(theta) / (1 + I / (m R^2)) = (5/7) g sin(theta)

On the floor, the horizontal speed decays under effective rolling resistance:
    x(t) = x_0 + v_0 t - 1/2 mu_r_hidden g t^2

At wall impact:
    v_after = -e_hidden v_before

This experiment complements sliding probes with explicit rolling dynamics.
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
    add_plain_ball,
    animate_transforms,
    build_base_world,
    cleanup_empty_collections,
    cleanup_orphan_data,
    look_at,
    print_summary,
    t,
)


OUTPUT_FILE = Path(__file__).with_name("baselinev3_E_ball_rolling_collision.blend")

PHYSICS = {
    "radius": 0.25,
    "theta_deg": 18.0,
    "ramp_length": 6.4,
    "x_top": -7.0,
    "z_top_surface": 3.2,
    "floor_run": 8.2,
    "g_known": 9.81,
    "mu_r_hidden": 0.075,
    "e_hidden": 0.68,
    "x_wall": 6.4,
    "y": 0.0,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v3_E"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V3-Pure E  Ball rolling + wall collision: rolling resistance + restitution")
    theta = math.radians(PHYSICS["theta_deg"])
    ramp_center_x = PHYSICS["x_top"] + 0.5 * PHYSICS["ramp_length"] * math.cos(theta)
    ramp_center_z = PHYSICS["z_top_surface"] - 0.5 * PHYSICS["ramp_length"] * math.sin(theta) - 0.12
    add_cube(
        "E_Ramp",
        (ramp_center_x, 0.0, ramp_center_z),
        (PHYSICS["ramp_length"] * 0.53, 0.82, 0.12),
        rotation=(0.0, math.radians(PHYSICS["theta_deg"]), 0.0),
        collection=probes,
        material=mats["wall"],
    )
    z_surface_end = PHYSICS["z_top_surface"] - PHYSICS["ramp_length"] * math.sin(theta)
    floor_center_x = PHYSICS["x_top"] + PHYSICS["ramp_length"] * math.cos(theta) + 0.5 * PHYSICS["floor_run"]
    add_cube("E_Floor", (floor_center_x, 0.0, z_surface_end - 0.08), (PHYSICS["floor_run"] * 0.52, 0.82, 0.08), collection=probes, material=mats["floor"])
    add_cube("E_Wall", (PHYSICS["x_wall"] + 0.10, 0.0, z_surface_end + 0.66), (0.10, 0.82, 0.74), collection=probes, material=mats["accent"])
    ball = add_plain_ball("E_Ball", (-6.8, 0.0, 3.1), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.0, -18.4, 3.1)
    cam_side.data.lens = 34
    cam_main.location = (6.5, -20.5, 5.6)
    cam_main.data.lens = 28
    look_at(cam_main, (-1.0, 0.0, 2.35))
    scene.camera = cam_main
    return ball


def sample_motion():
    p = PHYSICS
    theta = math.radians(p["theta_deg"])
    radius = p["radius"]
    a_ramp = (5.0 / 7.0) * p["g_known"] * math.sin(theta)
    t_ramp = math.sqrt(2.0 * p["ramp_length"] / a_ramp)
    v_end = a_ramp * t_ramp
    a_floor = p["mu_r_hidden"] * p["g_known"]
    x_surface_end = p["x_top"] + p["ramp_length"] * math.cos(theta)
    z_surface_end = p["z_top_surface"] - p["ramp_length"] * math.sin(theta)
    x_end = x_surface_end - radius * math.sin(theta)
    z_floor_center = z_surface_end + radius
    x_contact = p["x_wall"] - radius
    stop_distance = v_end * v_end / (2.0 * a_floor)
    will_hit = x_end + stop_distance >= x_contact
    t_hit = None
    v_rebound = 0.0
    t_rebound_stop = 0.0
    if will_hit:
        disc = v_end * v_end - 2.0 * a_floor * (x_contact - x_end)
        t_hit = (v_end - math.sqrt(max(0.0, disc))) / a_floor
        v_before = max(0.0, v_end - a_floor * t_hit)
        v_rebound = p["e_hidden"] * v_before
        t_rebound_stop = v_rebound / a_floor if v_rebound > 0.0 else 0.0
    else:
        t_floor = v_end / a_floor

    transforms = []
    angle = 0.0
    prev_x = x_end
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        if time_sec <= t_ramp:
            s = min(0.5 * a_ramp * time_sec * time_sec, p["ramp_length"])
            x_surface = p["x_top"] + s * math.cos(theta)
            z_surface = p["z_top_surface"] - s * math.sin(theta)
            x_now = x_surface - radius * math.sin(theta)
            z_now = z_surface + radius * math.cos(theta)
        else:
            tau_total = time_sec - t_ramp
            if will_hit and tau_total <= t_hit:
                tau = tau_total
                x_now = x_end + v_end * tau - 0.5 * a_floor * tau * tau
                z_now = z_floor_center
            elif will_hit:
                tau = min(tau_total - t_hit, t_rebound_stop)
                x_now = x_contact - v_rebound * tau + 0.5 * a_floor * tau * tau
                z_now = z_floor_center
            else:
                tau = min(tau_total, t_floor)
                x_now = x_end + v_end * tau - 0.5 * a_floor * tau * tau
                z_now = z_floor_center
        dx = x_now - prev_x if frame > 1 else 0.0
        angle -= dx / radius
        prev_x = x_now
        transforms.append((frame, (x_now, p["y"], z_now), Euler((0.0, angle, 0.0), "XYZ")))
    return transforms


ball = build()
transforms = sample_motion()
animate_transforms(ball, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v3_E",
    "L3",
    PHYSICS,
    {"E_Ball": transforms},
    hidden_params=["mu_r_hidden", "e_hidden"],
    known_params=["g_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["E_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
