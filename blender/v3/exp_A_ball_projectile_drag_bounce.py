"""
Experiment A: ball projectile with gravity, linear air drag, and bounce restitution.

Hidden parameters:
    g_hidden : gravity acceleration
    c_hidden : linear air drag coefficient
    e_hidden : floor restitution coefficient

Between impacts:
    dvx/dt = -c_hidden vx
    dvz/dt = -g_hidden - c_hidden vz

    vx(t) = vx0 exp(-c_hidden t)
    x(t)  = x0 + vx0 (1 - exp(-c_hidden t)) / c_hidden

    vz(t) = (vz0 + g_hidden / c_hidden) exp(-c_hidden t) - g_hidden / c_hidden
    z(t)  = z0 + (vz0 + g_hidden / c_hidden) (1 - exp(-c_hidden t)) / c_hidden - g_hidden t / c_hidden

At each floor impact z = z_floor:
    vz_after = e_hidden * |vz_before|
    vx_after = vx_before

This keeps the hidden quantities fit-friendly while allowing realistic bounce.
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


OUTPUT_FILE = Path(__file__).with_name("baselinev3_A_ball_projectile_drag_bounce.blend")

PHYSICS = {
    "x0": -6.0,
    "y": 0.0,
    "z0": 1.1,
    "vx0": 5.2,
    "vz0": 7.8,
    "radius": 0.24,
    "g_hidden": 9.81,
    "c_hidden": 0.42,
    "e_hidden": 0.78,
    "z_floor": 0.44,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v3_A"])


def solve_hit_time(z_start, vz_start, p, t_max):
    c = p["c_hidden"]
    g = p["g_hidden"]
    z_floor = p["z_floor"]

    def z_at(local_t):
        exp_term = math.exp(-c * local_t)
        return z_start + (vz_start + g / c) * (1.0 - exp_term) / c - g * local_t / c

    if z_at(t_max) > z_floor:
        return None
    lo, hi = 0.0, t_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if z_at(mid) > z_floor:
            lo = mid
        else:
            hi = mid
    return hi


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V3-Pure A  Ball projectile: gravity + drag + bounce")
    add_cube("A_Floor", (0.5, 0.0, 0.12), (8.0, 0.78, 0.08), collection=probes, material=mats["wall"])
    add_cube("A_StartMark", (-6.0, 0.0, 0.18), (0.08, 0.64, 0.02), collection=probes, material=mats["mark"])
    ball = add_plain_ball("A_Ball", (PHYSICS["x0"], 0.0, PHYSICS["z0"]), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.0, -16.0, 4.0)
    cam_side.data.lens = 34
    cam_main.location = (6.5, -17.5, 5.2)
    cam_main.data.lens = 30
    look_at(cam_main, (-0.5, 0.0, 1.55))
    scene.camera = cam_main
    return ball


def sample_motion():
    p = PHYSICS
    c = p["c_hidden"]
    g = p["g_hidden"]
    segments = []
    x_start = p["x0"]
    z_start = p["z0"]
    vx_start = p["vx0"]
    vz_start = p["vz0"]
    current_t = 0.0
    total_end = t(FRAME_END)

    while current_t < total_end:
        hit_local = solve_hit_time(z_start, vz_start, p, total_end - current_t)
        seg_end = total_end if hit_local is None else current_t + hit_local
        segments.append((current_t, seg_end, x_start, z_start, vx_start, vz_start))
        if hit_local is None:
            break
        exp_term = math.exp(-c * hit_local)
        x_hit = x_start + vx_start * (1.0 - exp_term) / c
        vz_before = (vz_start + g / c) * exp_term - g / c
        vx_before = vx_start * exp_term
        x_start = x_hit
        z_start = p["z_floor"]
        vx_start = vx_before
        vz_start = p["e_hidden"] * abs(vz_before)
        current_t = seg_end
        if vz_start < 0.05:
            segments.append((current_t, total_end, x_start, z_start, 0.0, 0.0))
            break

    transforms = []
    distance = 0.0
    last_x = p["x0"]
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        chosen = segments[-1]
        for segment in segments:
            if segment[0] <= time_sec <= segment[1]:
                chosen = segment
                break
        t0, t1, x0, z0, vx0, vz0 = chosen
        tau = max(0.0, min(time_sec - t0, t1 - t0))
        if abs(vx0) < 1e-10 and abs(vz0) < 1e-10:
            x_now = x0
            z_now = p["z_floor"]
        else:
            exp_term = math.exp(-c * tau)
            x_now = x0 + vx0 * (1.0 - exp_term) / c
            z_now = z0 + (vz0 + g / c) * (1.0 - exp_term) / c - g * tau / c
        distance += abs(x_now - last_x)
        last_x = x_now
        transforms.append((frame, (x_now, p["y"], max(p["z_floor"], z_now)), Euler((0.0, -distance / p["radius"], 0.0), "XYZ")))
    return transforms


ball = build()
transforms = sample_motion()
animate_transforms(ball, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v3_A",
    "L3/L4",
    PHYSICS,
    {"A_Ball": transforms},
    hidden_params=["g_hidden", "c_hidden", "e_hidden"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["A_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
