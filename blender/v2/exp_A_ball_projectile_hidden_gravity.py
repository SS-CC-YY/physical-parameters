"""
Experiment A: long projectile motion with ground bounce.

Hidden parameter:
    g_hidden : gravity acceleration

Known parameter:
    e_known : restitution coefficient
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
from blender_scene_utils_v2 import FRAME_END, add_cube, add_plain_ball, animate_transforms, build_base_world, cleanup_empty_collections, cleanup_orphan_data, look_at, print_summary, t


OUTPUT_FILE = Path(__file__).with_name("baselinev2_A_ball_projectile_hidden_gravity.blend")

PHYSICS = {
    "x0": -6.0,
    "y": 0.0,
    "z0": 1.2,
    "vx0": 4.8,
    "vz0": 7.8,
    "radius": 0.24,
    "z_floor": 0.44,
    "g_hidden": 9.81,
    "e_known": 0.74,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v2_A"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V2-A  Ball projectile + bounce: hidden gravity")
    add_cube("A_Floor", (1.0, 0.0, 0.12), (8.2, 0.78, 0.08), collection=probes, material=mats["wall"])
    ball = add_plain_ball("A_Ball", (PHYSICS["x0"], 0.0, PHYSICS["z0"]), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.4, -17.0, 3.8)
    cam_side.data.lens = 34
    cam_main.location = (16.5, -24.0, 6.8)
    cam_main.data.lens = 24
    look_at(cam_main, (8.5, 0.0, 1.75))
    scene.camera = cam_main
    return ball


def sample_motion():
    p = PHYSICS
    total_end = t(FRAME_END)
    segments = []
    x_start, z_start = p["x0"], p["z0"]
    vx_start, vz_start = p["vx0"], p["vz0"]
    current_t = 0.0
    while current_t < total_end:
        delta = z_start - p["z_floor"]
        disc = vz_start * vz_start + 2.0 * p["g_hidden"] * delta
        t_hit = (vz_start + math.sqrt(max(0.0, disc))) / p["g_hidden"]
        seg_end = min(total_end, current_t + t_hit)
        segments.append((current_t, seg_end, x_start, z_start, vx_start, vz_start))
        if current_t + t_hit >= total_end:
            break
        x_start = x_start + vx_start * t_hit
        vz_before = vz_start - p["g_hidden"] * t_hit
        z_start = p["z_floor"]
        vz_start = p["e_known"] * abs(vz_before)
        current_t += t_hit
        if vz_start < 0.06:
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
        x_now = x0 + vx0 * tau
        z_now = z0 + vz0 * tau - 0.5 * p["g_hidden"] * tau * tau
        distance += abs(x_now - last_x)
        last_x = x_now
        transforms.append((frame, (x_now, p["y"], max(p["z_floor"], z_now)), Euler((0.0, -distance / p["radius"], 0.0), "XYZ")))
    return transforms


ball = build()
transforms = sample_motion()
animate_transforms(ball, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v2_A",
    "L2",
    PHYSICS,
    {"A_Ball": transforms},
    hidden_params=["g_hidden"],
    known_params=["e_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["A_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
