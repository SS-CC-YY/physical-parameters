"""
Experiment F: one ball collides with another, and both separate while decelerating on the floor.

Hidden parameters:
    e_hidden : restitution coefficient
    c_hidden : effective linear drag / floor damping coefficient

Masses are fixed and explicit. The hidden parameters are chosen so that they can be
fitted from the two outgoing trajectories rather than from hidden object properties.
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


OUTPUT_FILE = Path(__file__).with_name("baselinev3_F_two_ball_collision_drag.blend")

PHYSICS = {
    "radius": 0.25,
    "y_track": 0.0,
    "track_top": 0.16,
    "z_center": 0.41,
    "x1_start": -6.6,
    "x2_start": -0.8,
    "m1_known": 1.0,
    "m2_known": 1.0,
    "u1_known": 4.2,
    "u2_known": 0.0,
    "e_hidden": 0.74,
    "c_hidden": 0.34,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v3_F"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V3-Pure F  Two-ball collision: restitution + drag")
    add_cube("F_Track", (0.0, 0.0, 0.08), (9.8, 1.2, 0.08), collection=probes, material=mats["floor"])
    ball_1 = add_plain_ball("F_Ball_1", (PHYSICS["x1_start"], PHYSICS["y_track"], PHYSICS["z_center"]), PHYSICS["radius"], probes, mats["rubber"])
    ball_2 = add_plain_ball("F_Ball_2", (PHYSICS["x2_start"], PHYSICS["y_track"], PHYSICS["z_center"]), PHYSICS["radius"], probes, mats["accent"])
    cam_side.location = (0.0, -17.8, 3.0)
    cam_side.data.lens = 38
    cam_main.location = (8.8, -22.0, 4.5)
    cam_main.data.lens = 28
    look_at(cam_main, (1.4, 0.0, 0.7))
    scene.camera = cam_main
    return ball_1, ball_2


def outgoing_velocities():
    p = PHYSICS
    m1 = p["m1_known"]
    m2 = p["m2_known"]
    u1 = p["u1_known"]
    u2 = p["u2_known"]
    e = p["e_hidden"]
    v1 = ((m1 - e * m2) * u1 + (1.0 + e) * m2 * u2) / (m1 + m2)
    v2 = ((m2 - e * m1) * u2 + (1.0 + e) * m1 * u1) / (m1 + m2)
    return v1, v2


def damped_position(x0, v0, tau):
    c = PHYSICS["c_hidden"]
    if c <= 1e-8:
        return x0 + v0 * tau
    return x0 + (v0 / c) * (1.0 - math.exp(-c * tau))


def sample_motion():
    p = PHYSICS
    radius = p["radius"]
    t_hit = (p["x2_start"] - p["x1_start"] - 2.0 * radius) / max(1e-8, p["u1_known"] - p["u2_known"])
    x_contact_1 = p["x1_start"] + p["u1_known"] * t_hit
    x_contact_2 = p["x2_start"] + p["u2_known"] * t_hit
    v1_after, v2_after = outgoing_velocities()
    tf_1 = []
    tf_2 = []
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        if time_sec <= t_hit:
            x1 = p["x1_start"] + p["u1_known"] * time_sec
            x2 = p["x2_start"] + p["u2_known"] * time_sec
        else:
            tau = time_sec - t_hit
            x1 = damped_position(x_contact_1, v1_after, tau)
            x2 = damped_position(x_contact_2, v2_after, tau)
        tf_1.append((frame, (x1, p["y_track"], p["z_center"]), Euler((0.0, 0.0, 0.0), "XYZ")))
        tf_2.append((frame, (x2, p["y_track"], p["z_center"]), Euler((0.0, 0.0, 0.0), "XYZ")))
    return tf_1, tf_2


ball_1, ball_2 = build()
transforms_1, transforms_2 = sample_motion()
animate_transforms(ball_1, transforms_1)
animate_transforms(ball_2, transforms_2)
export_ground_truth(
    OUTPUT_FILE,
    "v3_F",
    "L3/L4",
    PHYSICS,
    {"F_Ball_1": transforms_1, "F_Ball_2": transforms_2},
    hidden_params=["e_hidden", "c_hidden"],
    known_params=["m1_known", "m2_known", "u1_known", "u2_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["F_Ball_1", "F_Ball_2"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
