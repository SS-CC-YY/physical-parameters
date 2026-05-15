"""
Experiment E: repeated bounce on a tilted floor.

Hidden parameter:
    e_hidden : normal restitution coefficient

Known parameters:
    g_known, floor tilt angle, tangential impact retention

This is the second V2-B-style branch. The hidden parameter is still restitution,
but the impact normal is tilted, so the bounce chain couples vertical motion
with horizontal drift. It is intentionally not a duplicate of V2-A projectile
motion: the inference target is one constant e across repeated oblique impacts.
"""

from pathlib import Path
import math
import sys

import bpy
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_io import export_ground_truth, parse_variant_args
from benchmark_variants import PARAM_VARIANTS
from blender_scene_utils_v2 import FRAME_END, FPS, add_cube, add_plain_ball, animate_transforms, build_base_world, cleanup_empty_collections, cleanup_orphan_data, look_at, print_summary, t


OUTPUT_FILE = Path(__file__).with_name("baselinev2_E_ball_tilted_floor_bounce_hidden_restitution.blend")

PHYSICS = {
    "anchor_x": -5.6,
    "anchor_y": 0.0,
    "anchor_z": 1.12,
    "floor_angle_deg": 4.0,
    "floor_length": 18.0,
    "radius": 0.24,
    "x0": -3.2,
    "z0": 4.4,
    "g_known": 9.81,
    "tangent_retention_known": 0.68,
    "e_hidden": 0.84,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v2_E"])


def basis_vectors():
    alpha = math.radians(PHYSICS["floor_angle_deg"])
    tangent = Vector((math.cos(alpha), 0.0, -math.sin(alpha)))
    normal = Vector((math.sin(alpha), 0.0, math.cos(alpha)))
    return alpha, tangent, normal


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V2-E  Tilted-floor bounce: hidden restitution")
    alpha, tangent, normal = basis_vectors()
    anchor = Vector((PHYSICS["anchor_x"], 0.0, PHYSICS["anchor_z"]))
    floor_center = anchor + 0.5 * PHYSICS["floor_length"] * tangent - 0.10 * normal
    add_cube(
        "E_TiltedFloor",
        floor_center,
        (PHYSICS["floor_length"] * 0.5, 0.82, 0.10),
        rotation=(0.0, alpha, 0.0),
        collection=probes,
        material=mats["wall"],
    )
    ball = add_plain_ball("E_Ball", (PHYSICS["x0"], 0.0, PHYSICS["z0"]), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (1.9, -17.0, 2.7)
    cam_side.data.lens = 34
    cam_main.location = (6.6, -17.8, 4.2)
    cam_main.data.lens = 32
    look_at(cam_main, (1.9, 0.0, 1.45))
    scene.camera = cam_main
    return ball


def sample_motion():
    p = PHYSICS
    _alpha, tangent, normal = basis_vectors()
    anchor = Vector((p["anchor_x"], 0.0, p["anchor_z"]))
    pos = Vector((p["x0"], p["anchor_y"], p["z0"]))
    vel = Vector((0.0, 0.0, 0.0))
    gravity = Vector((0.0, 0.0, -p["g_known"]))
    dt = 1.0 / (FPS * 18.0)
    current_t = 0.0
    transforms = []
    next_frame = 1

    while next_frame <= FRAME_END:
        target_t = t(next_frame)
        while current_t + 1e-10 < target_t:
            h = min(dt, target_t - current_t)
            vel += gravity * h
            pos += vel * h
            signed_distance = (pos - anchor).dot(normal)
            normal_speed = vel.dot(normal)
            if signed_distance <= p["radius"] and normal_speed < 0.0:
                pos += normal * (p["radius"] - signed_distance)
                v_n = normal * normal_speed
                v_t = tangent * (p["tangent_retention_known"] * vel.dot(tangent))
                vel = v_t - p["e_hidden"] * v_n
            current_t += h
        transforms.append((next_frame, tuple(pos), Euler((0.0, 0.0, 0.0), "XYZ")))
        next_frame += 1
    return transforms


ball = build()
transforms = sample_motion()
animate_transforms(ball, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v2_E",
    "L2",
    PHYSICS,
    {"E_Ball": transforms},
    hidden_params=["e_hidden"],
    known_params=["g_known", "floor_angle_deg", "tangent_retention_known"],
    variant_info=VARIANT_INFO,
    fps=FPS,
    frame_end=FRAME_END,
)
print_summary(["E_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
