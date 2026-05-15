"""
Experiment B: long-chain repeated vertical bounce with fixed gravity.

Hidden parameter:
    e_hidden : restitution coefficient

Known parameters:
    g_known

Unlike V1-B, this L2 probe keeps the motion one-dimensional and extends only
the event chain length: the same hidden restitution must remain consistent
across many impacts.
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


OUTPUT_FILE = Path(__file__).with_name("baselinev2_B_ball_repeated_bounce_hidden_restitution.blend")

PHYSICS = {
    "x0": 0.0,
    "y": 0.0,
    "z0": 4.8,
    "radius": 0.24,
    "z_floor": 0.44,
    "g_known": 9.81,
    "e_hidden": 0.84,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v2_B"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V2-B  Ball repeated bounce: hidden restitution")
    add_cube("B_Floor", (0.0, 0.0, 0.12), (6.0, 0.78, 0.08), collection=probes, material=mats["wall"])
    ball = add_plain_ball("B_Ball", (PHYSICS["x0"], 0.0, PHYSICS["z0"]), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.0, -14.8, 4.0)
    cam_side.data.lens = 40
    cam_main.location = (6.2, -13.8, 5.0)
    cam_main.data.lens = 36
    look_at(cam_main, (0.0, 0.0, 2.45))
    scene.camera = cam_main
    return ball


def sample_motion():
    p = PHYSICS
    total_end = t(FRAME_END)
    segments = []
    z_start = p["z0"]
    vz_start = 0.0
    current_t = 0.0
    while current_t < total_end:
        delta = z_start - p["z_floor"]
        t_hit = (vz_start + math.sqrt(max(0.0, vz_start * vz_start + 2.0 * p["g_known"] * delta))) / p["g_known"]
        seg_end = min(total_end, current_t + t_hit)
        segments.append((current_t, seg_end, z_start, vz_start))
        if current_t + t_hit >= total_end:
            break
        vz_before = vz_start - p["g_known"] * t_hit
        vz_start = p["e_hidden"] * abs(vz_before)
        z_start = p["z_floor"]
        current_t += t_hit
        if vz_start < 0.05:
            segments.append((current_t, total_end, p["z_floor"], 0.0))
            break
    transforms = []
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        chosen = segments[-1]
        for segment in segments:
            if segment[0] <= time_sec <= segment[1]:
                chosen = segment
                break
        t0, t1, z0, vz0 = chosen
        tau = max(0.0, min(time_sec - t0, t1 - t0))
        z_now = z0 + vz0 * tau - 0.5 * p["g_known"] * tau * tau
        transforms.append((frame, (p["x0"], p["y"], max(p["z_floor"], z_now)), Euler((0.0, 0.0, 0.0), "XYZ")))
    return transforms


ball = build()
transforms = sample_motion()
animate_transforms(ball, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v2_B",
    "L2",
    PHYSICS,
    {"B_Ball": transforms},
    hidden_params=["e_hidden"],
    known_params=["g_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["B_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
