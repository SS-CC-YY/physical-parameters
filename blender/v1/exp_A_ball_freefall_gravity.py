"""
Experiment A: single ball free fall under gravity.

Hidden parameter:
    g_hidden : gravity acceleration

Model:
    z(t) = z0 - 1/2 g_hidden t^2
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
from blender_scene_utils_v1 import FRAME_END, add_cube, add_plain_ball, animate_transforms, build_base_world, cleanup_empty_collections, cleanup_orphan_data, look_at, print_summary, t


OUTPUT_FILE = Path(__file__).with_name("baselinev1_A_ball_freefall_gravity.blend")

PHYSICS = {
    "x0": 0.0,
    "y": 0.0,
    "z0": 4.2,
    "radius": 0.24,
    "z_floor": 0.44,
    "g_hidden": 9.81,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v1_A"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V1-A  Ball free fall: gravity")
    add_cube("A_Floor", (0.0, 0.0, 0.12), (5.8, 0.78, 0.08), collection=probes, material=mats["wall"])
    ball = add_plain_ball("A_Ball", (PHYSICS["x0"], 0.0, PHYSICS["z0"]), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.0, -13.8, 3.6)
    cam_side.data.lens = 42
    cam_main.location = (5.8, -12.8, 4.8)
    cam_main.data.lens = 36
    look_at(cam_main, (0.0, 0.0, 2.35))
    scene.camera = cam_main
    return ball


def sample_motion():
    p = PHYSICS
    transforms = []
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        z_now = p["z0"] - 0.5 * p["g_hidden"] * time_sec * time_sec
        transforms.append((frame, (p["x0"], p["y"], max(p["z_floor"], z_now)), Euler((0.0, 0.0, 0.0), "XYZ")))
    return transforms


ball = build()
transforms = sample_motion()
animate_transforms(ball, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v1_A",
    "L1",
    PHYSICS,
    {"A_Ball": transforms},
    hidden_params=["g_hidden"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["A_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
