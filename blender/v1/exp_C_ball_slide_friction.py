"""
Experiment C: single ball horizontal slide with floor friction.

Hidden parameter:
    mu_hidden : kinetic friction coefficient

Model:
    x(t) = x0 + v0 t - 1/2 mu_hidden g_known t^2
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


OUTPUT_FILE = Path(__file__).with_name("baselinev1_C_ball_slide_friction.blend")

PHYSICS = {
    "x0": -5.8,
    "y": 0.0,
    "z0": 0.44,
    "radius": 0.24,
    "v0": 4.2,
    "g_known": 9.81,
    "mu_hidden": 0.12,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v1_C"])


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V1-C  Ball floor slide: friction")
    add_cube("C_Floor", (0.5, 0.0, 0.12), (7.8, 0.78, 0.08), collection=probes, material=mats["floor"])
    ball = add_plain_ball("C_Ball", (PHYSICS["x0"], 0.0, PHYSICS["z0"]), PHYSICS["radius"], probes, mats["rubber"])
    cam_side.location = (0.0, -16.0, 3.0)
    cam_side.data.lens = 38
    cam_main.location = (3.8, -15.0, 3.2)
    cam_main.data.lens = 32
    look_at(cam_main, (-2.2, 0.0, 0.62))
    scene.camera = cam_main
    return ball


def sample_motion():
    p = PHYSICS
    a = p["mu_hidden"] * p["g_known"]
    t_stop = p["v0"] / a
    distance = 0.0
    last_x = p["x0"]
    transforms = []
    for frame in range(1, FRAME_END + 1):
        time_sec = min(t(frame), t_stop)
        x_now = p["x0"] + p["v0"] * time_sec - 0.5 * a * time_sec * time_sec
        distance += abs(x_now - last_x)
        last_x = x_now
        transforms.append((frame, (x_now, p["y"], p["z0"]), Euler((0.0, -distance / p["radius"], 0.0), "XYZ")))
    return transforms


ball = build()
transforms = sample_motion()
animate_transforms(ball, transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v1_C",
    "L1",
    PHYSICS,
    {"C_Ball": transforms},
    hidden_params=["mu_hidden"],
    known_params=["g_known"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["C_Ball"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
