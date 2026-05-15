"""
Experiment B: block spring-damper oscillation on a horizontal guide.

Hidden parameters:
    omega0_hidden : undamped natural frequency
    gamma_hidden  : linear damping coefficient

Model:
    x(t) = x_eq + A exp(-gamma_hidden t) cos(omega_d t)
    omega_d = sqrt(omega0_hidden^2 - gamma_hidden^2)

The block stays in contact with the guide, so the motion is fully determined by
the spring restoring force and linear damping. The trajectory is fit-friendly
through both its oscillation frequency and its exponential amplitude envelope.
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
    set_linear_interpolation,
    t,
)


OUTPUT_FILE = Path(__file__).with_name("baselinev3_B_block_spring_damper.blend")

PHYSICS = {
    "x_eq": -1.8,
    "amplitude": 1.65,
    "y": 0.0,
    "z_guide": 0.50,
    "omega0_hidden": 2.35,
    "gamma_hidden": 0.20,
    "guide_left": -7.0,
    "guide_right": 3.8,
    "anchor_x": -6.3,
    "block_half_x": 0.34,
}

OUTPUT_FILE, VARIANT_INFO = parse_variant_args(OUTPUT_FILE, PHYSICS, PARAM_VARIANTS["v3_B"])


def add_continuous_spring(name, start_x, end_x, y, z, radius, turns, points_per_turn, collection, material):
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 24
    curve.bevel_depth = 0.028
    curve.bevel_resolution = 6
    spline = curve.splines.new("POLY")
    total_points = turns * points_per_turn + 1
    spline.points.add(total_points - 1)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    if material:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    update_spring_points(obj, start_x, end_x, y, z, radius, turns, points_per_turn)
    return obj


def update_spring_points(obj, start_x, end_x, y, z, radius, turns, points_per_turn):
    total_points = turns * points_per_turn + 1
    spline = obj.data.splines[0]
    span = max(0.30, end_x - start_x)
    for idx, point in enumerate(spline.points):
        frac = idx / (total_points - 1)
        angle = 2.0 * math.pi * turns * frac
        x_now = start_x + span * frac
        y_now = y + radius * math.cos(angle)
        z_now = z + radius * math.sin(angle)
        point.co = (x_now, y_now, z_now, 1.0)


def build():
    scene, probes, mats, cam_main, cam_side = build_base_world("V3-Pure B  Block spring oscillation: stiffness + damping")
    add_cube("B_Floor", (-1.2, 0.0, 0.12), (7.8, 0.82, 0.08), collection=probes, material=mats["floor"])
    add_cube("B_Guide", (-1.6, 0.0, 0.32), (5.8, 0.28, 0.05), collection=probes, material=mats["metal"])
    add_cube("B_Wall", (-6.9, 0.0, 1.2), (0.16, 0.70, 1.0), collection=probes, material=mats["wall"])
    add_cube("B_Anchor", (PHYSICS["anchor_x"], 0.0, PHYSICS["z_guide"]), (0.06, 0.12, 0.12), collection=probes, material=mats["accent"])
    spring = add_continuous_spring(
        "B_Spring",
        PHYSICS["anchor_x"] + 0.12,
        PHYSICS["x_eq"] - PHYSICS["block_half_x"],
        0.0,
        PHYSICS["z_guide"],
        radius=0.18,
        turns=8,
        points_per_turn=20,
        collection=probes,
        material=mats["metal"],
    )
    block = add_cube("B_Block", (PHYSICS["x_eq"] + PHYSICS["amplitude"], 0.0, PHYSICS["z_guide"]), (PHYSICS["block_half_x"], 0.26, 0.30), collection=probes, material=mats["crate"])
    cam_side.location = (-1.4, -16.6, 2.9)
    cam_side.data.lens = 36
    cam_main.location = (3.6, -14.0, 3.4)
    cam_main.data.lens = 34
    look_at(cam_main, (-2.6, 0.0, 0.75))
    scene.camera = cam_main
    return block, spring


def sample_motion():
    p = PHYSICS
    omega_d = math.sqrt(max(0.08, p["omega0_hidden"] ** 2 - p["gamma_hidden"] ** 2))
    transforms = []
    for frame in range(1, FRAME_END + 1):
        time_sec = t(frame)
        x_now = p["x_eq"] + p["amplitude"] * math.exp(-p["gamma_hidden"] * time_sec) * math.cos(omega_d * time_sec)
        transforms.append((frame, (x_now, p["y"], p["z_guide"]), Euler((0.0, 0.0, 0.0), "XYZ")))
    return transforms


def animate_spring(spring_obj, block_transforms):
    p = PHYSICS
    start_x = p["anchor_x"] + 0.12
    spring_obj.animation_data_clear()
    points = spring_obj.data.splines[0].points
    for frame, location, _rotation in block_transforms:
        block_x = location[0]
        end_x = block_x - p["block_half_x"]
        update_spring_points(
            spring_obj,
            start_x,
            end_x,
            p["y"],
            p["z_guide"],
            radius=0.18,
            turns=8,
            points_per_turn=20,
        )
        for idx in range(len(points)):
            points[idx].keyframe_insert(data_path="co", frame=frame)
    set_linear_interpolation(spring_obj)


block, spring_obj = build()
block_transforms = sample_motion()
animate_transforms(block, block_transforms)
animate_spring(spring_obj, block_transforms)
export_ground_truth(
    OUTPUT_FILE,
    "v3_B",
    "L3",
    PHYSICS,
    {"B_Block": block_transforms},
    hidden_params=["omega0_hidden", "gamma_hidden"],
    variant_info=VARIANT_INFO,
    fps=24,
    frame_end=FRAME_END,
)
print_summary(["B_Block"])
cleanup_empty_collections()
cleanup_orphan_data()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))
