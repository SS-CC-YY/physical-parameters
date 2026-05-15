import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path


def parse_variant_args(default_output_file, physics, param_variants=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant-id", default="default")
    parser.add_argument("--override-json", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--output-dir", default=None)
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    args, _unknown = parser.parse_known_args(argv)

    selected = {}
    variants = param_variants or {}
    if args.variant_id != "default":
        if args.variant_id not in variants:
            raise ValueError(f"Unknown variant-id {args.variant_id!r}; available: {sorted(variants)}")
        selected.update(variants[args.variant_id])
    if args.override_json:
        selected.update(json.loads(args.override_json))

    physics.update(selected)
    output_file = Path(args.output_file) if args.output_file else Path(default_output_file)
    if args.output_dir:
        output_file = Path(args.output_dir) / output_file.name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file, {"variant_id": args.variant_id, "overrides": selected}


def transforms_to_rows(object_name, transforms, fps):
    rows = []
    for frame, location, rotation in transforms:
        rows.append(
            {
                "frame": int(frame),
                "time_s": round((int(frame) - 1) / fps, 8),
                "object": object_name,
                "x": round(float(location[0]), 8),
                "y": round(float(location[1]), 8),
                "z": round(float(location[2]), 8),
                "rot_x": round(float(rotation[0]), 8),
                "rot_y": round(float(rotation[1]), 8),
                "rot_z": round(float(rotation[2]), 8),
            }
        )
    return rows


def export_trajectory_csv(object_transforms, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame", "time_s", "object", "x", "y", "z", "rot_x", "rot_y", "rot_z"]
    rows = []
    for object_name, transforms in object_transforms.items():
        rows.extend(transforms_to_rows(object_name, transforms, fps))
    rows.sort(key=lambda row: (row["frame"], row["object"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def camera_metadata():
    try:
        import bpy
    except ImportError:
        return []
    cameras = []
    for obj in sorted((o for o in bpy.data.objects if o.type == "CAMERA"), key=lambda o: o.name):
        cameras.append(
            {
                "name": obj.name,
                "location": [round(float(v), 8) for v in obj.location],
                "rotation_euler": [round(float(v), 8) for v in obj.rotation_euler],
                "lens": round(float(obj.data.lens), 8),
            }
        )
    return cameras


def export_params_json(physics, path, *, experiment_id, level, hidden_params, known_params=None, variant_info=None, fps=None, frame_end=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    known_params = known_params or []
    data = {
        "experiment_id": experiment_id,
        "level": level,
        "fps": fps,
        "frame_end": frame_end,
        "variant": variant_info or {"variant_id": "default", "overrides": {}},
        "hidden_params": {key: physics[key] for key in hidden_params if key in physics},
        "known_params": {key: physics[key] for key in known_params if key in physics},
        "all_physics": deepcopy(physics),
        "cameras": camera_metadata(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_ground_truth(output_file, experiment_id, level, physics, object_transforms, hidden_params, known_params=None, variant_info=None, fps=None, frame_end=None):
    stem = Path(output_file).with_suffix("")
    export_trajectory_csv(object_transforms, stem.with_name(stem.name + "_trajectory.csv"), fps)
    export_params_json(
        physics,
        stem.with_name(stem.name + "_params.json"),
        experiment_id=experiment_id,
        level=level,
        hidden_params=hidden_params,
        known_params=known_params,
        variant_info=variant_info,
        fps=fps,
        frame_end=frame_end,
    )
