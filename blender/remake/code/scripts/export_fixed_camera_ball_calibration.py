#!/usr/bin/env python3
"""Export fixed-camera, frame-1 standard-ball calibration for all experiments.

Run this script with Blender, not the system Python::

    blender --background --python code/scripts/export_fixed_camera_ball_calibration.py -- \
      --workspace-root . --output-dir code/assets/fixed_camera_ball_calibration

The exporter discovers the frozen 13 experiments from
``benchmark_release_v1_0/experiment_registry.json``.  It only opens source
``.blend`` files and never invokes a Blender save operator.  SHA-256 digests
are computed before and after all reads, and output is written only after the
digests have been verified unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
EXPORTER_VERSION = "1.0.0"
DEFAULT_SCENES = (
    "baseline",
    "indoor1",
    "indoor2",
    "indoor3",
    "indoor4",
    "outdoor1",
    "outdoor2",
    "outdoor3",
    "outdoor4",
)
DEFAULT_CAMERAS = ("CAM_Main", "CAM_Side", "CAM_Top")
NEAR_SPHERE_MIN_AXIS_RATIO = 0.75


def _script_args(argv: list[str]) -> list[str]:
    """Return arguments following Blender's conventional ``--`` separator."""

    return argv[argv.index("--") + 1 :] if "--" in argv else []


def _parser() -> argparse.ArgumentParser:
    default_workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Read frozen benchmark .blend files without saving and export frame-1 "
            "standard-ball geometry plus fixed-camera OpenCV calibration."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=default_workspace,
        help="Benchmark workspace containing benchmark_release_v1_0 and v1/v2/v3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root for {experiment}/{scene}/{camera}.json and manifest.json.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment ids from the registry (default: all 13); 'all' is accepted.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEFAULT_SCENES),
        help="Scenario ids (default: baseline plus indoor1-4 and outdoor1-4).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="Camera object names (default: CAM_Main CAM_Side CAM_Top).",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace exporter-owned JSON files already present in output-dir.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _vector(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row][k] * right[k][column] for k in range(len(right)))
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _normalised_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _contains_standard_ball(name: str) -> bool:
    normalised = _normalised_name(name)
    return "standard_ball" in normalised or "standardball" in normalised


def _object_chain(obj: Any) -> list[Any]:
    chain: list[Any] = []
    current = obj
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.parent
    return chain


def _json_value(value: Any, depth: int = 0) -> Any:
    """Convert Blender ID properties to finite JSON-compatible values."""

    if depth > 16:
        return repr(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping) or hasattr(value, "keys"):
        try:
            return {
                str(key): _json_value(value[key], depth + 1)
                for key in sorted(value.keys(), key=lambda item: str(item))
            }
        except Exception:
            pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, depth + 1) for item in value]
    try:
        return [_json_value(item, depth + 1) for item in value]
    except Exception:
        return repr(value)


def _custom_properties(owner: Any) -> dict[str, Any]:
    return {
        str(key): _json_value(owner[key])
        for key in sorted(owner.keys(), key=str)
        if str(key) != "_RNA_UI"
    }


def _camera_intrinsics(camera_data: Any, scene: Any, width: int, height: int) -> dict[str, Any]:
    """Return a top-left/OpenCV pinhole K matching Blender's projection."""

    if str(camera_data.type) != "PERSP":
        raise RuntimeError(
            f"camera {camera_data.name!r} has unsupported type {camera_data.type!r}; "
            "a pinhole K requires a perspective camera"
        )
    pixel_aspect = float(scene.render.pixel_aspect_y) / float(scene.render.pixel_aspect_x)
    sensor_fit = str(camera_data.sensor_fit)
    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL" if width >= pixel_aspect * height else "VERTICAL"

    if sensor_fit == "VERTICAL":
        sensor_size_mm = float(camera_data.sensor_height)
        view_factor_px = pixel_aspect * height
    else:
        sensor_size_mm = float(camera_data.sensor_width)
        view_factor_px = float(width)

    focal_px_x = float(camera_data.lens) * view_factor_px / sensor_size_mm
    focal_px_y = focal_px_x / pixel_aspect
    principal_x = width * 0.5 - float(camera_data.shift_x) * view_factor_px
    principal_y = height * 0.5 + float(camera_data.shift_y) * view_factor_px / pixel_aspect
    matrix = [
        [focal_px_x, 0.0, principal_x],
        [0.0, focal_px_y, principal_y],
        [0.0, 0.0, 1.0],
    ]
    return {
        "matrix": matrix,
        "fx_px": focal_px_x,
        "fy_px": focal_px_y,
        "cx_px": principal_x,
        "cy_px": principal_y,
        "resolved_sensor_fit": sensor_fit,
        "pixel_aspect_y_over_x": pixel_aspect,
    }


def _opencv_extrinsics(camera: Any) -> dict[str, Any]:
    """Convert Blender's -Z-forward camera convention to OpenCV +Z-forward."""

    from mathutils import Matrix  # type: ignore

    rotation_world_to_blender_camera = camera.matrix_world.to_3x3().transposed()
    translation_world_to_blender_camera = -(
        rotation_world_to_blender_camera @ camera.matrix_world.translation
    )
    blender_camera_to_opencv = Matrix(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    )
    rotation_world_to_opencv = blender_camera_to_opencv @ rotation_world_to_blender_camera
    translation_world_to_opencv = blender_camera_to_opencv @ translation_world_to_blender_camera
    transform_world_to_opencv = Matrix.Identity(4)
    for row in range(3):
        for column in range(3):
            transform_world_to_opencv[row][column] = rotation_world_to_opencv[row][column]
        transform_world_to_opencv[row][3] = translation_world_to_opencv[row]
    return {
        "R_world_to_camera": _rows(rotation_world_to_opencv),
        "t_world_to_camera_m": _vector(translation_world_to_opencv),
        "T_world_to_camera": _rows(transform_world_to_opencv),
    }


def _project(
    K: list[list[float]],
    rotation: list[list[float]],
    translation: list[float],
    xyz: Iterable[Any],
) -> tuple[list[float], float]:
    point = [float(value) for value in xyz]
    camera_xyz = [
        sum(rotation[row][column] * point[column] for column in range(3))
        + translation[row]
        for row in range(3)
    ]
    depth = float(camera_xyz[2])
    if depth <= 0.0:
        raise RuntimeError(f"point is behind the OpenCV camera: xyz={point}, camera_xyz={camera_xyz}")
    homogeneous = [
        sum(K[row][column] * camera_xyz[column] for column in range(3))
        for row in range(3)
    ]
    return [homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]], depth


def _load_registry(workspace_root: Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    registry_path = workspace_root / "benchmark_release_v1_0" / "experiment_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"missing experiment registry: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    experiments_raw = registry.get("experiments")
    if not isinstance(experiments_raw, list):
        raise ValueError(f"{registry_path}: expected an experiments list")
    experiments: dict[str, dict[str, Any]] = {}
    for raw in experiments_raw:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            raise ValueError(f"{registry_path}: malformed experiment entry {raw!r}")
        experiment_id = str(raw["id"])
        if experiment_id in experiments:
            raise ValueError(f"{registry_path}: duplicate experiment id {experiment_id}")
        experiments[experiment_id] = raw
    expected_count = int(registry.get("frozen_experiment_count", len(experiments)))
    if len(experiments) != expected_count:
        raise ValueError(
            f"{registry_path}: registry declares {expected_count} experiments but contains {len(experiments)}"
        )
    return registry_path, registry, experiments


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _selected_experiments(
    requested: list[str] | None, experiments: dict[str, dict[str, Any]]
) -> list[str]:
    if requested is None or requested == ["all"]:
        return list(experiments)
    values = _deduplicate(requested)
    if "all" in values:
        raise ValueError("--experiments all cannot be combined with explicit experiment ids")
    unknown = [value for value in values if value not in experiments]
    if unknown:
        raise ValueError(f"unknown --experiments values: {unknown}; valid ids: {list(experiments)}")
    return values


def _selected_scenes(requested: list[str], registry: dict[str, Any]) -> list[str]:
    values = ["baseline" if value == "base" else value for value in _deduplicate(requested)]
    registry_scenes = registry.get("shared_protocol", {}).get("scenario_ids", list(DEFAULT_SCENES))
    valid = {str(value) for value in registry_scenes}
    unknown = [value for value in values if value not in valid]
    if unknown:
        raise ValueError(f"unknown --scenes values: {unknown}; valid ids: {sorted(valid)}")
    return values


def _selected_cameras(requested: list[str], registry: dict[str, Any]) -> list[str]:
    values = _deduplicate(requested)
    registry_cameras = registry.get("shared_protocol", {}).get("camera_names", list(DEFAULT_CAMERAS))
    valid = {str(value) for value in registry_cameras}
    unknown = [value for value in values if value not in valid]
    if unknown:
        raise ValueError(f"unknown --cameras values: {unknown}; valid names: {sorted(valid)}")
    return values


def _unique_match(paths: list[Path], description: str) -> Path:
    paths = sorted(path.resolve() for path in paths if path.is_file())
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one {description}, found {len(paths)}: "
            + json.dumps([str(path) for path in paths], ensure_ascii=False)
        )
    return paths[0]


def _resolve_source_blend(
    workspace_root: Path, experiment: dict[str, Any], scene_id: str
) -> tuple[Path, str]:
    canonical_text = experiment.get("canonical_blend")
    if not isinstance(canonical_text, str) or not canonical_text:
        raise ValueError(f"experiment {experiment.get('id')}: missing canonical_blend")
    canonical = (workspace_root / Path(canonical_text)).resolve()
    generated_dir = canonical.parent / "generated_scenarios"

    if scene_id == "baseline":
        generated_matches = (
            list(generated_dir.glob("*_base.blend")) if generated_dir.is_dir() else []
        )
        if len(generated_matches) > 1:
            _unique_match(
                generated_matches,
                f"generated baseline blend for experiment {experiment.get('id')}",
            )
        if len(generated_matches) == 1:
            return generated_matches[0].resolve(), "generated_scenarios_base"
        if not canonical.is_file():
            raise FileNotFoundError(
                f"experiment {experiment.get('id')}: no *_base.blend and missing canonical {canonical}"
            )
        return canonical, "registry_canonical_blend"

    if not generated_dir.is_dir():
        raise FileNotFoundError(
            f"experiment {experiment.get('id')}: missing generated_scenarios directory {generated_dir}"
        )
    match = _unique_match(
        list(generated_dir.glob(f"*_{scene_id}.blend")),
        f"*_{scene_id}.blend for experiment {experiment.get('id')}",
    )
    return match, "generated_scenarios_scene"


def _candidate_dimensions(obj: Any, depsgraph: Any) -> list[float]:
    evaluated = obj.evaluated_get(depsgraph)
    return [abs(float(value)) for value in evaluated.dimensions]


def _find_standard_ball_mesh(depsgraph: Any) -> tuple[Any, list[dict[str, Any]]]:
    import bpy  # type: ignore

    candidates: list[tuple[float, str, Any, dict[str, Any]]] = []
    named_meshes = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        chain = _object_chain(obj)
        matching_chain = [ancestor.name for ancestor in chain if _contains_standard_ball(ancestor.name)]
        if not matching_chain:
            continue
        named_meshes += 1
        dimensions = _candidate_dimensions(obj, depsgraph)
        maximum = max(dimensions, default=0.0)
        minimum = min(dimensions, default=0.0)
        axis_ratio = minimum / maximum if maximum > 0.0 else 0.0
        bbox_volume = math.prod(dimensions) if minimum > 0.0 else 0.0
        diagnostic = {
            "mesh_object": obj.name,
            "matching_name_or_ancestors": matching_chain,
            "world_dimensions_m": dimensions,
            "min_to_max_axis_ratio": axis_ratio,
            "bounding_box_volume_m3": bbox_volume,
            "near_spherical": axis_ratio >= NEAR_SPHERE_MIN_AXIS_RATIO,
        }
        if diagnostic["near_spherical"] and bbox_volume > 0.0:
            candidates.append((bbox_volume, obj.name, obj, diagnostic))

    if not candidates:
        raise RuntimeError(
            "could not find a near-spherical MESH whose name or ancestor contains "
            f"standard_ball (named mesh candidates={named_meshes}, "
            f"minimum axis ratio={NEAR_SPHERE_MIN_AXIS_RATIO})"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[0][2]
    diagnostics = [item[3] for item in candidates]
    return selected, diagnostics


def _evaluated_world_vertices(obj: Any, depsgraph: Any) -> list[Any]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    if mesh is None:
        raise RuntimeError(f"could not evaluate mesh {obj.name}")
    try:
        return [evaluated.matrix_world @ vertex.co.copy() for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


def _ball_geometry(
    ball_mesh: Any, depsgraph: Any, candidate_diagnostics: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[Any]]:
    from mathutils import Vector  # type: ignore

    vertices = _evaluated_world_vertices(ball_mesh, depsgraph)
    if not vertices:
        raise RuntimeError(f"selected standard-ball mesh {ball_mesh.name} has no vertices")
    minimum = Vector(tuple(min(vertex[axis] for vertex in vertices) for axis in range(3)))
    maximum = Vector(tuple(max(vertex[axis] for vertex in vertices) for axis in range(3)))
    center = 0.5 * (minimum + maximum)
    dimensions = maximum - minimum
    radial_distances = sorted(float((vertex - center).length) for vertex in vertices)
    radius = float(statistics.median(radial_distances))
    if not math.isfinite(radius) or radius <= 0.0:
        raise RuntimeError(f"selected standard-ball mesh {ball_mesh.name} has invalid radius {radius}")

    chain = _object_chain(ball_mesh)
    standard_ball_ancestor = next(
        (ancestor for ancestor in chain if _contains_standard_ball(ancestor.name)), None
    )
    pendulum_ancestor = next(
        (ancestor for ancestor in chain[1:] if "pendulum" in _normalised_name(ancestor.name)),
        None,
    )
    pendulum = None
    if pendulum_ancestor is not None:
        pendulum = {
            "ancestor_object": pendulum_ancestor.name,
            "pivot_world_m": _vector(pendulum_ancestor.matrix_world.translation),
            "matrix_world_blender": _rows(pendulum_ancestor.matrix_world),
            "custom_properties": _custom_properties(pendulum_ancestor),
        }

    return (
        {
            "object_id": "standard_ball",
            "selected_mesh_object": ball_mesh.name,
            "standard_ball_ancestor": (
                standard_ball_ancestor.name if standard_ball_ancestor is not None else None
            ),
            "ancestor_chain": [ancestor.name for ancestor in chain],
            "mesh_object_origin_world_m": _vector(ball_mesh.matrix_world.translation),
            "center_world_m": _vector(center),
            "radius_m": radius,
            "diameter_m": 2.0 * radius,
            "mesh_world_bbox_min_m": _vector(minimum),
            "mesh_world_bbox_max_m": _vector(maximum),
            "mesh_world_dimensions_m": _vector(dimensions),
            "radial_distance_min_m": min(radial_distances),
            "radial_distance_max_m": max(radial_distances),
            "radial_distance_median_m": radius,
            "pendulum_ancestor_pivot": pendulum,
            "selection_policy": {
                "requires_name_or_ancestor_containing": "standard_ball",
                "minimum_near_sphere_axis_ratio": NEAR_SPHERE_MIN_AXIS_RATIO,
                "ranking": "largest_world_axis_aligned_bounding_box_volume_then_name",
                "eligible_candidates": candidate_diagnostics,
            },
        },
        vertices,
    )


def _bbox(points: list[list[float]]) -> list[float]:
    if not points:
        raise ValueError("cannot calculate a bbox from no points")
    return [
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    ]


def _camera_record(
    *,
    experiment: dict[str, Any],
    scene_id: str,
    source_blend: Path,
    source_resolution_origin: str,
    source_hash: str,
    camera_name: str,
    width: int,
    height: int,
    original_render: dict[str, Any],
    ball: dict[str, Any],
    ball_vertices: list[Any],
    scene_properties: dict[str, Any],
) -> dict[str, Any]:
    import bpy  # type: ignore
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore
    from mathutils import Vector  # type: ignore

    scene = bpy.context.scene
    camera_original = bpy.data.objects.get(camera_name)
    if camera_original is None or camera_original.type != "CAMERA":
        raise RuntimeError(f"{source_blend.name}: missing camera object {camera_name}")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera = camera_original.evaluated_get(depsgraph)

    intrinsics = _camera_intrinsics(camera.data, scene, width, height)
    opencv = _opencv_extrinsics(camera)
    rotation = opencv["R_world_to_camera"]
    translation = opencv["t_world_to_camera_m"]
    extrinsic_3x4 = [rotation[row] + [translation[row]] for row in range(3)]
    projection = _matmul(intrinsics["matrix"], extrinsic_3x4)

    center_world = ball["center_world_m"]
    center_world_vector = Vector(center_world)
    center_opencv_uv, center_depth = _project(
        intrinsics["matrix"], rotation, translation, center_world
    )
    center_ndc = world_to_camera_view(scene, camera_original, center_world_vector)
    center_blender_uv = [float(center_ndc.x * width), float((1.0 - center_ndc.y) * height)]
    center_projection_error = max(
        abs(left - right) for left, right in zip(center_opencv_uv, center_blender_uv)
    )
    if center_projection_error > 1e-3:
        raise RuntimeError(
            f"{source_blend.name}/{camera_name}: Blender/OpenCV center projection mismatch "
            f"{center_projection_error:.6g}px"
        )

    opencv_vertices: list[list[float]] = []
    vertex_depths: list[float] = []
    blender_vertices: list[list[float]] = []
    for vertex in ball_vertices:
        uv, depth = _project(intrinsics["matrix"], rotation, translation, vertex)
        opencv_vertices.append(uv)
        vertex_depths.append(depth)
        ndc = world_to_camera_view(scene, camera_original, vertex)
        blender_vertices.append([float(ndc.x * width), float((1.0 - ndc.y) * height)])
    opencv_bbox = _bbox(opencv_vertices)
    blender_bbox = _bbox(blender_vertices)
    bbox_projection_error = max(
        abs(left - right) for left, right in zip(opencv_bbox, blender_bbox)
    )
    if bbox_projection_error > 1e-3:
        raise RuntimeError(
            f"{source_blend.name}/{camera_name}: Blender/OpenCV mesh bbox mismatch "
            f"{bbox_projection_error:.6g}px"
        )
    bbox_width = opencv_bbox[2] - opencv_bbox[0]
    bbox_height = opencv_bbox[3] - opencv_bbox[1]
    projected_radius = 0.25 * (abs(bbox_width) + abs(bbox_height))

    registry_fields = {
        key: _json_value(experiment[key])
        for key in (
            "id",
            "level",
            "title",
            "canonical_blend",
            "hidden_parameters",
            "known_parameters",
            "equations",
            "identifiability",
            "observation_policy",
        )
        if key in experiment
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "experiment_id": str(experiment["id"]),
        "scene_id": scene_id,
        "camera_name": camera_name,
        "frame": 1,
        "source": {
            "blend_path": str(source_blend),
            "blend_filename": source_blend.name,
            "blend_resolution_origin": source_resolution_origin,
            "blend_sha256": source_hash,
            "opened_read_only": True,
            "save_operator_invoked": False,
        },
        "source_image": {
            "width_px": width,
            "height_px": height,
            "origin": "top_left",
            "u_axis": "right",
            "v_axis": "down",
            "pixel_coordinate_convention": "OpenCV continuous pixel coordinates",
            "blender_stored_render_settings": original_render,
        },
        "camera": {
            "model": "pinhole",
            "blender_type": str(camera.data.type),
            "lens_mm": float(camera.data.lens),
            "sensor_width_mm": float(camera.data.sensor_width),
            "sensor_height_mm": float(camera.data.sensor_height),
            "sensor_fit": str(camera.data.sensor_fit),
            "shift_x": float(camera.data.shift_x),
            "shift_y": float(camera.data.shift_y),
            "clip_start_m": float(camera.data.clip_start),
            "clip_end_m": float(camera.data.clip_end),
            "K": intrinsics["matrix"],
            "intrinsic_parameters": {
                key: value for key, value in intrinsics.items() if key != "matrix"
            },
            "matrix_world_blender": _rows(camera.matrix_world),
            "T_world_from_camera_blender": _rows(camera.matrix_world),
            "T_camera_from_world_blender": _rows(camera.matrix_world.inverted()),
            "R_world_to_camera_opencv": rotation,
            "t_world_to_camera_opencv_m": translation,
            "T_world_to_camera_opencv": opencv["T_world_to_camera"],
            "P_world_to_image_opencv": projection,
            "custom_properties": _custom_properties(camera_original),
        },
        "standard_ball": ball,
        "frame1_projection": {
            "center_uv_px": center_opencv_uv,
            "center_camera_depth_m": center_depth,
            "mesh_vertex_bbox_xyxy_px": opencv_bbox,
            "mesh_vertex_bbox_width_px": bbox_width,
            "mesh_vertex_bbox_height_px": bbox_height,
            "mesh_vertex_projected_radius_px": projected_radius,
            "all_mesh_vertices_in_front_of_camera": all(depth > 0.0 for depth in vertex_depths),
            "minimum_mesh_vertex_camera_depth_m": min(vertex_depths),
            "maximum_mesh_vertex_camera_depth_m": max(vertex_depths),
        },
        "projection_validation": {
            "center_uv_from_blender": center_blender_uv,
            "center_uv_from_opencv_P": center_opencv_uv,
            "center_max_abs_error_px": center_projection_error,
            "mesh_bbox_from_blender_xyxy_px": blender_bbox,
            "mesh_bbox_from_opencv_P_xyxy_px": opencv_bbox,
            "mesh_bbox_max_abs_error_px": bbox_projection_error,
        },
        "scene": {
            "frame_current": int(scene.frame_current),
            "frame_start": int(scene.frame_start),
            "frame_end": int(scene.frame_end),
            "fps": float(scene.render.fps),
            "fps_base": float(scene.render.fps_base),
            "effective_fps": float(scene.render.fps) / float(scene.render.fps_base),
            "gravity_world_mps2": _vector(scene.gravity),
            "unit_system": str(scene.unit_settings.system),
            "unit_scale_length": float(scene.unit_settings.scale_length),
            "metric_assumption": "benchmark contract treats one Blender unit as one meter",
            "custom_properties": scene_properties,
        },
        "registry_experiment": registry_fields,
        "conventions": {
            "matrix_storage": "row_major JSON arrays; transforms multiply column homogeneous vectors",
            "blender_camera_axes": "+X right, +Y up, camera looks along local -Z",
            "opencv_camera_axes": "+X right, +Y down, +Z forward",
            "opencv_axis_conversion_from_blender_camera": "diag(1,-1,-1)",
            "projection_equation": (
                "s*[u,v,1]^T = K * [R_world_to_camera_opencv | "
                "t_world_to_camera_opencv_m] * [X,Y,Z,1]^T"
            ),
            "generated_video_note": (
                "K and P use source_image pixel coordinates. Compose the actual "
                "source-to-generated crop/resize transform before lifting generated-video pixels."
            ),
        },
    }


def _original_render_settings(scene: Any) -> dict[str, Any]:
    percentage = float(scene.render.resolution_percentage)
    width = int(scene.render.resolution_x)
    height = int(scene.render.resolution_y)
    return {
        "resolution_x_px": width,
        "resolution_y_px": height,
        "resolution_percentage": percentage,
        "effective_width_px": width * percentage / 100.0,
        "effective_height_px": height * percentage / 100.0,
        "pixel_aspect_x": float(scene.render.pixel_aspect_x),
        "pixel_aspect_y": float(scene.render.pixel_aspect_y),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _portable_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv if argv is None else argv
    args = _parser().parse_args(_script_args(raw_argv))
    import bpy  # type: ignore

    workspace_root = args.workspace_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")

    registry_path, registry, experiments = _load_registry(workspace_root)
    experiment_ids = _selected_experiments(args.experiments, experiments)
    scene_ids = _selected_scenes(args.scenes, registry)
    camera_names = _selected_cameras(args.cameras, registry)

    source_specs: dict[tuple[str, str], dict[str, Any]] = {}
    for experiment_id in experiment_ids:
        experiment = experiments[experiment_id]
        for scene_id in scene_ids:
            path, origin = _resolve_source_blend(workspace_root, experiment, scene_id)
            source_specs[(experiment_id, scene_id)] = {
                "path": path,
                "origin": origin,
            }

    planned_paths = [output_dir / "manifest.json"]
    for experiment_id in experiment_ids:
        for scene_id in scene_ids:
            for camera_name in camera_names:
                planned_paths.append(
                    output_dir / experiment_id / scene_id / f"{camera_name}.json"
                )
    existing = [str(path) for path in planned_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "exporter-owned output files already exist; choose a new --output-dir or pass "
            "--overwrite: "
            + json.dumps(existing[:20], ensure_ascii=False)
        )

    source_hashes_before = {
        key: _sha256(spec["path"]) for key, spec in source_specs.items()
    }
    pending_records: list[tuple[Path, dict[str, Any]]] = []
    manifest_records: list[dict[str, Any]] = []

    for experiment_id in experiment_ids:
        experiment = experiments[experiment_id]
        for scene_id in scene_ids:
            spec = source_specs[(experiment_id, scene_id)]
            source_blend: Path = spec["path"]
            bpy.ops.wm.open_mainfile(filepath=str(source_blend), load_ui=False)
            scene = bpy.context.scene
            original_render = _original_render_settings(scene)
            scene.frame_set(1)
            # These in-memory settings affect only projection calculation.  The
            # opened source is never saved, and the next source is freshly opened.
            scene.render.resolution_x = int(args.width)
            scene.render.resolution_y = int(args.height)
            scene.render.resolution_percentage = 100
            bpy.context.view_layer.update()

            depsgraph = bpy.context.evaluated_depsgraph_get()
            ball_mesh, candidate_diagnostics = _find_standard_ball_mesh(depsgraph)
            ball, ball_vertices = _ball_geometry(
                ball_mesh, depsgraph, candidate_diagnostics
            )
            scene_properties = _custom_properties(scene)

            for camera_name in camera_names:
                record = _camera_record(
                    experiment=experiment,
                    scene_id=scene_id,
                    source_blend=source_blend,
                    source_resolution_origin=str(spec["origin"]),
                    source_hash=source_hashes_before[(experiment_id, scene_id)],
                    camera_name=camera_name,
                    width=int(args.width),
                    height=int(args.height),
                    original_render=original_render,
                    ball=ball,
                    ball_vertices=ball_vertices,
                    scene_properties=scene_properties,
                )
                record["source"]["blend_path"] = _portable_path(source_blend, workspace_root)
                relative_path = (
                    Path(experiment_id) / scene_id / f"{camera_name}.json"
                )
                pending_records.append((output_dir / relative_path, record))
                manifest_records.append(
                    {
                        "experiment_id": experiment_id,
                        "scene_id": scene_id,
                        "camera_name": camera_name,
                        "path": relative_path.as_posix(),
                        "source_blend": _portable_path(source_blend, workspace_root),
                        "source_blend_resolution_origin": str(spec["origin"]),
                        "source_blend_sha256": source_hashes_before[
                            (experiment_id, scene_id)
                        ],
                        "standard_ball_mesh": ball["selected_mesh_object"],
                        "standard_ball_radius_m": ball["radius_m"],
                    }
                )

    source_hashes_after = {
        key: _sha256(spec["path"]) for key, spec in source_specs.items()
    }
    changed = {
        f"{key[0]}/{key[1]}": {
            "path": str(source_specs[key]["path"]),
            "before": source_hashes_before[key],
            "after": source_hashes_after[key],
        }
        for key in source_specs
        if source_hashes_before[key] != source_hashes_after[key]
    }
    if changed:
        raise RuntimeError(
            "one or more source .blend hashes changed during read-only export: "
            + json.dumps(changed, ensure_ascii=False)
        )

    for path, record in pending_records:
        _write_json(path, record)

    manifest_sources = [
        {
            "experiment_id": experiment_id,
            "scene_id": scene_id,
            "blend_path": _portable_path(
                source_specs[(experiment_id, scene_id)]["path"], workspace_root
            ),
            "resolution_origin": str(source_specs[(experiment_id, scene_id)]["origin"]),
            "sha256_before": source_hashes_before[(experiment_id, scene_id)],
            "sha256_after": source_hashes_after[(experiment_id, scene_id)],
            "unchanged": True,
        }
        for experiment_id in experiment_ids
        for scene_id in scene_ids
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "workspace_root": ".",
        "output_dir": _portable_path(output_dir, workspace_root),
        "registry_path": _portable_path(registry_path, workspace_root),
        "registry_sha256": _sha256(registry_path),
        "registry_release_id": registry.get("release_id"),
        "registry_frozen_experiment_count": registry.get("frozen_experiment_count"),
        "blender_version": bpy.app.version_string,
        "frame": 1,
        "source_image_size_px": [int(args.width), int(args.height)],
        "experiments": experiment_ids,
        "scenes": scene_ids,
        "cameras": camera_names,
        "source_count": len(source_specs),
        "record_count": len(manifest_records),
        "source_blends_unchanged": True,
        "source_blends": manifest_sources,
        "read_only_contract": {
            "source_blends_opened_only": True,
            "save_operator_invoked": False,
            "hashes_verified_before_and_after": True,
            "writes_confined_to_output_dir": True,
        },
        "records": manifest_records,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
