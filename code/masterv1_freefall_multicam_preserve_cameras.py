
import bpy
import math
import os
import json
import csv
from mathutils import Vector, Euler

# ============================================================
# masterv1 freefall - camera-preserving, collection-aware version
# - keeps camera1/camera2/camera3 transforms and lens unchanged
# - uses existing table/cabinet collections to place the falling props
# - leaves all three experiments visible after execution
# - optional per-camera rendering without modifying camera parameters
# ============================================================

CFG = {
    "fps": 24,
    "seconds": 3.2,
    "render_engine": "CYCLES",
    "resolution_x": 1280,
    "resolution_y": 720,
    "cycles_samples": 96,
    "cycles_preview_samples": 32,
    "cycles_use_denoise": True,
    "cycles_device": "GPU",

    # Keep this False if you mainly want to inspect the animation in Layout.
    "render_video": False,
    "video_format": "AUTO",
    "ffmpeg_container": "MPEG4",
    "ffmpeg_codec": "H264",

    "output_dir": "//outputs_masterv1",
    "bench_root_collection": "LAB_BENCHMARKS_FREEFALL_REAL",

    # We keep the real cameras untouched and only switch scene.camera temporarily if rendering.
    "camera_targets": ["camera1", "camera2", "camera3"],

    # Collection hints. You can add your exact collection names here.
    "table_collection_candidates": [
        "table", "Table", "GRtable", "desk", "Desk", "TABLE", "DESK"
    ],
    "cabinet_collection_candidates": [
        "chest", "Chest", "cabinet", "Cabinet", "CABINET", "GRchest"
    ],

    # Optional object hints
    "table_proxy_name": "BallDrop_TableProxy",
    "cabinet_top_object_candidates": ["G-Cabinet_Top", "Cabinet_Top"],

    "freefall": {
        "g": 9.81,
        "v0": 0.0,
    },

    "collision_detection": {
        "enabled": True,
        "freeze_on_contact": True,
        "raycast_max_distance": 40.0,
        "raycast_epsilon": 0.002,
        "contact_offset": 0.0,
        "substeps": 12,
        "support_object_names": [],
        "upward_normal_min": 0.45,
        "debug_print": False,
    },
}


# ------------------------------------------------------------
# small utils
# ------------------------------------------------------------

def ensure_dir(path):
    abs_path = bpy.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def enum_identifiers(bl_rna_owner, prop_name):
    try:
        prop = bl_rna_owner.bl_rna.properties[prop_name]
        return {item.identifier for item in prop.enum_items}
    except Exception:
        return set()


def pick_render_engine(preferred="CYCLES"):
    prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
    supported = {item.identifier for item in prop.enum_items}
    if preferred in supported:
        return preferred
    for eng in ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH"]:
        if eng in supported:
            return eng
    raise RuntimeError(f"No supported render engine found: {sorted(supported)}")


def configure_output_format(scene, cfg, out_dir, exp_name):
    settings = scene.render.image_settings
    requested = str(cfg.get("video_format", "AUTO")).upper()
    file_format_ids = enum_identifiers(type(settings), "file_format")
    media_type_ids = enum_identifiers(type(settings), "media_type")

    if cfg.get("render_video", False):
        if "VIDEO" in media_type_ids:
            settings.media_type = "VIDEO"
            scene.render.filepath = os.path.join(out_dir, f"{exp_name}.mp4")
            if hasattr(scene.render, "ffmpeg"):
                try:
                    scene.render.ffmpeg.format = cfg.get("ffmpeg_container", "MPEG4")
                    scene.render.ffmpeg.codec = cfg.get("ffmpeg_codec", "H264")
                except Exception:
                    pass
            return scene.render.filepath

        if requested in {"AUTO", "VIDEO", "FFMPEG"} and "FFMPEG" in file_format_ids:
            settings.file_format = "FFMPEG"
            scene.render.filepath = os.path.join(out_dir, f"{exp_name}.mp4")
            if hasattr(scene.render, "ffmpeg"):
                try:
                    scene.render.ffmpeg.format = cfg.get("ffmpeg_container", "MPEG4")
                    scene.render.ffmpeg.codec = cfg.get("ffmpeg_codec", "H264")
                except Exception:
                    pass
            return scene.render.filepath

    if "PNG" in file_format_ids:
        settings.file_format = "PNG"
    scene.render.filepath = os.path.join(out_dir, f"{exp_name}_")
    return scene.render.filepath


def set_render_settings(cfg, exp_name):
    scn = bpy.context.scene
    scn.render.engine = pick_render_engine(cfg.get("render_engine", "CYCLES"))
    scn.render.resolution_x = cfg["resolution_x"]
    scn.render.resolution_y = cfg["resolution_y"]
    scn.render.fps = cfg["fps"]
    scn.frame_start = 1
    scn.frame_end = int(round(cfg["seconds"] * cfg["fps"]))

    if scn.render.engine == 'CYCLES':
        try:
            scn.cycles.samples = int(cfg.get('cycles_samples', 96))
            scn.cycles.preview_samples = int(cfg.get('cycles_preview_samples', 32))
            scn.cycles.use_denoising = bool(cfg.get('cycles_use_denoise', True))
            req = str(cfg.get('cycles_device', 'GPU')).upper()
            if req in {'GPU', 'CPU'}:
                scn.cycles.device = req
        except Exception:
            pass

    out_dir = ensure_dir(os.path.join(cfg["output_dir"], exp_name))
    configure_output_format(scn, cfg, out_dir, exp_name)
    return out_dir


def get_or_create_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        if parent is None:
            bpy.context.scene.collection.children.link(col)
        else:
            parent.children.link(col)
    return col


def clear_collection_objects(col):
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def link_object(obj, col):
    if col is not None and obj.name not in col.objects:
        col.objects.link(obj)
    for c in list(obj.users_collection):
        if col is not None and c != col:
            try:
                c.objects.unlink(obj)
            except RuntimeError:
                pass


def find_node_by_type(nodes, node_type):
    for n in nodes:
        if n.type == node_type:
            return n
    return None


def set_socket_default(node, socket_name, value):
    sock = node.inputs.get(socket_name)
    if sock is not None:
        sock.default_value = value


def ensure_principled_material(mat):
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    out = find_node_by_type(nodes, 'OUTPUT_MATERIAL')
    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)
    bsdf = find_node_by_type(nodes, 'BSDF_PRINCIPLED')
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
    surf = out.inputs.get("Surface")
    out_bsdf = bsdf.outputs.get("BSDF")
    if surf is not None and out_bsdf is not None:
        linked = False
        for lk in links:
            if lk.from_node == bsdf and lk.to_node == out and lk.to_socket == surf:
                linked = True
                break
        if not linked:
            for lk in list(links):
                if lk.to_node == out and lk.to_socket == surf:
                    links.remove(lk)
            links.new(out_bsdf, surf)
    return bsdf


def make_material(name, base_color=(1,1,1,1), metallic=0.0, roughness=0.5, transmission=0.0, ior=1.45):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    bsdf = ensure_principled_material(mat)
    set_socket_default(bsdf, "Base Color", base_color)
    set_socket_default(bsdf, "Metallic", metallic)
    set_socket_default(bsdf, "Roughness", roughness)
    if bsdf.inputs.get("Transmission Weight") is not None:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif bsdf.inputs.get("Transmission") is not None:
        bsdf.inputs["Transmission"].default_value = transmission
    set_socket_default(bsdf, "IOR", ior)
    return mat


def assign_material(obj, mat):
    if obj.data and hasattr(obj.data, "materials"):
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat


def parent_and_set_local_transform(obj, parent=None, location=(0,0,0), rotation=(0,0,0), scale=None):
    if parent is not None:
        obj.parent = parent
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation)
    if scale is not None:
        obj.scale = Vector(scale)
    return obj


def create_empty(name, location=(0,0,0), parent=None, col=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = 'PLAIN_AXES'
    if col is not None:
        link_object(obj, col)
    else:
        bpy.context.scene.collection.objects.link(obj)
    parent_and_set_local_transform(obj, parent=parent, location=location)
    return obj


def create_cube(name, size=1.0, location=(0,0,0), rotation=(0,0,0), scale=(1,1,1), parent=None, col=None):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0,0,0), rotation=(0,0,0))
    obj = bpy.context.active_object
    obj.name = name
    if col is not None:
        link_object(obj, col)
    parent_and_set_local_transform(obj, parent=parent, location=location, rotation=rotation, scale=scale)
    return obj


def create_cylinder(name, radius=0.05, depth=1.0, location=(0,0,0), rotation=(0,0,0), scale=(1,1,1), parent=None, col=None, vertices=48):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=(0,0,0), rotation=(0,0,0), vertices=vertices)
    obj = bpy.context.active_object
    obj.name = name
    if col is not None:
        link_object(obj, col)
    parent_and_set_local_transform(obj, parent=parent, location=location, rotation=rotation, scale=scale)
    return obj


def add_keyframe(obj, frame):
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def export_csv(csv_path, rows, header):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def export_json(json_path, data):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def frame_to_time(frame, fps):
    return (frame - 1) / fps


def world_xyz(obj):
    p = obj.matrix_world.translation
    return p.x, p.y, p.z


def bbox_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [p.x for p in corners]
    ys = [p.y for p in corners]
    zs = [p.z for p in corners]
    return {
        "min": Vector((min(xs), min(ys), min(zs))),
        "max": Vector((max(xs), max(ys), max(zs))),
        "center": Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)),
        "size": Vector((max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))),
    }


def get_collection_objects_recursive(col):
    if col is None:
        return []
    try:
        return list(col.all_objects)
    except Exception:
        objs = list(col.objects)
        for child in col.children:
            objs.extend(get_collection_objects_recursive(child))
        return objs


def union_bbox_world(objs):
    pts = []
    for obj in objs:
        try:
            if hasattr(obj, 'bound_box') and obj.type in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}:
                pts.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
            else:
                pts.append(obj.matrix_world.translation.copy())
        except Exception:
            pass
    if not pts:
        return None
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    return {
        "min": Vector((min(xs), min(ys), min(zs))),
        "max": Vector((max(xs), max(ys), max(zs))),
        "center": Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)),
        "size": Vector((max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))),
    }


def find_collection_by_candidates(candidates):
    for name in candidates:
        col = bpy.data.collections.get(name)
        if col is not None:
            return col
    lowered = [c.lower() for c in candidates]
    for col in bpy.data.collections:
        n = col.name.lower()
        if any(key.lower() == n for key in candidates):
            return col
    for col in bpy.data.collections:
        n = col.name.lower()
        if any(key.lower() in n for key in candidates):
            return col
    return None


def find_named_object(names, collection=None):
    objs = get_collection_objects_recursive(collection) if collection is not None else list(bpy.data.objects)
    by_name = {obj.name: obj for obj in objs}
    for name in names:
        if name in by_name:
            return by_name[name]
    for obj in objs:
        for name in names:
            if obj.name.startswith(name):
                return obj
    return None


def collection_anchor_from_bbox(collection):
    objs = [o for o in get_collection_objects_recursive(collection) if o.type in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}]
    bb = union_bbox_world(objs)
    if bb is None:
        return None
    return {
        "collection": collection,
        "collection_name": collection.name,
        "center": bb["center"],
        "size": bb["size"],
        "top_z": bb["max"].z,
        "left": bb["min"].x,
        "right": bb["max"].x,
        "front": bb["min"].y,
        "back": bb["max"].y,
        "object_names": [o.name for o in objs],
    }


def get_table_anchor(cfg):
    proxy = bpy.data.objects.get(cfg.get("table_proxy_name", ""))
    if proxy is not None:
        bb = bbox_world(proxy)
        return {
            "collection": None,
            "collection_name": None,
            "proxy_name": proxy.name,
            "center": bb["center"],
            "size": bb["size"],
            "top_z": bb["max"].z,
            "left": bb["min"].x,
            "right": bb["max"].x,
            "front": bb["min"].y,
            "back": bb["max"].y,
            "object_names": [],
        }
    col = find_collection_by_candidates(cfg["table_collection_candidates"])
    anchor = collection_anchor_from_bbox(col) if col else None
    if anchor is not None:
        anchor["proxy_name"] = None
        return anchor
    return {
        "collection": None,
        "collection_name": None,
        "proxy_name": None,
        "center": Vector((0.0, 0.0, 0.75)),
        "size": Vector((1.20, 0.70, 0.05)),
        "top_z": 0.75,
        "left": -0.60,
        "right": 0.60,
        "front": -0.35,
        "back": 0.35,
        "object_names": [],
    }


def get_cabinet_anchor(cfg):
    top_obj = find_named_object(cfg.get("cabinet_top_object_candidates", []))
    if top_obj is not None:
        bb = bbox_world(top_obj)
        return {
            "collection": None,
            "collection_name": None,
            "proxy_name": top_obj.name,
            "center": bb["center"],
            "size": bb["size"],
            "top_z": bb["max"].z,
            "left": bb["min"].x,
            "right": bb["max"].x,
            "front": bb["min"].y,
            "back": bb["max"].y,
            "object_names": [],
        }
    col = find_collection_by_candidates(cfg["cabinet_collection_candidates"])
    anchor = collection_anchor_from_bbox(col) if col else None
    if anchor is not None:
        anchor["proxy_name"] = None
        return anchor
    table = get_table_anchor(cfg)
    return {
        "collection": None,
        "collection_name": None,
        "proxy_name": None,
        "center": table["center"] + Vector((0.0, 0.30, 0.65)),
        "size": Vector((0.65, 0.35, 0.03)),
        "top_z": table["top_z"] + 0.65,
        "left": table["center"].x - 0.325,
        "right": table["center"].x + 0.325,
        "front": table["center"].y - 0.175,
        "back": table["center"].y + 0.175,
        "object_names": [],
    }


def get_depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def raycast_down_filtered(origin_world, ignore_objects=None, allowed_object_names=None, max_distance=40.0, epsilon=0.002, upward_normal_min=None):
    scene = bpy.context.scene
    depsgraph = get_depsgraph()
    ignore_names = {obj.name for obj in (ignore_objects or []) if obj is not None}
    allowed_names = set(allowed_object_names or [])
    direction = Vector((0.0, 0.0, -1.0))
    current_origin = Vector(origin_world)
    remaining = max_distance

    while remaining > 0.0:
        hit, loc, normal, face_index, obj, matrix = scene.ray_cast(depsgraph, current_origin, direction, distance=remaining)
        if not hit:
            return None
        travelled = (current_origin - loc).length
        if obj is None:
            return None

        accept = True
        if obj.name in ignore_names:
            accept = False
        if accept and allowed_names and obj.name not in allowed_names:
            accept = False
        if accept and upward_normal_min is not None and Vector(normal).z < upward_normal_min:
            accept = False

        if accept:
            return {
                "location": Vector(loc),
                "normal": Vector(normal),
                "object": obj,
                "distance": travelled,
            }

        step = max(travelled + epsilon, epsilon)
        current_origin = current_origin + direction * step
        remaining -= step
    return None


def query_support_contact_local_z(root, sample_local_pos, probe_local_z, radius, ignore_objects=None, fallback_floor_local_z=None, collision_cfg=None):
    collision_cfg = collision_cfg or {}
    max_distance = collision_cfg.get("raycast_max_distance", 40.0)
    eps = collision_cfg.get("raycast_epsilon", 0.002)
    contact_offset = collision_cfg.get("contact_offset", 0.0)
    allowed_names = collision_cfg.get("support_object_names", [])
    upward_normal_min = collision_cfg.get("upward_normal_min", 0.45)

    sample_local_pos = Vector(sample_local_pos)
    probe_local = Vector((sample_local_pos.x, sample_local_pos.y, probe_local_z))
    probe_world_center = root.matrix_world @ probe_local
    ray_origin = probe_world_center + Vector((0.0, 0.0, radius + eps))

    hit = raycast_down_filtered(
        ray_origin,
        ignore_objects=ignore_objects,
        allowed_object_names=allowed_names,
        max_distance=max_distance,
        epsilon=eps,
        upward_normal_min=upward_normal_min,
    )
    if hit is not None:
        contact_world_z = hit["location"].z + radius + contact_offset
        contact_world_center = Vector((probe_world_center.x, probe_world_center.y, contact_world_z))
        contact_local_center = root.matrix_world.inverted() @ contact_world_center
        return {
            "local_z": contact_local_center.z,
            "world_z": contact_world_z,
            "hit_object_name": hit["object"].name,
            "normal_z": hit["normal"].z,
        }

    if fallback_floor_local_z is not None:
        fallback_world_contact = root.matrix_world @ Vector((sample_local_pos.x, sample_local_pos.y, fallback_floor_local_z + radius))
        fallback_world_contact_z = fallback_world_contact.z
        contact_world_center = Vector((probe_world_center.x, probe_world_center.y, fallback_world_contact_z))
        contact_local_center = root.matrix_world.inverted() @ contact_world_center
        return {
            "local_z": contact_local_center.z,
            "world_z": fallback_world_contact_z,
            "hit_object_name": "fallback_floor",
            "normal_z": 1.0,
        }
    return None


def detect_segment_crossing(root, prev_local_pos, curr_local_pos, radius, ignore_objects=None, fallback_floor_local_z=None, collision_cfg=None):
    prev_local_pos = Vector(prev_local_pos)
    curr_local_pos = Vector(curr_local_pos)
    probe_local_z = max(prev_local_pos.z, curr_local_pos.z)
    contact = query_support_contact_local_z(
        root=root,
        sample_local_pos=curr_local_pos,
        probe_local_z=probe_local_z,
        radius=radius,
        ignore_objects=ignore_objects,
        fallback_floor_local_z=fallback_floor_local_z,
        collision_cfg=collision_cfg,
    )
    if contact is None:
        return None

    contact_local_z = contact["local_z"]
    if prev_local_pos.z > contact_local_z and curr_local_pos.z <= contact_local_z:
        denom = prev_local_pos.z - curr_local_pos.z
        frac = 1.0 if abs(denom) < 1e-12 else (prev_local_pos.z - contact_local_z) / denom
        frac = max(0.0, min(1.0, frac))
        hit_local_pos = prev_local_pos.lerp(curr_local_pos, frac)
        hit_local_pos.z = contact_local_z
        return {**contact, "frac": frac, "local_pos": hit_local_pos}
    return None


# ------------------------------------------------------------
# scene-specific helpers
# ------------------------------------------------------------

def pick_camera_by_name(name):
    obj = bpy.data.objects.get(name)
    return obj if (obj is not None and obj.type == 'CAMERA') else None


def add_or_update_marker(frame, name, camera=None):
    scene = bpy.context.scene
    marker = None
    for m in scene.timeline_markers:
        if m.name == name:
            marker = m
            break
    if marker is None:
        marker = scene.timeline_markers.new(name=name, frame=frame)
    marker.frame = frame
    if camera is not None:
        try:
            marker.camera = camera
        except Exception:
            pass
    return marker


def prepare_experiment_collection(exp_name, cfg):
    root_col = get_or_create_collection(cfg["bench_root_collection"])
    exp_col = get_or_create_collection(exp_name, parent=root_col)
    clear_collection_objects(exp_col)
    try:
        exp_col.hide_render = False
        exp_col.hide_viewport = False
    except Exception:
        pass
    return exp_col


# ------------------------------------------------------------
# prop builders
# ------------------------------------------------------------

def apply_bevel(obj, width=0.0025, segments=2):
    try:
        mod = obj.modifiers.new(name="Bevel", type='BEVEL')
        mod.width = width
        mod.segments = segments
    except Exception:
        pass
    return obj


def build_eraser_item(parent, col, h0):
    mats = {
        "body": make_material("FFR_Eraser_Body", (0.93, 0.61, 0.70, 1.0), roughness=0.88),
        "body2": make_material("FFR_Eraser_Body2", (0.95, 0.82, 0.78, 1.0), roughness=0.90),
        "sleeve": make_material("FFR_Eraser_Sleeve", (0.17, 0.36, 0.72, 1.0), roughness=0.55),
        "paper": make_material("FFR_Eraser_Paper", (0.96, 0.94, 0.88, 1.0), roughness=0.92),
    }
    rig = create_empty("FF_ItemRig", location=(0.0, 0.0, h0), parent=parent, col=col)
    rig.rotation_euler = Euler((math.radians(8.0), math.radians(12.0), math.radians(6.0)))

    body1 = create_cube("FF_EraserBodyA", size=1.0, location=(0.0, 0.0, 0.0), scale=(0.028, 0.016, 0.010), parent=rig, col=col)
    assign_material(body1, mats["body"])
    apply_bevel(body1, 0.0022, 3)

    body2 = create_cube("FF_EraserBodyB", size=1.0, location=(0.010, 0.0, 0.0), scale=(0.018, 0.016, 0.010), parent=rig, col=col)
    assign_material(body2, mats["body2"])
    apply_bevel(body2, 0.0022, 3)

    sleeve = create_cube("FF_EraserSleeve", size=1.0, location=(0.0, 0.0, -0.001), scale=(0.018, 0.0172, 0.0106), parent=rig, col=col)
    assign_material(sleeve, mats["sleeve"])
    apply_bevel(sleeve, 0.0016, 2)

    paper = create_cube("FF_EraserPaper", size=1.0, location=(0.0, 0.0, 0.0108), scale=(0.014, 0.011, 0.0008), parent=rig, col=col)
    assign_material(paper, mats["paper"])

    return {"rig": rig, "parts": [body1, body2, sleeve, paper], "radius": 0.034}


def build_bottle_item(parent, col, h0):
    mats = {
        "body": make_material("FFR_Bottle_Body", (0.96, 0.96, 0.94, 1.0), roughness=0.70),
        "cap": make_material("FFR_Bottle_Cap", (0.10, 0.23, 0.60, 1.0), roughness=0.45),
        "label": make_material("FFR_Bottle_Label", (0.89, 0.80, 0.22, 1.0), roughness=0.55),
    }
    rig = create_empty("FF_ItemRig", location=(0.0, 0.0, h0), parent=parent, col=col)
    rig.rotation_euler = Euler((math.radians(2.0), math.radians(0.0), math.radians(5.0)))

    body = create_cylinder("FF_BottleBody", radius=0.019, depth=0.056, location=(0.0, 0.0, 0.0), parent=rig, col=col, vertices=48)
    assign_material(body, mats["body"])
    apply_bevel(body, 0.0018, 2)

    shoulder = create_cylinder("FF_BottleShoulder", radius=0.014, depth=0.010, location=(0.0, 0.0, 0.033), parent=rig, col=col, vertices=36)
    assign_material(shoulder, mats["body"])

    cap = create_cylinder("FF_BottleCap", radius=0.013, depth=0.016, location=(0.0, 0.0, 0.045), parent=rig, col=col, vertices=36)
    assign_material(cap, mats["cap"])

    label = create_cylinder("FF_BottleLabel", radius=0.0196, depth=0.020, location=(0.0, 0.0, -0.004), parent=rig, col=col, vertices=48)
    assign_material(label, mats["label"])

    return {"rig": rig, "parts": [body, shoulder, cap, label], "radius": 0.040}


def build_box_item(parent, col, h0):
    mats = {
        "cardboard": make_material("FFR_Box_Cardboard", (0.63, 0.45, 0.27, 1.0), roughness=0.92),
        "tape": make_material("FFR_Box_Tape", (0.77, 0.64, 0.42, 1.0), roughness=0.58),
        "label": make_material("FFR_Box_Label", (0.93, 0.92, 0.87, 1.0), roughness=0.95),
    }
    rig = create_empty("FF_ItemRig", location=(0.0, 0.0, h0), parent=parent, col=col)
    rig.rotation_euler = Euler((math.radians(3.0), math.radians(9.0), math.radians(0.0)))

    body = create_cube("FF_BoxBody", size=1.0, location=(0.0, 0.0, 0.0), scale=(0.045, 0.028, 0.026), parent=rig, col=col)
    assign_material(body, mats["cardboard"])
    apply_bevel(body, 0.0022, 2)

    tape = create_cube("FF_BoxTape", size=1.0, location=(0.0, 0.0, 0.0265), scale=(0.008, 0.029, 0.0012), parent=rig, col=col)
    assign_material(tape, mats["tape"])

    label = create_cube("FF_BoxLabel", size=1.0, location=(0.010, -0.0292, 0.0), scale=(0.012, 0.0007, 0.009), rotation=(math.radians(90.0), 0.0, 0.0), parent=rig, col=col)
    assign_material(label, mats["label"])

    return {"rig": rig, "parts": [body, tape, label], "radius": 0.056}


def build_freefall_item(builder_name, parent, col, h0):
    if builder_name == "eraser":
        return build_eraser_item(parent, col, h0)
    if builder_name == "bottle":
        return build_bottle_item(parent, col, h0)
    if builder_name == "box":
        return build_box_item(parent, col, h0)
    raise ValueError(f"Unknown freefall item builder: {builder_name}")


# ------------------------------------------------------------
# presets
# ------------------------------------------------------------

def get_freefall_camera_presets(cfg):
    table = get_table_anchor(cfg)
    cabinet = get_cabinet_anchor(cfg)

    # overhang pushes object center slightly beyond the support edge
    return {
        "camera1": {
            "exp_name": "FREEFALL_camera1_eraser",
            "builder": "eraser",
            "anchor_kind": "table",
            "anchor": table,
            "drop_height": 0.12,
            "root_loc": Vector((
                table["right"] + 0.012,
                table["center"].y - table["size"].y * 0.16,
                table["top_z"],
            )),
            "fallback_floor_world_z": 0.0,
        },
        "camera2": {
            "exp_name": "FREEFALL_camera2_bottle",
            "builder": "bottle",
            "anchor_kind": "table",
            "anchor": table,
            "drop_height": 0.13,
            "root_loc": Vector((
                table["center"].x + table["size"].x * 0.14,
                table["front"] - 0.016,
                table["top_z"],
            )),
            "fallback_floor_world_z": 0.0,
        },
        "camera3": {
            "exp_name": "FREEFALL_camera3_cabinet_box",
            "builder": "box",
            "anchor_kind": "cabinet",
            "anchor": cabinet,
            "drop_height": 0.15,
            "root_loc": Vector((
                cabinet["center"].x + cabinet["size"].x * 0.10,
                cabinet["front"] - 0.018,
                cabinet["top_z"],
            )),
            "fallback_floor_world_z": 0.0,
        },
    }


# ------------------------------------------------------------
# animation
# ------------------------------------------------------------

def animate_freefall_rig(cfg, exp_name, out_dir, root, rig, ignore_objects, radius, fallback_floor_local_z):
    p = cfg["freefall"]
    fps = cfg["fps"]
    rows = []
    collision_cfg = dict(cfg.get("collision_detection", {}))

    contacted = False
    frozen_local_z = None
    frozen_hit_name = None
    substeps = max(1, int(collision_cfg.get("substeps", 12)))
    prev_local_pos = Vector(rig.location)

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        t = frame_to_time(frame, fps)
        predicted_local_z = rig.location.z if frame == bpy.context.scene.frame_start else (p["h0"] + p["v0"] * t - 0.5 * p["g"] * t * t)
        hit_name = None

        if frame == bpy.context.scene.frame_start:
            z_local = p["h0"]
        elif contacted and frozen_local_z is not None:
            z_local = frozen_local_z
            hit_name = frozen_hit_name
        else:
            z_local = predicted_local_z
            if collision_cfg.get("enabled", True):
                prev_t = frame_to_time(frame - 1, fps)
                seg_prev = Vector(prev_local_pos)
                for s in range(1, substeps + 1):
                    alpha = s / substeps
                    seg_t = prev_t + (t - prev_t) * alpha
                    seg_curr = Vector((0.0, 0.0, p["h0"] + p["v0"] * seg_t - 0.5 * p["g"] * seg_t * seg_t))
                    seg_hit = detect_segment_crossing(
                        root=root,
                        prev_local_pos=seg_prev,
                        curr_local_pos=seg_curr,
                        radius=radius,
                        ignore_objects=ignore_objects,
                        fallback_floor_local_z=fallback_floor_local_z,
                        collision_cfg=collision_cfg,
                    )
                    if seg_hit is not None:
                        z_local = seg_hit["local_pos"].z
                        hit_name = seg_hit["hit_object_name"]
                        if collision_cfg.get("freeze_on_contact", True):
                            contacted = True
                            frozen_local_z = z_local
                            frozen_hit_name = hit_name
                        break
                    seg_prev = seg_curr

        rig.location = Vector((0.0, 0.0, z_local))
        add_keyframe(rig, frame)
        prev_local_pos = Vector(rig.location)

        wx, wy, wz = world_xyz(rig)
        rows.append([frame, round(t, 8), wx, wy, wz, 0.0, 0.0, z_local, int(contacted), hit_name or ""])

    export_csv(
        os.path.join(out_dir, f"{exp_name}_trajectory.csv"),
        rows,
        ["frame", "t", "world_x", "world_y", "world_z", "local_x", "local_y", "local_z", "contacted", "hit_object"],
    )


def run_freefall_preset(cfg, preset_name):
    presets = get_freefall_camera_presets(cfg)
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}; available={sorted(presets)}")

    preset = presets[preset_name]
    exp_name = preset["exp_name"]
    out_dir = set_render_settings(cfg, exp_name)
    exp_col = prepare_experiment_collection(exp_name, cfg)

    root_loc = preset["root_loc"]
    root = create_empty(f"FF_ROOT_{preset_name}", location=tuple(root_loc), col=exp_col)

    cfg["freefall"]["h0"] = float(preset["drop_height"])
    item = build_freefall_item(preset["builder"], root, exp_col, cfg["freefall"]["h0"])

    # Keep camera parameters unchanged. We only attach timeline markers and use scene.camera for rendering if needed.
    cam = pick_camera_by_name(preset_name)
    add_or_update_marker(1, f"{preset_name}_start", camera=cam)

    fallback_floor_local_z = preset["fallback_floor_world_z"] - root_loc.z
    ignore_objects = list(item["parts"]) + [item["rig"]]

    animate_freefall_rig(
        cfg=cfg,
        exp_name=exp_name,
        out_dir=out_dir,
        root=root,
        rig=item["rig"],
        ignore_objects=ignore_objects,
        radius=item["radius"],
        fallback_floor_local_z=fallback_floor_local_z,
    )

    export_json(
        os.path.join(out_dir, f"{exp_name}_params.json"),
        {
            "camera_name": preset_name,
            "item_type": preset["builder"],
            "anchor_kind": preset["anchor_kind"],
            "anchor_collection": preset["anchor"].get("collection_name"),
            "anchor_proxy": preset["anchor"].get("proxy_name"),
            "root_world": [float(v) for v in root_loc],
            "anchor_top_z": float(preset["anchor"]["top_z"]),
            "drop_height": float(preset["drop_height"]),
            "camera_preserved": True,
        },
    )
    return cam


def render_from_camera(cfg, exp_name, cam):
    if not cfg.get("render_video", False):
        return
    if cam is None:
        return
    scene = bpy.context.scene
    original_camera = scene.camera
    try:
        scene.camera = cam
        out_dir = set_render_settings(cfg, exp_name)
        scene.render.filepath = os.path.join(out_dir, f"{exp_name}.mp4")
        bpy.ops.render.render(animation=True)
    finally:
        scene.camera = original_camera


def main(cfg):
    scene = bpy.context.scene
    original_camera = scene.camera
    original_frame = scene.frame_current

    built = []
    for preset_name in cfg.get("camera_targets", ["camera1", "camera2", "camera3"]):
        scene.frame_set(1)
        cam = run_freefall_preset(cfg, preset_name)
        built.append((preset_name, cam))

    # Optional render pass per camera. Camera transforms stay unchanged.
    if cfg.get("render_video", False):
        for preset_name, cam in built:
            exp_name = get_freefall_camera_presets(cfg)[preset_name]["exp_name"]
            render_from_camera(cfg, exp_name, cam)

    # Leave the scene friendly for Layout inspection.
    try:
        if original_camera is not None:
            scene.camera = original_camera
    except Exception:
        pass
    scene.frame_set(1)

    print("[DONE] Built freefall presets and left all collections visible:")
    for preset_name, _ in built:
        print("  -", preset_name)


if __name__ == "__main__":
    main(CFG)
