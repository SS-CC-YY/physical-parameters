import bpy
import math
import os
import json
import csv
from mathutils import Vector, Euler

# ============================================================
# Blender V1 indoor-lab benchmark template
# Local-coordinate edition:
#   - Move only EXP_ROOT to reposition the entire experiment
#   - All experiment objects are built in local coordinates
#   - Motions are analytic / formula-driven
#   - Optional downward contact detection for falling objects
#   - Exports RGB render (optional) + trajectory CSV + params JSON
# Compatible with Blender versions whose render engine enum includes
# BLENDER_EEVEE / CYCLES / BLENDER_WORKBENCH.
# ============================================================

CFG = {
    "experiment": "freefall",   # freefall | projectile | spring | pendulum | rotation
    "fps": 24,
    "seconds": 5.0,
    "camera_name": "Camera",
    "render_engine": "CYCLES",  # default to CYCLES; auto-fallback if unavailable
    "resolution_x": 1280,
    "resolution_y": 720,
    "output_dir": "//outputs",

    # Fixed camera pose (used when camera object exists or will be created)
    "camera_location": (0.0, -2.0, 1.0),
    "camera_rotation_deg": (90.0, 0.0, 0.0),

    # Cycles defaults
    "cycles_samples": 128,
    "cycles_preview_samples": 64,
    "cycles_use_denoise": True,
    "cycles_device": "GPU",  # fallback to CPU if unavailable
    "render_video": False,
    "video_format": "AUTO",          # AUTO | VIDEO | FFMPEG | PNG
    "ffmpeg_container": "MPEG4",
    "ffmpeg_codec": "H264",

    # ---------------- Clean studio scene ----------------
    "use_clean_studio_scene": True,
    "studio_scene": {
        "background_mode": "CURVE",   # SOLID | CURVE
        "studio_root_name": "STUDIO_ROOT",
        "clear_default_startup_objects": True,
        "background_color": (0.60, 0.60, 0.60, 1.0),
        "floor_color": (0.60, 0.60, 0.60, 1.0),
        "backdrop_width": 14.0,
        "backdrop_depth": 14.0,
        "backdrop_height": 8.0,
        "backdrop_curve_radius": 2.20,
        "backdrop_y_shift": 4.0,
        "key_energy": 2200.0,
        "fill_energy": 1200.0,
        "rim_energy": 700.0,
        "top_energy": 1400.0,
        "area_size": 5.5,
        "top_area_size": 7.0,
        "world_strength": 0.10,
        "view_exposure": 0.25,
    },

    # Existing environment hooks
    "table_object_name": None,
    "env_collection_names": [],

    # Global experiment placement
    "exp_root_name": "EXP_ROOT",
    "lab_origin": (0.0, 0.0, 0.9),   # Move this to reposition the entire rig
    "glass_thickness": 0.008,
    "use_glass_enclosure": False,
    "use_exp_platform": False,

    # Materials / appearance
    "ball_radius": 0.06,
    "ball_color": (0.82, 0.12, 0.12, 1.0),
    "metal_color": (0.55, 0.57, 0.60, 1.0),
    "plastic_color": (0.10, 0.10, 0.12, 1.0),
    "accent_color": (0.95, 0.80, 0.15, 1.0),
    "base_color": (0.78, 0.78, 0.78, 1.0),

    # Contact detection for falling objects
    "collision_detection": {
        "enabled": True,
        "freeze_on_contact": True,
        "raycast_max_distance": 20.0,
        "raycast_epsilon": 0.002,
        "contact_offset": 0.0,
        "substeps": 8,                 # swept/contact substeps per rendered frame
        "support_object_names": [],    # e.g. ["TableTop"]
        "upward_normal_min": 0.5,      # only accept upward-facing support surfaces
        "debug_print": False,
    },

    # ---------------- Free fall ----------------
    "freefall": {
        "g": 9.81,
        "h0": 0.70,          # local Z relative to EXP_ROOT
        "v0": 0.0,
        "tube_height": 1.10,
        "tube_radius": 0.12,
        "stop_before_hit": True,
        "floor_y": None,     # fallback local Z of floor contact plane
    },

    # ---------------- Projectile ----------------
    "projectile": {
        "g": 9.81,
        "x0": 0.00,
        "y0": 0.38,          # local Z launch height
        "v0": 2.2,
        "theta_deg": 18.0,
        "box_length": 1.80,
        "box_height": 0.90,
        "box_depth": 0.35,
        "floor_y": None,     # fallback local Z of floor contact plane
        "stop_before_hit": True,
    },

    # ---------------- Spring ----------------
    "spring": {
        "A": 0.22,
        "k": 12.0,
        "m": 0.60,
        "phi_deg": 0.0,
        "rail_length": 1.20,
        "spring_rest": 0.35,
    },

    # ---------------- Pendulum ----------------
    "pendulum": {
        "g": 9.81,
        "L": 0.80,
        "Theta_deg": 8.0,
        "phi_deg": 0.0,
        "stand_height": 1.10,
    },

    # ---------------- Rotation ----------------
    "rotation": {
        "theta0_deg": 0.0,
        "omega0_deg_s": 90.0,
        "alpha_deg_s2": 0.0,
        "disc_radius": 0.16,
        "base_height": 0.12,
        "constant_accel": False,
    },
}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def ensure_dir(path):
    abs_path = bpy.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def pick_render_engine(preferred="BLENDER_EEVEE"):
    prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
    supported = {item.identifier for item in prop.enum_items}

    if preferred in supported:
        return preferred

    fallback_order = [
        "BLENDER_EEVEE",
        "BLENDER_EEVEE_NEXT",
        "CYCLES",
        "BLENDER_WORKBENCH",
    ]
    for eng in fallback_order:
        if eng in supported:
            print(f"[INFO] render engine '{preferred}' not supported, fallback to '{eng}'")
            return eng

    raise RuntimeError(f"No supported render engine found. Supported: {sorted(supported)}")


def enum_identifiers(bl_rna_owner, prop_name):
    try:
        prop = bl_rna_owner.bl_rna.properties[prop_name]
        return {item.identifier for item in prop.enum_items}
    except Exception:
        return set()


def configure_output_format(scene, cfg, out_dir):
    settings = scene.render.image_settings
    requested = str(cfg.get("video_format", "AUTO")).upper()
    file_format_ids = enum_identifiers(type(settings), "file_format")
    media_type_ids = enum_identifiers(type(settings), "media_type")

    exp_name = cfg["experiment"]

    # Newer Blender builds may separate image type from media type.
    if cfg.get("render_video", False):
        if "VIDEO" in media_type_ids:
            settings.media_type = "VIDEO"
            scene.render.filepath = os.path.join(out_dir, f"{exp_name}.mp4")
            if hasattr(scene.render, "ffmpeg"):
                try:
                    scene.render.ffmpeg.format = cfg.get("ffmpeg_container", "MPEG4")
                except Exception:
                    pass
                try:
                    scene.render.ffmpeg.codec = cfg.get("ffmpeg_codec", "H264")
                except Exception:
                    pass
            return {
                "mode": "VIDEO",
                "filepath": scene.render.filepath,
            }

        if requested in {"AUTO", "VIDEO", "FFMPEG"} and "FFMPEG" in file_format_ids:
            settings.file_format = "FFMPEG"
            scene.render.filepath = os.path.join(out_dir, f"{exp_name}.mp4")
            if hasattr(scene.render, "ffmpeg"):
                try:
                    scene.render.ffmpeg.format = cfg.get("ffmpeg_container", "MPEG4")
                except Exception:
                    pass
                try:
                    scene.render.ffmpeg.codec = cfg.get("ffmpeg_codec", "H264")
                except Exception:
                    pass
            return {
                "mode": "FFMPEG",
                "filepath": scene.render.filepath,
            }

        # Last fallback: render PNG sequence instead of crashing
        print("[WARN] Current Blender build does not expose VIDEO/FFMPEG output in Python. Fallback to PNG sequence.")
        if "PNG" in file_format_ids:
            settings.file_format = "PNG"
        scene.render.filepath = os.path.join(out_dir, f"{exp_name}_")
        return {
            "mode": "PNG_SEQUENCE",
            "filepath": scene.render.filepath,
        }

    # Non-video render path
    if requested == "PNG" and "PNG" in file_format_ids:
        settings.file_format = "PNG"
        scene.render.filepath = os.path.join(out_dir, f"{exp_name}_")
        return {
            "mode": "PNG_SEQUENCE",
            "filepath": scene.render.filepath,
        }

    if "PNG" in file_format_ids:
        settings.file_format = "PNG"
    scene.render.filepath = os.path.join(out_dir, f"{exp_name}_")
    return {
        "mode": "PNG_SEQUENCE",
        "filepath": scene.render.filepath,
    }


def set_cycles_settings(cfg):
    scn = bpy.context.scene
    if scn.render.engine != 'CYCLES':
        return

    try:
        scn.cycles.samples = int(cfg.get('cycles_samples', 128))
    except Exception:
        pass
    try:
        scn.cycles.preview_samples = int(cfg.get('cycles_preview_samples', 64))
    except Exception:
        pass
    try:
        scn.cycles.use_denoising = bool(cfg.get('cycles_use_denoise', True))
    except Exception:
        pass
    try:
        requested_device = str(cfg.get('cycles_device', 'GPU')).upper()
        if requested_device in {'GPU', 'CPU'}:
            scn.cycles.device = requested_device
    except Exception:
        pass


def set_render_settings(cfg):
    scn = bpy.context.scene
    scn.render.engine = pick_render_engine(cfg.get("render_engine", "BLENDER_EEVEE"))
    set_cycles_settings(cfg)
    scn.render.resolution_x = cfg["resolution_x"]
    scn.render.resolution_y = cfg["resolution_y"]
    scn.render.fps = cfg["fps"]
    scn.frame_start = 1
    scn.frame_end = int(round(cfg["seconds"] * cfg["fps"]))

    out_dir = ensure_dir(cfg["output_dir"])
    configure_output_format(scn, cfg, out_dir)
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
    objs = list(col.objects)
    for obj in objs:
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
    if nt is None:
        raise RuntimeError(f"Material '{mat.name}' has no node tree.")

    nodes = nt.nodes
    links = nt.links

    output = find_node_by_type(nodes, 'OUTPUT_MATERIAL')
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (300, 0)

    bsdf = find_node_by_type(nodes, 'BSDF_PRINCIPLED')
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)

    surface_in = output.inputs.get("Surface")
    bsdf_out = bsdf.outputs.get("BSDF")
    if surface_in is not None and bsdf_out is not None:
        linked = False
        for lk in links:
            if lk.from_node == bsdf and lk.to_node == output and lk.to_socket == surface_in:
                linked = True
                break
        if not linked:
            for lk in list(links):
                if lk.to_node == output and lk.to_socket == surface_in:
                    links.remove(lk)
            links.new(bsdf_out, surface_in)

    return bsdf


def make_material(name, base_color=(1, 1, 1, 1), metallic=0.0, roughness=0.5,
                  transmission=0.0, ior=1.45, alpha=1.0):
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
    set_socket_default(bsdf, "Alpha", alpha)

    if hasattr(mat, "blend_method"):
        mat.blend_method = 'BLEND' if (alpha < 1.0 or transmission > 0.0) else 'OPAQUE'
    if hasattr(mat, "shadow_method"):
        try:
            mat.shadow_method = 'HASHED'
        except Exception:
            pass

    return mat


def assign_material(obj, mat):
    if obj.data and hasattr(obj.data, "materials"):
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat


def parent_and_set_local_transform(obj, parent=None, location=(0, 0, 0), rotation=(0, 0, 0), scale=None):
    if parent is not None:
        obj.parent = parent
    obj.location = Vector(location)
    obj.rotation_euler = Euler(rotation)
    if scale is not None:
        obj.scale = Vector(scale)
    return obj


def create_empty(name, location=(0, 0, 0), parent=None, col=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = 'PLAIN_AXES'
    if col is not None:
        link_object(obj, col)
    else:
        bpy.context.scene.collection.objects.link(obj)
    parent_and_set_local_transform(obj, parent=parent, location=location)
    return obj


def create_cube(name, size=1.0, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), parent=None, col=None):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0), rotation=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    if col is not None:
        link_object(obj, col)
    parent_and_set_local_transform(obj, parent=parent, location=location, rotation=rotation, scale=scale)
    return obj


def create_uv_sphere(name, radius=0.05, location=(0, 0, 0), rotation=(0, 0, 0), parent=None, col=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0), segments=64, ring_count=32)
    obj = bpy.context.active_object
    obj.name = name
    if col is not None:
        link_object(obj, col)
    parent_and_set_local_transform(obj, parent=parent, location=location, rotation=rotation)
    return obj


def create_cylinder(name, radius=0.05, depth=1.0, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), parent=None, col=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=(0, 0, 0), rotation=(0, 0, 0), vertices=64)
    obj = bpy.context.active_object
    obj.name = name
    if col is not None:
        link_object(obj, col)
    parent_and_set_local_transform(obj, parent=parent, location=location, rotation=rotation, scale=scale)
    return obj


def add_wireframe_modifier(obj, thickness=0.01):
    mod = obj.modifiers.new(name="Wireframe", type='WIREFRAME')
    mod.thickness = thickness
    mod.use_replace = False
    return mod


def create_glass_box(name, dims=(1, 1, 1), location=(0, 0, 0), parent=None, col=None, thickness=0.01):
    box = create_cube(
        name,
        size=1.0,
        location=location,
        scale=(dims[0] / 2, dims[1] / 2, dims[2] / 2),
        parent=parent,
        col=col,
    )
    add_wireframe_modifier(box, thickness=thickness)
    return box


def create_ring_arrow_disc(name, radius=0.15, thickness=0.02, location=(0, 0, 0), parent=None, col=None):
    return create_cylinder(
        name,
        radius=radius,
        depth=thickness,
        location=location,
        rotation=(math.pi / 2, 0, 0),
        parent=parent,
        col=col,
    )


def remove_objects_by_prefix(prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    to_remove = []
    for obj in bpy.data.objects:
        for p in prefixes:
            if obj.name.startswith(p):
                to_remove.append(obj)
                break
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


def remove_default_startup_objects(keep_camera_name=None):
    default_names = {"Cube", "Light", "Area", "Point", "Sun", "Spot"}
    for obj in list(bpy.data.objects):
        if obj.name == keep_camera_name:
            continue
        if obj.name in default_names:
            bpy.data.objects.remove(obj, do_unlink=True)


def ensure_world_nodes():
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links

    bg = None
    out = None
    for n in nodes:
        if n.type == 'BACKGROUND':
            bg = n
        elif n.type == 'OUTPUT_WORLD':
            out = n

    if bg is None:
        bg = nodes.new("ShaderNodeBackground")
        bg.location = (0, 0)
    if out is None:
        out = nodes.new("ShaderNodeOutputWorld")
        out.location = (250, 0)

    linked = False
    for lk in links:
        if lk.from_node == bg and lk.to_node == out:
            linked = True
            break
    if not linked:
        for lk in list(links):
            if lk.to_node == out:
                links.remove(lk)
        links.new(bg.outputs["Background"], out.inputs["Surface"])

    return world, bg, out


def create_area_light(name, location=(0, 0, 0), rotation=(0, 0, 0), energy=1000.0, size=2.0, parent=None, col=None):
    light_data = bpy.data.lights.get(name)
    if light_data is None:
        light_data = bpy.data.lights.new(name=name, type='AREA')
    else:
        light_data.type = 'AREA'

    light_data.energy = energy
    light_data.shape = 'RECTANGLE'
    light_data.size = size
    light_data.size_y = size

    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, light_data)
        if col is not None:
            link_object(obj, col)
        else:
            bpy.context.scene.collection.objects.link(obj)
    else:
        obj.data = light_data
        if col is not None and col not in obj.users_collection:
            link_object(obj, col)

    parent_and_set_local_transform(obj, parent=parent, location=location, rotation=rotation)
    return obj


def setup_color_management_for_studio(exposure=0.0):
    scene = bpy.context.scene
    try:
        scene.view_settings.view_transform = 'Standard'
    except Exception:
        pass
    try:
        scene.view_settings.look = 'None'
    except Exception:
        pass
    try:
        scene.view_settings.exposure = exposure
    except Exception:
        pass
    try:
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def create_curved_backdrop(name, width=4.0, depth=4.0, height=3.0, curve_radius=0.6, location=(0, 0, 0), parent=None, col=None):
    ground = create_cube(
        name + "_Ground",
        size=1.0,
        location=(location[0], location[1], location[2]),
        scale=(width / 2, depth / 2, 0.02),
        parent=parent,
        col=col,
    )

    back = create_cube(
        name + "_Back",
        size=1.0,
        location=(location[0], location[1] + depth / 2 - 0.02, location[2] + height / 2),
        scale=(width / 2, 0.02, height / 2),
        parent=parent,
        col=col,
    )

    curve = create_cylinder(
        name + "_Curve",
        radius=curve_radius,
        depth=width,
        location=(location[0], location[1] + depth / 2 - curve_radius, location[2] + curve_radius),
        rotation=(0, math.pi / 2, 0),
        parent=parent,
        col=col,
    )

    return {"ground": ground, "back": back, "curve": curve}


def setup_clean_studio_scene(cfg, exp_col, root):
    scfg = cfg["studio_scene"]

    remove_objects_by_prefix(["STUDIO_", "BG_", "LIGHT_"])
    if scfg.get("clear_default_startup_objects", True):
        remove_default_startup_objects(keep_camera_name=cfg.get("camera_name", "Camera"))

    world, bg, out = ensure_world_nodes()
    bg.inputs["Color"].default_value = scfg["background_color"]
    bg.inputs["Strength"].default_value = scfg["world_strength"]

    setup_color_management_for_studio(exposure=scfg.get("view_exposure", -1.0))

    bg_mat = make_material(
        "STUDIO_MAT_Background",
        base_color=scfg["background_color"],
        metallic=0.0,
        roughness=1.0
    )
    floor_mat = make_material(
        "STUDIO_MAT_Floor",
        base_color=scfg["floor_color"],
        metallic=0.0,
        roughness=1.0
    )

    studio_root = create_empty(scfg.get("studio_root_name", "STUDIO_ROOT"), location=(0.0, 0.0, 0.0), parent=None, col=exp_col)
    y_shift = scfg.get("backdrop_y_shift", 0.0)

    if scfg.get("background_mode", "SOLID") == "CURVE":
        bg_objs = create_curved_backdrop(
            "BG_CurveBackdrop",
            width=scfg["backdrop_width"],
            depth=scfg["backdrop_depth"],
            height=scfg["backdrop_height"],
            curve_radius=scfg["backdrop_curve_radius"],
            location=(0.0, y_shift, -0.02),
            parent=studio_root,
            col=exp_col,
        )
        assign_material(bg_objs["ground"], floor_mat)
        assign_material(bg_objs["back"], bg_mat)
        assign_material(bg_objs["curve"], bg_mat)
    else:
        floor = create_cube(
            "BG_Floor",
            size=1.0,
            location=(0.0, y_shift, -0.02),
            scale=(scfg["backdrop_width"] / 2, scfg["backdrop_depth"] / 2, 0.02),
            parent=studio_root,
            col=exp_col,
        )
        assign_material(floor, floor_mat)

        back = create_cube(
            "BG_BackWall",
            size=1.0,
            location=(0.0, y_shift + scfg["backdrop_depth"] / 2 - 0.02, scfg["backdrop_height"] / 2 - 0.02),
            scale=(scfg["backdrop_width"] / 2, 0.02, scfg["backdrop_height"] / 2),
            parent=studio_root,
            col=exp_col,
        )
        assign_material(back, bg_mat)

    area_size = scfg["area_size"]

    key = create_area_light(
        "LIGHT_Key",
        location=(2.6, -2.8, 3.8),
        rotation=(math.radians(52), 0.0, math.radians(30)),
        energy=scfg["key_energy"],
        size=area_size,
        parent=studio_root,
        col=exp_col,
    )

    fill = create_area_light(
        "LIGHT_Fill",
        location=(-2.8, -2.6, 3.2),
        rotation=(math.radians(58), 0.0, math.radians(-32)),
        energy=scfg["fill_energy"],
        size=area_size * 1.15,
        parent=studio_root,
        col=exp_col,
    )

    top = create_area_light(
        "LIGHT_Top",
        location=(0.0, -0.8, 5.0),
        rotation=(math.radians(180), 0.0, 0.0),
        energy=scfg["top_energy"],
        size=scfg["top_area_size"],
        parent=studio_root,
        col=exp_col,
    )

    rim = create_area_light(
        "LIGHT_Rim",
        location=(0.0, 3.8, 3.2),
        rotation=(math.radians(-70), 0.0, math.radians(180)),
        energy=scfg["rim_energy"],
        size=area_size * 0.9,
        parent=studio_root,
        col=exp_col,
    )

    scene = bpy.context.scene
    if scene.render.engine in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        eevee = scene.eevee
        try:
            eevee.taa_render_samples = 64
        except Exception:
            pass
        try:
            eevee.use_gtao = True
        except Exception:
            pass
        try:
            eevee.use_bloom = False
        except Exception:
            pass

    return {
        "studio_root": studio_root,
        "lights": [key, fill, top, rim],
    }


def set_camera_to_scene(cfg):
    cam = bpy.data.objects.get(cfg["camera_name"])
    if cam is None:
        cam_data = bpy.data.cameras.new(cfg["camera_name"])
        cam = bpy.data.objects.new(cfg["camera_name"], cam_data)
        bpy.context.scene.collection.objects.link(cam)

    loc = cfg.get("camera_location")
    rot_deg = cfg.get("camera_rotation_deg")
    if loc is not None:
        cam.location = Vector(loc)
    if rot_deg is not None:
        cam.rotation_euler = Euler(tuple(math.radians(v) for v in rot_deg), 'XYZ')

    bpy.context.scene.camera = cam
    return cam


def set_collection_visibility(target_col_name):
    exp_root = bpy.data.collections.get("EXP")
    if exp_root is None:
        return
    for col in exp_root.children:
        visible = (col.name == target_col_name)
        col.hide_render = not visible
        col.hide_viewport = not visible


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


def safe_ball_y(y, radius, floor_y):
    return max(y, floor_y + radius)


def world_xyz(obj):
    p = obj.matrix_world.translation
    return p.x, p.y, p.z


def get_depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def raycast_down_filtered(origin_world, ignore_objects=None, allowed_object_names=None, max_distance=20.0, epsilon=0.002, upward_normal_min=None):
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
    max_distance = collision_cfg.get("raycast_max_distance", 20.0)
    eps = collision_cfg.get("raycast_epsilon", 0.002)
    contact_offset = collision_cfg.get("contact_offset", 0.0)
    allowed_names = collision_cfg.get("support_object_names", [])
    upward_normal_min = collision_cfg.get("upward_normal_min", 0.5)

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
        if abs(denom) < 1e-12:
            frac = 1.0
        else:
            frac = (prev_local_pos.z - contact_local_z) / denom
        frac = max(0.0, min(1.0, frac))
        hit_local_pos = prev_local_pos.lerp(curr_local_pos, frac)
        hit_local_pos.z = contact_local_z
        return {
            **contact,
            "frac": frac,
            "local_pos": hit_local_pos,
        }

    return None


# ------------------------------------------------------------
# Scene bootstrap
# ------------------------------------------------------------

def bootstrap_scene(cfg):
    out_dir = set_render_settings(cfg)
    set_camera_to_scene(cfg)

    exp_root = get_or_create_collection("EXP")
    exp_col_name = f"EXP_{cfg['experiment'].upper()}"
    exp_col = get_or_create_collection(exp_col_name, parent=exp_root)
    clear_collection_objects(exp_col)
    set_collection_visibility(exp_col_name)

    mats = {
        "ball": make_material("EXP_MAT_Ball", cfg["ball_color"], metallic=0.0, roughness=0.3),
        "metal": make_material("EXP_MAT_Metal", cfg["metal_color"], metallic=0.9, roughness=0.3),
        "plastic": make_material("EXP_MAT_Plastic", cfg["plastic_color"], metallic=0.0, roughness=0.55),
        "accent": make_material("EXP_MAT_Accent", cfg["accent_color"], metallic=0.0, roughness=0.45),
        "glass": make_material("EXP_MAT_Glass", (0.95, 0.98, 1.0, 1.0), metallic=0.0, roughness=0.03, transmission=1.0, ior=1.45),
        "base": make_material("EXP_MAT_Base", cfg["base_color"], metallic=0.0, roughness=0.65),
    }

    root = create_empty(cfg["exp_root_name"], location=cfg["lab_origin"], col=exp_col)

    if cfg.get("use_exp_platform", False):
        base = create_cube(
            "EXP_Platform",
            size=1.0,
            location=(0.0, 0.0, -0.03),
            scale=(0.45, 0.30, 0.03),
            parent=root,
            col=exp_col,
        )
        assign_material(base, mats["base"])

    if cfg.get("use_clean_studio_scene", True):
        setup_clean_studio_scene(cfg, exp_col, root)

    return out_dir, exp_col, root, mats


# ------------------------------------------------------------
# Experiment builders (all local coordinates under EXP_ROOT)
# ------------------------------------------------------------

def build_freefall(cfg, exp_col, root, mats):
    p = cfg["freefall"]
    o = Vector((0, 0, 0))

    if cfg.get("use_glass_enclosure", False):
        tube = create_glass_box(
            "FF_GlassTube",
            dims=(p["tube_radius"] * 2.0, p["tube_radius"] * 2.0, p["tube_height"]),
            location=tuple(o + Vector((0, 0, p["tube_height"] / 2))),
            parent=root,
            col=exp_col,
            thickness=cfg["glass_thickness"],
        )
        assign_material(tube, mats["glass"])

    cap = create_cylinder(
        "FF_TopCap",
        radius=p["tube_radius"] * 1.05,
        depth=0.03,
        location=tuple(o + Vector((0, 0, p["tube_height"] + 0.015))),
        parent=root,
        col=exp_col,
    )
    assign_material(cap, mats["metal"])

    floor = create_cylinder(
        "FF_BottomBase",
        radius=p["tube_radius"] * 1.10,
        depth=0.04,
        location=tuple(o + Vector((0, 0, 0))),
        parent=root,
        col=exp_col,
    )
    assign_material(floor, mats["metal"])

    ball = create_uv_sphere(
        "FF_Ball",
        radius=cfg["ball_radius"],
        location=tuple(o + Vector((0, 0, p["h0"]))),
        parent=root,
        col=exp_col,
    )
    assign_material(ball, mats["ball"])

    return {"ball": ball, "root": root}


def animate_freefall(cfg, obj_map, out_dir):
    p = cfg["freefall"]
    collision_cfg = cfg.get("collision_detection", {})
    ball = obj_map["ball"]
    root = obj_map["root"]
    fps = cfg["fps"]
    rows = []
    radius = cfg["ball_radius"]
    contacted = False
    frozen_local_z = None
    frozen_hit_name = None
    substeps = max(1, int(collision_cfg.get("substeps", 1)))
    debug_print = bool(collision_cfg.get("debug_print", False))
    prev_local_pos = Vector((0.0, 0.0, p["h0"]))

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        t = frame_to_time(frame, fps)
        predicted_local_z = p["h0"] + p["v0"] * t - 0.5 * p["g"] * t * t
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
                seg_hit = None

                for s in range(1, substeps + 1):
                    alpha = s / substeps
                    seg_t = prev_t + (t - prev_t) * alpha
                    seg_curr = Vector((0.0, 0.0, p["h0"] + p["v0"] * seg_t - 0.5 * p["g"] * seg_t * seg_t))
                    seg_hit = detect_segment_crossing(
                        root=root,
                        prev_local_pos=seg_prev,
                        curr_local_pos=seg_curr,
                        radius=radius,
                        ignore_objects=[ball],
                        fallback_floor_local_z=p.get("floor_y"),
                        collision_cfg=collision_cfg,
                    )
                    if seg_hit is not None:
                        z_local = seg_hit["local_pos"].z
                        hit_name = seg_hit["hit_object_name"]
                        if debug_print:
                            print(f"[freefall swept] frame={frame} step={s}/{substeps} hit={hit_name} local_z={z_local:.6f}")
                        if collision_cfg.get("freeze_on_contact", True):
                            contacted = True
                            frozen_local_z = z_local
                            frozen_hit_name = hit_name
                        break
                    seg_prev = seg_curr
            elif p.get("stop_before_hit", True) and p.get("floor_y") is not None:
                z_local = safe_ball_y(z_local, radius, p["floor_y"])

        ball.location = Vector((0.0, 0.0, z_local))
        ball.rotation_euler = Euler((0.0, 0.0, 0.0))
        add_keyframe(ball, frame)
        prev_local_pos = Vector(ball.location)

        wx, wy, wz = world_xyz(ball)
        rows.append([
            frame, round(t, 8), wx, wy, wz,
            0.0, 0.0, 0.0,
            0.0, 0.0, z_local,
            int(contacted), hit_name or "",
        ])

    export_csv(
        os.path.join(out_dir, "freefall_trajectory.csv"),
        rows,
        ["frame", "t", "world_x", "world_y", "world_z", "rx", "ry", "rz", "local_x", "local_y", "local_z", "contacted", "hit_object"],
    )
    export_json(os.path.join(out_dir, "freefall_params.json"), {**p, "collision_detection": collision_cfg})


def build_projectile(cfg, exp_col, root, mats):
    p = cfg["projectile"]
    o = Vector((0, 0, 0))

    if cfg.get("use_glass_enclosure", False):
        box = create_glass_box(
            "PR_GlassBox",
            dims=(p["box_length"], p["box_depth"], p["box_height"]),
            location=tuple(o + Vector((p["box_length"] / 2 - 0.05, 0, p["box_height"] / 2))),
            parent=root,
            col=exp_col,
            thickness=cfg["glass_thickness"],
        )
        assign_material(box, mats["glass"])

    launcher = create_cube(
        "PR_Launcher",
        size=1.0,
        location=tuple(o + Vector((-0.18, 0, p["y0"]))),
        scale=(0.16, 0.05, 0.05),
        parent=root,
        col=exp_col,
    )
    assign_material(launcher, mats["metal"])

    guide = create_cube(
        "PR_GuideRail",
        size=1.0,
        location=tuple(o + Vector((0.00, 0, p["y0"]))),
        scale=(0.25, 0.015, 0.015),
        parent=root,
        col=exp_col,
    )
    assign_material(guide, mats["accent"])

    floor = create_cube(
        "PR_Floor",
        size=1.0,
        location=tuple(o + Vector((p["box_length"] / 2 - 0.05, 0, 0))),
        scale=(p["box_length"] / 2, p["box_depth"] / 2, 0.02),
        parent=root,
        col=exp_col,
    )
    assign_material(floor, mats["metal"])

    ball = create_uv_sphere(
        "PR_Ball",
        radius=cfg["ball_radius"],
        location=tuple(o + Vector((p["x0"], 0, p["y0"]))),
        parent=root,
        col=exp_col,
    )
    assign_material(ball, mats["ball"])

    return {"ball": ball, "root": root}


def animate_projectile(cfg, obj_map, out_dir):
    p = cfg["projectile"]
    collision_cfg = cfg.get("collision_detection", {})
    ball = obj_map["ball"]
    root = obj_map["root"]
    fps = cfg["fps"]
    rows = []
    theta = math.radians(p["theta_deg"])
    v0x = p["v0"] * math.cos(theta)
    v0y = p["v0"] * math.sin(theta)
    radius = cfg["ball_radius"]
    contacted = False
    frozen_local = None
    frozen_hit_name = None
    substeps = max(1, int(collision_cfg.get("substeps", 1)))
    debug_print = bool(collision_cfg.get("debug_print", False))
    prev_local_pos = Vector((p["x0"], 0.0, p["y0"]))

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        t = frame_to_time(frame, fps)
        predicted_x = p["x0"] + v0x * t
        predicted_z = p["y0"] + v0y * t - 0.5 * p["g"] * t * t
        hit_name = None

        if frame == bpy.context.scene.frame_start:
            x_local, z_local = p["x0"], p["y0"]
        elif contacted and frozen_local is not None:
            x_local, z_local = frozen_local
            hit_name = frozen_hit_name
        else:
            x_local, z_local = predicted_x, predicted_z
            if collision_cfg.get("enabled", True):
                prev_t = frame_to_time(frame - 1, fps)
                seg_prev = Vector(prev_local_pos)
                seg_hit = None

                for s in range(1, substeps + 1):
                    alpha = s / substeps
                    seg_t = prev_t + (t - prev_t) * alpha
                    seg_curr = Vector((
                        p["x0"] + v0x * seg_t,
                        0.0,
                        p["y0"] + v0y * seg_t - 0.5 * p["g"] * seg_t * seg_t,
                    ))
                    seg_hit = detect_segment_crossing(
                        root=root,
                        prev_local_pos=seg_prev,
                        curr_local_pos=seg_curr,
                        radius=radius,
                        ignore_objects=[ball],
                        fallback_floor_local_z=p.get("floor_y"),
                        collision_cfg=collision_cfg,
                    )
                    if seg_hit is not None:
                        x_local = seg_hit["local_pos"].x
                        z_local = seg_hit["local_pos"].z
                        hit_name = seg_hit["hit_object_name"]
                        if debug_print:
                            print(f"[projectile swept] frame={frame} step={s}/{substeps} hit={hit_name} local=({x_local:.6f}, {z_local:.6f})")
                        if collision_cfg.get("freeze_on_contact", True):
                            contacted = True
                            frozen_local = (x_local, z_local)
                            frozen_hit_name = hit_name
                        break
                    seg_prev = seg_curr
            elif p.get("stop_before_hit", True) and p.get("floor_y") is not None:
                z_local = safe_ball_y(z_local, radius, p["floor_y"])

        ball.location = Vector((x_local, 0.0, z_local))
        ball.rotation_euler = Euler((0.0, 0.0, 0.0))
        add_keyframe(ball, frame)
        prev_local_pos = Vector(ball.location)

        wx, wy, wz = world_xyz(ball)
        rows.append([
            frame, round(t, 8), wx, wy, wz,
            x_local, 0.0, z_local,
            v0x, v0y, p["g"],
            int(contacted), hit_name or "",
        ])

    export_csv(
        os.path.join(out_dir, "projectile_trajectory.csv"),
        rows,
        ["frame", "t", "world_x", "world_y", "world_z", "local_x", "local_y", "local_z", "v0x", "v0y", "g", "contacted", "hit_object"],
    )
    export_json(os.path.join(out_dir, "projectile_params.json"), {**p, "v0x": v0x, "v0y": v0y, "collision_detection": collision_cfg})


def build_spring(cfg, exp_col, root, mats):
    p = cfg["spring"]
    o = Vector((0, 0, 0))

    stand = create_cube(
        "SP_Stand",
        size=1.0,
        location=tuple(o + Vector((-0.48, 0, 0.16))),
        scale=(0.04, 0.08, 0.16),
        parent=root,
        col=exp_col,
    )
    assign_material(stand, mats["metal"])

    rail = create_cube(
        "SP_Rail",
        size=1.0,
        location=tuple(o + Vector((0, 0, 0.07))),
        scale=(p["rail_length"] / 2, 0.05, 0.02),
        parent=root,
        col=exp_col,
    )
    assign_material(rail, mats["metal"])

    slider = create_cube(
        "SP_Slider",
        size=1.0,
        location=tuple(o + Vector((0, 0, 0.11))),
        scale=(0.08, 0.06, 0.04),
        parent=root,
        col=exp_col,
    )
    assign_material(slider, mats["accent"])

    spring_vis = create_cylinder(
        "SP_SpringVis",
        radius=0.02,
        depth=p["spring_rest"],
        location=tuple(o + Vector((-0.30, 0, 0.11))),
        rotation=(0, math.pi / 2, 0),
        parent=root,
        col=exp_col,
    )
    assign_material(spring_vis, mats["metal"])

    return {"slider": slider, "spring_vis": spring_vis}


def animate_spring(cfg, obj_map, out_dir):
    p = cfg["spring"]
    slider = obj_map["slider"]
    spring_vis = obj_map["spring_vis"]
    fps = cfg["fps"]
    rows = []
    omega = math.sqrt(p["k"] / p["m"])
    phi = math.radians(p["phi_deg"])
    anchor_x = -0.48
    z = 0.11

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        t = frame_to_time(frame, fps)
        x_local = p["A"] * math.cos(omega * t + phi)
        slider.location = Vector((x_local, 0, z))
        slider.rotation_euler = Euler((0, 0, 0))

        length = max(0.05, (slider.location.x - anchor_x))
        spring_vis.location = Vector((anchor_x + length / 2, 0, z))
        spring_vis.rotation_euler = Euler((0, math.pi / 2, 0))
        spring_vis.scale = Vector((1.0, 1.0, length / p["spring_rest"]))

        add_keyframe(slider, frame)
        spring_vis.keyframe_insert(data_path="location", frame=frame)
        spring_vis.keyframe_insert(data_path="rotation_euler", frame=frame)
        spring_vis.keyframe_insert(data_path="scale", frame=frame)

        wx, wy, wz = world_xyz(slider)
        rows.append([frame, round(t, 8), wx, wy, wz, x_local, 0.0, z, omega])

    export_csv(
        os.path.join(out_dir, "spring_trajectory.csv"),
        rows,
        ["frame", "t", "world_x", "world_y", "world_z", "local_x", "local_y", "local_z", "omega"],
    )
    export_json(os.path.join(out_dir, "spring_params.json"), {**p, "omega": omega})


def build_pendulum(cfg, exp_col, root, mats):
    p = cfg["pendulum"]
    o = Vector((0, 0, 0))

    post = create_cube(
        "PD_Post",
        size=1.0,
        location=tuple(o + Vector((0, 0, p["stand_height"] / 2))),
        scale=(0.03, 0.03, p["stand_height"] / 2),
        parent=root,
        col=exp_col,
    )
    assign_material(post, mats["metal"])

    arm = create_cube(
        "PD_Arm",
        size=1.0,
        location=tuple(o + Vector((0.12, 0, p["stand_height"]))),
        scale=(0.16, 0.02, 0.02),
        parent=root,
        col=exp_col,
    )
    assign_material(arm, mats["metal"])

    pivot = create_empty(
        "PD_Pivot",
        location=tuple(o + Vector((0.24, 0, p["stand_height"]))),
        parent=root,
        col=exp_col,
    )

    line = create_cylinder(
        "PD_Line",
        radius=0.005,
        depth=p["L"],
        location=(0, 0, -p["L"] / 2),
        parent=pivot,
        col=exp_col,
    )
    assign_material(line, mats["plastic"])

    bob = create_uv_sphere(
        "PD_Bob",
        radius=cfg["ball_radius"],
        location=(0, 0, -p["L"]),
        parent=pivot,
        col=exp_col,
    )
    assign_material(bob, mats["ball"])

    return {"pivot": pivot, "bob": bob, "line": line}


def animate_pendulum(cfg, obj_map, out_dir):
    p = cfg["pendulum"]
    pivot = obj_map["pivot"]
    bob = obj_map["bob"]
    fps = cfg["fps"]
    rows = []
    omega = math.sqrt(p["g"] / p["L"])
    Theta = math.radians(p["Theta_deg"])
    phi = math.radians(p["phi_deg"])

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        t = frame_to_time(frame, fps)
        theta = Theta * math.cos(omega * t + phi)
        pivot.rotation_euler = Euler((0, theta, 0))
        pivot.keyframe_insert(data_path="rotation_euler", frame=frame)

        wx, wy, wz = world_xyz(bob)
        rows.append([frame, round(t, 8), wx, wy, wz, theta, omega])

    export_csv(
        os.path.join(out_dir, "pendulum_trajectory.csv"),
        rows,
        ["frame", "t", "world_x", "world_y", "world_z", "theta", "omega"],
    )
    export_json(os.path.join(out_dir, "pendulum_params.json"), {**p, "omega": omega})


def build_rotation(cfg, exp_col, root, mats):
    p = cfg["rotation"]
    o = Vector((0, 0, 0))

    base = create_cylinder(
        "RT_Base",
        radius=p["disc_radius"] * 0.75,
        depth=p["base_height"],
        location=tuple(o + Vector((0, 0, p["base_height"] / 2))),
        parent=root,
        col=exp_col,
    )
    assign_material(base, mats["metal"])

    disc = create_ring_arrow_disc(
        "RT_Disc",
        radius=p["disc_radius"],
        thickness=0.03,
        location=tuple(o + Vector((0, 0, p["base_height"] + 0.03 / 2))),
        parent=root,
        col=exp_col,
    )
    assign_material(disc, mats["plastic"])

    marker = create_cube(
        "RT_Marker",
        size=1.0,
        location=(p["disc_radius"] * 0.55, 0, 0.002),
        scale=(p["disc_radius"] * 0.35, 0.015, 0.003),
        parent=disc,
        col=exp_col,
    )
    assign_material(marker, mats["accent"])

    return {"disc": disc}


def animate_rotation(cfg, obj_map, out_dir):
    p = cfg["rotation"]
    disc = obj_map["disc"]
    fps = cfg["fps"]
    rows = []
    theta0 = math.radians(p["theta0_deg"])
    omega0 = math.radians(p["omega0_deg_s"])
    alpha = math.radians(p["alpha_deg_s2"])

    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        t = frame_to_time(frame, fps)
        theta = theta0 + omega0 * t
        if p["constant_accel"]:
            theta += 0.5 * alpha * t * t
        disc.rotation_euler = Euler((0, 0, theta))
        add_keyframe(disc, frame)
        rows.append([frame, round(t, 8), theta, omega0, alpha])

    export_csv(
        os.path.join(out_dir, "rotation_trajectory.csv"),
        rows,
        ["frame", "t", "theta_rad", "omega0_rad_s", "alpha_rad_s2"],
    )
    export_json(os.path.join(out_dir, "rotation_params.json"), p)


# ------------------------------------------------------------
# Main dispatch
# ------------------------------------------------------------

def main(cfg):
    out_dir, exp_col, root, mats = bootstrap_scene(cfg)
    exp = cfg["experiment"].lower()

    builders = {
        "freefall": (build_freefall, animate_freefall),
        "projectile": (build_projectile, animate_projectile),
        "spring": (build_spring, animate_spring),
        "pendulum": (build_pendulum, animate_pendulum),
        "rotation": (build_rotation, animate_rotation),
    }
    if exp not in builders:
        raise ValueError(f"Unknown experiment: {exp}")

    build_fn, anim_fn = builders[exp]
    obj_map = build_fn(cfg, exp_col, root, mats)
    anim_fn(cfg, obj_map, out_dir)

    if cfg["render_video"]:
        bpy.ops.render.render(animation=True)

    print(f"[DONE] Experiment={exp} output_dir={out_dir}")


if __name__ == "__main__":
    main(CFG)
