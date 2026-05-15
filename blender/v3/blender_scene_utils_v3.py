import math

import bpy
from mathutils import Euler, Vector


FPS = 24
FRAME_END = 240


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.collections,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def setup_scene():
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    scene.render.fps = FPS
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.unit_settings.system = "METRIC"
    scene.world.color = (0.018, 0.019, 0.022)
    if hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 96
    return scene


def make_collection(name):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(obj, collection):
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)


def make_material(name, base_color, roughness=0.45, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    principled = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Metallic"].default_value = metallic
    return mat


def default_materials():
    return {
        "floor": make_material("MAT_Floor", (0.78, 0.79, 0.82, 1.0), roughness=0.92),
        "wall": make_material("MAT_Wall", (0.92, 0.93, 0.95, 1.0), roughness=0.94),
        "accent": make_material("MAT_Accent", (0.18, 0.44, 0.78, 1.0), roughness=0.36),
        "metal": make_material("MAT_Metal", (0.62, 0.65, 0.70, 1.0), roughness=0.24, metallic=0.9),
        "rubber": make_material("MAT_Rubber", (0.79, 0.44, 0.15, 1.0), roughness=0.52),
        "rubber_dark": make_material("MAT_RubberDark", (0.07, 0.07, 0.08, 1.0), roughness=0.62),
        "crate": make_material("MAT_Crate", (0.71, 0.50, 0.33, 1.0), roughness=0.88),
        "crate_dark": make_material("MAT_CrateDark", (0.44, 0.30, 0.19, 1.0), roughness=0.82),
        "suitcase": make_material("MAT_Suitcase", (0.18, 0.24, 0.36, 1.0), roughness=0.34),
        "plastic": make_material("MAT_Plastic", (0.11, 0.12, 0.14, 1.0), roughness=0.46),
        "bowling": make_material("MAT_Bowling", (0.12, 0.16, 0.42, 1.0), roughness=0.22),
        "softbag": make_material("MAT_SoftBag", (0.48, 0.24, 0.18, 1.0), roughness=0.74),
        "rope": make_material("MAT_Rope", (0.42, 0.34, 0.22, 1.0), roughness=0.98),
        "mark": make_material("MAT_Mark", (0.92, 0.92, 0.22, 1.0), roughness=0.58),
        "text": make_material("MAT_Text", (0.10, 0.11, 0.13, 1.0), roughness=0.55),
        "rough": make_material("MAT_Rough", (0.48, 0.35, 0.20, 1.0), roughness=0.96),
    }


def assign_material(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def add_cube(name, location, scale, rotation=(0.0, 0.0, 0.0), collection=None, material=None):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    if material:
        assign_material(obj, material)
    if collection:
        link_to_collection(obj, collection)
    return obj


def add_plane(name, location, size, rotation=(0.0, 0.0, 0.0), collection=None, material=None):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    if material:
        assign_material(obj, material)
    if collection:
        link_to_collection(obj, collection)
    return obj


def add_uv_sphere(name, location, radius, collection=None, material=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=48, ring_count=24)
    obj = bpy.context.active_object
    obj.name = name
    if material:
        assign_material(obj, material)
    if collection:
        link_to_collection(obj, collection)
    return obj


def add_cylinder(name, location, radius, depth, rotation=(0.0, 0.0, 0.0), collection=None, material=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation, vertices=32)
    obj = bpy.context.active_object
    obj.name = name
    if material:
        assign_material(obj, material)
    if collection:
        link_to_collection(obj, collection)
    return obj


def add_torus(name, location, major_radius, minor_radius, rotation=(0.0, 0.0, 0.0), collection=None, material=None):
    bpy.ops.mesh.primitive_torus_add(
        location=location,
        rotation=rotation,
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=48,
        minor_segments=16,
    )
    obj = bpy.context.active_object
    obj.name = name
    if material:
        assign_material(obj, material)
    if collection:
        link_to_collection(obj, collection)
    return obj


def add_text(name, body, location, rotation, scale, collection, material):
    bpy.ops.object.text_add(location=location, rotation=rotation)
    text = bpy.context.active_object
    text.name = name
    text.data.body = body
    text.data.extrude = 0.02
    text.data.align_x = "CENTER"
    text.scale = scale
    assign_material(text, material)
    link_to_collection(text, collection)
    return text


def add_camera(name, location, rotation_deg, collection, lens=45):
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    cam_obj.location = location
    cam_obj.rotation_euler = Euler(tuple(math.radians(v) for v in rotation_deg), "XYZ")
    cam_obj.data.lens = lens
    collection.objects.link(cam_obj)
    return cam_obj


def look_at(obj, target):
    target = Vector(target)
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, rotation_deg, power, size, color, collection):
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = power
    light_data.color = color
    light_data.shape = "RECTANGLE"
    light_data.size = size
    light_data.size_y = size * 0.62
    light_obj = bpy.data.objects.new(name, light_data)
    light_obj.location = location
    light_obj.rotation_euler = Euler(tuple(math.radians(v) for v in rotation_deg), "XYZ")
    collection.objects.link(light_obj)
    return light_obj


def add_basketball(name, location, radius, collection, mats):
    ball = add_uv_sphere(name, location, radius, collection=collection, material=mats["rubber"])
    seams = [
        add_torus(f"{name}_Seam_1", location, radius * 0.98, radius * 0.05, rotation=(math.radians(90), 0.0, 0.0), collection=collection, material=mats["rubber_dark"]),
        add_torus(f"{name}_Seam_2", location, radius * 0.98, radius * 0.05, rotation=(0.0, math.radians(90), 0.0), collection=collection, material=mats["rubber_dark"]),
        add_torus(f"{name}_Seam_3", location, radius * 0.70, radius * 0.04, rotation=(0.0, math.radians(48), 0.0), collection=collection, material=mats["rubber_dark"]),
        add_torus(f"{name}_Seam_4", location, radius * 0.70, radius * 0.04, rotation=(0.0, math.radians(-48), 0.0), collection=collection, material=mats["rubber_dark"]),
    ]
    for seam in seams:
        seam.parent = ball
        seam.matrix_parent_inverse = ball.matrix_world.inverted()
    return ball


def add_plain_ball(name, location, radius, collection, material):
    return add_uv_sphere(name, location, radius, collection=collection, material=material)


def add_shipping_crate(name, location, scale, collection, mats):
    crate = add_cube(name, location, scale, collection=collection, material=mats["crate"])
    ribs = [
        add_cube(f"{name}_RibX1", (location[0], location[1], location[2] + scale[2] * 0.92), (scale[0], scale[1] * 0.12, scale[2] * 0.05), collection=collection, material=mats["crate_dark"]),
        add_cube(f"{name}_RibX2", (location[0], location[1], location[2] - scale[2] * 0.92), (scale[0], scale[1] * 0.12, scale[2] * 0.05), collection=collection, material=mats["crate_dark"]),
        add_cube(f"{name}_RibY1", (location[0] - scale[0] * 0.92, location[1], location[2]), (scale[0] * 0.05, scale[1] * 0.12, scale[2]), collection=collection, material=mats["crate_dark"]),
        add_cube(f"{name}_RibY2", (location[0] + scale[0] * 0.92, location[1], location[2]), (scale[0] * 0.05, scale[1] * 0.12, scale[2]), collection=collection, material=mats["crate_dark"]),
    ]
    for rib in ribs:
        rib.parent = crate
        rib.matrix_parent_inverse = crate.matrix_world.inverted()
    return crate


def add_suitcase(name, location, scale, collection, mats):
    body = add_cube(name, location, scale, collection=collection, material=mats["suitcase"])
    handle = add_cylinder(
        f"{name}_Handle",
        (location[0], location[1] - scale[1] - 0.05, location[2] + scale[2] * 0.75),
        0.03,
        scale[0] * 0.7,
        rotation=(0.0, 0.0, math.radians(90)),
        collection=collection,
        material=mats["metal"],
    )
    wheel_1 = add_cylinder(
        f"{name}_Wheel1",
        (location[0] - scale[0] * 0.72, location[1], location[2] - scale[2] - 0.02),
        0.045,
        0.05,
        rotation=(math.radians(90), 0.0, 0.0),
        collection=collection,
        material=mats["plastic"],
    )
    wheel_2 = add_cylinder(
        f"{name}_Wheel2",
        (location[0] + scale[0] * 0.72, location[1], location[2] - scale[2] - 0.02),
        0.045,
        0.05,
        rotation=(math.radians(90), 0.0, 0.0),
        collection=collection,
        material=mats["plastic"],
    )
    for child in (handle, wheel_1, wheel_2):
        child.parent = body
        child.matrix_parent_inverse = body.matrix_world.inverted()
    return body


def add_bowling_ball(name, location, radius, collection, mats):
    ball = add_uv_sphere(name, location, radius, collection=collection, material=mats["bowling"])
    for idx, dx in enumerate((-0.08, 0.0, 0.08), start=1):
        hole = add_cylinder(
            f"{name}_Hole_{idx}",
            (location[0] + dx, location[1], location[2] + radius * 0.78),
            0.03,
            0.04,
            rotation=(math.radians(90), 0.0, 0.0),
            collection=collection,
            material=mats["plastic"],
        )
        hole.parent = ball
        hole.matrix_parent_inverse = ball.matrix_world.inverted()
    return ball


def add_hanging_bag(name, location, scale, collection, mats):
    bag = add_cylinder(name, location, radius=scale[0], depth=scale[2] * 2.0, collection=collection, material=mats["softbag"])
    bag.scale.y = scale[1] / scale[0]
    strap = add_cylinder(
        f"{name}_Strap",
        (location[0], location[1], location[2] + scale[2] + 0.18),
        0.025,
        0.38,
        collection=collection,
        material=mats["rope"],
    )
    strap.parent = bag
    strap.matrix_parent_inverse = bag.matrix_world.inverted()
    return bag


def add_coil_spring(name, start_x, end_x, y, z, coil_radius, wire_radius, turns, collection, material):
    span = end_x - start_x
    for idx in range(turns):
        frac = (idx + 0.5) / turns
        x = start_x + span * frac
        torus = add_torus(
            f"{name}_{idx+1}",
            (x, y, z),
            major_radius=coil_radius,
            minor_radius=wire_radius,
            rotation=(0.0, math.radians(90), 0.0),
            collection=collection,
            material=material,
        )
        torus.scale.x = max(0.25, span / (2.2 * turns * coil_radius))
    return


def build_base_world(label_text):
    reset_scene()
    scene = setup_scene()
    env = make_collection("ENV")
    probes = make_collection("PROBES")
    cameras = make_collection("CAMERAS")
    lights = make_collection("LIGHTS")
    labels = make_collection("LABELS")
    mats = default_materials()

    add_plane("ENV_Floor", (0.0, 0.0, 0.0), 46, collection=env, material=mats["floor"])
    wall_height = 7.5
    wall_center_z = wall_height
    add_cube("ENV_BackWall", (0.0, 11.0, wall_center_z), (20.0, 0.10, wall_height), collection=env, material=mats["wall"])
    add_cube("ENV_LeftWall", (-20.0, 0.0, wall_center_z), (0.10, 11.0, wall_height), collection=env, material=mats["wall"])
    add_cube("ENV_RightWall", (20.0, 0.0, wall_center_z), (0.10, 11.0, wall_height), collection=env, material=mats["wall"])

    camera_target = (0.0, 0.0, 1.3)
    cam_main_location = (6.5, -14.5, 5.0)
    cam_main = add_camera("CAM_Main", cam_main_location, (74, 0, 0), cameras, lens=48)
    look_at(cam_main, camera_target)
    cam_side = add_camera("CAM_Side", (0.0, -17.2, 3.8), (84, 0, 0), cameras, lens=58)
    cam_top_location = (-cam_main_location[0], cam_main_location[1], cam_main_location[2])
    cam_top = add_camera("CAM_Top", cam_top_location, (74, 0, 0), cameras, lens=48)
    look_at(cam_top, camera_target)
    scene.camera = cam_main

    add_area_light("LIGHT_Key", (7.0, -7.0, 8.8), (58, 0, 34), 3000, 8.0, (1.0, 0.96, 0.92), lights)
    add_area_light("LIGHT_Fill", (-5.5, -8.8, 5.9), (70, 0, -24), 1200, 6.0, (0.84, 0.9, 1.0), lights)
    add_area_light("LIGHT_Top", (0.0, 0.0, 11.0), (180, 0, 0), 1500, 10.0, (1.0, 1.0, 1.0), lights)
    return scene, probes, mats, cam_main, cam_side


def t(frame):
    return (frame - 1) / FPS


def set_linear_interpolation(obj):
    if not obj.animation_data or not obj.animation_data.action:
        return
    action = obj.animation_data.action
    if hasattr(action, "fcurves"):
        for fcurve in action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            channelbags = getattr(strip, "channelbags", None)
            if not channelbags:
                continue
            for channelbag in channelbags:
                for fcurve in getattr(channelbag, "fcurves", []):
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = "LINEAR"


def animate_transforms(obj, transforms):
    obj.animation_data_clear()
    for frame, location, rotation in transforms:
        bpy.context.scene.frame_set(frame)
        obj.location = location
        obj.rotation_euler = rotation
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
    set_linear_interpolation(obj)


def cleanup_empty_collections():
    changed = True
    while changed:
        changed = False
        for collection in list(bpy.data.collections):
            if collection == bpy.context.scene.collection:
                continue
            if len(collection.objects) == 0 and len(collection.children) == 0:
                bpy.data.collections.remove(collection)
                changed = True
                break


def cleanup_orphan_data():
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def print_summary(names):
    scene = bpy.context.scene
    print("SUMMARY")
    for frame in (1, 30, 60, 90, 120, 180, 240):
        scene.frame_set(frame)
        print(f"FRAME {frame}")
        for name in names:
            obj = bpy.data.objects.get(name)
            if obj:
                loc = tuple(round(v, 4) for v in obj.matrix_world.translation)
                print(" ", name, loc)
