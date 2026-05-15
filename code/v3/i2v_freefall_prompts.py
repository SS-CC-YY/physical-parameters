# -*- coding: utf-8 -*-
NEGATIVE = (
    "no extra objects, no people, no hands, no camera motion, no zoom, no pan, "
    "no text, no subtitles, no watermark, no duplicated ball, no deformation, "
    "no motion blur that hides the ball, no bounce unless specified, no rolling"
)

def _common_prefix(params):
    object_name = params.get("object_name", "matte ball")
    color = params.get("color", "bright red")
    return (
        f"The input image already shows one single {color} {object_name} near the top of the frame. "
        f"Preserve the same object identity, same scene layout, same viewpoint, and the same fixed camera. "
        f"Continue the scene from this exact initial frame. "
    )

def freefall_i2v_prompt(params, style="structured_detailed"):
    g = params["g"]
    v0 = params["v0"]
    h0 = params.get("h0", None)
    object_name = params.get("object_name", "matte ball")
    color = params.get("color", "bright red")

    if style == "short_raw":
        return (
            f"{_common_prefix(params)} "
            f"A single {color} {object_name} is released and undergoes pure vertical free fall in vacuum. "
            f"Gravity is {g} m/s^2 and initial vertical velocity is {v0} m/s. "
            f"The motion should follow a smooth quadratic fall with no horizontal drift. "
            f"{NEGATIVE}."
        )

    if style == "structured_raw":
        return (
            f"Subject: one single {color} {object_name}. "
            f"Initial frame: use the provided image as the exact start state. "
            f"Camera: static fixed camera, same viewpoint as the input image. "
            f"Motion: pure vertical free fall only. "
            f"Physics: gravity = {g} m/s^2, initial vertical velocity = {v0} m/s, vacuum environment, no air resistance. "
            f"Constraint: the object center should move only downward, with negligible horizontal displacement, and the fall should be smooth and quadratic. "
            f"{NEGATIVE}."
        )

    extra_h = ""
    if h0 is not None:
        extra_h = (
            f" The physical height parameter in the prompt is h0 = {h0} meters. "
            f"Keep the visible initial frame unchanged, but the generated motion should be consistent with the requested physics parameters."
        )

    return (
        f"Subject: one single {color} {object_name}, clearly visible and easy to track. "
        f"Initial frame: the provided image is the exact initial state at time t = 0. Keep the same ball, same scene, same wall, same floor, same lighting, and same fixed camera. "
        f"Camera: completely static, no pan, no tilt, no zoom, no scene change. "
        f"Motion type: pure free fall in one vertical plane, with only one dominant motion and no extra events. "
        f"Physics: gravity = {g} m/s^2, initial vertical velocity = {v0} m/s, vacuum environment, no air resistance, no external force except gravity.{extra_h} "
        f"Trajectory constraint: after release, the object center should follow a smooth vertical quadratic trajectory y(t)=h0+v0*t-0.5*g*t^2, with nearly zero horizontal drift. "
        f"Geometric constraint: keep the object as a single compact ball with stable shape and stable size. "
        f"Contact constraint: no bounce, no rolling, no collision artifacts, no penetration through the floor. "
        f"{NEGATIVE}."
    )
