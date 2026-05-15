from __future__ import annotations

from typing import Any, Dict

COMMON_NEGATIVE_PROMPT = (
    "cartoon, animation, illustration, painting, CGI, stylized motion, surreal motion, "
    "camera shake, camera pan, camera zoom, camera roll, moving camera, handheld, slow motion, "
    "time-lapse, motion blur, extra objects, multiple objects, occlusion, partial visibility, "
    "cropped object, deformed object, flicker, frame inconsistency, background clutter, text overlay, subtitles"
)

PROMPT_STYLES = [
    "short_raw",
    "structured_raw",
    "structured_detailed",
]

SCENE_LAB = (
    "A controlled physics experiment recorded as realistic video in a clean indoor laboratory with a plain light background. "
)

CAMERA_STATIC = (
    "Static side-view camera on a tripod. The camera does not move, zoom, roll, or shake. "
)

VISIBILITY = (
    "The full object remains visible for the entire clip and the full trajectory stays inside the frame. "
)

TRACKABLE_OBJECT = (
    "Use a single high-contrast object with sharp boundaries and stable shape so the center can be tracked clearly in every frame. "
)


def _fmt(x: float) -> str:
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _task_semantics(task: str, params: Dict[str, Any]) -> Dict[str, str]:
    task = task.lower()

    if task == "free_fall":
        g = _fmt(params["g"])
        h0 = _fmt(params["h0"])
        v0 = _fmt(params["v0"])
        color = params.get("color", "bright red")
        obj = params.get("object_name", "sphere")
        return {
            "subject": f"one single {color} {obj}",
            "motion": (
                f"The object starts at a visible height of {h0} meters with initial vertical velocity {v0} m/s "
                f"and then falls vertically downward under constant gravity {g} m/s^2 in vacuum."
            ),
            "constraints": "No air resistance, no bounce, no rotation, no collision, no extra objects.",
            "physics": "Single-object ideal free fall under constant downward gravity.",
        }

    if task == "projectile":
        g = _fmt(params["g"])
        x0 = _fmt(params["x0"])
        y0 = _fmt(params["y0"])
        v0 = _fmt(params["v0"])
        theta = _fmt(params["theta_deg"])
        color = params.get("color", "bright blue")
        obj = params.get("object_name", "sphere")
        return {
            "subject": f"one single {color} {obj}",
            "motion": (
                f"The object starts at x={x0} meters and y={y0} meters, and is launched with speed {v0} m/s "
                f"at {theta} degrees above horizontal. It follows a clean parabolic trajectory under constant gravity {g} m/s^2 in vacuum."
            ),
            "constraints": "No air resistance, no bounce, no spin, no collision, no extra objects.",
            "physics": "Single-object ideal projectile motion under constant downward gravity.",
        }

    if task == "spring":
        k = _fmt(params["k"])
        m = _fmt(params["m"])
        A = _fmt(params["A"])
        phi = _fmt(params["phi"])
        color = params.get("color", "bright yellow")
        obj = params.get("object_name", "block")
        return {
            "subject": f"one single {color} {obj} attached to a horizontal spring",
            "motion": (
                f"The object performs simple harmonic motion on a frictionless straight track with spring constant {k} N/m, "
                f"mass {m} kg, amplitude {A} meters, and initial phase {phi} radians."
            ),
            "constraints": "No damping, no friction, no rotation, no extra objects.",
            "physics": "Ideal undamped horizontal spring oscillation.",
        }

    if task == "pendulum":
        L = _fmt(params["L"])
        theta = _fmt(params["theta_max_deg"])
        phi = _fmt(params["phi"])
        color = params.get("color", "bright green")
        obj = params.get("object_name", "pendulum bob")
        return {
            "subject": f"one single {color} {obj} attached to a rigid massless rod",
            "motion": (
                f"The pendulum length is {L} meters, the maximum initial angular displacement is {theta} degrees, "
                f"and the initial phase is {phi} radians. The bob swings in a single plane with a small angle."
            ),
            "constraints": "Fixed pivot, no air resistance, no damping, no camera motion, no extra objects.",
            "physics": "Ideal small-angle pendulum motion.",
        }

    if task == "rotation":
        theta0 = _fmt(params["theta0_deg"])
        omega0 = _fmt(params["omega0_deg_s"])
        alpha = _fmt(params["alpha_deg_s2"])
        color = params.get("color", "white")
        obj = params.get("object_name", "rigid bar")
        return {
            "subject": f"one single {color} {obj} suspended in space",
            "motion": (
                f"The object rotates around a fixed central axis with initial angle {theta0} degrees, "
                f"initial angular velocity {omega0} degrees per second, and angular acceleration {alpha} degrees per second squared."
            ),
            "constraints": "No translation, no wobble, no deformation, no extra objects.",
            "physics": "Ideal rigid-body planar rotation about a fixed axis.",
        }

    raise ValueError(f"Unsupported task: {task}")


def _short_raw(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"{sem['subject']}. "
        f"{SCENE_LAB}"
        f"{CAMERA_STATIC}"
        f"{VISIBILITY}"
        f"{sem['motion']} "
        f"{sem['constraints']} "
        "No text, no subtitles."
    )


def _structured_raw(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"Subject: {sem['subject']}. "
        f"Scene: clean indoor laboratory, plain background, empty scene. "
        f"Camera: static side view, fixed tripod, medium-wide shot, full trajectory stays in frame. "
        f"Motion: {sem['motion']} "
        f"Physics: {sem['physics']} "
        f"Constraints: {sem['constraints']} No people, no hands, no text, no subtitles."
    )


def _structured_detailed(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"Subject: {sem['subject']} with strong color contrast against the background and a stable silhouette. "
        f"Scene: clean physics laboratory, plain matte background, no clutter, even lighting. "
        f"Camera: static side-view camera on a tripod, no camera motion or zoom, medium-wide framing, the object stays fully visible for the whole clip. "
        f"Motion: {sem['motion']} "
        f"Physics: {sem['physics']} "
        f"Tracking constraint: the object boundary should remain sharp enough for frame-by-frame center tracking. "
        f"Constraints: {sem['constraints']} No people, no hands, no text, no subtitles, no motion blur that hides the object."
    )


STYLE_BUILDERS = {
    "short_raw": _short_raw,
    "structured_raw": _structured_raw,
    "structured_detailed": _structured_detailed,
}


def build_prompt(task: str, params: Dict[str, Any], prompt_style: str = "structured_raw") -> str:
    if prompt_style not in STYLE_BUILDERS:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")
    return STYLE_BUILDERS[prompt_style](task, params)
