from __future__ import annotations

import math
from typing import Any, Dict

COMMON_NEGATIVE_PROMPT = (
    "cartoon, animation, illustration, painting, CGI, stylized motion, surreal motion, "
    "camera shake, camera pan, camera zoom, camera roll, moving camera, handheld, slow motion, "
    "time-lapse, motion blur, extra objects, multiple objects, occlusion, partial visibility, "
    "cropped object, deformed object, flicker, frame inconsistency, background clutter, text overlay, subtitles, watermark"
)

PROMPT_STYLES = [
    "short_raw",
    "structured_raw",
    "structured_detailed",
]

SCENE_LAB = (
    "Realistic physics experiment video in a clean indoor laboratory with a plain light background and even lighting. "
)

CAMERA_SIDE = (
    "Static side-view camera on a fixed tripod. No pan, no tilt, no zoom, no roll, no shake. "
)

CAMERA_FRONT = (
    "Static front-view camera on a fixed tripod. No pan, no tilt, no zoom, no roll, no shake. "
)

FULL_VIS = (
    "The full object stays visible for the whole clip and the full motion trajectory remains inside the frame. "
)

TRACKABLE = (
    "Use one single high-contrast object with sharp boundaries, stable shape, and clear center for frame-by-frame tracking. "
)


def _fmt(x: float) -> str:
    x = float(x)
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _safe_get(params: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in params:
            return params[k]
    return default


def _projectile_components(params: Dict[str, Any]) -> Dict[str, float]:
    g = float(_safe_get(params, "g", default=9.8))
    x0 = float(_safe_get(params, "x0", default=0.0))
    y0 = float(_safe_get(params, "y0", default=1.5))

    if "v0x" in params or "v0y" in params:
        v0x = float(_safe_get(params, "v0x", default=0.0))
        v0y = float(_safe_get(params, "v0y", default=0.0))
        v0 = math.sqrt(v0x * v0x + v0y * v0y)
        theta_deg = math.degrees(math.atan2(v0y, v0x)) if abs(v0) > 1e-9 else 0.0
    else:
        v0 = float(_safe_get(params, "v0", default=3.0))
        theta_deg = float(_safe_get(params, "theta_deg", "theta", default=0.0))
        theta_rad = math.radians(theta_deg)
        v0x = v0 * math.cos(theta_rad)
        v0y = v0 * math.sin(theta_rad)

    return {
        "g": g,
        "x0": x0,
        "y0": y0,
        "v0": v0,
        "theta_deg": theta_deg,
        "v0x": v0x,
        "v0y": v0y,
    }


def _oscillation_cycles(omega: float, duration: float | None) -> str:
    if duration is None or duration <= 0:
        return "several"
    cycles = omega * duration / (2 * math.pi)
    if cycles < 1.5:
        return "about 1"
    if cycles < 2.5:
        return "about 2"
    if cycles < 3.5:
        return "about 3"
    return f"about {_fmt(round(cycles))}"


def _semantics_free_fall(params: Dict[str, Any]) -> Dict[str, str]:
    g = _fmt(params["g"])
    h0 = _fmt(params["h0"])
    v0 = float(params["v0"])
    v0_s = _fmt(v0)
    color = params.get("color", "bright red")
    obj = params.get("object_name", "matte ball")
    released = "released from rest" if abs(v0) < 1e-9 else f"given an initial vertical velocity of {v0_s} m/s"

    return {
        "camera": CAMERA_SIDE,
        "subject": f"one single {color} {obj}",
        "motion_short": f"The object starts at height {h0} meters, is {released}, and falls straight downward under gravity {g} m/s^2 in vacuum.",
        "motion_detailed": f"The object starts clearly high above the ground at about {h0} meters, is {released}, moves only in the vertical direction, and its downward speed increases visibly over time.",
        "physics": f"Gravity is {g} m/s^2, initial height is {h0} meters, initial vertical velocity is {v0_s} m/s, vacuum environment, ideal single-object free fall.",
        "constraints": "No air resistance, no bounce, no collision before the clip ends, no rolling, no rotation, no extra objects.",
    }


def _semantics_projectile(params: Dict[str, Any]) -> Dict[str, str]:
    c = _projectile_components(params)
    g = _fmt(c["g"])
    x0 = _fmt(c["x0"])
    y0 = _fmt(c["y0"])
    v0 = _fmt(c["v0"])
    theta = _fmt(c["theta_deg"])
    v0x = _fmt(c["v0x"])
    v0y = _fmt(c["v0y"])
    color = params.get("color", "bright blue")
    obj = params.get("object_name", "matte ball")

    horizontal_throw = abs(c["v0y"]) < 1e-6 or abs(c["theta_deg"]) < 1e-6
    if horizontal_throw:
        motion_short = (
            f"The object starts at x={x0} meters and y={y0} meters and is launched horizontally from left to right with horizontal speed {v0x} m/s in vacuum under gravity {g} m/s^2."
        )
        motion_detailed = (
            f"The object starts from a clearly visible height of about {y0} meters, initially has no upward motion, moves with nearly constant horizontal speed, and follows a smooth parabolic path while descending faster and faster."
        )
        physics = (
            f"Gravity is {g} m/s^2, initial position is ({x0}, {y0}) meters, horizontal initial speed is {v0x} m/s, vertical initial speed is {v0y} m/s, vacuum environment, ideal horizontal projectile motion."
        )
    else:
        motion_short = (
            f"The object starts at x={x0} meters and y={y0} meters and is launched from left to right at speed {v0} m/s with angle {theta} degrees in vacuum under gravity {g} m/s^2."
        )
        motion_detailed = (
            f"The object first rises, reaches one visible highest point, then falls along a smooth parabolic trajectory. The horizontal motion remains steady while the vertical velocity changes from upward to downward."
        )
        physics = (
            f"Gravity is {g} m/s^2, initial position is ({x0}, {y0}) meters, launch speed is {v0} m/s, launch angle is {theta} degrees, horizontal initial speed is {v0x} m/s, vertical initial speed is {v0y} m/s, vacuum environment, ideal oblique projectile motion."
        )

    return {
        "camera": CAMERA_SIDE,
        "subject": f"one single {color} {obj}",
        "motion_short": motion_short,
        "motion_detailed": motion_detailed,
        "physics": physics,
        "constraints": "No air resistance, no bounce, no collision before the clip ends, no spin, no extra objects.",
    }


def _semantics_spring(params: Dict[str, Any]) -> Dict[str, str]:
    k = float(params["k"])
    m = float(params["m"])
    A = _fmt(params["A"])
    phi = _fmt(params.get("phi", 0.0))
    omega = math.sqrt(k / m)
    omega_s = _fmt(omega)
    duration = _safe_get(params, "duration_s", default=None)
    n_cycles = params.get("n_cycles") or _oscillation_cycles(omega, duration)
    color = params.get("color", "bright yellow")
    obj = params.get("object_name", "rectangular block")

    return {
        "camera": CAMERA_FRONT,
        "subject": f"one single {color} {obj} attached to a clearly visible horizontal spring",
        "motion_short": f"The block oscillates left and right around one equilibrium position with amplitude {A} meters and angular frequency {omega_s} rad/s.",
        "motion_detailed": f"The block performs smooth periodic horizontal motion with nearly constant amplitude, completes {n_cycles} visible oscillations during the clip, and stays on one straight horizontal line around the central equilibrium point.",
        "physics": f"Spring stiffness is {_fmt(k)} N/m, mass is {_fmt(m)} kg, amplitude is {A} meters, initial phase is {phi} radians, angular frequency is {omega_s} rad/s, ideal undamped horizontal spring oscillation.",
        "constraints": "No damping, no friction, no external disturbance, no rotation of the block, no extra objects.",
    }


def _semantics_pendulum(params: Dict[str, Any]) -> Dict[str, str]:
    L = float(params["L"])
    g = float(_safe_get(params, "g", default=9.8))
    theta_max = float(_safe_get(params, "theta_max_deg", "Theta", default=10.0))
    phi = _fmt(params.get("phi", 0.0))
    T = 2 * math.pi * math.sqrt(L / g)
    duration = _safe_get(params, "duration_s", default=None)
    n_cycles = params.get("n_cycles") or (_oscillation_cycles(2 * math.pi / T, duration))
    color = params.get("color", "bright green")
    bob = params.get("object_name", "pendulum bob")

    return {
        "camera": CAMERA_FRONT,
        "subject": f"one single {color} spherical {bob} attached to a thin dark string",
        "motion_short": f"The pendulum performs small-angle oscillation with string length {_fmt(L)} meters and initial angle amplitude {_fmt(theta_max)} degrees under gravity {_fmt(g)} m/s^2.",
        "motion_detailed": f"The pendulum starts near one side, swings smoothly through the center, stays close to the vertical line, and completes {n_cycles} visible oscillations with approximately symmetric left-right motion.",
        "physics": f"String length is {_fmt(L)} meters, gravity is {_fmt(g)} m/s^2, initial angle amplitude is {_fmt(theta_max)} degrees, initial phase is {phi} radians, ideal small-angle pendulum motion.",
        "constraints": "Fixed pivot, no damping, no air resistance, no external disturbance, no extra objects.",
    }


def _semantics_rotation(params: Dict[str, Any]) -> Dict[str, str]:
    theta0 = _fmt(_safe_get(params, "theta0_deg", default=0.0))
    omega0 = float(_safe_get(params, "omega0_deg_s", "omega0", default=90.0))
    alpha = float(_safe_get(params, "alpha_deg_s2", "alpha", default=0.0))
    omega0_s = _fmt(omega0)
    alpha_s = _fmt(alpha)
    color = params.get("color", "white")
    obj = params.get("object_name", "flat circular disk")
    if abs(alpha) < 1e-9:
        trend = "The rotation speed remains visually constant throughout the clip."
        phys = f"Initial angle is {theta0} degrees, angular velocity is constant at {omega0_s} degrees per second, no angular acceleration, ideal uniform rotation about a fixed central axis."
    elif alpha > 0:
        trend = "The rotation speed increases steadily and visibly over time."
        phys = f"Initial angle is {theta0} degrees, initial angular velocity is {omega0_s} degrees per second, angular acceleration is {alpha_s} degrees per second squared, ideal accelerated rotation about a fixed central axis."
    else:
        trend = "The rotation speed decreases steadily and visibly over time."
        phys = f"Initial angle is {theta0} degrees, initial angular velocity is {omega0_s} degrees per second, angular acceleration is {alpha_s} degrees per second squared, ideal decelerated rotation about a fixed central axis."

    return {
        "camera": CAMERA_FRONT,
        "subject": f"one single {color} {obj} with a bold black radial orientation marker and a colored center marker",
        "motion_short": f"The disk rotates in place around its central axis from initial angle {theta0} degrees with initial angular velocity {omega0_s} degrees per second and angular acceleration {alpha_s} degrees per second squared.",
        "motion_detailed": f"The object performs pure rotation only, its center stays fixed in space with no translation, and the orientation marker is clearly visible in every frame. {trend}",
        "physics": phys,
        "constraints": "No translation of the object center, no wobble, no deformation, no extra objects.",
    }


def _task_semantics(task: str, params: Dict[str, Any]) -> Dict[str, str]:
    task = task.lower()
    if task == "free_fall":
        return _semantics_free_fall(params)
    if task == "projectile":
        return _semantics_projectile(params)
    if task == "spring":
        return _semantics_spring(params)
    if task == "pendulum":
        return _semantics_pendulum(params)
    if task == "rotation":
        return _semantics_rotation(params)
    raise ValueError(f"Unsupported task: {task}")


def _short_raw(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"{sem['subject']}. "
        f"{SCENE_LAB}"
        f"{sem['camera']}"
        f"{FULL_VIS}"
        f"{sem['motion_short']} "
        f"{sem['motion_detailed']} "
        f"{sem['constraints']} "
        "No people, no hands, no text, no subtitles."
    )


def _structured_raw(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"Subject: {sem['subject']}. "
        f"Scene: clean indoor laboratory, plain background, empty scene, even lighting. "
        f"Camera: {sem['camera']}"
        f"Visibility: the full object and the full motion stay inside the frame. "
        f"Motion: {sem['motion_short']} {sem['motion_detailed']} "
        f"Physics: {sem['physics']} "
        f"Constraints: {sem['constraints']} No people, no hands, no text, no subtitles."
    )


def _structured_detailed(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"Subject: {sem['subject']} with strong color contrast against the background, stable silhouette, sharp boundary, and a clearly trackable center. "
        f"Scene: clean and minimal physics laboratory, plain matte pale background, no clutter, no distracting objects, realistic lighting. "
        f"Camera: {sem['camera']}"
        f"Visibility: the object remains fully visible for the entire clip and the whole physical trajectory stays inside the frame from start to end. "
        f"Tracking requirement: {TRACKABLE}"
        f"Motion: {sem['motion_short']} {sem['motion_detailed']} "
        f"Physics: {sem['physics']} "
        f"Constraints: single-object experiment only. {sem['constraints']} No people, no hands, no camera motion, no scene change, no text, no subtitles, no watermark, no motion blur that hides the object."
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
