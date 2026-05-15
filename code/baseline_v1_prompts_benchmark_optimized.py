from __future__ import annotations

import math
from typing import Any, Dict

# Benchmark-inspired prompt design principles:
# - Use disentangled, explicit sections and tailored prompts for controlled evaluation.
# - Specify initial conditions, geometric constraints, camera constraints, and force constraints.
# - Keep a single salient moving subject and a minimal environment for easier tracking.
# - Express physical parameters both numerically and visually.

COMMON_NEGATIVE_PROMPT = (
    "cartoon, animation, illustration, painting, CGI, stylized motion, surreal motion, "
    "camera shake, camera pan, camera zoom, camera roll, moving camera, handheld, dolly, tracking shot, "
    "slow motion, time-lapse, motion blur, extra objects, multiple moving objects, occlusion, partial visibility, "
    "cropped object, deformed object, flicker, frame inconsistency, background clutter, text overlay, subtitles, watermark"
)

PROMPT_STYLES = ["short_raw", "structured_raw", "structured_detailed"]

SCENE_LAB = (
    "Realistic physics experiment video in a clean indoor laboratory with a plain matte light background, even lighting, and no clutter. "
)

CAMERA_SIDE = (
    "Static side-view camera on a fixed tripod. No pan, no tilt, no zoom, no roll, no shake. "
)

CAMERA_FRONT = (
    "Static front-view camera on a fixed tripod. No pan, no tilt, no zoom, no roll, no shake. "
)

FULL_VIS = (
    "The moving object stays fully visible for the whole clip and the entire motion trajectory remains inside the frame. "
)

TRACKABLE = (
    "Use one single high-contrast moving object with a stable shape, sharp boundary, and a clearly trackable center for frame-by-frame tracking and curve fitting. "
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
    y0 = float(_safe_get(params, "y0", default=1.2))

    if "v0x" in params or "v0y" in params:
        v0x = float(_safe_get(params, "v0x", default=2.0))
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
    g = _fmt(_safe_get(params, "g", default=9.8))
    h0 = _fmt(_safe_get(params, "h0", default=3.0))
    v0 = float(_safe_get(params, "v0", default=0.0))
    v0_s = _fmt(v0)
    color = params.get("color", "bright red")
    obj = params.get("object_name", "matte ball")
    released = "released from rest" if abs(v0) < 1e-9 else f"given an initial vertical velocity of {v0_s} m/s"
    start_state = f"One single {color} {obj} is initially stationary at a height of about {h0} meters above the floor, with no contact with any surface."
    force_state = f"Only gravity acts on the object. Gravity is {g} m/s^2. Vacuum environment with no air resistance."
    motion_short = f"The object is {released} and then falls straight downward only, with its horizontal position staying approximately constant."
    motion_detailed = (
        "This is an ideal single-object free-fall experiment. The motion is purely vertical along the y-axis, the path is a straight vertical line, and the downward speed increases visibly over time."
    )
    physics = f"Initial height is {h0} meters, initial vertical velocity is {v0_s} m/s, horizontal velocity is zero, only one force is present: gravity."
    constraints = (
        "No air resistance, no bounce, no collision before the clip ends, no rolling, no spin, no extra moving objects, and no change of camera pose."
    )
    return {
        "camera": CAMERA_SIDE,
        "subject": f"one single {color} {obj}",
        "setup": "A minimal free-fall setup with one object and empty space around it.",
        "start_state": start_state,
        "force_state": force_state,
        "motion_short": motion_short,
        "motion_detailed": motion_detailed,
        "physics": physics,
        "constraints": constraints,
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
        setup = "A single small ball moves uniformly along a flat table and then leaves the table edge into open space."
        start_state = (
            f"One single {color} {obj} starts on a flat tabletop at height about {y0} meters, moves from left to right with constant horizontal speed, and reaches the table edge."
        )
        force_state = (
            f"After leaving the table edge, only gravity acts on the ball. Gravity is {g} m/s^2. Air resistance is absent."
        )
        motion_short = (
            f"The ball leaves the table horizontally with horizontal speed {v0x} m/s and zero initial vertical speed, then follows a smooth parabolic path downward."
        )
        motion_detailed = (
            "Before leaving the table, the motion is straight and uniform along the x-axis. After leaving the edge, the ball keeps nearly constant horizontal speed while accelerating downward along the y-axis."
        )
        physics = (
            f"Initial launch point is near (x={x0}, y={y0}) meters at the table edge, horizontal initial speed is {v0x} m/s, vertical initial speed is {v0y} m/s, and the only post-launch force is gravity."
        )
        constraints = (
            "Exactly one moving ball, one simple table, no bouncing, no contact after leaving the edge, no spin, no extra moving objects, and no camera motion."
        )
    else:
        setup = "A single ball is launched into open space in a clean projectile-motion experiment."
        start_state = (
            f"One single {color} {obj} starts near position (x={x0}, y={y0}) meters and is launched from left to right with speed {v0} m/s at angle {theta} degrees."
        )
        force_state = f"Only gravity acts on the ball after launch. Gravity is {g} m/s^2. Vacuum environment with no air resistance."
        motion_short = (
            "The ball first rises, reaches one highest point, and then falls. The trajectory is a single smooth parabola in one vertical plane."
        )
        motion_detailed = (
            f"Horizontal velocity starts at about {v0x} m/s and stays approximately constant, while vertical velocity starts at about {v0y} m/s and changes continuously due to gravity."
        )
        physics = (
            f"Initial launch speed is {v0} m/s, launch angle is {theta} degrees, horizontal initial speed is {v0x} m/s, vertical initial speed is {v0y} m/s, and only one force is present after launch: gravity."
        )
        constraints = (
            "Exactly one moving ball, no bounce before the clip ends, no collision, no spin, no extra moving objects, and no camera motion."
        )

    return {
        "camera": CAMERA_SIDE,
        "subject": f"one single {color} {obj}",
        "setup": setup,
        "start_state": start_state,
        "force_state": force_state,
        "motion_short": motion_short,
        "motion_detailed": motion_detailed,
        "physics": physics,
        "constraints": constraints,
    }


def _semantics_spring(params: Dict[str, Any]) -> Dict[str, str]:
    k = float(_safe_get(params, "k", default=8.0))
    m = float(_safe_get(params, "m", default=1.0))
    A = _fmt(_safe_get(params, "A", default=0.2))
    phi = _fmt(_safe_get(params, "phi", default=0.0))
    omega = math.sqrt(k / m)
    omega_s = _fmt(omega)
    duration = _safe_get(params, "duration_s", default=None)
    n_cycles = params.get("n_cycles") or _oscillation_cycles(omega, duration)
    color = params.get("color", "bright yellow")
    obj = params.get("object_name", "rectangular block")
    start_state = (
        f"One single {color} {obj} is attached to a clearly visible horizontal spring and starts displaced from equilibrium with amplitude {A} meters."
    )
    force_state = (
        f"The block moves on a frictionless horizontal surface or track. Motion is constrained to the x-axis only. No vertical motion is allowed. The restoring force is from the spring only."
    )
    motion_short = (
        f"The block oscillates left and right only, with angular frequency {omega_s} rad/s, completing {n_cycles} visible oscillations during the clip."
    )
    motion_detailed = (
        "This is ideal one-dimensional simple harmonic motion. The block stays on one straight horizontal line, keeps nearly constant amplitude, and does not rotate."
    )
    physics = (
        f"Spring stiffness is {_fmt(k)} N/m, mass is {_fmt(m)} kg, amplitude is {A} meters, initial phase is {phi} radians, and angular frequency is {omega_s} rad/s."
    )
    constraints = (
        "Exactly one moving block and one spring, no damping, no friction, no external push after release, no y-axis motion, no z-axis motion, no rotation of the block, and no extra moving objects."
    )
    return {
        "camera": CAMERA_FRONT,
        "subject": f"one single {color} {obj} attached to a horizontal spring",
        "setup": "A minimal spring-mass experiment on a frictionless horizontal surface.",
        "start_state": start_state,
        "force_state": force_state,
        "motion_short": motion_short,
        "motion_detailed": motion_detailed,
        "physics": physics,
        "constraints": constraints,
    }


def _semantics_pendulum(params: Dict[str, Any]) -> Dict[str, str]:
    L = float(_safe_get(params, "L", default=1.0))
    g = float(_safe_get(params, "g", default=9.8))
    theta_max = float(_safe_get(params, "theta_max_deg", "Theta", default=10.0))
    phi = _fmt(_safe_get(params, "phi", default=0.0))
    T = 2 * math.pi * math.sqrt(L / g)
    duration = _safe_get(params, "duration_s", default=None)
    n_cycles = params.get("n_cycles") or _oscillation_cycles(2 * math.pi / T, duration)
    color = params.get("color", "bright green")
    bob = params.get("object_name", "pendulum bob")
    start_state = (
        f"One single {color} spherical {bob} hangs from a thin dark string of length {_fmt(L)} meters and starts from a small angular displacement of about {_fmt(theta_max)} degrees."
    )
    force_state = (
        f"The pivot is fixed, the string length stays constant, and the bob swings in one vertical plane only under gravity { _fmt(g) } m/s^2."
    )
    motion_short = (
        f"The pendulum performs small-angle oscillation, swings left and right through the lowest point, and completes {n_cycles} visible oscillations."
    )
    motion_detailed = (
        "This is an ideal simple pendulum in the small-angle regime. The bob stays in one plane, the arc is symmetric, and the motion remains close to the vertical line."
    )
    physics = (
        f"String length is {_fmt(L)} meters, gravity is {_fmt(g)} m/s^2, initial angle amplitude is {_fmt(theta_max)} degrees, and initial phase is {phi} radians."
    )
    constraints = (
        "Exactly one moving bob and one string, fixed pivot, no damping, no air resistance, no external disturbance, no out-of-plane motion, and no extra moving objects."
    )
    return {
        "camera": CAMERA_FRONT,
        "subject": f"one single {color} spherical {bob} attached to a thin dark string",
        "setup": "A minimal single-pendulum experiment with one bob and one fixed pivot.",
        "start_state": start_state,
        "force_state": force_state,
        "motion_short": motion_short,
        "motion_detailed": motion_detailed,
        "physics": physics,
        "constraints": constraints,
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
        trend = "The angular speed stays constant throughout the clip."
        phys = (
            f"Initial angle is {theta0} degrees, angular velocity is constant at {omega0_s} degrees per second, and angular acceleration is zero."
        )
    elif alpha > 0:
        trend = "The angular speed increases steadily over time."
        phys = (
            f"Initial angle is {theta0} degrees, initial angular velocity is {omega0_s} degrees per second, and angular acceleration is {alpha_s} degrees per second squared."
        )
    else:
        trend = "The angular speed decreases steadily over time."
        phys = (
            f"Initial angle is {theta0} degrees, initial angular velocity is {omega0_s} degrees per second, and angular acceleration is {alpha_s} degrees per second squared."
        )

    start_state = (
        f"One single {color} {obj} with a bold black radial marker and a colored center marker starts from orientation angle {theta0} degrees."
    )
    force_state = (
        "The object rotates around one fixed central axis only. The center stays fixed in space and there is no translation."
    )
    motion_short = (
        f"The object performs pure in-place rotation with initial angular velocity {omega0_s} degrees per second. {trend}"
    )
    motion_detailed = (
        "This is a single rigid-body rotation experiment. The orientation marker stays clearly visible in each frame for angular tracking, and the object shows no wobble or shape deformation."
    )
    constraints = (
        "Exactly one rotating object, no translation of the center, no wobble, no precession, no deformation, no extra moving objects, and no camera motion."
    )
    return {
        "camera": CAMERA_FRONT,
        "subject": f"one single {color} {obj} with a bold black radial marker and a colored center marker",
        "setup": "A minimal rigid-body rotation experiment with one marked disk rotating around a fixed axis.",
        "start_state": start_state,
        "force_state": force_state,
        "motion_short": motion_short,
        "motion_detailed": motion_detailed,
        "physics": phys,
        "constraints": constraints,
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
        f"Setup: {sem['setup']} "
        f"Initial state: {sem['start_state']} "
        f"Dynamics: {sem['motion_short']} "
        f"Forces and constraints: {sem['force_state']} {sem['constraints']} "
        "No people, no hands, no text, no subtitles."
    )


def _structured_raw(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"Subject: {sem['subject']}. "
        f"Scene: clean indoor laboratory, plain matte background, empty scene, even lighting. "
        f"Camera: {sem['camera']}"
        f"Visibility: the full moving object and the full motion remain inside the frame. "
        f"Setup: {sem['setup']} "
        f"Initial state: {sem['start_state']} "
        f"Dynamics: {sem['motion_short']} {sem['motion_detailed']} "
        f"Forces: {sem['force_state']} "
        f"Physical quantities: {sem['physics']} "
        f"Constraints: {sem['constraints']} No people, no hands, no text, no subtitles."
    )


def _structured_detailed(task: str, params: Dict[str, Any]) -> str:
    sem = _task_semantics(task, params)
    return (
        f"Subject: {sem['subject']} with strong color contrast against the background, a stable silhouette, a sharp boundary, and a clearly trackable center. "
        f"Scene: clean and minimal physics laboratory, plain pale matte background, no clutter, no distracting objects, realistic lighting, only the minimal apparatus needed for the experiment. "
        f"Camera: {sem['camera']}"
        f"Visibility: the object remains fully visible for the entire clip and the whole physical trajectory stays inside the frame from start to end. "
        f"Tracking requirement: {TRACKABLE}"
        f"Experimental setup: {sem['setup']} "
        f"Initial state: {sem['start_state']} "
        f"Dynamics: {sem['motion_short']} {sem['motion_detailed']} "
        f"Force model: {sem['force_state']} "
        f"Physical quantities: {sem['physics']} "
        f"Hard constraints: single-motion, single-subject experiment. {sem['constraints']} No people, no hands, no camera motion, no scene change, no text, no subtitles, no watermark, and no motion blur that hides the moving object."
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
