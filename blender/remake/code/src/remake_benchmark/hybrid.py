"""Camera-motion routing shared by calibrated and 4D reconstruction paths.

Side-view videos are treated specially.  Their frozen Blender camera is close
to orthographic, so a calibrated image-plane trajectory is the primary signal.
We only pay for a dynamic 3D reconstruction when the camera audit measures
persistent, high-quality translation large enough to provide useful parallax.

For Main/Top, a camera-audit ``review`` label is not itself evidence of useful
camera motion.  A reviewed sample may return to calibrated reconstruction only
when the measured motion is small, non-persistent, cut-free, and supported by a
high-quality background registration.  This prevents a single noisy maximum
from needlessly replacing a clean calibrated trajectory with under-constrained
monocular 3-D, while still routing sustained pan/zoom/orbit motion to the
dynamic branch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


STATIC_CATEGORIES = {
    "fixed",
    "no_significant_camera_change",
    "calibrated_motion_within_tolerance",
}
SIDE_2D_EFFECTIVE_CATEGORY = "side_2d_motion_within_tolerance"
CALIBRATED_2D_EFFECTIVE_CATEGORY = "calibrated_motion_within_tolerance"
DYNAMIC_CATEGORIES = {
    "changed",
    "camera_changed",
    "review",
    "borderline_below_threshold",
    "error",
}

# Frozen-policy defaults for 864 x 496 Seedance videos.  The displacement
# threshold is normalized by the image diagonal so the rule also applies to
# other resolutions.  A persistent cluster prevents a single bad audit pair
# from sending an otherwise fixed Side video to 3D.
# A one-percent threshold admitted a reviewed false positive whose direct
# frame-0 registration jumped by 12 px even though adjacent-frame motion and
# the visible Side view remained stable.  Require a clearer displacement
# before paying the accuracy cost of under-constrained monocular 3-D.
SIDE_3D_MIN_TRANSLATION_DIAGONAL_FRACTION = 0.02
SIDE_3D_MIN_DIRECT_MOTION_CLUSTER = 5
SIDE_3D_MIN_VALID_PAIR_FRACTION = 0.80
SIDE_3D_MIN_MEDIAN_INLIER_RATIO = 0.70

# A non-Side review/borderline result can use frozen calibration only when the
# audit supplies positive evidence that motion is both small and non-persistent.
# Fractions are resolution-independent.  These are *2-D tolerance* gates, not
# claims that motion above them necessarily supplies enough parallax for 3-D.
CALIBRATED_2D_MAX_TRANSLATION_DIAGONAL_FRACTION = 0.01
CALIBRATED_2D_MAX_ROTATION_DEG = 0.75
CALIBRATED_2D_MAX_SCALE_CHANGE = 0.03
# Chained adjacent-frame transforms catch smooth tracking/zoom that can become
# too large for a reliable direct frame-0 match.  They receive slightly wider
# tolerances because chaining accumulates registration noise, but must still be
# small before a reviewed Main/Top sample may use frozen calibration.
CALIBRATED_2D_MAX_GLOBAL_TRANSLATION_DIAGONAL_FRACTION = 0.02
CALIBRATED_2D_MAX_GLOBAL_ROTATION_DEG = 1.00
CALIBRATED_2D_MAX_GLOBAL_SCALE_CHANGE = 0.05
CALIBRATED_2D_MAX_RESIDUAL_DIAGONAL_FRACTION = 0.002
CALIBRATED_2D_MIN_VALID_PAIR_FRACTION = 0.80
CALIBRATED_2D_MIN_MEDIAN_INLIER_RATIO = 0.65


def camera_motion_category(evidence: dict[str, Any] | None) -> str | None:
    """Return a normalized audit category without guessing missing evidence."""

    if not evidence:
        return None
    category = evidence.get("final_category") or evidence.get("decision")
    return None if category is None else str(category)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _camera_name(video_filename: str | Path, camera_name: str | None) -> str | None:
    if camera_name:
        return str(camera_name)
    filename = Path(video_filename).name
    for candidate in ("CAM_Side", "CAM_Main", "CAM_Top"):
        if f"__{candidate}__" in filename:
            return candidate
    return None


def _side_3d_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    width = _finite_float(evidence.get("width"))
    height = _finite_float(evidence.get("height"))
    translation_px = _finite_float(evidence.get("max_direct_translation_px"))
    valid_pair_fraction = _finite_float(evidence.get("valid_pair_fraction"))
    median_inlier_ratio = _finite_float(evidence.get("median_inlier_ratio"))
    motion_cluster = _finite_float(evidence.get("direct_motion_cluster_max"))
    diagonal_px = (
        math.hypot(width, height)
        if width is not None and height is not None and width > 0.0 and height > 0.0
        else None
    )
    translation_fraction = (
        translation_px / diagonal_px
        if translation_px is not None and diagonal_px is not None and diagonal_px > 0.0
        else None
    )
    cut_pair_count = int(evidence.get("cut_pair_count") or 0)
    status_ok = str(evidence.get("status") or "").lower() == "ok"

    threshold_checks = {
        "translation": bool(
            translation_fraction is not None
            and translation_fraction >= SIDE_3D_MIN_TRANSLATION_DIAGONAL_FRACTION
        ),
        "persistence": bool(
            motion_cluster is not None
            and motion_cluster >= SIDE_3D_MIN_DIRECT_MOTION_CLUSTER
        ),
        "valid_pairs": bool(
            valid_pair_fraction is not None
            and valid_pair_fraction >= SIDE_3D_MIN_VALID_PAIR_FRACTION
        ),
        "inlier_quality": bool(
            median_inlier_ratio is not None
            and median_inlier_ratio >= SIDE_3D_MIN_MEDIAN_INLIER_RATIO
        ),
        "no_scene_cut": cut_pair_count == 0,
        "audit_ok": status_ok,
    }
    return {
        "eligible": all(threshold_checks.values()),
        "translation_px": translation_px,
        "image_diagonal_px": diagonal_px,
        "translation_diagonal_fraction": translation_fraction,
        "direct_motion_cluster_max": motion_cluster,
        "valid_pair_fraction": valid_pair_fraction,
        "median_inlier_ratio": median_inlier_ratio,
        "cut_pair_count": cut_pair_count,
        "threshold_checks": threshold_checks,
        "thresholds": {
            "min_translation_diagonal_fraction": SIDE_3D_MIN_TRANSLATION_DIAGONAL_FRACTION,
            "min_direct_motion_cluster": SIDE_3D_MIN_DIRECT_MOTION_CLUSTER,
            "min_valid_pair_fraction": SIDE_3D_MIN_VALID_PAIR_FRACTION,
            "min_median_inlier_ratio": SIDE_3D_MIN_MEDIAN_INLIER_RATIO,
        },
    }


def _preferred_metric(
    evidence: dict[str, Any],
    preferred_key: str,
    fallback_key: str,
) -> float | None:
    value = _finite_float(evidence.get(preferred_key))
    return value if value is not None else _finite_float(evidence.get(fallback_key))


def _reviewed_calibrated_2d_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the positive evidence required to treat a review as 2-D-safe.

    P95 direct-to-frame-0 metrics are preferred because an isolated maximum is
    exactly what created the reviewed Top false positive.  Older frozen assets
    do not contain P95 fields, so their maxima are used as a conservative
    fallback; all hit and persistence counters must still be present and zero.
    """

    width = _finite_float(evidence.get("width"))
    height = _finite_float(evidence.get("height"))
    diagonal_px = (
        math.hypot(width, height)
        if width is not None and height is not None and width > 0.0 and height > 0.0
        else None
    )
    translation_px = _preferred_metric(
        evidence,
        "p95_direct_translation_px",
        "max_direct_translation_px",
    )
    rotation_deg = _preferred_metric(
        evidence,
        "p95_direct_rotation_deg",
        "max_direct_rotation_deg",
    )
    scale_change = _preferred_metric(
        evidence,
        "p95_direct_scale_change",
        "max_direct_scale_change",
    )
    global_translation_px = _preferred_metric(
        evidence,
        "p95_global_translation_px",
        "max_global_translation_px",
    )
    global_rotation_deg = _preferred_metric(
        evidence,
        "p95_global_rotation_deg",
        "max_global_rotation_deg",
    )
    global_scale_change = _preferred_metric(
        evidence,
        "p95_global_scale_change",
        "max_global_scale_change",
    )
    residual_px = _finite_float(evidence.get("median_residual_px"))
    translation_fraction = (
        translation_px / diagonal_px
        if translation_px is not None and diagonal_px is not None and diagonal_px > 0.0
        else None
    )
    residual_fraction = (
        residual_px / diagonal_px
        if residual_px is not None and diagonal_px is not None and diagonal_px > 0.0
        else None
    )
    global_translation_fraction = (
        global_translation_px / diagonal_px
        if global_translation_px is not None and diagonal_px is not None and diagonal_px > 0.0
        else None
    )
    valid_pair_fraction = _finite_float(evidence.get("valid_pair_fraction"))
    median_inlier_ratio = _finite_float(evidence.get("median_inlier_ratio"))
    motion_cluster = _finite_float(evidence.get("direct_motion_cluster_max"))
    hit_values = [
        _finite_float(evidence.get("direct_translation_hit_count")),
        _finite_float(evidence.get("direct_rotation_hit_count")),
        _finite_float(evidence.get("direct_scale_hit_count")),
    ]
    cut_pair_count = int(evidence.get("cut_pair_count") or 0)
    status_ok = str(evidence.get("status") or "").lower() == "ok"

    threshold_checks = {
        "translation_small": bool(
            translation_fraction is not None
            and translation_fraction <= CALIBRATED_2D_MAX_TRANSLATION_DIAGONAL_FRACTION
        ),
        "rotation_small": bool(
            rotation_deg is not None and rotation_deg <= CALIBRATED_2D_MAX_ROTATION_DEG
        ),
        "scale_small": bool(
            scale_change is not None and scale_change <= CALIBRATED_2D_MAX_SCALE_CHANGE
        ),
        "global_translation_small": bool(
            global_translation_fraction is not None
            and global_translation_fraction
            <= CALIBRATED_2D_MAX_GLOBAL_TRANSLATION_DIAGONAL_FRACTION
        ),
        "global_rotation_small": bool(
            global_rotation_deg is not None
            and global_rotation_deg <= CALIBRATED_2D_MAX_GLOBAL_ROTATION_DEG
        ),
        "global_scale_small": bool(
            global_scale_change is not None
            and global_scale_change <= CALIBRATED_2D_MAX_GLOBAL_SCALE_CHANGE
        ),
        "registration_residual_small": bool(
            residual_fraction is not None
            and residual_fraction <= CALIBRATED_2D_MAX_RESIDUAL_DIAGONAL_FRACTION
        ),
        "no_threshold_hits": bool(
            all(value is not None for value in hit_values)
            and sum(value for value in hit_values if value is not None) == 0.0
        ),
        "no_persistent_cluster": bool(motion_cluster is not None and motion_cluster == 0.0),
        "valid_pairs": bool(
            valid_pair_fraction is not None
            and valid_pair_fraction >= CALIBRATED_2D_MIN_VALID_PAIR_FRACTION
        ),
        "inlier_quality": bool(
            median_inlier_ratio is not None
            and median_inlier_ratio >= CALIBRATED_2D_MIN_MEDIAN_INLIER_RATIO
        ),
        "no_scene_cut": cut_pair_count == 0,
        "audit_ok": status_ok,
    }
    return {
        "eligible": all(threshold_checks.values()),
        "translation_px": translation_px,
        "image_diagonal_px": diagonal_px,
        "translation_diagonal_fraction": translation_fraction,
        "rotation_deg": rotation_deg,
        "scale_change": scale_change,
        "global_translation_px": global_translation_px,
        "global_translation_diagonal_fraction": global_translation_fraction,
        "global_rotation_deg": global_rotation_deg,
        "global_scale_change": global_scale_change,
        "median_residual_px": residual_px,
        "residual_diagonal_fraction": residual_fraction,
        "direct_hit_count": (
            sum(value for value in hit_values if value is not None)
            if all(value is not None for value in hit_values)
            else None
        ),
        "direct_motion_cluster_max": motion_cluster,
        "valid_pair_fraction": valid_pair_fraction,
        "median_inlier_ratio": median_inlier_ratio,
        "cut_pair_count": cut_pair_count,
        "threshold_checks": threshold_checks,
        "thresholds": {
            "max_translation_diagonal_fraction": (
                CALIBRATED_2D_MAX_TRANSLATION_DIAGONAL_FRACTION
            ),
            "max_rotation_deg": CALIBRATED_2D_MAX_ROTATION_DEG,
            "max_scale_change": CALIBRATED_2D_MAX_SCALE_CHANGE,
            "max_global_translation_diagonal_fraction": (
                CALIBRATED_2D_MAX_GLOBAL_TRANSLATION_DIAGONAL_FRACTION
            ),
            "max_global_rotation_deg": CALIBRATED_2D_MAX_GLOBAL_ROTATION_DEG,
            "max_global_scale_change": CALIBRATED_2D_MAX_GLOBAL_SCALE_CHANGE,
            "max_residual_diagonal_fraction": (
                CALIBRATED_2D_MAX_RESIDUAL_DIAGONAL_FRACTION
            ),
            "min_valid_pair_fraction": CALIBRATED_2D_MIN_VALID_PAIR_FRACTION,
            "min_median_inlier_ratio": CALIBRATED_2D_MIN_MEDIAN_INLIER_RATIO,
        },
    }
def choose_reconstruction_route(
    evidence: dict[str, Any] | None,
    video_filename: str | Path,
    *,
    camera_name: str | None = None,
) -> dict[str, Any]:
    """Choose the fast calibrated path only from matching, positive evidence.

    Missing, failed, explicitly changed, and under-specified review evidence is
    sent to the dynamic 3D/4D path.  A review can return to calibrated 2-D only
    through the explicit small/non-persistent/quality gates above.
    """

    filename = Path(video_filename).name
    resolved_camera = _camera_name(video_filename, camera_name)
    if not evidence:
        return {
            "route": "spatialtrackerv2_dynamic",
            "camera_motion_category": None,
            "raw_camera_motion_category": None,
            "effective_camera_motion_category": None,
            "reason": "missing_camera_motion_evidence",
            "static_calibration_allowed": False,
            "camera": resolved_camera,
        }

    evidence_filename = evidence.get("filename")
    if evidence_filename is None and evidence.get("video"):
        evidence_filename = Path(str(evidence["video"])).name
    if evidence_filename != filename:
        return {
            "route": "spatialtrackerv2_dynamic",
            "camera_motion_category": camera_motion_category(evidence),
            "raw_camera_motion_category": camera_motion_category(evidence),
            "effective_camera_motion_category": camera_motion_category(evidence),
            "reason": "camera_motion_evidence_filename_mismatch",
            "static_calibration_allowed": False,
            "evidence_filename": evidence_filename,
            "camera": resolved_camera,
        }

    category = camera_motion_category(evidence)
    if evidence.get("status") == "error" or category == "error":
        return {
            "route": "spatialtrackerv2_dynamic",
            "camera_motion_category": category,
            "raw_camera_motion_category": category,
            "effective_camera_motion_category": category,
            "reason": "camera_motion_audit_error",
            "static_calibration_allowed": False,
            "camera": resolved_camera,
        }
    if category in STATIC_CATEGORIES:
        return {
            "route": "calibrated_static_sphere",
            "camera_motion_category": category,
            "raw_camera_motion_category": category,
            "effective_camera_motion_category": category,
            "reason": "camera_motion_audit_confirmed_no_significant_change",
            "static_calibration_allowed": True,
            "camera": resolved_camera,
        }

    if resolved_camera == "CAM_Side":
        side_evidence = _side_3d_evidence(evidence)
        if side_evidence["eligible"]:
            return {
                "route": "spatialtrackerv2_dynamic",
                "camera_motion_category": category,
                "raw_camera_motion_category": category,
                "effective_camera_motion_category": category,
                "reason": "side_persistent_translation_supports_3d",
                "static_calibration_allowed": False,
                "camera": resolved_camera,
                "side_3d_evidence": side_evidence,
            }
        return {
            "route": "calibrated_static_sphere",
            "camera_motion_category": category,
            "raw_camera_motion_category": category,
            "effective_camera_motion_category": SIDE_2D_EFFECTIVE_CATEGORY,
            "reason": "side_motion_below_reliable_3d_threshold",
            "static_calibration_allowed": True,
            "calibrated_2d_approximation": True,
            "camera": resolved_camera,
            "side_3d_evidence": side_evidence,
        }

    reviewed_2d_evidence = _reviewed_calibrated_2d_evidence(evidence)
    if category in {"review", "borderline_below_threshold"} and reviewed_2d_evidence[
        "eligible"
    ]:
        return {
            "route": "calibrated_static_sphere",
            "camera_motion_category": category,
            "raw_camera_motion_category": category,
            "effective_camera_motion_category": CALIBRATED_2D_EFFECTIVE_CATEGORY,
            "reason": "reviewed_motion_within_calibrated_2d_tolerance",
            "static_calibration_allowed": True,
            "calibrated_2d_approximation": True,
            "camera": resolved_camera,
            "calibrated_2d_evidence": reviewed_2d_evidence,
        }

    reason = (
        "camera_motion_audit_detected_change"
        if category in {"changed", "camera_changed"}
        else "camera_motion_not_confidently_static"
    )
    return {
        "route": "spatialtrackerv2_dynamic",
        "camera_motion_category": category,
        "raw_camera_motion_category": category,
        "effective_camera_motion_category": category,
        "reason": reason,
        "static_calibration_allowed": False,
        "camera": resolved_camera,
        "calibrated_2d_evidence": reviewed_2d_evidence,
    }


__all__ = [
    "DYNAMIC_CATEGORIES",
    "CALIBRATED_2D_EFFECTIVE_CATEGORY",
    "CALIBRATED_2D_MAX_RESIDUAL_DIAGONAL_FRACTION",
    "CALIBRATED_2D_MAX_GLOBAL_ROTATION_DEG",
    "CALIBRATED_2D_MAX_GLOBAL_SCALE_CHANGE",
    "CALIBRATED_2D_MAX_GLOBAL_TRANSLATION_DIAGONAL_FRACTION",
    "CALIBRATED_2D_MAX_ROTATION_DEG",
    "CALIBRATED_2D_MAX_SCALE_CHANGE",
    "CALIBRATED_2D_MAX_TRANSLATION_DIAGONAL_FRACTION",
    "CALIBRATED_2D_MIN_MEDIAN_INLIER_RATIO",
    "CALIBRATED_2D_MIN_VALID_PAIR_FRACTION",
    "SIDE_2D_EFFECTIVE_CATEGORY",
    "SIDE_3D_MIN_DIRECT_MOTION_CLUSTER",
    "SIDE_3D_MIN_MEDIAN_INLIER_RATIO",
    "SIDE_3D_MIN_TRANSLATION_DIAGONAL_FRACTION",
    "SIDE_3D_MIN_VALID_PAIR_FRACTION",
    "STATIC_CATEGORIES",
    "camera_motion_category",
    "choose_reconstruction_route",
]
