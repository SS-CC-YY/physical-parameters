"""Camera-motion routing shared by calibrated and 4D reconstruction paths.

Side-view videos are treated specially.  Their frozen Blender camera is close
to orthographic, so a calibrated image-plane trajectory is the primary signal.
We only pay for a dynamic 3D reconstruction when the camera audit measures
persistent, high-quality translation large enough to provide useful parallax.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


STATIC_CATEGORIES = {"fixed", "no_significant_camera_change"}
SIDE_2D_EFFECTIVE_CATEGORY = "side_2d_motion_within_tolerance"
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
SIDE_3D_MIN_TRANSLATION_DIAGONAL_FRACTION = 0.01
SIDE_3D_MIN_DIRECT_MOTION_CLUSTER = 5
SIDE_3D_MIN_VALID_PAIR_FRACTION = 0.80
SIDE_3D_MIN_MEDIAN_INLIER_RATIO = 0.70


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


def choose_reconstruction_route(
    evidence: dict[str, Any] | None,
    video_filename: str | Path,
    *,
    camera_name: str | None = None,
) -> dict[str, Any]:
    """Choose the fast calibrated path only from matching, positive evidence.

    Anything missing, borderline, failed, or explicitly changed is sent to the
    dynamic 3D/4D path.  This makes the expensive route conservative: a weak
    camera-motion detector can cost runtime, but cannot silently make static
    calibration valid for a moving camera.
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
    }


__all__ = [
    "DYNAMIC_CATEGORIES",
    "SIDE_2D_EFFECTIVE_CATEGORY",
    "SIDE_3D_MIN_DIRECT_MOTION_CLUSTER",
    "SIDE_3D_MIN_MEDIAN_INLIER_RATIO",
    "SIDE_3D_MIN_TRANSLATION_DIAGONAL_FRACTION",
    "SIDE_3D_MIN_VALID_PAIR_FRACTION",
    "STATIC_CATEGORIES",
    "camera_motion_category",
    "choose_reconstruction_route",
]
