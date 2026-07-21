"""Camera-motion routing shared by calibrated and 4D reconstruction paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


STATIC_CATEGORIES = {"fixed", "no_significant_camera_change"}
DYNAMIC_CATEGORIES = {
    "changed",
    "camera_changed",
    "review",
    "borderline_below_threshold",
    "error",
}


def camera_motion_category(evidence: dict[str, Any] | None) -> str | None:
    """Return a normalized audit category without guessing missing evidence."""

    if not evidence:
        return None
    category = evidence.get("final_category") or evidence.get("decision")
    return None if category is None else str(category)


def choose_reconstruction_route(
    evidence: dict[str, Any] | None,
    video_filename: str | Path,
) -> dict[str, Any]:
    """Choose the fast calibrated path only from matching, positive evidence.

    Anything missing, borderline, failed, or explicitly changed is sent to the
    dynamic 3D/4D path.  This makes the expensive route conservative: a weak
    camera-motion detector can cost runtime, but cannot silently make static
    calibration valid for a moving camera.
    """

    filename = Path(video_filename).name
    if not evidence:
        return {
            "route": "spatialtrackerv2_dynamic",
            "camera_motion_category": None,
            "reason": "missing_camera_motion_evidence",
            "static_calibration_allowed": False,
        }

    evidence_filename = evidence.get("filename")
    if evidence_filename is None and evidence.get("video"):
        evidence_filename = Path(str(evidence["video"])).name
    if evidence_filename != filename:
        return {
            "route": "spatialtrackerv2_dynamic",
            "camera_motion_category": camera_motion_category(evidence),
            "reason": "camera_motion_evidence_filename_mismatch",
            "static_calibration_allowed": False,
            "evidence_filename": evidence_filename,
        }

    category = camera_motion_category(evidence)
    if evidence.get("status") == "error" or category == "error":
        return {
            "route": "spatialtrackerv2_dynamic",
            "camera_motion_category": category,
            "reason": "camera_motion_audit_error",
            "static_calibration_allowed": False,
        }
    if category in STATIC_CATEGORIES:
        return {
            "route": "calibrated_static_sphere",
            "camera_motion_category": category,
            "reason": "camera_motion_audit_confirmed_no_significant_change",
            "static_calibration_allowed": True,
        }

    reason = (
        "camera_motion_audit_detected_change"
        if category in {"changed", "camera_changed"}
        else "camera_motion_not_confidently_static"
    )
    return {
        "route": "spatialtrackerv2_dynamic",
        "camera_motion_category": category,
        "reason": reason,
        "static_calibration_allowed": False,
    }


__all__ = [
    "DYNAMIC_CATEGORIES",
    "STATIC_CATEGORIES",
    "camera_motion_category",
    "choose_reconstruction_route",
]
