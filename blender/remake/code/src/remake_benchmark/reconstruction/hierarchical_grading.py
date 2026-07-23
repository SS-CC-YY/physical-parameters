"""Auditable G0--G4 hierarchical grading over frozen trajectory results.

Scope is intentionally explicit:

* G0, G1, PASS_TO_SCAN and U are per-video gates.
* G2, G3, G4 and U are matched parameter-scan conclusions.

No single video is ever labelled G2/G3/G4.  U is an evidence state and is not
ordered below G0.  Parameter targets are used only after target-free tracking,
reconstruction, G0/G1 checks and inverse fitting have completed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contact_geometry import evaluate_contact_geometry
from .generation_validity import evaluate_generation_validity
from .motion_type_validity import evaluate_motion_type
from .physics_parameters import fit_physics_parameters


GRADE_SCHEMA_VERSION = "1.0.0"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _typed(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "nan", ""}:
        return None
    number = _number(stripped)
    return number if number is not None else value


def read_trajectory_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: _typed(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def resolve_trajectory_path(result: Mapping[str, Any]) -> Path | None:
    for key in ("source_trajectory", "trajectory_csv"):
        value = result.get(key)
        if value and Path(str(value)).is_file():
            return Path(str(value))
    source_extraction = result.get("source_extraction")
    if source_extraction:
        candidate = Path(str(source_extraction)).parent / "trajectory_frames.csv"
        if candidate.is_file():
            return candidate
    return None


def _result_job_id(result: Mapping[str, Any]) -> str:
    job = result.get("job", {})
    video_name = job.get("video_name")
    if video_name:
        return Path(str(video_name)).stem
    video_path = job.get("video_path")
    if video_path:
        return Path(str(video_path)).stem
    fields = (
        job.get("experiment_id"),
        job.get("parameter_tuple_id"),
        job.get("scene_id"),
        job.get("object_id"),
        job.get("camera_name"),
        job.get("seed"),
    )
    if all(value is not None for value in fields):
        return "__".join(str(value) for value in fields)
    raise ValueError("result has no stable job id")


def _existing_visuals(result: Mapping[str, Any]) -> dict[str, Any]:
    visuals = dict(result.get("visual_evidence", {}))
    source = result.get("source_extraction")
    if source and Path(str(source)).is_file():
        source_dir = Path(str(source)).parent
        try:
            extraction = load_json(Path(str(source)))
        except (OSError, ValueError, json.JSONDecodeError):
            extraction = {}
        for key in (
            "object_track_overlay",
            "validity_object_track_overlay",
            "validity_evidence_contact_sheet",
        ):
            if extraction.get(key) and key not in visuals:
                visuals[key] = extraction[key]
        for key, value in extraction.get("visual_evidence", {}).items():
            visuals.setdefault(key, value)
        for key, filename in (
            ("object_track_overlay", "object_track_overlay.mp4"),
            ("validity_object_track_overlay", "validity_object_track_overlay.mp4"),
            ("validity_evidence_contact_sheet", "validity_evidence_contact_sheet.jpg"),
        ):
            candidate = source_dir / filename
            if candidate.is_file():
                visuals.setdefault(key, str(candidate))
    return visuals


def _fit_series_records(value: Any, *, path: str = "diagnostics") -> list[dict[str, Any]]:
    """Return every persisted inverse-model series with its quality fields."""

    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("fit_series"), Mapping):
            series = value["fit_series"]
            records.append(
                {
                    "path": path,
                    "series_name": series.get("series_name"),
                    "fit_points": int(value.get("fit_points", len(series.get("observed", []))) or 0),
                    "fit_nrmse": _number(value.get("fit_nrmse")),
                    "fit_r2": _number(value.get("fit_r2")),
                }
            )
        for key, item in value.items():
            if key != "fit_series":
                records.extend(_fit_series_records(item, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            records.extend(_fit_series_records(item, path=f"{path}[{index}]"))
    return records


def _parameter_fit_evidence(
    fit: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Normalize new parameter-specific fit evidence with a legacy fallback.

    ``metrics.fit_complete`` describes the whole experiment fit.  It cannot be
    used as the admission flag for a single target in a multi-parameter
    experiment: one identifiable parameter must not be discarded merely
    because another parameter is unavailable.  New fitters persist
    ``parameter_attribution``; old result JSON is interpreted from the
    historical ``parameter_observed`` contract.
    """

    estimates = fit.get("parameter_estimates", {})
    raw_estimates = fit.get("raw_parameter_estimates", {})
    observed = fit.get("parameter_observed", {})
    attribution = fit.get("parameter_attribution", {})
    metric_parameters = metrics.get("parameters", {})
    mappings = [
        value
        for value in (estimates, raw_estimates, observed, attribution, metric_parameters)
        if isinstance(value, Mapping)
    ]
    names = sorted({str(name) for value in mappings for name in value})
    output: dict[str, dict[str, Any]] = {}
    accepted_statuses = {"pass", "ok", "accepted"}
    for name in names:
        estimate = _number(estimates.get(name)) if isinstance(estimates, Mapping) else None
        raw_estimate = _number(raw_estimates.get(name)) if isinstance(raw_estimates, Mapping) else None
        observed_declared = isinstance(observed, Mapping) and name in observed
        observed_value = observed.get(name) is True if observed_declared else False
        metric_entry = metric_parameters.get(name, {}) if isinstance(metric_parameters, Mapping) else {}
        if not observed_declared and isinstance(metric_entry, Mapping) and "observed" in metric_entry:
            observed_declared = True
            observed_value = metric_entry.get("observed") is True

        attribution_entry = attribution.get(name, {}) if isinstance(attribution, Mapping) else {}
        attribution_declared = isinstance(attribution_entry, Mapping) and bool(attribution_entry)
        attribution_status = (
            str(attribution_entry.get("status", "")).strip().lower()
            if attribution_declared
            else None
        )
        # Some early result files have only parameter_observed.  Preserve that
        # contract unless the newer attribution field explicitly rejects it.
        attribution_accepted = (
            attribution_status in accepted_statuses
            if attribution_declared and attribution_status
            else True
        )
        reason_codes = []
        if attribution_declared:
            values = attribution_entry.get("reason_codes", [])
            if isinstance(values, str):
                values = [values]
            reason_codes = [str(value) for value in values if str(value)]
        admissible = bool(observed_value and attribution_accepted)
        output[name] = {
            "parameter_name": name,
            "observed": bool(observed_value),
            "observed_declared": bool(observed_declared),
            "estimate_available": estimate is not None,
            "raw_estimate_available": raw_estimate is not None,
            "attribution_status": attribution_status,
            "attribution_declared": attribution_declared,
            "attribution_reason_codes": reason_codes,
            "admissible": admissible,
            "scan_ready": bool(admissible and estimate is not None),
        }
    return output


def _inverse_model_mismatch_evidence(fit: Mapping[str, Any]) -> dict[str, Any]:
    """Identify an explicit dynamics-rule mismatch, not missing evidence."""

    fit_status = str(fit.get("status", "")).strip().lower()
    family = fit.get("rule_family_evaluation", {})
    family = family if isinstance(family, Mapping) else {}
    family_status = str(family.get("status", "")).strip().lower()
    validity = fit.get("fit_validity", {})
    validity = validity if isinstance(validity, Mapping) else {}
    validity_status = str(validity.get("status", "")).strip().lower()
    validity_category = str(validity.get("category", "")).strip().lower()
    explicit_mismatch = (
        fit_status == "model_mismatch"
        or family_status == "fail"
        or (
            validity_status in {"fail", "failed", "rejected"}
            and validity_category in {"model_mismatch", "rule_family_mismatch"}
        )
    )
    family_reasons = family.get("reason_codes", [])
    if isinstance(family_reasons, str):
        family_reasons = [family_reasons]
    reasons = ["inverse_rule_family_model_mismatch"] if explicit_mismatch else []
    reasons.extend(str(value) for value in family_reasons if str(value))
    primary = validity.get("primary_reason_code")
    if explicit_mismatch and primary:
        reasons.append(str(primary))
    secondary = validity.get("secondary_reason_codes", [])
    if isinstance(secondary, str):
        secondary = [secondary]
    if explicit_mismatch:
        reasons.extend(str(value) for value in secondary if str(value))
    return {
        "is_model_mismatch": explicit_mismatch,
        "status": "model_mismatch" if explicit_mismatch else "not_detected",
        "reason_codes": list(dict.fromkeys(reasons)),
        "fit_status": fit.get("status"),
        "rule_family_status": family.get("status"),
        "fit_validity_status": validity.get("status"),
        "fit_validity_category": validity.get("category"),
    }


def _assess_inverse_fit_evidence(
    result: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate scan admission on target-free, sufficiently supported inverse fits."""

    fit = dict(result.get("fit", {}))
    metrics = dict(result.get("metrics", {}))
    thresholds = dict(policy.get("video_gate", {}))
    reasons: list[str] = []
    if fit.get("target_not_used_for_fit") is not True:
        reasons.append("inverse_fit_target_independence_not_proven")
    parameter_evidence = _parameter_fit_evidence(fit, metrics)
    admissible_parameters = sorted(
        name for name, evidence in parameter_evidence.items() if evidence.get("admissible") is True
    )
    # A global incomplete flag is expected for a valid partial fit.  It blocks
    # admission only when no individual parameter has usable evidence.
    if metrics.get("fit_complete") is not True and not admissible_parameters:
        reasons.append("inverse_fit_incomplete")

    total_rows = len(trajectory_rows)
    fit_rows = [
        row
        for row in trajectory_rows
        if _truth(row.get("physics_fit_used")) or _truth(row.get("fit_eligible"))
    ]
    fit_measurements = len(fit_rows)
    fit_fraction = fit_measurements / total_rows if total_rows else 0.0
    minimum_measurements = int(thresholds.get("minimum_fit_measurements", 0))
    minimum_fraction = float(thresholds.get("minimum_fit_eligible_fraction", 0.0))
    if fit_measurements < minimum_measurements:
        reasons.append("insufficient_inverse_fit_measurements")
    if fit_fraction + 1e-12 < minimum_fraction:
        reasons.append("insufficient_inverse_fit_fraction")

    diagnostics_source = "persisted_inverse_fit"
    diagnostics = fit.get("diagnostics", {})
    series = _fit_series_records(diagnostics)
    recompute_error: str | None = None
    if not series and trajectory_rows and fit.get("target_not_used_for_fit") is True:
        # Backward-compatible evidence recovery for old result JSON and small
        # synthetic fixtures.  This reruns only the target-free inverse model;
        # it never looks up the requested registry tuple.
        try:
            recomputed = fit_physics_parameters(experiment_id, trajectory_rows)
            series = _fit_series_records(recomputed.get("diagnostics", {}))
            diagnostics_source = "recomputed_target_free_inverse_fit"
        except Exception as error:  # Evidence failure becomes U, not a crash.
            recompute_error = f"{type(error).__name__}: {error}"
    if not series:
        reasons.append("inverse_fit_series_evidence_missing")

    missing_nrmse = [item["path"] for item in series if item.get("fit_nrmse") is None]
    missing_r2 = [item["path"] for item in series if item.get("fit_r2") is None]
    if missing_nrmse:
        reasons.append("inverse_fit_nrmse_missing")
    if missing_r2:
        reasons.append("inverse_fit_r2_missing")
    nrmse_values = [float(item["fit_nrmse"]) for item in series if item.get("fit_nrmse") is not None]
    r2_values = [float(item["fit_r2"]) for item in series if item.get("fit_r2") is not None]
    maximum_nrmse = float(thresholds.get("maximum_motion_model_nrmse", float("inf")))
    minimum_r2 = float(thresholds.get("minimum_motion_model_r2", -float("inf")))
    if nrmse_values and max(nrmse_values) > maximum_nrmse:
        reasons.append("inverse_fit_nrmse_exceeds_limit")
    if r2_values and min(r2_values) < minimum_r2:
        reasons.append("inverse_fit_r2_below_limit")
    return {
        "status": "pass" if not reasons else "indeterminate",
        "reason_codes": list(dict.fromkeys(reasons)),
        "target_not_used_for_fit": fit.get("target_not_used_for_fit"),
        "fit_complete": metrics.get("fit_complete"),
        "fit_completeness_scope": (
            "complete"
            if metrics.get("fit_complete") is True
            else "partial_parameters"
            if admissible_parameters
            else "none"
        ),
        "parameter_evidence": parameter_evidence,
        "admissible_parameters": admissible_parameters,
        "fit_measurement_count": fit_measurements,
        "trajectory_row_count": total_rows,
        "fit_eligible_fraction": fit_fraction,
        "minimum_fit_measurements": minimum_measurements,
        "minimum_fit_eligible_fraction": minimum_fraction,
        "series_evidence_source": diagnostics_source,
        "series": series,
        "per_job_worst_fit_nrmse": max(nrmse_values) if nrmse_values else None,
        "per_job_worst_fit_r2": min(r2_values) if r2_values else None,
        "maximum_allowed_fit_nrmse": maximum_nrmse,
        "minimum_allowed_fit_r2": minimum_r2,
        "missing_nrmse_series": missing_nrmse,
        "missing_r2_series": missing_r2,
        "recompute_error": recompute_error,
    }


def _normalize_adjudication_override(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    status = str(value.get("g0_status", "")).strip().lower()
    if status not in {"pass", "fail", "indeterminate"}:
        raise ValueError("adjudication_override.g0_status must be pass, fail, or indeterminate")
    reasons = value.get("reason_codes", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    paths = value.get("evidence_paths", {})
    if not isinstance(paths, Mapping):
        raise ValueError("adjudication_override.evidence_paths must be an object")
    return {
        "g0_status": status,
        "reason_codes": [str(reason) for reason in reasons if str(reason)],
        "reviewer": value.get("reviewer"),
        "note": value.get("note"),
        "evidence_paths": {str(key): item for key, item in paths.items()},
    }


def grade_video_result(
    result: Mapping[str, Any],
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    motion_profile: Mapping[str, Any],
    adjudication_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the G0 then G1 gates to one already-extracted trajectory."""

    job = dict(result.get("job", {}))
    job_id = _result_job_id(result)
    experiment_id = str(job.get("experiment_id"))
    automatic_generation = dict(result.get("video_generation_validity", {}))
    adjudication = _normalize_adjudication_override(adjudication_override)
    generation = dict(automatic_generation)
    if adjudication is not None:
        generation["status"] = adjudication["g0_status"]
        generation["adjudication_applied"] = True
        generation["automatic_status"] = automatic_generation.get("status")
    generation_status = str(generation.get("status", "indeterminate")).lower()
    contact = evaluate_contact_geometry(experiment_id, trajectory_rows, motion_profile)
    eligibility = dict(result.get("trajectory_fit_eligibility", {}))
    dynamic_rigidity: dict[str, Any] = {"status": "not_applicable"}
    if str(result.get("reconstruction_route")) == "spatialtrackerv2_dynamic":
        dynamic_payload = result.get("pipeline", {}).get("dynamic_reconstruction", {})
        rigidity = dynamic_payload.get("rigidity_evidence") if isinstance(dynamic_payload, Mapping) else None
        if isinstance(rigidity, Mapping):
            unified = evaluate_generation_validity(
                trajectory_rows,
                camera_motion_evidence=result.get("camera_motion_evidence", {}),
                trajectory_evidence=trajectory_rows,
                dynamic_rigidity_evidence=rigidity,
            )
            object_check = dict(unified.get("checks", {}).get("object_rigidity_3d", {}))
            scene_check = dict(unified.get("checks", {}).get("scene_rigidity_3d", {}))
            if "fail" in {object_check.get("status"), scene_check.get("status")}:
                dynamic_status = "fail"
            elif object_check.get("status") == "pass" and scene_check.get("status") == "pass":
                dynamic_status = "pass"
            else:
                dynamic_status = "indeterminate"
            dynamic_rigidity = {
                "status": dynamic_status,
                "object_rigidity_3d": object_check,
                "scene_rigidity_3d": scene_check,
                "failure_codes": [
                    code
                    for code in unified.get("failure_codes", [])
                    if code in {
                        "persistent_experiment_object_deformation_3d",
                        "persistent_non_experimental_scene_deformation_3d",
                    }
                ],
                "thresholds": {
                    "object_rigidity_3d": unified.get("thresholds", {}).get("object_rigidity_3d"),
                    "scene_rigidity_3d": unified.get("thresholds", {}).get("scene_rigidity_3d"),
                },
                "policy": "unified_generation_validity_3d_rigidity_thresholds",
            }
        else:
            dynamic_rigidity = {
                "status": "indeterminate",
                "indeterminate_codes": ["dynamic_route_rigidity_evidence_missing"],
            }

    reasons: list[str] = []
    fit_evidence: dict[str, Any] = {"status": "not_run", "reason_codes": []}
    if generation_status in {"fail", "failed"}:
        stage = "G0"
        if adjudication is not None:
            reasons.extend(adjudication["reason_codes"])
        else:
            reasons.extend(str(value) for value in generation.get("failure_codes", []))
        if not reasons:
            reasons.append("generation_validity_fail")
        motion = {
            "schema_version": "1.0.0",
            "status": "not_run",
            "reason": "blocked_by_g0",
            "target_parameters_used": False,
        }
    elif contact.get("status") == "fail":
        stage = "G0"
        reasons.extend(str(value) for value in contact.get("failure_codes", []))
        motion = {
            "schema_version": "1.0.0",
            "status": "not_run",
            "reason": "blocked_by_g0_contact_geometry",
            "target_parameters_used": False,
        }
    elif dynamic_rigidity.get("status") == "fail":
        stage = "G0"
        reasons.extend(str(value) for value in dynamic_rigidity.get("failure_codes", []))
        motion = {
            "schema_version": "1.0.0",
            "status": "not_run",
            "reason": "blocked_by_g0_dynamic_rigidity",
            "target_parameters_used": False,
        }
    elif generation_status != "pass":
        stage = "U"
        if adjudication is not None:
            reasons.extend(adjudication["reason_codes"])
        else:
            reasons.extend(str(value) for value in generation.get("indeterminate_codes", []))
        reasons.append("generation_validity_evidence_insufficient")
        motion = {
            "schema_version": "1.0.0",
            "status": "not_run",
            "reason": "blocked_by_indeterminate_g0",
            "target_parameters_used": False,
        }
    elif contact.get("status") == "indeterminate":
        stage = "U"
        reasons.extend(str(value) for value in contact.get("indeterminate_codes", []))
        motion = {
            "schema_version": "1.0.0",
            "status": "not_run",
            "reason": "contact_geometry_evidence_insufficient",
            "target_parameters_used": False,
        }
    elif dynamic_rigidity.get("status") == "indeterminate":
        stage = "U"
        reasons.extend(str(value) for value in dynamic_rigidity.get("indeterminate_codes", []))
        reasons.append("dynamic_rigidity_evidence_insufficient")
        motion = {
            "schema_version": "1.0.0",
            "status": "not_run",
            "reason": "dynamic_rigidity_evidence_insufficient",
            "target_parameters_used": False,
        }
    elif eligibility.get("eligible") is not True:
        stage = "U"
        reasons.append(str(eligibility.get("reason") or "trajectory_not_fit_eligible"))
        motion = {
            "schema_version": "1.0.0",
            "status": "not_run",
            "reason": "trajectory_evidence_insufficient",
            "target_parameters_used": False,
        }
    else:
        motion = evaluate_motion_type(experiment_id, trajectory_rows, motion_profile)
        if motion.get("status") == "fail":
            stage = "G1"
            reasons.extend(str(value) for value in motion.get("failure_codes", []))
        elif motion.get("status") == "indeterminate":
            stage = "U"
            reasons.extend(str(value) for value in motion.get("indeterminate_codes", []))
        else:
            fit = dict(result.get("fit", {}))
            mismatch = _inverse_model_mismatch_evidence(fit)
            if mismatch.get("is_model_mismatch") is True:
                # At this point G0, contact geometry, trajectory eligibility and
                # the coarse motion family have all passed.  An explicit
                # rule-family rejection is therefore observed model behaviour,
                # not an absence of evaluable evidence.
                stage = "G1"
                reasons.extend(str(value) for value in mismatch.get("reason_codes", []))
                fit_evidence = {
                    **mismatch,
                    "parameter_evidence": _parameter_fit_evidence(
                        fit,
                        dict(result.get("metrics", {})),
                    ),
                    "classification": "model_physics_failure",
                }
            else:
                fit_evidence = _assess_inverse_fit_evidence(
                    result,
                    trajectory_rows,
                    experiment_id=experiment_id,
                    policy=policy,
                )
                if fit_evidence.get("status") != "pass":
                    stage = "U"
                    reasons.extend(str(value) for value in fit_evidence.get("reason_codes", []))
                else:
                    stage = "PASS_TO_SCAN"
                    reasons.append("g0_g1_and_inverse_fit_evidence_pass")

    fit = dict(result.get("fit", {}))
    metrics = dict(result.get("metrics", {}))
    return {
        "schema_version": GRADE_SCHEMA_VERSION,
        "entity_scope": "single_video",
        "job_id": job_id,
        "job": job,
        "video_stage": stage,
        "terminal_grade": stage if stage in {"G0", "G1", "U"} else None,
        "eligible_for_parameter_scan": stage == "PASS_TO_SCAN",
        "reason_codes": list(dict.fromkeys(value for value in reasons if value)),
        "g0_generation_validity": generation,
        "g0_generation_validity_automatic": automatic_generation,
        "g0_generation_validity_adjudication": adjudication,
        "g0_contact_geometry": contact,
        "g0_dynamic_rigidity": dynamic_rigidity,
        "g1_motion_type_validity": motion,
        "inverse_fit_evidence": fit_evidence,
        "inverse_fit": {
            "status": fit.get("status"),
            "parameter_observed": fit.get("parameter_observed", {}),
            "parameter_attribution": fit.get("parameter_attribution", {}),
            "fit_validity": fit.get("fit_validity", {}),
            "rule_family_evaluation": fit.get("rule_family_evaluation", {}),
            "fit_complete": metrics.get("fit_complete"),
            "target_not_used_for_fit": fit.get("target_not_used_for_fit"),
        },
        "evidence_paths": {
            "trajectory_csv": str(resolve_trajectory_path(result)) if resolve_trajectory_path(result) else None,
            **_existing_visuals(result),
            **({f"adjudication_{key}": value for key, value in adjudication["evidence_paths"].items()} if adjudication else {}),
        },
        "policy_id": policy.get("policy_id"),
        "scope_note": "G2/G3/G4 require a matched parameter scan and are never assigned here.",
    }


def missing_video_grade(job_id: str, job: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": GRADE_SCHEMA_VERSION,
        "entity_scope": "single_video",
        "job_id": job_id,
        "job": dict(job),
        "video_stage": "U",
        "terminal_grade": "U",
        "eligible_for_parameter_scan": False,
        "reason_codes": ["evaluation_result_missing"],
        "g0_generation_validity": {"status": "indeterminate"},
        "g0_contact_geometry": {"status": "not_run"},
        "g0_dynamic_rigidity": {"status": "not_run"},
        "g1_motion_type_validity": {"status": "not_run"},
        "inverse_fit": {"status": "not_run", "fit_complete": False},
        "evidence_paths": {},
        "policy_id": policy.get("policy_id"),
        "scope_note": "Missing evidence is U, not a model failure.",
    }


def _registry_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for experiment in registry.get("experiments", []):
        copied = dict(experiment)
        copied["parameters_by_name"] = {
            str(parameter["name"]): dict(parameter)
            for parameter in experiment.get("hidden_parameters", [])
        }
        copied["anchors_by_id"] = {
            str(anchor["id"]): {
                str(key): float(value)
                for key, value in anchor.items()
                if key != "id"
            }
            for anchor in experiment.get("anchor_tuples", [])
        }
        output[str(experiment["id"])] = copied
    return output


def _manifest_job(row: Mapping[str, Any]) -> dict[str, Any]:
    factors = dict(row.get("factors", {}))
    return {
        "experiment_id": str(row.get("experiment_id") or factors.get("experiment_id")),
        "parameter_tuple_id": str(factors.get("parameter_tuple_id") or row.get("parameter_tuple_id")),
        "scene_id": str(factors.get("scene_id") or factors.get("scene") or row.get("scene_id")),
        "object_id": str(factors.get("object_id") or factors.get("object") or row.get("object_id") or "standard_ball"),
        "camera_name": str(factors.get("camera") or factors.get("camera_name") or row.get("camera_name")),
        "seed": int(row.get("seed", factors.get("seed", 0))),
        "video_name": f"{row['job_id']}.mp4",
    }


def _reference_anchor(spec: Mapping[str, Any]) -> dict[str, float]:
    anchors = dict(spec.get("anchors_by_id", {}))
    if "default" in anchors:
        return dict(anchors["default"])
    parameters = spec.get("parameters_by_name", {})
    def distance(anchor: Mapping[str, float]) -> float:
        values: list[float] = []
        for name, parameter in parameters.items():
            low, high = (float(value) for value in parameter["valid_range"])
            midpoint = 0.5 * (low + high)
            span = max(high - low, 1e-12)
            values.append(abs(float(anchor[name]) - midpoint) / span)
        return float(sum(values))
    return dict(min(anchors.values(), key=distance)) if anchors else {}


def _signature_id(values: Mapping[str, float]) -> str:
    text = json.dumps({key: values[key] for key in sorted(values)}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end + 1) / 2.0
        for position in range(cursor, end):
            output[order[position]] = rank
        cursor = end
    return output


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(first - first.mean()) * np.linalg.norm(second - second.mean()))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first - first.mean(), second - second.mean()) / denominator)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_ranks(left), _ranks(right))


def _theil_sen_slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    slopes = [
        (float(y[right]) - float(y[left])) / (float(x[right]) - float(x[left]))
        for left in range(len(x))
        for right in range(left + 1, len(x))
        if abs(float(x[right]) - float(x[left])) > 1e-12
    ]
    return float(np.median(slopes)) if slopes else None


def _bootstrap_median_ci(values: Sequence[float], *, seed: int, samples: int = 1000) -> list[float] | None:
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    random = np.random.default_rng(seed)
    medians = np.median(random.choice(array, size=(samples, len(array)), replace=True), axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def _bootstrap_matched_block_slope_ci(
    blocks: Sequence[Mapping[float, float]],
    targets: Sequence[float],
    *,
    seed: int,
    samples: int = 1000,
) -> list[float] | None:
    """Bootstrap whole scene blocks, preserving every within-scene contrast."""

    if len(blocks) < 2 or len(targets) < 2:
        return None
    random = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(samples):
        selected = random.integers(0, len(blocks), size=len(blocks))
        estimates = [
            float(np.median([float(blocks[index][target]) for index in selected]))
            for target in targets
        ]
        slope = _theil_sen_slope(targets, estimates)
        if slope is not None and math.isfinite(slope):
            slopes.append(float(slope))
    if not slopes:
        return None
    return [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]


def _fit_nrmse_values(value: Any) -> list[float]:
    output: list[float] = []
    if isinstance(value, Mapping):
        nrmse = _number(value.get("fit_nrmse"))
        if nrmse is not None:
            output.append(nrmse)
        for key, item in value.items():
            if key != "fit_series":
                output.extend(_fit_nrmse_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            output.extend(_fit_nrmse_values(item))
    return output


def _score_response(
    levels: Sequence[Mapping[str, Any]],
    parameter_spec: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    slope_ci95: Sequence[float] | None = None,
    require_slope_ci: bool = False,
) -> dict[str, Any]:
    thresholds = dict(policy["parameter_response"])
    low, high = (float(value) for value in parameter_spec["valid_range"])
    valid_span = max(high - low, 1e-12)
    ordered_levels = sorted(levels, key=lambda level: float(level["target_value"]))
    planned = [float(level["target_value"]) for level in ordered_levels]
    usable_levels = [level for level in ordered_levels if level.get("median_estimate") is not None]
    targets = [float(level["target_value"]) for level in usable_levels]
    estimates = [float(level["median_estimate"]) for level in usable_levels]
    planned_jobs = sum(int(level.get("planned_job_count", 0)) for level in levels)
    usable_jobs = sum(int(level.get("usable_job_count", 0)) for level in levels)
    coverage = len(usable_levels) / max(len(levels), 1)
    job_coverage = usable_jobs / max(planned_jobs, 1)
    evidence = {
        "planned_level_count": len(planned),
        "usable_level_count": len(usable_levels),
        "planned_job_count": planned_jobs,
        "usable_job_count": usable_jobs,
        "level_coverage_fraction": coverage,
        "job_coverage_fraction": job_coverage,
    }
    u_reasons: list[str] = []
    if len(planned) < int(thresholds["minimum_planned_levels"]):
        u_reasons.append("fewer_than_minimum_planned_levels")
    if len(usable_levels) < int(thresholds["minimum_usable_levels"]):
        u_reasons.append("fewer_than_minimum_usable_levels")
    # JSON cannot represent 2/3 exactly.  A numerical tolerance must not turn
    # the documented >=2/3 rule into a strict >2/3 rule.
    coverage_tolerance = 1e-9
    if coverage + coverage_tolerance < float(thresholds["minimum_coverage_fraction"]):
        u_reasons.append("insufficient_level_coverage")
    if job_coverage + coverage_tolerance < float(thresholds.get("minimum_job_coverage_fraction", 0.0)):
        u_reasons.append("insufficient_job_coverage")
    if require_slope_ci and (slope_ci95 is None or len(slope_ci95) != 2):
        u_reasons.append("matched_block_slope_ci_unavailable")
    if u_reasons:
        return {
            "grade": "U",
            "reason_codes": u_reasons,
            "evidence": evidence,
            "direction": {},
            "numeric_accuracy": {},
            "threshold_snapshot": thresholds,
        }

    deadband = float(thresholds["flat_response_valid_range_fraction"]) * valid_span
    concordant = discordant = ties = 0
    for left in range(len(targets)):
        for right in range(left + 1, len(targets)):
            delta = estimates[right] - estimates[left]
            if delta > deadband:
                concordant += 1
            elif delta < -deadband:
                discordant += 1
            else:
                ties += 1
    comparisons = concordant + discordant + ties
    concordance = concordant / max(comparisons, 1)
    spearman = _spearman(targets, estimates)
    slope = _theil_sen_slope(targets, estimates)
    response_span = (max(estimates) - min(estimates)) / valid_span
    flat = response_span < float(thresholds["flat_response_valid_range_fraction"])
    direction_checks = {
        "not_flat": not flat,
        "positive_slope": slope is not None and slope > float(thresholds["minimum_positive_slope"]),
        "pairwise_concordance": concordance >= float(thresholds["minimum_pairwise_direction_concordance"]),
        "spearman_for_three_or_more": (
            True
            if len(targets) < 3
            else spearman is not None and spearman >= float(thresholds["minimum_spearman_rho_for_three_levels"])
        ),
    }
    if require_slope_ci:
        direction_checks["positive_matched_block_slope_ci"] = bool(
            slope_ci95 is not None and len(slope_ci95) == 2 and float(slope_ci95[0]) > 0.0
        )
    direction_pass = all(direction_checks.values())
    direction = {
        "expected_relationship": "estimated parameter increases with requested value",
        "pairwise_comparison_count": comparisons,
        "pairwise_concordant_count": concordant,
        "pairwise_discordant_count": discordant,
        "pairwise_tie_count": ties,
        "pairwise_direction_concordance": concordance,
        "spearman_rho": spearman,
        "theil_sen_slope": slope,
        "theil_sen_slope_ci95": None if slope_ci95 is None else [float(value) for value in slope_ci95],
        "response_span_over_valid_range": response_span,
        "flat_response": flat,
        "checks": direction_checks,
        "passed": direction_pass,
    }
    if not direction_pass:
        failed = [name for name, passed in direction_checks.items() if not passed]
        return {
            "grade": "G2",
            "reason_codes": [f"direction_gate_failed:{name}" for name in failed],
            "evidence": evidence,
            "direction": direction,
            "numeric_accuracy": {},
            "threshold_snapshot": thresholds,
        }

    per_job_errors = [
        float(value)
        for level in usable_levels
        for value in level.get("per_job_normalized_errors", [])
        if _number(value) is not None
    ]
    errors = (
        per_job_errors
        if per_job_errors
        else [(estimate - target) / valid_span for target, estimate in zip(targets, estimates)]
    )
    absolute = [abs(value) for value in errors]
    median_nae = float(np.median(absolute))
    p90_nae = float(np.percentile(absolute, 90))
    normalized_bias = float(np.median(errors))
    slope_error = None if slope is None else abs(float(slope) - 1.0)
    per_job_worst_fit_nrmse = [
        float(value)
        for level in usable_levels
        for value in level.get("per_job_worst_fit_nrmse", level.get("fit_nrmse_values", []))
        if _number(value) is not None
    ]
    p90_fit_nrmse = float(np.percentile(per_job_worst_fit_nrmse, 90)) if per_job_worst_fit_nrmse else None
    maximum_job_fit_nrmse = max(per_job_worst_fit_nrmse) if per_job_worst_fit_nrmse else None
    maximum_fit_nrmse = float(policy["video_gate"]["maximum_motion_model_nrmse"])
    off_target_names = sorted({
        str(name)
        for level in usable_levels
        for name in level.get("off_target_valid_ranges", {})
    })
    off_target_drift: dict[str, float | None] = {}
    off_target_coverage: dict[str, dict[str, Any]] = {}
    for name in off_target_names:
        medians = [
            _number(level.get("off_target_median_estimates", {}).get(name))
            for level in usable_levels
        ]
        planned_off_target_jobs = sum(int(level.get("planned_job_count", 0)) for level in usable_levels)
        observed_off_target_jobs = sum(
            len(level.get("off_target_estimates", {}).get(name, []))
            or int(_number(level.get("off_target_median_estimates", {}).get(name)) is not None)
            for level in usable_levels
        )
        level_observed = sum(
            bool(level.get("off_target_estimates", {}).get(name, []))
            or _number(level.get("off_target_median_estimates", {}).get(name)) is not None
            for level in usable_levels
        )
        coverage_value = observed_off_target_jobs / max(planned_off_target_jobs, 1)
        off_target_coverage[name] = {
            "planned_job_count": planned_off_target_jobs,
            "observed_job_count": observed_off_target_jobs,
            "job_coverage_fraction": coverage_value,
            "observed_level_count": level_observed,
            "required_level_count": len(usable_levels),
        }
        if (
            any(value is None for value in medians)
            or level_observed != len(usable_levels)
            or coverage_value + coverage_tolerance
            < float(thresholds.get("minimum_job_coverage_fraction", 0.0))
        ):
            off_target_drift[name] = None
            continue
        valid_range = next(
            level.get("off_target_valid_ranges", {}).get(name)
            for level in usable_levels
            if name in level.get("off_target_valid_ranges", {})
        )
        other_span = max(float(valid_range[1]) - float(valid_range[0]), 1e-12)
        values = [float(value) for value in medians if value is not None]
        off_target_drift[name] = (max(values) - min(values)) / other_span
    if not off_target_names:
        off_target_check: bool | None = True
    elif any(value is None for value in off_target_drift.values()):
        off_target_check = None
    else:
        off_target_check = max(float(value) for value in off_target_drift.values() if value is not None) <= float(
            thresholds["g4_max_off_target_drift_nad"]
        )
    numeric_checks = {
        "median_valid_range_nae": median_nae <= float(thresholds["g4_median_valid_range_nae"]),
        "p90_valid_range_nae": p90_nae <= float(thresholds["g4_p90_valid_range_nae"]),
        "slope_calibration": slope_error is not None and slope_error <= float(thresholds["g4_max_absolute_slope_error"]),
        "normalized_bias": abs(normalized_bias) <= float(thresholds["g4_max_absolute_normalized_bias"]),
        "full_level_coverage": (
            not _truth(thresholds.get("g4_requires_full_level_coverage"))
            or len(usable_levels) == len(levels)
        ),
        "trajectory_model_residual": (
            None if p90_fit_nrmse is None else p90_fit_nrmse <= maximum_fit_nrmse
        ),
        "off_target_parameter_stability": off_target_check,
    }
    known_numeric_checks = {name: passed for name, passed in numeric_checks.items() if passed is not None}
    known_numeric_pass = all(passed is True for passed in known_numeric_checks.values())
    missing_numeric_evidence = [name for name, passed in numeric_checks.items() if passed is None]
    if known_numeric_pass and missing_numeric_evidence:
        return {
            "grade": "U",
            "reason_codes": [f"missing_numeric_evidence:{name}" for name in missing_numeric_evidence],
            "evidence": evidence,
            "direction": direction,
            "numeric_accuracy": {
                "median_valid_range_nae": median_nae,
                "p90_valid_range_nae": p90_nae,
                "normalized_bias": normalized_bias,
                "absolute_slope_error": slope_error,
                "p90_per_job_worst_trajectory_fit_nrmse": p90_fit_nrmse,
                "maximum_per_job_worst_trajectory_fit_nrmse": maximum_job_fit_nrmse,
                "maximum_allowed_trajectory_fit_nrmse": maximum_fit_nrmse,
                "off_target_drift_nad": off_target_drift,
                "off_target_coverage": off_target_coverage,
                "maximum_allowed_off_target_drift_nad": thresholds["g4_max_off_target_drift_nad"],
                "checks": numeric_checks,
                "passed": None,
            },
            "threshold_snapshot": thresholds,
        }
    numeric_pass = all(passed is True for passed in numeric_checks.values())
    numeric = {
        "median_valid_range_nae": median_nae,
        "p90_valid_range_nae": p90_nae,
        "normalized_bias": normalized_bias,
        "absolute_slope_error": slope_error,
        "p90_per_job_worst_trajectory_fit_nrmse": p90_fit_nrmse,
        "maximum_per_job_worst_trajectory_fit_nrmse": maximum_job_fit_nrmse,
        "maximum_allowed_trajectory_fit_nrmse": maximum_fit_nrmse,
        "off_target_drift_nad": off_target_drift,
        "off_target_coverage": off_target_coverage,
        "maximum_allowed_off_target_drift_nad": thresholds["g4_max_off_target_drift_nad"],
        "checks": numeric_checks,
        "passed": numeric_pass,
    }
    grade = "G4" if numeric_pass else "G3"
    failed = [name for name, passed in numeric_checks.items() if passed is False]
    return {
        "grade": grade,
        "reason_codes": ["direction_correct_and_numeric_accurate"] if grade == "G4" else [f"numeric_gate_failed:{name}" for name in failed],
        "evidence": evidence,
        "direction": direction,
        "numeric_accuracy": numeric,
        "threshold_snapshot": thresholds,
    }


def score_parameter_response(
    levels: Sequence[Mapping[str, Any]],
    parameter_spec: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Public, deterministic scorer used by tests and non-Seedance adapters."""

    return _score_response(levels, parameter_spec, policy)


def _level_summary(
    target: float,
    jobs: Sequence[Mapping[str, Any]],
    *,
    results_by_id: Mapping[str, Mapping[str, Any]],
    grades_by_id: Mapping[str, Mapping[str, Any]],
    parameter_name: str,
    off_target_specs: Mapping[str, Mapping[str, Any]],
    valid_span: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    planned_ids = [str(item["job_id"]) for item in jobs]
    stage_ids: dict[str, list[str]] = defaultdict(list)
    estimates: list[float] = []
    fit_nrmse_values: list[float] = []
    per_job_worst_fit_nrmse: list[float] = []
    off_target_estimates: dict[str, list[float]] = defaultdict(list)
    off_target_usable_job_ids: dict[str, list[str]] = defaultdict(list)
    usable_ids: list[str] = []
    scan_admission_rejections: dict[str, list[str]] = {}
    for job_id in planned_ids:
        video_grade = grades_by_id.get(job_id, {})
        stage = str(video_grade.get("video_stage", "U"))
        stage_ids[stage].append(job_id)
        result = results_by_id.get(job_id)
        if stage != "PASS_TO_SCAN" or result is None:
            continue
        fit = result.get("fit", {})
        metrics = result.get("metrics", {})
        admission_reasons: list[str] = []
        # Recheck the persisted fit contract here as well as in the per-video
        # gate.  This prevents a stale pre-v1 grade.json from admitting a
        # target-informed or otherwise unaudited fit when users run only the
        # response stage.
        if fit.get("target_not_used_for_fit") is not True:
            admission_reasons.append("inverse_fit_target_independence_not_proven")
        mismatch = _inverse_model_mismatch_evidence(fit)
        if mismatch.get("is_model_mismatch") is True:
            admission_reasons.append("inverse_rule_family_model_mismatch")
        parameter_evidence = _parameter_fit_evidence(fit, metrics)
        target_evidence = parameter_evidence.get(parameter_name, {})
        if target_evidence.get("observed") is not True:
            admission_reasons.append("target_parameter_not_observed")
        elif target_evidence.get("admissible") is not True:
            admission_reasons.append("target_parameter_quality_not_accepted")
        persisted_evidence = video_grade.get("inverse_fit_evidence", {})
        if persisted_evidence.get("status") not in {"pass", "partial"}:
            admission_reasons.append("inverse_fit_evidence_gate_not_passed")
        persisted_parameters = persisted_evidence.get("parameter_evidence", {})
        if isinstance(persisted_parameters, Mapping) and parameter_name in persisted_parameters:
            persisted_target = persisted_parameters.get(parameter_name, {})
            if not isinstance(persisted_target, Mapping) or persisted_target.get("admissible") is not True:
                admission_reasons.append("target_parameter_evidence_gate_not_passed")
        if admission_reasons:
            scan_admission_rejections[job_id] = list(dict.fromkeys(admission_reasons))
            continue
        estimate = _number(fit.get("parameter_estimates", {}).get(parameter_name))
        if estimate is None:
            scan_admission_rejections[job_id] = ["target_parameter_estimate_missing"]
            continue
        estimates.append(estimate)
        usable_ids.append(job_id)
        job_nrmse = _fit_nrmse_values(result.get("fit", {}).get("diagnostics", {}))
        if not job_nrmse:
            recovered_nrmse = _number(
                grades_by_id.get(job_id, {})
                .get("inverse_fit_evidence", {})
                .get("per_job_worst_fit_nrmse")
            )
            if recovered_nrmse is not None:
                job_nrmse = [recovered_nrmse]
        fit_nrmse_values.extend(job_nrmse)
        if job_nrmse:
            per_job_worst_fit_nrmse.append(max(job_nrmse))
        fitted_parameters = result.get("fit", {}).get("parameter_estimates", {})
        for name in off_target_specs:
            off_target_evidence = parameter_evidence.get(name, {})
            if off_target_evidence.get("admissible") is not True:
                continue
            off_target = _number(fitted_parameters.get(name))
            if off_target is not None:
                off_target_estimates[name].append(off_target)
                off_target_usable_job_ids[name].append(job_id)
    median = float(np.median(estimates)) if estimates else None
    return {
        "target_value": float(target),
        "planned_job_count": len(planned_ids),
        "usable_job_count": len(usable_ids),
        "planned_job_ids": planned_ids,
        "usable_job_ids": usable_ids,
        "g0_job_ids": stage_ids.get("G0", []),
        "g1_job_ids": stage_ids.get("G1", []),
        "u_job_ids": stage_ids.get("U", []),
        "scan_admission_rejected_job_ids": sorted(scan_admission_rejections),
        "scan_admission_rejection_reasons": scan_admission_rejections,
        "estimates": estimates,
        "median_estimate": median,
        "median_estimate_ci95": _bootstrap_median_ci(estimates, seed=bootstrap_seed),
        "median_valid_range_nae": None if median is None else abs(median - float(target)) / max(valid_span, 1e-12),
        "per_job_normalized_errors": [
            (float(estimate) - float(target)) / max(valid_span, 1e-12)
            for estimate in estimates
        ],
        "fit_nrmse_values": fit_nrmse_values,
        "per_job_worst_fit_nrmse": per_job_worst_fit_nrmse,
        "off_target_estimates": dict(off_target_estimates),
        "off_target_usable_job_ids": dict(off_target_usable_job_ids),
        "off_target_median_estimates": {
            name: float(np.median(values))
            for name, values in off_target_estimates.items()
            if values
        },
        "off_target_valid_ranges": {
            name: [float(value) for value in spec["valid_range"]]
            for name, spec in off_target_specs.items()
        },
    }


def build_parameter_scan_grades(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    results_by_id: Mapping[str, Mapping[str, Any]],
    grades_by_id: Mapping[str, Mapping[str, Any]],
    registry: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build every honest one-factor scan and mark one canonical scan per target."""

    specs = _registry_index(registry)
    scopes: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        job = _manifest_job(row)
        key = (
            job["experiment_id"],
            job["scene_id"],
            job["camera_name"],
            job["object_id"],
            int(job["seed"]),
        )
        scopes[key].append({"job_id": str(row["job_id"]), "job": job})

    output: list[dict[str, Any]] = []
    for scope, planned in sorted(scopes.items(), key=lambda item: tuple(str(value) for value in item[0])):
        experiment_id, scene_id, camera_name, object_id, seed = scope
        spec = specs.get(str(experiment_id))
        if spec is None:
            continue
        anchors = spec["anchors_by_id"]
        parameters = spec["parameters_by_name"]
        required_parameters = sorted(str(name) for name in parameters)
        reference = _reference_anchor(spec)
        candidates_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for parameter_name, parameter_spec in parameters.items():
            other_names = [name for name in parameters if name != parameter_name]
            families: dict[tuple[float, ...], list[dict[str, Any]]] = defaultdict(list)
            for item in planned:
                tuple_id = str(item["job"]["parameter_tuple_id"])
                target_values = anchors.get(tuple_id)
                if target_values is None or parameter_name not in target_values:
                    continue
                signature = tuple(float(target_values[name]) for name in other_names)
                families[signature].append({**item, "target_values": target_values})
            for signature, family_jobs in families.items():
                level_map: dict[float, list[dict[str, Any]]] = defaultdict(list)
                for item in family_jobs:
                    level_map[float(item["target_values"][parameter_name])].append(item)
                if len(level_map) < int(policy["parameter_response"]["minimum_planned_levels"]):
                    continue
                fixed = {name: float(value) for name, value in zip(other_names, signature)}
                off_target_specs = {name: parameters[name] for name in other_names}
                signature_id = _signature_id(fixed)
                low, high = (float(value) for value in parameter_spec["valid_range"])
                span = max(high - low, 1e-12)
                levels = [
                    _level_summary(
                        target,
                        jobs,
                        results_by_id=results_by_id,
                        grades_by_id=grades_by_id,
                        parameter_name=parameter_name,
                        off_target_specs=off_target_specs,
                        valid_span=span,
                        bootstrap_seed=int(hashlib.sha256(f"{scope}|{parameter_name}|{target}".encode()).hexdigest()[:8], 16),
                    )
                    for target, jobs in sorted(level_map.items())
                ]
                decision = _score_response(levels, parameter_spec, policy)
                reference_distance = sum(
                    abs(float(fixed[name]) - float(reference.get(name, fixed[name])))
                    / max(float(parameters[name]["valid_range"][1]) - float(parameters[name]["valid_range"][0]), 1e-12)
                    for name in fixed
                )
                scan_id = (
                    f"{experiment_id}__{parameter_name}__{scene_id}__{camera_name}__"
                    f"seed-{seed}__fixed-{signature_id}"
                )
                candidates_by_target[parameter_name].append(
                    {
                        "schema_version": GRADE_SCHEMA_VERSION,
                        "entity_scope": "parameter_scan",
                        "scan_id": scan_id,
                        "experiment_id": experiment_id,
                        "scene_id": scene_id,
                        "camera_name": camera_name,
                        "object_id": object_id,
                        "seed": int(seed),
                        "target_parameter": parameter_name,
                        "required_parameters": required_parameters,
                        "parameter_unit": parameter_spec.get("unit"),
                        "valid_range": [low, high],
                        "fixed_other_parameters": fixed,
                        "confounded_intervention": False,
                        "canonical_for_target": False,
                        "canonical_reference_distance": reference_distance,
                        "levels": levels,
                        "response_grade": decision["grade"],
                        "decision": decision,
                        "policy_id": policy.get("policy_id"),
                        "evidence_strength": (
                            "replicated_within_level" if any(len(level["estimates"]) >= 2 for level in levels)
                            else "descriptive_single_realization_per_level"
                        ),
                    }
                )
        for parameter_name in required_parameters:
            candidates = candidates_by_target.get(parameter_name, [])
            if not candidates:
                target_jobs: dict[float, list[str]] = defaultdict(list)
                for item in planned:
                    tuple_id = str(item["job"]["parameter_tuple_id"])
                    values = anchors.get(tuple_id, {})
                    if parameter_name in values:
                        target_jobs[float(values[parameter_name])].append(str(item["job_id"]))
                is_confounded = len(target_jobs) >= int(
                    policy["parameter_response"]["minimum_planned_levels"]
                )
                reason = (
                    "confounded_intervention_no_fixed_other_parameter_family"
                    if is_confounded
                    else "required_parameter_not_varied_in_scope"
                )
                levels = [
                    {
                        "target_value": float(target),
                        "planned_job_count": len(job_ids),
                        "usable_job_count": 0,
                        "planned_job_ids": sorted(job_ids),
                        "usable_job_ids": [],
                        "g0_job_ids": [],
                        "g1_job_ids": [],
                        "u_job_ids": sorted(job_ids),
                        "estimates": [],
                        "median_estimate": None,
                        "median_estimate_ci95": None,
                        "median_valid_range_nae": None,
                        "per_job_normalized_errors": [],
                        "fit_nrmse_values": [],
                        "per_job_worst_fit_nrmse": [],
                        "off_target_estimates": {},
                        "off_target_median_estimates": {},
                        "off_target_valid_ranges": {},
                    }
                    for target, job_ids in sorted(target_jobs.items())
                ]
                parameter_spec = parameters[parameter_name]
                low, high = (float(value) for value in parameter_spec["valid_range"])
                scan_id = (
                    f"{experiment_id}__{parameter_name}__{scene_id}__{camera_name}__"
                    f"seed-{seed}__unidentifiable"
                )
                candidates = [
                    {
                        "schema_version": GRADE_SCHEMA_VERSION,
                        "entity_scope": "parameter_scan",
                        "scan_id": scan_id,
                        "experiment_id": experiment_id,
                        "scene_id": scene_id,
                        "camera_name": camera_name,
                        "object_id": object_id,
                        "seed": int(seed),
                        "target_parameter": parameter_name,
                        "required_parameters": required_parameters,
                        "parameter_unit": parameter_spec.get("unit"),
                        "valid_range": [low, high],
                        "fixed_other_parameters": {},
                        "confounded_intervention": is_confounded,
                        "canonical_for_target": True,
                        "canonical_reference_distance": None,
                        "levels": levels,
                        "response_grade": "U",
                        "decision": {
                            "grade": "U",
                            "reason_codes": [reason],
                            "evidence": {
                                "planned_level_count": len(levels),
                                "usable_level_count": 0,
                                "planned_job_count": sum(len(ids) for ids in target_jobs.values()),
                                "usable_job_count": 0,
                                "level_coverage_fraction": 0.0,
                                "job_coverage_fraction": 0.0,
                            },
                            "direction": {},
                            "numeric_accuracy": {},
                            "threshold_snapshot": dict(policy["parameter_response"]),
                        },
                        "policy_id": policy.get("policy_id"),
                        "evidence_strength": "insufficient_or_confounded_design",
                    }
                ]
                candidates_by_target[parameter_name] = candidates
            canonical = min(
                candidates,
                key=lambda row: (
                    -len(row["levels"]),
                    float(row["canonical_reference_distance"])
                    if row.get("canonical_reference_distance") is not None
                    else float("inf"),
                    row["scan_id"],
                ),
            )
            canonical["canonical_for_target"] = True
            output.extend(candidates)
    return sorted(output, key=lambda row: row["scan_id"])


def build_cross_scene_parameter_scan_grades(
    scans: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    seed: int = 341867882,
) -> list[dict[str, Any]]:
    """Pool canonical matched scans across scenes for model-level CIs.

    Atomic scene grades remain the primary audit records.  This aggregation
    treats scenes as repeated benchmark conditions and bootstraps the median
    estimate at each requested level; it never mixes camera views or seeds.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for scan in scans:
        if not scan.get("canonical_for_target") or int(scan.get("seed", -1)) != int(seed):
            continue
        fixed = json.dumps(scan.get("fixed_other_parameters", {}), sort_keys=True, separators=(",", ":"))
        key = (
            scan.get("experiment_id"),
            scan.get("camera_name"),
            scan.get("object_id"),
            scan.get("target_parameter"),
            fixed,
        )
        groups[key].append(scan)

    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        experiment_id, camera_name, object_id, target_parameter, fixed_json = key
        fixed = json.loads(str(fixed_json))
        valid_range = [float(value) for value in items[0]["valid_range"]]
        valid_span = max(valid_range[1] - valid_range[0], 1e-12)
        targets = sorted({
            float(level["target_value"])
            for item in items
            for level in item.get("levels", [])
        })
        item_levels = {
            str(item["scan_id"]): {
                float(level["target_value"]): level
                for level in item.get("levels", [])
            }
            for item in items
        }
        matched_items = [
            item
            for item in items
            if all(
                target in item_levels[str(item["scan_id"])]
                and item_levels[str(item["scan_id"])][target].get("median_estimate") is not None
                for target in targets
            )
        ]
        matched_ids = {str(item["scan_id"]) for item in matched_items}
        matched_blocks = [
            {
                target: float(item_levels[str(item["scan_id"])][target]["median_estimate"])
                for target in targets
            }
            for item in matched_items
        ]
        slope_seed = int(hashlib.sha256(f"matched-slope|{key}".encode()).hexdigest()[:8], 16)
        slope_ci95 = _bootstrap_matched_block_slope_ci(
            matched_blocks,
            targets,
            seed=slope_seed,
        )

        levels: list[dict[str, Any]] = []
        for target in targets:
            all_parts = [
                item_levels[str(item["scan_id"])][target]
                for item in items
                if target in item_levels[str(item["scan_id"])]
            ]
            matched_parts = [
                item_levels[str(item["scan_id"])][target]
                for item in matched_items
            ]
            estimates = [float(value) for part in matched_parts for value in part.get("estimates", [])]
            off_target_names = sorted({
                str(name)
                for part in all_parts
                for name in part.get("off_target_valid_ranges", {})
            })
            off_target_estimates = {
                name: [
                    float(value)
                    for part in matched_parts
                    for value in part.get("off_target_estimates", {}).get(name, [])
                ]
                for name in off_target_names
            }
            off_target_ranges = {
                name: next(
                    [float(value) for value in part.get("off_target_valid_ranges", {})[name]]
                    for part in all_parts
                    if name in part.get("off_target_valid_ranges", {})
                )
                for name in off_target_names
            }
            planned_ids = sorted({str(value) for part in all_parts for value in part.get("planned_job_ids", [])})
            usable_ids = sorted({str(value) for part in matched_parts for value in part.get("usable_job_ids", [])})
            unmatched_usable_ids = sorted({
                str(value)
                for item in items
                if str(item["scan_id"]) not in matched_ids
                for value in item_levels[str(item["scan_id"])].get(target, {}).get("usable_job_ids", [])
            })
            median = float(np.median(estimates)) if estimates else None
            level_seed = int(hashlib.sha256(f"aggregate|{key}|{target}".encode()).hexdigest()[:8], 16)
            levels.append(
                {
                    "target_value": target,
                    "planned_job_count": len(planned_ids),
                    "usable_job_count": len(usable_ids),
                    "planned_job_ids": planned_ids,
                    "usable_job_ids": usable_ids,
                    "unmatched_usable_job_ids_excluded": unmatched_usable_ids,
                    "g0_job_ids": sorted({str(value) for part in all_parts for value in part.get("g0_job_ids", [])}),
                    "g1_job_ids": sorted({str(value) for part in all_parts for value in part.get("g1_job_ids", [])}),
                    "u_job_ids": sorted({str(value) for part in all_parts for value in part.get("u_job_ids", [])}),
                    "estimates": estimates,
                    "median_estimate": median,
                    "median_estimate_ci95": _bootstrap_median_ci(estimates, seed=level_seed),
                    "median_valid_range_nae": None if median is None else abs(median - target) / valid_span,
                    "per_job_normalized_errors": [
                        (float(value) - target) / valid_span for value in estimates
                    ],
                    "fit_nrmse_values": [
                        float(value)
                        for part in matched_parts
                        for value in part.get("fit_nrmse_values", [])
                    ],
                    "per_job_worst_fit_nrmse": [
                        float(value)
                        for part in matched_parts
                        for value in part.get("per_job_worst_fit_nrmse", part.get("fit_nrmse_values", []))
                    ],
                    "off_target_estimates": off_target_estimates,
                    "off_target_median_estimates": {
                        name: float(np.median(values))
                        for name, values in off_target_estimates.items()
                        if values
                    },
                    "off_target_valid_ranges": off_target_ranges,
                    "scene_count": len(all_parts),
                    "matched_scene_count": len(matched_parts),
                }
            )
        parameter_spec = {
            "name": target_parameter,
            "unit": items[0].get("parameter_unit"),
            "valid_range": valid_range,
        }
        decision = _score_response(
            levels,
            parameter_spec,
            policy,
            slope_ci95=slope_ci95,
            require_slope_ci=True,
        )
        if any(bool(item.get("confounded_intervention")) for item in items):
            decision = {
                "grade": "U",
                "reason_codes": ["confounded_intervention_no_fixed_other_parameter_family"],
                "evidence": decision.get("evidence", {}),
                "direction": {},
                "numeric_accuracy": {},
                "threshold_snapshot": dict(policy["parameter_response"]),
            }
        signature_id = _signature_id({str(name): float(value) for name, value in fixed.items()})
        scan_id = f"aggregate__{experiment_id}__{target_parameter}__{camera_name}__seed-{seed}__fixed-{signature_id}"
        scenes = sorted({str(item.get("scene_id")) for item in items})
        matched_scenes = sorted({str(item.get("scene_id")) for item in matched_items})
        output.append(
            {
                "schema_version": GRADE_SCHEMA_VERSION,
                "entity_scope": "cross_scene_parameter_scan",
                "scan_id": scan_id,
                "experiment_id": experiment_id,
                "scene_id": "ALL_SCENES",
                "camera_name": camera_name,
                "object_id": object_id,
                "seed": int(seed),
                "target_parameter": target_parameter,
                "required_parameters": list(items[0].get("required_parameters", [target_parameter])),
                "parameter_unit": items[0].get("parameter_unit"),
                "valid_range": valid_range,
                "fixed_other_parameters": fixed,
                "confounded_intervention": any(bool(item.get("confounded_intervention")) for item in items),
                "canonical_for_target": True,
                "levels": levels,
                "response_grade": decision["grade"],
                "decision": decision,
                "policy_id": policy.get("policy_id"),
                "evidence_strength": "matched_scene_block_bootstrap" if slope_ci95 is not None else "insufficient_matched_scene_blocks",
                "scene_count": len(scenes),
                "matched_scene_count": len(matched_scenes),
                "scenes": scenes,
                "matched_scenes": matched_scenes,
                "matched_block_bootstrap": {
                    "unit": "scene",
                    "samples": 1000,
                    "seed": slope_seed,
                    "theil_sen_slope_ci95": slope_ci95,
                },
                "source_atomic_scan_ids": [str(item["scan_id"]) for item in items],
                "aggregation_note": "Only complete matched scene blocks enter response, error, residual and bootstrap statistics; camera and seed remain fixed.",
            }
        )
    return output


def aggregate_experiment_scan_grades(
    scans: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Conservatively combine canonical parameter grades within an experiment scope."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for scan in scans:
        if not scan.get("canonical_for_target"):
            continue
        key = (
            scan["experiment_id"],
            scan["scene_id"],
            scan["camera_name"],
            scan["object_id"],
            scan["seed"],
        )
        groups[key].append(scan)
    order = {"G4": 3, "G3": 2, "G2": 1}
    registry_required = {
        str(experiment["id"]): sorted(
            str(parameter["name"])
            for parameter in experiment.get("hidden_parameters", [])
        )
        for experiment in (registry or {}).get("experiments", [])
    }
    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        grades = {str(item["target_parameter"]): str(item["response_grade"]) for item in items}
        required = registry_required.get(str(key[0])) or sorted({
            str(name)
            for item in items
            for name in item.get("required_parameters", [item.get("target_parameter")])
            if name is not None
        })
        missing = sorted(set(required) - set(grades))
        for name in missing:
            grades[name] = "U"
        if any(value == "U" for value in grades.values()):
            overall = "U"
            reason = "at_least_one_required_parameter_has_insufficient_evidence"
        else:
            overall = min(grades.values(), key=lambda grade: order[grade])
            reason = "worst_required_parameter_grade"
        output.append(
            {
                "schema_version": GRADE_SCHEMA_VERSION,
                "entity_scope": "experiment_parameter_response",
                "experiment_id": key[0],
                "scene_id": key[1],
                "camera_name": key[2],
                "object_id": key[3],
                "seed": key[4],
                "required_parameters": required,
                "missing_required_parameters": missing,
                "parameter_grades": grades,
                "overall_response_grade": overall,
                "aggregation_rule": reason,
                "scan_ids": [item["scan_id"] for item in items],
            }
        )
    return output


def video_grade_summary_row(grade: Mapping[str, Any]) -> dict[str, Any]:
    job = grade.get("job", {})
    return {
        "job_id": grade.get("job_id"),
        "experiment_id": job.get("experiment_id"),
        "parameter_tuple_id": job.get("parameter_tuple_id"),
        "scene_id": job.get("scene_id"),
        "camera_name": job.get("camera_name"),
        "seed": job.get("seed"),
        "video_stage": grade.get("video_stage"),
        "eligible_for_parameter_scan": grade.get("eligible_for_parameter_scan"),
        "reason_codes": ";".join(str(value) for value in grade.get("reason_codes", [])),
        "g0_generation_status": grade.get("g0_generation_validity", {}).get("status"),
        "g0_contact_status": grade.get("g0_contact_geometry", {}).get("status"),
        "g0_dynamic_rigidity_status": grade.get("g0_dynamic_rigidity", {}).get("status"),
        "g1_motion_status": grade.get("g1_motion_type_validity", {}).get("status"),
        "fit_status": grade.get("inverse_fit", {}).get("status"),
        "fit_complete": grade.get("inverse_fit", {}).get("fit_complete"),
        "inverse_fit_evidence_status": grade.get("inverse_fit_evidence", {}).get("status"),
        "inverse_fit_worst_nrmse": grade.get("inverse_fit_evidence", {}).get("per_job_worst_fit_nrmse"),
        "inverse_fit_worst_r2": grade.get("inverse_fit_evidence", {}).get("per_job_worst_fit_r2"),
    }


def scan_grade_summary_row(scan: Mapping[str, Any]) -> dict[str, Any]:
    decision = scan.get("decision", {})
    evidence = decision.get("evidence", {})
    direction = decision.get("direction", {})
    numeric = decision.get("numeric_accuracy", {})
    off_target = [
        float(value)
        for value in numeric.get("off_target_drift_nad", {}).values()
        if _number(value) is not None
    ]
    return {
        "scan_id": scan.get("scan_id"),
        "experiment_id": scan.get("experiment_id"),
        "scene_id": scan.get("scene_id"),
        "camera_name": scan.get("camera_name"),
        "seed": scan.get("seed"),
        "target_parameter": scan.get("target_parameter"),
        "fixed_other_parameters_json": json.dumps(scan.get("fixed_other_parameters", {}), sort_keys=True),
        "canonical_for_target": scan.get("canonical_for_target"),
        "response_grade": scan.get("response_grade"),
        "reason_codes": ";".join(str(value) for value in decision.get("reason_codes", [])),
        "planned_level_count": evidence.get("planned_level_count"),
        "usable_level_count": evidence.get("usable_level_count"),
        "level_coverage_fraction": evidence.get("level_coverage_fraction"),
        "pairwise_direction_concordance": direction.get("pairwise_direction_concordance"),
        "spearman_rho": direction.get("spearman_rho"),
        "theil_sen_slope": direction.get("theil_sen_slope"),
        "theil_sen_slope_ci95_json": json.dumps(direction.get("theil_sen_slope_ci95")),
        "median_valid_range_nae": numeric.get("median_valid_range_nae"),
        "p90_valid_range_nae": numeric.get("p90_valid_range_nae"),
        "maximum_off_target_drift_nad": max(off_target) if off_target else None,
        "p90_per_job_worst_trajectory_fit_nrmse": numeric.get("p90_per_job_worst_trajectory_fit_nrmse"),
        "matched_scene_count": scan.get("matched_scene_count"),
        "evidence_strength": scan.get("evidence_strength"),
    }


__all__ = [
    "GRADE_SCHEMA_VERSION",
    "aggregate_experiment_scan_grades",
    "build_parameter_scan_grades",
    "build_cross_scene_parameter_scan_grades",
    "grade_video_result",
    "load_json",
    "missing_video_grade",
    "read_trajectory_csv",
    "resolve_trajectory_path",
    "score_parameter_response",
    "scan_grade_summary_row",
    "video_grade_summary_row",
    "write_csv",
    "write_json",
    "write_jsonl",
]
