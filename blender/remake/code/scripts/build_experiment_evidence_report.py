#!/usr/bin/env python3
"""Build an experiment-level PhysParamBench evidence matrix.

This report is deliberately simpler than the parameter-channel report.  It
answers one question per model and experiment:

* A: numerically faithful response;
* D: correct parameter-response direction, but inaccurate magnitude;
* N: usable numerical evidence, but no correct parameter response;
* M: the generated motion violates the prescribed rule or misses a required
  event despite adequate tracking;
* V: clear generation-validity failure; or
* R: the trajectory is present but a scene-only gate requires refitting; or
* U: the observation/tracking evidence is genuinely insufficient.

Missing parameter estimates are never converted to parameter errors.  A rule
failure (M) is evidence about motion generation, not a fabricated parameter
score.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.deadline_report import (  # noqa: E402
    parse_model_arguments,
)


SCHEMA_VERSION = "1.1.0"
EVENT_TRACK_AUTO_MIN = 0.90
EVENT_TRACK_REVIEW_MIN = 0.75
MIN_NEGATIVE_TUPLES = 2
MIN_NEGATIVE_TUPLE_FRACTION = 0.50

EXPERIMENT_TITLES = {
    "v1_A": "自由落体 / 重力",
    "v1_B": "单次墙碰撞 / 恢复系数",
    "v1_C": "单表面滑动 / 摩擦",
    "v1_D": "阻尼摆 / 衰减",
    "v2_A": "周期轨道 / 重力",
    "v2_B": "多次墙碰撞 / 恢复系数",
    "v2_C": "双表面滑动 / 双摩擦",
    "v2_D": "重力 + 阻尼摆",
    "v2_E": "重力 + 地面反弹",
    "v3_A": "重力 + 阻力 + 反弹",
    "v3_B": "摩擦 + 左右墙碰撞",
    "v3_C": "斜坡 + 地面 + 墙碰撞",
    "v3_D": "重力 + 阻尼 + 磁力摆",
}

EXPERIMENT_CONCLUSIONS = {
    "v1_A": "通常能生成下落外形，部分模型会随重力增大而给出同向变化，但加速度量级往往不准确。",
    "v1_B": "短时碰撞最不稳定；常见速度反转缺失、碰撞粘滞、非匀速或碰撞位置错误，恢复系数难以可靠控制。",
    "v1_C": "应在连续一维滑动段上直接拟合摩擦；若只有静态场景刚性告警而物体跟踪完整，应重拟合而非直接判失败。",
    "v1_D": "能够生成振荡外形不等于理解阻尼；完整周期和指数衰减经常缺失，衰减率响应通常不精确。",
    "v2_A": "周期运动有时成立，但周期与指定重力的映射不稳定；视觉周期性不能证明参数忠实。",
    "v2_B": "重复碰撞没有稳定恢复系数；反弹序列、碰撞间匀速和非粘滞条件经常同时被破坏。",
    "v2_C": "模型难以同时维持唯一表面切换、速度连续性和两段恒定摩擦，两个摩擦参数通常不能独立控制。",
    "v2_D": "模型可能局部表现阻尼趋势，但常不能用同一方程同时解释重力与阻尼，体现组合参数分离失败。",
    "v2_E": "连续弹道段通常比瞬时碰撞更容易；重力有时可恢复，但恢复系数或跨反弹一致性仍常失败。",
    "v3_A": "拆轴后偶尔能恢复重力或反弹线索，但水平阻力通常错误，三个参数无法联合忠实实现。",
    "v3_B": "局部墙碰撞可能可辨识，但摩擦与左右恢复系数难以同时保持，组合后只剩局部物理线索。",
    "v3_C": "斜坡到地面再到墙面的事件链经常不完整，说明长事件链和混合规则是明显薄弱点。",
    "v3_D": "视频可呈现周期外观，但前向方程、阻尼物理域和磁参数辨识通常失败，三参数联合理解不足。",
}

STATE_LABELS = {
    "A": "数值准确",
    "D": "方向正确、数值不准",
    "N": "有数值证据、参数无正确响应",
    "M": "规则或必要事件失败",
    "V": "生成有效性失败",
    "R": "轨迹可用，需解除场景门控后重拟合",
    "U": "观测证据不足",
}

EVENT_REASON_TOKENS = {
    "first_floor_contact_not_observed",
    "wall_velocity_reversal_not_found",
    "first_complete_period_not_observed",
    "no_complete_cycloid_period",
    "no_valid_frozen_wall_impact",
    "no_valid_left_wall_impact",
    "no_valid_right_wall_impact",
    "both_friction_surfaces_not_observed",
    "surface_transition_not_observed",
    "expected_exactly_one_surface_transition",
    "right_wall_impact_not_observed",
    "left_wall_impact_not_observed",
    "ramp_to_floor_transition_not_observed",
    "required_segments_missing",
    "ramp_pre_wall_and_post_wall_segments_required",
    "no_complete_magnetic_pendulum_period",
}

EVIDENCE_TIER_LABELS = {
    "E3": "高：跨目标/重复证据收敛",
    "E2": "中：证据足以给出实验结论",
    "E1": "待重拟合：轨迹存在但当前无正式结论",
    "E0": "不足：跟踪或观测证据不够",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _boolean(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return fallback
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _reason_tokens(row: Mapping[str, Any]) -> set[str]:
    values: list[str] = []
    for key in (
        "fit_validity_primary_reason",
        "rule_family_reason_codes",
        "generation_failure_codes",
        "generation_warning_codes",
        "generation_review_codes",
    ):
        values.append(str(row.get(key) or ""))
    attribution = str(row.get("parameter_attribution_json") or "")
    values.append(attribution)
    text = ";".join(values)
    for punctuation in ",:[]{}\"'":
        text = text.replace(punctuation, ";")
    return {token.strip() for token in text.split(";") if token.strip()}


def _contains_event_reason(tokens: Iterable[str]) -> bool:
    for token in tokens:
        leaf = token.rsplit(":", 1)[-1]
        if leaf in EVENT_REASON_TOKENS:
            return True
        if any(marker in leaf for marker in EVENT_REASON_TOKENS):
            return True
    return False


def _scene_only_refit_candidate(
    row: Mapping[str, Any],
    *,
    tracked_fraction: float | None,
    tokens: set[str],
) -> bool:
    if str(row.get("fit_status") or "") != "skipped_trajectory_not_eligible":
        return False
    if tracked_fraction is None or tracked_fraction < EVENT_TRACK_AUTO_MIN:
        return False
    if "static_scene_rigidity_unresolved" not in tokens:
        return False
    hard_markers = (
        "persistent_object_shape_deformation",
        "object_identity_unresolved",
        "experimental_object_missing",
    )
    return not any(
        any(marker in token for marker in hard_markers) for token in tokens
    )


def _classify_job(row: Mapping[str, Any]) -> dict[str, Any]:
    fit_status = str(row.get("fit_status") or "")
    generation_status = str(row.get("generation_validity_status") or "")
    tracked_fraction = _number(row.get("tracked_fraction"))
    tokens = _reason_tokens(row)
    event_reason = _contains_event_reason(tokens)

    if generation_status == "fail" or _boolean(row.get("generation_hard_failure")):
        evidence = "VALIDITY_FAIL"
        state = "V"
    elif fit_status == "model_mismatch":
        evidence = "RULE_FAIL"
        state = "M"
    elif fit_status in {"ok", "partial"}:
        evidence = "NUMERIC"
        state = "NUMERIC"
    elif fit_status == "insufficient_evidence" and event_reason:
        if tracked_fraction is not None and tracked_fraction >= EVENT_TRACK_AUTO_MIN:
            evidence = "EVENT_FAIL"
            state = "M"
        elif (
            tracked_fraction is not None
            and tracked_fraction >= EVENT_TRACK_REVIEW_MIN
        ):
            evidence = "EVENT_REVIEW"
            state = "U"
        else:
            evidence = "TRUE_U"
            state = "U"
    elif _scene_only_refit_candidate(
        row, tracked_fraction=tracked_fraction, tokens=tokens
    ):
        evidence = "REFIT_CANDIDATE"
        state = "U"
    else:
        evidence = "TRUE_U"
        state = "U"

    return {
        **dict(row),
        "tracked_fraction_numeric": tracked_fraction,
        "reason_tokens": sorted(tokens),
        "event_reason_detected": event_reason,
        "job_evidence_class": evidence,
        "job_state": state,
    }


def _load_jobs(models: Mapping[str, Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, root in models.items():
        all_jobs = root if root.is_file() else root / "all_jobs.csv"
        if not all_jobs.is_file():
            raise FileNotFoundError(f"missing all_jobs.csv for {model}: {all_jobs}")
        evaluation_root = all_jobs.parent
        for row in _read_csv(all_jobs):
            if (
                row.get("scene_id") != "baseline"
                or row.get("camera_name") != "CAM_Side"
            ):
                continue
            classified = _classify_job(row)
            job_id = str(row.get("job_id") or Path(row.get("video_name") or "").stem)
            result_json = evaluation_root / "jobs" / job_id / "result.json"
            classified.update(
                {
                    "model": model,
                    "job_id": job_id,
                    "result_json": str(result_json) if result_json.is_file() else None,
                }
            )
            output.append(classified)
    return output


def _scan_index(scan_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[str]]:
    output: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in scan_rows:
        output[
            (str(row.get("model") or ""), str(row.get("experiment_id") or ""))
        ].append(str(row.get("grade") or "U"))
    return output


def _scan_evidence_index(
    scan_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in scan_rows:
        row = dict(raw)
        levels = _json_value(row.get("target_levels"), [])
        if not isinstance(levels, list):
            levels = []
        seed_counts = [
            int(value)
            for level in levels
            if isinstance(level, Mapping)
            and (value := _number(level.get("seed_count"))) is not None
        ]
        row["target_levels_parsed"] = levels
        row["target_level_count_numeric"] = len(levels)
        row["total_seed_evidence_count"] = sum(seed_counts)
        row["minimum_seed_count_per_level"] = min(seed_counts) if seed_counts else 0
        output[
            (str(row.get("model") or ""), str(row.get("experiment_id") or ""))
        ].append(row)
    return output


def _evidence_tier(
    *,
    state: str,
    scan_rows: Sequence[Mapping[str, Any]],
    conclusive_tuple_count: int,
    conclusive_tuple_fraction: float,
    conclusive_job_count: int,
) -> str:
    if state == "R":
        return "E1"
    if state == "U":
        return "E0"
    if state in {"A", "D", "N"}:
        evaluable = [
            row for row in scan_rows if str(row.get("grade") or "U") != "U"
        ]
        # E3 requires both a three-level scan and repeated generation evidence
        # at every level.  Three single samples remain useful, but only E2.
        if any(
            int(row.get("target_level_count_numeric") or 0) >= 3
            and int(row.get("minimum_seed_count_per_level") or 0) >= 2
            for row in evaluable
        ):
            return "E3"
        return "E2"
    # M/V conclusions are E3 only when the same failure is broad across target
    # tuples and repeated jobs; otherwise the preregistered 50%/two-tuple rule
    # supports an E2 conclusion.
    if (
        conclusive_tuple_count >= 3
        and conclusive_tuple_fraction >= 0.75
        and conclusive_job_count >= 2 * conclusive_tuple_count
    ):
        return "E3"
    return "E2"


def _representative_job(rows: Sequence[Mapping[str, Any]], state: str) -> str | None:
    priority = {
        "D": ("NUMERIC", "RULE_FAIL", "EVENT_FAIL", "TRUE_U"),
        "N": ("NUMERIC", "RULE_FAIL", "EVENT_FAIL", "TRUE_U"),
        "M": ("RULE_FAIL", "EVENT_FAIL", "VALIDITY_FAIL", "TRUE_U"),
        "V": ("VALIDITY_FAIL", "RULE_FAIL", "TRUE_U"),
        "R": ("REFIT_CANDIDATE", "TRUE_U"),
        "U": ("REFIT_CANDIDATE", "EVENT_REVIEW", "TRUE_U"),
        "A": ("NUMERIC",),
    }
    for evidence in priority.get(state, ()):
        candidates = [
            row for row in rows if row.get("job_evidence_class") == evidence
        ]
        if candidates:
            candidates.sort(
                key=lambda row: (
                    -(_number(row.get("tracked_fraction_numeric")) or 0.0),
                    str(row.get("job_id") or ""),
                )
            )
            return str(candidates[0].get("job_id") or "")
    return None


def _experiment_rows(
    jobs: Sequence[Mapping[str, Any]],
    scans: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in jobs:
        by_cell[
            (str(row.get("model") or ""), str(row.get("experiment_id") or ""))
        ].append(row)
    scan_by_cell = _scan_index(scans)
    scan_evidence_by_cell = _scan_evidence_index(scans)
    output: list[dict[str, Any]] = []
    for experiment_id in EXPERIMENT_TITLES:
        for model in models:
            cell = by_cell.get((model, experiment_id), [])
            counts = Counter(str(row.get("job_evidence_class") or "") for row in cell)
            tuple_ids = {
                str(row.get("parameter_tuple_id") or "") for row in cell
            }
            negative_tuples = {
                str(row.get("parameter_tuple_id") or "")
                for row in cell
                if row.get("job_state") == "M"
            }
            validity_tuples = {
                str(row.get("parameter_tuple_id") or "")
                for row in cell
                if row.get("job_state") == "V"
            }
            refit_tuples = {
                str(row.get("parameter_tuple_id") or "")
                for row in cell
                if row.get("job_evidence_class") == "REFIT_CANDIDATE"
            }
            numeric_tuples = {
                str(row.get("parameter_tuple_id") or "")
                for row in cell
                if row.get("job_evidence_class") == "NUMERIC"
            }
            conclusive_tuples = negative_tuples | validity_tuples | numeric_tuples
            refit_only_tuples = refit_tuples - conclusive_tuples
            unresolved_tuples = tuple_ids - conclusive_tuples - refit_only_tuples
            negative_fraction = (
                len(negative_tuples) / len(tuple_ids) if tuple_ids else 0.0
            )
            validity_fraction = (
                len(validity_tuples) / len(tuple_ids) if tuple_ids else 0.0
            )
            refit_fraction = (
                len(refit_tuples) / len(tuple_ids) if tuple_ids else 0.0
            )
            grades = scan_by_cell.get((model, experiment_id), [])
            scan_evidence = scan_evidence_by_cell.get((model, experiment_id), [])
            if any(grade.startswith("G4") for grade in grades):
                state = "A"
            elif any(grade.startswith("G3") for grade in grades):
                state = "D"
            elif any(grade.startswith("G2") for grade in grades):
                state = "N"
            elif (
                len(negative_tuples) >= MIN_NEGATIVE_TUPLES
                and negative_fraction >= MIN_NEGATIVE_TUPLE_FRACTION
            ):
                state = "M"
            elif (
                len(validity_tuples) >= MIN_NEGATIVE_TUPLES
                and validity_fraction >= MIN_NEGATIVE_TUPLE_FRACTION
            ):
                state = "V"
            elif (
                len(refit_tuples) >= MIN_NEGATIVE_TUPLES
                and refit_fraction >= MIN_NEGATIVE_TUPLE_FRACTION
            ):
                state = "R"
            else:
                state = "U"
            conclusive_job_count = (
                counts["NUMERIC"]
                + counts["RULE_FAIL"]
                + counts["EVENT_FAIL"]
                + counts["VALIDITY_FAIL"]
            )
            unresolved_job_count = counts["EVENT_REVIEW"] + counts["TRUE_U"]
            evidence_tier = _evidence_tier(
                state=state,
                scan_rows=scan_evidence,
                conclusive_tuple_count=len(conclusive_tuples),
                conclusive_tuple_fraction=(
                    len(conclusive_tuples) / len(tuple_ids) if tuple_ids else 0.0
                ),
                conclusive_job_count=conclusive_job_count,
            )
            representative = _representative_job(cell, state)
            representative_row = next(
                (
                    row
                    for row in cell
                    if str(row.get("job_id") or "") == representative
                ),
                {},
            )
            output.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_title_zh": EXPERIMENT_TITLES[experiment_id],
                    "model": model,
                    "state": state,
                    "state_label_zh": STATE_LABELS[state],
                    "evidence_tier": evidence_tier,
                    "evidence_tier_label_zh": EVIDENCE_TIER_LABELS[evidence_tier],
                    "scan_grades": grades,
                    "scan_channel_count": len(scan_evidence),
                    "evaluable_scan_channel_count": sum(
                        str(row.get("grade") or "U") != "U"
                        for row in scan_evidence
                    ),
                    "directional_scan_channel_count": sum(
                        str(row.get("grade") or "").startswith(("G3", "G4"))
                        for row in scan_evidence
                    ),
                    "maximum_available_target_levels": max(
                        (
                            int(row.get("target_level_count_numeric") or 0)
                            for row in scan_evidence
                        ),
                        default=0,
                    ),
                    "total_scan_seed_evidence_count": sum(
                        int(row.get("total_seed_evidence_count") or 0)
                        for row in scan_evidence
                    ),
                    "baseline_side_jobs": len(cell),
                    "numeric_jobs": counts["NUMERIC"],
                    "rule_fail_jobs": counts["RULE_FAIL"],
                    "event_fail_jobs": counts["EVENT_FAIL"],
                    "validity_fail_jobs": counts["VALIDITY_FAIL"],
                    "event_review_jobs": counts["EVENT_REVIEW"],
                    "refit_candidate_jobs": counts["REFIT_CANDIDATE"],
                    "true_u_jobs": counts["TRUE_U"],
                    "conclusive_job_count": conclusive_job_count,
                    "conclusive_job_fraction": (
                        conclusive_job_count / len(cell) if cell else 0.0
                    ),
                    "pending_refit_job_count": counts["REFIT_CANDIDATE"],
                    "genuine_unresolved_job_count": unresolved_job_count,
                    "genuine_unresolved_job_fraction": (
                        unresolved_job_count / len(cell) if cell else 0.0
                    ),
                    "conclusive_target_tuple_count": len(conclusive_tuples),
                    "conclusive_target_tuple_fraction": (
                        len(conclusive_tuples) / len(tuple_ids)
                        if tuple_ids
                        else 0.0
                    ),
                    "pending_refit_target_tuple_count": len(refit_only_tuples),
                    "genuine_unresolved_target_tuple_count": len(unresolved_tuples),
                    "genuine_unresolved_target_tuple_fraction": (
                        len(unresolved_tuples) / len(tuple_ids)
                        if tuple_ids
                        else 0.0
                    ),
                    "negative_target_tuple_count": len(negative_tuples),
                    "target_tuple_count": len(tuple_ids),
                    "negative_target_tuple_fraction": negative_fraction,
                    "validity_target_tuple_fraction": validity_fraction,
                    "refit_target_tuple_count": len(refit_tuples),
                    "refit_target_tuple_fraction": refit_fraction,
                    "representative_job_id": representative,
                    "representative_job_evidence_class": representative_row.get(
                        "job_evidence_class"
                    ),
                    "representative_fit_reason": representative_row.get(
                        "fit_validity_primary_reason"
                    ),
                    "representative_rule_reason_codes": representative_row.get(
                        "rule_family_reason_codes"
                    ),
                    "representative_tracked_fraction": representative_row.get(
                        "tracked_fraction_numeric"
                    ),
                    "representative_result_json": representative_row.get(
                        "result_json"
                    ),
                    "experiment_conclusion_zh": EXPERIMENT_CONCLUSIONS[experiment_id],
                }
            )
    return output


def _matrix_svg(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> None:
    cell_w = 150
    cell_h = 42
    left = 330
    top = 94
    width = max(left + len(models) * cell_w + 40, 1180)
    height = top + len(EXPERIMENT_TITLES) * cell_h + 100
    colors = {
        "A": "#1b9e77",
        "D": "#66a61e",
        "N": "#e6ab02",
        "M": "#d95f02",
        "V": "#e7298a",
        "R": "#1f78b4",
        "U": "#7570b3",
    }
    by_cell = {
        (str(row["experiment_id"]), str(row["model"])): row for row in rows
    }
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#222}",
        ".title{font-size:22px;font-weight:700}.head{font-size:13px;font-weight:700}",
        ".label{font-size:13px}.cell{font-size:16px;font-weight:700;fill:white}",
        ".legend{font-size:12px}",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="32" class="title">PhysParamBench experiment-level evidence matrix</text>',
        '<text x="20" y="56" class="label">baseline + CAM_Side; U only means genuine observation insufficiency</text>',
    ]
    for col, model in enumerate(models):
        x = left + col * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{top - 22}" text-anchor="middle" class="head">'
            f"{html.escape(model)}</text>"
        )
    for row_index, (experiment, title) in enumerate(EXPERIMENT_TITLES.items()):
        y = top + row_index * cell_h
        parts.append(
            f'<text x="20" y="{y + 27}" class="label">'
            f"{html.escape(experiment)}  {html.escape(title)}</text>"
        )
        for col, model in enumerate(models):
            item = by_cell[(experiment, model)]
            state = str(item["state"])
            x = left + col * cell_w
            parts.append(
                f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 6}" '
                f'height="{cell_h - 6}" rx="5" fill="{colors[state]}">'
                f"<title>{html.escape(model)} · {html.escape(experiment)}: "
                f"{html.escape(STATE_LABELS[state])}</title></rect>"
            )
            parts.append(
                f'<text x="{x + cell_w / 2}" y="{y + 27}" '
                f'text-anchor="middle" class="cell">{state}</text>'
            )
    legend_y = top + len(EXPERIMENT_TITLES) * cell_h + 24
    cursor = 20
    for state in ("A", "D", "N", "M", "V", "R", "U"):
        parts.append(
            f'<rect x="{cursor}" y="{legend_y}" width="16" height="16" rx="3" '
            f'fill="{colors[state]}"/>'
        )
        parts.append(
            f'<text x="{cursor + 22}" y="{legend_y + 13}" class="legend">'
            f"{state} {html.escape(STATE_LABELS[state])}</text>"
        )
        cursor += 160
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _model_evidence_summaries(
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in rows if row.get("model") == model]
        model_jobs = [row for row in jobs if row.get("model") == model]
        states = Counter(str(row.get("state") or "U") for row in model_rows)
        tiers = Counter(str(row.get("evidence_tier") or "E0") for row in model_rows)
        job_classes = Counter(
            str(row.get("job_evidence_class") or "TRUE_U") for row in model_jobs
        )
        conclusive_cells = sum(
            states.get(state, 0) for state in ("A", "D", "N", "M", "V")
        )
        conclusive_jobs = sum(
            job_classes.get(evidence, 0)
            for evidence in ("NUMERIC", "RULE_FAIL", "EVENT_FAIL", "VALIDITY_FAIL")
        )
        genuine_u_jobs = (
            job_classes.get("TRUE_U", 0) + job_classes.get("EVENT_REVIEW", 0)
        )
        output.append(
            {
                "model": model,
                "experiment_count": len(model_rows),
                "experiment_conclusion_count": conclusive_cells,
                "experiment_conclusion_coverage": (
                    conclusive_cells / len(model_rows) if model_rows else None
                ),
                "accurate_experiment_count": states.get("A", 0),
                "directional_experiment_count": states.get("D", 0),
                "no_response_experiment_count": states.get("N", 0),
                "rule_failure_experiment_count": states.get("M", 0),
                "validity_failure_experiment_count": states.get("V", 0),
                "pending_refit_experiment_count": states.get("R", 0),
                "genuine_u_experiment_count": states.get("U", 0),
                "e3_experiment_count": tiers.get("E3", 0),
                "e2_experiment_count": tiers.get("E2", 0),
                "e1_experiment_count": tiers.get("E1", 0),
                "e0_experiment_count": tiers.get("E0", 0),
                "baseline_side_job_count": len(model_jobs),
                "conclusive_job_count": conclusive_jobs,
                "conclusive_job_fraction": (
                    conclusive_jobs / len(model_jobs) if model_jobs else None
                ),
                "pending_refit_job_count": job_classes.get("REFIT_CANDIDATE", 0),
                "genuine_u_job_count": genuine_u_jobs,
                "genuine_u_job_fraction": (
                    genuine_u_jobs / len(model_jobs) if model_jobs else None
                ),
            }
        )
    return output


def _experiment_conclusion_summaries(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for experiment_id, title in EXPERIMENT_TITLES.items():
        selected = [
            row for row in rows if row.get("experiment_id") == experiment_id
        ]
        states = Counter(str(row.get("state") or "U") for row in selected)
        tiers = Counter(str(row.get("evidence_tier") or "E0") for row in selected)
        concluded = sum(
            states.get(state, 0) for state in ("A", "D", "N", "M", "V")
        )
        output.append(
            {
                "experiment_id": experiment_id,
                "experiment_title_zh": title,
                "model_count": len(models),
                "model_conclusion_count": concluded,
                "model_conclusion_coverage": (
                    concluded / len(models) if models else None
                ),
                "accurate_model_count": states.get("A", 0),
                "directional_model_count": states.get("D", 0),
                "no_response_model_count": states.get("N", 0),
                "rule_failure_model_count": states.get("M", 0),
                "validity_failure_model_count": states.get("V", 0),
                "pending_refit_model_count": states.get("R", 0),
                "genuine_u_model_count": states.get("U", 0),
                "e3_model_count": tiers.get("E3", 0),
                "e2_model_count": tiers.get("E2", 0),
                "experiment_conclusion_zh": EXPERIMENT_CONCLUSIONS[experiment_id],
            }
        )
    return output


def _merge_paper_metrics(
    model_summaries: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_model = {str(row.get("model") or ""): row for row in metric_rows}
    output: list[dict[str, Any]] = []
    for evidence in model_summaries:
        model = str(evidence.get("model") or "")
        metric = by_model.get(model, {})
        g3 = int(_number(metric.get("g3_directional_channel_count")) or 0)
        g4 = int(_number(metric.get("g4_accurate_channel_count")) or 0)
        total = int(_number(metric.get("canonical_channel_count")) or 0)
        evaluable = int(_number(metric.get("evaluable_channel_count")) or 0)
        output.append(
            {
                "model": model,
                "equation_supported_parameters": metric.get(
                    "equation_supported_parameter_count"
                ),
                "expected_parameters": metric.get(
                    "expected_parameter_observation_count"
                ),
                "equation_support_coverage": metric.get(
                    "equation_supported_parameter_coverage"
                ),
                "parameter_success_at_25_count": metric.get(
                    "parameter_success_count"
                ),
                "parameter_success_at_25_all": metric.get(
                    "parameter_success_at_25pct_all"
                ),
                "median_bnae_supported": metric.get(
                    "median_candidate_bnae_supported"
                ),
                "median_trajectory_r2_supported": metric.get(
                    "median_trajectory_r2_supported"
                ),
                "median_trajectory_nrmse_supported": metric.get(
                    "median_trajectory_nrmse_supported"
                ),
                "directional_or_accurate_channels": g3 + g4,
                "evaluable_channels": evaluable,
                "canonical_channels": total,
                "scan_evidence_coverage": evaluable / total if total else None,
                "directional_rate_all_channels": (
                    (g3 + g4) / total if total else None
                ),
                "directional_rate_evaluable_channels": (
                    (g3 + g4) / evaluable if evaluable else None
                ),
                **{
                    key: value
                    for key, value in evidence.items()
                    if key != "model"
                },
            }
        )
    return output


def _format_fraction(numerator: Any, denominator: Any) -> str:
    left = int(_number(numerator) or 0)
    right = int(_number(denominator) or 0)
    return f"{left}/{right}"


def _format_percent(value: Any) -> str:
    numeric = _number(value)
    return "—" if numeric is None else f"{100.0 * numeric:.1f}%"


def _format_metric(value: Any, digits: int = 3) -> str:
    numeric = _number(value)
    return "—" if numeric is None else f"{numeric:.{digits}f}"


def _paper_main_table_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# 可审计的主结果表",
        "",
        "参数准确性与证据覆盖必须同时报告。`Eq. support` 和 `Acc@25` 的分母均为全部应评参数；缺失不会被填成 0。`Dir.` 同时给出全部预注册通道分母，避免只在成功拟合的子集上放大响应率。",
        "",
        "| 模型 | Eq. support | Acc@25 | Median BNAE† | Fit R² / NRMSE† | Dir./all | Scan coverage | 实验结论 | 待重拟合 | 真 U |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {eq} ({eq_pct}) | {acc} ({acc_pct}) | {bnae} | "
            "{r2} / {nrmse} | "
            "{direction} | {scan_pct} | {conclusions} ({conclusion_pct}) | "
            "{pending} | {true_u} |".format(
                model=row.get("model"),
                eq=_format_fraction(
                    row.get("equation_supported_parameters"),
                    row.get("expected_parameters"),
                ),
                eq_pct=_format_percent(row.get("equation_support_coverage")),
                acc=_format_fraction(
                    row.get("parameter_success_at_25_count"),
                    row.get("expected_parameters"),
                ),
                acc_pct=_format_percent(row.get("parameter_success_at_25_all")),
                bnae=_format_metric(row.get("median_bnae_supported")),
                r2=_format_metric(row.get("median_trajectory_r2_supported")),
                nrmse=_format_metric(
                    row.get("median_trajectory_nrmse_supported")
                ),
                direction=_format_fraction(
                    row.get("directional_or_accurate_channels"),
                    row.get("canonical_channels"),
                ),
                scan_pct=_format_percent(row.get("scan_evidence_coverage")),
                conclusions=_format_fraction(
                    row.get("experiment_conclusion_count"),
                    row.get("experiment_count"),
                ),
                conclusion_pct=_format_percent(
                    row.get("experiment_conclusion_coverage")
                ),
                pending=int(_number(row.get("pending_refit_experiment_count")) or 0),
                true_u=int(_number(row.get("genuine_u_experiment_count")) or 0),
            )
        )
    lines.extend(
        [
            "",
            "† Median BNAE 只在方程支持的参数估计上计算，因此必须和 Eq. support 一起解读；它不是全数据集平均误差。",
            "",
            "## 指标口径",
            "",
            "- `AE = |θ̂−θ|`：单个参数、原单位误差，最直观，但不能跨重力/摩擦/阻尼直接平均。",
            "- `BNAE = AE/(该实验参数的预注册目标最大值−最小值)`：无量纲且不截断，用于跨物理量汇总；模型是否知道这个跨度与评测是否可用无关，它只是 benchmark 的评分尺。",
            "- `Trajectory R² / NRMSE`：只判断选定方程能否解释轨迹，不等价于参数正确；主表给方程支持样本的中位数作为拟合质量诊断。",
            "- `Eq. support coverage`：通过轨迹方程支持门槛的参数数/全部应评参数数。它防止只展示少量成功案例。",
            "- `Acc@25`：方程支持且 `BNAE≤0.25` 的参数数/全部应评参数数，是端到端参数忠实度。",
            "- `Direction`：先按目标值跨 seed 取参数估计中位数，再计算两两方向一致率与 Theil–Sen 斜率；至少两个目标值、方向一致率>0.5 且斜率≥0.10 才算同向响应。",
            "- `实验结论覆盖`：13 个实验中有数值响应、明确规则失败或明确生成失败的实验数。它与参数估计覆盖分开，不把规则失败伪装成参数分数。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> None:
    by_cell = {
        (str(row["experiment_id"]), str(row["model"])): row for row in rows
    }
    lines = [
        "# 13 个实验的可读证据结论",
        "",
        "主统计范围：`baseline + CAM_Side`。本表是实验级证据状态，不把缺失参数估计填成 0，也不把跟踪失败伪装成模型物理失败。",
        "",
        "- `A`：方向与数值均准确；`D`：方向正确但数值不准；`N`：有数值轨迹证据但参数扫描无正确响应。",
        "- `M`：轨迹明确不满足规则，或在跟踪充分时缺少设计要求的事件；`V`：明确生成有效性失败。",
        "- `R`：轨迹数据已存在，但仅被场景门控阻断，需重拟合；`U`：真正的观测证据不足。",
        "- `E3/E2` 是高/中可信证据；`E1` 表示已有轨迹、等待重拟合；`E0` 才是真正无法下结论。",
        "",
        "![实验级证据矩阵](experiment_evidence_matrix.svg)",
        "",
        "| 实验 | " + " | ".join(models) + " | 可写入论文/汇报的结论 |",
        "|---|" + "|".join("---" for _ in models) + "|---|",
    ]
    for experiment_id, title in EXPERIMENT_TITLES.items():
        states = [
            str(by_cell[(experiment_id, model)]["state"]) for model in models
        ]
        lines.append(
            f"| {experiment_id} {title} | "
            + " | ".join(states)
            + f" | {EXPERIMENT_CONCLUSIONS[experiment_id]} |"
        )
    lines.extend(
        [
            "",
            "## 三条总叙事",
            "",
            "1. 单段连续运动最容易获得定性响应：模型常能生成下落、周期或衰减外形，但反推参数量级通常不忠实。",
            "2. 碰撞是稳定薄弱点：从单次碰撞到重复碰撞和组合碰撞，持续出现速度反转缺失、粘滞、突然加减速与恢复系数不一致。",
            "3. 参数组合体现“局部理解、联合失败”：某个分量偶尔可恢复，并不意味着同一轨迹中的多个物理参数能同时被忠实实现。",
            "",
            "## 证据链使用方式",
            "",
            "- `D/N/A` 单元：展示不同目标参数的原视频对比、overlay、观测轨迹与拟合轨迹、target→estimate。",
            "- `M` 单元：展示原视频和 overlay，并报告失败的必要条件（例如无速度反转、无表面切换、方程残差过大）；不强行给参数分数。",
            "- `V` 单元：展示形变、消失、穿模或非实验物体刚性变化的关键帧。",
            "- `R` 单元：不对模型下结论；解除场景门控并复用已有轨迹重拟合，不需要重跑视频或跟踪。",
            "- `U` 单元：只用于物体身份、跟踪、遮挡或 3D 重建确实不能提供判断证据的情况。",
            "",
            "每格的计数、缺失率、证据等级、代表 job 和机器可读原因见 `experiment_evidence_matrix.csv`；逐视频重分类见 `experiment_evidence_jobs.csv`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    models: Mapping[str, Path],
    *,
    scan_csv: Path,
    metric_table: Path | None,
    output: Path,
) -> dict[str, Any]:
    scans = _read_csv(scan_csv)
    jobs = _load_jobs(models)
    matrix = _experiment_rows(jobs, scans, list(models))
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "experiment_evidence_jobs.csv", jobs)
    _write_csv(output / "experiment_evidence_matrix.csv", matrix)
    model_summaries = _model_evidence_summaries(matrix, jobs, list(models))
    experiment_summaries = _experiment_conclusion_summaries(matrix, list(models))
    _write_csv(output / "model_evidence_summary.csv", model_summaries)
    _write_csv(
        output / "experiment_conclusion_summary.csv", experiment_summaries
    )
    paper_rows: list[dict[str, Any]] = []
    if metric_table is not None:
        paper_rows = _merge_paper_metrics(
            model_summaries,
            _read_csv(metric_table),
        )
        _write_csv(output / "paper_evidence_main_table.csv", paper_rows)
        _paper_main_table_markdown(
            output / "PAPER_MAIN_TABLE_AUDITABLE_ZH.md", paper_rows
        )
    _matrix_svg(output / "experiment_evidence_matrix.svg", matrix, list(models))
    _markdown_report(
        output / "EXPERIMENT_CONCLUSIONS_ZH.md", matrix, list(models)
    )
    model_state_counts = {
        model: dict(
            Counter(
                str(row["state"]) for row in matrix if row.get("model") == model
            )
        )
        for model in models
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": "baseline_CAM_Side",
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "policy": {
            "event_fail_auto_min_tracked_fraction": EVENT_TRACK_AUTO_MIN,
            "event_fail_review_min_tracked_fraction": EVENT_TRACK_REVIEW_MIN,
            "minimum_negative_target_tuples": MIN_NEGATIVE_TUPLES,
            "minimum_negative_target_tuple_fraction": MIN_NEGATIVE_TUPLE_FRACTION,
            "missing_parameter_estimate_is_not_zero": True,
            "rule_or_event_failure_is_not_a_parameter_score": True,
        },
        "job_count": len(jobs),
        "experiment_model_cell_count": len(matrix),
        "model_state_counts": model_state_counts,
        "model_evidence_summary": model_summaries,
        "experiment_conclusion_summary": experiment_summaries,
        "metric_table": str(metric_table.resolve()) if metric_table else None,
        "paper_main_table_written": bool(paper_rows),
        "output": str(output.resolve()),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=EVALUATION_ROOT",
    )
    parser.add_argument(
        "--metric-table",
        type=Path,
        help="Optional paper_main_table.csv from build_candidate_physics_report.py",
    )
    parser.add_argument(
        "--scan-csv",
        type=Path,
        required=True,
        help="scan_response_channels.csv from build_candidate_physics_report.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_report(
        parse_model_arguments(args.model),
        scan_csv=args.scan_csv.resolve(),
        metric_table=args.metric_table.resolve() if args.metric_table else None,
        output=args.output.resolve(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
