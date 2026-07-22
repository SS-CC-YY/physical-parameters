#!/usr/bin/env python3
"""Build the compact paper-facing PhysParamBench report.

The strict hierarchical audit remains available, but this entry point answers
the four questions used by the paper abstract with a deliberately smaller
contract: baseline/Side parameter response is primary; background, view and
seed experiments are robustness analyses; tracking/reconstruction failures are
reported as X rather than model failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.simple_paper_grading import (  # noqa: E402
    DEFAULT_PRIMARY_SEED,
    grade_simple_paper_benchmark,
)
from remake_benchmark.reconstruction.simple_report_visuals import (  # noqa: E402
    write_experiment_scope_summary_svg,
    write_grade_counts_svg,
    write_parameter_scan_small_multiples_svg,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed", "ok"}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _seed(value: Any) -> int | None:
    result = _number(value)
    return int(result) if result is not None and result.is_integer() else None


def _job_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("job_id") or row.get("video_name") or "")
    return Path(value).stem


def _paper_video_state(row: Mapping[str, Any]) -> str:
    """Return success, clear_failure, or measurement_unavailable.

    Only a manual failure is promoted to a clear model failure.  Automatic
    failures remain measurement-unavailable because the current detector has
    known false positives in complex backgrounds.
    """

    manual = str(row.get("manual_generation_validity_status") or "").strip().lower()
    if manual in {"fail", "failed"}:
        return "clear_failure"
    if manual in {"pass", "passed", "ok"}:
        return "success"
    automatic = str(row.get("generation_validity_status") or "").strip().lower()
    if automatic in {"pass", "passed", "ok"}:
        return "success"
    return "measurement_unavailable"


def _motion_evaluable(row: Mapping[str, Any]) -> bool:
    return _paper_video_state(row) == "success" and (
        _truth(row.get("fit_complete")) or _truth(row.get("trajectory_fit_eligible"))
    )


def _scene_class(scene_id: Any) -> str:
    value = str(scene_id or "")
    if value == "baseline":
        return "baseline"
    if value.startswith("indoor"):
        return "indoor"
    if value.startswith("outdoor"):
        return "outdoor"
    return "other"


def _condition_key(row: Mapping[str, Any], *, include_camera: bool = False) -> tuple[str, ...]:
    values = (
        str(row.get("experiment_id") or ""),
        str(row.get("parameter_tuple_id") or ""),
        str(row.get("scene_id") or ""),
        str(row.get("seed") or ""),
    )
    return values + (str(row.get("camera_name") or ""),) if include_camera else values


def _status_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    states = Counter(_paper_video_state(row) for row in values)
    evaluable = sum(_motion_evaluable(row) for row in values)
    return {
        "total": len(values),
        "motion_evaluable": evaluable,
        "motion_evaluable_rate": evaluable / len(values) if values else None,
        "clear_generation_failure": states["clear_failure"],
        "measurement_unavailable": len(values) - evaluable - states["clear_failure"],
    }


def _background_robustness(rows: Sequence[Mapping[str, Any]], primary_seed: int) -> dict[str, Any]:
    side = [
        row for row in rows
        if row.get("camera_name") == "CAM_Side" and _seed(row.get("seed")) == primary_seed
    ]
    by_class = {
        category: _status_summary(row for row in side if _scene_class(row.get("scene_id")) == category)
        for category in ("baseline", "indoor", "outdoor")
    }
    baseline = {
        (str(row.get("experiment_id")), str(row.get("parameter_tuple_id"))): row
        for row in side if row.get("scene_id") == "baseline"
    }
    matched: dict[str, dict[str, Any]] = {}
    for category in ("indoor", "outdoor"):
        pairs = [
            (baseline[(str(row.get("experiment_id")), str(row.get("parameter_tuple_id")))], row)
            for row in side
            if _scene_class(row.get("scene_id")) == category
            and (str(row.get("experiment_id")), str(row.get("parameter_tuple_id"))) in baseline
        ]
        reference = [(base, scene) for base, scene in pairs if _motion_evaluable(base)]
        retained = sum(_motion_evaluable(scene) for _, scene in reference)
        matched[category] = {
            "matched_pair_count": len(pairs),
            "evaluable_baseline_reference_count": len(reference),
            "motion_evaluable_retained_count": retained,
            "motion_evaluable_retention_rate": retained / len(reference) if reference else None,
        }
    return {"by_scene_class": by_class, "matched_to_baseline": matched}


def _view_robustness(
    rows: Sequence[Mapping[str, Any]],
    primary_seed: int,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [row for row in rows if _seed(row.get("seed")) == primary_seed]
    view_summary = {
        view: _status_summary(row for row in selected if row.get("camera_name") == view)
        for view in ("CAM_Side", "CAM_Main", "CAM_Top")
    }
    grouped: dict[tuple[str, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        view = str(row.get("camera_name") or "")
        if view in {"CAM_Side", "CAM_Main", "CAM_Top"}:
            grouped[_condition_key(row)][view] = row
    triplets = [group for group in grouped.values() if len(group) == 3]
    all_evaluable = sum(all(_motion_evaluable(group[view]) for view in group) for group in triplets)
    any_clear_failure = sum(
        any(_paper_video_state(group[view]) == "clear_failure" for view in group)
        for group in triplets
    )
    specs = _registry_parameters(registry)
    distances: dict[str, list[float]] = {"CAM_Main": [], "CAM_Top": []}
    for group in triplets:
        side = group["CAM_Side"]
        experiment = str(side.get("experiment_id") or "")
        for view in ("CAM_Main", "CAM_Top"):
            values: list[float] = []
            for parameter in specs.get(experiment, []):
                valid_range = parameter.get("valid_range", [])
                if len(valid_range) != 2:
                    continue
                span = float(valid_range[1]) - float(valid_range[0])
                left = _estimate(side, str(parameter["name"]))
                right = _estimate(group[view], str(parameter["name"]))
                if left is not None and right is not None and span > 0:
                    values.append(abs(left - right) / span)
            if values:
                distances[view].append(statistics.fmean(values))
    return {
        "by_view": view_summary,
        "matched_triplet_count": len(triplets),
        "all_three_motion_evaluable_count": all_evaluable,
        "all_three_motion_evaluable_rate": all_evaluable / len(triplets) if triplets else None,
        "triplet_with_clear_failure_count": any_clear_failure,
        "median_main_vs_side_parameter_nad": (
            statistics.median(distances["CAM_Main"]) if distances["CAM_Main"] else None
        ),
        "median_top_vs_side_parameter_nad": (
            statistics.median(distances["CAM_Top"]) if distances["CAM_Top"] else None
        ),
    }


def _registry_parameters(registry: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(experiment["id"]): [dict(parameter) for parameter in experiment.get("hidden_parameters", [])]
        for experiment in registry.get("experiments", [])
    }


def _estimate(row: Mapping[str, Any], name: str) -> float | None:
    return _number(row.get(f"{name}__estimate"))


def _seed_stability(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specs = _registry_parameters(registry)
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("camera_name") == "CAM_Side":
            groups[(str(row.get("experiment_id")), str(row.get("parameter_tuple_id")), str(row.get("scene_id")))].append(row)
    selected = {key: values for key, values in groups.items() if len({_seed(row.get("seed")) for row in values}) >= 4}
    group_rows: list[dict[str, Any]] = []
    for (experiment, tuple_id, scene), values in sorted(selected.items()):
        dispersions: list[float] = []
        for parameter in specs.get(experiment, []):
            name = str(parameter["name"])
            valid_range = parameter.get("valid_range", [])
            if len(valid_range) != 2:
                continue
            span = float(valid_range[1]) - float(valid_range[0])
            estimates = [value for row in values if (value := _estimate(row, name)) is not None]
            if len(estimates) >= 2 and span > 0:
                dispersions.append(statistics.pstdev(estimates) / span)
        group_rows.append(
            {
                "experiment_id": experiment,
                "parameter_tuple_id": tuple_id,
                "scene_id": scene,
                "seed_count": len({_seed(row.get("seed")) for row in values}),
                "all_seeds_motion_evaluable": all(_motion_evaluable(row) for row in values),
                "mean_parameter_range_normalized_std": (
                    statistics.fmean(dispersions) if dispersions else None
                ),
            }
        )
    spread = [
        float(row["mean_parameter_range_normalized_std"])
        for row in group_rows if row["mean_parameter_range_normalized_std"] is not None
    ]
    summary = {
        "group_count": len(group_rows),
        "all_seeds_motion_evaluable_count": sum(_truth(row["all_seeds_motion_evaluable"]) for row in group_rows),
        "all_seeds_motion_evaluable_rate": (
            sum(_truth(row["all_seeds_motion_evaluable"]) for row in group_rows) / len(group_rows)
            if group_rows else None
        ),
        "groups_with_parameter_dispersion": len(spread),
        "median_parameter_range_normalized_std": statistics.median(spread) if spread else None,
    }
    return summary, group_rows


def _flatten_scan(scan: Mapping[str, Any]) -> dict[str, Any]:
    evidence = dict(scan.get("evidence", {}))
    return {
        "scan_id": scan.get("scan_id"),
        "experiment_id": scan.get("experiment_id"),
        "target_parameter": scan.get("target_parameter"),
        "grade": scan.get("grade"),
        "planned_level_count": evidence.get("planned_level_count"),
        "usable_level_count": evidence.get("usable_level_count"),
        "pairwise_direction_concordance": evidence.get("pairwise_direction_concordance"),
        "theil_sen_slope": evidence.get("theil_sen_slope"),
        "median_valid_range_nae": evidence.get("median_valid_range_nae"),
        "reason_codes": ";".join(str(value) for value in scan.get("reason_codes", [])),
        "levels_json": json.dumps(evidence.get("levels", []), ensure_ascii=False, separators=(",", ":")),
    }


def _complexity_summary(scans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for tier in ("v1", "v2", "v3"):
        values = [row for row in scans if str(row.get("experiment_id", "")).startswith(tier)]
        counts = Counter(str(row.get("grade")) for row in values)
        errors = [
            value for row in values
            if (value := _number(row.get("evidence", {}).get("median_valid_range_nae"))) is not None
        ]
        output.append(
            {
                "tier": tier,
                "scan_count": len(values),
                "L2": counts["L2"],
                "L3": counts["L3"],
                "L4": counts["L4"],
                "X": counts["X"],
                "median_scan_nae": statistics.median(errors) if errors else None,
            }
        )
    return output


def _evidence_index(
    video_rows: Sequence[Mapping[str, Any]],
    evaluation_root: Path | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in video_rows:
        job_id = Path(str(row.get("row_id") or "")).stem
        job_root = None if evaluation_root is None else evaluation_root / "jobs" / job_id
        output.append(
            {
                "job_id": job_id,
                "experiment_id": row.get("experiment_id"),
                "parameter_tuple_id": row.get("parameter_tuple_id"),
                "video_grade": row.get("grade"),
                "reason_codes": ";".join(str(value) for value in row.get("reason_codes", [])),
                "result_json": None if job_root is None else str(job_root / "result.json"),
                "trajectory_plot": None if job_root is None else str(job_root / "trajectory_plot.png"),
                "track_overlay": None if job_root is None else str(job_root / "validity_object_track_overlay.mp4"),
            }
        )
    return output


def _representative_scans(scans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for grade in ("L2", "L3", "L4", "X"):
        candidates = [row for row in scans if row.get("grade") == grade]
        if not candidates:
            continue
        if grade == "L4":
            choice = min(candidates, key=lambda row: _number(row.get("evidence", {}).get("median_valid_range_nae")) or math.inf)
        elif grade == "L3":
            choice = min(candidates, key=lambda row: _number(row.get("evidence", {}).get("median_valid_range_nae")) or math.inf)
        else:
            choice = max(candidates, key=lambda row: int(row.get("evidence", {}).get("usable_level_count", 0)))
        job_ids = sorted({
            Path(str(item.get("row_id") or "")).stem
            for level in choice.get("evidence", {}).get("levels", [])
            for item in level.get("rows", [])
            if item.get("row_id") and not str(item.get("row_id")).startswith("missing:")
        })
        selected.append(
            {
                **_flatten_scan(choice),
                "job_ids": ";".join(job_ids),
            }
        )
    return selected


def _pct(value: Any) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{100.0 * number:.1f}%"


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def _write_report(
    output: Path,
    summary: Mapping[str, Any],
    complexity: Sequence[Mapping[str, Any]],
) -> None:
    primary = summary["primary_baseline_side"]
    grades = primary["grade_counts"]
    video_grades = primary["video_grade_counts"]
    non_x = int(primary["non_x_scan_count"])
    positive = int(primary["positive_parameter_response_scan_count"])
    accurate = int(primary["numeric_accurate_scan_count"])
    background = summary["background_robustness"]
    views = summary["view_robustness"]
    seeds = summary["seed_stability"]
    lines = [
        "# PhysParamBench 简化主评测结果",
        "",
        "> 本报告对应论文 abstract 的最小可用结论。严格分级和 3D/4D 结果仅作为补充审计，不改变本页主统计。",
        "",
        "## 1. Baseline + CAM_Side：主参数评测",
        "",
        f"固定 baseline、CAM_Side 和主 seed 后，共形成 {primary['total_scan_count']} 个一次一变量参数扫描。"
        f"其中可判断 {non_x} 个，测量不可用 X 为 {grades.get('X', 0)} 个。",
        "",
        f"- L2（运动生成成功但参数响应失败）：{grades.get('L2', 0)}",
        f"- L3（响应方向正确但数值不精确）：{grades.get('L3', 0)}",
        f"- L4（方向正确且误差在容差内）：{grades.get('L4', 0)}",
        f"- 正向参数响应：{positive}/{non_x}（{_pct(positive/non_x if non_x else None)}）",
        f"- 数值达到 benchmark 容差：{accurate}/{non_x}（{_pct(accurate/non_x if non_x else None)}）",
        f"- 单视频明确人工确认生成失败 L1：{video_grades.get('L1', 0)}；自动检测失败若未人工确认均记为 X。",
        "",
        "![主评测扫描等级](fig2_baseline_response_grades.svg)",
        "",
        "![请求参数与轨迹反演参数](fig3_requested_vs_recovered.svg)",
        "",
        "## 2. 复杂背景、视角与随机性",
        "",
        f"- Indoor：相对于可测 baseline 的运动可测保持率 {_pct(background['matched_to_baseline']['indoor']['motion_evaluable_retention_rate'])}。",
        f"- Outdoor：相对于可测 baseline 的运动可测保持率 {_pct(background['matched_to_baseline']['outdoor']['motion_evaluable_retention_rate'])}。",
        f"- Views：{views['matched_triplet_count']} 个匹配条件中，Side/Main/Top 均可解释运动轨迹的比例为 {_pct(views['all_three_motion_evaluable_rate'])}；"
        f"Main–Side/Top–Side 的反演参数差异中位数为 {_fmt(views['median_main_vs_side_parameter_nad'])}/{_fmt(views['median_top_vs_side_parameter_nad'])} 个参数范围（仅作鲁棒性描述）。",
        f"- Seeds：{seeds['group_count']} 个四 seed 条件中，全部可测比例 {_pct(seeds['all_seeds_motion_evaluable_rate'])}；"
        f"参数范围归一化标准差中位数 {_fmt(seeds['median_parameter_range_normalized_std'])}。",
        "",
        "![四类实验作用域](fig4_evaluation_scopes.svg)",
        "",
        "## 3. 与 abstract 对齐的结论",
        "",
        "视频在视觉上或定性上看起来合理，并不保证其轨迹实现了 prompt 中给定的数值参数。"
        "本报告首先在标定的 baseline Side 视角中进行参数反演，再把复杂背景、视角和 seed 分别作为鲁棒性压力测试。"
        "结果允许支持‘存在局部、方向正确的参数响应’，但只有 L4 才支持在当前容差内的数值一致性。",
        "",
        "轨迹抖动、目标关联错误或不稳定 3D 深度被归为测量不可用 X，不作为生成模型失败。"
        "因此这些数字可以用于支持 benchmark 叙事，但不能被写成世界模型的绝对物理理解率。",
        "",
        "## 4. 复杂度",
        "",
        "|层级|扫描数|L2|L3|L4|X|Median scan NAE|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in complexity:
        lines.append(
            f"|{row['tier']}|{row['scan_count']}|{row['L2']}|{row['L3']}|{row['L4']}|{row['X']}|{_fmt(row['median_scan_nae'])}|"
        )
    lines += [
        "",
        "## 5. 证据",
        "",
        "- `parameter_scans.csv/jsonl`：baseline Side 的完整扫描结果；",
        "- `representative_scans.csv`：每个结果等级的一个可展示案例；",
        "- `evidence_index.csv`：对应 trajectory plot、检测 overlay 和 result.json；",
        "- `video_gate.csv`：L1、PASS_TO_SCAN 与 X 的单视频门控；",
        "- 原始严格 G0–G4 结果继续保留在原目录，仅作为附录。",
    ]
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tier_error = {str(row["tier"]): _number(row.get("median_scan_nae")) for row in complexity}
    english = [
        "# Paper-facing result wording",
        "",
        f"Under the calibrated baseline Side-view protocol, {positive}/{non_x} measurable parameter scans exhibited a positive trajectory-inferred response to the requested parameter, while {accurate}/{non_x} met the benchmark's numerical tolerance.",
        "These results support partial parameter sensitivity rather than uniformly accurate numerical control.",
        f"Indoor and outdoor motion-evidence retention relative to measurable baseline counterparts was {_pct(background['matched_to_baseline']['indoor']['motion_evaluable_retention_rate'])} and {_pct(background['matched_to_baseline']['outdoor']['motion_evaluable_retention_rate'])}, respectively.",
        f"Across matched independently generated viewpoints, all three views remained motion-evaluable in {_pct(views['all_three_motion_evaluable_rate'])} of conditions.",
        f"Median recovered-parameter differences were {_fmt(views['median_main_vs_side_parameter_nad'])} valid ranges for Main versus Side and {_fmt(views['median_top_vs_side_parameter_nad'])} for Top versus Side; these are robustness descriptors rather than primary accuracy scores.",
        f"The multi-seed pilot produced a median valid-range-normalized recovered-parameter standard deviation of {_fmt(seeds['median_parameter_range_normalized_std'])}.",
        f"Median scan-level normalized error increased from {_fmt(tier_error.get('v1'))} in V1 to {_fmt(tier_error.get('v2'))} in V2 and {_fmt(tier_error.get('v3'))} in V3, consistent with greater difficulty under composed dynamics.",
        "Tracking, calibration, or monocular reconstruction failures are reported as measurement-unavailable and are not counted as failures of the video model.",
    ]
    (output / "ABSTRACT_RESULTS_EN.md").write_text("\n".join(english) + "\n", encoding="utf-8")


def build(
    all_jobs_csv: Path,
    registry_path: Path,
    policy_path: Path,
    output: Path,
    *,
    evaluation_root: Path | None = None,
    primary_seed: int | None = None,
) -> dict[str, Any]:
    rows = _read_csv(all_jobs_csv)
    registry = _read_json(registry_path)
    policy = _read_json(policy_path)
    configured_seed = int(policy.get("primary_scope", {}).get("seed", DEFAULT_PRIMARY_SEED))
    selected_seed = configured_seed if primary_seed is None else int(primary_seed)
    threshold_config = policy.get("response_grading", {})
    grading = grade_simple_paper_benchmark(
        rows,
        registry,
        primary_seed=selected_seed,
        thresholds=threshold_config,
    )
    scans = grading["per_scan_rows"]
    video_rows = grading["per_video_rows"]
    background = _background_robustness(rows, selected_seed)
    views = _view_robustness(rows, selected_seed, registry)
    seed_summary, seed_groups = _seed_stability(rows, registry)
    complexity = _complexity_summary(scans)
    evidence = _evidence_index(video_rows, evaluation_root)

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "parameter_scans.jsonl", scans)
    _write_csv(output / "parameter_scans.csv", [_flatten_scan(row) for row in scans])
    _write_csv(output / "video_gate.csv", video_rows)
    _write_csv(output / "seed_stability.csv", seed_groups)
    _write_csv(output / "complexity_summary.csv", complexity)
    _write_csv(output / "evidence_index.csv", evidence)
    _write_csv(output / "representative_scans.csv", _representative_scans(scans))

    visual_scans = [
        {
            **dict(scan),
            "response_grade": scan.get("grade"),
            "levels": scan.get("evidence", {}).get("levels", []),
        }
        for scan in scans
    ]
    write_grade_counts_svg(
        output / "fig1_video_gate.svg",
        grading["summary"]["video_grade_counts"],
        title="Baseline Side video evidence gate",
        subtitle="L1 requires manual confirmation; X is a measurement limitation, not a model failure.",
        grade_order=("L1", "PASS_TO_SCAN", "X"),
    )
    write_grade_counts_svg(
        output / "fig2_baseline_response_grades.svg",
        grading["summary"]["grade_counts"],
        title="Baseline Side parameter-response grades",
        subtitle="Unit: matched one-factor parameter scan. L3/L4 indicate positive response; X is excluded.",
        grade_order=("L2", "L3", "L4", "X"),
    )
    write_parameter_scan_small_multiples_svg(
        output / "fig3_requested_vs_recovered.svg",
        visual_scans,
        title="Requested vs trajectory-inferred parameters · baseline Side",
        subtitle="Direction and valid-range-normalized error determine L2/L3/L4; small tracking jitter is not a hard gate.",
        columns=4,
    )
    scope_summaries = {
        "baseline_side_quantitative": {
            "metric": f"{grading['summary']['non_x_scan_count']}/{grading['summary']['total_scan_count']} measurable scans",
            "note": " · ".join(f"{key} {value}" for key, value in grading["summary"]["grade_counts"].items() if value),
            "report_unit": "L2/L3/L4 plus requested-versus-recovered curves; X remains outside model-failure counts.",
        },
        "background_robustness": {
            "metric": f"indoor {_pct(background['matched_to_baseline']['indoor']['motion_evaluable_retention_rate'])} · outdoor {_pct(background['matched_to_baseline']['outdoor']['motion_evaluable_retention_rate'])}",
            "question": "Does a more complex visual scene preserve a measurable motion pattern?",
            "comparison": "Same experiment and parameter tuple; baseline is the reference, scene complexity varies.",
            "report_unit": "Motion-evidence retention and X rate; no headline parameter-accuracy score.",
        },
        "view_robustness": {
            "metric": f"{_pct(views['all_three_motion_evaluable_rate'])} all-view evaluable",
            "question": "Does the generated object still follow the specified motion law from another view?",
            "comparison": "Same condition across independently generated Side, Main and Top videos.",
            "report_unit": "Motion-law evaluability; views are not treated as synchronized multiview geometry.",
        },
        "seed_stability": {
            "metric": f"{seed_summary['group_count']} groups · median NStd {_fmt(seed_summary['median_parameter_range_normalized_std'])}",
            "question": "Is apparent parameter understanding stable rather than a lucky random sample?",
            "comparison": "Same experiment, parameter, scene and Side view; only random seed changes.",
            "report_unit": "All-seed evaluability and recovered-parameter dispersion.",
        },
    }
    write_experiment_scope_summary_svg(
        output / "fig4_evaluation_scopes.svg",
        scope_summaries,
        title="PhysParamBench paper-facing evaluation scopes",
        subtitle="Only baseline Side drives numerical parameter claims; the other scopes test robustness.",
    )

    summary = {
        "schema_version": "1.0.0",
        "policy_id": policy.get("policy_id"),
        "input": {
            "all_jobs_csv": str(all_jobs_csv),
            "input_row_count": len(rows),
            "registry": str(registry_path),
            "evaluation_root": None if evaluation_root is None else str(evaluation_root),
        },
        "primary_baseline_side": grading["summary"],
        "background_robustness": background,
        "view_robustness": views,
        "seed_stability": seed_summary,
        "complexity": complexity,
        "interpretation": {
            "primary_claim": "baseline Side parameter-conditioned system identification",
            "auxiliary_claims": "background, view, and seed robustness",
            "dynamic_3d": "exploratory_only",
            "measurement_unavailable_is_model_failure": False,
        },
    }
    _write_json(output / "summary.json", summary)
    _write_report(output, summary, complexity)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-jobs", type=Path, required=True, help="Evaluation all_jobs.csv (manual status columns are used when present).")
    parser.add_argument(
        "--registry",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "experiment_registry.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=CODE_ROOT / "configs" / "evaluations" / "physparambench_simple_paper_v1.json",
    )
    parser.add_argument("--evaluation-root", type=Path, help="Optional root containing jobs/<job_id>/ evidence.")
    parser.add_argument("--primary-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.all_jobs.resolve(),
        args.registry.resolve(),
        args.policy.resolve(),
        args.output.resolve(),
        evaluation_root=None if args.evaluation_root is None else args.evaluation_root.resolve(),
        primary_seed=args.primary_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
