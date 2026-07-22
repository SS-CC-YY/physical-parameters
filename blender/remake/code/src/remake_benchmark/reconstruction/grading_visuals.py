"""Publication-oriented evidence plots for hierarchical physics grades."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .hierarchical_grading import read_trajectory_csv, resolve_trajectory_path, write_json


GRADE_COLOURS = {
    "G0": "#b2182b",
    "G1": "#ef8a62",
    "G2": "#d6604d",
    "G3": "#f6c85f",
    "G4": "#2ca25f",
    "U": "#7f8c8d",
    "PASS_TO_SCAN": "#3288bd",
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, png_path: Path, *, svg: bool = True) -> dict[str, str]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    output = {"png": str(png_path)}
    if svg:
        svg_path = png_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        output["svg"] = str(svg_path)
    return output


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}g}"


def _trajectory_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    output: dict[str, list[float]] = {
        "frame": [], "time": [], "x": [], "y": [], "z": [], "used": [], "interpolated": []
    }
    for ordinal, row in enumerate(rows):
        time = _number(row.get("time_s"))
        x = _number(row.get("fit_x_m"))
        y = _number(row.get("fit_y_m"))
        z = _number(row.get("fit_z_m"))
        if x is None:
            x = _number(row.get("x_m"))
        if y is None:
            y = _number(row.get("y_m"))
        if z is None:
            z = _number(row.get("z_m"))
        if time is None or x is None or z is None:
            continue
        output["frame"].append(float(row.get("source_frame_index", row.get("frame_index", ordinal))))
        output["time"].append(time)
        output["x"].append(x)
        output["y"].append(0.0 if y is None else y)
        output["z"].append(z)
        output["used"].append(float(_truth(row.get("physics_fit_used", row.get("fit_eligible", True)))))
        source = str(row.get("continuous_source") or row.get("measurement_source") or "").lower()
        interpolated = _truth(row.get("interpolated")) or "interpol" in source
        output["interpolated"].append(float(interpolated))
    return {key: np.asarray(value, dtype=float) for key, value in output.items()}


def _collect_fit_series(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        series = value.get("fit_series")
        if isinstance(series, Mapping):
            copied = dict(series)
            copied["diagnostic_path"] = prefix.rstrip(".") or "diagnostics"
            output.append(copied)
        for key, item in value.items():
            if key != "fit_series":
                output.extend(_collect_fit_series(item, f"{prefix}{key}."))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            output.extend(_collect_fit_series(item, f"{prefix}{index}."))
    return output


def _support_parameters(series: Mapping[str, Any]) -> list[str]:
    value = series.get("supports_parameters")
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    return []


def _event_times(
    data: Mapping[str, np.ndarray],
    grade: Mapping[str, Any],
) -> list[tuple[float, str]]:
    frame_to_time = {
        int(frame): float(time)
        for frame, time in zip(data["frame"], data["time"])
    }
    output: list[tuple[float, str]] = []
    for event in grade.get("g1_motion_type_validity", {}).get("event_sequence", []):
        frame = event.get("frame")
        if frame is None or int(frame) not in frame_to_time:
            continue
        output.append((frame_to_time[int(frame)], str(event.get("name"))))
    return output


def write_motion_type_plot(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    grade: Mapping[str, Any],
) -> dict[str, str] | None:
    data = _trajectory_arrays(rows)
    if not len(data["time"]):
        return None
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    motion = grade.get("g1_motion_type_validity", {})
    stage = str(grade.get("video_stage"))
    colour = GRADE_COLOURS.get(stage, "#333333")
    used = data["used"] > 0.5
    interpolated = data["interpolated"] > 0.5
    direct_used = used & ~interpolated
    excluded = ~used & ~interpolated
    axes[0, 0].plot(data["x"], data["z"], color="#4c78a8", linewidth=1.6)
    axes[0, 0].scatter(data["x"][direct_used], data["z"][direct_used], s=12, color="#1f77b4", label="direct fit-used")
    if np.any(excluded):
        axes[0, 0].scatter(data["x"][excluded], data["z"][excluded], s=18, facecolors="none", edgecolors="#d62728", label="excluded")
    if np.any(interpolated):
        axes[0, 0].scatter(data["x"][interpolated], data["z"][interpolated], s=18, marker="x", color="#17becf", label="interpolated/display-only")
    axes[0, 0].scatter(data["x"][0], data["z"][0], marker="o", s=70, color="#2ca02c", label="start")
    axes[0, 0].scatter(data["x"][-1], data["z"][-1], marker="s", s=55, color="#ff7f0e", label="end")
    axes[0, 0].set(title="Metric trajectory and event geometry", xlabel="x (m)", ylabel="z (m)")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="best")

    axes[0, 1].plot(data["time"], data["x"], label="x", color="#4c78a8")
    axes[0, 1].plot(data["time"], data["z"], label="z", color="#e45756")
    for event_time, event_name in _event_times(data, grade):
        axes[0, 1].axvline(event_time, color="#9467bd", alpha=0.45, linewidth=1)
        axes[0, 1].text(event_time, axes[0, 1].get_ylim()[1], event_name, rotation=90, va="top", ha="right", fontsize=7)
    axes[0, 1].set(title="Coordinates and detected events", xlabel="decoded-video time (s)", ylabel="position (m)")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    if len(data["time"]) >= 2:
        vx = np.gradient(data["x"], data["time"])
        vz = np.gradient(data["z"], data["time"])
        axes[1, 0].plot(data["time"], vx, label="vx", color="#4c78a8")
        axes[1, 0].plot(data["time"], vz, label="vz", color="#e45756")
        axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[1, 0].set(title="Kinematic sign evidence", xlabel="decoded-video time (s)", ylabel="velocity (m/s)")
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend()
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "Velocity unavailable: fewer than two metric samples", ha="center", va="center")

    axes[1, 1].axis("off")
    lines = [
        f"Video stage: {stage}",
        f"Expected motion: {motion.get('motion_type')}",
        f"G1 status: {motion.get('status')}",
        "",
        "Target-free predicates:",
    ]
    for check in motion.get("checks", []):
        marker = {"pass": "PASS", "fail": "FAIL", "indeterminate": "U"}.get(str(check.get("status")), "-")
        lines.append(f"[{marker}] {check.get('name')}")
    lines.extend(["", "Reason codes:"] + [f"- {value}" for value in grade.get("reason_codes", [])])
    axes[1, 1].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9, color=colour)
    fig.suptitle(str(grade.get("job_id")), fontsize=13, fontweight="bold")
    output = _save(fig, path)
    plt.close(fig)
    return output


def write_trajectory_fit_evidence(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    grade: Mapping[str, Any],
) -> dict[str, Any] | None:
    data = _trajectory_arrays(rows)
    if not len(data["time"]):
        return None
    fit = result.get("fit", {})
    series = _collect_fit_series(fit.get("diagnostics", {}))
    series.sort(key=lambda item: str(item.get("diagnostic_path", "")))
    target_not_used = fit.get("target_not_used_for_fit")
    target_contract_passed = target_not_used is True
    fit_csv = output_dir / "fit_series.csv"
    fit_csv.parent.mkdir(parents=True, exist_ok=True)
    with fit_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "series_id", "diagnostic_path", "series_name", "supports_parameters",
            "sample_index", "time_s", "observed", "predicted", "residual",
            "time_basis", "target_not_used_for_fit",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for series_id, item in enumerate(series):
            observed = item.get("observed", [])
            predicted = item.get("predicted", [])
            residual = item.get("residual", [])
            times = item.get("time_s", list(range(len(observed))))
            for index in range(min(len(observed), len(predicted), len(residual), len(times))):
                writer.writerow({
                    "series_id": series_id,
                    "diagnostic_path": item.get("diagnostic_path"),
                    "series_name": item.get("series_name"),
                    "supports_parameters": ";".join(_support_parameters(item)),
                    "sample_index": index,
                    "time_s": times[index],
                    "observed": observed[index],
                    "predicted": predicted[index],
                    "residual": residual[index],
                    "time_basis": item.get("time_basis"),
                    "target_not_used_for_fit": target_not_used,
                })
    events_path = output_dir / "fit_events.json"
    write_json(events_path, {
        "schema_version": "1.0.0",
        "target_not_used_for_fit": target_not_used,
        "target_parameters_used": None if target_not_used is None else not bool(target_not_used),
        "target_contract_warning": None if target_contract_passed else "fit does not affirm target_not_used_for_fit=true",
        "events": grade.get("g1_motion_type_validity", {}).get("event_sequence", []),
    })

    plt = _pyplot()
    series_rows = max(1, len(series))
    fig, axes = plt.subplots(
        1 + series_rows,
        2,
        figsize=(14.2, 4.0 + 3.25 * series_rows),
        constrained_layout=True,
        squeeze=False,
    )
    used = data["used"] > 0.5
    interpolated = data["interpolated"] > 0.5
    direct_used = used & ~interpolated
    excluded = ~used & ~interpolated
    axes[0, 0].plot(data["x"], data["z"], color="#9ecae1", linewidth=1.2)
    axes[0, 0].scatter(data["x"][direct_used], data["z"][direct_used], s=13, color="#2171b5", label="direct fit-used")
    if np.any(excluded):
        axes[0, 0].scatter(data["x"][excluded], data["z"][excluded], s=18, facecolors="none", edgecolors="#cb181d", label="excluded")
    if np.any(interpolated):
        axes[0, 0].scatter(data["x"][interpolated], data["z"][interpolated], s=18, marker="x", color="#17becf", label="interpolated/display-only")
    axes[0, 0].set(title="Metric trajectory", xlabel="x (m)", ylabel="z (m)")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].axis("off")
    metrics = result.get("metrics", {})
    estimates = fit.get("parameter_estimates", {})
    contract_line = f"target_not_used_for_fit = {target_not_used!r}"
    lines = [
        f"Video stage: {grade.get('video_stage')}",
        f"Fit method: {fit.get('method')}",
        contract_line,
        "",
        "Parameter              GT        estimate     NAE",
    ]
    for name, estimate in estimates.items():
        metric = metrics.get("parameters", {}).get(name, {})
        gt = metric.get("gt")
        nae = metric.get("normalized_absolute_error")
        lines.append(f"{name[:20]:20s} {_fmt(gt, 5):>9s} {_fmt(estimate, 5):>12s} {_fmt(nae, 4):>9s}")
    lines.extend([
        "",
        f"experiment NMAE: {_fmt(metrics.get('experiment_nmae'), 4)}",
        f"fit complete: {metrics.get('fit_complete')}",
        f"standardized series count: {len(series)}",
        "",
        "Series-to-parameter support:",
    ])
    for index, item in enumerate(series):
        support = _support_parameters(item)
        support_text = ",".join(support) if support else "not explicitly declared by fitter"
        lines.append(f"[{index}] {item.get('diagnostic_path')} -> {support_text}")
    if series and not any(_support_parameters(item) for item in series):
        lines.extend([
            "",
            "LIMIT: residual panels validate diagnostic fits only;",
            "they do not independently prove every parameter's identifiability.",
        ])
    axes[0, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.2,
        color="#b2182b" if not target_contract_passed else "#222222",
    )

    event_times = _event_times(data, grade)
    if series:
        for row_index, item in enumerate(series, 1):
            observed = np.asarray(item.get("observed", []), dtype=float)
            predicted = np.asarray(item.get("predicted", []), dtype=float)
            residual = np.asarray(item.get("residual", []), dtype=float)
            times = np.asarray(item.get("time_s", []), dtype=float)
            length = min(len(observed), len(predicted), len(residual), len(times))
            observed, predicted, residual, times = (
                observed[:length], predicted[:length], residual[:length], times[:length]
            )
            support = _support_parameters(item)
            support_text = ", ".join(support) if support else "support mapping not declared"
            title = (
                f"Series {row_index - 1}: {item.get('series_name')} | "
                f"{item.get('diagnostic_path')} | {support_text}"
            )
            if length:
                axes[row_index, 0].scatter(times, observed, s=14, color="#2171b5", label="observed used by fitter")
                axes[row_index, 0].plot(times, predicted, color="#cb181d", linewidth=1.8, label="physical fit prediction")
                axes[row_index, 0].set(title=title, xlabel=str(item.get("time_basis") or "fit sample"), ylabel=str(item.get("series_name")))
                axes[row_index, 0].grid(alpha=0.25)
                axes[row_index, 0].legend(fontsize=8)
                axes[row_index, 1].scatter(times, residual, s=12, color="#6a51a3")
                axes[row_index, 1].axhline(0.0, color="black", linewidth=0.9)
                rmse = float(np.sqrt(np.mean(residual**2)))
                axes[row_index, 1].set(title=f"Residual for series {row_index - 1} (RMSE={rmse:.4g})", xlabel=str(item.get("time_basis") or "fit sample"), ylabel="observed - fitted")
                axes[row_index, 1].grid(alpha=0.25)
                if item.get("time_basis") == "video_time_s":
                    for event_time, event_name in event_times:
                        for axis in (axes[row_index, 0], axes[row_index, 1]):
                            axis.axvline(event_time, color="#9467bd", alpha=0.5, linewidth=1)
                        axes[row_index, 0].text(event_time, axes[row_index, 0].get_ylim()[1], event_name, rotation=90, va="top", ha="right", fontsize=7)
            else:
                for axis in axes[row_index]:
                    axis.axis("off")
                    axis.text(0.5, 0.5, f"{title}\nno aligned observed/predicted/residual samples", ha="center", va="center")
    else:
        for axis, title in zip(
            axes[1],
            ("Standardized fitted series unavailable", "Residual unavailable"),
        ):
            axis.axis("off")
            axis.text(0.5, 0.5, title, ha="center", va="center", color="#7f8c8d")

    warning = "" if target_contract_passed else " — WARNING: target-leakage contract is false or missing"
    fig.suptitle(f"{grade.get('job_id')}{warning}", fontsize=13, fontweight="bold", color="#b2182b" if warning else "#222222")
    plots = _save(fig, output_dir / "trajectory_fit.png")
    plt.close(fig)
    return {
        "fit_series_csv": str(fit_csv),
        "fit_events_json": str(events_path),
        "trajectory_fit_plot": plots,
        "standardized_fit_series_count": len(series),
        "target_not_used_for_fit": target_not_used,
        "series_support_mapping_complete": bool(series) and all(bool(_support_parameters(item)) for item in series),
    }


def write_video_grade_card(path: Path, grade: Mapping[str, Any]) -> dict[str, str]:
    plt = _pyplot()
    fig = plt.figure(figsize=(12.0, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.5])
    left = fig.add_subplot(grid[0, 0])
    right = fig.add_subplot(grid[0, 1])
    stage = str(grade.get("video_stage"))
    colour = GRADE_COLOURS.get(stage, "#333333")
    left.axis("off")
    left.text(0.5, 0.75, stage, ha="center", va="center", fontsize=56, fontweight="bold", color=colour)
    left.text(0.5, 0.57, "single-video gate", ha="center", va="center", fontsize=12)
    gates = [
        ("G0 appearance/rigidity", grade.get("g0_generation_validity", {}).get("status")),
        ("G0 contact geometry", grade.get("g0_contact_geometry", {}).get("status")),
        ("G0 dynamic 3D rigidity", grade.get("g0_dynamic_rigidity", {}).get("status")),
        ("G1 motion topology", grade.get("g1_motion_type_validity", {}).get("status")),
        ("inverse fit", grade.get("inverse_fit", {}).get("status")),
    ]
    y = 0.43
    for name, status in gates:
        left.text(0.08, y, f"{name}: {status}", ha="left", va="center", family="monospace", fontsize=10)
        y -= 0.08
    right.axis("off")
    lines = [
        str(grade.get("job_id")),
        "",
        "Decision reasons:",
        *[f"- {value}" for value in grade.get("reason_codes", [])],
        "",
        "Evidence contract:",
        "- G0/G1 use no target parameter value.",
        "- U means missing evidence, not model failure.",
        "- G2/G3/G4 are decided only on matched scans.",
        "",
        "Referenced artifacts:",
    ]
    for name, value in grade.get("evidence_paths", {}).items():
        if value:
            display = value.get("png") if isinstance(value, Mapping) else value
            try:
                display = Path(str(display)).name
            except (TypeError, ValueError):
                display = str(display)
            lines.append(f"- {name}: {display}")
    right.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5, wrap=True)
    output = _save(fig, path)
    plt.close(fig)
    return output


def write_parameter_response_plot(path: Path, scan: Mapping[str, Any]) -> dict[str, str]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.7, 6.5), constrained_layout=True)
    low, high = (float(value) for value in scan["valid_range"])
    span = high - low
    levels = scan.get("levels", [])
    targets = np.asarray([float(level["target_value"]) for level in levels], dtype=float)
    medians = np.asarray([np.nan if level.get("median_estimate") is None else float(level["median_estimate"]) for level in levels], dtype=float)
    tolerance = float(scan.get("decision", {}).get("threshold_snapshot", {}).get("g4_median_valid_range_nae", 0.10)) * span
    xline = np.linspace(min(low, float(np.min(targets))), max(high, float(np.max(targets))), 200)
    axis.fill_between(xline, xline - tolerance, xline + tolerance, color="#c7e9c0", alpha=0.45, label="G4 point-error band")
    axis.plot(xline, xline, linestyle="--", color="#238b45", linewidth=1.6, label="ideal y=x")
    for level, x, y in zip(levels, targets, medians):
        if not math.isfinite(float(y)):
            axis.scatter([x], [x], marker="x", s=80, color=GRADE_COLOURS["U"], label="missing estimate")
            continue
        ci = level.get("median_estimate_ci95")
        if ci:
            axis.errorbar([x], [y], yerr=[[y - float(ci[0])], [float(ci[1]) - y]], fmt="o", color="#2b8cbe", capsize=4)
        else:
            axis.scatter([x], [y], s=55, color="#2b8cbe")
        axis.annotate(f"{level.get('usable_job_count')}/{level.get('planned_job_count')}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    finite = np.isfinite(medians)
    if int(np.sum(finite)) >= 2:
        axis.plot(targets[finite], medians[finite], color="#2b8cbe", alpha=0.65)
    decision = scan.get("decision", {})
    direction = decision.get("direction", {})
    numeric = decision.get("numeric_accuracy", {})
    off_target_values = [
        float(value)
        for value in numeric.get("off_target_drift_nad", {}).values()
        if _number(value) is not None
    ]
    trajectory_quality = numeric.get("p90_per_job_worst_trajectory_fit_nrmse")
    trajectory_quality_label = "p90 per-job worst trajectory NRMSE"
    if trajectory_quality is None:
        trajectory_quality = numeric.get("median_trajectory_fit_nrmse")
        trajectory_quality_label = "median trajectory NRMSE (legacy)"
    grade = str(scan.get("response_grade"))
    has_ci = any(level.get("median_estimate_ci95") for level in levels)
    if scan.get("entity_scope") == "cross_scene_parameter_scan" and has_ci:
        ci_text = "95% CI: bootstrap of the cross-scene median (camera/seed fixed)"
    else:
        ci_text = "95% CI: unavailable for this single-scene/single-seed atomic scan"
    axis.set(
        title=f"{scan.get('experiment_id')} / {scan.get('target_parameter')} — {grade}",
        xlabel=f"requested {scan.get('target_parameter')} ({scan.get('parameter_unit')})",
        ylabel=f"trajectory-inferred {scan.get('target_parameter')} ({scan.get('parameter_unit')})",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    text = (
        f"levels usable/planned: {decision.get('evidence', {}).get('usable_level_count')}/{decision.get('evidence', {}).get('planned_level_count')}\n"
        f"pairwise direction: {_fmt(direction.get('pairwise_direction_concordance'))}\n"
        f"Spearman rho: {_fmt(direction.get('spearman_rho'))}\n"
        f"Theil-Sen slope: {_fmt(direction.get('theil_sen_slope'))}\n"
        f"median NAE: {_fmt(numeric.get('median_valid_range_nae'))}\n"
        f"{trajectory_quality_label}: {_fmt(trajectory_quality)}\n"
        f"max off-target drift: {_fmt(max(off_target_values) if off_target_values else None)}\n"
        f"evidence: {scan.get('evidence_strength')}\n"
        f"{ci_text}"
    )
    axis.text(0.02, 0.98, text, transform=axis.transAxes, va="top", ha="left", fontsize=8.5, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": GRADE_COLOURS.get(grade, "#777777")})
    output = _save(fig, path)
    plt.close(fig)
    return output


def write_trajectory_comparison(
    path: Path,
    scan: Mapping[str, Any],
    results_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    curves: list[dict[str, Any]] = []
    for level in scan.get("levels", []):
        usable = level.get("usable_job_ids", [])
        for job_value in usable:
            job_id = str(job_value)
            result = results_by_id.get(job_id)
            path_value = None if result is None else resolve_trajectory_path(result)
            if path_value is None:
                continue
            rows = read_trajectory_csv(path_value)
            data = _trajectory_arrays(rows)
            if not len(data["time"]):
                continue
            job = result.get("job", {}) if isinstance(result, Mapping) else {}
            scene = str(job.get("scene_id") or "unknown_scene")
            curves.append({
                "target": float(level["target_value"]),
                "job_id": job_id,
                "scene": scene,
                "data": data,
            })
    if not curves:
        return None
    plt = _pyplot()
    targets = sorted({float(curve["target"]) for curve in curves})
    scenes = sorted({str(curve["scene"]) for curve in curves})
    scene_colours = {
        scene: plt.cm.tab10(index % 10)
        for index, scene in enumerate(scenes)
    }
    fig, axes = plt.subplots(
        len(targets),
        3,
        figsize=(16.0, max(4.8, 3.75 * len(targets))),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, target in enumerate(targets):
        target_curves = [curve for curve in curves if float(curve["target"]) == target]
        for curve in target_curves:
            data = curve["data"]
            scene = str(curve["scene"])
            colour = scene_colours[scene]
            label = f"{scene} | {curve['job_id']}"
            axes[row_index, 0].plot(data["x"], data["z"], color=colour, alpha=0.58, linewidth=1.25, label=label)
            axes[row_index, 1].plot(data["time"], data["x"], color=colour, alpha=0.58, linewidth=1.25)
            axes[row_index, 2].plot(data["time"], data["z"], color=colour, alpha=0.58, linewidth=1.25)
        axes[row_index, 0].set(title=f"target={target:g}: all usable scenes, metric path", xlabel="x (m)", ylabel="z (m)")
        axes[row_index, 1].set(title=f"target={target:g}: x(t), original video time", xlabel="time (s)", ylabel="x (m)")
        axes[row_index, 2].set(title=f"target={target:g}: z(t), original video time", xlabel="time (s)", ylabel="z (m)")
        axes[row_index, 0].legend(fontsize=6.4, loc="best")
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)

    all_x = np.concatenate([curve["data"]["x"] for curve in curves])
    all_z = np.concatenate([curve["data"]["z"] for curve in curves])
    all_t = np.concatenate([curve["data"]["time"] for curve in curves])

    def padded_limits(values: np.ndarray) -> tuple[float, float]:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if high - low <= 1e-12:
            padding = max(abs(low) * 0.05, 0.5)
        else:
            padding = 0.05 * (high - low)
        return low - padding, high + padding

    x_limits = padded_limits(all_x)
    z_limits = padded_limits(all_z)
    t_limits = padded_limits(all_t)
    for row_index in range(len(targets)):
        axes[row_index, 0].set_xlim(*x_limits)
        axes[row_index, 0].set_ylim(*z_limits)
        axes[row_index, 0].set_aspect("equal", adjustable="box")
        axes[row_index, 1].set_xlim(*t_limits)
        axes[row_index, 1].set_ylim(*x_limits)
        axes[row_index, 2].set_xlim(*t_limits)
        axes[row_index, 2].set_ylim(*z_limits)

    scope_note = (
        "Cross-scene aggregate: every usable scene is drawn and labelled; "
        "no first-job or synthetic median trajectory is presented."
        if scan.get("entity_scope") == "cross_scene_parameter_scan"
        else "Atomic scan: every usable job is drawn; all rows share global metric/time limits."
    )
    fig.suptitle(f"{scan.get('scan_id')}\n{scope_note}", fontsize=10.5, fontweight="bold")
    output = _save(fig, path)
    plt.close(fig)
    output["selection_rule"] = "all_usable_jobs_grouped_by_target_and_labelled_by_scene"
    output["plotted_job_ids"] = [str(curve["job_id"]) for curve in curves]
    output["plotted_scenes"] = scenes
    return output


def write_grade_funnel(path: Path, video_grades: Sequence[Mapping[str, Any]], scans: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    plt = _pyplot()
    stages = Counter(str(row.get("video_stage")) for row in video_grades)
    response = Counter(str(row.get("response_grade")) for row in scans if row.get("canonical_for_target"))
    labels = ["G0", "G1", "U(video)", "PASS_TO_SCAN", "G2", "G3", "G4", "U(scan)"]
    values = [stages.get("G0", 0), stages.get("G1", 0), stages.get("U", 0), stages.get("PASS_TO_SCAN", 0), response.get("G2", 0), response.get("G3", 0), response.get("G4", 0), response.get("U", 0)]
    colours = [GRADE_COLOURS["G0"], GRADE_COLOURS["G1"], GRADE_COLOURS["U"], GRADE_COLOURS["PASS_TO_SCAN"], GRADE_COLOURS["G2"], GRADE_COLOURS["G3"], GRADE_COLOURS["G4"], GRADE_COLOURS["U"]]
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    bars = axis.bar(labels, values, color=colours)
    axis.bar_label(bars, padding=3)
    axis.set(title="Hierarchical grading evidence counts (scopes kept separate)", ylabel="entity count")
    axis.grid(axis="y", alpha=0.2)
    axis.axvline(3.5, color="black", linestyle="--", alpha=0.6)
    axis.text(1.5, max(values or [1]) * 1.05, "single-video gates", ha="center", fontsize=9)
    axis.text(5.5, max(values or [1]) * 1.05, "canonical parameter scans", ha="center", fontsize=9)
    output = _save(fig, path)
    plt.close(fig)
    return output


__all__ = [
    "write_grade_funnel",
    "write_motion_type_plot",
    "write_parameter_response_plot",
    "write_trajectory_comparison",
    "write_trajectory_fit_evidence",
    "write_video_grade_card",
]
