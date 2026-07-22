"""Dependency-free SVG figures for grading reports.

The main grading pipeline may run in small server environments where
``matplotlib`` is deliberately absent.  This module therefore uses only the
Python standard library and writes self-contained, editable SVG files.

The functions accept ordinary mappings/sequences and, where useful, JSON,
JSONL, or CSV paths.  They do not mutate the supplied grading records.
"""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_FONT = "Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif"
_GRADE_ORDER = (
    "L1", "L2", "L3", "L4",
    "G0", "G1", "G2", "G3", "G4", "U", "X", "PASS_TO_SCAN",
)
_GRADE_COLOURS = {
    "L1": "#B42318",
    "L2": "#64748B",
    "L3": "#2563EB",
    "L4": "#16803B",
    "G0": "#B42318",
    "G1": "#C65D07",
    "G2": "#64748B",
    "G3": "#2563EB",
    "G4": "#16803B",
    "U": "#6B7280",
    "X": "#7C3AED",
    "PASS_TO_SCAN": "#0F766E",
}
_SCOPE_COLOURS = {
    "baseline_side_quantitative": "#2563EB",
    "background_robustness": "#16803B",
    "view_robustness": "#7C3AED",
    "seed_stability": "#C65D07",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_svg(
    path: Path | str,
    *,
    width: int,
    height: int,
    title: str,
    description: str,
    body: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="figure-title figure-description">'
        ),
        f'<title id="figure-title">{_escape(title)}</title>',
        f'<desc id="figure-description">{_escape(description)}</desc>',
        "<defs>",
        "<style>",
        f"text {{ font-family: {_FONT}; fill: #111827; }}",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #4B5563; }",
        ".label { font-size: 13px; font-weight: 600; }",
        ".value { font-size: 13px; font-weight: 700; }",
        ".axis { font-size: 10px; fill: #4B5563; }",
        ".small { font-size: 10px; fill: #4B5563; }",
        ".body { font-size: 12px; fill: #374151; }",
        ".panel-title { font-size: 14px; font-weight: 700; }",
        ".grid { stroke: #D1D5DB; stroke-width: 1; }",
        ".frame { fill: #FFFFFF; stroke: #D1D5DB; stroke-width: 1; }",
        "</style>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        *body,
        "</svg>",
    ]
    output.write_text("\n".join(document) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "svg": str(output),
        "width": width,
        "height": height,
    }
    if metadata:
        result.update(metadata)
    return result


def _load_path(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(dict(value))
        return rows
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"unsupported report input: {path}")


def _records(source: Any) -> list[dict[str, Any]]:
    """Normalize a path, mapping, or sequence to independent record dicts."""

    base_dir: Path | None = None
    if isinstance(source, (str, Path)):
        input_path = Path(source)
        base_dir = input_path.parent
        source = _load_path(input_path)
    if isinstance(source, Mapping):
        for key in (
            "scans",
            "rows",
            "records",
            "parameter_scans",
            "parameter_scan_grades",
            "cross_scene_parameter_scans",
        ):
            value = source.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                source = value
                break
        else:
            if source and all(isinstance(value, Mapping) for value in source.values()):
                source = list(source.values())
            else:
                source = [source]
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise TypeError("expected a Path, mapping, or sequence of mappings")

    output: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, Mapping):
            raise TypeError("every report row must be a mapping")
        row = dict(item)
        if not isinstance(row.get("levels"), Sequence):
            # Evidence-index rows can point at the authoritative group JSON.
            for key in ("grade_json", "group_grade_json", "evidence_json"):
                candidate = row.get(key)
                if not candidate:
                    continue
                candidate_path = Path(str(candidate))
                if not candidate_path.is_absolute() and base_dir is not None:
                    candidate_path = base_dir / candidate_path
                if candidate_path.is_file():
                    loaded = _load_path(candidate_path)
                    if isinstance(loaded, Mapping):
                        row = {**row, **dict(loaded)}
                    break
        output.append(row)
    return output


def _coerce_counts(source: Any) -> dict[str, int]:
    if isinstance(source, (str, Path)):
        source = _load_path(Path(source))
    if isinstance(source, Mapping):
        for key in ("counts", "grade_counts", "response_grade_counts"):
            nested = source.get(key)
            if isinstance(nested, Mapping):
                return _coerce_counts(nested)
        direct: dict[str, int] = {}
        for key, value in source.items():
            numeric = _number(value)
            if numeric is not None and numeric >= 0:
                direct[str(key)] = int(round(numeric))
        if direct:
            return direct
        raise ValueError("count mapping contains no non-negative numeric values")
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        values: list[str] = []
        for row in source:
            if isinstance(row, Mapping):
                grade = next(
                    (
                        row.get(key)
                        for key in ("grade", "response_grade", "video_stage", "terminal_grade")
                        if row.get(key) not in (None, "")
                    ),
                    None,
                )
                if grade is not None:
                    values.append(str(grade))
            elif row not in (None, ""):
                values.append(str(row))
        return dict(Counter(values))
    raise TypeError("counts must be a Path, mapping, or sequence")


def write_grade_counts_svg(
    path: Path | str,
    counts: Any,
    *,
    title: str = "Hierarchical grading counts",
    subtitle: str = "Counts are shown for the supplied evidence scope only.",
    grade_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write a directly labelled horizontal grade-count chart.

    ``counts`` may be a ``{grade: count}`` mapping, a mapping containing a
    ``counts``/``grade_counts`` child, or a sequence of grade records.  G0--G4,
    U, and an optional X category are ordered consistently; any custom labels
    are appended in their input order.
    """

    normalized = _coerce_counts(counts)
    preferred = tuple(str(value) for value in (grade_order or _GRADE_ORDER))
    labels = [value for value in preferred if value in normalized]
    labels.extend(value for value in normalized if value not in labels)
    if not labels:
        raise ValueError("at least one grade count is required")
    total = sum(max(0, int(normalized[label])) for label in labels)
    maximum = max([int(normalized[label]) for label in labels] + [1])
    width = 1000
    top = 112
    row_height = 48
    height = top + len(labels) * row_height + 54
    plot_x = 176
    plot_width = 690
    body = [
        f'<text x="48" y="42" class="title">{_escape(title)}</text>',
        f'<text x="48" y="66" class="subtitle">{_escape(subtitle)}</text>',
        f'<text x="952" y="42" text-anchor="end" class="value">N = {total}</text>',
    ]
    for tick in range(5):
        value = maximum * tick / 4
        x = plot_x + plot_width * tick / 4
        body.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 42}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height - 20}" text-anchor="middle" class="axis">{value:.0f}</text>')
    for index, label in enumerate(labels):
        count = max(0, int(normalized[label]))
        y = top + index * row_height
        bar_width = plot_width * count / maximum
        colour = _GRADE_COLOURS.get(label, "#475569")
        percent = 100.0 * count / total if total else 0.0
        body.extend(
            [
                f'<text x="48" y="{y + 21}" class="label">{_escape(label)}</text>',
                f'<rect x="{plot_x}" y="{y}" width="{bar_width:.1f}" height="28" rx="3" fill="{colour}" data-grade="{_escape(label)}"/>',
                f'<text x="{plot_x + bar_width + 10:.1f}" y="{y + 20}" class="value">{count} <tspan class="small">({percent:.1f}%)</tspan></text>',
            ]
        )
    return _write_svg(
        path,
        width=width,
        height=height,
        title=title,
        description="Horizontal bars report the count and percentage of every supplied grade category.",
        body=body,
        metadata={"counts": {label: int(normalized[label]) for label in labels}, "total": total},
    )


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _ci(level: Mapping[str, Any]) -> tuple[float, float] | None:
    value = level.get("median_estimate_ci95", level.get("recovered_ci95"))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        low, high = _number(value[0]), _number(value[1])
    else:
        low = _first_number(level, ("ci_low", "recovered_ci_low"))
        high = _first_number(level, ("ci_high", "recovered_ci_high"))
    if low is None or high is None:
        return None
    return (min(low, high), max(low, high))


def _scan_levels(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    nested = [dict(row) for row in records if isinstance(row.get("levels"), Sequence)]
    flat = [row for row in records if not isinstance(row.get("levels"), Sequence)]
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in flat:
        if _first_number(row, ("target_value", "target", "requested_value", "requested")) is None:
            continue
        scan_id = str(row.get("scan_id") or row.get("target_parameter") or "parameter_scan")
        groups.setdefault(scan_id, []).append(row)
    for scan_id, rows in groups.items():
        first = rows[0]
        nested.append(
            {
                "scan_id": scan_id,
                "experiment_id": first.get("experiment_id"),
                "target_parameter": first.get("target_parameter", first.get("parameter")),
                "parameter_unit": first.get("parameter_unit", first.get("unit")),
                "response_grade": first.get("response_grade", first.get("grade")),
                "valid_range": first.get("valid_range"),
                "levels": [dict(row) for row in rows],
            }
        )
    return nested


def _level_data(level: Mapping[str, Any]) -> dict[str, Any] | None:
    target = _first_number(level, ("target_value", "target", "requested_value", "requested"))
    if target is None:
        return None
    estimates_value = level.get("estimates", [])
    if isinstance(estimates_value, str):
        try:
            estimates_value = json.loads(estimates_value)
        except json.JSONDecodeError:
            estimates_value = []
    estimates = [
        value
        for value in (_number(item) for item in estimates_value if not isinstance(estimates_value, (str, bytes)))
        if value is not None
    ] if isinstance(estimates_value, Sequence) else []
    recovered = _first_number(
        level,
        ("median_estimate", "recovered_value", "estimated_value", "estimate", "recovered"),
    )
    if recovered is None and estimates:
        recovered = float(statistics.median(estimates))
    return {
        "target": target,
        "recovered": recovered,
        "ci": _ci(level),
        "estimates": estimates,
        "usable": level.get("usable_job_count"),
        "planned": level.get("planned_job_count"),
    }


def write_parameter_scan_small_multiples_svg(
    path: Path | str,
    scans: Any,
    *,
    title: str = "Requested vs trajectory-inferred parameters",
    subtitle: str = "Each panel uses an equal x/y scale; dashed line denotes ideal recovery.",
    columns: int = 3,
    max_panels: int | None = None,
) -> dict[str, Any]:
    """Write target-vs-recovered small multiples from scan/evidence rows.

    Nested grading rows (``levels``), flat level rows, or evidence-index rows
    with a readable ``grade_json`` are accepted.  Individual estimates are
    hollow circles, the persisted/recomputed median is a filled diamond, and a
    persisted confidence interval is a vertical whisker.  No confidence
    interval is invented for atomic scans.
    """

    records = _records(scans)
    normalized = _scan_levels(records)
    if max_panels is not None:
        normalized = normalized[: max(0, int(max_panels))]
    if not normalized:
        raise ValueError("no parameter scan with target levels was found")
    columns = max(1, min(int(columns), 4))
    rows = math.ceil(len(normalized) / columns)
    panel_width, panel_height = 350, 330
    margin_x, top = 42, 104
    width = margin_x * 2 + columns * panel_width
    height = top + rows * panel_height + 34
    body = [
        f'<text x="{margin_x}" y="40" class="title">{_escape(title)}</text>',
        f'<text x="{margin_x}" y="64" class="subtitle">{_escape(subtitle)}</text>',
        '<line x1="42" y1="80" x2="62" y2="80" stroke="#6B7280" stroke-width="1.5" stroke-dasharray="5 4"/>',
        '<text x="68" y="84" class="small">ideal y = x</text>',
        '<circle cx="168" cy="80" r="3.5" fill="#FFFFFF" stroke="#2563EB"/>',
        '<text x="177" y="84" class="small">individual estimate</text>',
        '<path d="M 300 75 l 5 5 l -5 5 l -5 -5 z" fill="#2563EB"/>',
        '<text x="312" y="84" class="small">median</text>',
    ]
    panel_metadata: list[dict[str, Any]] = []
    for index, scan in enumerate(normalized):
        col, row = index % columns, index // columns
        x0 = margin_x + col * panel_width
        y0 = top + row * panel_height
        plot_x, plot_y = x0 + 54, y0 + 62
        plot_size = 226
        points = [value for value in (_level_data(level) for level in scan.get("levels", [])) if value]
        values = [value for point in points for value in (point["target"], point["recovered"]) if value is not None]
        for point in points:
            if point["ci"]:
                values.extend(point["ci"])
            values.extend(point["estimates"])
        valid_range = scan.get("valid_range")
        if isinstance(valid_range, str):
            try:
                valid_range = json.loads(valid_range)
            except json.JSONDecodeError:
                valid_range = None
        if isinstance(valid_range, Sequence) and not isinstance(valid_range, (str, bytes)):
            values.extend(value for value in (_number(item) for item in valid_range[:2]) if value is not None)
        if not values:
            values = [0.0, 1.0]
        low, high = min(values), max(values)
        if high - low <= 1e-12:
            padding = max(abs(low) * 0.1, 0.5)
        else:
            padding = 0.08 * (high - low)
        low, high = low - padding, high + padding

        def sx(value: float) -> float:
            return plot_x + plot_size * (value - low) / (high - low)

        def sy(value: float) -> float:
            return plot_y + plot_size - plot_size * (value - low) / (high - low)

        scan_id = str(scan.get("scan_id") or f"scan-{index + 1}")
        parameter = str(scan.get("target_parameter") or scan.get("parameter") or "parameter")
        experiment = str(scan.get("experiment_id") or "")
        grade = str(scan.get("response_grade") or scan.get("grade") or "")
        unit = str(scan.get("parameter_unit") or scan.get("unit") or "").strip()
        heading = f"{experiment} · {parameter}" if experiment else parameter
        body.extend(
            [
                f'<g data-scan-id="{_escape(scan_id)}">',
                f'<rect x="{x0 + 6}" y="{y0 + 4}" width="{panel_width - 18}" height="{panel_height - 18}" rx="5" class="frame"/>',
                f'<text x="{x0 + 20}" y="{y0 + 29}" class="panel-title">{_escape(heading)}</text>',
                f'<text x="{x0 + panel_width - 28}" y="{y0 + 29}" text-anchor="end" class="value" fill="{_GRADE_COLOURS.get(grade, "#475569")}">{_escape(grade)}</text>',
            ]
        )
        for tick in range(5):
            value = low + (high - low) * tick / 4
            x, y = sx(value), sy(value)
            body.extend(
                [
                    f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_size}" class="grid"/>',
                    f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_size}" y2="{y:.1f}" class="grid"/>',
                    f'<text x="{x:.1f}" y="{plot_y + plot_size + 15}" text-anchor="middle" class="axis">{value:.3g}</text>',
                    f'<text x="{plot_x - 7}" y="{y + 3:.1f}" text-anchor="end" class="axis">{value:.3g}</text>',
                ]
            )
        body.append(
            f'<line x1="{sx(low):.1f}" y1="{sy(low):.1f}" x2="{sx(high):.1f}" y2="{sy(high):.1f}" stroke="#6B7280" stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        rendered_points = 0
        missing_points = 0
        for level_index, point in enumerate(points):
            x = sx(point["target"])
            for estimate_index, estimate in enumerate(point["estimates"]):
                # Deterministic sub-pixel spread keeps coincident replicates visible.
                dx = ((estimate_index % 5) - 2) * 1.4
                body.append(
                    f'<circle cx="{x + dx:.1f}" cy="{sy(estimate):.1f}" r="3.2" fill="#FFFFFF" stroke="#2563EB" stroke-width="1.1"/>'
                )
            recovered = point["recovered"]
            if recovered is None:
                missing_points += 1
                y = plot_y + plot_size - 4
                body.extend(
                    [
                        f'<line x1="{x - 4:.1f}" y1="{y - 4:.1f}" x2="{x + 4:.1f}" y2="{y + 4:.1f}" stroke="#B42318" stroke-width="1.8"/>',
                        f'<line x1="{x - 4:.1f}" y1="{y + 4:.1f}" x2="{x + 4:.1f}" y2="{y - 4:.1f}" stroke="#B42318" stroke-width="1.8"/>',
                    ]
                )
                continue
            rendered_points += 1
            y = sy(recovered)
            if point["ci"]:
                ci_low, ci_high = point["ci"]
                body.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{sy(ci_low):.1f}" x2="{x:.1f}" y2="{sy(ci_high):.1f}" stroke="#1D4ED8" stroke-width="1.4"/>',
                        f'<line x1="{x - 4:.1f}" y1="{sy(ci_low):.1f}" x2="{x + 4:.1f}" y2="{sy(ci_low):.1f}" stroke="#1D4ED8"/>',
                        f'<line x1="{x - 4:.1f}" y1="{sy(ci_high):.1f}" x2="{x + 4:.1f}" y2="{sy(ci_high):.1f}" stroke="#1D4ED8"/>',
                    ]
                )
            body.append(
                f'<path d="M {x:.1f} {y - 5:.1f} l 5 5 l -5 5 l -5 -5 z" fill="#2563EB" stroke="#FFFFFF" stroke-width="0.8"/>'
            )
        unit_suffix = f" ({unit})" if unit else ""
        body.extend(
            [
                f'<text x="{plot_x + plot_size / 2:.1f}" y="{plot_y + plot_size + 35}" text-anchor="middle" class="axis">requested { _escape(parameter) }{_escape(unit_suffix)}</text>',
                f'<text x="{plot_x - 38}" y="{plot_y + plot_size / 2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 {plot_x - 38} {plot_y + plot_size / 2:.1f})">recovered { _escape(parameter) }{_escape(unit_suffix)}</text>',
                f'<text x="{x0 + 20}" y="{y0 + panel_height - 28}" class="small">levels: {len(points)} · recovered: {rendered_points} · missing: {missing_points}</text>',
                "</g>",
            ]
        )
        panel_metadata.append(
            {
                "scan_id": scan_id,
                "parameter": parameter,
                "grade": grade or None,
                "level_count": len(points),
                "recovered_level_count": rendered_points,
                "missing_level_count": missing_points,
            }
        )
    return _write_svg(
        path,
        width=width,
        height=height,
        title=title,
        description="Small multiples compare requested parameter values with trajectory-inferred values on equal axes.",
        body=body,
        metadata={"panel_count": len(panel_metadata), "panels": panel_metadata},
    )


_DEFAULT_SCOPES: tuple[dict[str, str], ...] = (
    {
        "scope_id": "baseline_side_quantitative",
        "title": "1. Baseline · Side quantitative",
        "question": "Does recovered physics respond to the requested parameter?",
        "comparison": "Same scene, Side view and seed; matched one-factor parameter scan.",
        "report_unit": "G2/G3/G4 plus requested↔recovered curve and trajectory residuals.",
    },
    {
        "scope_id": "background_robustness",
        "title": "2. Background robustness",
        "question": "Does the response survive indoor/outdoor scene complexity?",
        "comparison": "Matched parameter scan across baseline, indoor and outdoor scenes.",
        "report_unit": "Per-scene coverage/error and cross-scene aggregate grade.",
    },
    {
        "scope_id": "view_robustness",
        "title": "3. View robustness",
        "question": "Is the inferred response consistent across Side/Main/Top views?",
        "comparison": "Same experiment, parameter tuple, scene and seed; camera varies.",
        "report_unit": "Per-view grade and consistency; report 3D-route evidence separately.",
    },
    {
        "scope_id": "seed_stability",
        "title": "4. Seed stability",
        "question": "Is the physical response stable under stochastic sampling?",
        "comparison": "Same condition and view; only the generation seed varies.",
        "report_unit": "Dispersion, direction concordance and validity-failure rate.",
    },
)


def _is_scan_collection(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and any(
        any(key in row for key in ("experiment_id", "scene_id", "camera_name", "response_grade"))
        for row in rows
    ) and not any("scope_id" in row for row in rows)


def _grade_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(
        str(row.get("response_grade") or row.get("grade"))
        for row in rows
        if row.get("response_grade") or row.get("grade")
    )
    return " · ".join(f"{grade} {counts[grade]}" for grade in _GRADE_ORDER if counts.get(grade)) or "No graded scans"


def _derive_scope_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    def scene(row: Mapping[str, Any]) -> str:
        return str(row.get("scene_id") or "").lower()

    def camera(row: Mapping[str, Any]) -> str:
        return str(row.get("camera_name") or row.get("camera") or "").lower()

    baseline_side = [row for row in rows if scene(row) == "baseline" and "side" in camera(row)]
    backgrounds = [row for row in rows if scene(row) and scene(row) != "baseline"]
    views = [row for row in rows if camera(row) and "side" not in camera(row)]
    seed_rows = [row for row in rows if scene(row) == "baseline" and "side" in camera(row)]
    selected = {
        "baseline_side_quantitative": baseline_side,
        "background_robustness": backgrounds,
        "view_robustness": views,
        "seed_stability": seed_rows,
    }
    output: dict[str, dict[str, Any]] = {}
    for key, subset in selected.items():
        unique_seeds = {str(row.get("seed")) for row in subset if row.get("seed") is not None}
        output[key] = {
            "metric": f"{len(subset)} scan{'s' if len(subset) != 1 else ''}",
            "note": _grade_digest(subset),
        }
        if key == "seed_stability":
            output[key]["metric"] = f"{len(unique_seeds)} seed{'s' if len(unique_seeds) != 1 else ''} · {len(subset)} scans"
    return output


def _scope_overrides(source: Any) -> dict[str, dict[str, Any]]:
    if source is None:
        return {}
    rows = _records(source)
    if _is_scan_collection(rows):
        return _derive_scope_summaries(rows)
    output: dict[str, dict[str, Any]] = {}
    if isinstance(source, Mapping) and any(key in source for key in _SCOPE_COLOURS):
        for key in _SCOPE_COLOURS:
            value = source.get(key)
            if isinstance(value, Mapping):
                output[key] = dict(value)
        return output
    for row in rows:
        scope_id = str(row.get("scope_id") or row.get("id") or "")
        if scope_id:
            output[scope_id] = dict(row)
    return output


def _wrapped_svg_text(
    text: Any,
    *,
    x: float,
    y: float,
    width_chars: int,
    css_class: str,
    line_height: int,
    max_lines: int,
) -> tuple[list[str], float]:
    lines = textwrap.wrap(str(text or ""), width=max(10, width_chars), break_long_words=False) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .") + "…"
    values = [f'<text x="{x}" y="{y}" class="{css_class}">']
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        values.append(f'<tspan x="{x}" dy="{dy}">{_escape(line)}</tspan>')
    values.append("</text>")
    return values, y + max(0, len(lines) - 1) * line_height


def write_experiment_scope_summary_svg(
    path: Path | str,
    summaries: Any = None,
    *,
    title: str = "Four complementary evaluation scopes",
    subtitle: str = "Quantitative response is established first; background, view and seed tests then measure robustness.",
) -> dict[str, Any]:
    """Write a four-panel benchmark-scope schematic and compact summary.

    ``summaries`` may be keyed by the four scope IDs, contain rows with a
    ``scope_id``, or be a sequence of parameter-scan rows.  For scan rows, the
    function derives transparent counts and a grade digest; it does not infer a
    robustness conclusion from counts alone.
    """

    overrides = _scope_overrides(summaries)
    scopes = []
    for default in _DEFAULT_SCOPES:
        scope = dict(default)
        scope.update(overrides.get(scope["scope_id"], {}))
        scopes.append(scope)
    width, height = 1200, 746
    card_width, card_height = 536, 262
    positions = ((50, 112), (614, 112), (50, 414), (614, 414))
    body = [
        f'<text x="50" y="42" class="title">{_escape(title)}</text>',
        f'<text x="50" y="68" class="subtitle">{_escape(subtitle)}</text>',
        '<path d="M 586 236 L 608 236" stroke="#9CA3AF" stroke-width="2" marker-end="url(#scope-arrow)"/>',
        '<path d="M 318 378 L 318 406" stroke="#9CA3AF" stroke-width="2"/>',
        '<path d="M 882 378 L 882 406" stroke="#9CA3AF" stroke-width="2"/>',
    ]
    # Insert a reusable arrow marker without requiring JavaScript or a library.
    body.insert(
        0,
        '<defs><marker id="scope-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#9CA3AF"/></marker></defs>',
    )
    summary_meta: list[dict[str, Any]] = []
    for scope, (x, y) in zip(scopes, positions):
        scope_id = str(scope["scope_id"])
        colour = _SCOPE_COLOURS[scope_id]
        metric = str(scope.get("metric") or scope.get("value") or "Evidence summary pending")
        note = str(scope.get("note") or scope.get("status") or "")
        body.extend(
            [
                f'<g data-scope-id="{_escape(scope_id)}">',
                f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" rx="7" class="frame"/>',
                f'<rect x="{x}" y="{y}" width="8" height="{card_height}" rx="4" fill="{colour}"/>',
                f'<text x="{x + 26}" y="{y + 32}" class="panel-title">{_escape(scope.get("title"))}</text>',
                f'<rect x="{x + card_width - 198}" y="{y + 13}" width="176" height="28" rx="14" fill="{colour}" opacity="0.12"/>',
                f'<text x="{x + card_width - 110}" y="{y + 32}" text-anchor="middle" class="label">{_escape(metric)}</text>',
                f'<text x="{x + 26}" y="{y + 66}" class="small">SCIENTIFIC QUESTION</text>',
            ]
        )
        lines, cursor = _wrapped_svg_text(scope.get("question"), x=x + 26, y=y + 84, width_chars=70, css_class="body", line_height=16, max_lines=2)
        body.extend(lines)
        body.append(f'<text x="{x + 26}" y="{cursor + 27}" class="small">MATCHED COMPARISON</text>')
        lines, cursor = _wrapped_svg_text(scope.get("comparison"), x=x + 26, y=cursor + 45, width_chars=70, css_class="body", line_height=16, max_lines=2)
        body.extend(lines)
        body.append(f'<text x="{x + 26}" y="{cursor + 27}" class="small">REPORT</text>')
        lines, cursor = _wrapped_svg_text(scope.get("report_unit"), x=x + 26, y=cursor + 45, width_chars=70, css_class="body", line_height=16, max_lines=2)
        body.extend(lines)
        if note:
            lines, _ = _wrapped_svg_text(note, x=x + 26, y=y + card_height - 18, width_chars=76, css_class="small", line_height=14, max_lines=1)
            body.extend(lines)
        body.append("</g>")
        summary_meta.append({"scope_id": scope_id, "metric": metric, "note": note or None})
    body.extend(
        [
            '<text x="50" y="722" class="small">Interpretation rule: robustness panels complement—not replace—the baseline Side quantitative result.</text>',
            '<text x="1150" y="722" text-anchor="end" class="small">Keep video-level validity and scan-level response counts separate.</text>',
        ]
    )
    return _write_svg(
        path,
        width=width,
        height=height,
        title=title,
        description="Four panels define baseline quantitative, background, view, and seed evaluation scopes and their matched comparisons.",
        body=body,
        metadata={"scope_count": 4, "scopes": summary_meta},
    )


# Readable aliases for callers that name the chart rather than the file format.
write_grade_count_bar_chart = write_grade_counts_svg
write_parameter_scan_small_multiples = write_parameter_scan_small_multiples_svg
write_experiment_scope_summary = write_experiment_scope_summary_svg


__all__ = [
    "write_experiment_scope_summary",
    "write_experiment_scope_summary_svg",
    "write_grade_count_bar_chart",
    "write_grade_counts_svg",
    "write_parameter_scan_small_multiples",
    "write_parameter_scan_small_multiples_svg",
]
