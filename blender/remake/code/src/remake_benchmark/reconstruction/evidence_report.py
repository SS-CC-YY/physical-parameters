"""Build a human-readable, video-first PhysParamBench evidence package.

This module is deliberately a *renderer*, not another evaluator.  It consumes
the frozen per-video evaluation results and the compact paper-facing grades,
then makes their provenance inspectable through videos, trajectory figures and
short formula cards.  Missing tracking/reconstruction evidence stays missing;
the renderer never changes a grade or fabricates a fitted value.

The report keeps the benchmark scopes separate:

* baseline + CAM_Side + primary seed: quantitative parameter identification;
* indoor/outdoor: retention relative to the same baseline setting;
* CAM_Main/CAM_Top: view robustness relative to CAM_Side; and
* dynamically reconstructed videos: an explicit camera-drift appendix.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from html import escape
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

from .evidence_formulas import get_formula_spec
from .evidence_media import (
    materialize_media,
    render_fit_process_card,
    render_trajectory_diagnostics,
    write_placeholder_image,
    write_video_montage,
)


SCHEMA_VERSION = "1.0.0"
PRIMARY_SEED = 341867882
GOOD_FIT_STATUS = {"ok", "pass", "passed", "success", "succeeded", "complete", "completed"}
PASS_STATUS = {"pass", "passed", "ok", "success", "succeeded", "valid", "true", "1"}
FAIL_STATUS = {"fail", "failed", "invalid", "g0", "g1", "false", "0"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed", "ok"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed"}:
        return False
    return None


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _job_id(value: Any) -> str:
    text = str(value or "").strip()
    return Path(text).stem


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or ""))
    return text.strip("._") or "unnamed"


def _rel(path: Path | None, page: Path) -> str | None:
    if path is None:
        return None
    try:
        return Path(os.path.relpath(path, page.parent)).as_posix()
    except ValueError:
        return path.as_posix()


def _first_file(candidates: Iterable[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _codes(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]


def _parameter_specs(registry: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for experiment in registry.get("experiments", []):
        if isinstance(experiment, Mapping):
            output[str(experiment.get("id"))] = [
                dict(parameter)
                for parameter in experiment.get("hidden_parameters", [])
                if isinstance(parameter, Mapping)
            ]
    return output


def _experiment_registry(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id")): dict(item)
        for item in registry.get("experiments", [])
        if isinstance(item, Mapping) and item.get("id")
    }


def _estimate(row: Mapping[str, Any], parameter: str) -> float | None:
    return _number(row.get(f"{parameter}__estimate"))


def _target(row: Mapping[str, Any], parameter: str) -> float | None:
    return _number(row.get(f"{parameter}__gt"))


def _normalised_difference(a: float | None, b: float | None, valid_range: Sequence[Any]) -> float | None:
    if a is None or b is None or len(valid_range) != 2:
        return None
    lo, hi = _number(valid_range[0]), _number(valid_range[1])
    if lo is None or hi is None or hi <= lo:
        return None
    return abs(a - b) / (hi - lo)


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def _format_list(values: Sequence[str]) -> str:
    return "；".join(values) if values else "未记录原因码"


def _outcome_available(result: Mapping[str, Any]) -> bool:
    """Treat partial visual evidence as inspectable, never as a fake success."""

    if result.get("available") is True:
        return True
    return str(result.get("status") or "").lower() in {"ok", "partial", "rendered", "linked", "copied"}


def _formula_text(spec: Mapping[str, Any]) -> str:
    model = spec.get("model_plain")
    estimate = spec.get("estimate_plain")
    if model and estimate:
        return f"{model}; {estimate}"
    for key in ("estimate_plain", "model_plain", "recovery_formula", "formula", "display_formula", "equation", "model_equation"):
        value = spec.get(key)
        if value:
            if isinstance(value, Sequence) and not isinstance(value, str):
                return "；".join(str(item) for item in value)
            return str(value)
    equations = spec.get("equations")
    if isinstance(equations, Sequence) and not isinstance(equations, str):
        return "；".join(str(item) for item in equations)
    return "见单视频拟合卡中的实际 fitter method；本报告不补造缺失系数。"


def _formula_steps(spec: Mapping[str, Any]) -> list[str]:
    value = spec.get("estimate_steps_zh") or spec.get("steps") or spec.get("recovery_steps") or spec.get("process")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return [
        "从视频检测/重建得到逐帧位置与时间戳。",
        "在目标参数未参与拟合的前提下，用对应运动方程拟合轨迹或事件前后速度。",
        "从拟合系数恢复有效物理参数，并与 prompt 请求值做 valid-range 归一化误差比较。",
    ]


def _card_formula_spec(spec: Mapping[str, Any], experiment_id: str, parameter: str) -> dict[str, Any]:
    return {
        "title": f"{spec.get('experiment_title_zh', experiment_id)} · {spec.get('parameter_title_zh', parameter)}",
        "equations": [value for value in (spec.get("model_plain"), spec.get("estimate_plain")) if value],
        "observables": spec.get("observables", []),
        "parameter_transform": spec.get("estimate_plain"),
        "notes": [spec.get("fitter_summary_zh"), *list(spec.get("estimate_steps_zh", []))],
    }


def _html_page(title: str, body: str, *, root_prefix: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>
:root{{--ink:#18212b;--muted:#667085;--line:#d9e1ea;--paper:#fff;--bg:#f5f7fa;--accent:#165dff;--bad:#b42318;--ok:#067647;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1320px;margin:auto;padding:28px}} h1,h2,h3{{line-height:1.25}} h1{{font-size:29px}} h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 2px 8px #102a4310}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}} .media{{width:100%;max-height:520px;background:#111;border-radius:8px}}
img.media{{object-fit:contain}} video.media{{object-fit:contain}} table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}} th{{background:#eef3f8;position:sticky;top:0}}
.muted{{color:var(--muted)}} .bad{{color:var(--bad);font-weight:700}} .ok{{color:var(--ok);font-weight:700}} .tag{{display:inline-block;padding:2px 8px;margin:2px;border-radius:999px;background:#e9f0ff;color:#174ea6}}
code{{background:#eef1f5;padding:1px 4px;border-radius:4px}} a{{color:var(--accent)}} nav a{{margin-right:14px}} details{{margin:8px 0}}
</style></head><body><main><nav><a href="{root_prefix}index.html">报告首页</a><a href="{root_prefix}01_unusable/index.html">不可用案例</a><a href="{root_prefix}02_parameter_scans/index.html">参数扫描</a><a href="{root_prefix}03_background/index.html">复杂背景</a><a href="{root_prefix}04_views/index.html">视角</a><a href="{root_prefix}05_camera_drift/index.html">相机漂移</a></nav>{body}</main></body></html>"""


def _media_block(label: str, path: Path | None, page: Path, kind: str) -> str:
    rel = _rel(path, page)
    if rel is None or path is None or not path.is_file():
        return f'<div class="card"><h3>{escape(label)}</h3><p class="bad">文件不可用；报告没有用占位数值替代。</p></div>'
    if kind == "video":
        element = f'<video class="media" controls preload="metadata" src="{escape(rel)}"></video>'
    else:
        element = f'<a href="{escape(rel)}"><img class="media" loading="lazy" src="{escape(rel)}" alt="{escape(label)}"></a>'
    return f'<div class="card"><h3>{escape(label)}</h3>{element}<p><a href="{escape(rel)}">打开原文件</a></p></div>'


def _document_block(label: str, path: Path | None, page: Path) -> str:
    rel = _rel(path, page)
    if rel is None or path is None or not path.is_file():
        return f'<div class="card"><h3>{escape(label)}</h3><p class="bad">拟合过程文件不可用。</p></div>'
    return (
        f'<div class="card"><h3>{escape(label)}</h3>'
        f'<iframe src="{escape(rel)}" title="{escape(label)}" style="width:100%;height:580px;border:1px solid #d9e1ea;border-radius:8px"></iframe>'
        f'<p><a href="{escape(rel)}">在新页面打开完整拟合过程</a></p></div>'
    )


@dataclass
class LocatedArtifacts:
    job_id: str
    original: Path | None
    overlay: Path | None
    trajectory_csv: Path | None
    native_3d_plot: Path | None
    result_json: Path | None


class ArtifactLocator:
    def __init__(self, videos_root: Path, evaluation_root: Path, tracks_root: Path | None) -> None:
        self.videos_root = videos_root
        self.evaluation_root = evaluation_root
        self.tracks_root = tracks_root

    def locate(self, job_id: str) -> LocatedArtifacts:
        eval_job = self.evaluation_root / "jobs" / job_id
        track_job = None if self.tracks_root is None else self.tracks_root / "jobs" / job_id
        dynamic_job = None if self.tracks_root is None else self.tracks_root / "dynamic_native" / job_id
        result_path = eval_job / "result.json"
        result = _read_json(result_path)
        reported_video = result.get("job", {}).get("video_path") if isinstance(result.get("job"), Mapping) else None
        reported_overlay = result.get("visual_evidence", {}).get("overlay_video") if isinstance(result.get("visual_evidence"), Mapping) else None
        original = _first_file(
            [
                self.videos_root / f"{job_id}.mp4",
                Path(str(reported_video)) if reported_video else None,
            ]
        )
        overlay_candidates: list[Path | None] = []
        if track_job is not None:
            overlay_candidates += [track_job / "object_track_overlay.mp4", track_job / "validity_object_track_overlay.mp4"]
        if dynamic_job is not None:
            overlay_candidates.append(dynamic_job / "object_track_overlay.mp4")
        overlay_candidates += [eval_job / "object_track_overlay.mp4", eval_job / "validity_object_track_overlay.mp4"]
        if reported_overlay:
            overlay_candidates += [Path(str(reported_overlay)), eval_job / Path(str(reported_overlay)).name]
        trajectory_candidates: list[Path | None] = [
            eval_job / "trajectory_frames.csv",
            eval_job / "trajectory.csv",
        ]
        if track_job is not None:
            trajectory_candidates += [track_job / "trajectory_frames.csv", track_job / "trajectory.csv"]
        if dynamic_job is not None:
            trajectory_candidates += [dynamic_job / "trajectory_world.csv", dynamic_job / "trajectory_frames.csv"]
        native_candidates: list[Path | None] = [eval_job / "trajectory_3d.png"]
        if dynamic_job is not None:
            native_candidates += [dynamic_job / "trajectory_3d.png", dynamic_job / "trajectory_world.png"]
        return LocatedArtifacts(
            job_id=job_id,
            original=original,
            overlay=_first_file(overlay_candidates),
            trajectory_csv=_first_file(trajectory_candidates),
            native_3d_plot=_first_file(native_candidates),
            result_json=result_path if result_path.is_file() else None,
        )


class EvidenceReportBuilder:
    def __init__(
        self,
        *,
        all_jobs: Sequence[Mapping[str, Any]],
        scans: Sequence[Mapping[str, Any]],
        video_gate: Sequence[Mapping[str, Any]],
        registry: Mapping[str, Any],
        simple_report_root: Path,
        evaluation_root: Path,
        tracks_root: Path | None,
        videos_root: Path,
        output: Path,
        media_mode: str,
        build_montages: bool,
        primary_seed: int,
    ) -> None:
        self.rows = [dict(row) for row in all_jobs]
        self.scans = [dict(row) for row in scans]
        self.video_gate = {_job_id(row.get("row_id")): dict(row) for row in video_gate}
        self.registry = dict(registry)
        self.experiments = _experiment_registry(registry)
        self.parameters = _parameter_specs(registry)
        self.simple_report_root = simple_report_root
        self.evaluation_root = evaluation_root
        self.tracks_root = tracks_root
        self.videos_root = videos_root
        self.output = output
        self.media_mode = media_mode
        self.build_montages = build_montages
        self.primary_seed = primary_seed
        self.locator = ArtifactLocator(videos_root, evaluation_root, tracks_root)
        self.row_by_job = {_job_id(row.get("video_name") or row.get("job_id")): row for row in self.rows}
        self.dynamic_gate_by_job = {
            _job_id(row.get("job_id")): row
            for row in _read_csv(simple_report_root / "dynamic_3d_gate.csv")
        }
        self.asset_cache: dict[str, dict[str, Path | None]] = {}
        self.manifest: list[dict[str, Any]] = []

    def _materialize_job(self, job_id: str) -> dict[str, Path | None]:
        if job_id in self.asset_cache:
            return self.asset_cache[job_id]
        located = self.locator.locate(job_id)
        root = self.output / "_assets" / "jobs" / job_id
        root.mkdir(parents=True, exist_ok=True)
        original_dest = root / "original.mp4"
        overlay_dest = root / "track_overlay.mp4"
        original = None
        overlay = None
        if located.original is not None:
            outcome = materialize_media(located.original, original_dest, mode=self.media_mode)
            if _outcome_available(outcome):
                original = original_dest
        if located.overlay is not None:
            outcome = materialize_media(located.overlay, overlay_dest, mode=self.media_mode)
            if _outcome_available(outcome):
                overlay = overlay_dest
        trajectory_plot = root / "trajectory_2d_3d.png"
        if located.trajectory_csv is not None:
            result = render_trajectory_diagnostics(
                located.trajectory_csv,
                trajectory_plot,
                title=job_id,
                route=str(self.row_by_job.get(job_id, {}).get("reconstruction_route") or ""),
            )
            if not _outcome_available(result) and not trajectory_plot.is_file():
                write_placeholder_image(trajectory_plot, "轨迹图不可生成", str(result.get("reason_code") or result.get("error") or "unknown"))
        else:
            write_placeholder_image(trajectory_plot, "没有可用逐帧轨迹", "原视频仍保留；此处不补造 2D/3D 坐标。")
        native_3d = None
        if located.native_3d_plot is not None:
            native_dest = root / "trajectory_3d_native.png"
            outcome = materialize_media(located.native_3d_plot, native_dest, mode=self.media_mode)
            if _outcome_available(outcome):
                native_3d = native_dest
        assets = {
            "original": original,
            "overlay": overlay,
            "trajectory": trajectory_plot,
            "native_3d": native_3d,
            "result": located.result_json,
            "trajectory_csv": located.trajectory_csv,
        }
        self.asset_cache[job_id] = assets
        self.manifest.append(
            {
                "job_id": job_id,
                "source_original": None if located.original is None else str(located.original),
                "source_overlay": None if located.overlay is None else str(located.overlay),
                "source_trajectory": None if located.trajectory_csv is None else str(located.trajectory_csv),
                "source_result": None if located.result_json is None else str(located.result_json),
                "report_original": None if original is None else str(original),
                "report_overlay": None if overlay is None else str(overlay),
                "report_trajectory": str(trajectory_plot),
                "measurement_route": self.row_by_job.get(job_id, {}).get("reconstruction_route"),
            }
        )
        return assets

    def _unusable_decision(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        job_id = _job_id(row.get("video_name") or row.get("job_id"))
        gate = self.video_gate.get(job_id, {})
        gate_grade = str(gate.get("grade") or "")
        reasons = _codes(gate.get("reason_codes"))
        manual_generation = _status(row.get("manual_generation_validity_status"))
        generation = _status(row.get("generation_validity_status"))
        fit_eligible = _truth(row.get("trajectory_fit_eligible"))
        fit_complete = _truth(row.get("fit_complete"))
        fit_status = _status(row.get("fit_status"))
        error = str(row.get("error") or "").strip()
        category: str | None = None
        headline = ""
        if gate_grade == "L1" or manual_generation in FAIL_STATUS:
            category = "L1"
            headline = "人工确认的生成/运动有效性失败"
            reasons += ["manual_generation_or_motion_failure"]
        elif gate_grade == "X":
            category = "X"
            headline = "主评测的轨迹/反演证据不足（不计作模型失败）"
        elif manual_generation not in PASS_STATUS and generation in FAIL_STATUS:
            category = "REVIEW"
            headline = "自动有效性检查标记异常，需看视频复核"
            reasons += _codes(row.get("generation_failure_codes")) or ["automatic_generation_validity_failure"]
        elif fit_eligible is False:
            category = "X"
            headline = "轨迹不满足当前拟合输入要求（不计作模型失败）"
            reasons += _codes(row.get("trajectory_fit_eligibility")) or ["trajectory_fit_ineligible"]
        elif fit_complete is False or (fit_status and fit_status not in GOOD_FIT_STATUS):
            category = "X"
            headline = "运动方程拟合未完成（不计作模型失败）"
            reasons += [f"fit_status:{fit_status or 'incomplete'}"]
        elif error:
            category = "X"
            headline = "评测流程记录错误（不计作模型失败）"
            reasons += [error]
        if category is None:
            return None
        if not reasons:
            reasons = ["see_visual_evidence_and_result"]
        return {"category": category, "headline": headline, "reason_codes": list(dict.fromkeys(reasons))}

    def build_unusable(self) -> dict[str, Any]:
        root = self.output / "01_unusable"
        root.mkdir(parents=True, exist_ok=True)
        cases: list[dict[str, Any]] = []
        for row in self.rows:
            decision = self._unusable_decision(row)
            if decision is None:
                continue
            job_id = _job_id(row.get("video_name") or row.get("job_id"))
            if not job_id:
                continue
            assets = self._materialize_job(job_id)
            category = decision["category"]
            case_root = root / category / job_id
            case_root.mkdir(parents=True, exist_ok=True)
            result = _read_json(assets["result"]) if assets.get("result") else {}
            dynamic = result.get("dynamic_3d_inclusion", {}) if isinstance(result.get("dynamic_3d_inclusion"), Mapping) else {}
            camera = result.get("camera_motion_evidence", {}) if isinstance(result.get("camera_motion_evidence"), Mapping) else {}
            explanation = [
                f"# {job_id}",
                "",
                f"**结论：{decision['headline']}**",
                "",
                "这里把“模型生成失败”和“测量不可用”分开。X/REVIEW 不会被写成视频模型物理失败。",
                "",
                "## 为什么当前不可进入参数主统计",
                "",
            ]
            explanation += [f"- `{code}`" for code in decision["reason_codes"]]
            explanation += [
                "",
                "## 可核验证据",
                "",
                f"- 原始视频：{'已提供' if assets.get('original') else '缺失'}",
                f"- 轨迹检测视频：{'已提供' if assets.get('overlay') else '缺失'}",
                f"- 2D/3D 轨迹图：{'已提供' if assets.get('trajectory') else '缺失'}",
                f"- 重建路线：`{row.get('reconstruction_route') or 'unknown'}`",
                f"- 跟踪覆盖率：`{_fmt(row.get('tracked_fraction'))}`",
                f"- 拟合状态：`{row.get('fit_status') or 'unknown'}`",
                f"- 相机判断：`{camera.get('decision') or camera.get('effective_camera_motion_category') or 'not recorded'}`",
                f"- 3D 门控：`{dynamic.get('decision') or row.get('simple_dynamic_3d_decision') or 'not applicable'}`",
                "",
                "轨迹图中出现断点、离群或错误深度时，原因属于测量链路；只有原视频本身出现消失、明显形变、穿模或运动类型错误并经人工确认时，才归入 L1。",
            ]
            (case_root / "WHY_UNUSABLE_ZH.md").write_text("\n".join(explanation) + "\n", encoding="utf-8")
            page = case_root / "index.html"
            body = [
                f"<h1>{escape(job_id)}</h1><div class='card'><p class='bad'>{escape(decision['headline'])}</p>",
                "<p>原因：</p><ul>" + "".join(f"<li><code>{escape(code)}</code></li>" for code in decision["reason_codes"]) + "</ul>",
                f"<p>路线：<code>{escape(str(row.get('reconstruction_route') or 'unknown'))}</code>；跟踪覆盖率 {_fmt(row.get('tracked_fraction'))}；拟合状态 <code>{escape(str(row.get('fit_status') or 'unknown'))}</code></p></div>",
                "<div class='grid'>",
                _media_block("1. 原始生成视频", assets.get("original"), page, "video"),
                _media_block("2. 实验物体检测与轨迹 overlay", assets.get("overlay"), page, "video"),
                _media_block("3. 逐帧 2D / 公制 / 3D 轨迹", assets.get("trajectory"), page, "image"),
                _media_block("4. SpatialTrackerV2 原生 3D 图（若适用）", assets.get("native_3d"), page, "image"),
                "</div><p class='muted'>判断来自冻结 result/CSV；本页面只整理证据，不重跑检测，也不改判分。</p>",
            ]
            page.write_text(_html_page(job_id, "".join(body), root_prefix="../../../"), encoding="utf-8")
            cases.append({"job_id": job_id, "experiment_id": row.get("experiment_id"), **decision, "page": str(page)})
        counts = Counter(case["category"] for case in cases)
        page = root / "index.html"
        rows_html = "".join(
            f"<tr><td>{escape(case['category'])}</td><td>{escape(str(case['experiment_id']))}</td><td><a href='{escape(Path(os.path.relpath(case['page'], root)).as_posix())}'>{escape(case['job_id'])}</a></td><td>{escape(case['headline'])}</td><td>{escape(_format_list(case['reason_codes']))}</td></tr>"
            for case in cases
        )
        body = f"<h1>不可用与待复核视频证据</h1><div class='card'><p>L1={counts['L1']}，X={counts['X']}，REVIEW={counts['REVIEW']}。L1 只用于人工确认的模型失败；X 是轨迹/重建/拟合不可用；REVIEW 是自动门异常、仍需看原视频。</p></div><table><thead><tr><th>状态</th><th>实验</th><th>视频</th><th>可读结论</th><th>原因</th></tr></thead><tbody>{rows_html}</tbody></table>"
        page.write_text(_html_page("不可用证据", body, root_prefix="../"), encoding="utf-8")
        return {"case_count": len(cases), "counts": dict(counts), "cases": cases}

    def _scan_level_jobs(self, scan: Mapping[str, Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        experiment_id = str(scan.get("experiment_id"))
        seed = int(scan.get("seed") or self.primary_seed)
        for level in scan.get("evidence", {}).get("levels", []):
            if not isinstance(level, Mapping):
                continue
            target_value = _number(level.get("target_value"))
            level_rows = [row for row in level.get("rows", []) if isinstance(row, Mapping)]
            present_rows = [row for row in level_rows if row.get("row_id") and not str(row.get("row_id")).startswith("missing:")]
            median_estimate = _number(level.get("median_estimate"))
            usable_rows = [row for row in present_rows if str(row.get("state")) == "usable" and _number(row.get("estimate")) is not None]
            if usable_rows and median_estimate is not None:
                present = min(usable_rows, key=lambda row: abs(float(row["estimate"]) - median_estimate))
            else:
                present = usable_rows[0] if usable_rows else (present_rows[0] if present_rows else None)
            tuple_ids = list(level.get("parameter_tuple_ids") or [])
            tuple_id = str((present or {}).get("parameter_tuple_id") or (tuple_ids[0] if tuple_ids else "unknown"))
            if present is not None:
                job_id = _job_id(present.get("row_id"))
                estimate = median_estimate if median_estimate is not None else _number(present.get("estimate"))
                state = str(present.get("state") or "unknown")
                reasons = _codes(present.get("reason_codes"))
            else:
                job_id = f"{experiment_id}__{tuple_id}__baseline__standard_ball__CAM_Side__seed-{seed}"
                estimate = _number(level.get("median_estimate"))
                state = "measurement_or_tracking_insufficient"
                reasons = ["primary_slice_row_missing"]
            output.append(
                {
                    "job_id": job_id,
                    "parameter_tuple_id": tuple_id,
                    "target": target_value,
                    "estimate": estimate,
                    "state": state,
                    "reason_codes": reasons,
                }
            )
        return output

    def _write_response_svg(self, path: Path, scan: Mapping[str, Any], levels: Sequence[Mapping[str, Any]]) -> None:
        valid_range = scan.get("valid_range") or [0.0, 1.0]
        values = [v for item in levels for v in (_number(item.get("target")), _number(item.get("estimate"))) if v is not None]
        lo = min(values + [_number(valid_range[0]) or 0.0])
        hi = max(values + [_number(valid_range[1]) or 1.0])
        if hi <= lo:
            hi = lo + 1.0
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        width, height = 820, 520
        left, right, top, bottom = 90, 40, 55, 90
        pw, ph = width - left - right, height - top - bottom
        sx = lambda value: left + (float(value) - lo) / (hi - lo) * pw
        sy = lambda value: top + (hi - float(value)) / (hi - lo) * ph
        points = [(item, sx(item["target"]), sy(item["estimate"])) for item in levels if _number(item.get("target")) is not None and _number(item.get("estimate")) is not None]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for _, x, y in sorted(points, key=lambda value: value[0]["target"]))
        circles = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#165dff"/><text x="{x+9:.1f}" y="{y-8:.1f}" font-size="12">{escape(str(item["parameter_tuple_id"]))}: {_fmt(item["target"])}→{_fmt(item["estimate"])}</text>'
            for item, x, y in points
        )
        missing = [item for item in levels if _number(item.get("estimate")) is None]
        missing_text = "；".join(f"{item['parameter_tuple_id']} target={_fmt(item['target'])}" for item in missing) or "none"
        title = f"{scan.get('experiment_id')} · {scan.get('target_parameter')} · {scan.get('grade')}"
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><text x="{left}" y="28" font-family="sans-serif" font-size="21" font-weight="700">{escape(title)}</text><line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top}" stroke="#98a2b3" stroke-dasharray="6 5"/><line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#344054"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#344054"/>{f'<polyline points="{polyline}" fill="none" stroke="#165dff" stroke-width="3"/>' if polyline else ''}{circles}<text x="{left+pw/2}" y="{height-48}" text-anchor="middle" font-family="sans-serif">prompt requested parameter</text><text x="22" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 22 {top+ph/2})" font-family="sans-serif">trajectory-inferred parameter</text><text x="{left}" y="{height-18}" font-family="sans-serif" font-size="12" fill="#667085">Missing estimate: {escape(missing_text)} · diagonal = exact numerical recovery</text></svg>'''
        path.write_text(svg, encoding="utf-8")

    def build_parameter_scans(self) -> dict[str, Any]:
        root = self.output / "02_parameter_scans"
        root.mkdir(parents=True, exist_ok=True)
        rendered: list[dict[str, Any]] = []
        for scan in self.scans:
            grade = str(scan.get("grade") or "X")
            if grade not in {"L2", "L3", "L4", "X"}:
                continue
            experiment_id = str(scan.get("experiment_id"))
            parameter = str(scan.get("target_parameter"))
            scan_root = root / _safe_name(experiment_id) / _safe_name(parameter)
            scan_root.mkdir(parents=True, exist_ok=True)
            levels = self._scan_level_jobs(scan)
            raw_sources: list[Path] = []
            overlay_sources: list[Path] = []
            raw_labels: list[str] = []
            overlay_labels: list[str] = []
            level_cards: list[tuple[dict[str, Any], dict[str, Path | None], Path | None]] = []
            spec = get_formula_spec(experiment_id, parameter)
            for level in levels:
                assets = self._materialize_job(str(level["job_id"]))
                label = f"target={_fmt(level['target'])} / fit={_fmt(level['estimate'])}"
                if assets.get("original"):
                    raw_sources.append(assets["original"])
                    raw_labels.append(label)
                if assets.get("overlay"):
                    overlay_sources.append(assets["overlay"])
                    overlay_labels.append(label)
                fit_card = scan_root / "levels" / _safe_name(level["parameter_tuple_id"]) / "fit_process.html"
                fit_card.parent.mkdir(parents=True, exist_ok=True)
                result_path = assets.get("result")
                if result_path:
                    card_spec = _card_formula_spec(spec, experiment_id, parameter)
                    outcome = render_fit_process_card(result_path, fit_card, card_spec, grade=grade)
                    if not _outcome_available(outcome) and not fit_card.is_file():
                        fit_card.write_text("<!doctype html><meta charset='utf-8'><h1>拟合过程不可用</h1><p>冻结 result.json 无法读取；没有补造曲线。</p>", encoding="utf-8")
                else:
                    fit_card.write_text("<!doctype html><meta charset='utf-8'><h1>没有冻结拟合结果</h1><p>该参数档位保留在扫描设计中，但没有 result.json；没有补造系数或曲线。</p>", encoding="utf-8")
                level_cards.append((level, assets, fit_card))
            raw_montage = scan_root / "raw_parameter_comparison.mp4"
            overlay_montage = scan_root / "track_parameter_comparison.mp4"
            raw_status = {"available": False, "reason": "montages disabled"}
            overlay_status = {"available": False, "reason": "montages disabled"}
            if self.build_montages:
                raw_status = write_video_montage(raw_sources, raw_labels, raw_montage)
                overlay_status = write_video_montage(overlay_sources, overlay_labels, overlay_montage)
            response = scan_root / "requested_vs_recovered.svg"
            self._write_response_svg(response, scan, levels)
            evidence = scan.get("evidence", {}) if isinstance(scan.get("evidence"), Mapping) else {}
            formula_lines = [
                f"# {experiment_id} / {parameter} / {grade}",
                "",
                "## 为什么 13 个实验会有 24 个结果",
                "",
                "这里的一行不是一个实验，而是一个 **实验 × 被扫描参数**。多参数实验会分别固定其余参数，对每个参数做一次 OAT（一次一变量）扫描。",
                "",
                "## 轨迹反推参数（简版）",
                "",
                f"- 轨迹模型：`{_formula_text(spec)}`",
                f"- 实际 fitter：`{spec.get('method') or spec.get('fitter_method') or '以各 result.json 的 fit.method 为准'}`",
                "- 请求参数不参与检测和拟合：`target_not_used_for_fit=true`。请求值只在拟合完成后用于算误差与等级。",
                "",
            ]
            formula_lines += [f"{index}. {step}" for index, step in enumerate(_formula_steps(spec), 1)]
            formula_lines += [
                "",
                "## 每个参数档位",
                "",
                "|设定|目标值|轨迹反演值|状态|说明|",
                "|---|---:|---:|---|---|",
            ]
            for level in levels:
                formula_lines.append(f"|{level['parameter_tuple_id']}|{_fmt(level['target'])}|{_fmt(level['estimate'])}|{level['state']}|{_format_list(level['reason_codes'])}|")
            formula_lines += [
                "",
                "## 扫描级判定",
                "",
                f"- 成对方向一致率：`C={_fmt(evidence.get('pairwise_direction_concordance'))}`，其中 C = #((Δtarget)(Δfit)>0) / #pairs。",
                f"- Theil–Sen 响应斜率：`{_fmt(evidence.get('theil_sen_slope'))}`。",
                f"- 中位 valid-range 归一化误差：`{_fmt(evidence.get('median_valid_range_nae'))}`。",
                f"- 最终等级：**{grade}**；原因：`{_format_list(_codes(scan.get('reason_codes')))}`。",
                "",
                "L2 表示运动可测但参数响应方向不足；L3 表示方向响应成立但数值误差较大；L4 表示方向与当前 benchmark 容差内的数值均通过；X 表示可用档位不足或测量不可用，不计作模型参数理解失败。",
            ]
            fit_md = scan_root / "FIT_PROCESS_ZH.md"
            fit_md.write_text("\n".join(formula_lines) + "\n", encoding="utf-8")
            page = scan_root / "index.html"
            level_html = "".join(
                "<div class='card'><h3>" + escape(str(level["parameter_tuple_id"])) + f" · target {_fmt(level['target'])} → fit {_fmt(level['estimate'])}</h3><div class='grid'>" +
                _media_block("原视频", assets.get("original"), page, "video") +
                _media_block("检测轨迹视频", assets.get("overlay"), page, "video") +
                _media_block("2D/3D 轨迹", assets.get("trajectory"), page, "image") +
                _document_block("公式拟合过程", card, page) + "</div></div>"
                for level, assets, card in level_cards
            )
            raw_path = raw_montage if raw_montage.is_file() else None
            overlay_path = overlay_montage if overlay_montage.is_file() else None
            body = f"<h1>{escape(experiment_id)} · {escape(parameter)} · {escape(grade)}</h1><div class='card'><h2>简洁反推</h2><p><code>{escape(_formula_text(spec))}</code></p><ol>{''.join(f'<li>{escape(step)}</li>' for step in _formula_steps(spec))}</ol><p>方向一致率 {_fmt(evidence.get('pairwise_direction_concordance'))}；Theil–Sen 斜率 {_fmt(evidence.get('theil_sen_slope'))}；中位归一化误差 {_fmt(evidence.get('median_valid_range_nae'))}。</p><p><a href='FIT_PROCESS_ZH.md'>打开文字版计算过程</a></p></div><div class='grid'>{_media_block('不同参数设定：原始视频对比', raw_path, page, 'video')}{_media_block('不同参数设定：检测轨迹对比', overlay_path, page, 'video')}{_media_block('请求值 vs 轨迹反演值', response, page, 'image')}</div>{level_html}"
            page.write_text(_html_page(f"{experiment_id} {parameter}", body, root_prefix="../../../"), encoding="utf-8")
            rendered.append(
                {
                    "experiment_id": experiment_id,
                    "parameter": parameter,
                    "grade": grade,
                    "page": str(page),
                    "level_count": len(levels),
                    "raw_montage": raw_status,
                    "overlay_montage": overlay_status,
                }
            )
        page = root / "index.html"
        counts = Counter(item["grade"] for item in rendered)
        l2_l4_count = sum(counts[grade] for grade in ("L2", "L3", "L4"))
        rows_html = "".join(
            f"<tr><td>{escape(item['experiment_id'])}</td><td>{escape(item['parameter'])}</td><td><b>{escape(item['grade'])}</b></td><td>{item['level_count']}</td><td><a href='{escape(Path(os.path.relpath(item['page'], root)).as_posix())}'>原视频 + 轨迹视频 + 公式拟合</a></td></tr>"
            for item in rendered
        )
        body = f"<h1>13 个实验系统，24 个参数响应轴</h1><div class='card'><p>13 是物理实验设计数量；24 是所有实验的隐藏参数维度总和。V1 为 4 轴，V2 为 8 轴，V3 为 12 轴。多参数实验会产生多个 OAT 扫描，但并没有被重复称作新实验。</p><p>当前 24 轴中，L2–L4 共 {l2_l4_count} 个：L2={counts['L2']}、L3={counts['L3']}、L4={counts['L4']}；扫描级 X={counts['X']}。X 也保留同样的原视频、overlay、轨迹与缺失原因页，但不计作模型失败。</p></div><table><thead><tr><th>实验</th><th>被扫描参数</th><th>等级</th><th>档位</th><th>证据</th></tr></thead><tbody>{rows_html}</tbody></table>"
        page.write_text(_html_page("参数扫描证据", body, root_prefix="../"), encoding="utf-8")
        return {
            "rendered_scan_count": len(rendered),
            "l2_l4_scan_count": l2_l4_count,
            "x_scan_count": counts["X"],
            "grade_counts": dict(counts),
            "scans": rendered,
        }

    def _row_status_label(self, row: Mapping[str, Any]) -> str:
        manual = _status(row.get("manual_generation_validity_status"))
        generation = _status(row.get("generation_validity_status"))
        if manual in FAIL_STATUS:
            return "人工确认生成有效性失败"
        if manual not in PASS_STATUS and generation in FAIL_STATUS:
            return "生成有效性异常/待复核"
        if not self._measurement_evaluable(row):
            if _truth(row.get("trajectory_fit_eligible")) is False:
                return "轨迹不可用于拟合"
            if str(row.get("reconstruction_route") or "") == "spatialtrackerv2_dynamic":
                return "动态 3D 证据门未通过"
            return "拟合未完成或测量不可用"
        return "生成可用且已有拟合"

    def _measurement_evaluable(self, row: Mapping[str, Any]) -> bool:
        """Whether a frozen estimate may enter background/view arithmetic."""

        manual = _status(row.get("manual_generation_validity_status"))
        automatic = _status(row.get("generation_validity_status"))
        if manual in FAIL_STATUS:
            return False
        if manual not in PASS_STATUS and automatic in FAIL_STATUS:
            return False
        if _truth(row.get("trajectory_fit_eligible")) is False:
            return False
        if _truth(row.get("fit_complete")) is False:
            return False
        fit_status = _status(row.get("fit_status"))
        if fit_status and fit_status not in GOOD_FIT_STATUS:
            return False
        if str(row.get("error") or "").strip():
            return False
        if str(row.get("reconstruction_route") or "") == "spatialtrackerv2_dynamic":
            job_id = _job_id(row.get("video_name") or row.get("job_id"))
            gate = self.dynamic_gate_by_job.get(job_id, {})
            decision = _status(gate.get("decision"))
            route = _status(gate.get("measurement_route"))
            if decision not in {"include", "qualified_3d"} and route != "qualified_dynamic_3d":
                return False
        return True

    def _representative_tuple(self, experiment_id: str) -> str | None:
        anchors = self.experiments.get(experiment_id, {}).get("anchor_tuples", [])
        if not anchors:
            return None
        default = next((item for item in anchors if item.get("id") == "default"), None)
        choice = default or anchors[len(anchors) // 2]
        return str(choice.get("id"))

    def _representative_view_tuple(
        self,
        experiment_id: str,
        grouped: Mapping[tuple[str, str, str], Mapping[str, Mapping[str, Any]]],
    ) -> str | None:
        """Use maximum baseline view coverage, breaking ties by registry order."""

        anchors = self.experiments.get(experiment_id, {}).get("anchor_tuples", [])
        scored: list[tuple[int, int, str]] = []
        for ordinal, anchor in enumerate(anchors):
            tuple_id = str(anchor.get("id"))
            views = grouped.get((experiment_id, tuple_id, "baseline"), {})
            coverage = sum(name in views for name in ("CAM_Side", "CAM_Main", "CAM_Top"))
            scored.append((coverage, -ordinal, tuple_id))
        return max(scored)[2] if scored else None

    def _build_comparison_page(self, root: Path, title: str, rows: Sequence[Mapping[str, Any]], labels: Sequence[str], intro: str) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        sources_raw: list[Path] = []
        sources_overlay: list[Path] = []
        raw_labels: list[str] = []
        overlay_labels: list[str] = []
        cards = []
        for row, label in zip(rows, labels):
            job_id = _job_id(row.get("video_name") or row.get("job_id"))
            assets = self._materialize_job(job_id)
            if assets.get("original"):
                sources_raw.append(assets["original"])
                raw_labels.append(label)
            if assets.get("overlay"):
                sources_overlay.append(assets["overlay"])
                overlay_labels.append(label)
            cards.append((row, label, assets))
        raw = root / "raw_comparison.mp4"
        overlay = root / "track_comparison.mp4"
        raw_status = {"available": False, "reason": "montages disabled"}
        overlay_status = {"available": False, "reason": "montages disabled"}
        if self.build_montages:
            raw_status = write_video_montage(sources_raw, raw_labels, raw)
            overlay_status = write_video_montage(sources_overlay, overlay_labels, overlay)
        page = root / "index.html"
        cards_html = "".join(
            f"<div class='card'><h3>{escape(label)}</h3><p>{escape(self._row_status_label(row))}</p><div class='grid'>{_media_block('原始视频', assets.get('original'), page, 'video')}{_media_block('轨迹检测', assets.get('overlay'), page, 'video')}{_media_block('2D/3D 轨迹', assets.get('trajectory'), page, 'image')}</div></div>"
            for row, label, assets in cards
        )
        body = f"<h1>{escape(title)}</h1><div class='card'><p>{escape(intro)}</p></div><div class='grid'>{_media_block('原始视频拼接', raw if raw.is_file() else None, page, 'video')}{_media_block('轨迹检测视频拼接', overlay if overlay.is_file() else None, page, 'video')}</div>{cards_html}"
        page.write_text(_html_page(title, body, root_prefix="../../../"), encoding="utf-8")
        return {"page": str(page), "raw_montage": raw_status, "overlay_montage": overlay_status}

    def build_background(self) -> dict[str, Any]:
        root = self.output / "03_background"
        root.mkdir(parents=True, exist_ok=True)
        selected = [row for row in self.rows if _number(row.get("seed")) == self.primary_seed and row.get("camera_name") == "CAM_Side"]
        baseline = {(str(row.get("experiment_id")), str(row.get("parameter_tuple_id"))): row for row in selected if row.get("scene_id") == "baseline"}
        comparisons: list[dict[str, Any]] = []
        for row in selected:
            scene = str(row.get("scene_id") or "")
            if scene == "baseline":
                continue
            key = (str(row.get("experiment_id")), str(row.get("parameter_tuple_id")))
            base = baseline.get(key)
            if base is None:
                continue
            base_evaluable = self._measurement_evaluable(base)
            scene_evaluable = self._measurement_evaluable(row)
            for spec in self.parameters.get(key[0], []):
                parameter = str(spec.get("name"))
                base_est = _estimate(base, parameter) if base_evaluable else None
                scene_est = _estimate(row, parameter) if scene_evaluable else None
                comparisons.append(
                    {
                        "experiment_id": key[0],
                        "parameter_tuple_id": key[1],
                        "scene_id": scene,
                        "parameter": parameter,
                        "generation_status": self._row_status_label(row),
                        "baseline_estimate": base_est,
                        "scene_estimate": scene_est,
                        "normalized_difference": _normalised_difference(base_est, scene_est, spec.get("valid_range", [])),
                        "job_id": _job_id(row.get("video_name")),
                    }
                )
        examples: list[dict[str, Any]] = []
        for experiment_id in self.experiments:
            tuple_id = self._representative_tuple(experiment_id)
            candidates = [row for row in selected if row.get("experiment_id") == experiment_id and row.get("parameter_tuple_id") == tuple_id]
            base = next((row for row in candidates if row.get("scene_id") == "baseline"), None)
            indoor = next((item for item in sorted(candidates, key=lambda value: str(value.get("scene_id"))) if str(item.get("scene_id", "")).startswith("indoor")), None)
            outdoor = next((item for item in sorted(candidates, key=lambda value: str(value.get("scene_id"))) if str(item.get("scene_id", "")).startswith("outdoor")), None)
            chosen = [row for row in (base, indoor, outdoor) if row is not None]
            if len(chosen) < 2:
                continue
            labels = [str(row.get("scene_id")) for row in chosen]
            comparison = self._build_comparison_page(
                root / "examples" / experiment_id,
                f"{experiment_id} · baseline / indoor / outdoor",
                chosen,
                labels,
                "固定选择规则：使用注册表 default（若无则中间 anchor）参数档位，并按场景名选首个 indoor/outdoor；不按结果好坏挑样本。数值以同设定 baseline 为参照。",
            )
            examples.append({"experiment_id": experiment_id, "parameter_tuple_id": tuple_id, **comparison})
        page = root / "index.html"
        table_rows = "".join(
            f"<tr><td>{escape(str(item['experiment_id']))}</td><td>{escape(str(item['parameter_tuple_id']))}</td><td>{escape(str(item['scene_id']))}</td><td>{escape(str(item['parameter']))}</td><td>{escape(str(item['generation_status']))}</td><td>{_fmt(item['baseline_estimate'])}</td><td>{_fmt(item['scene_estimate'])}</td><td>{_fmt(item['normalized_difference'])}</td></tr>"
            for item in comparisons
        )
        example_links = "".join(f"<li><a href='{escape(Path(os.path.relpath(item['page'], root)).as_posix())}'>{escape(item['experiment_id'])} 视频/轨迹对比</a></li>" for item in examples)
        body = f"<h1>复杂背景鲁棒性</h1><div class='card'><p>先看复杂场景是否正常生成和可测；只有 baseline 与场景两边都有有效反演值时，才计算 <code>D_scene=|θ̂_scene-θ̂_baseline|/(θmax-θmin)</code>。这不是新的主精度分数，而是同参数设定下背景造成的相对漂移。</p><ul>{example_links}</ul></div><table><thead><tr><th>实验</th><th>设定</th><th>场景</th><th>参数</th><th>生成/测量状态</th><th>baseline 反演</th><th>场景反演</th><th>D_scene</th></tr></thead><tbody>{table_rows}</tbody></table>"
        page.write_text(_html_page("复杂背景鲁棒性", body, root_prefix="../"), encoding="utf-8")
        return {"comparison_count": len(comparisons), "example_count": len(examples), "examples": examples}

    def build_views(self) -> dict[str, Any]:
        root = self.output / "04_views"
        root.mkdir(parents=True, exist_ok=True)
        selected = [row for row in self.rows if _number(row.get("seed")) == self.primary_seed]
        grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in selected:
            grouped[(str(row.get("experiment_id")), str(row.get("parameter_tuple_id")), str(row.get("scene_id")))][str(row.get("camera_name"))] = row
        comparisons: list[dict[str, Any]] = []
        for (experiment_id, tuple_id, scene_id), views in grouped.items():
            side = views.get("CAM_Side")
            if side is None:
                continue
            for view_name in ("CAM_Main", "CAM_Top"):
                other = views.get(view_name)
                if other is None:
                    continue
                side_evaluable = self._measurement_evaluable(side)
                view_evaluable = self._measurement_evaluable(other)
                for spec in self.parameters.get(experiment_id, []):
                    parameter = str(spec.get("name"))
                    side_est = _estimate(side, parameter) if side_evaluable else None
                    view_est = _estimate(other, parameter) if view_evaluable else None
                    comparisons.append(
                        {
                            "experiment_id": experiment_id,
                            "parameter_tuple_id": tuple_id,
                            "scene_id": scene_id,
                            "view": view_name,
                            "parameter": parameter,
                            "view_status": self._row_status_label(other),
                            "side_estimate": side_est,
                            "view_estimate": view_est,
                            "normalized_difference": _normalised_difference(side_est, view_est, spec.get("valid_range", [])),
                            "view_route": other.get("reconstruction_route"),
                        }
                    )
        examples: list[dict[str, Any]] = []
        for experiment_id in self.experiments:
            tuple_id = self._representative_view_tuple(experiment_id, grouped)
            views = grouped.get((experiment_id, str(tuple_id), "baseline"), {})
            chosen = [views[name] for name in ("CAM_Side", "CAM_Main", "CAM_Top") if name in views]
            if len(chosen) < 2:
                continue
            labels = [str(row.get("camera_name")) for row in chosen]
            comparison = self._build_comparison_page(
                root / "examples" / experiment_id,
                f"{experiment_id} · Side / Main / Top",
                chosen,
                labels,
                "这些视频是相同物理设定下分别生成的单目样本，并非同步多视角。Side 是主数值参照；Main/Top 用于检查运动规律与反演参数是否在视角变化后仍大体保持。",
            )
            examples.append({"experiment_id": experiment_id, "parameter_tuple_id": tuple_id, **comparison})
        page = root / "index.html"
        table_rows = "".join(
            f"<tr><td>{escape(str(item['experiment_id']))}</td><td>{escape(str(item['parameter_tuple_id']))}</td><td>{escape(str(item['scene_id']))}</td><td>{escape(str(item['view']))}</td><td>{escape(str(item['parameter']))}</td><td>{escape(str(item['view_status']))}</td><td>{escape(str(item['view_route']))}</td><td>{_fmt(item['side_estimate'])}</td><td>{_fmt(item['view_estimate'])}</td><td>{_fmt(item['normalized_difference'])}</td></tr>"
            for item in comparisons
        )
        links = "".join(f"<li><a href='{escape(Path(os.path.relpath(item['page'], root)).as_posix())}'>{escape(item['experiment_id'])} 三视角视频/轨迹</a></li>" for item in examples)
        body = f"<h1>视角鲁棒性</h1><div class='card'><p><code>D_view=|θ̂_view-θ̂_side|/(θmax-θmin)</code>。先看视频能否正常生成和轨迹是否可测，再比较反演参数。N/A 表示不能诚实判断，不会被填成失败或零差异。</p><ul>{links}</ul></div><table><thead><tr><th>实验</th><th>设定</th><th>场景</th><th>视角</th><th>参数</th><th>状态</th><th>测量路线</th><th>Side</th><th>当前视角</th><th>D_view</th></tr></thead><tbody>{table_rows}</tbody></table>"
        page.write_text(_html_page("视角鲁棒性", body, root_prefix="../"), encoding="utf-8")
        return {"comparison_count": len(comparisons), "example_count": len(examples), "examples": examples}

    def build_camera_drift(self) -> dict[str, Any]:
        root = self.output / "05_camera_drift"
        root.mkdir(parents=True, exist_ok=True)
        gate_rows = {_job_id(row.get("job_id")): row for row in _read_csv(self.simple_report_root / "dynamic_3d_gate.csv")}
        dynamic_rows = [row for row in self.rows if str(row.get("reconstruction_route") or "") == "spatialtrackerv2_dynamic"]
        cases: list[dict[str, Any]] = []
        for row in dynamic_rows:
            job_id = _job_id(row.get("video_name") or row.get("job_id"))
            assets = self._materialize_job(job_id)
            gate = gate_rows.get(job_id, {})
            camera_name = str(row.get("camera_name") or "")
            qualified = _status(gate.get("decision")) in {"include", "qualified_3d"} or _status(gate.get("measurement_route")) == "qualified_dynamic_3d"
            key = (str(row.get("experiment_id")), str(row.get("parameter_tuple_id")), str(row.get("scene_id")), str(row.get("seed")))
            side = next((candidate for candidate in self.rows if (str(candidate.get("experiment_id")), str(candidate.get("parameter_tuple_id")), str(candidate.get("scene_id")), str(candidate.get("seed"))) == key and candidate.get("camera_name") == "CAM_Side"), None)
            comparisons = []
            fit_cards: list[tuple[str, Path]] = []
            for spec in self.parameters.get(str(row.get("experiment_id")), []):
                parameter = str(spec.get("name"))
                current_est = _estimate(row, parameter) if self._measurement_evaluable(row) else None
                if camera_name == "CAM_Side":
                    reference_est = _target(row, parameter)
                    reference_label = "prompt target"
                else:
                    reference_est = _estimate(side or {}, parameter) if side is not None and self._measurement_evaluable(side) else None
                    reference_label = "CAM_Side"
                comparisons.append((parameter, current_est, reference_est, _normalised_difference(current_est, reference_est, spec.get("valid_range", [])), reference_label))
            if camera_name == "CAM_Side":
                headline = "该 Side 视频超过固定相机阈值，属于严重相机漂移；固定相机 2D 不再使用，必须检查 3D 重建门与拟合。"
            else:
                headline = "该非 Side 视频超过固定相机阈值；用合格 3D 轨迹检查整段运动规律，并与同设定 Side 的反演参数比较。"
            case_root = root / job_id
            case_root.mkdir(parents=True, exist_ok=True)
            if assets.get("result"):
                for parameter_spec in self.parameters.get(str(row.get("experiment_id")), []):
                    parameter = str(parameter_spec.get("name"))
                    formula = get_formula_spec(str(row.get("experiment_id")), parameter)
                    card_path = case_root / f"fit_process_{_safe_name(parameter)}.html"
                    render_fit_process_card(
                        assets["result"],
                        card_path,
                        _card_formula_spec(formula, str(row.get("experiment_id")), parameter),
                        grade="dynamic_3d_fit" if qualified else "X",
                    )
                    fit_cards.append((parameter, card_path))
            lines = [
                f"# {job_id}", "", headline, "",
                f"- 3D 门控：`{'QUALIFIED_3D' if qualified else 'X'}`",
                f"- 原因：`{_format_list(_codes(gate.get('reason_codes') or gate.get('simple_dynamic_3d_reason_codes')))}`",
                f"- 有效 3D 比例：`{_fmt(gate.get('valid_fraction') or gate.get('metrics.valid_fraction'))}`",
                f"- 主运动范围：`{_fmt(gate.get('main_motion_range_m'))} m`",
                f"- 离预期运动流形范围比：`{_fmt(gate.get('off_manifold_range_ratio'))}`",
                f"- 运动方程 NRMSE：`{_fmt(gate.get('motion_law_median_nrmse'))}`",
                "", "## 参数一致性", "", "|参数|当前反演|参照|参照值|归一化差异|", "|---|---:|---|---:|---:|",
            ]
            lines += [f"|{name}|{_fmt(current)}|{reference_label}|{_fmt(reference)}|{_fmt(diff)}|" for name, current, reference, diff, reference_label in comparisons]
            lines += ["", "当前 fitter 给出的是整条轨迹的单一有效参数，所以这里只能声明 **whole-trajectory consistency**；若要证明漂移过程中参数逐时刻连续，需要另做滑窗拟合，当前报告不会夸大为时间连续证明。"]
            (case_root / "DRIFT_ASSESSMENT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            page = case_root / "index.html"
            table = "".join(f"<tr><td>{escape(name)}</td><td>{_fmt(current)}</td><td>{escape(reference_label)}</td><td>{_fmt(reference)}</td><td>{_fmt(diff)}</td></tr>" for name, current, reference, diff, reference_label in comparisons)
            fit_html = "".join(_document_block(f"{parameter} 公式与拟合过程", card, page) for parameter, card in fit_cards)
            body = f"<h1>{escape(job_id)}</h1><div class='card'><p class='bad'>{escape(headline)}</p><p>3D 门控：<b>{'QUALIFIED_3D' if qualified else 'X'}</b>；原因：{escape(_format_list(_codes(gate.get('reason_codes') or gate.get('simple_dynamic_3d_reason_codes'))))}</p></div><div class='grid'>{_media_block('原始漂移视频', assets.get('original'), page, 'video')}{_media_block('3D/2D 轨迹检测 overlay', assets.get('overlay'), page, 'video')}{_media_block('2D 与 3D 轨迹', assets.get('trajectory'), page, 'image')}{_media_block('SpatialTrackerV2 原生 3D 图', assets.get('native_3d'), page, 'image')}</div><h2>{'相对 prompt target 的拟合精度' if camera_name == 'CAM_Side' else '相对同设定 Side 的反演参数'}</h2><table><tr><th>参数</th><th>当前</th><th>参照</th><th>参照值</th><th>归一化差异</th></tr>{table}</table><h2>冻结公式与拟合过程</h2>{fit_html}<p class='muted'>这里评估整段轨迹一致性，不声称逐时刻参数恒定。</p>"
            page.write_text(_html_page(job_id, body, root_prefix="../../"), encoding="utf-8")
            cases.append({"job_id": job_id, "camera_name": camera_name, "qualified_3d": qualified, "page": str(page)})
        page = root / "index.html"
        links = "".join(f"<li><a href='{escape(Path(os.path.relpath(item['page'], root)).as_posix())}'>{escape(item['job_id'])}</a> · {escape(item['camera_name'])} · {'QUALIFIED_3D' if item['qualified_3d'] else 'X'}</li>" for item in cases)
        body = f"<h1>相机漂移与动态 3D</h1><div class='card'><p>Side 漂移先明确标为严重相机漂移，再检查 3D 是否落在该实验预期的一维轴/二维平面运动流形内；通过 target-independent 门后才拟合。Main/Top 漂移则比较整段反演参数与 Side 是否近似。未通过 3D 门的样本是测量 X，不冒充模型失败。</p><ul>{links}</ul></div>"
        page.write_text(_html_page("相机漂移", body, root_prefix="../"), encoding="utf-8")
        return {"dynamic_case_count": len(cases), "qualified_count": sum(item["qualified_3d"] for item in cases), "cases": cases}

    def build_overview(self, sections: Mapping[str, Any]) -> None:
        overview = self.output / "00_overview"
        overview.mkdir(parents=True, exist_ok=True)
        figures = []
        for name in ("fig2_baseline_response_grades.svg", "fig3_requested_vs_recovered.svg", "fig4_evaluation_scopes.svg", "fig5_dynamic_3d_gate.svg"):
            source = self.simple_report_root / name
            if source.is_file():
                destination = overview / name
                materialize_media(source, destination, mode=self.media_mode)
                figures.append(destination)
        experiment_rows = []
        axis_total = 0
        for experiment_id, specs in self.parameters.items():
            count = len(specs)
            axis_total += count
            experiment_rows.append(f"<tr><td>{escape(experiment_id)}</td><td>{count}</td><td>{escape(', '.join(str(spec.get('name')) for spec in specs))}</td></tr>")
        body = [
            "<h1>PhysParamBench 人可读视频证据报告</h1>",
            "<div class='card'><h2>先回答 13 与 24</h2>",
            f"<p><b>{len(self.experiments)} 个实验系统</b>包含<b>{axis_total} 个待反演物理参数轴</b>。等级 L2–L4 的统计单位是 OAT 参数扫描，而不是实验编号；所以多参数的 V2/V3 每个实验会有 2–3 个结果。</p>",
            "<table><tr><th>实验</th><th>扫描轴数</th><th>参数</th></tr>" + "".join(experiment_rows) + "</table></div>",
            "<div class='grid'>",
            f"<div class='card'><h2>不可用证据</h2><p>{sections['unusable']['case_count']} 个案例页：原视频、overlay、2D/3D 轨迹与原因。</p><a href='01_unusable/index.html'>打开</a></div>",
            f"<div class='card'><h2>参数响应</h2><p>24 个响应轴均有页面；其中 {sections['scans']['l2_l4_scan_count']} 个 L2–L4，{sections['scans']['x_scan_count']} 个扫描级 X。包含不同设定原视频/轨迹视频与公式反推。</p><a href='02_parameter_scans/index.html'>打开</a></div>",
            f"<div class='card'><h2>复杂背景</h2><p>{sections['background']['comparison_count']} 个同设定 baseline 对照条目。</p><a href='03_background/index.html'>打开</a></div>",
            f"<div class='card'><h2>视角</h2><p>{sections['views']['comparison_count']} 个 Main/Top 相对 Side 的参数对照条目。</p><a href='04_views/index.html'>打开</a></div>",
            f"<div class='card'><h2>相机漂移</h2><p>{sections['drift']['dynamic_case_count']} 个动态 3D 路由案例，合格 {sections['drift']['qualified_count']}。</p><a href='05_camera_drift/index.html'>打开</a></div>",
            "</div>",
            "<div class='card'><h2>读图原则</h2><ul><li>baseline + Side + 主 seed 是数值参数主统计。</li><li>indoor/outdoor 看复杂视觉输入是否破坏生成与同设定参数保持。</li><li>Main/Top 是独立生成视角鲁棒性，不伪装成同步多视角。</li><li>Side 严重漂移转 3D；只有通过目标参数无关的运动流形门才拟合。</li><li>X 是测量证据不足，不是模型失败；只有人工确认的形变/消失/穿模/错误运动才是 L1。</li></ul></div>",
        ]
        if figures:
            body.append("<h2>已有总览图</h2><div class='grid'>")
            page = self.output / "index.html"
            body.extend(_media_block(path.stem, path, page, "image") for path in figures)
            body.append("</div>")
        (self.output / "index.html").write_text(_html_page("PhysParamBench 视频证据", "".join(body)), encoding="utf-8")
        readme = [
            "# PhysParamBench 人可读证据包",
            "",
            "从 `index.html` 开始查看。本目录由冻结评测结果只读生成，不重新跟踪、不重新拟合、不修改等级。",
            "",
            f"- 实验系统：{len(self.experiments)}",
            f"- 参数响应轴：{axis_total}",
            f"- 参数扫描证据页：{sections['scans']['rendered_scan_count']}（L2–L4={sections['scans']['l2_l4_scan_count']}，X={sections['scans']['x_scan_count']}）",
            f"- 不可用/待复核案例页：{sections['unusable']['case_count']}",
            "",
            "视频与图像是主要交付；`_machine/evidence_manifest.jsonl` 仅用于追溯文件来源，无需人工阅读。",
        ]
        (self.output / "README_ZH.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    def build(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        unusable = self.build_unusable()
        scans = self.build_parameter_scans()
        background = self.build_background()
        views = self.build_views()
        drift = self.build_camera_drift()
        sections = {"unusable": unusable, "scans": scans, "background": background, "views": views, "drift": drift}
        self.build_overview(sections)
        _write_jsonl(self.output / "_machine" / "evidence_manifest.jsonl", self.manifest)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "report_type": "human_readable_video_first_evidence",
            "read_only_renderer": True,
            "experiment_count": len(self.experiments),
            "parameter_response_axis_count": sum(len(value) for value in self.parameters.values()),
            "input_video_count": len(self.rows),
            "sections": {
                "unusable": {key: value for key, value in unusable.items() if key != "cases"},
                "parameter_scans": {key: value for key, value in scans.items() if key != "scans"},
                "background": {key: value for key, value in background.items() if key != "examples"},
                "views": {key: value for key, value in views.items() if key != "examples"},
                "camera_drift": {key: value for key, value in drift.items() if key != "cases"},
            },
            "entrypoint": str(self.output / "index.html"),
            "scope_note": "baseline Side is primary; background/view/seed are robustness; dynamic 3D enters only after its evidence gate",
        }
        _write_json(self.output / "summary.json", summary)
        return summary


def _prepare_owned_output(output: Path) -> None:
    """Clear only a directory previously owned by this report renderer."""

    marker = output / ".physparambench_evidence_report"
    summary = _read_json(output / "summary.json") if output.is_dir() else {}
    legacy_owned = summary.get("report_type") == "human_readable_video_first_evidence"
    if output.exists():
        entries = list(output.iterdir()) if output.is_dir() else []
        if not output.is_dir():
            raise ValueError(f"--output must be a directory path: {output}")
        if entries and not marker.is_file() and not legacy_owned:
            raise ValueError(
                f"Refusing to overwrite non-report directory: {output}. "
                "Choose an empty output or a directory containing the report ownership marker."
            )
    else:
        output.mkdir(parents=True, exist_ok=True)
    if marker.is_file() or legacy_owned:
        for name in (
            "00_overview", "01_unusable", "02_parameter_scans", "03_background",
            "04_views", "05_camera_drift", "_assets", "_machine",
        ):
            path = output / name
            if path.is_dir():
                shutil.rmtree(path)
        for name in ("index.html", "README_ZH.md", "summary.json"):
            path = output / name
            if path.is_file():
                path.unlink()
    marker.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "owner": "PhysParamBench evidence renderer"}) + "\n",
        encoding="utf-8",
    )


def build_evidence_report(
    *,
    all_jobs_csv: Path,
    simple_report_root: Path,
    evaluation_root: Path,
    videos_root: Path,
    output: Path,
    registry_path: Path,
    tracks_root: Path | None = None,
    media_mode: str = "hardlink",
    build_montages: bool = True,
    primary_seed: int = PRIMARY_SEED,
) -> dict[str, Any]:
    """Build the report from frozen evaluation outputs."""

    required = {
        "all_jobs_csv": all_jobs_csv,
        "simple parameter scans": simple_report_root / "parameter_scans.jsonl",
        "experiment registry": registry_path,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report input(s):\n  - " + "\n  - ".join(missing))
    registry = _read_json(registry_path)
    if not registry:
        raise ValueError(f"Invalid/empty experiment registry: {registry_path}")
    _prepare_owned_output(output)
    builder = EvidenceReportBuilder(
        all_jobs=_read_csv(all_jobs_csv),
        scans=_read_jsonl(simple_report_root / "parameter_scans.jsonl"),
        video_gate=_read_csv(simple_report_root / "video_gate.csv"),
        registry=registry,
        simple_report_root=simple_report_root,
        evaluation_root=evaluation_root,
        tracks_root=tracks_root,
        videos_root=videos_root,
        output=output,
        media_mode=media_mode,
        build_montages=build_montages,
        primary_seed=int(primary_seed),
    )
    return builder.build()


__all__ = ["ArtifactLocator", "EvidenceReportBuilder", "build_evidence_report"]
