"""Resumable, view-aware evaluation orchestration for the frozen Seedance 978 set."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from remake_benchmark.hybrid import choose_reconstruction_route

from .fixed_camera_ball import calibration_path, parse_video_job
from .physics_evaluation import (
    DEFAULT_BENCHMARK_SEED,
    _summary_row,
    aggregate_results,
    benchmark_split,
    load_experiment_registry,
    run_physics_job,
)


EXPECTED_COUNTS = {
    "all": 978,
    "CAM_Side": 510,
    "side_primary": 432,
    "side_seed_stability_extra": 78,
    "CAM_Main": 234,
    "CAM_Top": 234,
    "view_triplets": 234,
    "seed_stability_groups": 26,
}
CAMERA_ORDER = {"CAM_Side": 0, "CAM_Main": 1, "CAM_Top": 2}


def _run_physics_job_process(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pickle-safe worker for CPU-parallel per-video evaluation."""

    values = dict(payload)
    job_id = str(values.pop("job_id"))
    return job_id, run_physics_job(**values)


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _manifest_job_id(row: Mapping[str, Any]) -> str:
    return str(row["job_id"])


def validate_978_inputs(videos_dir: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    videos = sorted(videos_dir.glob("*.mp4"))
    video_ids = [path.stem for path in videos]
    manifest_ids = [_manifest_job_id(row) for row in manifest_rows]
    duplicates = sorted({value for value in manifest_ids if manifest_ids.count(value) > 1})
    missing = sorted(set(manifest_ids) - set(video_ids))
    extra = sorted(set(video_ids) - set(manifest_ids))
    if duplicates or missing or extra or len(manifest_ids) != EXPECTED_COUNTS["all"]:
        raise ValueError(
            "Seedance 978 coverage mismatch: "
            f"manifest={len(manifest_ids)} videos={len(video_ids)} duplicates={len(duplicates)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    jobs = [parse_video_job(videos_dir / f"{job_id}.mp4") for job_id in manifest_ids]
    counts: dict[str, int] = defaultdict(int)
    for job in jobs:
        counts[str(job["camera_name"])] += 1
        counts[benchmark_split(job)] += 1
    expected_subset = {
        "CAM_Side": 510,
        "side_primary": 432,
        "side_seed_stability_extra": 78,
        "CAM_Main": 234,
        "CAM_Top": 234,
    }
    wrong = {key: (counts.get(key, 0), expected) for key, expected in expected_subset.items() if counts.get(key, 0) != expected}
    if wrong:
        raise ValueError(f"Seedance 978 split-count mismatch: {wrong}")
    return {"manifest_jobs": len(manifest_ids), "video_jobs": len(video_ids), "counts": dict(counts)}


def _audit_index(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        filename = row.get("filename")
        if not filename and row.get("video"):
            filename = Path(str(row["video"])).name
        if not filename:
            raise ValueError(f"camera audit row has no filename: {row}")
        if filename in output:
            raise ValueError(f"duplicate camera audit row: {filename}")
        output[str(filename)] = row
    return output


def _ordered_manifest(manifest_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in manifest_rows),
        key=lambda row: (
            CAMERA_ORDER.get(str(row.get("factors", {}).get("camera")), 99),
            str(row["job_id"]),
        ),
    )


def _runtime_spatracker_manifest(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    videos_dir: Path,
    workspace_root: Path,
    calibration_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in manifest_rows:
        factors = source["factors"]
        job = parse_video_job(videos_dir / f"{source['job_id']}.mp4")
        first_frame = workspace_root / str(source["inputs"]["image"])
        rows.append(
            {
                "job_id": source["job_id"],
                "experiment_id": source["experiment_id"],
                "parameter_tuple_id": factors["parameter_tuple_id"],
                "scene": factors["scene_id"],
                "object": factors["object_id"],
                "camera": factors["camera"],
                "seed": int(source["seed"]),
                "video": str((videos_dir / f"{source['job_id']}.mp4").resolve()),
                "first_frame": str(first_frame.resolve()),
                "calibration": str(calibration_path(calibration_root, job).resolve()),
                "source_fps": float(source.get("inputs", {}).get("source_fps") or 24.0),
            }
        )
    return rows


def _result_path(output_root: Path, job_id: str) -> Path:
    return output_root / "jobs" / job_id / "result.json"


def _load_result(output_root: Path, job_id: str) -> dict[str, Any] | None:
    path = _result_path(output_root, job_id)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _range_map(registry: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for experiment in registry.get("experiments", []):
        ranges: dict[str, float] = {}
        for parameter in experiment.get("hidden_parameters", []):
            low, high = parameter["valid_range"]
            ranges[str(parameter["name"])] = float(high) - float(low)
        output[str(experiment["id"])] = ranges
    return output


def _estimate_difference(
    reference: Mapping[str, Any],
    comparison: Mapping[str, Any],
    ranges: Mapping[str, float],
) -> tuple[float | None, dict[str, float]]:
    first = reference.get("fit", {}).get("parameter_estimates", {})
    second = comparison.get("fit", {}).get("parameter_estimates", {})
    values: dict[str, float] = {}
    for name, span in ranges.items():
        a, b = first.get(name), second.get(name)
        if a is None or b is None or span <= 0:
            continue
        values[name] = abs(float(b) - float(a)) / span
    return (float(np.mean(list(values.values()))) if values else None), values


def _view_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    job = result["job"]
    return (
        job["experiment_id"],
        job["parameter_tuple_id"],
        job["scene_id"],
        job["object_id"],
        int(job["seed"]),
    )


def _viewpoint_rows(
    results: Sequence[Mapping[str, Any]],
    ranges: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for result in results:
        grouped[_view_key(result)][str(result["job"]["camera_name"])] = result
    rows: list[dict[str, Any]] = []
    for key, views in sorted(grouped.items()):
        if not {"CAM_Side", "CAM_Main", "CAM_Top"}.issubset(views):
            continue
        side, main, top = views["CAM_Side"], views["CAM_Main"], views["CAM_Top"]
        experiment_id = str(key[0])
        main_delta, main_parameters = _estimate_difference(side, main, ranges[experiment_id])
        top_delta, top_parameters = _estimate_difference(side, top, ranges[experiment_id])
        row = {
            "experiment_id": key[0],
            "parameter_tuple_id": key[1],
            "scene_id": key[2],
            "object_id": key[3],
            "seed": key[4],
            "side_validity": side.get("video_generation_validity", {}).get("status"),
            "main_validity": main.get("video_generation_validity", {}).get("status"),
            "top_validity": top.get("video_generation_validity", {}).get("status"),
            "all_three_generation_valid": all(
                item.get("video_generation_validity", {}).get("status") == "pass"
                for item in (side, main, top)
            ),
            "side_fit_complete": side.get("metrics", {}).get("fit_complete"),
            "main_fit_complete": main.get("metrics", {}).get("fit_complete"),
            "top_fit_complete": top.get("metrics", {}).get("fit_complete"),
            "side_score_0_100": side.get("metrics", {}).get("experiment_score_0_100"),
            "main_score_0_100": main.get("metrics", {}).get("experiment_score_0_100"),
            "top_score_0_100": top.get("metrics", {}).get("experiment_score_0_100"),
            "main_vs_side_parameter_range_normalized_mae": main_delta,
            "top_vs_side_parameter_range_normalized_mae": top_delta,
            "main_vs_side_parameter_differences_json": json.dumps(main_parameters, sort_keys=True),
            "top_vs_side_parameter_differences_json": json.dumps(top_parameters, sort_keys=True),
        }
        side_score = row["side_score_0_100"]
        main_score = row["main_score_0_100"]
        top_score = row["top_score_0_100"]
        row["main_vs_side_score_abs_delta"] = (
            None if side_score is None or main_score is None else abs(float(main_score) - float(side_score))
        )
        row["top_vs_side_score_abs_delta"] = (
            None if side_score is None or top_score is None else abs(float(top_score) - float(side_score))
        )
        rows.append(row)
    return rows


def _seed_stability_rows(
    results: Sequence[Mapping[str, Any]],
    ranges: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        job = result["job"]
        if job["camera_name"] != "CAM_Side" or job["scene_id"] != "baseline":
            continue
        groups[(job["experiment_id"], job["parameter_tuple_id"], job["scene_id"], job["object_id"])].append(result)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        if len({int(item["job"]["seed"]) for item in items}) <= 1:
            continue
        experiment_id = key[0]
        parameter_stds: dict[str, float] = {}
        for name, span in ranges[experiment_id].items():
            estimates = [
                item.get("fit", {}).get("parameter_estimates", {}).get(name)
                for item in items
                if item.get("video_generation_validity", {}).get("status") == "pass"
            ]
            values = [float(value) for value in estimates if value is not None]
            if len(values) >= 2 and span > 0:
                parameter_stds[name] = float(np.std(values, ddof=1) / span)
        rows.append(
            {
                "experiment_id": experiment_id,
                "parameter_tuple_id": key[1],
                "scene_id": key[2],
                "object_id": key[3],
                "seed_count": len({int(item["job"]["seed"]) for item in items}),
                "generation_valid_count": sum(
                    item.get("video_generation_validity", {}).get("status") == "pass"
                    for item in items
                ),
                "complete_fit_count": sum(bool(item.get("metrics", {}).get("fit_complete")) for item in items),
                "mean_parameter_range_normalized_std": (
                    float(np.mean(list(parameter_stds.values()))) if parameter_stds else None
                ),
                "parameter_range_normalized_std_json": json.dumps(parameter_stds, sort_keys=True),
                "seeds_json": json.dumps(sorted(int(item["job"]["seed"]) for item in items)),
            }
        )
    return rows


def write_seedance978_reports(
    output_root: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = _ordered_manifest(manifest_rows)
    results = [
        result
        for row in ordered
        if (result := _load_result(output_root, str(row["job_id"]))) is not None
    ]
    summary_rows = [_summary_row(result) for result in results]
    _write_csv(output_root / "all_jobs.csv", summary_rows)
    side_primary = [
        result
        for result in results
        if (result.get("benchmark_split") or benchmark_split(result["job"])) == "side_primary"
    ]
    main = [
        result
        for result in results
        if (result.get("benchmark_split") or benchmark_split(result["job"])) == "main_robustness"
    ]
    top = [
        result
        for result in results
        if (result.get("benchmark_split") or benchmark_split(result["job"])) == "top_robustness"
    ]
    seed_extra = [
        result
        for result in results
        if (result.get("benchmark_split") or benchmark_split(result["job"]))
        == "side_seed_stability_extra"
    ]
    _write_csv(output_root / "side_primary_summary.csv", [_summary_row(item) for item in side_primary])
    ranges = _range_map(registry)
    view_rows = _viewpoint_rows(results, ranges)
    seed_rows = _seed_stability_rows(results, ranges)
    _write_csv(output_root / "viewpoint_robustness.csv", view_rows)
    _write_csv(output_root / "seed_stability.csv", seed_rows)

    def mean_present(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    aggregate = {
        "schema_version": "2.0.0",
        "expected_counts": EXPECTED_COUNTS,
        "result_count": len(results),
        "missing_result_count": len(manifest_rows) - len(results),
        "execution_order": ["CAM_Side", "CAM_Main", "CAM_Top"],
        "headline": {
            "name": "CAM_Side primary physics identification",
            "scheduled_count": EXPECTED_COUNTS["side_primary"],
            "result_count": len(side_primary),
            "metrics": aggregate_results(side_primary),
        },
        "main_robustness": {"result_count": len(main), "metrics": aggregate_results(main)},
        "top_robustness": {"result_count": len(top), "metrics": aggregate_results(top)},
        "viewpoint_robustness": {
            "expected_triplet_count": EXPECTED_COUNTS["view_triplets"],
            "triplet_result_count": len(view_rows),
            "all_three_generation_valid_count": sum(bool(row["all_three_generation_valid"]) for row in view_rows),
            "mean_main_vs_side_parameter_range_normalized_mae": mean_present(
                view_rows, "main_vs_side_parameter_range_normalized_mae"
            ),
            "mean_top_vs_side_parameter_range_normalized_mae": mean_present(
                view_rows, "top_vs_side_parameter_range_normalized_mae"
            ),
            "mean_main_vs_side_score_abs_delta": mean_present(view_rows, "main_vs_side_score_abs_delta"),
            "mean_top_vs_side_score_abs_delta": mean_present(view_rows, "top_vs_side_score_abs_delta"),
            "note": "Views are independently generated monocular samples; comparison is robustness, not synchronized triangulation.",
        },
        "seed_stability": {
            "extra_result_count": len(seed_extra),
            "expected_group_count": EXPECTED_COUNTS["seed_stability_groups"],
            "group_result_count": len(seed_rows),
            "complete_four_seed_group_count": sum(int(row["seed_count"]) == 4 for row in seed_rows),
            "mean_parameter_range_normalized_std": mean_present(
                seed_rows, "mean_parameter_range_normalized_std"
            ),
        },
        "reporting_policy": {
            "generation_validity_denominator": "all scheduled videos, including failures and indeterminate cases",
            "physics_accuracy": "reported for completed fits from trajectory-usable videos; fail videos are never fitted, and indeterminate videos require a separately audited partial trajectory",
            "side_primary_excludes_extra_seeds": True,
            "main_top_are_robustness_not_headline": True,
        },
    }
    _write_json(output_root / "aggregate.json", aggregate)
    return aggregate


def _phase_rows(rows: Sequence[Mapping[str, Any]], phase: str) -> list[dict[str, Any]]:
    if phase == "side":
        return [dict(row) for row in rows if row.get("factors", {}).get("camera") == "CAM_Side"]
    if phase == "robustness":
        return [dict(row) for row in rows if row.get("factors", {}).get("camera") in {"CAM_Main", "CAM_Top"}]
    if phase == "main":
        return [dict(row) for row in rows if row.get("factors", {}).get("camera") == "CAM_Main"]
    if phase == "top":
        return [dict(row) for row in rows if row.get("factors", {}).get("camera") == "CAM_Top"]
    if phase == "all":
        return [dict(row) for row in rows]
    raise ValueError(f"unknown phase: {phase}")


def _route_adjusted_camera_evidence(
    evidence: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the frozen audit intact while exposing the route's effective category."""

    adjusted = dict(evidence)
    adjusted["raw_final_category"] = evidence.get("final_category")
    adjusted["effective_camera_motion_category"] = route.get(
        "effective_camera_motion_category"
    )
    adjusted["reconstruction_route"] = route.get("route")
    adjusted["reconstruction_route_reason"] = route.get("reason")
    if route.get("side_3d_evidence") is not None:
        adjusted["side_3d_evidence"] = route["side_3d_evidence"]
    return adjusted


def run_seedance978_evaluation(
    *,
    workspace_root: Path,
    videos_dir: Path,
    manifest_path: Path,
    audit_jsonl: Path,
    calibration_root: Path,
    registry_path: Path,
    output_root: Path,
    spatialtracker_script: Path,
    phase: str = "all",
    job_ids: Sequence[str] | None = None,
    max_jobs: int | None = None,
    overwrite: bool = False,
    assess_background_rigidity: bool = True,
    overlay_count: int = 12,
    run_dynamic: bool = False,
    dynamic_frame_stride: int = 1,
    isolated_process: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    manifest_rows = _ordered_manifest(_read_jsonl(manifest_path))
    coverage = validate_978_inputs(videos_dir, manifest_rows)
    audit = _audit_index(audit_jsonl)
    missing_audit = sorted(
        f"{row['job_id']}.mp4" for row in manifest_rows if f"{row['job_id']}.mp4" not in audit
    )
    if missing_audit:
        raise ValueError(f"camera audit is missing {len(missing_audit)} videos")
    registry = load_experiment_registry(registry_path)
    runtime_manifest = _runtime_spatracker_manifest(
        manifest_rows,
        videos_dir=videos_dir,
        workspace_root=workspace_root,
        calibration_root=calibration_root,
    )
    runtime_manifest_path = output_root / "runtime_spatialtracker_manifest.jsonl"
    _write_jsonl(runtime_manifest_path, runtime_manifest)
    selected = _phase_rows(manifest_rows, phase)
    requested_job_ids = list(dict.fromkeys(str(value) for value in (job_ids or [])))
    if requested_job_ids:
        all_job_ids = {str(row["job_id"]) for row in manifest_rows}
        unknown = sorted(set(requested_job_ids) - all_job_ids)
        if unknown:
            raise ValueError(f"unknown manifest job_id(s): {', '.join(unknown)}")
        phase_job_ids = {str(row["job_id"]) for row in selected}
        outside_phase = sorted(set(requested_job_ids) - phase_job_ids)
        if outside_phase:
            raise ValueError(
                f"requested job_id(s) are outside phase={phase}: {', '.join(outside_phase)}"
            )
        requested = set(requested_job_ids)
        selected = [row for row in selected if str(row["job_id"]) in requested]
    if max_jobs is not None:
        selected = selected[: max(0, int(max_jobs))]

    if requested_job_ids:
        selection_digest = hashlib.sha256(
            "\n".join(requested_job_ids).encode("utf-8")
        ).hexdigest()[:10]
        artifact_suffix = f"targeted_{phase}_{selection_digest}"
        routing_path = output_root / f"routing_{artifact_suffix}.jsonl"
        metadata_path = output_root / f"run_metadata_{artifact_suffix}.json"
    else:
        artifact_suffix = phase
        routing_path = output_root / f"routing_{phase}.jsonl"
        metadata_path = output_root / "run_metadata.json"

    routes: list[dict[str, Any]] = []
    evaluation_evidence_by_id: dict[str, dict[str, Any]] = {}
    for row in selected:
        job_id = str(row["job_id"])
        filename = f"{job_id}.mp4"
        camera = str(row.get("factors", {}).get("camera") or "")
        decision = choose_reconstruction_route(
            audit[filename],
            filename,
            camera_name=camera,
        )
        routes.append({"job_id": job_id, **decision})
        evaluation_evidence_by_id[job_id] = _route_adjusted_camera_evidence(
            audit[filename], decision
        )
    _write_jsonl(routing_path, routes)
    route_by_id = {row["job_id"]: row for row in routes}

    worker_count = max(1, int(workers))
    overlay_remaining = max(0, int(overlay_count))
    dynamic_candidates: list[str] = []
    payloads: list[dict[str, Any]] = []
    route_and_dynamic_dir: dict[str, tuple[str, Path]] = {}
    for ordinal, row in enumerate(selected, 1):
        job_id = str(row["job_id"])
        filename = f"{job_id}.mp4"
        route = str(route_by_id[job_id]["route"])
        dynamic_dir = output_root / "dynamic" / job_id
        route_and_dynamic_dir[job_id] = (route, dynamic_dir)
        payloads.append(
            {
                "job_id": job_id,
                "video_path": videos_dir / filename,
                "calibration_root": calibration_root,
                "registry": registry,
                "output_root": output_root / "jobs",
                "overwrite": overwrite,
                "camera_motion_evidence": evaluation_evidence_by_id[job_id],
                "reconstruction_route": route,
                "dynamic_result_dir": dynamic_dir,
                "assess_background_rigidity": assess_background_rigidity,
                # Parallel runs choose overlays deterministically by manifest
                # order.  Invalid examples are useful diagnostics too.
                "make_overlay": ordinal <= max(0, int(overlay_count)),
            }
        )

    completed_results: dict[str, dict[str, Any]] = {}
    if worker_count == 1:
        for ordinal, payload in enumerate(payloads, 1):
            job_id, result = _run_physics_job_process(payload)
            completed_results[job_id] = result
            if result.get("video_generation_validity", {}).get("status") == "pass" and overlay_remaining > 0:
                overlay_remaining -= 1
            route, dynamic_dir = route_and_dynamic_dir[job_id]
            if (
                route == "spatialtrackerv2_dynamic"
                and result.get("video_generation_validity", {}).get("status") != "fail"
                and not (dynamic_dir / "result.json").is_file()
            ):
                dynamic_candidates.append(job_id)
            print(
                f"[{ordinal}/{len(selected)}] {job_id} route={route} "
                f"validity={result.get('video_generation_validity', {}).get('status')} "
                f"fit={result.get('fit', {}).get('status')}",
                flush=True,
            )
            if ordinal % 25 == 0:
                write_seedance978_reports(output_root, manifest_rows, registry)
    else:
        overlay_remaining = 0
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {
                executor.submit(_run_physics_job_process, payload): str(payload["job_id"])
                for payload in payloads
            }
            for completed, future in enumerate(as_completed(future_to_job), 1):
                job_id, result = future.result()
                completed_results[job_id] = result
                route, dynamic_dir = route_and_dynamic_dir[job_id]
                if (
                    route == "spatialtrackerv2_dynamic"
                    and result.get("video_generation_validity", {}).get("status") != "fail"
                    and not (dynamic_dir / "result.json").is_file()
                ):
                    dynamic_candidates.append(job_id)
                print(
                    f"[{completed}/{len(selected)}] {job_id} route={route} "
                    f"validity={result.get('video_generation_validity', {}).get('status')} "
                    f"fit={result.get('fit', {}).get('status')}",
                    flush=True,
                )

        selected_order = {str(row["job_id"]): index for index, row in enumerate(selected)}
        dynamic_candidates.sort(key=lambda value: selected_order[value])

    if run_dynamic and dynamic_candidates:
        command = [
            sys.executable,
            str(spatialtracker_script),
            "--manifest",
            str(runtime_manifest_path),
            "--output-root",
            str(output_root / "dynamic"),
            "--work-root",
            str(output_root / "dynamic_work"),
            "--frame-stride",
            str(max(1, int(dynamic_frame_stride))),
        ]
        for job_id in dynamic_candidates:
            command.extend(["--job-id", job_id])
        if isolated_process:
            command.append("--isolated-process")
        if overwrite:
            command.append("--rerun")
        subprocess.run(command, check=True, cwd=spatialtracker_script.parent.parent)
        for ordinal, job_id in enumerate(dynamic_candidates, 1):
            filename = f"{job_id}.mp4"
            result = run_physics_job(
                videos_dir / filename,
                calibration_root=calibration_root,
                registry=registry,
                output_root=output_root / "jobs",
                overwrite=True,
                camera_motion_evidence=evaluation_evidence_by_id[job_id],
                reconstruction_route="spatialtrackerv2_dynamic",
                dynamic_result_dir=output_root / "dynamic" / job_id,
                assess_background_rigidity=False,
                make_overlay=overlay_remaining > 0,
            )
            if result.get("video_generation_validity", {}).get("status") == "pass" and overlay_remaining > 0:
                overlay_remaining -= 1
            print(
                f"[dynamic {ordinal}/{len(dynamic_candidates)}] {job_id} "
                f"validity={result.get('video_generation_validity', {}).get('status')} "
                f"fit={result.get('fit', {}).get('status')}",
                flush=True,
            )

    aggregate = write_seedance978_reports(output_root, manifest_rows, registry)
    _write_json(
        metadata_path,
        {
            "schema_version": "1.0.0",
            "phase": phase,
            "requested_job_ids": requested_job_ids,
            "artifact_suffix": artifact_suffix,
            "selected_jobs": len(selected),
            "coverage": coverage,
            "dynamic_candidate_count": len(dynamic_candidates),
            "dynamic_executed": bool(run_dynamic),
            "workers": worker_count,
            "route_counts": {
                route_name: sum(1 for item in routes if item.get("route") == route_name)
                for route_name in sorted({str(item.get("route")) for item in routes})
            },
            "paths": {
                "videos": str(videos_dir),
                "manifest": str(manifest_path),
                "camera_audit": str(audit_jsonl),
                "calibration_root": str(calibration_root),
                "registry": str(registry_path),
                "output": str(output_root),
                "routing": str(routing_path),
            },
        },
    )
    return aggregate


__all__ = [
    "EXPECTED_COUNTS",
    "run_seedance978_evaluation",
    "validate_978_inputs",
    "write_seedance978_reports",
]
