#!/usr/bin/env python3
"""Route fixed V1A videos to calibration and drifting videos to SpaTrackerV2."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REMAKE_ROOT = PACKAGE_ROOT.parents[1]
CODE_ROOT = REMAKE_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.hybrid import choose_reconstruction_route  # noqa: E402


DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "v1a_seedance27.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "v1a_hybrid"
DEFAULT_CONFIG = CODE_ROOT / "configs" / "reconstruction" / "v1a_metric_v1.yaml"
DEFAULT_CALIBRATION = CODE_ROOT / "assets" / "v1a_calibration"

AUDIT_FIELDS = {
    "filename",
    "video",
    "status",
    "decision",
    "final_category",
    "reasons",
    "measurement_source",
    "max_direct_translation_px",
    "max_direct_rotation_deg",
    "max_direct_scale_change",
    "direct_motion_cluster_max",
    "direct_translation_hit_count",
    "direct_rotation_hit_count",
    "direct_scale_hit_count",
    "direct_strong_motion_sample_count",
    "strong_adjacent_pair_count",
    "valid_pair_fraction",
    "median_inlier_ratio",
    "median_residual_px",
    "cut_pair_count",
    "error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument(
        "--audit-jsonl",
        type=Path,
        default=None,
        help="Existing raw/final camera-motion JSONL. If omitted, run the audit first.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workspace-root", type=Path, default=REMAKE_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--overlay-count", type=int, default=6)
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--audit-stride", type=int, default=4)
    parser.add_argument("--dynamic-frame-stride", type=int, default=1)
    parser.add_argument("--isolated-process", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _compact_evidence(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in AUDIT_FIELDS if key in row}


def _index_audit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        filename = row.get("filename")
        if not filename and row.get("video"):
            filename = Path(str(row["video"])).name
        if not filename:
            raise ValueError("camera-motion audit row has no filename or video")
        if filename in output:
            raise ValueError(f"duplicate camera-motion audit filename: {filename}")
        output[str(filename)] = row
    return output


def _ensure_audit(args: argparse.Namespace) -> Path:
    if args.audit_jsonl is not None:
        path = args.audit_jsonl.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    audit_dir = args.output_root.resolve() / "camera_motion_audit"
    audit_path = audit_dir / "camera_motion_audit.jsonl"
    if audit_path.is_file() and not args.rerun:
        return audit_path
    command = [
        sys.executable,
        str(CODE_ROOT / "scripts" / "audit_camera_motion.py"),
        "--videos",
        str(args.videos.resolve()),
        "--output",
        str(audit_dir),
        "--workers",
        str(max(1, args.audit_workers)),
        "--stride",
        str(max(1, args.audit_stride)),
        "--no-contact-sheets",
    ]
    subprocess.run(command, check=True)
    return audit_path


def _resolve_video(job: dict[str, Any], videos_root: Path) -> Path:
    expected_name = Path(str(job["video"])).name
    direct = videos_root / expected_name
    if direct.is_file():
        return direct.resolve()
    matches = list(videos_root.rglob(expected_name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {expected_name} under {videos_root}, found {len(matches)}"
        )
    return matches[0].resolve()


def _write_routing_manifest(path: Path, routes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in routes:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _normalize_static(source: Path, target: Path, route: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_index",
        "source_frame",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "valid",
        "route",
        "geometry_source",
        "object_radius_px",
        "expected_radius_px",
        "sphere_radius_ratio",
        "quality_value",
    ]
    with source.open(encoding="utf-8-sig", newline="") as src, target.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            valid = _is_true(row.get("measurement_valid"))
            writer.writerow(
                {
                    "sample_index": row.get("frame_index"),
                    "source_frame": row.get("frame_index"),
                    "time_s": row.get("time_s"),
                    "x_m": row.get("x_m") if valid else "",
                    "y_m": row.get("y_m") if valid else "",
                    "z_m": row.get("z_m") if valid else "",
                    "valid": str(valid).lower(),
                    "route": route["route"],
                    "geometry_source": row.get("geometry_source"),
                    "object_radius_px": row.get("measurement_radius_px"),
                    "expected_radius_px": row.get("expected_radius_px"),
                    "sphere_radius_ratio": row.get("sphere_radius_ratio"),
                    "quality_value": row.get("z_sigma_m"),
                }
            )


def _normalize_dynamic(source: Path, target: Path, route: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_index",
        "source_frame",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "valid",
        "route",
        "geometry_source",
        "object_radius_px",
        "expected_radius_px",
        "sphere_radius_ratio",
        "quality_value",
    ]
    with source.open(encoding="utf-8-sig", newline="") as src, target.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            xyz = [row.get("x_m"), row.get("y_m"), row.get("z_m")]
            valid = all(value not in {None, ""} for value in xyz)
            writer.writerow(
                {
                    "sample_index": row.get("sample_index"),
                    "source_frame": row.get("source_frame"),
                    "time_s": row.get("time_s"),
                    "x_m": row.get("x_m"),
                    "y_m": row.get("y_m"),
                    "z_m": row.get("z_m"),
                    "valid": str(valid).lower(),
                    "route": route["route"],
                    "geometry_source": "spatialtrackerv2_sim3_static_anchors",
                    "object_radius_px": "",
                    "expected_radius_px": "",
                    "sphere_radius_ratio": "",
                    "quality_value": row.get("object_rigid_rmse_m"),
                }
            )


def _run_dynamic(
    args: argparse.Namespace,
    routes: list[dict[str, Any]],
    output_root: Path,
    runtime_manifest: Path,
) -> None:
    selected = [row["job_id"] for row in routes if row["route"] == "spatialtrackerv2_dynamic"]
    if not selected:
        return
    command = [
        sys.executable,
        str(HERE / "run_batch.py"),
        "--manifest",
        str(runtime_manifest),
        "--output-root",
        str(output_root / "dynamic"),
        "--work-root",
        str(output_root / "work"),
        "--frame-stride",
        str(args.dynamic_frame_stride),
    ]
    for job_id in selected:
        command.extend(["--job-id", job_id])
    if args.isolated_process:
        command.append("--isolated-process")
    if args.rerun:
        command.append("--rerun")
    subprocess.run(command, check=True, cwd=PACKAGE_ROOT)


def _result_status(path: Path, route: str) -> tuple[str, bool | None, str | None]:
    if not path.is_file():
        return "missing", None, "result.json not found"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", None, str(exc)
    if route == "spatialtrackerv2_dynamic":
        return str(result.get("status") or "unknown"), result.get("quality_pass"), result.get("error")
    if result.get("status") == "failed":
        return "failed", False, result.get("error")
    validity = result.get("reconstruction_validity", {})
    return "succeeded", validity.get("airborne_metric_trajectory_valid"), None


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit_path = _ensure_audit(args)
    audit_index = _index_audit(_read_jsonl(audit_path))
    manifest = _read_jsonl(args.manifest.resolve())
    runtime_manifest_rows: list[dict[str, Any]] = []
    for job in manifest:
        runtime_job = dict(job)
        try:
            runtime_job["video"] = str(_resolve_video(job, args.videos.resolve()))
        except FileNotFoundError:
            # Selection below still fails immediately for a requested missing
            # video; retaining the original path keeps unrelated manifest rows
            # available to run_batch's complete 3-view summary logic.
            pass
        runtime_manifest_rows.append(runtime_job)
    runtime_manifest = output_root / "runtime_manifest.jsonl"
    _write_routing_manifest(runtime_manifest, runtime_manifest_rows)
    selected_ids = set(args.job_id)
    selected_jobs = [job for job in manifest if not selected_ids or job["job_id"] in selected_ids]
    unknown = selected_ids - {job["job_id"] for job in selected_jobs}
    if unknown:
        raise ValueError(f"unknown job ids: {sorted(unknown)}")
    if args.max_jobs is not None:
        selected_jobs = selected_jobs[: max(0, args.max_jobs)]
    if not selected_jobs:
        raise ValueError("no V1A jobs selected")

    routes: list[dict[str, Any]] = []
    evidence_by_job: dict[str, dict[str, Any] | None] = {}
    video_by_job: dict[str, Path] = {}
    for job in selected_jobs:
        video = _resolve_video(job, args.videos.resolve())
        evidence = _compact_evidence(audit_index.get(video.name))
        decision = choose_reconstruction_route(evidence, video.name)
        route = {
            "job_id": job["job_id"],
            "video": str(video),
            "scene": job["scene"],
            "camera": job["camera"],
            **decision,
            "camera_motion_evidence": evidence,
            "camera_motion_audit": str(audit_path),
        }
        routes.append(route)
        evidence_by_job[job["job_id"]] = evidence
        video_by_job[job["job_id"]] = video

    routing_path = output_root / "routing_manifest.jsonl"
    _write_routing_manifest(routing_path, routes)
    counts: dict[str, int] = {}
    for route in routes:
        counts[route["route"]] = counts.get(route["route"], 0) + 1
    write_json(
        output_root / "routing_summary.json",
        {
            "schema_version": "1.0.0",
            "jobs": len(routes),
            "route_counts": counts,
            "policy": {
                "confirmed_fixed": "calibrated_static_sphere",
                "changed_review_missing_or_error": "spatialtrackerv2_dynamic",
            },
            "routing_manifest": str(routing_path),
        },
    )
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True), flush=True)
    if args.dry_run:
        return

    # Keep route-only audits runnable on CPU machines without OpenCV/Matplotlib.
    # Heavy calibrated reconstruction dependencies are imported only when a
    # reconstruction is actually requested.
    from remake_benchmark.reconstruction.v1a_metric import (
        load_v1a_config,
        parse_v1a_video_job,
        run_v1a_metric_job,
    )

    config = load_v1a_config(args.config.resolve())
    overlay_remaining = max(0, args.overlay_count)
    for route in routes:
        if route["route"] != "calibrated_static_sphere":
            continue
        job_id = route["job_id"]
        result_path = output_root / "static" / job_id / "result.json"
        if result_path.is_file() and not args.rerun:
            print(f"STATIC SKIP {job_id}", flush=True)
            continue
        print(f"STATIC START {job_id}", flush=True)
        try:
            metric_job = parse_v1a_video_job(video_by_job[job_id])
            result = run_v1a_metric_job(
                metric_job,
                args.workspace_root.resolve(),
                args.calibration_root.resolve(),
                output_root / "static",
                config=deepcopy(config),
                make_overlay=overlay_remaining > 0,
                camera_motion_evidence=evidence_by_job[job_id],
            )
            result["hybrid_routing"] = route
            write_json(result_path, result)
            if overlay_remaining > 0:
                overlay_remaining -= 1
            print(
                f"STATIC DONE {job_id} valid={result['reconstruction_validity']['airborne_metric_trajectory_valid']}",
                flush=True,
            )
        except Exception as exc:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                result_path,
                {
                    "status": "failed",
                    "job_id": job_id,
                    "hybrid_routing": route,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"STATIC ERROR {job_id}: {exc}", file=sys.stderr, flush=True)

    _run_dynamic(args, routes, output_root, runtime_manifest)

    summary_rows: list[dict[str, Any]] = []
    for route in routes:
        job_id = route["job_id"]
        native_root = output_root / (
            "static" if route["route"] == "calibrated_static_sphere" else "dynamic"
        )
        native_dir = native_root / job_id
        source_csv = native_dir / (
            "trajectory_metric.csv"
            if route["route"] == "calibrated_static_sphere"
            else "trajectory_world.csv"
        )
        normalized = output_root / "normalized" / job_id / "trajectory_world.csv"
        if source_csv.is_file():
            if route["route"] == "calibrated_static_sphere":
                _normalize_static(source_csv, normalized, route)
            else:
                _normalize_dynamic(source_csv, normalized, route)
        write_json(
            output_root / "normalized" / job_id / "route.json",
            {
                **route,
                "native_output_dir": str(native_dir),
                "normalized_trajectory": str(normalized) if normalized.is_file() else None,
            },
        )
        status, quality_pass, error = _result_status(native_dir / "result.json", route["route"])
        summary_rows.append(
            {
                "job_id": job_id,
                "scene": route["scene"],
                "camera": route["camera"],
                "route": route["route"],
                "camera_motion_category": route["camera_motion_category"],
                "status": status,
                "quality_pass": quality_pass,
                "normalized_trajectory": str(normalized) if normalized.is_file() else None,
                "error": error,
            }
        )
    with (output_root / "hybrid_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(
        output_root / "hybrid_summary.json",
        {"jobs": len(summary_rows), "route_counts": counts, "rows": summary_rows},
    )
    print(f"summary: {output_root / 'hybrid_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
