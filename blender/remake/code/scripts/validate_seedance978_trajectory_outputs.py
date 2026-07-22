#!/usr/bin/env python3
"""Verify that every selected video has a full-frame CSV and overlay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.seedance978 import (  # noqa: E402
    _ordered_manifest,
    _phase_rows,
    _read_jsonl,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl",
    )
    parser.add_argument("--phase", choices=["all", "side", "main", "top", "robustness"], default="all")
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Default: <tracks>/artifact_validation_<phase>.json",
    )
    return parser


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _video_properties(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opened": False, "frames": 0, "fps": 0.0, "width": 0, "height": 0}
    frames = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.grab():
        frames += 1
    value: dict[str, float | int] = {
        "opened": True,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
    }
    cap.release()
    return value


def _validate_job(job_id: str, videos: Path, tracks: Path) -> dict[str, object]:
    errors: list[str] = []
    video_path = videos / f"{job_id}.mp4"
    job_dir = tracks / "jobs" / job_id
    result_path = job_dir / "track_result.json"
    trajectory_path = job_dir / "trajectory_frames.csv"
    overlay_path = job_dir / "object_track_overlay.mp4"
    source = _video_properties(video_path)
    overlay = _video_properties(overlay_path)
    result: dict[str, object] = {}
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_track_result:{exc}")
    else:
        errors.append("missing_track_result")
    rows: list[dict[str, str]] = []
    if trajectory_path.is_file():
        with trajectory_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        errors.append("missing_trajectory_frames_csv")
    if not source["opened"]:
        errors.append("source_video_unreadable")
    if not overlay["opened"]:
        errors.append("overlay_video_unreadable")
    source_frames = int(source["frames"])
    if source_frames != len(rows):
        errors.append(f"csv_row_count:{len(rows)}!=source_frames:{source_frames}")
    if source_frames != int(overlay["frames"]):
        errors.append(f"overlay_frames:{overlay['frames']}!=source_frames:{source_frames}")
    if source["opened"] and overlay["opened"]:
        if (source["width"], source["height"]) != (overlay["width"], overlay["height"]):
            errors.append("overlay_resolution_mismatch")
        if abs(float(source["fps"]) - float(overlay["fps"])) > 0.05:
            errors.append("overlay_fps_mismatch")
    for expected, row in enumerate(rows):
        try:
            frame_index = int(row["frame_index"])
            source_index = int(row["source_frame_index"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"frame_{expected}:invalid_frame_index")
            continue
        if frame_index != expected or source_index != expected:
            errors.append(f"frame_{expected}:non_contiguous_index")
        valid_2d = _truth(row.get("valid_2d"))
        valid_3d = _truth(row.get("valid_3d"))
        interpolated = _truth(row.get("interpolated"))
        measurement = _truth(row.get("measurement_valid"))
        fit = _truth(row.get("fit_eligible"))
        if valid_2d:
            if not (_finite(row.get("center_u_px")) and _finite(row.get("center_v_px"))):
                errors.append(f"frame_{expected}:valid_2d_without_finite_position")
            else:
                u, v = float(row["center_u_px"]), float(row["center_v_px"])
                if not (0 <= u < int(source["width"]) and 0 <= v < int(source["height"])):
                    errors.append(f"frame_{expected}:2d_position_outside_frame")
        if valid_3d and not all(_finite(row.get(name)) for name in ("x_m", "y_m", "z_m")):
            errors.append(f"frame_{expected}:valid_3d_without_finite_position")
        if fit and (not valid_3d or not measurement or interpolated):
            errors.append(f"frame_{expected}:invalid_fit_eligibility")
        if fit and not all(_finite(row.get(name)) for name in ("fit_x_m", "fit_y_m", "fit_z_m")):
            errors.append(f"frame_{expected}:fit_eligible_without_direct_fit_position")
        if not fit and any(_finite(row.get(name)) for name in ("fit_x_m", "fit_y_m", "fit_z_m")):
            errors.append(f"frame_{expected}:noneligible_frame_contains_fit_position")
    if result and result.get("target_parameters_used") is not False:
        errors.append("target_parameters_used_during_extraction")
    return {
        "job_id": job_id,
        "ok": not errors,
        "status": result.get("status"),
        "route": result.get("reconstruction_route"),
        "source_frames": source_frames,
        "csv_rows": len(rows),
        "overlay_frames": int(overlay["frames"]),
        "errors": errors,
    }


def main() -> None:
    args = _parser().parse_args()
    tracks = args.tracks.resolve()
    videos = args.videos.resolve()
    manifest = _ordered_manifest(_read_jsonl(args.manifest.resolve()))
    selected = _phase_rows(manifest, args.phase)
    if args.job_id:
        requested = set(args.job_id)
        selected = [row for row in selected if str(row["job_id"]) in requested]
        missing = requested - {str(row["job_id"]) for row in selected}
        if missing:
            raise ValueError(f"unknown/out-of-phase job ids: {sorted(missing)}")
    rows = [_validate_job(str(row["job_id"]), videos, tracks) for row in selected]
    report = {
        "schema_version": "1.0.0",
        "phase": args.phase,
        "selected_jobs": len(rows),
        "passed_jobs": sum(bool(row["ok"]) for row in rows),
        "failed_jobs": sum(not bool(row["ok"]) for row in rows),
        "all_passed": all(bool(row["ok"]) for row in rows),
        "rows": rows,
    }
    report_path = (args.report or (tracks / f"artifact_validation_{args.phase}.json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tsv_path = report_path.with_suffix(".tsv")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_id", "ok", "status", "route", "source_frames", "csv_rows", "overlay_frames", "errors"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "errors": ";".join(row["errors"])})
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"report: {report_path}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
