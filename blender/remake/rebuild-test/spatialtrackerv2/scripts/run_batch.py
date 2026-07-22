#!/usr/bin/env python3
"""Sequential, resumable SpatialTrackerV2 runner for benchmark manifests."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from postprocess_result import postprocess
from prepare_queries import prepare_queries


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REMAKE_ROOT = PACKAGE_ROOT.parents[1]
UPSTREAM = Path(
    os.environ.get(
        "SPATIALTRACKERV2_ROOT",
        str(PACKAGE_ROOT / "upstream" / "SpaTrackerV2"),
    )
).expanduser().resolve()
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "v1a_seedance27.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs" / "v1a_seedance27"
DEFAULT_WORK = PACKAGE_ROOT / "work" / "v1a_seedance27"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--track-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--object-points", type=int, default=64)
    parser.add_argument("--anchor-points", type=int, default=128)
    parser.add_argument("--align-threshold-m", type=float, default=0.30)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--isolated-process",
        action="store_true",
        help="Reload models for each job; slower, but useful to isolate a crashing video",
    )
    parser.add_argument("--job-id", action="append", default=[], help="Run only an exact job id; repeatable")
    return parser.parse_args()


def resolve_remake(relative: str) -> Path:
    return REMAKE_ROOT / Path(relative)


def successful_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        required = [
            path.parent / "raw_spatialtrackerv2.npz",
            path.parent / "trajectory_world.csv",
            path.parent / "object_track_overlay.mp4",
        ]
        return result.get("status") == "succeeded" and all(item.is_file() for item in required)
    except (OSError, json.JSONDecodeError):
        return False


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def upstream_git_state() -> dict[str, object]:
    if not (UPSTREAM / ".git").exists():
        return {"root": str(UPSTREAM), "git_commit": None, "git_dirty": None}
    try:
        git_prefix = ["git", "-c", f"safe.directory={UPSTREAM}", "-C", str(UPSTREAM)]
        commit = subprocess.check_output(
            [*git_prefix, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                [*git_prefix, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"root": str(UPSTREAM), "git_commit": commit, "git_dirty": dirty}


def write_failed_result(result_path: Path, job: dict, error: str, elapsed: float) -> None:
    write_json_atomic(
        result_path,
        {
            "status": "failed",
            "job_id": job["job_id"],
            "error": error,
            "elapsed_seconds": elapsed,
        },
    )


def refresh_summary(manifest: list[dict], output_root: Path) -> None:
    rows = []
    for job in manifest:
        result_path = output_root / job["job_id"] / "result.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result = {"status": "invalid_result_json"}
        else:
            result = {"status": "pending"}
        rows.append(
            {
                "job_id": job["job_id"],
                "experiment_id": job.get("experiment_id"),
                "parameter_tuple_id": job.get("parameter_tuple_id", job.get("factor_token")),
                "scene": job["scene"],
                "camera": job["camera"],
                "seed": job.get("seed"),
                "status": result.get("status"),
                "quality_pass": result.get("quality_pass"),
                "trajectory_valid_fraction": result.get("trajectory_valid_fraction"),
                "alignment_direct_fraction": result.get("alignment_direct_fraction"),
                "median_anchor_rmse_m": result.get("median_anchor_rmse_m"),
                "median_object_rigid_rmse_m": result.get("median_object_rigid_rmse_m"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "error": result.get("error"),
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    condition_rows = []
    condition_keys = dict.fromkeys(
        (
            job.get("experiment_id"),
            job.get("parameter_tuple_id", job.get("factor_token")),
            job["scene"],
            job.get("seed"),
        )
        for job in manifest
    )
    for experiment_id, parameter_tuple_id, scene, seed in condition_keys:
        group = [
            row
            for row in rows
            if (
                row["experiment_id"],
                row["parameter_tuple_id"],
                row["scene"],
                row["seed"],
            )
            == (experiment_id, parameter_tuple_id, scene, seed)
        ]
        status_by_camera = {row["camera"]: row["status"] for row in group}
        condition_rows.append(
            {
                "experiment_id": experiment_id,
                "parameter_tuple_id": parameter_tuple_id,
                "scene": scene,
                "seed": seed,
                "succeeded_views": sum(row["status"] == "succeeded" for row in group),
                "quality_pass_views": sum(row["quality_pass"] is True for row in group),
                "CAM_Main_status": status_by_camera.get("CAM_Main"),
                "CAM_Side_status": status_by_camera.get("CAM_Side"),
                "CAM_Top_status": status_by_camera.get("CAM_Top"),
            }
        )
    with (output_root / "summary_by_condition.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
    # Keep the legacy filename for the original 27-job workflow.
    with (output_root / "summary_by_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "jobs": len(rows),
                "status_counts": counts,
                "condition_rows": condition_rows,
                "scene_rows": condition_rows,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if not args.manifest.is_file():
        subprocess.run([sys.executable, str(HERE / "build_manifest.py"), "--strict"], check=True)
    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected_ids = set(args.job_id)
    selected = [job for job in manifest if not selected_ids or job["job_id"] in selected_ids]
    if selected_ids - {job["job_id"] for job in selected}:
        raise ValueError(f"Unknown job ids: {sorted(selected_ids - {job['job_id'] for job in selected})}")

    model_cache = PACKAGE_ROOT / "models" / "huggingface"
    model_cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(model_cache))
    environment.setdefault("HUGGINGFACE_HUB_CACHE", str(model_cache / "hub"))
    environment["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("HF_HOME", environment["HF_HOME"])
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", environment["HUGGINGFACE_HUB_CACHE"])

    attempted = 0
    session = None
    upstream_state = upstream_git_state()
    for ordinal, job in enumerate(selected, start=1):
        output_dir = args.output_root / job["job_id"]
        result_path = output_dir / "result.json"
        if not args.rerun and successful_result(result_path):
            print(f"[{ordinal}/{len(selected)}] SKIP {job['job_id']}")
            continue
        if args.max_jobs is not None and attempted >= args.max_jobs:
            break
        attempted += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = args.work_root / job["job_id"]
        work_dir.mkdir(parents=True, exist_ok=True)
        video = resolve_remake(job["video"])
        first_frame = resolve_remake(job["first_frame"])
        calibration = resolve_remake(job["calibration"])
        queries_path = work_dir / "queries.npz"
        start = time.perf_counter()
        print(f"[{ordinal}/{len(selected)}] START {job['job_id']}", flush=True)
        try:
            prepare_queries(
                video,
                first_frame,
                calibration,
                queries_path,
                output_dir / "queries_frame0.png",
                args.object_points,
                args.anchor_points,
            )
            log_path = output_dir / "spatialtrackerv2.log"
            raw_path = output_dir / "raw_spatialtrackerv2.npz"
            if raw_path.exists():
                raw_path.unlink()
            if args.isolated_process:
                command = [
                    sys.executable,
                    str(HERE / "run_single_inference.py"),
                    "--video",
                    str(video),
                    "--queries",
                    str(queries_path),
                    "--output",
                    str(raw_path),
                    "--source-fps",
                    str(job["source_fps"]),
                    "--frame-stride",
                    str(args.frame_stride),
                    "--track-mode",
                    args.track_mode,
                    "--vo-points",
                    str(max(256, args.object_points + args.anchor_points)),
                ]
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=UPSTREAM,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if process.returncode != 0:
                    raise RuntimeError(f"SpatialTrackerV2 exited with code {process.returncode}; see {log_path}")
                if not raw_path.is_file():
                    raise FileNotFoundError(f"Missing isolated inference result: {raw_path}")
            else:
                if session is None:
                    from spatialtracker_session import SpatialTrackerSession

                    session = SpatialTrackerSession(
                        args.track_mode,
                        max(256, args.object_points + args.anchor_points),
                    )
                with (
                    log_path.open("w", encoding="utf-8") as log,
                    contextlib.redirect_stdout(log),
                    contextlib.redirect_stderr(log),
                ):
                    session.run(
                        video,
                        queries_path,
                        raw_path,
                        float(job["source_fps"]),
                        args.frame_stride,
                    )
            result = postprocess(raw_path, video, output_dir, args.align_threshold_m)
            elapsed = time.perf_counter() - start
            result.update(
                {
                    "job_id": job["job_id"],
                    "scene": job["scene"],
                    "camera": job["camera"],
                    "frame_stride": args.frame_stride,
                    "elapsed_seconds": elapsed,
                    "ground_truth_comparison": False,
                    "spatialtrackerv2_upstream": upstream_state,
                }
            )
            write_json_atomic(result_path, result)
            print(
                f"[{ordinal}/{len(selected)}] DONE {job['job_id']} "
                f"quality_pass={result['quality_pass']} elapsed={elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            with (output_dir / "spatialtrackerv2.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            write_failed_result(result_path, job, f"{type(exc).__name__}: {exc}", elapsed)
            print(f"[{ordinal}/{len(selected)}] ERROR {job['job_id']}: {exc}", file=sys.stderr, flush=True)
        refresh_summary(manifest, args.output_root)

    refresh_summary(manifest, args.output_root)
    print(f"summary: {args.output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
