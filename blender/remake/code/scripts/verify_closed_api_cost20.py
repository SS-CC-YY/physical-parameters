#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MATCH_FIELDS = (
    "job_id",
    "experiment_id",
    "task_type",
    "case_id",
    "repeat_id",
    "seed",
    "factors",
    "targets",
    "units",
    "inputs",
    "prompt",
    "negative_prompt",
    "generation",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_and_export(
    seedance_manifest: Path,
    kling_manifest: Path,
    workspace_root: Path,
    output_dir: Path,
    expected_jobs: int,
) -> dict[str, Any]:
    seedance_jobs = _read_jsonl(seedance_manifest)
    kling_jobs = _read_jsonl(kling_manifest)
    if len(seedance_jobs) != expected_jobs or len(kling_jobs) != expected_jobs:
        raise ValueError(
            f"expected exactly {expected_jobs} jobs per provider, got "
            f"Seedance={len(seedance_jobs)}, Kling={len(kling_jobs)}"
        )
    if len({job.get("job_id") for job in seedance_jobs}) != expected_jobs:
        raise ValueError("Seedance manifest contains duplicate job_id values")
    if len({job.get("job_id") for job in kling_jobs}) != expected_jobs:
        raise ValueError("Kling manifest contains duplicate job_id values")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []
    for index, (seedance_job, kling_job) in enumerate(zip(seedance_jobs, kling_jobs), 1):
        mismatches = [field for field in MATCH_FIELDS if seedance_job.get(field) != kling_job.get(field)]
        if mismatches:
            raise ValueError(f"job {index} is not paired across providers; mismatched fields: {mismatches}")
        image_reference = Path(str(seedance_job["inputs"]["image"]))
        image_path = image_reference if image_reference.is_absolute() else workspace_root / image_reference
        if not image_path.is_file():
            raise FileNotFoundError(f"missing first frame for job {index}: {image_path}")
        factors = seedance_job.get("factors", {})
        report_rows.append(
            {
                "index": index,
                "job_id": seedance_job["job_id"],
                "scene_id": factors.get("scene_id"),
                "object_id": factors.get("object_id"),
                "camera": factors.get("camera"),
                "seed": seedance_job.get("seed"),
                "targets_json": json.dumps(seedance_job.get("targets", {}), ensure_ascii=False, sort_keys=True),
                "first_frame": str(image_reference).replace("\\", "/"),
                "first_frame_sha256": _sha256(image_path),
                "prompt": seedance_job.get("prompt", ""),
                "negative_prompt": seedance_job.get("negative_prompt", ""),
            }
        )

    csv_path = output_dir / "selected_20_jobs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    jsonl_path = output_dir / "selected_20_jobs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in report_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    scenes = sorted({str(row["scene_id"]) for row in report_rows})
    objects = sorted({str(row["object_id"]) for row in report_rows})
    cameras = sorted({str(row["camera"]) for row in report_rows})
    summary = {
        "schema_version": "1.0.0",
        "paired": True,
        "jobs_per_provider": expected_jobs,
        "total_billable_tasks_planned": expected_jobs * 2,
        "scenes": scenes,
        "objects": objects,
        "cameras": cameras,
        "seedance_manifest": str(seedance_manifest.resolve()),
        "kling_manifest": str(kling_manifest.resolve()),
        "csv": str(csv_path.resolve()),
        "jsonl": str(jsonl_path.resolve()),
    }
    summary_path = output_dir / "selected_20_jobs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and export the paired Seedance/Kling 20-job manifests")
    parser.add_argument("--seedance-manifest", type=Path, required=True)
    parser.add_argument("--kling-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-jobs", type=int, default=20)
    args = parser.parse_args()
    if args.expected_jobs <= 0:
        raise SystemExit("--expected-jobs must be positive")
    summary = verify_and_export(
        args.seedance_manifest,
        args.kling_manifest,
        args.workspace_root.resolve(),
        args.output_dir.resolve(),
        args.expected_jobs,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
