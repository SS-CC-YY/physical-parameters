#!/usr/bin/env python3
"""Compact the frozen 978 generation manifest and camera audit for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPECTED_JOBS = 978
MANIFEST_FIELDS = {"job_id", "experiment_id", "seed", "inputs", "factors"}
AUDIT_FIELDS = {
    "filename",
    "status",
    "decision",
    "final_category",
    "reasons",
    "review_note",
    "measurement_source",
    "width",
    "height",
    "fps",
    "reported_frames",
    "decoded_frames",
    "sample_count",
    "valid_pair_fraction",
    "median_inlier_ratio",
    "median_residual_px",
    "max_direct_translation_px",
    "max_direct_rotation_deg",
    "max_direct_scale_change",
    "direct_translation_hit_count",
    "direct_rotation_hit_count",
    "direct_scale_hit_count",
    "direct_motion_cluster_max",
    "cut_pair_count",
    "cut_pairs",
    "error",
}


def _read(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            output.append(value)
    return output


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def build(manifest: Path, audit: Path, output_dir: Path) -> dict[str, Any]:
    manifest_rows = _read(manifest)
    audit_rows = _read(audit)
    compact_manifest = []
    for row in manifest_rows:
        selected = {key: row[key] for key in MANIFEST_FIELDS}
        selected["inputs"] = {
            key: row["inputs"][key]
            for key in ("image", "source_fps", "conditioning_frame_index")
            if key in row["inputs"]
        }
        selected["factors"] = {
            key: row["factors"][key]
            for key in ("parameter_tuple_id", "scene_id", "object_id", "camera")
        }
        compact_manifest.append(selected)
    compact_audit = []
    for row in audit_rows:
        selected = {key: row[key] for key in AUDIT_FIELDS if key in row}
        if "filename" not in selected and row.get("video"):
            selected["filename"] = Path(str(row["video"])).name
        compact_audit.append(selected)
    manifest_ids = [str(row["job_id"]) for row in compact_manifest]
    audit_ids = [Path(str(row["filename"])).stem for row in compact_audit]
    if len(manifest_ids) != EXPECTED_JOBS or len(set(manifest_ids)) != EXPECTED_JOBS:
        raise ValueError(f"expected {EXPECTED_JOBS} unique manifest jobs, got {len(manifest_ids)}")
    if set(manifest_ids) != set(audit_ids):
        raise ValueError(
            f"manifest/audit mismatch: missing={len(set(manifest_ids)-set(audit_ids))} "
            f"extra={len(set(audit_ids)-set(manifest_ids))}"
        )
    _write(output_dir / "manifest.jsonl", compact_manifest)
    _write(output_dir / "camera_motion.jsonl", compact_audit)
    summary = {
        "schema_version": "1.0.0",
        "job_count": EXPECTED_JOBS,
        "manifest_source": str(manifest),
        "camera_audit_source": str(audit),
        "files": ["manifest.jsonl", "camera_motion.jsonl"],
        "note": "Compact immutable inputs for the Seedance 978 evaluation; generated videos are not duplicated.",
    }
    (output_dir / "README.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.camera_audit, args.output), indent=2))


if __name__ == "__main__":
    main()
