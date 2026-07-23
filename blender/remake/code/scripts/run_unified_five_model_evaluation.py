#!/usr/bin/env python3
"""Evaluate five frozen trajectory sets with one contract and build one report.

The command deliberately starts from ``tracks/jobs/<job_id>`` rather than from
videos.  It therefore never reruns object tracking or SpaTrackerV2.  A report
is built only after all models contain the exact frozen manifest and every
result has the same evaluator/registry/fitter/gate lineage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
REMAKE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

FROZEN_EVALUATOR_VERSION = "1.3.0"
EXPECTED_MODEL_COUNT = 5
COMMON_LINEAGE_FIELDS = (
    "registry_sha256",
    "evaluator_source_sha256",
    "physics_fitter_source_sha256",
    "dynamic_3d_gate_source_sha256",
    "evaluator_version",
)
PER_VIDEO_LINEAGE_FIELDS = (
    "trajectory_sha256",
    "track_result_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_manifest_job_ids(path: Path) -> list[str]:
    job_ids: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        job_id = row.get("job_id") if isinstance(row, Mapping) else None
        if not job_id:
            raise ValueError(f"missing job_id at {path}:{line_number}")
        job_ids.append(str(job_id))
    duplicate = sorted(
        value for value, count in Counter(job_ids).items() if count > 1
    )
    if duplicate:
        raise ValueError(f"duplicate manifest job_id(s): {duplicate[:5]}")
    return job_ids


def _parse_model_specs(values: Sequence[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be NAME=TRACKS_ROOT, got: {value}")
        name, path_text = value.split("=", 1)
        name = name.strip()
        if not name or not path_text.strip():
            raise ValueError(f"--model must be NAME=TRACKS_ROOT, got: {value}")
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        models[name] = Path(path_text).expanduser().resolve()
    return models


def _slug(value: str) -> str:
    output = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not output:
        raise ValueError(f"cannot derive output directory from model name: {value!r}")
    return output


def _evaluation_roots(models: Mapping[str, Path], output_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    used: dict[str, str] = {}
    for name in models:
        slug = _slug(name)
        if slug in used:
            raise ValueError(
                f"model names {used[slug]!r} and {name!r} produce the same slug {slug!r}"
            )
        used[slug] = name
        roots[name] = output_root / "models" / slug / "evaluation"
    return roots


def validate_unified_lineage(
    evaluations: Mapping[str, Path],
    *,
    manifest_path: Path,
    registry_path: Path,
    expected_version: str = FROZEN_EVALUATOR_VERSION,
) -> dict[str, Any]:
    """Reject incomplete, mixed-version, or mixed-source model evaluations."""

    expected_job_ids = _read_manifest_job_ids(manifest_path)
    expected_set = set(expected_job_ids)
    expected_registry_hash = _sha256(registry_path)
    signatures: set[tuple[str, ...]] = set()
    model_rows: list[dict[str, Any]] = []

    for model, evaluation_root in evaluations.items():
        result_paths = sorted((evaluation_root / "jobs").glob("*/result.json"))
        actual_ids = {path.parent.name for path in result_paths}
        missing = sorted(expected_set - actual_ids)
        unexpected = sorted(actual_ids - expected_set)
        if missing or unexpected or len(result_paths) != len(expected_job_ids):
            raise RuntimeError(
                f"{model}: evaluation inventory mismatch; "
                f"results={len(result_paths)} expected={len(expected_job_ids)} "
                f"missing={len(missing)} unexpected={len(unexpected)}; "
                f"examples missing={missing[:3]} unexpected={unexpected[:3]}"
            )

        per_model_signatures: set[tuple[str, ...]] = set()
        missing_lineage: list[str] = []
        bad_execution_contract: list[str] = []
        missing_portable_trajectory: list[str] = []
        for path in result_paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            lineage = result.get("evaluation_lineage")
            if not isinstance(lineage, Mapping):
                missing_lineage.append(path.parent.name)
                continue
            absent = [
                field
                for field in (*COMMON_LINEAGE_FIELDS, *PER_VIDEO_LINEAGE_FIELDS)
                if not lineage.get(field)
            ]
            if absent:
                missing_lineage.append(f"{path.parent.name}:{','.join(absent)}")
                continue
            signature = tuple(str(lineage[field]) for field in COMMON_LINEAGE_FIELDS)
            per_model_signatures.add(signature)
            if (
                result.get("evaluation_reads_video") is not False
                or result.get("evaluation_invokes_tracker") is not False
            ):
                bad_execution_contract.append(path.parent.name)
            if not (path.parent / "trajectory_frames.csv").is_file():
                missing_portable_trajectory.append(path.parent.name)

        if missing_lineage:
            raise RuntimeError(
                f"{model}: missing/incomplete evaluation lineage in "
                f"{len(missing_lineage)} result(s); first={missing_lineage[:3]}"
            )
        if len(per_model_signatures) != 1:
            raise RuntimeError(
                f"{model}: mixed evaluator lineage ({len(per_model_signatures)} signatures)"
            )
        if bad_execution_contract:
            raise RuntimeError(
                f"{model}: {len(bad_execution_contract)} result(s) do not declare "
                "trajectory-only evaluation"
            )
        if missing_portable_trajectory:
            raise RuntimeError(
                f"{model}: {len(missing_portable_trajectory)} portable trajectory "
                f"CSV(s) missing; first={missing_portable_trajectory[:3]}"
            )

        signature = next(iter(per_model_signatures))
        signatures.add(signature)
        signature_map = dict(zip(COMMON_LINEAGE_FIELDS, signature))
        if signature_map["evaluator_version"] != expected_version:
            raise RuntimeError(
                f"{model}: evaluator_version={signature_map['evaluator_version']!r}, "
                f"expected {expected_version!r}"
            )
        if signature_map["registry_sha256"] != expected_registry_hash:
            raise RuntimeError(
                f"{model}: registry hash does not match {registry_path}"
            )

        metadata_path = evaluation_root / "evaluation_metadata_all.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"{model}: missing {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("phase") != "all"
            or int(metadata.get("selected_jobs", -1)) != len(expected_job_ids)
            or metadata.get("evaluation_reads_video") is not False
            or metadata.get("evaluation_invokes_tracker") is not False
        ):
            raise RuntimeError(f"{model}: invalid phase/all evaluation metadata")

        model_rows.append(
            {
                "model": model,
                "evaluation_root": str(evaluation_root),
                "result_count": len(result_paths),
                "phase": "all",
                **signature_map,
            }
        )

    if len(signatures) != 1:
        raise RuntimeError(
            "models were not produced by one evaluation contract: "
            f"{len(signatures)} common-lineage signatures found"
        )

    common_signature = dict(
        zip(COMMON_LINEAGE_FIELDS, next(iter(signatures)))
    )
    return {
        "schema_version": "1.0.0",
        "status": "passed",
        "model_count": len(evaluations),
        "manifest_job_count": len(expected_job_ids),
        "phase": "all",
        "common_lineage": common_signature,
        "per_video_lineage_fields": list(PER_VIDEO_LINEAGE_FIELDS),
        "models": model_rows,
    }


def _run_report_command(
    script: Path,
    evaluations: Mapping[str, Path],
    *,
    extra: Sequence[str],
) -> None:
    command = [sys.executable, str(script)]
    for model, root in evaluations.items():
        command.extend(["--model", f"{model}={root}"])
    command.extend(extra)
    subprocess.run(command, check=True, cwd=REMAKE_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=TRACKS_ROOT",
        help="Repeat exactly five times. The name is retained in all tables.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CODE_ROOT / "assets" / "seedance978_evaluation" / "manifest.jsonl",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=CODE_ROOT
        / "assets"
        / "seedance978_evaluation"
        / "experiment_registry.json",
    )
    parser.add_argument(
        "--phase",
        choices=("all",),
        default="all",
        help="Frozen to all so all five models use the same 978-job scope.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Import the OpenCV-dependent evaluator only for a real run so --help and
    # the pure lineage audit remain usable in a lightweight local environment.
    from remake_benchmark.reconstruction.trajectory_pipeline import (  # noqa: E402
        EVALUATOR_VERSION,
        evaluate_extracted_seedance978,
    )

    if EVALUATOR_VERSION != FROZEN_EVALUATOR_VERSION:
        parser.error(
            "this entry point freezes evaluator version "
            f"{FROZEN_EVALUATOR_VERSION}, but the checkout contains "
            f"{EVALUATOR_VERSION}"
        )
    models = _parse_model_specs(args.model)
    if len(models) != EXPECTED_MODEL_COUNT:
        parser.error(
            f"expected {EXPECTED_MODEL_COUNT} models, received {len(models)}"
        )
    missing_tracks = [
        f"{name}={path}"
        for name, path in models.items()
        if not (path / "jobs").is_dir()
    ]
    if missing_tracks:
        parser.error(
            "missing standardized tracks/jobs directory: "
            + ", ".join(missing_tracks)
        )

    output_root = args.output_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    evaluations = _evaluation_roots(models, output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for ordinal, (model, tracks_root) in enumerate(models.items(), 1):
        print(
            f"=== [{ordinal}/{len(models)}] {model}: trajectory-only evaluation ===",
            flush=True,
        )
        evaluate_extracted_seedance978(
            extraction_root=tracks_root,
            manifest_path=manifest_path,
            registry_path=registry_path,
            output_root=evaluations[model],
            phase="all",
            overwrite=args.overwrite,
        )

    audit = validate_unified_lineage(
        evaluations,
        manifest_path=manifest_path,
        registry_path=registry_path,
    )
    audit_path = output_root / "lineage_audit.json"
    _write_json(audit_path, audit)
    print(f"LINEAGE PASS: {audit_path}", flush=True)

    report_root = output_root / "report"
    _run_report_command(
        CODE_ROOT / "scripts" / "build_candidate_physics_report.py",
        evaluations,
        extra=[
            "--registry",
            str(registry_path),
            "--output",
            str(report_root),
        ],
    )
    evidence_root = output_root / "experiment_evidence"
    _run_report_command(
        CODE_ROOT / "scripts" / "build_experiment_evidence_report.py",
        evaluations,
        extra=[
            "--scan-csv",
            str(report_root / "scan_response_channels.csv"),
            "--metric-table",
            str(report_root / "paper_main_table.csv"),
            "--output",
            str(evidence_root),
        ],
    )

    summary = {
        "schema_version": "1.0.0",
        "status": "completed",
        "phase": "all",
        "evaluator_version": FROZEN_EVALUATOR_VERSION,
        "manifest": str(manifest_path),
        "registry": str(registry_path),
        "tracks": {name: str(path) for name, path in models.items()},
        "evaluations": {name: str(path) for name, path in evaluations.items()},
        "lineage_audit": str(audit_path),
        "report": str(report_root),
        "experiment_evidence": str(evidence_root),
    }
    _write_json(output_root / "unified_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
