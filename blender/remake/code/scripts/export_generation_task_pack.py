#!/usr/bin/env python3
"""Export the frozen 978-job standard-ball generation handoff package.

The exported manifest is the benchmark contract.  Model owners may replace the
model adapter, but must not regenerate jobs, prompts, first frames, or seeds.
This script intentionally uses only the Python standard library so it can run in
the lightweight environment used to prepare a handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


PACK_ID = "standard_ball_factorized978_v1"
EXPECTED_BUILD_ID = (
    "standard_ball_all13_factorized978_random_seed_audit_wan22_i2v_generation"
)
EXPECTED_JOB_COUNT = 978
EXPECTED_IMAGE_COUNT = 351
EXPECTED_EXPERIMENTS = {
    "v1_A",
    "v1_B",
    "v1_C",
    "v1_D",
    "v2_A",
    "v2_B",
    "v2_C",
    "v2_D",
    "v2_E",
    "v3_A",
    "v3_B",
    "v3_C",
    "v3_D",
}
PRIMARY_SEED = 341867882
AUDIT_SEEDS = {1750912582, 265635392, 135883006}
EXPECTED_SEEDS = {PRIMARY_SEED, *AUDIT_SEEDS}
EXPECTED_SCENES = {
    "baseline",
    "indoor1",
    "indoor2",
    "indoor3",
    "indoor4",
    "outdoor1",
    "outdoor2",
    "outdoor3",
    "outdoor4",
}
EXPECTED_MEDIA = {"width": 832, "height": 480, "num_frames": 81, "fps": 16.0}
EXPECTED_TRACK_COUNTS = {
    "physics_identification_side": 432,
    "viewpoint_robustness_main_top": 468,
    "random_seed_stability_audit": 78,
}
EXPECTED_CAMERA_COUNTS = {"CAM_Side": 510, "CAM_Main": 234, "CAM_Top": 234}
EXPECTED_PARAMETER_TUPLE_COUNT = 48
EXPECTED_MANIFEST_SHA256 = "0239b849a8d2b2922b460d6586e3898199907fed9602dcaf84882049091f565a"
EXPECTED_SELECTION_SHA256 = "fc9735fa42c86b45c26e7ed70525d925b81850f852a7b77508b80ebe71a6d6ed"
EXPECTED_RESOLVED_BUILD_SHA256 = "32e49d12a25299663555669882580f35b997b2d9953b2d9f59cdc79d0b71eb61"

CANONICAL_TASK_FILES = [
    "manifest.jsonl",
    "manifest.selection.json",
    "first_frames_records.jsonl",
    "code/builds/standard_ball_factorized978_wan22_generation.yaml",
    "code/configs/experiments/all_experiments_full.yaml",
    "code/configs/prompts/common_physics_video_v1.yaml",
    "code/configs/prompts/all_experiments_explicit_v1.yaml",
    "code/schemas/build.schema.json",
    "code/schemas/job.schema.json",
    "code/schemas/prompt.schema.json",
]

PROVENANCE_ONLY_FILES = [
    "provenance/source_reference_resolved_build.yaml",
    "code/configs/models/wan22_i2v_a14b_full_server.yaml",
    "code/configs/evaluations/basic_video_v1.yaml",
]

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PackExportError(RuntimeError):
    """Raised when a source or destination violates the frozen pack contract."""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_existing_dir(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise PackExportError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise PackExportError(f"{label} is not a directory: {resolved}")
    return resolved


def _resolve_destination(path: Path, workspace_root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not _is_relative_to(resolved, workspace_root) or resolved == workspace_root:
        raise PackExportError(
            f"{label} must be a child of workspace root {workspace_root}: {resolved}"
        )
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or _is_relative_to(first, second) or _is_relative_to(second, first)


def _safe_relative_path(raw: str, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw.strip():
        raise PackExportError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PackExportError(f"unsafe {label}: {raw!r}")
    if path.parts[0].endswith(":"):
        raise PackExportError(f"unsafe {label}: {raw!r}")
    return path


def _source_under_workspace(workspace_root: Path, relative: PurePosixPath, label: str) -> Path:
    candidate = workspace_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PackExportError(f"missing {label}: {relative.as_posix()}") from exc
    if not _is_relative_to(resolved, workspace_root):
        raise PackExportError(f"{label} escapes workspace through a symlink: {relative}")
    if not resolved.is_file():
        raise PackExportError(f"{label} is not a file: {relative}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PackExportError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise PackExportError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _classify_track(job: dict[str, Any]) -> str:
    seed = job.get("seed")
    factors = job.get("factors")
    if not isinstance(factors, dict):
        raise PackExportError(f"job {job.get('job_id')!r} has no factors object")
    camera = factors.get("camera")
    if seed in AUDIT_SEEDS:
        return "random_seed_stability_audit"
    if seed == PRIMARY_SEED and camera == "CAM_Side":
        return "physics_identification_side"
    if seed == PRIMARY_SEED and camera in {"CAM_Main", "CAM_Top"}:
        return "viewpoint_robustness_main_top"
    raise PackExportError(
        f"job {job.get('job_id')!r} cannot be assigned to a frozen coverage track"
    )


def validate_frozen_jobs(jobs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate and summarize the exact standard-ball factorized978 contract."""

    if len(jobs) != EXPECTED_JOB_COUNT:
        raise PackExportError(f"expected {EXPECTED_JOB_COUNT} jobs, found {len(jobs)}")

    ids: list[str] = []
    images: set[str] = set()
    tracks: Counter[str] = Counter()
    cameras: Counter[str] = Counter()
    seeds: Counter[int] = Counter()
    experiments: set[str] = set()
    scenes: set[str] = set()
    parameter_tuples: set[tuple[str, str]] = set()

    for index, job in enumerate(jobs, start=1):
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise PackExportError(f"job #{index} has unsafe or missing job_id: {job_id!r}")
        ids.append(job_id)

        if job.get("build_id") != EXPECTED_BUILD_ID:
            raise PackExportError(f"job {job_id} has unexpected build_id {job.get('build_id')!r}")
        if job.get("task_type") != "i2v":
            raise PackExportError(f"job {job_id} is not an i2v task")

        experiment_id = job.get("experiment_id")
        factors = job.get("factors")
        inputs = job.get("inputs")
        if not isinstance(experiment_id, str) or not isinstance(factors, dict):
            raise PackExportError(f"job {job_id} is missing experiment/factors")
        if not isinstance(inputs, dict):
            raise PackExportError(f"job {job_id} has no inputs object")
        image_path = _safe_relative_path(inputs.get("image"), f"inputs.image for {job_id}")
        if image_path.parts[:2] != ("first_frames_v1_0", "images"):
            raise PackExportError(f"job {job_id} image is outside first_frames_v1_0/images")
        if "standard_ball" not in image_path.parts:
            raise PackExportError(f"job {job_id} does not use a standard_ball first frame")
        images.add(image_path.as_posix())

        object_id = factors.get("object_id")
        camera = factors.get("camera")
        scene = factors.get("scene_id")
        tuple_id = factors.get("parameter_tuple_id")
        if object_id != "standard_ball":
            raise PackExportError(f"job {job_id} has object_id={object_id!r}, expected standard_ball")
        if not isinstance(camera, str) or not isinstance(scene, str) or not isinstance(tuple_id, str):
            raise PackExportError(f"job {job_id} has incomplete factors")

        generation = job.get("generation")
        if not isinstance(generation, dict):
            raise PackExportError(f"job {job_id} has no generation object")
        for key, expected in EXPECTED_MEDIA.items():
            actual = generation.get(key)
            if isinstance(expected, float):
                matches = isinstance(actual, (int, float)) and float(actual) == expected
            else:
                matches = actual == expected
            if not matches:
                raise PackExportError(
                    f"job {job_id} generation.{key}={actual!r}, expected {expected!r}"
                )

        seed = job.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise PackExportError(f"job {job_id} has invalid seed {seed!r}")
        if seed not in EXPECTED_SEEDS:
            raise PackExportError(f"job {job_id} has unexpected seed {seed}")
        track = _classify_track(job)
        if track == "random_seed_stability_audit":
            if camera != "CAM_Side" or scene != "baseline":
                raise PackExportError(
                    f"audit job {job_id} must be baseline/CAM_Side, found {scene}/{camera}"
                )

        tracks[track] += 1
        cameras[camera] += 1
        seeds[seed] += 1
        experiments.add(experiment_id)
        scenes.add(scene)
        parameter_tuples.add((experiment_id, tuple_id))

    duplicate_ids = [job_id for job_id, count in Counter(ids).items() if count != 1]
    if duplicate_ids:
        raise PackExportError(f"manifest has duplicate job_id values: {duplicate_ids[:5]}")
    if len(images) != EXPECTED_IMAGE_COUNT:
        raise PackExportError(
            f"expected {EXPECTED_IMAGE_COUNT} unique input images, found {len(images)}"
        )
    if experiments != EXPECTED_EXPERIMENTS:
        raise PackExportError(
            f"experiment set differs: missing={sorted(EXPECTED_EXPERIMENTS - experiments)}, "
            f"extra={sorted(experiments - EXPECTED_EXPERIMENTS)}"
        )
    if scenes != EXPECTED_SCENES:
        raise PackExportError(
            f"scene set differs: missing={sorted(EXPECTED_SCENES - scenes)}, "
            f"extra={sorted(scenes - EXPECTED_SCENES)}"
        )
    if dict(tracks) != EXPECTED_TRACK_COUNTS:
        raise PackExportError(f"coverage track counts {dict(tracks)} != {EXPECTED_TRACK_COUNTS}")
    if dict(cameras) != EXPECTED_CAMERA_COUNTS:
        raise PackExportError(f"camera counts {dict(cameras)} != {EXPECTED_CAMERA_COUNTS}")
    if set(seeds) != EXPECTED_SEEDS:
        raise PackExportError(f"seed set {sorted(seeds)} != {sorted(EXPECTED_SEEDS)}")
    expected_seed_counts = {PRIMARY_SEED: 900, **{seed: 26 for seed in AUDIT_SEEDS}}
    if dict(seeds) != expected_seed_counts:
        raise PackExportError(f"seed counts {dict(seeds)} != {expected_seed_counts}")
    if len(parameter_tuples) != EXPECTED_PARAMETER_TUPLE_COUNT:
        raise PackExportError(
            f"expected {EXPECTED_PARAMETER_TUPLE_COUNT} experiment/tuple pairs, "
            f"found {len(parameter_tuples)}"
        )

    return {
        "jobs": len(jobs),
        "unique_job_ids": len(ids),
        "unique_input_images": len(images),
        "experiments": len(experiments),
        "scenes": len(scenes),
        "parameter_tuples": len(parameter_tuples),
        "track_counts": dict(sorted(tracks.items())),
        "camera_counts": dict(sorted(cameras.items())),
        "seed_counts": {str(key): seeds[key] for key in sorted(seeds)},
        "input_images": sorted(images),
    }


def _copy_code_snapshot(workspace_root: Path, destination: Path) -> None:
    source = workspace_root / "code"
    if not source.is_dir():
        raise PackExportError(f"missing code snapshot source: {source}")
    for item in source.rglob("*"):
        if item.is_symlink() and not _is_relative_to(item.resolve(strict=True), workspace_root):
            raise PackExportError(f"code symlink escapes workspace: {item}")

    def ignore(_directory: str, names: list[str]) -> set[str]:
        ignored = {
            name
            for name in names
            if name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
            or name.endswith((".pyc", ".pyo"))
        }
        return ignored

    shutil.copytree(source, destination / "code", ignore=ignore)


def _copy_first_frames(
    workspace_root: Path,
    destination: Path,
    image_paths: Sequence[str],
) -> dict[str, str]:
    copied_hashes: dict[str, str] = {}
    for raw in image_paths:
        relative = _safe_relative_path(raw, "manifest image")
        source = _source_under_workspace(workspace_root, relative, "first-frame image")
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_hashes[relative.as_posix()] = _sha256_file(target)
    return copied_hashes


def _filter_first_frame_records(
    workspace_root: Path,
    destination: Path,
    image_paths: Sequence[str],
    copied_hashes: dict[str, str],
) -> int | None:
    records_path = workspace_root / "first_frames_v1_0" / "records.jsonl"
    if not records_path.exists():
        return None

    needed = set(image_paths)
    selected: list[dict[str, Any]] = []
    found: set[str] = set()
    for row in _read_jsonl(records_path):
        raw = row.get("relative_path")
        if not isinstance(raw, str):
            continue
        normalized = _safe_relative_path(raw, "first-frame record relative_path").as_posix()
        full = f"first_frames_v1_0/{normalized}"
        if full not in needed:
            continue
        if full in found:
            raise PackExportError(f"duplicate first-frame record for {full}")
        recorded_hash = row.get("sha256")
        if recorded_hash and recorded_hash != copied_hashes[full]:
            raise PackExportError(
                f"first-frame SHA-256 mismatch for {full}: record={recorded_hash}, "
                f"actual={copied_hashes[full]}"
            )
        selected.append(row)
        found.add(full)
    missing = needed - found
    if missing:
        raise PackExportError(
            f"first_frames_v1_0/records.jsonl is missing {len(missing)} used images, "
            f"for example {sorted(missing)[:3]}"
        )
    _write_jsonl(destination / "first_frames_records.jsonl", selected)
    _write_jsonl(destination / "first_frames_v1_0" / "records.jsonl", selected)
    (destination / "first_frames_v1_0" / "README.md").write_text(
        "# Factorized978 standard-ball first-frame subset\n\n"
        "This task pack contains exactly the 351 PNGs referenced by the frozen "
        "978-job manifest: 13 experiments x 9 scenes x 3 cameras, for "
        "`standard_ball` only. `records.jsonl` is the filtered 351-row index.\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(selected)


def _git_commit(workspace_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"
    commit = result.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-fA-F]{40}", commit) else "unavailable"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "sha256sums.txt":
            continue
        relative = path.relative_to(root).as_posix()
        hashes[relative] = _sha256_file(path)
    return hashes


def _write_checksum_index(root: Path) -> dict[str, str]:
    hashes = _content_hashes(root)
    lines = [f"{digest}  {relative}" for relative, digest in sorted(hashes.items())]
    (root / "sha256sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return hashes


MODEL_RUNNER = r'''#!/usr/bin/env python3
"""Persistent-model runner template for one frozen generation manifest."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_model_once(config: dict[str, Any]) -> Any:
    """Load weights once, before the manifest loop.

    TODO(model owner): initialize Helios/Cosmos/LongLive/etc. here.  Never load
    model weights inside generate_one().  Do not put API keys in config files.
    """
    raise NotImplementedError("implement load_model_once() for the target model")


def generate_one(
    model: Any,
    job: dict[str, Any],
    image_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate exactly one MP4 and return provider/model metadata.

    TODO(model owner): consume job['prompt'], job['negative_prompt'], job['seed'],
    job['generation'], and image_path.  Write the original result directly to
    output_path.  If the provider lacks a negative-prompt channel, record the
    provider prompt you actually used in the returned metadata.
    """
    del model, job, image_path, output_path, config
    raise NotImplementedError("implement generate_one() for the target model")


def _safe_input(bundle_root: Path, raw: str) -> Path:
    relative = Path(raw.replace("/", str(Path('/'))))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe inputs.image: {raw!r}")
    resolved = (bundle_root / relative).resolve(strict=True)
    if bundle_root != resolved and bundle_root not in resolved.parents:
        raise ValueError(f"inputs.image escapes bundle: {raw!r}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--bundle-root", type=Path, default=default_root)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("model_config.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve(strict=True)
    manifest = (args.manifest or bundle_root / "manifest.jsonl").resolve(strict=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve(strict=False)
    videos_dir = output_dir / "videos"
    metadata_dir = output_dir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Critical: weights are loaded exactly once, outside the job loop.
    model = load_model_once(config)
    processed = 0
    with manifest.open("r", encoding="utf-8") as source, (output_dir / "run_state.jsonl").open(
        "a", encoding="utf-8", buffering=1
    ) as state:
        for line in source:
            if not line.strip():
                continue
            if args.max_jobs is not None and processed >= args.max_jobs:
                break
            job = json.loads(line)
            job_id = job["job_id"]
            if not JOB_ID_RE.fullmatch(job_id):
                raise ValueError(f"unsafe job_id: {job_id!r}")
            image_path = _safe_input(bundle_root, job["inputs"]["image"])
            output_path = videos_dir / f"{job_id}.mp4"
            started = time.time()
            provider_metadata = generate_one(model, job, image_path, output_path, config) or {}
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"generate_one did not create a non-empty MP4: {output_path}")
            metadata = {
                "job_id": job_id,
                "status": "succeeded",
                "canonical_prompt": job["prompt"],
                "canonical_negative_prompt": job.get("negative_prompt", ""),
                "requested_seed": job["seed"],
                "requested_generation": job["generation"],
                "input_image": job["inputs"]["image"],
                "output_video": f"videos/{job_id}.mp4",
                "wall_seconds": round(time.time() - started, 3),
                "model": {key: config.get(key) for key in ("model_id", "checkpoint", "revision")},
                "provider": provider_metadata,
            }
            (metadata_dir / f"{job_id}.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            state.write(json.dumps({"job_id": job_id, "status": "succeeded"}) + "\n")
            processed += 1
    print(json.dumps({"processed": processed, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


MODEL_RUNNER_README = """\
# External model runner template

This template is deliberately model-neutral. Implement only `load_model_once()`
and `generate_one()` in `run_model.py`.

Critical rules:

1. Load model weights once before iterating over the manifest. Reloading a 14B
   model for each of 978 jobs is not an acceptable integration.
2. Read `prompt`, `negative_prompt`, `seed`, `generation`, and `inputs.image`
   directly from each manifest row. Do not rebuild prompts or resample seeds.
3. Save the original MP4 as `videos/<job_id>.mp4`; do not append model names,
   timestamps, or `_final` to the basename.
4. Record actual model/checkpoint/revision, provider prompt, actual media
   properties, timing, retries, and provider task IDs in metadata. Never save an
   API key or bearer token.
5. First run `--max-jobs 1`, inspect the MP4 and metadata, then continue the
   complete manifest in the same persistent model process.

Example after implementing the two functions:

```bash
cp model_config.example.json model_config.json
python model_runner_template/run_model.py \
  --bundle-root . \
  --config model_runner_template/model_config.json \
  --output-dir outputs/<model_id> \
  --max-jobs 1
```
"""


MODEL_CONFIG_EXAMPLE = {
    "model_id": "replace-with-model-id",
    "checkpoint": "/absolute/server/path/to/checkpoint",
    "revision": "record-an-immutable-revision",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "notes": "Do not put API keys or bearer tokens in this file.",
}


BUNDLE_VALIDATOR = r'''#!/usr/bin/env python3
"""Verify handoff hashes and the frozen 978/351 manifest contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

EXPECTED_SEEDS = {341867882, 1750912582, 265635392, 135883006}
AUDIT_SEEDS = EXPECTED_SEEDS - {341867882}
EXPECTED_TRACKS = {
    "physics_identification_side": 432,
    "viewpoint_robustness_main_top": 468,
    "random_seed_stability_audit": 78,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(root: Path, raw: str) -> Path:
    relative = PurePosixPath(raw.replace("\\", "/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe bundle path: {raw!r}")
    candidate = root.joinpath(*relative.parts).resolve(strict=True)
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"bundle path escapes root: {raw!r}")
    if not candidate.is_file():
        raise ValueError(f"not a file: {raw!r}")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.bundle_root.resolve(strict=True)

    checked = 0
    indexed = set()
    for line_number, line in enumerate((root / "sha256sums.txt").read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"invalid sha256sums.txt line {line_number}")
        expected, raw = match.groups()
        if raw in indexed:
            raise ValueError(f"duplicate checksum path: {raw}")
        indexed.add(raw)
        path = safe_file(root, raw)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {raw}")
        checked += 1

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "sha256sums.txt"
    }
    if actual != indexed:
        raise ValueError(
            "checksum index coverage differs: "
            f"unindexed={sorted(actual - indexed)[:5]}, missing={sorted(indexed - actual)[:5]}"
        )

    jobs = []
    with (root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        jobs = [json.loads(line) for line in handle if line.strip()]
    if len(jobs) != 978 or len({job["job_id"] for job in jobs}) != 978:
        raise ValueError("manifest must contain 978 unique jobs")
    images = {job["inputs"]["image"] for job in jobs}
    if len(images) != 351:
        raise ValueError(f"manifest must reference 351 images, found {len(images)}")
    for image in images:
        safe_file(root, image)

    tracks = Counter()
    seeds = Counter()
    for job in jobs:
        seed = job["seed"]
        camera = job["factors"]["camera"]
        scene = job["factors"]["scene_id"]
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected seed in {job['job_id']}")
        if job["generation"] != {"width": 832, "height": 480, "num_frames": 81, "fps": 16.0}:
            raise ValueError(f"unexpected media request in {job['job_id']}")
        if seed in AUDIT_SEEDS:
            if camera != "CAM_Side" or scene != "baseline":
                raise ValueError(f"invalid audit job {job['job_id']}")
            track = "random_seed_stability_audit"
        elif camera == "CAM_Side":
            track = "physics_identification_side"
        elif camera in {"CAM_Main", "CAM_Top"}:
            track = "viewpoint_robustness_main_top"
        else:
            raise ValueError(f"unexpected camera in {job['job_id']}")
        tracks[track] += 1
        seeds[seed] += 1
    if dict(tracks) != EXPECTED_TRACKS:
        raise ValueError(f"track counts differ: {dict(tracks)}")
    if seeds[341867882] != 900 or any(seeds[seed] != 26 for seed in AUDIT_SEEDS):
        raise ValueError(f"seed counts differ: {dict(seeds)}")

    print(json.dumps({"status": "ok", "hashed_files": checked, "jobs": 978, "images": 351}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


HANDOFF = """\
# Frozen standard-ball factorized978 generation pack

This directory is the complete handoff for generating the same 978 benchmark
videos with another model. `manifest.jsonl` is the immutable task contract and
all `inputs.image` paths resolve directly from this directory.

Before use, run:

```bash
python tools/validate_bundle.py .
```

The package contains 978 jobs, 351 first-frame PNGs, 13 experiments, 9 scenes,
3 camera names, and four frozen non-consecutive seeds. Requested output is
832x480, 81 frames at 16 fps (5 seconds by the benchmark convention).

Do not change prompts, negative prompts, seeds, first frames, task selection, or
job IDs. The Wan model profile, evaluation profile, and `resolved_build.yaml`
are included only as preparation provenance; they do not define the semantics
of Helios, Cosmos-Predict2.5, LongLive2.0, or any other replacement model.

Use `model_runner_template/` as the integration starting point. It loads a model
once and then streams the manifest. Each original video must be returned at
`videos/<job_id>.mp4`, together with `metadata/<job_id>.json`, logs,
`run_state.jsonl`, and a run summary. Preserve provider-original MP4s; record
actual output FPS/frame count/size rather than silently transcoding them.

Never include API keys, bearer tokens, model weights, or generated videos when
redistributing this task pack.
"""


def _write_generated_helpers(destination: Path) -> None:
    runner_source = destination / "code" / "templates" / "generation_model_runner"
    if not runner_source.is_dir():
        raise PackExportError(f"missing source-controlled model runner template: {runner_source}")
    shutil.copytree(runner_source, destination / "model_runner_template")
    tools_dir = destination / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "validate_bundle.py").write_text(BUNDLE_VALIDATOR, encoding="utf-8", newline="\n")
    shutil.copy2(
        destination / "code" / "scripts" / "validate_generation_outputs.py",
        tools_dir / "validate_generation_outputs.py",
    )
    shutil.copy2(
        destination / "code" / "docs" / "generation_task_pack_978.md",
        destination / "HANDOFF.md",
    )
    shutil.copy2(
        destination / "code" / "handoffs" / "generation_factorized978_v1.yaml",
        destination / "TASK_CONTRACT.yaml",
    )


def _validate_selection(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackExportError(f"invalid manifest.selection.json: {exc}") from exc
    if not isinstance(value, dict):
        raise PackExportError("manifest.selection.json must contain an object")
    for key in ("planned_job_count", "job_count"):
        if value.get(key) != EXPECTED_JOB_COUNT:
            raise PackExportError(
                f"manifest.selection.json {key}={value.get(key)!r}, expected {EXPECTED_JOB_COUNT}"
            )
    if value.get("planned_parameter_tuple_count") != EXPECTED_PARAMETER_TUPLE_COUNT:
        raise PackExportError(
            "manifest.selection.json planned_parameter_tuple_count must be "
            f"{EXPECTED_PARAMETER_TUPLE_COUNT}"
        )
    return value


def _check_required_code_files(workspace_root: Path) -> None:
    for relative in CANONICAL_TASK_FILES[3:] + PROVENANCE_ONLY_FILES[1:]:
        _source_under_workspace(
            workspace_root,
            _safe_relative_path(relative, "required code file"),
            "required code file",
        )


def _validate_exported_paths(destination: Path, jobs: Sequence[dict[str, Any]]) -> None:
    for job in jobs:
        relative = _safe_relative_path(job["inputs"]["image"], "exported inputs.image")
        target = destination.joinpath(*relative.parts).resolve(strict=True)
        if not _is_relative_to(target, destination) or not target.is_file():
            raise PackExportError(f"exported input does not resolve from package root: {relative}")
    for relative in CANONICAL_TASK_FILES + PROVENANCE_ONLY_FILES:
        target = destination.joinpath(*PurePosixPath(relative).parts)
        if not target.is_file():
            raise PackExportError(f"exported package is missing required file: {relative}")


def export_package(
    *,
    workspace_root: Path,
    prepared_run: Path,
    output: Path,
    archive: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export one complete frozen package and optionally a ZIP archive."""

    workspace_root = _resolve_existing_dir(workspace_root, "workspace root")
    prepared_run = _resolve_existing_dir(prepared_run, "prepared run")
    if not _is_relative_to(prepared_run, workspace_root):
        raise PackExportError("prepared run must be inside workspace root")
    output = _resolve_destination(output, workspace_root, "output")

    protected = [workspace_root / "code", workspace_root / "first_frames_v1_0", prepared_run]
    for path in protected:
        if _paths_overlap(output, path.resolve(strict=True)):
            raise PackExportError(f"output overlaps protected source path: {path}")
    if output.exists() and not overwrite:
        raise PackExportError(f"output already exists (use --overwrite): {output}")
    if output.exists() and not output.is_dir():
        raise PackExportError(f"output exists and is not a directory: {output}")

    archive_path: Path | None = None
    if archive is not None:
        archive_path = _resolve_destination(archive, workspace_root, "archive")
        if archive_path.suffix.lower() != ".zip":
            raise PackExportError(f"archive path must end in .zip: {archive_path}")
        if _paths_overlap(archive_path, output):
            raise PackExportError("archive must be outside the output directory")
        if archive_path.exists() and not overwrite:
            raise PackExportError(f"archive already exists (use --overwrite): {archive_path}")
        if archive_path.exists() and not archive_path.is_file():
            raise PackExportError(f"archive exists and is not a file: {archive_path}")

    manifest_source = prepared_run / "manifest.jsonl"
    selection_source = prepared_run / "manifest.selection.json"
    resolved_build_source = prepared_run / "resolved_build.yaml"
    for path in (manifest_source, selection_source, resolved_build_source):
        if not path.is_file():
            raise PackExportError(f"prepared run is missing {path.name}: {path}")

    frozen_hashes = {
        "manifest.jsonl": (manifest_source, EXPECTED_MANIFEST_SHA256),
        "manifest.selection.json": (selection_source, EXPECTED_SELECTION_SHA256),
        "resolved_build.yaml": (resolved_build_source, EXPECTED_RESOLVED_BUILD_SHA256),
    }
    for label, (path, expected_hash) in frozen_hashes.items():
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise PackExportError(
                f"{label} is not the frozen task artifact: sha256={actual_hash}, "
                f"expected={expected_hash}"
            )

    jobs = _read_jsonl(manifest_source)
    summary = validate_frozen_jobs(jobs)
    _validate_selection(selection_source)
    _check_required_code_files(workspace_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        shutil.copy2(manifest_source, staging / "manifest.jsonl")
        shutil.copy2(selection_source, staging / "manifest.selection.json")
        provenance_dir = staging / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            resolved_build_source,
            provenance_dir / "source_reference_resolved_build.yaml",
        )
        _copy_code_snapshot(workspace_root, staging)
        copied_hashes = _copy_first_frames(
            workspace_root, staging, summary["input_images"]
        )
        records_count = _filter_first_frame_records(
            workspace_root,
            staging,
            summary["input_images"],
            copied_hashes,
        )
        if records_count is None:
            # Keep the canonical path present while making the missing source explicit.
            _write_jsonl(staging / "first_frames_records.jsonl", [])
        elif records_count != EXPECTED_IMAGE_COUNT:
            raise PackExportError(
                f"expected {EXPECTED_IMAGE_COUNT} filtered records, found {records_count}"
            )

        commit = _git_commit(workspace_root)
        (staging / "git_commit.txt").write_text(commit + "\n", encoding="utf-8", newline="\n")
        _write_generated_helpers(staging)

        package_manifest = {
            "schema_version": "1.0.0",
            "pack_id": PACK_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_build_id": EXPECTED_BUILD_ID,
            "source_git_commit": commit,
            "counts": {key: value for key, value in summary.items() if key != "input_images"},
            "requested_media": EXPECTED_MEDIA,
            "frozen_seeds": [PRIMARY_SEED, *sorted(AUDIT_SEEDS)],
            "canonical_files": CANONICAL_TASK_FILES,
            "provenance_only_files": PROVENANCE_ONLY_FILES,
            "provenance_note": (
                "Wan model/evaluation configuration and the source reference resolved build "
                "document how the manifest was prepared; they are not replacement-model "
                "semantics and must not be executed on the receiver."
            ),
            "replaceable_model_scope": [
                "model_runner_template/run_model.py:load_model_once",
                "model_runner_template/run_model.py:generate_one",
                "model_runner_template/model_config.json (local, do not redistribute secrets)",
            ],
            "output_contract": {
                "video": "videos/<job_id>.mp4",
                "metadata": "metadata/<job_id>.json",
                "preserve_provider_original_video": True,
            },
            "integrity_index": "sha256sums.txt",
        }
        _write_json(staging / "PACKAGE_MANIFEST.json", package_manifest)
        _validate_exported_paths(staging, jobs)
        hashes = _write_checksum_index(staging)

        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        staging = None

        if archive_path is not None:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            if archive_path.exists():
                archive_path.unlink()
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
                allowZip64=True,
            ) as bundle:
                for path in sorted(output.rglob("*")):
                    if path.is_file():
                        arcname = (Path(output.name) / path.relative_to(output)).as_posix()
                        bundle.write(path, arcname)

        return {
            "pack_id": PACK_ID,
            "output": str(output),
            "archive": str(archive_path) if archive_path else None,
            "jobs": EXPECTED_JOB_COUNT,
            "unique_input_images": EXPECTED_IMAGE_COUNT,
            "hashed_files": len(hashes),
            "manifest_sha256": hashes["manifest.jsonl"],
            "git_commit": commit,
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--prepared-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--archive",
        nargs="?",
        const="__AUTO__",
        help="also create a ZIP; omit the value to use <output>.zip",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    archive: Path | None
    if args.archive == "__AUTO__":
        archive = args.output.with_suffix(".zip")
    elif args.archive:
        archive = Path(args.archive)
    else:
        archive = None
    try:
        result = export_package(
            workspace_root=args.workspace_root,
            prepared_run=args.prepared_run,
            output=args.output,
            archive=archive,
            overwrite=args.overwrite,
        )
    except PackExportError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
