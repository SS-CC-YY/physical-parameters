#!/usr/bin/env python3
"""Export a small, model-neutral research pack for the 978 video tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import yaml


EXPECTED_JOBS = 978
EXPECTED_IMAGES = 351
EXPECTED_MANIFEST_SHA256 = "0239b849a8d2b2922b460d6586e3898199907fed9602dcaf84882049091f565a"
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ExportError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(raw: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise ExportError("input image path must be a non-empty string")
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExportError(f"unsafe relative path: {raw!r}")
    return path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_jobs(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExportError(f"invalid JSON on manifest line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ExportError(f"manifest line {line_number} is not an object")
            jobs.append(row)
    return jobs


def _validate_jobs(jobs: list[dict[str, Any]]) -> list[str]:
    if len(jobs) != EXPECTED_JOBS:
        raise ExportError(f"expected {EXPECTED_JOBS} jobs, found {len(jobs)}")
    seen: set[str] = set()
    images: set[str] = set()
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise ExportError(f"invalid job_id: {job_id!r}")
        if job_id in seen:
            raise ExportError(f"duplicate job_id: {job_id}")
        seen.add(job_id)
        inputs = job.get("inputs")
        if not isinstance(inputs, dict):
            raise ExportError(f"{job_id}: inputs is missing")
        image = _safe_relative(inputs.get("image"))
        if image.parts[:2] != ("first_frames_v1_0", "images"):
            raise ExportError(f"{job_id}: unexpected first-frame path {image}")
        images.add(image.as_posix())
        for key in ("prompt", "negative_prompt", "seed", "targets", "generation", "factors"):
            if key not in job:
                raise ExportError(f"{job_id}: missing {key}")
    if len(images) != EXPECTED_IMAGES:
        raise ExportError(f"expected {EXPECTED_IMAGES} first frames, found {len(images)}")
    return sorted(images)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_prompt_profile(common_path: Path, explicit_path: Path) -> dict[str, Any]:
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    explicit = yaml.safe_load(explicit_path.read_text(encoding="utf-8"))
    if not isinstance(common, dict) or not isinstance(explicit, dict):
        raise ExportError("prompt YAML roots must be mappings")
    explicit = dict(explicit)
    explicit.pop("extends", None)
    return _deep_merge(common, explicit)


def simplify_job(job: dict[str, Any]) -> dict[str, Any]:
    factors = job["factors"]
    generation = job["generation"]
    fps = float(generation["fps"])
    frames = int(generation["num_frames"])
    return {
        "job_id": job["job_id"],
        "input_image": job["inputs"]["image"],
        "output_video": f"videos/{job['job_id']}.mp4",
        "experiment_id": job["experiment_id"],
        "parameter_tuple_id": factors["parameter_tuple_id"],
        "scene_id": factors["scene_id"],
        "object_id": factors["object_id"],
        "camera": factors["camera"],
        "seed": job["seed"],
        "targets": job["targets"],
        "units": job.get("units", {}),
        "prompt": job["prompt"],
        "negative_prompt": job["negative_prompt"],
        "video": {
            "duration_seconds": (frames - 1) / fps,
            "width": int(generation["width"]),
            "height": int(generation["height"]),
            "fps": fps,
            "num_frames": frames,
        },
    }


README = """\
# 978 条物理视频生成任务

这是一个给不同图生视频模型使用的精简科研数据包。

- `tasks.jsonl`：978 条任务，每行一条；
- `prompts.csv`：相同任务的表格版本，方便直接查看；
- `first_frames_v1_0/images/`：任务实际使用的 351 张首帧；
- `prompt_setup/`：统一 prompt 规则、13 个实验模板和重建脚本。

每条任务已经包含 `input_image`、`prompt`、`negative_prompt`、`seed`、
物理参数和建议的 5 秒视频设置。模型代码只需逐行读取、调用模型，并把视频保存到
`output_video` 指定的位置。

```python
import json
from pathlib import Path

root = Path("/path/to/this/folder")
for line in (root / "tasks.jsonl").open(encoding="utf-8"):
    task = json.loads(line)
    image = root / task["input_image"]
    # video = your_model(image, task["prompt"], task["negative_prompt"], task["seed"])
    output = root / task["output_video"]
    output.parent.mkdir(parents=True, exist_ok=True)
    # save video to output
```

为了便于不同模型间比较，请直接使用给定首帧和 prompt，不再自行改写。模型可以使用
最接近的原生分辨率和帧率，但视频时长应约为 5 秒。各模型代码、依赖和权重需由对应
负责人单独安装，本包不包含这些内容。
"""


PROMPT_README = """\
# Prompt 搭建方式

所有任务都按照相同的七段顺序构建：

```text
task -> scene -> camera -> dynamics -> parameter -> terminal -> quality
```

`common_physics_video_v1.yaml` 保存公共的 task、scene、camera 和 quality；
`all_experiments_explicit_v1.yaml` 保存 13 个实验各自的 dynamics、parameter 和
terminal；`prompt_templates.json` 是合并后的版本，供独立脚本直接读取。

最终文字已经写入 `../tasks.jsonl` 的每条任务。验证模板能否重建全部 978 条 prompt：

```bash
python prompt_setup/build_prompts.py --check
```

另存一份重新生成的 prompt：

```bash
python prompt_setup/build_prompts.py --output rebuilt_prompts.jsonl
```
"""


PROMPT_BUILDER = r'''#!/usr/bin/env python3
"""Rebuild the prompt text in tasks.jsonl from prompt_templates.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class StrictValues(dict):
    def __missing__(self, key):
        raise KeyError(f"prompt template references missing value: {key}")


def render(profile, task):
    object_id = task["object_id"]
    values = StrictValues(
        experiment_id=task["experiment_id"],
        scene_id=task["scene_id"],
        object_id=object_id,
        object_description=profile.get("object_descriptions", {}).get(
            object_id, object_id.replace("_", " ")
        ),
        camera=task["camera"],
        **task["targets"],
    )
    experiment = profile["experiment_templates"][task["experiment_id"]]
    sections = {
        "task": profile["shared_templates"]["task"],
        "scene": profile["shared_templates"]["scene"],
        "camera": profile["shared_templates"]["camera"],
        "dynamics": experiment["dynamics"],
        "parameter": experiment["parameter"],
        "terminal": experiment["terminal"],
        "quality": profile["shared_templates"]["quality"],
    }
    prompt = " ".join(
        str(sections[name]).format_map(values).strip()
        for name in profile["section_order"]
        if str(sections[name]).strip()
    )
    negative = str(profile["negative_prompt"]).format_map(values).strip()
    extra = str(profile.get("additional_negative_prompt", "")).format_map(values).strip()
    if extra:
        negative = f"{negative}, {extra}"
    return prompt, negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    setup = Path(__file__).resolve().parent
    root = setup.parent
    tasks_path = (args.tasks or root / "tasks.jsonl").resolve()
    profile = json.loads((setup / "prompt_templates.json").read_text(encoding="utf-8"))
    rebuilt = []
    mismatches = []
    with tasks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            prompt, negative = render(profile, task)
            rebuilt.append(
                {"job_id": task["job_id"], "prompt": prompt, "negative_prompt": negative}
            )
            if prompt != task["prompt"] or negative != task["negative_prompt"]:
                mismatches.append(task["job_id"])

    if args.output:
        with args.output.open("w", encoding="utf-8", newline="\n") as output:
            for row in rebuilt:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.check:
        if mismatches:
            raise SystemExit(f"prompt mismatch in {len(mismatches)} tasks: {mismatches[:5]}")
        print(f"OK: rebuilt {len(rebuilt)} prompts")
    elif not args.output:
        print(json.dumps(rebuilt[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def _write_tasks(root: Path, tasks: list[dict[str, Any]]) -> None:
    with (root / "tasks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")

    columns = [
        "job_id",
        "input_image",
        "output_video",
        "experiment_id",
        "parameter_tuple_id",
        "scene_id",
        "object_id",
        "camera",
        "seed",
        "targets_json",
        "duration_seconds",
        "prompt",
        "negative_prompt",
    ]
    with (root / "prompts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task in tasks:
            writer.writerow(
                {
                    "job_id": task["job_id"],
                    "input_image": task["input_image"],
                    "output_video": task["output_video"],
                    "experiment_id": task["experiment_id"],
                    "parameter_tuple_id": task["parameter_tuple_id"],
                    "scene_id": task["scene_id"],
                    "object_id": task["object_id"],
                    "camera": task["camera"],
                    "seed": task["seed"],
                    "targets_json": json.dumps(task["targets"], ensure_ascii=False, sort_keys=True),
                    "duration_seconds": task["video"]["duration_seconds"],
                    "prompt": task["prompt"],
                    "negative_prompt": task["negative_prompt"],
                }
            )


def export_pack(
    workspace_root: Path,
    manifest: Path,
    output: Path,
    archive: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    output = output.resolve(strict=False)
    if not workspace_root.is_dir() or not _inside(manifest, workspace_root):
        raise ExportError("manifest must be inside the workspace")
    if not _inside(output, workspace_root) or output == workspace_root:
        raise ExportError("output must be a child of the workspace")
    if _sha256(manifest) != EXPECTED_MANIFEST_SHA256:
        raise ExportError("the selected manifest is not the frozen 978-task manifest")
    if output.exists() and not overwrite:
        raise ExportError(f"output already exists: {output}")
    if archive is not None:
        archive = archive.resolve(strict=False)
        if not _inside(archive, workspace_root) or archive.suffix.lower() != ".zip":
            raise ExportError("archive must be a .zip inside the workspace")
        if archive.exists() and not overwrite:
            raise ExportError(f"archive already exists: {archive}")

    jobs = _read_jobs(manifest)
    image_paths = _validate_jobs(jobs)
    tasks = [simplify_job(job) for job in jobs]
    common_source = workspace_root / "code/configs/prompts/common_physics_video_v1.yaml"
    explicit_source = workspace_root / "code/configs/prompts/all_experiments_explicit_v1.yaml"
    profile = load_prompt_profile(common_source, explicit_source)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (staging / "README.md").write_text(README, encoding="utf-8", newline="\n")
        _write_tasks(staging, tasks)

        setup = staging / "prompt_setup"
        setup.mkdir()
        shutil.copy2(common_source, setup / common_source.name)
        explicit_text = explicit_source.read_text(encoding="utf-8").replace(
            "extends: configs/prompts/common_physics_video_v1.yaml",
            "extends: common_physics_video_v1.yaml",
            1,
        )
        (setup / explicit_source.name).write_text(explicit_text, encoding="utf-8", newline="\n")
        (setup / "prompt_templates.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        (setup / "build_prompts.py").write_text(
            PROMPT_BUILDER, encoding="utf-8", newline="\n"
        )
        (setup / "README.md").write_text(PROMPT_README, encoding="utf-8", newline="\n")

        for raw in image_paths:
            relative = _safe_relative(raw)
            source = workspace_root.joinpath(*relative.parts).resolve(strict=True)
            if not _inside(source, workspace_root) or not source.is_file():
                raise ExportError(f"missing or unsafe first frame: {raw}")
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        staging = None

        if archive is not None:
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                archive.unlink()
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
            ) as bundle:
                for path in sorted(output.rglob("*")):
                    if path.is_file():
                        name = (Path(output.name) / path.relative_to(output)).as_posix()
                        bundle.write(path, name)

        return {
            "output": str(output),
            "archive": str(archive) if archive else None,
            "tasks": len(tasks),
            "first_frames": len(image_paths),
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", dest="archive", nargs="?", const="__AUTO__")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    archive = None
    if args.archive == "__AUTO__":
        archive = args.output.with_suffix(".zip")
    elif args.archive:
        archive = Path(args.archive)
    try:
        result = export_pack(
            args.workspace_root, args.manifest, args.output, archive, args.overwrite
        )
    except (ExportError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
