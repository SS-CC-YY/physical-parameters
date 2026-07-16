from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _probe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    executable = shutil.which("ffprobe")
    if executable is None:
        return None, "ffprobe_not_found"
    process = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return None, process.stderr.strip() or f"ffprobe_returncode_{process.returncode}"
    payload = json.loads(process.stdout)
    streams = payload.get("streams", [])
    return (streams[0] if streams else None), (None if streams else "no_video_stream")


def evaluate_video(job: dict[str, Any], video_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    metrics: dict[str, Any] = {"exists": video_path.is_file()}
    if not video_path.is_file():
        flags.append("missing_video")
        return {"status": "invalid", "quality_flags": flags, "metrics": metrics}
    size_bytes = video_path.stat().st_size
    metrics["size_bytes"] = size_bytes
    if size_bytes < int(config.get("min_bytes", 1024)):
        flags.append("video_too_small")
    if bool(config.get("ffprobe", True)):
        probe, error = _probe(video_path)
        metrics["ffprobe"] = probe
        if error:
            flags.append(error)
            if not bool(config.get("ffprobe_required", False)) and error == "ffprobe_not_found":
                flags.remove(error)
    return {"status": "ok" if not flags else "invalid", "quality_flags": flags, "metrics": metrics}
