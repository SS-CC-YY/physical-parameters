#!/usr/bin/env python3
"""Create visual diagnostics for V1 inverse-physics evaluation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_LABELS = {
    "v1_A": "Free fall gravity",
    "v1_B": "Bounce restitution",
    "v1_C": "Slide friction",
    "v1_D": "Pendulum damping",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--make-overlays", action="store_true")
    parser.add_argument("--overlay-max-jobs", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--font-size", type=int, default=10)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out[item["job_id"]] = item
    return out


def to_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def to_int(value: Any, default: int = 0) -> int:
    val = to_float(value, float(default))
    if not math.isfinite(val):
        return default
    return int(round(val))


def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    return "".join(keep)


def locate_video(generated_root: Path, row: dict[str, str]) -> Path | None:
    direct = row.get("video", "")
    if direct:
        path = Path(direct)
        if path.exists():
            return path
    job_id = row["job_id"]
    candidates = [
        generated_root / "videos" / f"{job_id}.mp4",
        generated_root / f"{job_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(generated_root.rglob(f"{job_id}.mp4"))
    return matches[0] if matches else None


def load_track(track_path: Path) -> dict[str, np.ndarray]:
    rows = read_csv_rows(track_path)
    frame = np.array([to_int(r.get("frame_idx")) for r in rows], dtype=np.int32)
    time_s = np.array([to_float(r.get("time_s")) for r in rows], dtype=np.float64)
    x = np.array([to_float(r.get("cx_px")) for r in rows], dtype=np.float64)
    y = np.array([to_float(r.get("cy_px")) for r in rows], dtype=np.float64)
    found = np.array([str(r.get("found", "")).strip().lower() == "true" for r in rows], dtype=bool)
    return {"frame": frame, "time_s": time_s, "x": x, "y": y, "found": found}


def row_title(row: dict[str, str]) -> str:
    target = to_float(row.get("target_param_value"))
    estimate = to_float(row.get("estimate_value"))
    rel = to_float(row.get("rel_error"))
    return (
        f"{row.get('job_id', '')}\n"
        f"target={target:.5g}  estimate={estimate:.5g}  rel_error={rel:.3g}"
    )


def plot_job_track(row: dict[str, str], track: dict[str, np.ndarray], outpath: Path, font_size: int) -> None:
    t = track["time_s"]
    x = track["x"]
    y = track["y"]
    found = track["found"]
    valid = np.isfinite(x) & np.isfinite(y)

    plt.rcParams.update({"font.size": font_size})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fig.suptitle(row_title(row), fontsize=font_size + 1)

    axes[0].plot(t[valid], x[valid], label="x_px", color="#2f6fbb", linewidth=1.5)
    axes[0].plot(t[valid], y[valid], label="y_px", color="#d95f02", linewidth=1.5)
    if np.any(found & valid):
        axes[0].scatter(t[found & valid], y[found & valid], s=8, color="#d95f02", alpha=0.45)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("image coordinate (px)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(x[valid], y[valid], color="#222222", linewidth=1.2)
    axes[1].scatter(x[valid], y[valid], s=10, c=t[valid], cmap="viridis", alpha=0.8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("x_px")
    axes[1].set_ylabel("y_px")
    axes[1].set_title("image-plane track")
    axes[1].grid(True, alpha=0.25)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def group_ok_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(row.get("experiment", "unknown"), []).append(row)
    return grouped


def plot_experiment_summaries(rows: list[dict[str, str]], outdir: Path, font_size: int) -> list[Path]:
    paths: list[Path] = []
    grouped = group_ok_rows(rows)
    plt.rcParams.update({"font.size": font_size})
    for experiment, exp_rows in sorted(grouped.items()):
        if not exp_rows:
            continue
        exp_rows = sorted(exp_rows, key=lambda r: to_float(r.get("target_param_value")))
        targets = np.array([to_float(r.get("target_param_value")) for r in exp_rows], dtype=np.float64)
        estimates = np.array([to_float(r.get("estimate_value")) for r in exp_rows], dtype=np.float64)
        rel_errors = np.array([to_float(r.get("rel_error")) for r in exp_rows], dtype=np.float64)
        labels = [r.get("variant", r.get("job_id", "")) for r in exp_rows]
        valid = np.isfinite(targets) & np.isfinite(estimates)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        fig.suptitle(f"{experiment}: {EXPERIMENT_LABELS.get(experiment, experiment)}", fontsize=font_size + 2)

        axes[0].scatter(targets[valid], estimates[valid], s=56, color="#2f6fbb")
        if np.any(valid):
            lo = float(np.nanmin([np.nanmin(targets[valid]), np.nanmin(estimates[valid])]))
            hi = float(np.nanmax([np.nanmax(targets[valid]), np.nanmax(estimates[valid])]))
            pad = 0.05 * max(hi - lo, 1e-6)
            axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#777777", linewidth=1)
        for label, tx, est in zip(labels, targets, estimates):
            if np.isfinite(tx) and np.isfinite(est):
                axes[0].annotate(label, (tx, est), textcoords="offset points", xytext=(5, 4), fontsize=max(font_size - 2, 7))
        axes[0].set_xlabel("target")
        axes[0].set_ylabel("estimate")
        axes[0].grid(True, alpha=0.25)

        x_pos = np.arange(len(exp_rows))
        axes[1].bar(x_pos, rel_errors, color="#d95f02", alpha=0.85)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=30, ha="right")
        axes[1].set_ylabel("relative error")
        axes[1].grid(True, axis="y", alpha=0.25)

        outpath = outdir / "summary" / f"{safe_name(experiment)}_summary.png"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=160)
        plt.close(fig)
        paths.append(outpath)
    return paths


def draw_overlay(video_path: Path, row: dict[str, str], track: dict[str, np.ndarray], outpath: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or to_float(row.get("fps"), 16.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_to_idx = {int(f): i for i, f in enumerate(track["frame"])}
    trail: list[tuple[int, int]] = []
    frame_idx = 0
    text = (
        f"{row.get('experiment')} {row.get('variant')} "
        f"target={to_float(row.get('target_param_value')):.4g} "
        f"est={to_float(row.get('estimate_value')):.4g}"
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = frame_to_idx.get(frame_idx)
        if idx is not None:
            x = track["x"][idx]
            y = track["y"][idx]
            if np.isfinite(x) and np.isfinite(y):
                pt = (int(round(x)), int(round(y)))
                trail.append(pt)
                if len(trail) > 120:
                    trail = trail[-120:]
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        if len(trail) >= 2:
            cv2.polylines(frame, [np.array(trail, dtype=np.int32)], False, (255, 255, 0), 2)
        cv2.rectangle(frame, (8, 8), (min(width - 8, 8 + 760), 68), (0, 0, 0), -1)
        cv2.putText(frame, row.get("job_id", ""), (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()


def relative_link(base: Path, target: Path) -> str:
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return target.as_posix()


def write_html_report(
    outdir: Path,
    rows: list[dict[str, str]],
    job_plots: dict[str, Path],
    overlays: dict[str, Path],
    summary_plots: list[Path],
) -> None:
    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>V1 Wan2.2 I2V Evaluation Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#fafafa;color:#222}",
        "table{border-collapse:collapse;width:100%;background:white}",
        "th,td{border:1px solid #ddd;padding:6px 8px;font-size:13px}",
        "th{background:#f0f0f0;text-align:left}",
        "img{max-width:100%;border:1px solid #ddd;background:white}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #ddd;padding:12px;margin:14px 0}",
        ".bad{color:#b00020}",
        "</style></head><body>",
        "<h1>V1 Wan2.2 I2V Evaluation Report</h1>",
        "<h2>Experiment Summary</h2>",
        "<div class='grid'>",
    ]
    for plot in summary_plots:
        html_parts.append(f"<div class='card'><img src='{html.escape(relative_link(outdir, plot))}'></div>")
    html_parts.append("</div>")

    html_parts.append("<h2>Jobs</h2>")
    html_parts.append("<table><tr><th>job</th><th>experiment</th><th>variant</th><th>status</th><th>target</th><th>estimate</th><th>rel_error</th><th>plot</th><th>overlay</th></tr>")
    for row in rows:
        job_id = row.get("job_id", "")
        status = row.get("status", "")
        cls = " class='bad'" if status not in {"ok", ""} else ""
        plot_link = ""
        if job_id in job_plots:
            rel = html.escape(relative_link(outdir, job_plots[job_id]))
            plot_link = f"<a href='{rel}'>track</a>"
        overlay_link = ""
        if job_id in overlays:
            rel = html.escape(relative_link(outdir, overlays[job_id]))
            overlay_link = f"<a href='{rel}'>overlay</a>"
        html_parts.append(
            f"<tr{cls}><td>{html.escape(job_id)}</td><td>{html.escape(row.get('experiment',''))}</td>"
            f"<td>{html.escape(row.get('variant',''))}</td><td>{html.escape(status)}</td>"
            f"<td>{html.escape(row.get('target_param_value',''))}</td><td>{html.escape(row.get('estimate_value',''))}</td>"
            f"<td>{html.escape(row.get('rel_error',''))}</td><td>{plot_link}</td><td>{overlay_link}</td></tr>"
        )
    html_parts.append("</table></body></html>")
    (outdir / "index.html").write_text("\n".join(html_parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (args.eval_dir / "visualization")
    outdir.mkdir(parents=True, exist_ok=True)

    _manifest = read_manifest(args.manifest)
    summary_path = args.eval_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")
    rows = read_csv_rows(summary_path)
    if args.max_jobs is not None:
        rows = rows[: args.max_jobs]

    job_plots: dict[str, Path] = {}
    overlays: dict[str, Path] = {}
    overlay_count = 0

    for row in rows:
        if row.get("status") != "ok":
            continue
        job_id = row["job_id"]
        track_path = args.eval_dir / "tracks" / f"{job_id}.csv"
        if not track_path.exists():
            print(f"skip plot, missing track: {job_id}")
            continue
        track = load_track(track_path)
        plot_path = outdir / "jobs" / f"{safe_name(job_id)}_track.png"
        plot_job_track(row, track, plot_path, args.font_size)
        job_plots[job_id] = plot_path
        print(f"wrote {plot_path}")

        if args.make_overlays:
            if args.overlay_max_jobs is not None and overlay_count >= args.overlay_max_jobs:
                continue
            video_path = locate_video(args.generated_root, row)
            if video_path is None:
                print(f"skip overlay, missing video: {job_id}")
                continue
            overlay_path = outdir / "overlays" / f"{safe_name(job_id)}_overlay.mp4"
            draw_overlay(video_path, row, track, overlay_path)
            overlays[job_id] = overlay_path
            overlay_count += 1
            print(f"wrote {overlay_path}")

    summary_plots = plot_experiment_summaries(rows, outdir, args.font_size)
    write_html_report(outdir, rows, job_plots, overlays, summary_plots)
    print(f"wrote {outdir / 'index.html'}")


if __name__ == "__main__":
    main()
