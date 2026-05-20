#!/usr/bin/env python3
"""
Extract "seed" (first N frames) from each rendered video into two formats:

  1. seed_<N>frames.mp4  - a short video clip (for video-input models like
                           Gemini / Sora / Runway / Luma / Kling — used as
                           video continuation condition).
  2. frame_01.png ... frame_NN.png  - a frame sequence (for image-sequence
                           models like GPT-4o / Claude / Qwen-VL — used as
                           keyframe/first-frame continuation condition).

The corresponding prompt.md (already sitting next to the source video) is
also copied so that (seed, prompt) travel together for downstream continuation.

Scans: renders/<version>/<exp_id>/<variant_id>/<camera>/*.mp4
Writes: seeds/<version>/<exp_id>/<variant_id>/<camera>/
          seed_<N>frames.mp4
          frame_01.png ... frame_NN.png
          prompt.md

Default mode is dry-run. Pass --execute to actually write files.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", default="renders",
                        help="Directory to scan for source mp4 files (relative to --root).")
    parser.add_argument("--seed-root", default="seeds",
                        help="Output directory (relative to --root).")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--n-frames", type=int, default=10,
                        help="Number of leading frames to extract. Default 10.")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Optional filter, e.g. v1_A v1_B.")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Optional camera filter, e.g. CAM_Side.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing seed outputs.")
    return parser.parse_args()


def iter_rendered_videos(render_root, versions, exps, variants, cameras):
    """Yield (mp4_path, version, exp_id, variant_id, camera)."""
    for version in versions:
        version_dir = render_root / version
        if not version_dir.exists():
            continue
        for exp_dir in sorted(version_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_id = exp_dir.name
            if exps and exp_id not in exps:
                continue
            for variant_dir in sorted(exp_dir.iterdir()):
                if not variant_dir.is_dir():
                    continue
                variant_id = variant_dir.name
                if variants and variant_id not in variants:
                    continue
                for camera_dir in sorted(variant_dir.iterdir()):
                    if not camera_dir.is_dir():
                        continue
                    camera = camera_dir.name
                    if cameras and camera not in cameras:
                        continue
                    mp4_files = sorted(camera_dir.glob("*.mp4"))
                    if not mp4_files:
                        continue
                    mp4 = max(mp4_files, key=lambda p: (p.stat().st_mtime, p.name))
                    yield mp4, version, exp_id, variant_id, camera


def extract_video_seed(src_mp4, dst_mp4, n_frames, force):
    if dst_mp4.exists() and not force:
        return "skip"
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-i", str(src_mp4),
        "-frames:v", str(n_frames),
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return "ok"


def extract_frame_sequence(src_mp4, dst_dir, n_frames, force):
    existing = sorted(dst_dir.glob("frame_*.png"))
    if len(existing) >= n_frames and not force:
        return "skip"
    dst_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(dst_dir / "frame_%02d.png")
    cmd = [
        FFMPEG, "-y",
        "-i", str(src_mp4),
        "-frames:v", str(n_frames),
        "-start_number", "1",
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return "ok"


def copy_prompt(src_mp4, dst_dir):
    src_prompt = src_mp4.parent / "prompt.md"
    if not src_prompt.exists():
        return None
    dst_prompt = dst_dir / "prompt.md"
    shutil.copy2(src_prompt, dst_prompt)
    return dst_prompt


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    render_root = (root / args.render_root).resolve()
    seed_root = (root / args.seed_root).resolve()

    if not render_root.exists():
        print(f"Render root does not exist: {render_root}")
        return

    exps_filter = set(args.experiments) if args.experiments else None
    variants_filter = set(args.variants) if args.variants else None
    cameras_filter = set(args.cameras) if args.cameras else None

    jobs = list(iter_rendered_videos(render_root, args.versions,
                                      exps_filter, variants_filter, cameras_filter))

    if not jobs:
        print("No rendered mp4 files found under", render_root)
        return

    print(f"Found {len(jobs)} rendered videos. N_frames = {args.n_frames}.")
    for mp4, version, exp_id, variant_id, camera in jobs:
        dst_dir = seed_root / version / exp_id / variant_id / camera
        seed_video = dst_dir / f"seed_{args.n_frames}frames.mp4"
        print(f"  {mp4.relative_to(root)} -> {dst_dir.relative_to(root)}/")
        if args.execute:
            v_status = extract_video_seed(mp4, seed_video, args.n_frames, args.force)
            f_status = extract_frame_sequence(mp4, dst_dir, args.n_frames, args.force)
            p_dst = copy_prompt(mp4, dst_dir)
            p_status = "ok" if p_dst else "missing"
            print(f"    video:{v_status}  frames:{f_status}  prompt:{p_status}")

    if not args.execute:
        print(f"\nDry run only: {len(jobs)} seed jobs planned. Re-run with --execute.")


if __name__ == "__main__":
    main()
