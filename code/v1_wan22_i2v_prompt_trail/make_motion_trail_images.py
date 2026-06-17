#!/usr/bin/env python3
"""Create single-image motion-trail conditioning images for V1 I2V ablations."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_EXPERIMENTS = ["v1_A", "v1_B", "v1_C", "v1_D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-root", type=Path, default=Path("blender/seeds"))
    parser.add_argument("--renders-root", type=Path, default=Path("blender/renders"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--cameras", nargs="+", default=["CAM_Side"])
    parser.add_argument("--modes", nargs="+", default=["raw_frame10", "center_10f"])
    parser.add_argument("--mask-mode", choices=["orange"], default="orange")
    parser.add_argument("--ghost-min-alpha", type=float, default=0.12)
    parser.add_argument("--ghost-max-alpha", type=float, default=0.42)
    parser.add_argument("--center-dot-radius", type=int, default=5)
    parser.add_argument("--center-current-radius", type=int, default=6)
    parser.add_argument("--center-history-color", default="40,120,255")
    parser.add_argument("--center-current-color", default="255,40,40")
    parser.add_argument("--center-draw-line", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def parse_mode(mode: str) -> tuple[str, int]:
    if mode == "raw_frame10":
        return ("raw", 10)
    match = re.match(r"^(trail|center)_(\d+)f$", mode)
    if not match:
        raise ValueError(f"unknown conditioning mode: {mode}")
    kind, count = match.groups()
    return (kind, int(count))


def frame_index(path: Path) -> int:
    match = re.search(r"frame_(\d+)\.png$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def read_png_frames(seed_camera_dir: Path, count: int) -> list:
    import numpy as np
    from PIL import Image

    paths = sorted(seed_camera_dir.glob("frame_*.png"), key=frame_index)
    selected = [p for p in paths if 1 <= frame_index(p) <= count]
    if len(selected) < count:
        return []
    frames = []
    for path in selected[:count]:
        image = Image.open(path).convert("RGB")
        frames.append(np.asarray(image, dtype=np.uint8))
    return frames


def find_render_video(renders_root: Path, experiment: str, variant: str, camera: str) -> Path | None:
    camera_dir = renders_root / "v1" / experiment / variant / camera
    if not camera_dir.exists():
        return None
    videos = sorted(camera_dir.glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
    return videos[0] if videos else None


def read_video_frames(video_path: Path, count: int) -> list:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames if len(frames) >= count else []


def orange_mask(frame) -> object:
    import numpy as np

    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    mask = (
        (r > 120)
        & (g > 45)
        & (g < 205)
        & (b < 150)
        & (r > g + 20)
        & (g > b + 10)
    )
    return mask.astype(np.float32)


def object_center(frame) -> tuple[float, float] | None:
    import numpy as np

    mask = orange_mask(frame)
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) < 8:
        return None
    return (float(xs.mean()), float(ys.mean()))


def make_trail(frames: list, min_alpha: float, max_alpha: float):
    if len(frames) < 2:
        raise ValueError("at least two frames are required for trail image")
    current = frames[-1].copy()
    out = current.astype("float32")
    history = frames[:-1]
    if not history:
        return current
    for idx, frame in enumerate(history):
        alpha = min_alpha + (max_alpha - min_alpha) * (idx / max(1, len(history) - 1))
        mask = orange_mask(frame)
        ghost = frame.astype("float32")
        mask_f = mask[:, :, None]
        out = out * (1.0 - alpha * mask_f) + ghost * (alpha * mask_f)
    out = out.clip(0, 255).astype("uint8")
    return out


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected R,G,B color, got: {value}")
    return tuple(max(0, min(255, part)) for part in parts)


def make_center_marks(
    frames: list,
    dot_radius: int,
    current_radius: int,
    history_color: str,
    current_color: str,
    draw_line: bool,
):
    from PIL import Image, ImageDraw
    import numpy as np

    if len(frames) < 2:
        raise ValueError("at least two frames are required for center-mark image")
    current = frames[-1].copy()
    centers = [object_center(frame) for frame in frames]
    valid_centers = [center for center in centers if center is not None]
    if not valid_centers:
        return current

    image = Image.fromarray(current)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    hist_rgb = parse_rgb(history_color)
    cur_rgb = parse_rgb(current_color)

    if draw_line and len(valid_centers) > 1:
        line_points = [(float(x), float(y)) for x, y in valid_centers]
        draw.line(line_points, fill=(*hist_rgb, 110), width=max(1, dot_radius // 2))

    for idx, center in enumerate(centers):
        if center is None:
            continue
        x, y = center
        is_current = idx == len(centers) - 1
        radius = current_radius if is_current else dot_radius
        rgb = cur_rgb if is_current else hist_rgb
        alpha = 220 if is_current else int(70 + 130 * (idx / max(1, len(centers) - 2)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*rgb, alpha), outline=(255, 255, 255, 180))

    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(composed)


def variant_sort_key(path: Path) -> tuple[str, float | str]:
    match = re.search(r"(\d+)(?:p(\d+))?", path.name)
    if not match:
        return (path.name, path.name)
    whole, frac = match.groups()
    value = float(whole if frac is None else f"{whole}.{frac}")
    return (re.sub(r"\d.*$", "", path.name), value)


def write_image(path: Path, image) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def build_one(args: argparse.Namespace, mode: str, experiment: str, variant_dir: Path, camera: str) -> bool:
    kind, count = parse_mode(mode)
    seed_camera_dir = variant_dir / camera
    dst = args.output_root / mode / experiment / variant_dir.name / camera / "conditioning.png"
    dst_meta = dst.with_suffix(".txt")
    if kind == "raw":
        src = seed_camera_dir / "frame_10.png"
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        dst_meta.write_text(f"mode={mode}\nsource={src.as_posix()}\ncurrent_frame=10\n", encoding="utf-8")
        return True

    frames = read_png_frames(seed_camera_dir, count)
    source = f"{seed_camera_dir.as_posix()} frame pngs"
    if not frames:
        video = find_render_video(args.renders_root, experiment, variant_dir.name, camera)
        if video is None:
            return False
        frames = read_video_frames(video, count)
        source = video.as_posix()
    if not frames:
        return False
    if kind == "trail":
        image = make_trail(frames, args.ghost_min_alpha, args.ghost_max_alpha)
    elif kind == "center":
        image = make_center_marks(
            frames=frames,
            dot_radius=args.center_dot_radius,
            current_radius=args.center_current_radius,
            history_color=args.center_history_color,
            current_color=args.center_current_color,
            draw_line=args.center_draw_line,
        )
    else:
        raise ValueError(f"unsupported mode kind: {kind}")
    write_image(dst, image)
    dst_meta.write_text(f"mode={mode}\nsource={source}\ncurrent_frame={count}\n", encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    total = 0
    missing = []
    for experiment in args.experiments:
        exp_dir = args.seeds_root / "v1" / experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"seed experiment directory not found: {exp_dir}")
        variants = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=variant_sort_key)
        for variant_dir in variants:
            for camera in args.cameras:
                for mode in args.modes:
                    ok = build_one(args, mode, experiment, variant_dir, camera)
                    if ok:
                        total += 1
                    else:
                        missing.append(f"{mode}/{experiment}/{variant_dir.name}/{camera}")
    if missing and not args.allow_missing:
        raise RuntimeError("missing conditioning images:\n" + "\n".join(missing[:50]))
    print(f"wrote {total} conditioning images under {args.output_root}")
    if missing:
        print(f"missing/skipped: {len(missing)}")


if __name__ == "__main__":
    main()
