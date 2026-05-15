#!/usr/bin/env python3
"""
Batch render helper for benchmark .blend files.

Scans both:
  - root-level baselines (e.g. v1/baseline*.blend)  - only when --include-baseline
  - variant blends: variants/v{1,2,3}/<exp_id>/<variant_id>/*.blend

For each render, the corresponding prompt file from prompts/ is copied next
to the output so that (video, prompt) travel together.

Output layout:
  renders/<version>/<exp_id>/<variant_id>/<camera>/
    <exp_id>_<variant_id>_<camera>_0001-0192.mp4
    prompt.md

Default mode is dry-run. Pass --execute to actually run Blender.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


CAMERAS = ("CAM_Main", "CAM_Side", "CAM_Top")

EXP_ID_PATTERN = re.compile(r"^v[123]_[A-G]$")

PROMPT_MAP = {
    "v1_A": "v1_A_freefall_gravity.md",
    "v1_B": "v1_B_bounce_restitution.md",
    "v1_C": "v1_C_slide_friction.md",
    "v1_D": "v1_D_pendulum_damping.md",
    "v2_A": "v2_A_projectile_gravity.md",
    "v2_B": "v2_B_repeated_bounce.md",
    "v2_C": "v2_C_incline_friction.md",
    "v2_D": "v2_D_long_pendulum.md",
    "v2_E": "v2_E_tilted_bounce.md",
    "v2_F": "v2_F_forced_pendulum.md",
    "v3_A": "v3_A_projectile_drag_bounce.md",
    "v3_B": "v3_B_spring_damper.md",
    "v3_C": "v3_C_incline_wall.md",
    "v3_D": "v3_D_em_pendulum.md",
    "v3_E": "v3_E_rolling_collision.md",
    "v3_F": "v3_F_two_ball_collision.md",
    "v3_G": "v3_G_spatial_friction.md",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"], choices=["v1", "v2", "v3"])
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Optional filter, e.g. v1_A v1_B. Default: all within --versions.")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Optional variant ID filter, e.g. g9p81 baseline. Default: all.")
    parser.add_argument("--output-root", default=None, help="Default: <root>/renders")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    parser.add_argument("--include-baseline", action="store_true",
                        help="Also render the root-level baseline blends (v1/baseline*.blend).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a camera output directory when a matching mp4 already exists.")
    parser.add_argument("--execute", action="store_true", help="Actually run Blender.")
    return parser.parse_args()


def exp_id_from_blend(blend_path):
    for part in blend_path.parts:
        if EXP_ID_PATTERN.fullmatch(part):
            return part
    m = re.search(r"(v[123])_([A-G])", blend_path.stem)
    return f"{m.group(1)}_{m.group(2)}" if m else "unknown"


def variant_id_from_blend(blend_path):
    if "variants" in blend_path.parts:
        return blend_path.parent.name
    return "baseline"


def iter_blend_files(root, versions, experiments_filter, variants_filter, include_baseline):
    for version in versions:
        if include_baseline:
            for bp in sorted((root / version).glob("baseline*.blend")):
                exp = exp_id_from_blend(bp)
                if experiments_filter and exp not in experiments_filter:
                    continue
                if variants_filter and "baseline" not in variants_filter:
                    continue
                yield bp
        variants_root = root / "variants" / version
        if variants_root.exists():
            for bp in sorted(variants_root.rglob("baseline*.blend")):
                exp = exp_id_from_blend(bp)
                if experiments_filter and exp not in experiments_filter:
                    continue
                if variants_filter and variant_id_from_blend(bp) not in variants_filter:
                    continue
                yield bp


def build_python_expr(camera_name):
    # Blender 5.0+ requires media_type='VIDEO' before FFMPEG becomes a valid file_format.
    return (
        "import bpy; "
        f"bpy.context.scene.camera=bpy.data.objects['{camera_name}']; "
        "s=bpy.context.scene.render; "
        "s.image_settings.media_type='VIDEO'; "
        "s.image_settings.file_format='FFMPEG'; "
        "s.ffmpeg.format='MPEG4'; "
        "s.ffmpeg.codec='H264'; "
        "s.ffmpeg.constant_rate_factor='MEDIUM'"
    )


def command_for(blender, blend_path, output_dir, camera_name, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        blender,
        "-b",
        str(blend_path),
        "--python-expr",
        build_python_expr(camera_name),
        "-o",
        str(output_dir / f"{stem}_####"),
        "-a",
    ]


def copy_prompt(root, exp_id, output_dir):
    if exp_id not in PROMPT_MAP:
        return None
    src = root / "prompts" / PROMPT_MAP[exp_id]
    if not src.exists():
        return None
    dst = output_dir / "prompt.md"
    shutil.copy2(src, dst)
    return dst


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else root / "renders"

    experiments_filter = set(args.experiments) if args.experiments else None
    variants_filter = set(args.variants) if args.variants else None

    plans = []
    for blend_path in iter_blend_files(root, args.versions, experiments_filter,
                                        variants_filter, args.include_baseline):
        exp_id = exp_id_from_blend(blend_path)
        variant_id = variant_id_from_blend(blend_path)
        version = exp_id.split("_")[0]
        for camera_name in args.cameras:
            output_dir = output_root / version / exp_id / variant_id / camera_name
            stem = f"{exp_id}_{variant_id}_{camera_name}"
            if args.skip_existing and any(output_dir.glob(f"{stem}_*.mp4")):
                print(f"skip existing: {output_dir}")
                continue
            plans.append((blend_path, output_dir, camera_name, exp_id, stem))

    for blend_path, output_dir, camera_name, exp_id, stem in plans:
        cmd = command_for(args.blender, blend_path, output_dir, camera_name, stem)
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        if args.execute:
            subprocess.run(cmd, check=True)
            copied = copy_prompt(root, exp_id, output_dir)
            if copied:
                print(f"  -> prompt copied to {copied}")

    if not plans:
        print("No render commands generated after filters/skips.")
        print("If this is unexpected, check that variant .blend files exist and filters match.")
    elif not args.execute:
        print(f"\nDry run only: {len(plans)} render commands generated. Re-run with --execute to render.")


if __name__ == "__main__":
    main()
