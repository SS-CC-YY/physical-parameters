#!/usr/bin/env python3
"""
Generate commands for building parameter-variant .blend files and GT CSV/JSON.

Default mode is dry-run. Pass --execute to run Blender scene-generation scripts.
This script does not render videos.
"""

import argparse
import subprocess
from pathlib import Path

from benchmark_variants import PARAM_VARIANTS


SCRIPTS = {
    "v1_A": ("v1", "exp_A_ball_freefall_gravity.py", "baselinev1_A_ball_freefall_gravity"),
    "v1_B": ("v1", "exp_B_ball_bounce_restitution.py", "baselinev1_B_ball_bounce_restitution"),
    "v1_C": ("v1", "exp_C_ball_slide_friction.py", "baselinev1_C_ball_slide_friction"),
    "v1_D": ("v1", "exp_D_ball_pendulum_damping.py", "baselinev1_D_ball_pendulum_damping"),
    "v2_A": ("v2", "exp_A_ball_projectile_hidden_gravity.py", "baselinev2_A_ball_projectile_hidden_gravity"),
    "v2_B": ("v2", "exp_B_ball_repeated_bounce_hidden_restitution.py", "baselinev2_B_ball_repeated_bounce_hidden_restitution"),
    "v2_C": ("v2", "exp_C_block_incline_floor_hidden_friction.py", "baselinev2_C_block_incline_floor_hidden_friction"),
    "v2_D": ("v2", "exp_D_ball_long_pendulum_hidden_damping.py", "baselinev2_D_ball_long_pendulum_hidden_damping"),
    "v2_E": ("v2", "exp_E_ball_tilted_floor_bounce_hidden_restitution.py", "baselinev2_E_ball_tilted_floor_bounce_hidden_restitution"),
    "v2_F": ("v2", "exp_F_forced_pendulum_hidden_damping.py", "baselinev2_F_forced_pendulum_hidden_damping"),
    "v3_A": ("v3", "exp_A_ball_projectile_drag_bounce.py", "baselinev3_A_ball_projectile_drag_bounce"),
    "v3_B": ("v3", "exp_B_block_spring_damper.py", "baselinev3_B_block_spring_damper"),
    "v3_C": ("v3", "exp_C_block_incline_floor_friction.py", "baselinev3_C_block_incline_floor_friction"),
    "v3_D": ("v3", "exp_D_ball_em_damping_pendulum.py", "baselinev3_D_ball_em_damping_pendulum"),
    "v3_E": ("v3", "exp_E_ball_rolling_collision.py", "baselinev3_E_ball_rolling_collision"),
    "v3_F": ("v3", "exp_F_two_ball_collision_drag.py", "baselinev3_F_two_ball_collision_drag"),
    "v3_G": ("v3", "exp_G_block_spatial_friction.py", "baselinev3_G_block_spatial_friction"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--experiments", nargs="+", default=sorted(SCRIPTS))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else root / "variants"
    commands = []
    for exp_id in args.experiments:
        version, script_name, stem = SCRIPTS[exp_id]
        script_path = root / version / script_name
        for variant_id in sorted(PARAM_VARIANTS[exp_id]):
            out_file = output_root / version / exp_id / variant_id / f"{stem}_{variant_id}.blend"
            commands.append([args.blender, "-b", "--python", str(script_path), "--", "--variant-id", variant_id, "--output-file", str(out_file)])

    for cmd in commands:
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        if args.execute:
            subprocess.run(cmd, check=True)

    if not args.execute:
        print(f"\nDry run only: {len(commands)} variant build commands generated. Re-run with --execute to build .blend/GT files.")


if __name__ == "__main__":
    main()
