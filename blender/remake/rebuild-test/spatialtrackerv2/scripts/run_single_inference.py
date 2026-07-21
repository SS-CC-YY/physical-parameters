#!/usr/bin/env python3
"""Run one benchmark video in an isolated process using the official models."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialtracker_session import SpatialTrackerSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--track-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--vo-points", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = SpatialTrackerSession(args.track_mode, args.vo_points)
    session.run(
        args.video,
        args.queries,
        args.output,
        args.source_fps,
        args.frame_stride,
    )


if __name__ == "__main__":
    main()
