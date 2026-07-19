#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction import run_reconstruction_batch  # noqa: E402
from remake_benchmark.reconstruction.mvp import load_config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    workspace = CODE_ROOT.parent
    parser = argparse.ArgumentParser(
        description="Validate fast object-centric reconstruction on generated benchmark MP4 files."
    )
    parser.add_argument("--workspace-root", type=Path, default=workspace)
    parser.add_argument("--videos", type=Path, default=workspace / "rebuild-test" / "test-videos" / "videos")
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "rebuild-test" / "outputs" / "seedance20_reconstruction_mvp",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CODE_ROOT / "configs" / "reconstruction" / "object_centric_mvp_v1.yaml",
    )
    parser.add_argument("--workers", type=int, default=1, help="Use process workers; 4 is suitable for the 20-video batch.")
    parser.add_argument("--overlay-count", type=int, default=5, help="Create one inspectable overlay per scene by default.")
    parser.add_argument("--limit", type=int, default=None, help="Optional deterministic smoke-test limit.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    aggregate = run_reconstruction_batch(
        workspace_root=args.workspace_root.resolve(),
        videos_dir=args.videos.resolve(),
        output_root=args.output.resolve(),
        config=config,
        workers=args.workers,
        overlay_count=args.overlay_count,
        limit=args.limit,
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
