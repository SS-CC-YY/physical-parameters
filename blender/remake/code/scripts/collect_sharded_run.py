from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.orchestration.sharding import collect_sharded_run  # noqa: E402
from remake_benchmark.core.errors import ConfigError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate four-GPU shards and hard-link them into one canonical run directory."
    )
    parser.add_argument("run_root", type=Path, help="Parent output directory containing control/ and shards/.")
    args = parser.parse_args()
    try:
        summary = collect_sharded_run(args.run_root)
    except ConfigError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
