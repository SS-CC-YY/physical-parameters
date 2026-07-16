#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
