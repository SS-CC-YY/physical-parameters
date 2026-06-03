#!/usr/bin/env python3
"""V3 wrapper around the V2 visualization tool."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2_wan22_i2v_full"))

from visualize_v2_eval import main  # noqa: E402


if __name__ == "__main__":
    main()

