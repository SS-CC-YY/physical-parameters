#!/usr/bin/env python3
"""Patch official Wan2.2 to use PyTorch SDPA when flash-attn is unavailable.

Wan2.2's model.py imports `flash_attention` directly, so the fallback function
in wan/modules/attention.py is not used.  This patch changes the import to:

    from .attention import attention as flash_attention

The fallback is slower and may use more memory, but it is useful for smoke tests
on servers where flash-attn wheels cannot be downloaded or compiled.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OLD = "from .attention import flash_attention"
NEW = "from .attention import attention as flash_attention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-repo", type=Path, required=True, help="Path to official Wan2.2 repository.")
    parser.add_argument("--restore", action="store_true", help="Restore model.py from model.py.bak_phase1 if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_py = args.wan_repo.expanduser().resolve() / "wan" / "modules" / "model.py"
    backup = model_py.with_name("model.py.bak_phase1")
    if not model_py.exists():
        raise FileNotFoundError(model_py)

    if args.restore:
        if not backup.exists():
            raise FileNotFoundError(f"No backup found: {backup}")
        shutil.copy2(backup, model_py)
        print(f"restored: {model_py}")
        return

    text = model_py.read_text(encoding="utf-8")
    if NEW in text:
        print(f"already patched: {model_py}")
        return
    if OLD not in text:
        raise RuntimeError(f"Expected import not found in {model_py}")
    if not backup.exists():
        shutil.copy2(model_py, backup)
        print(f"backup: {backup}")
    model_py.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched: {model_py}")


if __name__ == "__main__":
    main()
