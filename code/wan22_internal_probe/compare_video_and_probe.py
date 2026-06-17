#!/usr/bin/env python3
"""Compare output-video recovery with internal-state probe performance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-summary", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def corrcoef(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def best_probe(rows: list[dict[str, str]]) -> dict[str, Any]:
    def score(row: dict[str, str]) -> float:
        value = finite_float(row.get("r2", ""))
        return -1e9 if value is None else value

    if not rows:
        return {}
    return max(rows, key=score)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    video_rows = read_csv(args.video_summary)
    probe_rows = read_csv(args.probe_summary)
    targets: list[float] = []
    estimates: list[float] = []
    rel_errors: list[float] = []
    for row in video_rows:
        target = finite_float(row.get("target_param_value", ""))
        estimate = finite_float(row.get("estimate_value", ""))
        rel_error = finite_float(row.get("rel_error", ""))
        if target is not None and estimate is not None:
            targets.append(target)
            estimates.append(estimate)
        if rel_error is not None:
            rel_errors.append(rel_error)

    summary = {
        "video_num_rows": len(video_rows),
        "video_target_estimate_pearson": corrcoef(targets, estimates),
        "video_mean_relative_error": float(np.mean(rel_errors)) if rel_errors else float("nan"),
        "video_median_relative_error": float(np.median(rel_errors)) if rel_errors else float("nan"),
        "best_internal_probe": best_probe(probe_rows),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "comparison_summary.json"
    summary = json_safe(summary)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
