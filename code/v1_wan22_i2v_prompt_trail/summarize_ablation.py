#!/usr/bin/env python3
"""Summarize V1 prompt/trail ablation metrics across conditioning modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def corrcoef(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def summarize_group(rows: list[dict[str, str]]) -> dict[str, object]:
    rel_errors: list[float] = []
    abs_errors: list[float] = []
    targets: list[float] = []
    estimates: list[float] = []
    detection_rates: list[float] = []
    for row in rows:
        rel = finite_float(row.get("rel_error", ""))
        abs_err = finite_float(row.get("abs_error", ""))
        target = finite_float(row.get("target_param_value", ""))
        estimate = finite_float(row.get("estimate_value", ""))
        det = finite_float(row.get("detection_rate", ""))
        if rel is not None:
            rel_errors.append(rel)
        if abs_err is not None:
            abs_errors.append(abs_err)
        if target is not None and estimate is not None:
            targets.append(target)
            estimates.append(estimate)
        if det is not None:
            detection_rates.append(det)
    return {
        "n": len(rows),
        "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else None,
        "median_relative_error": float(np.median(rel_errors)) if rel_errors else None,
        "mean_absolute_error": float(np.mean(abs_errors)) if abs_errors else None,
        "target_estimate_pearson": corrcoef(targets, estimates),
        "mean_detection_rate": float(np.mean(detection_rates)) if detection_rates else None,
    }


def main() -> None:
    args = parse_args()
    summary: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    for mode in args.modes:
        rows = read_rows(args.out_root / mode / "eval" / "summary.csv")
        mode_summary: dict[str, object] = {"all": summarize_group(rows), "experiments": {}}
        experiments = sorted({row.get("experiment", "") for row in rows if row.get("experiment")})
        for experiment in experiments:
            exp_rows = [row for row in rows if row.get("experiment") == experiment]
            exp_summary = summarize_group(exp_rows)
            mode_summary["experiments"][experiment] = exp_summary
            csv_rows.append({"mode": mode, "experiment": experiment, **exp_summary})
        summary[mode] = mode_summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fieldnames = [
        "mode",
        "experiment",
        "n",
        "mean_relative_error",
        "median_relative_error",
        "mean_absolute_error",
        "target_estimate_pearson",
        "mean_detection_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {args.output}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
