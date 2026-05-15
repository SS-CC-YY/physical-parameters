#!/usr/bin/env python3
"""Aggregate per-view evaluation JSON files into L4 consistency metrics."""

import argparse
import json
import statistics
from pathlib import Path


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", nargs="+", required=True, help="per-view outputs from benchmark_evaluation.py")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    evals = [read_json(path) for path in args.eval_json]
    estimates = {}
    event_errors = {}
    for path, data in zip(args.eval_json, evals):
        view_name = Path(path).stem
        for key, item in data.get("param_errors", {}).items():
            estimates.setdefault(key, {})[view_name] = item["estimate"]
        for key, value in data.get("event_time_errors_s", {}).items():
            event_errors.setdefault(key, {})[view_name] = value

    param_consistency = {}
    for key, by_view in estimates.items():
        vals = list(by_view.values())
        param_consistency[key] = {
            "by_view": by_view,
            "mean": statistics.fmean(vals) if vals else None,
            "variance": statistics.pvariance(vals) if len(vals) > 1 else 0.0,
            "range": max(vals) - min(vals) if vals else None,
        }

    event_consistency = {}
    for key, by_view in event_errors.items():
        vals = list(by_view.values())
        event_consistency[key] = {
            "by_view": by_view,
            "mean_abs_error_s": statistics.fmean(vals) if vals else None,
            "variance": statistics.pvariance(vals) if len(vals) > 1 else 0.0,
        }

    result = {
        "views": [Path(path).stem for path in args.eval_json],
        "parameter_cross_view_consistency": param_consistency,
        "event_cross_view_consistency": event_consistency,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
