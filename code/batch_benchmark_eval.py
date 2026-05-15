#!/usr/bin/env python3
"""
Batch benchmark evaluation runner.

This script closes the last manual gap between:
  1. Blender GT export in blender/variants/
  2. model-predicted trajectory CSVs
  3. per-view evaluation JSONs
  4. cross-view consistency aggregation

Expected prediction layout:
  <pred-root>/<version>/<exp_id>/<variant_id>/<camera>/trajectory.csv

Output layout:
  <out-root>/<version>/<exp_id>/<variant_id>/<camera>/eval.json
  <out-root>/<version>/<exp_id>/<variant_id>/cross_view_consistency.json
  <out-root>/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark_evaluation import evaluate
from cross_view_consistency import read_json


CAMERAS = ("CAM_Main", "CAM_Side", "CAM_Top")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--variants-root", default="blender/variants")
    parser.add_argument("--pred-root", required=True,
                        help="Root directory containing predicted trajectory CSVs.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    return parser.parse_args()


def find_gt_triplets(variants_root: Path, versions, experiments=None, variant_ids=None):
    experiments = set(experiments) if experiments else None
    variant_ids = set(variant_ids) if variant_ids else None
    for version in versions:
        version_root = variants_root / version
        if not version_root.exists():
            continue
        for exp_dir in sorted(p for p in version_root.iterdir() if p.is_dir()):
            exp_id = exp_dir.name
            if experiments and exp_id not in experiments:
                continue
            for variant_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                variant_id = variant_dir.name
                if variant_ids and variant_id not in variant_ids:
                    continue
                gt_csvs = sorted(variant_dir.glob("*_trajectory.csv"))
                params_jsons = sorted(variant_dir.glob("*_params.json"))
                if not gt_csvs or not params_jsons:
                    continue
                yield version, exp_id, variant_id, gt_csvs[0], params_jsons[0]


def find_pred_csv(pred_root: Path, version: str, exp_id: str, variant_id: str, camera: str):
    direct = pred_root / version / exp_id / variant_id / camera / "trajectory.csv"
    if direct.exists():
        return direct
    camera_dir = pred_root / version / exp_id / variant_id / camera
    if camera_dir.exists():
        csvs = sorted(p for p in camera_dir.glob("*.csv") if p.name != "prompt.csv")
        if csvs:
            return csvs[0]
    return None


def aggregate_cross_view(eval_json_paths):
    evals = [read_json(str(path)) for path in eval_json_paths]
    estimates = {}
    event_errors = {}
    for path, data in zip(eval_json_paths, evals):
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
            "mean": sum(vals) / len(vals) if vals else None,
            "variance": 0.0 if len(vals) <= 1 else sum((v - (sum(vals) / len(vals))) ** 2 for v in vals) / len(vals),
            "range": max(vals) - min(vals) if vals else None,
        }

    event_consistency = {}
    for key, by_view in event_errors.items():
        vals = list(by_view.values())
        event_consistency[key] = {
            "by_view": by_view,
            "mean_abs_error_s": sum(vals) / len(vals) if vals else None,
            "variance": 0.0 if len(vals) <= 1 else sum((v - (sum(vals) / len(vals))) ** 2 for v in vals) / len(vals),
        }

    return {
        "views": [Path(path).stem for path in eval_json_paths],
        "parameter_cross_view_consistency": param_consistency,
        "event_cross_view_consistency": event_consistency,
    }


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    variants_root = (workspace_root / args.variants_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    out_root = Path(args.out_root).resolve()

    summary_rows = []

    for version, exp_id, variant_id, gt_csv, params_json in find_gt_triplets(
        variants_root, args.versions, args.experiments, args.variants
    ):
        per_view_eval_paths = []
        for camera in args.cameras:
            pred_csv = find_pred_csv(pred_root, version, exp_id, variant_id, camera)
            if pred_csv is None:
                continue

            result = evaluate(str(gt_csv), str(pred_csv), str(params_json))
            eval_dir = out_root / version / exp_id / variant_id / camera
            eval_path = eval_dir / "eval.json"
            write_json(eval_path, result)
            per_view_eval_paths.append(eval_path)

            obj_metrics = result.get("objects", {})
            position_mse = None
            velocity_mse = None
            dtw = None
            if obj_metrics:
                first_obj = next(iter(obj_metrics.values()))
                position_mse = first_obj.get("position_mse")
                velocity_mse = first_obj.get("velocity_mse")
                dtw = first_obj.get("trajectory_dtw")

            summary_rows.append(
                {
                    "version": version,
                    "experiment_id": exp_id,
                    "variant_id": variant_id,
                    "camera": camera,
                    "gt_csv": str(gt_csv),
                    "pred_csv": str(pred_csv),
                    "position_mse": position_mse,
                    "velocity_mse": velocity_mse,
                    "trajectory_dtw": dtw,
                    "param_error_count": len(result.get("param_errors", {})),
                    "event_error_count": len(result.get("event_time_errors_s", {})),
                }
            )

        if len(per_view_eval_paths) >= 2:
            cross_view = aggregate_cross_view(per_view_eval_paths)
            write_json(out_root / version / exp_id / variant_id / "cross_view_consistency.json", cross_view)

    write_csv(out_root / "summary.csv", summary_rows)
    print(f"Saved batch evaluation outputs to: {out_root}")


if __name__ == "__main__":
    main()
