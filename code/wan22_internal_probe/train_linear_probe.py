#!/usr/bin/env python3
"""Train block-wise ridge probes from captured Wan2.2 activation features."""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-key", default="target_param_value")
    parser.add_argument("--group-key", default="variant")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top-blocks", type=int, default=8)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train_n = (x_train - mean) / std
    x_test_n = (x_test - mean) / std
    x_aug = np.concatenate([np.ones((x_train_n.shape[0], 1)), x_train_n], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    weights = np.linalg.pinv(xtx + reg) @ x_aug.T @ y_train
    x_test_aug = np.concatenate([np.ones((x_test_n.shape[0], 1)), x_test_n], axis=1)
    return x_test_aug @ weights


def load_block_features(feature_file: Path) -> dict[str, np.ndarray]:
    data = np.load(feature_file, allow_pickle=False)
    features = data["features"].astype(np.float64)
    records = json.loads(str(data["records_json"]))
    by_block: dict[str, list[np.ndarray]] = {}
    for idx, record in enumerate(records):
        block = record["block_label"]
        dim = int(record.get("feature_dim", features.shape[1]))
        by_block.setdefault(block, []).append(features[idx, :dim])
    aggregated: dict[str, np.ndarray] = {}
    for block, vectors in by_block.items():
        max_dim = max(vec.shape[0] for vec in vectors)
        mat = np.zeros((len(vectors), max_dim), dtype=np.float64)
        for row, vec in enumerate(vectors):
            mat[row, : vec.shape[0]] = vec
        aggregated[block] = mat.mean(axis=0)
    return aggregated


def collect_dataset(jobs: list[dict[str, Any]], features_dir: Path, target_key: str) -> tuple[dict[str, list[np.ndarray]], np.ndarray, list[str], list[str]]:
    block_to_x: dict[str, list[np.ndarray]] = {}
    y_values: list[float] = []
    job_ids: list[str] = []
    groups: list[str] = []
    for job in jobs:
        feature_file = features_dir / f"{job['job_id']}.npz"
        if not feature_file.exists():
            continue
        block_features = load_block_features(feature_file)
        if not block_features:
            continue
        y = float(job[target_key])
        y_values.append(y)
        job_ids.append(str(job["job_id"]))
        groups.append(str(job.get("variant", job["job_id"])))
        for block, vector in block_features.items():
            block_to_x.setdefault(block, []).append(vector)
    return block_to_x, np.asarray(y_values, dtype=np.float64), job_ids, groups


def align_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    max_dim = max(vec.shape[0] for vec in vectors)
    mat = np.zeros((len(vectors), max_dim), dtype=np.float64)
    for idx, vec in enumerate(vectors):
        mat[idx, : vec.shape[0]] = vec
    return mat


def leave_group_out_predictions(x: np.ndarray, y: np.ndarray, groups: list[str], alpha: float) -> np.ndarray:
    unique_groups = sorted(set(groups))
    pred = np.full_like(y, fill_value=np.nan, dtype=np.float64)
    groups_arr = np.asarray(groups)
    for group in unique_groups:
        test_mask = groups_arr == group
        train_mask = ~test_mask
        if int(train_mask.sum()) < 2:
            continue
        pred[test_mask] = ridge_fit_predict(x[train_mask], y[train_mask], x[test_mask], alpha)
    return pred


def summarize_block(block: str, x: np.ndarray, y: np.ndarray, groups: list[str], alpha: float) -> dict[str, Any]:
    pred = leave_group_out_predictions(x, y, groups, alpha)
    valid = ~np.isnan(pred)
    if int(valid.sum()) < 2:
        return {
            "block": block,
            "n": int(len(y)),
            "feature_dim": int(x.shape[1]),
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    yv = y[valid]
    pv = pred[valid]
    return {
        "block": block,
        "n": int(valid.sum()),
        "feature_dim": int(x.shape[1]),
        "r2": r2_score(yv, pv),
        "mae": float(np.mean(np.abs(yv - pv))),
        "rmse": float(math.sqrt(float(np.mean((yv - pv) ** 2)))),
        "pearson": corrcoef(yv, pv),
        "spearman": corrcoef(rankdata(yv), rankdata(pv)),
    }


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
    jobs = read_manifest(args.manifest)
    block_to_x, y, job_ids, groups = collect_dataset(jobs, args.features_dir, args.target_key)
    if len(y) < 4:
        raise RuntimeError(f"too few usable feature files: {len(y)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for block, vectors in sorted(block_to_x.items()):
        if len(vectors) != len(y):
            continue
        x = align_matrix(vectors)
        rows.append(summarize_block(block, x, y, groups, args.alpha))

    rows.sort(key=lambda row: (-999.0 if math.isnan(float(row["r2"])) else -float(row["r2"]), row["block"]))
    summary_csv = args.outdir / "probe_summary.csv"
    fieldnames = ["block", "n", "feature_dim", "r2", "mae", "rmse", "pearson", "spearman"]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0] if rows else {}
    (args.outdir / "best_probe.json").write_text(json.dumps(json_safe(best), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.outdir / "dataset_meta.json").write_text(
        json.dumps(
            json_safe(
                {
                "num_jobs": len(job_ids),
                "target_key": args.target_key,
                "group_key": args.group_key,
                "alpha": args.alpha,
                "groups": sorted(set(groups)),
                "target_values": sorted(set(float(v) for v in y)),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_csv}")
    print(json.dumps(json_safe(best), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
