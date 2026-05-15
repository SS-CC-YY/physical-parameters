#!/usr/bin/env python3
"""
Trajectory-level benchmark evaluation.

Inputs:
  --gt-csv      ground-truth trajectory exported by blender/benchmark_io.py
  --pred-csv    model/video-derived trajectory with matching columns
  --params-json ground-truth params JSON exported by blender/benchmark_io.py

The script reports trajectory errors, event timing deltas, cross-view parameter
variance when given multiple prediction files, and lightweight parameter fits
for the experiments with stable closed-form estimates.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_params(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_series(rows):
    series = defaultdict(list)
    for row in rows:
        obj = row["object"]
        series[obj].append(
            {
                "frame": int(row["frame"]),
                "time_s": float(row["time_s"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            }
        )
    for values in series.values():
        values.sort(key=lambda item: item["frame"])
    return dict(series)


def align_by_frame(gt, pred):
    pred_by_frame = {row["frame"]: row for row in pred}
    pairs = []
    for gt_row in gt:
        pred_row = pred_by_frame.get(gt_row["frame"])
        if pred_row:
            pairs.append((gt_row, pred_row))
    return pairs


def mse(values):
    values = list(values)
    return sum(v * v for v in values) / len(values) if values else None


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def position_mse(gt, pred):
    pairs = align_by_frame(gt, pred)
    return mean(
        (g["x"] - p["x"]) ** 2 + (g["y"] - p["y"]) ** 2 + (g["z"] - p["z"]) ** 2
        for g, p in pairs
    )


def velocities(series):
    out = []
    for a, b in zip(series, series[1:]):
        dt = b["time_s"] - a["time_s"]
        if dt <= 1e-9:
            continue
        out.append(
            {
                "time_s": 0.5 * (a["time_s"] + b["time_s"]),
                "vx": (b["x"] - a["x"]) / dt,
                "vy": (b["y"] - a["y"]) / dt,
                "vz": (b["z"] - a["z"]) / dt,
            }
        )
    return out


def velocity_mse(gt, pred):
    return position_mse(
        [{"frame": i, "x": v["vx"], "y": v["vy"], "z": v["vz"]} for i, v in enumerate(velocities(gt))],
        [{"frame": i, "x": v["vx"], "y": v["vy"], "z": v["vz"]} for i, v in enumerate(velocities(pred))],
    )


def dtw_distance(gt, pred):
    n, m = len(gt), len(pred)
    if not n or not m:
        return None
    prev = [float("inf")] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [float("inf")] * (m + 1)
        for j in range(1, m + 1):
            a, b = gt[i - 1], pred[j - 1]
            cost = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)
            cur[j] = cost + min(cur[j - 1], prev[j], prev[j - 1])
        prev = cur
    return prev[m] / (n + m)


def polyfit2(xs, ys):
    n = len(xs)
    sx = sum(xs)
    sx2 = sum(x * x for x in xs)
    sx3 = sum(x**3 for x in xs)
    sx4 = sum(x**4 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2y = sum(x * x * y for x, y in zip(xs, ys))
    a = [[sx4, sx3, sx2], [sx3, sx2, sx], [sx2, sx, n]]
    b = [sx2y, sxy, sy]
    return solve3(a, b)


def solve3(a, b):
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        if abs(div) < 1e-12:
            return None
        for k in range(col, 4):
            m[col][k] /= div
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            for k in range(col, 4):
                m[r][k] -= factor * m[col][k]
    return [m[i][3] for i in range(3)]


def fit_basic_params(exp_id, series, params):
    hidden = {}
    first_obj = next(iter(series.values()), [])
    if len(first_obj) < 5:
        return hidden
    times = [r["time_s"] - first_obj[0]["time_s"] for r in first_obj]

    if exp_id in {"v1_A", "v2_A"}:
        coeff = polyfit2(times, [r["z"] for r in first_obj])
        if coeff:
            hidden["g_hidden"] = -2.0 * coeff[0]

    if exp_id in {"v1_C", "v2_C"}:
        coeff = polyfit2(times, [r["x"] for r in first_obj])
        g = params["known_params"].get("g_known", params["all_physics"].get("g_known"))
        if coeff and g:
            hidden["mu_hidden"] = max(0.0, -2.0 * coeff[0] / g)

    if exp_id == "v3_G":
        x_boundary = params["all_physics"]["x_boundary"]
        g = params["known_params"]["g_known"]
        left = [r for r in first_obj if r["x"] <= x_boundary]
        right = [r for r in first_obj if r["x"] >= x_boundary]
        if right:
            max_x = max(r["x"] for r in right)
            right = [r for r in right if r["x"] < max_x - 1e-5]
        if len(left) >= 5:
            t0 = left[0]["time_s"]
            coeff = polyfit2([r["time_s"] - t0 for r in left], [r["x"] for r in left])
            if coeff:
                hidden["mu_1_hidden"] = max(0.0, -2.0 * coeff[0] / g)
        if len(right) >= 5:
            t0 = right[0]["time_s"]
            coeff = polyfit2([r["time_s"] - t0 for r in right], [r["x"] for r in right])
            if coeff:
                hidden["mu_2_hidden"] = max(0.0, -2.0 * coeff[0] / g)

    return hidden


def event_times(series):
    events = {}
    for obj, rows in series.items():
        if rows:
            events[f"{obj}_min_z_time"] = min(rows, key=lambda r: r["z"])["time_s"]
    if len(series) == 2:
        names = sorted(series)
        a, b = series[names[0]], series[names[1]]
        pairs = align_by_frame(a, b)
        if pairs:
            events["two_object_closest_time"] = min(
                pairs,
                key=lambda pair: (pair[0]["x"] - pair[1]["x"]) ** 2 + (pair[0]["y"] - pair[1]["y"]) ** 2 + (pair[0]["z"] - pair[1]["z"]) ** 2,
            )[0]["time_s"]
    return events


def evaluate(gt_csv, pred_csv, params_json):
    params = read_params(params_json)
    gt = to_series(read_rows(gt_csv))
    pred = to_series(read_rows(pred_csv))
    objects = sorted(set(gt) & set(pred))
    object_metrics = {}
    for obj in objects:
        object_metrics[obj] = {
            "position_mse": position_mse(gt[obj], pred[obj]),
            "velocity_mse": velocity_mse(gt[obj], pred[obj]),
            "trajectory_dtw": dtw_distance(gt[obj], pred[obj]),
        }
    fit = fit_basic_params(params["experiment_id"], pred, params)
    param_errors = {}
    for key, estimate in fit.items():
        truth = params["hidden_params"].get(key)
        if truth is not None:
            param_errors[key] = {"estimate": estimate, "truth": truth, "abs_error": abs(estimate - truth), "rel_error": abs(estimate - truth) / abs(truth) if abs(truth) > 1e-12 else None}
    gt_events = event_times(gt)
    pred_events = event_times(pred)
    event_errors = {key: abs(pred_events[key] - value) for key, value in gt_events.items() if key in pred_events}
    return {"experiment_id": params["experiment_id"], "objects": object_metrics, "param_errors": param_errors, "event_time_errors_s": event_errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    result = evaluate(args.gt_csv, args.pred_csv, args.params_json)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
