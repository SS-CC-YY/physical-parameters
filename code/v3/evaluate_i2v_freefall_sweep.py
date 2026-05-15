#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, math
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_PRESETS = {
    "red": [((0, 80, 60), (12, 255, 255)), ((170, 80, 60), (179, 255, 255))],
    "blue": [((90, 80, 60), (130, 255, 255))],
    "green": [((35, 60, 50), (90, 255, 255))],
    "yellow": [((18, 80, 80), (40, 255, 255))],
    "white": [((0, 0, 160), (179, 70, 255))],
}

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def normalize_color_name(text, default="red"):
    if text is None:
        return default
    s = str(text).lower()
    if "red" in s: return "red"
    if "blue" in s: return "blue"
    if "green" in s: return "green"
    if "yellow" in s: return "yellow"
    if "white" in s: return "white"
    return default

def ensure_odd(k):
    if k <= 0:
        return 0
    return k if k % 2 == 1 else k + 1

def build_mask(hsv, color_name):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_PRESETS[color_name]:
        lo = np.array(lo, dtype=np.uint8)
        hi = np.array(hi, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask

def detect_ball(frame_bgr, color_name, min_area=30, blur_ksize=5, open_kernel=5, close_kernel=7, sig_ratio=0.15):
    blur_ksize = ensure_odd(blur_ksize)
    img = frame_bgr.copy()
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color_name)
    if open_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, 0
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    areas = [float(cv2.contourArea(c)) for c in contours]
    if areas[0] < min_area:
        return None, mask, 0
    num_sig = sum(1 for a in areas if a >= max(min_area, sig_ratio * areas[0]))
    c = contours[0]
    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-8:
        return None, mask, num_sig
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    (_, _), radius = cv2.minEnclosingCircle(c)
    return {"cx": cx, "cy": cy, "radius_px": float(radius), "area_px": areas[0]}, mask, num_sig

def interpolate_small_gaps(values, max_gap=5):
    out = values[:]
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        start = i - 1
        j = i
        while j < n and out[j] is None:
            j += 1
        end = j
        gap = end - i
        if start >= 0 and end < n and out[start] is not None and out[end] is not None and gap <= max_gap:
            v0, v1 = out[start], out[end]
            for k in range(1, gap + 1):
                alpha = k / (gap + 1)
                out[start + k] = (1 - alpha) * v0 + alpha * v1
        i = j
    return out

def fit_quadratic(t, y):
    coeff = np.polyfit(t, y, 2)
    a, b, c = [float(v) for v in coeff]
    y_hat = np.polyval(coeff, t)
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"a": a, "b": b, "c": c, "rmse": rmse, "r2": r2, "y_hat": y_hat, "resid": resid}

def fit_linear(t, x):
    coeff = np.polyfit(t, x, 1)
    m, q = [float(v) for v in coeff]
    x_hat = np.polyval(coeff, t)
    resid = x - x_hat
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {"m": m, "q": q, "rmse": rmse, "x_hat": x_hat, "resid": resid}

def locate_video(generated_root: Path, job_id: str):
    matches = list(generated_root.rglob(f"{job_id}.mp4"))
    return matches[0] if matches else None

def build_plots(outdir, t_fit, y_fit_vals, fit_y, x_fit_vals, fit_x, unit_name="D"):
    t_dense = np.linspace(t_fit.min(), t_fit.max(), 400)
    y_dense = fit_y["a"] * t_dense**2 + fit_y["b"] * t_dense + fit_y["c"]

    plt.figure(figsize=(8, 5))
    plt.scatter(t_fit, y_fit_vals, s=22, label="tracked y(t)")
    plt.plot(t_dense, y_dense, linewidth=2, label="quadratic fit")
    plt.xlabel("time (s)")
    plt.ylabel(f"y ({unit_name})")
    plt.title("Free-fall fit in ball-diameter units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit_plot_y.png", dpi=180)
    plt.close()

    x_dense = fit_x["m"] * t_dense + fit_x["q"]
    plt.figure(figsize=(8, 5))
    plt.scatter(t_fit, x_fit_vals, s=22, label="tracked x(t)")
    plt.plot(t_dense, x_dense, linewidth=2, label="linear fit")
    plt.xlabel("time (s)")
    plt.ylabel(f"x ({unit_name})")
    plt.title("Horizontal drift check")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fit_plot_x.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_fit_vals, y_fit_vals, s=18)
    plt.plot(x_fit_vals, y_fit_vals, linewidth=1)
    plt.xlabel(f"x ({unit_name})")
    plt.ylabel(f"y ({unit_name})")
    plt.title("Tracked trajectory")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outdir / "trajectory_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.axhline(0.0, linewidth=1)
    plt.scatter(t_fit, fit_y["resid"], s=22)
    plt.xlabel("time (s)")
    plt.ylabel(f"residual ({unit_name})")
    plt.title("Vertical fit residuals")
    plt.tight_layout()
    plt.savefig(outdir / "residual_plot.png", dpi=180)
    plt.close()

def make_overlay(video_path, outpath, fps, width, height, frame_rows, t0_fit, fit_y, fit_x, D_px, x0_px, ground_y_px=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(outpath), fourcc, fps, (width, height))
    row_map = {r["frame_idx"]: r for r in frame_rows}
    a, b, c = fit_y["a"], fit_y["b"], fit_y["c"]
    m, q = fit_x["m"], fit_x["q"]
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        if ground_y_px is not None:
            cv2.line(vis, (0, int(round(ground_y_px))), (width - 1, int(round(ground_y_px))), (0,255,255), 2)
        r = row_map.get(idx)
        if r is not None:
            if r["cx_px"] is not None and r["cy_px"] is not None:
                cx, cy = int(round(r["cx_px"])), int(round(r["cy_px"]))
                cv2.circle(vis, (cx, cy), 5, (0,255,0), -1)
                cv2.putText(vis, "tracked", (cx + 7, cy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
            tloc = float(r["time_s"]) - t0_fit
            if tloc >= 0:
                x_pred_D = m * tloc + q
                y_pred_D = a * tloc**2 + b * tloc + c
                cx_fit = int(round(x0_px + x_pred_D * D_px))
                ground_ref = height - 1 if ground_y_px is None else ground_y_px
                cy_fit = int(round(ground_ref - y_pred_D * D_px))
                if 0 <= cx_fit < width and 0 <= cy_fit < height:
                    cv2.circle(vis, (cx_fit, cy_fit), 5, (255,0,0), -1)
                    cv2.putText(vis, "fit", (cx_fit + 7, cy_fit + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1)
        writer.write(vis)
        idx += 1
    cap.release()
    writer.release()

def regression_metrics(prompt_vals, est_vals):
    x = np.array(prompt_vals, dtype=float)
    y = np.array(est_vals, dtype=float)
    if len(x) < 2:
        return None
    coeff = np.polyfit(x, y, 1)
    slope, intercept = [float(v) for v in coeff]
    y_hat = np.polyval(coeff, x)
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
    order = np.argsort(x)
    y_sorted = y[order]
    if len(y_sorted) >= 2:
        dy = np.diff(y_sorted)
        nondecreasing = float(np.mean(dy >= -1e-8))
        nonincreasing = float(np.mean(dy <= 1e-8))
        monotonic_score = max(nondecreasing, nonincreasing)
    else:
        monotonic_score = float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2, "pearson": pearson, "monotonic_score": monotonic_score}

def plot_axis_regression(outdir, axis_name, prompt_vals, est_vals, est_label):
    x = np.array(prompt_vals, dtype=float)
    y = np.array(est_vals, dtype=float)
    if len(x) < 2:
        return
    coeff = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = np.polyval(coeff, xx)
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=35)
    plt.plot(xx, yy, linewidth=2)
    plt.xlabel(f"prompted {axis_name}")
    plt.ylabel(est_label)
    plt.title(f"{axis_name} sweep: prompted vs estimated")
    plt.tight_layout()
    plt.savefig(outdir / f"{axis_name}_regression_plot.png", dpi=180)
    plt.close()

def evaluate_job(job, generated_root: Path, outdir: Path, default_color="red", ground_y_px=None, min_area=30, blur_ksize=5, open_kernel=5, close_kernel=7, max_interp_gap=5):
    video = locate_video(generated_root, job["id"])
    if video is None:
        raise FileNotFoundError(f"missing video for job id: {job['id']}")
    color_name = normalize_color_name(job.get("parameters", {}).get("color"), default=default_color)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6 or np.isnan(fps):
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ground_ref = float(ground_y_px) if ground_y_px is not None else float(height - 1)

    rows = []
    first_ball = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det, _, num_sig = detect_ball(frame, color_name, min_area=min_area, blur_ksize=blur_ksize, open_kernel=open_kernel, close_kernel=close_kernel)
        row = {"frame_idx": frame_idx, "time_s": frame_idx / fps, "num_significant_contours": num_sig}
        if det is not None:
            row.update({"found": True, "cx_px_raw": det["cx"], "cy_px_raw": det["cy"], "radius_px": det["radius_px"], "area_px": det["area_px"]})
            if first_ball is None:
                first_ball = det
        else:
            row.update({"found": False, "cx_px_raw": None, "cy_px_raw": None, "radius_px": None, "area_px": None})
        rows.append(row)
        frame_idx += 1
    cap.release()

    if first_ball is None:
        raise RuntimeError(f"no ball detected: {video}")

    D_px = 2.0 * first_ball["radius_px"]
    x0_px = first_ball["cx"]

    cx_list = [r["cx_px_raw"] for r in rows]
    cy_list = [r["cy_px_raw"] for r in rows]
    cx_interp = interpolate_small_gaps(cx_list, max_gap=max_interp_gap)
    cy_interp = interpolate_small_gaps(cy_list, max_gap=max_interp_gap)

    valid_idx, valid_t, valid_x, valid_y = [], [], [], []
    for i, r in enumerate(rows):
        r["cx_px"] = cx_interp[i]
        r["cy_px"] = cy_interp[i]
        if r["cx_px"] is not None and r["cy_px"] is not None:
            xD = (r["cx_px"] - x0_px) / D_px
            yD = (ground_ref - r["cy_px"]) / D_px
            r["x_D"] = xD; r["y_D"] = yD
            valid_idx.append(i); valid_t.append(r["time_s"]); valid_x.append(xD); valid_y.append(yD)
        else:
            r["x_D"] = None; r["y_D"] = None

    if len(valid_t) < 5:
        raise RuntimeError(f"too few valid points: {video}")

    valid_t = np.array(valid_t, dtype=float)
    valid_x = np.array(valid_x, dtype=float)
    valid_y = np.array(valid_y, dtype=float)

    contact_i = None
    for i, r in enumerate(rows):
        if r["cx_px"] is not None and r["cy_px"] is not None and r["radius_px"] is not None:
            if r["cy_px"] + r["radius_px"] >= ground_ref - 2.0:
                contact_i = i
                break

    if contact_i is None:
        fit_mask = np.array([True] * len(valid_t), dtype=bool)
    else:
        fit_mask = np.array([gi < contact_i for gi in valid_idx], dtype=bool)
        if fit_mask.sum() < 5:
            fit_mask[:] = True

    t_fit_all = valid_t[fit_mask]
    x_fit_all = valid_x[fit_mask]
    y_fit_all = valid_y[fit_mask]
    t0_fit = float(t_fit_all[0])
    t_fit = t_fit_all - t0_fit

    fit_y = fit_quadratic(t_fit, y_fit_all)
    fit_x = fit_linear(t_fit, x_fit_all)

    g_hat_D = -2.0 * fit_y["a"]
    v0_hat_D = fit_y["b"]
    h0_hat_D = fit_y["c"]

    detection_rate = sum(int(bool(r["found"])) for r in rows) / len(rows)
    det_rows = [r for r in rows if r["found"] and r["cx_px"] is not None and r["cy_px"] is not None and r["radius_px"] is not None]
    single_object_rate = sum(int(r["num_significant_contours"] <= 1) for r in det_rows) / len(det_rows) if det_rows else 0.0
    in_frame_rate = sum(int(
        r["cx_px"] - r["radius_px"] >= 1.0 and
        r["cx_px"] + r["radius_px"] <= width - 2.0 and
        r["cy_px"] - r["radius_px"] >= 1.0 and
        r["cy_px"] + r["radius_px"] <= height - 2.0
    ) for r in det_rows) / len(det_rows) if det_rows else 0.0

    x_span = float(np.max(x_fit_all) - np.min(x_fit_all)) if len(x_fit_all) > 1 else 0.0
    y_span = float(np.max(y_fit_all) - np.min(y_fit_all)) if len(y_fit_all) > 1 else 0.0
    verticality_ratio = x_span / max(y_span, 1e-8)
    dy = np.diff(y_fit_all)
    monotonic_down_rate = float(np.mean(dy <= 0.01 * max(y_span, 1e-8))) if len(dy) > 0 else 0.0

    penetration_depth_px = 0.0
    for r in det_rows:
        penetration = (r["cy_px"] + r["radius_px"]) - ground_ref
        penetration_depth_px = max(penetration_depth_px, penetration)
    penetration_norm = penetration_depth_px / max(first_ball["radius_px"], 1e-8)

    bounce_height = 0.0
    if contact_i is not None:
        post_y = [yv for gi, yv in zip(valid_idx, valid_y) if gi >= contact_i]
        if len(post_y) >= 2:
            bounce_height = max(post_y) - post_y[0]
    bounce_norm = bounce_height / max(y_span, 1e-8) if y_span > 1e-8 else 0.0

    tracking_score = 0.5 * detection_rate + 0.25 * single_object_rate + 0.25 * in_frame_rate
    shape_score = max(0.0, min(1.0, (fit_y["r2"] - 0.90) / 0.09))
    no_horizontal_score = max(0.0, min(1.0, 1.0 - verticality_ratio / 0.15))
    no_penetration_score = 1.0 if penetration_depth_px <= 2.0 else max(0.0, 1.0 - penetration_norm / 0.5)
    no_bounce_score = 1.0 if bounce_norm <= 0.02 else max(0.0, 1.0 - bounce_norm / 0.2)
    per_video_score = 100.0 * (0.25 * tracking_score + 0.25 * shape_score + 0.20 * no_horizontal_score + 0.15 * no_penetration_score + 0.15 * no_bounce_score)

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "trajectory.csv", rows)
    build_plots(outdir, t_fit, y_fit_all, fit_y, x_fit_all, fit_x, unit_name="D")
    make_overlay(video, outdir / "fitted_overlay.mp4", fps, width, height, rows, t0_fit, fit_y, fit_x, D_px, x0_px, ground_ref)

    ev = {
        "job_id": job["id"],
        "axis_name": job["axis_name"],
        "axis_value_prompt": job["axis_value"],
        "prompt_style": job["prompt_style"],
        "video_path": str(video),
        "D_px_first_frame": D_px,
        "fit_recovered_proxy": {"g_hat_D": g_hat_D, "v0_hat_D": v0_hat_D, "h0_hat_D": h0_hat_D},
        "fit_y": {"a": fit_y["a"], "b": fit_y["b"], "c": fit_y["c"], "rmse": fit_y["rmse"], "r2": fit_y["r2"]},
        "fit_x": {"m": fit_x["m"], "q": fit_x["q"], "rmse": fit_x["rmse"]},
        "constraints": {
            "detection_rate": detection_rate,
            "single_object_rate": single_object_rate,
            "in_frame_rate": in_frame_rate,
            "verticality_ratio": verticality_ratio,
            "monotonic_down_rate": monotonic_down_rate,
            "penetration_depth_px": penetration_depth_px,
            "penetration_norm": penetration_norm,
            "bounce_norm": bounce_norm,
        },
        "per_video_score_100": per_video_score,
        "note": "本视频评分主要检查是否是干净的自由落体。物理量理解的核心评价在 axis 级别的线性关联。"
    }
    save_json(outdir / "evaluation.json", ev)
    return ev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--generated-root", required=True)
    ap.add_argument("--eval-outdir", required=True)
    ap.add_argument("--ground-y-px", type=float, default=None)
    ap.add_argument("--default-color", default="red", choices=list(COLOR_PRESETS.keys()))
    args = ap.parse_args()

    manifest = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    generated_root = Path(args.generated_root)
    eval_outdir = Path(args.eval_outdir)
    eval_outdir.mkdir(parents=True, exist_ok=True)

    per_video_rows = []
    axis_groups = {}
    errors = []

    for idx, job in enumerate(manifest, 1):
        try:
            outdir = eval_outdir / job["axis_name"] / job["prompt_style"] / job["id"]
            ev = evaluate_job(job, generated_root=generated_root, outdir=outdir, default_color=args.default_color, ground_y_px=args.ground_y_px)
            row = {
                "job_id": ev["job_id"],
                "axis_name": ev["axis_name"],
                "axis_value_prompt": ev["axis_value_prompt"],
                "prompt_style": ev["prompt_style"],
                "g_hat_D": ev["fit_recovered_proxy"]["g_hat_D"],
                "v0_hat_D": ev["fit_recovered_proxy"]["v0_hat_D"],
                "h0_hat_D": ev["fit_recovered_proxy"]["h0_hat_D"],
                "quadratic_r2": ev["fit_y"]["r2"],
                "verticality_ratio": ev["constraints"]["verticality_ratio"],
                "detection_rate": ev["constraints"]["detection_rate"],
                "single_object_rate": ev["constraints"]["single_object_rate"],
                "in_frame_rate": ev["constraints"]["in_frame_rate"],
                "penetration_norm": ev["constraints"]["penetration_norm"],
                "bounce_norm": ev["constraints"]["bounce_norm"],
                "per_video_score_100": ev["per_video_score_100"],
                "video_path": ev["video_path"],
            }
            per_video_rows.append(row)
            axis_groups.setdefault((ev["axis_name"], ev["prompt_style"]), []).append(row)
            print(f"[{idx}/{len(manifest)}] OK {job['id']} score={row['per_video_score_100']:.2f}")
        except Exception as e:
            errors.append({"job_id": job["id"], "error": f"{type(e).__name__}: {e}"})
            print(f"[{idx}/{len(manifest)}] ERR {job['id']} -> {type(e).__name__}: {e}")

    write_csv(eval_outdir / "summary.csv", per_video_rows)

    axis_summary_rows = []
    axis_report = {}

    for (axis_name, prompt_style), rows in axis_groups.items():
        rows = sorted(rows, key=lambda r: float(r["axis_value_prompt"]))
        prompt_vals = [float(r["axis_value_prompt"]) for r in rows]
        if axis_name == "g":
            est_vals = [float(r["g_hat_D"]) for r in rows]
            est_label = "estimated curvature proxy g_hat_D"
        elif axis_name == "v0":
            est_vals = [float(r["v0_hat_D"]) for r in rows]
            est_label = "estimated initial velocity proxy v0_hat_D"
        else:
            est_vals = [float(r["h0_hat_D"]) for r in rows]
            est_label = "estimated initial height proxy h0_hat_D"

        reg = regression_metrics(prompt_vals, est_vals)
        group_outdir = eval_outdir / axis_name / prompt_style
        group_outdir.mkdir(parents=True, exist_ok=True)
        plot_axis_regression(group_outdir, axis_name, prompt_vals, est_vals, est_label)

        mean_video_score = float(np.mean([r["per_video_score_100"] for r in rows])) if rows else 0.0
        mean_r2 = float(np.mean([r["quadratic_r2"] for r in rows])) if rows else 0.0
        mean_verticality = float(np.mean([r["verticality_ratio"] for r in rows])) if rows else 0.0
        mean_detection = float(np.mean([r["detection_rate"] for r in rows])) if rows else 0.0

        linearity_r2 = 0.0 if (reg is None or math.isnan(reg["r2"])) else max(0.0, min(1.0, reg["r2"]))
        pearson_abs = 0.0 if (reg is None or math.isnan(reg["pearson"])) else abs(reg["pearson"])
        mono = 0.0 if (reg is None or math.isnan(reg["monotonic_score"])) else reg["monotonic_score"]
        fit_quality_score = max(0.0, min(1.0, (mean_r2 - 0.90) / 0.09))
        constraint_score = (
            0.4 * mean_detection +
            0.3 * max(0.0, min(1.0, 1.0 - mean_verticality / 0.15)) +
            0.3 * float(np.mean([1.0 if (r["penetration_norm"] <= 0.1 and r["bounce_norm"] <= 0.05) else 0.0 for r in rows]))
        )
        axis_score = 100.0 * (0.40 * linearity_r2 + 0.20 * pearson_abs + 0.15 * mono + 0.15 * fit_quality_score + 0.10 * constraint_score)

        axis_item = {
            "axis_name": axis_name,
            "prompt_style": prompt_style,
            "count": len(rows),
            "linearity_r2": linearity_r2,
            "pearson_abs": pearson_abs,
            "monotonic_score": mono,
            "mean_per_video_score_100": mean_video_score,
            "mean_quadratic_r2": mean_r2,
            "mean_verticality_ratio": mean_verticality,
            "mean_detection_rate": mean_detection,
            "axis_score_100": axis_score,
        }
        if reg is not None:
            axis_item["regression_slope"] = reg["slope"]
            axis_item["regression_intercept"] = reg["intercept"]

        axis_summary_rows.append(axis_item)
        axis_report[f"{axis_name}::{prompt_style}"] = axis_item
        save_json(group_outdir / "axis_summary.json", axis_item)

    write_csv(eval_outdir / "axis_summary.csv", axis_summary_rows)
    final_report = {
        "axis_report": axis_report,
        "errors": errors,
        "scoring_explanation": {
            "per_video": "检查该视频是否是干净的自由落体：高 tracking、低横向漂移、低穿模/反弹、较高二次拟合 R²。",
            "axis_level": "核心看 prompt 参数与估计 proxy 的线性关联。axis_score=40%线性R²+20%|Pearson|+15%单调性+15%平均拟合质量+10%平均约束质量。"
        }
    }
    save_json(eval_outdir / "final_report.json", final_report)
    print(f"saved: {eval_outdir}")

if __name__ == "__main__":
    main()
