#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import cv2, numpy as np

COLOR_PRESETS = {
    "red": [((0, 80, 60), (12, 255, 255)), ((170, 80, 60), (179, 255, 255))],
    "blue": [((90, 80, 60), (130, 255, 255))],
    "green": [((35, 60, 50), (90, 255, 255))],
    "yellow": [((18, 80, 80), (40, 255, 255))],
    "white": [((0, 0, 160), (179, 70, 255))],
}

def build_mask(hsv, color_name):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_PRESETS[color_name]:
        lo = np.array(lo, dtype=np.uint8)
        hi = np.array(hi, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask

def detect_ball(img_bgr, color_name="red", min_area=30):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color_name)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    area = float(cv2.contourArea(c))
    if area < min_area:
        return None, mask
    M = cv2.moments(c)
    if abs(M["m00"]) < 1e-8:
        return None, mask
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    (_, _), radius = cv2.minEnclosingCircle(c)
    return {"cx": cx, "cy": cy, "radius_px": float(radius), "area_px": area}, mask

def enhance_image(img_bgr, color_name="red", saturation_gain=1.25, contrast_gain=1.06, sharpness_gain=0.2):
    img = cv2.convertScaleAbs(img_bgr, alpha=contrast_gain, beta=0)
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.0 + sharpness_gain, blur, -sharpness_gain, 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = build_mask(hsv.astype(np.uint8), color_name)
    hsv[:, :, 1][mask > 0] = np.clip(hsv[:, :, 1][mask > 0] * saturation_gain, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--color", default="red", choices=list(COLOR_PRESETS.keys()))
    ap.add_argument("--saturation-gain", type=float, default=1.25)
    ap.add_argument("--contrast-gain", type=float, default=1.06)
    ap.add_argument("--sharpness-gain", type=float, default=0.20)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    enhanced, mask = enhance_image(img, args.color, args.saturation_gain, args.contrast_gain, args.sharpness_gain)
    det, _ = detect_ball(enhanced, args.color)

    enhanced_path = outdir / "anchor_enhanced.png"
    mask_path = outdir / "anchor_mask.png"
    preview_path = outdir / "anchor_preview.png"
    meta_path = outdir / "anchor_metadata.json"

    cv2.imwrite(str(enhanced_path), enhanced)
    cv2.imwrite(str(mask_path), mask)

    preview = enhanced.copy()
    if det is not None:
        cx, cy = int(round(det["cx"])), int(round(det["cy"]))
        rr = int(round(det["radius_px"]))
        cv2.circle(preview, (cx, cy), max(rr, 3), (0, 255, 0), 2)
        cv2.circle(preview, (cx, cy), 3, (255, 0, 0), -1)
        cv2.putText(preview, f"r={det['radius_px']:.2f}px", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(str(preview_path), preview)

    meta = {
        "source_image": str(Path(args.image)),
        "enhanced_image": str(enhanced_path),
        "color": args.color,
        "image_width": int(enhanced.shape[1]),
        "image_height": int(enhanced.shape[0]),
        "ball_detection": det,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {enhanced_path}")
    print(f"saved: {meta_path}")

if __name__ == "__main__":
    main()
