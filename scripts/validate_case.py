#!/usr/bin/env python3
"""Validate one segmentation case against a running sam3-server.

Case directory layout:
  case_dir/
    image.png       input image (any format PIL can read; name configurable)
    mask_ref.png    reference binary mask (0/255), any size — resized to output
    case.json       prompts and expectations (see below)

case.json:
  {
    "type": "segment" | "detect" | "track",   // default "segment"
    "image": "image.png",                     // default "image.png"
    "mask_ref": "mask_ref.png",               // default "mask_ref.png"
    "points": [{"x": 315.0, "y": 250.0, "label": 1}, ...],
    "box": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
    "text_prompt": "a cat",                   // detect only (SAM3)
    "score_threshold": null,                 // detect: override score threshold
    "nms_threshold": null,                   // detect: override NMS threshold
    "multimask": false,                       // segment only
    "mask_index": 0,                          // -1 = pick best IoU among returned
    "aggregate": false,                       // true = compare UNION of all masks vs ref
    "threshold": 0.9                          // PASS iff iou >= t and pixel_match >= t
  }

Metrics (on binarized masks):
  IoU          = intersection / union
  pixel_match  = fraction of pixels equal (both 0 or both 255)

Exit code 0 on PASS, 1 on FAIL.

Usage:
  uv run python scripts/validate_case.py <server_url> <case_dir> [--threshold 0.9]
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


def http_post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read().decode())


def http_get(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.status, resp.read()


def load_case(case_dir):
    with open(os.path.join(case_dir, "case.json")) as f:
        case = json.load(f)
    image_name = case.get("image", "image.png")
    ref_name = case.get("mask_ref", "mask_ref.png")
    image_path = os.path.join(case_dir, image_name)
    ref_path = os.path.join(case_dir, ref_name)
    if not os.path.exists(image_path):
        raise RuntimeError(f"missing image file: {image_path}")
    if not os.path.exists(ref_path):
        raise RuntimeError(f"missing reference mask file: {ref_path}")
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    ref = np.asarray(Image.open(ref_path).convert("L"), dtype=np.uint8)
    return case, image_b64, ref


def mask_from_json(m):
    raw = base64.b64decode(m["mask_base64"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(m["height"], m["width"])


def run_segment(server_url, case, image_b64):
    body = {"image_base64": image_b64}
    if case.get("points"):
        body["points"] = case["points"]
    if case.get("box"):
        body["box"] = case["box"]
    if case.get("multimask"):
        body["multimask"] = case["multimask"]
    resp = http_post(server_url + "/segment", body)
    return [mask_from_json(m) for m in resp.get("masks", [])], resp.get("time_ms")


def run_detect(server_url, case, image_b64):
    body = {"image_base64": image_b64, "text_prompt": case["text_prompt"]}
    if case.get("score_threshold") is not None:
        body["score_threshold"] = case["score_threshold"]
    if case.get("nms_threshold") is not None:
        body["nms_threshold"] = case["nms_threshold"]
    resp = http_post(server_url + "/detect", body)
    return [mask_from_json(d["mask"]) for d in resp.get("detections", [])], resp.get("time_ms")


def run_track(server_url, case, image_b64):
    resp = http_post(server_url + "/track/new", {})
    sess_id = resp["id"]
    try:
        body = {"id": sess_id, "image_base64": image_b64}
        if case.get("points"):
            body["points"] = case["points"]
        if case.get("box"):
            body["box"] = case["box"]
        http_post(server_url + "/track/instances", body)
        resp = http_post(server_url + "/track/frame", {"id": sess_id, "image_base64": image_b64})
        return [mask_from_json(d["mask"]) for d in resp.get("detections", [])], resp.get("time_ms")
    finally:
        http_post(server_url + "/track/reset", {"id": sess_id})


def metrics(out, ref):
    if out.shape != ref.shape:
        ref = np.asarray(
            Image.fromarray(ref).resize((out.shape[1], out.shape[0]), Image.NEAREST),
            dtype=np.uint8,
        )
    a = (out > 0).astype(np.uint8)
    b = (ref > 0).astype(np.uint8)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    iou = inter / union if union > 0 else (1.0 if inter == 0 else 0.0)
    pixel_match = float((a == b).mean())
    return iou, pixel_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_url")
    parser.add_argument("case_dir")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    case, image_b64, ref = load_case(args.case_dir)
    case_type = case.get("type", "segment")
    threshold = args.threshold if args.threshold is not None else case.get("threshold", 0.9)
    mask_index = case.get("mask_index", 0)

    if case_type == "segment":
        masks, time_ms = run_segment(args.server_url, case, image_b64)
    elif case_type == "detect":
        masks, time_ms = run_detect(args.server_url, case, image_b64)
    elif case_type == "track":
        masks, time_ms = run_track(args.server_url, case, image_b64)
    else:
        raise RuntimeError(f"unknown case type: {case_type}")

    name = os.path.basename(os.path.normpath(args.case_dir))
    if not masks:
        print(f"[FAIL] {name}: no masks returned")
        sys.exit(1)

    timing = f" time_ms={time_ms:.1f}" if time_ms is not None else ""

    if case.get("aggregate"):
        union = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            union |= (m > 0).astype(np.uint8)
        union = np.where(union > 0, 255, 0).astype(np.uint8)
        iou, pm = metrics(union, ref)
        ok = iou >= threshold and pm >= threshold
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: aggregate of {len(masks)} masks "
              f"iou={iou:.4f} pixel_match={pm:.4f}{timing}")
        sys.exit(0 if ok else 1)

    if mask_index >= 0:
        if mask_index >= len(masks):
            print(f"[FAIL] {name}: mask_index {mask_index} out of range ({len(masks)} masks)")
            sys.exit(1)
        candidates = [(mask_index, masks[mask_index])]
    else:
        candidates = list(enumerate(masks))

    best = None
    for idx, out in candidates:
        iou, pm = metrics(out, ref)
        print(f"  {name} mask[{idx}]: iou={iou:.4f} pixel_match={pm:.4f} (threshold {threshold})")
        if best is None or iou > best[1][0]:
            best = (idx, (iou, pm))
    idx, (iou, pm) = best
    ok = iou >= threshold and pm >= threshold
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: best mask[{idx}] iou={iou:.4f} pixel_match={pm:.4f}{timing}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
