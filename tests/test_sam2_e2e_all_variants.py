#!/usr/bin/env python3
"""End-to-end comparison of all SAM2.1 variants: Python vs C++ ggml.

This script:
1. Runs Python SAM2 inference for all 4 variants (Tiny, Small, Base+, Large)
2. Runs C++ ggml inference for the same variants
3. Compares binary masks and IoU scores

Usage:
    cd ~/Documents/sam2
    python /path/to/test_sam2_e2e_all_variants.py \
        --image /path/to/test_image.jpg \
        --cpp-dir /path/to/sam3.cpp-sam2 \
        --point-x 600 --point-y 599

Requires: SAM2 Python package installed, C++ test binary built.
"""
import argparse, os, sys, subprocess, json, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Documents/sam2"))
import torch
torch.set_default_dtype(torch.float32)
from PIL import Image

VARIANTS = [
    {"name": "tiny",      "config": "configs/sam2.1/sam2.1_hiera_t.yaml"},
    {"name": "small",     "config": "configs/sam2.1/sam2.1_hiera_s.yaml"},
    {"name": "base_plus", "config": "configs/sam2.1/sam2.1_hiera_b+.yaml"},
    {"name": "large",     "config": "configs/sam2.1/sam2.1_hiera_l.yaml"},
]


def run_python_variant(variant_name, config, checkpoint, image_path, point_x, point_y, dump_dir):
    """Run Python SAM2 inference for one variant."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    os.makedirs(dump_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Python: {variant_name}")
    print(f"{'='*60}")

    t0 = time.time()
    model = build_sam2(config, checkpoint, device="cpu")
    model = model.float().eval()
    predictor = SAM2ImagePredictor(model)
    t_load = time.time() - t0

    image = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w = image.shape[:2]

    t0 = time.time()
    with torch.no_grad():
        predictor.set_image(image)
    t_encode = time.time() - t0

    point_coords = np.array([[point_x, point_y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    t0 = time.time()
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=True,
        )
    t_predict = time.time() - t0

    # Save preprocessed image for C++ consumption
    transforms = predictor._transforms
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    from torchvision.transforms.functional import resize, normalize
    img_resized = resize(img_tensor, [model.image_size, model.image_size], antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_resized - mean) / std
    preproc_path = os.path.join(dump_dir, "preprocessed.bin")
    img_normalized.numpy().astype(np.float32).tofile(preproc_path)

    # Save masks and scores
    logits.astype(np.float32).tofile(os.path.join(dump_dir, "logits.bin"))
    scores.astype(np.float32).tofile(os.path.join(dump_dir, "scores.bin"))
    masks.astype(np.uint8).tofile(os.path.join(dump_dir, "masks.bin"))

    # Save metadata
    meta = {
        "variant": variant_name, "img_w": img_w, "img_h": img_h,
        "point_x": point_x, "point_y": point_y,
        "n_masks": int(masks.shape[0]),
        "mask_h": int(masks.shape[1]), "mask_w": int(masks.shape[2]),
        "logit_h": int(logits.shape[1]), "logit_w": int(logits.shape[2]),
        "scores": scores.tolist(),
        "t_load": t_load, "t_encode": t_encode, "t_predict": t_predict,
    }
    with open(os.path.join(dump_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    best = int(np.argmax(scores))
    fg_pct = 100 * masks[best].sum() / masks[best].size
    print(f"  Load: {t_load:.1f}s  Encode: {t_encode:.1f}s  Predict: {t_predict:.3f}s")
    print(f"  Scores: {scores}")
    print(f"  Best mask {best}: {fg_pct:.1f}% foreground")

    return meta


def run_cpp_variant(variant_name, model_path, preproc_path, dump_dir, cpp_binary, point_x, point_y, img_w, img_h):
    """Run C++ ggml inference for one variant."""
    print(f"\n  C++: {variant_name}")

    os.makedirs(dump_dir, exist_ok=True)

    # Build a minimal C++ test that uses sam3_encode_image_from_preprocessed + sam3_segment_pvs
    # and writes raw mask logits + IoU scores
    env = os.environ.copy()
    env["SAM2_DUMP_DIR"] = dump_dir
    build_dir = os.path.dirname(cpp_binary)
    env["DYLD_LIBRARY_PATH"] = ":".join([
        os.path.join(build_dir, "ggml/src"),
        os.path.join(build_dir, "ggml/src/ggml-cpu"),
        os.path.join(build_dir, "ggml/src/ggml-metal"),
        os.path.join(build_dir, "ggml/src/ggml-blas"),
    ])

    cmd = [cpp_binary, model_path, preproc_path, dump_dir,
           str(img_w), str(img_h), str(point_x), str(point_y)]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    t_total = time.time() - t0

    if result.returncode != 0:
        print(f"  C++ FAILED (exit {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return None

    # Parse IoU from stderr
    ious = []
    for line in result.stderr.split("\n"):
        if "iou=" in line:
            # Parse: iou=[0.990, 0.994, 0.397, 0.977]
            import re
            m = re.search(r'iou=\[([\d., ]+)\]', line)
            if m:
                ious = [float(x.strip()) for x in m.group(1).split(",")]

    print(f"  Time: {t_total:.1f}s  IoU: {ious}")
    return {"ious": ious, "t_total": t_total}


def compare_variant(variant_name, py_dir, cpp_dir, py_meta):
    """Compare Python and C++ results for one variant."""
    print(f"\n  Comparison: {variant_name}")

    # Load Python masks and scores
    py_scores = np.fromfile(os.path.join(py_dir, "scores.bin"), dtype=np.float32)
    py_logits = np.fromfile(os.path.join(py_dir, "logits.bin"), dtype=np.float32)
    n_masks = py_meta["n_masks"]
    logit_h, logit_w = py_meta["logit_h"], py_meta["logit_w"]
    py_logits = py_logits.reshape(n_masks, logit_h, logit_w)

    # Load C++ mask logits: [H4*H4, 4, 1, 1] in ggml layout
    cpp_mask_path = os.path.join(cpp_dir, "cpp_pvs_masks.bin")
    if not os.path.exists(cpp_mask_path):
        print(f"  SKIP — C++ masks not found")
        return None

    cpp_masks_raw = np.fromfile(cpp_mask_path, dtype=np.float32)
    # C++ has 4 masks (including single-mask at index 0), Python has 3 (multimask only)
    # C++ ggml layout: [H4*H4, 4, 1, 1] where flat = pixel + mask*H4*H4
    # Figure out mask resolution from total size
    n_pixels = cpp_masks_raw.size // 4
    mask_hw = int(np.sqrt(n_pixels))
    cpp_masks_4 = cpp_masks_raw.reshape(4, n_pixels)

    # C++ IoU
    cpp_iou_path = os.path.join(cpp_dir, "cpp_pvs_iou.bin")
    cpp_ious = np.fromfile(cpp_iou_path, dtype=np.float32) if os.path.exists(cpp_iou_path) else np.zeros(4)

    # Compare multimask outputs (indices 1-3 in C++, 0-2 in Python)
    results = []
    for m in range(n_masks):
        py_m = py_logits[m].flatten()
        cpp_m = cpp_masks_4[m + 1]  # skip index 0 (single-mask)

        # Resize if needed (Python is logit_h × logit_w, C++ is mask_hw × mask_hw)
        if py_m.size != cpp_m.size:
            # Reshape and compare at the smaller resolution
            print(f"    Mask {m}: SIZE MISMATCH py={py_m.size} cpp={cpp_m.size}")
            results.append({"mask": m, "status": "SIZE_MISMATCH"})
            continue

        # Cosine similarity of logits
        cos = np.dot(py_m, cpp_m) / (np.linalg.norm(py_m) * np.linalg.norm(cpp_m) + 1e-8)

        # Binary mask agreement (threshold at 0)
        py_bin = (py_m > 0).astype(int)
        cpp_bin = (cpp_m > 0).astype(int)
        union = max((py_bin | cpp_bin).sum(), 1)
        intersect = (py_bin & cpp_bin).sum()
        binary_iou = intersect / union
        agree_pct = 100 * (py_bin == cpp_bin).sum() / len(py_bin)

        # IoU score comparison
        py_iou = py_scores[m]
        cpp_iou = cpp_ious[m + 1] if len(cpp_ious) > m + 1 else 0

        status = "PASS" if binary_iou > 0.90 else "FAIL"
        results.append({
            "mask": m, "status": status,
            "logit_cos": float(cos), "binary_iou": float(binary_iou),
            "agree_pct": float(agree_pct),
            "py_iou_score": float(py_iou), "cpp_iou_score": float(cpp_iou),
            "py_fg": int(py_bin.sum()), "cpp_fg": int(cpp_bin.sum()),
        })

        flag = "✓" if status == "PASS" else "✗"
        print(f"    {flag} Mask {m}: logit_cos={cos:.4f}  binary_IoU={binary_iou:.4f}  "
              f"agree={agree_pct:.1f}%  py_iou={py_iou:.3f}  cpp_iou={cpp_iou:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--cpp-dir", required=True, help="Path to sam3.cpp-sam2 root")
    parser.add_argument("--point-x", type=float, default=600)
    parser.add_argument("--point-y", type=float, default=599)
    parser.add_argument("--variants", default="tiny,small,base_plus,large",
                        help="Comma-separated list of variants to test")
    args = parser.parse_args()

    cpp_dir = args.cpp_dir
    cpp_binary = os.path.join(cpp_dir, "build", "test_sam2_pvs_compare")
    if not os.path.exists(cpp_binary):
        print(f"ERROR: C++ binary not found at {cpp_binary}")
        print(f"Build it first: cd {cpp_dir}/build && make test_sam2_pvs_compare")
        sys.exit(1)

    selected = args.variants.split(",")
    base_dump = "/tmp/sam2_e2e_test"
    os.makedirs(base_dump, exist_ok=True)

    image = np.array(Image.open(args.image).convert("RGB"))
    img_h, img_w = image.shape[:2]

    all_results = {}

    for v in VARIANTS:
        if v["name"] not in selected:
            continue

        name = v["name"]
        ckpt = os.path.join(cpp_dir, f"weights/sam2.1/sam2.1_hiera_{name}.pt")
        ggml = os.path.join(cpp_dir, f"weights/ggml/sam2.1_hiera_{name}_f32.ggml")
        config = v["config"]

        if not os.path.exists(ckpt):
            print(f"\nSKIP {name}: checkpoint not found at {ckpt}")
            continue
        if not os.path.exists(ggml):
            print(f"\nSKIP {name}: ggml model not found at {ggml}")
            continue

        py_dump = os.path.join(base_dump, f"py_{name}")
        cpp_dump = os.path.join(base_dump, f"cpp_{name}")

        # Run Python
        py_meta = run_python_variant(
            name, config, ckpt, args.image,
            args.point_x, args.point_y, py_dump
        )

        # Run C++
        cpp_result = run_cpp_variant(
            name, ggml,
            os.path.join(py_dump, "preprocessed.bin"),
            cpp_dump, cpp_binary,
            args.point_x, args.point_y, img_w, img_h
        )

        # Compare
        if cpp_result is not None:
            results = compare_variant(name, py_dump, cpp_dump, py_meta)
            all_results[name] = results

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: SAM2.1 End-to-End Comparison (all variants)")
    print(f"{'='*70}")
    print(f"Image: {args.image} ({img_w}x{img_h})")
    print(f"Point: ({args.point_x}, {args.point_y})")
    print()
    print(f"{'Variant':<12} {'Best Mask IoU':>14} {'Logit Cos':>10} {'Binary IoU':>11} {'Agreement':>10} {'Status':>8}")
    print("-" * 70)

    all_pass = True
    for name, results in all_results.items():
        if not results:
            print(f"{name:<12} {'ERROR':>14}")
            all_pass = False
            continue
        best_idx = max(range(len(results)), key=lambda i: results[i].get("py_iou_score", 0))
        r = results[best_idx]
        status = r["status"]
        if status != "PASS":
            all_pass = False
        print(f"{name:<12} {r.get('py_iou_score', 0):>14.4f} {r.get('logit_cos', 0):>10.4f} "
              f"{r.get('binary_iou', 0):>11.4f} {r.get('agree_pct', 0):>9.1f}% "
              f"{'  ✓' if status == 'PASS' else '  ✗':>8}")

    print()
    print(f"Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
