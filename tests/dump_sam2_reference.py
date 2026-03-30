#!/usr/bin/env python3
"""Dump SAM2 reference tensors for C++ comparison.

Usage:
    cd ~/Documents/sam2
    SAM2_DUMP_DIR=/tmp/debug_sam2_pipeline \
    python /path/to/dump_sam2_reference.py \
        --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
        --config configs/sam2.1/sam2.1_hiera_b+.yaml \
        --image /path/to/test_image.jpg \
        --stage all
"""

import argparse
import os
import sys
import numpy as np

# Add SAM2 to path
sys.path.insert(0, os.path.expanduser("~/Documents/sam2"))

import torch
torch.set_default_dtype(torch.float32)

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import dump utilities
sys.path.insert(0, os.path.expanduser("~/Documents/sam2"))
from debug_pipeline_utils import save_tensor, DUMP_DIR


def dump_preprocessing(predictor, image_np):
    """Stage 0: Dump preprocessed image tensor."""
    print("\n=== Stage 0: Preprocessing ===")
    # Run the transforms manually
    from sam2.utils.transforms import SAM2Transforms
    transforms = predictor._transforms

    # Get the preprocessed image
    from torchvision.transforms import ToTensor
    img_tensor = ToTensor()(Image.fromarray(image_np))  # [3, H, W] float [0,1]

    # Resize
    from torchvision.transforms.functional import resize
    img_resized = resize(img_tensor, [predictor.model.image_size, predictor.model.image_size],
                         antialias=True)

    # Normalize (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_normalized = (img_resized - mean) / std

    save_tensor("preprocessed_image", img_normalized.unsqueeze(0))
    print(f"  preprocessed_image: {img_normalized.shape}")

    # Also save as CHW float for C++ consumption
    chw = img_normalized.numpy().astype(np.float32)
    with open(os.path.join(DUMP_DIR, "preprocessed.bin"), "wb") as f:
        f.write(chw.tobytes())
    print(f"  preprocessed.bin: CHW float32, shape {chw.shape}")

    return img_normalized.unsqueeze(0)


def dump_backbone(predictor, img_batch):
    """Stage 1: Dump Hiera backbone outputs."""
    print("\n=== Stage 1: Hiera Backbone ===")
    model = predictor.model

    # Run backbone trunk
    with torch.no_grad():
        backbone_out = model.image_encoder.trunk(img_batch)

    # backbone_out is a list of intermediate features
    # For Hiera, the output is typically from the trunk's forward which returns intermediates
    print(f"  Backbone output type: {type(backbone_out)}")

    if isinstance(backbone_out, (list, tuple)):
        for i, feat in enumerate(backbone_out):
            save_tensor(f"hiera_stage_{i}", feat)
            print(f"  hiera_stage_{i}: {feat.shape}")
    elif isinstance(backbone_out, dict):
        for k, v in backbone_out.items():
            if isinstance(v, torch.Tensor):
                save_tensor(f"backbone_{k}", v)
                print(f"  backbone_{k}: {v.shape}")

    return backbone_out


def dump_fpn_neck(predictor, img_batch):
    """Stage 2: Dump FPN neck outputs (backbone + neck together)."""
    print("\n=== Stage 2: FPN Neck ===")
    model = predictor.model

    with torch.no_grad():
        backbone_out = model.image_encoder(img_batch)

    # backbone_out is a dict with:
    #   "vision_features": src
    #   "vision_pos_enc": pos
    #   "backbone_fpn": features
    for k, v in backbone_out.items():
        if isinstance(v, torch.Tensor):
            save_tensor(f"imgenc_{k}", v)
            print(f"  imgenc_{k}: {v.shape}")
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, torch.Tensor):
                    save_tensor(f"imgenc_{k}_{i}", item)
                    print(f"  imgenc_{k}_{i}: {item.shape}")

    return backbone_out


def dump_image_features(predictor, image_np):
    """Stage 3: Dump prepared image features (what set_image produces)."""
    print("\n=== Stage 3: Image Features ===")

    with torch.no_grad():
        predictor.set_image(image_np)

    # Access cached features
    features = predictor._features
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            save_tensor(f"features_{k}", v)
            print(f"  features_{k}: {v.shape}")
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, torch.Tensor):
                    save_tensor(f"features_{k}_{i}", item)
                    print(f"  features_{k}_{i}: {item.shape}")

    return features


def dump_pvs_inference(predictor, image_np, point_coords, point_labels):
    """Stage 4-5: Dump prompt encoding + mask decoding."""
    print("\n=== Stage 4-5: PVS Inference ===")

    with torch.no_grad():
        if not predictor._is_image_set:
            predictor.set_image(image_np)

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=True,
        )

    save_tensor("pvs_masks", torch.from_numpy(masks).float())
    save_tensor("pvs_scores", torch.from_numpy(scores).float())
    save_tensor("pvs_logits", torch.from_numpy(logits).float())
    print(f"  pvs_masks: {masks.shape}")
    print(f"  pvs_scores: {scores.shape}, values: {scores}")
    print(f"  pvs_logits: {logits.shape}")

    return masks, scores, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--stage", default="all",
                        help="all, preprocess, backbone, fpn, features, pvs")
    parser.add_argument("--point-x", type=float, default=None)
    parser.add_argument("--point-y", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(DUMP_DIR, exist_ok=True)
    print(f"Dump directory: {DUMP_DIR}")

    # Load model
    print(f"\nLoading model: {args.config}")
    model = build_sam2(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device="cpu",
    )
    model = model.float()
    model.eval()

    predictor = SAM2ImagePredictor(model)

    # Load image
    image = np.array(Image.open(args.image).convert("RGB"))
    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Default point at center of image
    px = args.point_x if args.point_x is not None else image.shape[1] / 2
    py = args.point_y if args.point_y is not None else image.shape[0] / 2
    point_coords = np.array([[px, py]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    print(f"Test point: ({px}, {py})")

    stages = args.stage.split(",") if args.stage != "all" else [
        "preprocess", "backbone", "fpn", "features", "pvs"
    ]

    img_batch = None

    if "preprocess" in stages:
        img_batch = dump_preprocessing(predictor, image)

    if "backbone" in stages:
        if img_batch is None:
            img_batch = dump_preprocessing(predictor, image)
        dump_backbone(predictor, img_batch)

    if "fpn" in stages:
        if img_batch is None:
            img_batch = dump_preprocessing(predictor, image)
        dump_fpn_neck(predictor, img_batch)

    if "features" in stages:
        dump_image_features(predictor, image)

    if "pvs" in stages:
        dump_pvs_inference(predictor, image, point_coords, point_labels)

    print(f"\nDone. Tensors saved to {DUMP_DIR}/")


if __name__ == "__main__":
    main()
