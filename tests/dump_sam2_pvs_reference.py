#!/usr/bin/env python3
"""Dump SAM2 PVS decoder reference tensors for C++ comparison.

Usage:
    cd ~/Documents/sam2
    SAM2_DUMP_DIR=/tmp/debug_sam2_pvs \
    python /path/to/dump_sam2_pvs_reference.py \
        --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
        --config configs/sam2.1/sam2.1_hiera_b+.yaml \
        --image /path/to/test_image.jpg \
        --point-x 600 --point-y 599
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("~/Documents/sam2"))
import torch
torch.set_default_dtype(torch.float32)
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DUMP_DIR = os.environ.get("SAM2_DUMP_DIR", "/tmp/debug_sam2_pvs")

def save(name, t):
    os.makedirs(DUMP_DIR, exist_ok=True)
    t = t.detach().cpu().float().contiguous()
    path = os.path.join(DUMP_DIR, name)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  {name}: {tuple(t.shape)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--point-x", type=float, default=600)
    parser.add_argument("--point-y", type=float, default=599)
    args = parser.parse_args()

    model = build_sam2(args.config, args.checkpoint, device="cpu")
    model = model.float().eval()
    predictor = SAM2ImagePredictor(model)

    image = np.array(Image.open(args.image).convert("RGB"))
    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Set image and get features
    with torch.no_grad():
        predictor.set_image(image)

    # Dump features stored after set_image
    print("\n=== Features after set_image ===")
    save("features_image_embed", predictor._features["image_embed"])
    for i, f in enumerate(predictor._features["high_res_feats"]):
        save(f"features_high_res_{i}", f)

    # Run PVS prediction
    point_coords = np.array([[args.point_x, args.point_y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    print(f"\n=== PVS with point ({args.point_x}, {args.point_y}) ===")

    # Manually trace the prediction path for dumps
    with torch.no_grad():
        # Step 1: Coordinate normalization (matching _prep_prompts)
        unnorm_coords = predictor._transforms.transform_coords(
            torch.from_numpy(point_coords).unsqueeze(0).float(),
            normalize=True,
            orig_hw=(image.shape[0], image.shape[1])
        )
        labels = torch.from_numpy(point_labels).unsqueeze(0).int()
        concat_points = (unnorm_coords, labels)
        save("unnorm_coords", unnorm_coords.squeeze(0))
        print(f"  unnorm_coords: {unnorm_coords.squeeze(0).numpy()}")

        # Step 2: Prompt encoder
        sparse_emb, dense_emb = model.sam_prompt_encoder(
            points=concat_points, boxes=None, masks=None
        )
        save("sparse_embeddings", sparse_emb)
        save("dense_embeddings", dense_emb)

        # Step 3: Get dense PE
        image_pe = model.sam_prompt_encoder.get_dense_pe()
        save("image_pe", image_pe)

        # Step 4: Prepare high-res features
        high_res_features = [
            feat_level[0].unsqueeze(0)
            for feat_level in predictor._features["high_res_feats"]
        ]
        for i, hrf in enumerate(high_res_features):
            save(f"high_res_for_decoder_{i}", hrf)

        # Step 5: Run mask decoder
        image_embed = predictor._features["image_embed"][0].unsqueeze(0)
        save("image_embed_for_decoder", image_embed)

        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        save("low_res_masks", low_res_masks)
        save("iou_predictions", iou_predictions)
        print(f"  iou_predictions: {iou_predictions.numpy()}")
        print(f"  masks range: [{low_res_masks.min():.4f}, {low_res_masks.max():.4f}]")

    # Also run through the full predict API for comparison
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            normalize_coords=True,
        )
    save("api_masks", torch.from_numpy(masks).float())
    save("api_scores", torch.from_numpy(scores).float())
    save("api_logits", torch.from_numpy(logits).float())
    print(f"\n  API scores: {scores}")
    print(f"  API masks shape: {masks.shape}")

    print(f"\nDone. Tensors in {DUMP_DIR}/")

if __name__ == "__main__":
    main()
