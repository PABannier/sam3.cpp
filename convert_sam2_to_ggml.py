#!/usr/bin/env python3
"""Convert SAM2 PyTorch checkpoint to ggml binary format.

Usage:
    uv run python convert_sam2_to_ggml.py --model sam2.1_hiera_large.pt \
        --config sam2.1_hiera_l.yaml --output sam2_large.ggml [--ftype 1]

ftype: 0 = float32, 1 = float16 (default)
"""

import argparse
import struct
import sys
import os
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

MAGIC   = 0x73616D32   # "sam2"
VERSION = 1
FTYPE_F32 = 0
FTYPE_F16 = 1

# ── Default hyperparameters (SAM2.1 HieraL) ─────────────────────────────────

DEFAULT_HPARAMS = {
    "image_size":              1024,
    "backbone_type":           1,     # 1 = hiera

    "hiera_embed_dim":         144,
    "hiera_num_heads":         2,
    "hiera_num_stages":        4,
    "hiera_stages":            [2, 6, 36, 4],
    "hiera_global_att_n":      3,
    "hiera_global_att_idx":    [23, 33, 43],
    "hiera_q_pool":            3,
    "hiera_window_spec":       [8, 4, 16, 8],
    "hiera_pos_embed_bkg_h":   7,
    "hiera_pos_embed_bkg_w":   7,
    "scalp":                   1,

    "neck_dim":                256,
    "fpn_top_down_levels_n":   2,
    "fpn_top_down_levels":     [2, 3],

    "sam_embed_dim":           256,
    "sam_dec_depth":           2,
    "sam_n_multimask":         3,
    "sam_iou_head_depth":      3,

    "mem_out_dim":             64,
    "mem_attn_layers":         4,
    "num_maskmem":             7,
    "max_obj_ptrs":            16,

    "sigmoid_scale_for_mem_enc_x100":  2000,
    "sigmoid_bias_for_mem_enc_x100":   -1000,

    "use_high_res_features":               1,
    "use_obj_ptrs_in_encoder":             1,
    "pred_obj_scores":                     1,
    "use_multimask_token_for_obj_ptr":     1,
    "directly_add_no_mem_embed":           1,
    "non_overlap_masks_for_mem_enc":       1,
    "binarize_mask_from_pts":              0,
    "multimask_output_for_tracking":       1,
    "multimask_min_pt_num":                0,
    "multimask_max_pt_num":                1,
    "fixed_no_obj_ptr":                    1,
    "iou_prediction_use_sigmoid":          1,
    "use_mask_input_as_output":            1,
    "multimask_output_in_sam":             1,
}


# ── Variant hyperparameters ──────────────────────────────────────────────────

VARIANTS = {
    "hiera_t": {
        "hiera_embed_dim": 96, "hiera_num_heads": 1,
        "hiera_stages": [1, 2, 7, 2],
        "hiera_global_att_idx": [5, 7, 9],
        "hiera_window_spec": [8, 4, 14, 7],
        "hiera_pos_embed_bkg_h": 7, "hiera_pos_embed_bkg_w": 7,
    },
    "hiera_s": {
        "hiera_embed_dim": 96, "hiera_num_heads": 1,
        "hiera_stages": [1, 2, 11, 2],
        "hiera_global_att_idx": [7, 10, 13],
        "hiera_window_spec": [8, 4, 14, 7],
        "hiera_pos_embed_bkg_h": 7, "hiera_pos_embed_bkg_w": 7,
    },
    "hiera_b+": {
        "hiera_embed_dim": 112, "hiera_num_heads": 2,
        "hiera_stages": [2, 3, 16, 3],
        "hiera_global_att_idx": [12, 16, 20],
        "hiera_window_spec": [8, 4, 14, 7],
        "hiera_pos_embed_bkg_h": 14, "hiera_pos_embed_bkg_w": 14,
    },
    "hiera_l": {
        "hiera_embed_dim": 144, "hiera_num_heads": 2,
        "hiera_stages": [2, 6, 36, 4],
        "hiera_global_att_idx": [23, 33, 43],
        "hiera_window_spec": [8, 4, 16, 8],
        "hiera_pos_embed_bkg_h": 7, "hiera_pos_embed_bkg_w": 7,
    },
}


# ── Key renaming ─────────────────────────────────────────────────────────────

def rename_key(k: str) -> str | None:
    """Map a SAM2 PyTorch state_dict key to a flat ggml tensor name.

    Returns None if the tensor should be skipped.
    """

    # ── Skip rules ────────────────────────────────────────────────────────
    skip_patterns = [
        "loss", "criterion", "_dn_", "label_enc",
    ]
    for pat in skip_patterns:
        if pat in k:
            return None

    # ── Hiera backbone ────────────────────────────────────────────────────
    k = k.replace("image_encoder.trunk.", "hiera.")
    k = k.replace("hiera.patch_embed.proj.", "hiera.patch_embed.")

    # Hiera blocks MLP: .mlp.layers.0 -> .mlp.fc1, .mlp.layers.1 -> .mlp.fc2
    # Only for Hiera blocks -- SAM decoder twoway MLP uses .mlp.lin1/.mlp.lin2
    if "hiera.blocks." in k:
        k = k.replace(".mlp.layers.0.", ".mlp.fc1.")
        k = k.replace(".mlp.layers.1.", ".mlp.fc2.")

    # Dimension projection at stage transitions
    # hiera.blocks.{i}.proj.weight/bias -> hiera.blocks.{i}.proj.weight/bias
    # (already correct after trunk replacement)

    # ── FPN Neck ──────────────────────────────────────────────────────────
    # image_encoder.neck.convs.{i}.conv.weight → fpn.convs.{i}.weight
    k = k.replace("image_encoder.neck.convs.", "fpn.convs.")
    k = k.replace(".conv.weight", ".weight")
    k = k.replace(".conv.bias", ".bias")

    # ── SAM prompt encoder ────────────────────────────────────────────────
    k = k.replace("sam_prompt_encoder.", "sam_pe.")
    k = k.replace("sam_pe.pe_layer.positional_encoding_gaussian_matrix",
                   "sam_pe.pe_gaussian")
    # sam_pe.point_embeddings., not_a_point_embed., no_mask_embed. stay as-is
    k = k.replace("sam_pe.mask_downscaling.", "sam_pe.mask_ds.")

    # ── SAM mask decoder ──────────────────────────────────────────────────
    k = k.replace("sam_mask_decoder.", "sam_dec.")
    k = k.replace("sam_dec.transformer.layers.", "sam_dec.twoway.")
    k = k.replace("sam_dec.transformer.final_attn_token_to_image.",
                   "sam_dec.final_attn.")
    k = k.replace("sam_dec.transformer.norm_final_attn.",
                   "sam_dec.final_norm.")
    k = k.replace("sam_dec.output_upscaling.", "sam_dec.upscale.")
    k = k.replace("sam_dec.output_hypernetworks_mlps.", "sam_dec.hyper.")

    # TwoWay block sub-attention renaming
    k = k.replace(".self_attn.", ".sa.")

    # SAM decoder MLP layers: .mlp.layers.{0,1} → .mlp.lin1/.mlp.lin2
    # But only inside twoway blocks
    if "sam_dec.twoway." in k and ".mlp." in k:
        k = k.replace(".mlp.layers.0.", ".mlp.lin1.")
        k = k.replace(".mlp.layers.1.", ".mlp.lin2.")

    # Hyper MLP layers: sam_dec.hyper.{m}.layers.{j} → sam_dec.hyper.{m}.layers.{j}
    # (keep .layers. naming)

    # IoU head: sam_dec.iou_prediction_head.layers.{j} → sam_dec.iou_prediction_head.layers.{j}
    # (keep as-is)

    # Object score head: sam_dec.pred_obj_score_head stays as-is

    # ── Memory encoder ────────────────────────────────────────────────────
    k = k.replace("memory_encoder.", "mem_enc.")
    k = k.replace("mem_enc.mask_downsampler.encoder.", "mem_enc.ds.")
    # mem_enc.pix_feat_proj stays as-is
    k = k.replace("mem_enc.fuser.layers.", "mem_enc.fuser.")

    # ── Memory attention ──────────────────────────────────────────────────
    k = k.replace("memory_attention.layers.", "mem_attn.layers.")
    k = k.replace("memory_attention.norm.", "mem_attn.norm.")
    # RoPE attention sub-module renaming
    # Note: .self_attn. -> .sa. was already applied above (line 164)
    k = k.replace(".cross_attn_image.", ".ca.")

    # ── Top-level SAM2Base parameters ─────────────────────────────────────
    k = k.replace("maskmem_tpos_enc", "mem_enc.tpos_enc")
    # obj_ptr_proj and obj_ptr_tpos_proj stay as-is
    k = k.replace("mask_downsample.", "trk_mask_ds.")
    # no_mem_embed, no_mem_pos_enc, no_obj_ptr, no_obj_embed_spatial
    # — these top-level names are kept as-is

    return k


# ── I/O helpers ──────────────────────────────────────────────────────────────

def write_header(fout, ftype: int, n_tensors: int, hparams: dict):
    """Write SAM2 file header: magic, version, ftype, n_tensors, hparams."""
    fout.write(struct.pack("<I", MAGIC))
    fout.write(struct.pack("<i", VERSION))
    fout.write(struct.pack("<i", ftype))
    fout.write(struct.pack("<i", n_tensors))

    hp = hparams

    # Image + backbone type
    fout.write(struct.pack("<i", hp["image_size"]))
    fout.write(struct.pack("<i", hp["backbone_type"]))

    # Hiera backbone
    fout.write(struct.pack("<i", hp["hiera_embed_dim"]))
    fout.write(struct.pack("<i", hp["hiera_num_heads"]))
    fout.write(struct.pack("<i", hp["hiera_num_stages"]))
    for i in range(4):
        fout.write(struct.pack("<i", hp["hiera_stages"][i] if i < len(hp["hiera_stages"]) else 0))
    fout.write(struct.pack("<i", hp["hiera_global_att_n"]))
    for i in range(8):
        idx = hp["hiera_global_att_idx"]
        fout.write(struct.pack("<i", idx[i] if i < len(idx) else 0))
    fout.write(struct.pack("<i", hp["hiera_q_pool"]))
    for i in range(4):
        fout.write(struct.pack("<i", hp["hiera_window_spec"][i] if i < len(hp["hiera_window_spec"]) else 0))
    fout.write(struct.pack("<i", hp["hiera_pos_embed_bkg_h"]))
    fout.write(struct.pack("<i", hp["hiera_pos_embed_bkg_w"]))
    fout.write(struct.pack("<i", hp["scalp"]))

    # FPN neck
    fout.write(struct.pack("<i", hp["neck_dim"]))
    fout.write(struct.pack("<i", hp["fpn_top_down_levels_n"]))
    for i in range(4):
        td = hp["fpn_top_down_levels"]
        fout.write(struct.pack("<i", td[i] if i < len(td) else 0))

    # SAM decoder
    fout.write(struct.pack("<i", hp["sam_embed_dim"]))
    fout.write(struct.pack("<i", hp["sam_dec_depth"]))
    fout.write(struct.pack("<i", hp["sam_n_multimask"]))
    fout.write(struct.pack("<i", hp["sam_iou_head_depth"]))

    # Memory
    fout.write(struct.pack("<i", hp["mem_out_dim"]))
    fout.write(struct.pack("<i", hp["mem_attn_layers"]))
    fout.write(struct.pack("<i", hp["num_maskmem"]))
    fout.write(struct.pack("<i", hp["max_obj_ptrs"]))

    # Sigmoid scale/bias
    fout.write(struct.pack("<i", hp["sigmoid_scale_for_mem_enc_x100"]))
    fout.write(struct.pack("<i", hp["sigmoid_bias_for_mem_enc_x100"]))

    # Boolean flags
    for flag in [
        "use_high_res_features",
        "use_obj_ptrs_in_encoder",
        "pred_obj_scores",
        "use_multimask_token_for_obj_ptr",
        "directly_add_no_mem_embed",
        "non_overlap_masks_for_mem_enc",
        "binarize_mask_from_pts",
        "multimask_output_for_tracking",
        "multimask_min_pt_num",
        "multimask_max_pt_num",
        "fixed_no_obj_ptr",
        "iou_prediction_use_sigmoid",
        "use_mask_input_as_output",
        "multimask_output_in_sam",
    ]:
        fout.write(struct.pack("<i", hp[flag]))


def write_tensor(fout, name: str, data: np.ndarray, ftype: int):
    """Write one tensor record with 32-byte aligned data."""
    n_dims = len(data.shape)
    name_bytes = name.encode("utf-8")

    # 1D tensors, embeddings, positions → always f32
    use_f16 = (ftype == FTYPE_F16 and n_dims >= 2
               and "embed" not in name
               and "pos_embed" not in name
               and "tpos" not in name
               and "pe_gaussian" not in name
               and "token" not in name
               and "no_obj" not in name
               and "no_mem" not in name
               and "gamma" not in name)

    dtype_id = FTYPE_F16 if use_f16 else FTYPE_F32

    if use_f16:
        data = data.astype(np.float16)
    else:
        data = data.astype(np.float32)

    fout.write(struct.pack("<i", n_dims))
    fout.write(struct.pack("<i", len(name_bytes)))
    fout.write(struct.pack("<i", dtype_id))

    # ggml expects dimensions in reverse order (column-major)
    for dim in reversed(data.shape):
        fout.write(struct.pack("<i", dim))

    fout.write(name_bytes)

    # Pad to 32-byte alignment
    pos = fout.tell()
    pad = (32 - pos % 32) % 32
    fout.write(b"\x00" * pad)

    fout.write(data.tobytes())


# ── Config parsing ───────────────────────────────────────────────────────────

def detect_variant_from_checkpoint(state_dict: dict) -> str:
    """Auto-detect SAM2 variant from checkpoint tensor shapes."""
    # Look at patch_embed.proj.weight shape to determine embed_dim
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            embed_dim = v.shape[0]
            # Also look at total block count
            block_keys = [kk for kk in state_dict if "trunk.blocks." in kk and ".norm1.weight" in kk]
            n_blocks = len(block_keys)
            if embed_dim == 144 and n_blocks == 48:
                return "hiera_l"
            elif embed_dim == 112 and n_blocks == 24:
                return "hiera_b+"
            elif embed_dim == 96 and n_blocks == 16:
                return "hiera_s"
            elif embed_dim == 96 and n_blocks == 12:
                return "hiera_t"
            else:
                print(f"  WARNING: unknown variant (embed_dim={embed_dim}, blocks={n_blocks}), defaulting to hiera_l")
                return "hiera_l"
    raise ValueError("Could not find patch_embed.proj.weight in checkpoint")


def load_config_yaml(config_path: str) -> dict:
    """Load SAM2 Hydra YAML config and extract hyperparameters."""
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML not installed. Install with: uv pip install pyyaml")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hp = dict(DEFAULT_HPARAMS)

    # Extract from config structure
    if "model" in cfg:
        model_cfg = cfg["model"]
    else:
        model_cfg = cfg

    # Image encoder settings
    img_enc = model_cfg.get("image_encoder", {})
    trunk = img_enc.get("trunk", {})
    neck = img_enc.get("neck", {})

    if "embed_dim" in trunk:
        hp["hiera_embed_dim"] = trunk["embed_dim"]
    if "num_heads" in trunk:
        hp["hiera_num_heads"] = trunk["num_heads"]
    if "stages" in trunk:
        hp["hiera_stages"] = list(trunk["stages"])
        hp["hiera_num_stages"] = len(trunk["stages"])
    if "global_att_blocks" in trunk:
        hp["hiera_global_att_idx"] = list(trunk["global_att_blocks"])
        hp["hiera_global_att_n"] = len(trunk["global_att_blocks"])
    if "q_pool" in trunk:
        hp["hiera_q_pool"] = trunk["q_pool"]
    if "window_spec" in trunk:
        hp["hiera_window_spec"] = list(trunk["window_spec"])
    if "window_pos_embed_bkg_spatial_size" in trunk:
        bkg = trunk["window_pos_embed_bkg_spatial_size"]
        hp["hiera_pos_embed_bkg_h"] = bkg[0] if isinstance(bkg, (list, tuple)) else bkg
        hp["hiera_pos_embed_bkg_w"] = bkg[1] if isinstance(bkg, (list, tuple)) else bkg

    if "scalp" in img_enc:
        hp["scalp"] = img_enc["scalp"]

    if "d_model" in neck:
        hp["neck_dim"] = neck["d_model"]
    if "fpn_top_down_levels" in neck:
        hp["fpn_top_down_levels"] = list(neck["fpn_top_down_levels"])
        hp["fpn_top_down_levels_n"] = len(neck["fpn_top_down_levels"])

    # Memory settings
    if "num_maskmem" in model_cfg:
        hp["num_maskmem"] = model_cfg["num_maskmem"]
    if "max_obj_ptrs_in_encoder" in model_cfg:
        hp["max_obj_ptrs"] = model_cfg["max_obj_ptrs_in_encoder"]

    # Sigmoid scale/bias
    if "sigmoid_scale_for_mem_enc" in model_cfg:
        hp["sigmoid_scale_for_mem_enc_x100"] = int(model_cfg["sigmoid_scale_for_mem_enc"] * 100)
    if "sigmoid_bias_for_mem_enc" in model_cfg:
        hp["sigmoid_bias_for_mem_enc_x100"] = int(model_cfg["sigmoid_bias_for_mem_enc"] * 100)

    # Boolean flags
    flag_map = {
        "use_high_res_features_in_sam": "use_high_res_features",
        "use_obj_ptrs_in_encoder": "use_obj_ptrs_in_encoder",
        "pred_obj_scores": "pred_obj_scores",
        "use_multimask_token_for_obj_ptr": "use_multimask_token_for_obj_ptr",
        "directly_add_no_mem_embed": "directly_add_no_mem_embed",
        "non_overlap_masks_for_mem_enc": "non_overlap_masks_for_mem_enc",
        "binarize_mask_from_pts_for_mem_enc": "binarize_mask_from_pts",
        "multimask_output_for_tracking": "multimask_output_for_tracking",
        "multimask_min_pt_num": "multimask_min_pt_num",
        "multimask_max_pt_num": "multimask_max_pt_num",
        "fixed_no_obj_ptr": "fixed_no_obj_ptr",
        "iou_prediction_use_sigmoid": "iou_prediction_use_sigmoid",
        "use_mask_input_as_output_without_sam": "use_mask_input_as_output",
        "multimask_output_in_sam": "multimask_output_in_sam",
    }
    for cfg_key, hp_key in flag_map.items():
        if cfg_key in model_cfg:
            hp[hp_key] = int(bool(model_cfg[cfg_key]))

    # SAM mask decoder settings
    sam_dec = model_cfg.get("sam_mask_decoder_extra_args", {})
    if "iou_prediction_use_sigmoid" in sam_dec:
        hp["iou_prediction_use_sigmoid"] = int(bool(sam_dec["iou_prediction_use_sigmoid"]))

    return hp


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert SAM2 PyTorch checkpoint to ggml format")
    parser.add_argument("--model", required=True, help="Path to SAM2 .pt checkpoint")
    parser.add_argument("--config", default=None, help="Path to SAM2 Hydra YAML config (optional, auto-detects variant if not given)")
    parser.add_argument("--variant", default=None, choices=list(VARIANTS.keys()),
                        help="SAM2 variant (auto-detected from checkpoint if not given)")
    parser.add_argument("--output", required=True, help="Output .ggml file path")
    parser.add_argument("--ftype", type=int, default=1, choices=[0, 1],
                        help="0=f32, 1=f16 (default)")
    args = parser.parse_args()

    import torch

    print(f"Loading checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"  {len(state_dict)} tensors in checkpoint")

    # ── Determine hyperparameters ──────────────────────────────────────
    if args.config:
        print(f"Loading config: {args.config}")
        hparams = load_config_yaml(args.config)
    else:
        # Auto-detect variant from checkpoint
        variant = args.variant or detect_variant_from_checkpoint(state_dict)
        print(f"  Auto-detected variant: {variant}")
        hparams = dict(DEFAULT_HPARAMS)
        hparams.update(VARIANTS[variant])

    # Print key hyperparameters
    print(f"  embed_dim={hparams['hiera_embed_dim']}, stages={hparams['hiera_stages']}")
    total_blocks = sum(hparams["hiera_stages"])
    print(f"  total_blocks={total_blocks}, q_pool={hparams['hiera_q_pool']}")
    print(f"  global_att_idx={hparams['hiera_global_att_idx'][:hparams['hiera_global_att_n']]}")

    # ── Rename and collect tensors ────────────────────────────────────
    renamed = {}
    skipped = []
    for k, v in state_dict.items():
        new_name = rename_key(k)
        if new_name is None:
            skipped.append(k)
            continue
        data = v.detach().float().numpy()
        renamed[new_name] = data

    print(f"  {len(renamed)} tensors to write, {len(skipped)} skipped")
    if skipped:
        for s in skipped[:10]:
            print(f"    SKIP: {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    # ── Print tensor inventory ────────────────────────────────────────
    n_hiera = sum(1 for k in renamed if k.startswith("hiera."))
    n_fpn = sum(1 for k in renamed if k.startswith("fpn."))
    n_sam_pe = sum(1 for k in renamed if k.startswith("sam_pe."))
    n_sam_dec = sum(1 for k in renamed if k.startswith("sam_dec."))
    n_mem_enc = sum(1 for k in renamed if k.startswith("mem_enc."))
    n_mem_attn = sum(1 for k in renamed if k.startswith("mem_attn."))
    n_other = len(renamed) - n_hiera - n_fpn - n_sam_pe - n_sam_dec - n_mem_enc - n_mem_attn
    print(f"  Hiera={n_hiera}, FPN={n_fpn}, SAM_PE={n_sam_pe}, SAM_DEC={n_sam_dec}, "
          f"MEM_ENC={n_mem_enc}, MEM_ATTN={n_mem_attn}, other={n_other}")

    # ── Write output ──────────────────────────────────────────────────
    print(f"Writing: {args.output}")
    with open(args.output, "wb") as fout:
        write_header(fout, args.ftype, len(renamed), hparams)

        for name in sorted(renamed.keys()):
            data = renamed[name]
            write_tensor(fout, name, data, args.ftype)

    file_size = os.path.getsize(args.output)
    print(f"Done. {len(renamed)} tensors, {file_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
