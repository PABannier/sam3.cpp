#!/usr/bin/env python3
"""Convert SAM 3 PyTorch checkpoint to ggml binary format.

Usage:
    python convert_sam3_to_ggml.py --model sam3.pt --output sam3.ggml [--ftype 1]

ftype: 0 = float32, 1 = float16 (default)
"""

import argparse
import struct
import sys
import os
import re
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

MAGIC   = 0x73616D33   # "sam3"
VERSION = 1
FTYPE_F32 = 0
FTYPE_F16 = 1

# ── Hyperparameter defaults ───────────────────────────────────────────────────

HPARAMS_FIELDS = [
    ("img_size",              1008),
    ("patch_size",              14),
    ("vit_embed_dim",         1024),
    ("vit_depth",               32),
    ("vit_num_heads",           16),
    ("vit_mlp_ratio_x1000",  4625),
    ("vit_window_size",         24),
    ("n_global_attn_blocks",     4),
    ("global_attn_idx_0",        7),
    ("global_attn_idx_1",       15),
    ("global_attn_idx_2",       23),
    ("global_attn_idx_3",       31),
    ("text_width",            1024),
    ("text_heads",              16),
    ("text_layers",             24),
    ("text_ctx_len",            32),
    ("text_vocab_size",      49408),
    ("text_out_dim",           256),
    ("neck_dim",               256),
    ("fenc_layers",              6),
    ("fenc_heads",               8),
    ("fenc_ffn_dim",          2048),
    ("ddec_layers",              6),
    ("ddec_heads",               8),
    ("ddec_ffn_dim",          2048),
    ("ddec_num_queries",       200),
    ("geom_layers",              3),
    ("n_presence_tokens",        1),
    ("n_geom_queries",           4),
    ("sam_embed_dim",          256),
    ("sam_dec_depth",            2),
    ("sam_n_multimask",          3),
    ("sam_iou_head_depth",       3),
    ("mem_out_dim",             64),
    ("mem_attn_layers",          4),
    ("num_maskmem",              7),
    ("max_obj_ptrs",            16),
    ("n_amb_experts",            2),
]


# ── Key renaming ──────────────────────────────────────────────────────────────

def rename_key(k: str) -> str | None:
    """Map a PyTorch state_dict key to the flat ggml name.

    Returns None if the tensor should be skipped.
    """

    # ── Skip rules ────────────────────────────────────────────────────────
    # Only skip tensors that are genuinely training-only
    skip_patterns = [
        "attn_mask",                       # causal mask (deterministic, recomputed)
        ".dac_",                           # DAC dual supervision (training)
        "_dn_",                            # denoising queries (training)
    ]
    for pat in skip_patterns:
        if pat in k:
            return None

    # ── Detector path ─────────────────────────────────────────────────────
    # ViT backbone
    k = k.replace("detector.backbone.vision_backbone.trunk.", "vit.")
    # ViT MLP uses fc1/fc2 in timm
    k = k.replace(".mlp.fc1.", ".mlp.lin1.")
    k = k.replace(".mlp.fc2.", ".mlp.lin2.")
    # Attention
    k = k.replace(".attn.qkv.", ".attn.qkv.")
    k = k.replace(".attn.proj.", ".attn.proj.")

    # Detector neck
    k = k.replace("detector.backbone.vision_backbone.convs.", "neck.det.")
    k = k.replace("detector.backbone.vision_backbone.sam2_convs.", "neck.trk.")

    # Text encoder
    k = k.replace("detector.backbone.language_backbone.encoder.transformer.resblocks.",
                   "text.blocks.")
    k = k.replace("detector.backbone.language_backbone.encoder.token_embedding.",
                   "text.token_embed.")
    k = k.replace("detector.backbone.language_backbone.encoder.positional_embedding",
                   "text.pos_embed")
    k = k.replace("detector.backbone.language_backbone.encoder.text_projection",
                   "text.text_projection")
    k = k.replace("detector.backbone.language_backbone.encoder.ln_final.",
                   "text.ln_final.")
    k = k.replace("detector.backbone.language_backbone.resizer.",
                   "text.resizer.")
    # Text block sub-keys
    k = k.replace(".attn.in_proj_weight", ".attn.in_proj.weight")
    k = k.replace(".attn.in_proj_bias",   ".attn.in_proj.bias")
    k = k.replace(".mlp.c_fc.",  ".mlp.fc1.")
    k = k.replace(".mlp.c_proj.", ".mlp.fc2.")

    # Fusion encoder
    k = k.replace("detector.transformer.encoder.layers.", "fenc.layers.")
    k = k.replace(".cross_attn_image.", ".ca.")

    # DETR decoder
    k = k.replace("detector.transformer.decoder.layers.", "ddec.layers.")
    k = k.replace("detector.transformer.decoder.", "ddec.")
    k = k.replace(".cross_attn.", ".ca.")
    k = k.replace(".self_attn.",  ".sa.")
    k = k.replace(".ca_text.",    ".ca_text.")
    k = k.replace(".catext_norm.", ".norm_ca_text.")

    # Geometry encoder
    k = k.replace("detector.geometry_encoder.", "geom.")
    k = k.replace("geom.encode.", "geom.layers.")

    # Segmentation head
    k = k.replace("detector.segmentation_head.", "seg.")

    # DotProductScoring
    k = k.replace("detector.dot_prod_scoring.", "scoring.")

    # ── Tracker path ──────────────────────────────────────────────────────
    # Memory attention transformer
    k = k.replace("tracker.transformer.encoder.layers.", "mem_attn.layers.")
    k = k.replace("tracker.transformer.encoder.norm.", "mem_attn.norm.")
    # RoPE attention: already uses q_proj/k_proj/v_proj/out_proj

    # Memory encoder (maskmem_backbone)
    k = k.replace("tracker.maskmem_backbone.", "mem_enc.")
    k = k.replace("mem_enc.fuser.layers.", "mem_enc.fuser.")
    k = k.replace("mem_enc.mask_downsampler.encoder.", "mem_enc.ds.")

    # SAM prompt encoder
    k = k.replace("tracker.sam_prompt_encoder.", "sam_pe.")
    k = k.replace("sam_pe.pe_layer.positional_encoding_gaussian_matrix",
                   "sam_pe.pe_gaussian")
    k = k.replace("sam_pe.mask_downscaling.", "sam_pe.mask_ds.")

    # SAM mask decoder
    k = k.replace("tracker.sam_mask_decoder.", "sam_dec.")
    k = k.replace("sam_dec.transformer.layers.", "sam_dec.twoway.")
    k = k.replace("sam_dec.transformer.final_attn_token_to_image.",
                   "sam_dec.final_attn.")
    k = k.replace("sam_dec.transformer.norm_final_attn.",
                   "sam_dec.final_norm.")
    k = k.replace("sam_dec.output_upscaling.", "sam_dec.upscale.")
    k = k.replace("sam_dec.output_hypernetworks_mlps.", "sam_dec.hyper.")

    # Object pointer projection
    k = k.replace("tracker.obj_ptr_proj.", "obj_ptr_proj.")
    k = k.replace("tracker.obj_ptr_tpos_proj.", "obj_ptr_tpos_proj.")
    k = k.replace("tracker.no_obj_ptr", "no_obj_ptr")
    k = k.replace("tracker.no_mem_embed", "no_mem_embed")
    k = k.replace("tracker.no_mem_pos_enc", "no_mem_pos_enc")
    k = k.replace("tracker.no_obj_embed_spatial", "no_obj_embed_spatial")
    k = k.replace("tracker.maskmem_tpos_enc", "mem_enc.tpos_enc")
    k = k.replace("tracker.mask_downsample.", "trk_mask_ds.")

    # ── Catch-all: remove any remaining prefixes ──────────────────────────
    k = k.replace("detector.", "det.")
    k = k.replace("tracker.", "trk.")

    return k


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_header(fout, ftype: int, n_tensors: int):
    """Write file header: magic, version, ftype, n_tensors, hparams."""
    fout.write(struct.pack("<I", MAGIC))
    fout.write(struct.pack("<i", VERSION))
    fout.write(struct.pack("<i", ftype))
    fout.write(struct.pack("<i", n_tensors))
    for _, val in HPARAMS_FIELDS:
        fout.write(struct.pack("<i", val))


def write_tensor(fout, name: str, data: np.ndarray, ftype: int):
    """Write one tensor record with 32-byte aligned data."""
    n_dims = len(data.shape)
    name_bytes = name.encode("utf-8")

    # Determine storage dtype
    # 1D tensors, embeddings, and positions → always f32
    use_f16 = (ftype == FTYPE_F16 and n_dims >= 2
               and "embed" not in name
               and "pos_embed" not in name
               and "tpos" not in name
               and "pe_gaussian" not in name
               and "freqs_cis" not in name
               and "text_projection" not in name
               and "token" not in name
               and "no_obj" not in name
               and "no_mem" not in name
               and "gamma" not in name)

    dtype_id = FTYPE_F16 if use_f16 else FTYPE_F32

    if use_f16:
        data = data.astype(np.float16)
    else:
        data = data.astype(np.float32)

    # Write: n_dims, name_len, dtype, shape (reversed), name, padding, data
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert SAM3 checkpoint to ggml format")
    parser.add_argument("--model",  required=True, help="Path to sam3.pt")
    parser.add_argument("--output", required=True, help="Output .ggml path")
    parser.add_argument("--ftype",  type=int, default=1, choices=[0, 1],
                        help="0=f32, 1=f16 (default)")
    args = parser.parse_args()

    import torch

    print(f"Loading checkpoint: {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)

    # Handle nested {"model": {...}} format
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    print(f"Checkpoint has {len(ckpt)} tensors")

    # ── First pass: rename keys, skip unwanted tensors ────────────────────
    renamed = {}
    skipped = []
    for k, v in ckpt.items():
        new_name = rename_key(k)
        if new_name is None:
            skipped.append(k)
            continue
        if isinstance(v, torch.Tensor):
            # Complex tensors (e.g., freqs_cis): convert to real pairs via view_as_real
            # [N, D] complex64 → [N, D, 2] float32 (re, im interleaved in last dim)
            if v.is_complex():
                v = torch.view_as_real(v).contiguous()
            data = v.numpy()
        else:
            data = v
        # vit.pos_embed: checkpoint stores [1, 577, 1024] (576 spatial + 1 cls token).
        # The C++ loader expects [24, 24, 1024] (spatial grid only, no cls token).
        # Strip the cls token and reshape to the pretrained spatial grid.
        if new_name == "vit.pos_embed" and isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[1] == 577:
                grid = int(np.sqrt(data.shape[1] - 1))
                assert grid * grid == data.shape[1] - 1, (
                    f"pos_embed spatial tokens ({data.shape[1]-1}) is not a perfect square"
                )
                data = data[:, 1:, :]             # [1, 576, 1024]
                data = data.reshape(grid, grid, -1)  # [24, 24, 1024]
                print(f"  vit.pos_embed: stripped cls token, reshaped to {list(data.shape)}")

        renamed[new_name] = data

    print(f"Kept:    {len(renamed)} tensors")
    print(f"Skipped: {len(skipped)} tensors")
    if skipped:
        print("  First 10 skipped:")
        for s in skipped[:10]:
            print(f"    {s}")

    # ── Write ─────────────────────────────────────────────────────────────
    print(f"\nWriting {args.output} (ftype={args.ftype}) ...")

    with open(args.output, "wb") as fout:
        write_header(fout, args.ftype, len(renamed))

        for i, (name, data) in enumerate(renamed.items()):
            write_tensor(fout, name, data, args.ftype)
            if (i + 1) % 100 == 0 or i == len(renamed) - 1:
                print(f"  [{i+1}/{len(renamed)}] {name}  {list(data.shape)}")

    file_size = os.path.getsize(args.output)
    print(f"\nDone. {len(renamed)} tensors, {file_size / 1e9:.2f} GB")


# ── Listing mode (no conversion, just prints keys) ───────────────────────────

def list_keys():
    """Quick utility: python convert_sam3_to_ggml.py --list --model sam3.pt"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not args.list:
        return False

    import torch
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    for k, v in sorted(ckpt.items()):
        shape = list(v.shape) if hasattr(v, "shape") else "?"
        new = rename_key(k)
        tag = "SKIP" if new is None else new
        print(f"{k:100s}  {str(shape):30s}  → {tag}")
    return True


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_keys()
    else:
        main()
