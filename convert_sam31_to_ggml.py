#!/usr/bin/env python3
"""Convert SAM 3.1 transformers checkpoint (safetensors) to ggml binary format.

Usage:
    python convert_sam31_to_ggml.py --model model.safetensors --output sam3.1-f16.ggml [--ftype 1] [--tokenizer <dir>]

ftype: 0 = float32, 1 = float16 (default)
The tokenizer (vocab.json + merges.txt) is embedded in the output file.

Input: transformers-style safetensors checkpoint (e.g. unsloth/sam3.1, facebook/sam3.1).
The checkpoint contains ONLY the detector (PCS) path: ViT + text encoder + DETR +
geometry encoder + mask decoder + dot-product scoring. Tracker tensors are absent,
so the produced file has no tracker weights (C++ runtime marks tracker as unavailable).

Every source key must map to a ggml name or be in the explicit skip list, otherwise
the converter aborts (strict validation).
"""

import argparse
import os
import re
import sys

import numpy as np

from convert_sam3_to_ggml import (HPARAMS_FIELDS, MAGIC, VERSION,
                                  write_header, write_tokenizer)
from ggml_writer import KEEP_F32, write_tensor

# ── Constants ──────────────────────────────────────────────────────────────────

GLOBAL_ATTN_BLOCKS = [7, 15, 23, 31]
WINDOW_SIZE = 24
GLOBAL_GRID = 72          # img_size 1008 / patch 14
ROPE_THETA = 10000.0
ROPE_DIM = 32             # half of head dim 64

# Explicitly unused tensors (training artifacts / unused projections).
SKIP_KEYS = {
    "text_encoder.text_projection.weight",
}

# Transform kinds
T_LINEAR  = "linear"     # nn.Linear: kept as-is (ggml_writer reverses dims at write time)
T_CONV2D  = "conv2d"     # Conv2d: kept as-is
T_CONVT2D = "convt2d"    # ConvTranspose2d: kept as-is
T_NONE    = "none"
T_QKV     = "qkv"        # concatenate q,k,v along dim 0, no transpose
T_POS     = "pos_embed"  # (1, 576, 1024) -> (24, 24, 1024) grid

QKV_ORDER = {"q": 0, "k": 1, "v": 2}


# ── Key mapping ────────────────────────────────────────────────────────────────

def map_key(k: str):
    """Map a transformers checkpoint key to (ggml_name, transform, cat_key, cat_idx).

    Returns None if the tensor is explicitly skipped.
    Raises ValueError if the key is not recognized (strict validation).
    """

    if k in SKIP_KEYS:
        return None

    # ── Text encoder ──────────────────────────────────────────────────────
    m = re.fullmatch(r"text_encoder\.text_model\.embeddings\.token_embedding\.weight", k)
    if m: return ("text.token_embed.weight", T_LINEAR, None, None)
    m = re.fullmatch(r"text_encoder\.text_model\.embeddings\.position_embedding\.weight", k)
    if m: return ("text.pos_embed", T_LINEAR, None, None)
    m = re.fullmatch(r"text_encoder\.text_model\.encoder\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"text.blocks.{m['i']}.attn.in_proj.{m['s']}", T_QKV, f"text.blocks.{m['i']}.attn.in_proj.{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"text_encoder\.text_model\.encoder\.layers\.(?P<i>\d+)\.self_attn\.out_proj\.(?P<s>weight|bias)", k)
    if m: return (f"text.blocks.{m['i']}.attn.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"text_encoder\.text_model\.encoder\.layers\.(?P<i>\d+)\.layer_norm(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"text.blocks.{m['i']}.ln_{m['n']}.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"text_encoder\.text_model\.encoder\.layers\.(?P<i>\d+)\.mlp\.fc(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"text.blocks.{m['i']}.mlp.fc{m['n']}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"text_encoder\.text_model\.final_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"text.ln_final.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"text_projection\.(?P<s>weight|bias)", k)
    if m: return (f"text.resizer.{m['s']}", T_LINEAR, None, None)

    # ── ViT backbone ──────────────────────────────────────────────────────
    m = re.fullmatch(r"vision_encoder\.backbone\.embeddings\.patch_embeddings\.projection\.weight", k)
    if m: return ("vit.patch_embed.proj.weight", T_CONV2D, None, None)
    m = re.fullmatch(r"vision_encoder\.backbone\.embeddings\.position_embeddings", k)
    if m: return ("vit.pos_embed", T_POS, None, None)
    m = re.fullmatch(r"vision_encoder\.backbone\.layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"vit.ln_pre.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"vision_encoder\.backbone\.layers\.(?P<i>\d+)\.attention\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"vit.blocks.{m['i']}.attn.qkv.{m['s']}", T_QKV, f"vit.blocks.{m['i']}.attn.qkv.{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"vision_encoder\.backbone\.layers\.(?P<i>\d+)\.attention.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"vit.blocks.{m['i']}.attn.proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"vision_encoder\.backbone\.layers\.(?P<i>\d+)\.layer_norm(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"vit.blocks.{m['i']}.norm{m['n']}.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"vision_encoder\.backbone\.layers\.(?P<i>\d+)\.mlp\.fc(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"vit.blocks.{m['i']}.mlp.lin{m['n']}.{m['s']}", T_LINEAR, None, None)

    # ── Neck ──────────────────────────────────────────────────────────────
    m = re.fullmatch(r"vision_encoder\.neck\.fpn_layers\.(?P<i>[0-3])\.proj1\.(?P<s>weight|bias)", k)
    if m: return (f"neck.det.{m['i']}.conv_1x1.{m['s']}", T_CONV2D, None, None)
    m = re.fullmatch(r"vision_encoder\.neck\.fpn_layers\.(?P<i>[0-3])\.proj2\.(?P<s>weight|bias)", k)
    if m: return (f"neck.det.{m['i']}.conv_3x3.{m['s']}", T_CONV2D, None, None)
    m = re.fullmatch(r"vision_encoder\.neck\.fpn_layers\.0\.scale_layers\.(?P<j>[02])\.(?P<s>weight|bias)", k)
    if m: return (f"neck.det.0.dconv_2x2_{0 if m['j'] == '0' else 1}.{m['s']}", T_CONVT2D, None, None)
    m = re.fullmatch(r"vision_encoder\.neck\.fpn_layers\.1\.scale_layers\.0\.(?P<s>weight|bias)", k)
    if m: return (f"neck.det.1.dconv_2x2.{m['s']}", T_CONVT2D, None, None)

    # ── Fusion encoder ────────────────────────────────────────────────────
    m = re.fullmatch(r"detr_encoder\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"fenc.layers.{m['i']}.sa.in_proj_{m['s']}", T_QKV, f"fenc.layers.{m['i']}.sa.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"detr_encoder\.layers\.(?P<i>\d+)\.self_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"fenc.layers.{m['i']}.sa.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_encoder\.layers\.(?P<i>\d+)\.cross_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"fenc.layers.{m['i']}.ca.in_proj_{m['s']}", T_QKV, f"fenc.layers.{m['i']}.ca.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"detr_encoder\.layers\.(?P<i>\d+)\.cross_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"fenc.layers.{m['i']}.ca.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_encoder\.layers\.(?P<i>\d+)\.layer_norm(?P<n>[123])\.(?P<s>weight|bias)", k)
    if m: return (f"fenc.layers.{m['i']}.norm{m['n']}.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"detr_encoder\.layers\.(?P<i>\d+)\.mlp\.fc(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"fenc.layers.{m['i']}.linear{m['n']}.{m['s']}", T_LINEAR, None, None)

    # ── DETR decoder ──────────────────────────────────────────────────────
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.sa.in_proj_{m['s']}", T_QKV, f"ddec.layers.{m['i']}.sa.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.self_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.sa.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.text_cross_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.ca_text.in_proj_{m['s']}", T_QKV, f"ddec.layers.{m['i']}.ca_text.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.text_cross_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.ca_text.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.vision_cross_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.ca.in_proj_{m['s']}", T_QKV, f"ddec.layers.{m['i']}.ca.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.vision_cross_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.ca.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.self_attn_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.norm2.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.text_cross_attn_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.norm_ca_text.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.vision_cross_attn_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.norm1.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.mlp_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.norm3.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"detr_decoder\.layers\.(?P<i>\d+)\.mlp\.fc(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.layers.{m['i']}.linear{m['n']}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.query_embed\.weight", k)
    if m: return ("ddec.query_embed.weight", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.reference_points\.weight", k)
    if m: return ("ddec.reference_points.weight", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.ref_point_head\.layer(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.ref_point_head.layers.{int(m['n']) - 1}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.box_head\.layer(?P<n>[123])\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.bbox_embed.layers.{int(m['n']) - 1}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.box_rpb_embed_(?P<a>[xy])\.layer(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.boxRPB_embed_{m['a']}.layers.{int(m['n']) - 1}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.presence_token\.weight", k)
    if m: return ("ddec.presence_token.weight", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.presence_head\.layer(?P<n>[123])\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.presence_token_head.layers.{int(m['n']) - 1}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"detr_decoder\.presence_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.presence_token_out_norm.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"detr_decoder\.output_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"ddec.norm.{m['s']}", T_NONE, None, None)

    # ── Geometry encoder ──────────────────────────────────────────────────
    m = re.fullmatch(r"geometry_encoder\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"geom.layers.{m['i']}.sa.in_proj_{m['s']}", T_QKV, f"geom.layers.{m['i']}.sa.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"geometry_encoder\.layers\.(?P<i>\d+)\.self_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"geom.layers.{m['i']}.sa.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.layers\.(?P<i>\d+)\.cross_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"geom.layers.{m['i']}.ca.in_proj_{m['s']}", T_QKV, f"geom.layers.{m['i']}.ca.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"geometry_encoder\.layers\.(?P<i>\d+)\.cross_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"geom.layers.{m['i']}.ca.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.layers\.(?P<i>\d+)\.layer_norm(?P<n>[123])\.(?P<s>weight|bias)", k)
    if m: return (f"geom.layers.{m['i']}.norm{m['n']}.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"geometry_encoder\.layers\.(?P<i>\d+)\.mlp\.fc(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"geom.layers.{m['i']}.linear{m['n']}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.boxes_direct_project\.(?P<s>weight|bias)", k)
    if m: return (f"geom.boxes_direct_project.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.boxes_pool_project\.(?P<s>weight|bias)", k)
    if m: return (f"geom.boxes_pool_project.{m['s']}", T_CONV2D if m['s'] == "weight" else T_NONE, None, None)
    m = re.fullmatch(r"geometry_encoder\.boxes_pos_enc_project\.(?P<s>weight|bias)", k)
    if m: return (f"geom.boxes_pos_enc_project.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.cls_embed\.weight", k)
    if m: return ("geom.cls_embed.weight", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.label_embed\.weight", k)
    if m: return ("geom.label_embed.weight", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.final_proj\.(?P<s>weight|bias)", k)
    if m: return (f"geom.final_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"geometry_encoder\.output_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"geom.norm.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"geometry_encoder\.vision_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"geom.img_pre_norm.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"geometry_encoder\.prompt_layer_norm\.(?P<s>weight|bias)", k)
    if m: return (f"geom.encode_norm.{m['s']}", T_NONE, None, None)

    # ── Mask decoder ──────────────────────────────────────────────────────
    m = re.fullmatch(r"mask_decoder\.pixel_decoder\.conv_layers\.(?P<i>[012])\.(?P<s>weight|bias)", k)
    if m: return (f"seg.pixel_decoder.conv_layers.{m['i']}.{m['s']}", T_CONV2D if m['s'] == "weight" else T_NONE, None, None)
    m = re.fullmatch(r"mask_decoder\.pixel_decoder\.norms\.(?P<i>[012])\.(?P<s>weight|bias)", k)
    if m: return (f"seg.pixel_decoder.norms.{m['i']}.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"mask_decoder\.mask_embedder\.layers\.(?P<i>[012])\.(?P<s>weight|bias)", k)
    if m: return (f"seg.mask_predictor.mask_embed.layers.{m['i']}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"mask_decoder\.prompt_cross_attn\.(?P<p>[qkv])_proj\.(?P<s>weight|bias)", k)
    if m: return (f"seg.cross_attend_prompt.in_proj_{m['s']}", T_QKV, f"seg.cross_attend_prompt.in_proj_{m['s']}", QKV_ORDER[m['p']])
    m = re.fullmatch(r"mask_decoder\.prompt_cross_attn.o_proj.(?P<s>weight|bias)", k)
    if m: return (f"seg.cross_attend_prompt.out_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"mask_decoder\.prompt_cross_attn_norm\.(?P<s>weight|bias)", k)
    if m: return (f"seg.cross_attn_norm.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"mask_decoder\.instance_projection\.(?P<s>weight|bias)", k)
    if m: return (f"seg.instance_seg_head.{m['s']}", T_CONV2D if m['s'] == "weight" else T_NONE, None, None)
    m = re.fullmatch(r"mask_decoder\.semantic_projection\.(?P<s>weight|bias)", k)
    if m: return (f"seg.semantic_seg_head.{m['s']}", T_CONV2D if m['s'] == "weight" else T_NONE, None, None)

    # ── Dot product scoring ───────────────────────────────────────────────
    m = re.fullmatch(r"dot_product_scoring\.text_mlp\.layer(?P<n>[12])\.(?P<s>weight|bias)", k)
    if m: return (f"scoring.prompt_mlp.layers.{int(m['n']) - 1}.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"dot_product_scoring\.text_mlp_out_norm\.(?P<s>weight|bias)", k)
    if m: return (f"scoring.prompt_mlp.out_norm.{m['s']}", T_NONE, None, None)
    m = re.fullmatch(r"dot_product_scoring\.text_proj\.(?P<s>weight|bias)", k)
    if m: return (f"scoring.prompt_proj.{m['s']}", T_LINEAR, None, None)
    m = re.fullmatch(r"dot_product_scoring\.query_proj\.(?P<s>weight|bias)", k)
    if m: return (f"scoring.hs_proj.{m['s']}", T_LINEAR, None, None)

    raise ValueError(f"unrecognized checkpoint key: {k}")


# ── Synthetic tensors ──────────────────────────────────────────────────────────

def make_freqs_cis(block_idx: int) -> np.ndarray:
    """Synthesize RoPE freqs_cis (numpy [N, 32, 2]: last axis = cos/sin) for a
    ViT block.

    Matches transformers Sam3ViTRotaryEmbedding: axial 2D RoPE, 16 inv_freq
    values; window blocks use a 24x24 grid with scale 1, global blocks use a
    72x72 grid with scale window_size/grid = 1/3.

    ggml_writer.write_tensor stores reversed(data.shape) as the file dims, so
    numpy [N, 32, 2] lands in the file as [2, 32, N] (C++ registers
    T3f(2, 32, rope_n) with dim0 = cos/sin).
    """
    if block_idx in GLOBAL_ATTN_BLOCKS:
        grid, scale = GLOBAL_GRID, WINDOW_SIZE / GLOBAL_GRID
    else:
        grid, scale = WINDOW_SIZE, 1.0

    n = grid * grid
    x = (np.arange(n) % grid).astype(np.float64)
    y = (np.arange(n) // grid).astype(np.float64)

    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, ROPE_DIM, 2, dtype=np.float64) / ROPE_DIM))
    angles = np.zeros((n, ROPE_DIM), dtype=np.float64)
    angles[:, :ROPE_DIM // 2] = x[:, None] * scale * inv_freq[None, :]
    angles[:, ROPE_DIM // 2:] = y[:, None] * scale * inv_freq[None, :]

    out = np.zeros((n, ROPE_DIM, 2), dtype=np.float32)
    out[..., 0] = np.cos(angles)
    out[..., 1] = np.sin(angles)
    return out


def make_zero_tensors() -> dict:
    """geom.points_* tensors exist in the C++ loader but not in the checkpoint
    (SAM 3.1 has no point prompts). Emit zeros so loading succeeds."""
    return {
        "geom.points_direct_project.weight": np.zeros((256, 2), dtype=np.float32),
        "geom.points_direct_project.bias": np.zeros((256,), dtype=np.float32),
        "geom.points_pool_project.weight": np.zeros((256, 256), dtype=np.float32),
        "geom.points_pool_project.bias": np.zeros((256,), dtype=np.float32),
        "geom.points_pos_enc_project.weight": np.zeros((256, 256), dtype=np.float32),
        "geom.points_pos_enc_project.bias": np.zeros((256,), dtype=np.float32),
    }


# ── Transforms ─────────────────────────────────────────────────────────────────

def apply_transform(data: np.ndarray, tform: str) -> np.ndarray:
    # ggml_writer.write_tensor stores reversed(data.shape) as the ggml dims and
    # the data with the last numpy axis fastest (= ggml ne0 fastest), so torch
    # weights are written as-is; no transposition is needed.
    if tform in (T_LINEAR, T_CONV2D, T_CONVT2D):
        return data
    if tform == T_QKV:
        raise RuntimeError("T_QKV handled in assembly step")
    if tform == T_POS:
        grid = int(round(np.sqrt(data.shape[1])))
        assert grid * grid == data.shape[1], f"pos_embed tokens {data.shape[1]} not a perfect square"
        # checkpoint [1, 576, 1024] -> [24, 24, 1024]; reversed() at write time
        # turns this into the [1024, 24, 24] layout C++ registers.
        return data.reshape(grid, grid, -1).astype(np.float32)
    return data


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert SAM 3.1 transformers checkpoint to ggml format")
    parser.add_argument("--model",  required=True, help="Path to model.safetensors")
    parser.add_argument("--output", required=True, help="Output .ggml path")
    parser.add_argument("--ftype",  type=int, default=1, choices=[0, 1],
                        help="0=f32, 1=f16 (default)")
    parser.add_argument("--tokenizer", default=None,
                        help="Directory containing vocab.json + merges.txt "
                             "(default: same directory as --model)")
    args = parser.parse_args()

    from safetensors.torch import load_file

    print(f"Loading checkpoint: {args.model}")
    ckpt = load_file(args.model)
    print(f"Checkpoint has {len(ckpt)} tensors")

    # ── Map + transform all keys ──────────────────────────────────────────
    out = {}          # ggml_name -> np.ndarray
    qkv_parts = {}    # cat_key -> [q, k, v] arrays
    skipped = []
    for k, v in sorted(ckpt.items()):
        res = map_key(k)
        if res is None:
            skipped.append(k)
            continue
        name, tform, cat_key, cat_idx = res
        data = v.float().numpy()
        if tform == T_QKV:
            assert cat_idx in (0, 1, 2)
            qkv_parts.setdefault(cat_key, [None, None, None])[cat_idx] = data
        else:
            out[name] = apply_transform(data, tform)

    for cat_key, parts in qkv_parts.items():
        assert all(p is not None for p in parts), f"incomplete qkv group: {cat_key}"
        assert all(p.shape == parts[0].shape for p in parts), f"qkv shape mismatch: {cat_key}"
        out[cat_key] = np.concatenate(parts, axis=0)

    # ── Synthetic tensors ────────────────────────────────────────────────
    for i in range(32):
        out[f"vit.blocks.{i}.attn.freqs_cis"] = make_freqs_cis(i)
    out.update(make_zero_tensors())

    print(f"Mapped:   {len(out)} tensors (incl. synthetic freqs_cis + geom.points_* zeros)")
    print(f"Skipped:  {len(skipped)} tensors")
    for s in skipped:
        print(f"    SKIP {s}")

    # ── Write ─────────────────────────────────────────────────────────────
    print(f"\nWriting {args.output} (ftype={args.ftype}) ...")

    with open(args.output, "wb") as fout:
        write_header(fout, args.ftype, len(out), visual_only=False)

        for i, (name, data) in enumerate(sorted(out.items())):
            write_tensor(fout, name, data, args.ftype, KEEP_F32 + ("freqs_cis",))
            if (i + 1) % 100 == 0 or i == len(out) - 1:
                print(f"  [{i+1}/{len(out)}] {name}  {list(data.shape)}")

        tok_dir = args.tokenizer if args.tokenizer else os.path.dirname(os.path.abspath(args.model))
        write_tokenizer(fout, tok_dir)

    file_size = os.path.getsize(args.output)
    print(f"\nDone. {len(out)} tensors, {file_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
