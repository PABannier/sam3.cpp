#!/usr/bin/env python3
"""
Dump Phase 6 (SAM prompt encoder + SAM mask decoder) reference tensors for
numerical comparison against sam3.cpp.

The script reuses tracker neck features from Phase 3 so Phase 6 can be audited
in isolation and the fixtures can be reused by later tracker stages.

Usage:
  uv run python tests/dump_phase6_reference.py \
      --checkpoint raw_weights/sam3.pt \
      --prephase-ref tests/ref_phase3 \
      --cases tests/phase6_cases.tsv \
      --outdir tests/ref_phase6
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PromptCase:
    case_id: str
    multimask: bool
    pos_points: List[List[float]]
    neg_points: List[List[float]]
    box: List[float]


def load_tensor(path: str) -> torch.Tensor:
    with open(path + ".shape", "r", encoding="utf-8") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    data = np.fromfile(path + ".bin", dtype=np.float32).reshape(shape)
    return torch.from_numpy(data)


def save_raw(path: str, arr: np.ndarray, shape) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(arr.astype(np.float32, copy=False).tobytes())
    with open(path + ".shape", "w", encoding="utf-8") as f:
        f.write(",".join(str(int(d)) for d in shape))


def save_ggml_bd(path: str, x_bd: torch.Tensor) -> None:
    assert x_bd.ndim == 2
    x_bd = x_bd.detach().cpu().float().contiguous()
    b, d = x_bd.shape
    save_raw(path, x_bd.numpy(), (d, b))


def save_ggml_bnd(path: str, x_bnd: torch.Tensor) -> None:
    assert x_bnd.ndim == 3
    x_bnd = x_bnd.detach().cpu().float().contiguous()
    b, n, d = x_bnd.shape
    save_raw(path, x_bnd.numpy(), (d, n, b))


def save_ggml_bhwc(path: str, x_bhwc: torch.Tensor) -> None:
    assert x_bhwc.ndim == 4
    x_bhwc = x_bhwc.detach().cpu().float().contiguous()
    b, h, w, c = x_bhwc.shape
    save_raw(path, x_bhwc.numpy(), (c, w, h, b))


def save_ggml_nchw(path: str, x_nchw: torch.Tensor) -> None:
    assert x_nchw.ndim == 4
    save_ggml_bhwc(path, x_nchw.permute(0, 2, 3, 1))


def save_ggml_whcb_from_nchw(path: str, x_nchw: torch.Tensor) -> None:
    assert x_nchw.ndim == 4
    x_nchw = x_nchw.detach().cpu().float().contiguous()
    b, c, h, w = x_nchw.shape
    save_raw(path, x_nchw.numpy(), (w, h, c, b))


def save_ggml_masks(path: str, masks_bchw: torch.Tensor) -> None:
    assert masks_bchw.ndim == 4
    b, c, h, w = masks_bchw.shape
    save_ggml_bnd(path, masks_bchw.reshape(b, c, h * w))


def parse_points(field: str) -> List[List[float]]:
    if not field:
        return []
    points = []
    for part in field.split("|"):
        x_str, y_str = part.split(":")
        points.append([float(x_str), float(y_str)])
    return points


def parse_box(field: str) -> List[float]:
    if not field:
        return []
    x0, y0, x1, y1 = field.split(":")
    return [float(x0), float(y0), float(x1), float(y1)]


def load_cases(path: str) -> List[PromptCase]:
    cases: List[PromptCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            while len(fields) < 5:
                fields.append("")
            cases.append(
                PromptCase(
                    case_id=fields[0],
                    multimask=fields[1] == "1",
                    pos_points=parse_points(fields[2]),
                    neg_points=parse_points(fields[3]),
                    box=parse_box(fields[4]),
                )
            )
    return cases


def build_point_inputs(case: PromptCase):
    coords: List[List[float]] = []
    labels: List[int] = []

    if case.box:
        coords.append([case.box[0], case.box[1]])
        coords.append([case.box[2], case.box[3]])
        labels.extend([2, 3])

    coords.extend(case.pos_points)
    labels.extend([1] * len(case.pos_points))

    coords.extend(case.neg_points)
    labels.extend([0] * len(case.neg_points))

    coords_t = torch.tensor([coords], dtype=torch.float32) if coords else torch.zeros(1, 0, 2)
    labels_t = torch.tensor([labels], dtype=torch.int64) if labels else torch.zeros(1, 0, dtype=torch.int64)
    return coords_t, labels_t


def position_encode(coords_norm: torch.Tensor, pe_gaussian: torch.Tensor) -> torch.Tensor:
    coords = 2.0 * coords_norm - 1.0
    enc = coords @ pe_gaussian
    enc = 2.0 * math.pi * enc
    return torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)


def build_prompt_encoder_outputs(weights: Dict[str, torch.Tensor], case: PromptCase):
    embed_dim = 256
    input_size = 1008
    feat_size = 72

    pe_gaussian = weights["pe_layer.positional_encoding_gaussian_matrix"].float()
    coords, labels = build_point_inputs(case)

    pad_point = torch.zeros((coords.shape[0], 1, 2), dtype=torch.float32)
    pad_label = -torch.ones((labels.shape[0], 1), dtype=torch.int64)
    coords = torch.cat([coords, pad_point], dim=1)
    labels = torch.cat([labels, pad_label], dim=1)

    coords_centered = coords + 0.5
    coords_norm = coords_centered.clone()
    coords_norm[:, :, 0] /= input_size
    coords_norm[:, :, 1] /= input_size
    point_embedding = position_encode(coords_norm, pe_gaussian)

    not_a_point = weights["not_a_point_embed.weight"].float()[0]
    point_embeds = [
        weights[f"point_embeddings.{i}.weight"].float()[0]
        for i in range(4)
    ]

    sparse = point_embedding.clone()
    for idx in range(labels.shape[1]):
        label = int(labels[0, idx].item())
        if label == -1:
            sparse[:, idx, :] = not_a_point
        else:
            sparse[:, idx, :] = sparse[:, idx, :] + point_embeds[label]

    ys = (torch.arange(feat_size, dtype=torch.float32) + 0.5) / feat_size
    xs = (torch.arange(feat_size, dtype=torch.float32) + 0.5) / feat_size
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dense_coords = torch.stack([xx, yy], dim=-1)
    image_pe = position_encode(dense_coords, pe_gaussian).permute(2, 0, 1).unsqueeze(0)

    no_mask = weights["no_mask_embed.weight"].float().view(1, embed_dim, 1, 1)
    dense = no_mask.expand(1, embed_dim, feat_size, feat_size).contiguous()
    return sparse, dense, image_pe


def attention_forward(q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      prefix: str,
                      weights: Dict[str, torch.Tensor],
                      num_heads: int) -> torch.Tensor:
    q_w = weights[prefix + ".q_proj.weight"].float()
    q_b = weights[prefix + ".q_proj.bias"].float()
    k_w = weights[prefix + ".k_proj.weight"].float()
    k_b = weights[prefix + ".k_proj.bias"].float()
    v_w = weights[prefix + ".v_proj.weight"].float()
    v_b = weights[prefix + ".v_proj.bias"].float()
    out_w = weights[prefix + ".out_proj.weight"].float()
    out_b = weights[prefix + ".out_proj.bias"].float()

    q_proj = F.linear(q, q_w, q_b)
    k_proj = F.linear(k, k_w, k_b)
    v_proj = F.linear(v, v_w, v_b)

    bsz, nq, dim = q_proj.shape
    nk = k_proj.shape[1]
    head_dim = dim // num_heads

    qh = q_proj.view(bsz, nq, num_heads, head_dim).permute(0, 2, 1, 3)
    kh = k_proj.view(bsz, nk, num_heads, head_dim).permute(0, 2, 1, 3)
    vh = v_proj.view(bsz, nk, num_heads, head_dim).permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(qh, kh, vh)
    out = out.permute(0, 2, 1, 3).reshape(bsz, nq, dim)
    return F.linear(out, out_w, out_b)


def mlp_forward(x: torch.Tensor,
                weights: Dict[str, torch.Tensor],
                prefix: str,
                num_layers: int,
                sigmoid_output: bool = False) -> torch.Tensor:
    for i in range(num_layers):
        w = weights[f"{prefix}.layers.{i}.weight"].float()
        b = weights[f"{prefix}.layers.{i}.bias"].float()
        x = F.linear(x, w, b)
        if i < num_layers - 1:
            x = F.relu(x)
    if sigmoid_output:
        x = torch.sigmoid(x)
    return x


def layer_norm_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    return weight[:, None, None] * x + bias[:, None, None]


def dump_case(case: PromptCase,
              prompt_weights: Dict[str, torch.Tensor],
              mask_weights: Dict[str, torch.Tensor],
              no_mem_embed: torch.Tensor,
              neck_trk_0: torch.Tensor,
              neck_trk_1: torch.Tensor,
              neck_trk_2: torch.Tensor,
              outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    sparse_embeddings, dense_embeddings, image_pe = build_prompt_encoder_outputs(prompt_weights, case)

    image_embeddings = neck_trk_2 + no_mem_embed.view(1, -1, 1, 1)
    feat_s0 = F.conv2d(
        neck_trk_0,
        mask_weights["conv_s0.weight"].float(),
        mask_weights["conv_s0.bias"].float(),
    )
    feat_s1 = F.conv2d(
        neck_trk_1,
        mask_weights["conv_s1.weight"].float(),
        mask_weights["conv_s1.bias"].float(),
    )

    save_ggml_bnd(os.path.join(outdir, "sam_pe_sparse"), sparse_embeddings)
    save_ggml_nchw(os.path.join(outdir, "sam_pe_dense"), dense_embeddings)
    save_ggml_nchw(os.path.join(outdir, "sam_pe_image_pe"), image_pe)
    save_ggml_nchw(os.path.join(outdir, "sam_dec_image_feats"), image_embeddings)

    output_tokens = torch.cat(
        [
            mask_weights["obj_score_token.weight"].float(),
            mask_weights["iou_token.weight"].float(),
            mask_weights["mask_tokens.weight"].float(),
        ],
        dim=0,
    ).unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse_embeddings), dim=1)
    save_ggml_bnd(os.path.join(outdir, "sam_dec_tokens_initial"), tokens)

    src = image_embeddings + dense_embeddings
    pos_src = image_pe.repeat(tokens.shape[0], 1, 1, 1)
    bsz, dim, h, w = src.shape
    keys = src.flatten(2).permute(0, 2, 1)
    key_pe = pos_src.flatten(2).permute(0, 2, 1)
    queries = tokens
    query_pe = tokens

    for idx in range(2):
        prefix = f"transformer.layers.{idx}"
        if idx == 0:
            queries = attention_forward(
                queries, queries, queries, prefix + ".self_attn", mask_weights, 8
            )
        else:
            q = queries + query_pe
            attn_out = attention_forward(q, q, queries, prefix + ".self_attn", mask_weights, 8)
            queries = queries + attn_out
        queries = F.layer_norm(
            queries,
            [dim],
            mask_weights[prefix + ".norm1.weight"].float(),
            mask_weights[prefix + ".norm1.bias"].float(),
        )

        q = queries + query_pe
        k = keys + key_pe
        attn_out = attention_forward(
            q, k, keys, prefix + ".cross_attn_token_to_image", mask_weights, 8
        )
        queries = F.layer_norm(
            queries + attn_out,
            [dim],
            mask_weights[prefix + ".norm2.weight"].float(),
            mask_weights[prefix + ".norm2.bias"].float(),
        )

        mlp_hidden = F.linear(
            queries,
            mask_weights[prefix + ".mlp.lin1.weight"].float(),
            mask_weights[prefix + ".mlp.lin1.bias"].float(),
        )
        mlp_hidden = F.relu(mlp_hidden)
        mlp_hidden = F.linear(
            mlp_hidden,
            mask_weights[prefix + ".mlp.lin2.weight"].float(),
            mask_weights[prefix + ".mlp.lin2.bias"].float(),
        )
        queries = F.layer_norm(
            queries + mlp_hidden,
            [dim],
            mask_weights[prefix + ".norm3.weight"].float(),
            mask_weights[prefix + ".norm3.bias"].float(),
        )

        q = queries + query_pe
        k = keys + key_pe
        attn_out = attention_forward(
            k, q, queries, prefix + ".cross_attn_image_to_token", mask_weights, 8
        )
        keys = F.layer_norm(
            keys + attn_out,
            [dim],
            mask_weights[prefix + ".norm4.weight"].float(),
            mask_weights[prefix + ".norm4.bias"].float(),
        )

        save_ggml_bnd(os.path.join(outdir, f"sam_dec_block{idx}_queries"), queries)
        save_ggml_bnd(os.path.join(outdir, f"sam_dec_block{idx}_keys"), keys)

    q = queries + query_pe
    k = keys + key_pe
    attn_out = attention_forward(
        q, k, keys, "transformer.final_attn_token_to_image", mask_weights, 8
    )
    queries = F.layer_norm(
        queries + attn_out,
        [dim],
        mask_weights["transformer.norm_final_attn.weight"].float(),
        mask_weights["transformer.norm_final_attn.bias"].float(),
    )
    save_ggml_bnd(os.path.join(outdir, "sam_dec_final_queries"), queries)

    iou_token_out = queries[:, 1, :]
    mask_tokens_out = queries[:, 2:6, :]
    obj_token_out = queries[:, 0, :]
    save_ggml_bnd(os.path.join(outdir, "sam_dec_mask_tokens"), mask_tokens_out)
    save_ggml_bd(os.path.join(outdir, "sam_dec_sam_token"), mask_tokens_out[:, 0, :])

    src_img = keys.transpose(1, 2).view(bsz, dim, h, w)
    up1 = F.conv_transpose2d(
        src_img,
        mask_weights["output_upscaling.0.weight"].float(),
        mask_weights["output_upscaling.0.bias"].float(),
        stride=2,
    )
    up1 = up1 + feat_s1
    up1 = layer_norm_2d(
        up1,
        mask_weights["output_upscaling.1.weight"].float(),
        mask_weights["output_upscaling.1.bias"].float(),
    )
    up1 = F.gelu(up1)

    up2 = F.conv_transpose2d(
        up1,
        mask_weights["output_upscaling.3.weight"].float(),
        mask_weights["output_upscaling.3.bias"].float(),
        stride=2,
    )
    up2 = F.gelu(up2 + feat_s0)

    save_ggml_whcb_from_nchw(os.path.join(outdir, "sam_dec_feat_s1_proj"), feat_s1)
    save_ggml_whcb_from_nchw(os.path.join(outdir, "sam_dec_feat_s0_proj"), feat_s0)
    save_ggml_nchw(os.path.join(outdir, "sam_dec_upscaled"), up2)

    hyper_in_list = [
        mlp_forward(
            mask_tokens_out[:, i, :],
            mask_weights,
            f"output_hypernetworks_mlps.{i}",
            3,
        )
        for i in range(4)
    ]
    hyper_in = torch.stack(hyper_in_list, dim=1)
    bsz, c, h, w = up2.shape
    masks = (hyper_in @ up2.view(bsz, c, h * w)).view(bsz, -1, h, w)
    iou_pred = mlp_forward(
        iou_token_out,
        mask_weights,
        "iou_prediction_head",
        3,
        sigmoid_output=True,
    )
    obj_score_logits = mlp_forward(
        obj_token_out,
        mask_weights,
        "pred_obj_score_head",
        3,
        sigmoid_output=False,
    )

    save_ggml_masks(os.path.join(outdir, "sam_dec_masks"), masks)
    save_ggml_bd(os.path.join(outdir, "sam_dec_iou"), iou_pred)
    save_ggml_bd(os.path.join(outdir, "sam_dec_obj_score"), obj_score_logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prephase-ref", default="tests/ref_phase3")
    parser.add_argument("--cases", default="tests/phase6_cases.tsv")
    parser.add_argument("--outdir", default="tests/ref_phase6")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    prompt_weights = {
        k[len("tracker.sam_prompt_encoder."):]: v
        for k, v in ckpt.items()
        if k.startswith("tracker.sam_prompt_encoder.")
    }
    mask_weights = {
        k[len("tracker.sam_mask_decoder."):]: v
        for k, v in ckpt.items()
        if k.startswith("tracker.sam_mask_decoder.")
    }
    no_mem_embed = ckpt["tracker.no_mem_embed"].float()

    neck_trk_0 = load_tensor(os.path.join(args.prephase_ref, "neck_trk_0")).float()
    neck_trk_1 = load_tensor(os.path.join(args.prephase_ref, "neck_trk_1")).float()
    neck_trk_2 = load_tensor(os.path.join(args.prephase_ref, "neck_trk_2")).float()

    cases = load_cases(args.cases)
    for case in cases:
        print(f"Dumping {case.case_id}...")
        dump_case(
            case,
            prompt_weights,
            mask_weights,
            no_mem_embed,
            neck_trk_0,
            neck_trk_1,
            neck_trk_2,
            os.path.join(args.outdir, case.case_id),
        )

    print(f"All tensors saved to: {args.outdir}")


if __name__ == "__main__":
    main()
