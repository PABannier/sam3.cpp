#!/usr/bin/env python3
"""
Dump Phase 7 (video tracking) reference tensors for numerical comparison.

This reuses Phase 3 tracker features and Phase 6 SAM outputs so later tracker work
can audit Phase 7 in isolation from earlier stages.

Usage:
  uv run python tests/dump_phase7_reference.py \
      --checkpoint raw_weights/sam3.pt \
      --prephase-ref tests/ref_phase3 \
      --phase6-ref tests/ref_phase6 \
      --cases tests/phase7_cases.tsv \
      --outdir tests/ref_phase7
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
class Phase7Case:
    case_id: str
    label: str
    mem_cases: List[str]
    spatial_tpos: List[int]
    ptr_tpos: List[int]


def load_shape(path: str) -> List[int]:
    with open(path + ".shape", "r", encoding="utf-8") as f:
        return [int(x) for x in f.read().strip().split(",") if x]


def load_tensor(path: str) -> torch.Tensor:
    shape = load_shape(path)
    data = np.fromfile(path + ".bin", dtype=np.float32).reshape(shape)
    return torch.from_numpy(data)


def load_ggml_bd(path: str) -> torch.Tensor:
    d, b = load_shape(path)
    data = np.fromfile(path + ".bin", dtype=np.float32).reshape((b, d))
    return torch.from_numpy(data)


def load_ggml_bnd(path: str) -> torch.Tensor:
    d, n, b = load_shape(path)
    data = np.fromfile(path + ".bin", dtype=np.float32).reshape((b, n, d))
    return torch.from_numpy(data)


def load_ggml_masks(path: str, h: int, w: int) -> torch.Tensor:
    x = load_ggml_bnd(path)
    b, n, hw = x.shape
    assert hw == h * w
    return x.view(b, n, h, w)


def save_raw(path: str, arr: np.ndarray, shape) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(arr.astype(np.float32, copy=False).tobytes())
    with open(path + ".shape", "w", encoding="utf-8") as f:
        f.write(",".join(str(int(d)) for d in shape))


def save_tensor(path: str, t: torch.Tensor) -> None:
    t = t.detach().cpu().float().contiguous()
    save_raw(path, t.numpy(), t.shape)


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


def save_ggml_masks(path: str, masks_bchw: torch.Tensor) -> None:
    assert masks_bchw.ndim == 4
    b, c, h, w = masks_bchw.shape
    save_ggml_bnd(path, masks_bchw.reshape(b, c, h * w))


def parse_int_list(field: str) -> List[int]:
    if not field:
        return []
    vals = []
    for part in field.split("|"):
        vals.append(-1 if part == "-" else int(part))
    return vals


def load_cases(path: str) -> List[Phase7Case]:
    cases: List[Phase7Case] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            while len(fields) < 5:
                fields.append("")
            mem_cases = fields[2].split("|") if fields[2] else []
            spatial_tpos = parse_int_list(fields[3])
            ptr_tpos = parse_int_list(fields[4])
            assert len(mem_cases) == len(spatial_tpos) == len(ptr_tpos)
            cases.append(
                Phase7Case(
                    case_id=fields[0],
                    label=fields[1],
                    mem_cases=mem_cases,
                    spatial_tpos=spatial_tpos,
                    ptr_tpos=ptr_tpos,
                )
            )
    return cases


def layer_norm_2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    return weight[:, None, None] * x + bias[:, None, None]


def sinusoidal_pe(h: int, w: int, d_model: int) -> torch.Tensor:
    half = d_model // 2
    scale = 2.0 * math.pi
    temperature = 10000.0

    ys = torch.arange(1, h + 1, dtype=torch.float32).view(h, 1).repeat(1, w)
    xs = torch.arange(1, w + 1, dtype=torch.float32).view(1, w).repeat(h, 1)
    ys = ys / h * scale
    xs = xs / w * scale

    dim_t = torch.arange(half, dtype=torch.float32)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / half)

    pos_x = xs[:, :, None] / dim_t
    pos_y = ys[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    return torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0).contiguous()


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


def init_t_xy(end_x, end_y, scale=1.0, offset=0, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0, device=None):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    shape = [d if i >= x.ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, repeat_freqs_k: bool = False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def attention_forward(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    prefix: str,
    weights: Dict[str, torch.Tensor],
    num_heads: int,
    rope_cis: torch.Tensor | None = None,
    repeat_freqs_k: bool = False,
    num_k_exclude_rope: int = 0,
) -> torch.Tensor:
    q = F.linear(q_in, weights[prefix + ".q_proj.weight"].float(), weights[prefix + ".q_proj.bias"].float())
    k = F.linear(k_in, weights[prefix + ".k_proj.weight"].float(), weights[prefix + ".k_proj.bias"].float())
    v = F.linear(v_in, weights[prefix + ".v_proj.weight"].float(), weights[prefix + ".v_proj.bias"].float())

    bsz, nq, dim = q.shape
    nk = k.shape[1]
    head_dim = dim // num_heads

    qh = q.view(bsz, nq, num_heads, head_dim).permute(0, 2, 1, 3)
    kh = k.view(bsz, nk, num_heads, head_dim).permute(0, 2, 1, 3)
    vh = v.view(bsz, nk, num_heads, head_dim).permute(0, 2, 1, 3)

    if rope_cis is not None:
        nk_rope = kh.shape[2] - num_k_exclude_rope
        if nk_rope > 0:
            qh, kh_rope = apply_rotary_enc(
                qh,
                kh[:, :, :nk_rope, :],
                rope_cis,
                repeat_freqs_k=repeat_freqs_k,
            )
            if nk_rope < kh.shape[2]:
                kh = torch.cat([kh_rope, kh[:, :, nk_rope:, :]], dim=2)
            else:
                kh = kh_rope

    out = F.scaled_dot_product_attention(qh, kh, vh)
    out = out.permute(0, 2, 1, 3).reshape(bsz, nq, dim)
    return F.linear(out, weights[prefix + ".out_proj.weight"].float(), weights[prefix + ".out_proj.bias"].float())


def mlp_forward(x: torch.Tensor, weights: Dict[str, torch.Tensor], prefix: str, num_layers: int) -> torch.Tensor:
    for i in range(num_layers):
        w = weights[f"{prefix}.layers.{i}.weight"].float()
        b = weights[f"{prefix}.layers.{i}.bias"].float()
        x = F.linear(x, w, b)
        if i < num_layers - 1:
            x = F.relu(x)
    return x


def build_no_point_prompt_encoder_outputs(prompt_weights: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embed_dim = 256
    feat_size = 72

    pe_gaussian = prompt_weights["pe_layer.positional_encoding_gaussian_matrix"].float()
    coords = torch.zeros((1, 1, 2), dtype=torch.float32)
    labels = -torch.ones((1, 1), dtype=torch.int64)

    coords_centered = coords + 0.5
    coords_norm = coords_centered.clone()
    coords_norm[:, :, 0] /= 1008.0
    coords_norm[:, :, 1] /= 1008.0
    coords_enc = 2.0 * coords_norm - 1.0
    coords_enc = coords_enc @ pe_gaussian
    coords_enc = 2.0 * math.pi * coords_enc
    point_embedding = torch.cat([torch.sin(coords_enc), torch.cos(coords_enc)], dim=-1)

    not_a_point = prompt_weights["not_a_point_embed.weight"].float()[0]
    sparse = point_embedding.clone()
    sparse[:, 0, :] = not_a_point

    ys = (torch.arange(feat_size, dtype=torch.float32) + 0.5) / feat_size
    xs = (torch.arange(feat_size, dtype=torch.float32) + 0.5) / feat_size
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dense_coords = torch.stack([xx, yy], dim=-1)
    dense_coords = 2.0 * dense_coords - 1.0
    dense_pe = dense_coords @ pe_gaussian
    dense_pe = 2.0 * math.pi * dense_pe
    image_pe = torch.cat([torch.sin(dense_pe), torch.cos(dense_pe)], dim=-1).permute(2, 0, 1).unsqueeze(0)

    no_mask = prompt_weights["no_mask_embed.weight"].float().view(1, embed_dim, 1, 1)
    dense = no_mask.expand(1, embed_dim, feat_size, feat_size).contiguous()
    return sparse, dense, image_pe


def run_sam_decoder(
    mask_weights: Dict[str, torch.Tensor],
    prompt_weights: Dict[str, torch.Tensor],
    image_embeddings: torch.Tensor,
    neck_trk_0: torch.Tensor,
    neck_trk_1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sparse_embeddings, dense_embeddings, image_pe = build_no_point_prompt_encoder_outputs(prompt_weights)

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

    output_tokens = torch.cat(
        [
            mask_weights["obj_score_token.weight"].float(),
            mask_weights["iou_token.weight"].float(),
            mask_weights["mask_tokens.weight"].float(),
        ],
        dim=0,
    ).unsqueeze(0)
    tokens = torch.cat((output_tokens, sparse_embeddings), dim=1)

    src = image_embeddings + dense_embeddings
    pos_src = image_pe
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

    iou_token_out = queries[:, 1, :]
    mask_tokens_out = queries[:, 2:6, :]
    obj_token_out = queries[:, 0, :]

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

    hyper_in_list = [
        mlp_forward(mask_tokens_out[:, i, :], mask_weights, f"output_hypernetworks_mlps.{i}", 3)
        for i in range(4)
    ]
    hyper_in = torch.stack(hyper_in_list, dim=1)
    _, c, uh, uw = up2.shape
    masks = (hyper_in @ up2.view(bsz, c, uh * uw)).view(bsz, -1, uh, uw)
    iou_pred = mlp_forward(iou_token_out, mask_weights, "iou_prediction_head", 3).sigmoid()
    obj_score_logits = mlp_forward(obj_token_out, mask_weights, "pred_obj_score_head", 3)

    return {
        "sparse": sparse_embeddings,
        "dense": dense_embeddings,
        "image_pe": image_pe,
        "mask_logits": masks[:, :1, :, :].contiguous(),
        "iou": iou_pred[:, :1].contiguous(),
        "obj_score": obj_score_logits.contiguous(),
        "sam_token": mask_tokens_out[:, 0, :].contiguous(),
    }


def run_memory_encoder(
    tracker_weights: Dict[str, torch.Tensor],
    neck_trk_2: torch.Tensor,
    low_res_mask_logits: torch.Tensor,
    obj_score_logit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    high_res_mask = F.interpolate(
        low_res_mask_logits.float(),
        size=(1008, 1008),
        mode="bilinear",
        align_corners=False,
    )
    mask_for_mem = torch.sigmoid(high_res_mask)
    mask_for_mem = mask_for_mem * 20.0 - 10.0

    x = F.interpolate(
        mask_for_mem.float(),
        size=(1152, 1152),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    conv_ids = [0, 3, 6, 9]
    norm_ids = [1, 4, 7, 10]
    for conv_id, norm_id in zip(conv_ids, norm_ids):
        x = F.conv2d(
            x,
            tracker_weights[f"maskmem_backbone.mask_downsampler.encoder.{conv_id}.weight"].float(),
            tracker_weights[f"maskmem_backbone.mask_downsampler.encoder.{conv_id}.bias"].float(),
            stride=2,
            padding=1,
        )
        x = layer_norm_2d(
            x,
            tracker_weights[f"maskmem_backbone.mask_downsampler.encoder.{norm_id}.weight"].float(),
            tracker_weights[f"maskmem_backbone.mask_downsampler.encoder.{norm_id}.bias"].float(),
        )
        x = F.gelu(x)

    x = F.conv2d(
        x,
        tracker_weights["maskmem_backbone.mask_downsampler.encoder.12.weight"].float(),
        tracker_weights["maskmem_backbone.mask_downsampler.encoder.12.bias"].float(),
    )

    pix_proj = F.conv2d(
        neck_trk_2,
        tracker_weights["maskmem_backbone.pix_feat_proj.weight"].float(),
        tracker_weights["maskmem_backbone.pix_feat_proj.bias"].float(),
    )
    fused = pix_proj + x

    block_outs = []
    h = fused
    for idx in range(2):
        prefix = f"maskmem_backbone.fuser.layers.{idx}"
        residual = h
        h = F.conv2d(
            h,
            tracker_weights[prefix + ".dwconv.weight"].float(),
            tracker_weights[prefix + ".dwconv.bias"].float(),
            padding=3,
            groups=h.shape[1],
        )
        h = layer_norm_2d(
            h,
            tracker_weights[prefix + ".norm.weight"].float(),
            tracker_weights[prefix + ".norm.bias"].float(),
        )
        h = h.permute(0, 2, 3, 1)
        h = F.linear(
            h,
            tracker_weights[prefix + ".pwconv1.weight"].float(),
            tracker_weights[prefix + ".pwconv1.bias"].float(),
        )
        h = F.gelu(h)
        h = F.linear(
            h,
            tracker_weights[prefix + ".pwconv2.weight"].float(),
            tracker_weights[prefix + ".pwconv2.bias"].float(),
        )
        gamma = tracker_weights[prefix + ".gamma"].float().view(1, 1, 1, -1)
        h = residual + (gamma * h).permute(0, 3, 1, 2)
        block_outs.append(h)

    out = F.conv2d(
        h,
        tracker_weights["maskmem_backbone.out_proj.weight"].float(),
        tracker_weights["maskmem_backbone.out_proj.bias"].float(),
    )

    is_obj_appearing = (obj_score_logit > 0).float().view(-1, 1, 1, 1)
    no_obj_embed = tracker_weights["no_obj_embed_spatial"].float().view(1, -1, 1, 1)
    out = out + (1.0 - is_obj_appearing) * no_obj_embed
    pos = sinusoidal_pe(72, 72, 64)

    return {
        "high_res_mask": high_res_mask,
        "mask_scaled": mask_for_mem,
        "mask_downsampled": x,
        "pix_proj": pix_proj,
        "fused_input": fused,
        "fuser0": block_outs[0],
        "fuser1": block_outs[1],
        "output": out,
        "pos": pos,
    }


def run_obj_ptr(
    tracker_weights: Dict[str, torch.Tensor],
    sam_token: torch.Tensor,
    obj_score_logit: torch.Tensor,
) -> torch.Tensor:
    h = sam_token
    for i in range(3):
        h = F.linear(
            h,
            tracker_weights[f"obj_ptr_proj.layers.{i}.weight"].float(),
            tracker_weights[f"obj_ptr_proj.layers.{i}.bias"].float(),
        )
        if i < 2:
            h = F.relu(h)
    no_obj_ptr = tracker_weights["no_obj_ptr"].float()
    is_obj = (obj_score_logit > 0).float()
    return is_obj * h + (1.0 - is_obj) * no_obj_ptr


def build_prompt_and_pos(
    tracker_weights: Dict[str, torch.Tensor],
    mem_outputs: List[dict[str, torch.Tensor]],
    obj_ptrs: List[torch.Tensor],
    case: Phase7Case,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_tokens: List[torch.Tensor] = []
    prompt_pos: List[torch.Tensor] = []

    for idx, mem_out in enumerate(mem_outputs):
        feats = mem_out["output"].flatten(2).permute(2, 0, 1).contiguous()
        pos = mem_out["pos"].flatten(2).permute(2, 0, 1).contiguous()
        tpos = case.spatial_tpos[idx]
        if tpos > 0:
            add = tracker_weights["maskmem_tpos_enc"].float()[7 - tpos - 1].view(1, 1, -1)
            pos = pos + add
        prompt_tokens.append(feats)
        prompt_pos.append(pos)

    num_obj_ptr_tokens = 0
    ptr_tokens_list: List[torch.Tensor] = []
    ptr_pos_list: List[torch.Tensor] = []
    for idx, obj_ptr in enumerate(obj_ptrs):
        ptr_rel = case.ptr_tpos[idx]
        if ptr_rel < 0:
            continue
        pos_ind = torch.tensor([ptr_rel / 15.0], dtype=torch.float32)
        obj_pos = get_1d_sine_pe(pos_ind, dim=256)
        obj_pos = F.linear(
            obj_pos,
            tracker_weights["obj_ptr_tpos_proj.weight"].float(),
            tracker_weights["obj_ptr_tpos_proj.bias"].float(),
        )
        ptr = obj_ptr.view(1, 1, 4, 64).permute(1, 2, 0, 3).reshape(4, 1, 64)
        pos = obj_pos.view(1, 1, 64).repeat_interleave(4, dim=0)
        ptr_tokens_list.append(ptr)
        ptr_pos_list.append(pos)

    if ptr_tokens_list:
        ptr_tokens = torch.cat(ptr_tokens_list, dim=0)
        ptr_pos = torch.cat(ptr_pos_list, dim=0)
        prompt_tokens.append(ptr_tokens)
        prompt_pos.append(ptr_pos)
        num_obj_ptr_tokens = ptr_tokens.shape[0]

    return torch.cat(prompt_tokens, dim=0), torch.cat(prompt_pos, dim=0), num_obj_ptr_tokens


def run_memory_attention(
    tracker_weights: Dict[str, torch.Tensor],
    curr_neck_trk_2: torch.Tensor,
    prompt: torch.Tensor,
    prompt_pos: torch.Tensor,
    num_obj_ptr_tokens: int,
) -> dict[str, torch.Tensor]:
    src = curr_neck_trk_2.flatten(2).permute(2, 0, 1).contiguous()
    src_pos = sinusoidal_pe(72, 72, 256).flatten(2).permute(2, 0, 1).contiguous()

    x = src.permute(1, 0, 2).contiguous() + 0.1 * src_pos.permute(1, 0, 2).contiguous()
    mem = prompt.permute(1, 0, 2).contiguous()
    mem_pos = prompt_pos.permute(1, 0, 2).contiguous()

    rope_cis = compute_axial_cis(dim=256, end_x=72, end_y=72)
    out = {"input": x}
    for idx in range(4):
        prefix = f"transformer.encoder.layers.{idx}"

        x_norm = F.layer_norm(
            x,
            [256],
            tracker_weights[prefix + ".norm1.weight"].float(),
            tracker_weights[prefix + ".norm1.bias"].float(),
        )
        sa = attention_forward(
            x_norm,
            x_norm,
            x_norm,
            prefix + ".self_attn",
            tracker_weights,
            1,
            rope_cis=rope_cis,
        )
        x = x + sa
        out[f"layer{idx}_after_sa"] = x

        x_norm = F.layer_norm(
            x,
            [256],
            tracker_weights[prefix + ".norm2.weight"].float(),
            tracker_weights[prefix + ".norm2.bias"].float(),
        )
        ca = attention_forward(
            x_norm,
            mem + mem_pos,
            mem,
            prefix + ".cross_attn_image",
            tracker_weights,
            1,
            rope_cis=rope_cis,
            repeat_freqs_k=True,
            num_k_exclude_rope=num_obj_ptr_tokens,
        )
        x = x + ca
        out[f"layer{idx}_after_ca"] = x

        x_norm = F.layer_norm(
            x,
            [256],
            tracker_weights[prefix + ".norm3.weight"].float(),
            tracker_weights[prefix + ".norm3.bias"].float(),
        )
        ffn = F.linear(
            x_norm,
            tracker_weights[prefix + ".linear1.weight"].float(),
            tracker_weights[prefix + ".linear1.bias"].float(),
        )
        ffn = F.relu(ffn)
        ffn = F.linear(
            ffn,
            tracker_weights[prefix + ".linear2.weight"].float(),
            tracker_weights[prefix + ".linear2.bias"].float(),
        )
        x = x + ffn
        out[f"layer{idx}_after_ffn"] = x

    x = F.layer_norm(
        x,
        [256],
        tracker_weights["transformer.encoder.norm.weight"].float(),
        tracker_weights["transformer.encoder.norm.bias"].float(),
    )
    out["output"] = x
    return out


def write_meta(case_dir: str, case: Phase7Case) -> None:
    with open(os.path.join(case_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"num_slots={len(case.mem_cases)}\n")
        for i, tpos in enumerate(case.spatial_tpos):
            f.write(f"slot{i}_spatial_tpos={tpos}\n")
            f.write(f"slot{i}_ptr_tpos={case.ptr_tpos[i]}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prephase-ref", default="tests/ref_phase3")
    parser.add_argument("--phase6-ref", default="tests/ref_phase6")
    parser.add_argument("--cases", default="tests/phase7_cases.tsv")
    parser.add_argument("--outdir", default="tests/ref_phase7")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    tracker_weights = {
        k[len("tracker."):]: v
        for k, v in ckpt.items()
        if k.startswith("tracker.")
    }
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

    neck_trk_0 = load_tensor(os.path.join(args.prephase_ref, "neck_trk_0")).float()
    neck_trk_1 = load_tensor(os.path.join(args.prephase_ref, "neck_trk_1")).float()
    neck_trk_2 = load_tensor(os.path.join(args.prephase_ref, "neck_trk_2")).float()

    cases = load_cases(args.cases)
    for case in cases:
        case_dir = os.path.join(args.outdir, case.case_id)
        os.makedirs(case_dir, exist_ok=True)
        write_meta(case_dir, case)

        save_tensor(os.path.join(case_dir, "input_curr_neck_trk_0"), neck_trk_0)
        save_tensor(os.path.join(case_dir, "input_curr_neck_trk_1"), neck_trk_1)
        save_tensor(os.path.join(case_dir, "input_curr_neck_trk_2"), neck_trk_2)

        mem_outputs = []
        obj_ptrs = []
        for idx, mem_case in enumerate(case.mem_cases):
            masks = load_ggml_masks(os.path.join(args.phase6_ref, mem_case, "sam_dec_masks"), 288, 288).float()
            low_res_mask = masks[:, :1, :, :].contiguous()
            sam_token = load_ggml_bd(os.path.join(args.phase6_ref, mem_case, "sam_dec_sam_token")).float()
            obj_score = load_ggml_bd(os.path.join(args.phase6_ref, mem_case, "sam_dec_obj_score")).float()

            save_tensor(os.path.join(case_dir, f"input_mem_mask_logits_{idx}"), low_res_mask)
            save_tensor(os.path.join(case_dir, f"input_mem_sam_token_{idx}"), sam_token)
            save_tensor(os.path.join(case_dir, f"input_mem_obj_score_{idx}"), obj_score)

            mem_out = run_memory_encoder(tracker_weights, neck_trk_2, low_res_mask, obj_score)
            obj_ptr = run_obj_ptr(tracker_weights, sam_token, obj_score)
            mem_outputs.append(mem_out)
            obj_ptrs.append(obj_ptr)

            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_high_res_mask"), mem_out["high_res_mask"])
            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_mask_scaled"), mem_out["mask_scaled"])
            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_mask_downsampled"), mem_out["mask_downsampled"])
            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_pix_proj"), mem_out["pix_proj"])
            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_fused_input"), mem_out["fused_input"])
            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_fuser0"), mem_out["fuser0"])
            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_fuser1"), mem_out["fuser1"])
            save_ggml_nchw(os.path.join(case_dir, f"phase7_mem{idx}_output"), mem_out["output"])
            save_ggml_bd(os.path.join(case_dir, f"phase7_obj_ptr{idx}"), obj_ptr)

        prompt, prompt_pos, num_obj_ptr_tokens = build_prompt_and_pos(
            tracker_weights, mem_outputs, obj_ptrs, case
        )
        mem_attn = run_memory_attention(
            tracker_weights,
            neck_trk_2,
            prompt,
            prompt_pos,
            num_obj_ptr_tokens,
        )

        save_ggml_bnd(os.path.join(case_dir, "phase7_mem_attn_input"), mem_attn["input"])
        for idx in range(4):
            save_ggml_bnd(os.path.join(case_dir, f"phase7_mem_attn_layer{idx}_after_sa"), mem_attn[f"layer{idx}_after_sa"])
            save_ggml_bnd(os.path.join(case_dir, f"phase7_mem_attn_layer{idx}_after_ca"), mem_attn[f"layer{idx}_after_ca"])
            save_ggml_bnd(os.path.join(case_dir, f"phase7_mem_attn_layer{idx}_after_ffn"), mem_attn[f"layer{idx}_after_ffn"])
        save_ggml_bnd(os.path.join(case_dir, "phase7_mem_attn_output"), mem_attn["output"])

        conditioned = mem_attn["output"].permute(0, 2, 1).reshape(1, 256, 72, 72).contiguous()
        prop = run_sam_decoder(mask_weights, prompt_weights, conditioned, neck_trk_0, neck_trk_1)
        save_ggml_masks(os.path.join(case_dir, "phase7_prop_masks"), prop["mask_logits"])
        save_ggml_bd(os.path.join(case_dir, "phase7_prop_iou"), prop["iou"])
        save_ggml_bd(os.path.join(case_dir, "phase7_prop_obj_score"), prop["obj_score"])
        save_ggml_bd(os.path.join(case_dir, "phase7_prop_sam_token"), prop["sam_token"])

    print(f"Saved Phase 7 references to {args.outdir}")


if __name__ == "__main__":
    main()
