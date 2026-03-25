#!/usr/bin/env python3
"""
Dump Phase 4 text encoder reference tensors for numerical comparison.

Usage:
  uv run python tests/dump_phase4_reference.py \
      --checkpoint raw_weights/sam3.pt \
      --tokenizer raw_weights/tokenizer.json \
      --prompts tests/phase4_prompts.tsv \
      --outdir tests/ref_phase4
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


def save_tensor(path: str, t: torch.Tensor) -> None:
    t = t.detach().cpu().float().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))


def save_i32_tensor(path: str, values: List[int]) -> None:
    arr = np.asarray(values, dtype=np.int32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(arr.tobytes())
    with open(path + ".shape", "w") as f:
        f.write(str(arr.shape[0]))


def load_prompt_cases(path: str) -> List[Tuple[str, str]]:
    cases: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                prompt_id, text = line.split("\t", 1)
            else:
                prompt_id, text = f"prompt_{len(cases):02d}", line
            cases.append((prompt_id, text))
    return cases


def save_ggml_2d(path: str, t_td: torch.Tensor) -> None:
    # ggml stores a [D, T] tensor with D contiguous. A contiguous PyTorch [T, D]
    # tensor has the same byte order, so we write those bytes but record ggml dims.
    assert t_td.ndim == 2
    t_td = t_td.detach().cpu().float().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t_td.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(f"{t_td.shape[1]},{t_td.shape[0]}")

def save_ggml_2d_batch_first(path: str, x: torch.Tensor) -> None:
    assert x.ndim == 3 and x.shape[0] == 1
    save_ggml_2d(path, x[0])


def save_ggml_2d_seq_first(path: str, x: torch.Tensor) -> None:
    assert x.ndim == 3 and x.shape[1] == 1
    save_ggml_2d(path, x[:, 0, :])


def tokenize_prompt(tokenizer: Tokenizer, text: str, context_length: int) -> List[int]:
    sot_token = 49406
    eot_token = 49407

    token_ids = list(tokenizer.encode(text).ids)
    if not token_ids or token_ids[0] != sot_token:
        token_ids = [sot_token] + token_ids
    if token_ids[-1] != eot_token:
        token_ids.append(eot_token)

    if len(token_ids) > context_length:
        token_ids = token_ids[:context_length]
        token_ids[-1] = eot_token
    while len(token_ids) < context_length:
        token_ids.append(0)
    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="raw_weights/tokenizer.json")
    parser.add_argument("--prompts", default="tests/phase4_prompts.tsv")
    parser.add_argument("--outdir", default="tests/ref_phase4")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cpu"

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    text_prefix = "detector.backbone.language_backbone.encoder."
    text_sd = {k[len(text_prefix):]: v for k, v in ckpt.items() if k.startswith(text_prefix)}
    resizer_prefix = "detector.backbone.language_backbone.resizer."
    resizer_sd = {
        k[len(resizer_prefix):]: v for k, v in ckpt.items() if k.startswith(resizer_prefix)
    }

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompts = load_prompt_cases(args.prompts)

    text_width = 1024
    text_heads = 16
    text_layers = 24
    context_length = 32

    causal_mask = torch.full((context_length, context_length), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    with torch.no_grad():
        for prompt_id, prompt_text in prompts:
            prompt_dir = os.path.join(args.outdir, prompt_id)
            os.makedirs(prompt_dir, exist_ok=True)

            token_ids = tokenize_prompt(tokenizer, prompt_text, context_length)
            with open(os.path.join(prompt_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt_text)

            save_i32_tensor(os.path.join(prompt_dir, "token_ids"), token_ids)
            save_tensor(os.path.join(prompt_dir, "causal_mask"), causal_mask)

            tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

            x_bf = F.embedding(tokens, text_sd["token_embedding.weight"])
            save_ggml_2d_batch_first(os.path.join(prompt_dir, "text_token_embed"), x_bf)

            x_bf = x_bf + text_sd["positional_embedding"][:context_length]
            save_ggml_2d_batch_first(os.path.join(prompt_dir, "text_after_pos_embed"), x_bf)

            x = x_bf.transpose(0, 1).contiguous()  # [T, 1, D]

            mha = nn.MultiheadAttention(text_width, text_heads, batch_first=False).to(device)
            mha.eval()

            for i in range(text_layers):
                prefix = f"transformer.resblocks.{i}."

                xn1 = F.layer_norm(
                    x,
                    (text_width,),
                    text_sd[prefix + "ln_1.weight"],
                    text_sd[prefix + "ln_1.bias"],
                )
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_after_ln1"),
                    xn1[:, 0, :],
                )

                qkv = F.linear(
                    xn1,
                    text_sd[prefix + "attn.in_proj_weight"],
                    text_sd[prefix + "attn.in_proj_bias"],
                )
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_qkv"),
                    qkv[:, 0, :],
                )

                mha.in_proj_weight.copy_(text_sd[prefix + "attn.in_proj_weight"])
                mha.in_proj_bias.copy_(text_sd[prefix + "attn.in_proj_bias"])
                mha.out_proj.weight.copy_(text_sd[prefix + "attn.out_proj.weight"])
                mha.out_proj.bias.copy_(text_sd[prefix + "attn.out_proj.bias"])

                attn_out = mha(xn1, xn1, xn1, need_weights=False, attn_mask=causal_mask)[0]
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_attn_out"),
                    attn_out[:, 0, :],
                )

                x = x + attn_out
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_after_attn_residual"),
                    x[:, 0, :],
                )

                xn2 = F.layer_norm(
                    x,
                    (text_width,),
                    text_sd[prefix + "ln_2.weight"],
                    text_sd[prefix + "ln_2.bias"],
                )
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_after_ln2"),
                    xn2[:, 0, :],
                )

                mlp_fc1 = F.linear(
                    xn2,
                    text_sd[prefix + "mlp.c_fc.weight"],
                    text_sd[prefix + "mlp.c_fc.bias"],
                )
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_mlp_fc1"),
                    mlp_fc1[:, 0, :],
                )

                mlp_gelu = F.gelu(mlp_fc1)
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_mlp_gelu"),
                    mlp_gelu[:, 0, :],
                )

                mlp_out = F.linear(
                    mlp_gelu,
                    text_sd[prefix + "mlp.c_proj.weight"],
                    text_sd[prefix + "mlp.c_proj.bias"],
                )
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_mlp_out"),
                    mlp_out[:, 0, :],
                )

                x = x + mlp_out
                save_ggml_2d(
                    os.path.join(prompt_dir, f"text_block_{i:02d}_out"),
                    x[:, 0, :],
                )

            x = F.layer_norm(
                x,
                (text_width,),
                text_sd["ln_final.weight"],
                text_sd["ln_final.bias"],
            )
            save_ggml_2d_seq_first(os.path.join(prompt_dir, "text_final_ln"), x)

            text_features = F.linear(x, resizer_sd["weight"], resizer_sd["bias"])
            save_ggml_2d_seq_first(
                os.path.join(prompt_dir, "text_features_2d"),
                text_features,
            )

            print(f"[done] {prompt_id}: \"{prompt_text}\"")


if __name__ == "__main__":
    main()
