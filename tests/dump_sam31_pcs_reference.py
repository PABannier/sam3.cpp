#!/usr/bin/env python3
"""
Dump PCS reference tensors from the SAM 3.1 transformers model (unsloth checkpoint)
for verification against sam3.cpp C++ implementation (test_e2e_pcs).

Saves .bin files with raw float32 data and .shape files with tensor dimensions,
matching the format expected by tests/test_e2e_pcs.cpp (ref_dir/pcs_ref/).

Usage:
    uv run python tests/dump_sam31_pcs_reference.py
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from transformers import Sam3Model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sam3.1")
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat.jpg")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcs_ref_sam31", "pcs_ref")

PROMPT = "cat"
CTX_LEN = 32
SOT = 49406
EOT = 49407
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.1
IMG_SIZE = 1008


# ── CLIP BPE tokenizer (matches sam3_tokenize in sam3.cpp) ──────────────────
def _init_byte_encoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    enc = {b: chr(c) for b, c in zip(bs, cs)}
    return enc


def _bpe_encode(word, merge_ranks):
    word = list(word)
    if not word:
        return ""
    word[-1] += "</w>"
    if len(word) == 1:
        return word[0]
    while True:
        best = None
        best_rank = None
        for i in range(len(word) - 1):
            r = merge_ranks.get(word[i] + "\x1f" + word[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best = i
        if best is None:
            break
        merged = word[best] + word[best + 1]
        new_word = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] + word[i + 1] == merged:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
    return " ".join(word)


def _pretokenize(text):
    # Simplified: split into words/punct matching sam3_pretokenize for plain ASCII
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isalpha():
            start = i
            while i < n and text[i].isalpha():
                i += 1
            tokens.append(text[start:i])
        elif c.isdigit():
            tokens.append(c)
            i += 1
        else:
            start = i
            while i < n and not text[i].isspace() and not text[i].isalpha() and not text[i].isdigit():
                i += 1
            if i > start:
                tokens.append(text[start:i])
    return tokens


def make_tokenizer(vocab_path, merges_path):
    vocab = json.load(open(vocab_path))
    encoder = {tok: int(i) for tok, i in vocab.items()}
    merges = open(merges_path).read().splitlines()
    if merges and merges[0].startswith("#version"):
        merges = merges[1:]
    merge_ranks = {}
    for rank, line in enumerate(merges):
        parts = line.split()
        if len(parts) == 2:
            merge_ranks[parts[0] + "\x1f" + parts[1]] = rank
    byte_encoder = _init_byte_encoder()

    def tokenize(text, ctx_len=CTX_LEN):
        lower = text.lower()
        clean = " ".join(lower.split())
        ids = [SOT]
        for word in _pretokenize(clean):
            encoded = "".join(byte_encoder.get(b, "") for b in word.encode("utf-8"))
            for piece in _bpe_encode(encoded, merge_ranks).split(" "):
                if piece in encoder:
                    ids.append(encoder[piece])
        ids.append(EOT)
        if len(ids) > ctx_len:
            ids = ids[:ctx_len]
            ids[-1] = EOT
        ids += [0] * (ctx_len - len(ids))
        return ids

    return tokenize


def save_tensor(path, t):
    t = t.detach().cpu().float().contiguous()
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path} shape={list(t.shape)} bytes={t.numel()*4}")


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def box_iou(b1, b2):
    ix0 = max(b1[0], b2[0]); iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2]); iy1 = min(b1[3], b2[3])
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh):
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    suppressed = [False] * len(scores)
    keep = []
    for i in idx:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in idx:
            if suppressed[j] or j == i:
                continue
            if box_iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed[j] = True
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--image", default=IMAGE_PATH)
    parser.add_argument("--outdir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)

    print("Loading SAM 3.1 model (transformers)...")
    import json as _json
    _cfg_path = os.path.join(args.model_dir, "config.json")
    with open(_cfg_path) as _f:
        _cfg = _json.load(_f)
    _cfg["vision_config"]["layer_norm_eps"] = 1e-5
    _cfg["vision_config"]["backbone_config"]["layer_norm_eps"] = 1e-5
    _tmp_cfg = os.path.join(args.outdir, "config_eps1e5.json")
    with open(_tmp_cfg, "w") as _f:
        _json.dump(_cfg, _f)
    model = Sam3Model.from_pretrained(
        args.model_dir, torch_dtype=torch.float32, attn_implementation="eager",
        config=_tmp_cfg,
    )
    model.eval()

    tokenize = make_tokenizer(
        os.path.join(args.model_dir, "vocab.json"),
        os.path.join(args.model_dir, "merges.txt"),
    )

    # ── Preprocess image ────────────────────────────────────────────────────
    img = Image.open(args.image).convert("RGB")
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(IMG_SIZE, IMG_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = v2.functional.to_image(img)
    img_preprocessed = transform(img_tensor).unsqueeze(0)
    print(f"  preprocessed: {list(img_preprocessed.shape)}")

    token_ids = tokenize(PROMPT)
    print(f"  token_ids ({PROMPT!r}): {token_ids[:8]}")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = (input_ids != 0).long()

    # ── Hooks ───────────────────────────────────────────────────────────────
    captured = {}
    handles = []

    vit_idx = {0: "00", 7: "07", 15: "15", 23: "23", 31: "31"}
    for i, tag in vit_idx.items():
        def make_vit_hook(i=i, tag=tag):
            def hook(module, args, output):
                captured[f"vit_block_{tag}_out"] = output.detach().float().contiguous()
            return hook
        handles.append(model.vision_encoder.backbone.layers[i].register_forward_hook(make_vit_hook()))

    def vision_hook(module, args, output):
        captured["neck_det_raw"] = output.fpn_hidden_states
    handles.append(model.vision_encoder.register_forward_hook(vision_hook))

    for i in range(6):
        def make_fenc_hook(i=i):
            def hook(module, args, output):
                captured[f"fenc_layer{i}_out"] = output.detach().float().contiguous()
            return hook
        handles.append(model.detr_encoder.layers[i].register_forward_hook(make_fenc_hook()))

    for i in range(6):
        def make_ddec_hook(i=i):
            def hook(module, args, output):
                captured[f"ddec_layer{i}_hidden"] = output.detach().float().contiguous()
            return hook
        handles.append(model.detr_decoder.layers[i].register_forward_hook(make_ddec_hook()))

    def decoder_hook(module, args, output):
        captured["decoder_out"] = output
    handles.append(model.detr_decoder.register_forward_hook(decoder_hook))

    # ── Forward ─────────────────────────────────────────────────────────────
    print("Running forward pass...")
    with torch.no_grad():
        text_out = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out.pooler_output  # (1, 32, 256)
        out = model(pixel_values=img_preprocessed, input_ids=input_ids, attention_mask=attention_mask)

    for h in handles:
        h.remove()

    decoder_out = captured["decoder_out"]
    fpn = captured["neck_det_raw"]

    # C++ ddec_pred_boxes: reference boxes after ALL 6 layers (cxcywh)
    ref_after_l4 = decoder_out.reference_boxes[5]  # (1, 200, 4)
    hidden_after_l5 = captured["ddec_layer5_hidden"]  # (1, 201, 256)
    delta = model.detr_decoder.box_head(
        model.detr_decoder.output_layer_norm(hidden_after_l5[:, 1:])
    )
    final_ref_boxes = torch.sigmoid(inverse_sigmoid(ref_after_l4) + delta)  # (1, 200, 4)

    # C++ ddec_presence_logit: raw presence logit (no clamp)
    presence_raw = model.detr_decoder.presence_head(
        model.detr_decoder.presence_layer_norm(hidden_after_l5[:, :1])
    ).squeeze(-1)  # (1,)

    # ── Save reference files ────────────────────────────────────────────────
    save_tensor(os.path.join(args.outdir, "preprocessed_chw"), img_preprocessed.squeeze(0).contiguous())
    save_tensor(os.path.join(args.outdir, "token_ids"), torch.tensor(token_ids, dtype=torch.float32))

    for i, tag in vit_idx.items():
        save_tensor(os.path.join(args.outdir, f"vit_block_{tag}_out"), captured[f"vit_block_{tag}_out"])

    for i in range(3):
        f = fpn[i]  # (1, C, H, W)
        neck = f.squeeze(0).permute(1, 2, 0).contiguous()  # (H, W, C), C-fastest == ggml [C,W,H,1] byte order
        save_tensor(os.path.join(args.outdir, f"neck_det_{i}"), neck)

    save_tensor(os.path.join(args.outdir, "text_features"), text_features)
    for i in range(6):
        save_tensor(os.path.join(args.outdir, f"fenc_layer{i}_out"), captured[f"fenc_layer{i}_out"])
    save_tensor(os.path.join(args.outdir, "fenc_output"), captured["fenc_layer5_out"])
    for i in range(6):
        save_tensor(os.path.join(args.outdir, f"ddec_layer{i}_out"), captured[f"ddec_layer{i}_hidden"][:, 1:])
    save_tensor(os.path.join(args.outdir, "scoring_output"), out.pred_logits)
    save_tensor(os.path.join(args.outdir, "seg_pred_masks"), out.pred_masks)
    save_tensor(os.path.join(args.outdir, "pred_boxes"), final_ref_boxes.squeeze(0).contiguous())
    save_tensor(os.path.join(args.outdir, "presence_logit"), presence_raw)

    # ── Postprocessing (C++ sam3_segment_pcs semantics) ─────────────────────
    logits = out.pred_logits[0]  # (200,)
    presence_prob = torch.sigmoid(presence_raw[0]).item()
    class_prob = torch.sigmoid(logits)
    scores = (class_prob * presence_prob).cpu().numpy()
    boxes_np = final_ref_boxes[0].detach().cpu().numpy()  # cxcywh in [0,1]
    masks = out.pred_masks[0]  # (200, 288, 288)

    det_boxes = []
    det_scores = []
    det_masks = []
    for q in range(200):
        if scores[q] < SCORE_THRESHOLD:
            continue
        cx, cy, w, h = boxes_np[q]
        x0 = max(0.0, min((cx - 0.5 * w) * IMG_SIZE, IMG_SIZE))
        y0 = max(0.0, min((cy - 0.5 * h) * IMG_SIZE, IMG_SIZE))
        x1 = max(0.0, min((cx + 0.5 * w) * IMG_SIZE, IMG_SIZE))
        y1 = max(0.0, min((cy + 0.5 * h) * IMG_SIZE, IMG_SIZE))
        m = F.interpolate(masks[q].unsqueeze(0).unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False)[0, 0]
        bin_mask = (m > 0.0).float()
        det_boxes.append([x0, y0, x1, y1])
        det_scores.append(scores[q])
        det_masks.append(bin_mask)

    keep = nms(det_boxes, det_scores, NMS_THRESHOLD)
    print(f"  detections above threshold: {len(det_scores)}, after NMS: {len(keep)}")
    for i, top in enumerate(keep[:5]):
        print(f"    #{i}: score={det_scores[top]:.4f} box={[round(v,1) for v in det_boxes[top]]}")

    final_scores = torch.tensor([det_scores[i] for i in keep], dtype=torch.float32)
    final_boxes = torch.tensor([det_boxes[i] for i in keep], dtype=torch.float32)
    final_masks = torch.stack([det_masks[i] for i in keep]) if keep else torch.zeros(0, IMG_SIZE, IMG_SIZE)

    save_tensor(os.path.join(args.outdir, "final_scores"), final_scores)
    save_tensor(os.path.join(args.outdir, "final_boxes_xyxy"), final_boxes)
    save_tensor(os.path.join(args.outdir, "final_masks_binary"), final_masks)
    print("Done.")


if __name__ == "__main__":
    main()
