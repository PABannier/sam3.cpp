# EdgeTAM Support in sam3.cpp — Complete Implementation Plan

> Add EdgeTAM (On-Device Track Anything Model) as a fourth model family in sam3.cpp.
> EdgeTAM is a distilled SAM2 variant using a RepViT backbone and a Perceiver memory compressor.
> It tracks objects 22x faster than SAM2 on mobile devices.

---

## Table of Contents

1. [Context & Motivation](#1-context--motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [EdgeTAM vs SAM2 — Detailed Diff](#3-edgetam-vs-sam2--detailed-diff)
4. [Binary Weight Format Extension](#4-binary-weight-format-extension)
5. [Python Weight Conversion Script](#5-python-weight-conversion-script)
6. [C++ Weight Structs](#6-c-weight-structs)
7. [RepViT Backbone Forward Pass](#7-repvit-backbone-forward-pass)
8. [Spatial Perceiver Forward Pass](#8-spatial-perceiver-forward-pass)
9. [Memory Attention with RoPEv2](#9-memory-attention-with-ropev2)
10. [Reusable Components (No Changes Needed)](#10-reusable-components-no-changes-needed)
11. [Model Loading & Dispatch](#11-model-loading--dispatch)
12. [Pipeline Integration](#12-pipeline-integration)
13. [Tensor Name Mapping Reference](#13-tensor-name-mapping-reference)
14. [Implementation Order (Phased)](#14-implementation-order-phased)
15. [Verification Strategy](#15-verification-strategy)
16. [Appendix: RepViT Block Architecture](#16-appendix-repvit-block-architecture)
17. [Appendix: Perceiver Weight Shapes](#17-appendix-perceiver-weight-shapes)
18. [Appendix: Full Checkpoint Tensor Inventory](#18-appendix-full-checkpoint-tensor-inventory)

---

## 1. Context & Motivation

### What is EdgeTAM?

EdgeTAM is Meta's on-device variant of SAM2, achieving **16 FPS on iPhone 15 Pro Max** (vs 0.7 FPS for SAM2 Base+). It maintains competitive segmentation accuracy (72.3 J&F on SA-V val) while being dramatically faster.

### Key innovations over SAM2:
- **RepViT-M1 backbone** (mobile-optimized convolutional net) replaces Hiera (hierarchical ViT)
- **Spatial Perceiver** compresses memory features from 4096 spatial tokens to 512 latents (8x)
- **2 memory attention layers** instead of 4 (50% reduction)
- **RoPEAttentionv2** handles asymmetric q/k spatial resolutions in cross-attention

### Why add it to sam3.cpp?

EdgeTAM is tiny (~20 MB) and extremely fast. It's the ideal model for real-time tracking on CPU/Metal. Combined with quantization (Q4_0), it could fit in **~5 MB** and achieve interactive frame rates on Apple Silicon.

### What already works in sam3.cpp?

The library supports SAM2 (Hiera backbone) and SAM3 (ViT backbone). EdgeTAM shares ~70% of its pipeline with SAM2 — the FPN neck, prompt encoder, mask decoder, memory encoder, and video tracking logic are identical or nearly identical. Only the backbone, perceiver, and memory attention differ.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      EdgeTAM Architecture                            │
│                                                                      │
│  ┌─────────────┐   ┌──────────┐                                    │
│  │  RepViT-M1  │──▶│ FPN Neck │──▶ 3 feature levels (256-dim)      │
│  │ (20 blocks, │   │(4 1×1 +  │    ┌─────────────────────────┐     │
│  │  4 stages)  │   │ top-down)│    │ feat[0]: 256×256 (4×)   │     │
│  │ [48→384 ch] │   │ scalp=1  │    │ feat[1]: 128×128 (8×)   │     │
│  └─────────────┘   └──────────┘    │ feat[2]:  64× 64 (16×)  │     │
│                                     └───────────┬───────────┘       │
│                                                  │                   │
│  ┌───────────────────────────────────────────────┤                   │
│  │                TRACKER PATH                    │                   │
│  │                                                │                   │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────┐ │                   │
│  │  │  Memory    │  │  Memory  │  │  SAM Mask  │ │                   │
│  │  │ Attention  │  │  Bank    │  │  Decoder   │ │                   │
│  │  │ (2 layers) │  │(7 slots) │  │(2WayTrfm) │ │                   │
│  │  │ RoPEv2 CA  │  │          │  │ 8 heads    │ │                   │
│  │  └────────────┘  └────┬─────┘  └───────────┘ │                   │
│  │                       │                        │                   │
│  │  ┌────────────┐  ┌────┴────────┐              │                   │
│  │  │  Spatial   │  │  Memory     │              │                   │
│  │  │ Perceiver  │──▶│  Encoder   │              │                   │
│  │  │(2 layers,  │  │ (CXBlock    │              │                   │
│  │  │ 512 latent)│  │  fuser)     │              │                   │
│  │  └────────────┘  └─────────────┘              │                   │
│  └────────────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Tensor flow through the pipeline:

```
Image (3, 1024, 1024)
  │
  ▼
RepViT Stem: Conv(3→24, k=3, s=2) + BN + GELU → Conv(24→48, k=3, s=2) + BN
  │
  ▼ (48, 256, 256)
Stage 0: 2 RepViT blocks (ch=48)
  │ output → FPN xs[0] (256×256, 48ch)
  ▼ Downsample (48→96, spatial/2)
Stage 1: 2 RepViT blocks (ch=96)
  │ output → FPN xs[1] (128×128, 96ch)
  ▼ Downsample (96→192, spatial/2)
Stage 2: 14 RepViT blocks (ch=192)
  │ output → FPN xs[2] (64×64, 192ch) — primary feature map
  ▼ Downsample (192→384, spatial/2)
Stage 3: 2 RepViT blocks (ch=384)
  │ output → FPN xs[3] (32×32, 384ch) — coarsest
  ▼

FPN: 4 lateral 1×1 convs (all → 256-dim) + top-down fusion at indices [2, 3]
  │ Output: features[0..3] = [256×256, 128×128, 64×64, 32×32] (fine→coarse)
  │ scalp=1: discard features[-1] (32×32, coarsest)
  │ Keep: features[0..2] = [256×256, 128×128, 64×64]
  ▼

features[-1] = 64×64 (256-dim) → vision_features → memory attention → SAM decoder → mask
features[0] + features[1] → high-res features for SAM decoder upscaling
```

---

## 3. EdgeTAM vs SAM2 — Detailed Diff

### Components that are IDENTICAL (reuse existing SAM2 code):

| Component | Notes |
|-----------|-------|
| FPN Neck | Same architecture: 4 lateral 1×1 convs → 256d, top-down at [2,3], scalp=1 |
| Prompt Encoder | Same: PositionEmbeddingRandom + point/box/mask embeddings |
| SAM Mask Decoder | Same: TwoWayTransformer(depth=2, 8 heads) + hypernetwork MLPs |
| Memory Encoder | Same: MaskDownSampler + CXBlock fuser + out_proj(256→64) |
| Object Pointer Projection | Same: MLP(256→256→256→256) with 3 layers |
| Video Tracking State | Same: memory bank (7 slots), object pointers, temporal encoding |

### Components that DIFFER:

| Component | SAM2 (Hiera) | EdgeTAM (RepViT) |
|-----------|-------------|-------------------|
| **Backbone** | Hiera transformer (12-48 blocks, window attn, Q-stride) | RepViT-M1 (20 blocks, DW-conv + SE + channel mixer) |
| **Backbone channels** | [96-1152] growing via 2× pooling | [48, 96, 192, 384] fixed per stage |
| **Backbone normalization** | LayerNorm | BatchNorm (fused into conv at export) |
| **Memory attention layers** | 4 | **2** |
| **Cross-attention type** | RoPEAttention (same q/k sizes) | **RoPEAttentionv2** (asymmetric q/k sizes) |
| **Cross-attn kv_in_dim** | 64 | 64 (same, but via RoPEv2) |
| **Cross-attn q_sizes** | [32, 32] | **[64, 64]** |
| **Cross-attn k_sizes** | [32, 32] | **[16, 16]** |
| **Spatial Perceiver** | None | **PerceiverResampler (2 layers, 512 latents)** |
| **Mask downsampler stride** | 4 (k=4, s=4, p=0) per layer, 2 layers → total 16× | **2 (k=3, s=2, p=1) per layer, 4 layers → total 16×** |
| **no_obj_embed_spatial** | Present in SAM2.1 | **Not present** |

### Config values from `edgetam.yaml`:

```yaml
image_size: 1024
backbone_channel_list: [384, 192, 96, 48]
fpn_top_down_levels: [2, 3]
scalp: 1
d_model: 256
mem_out_dim: 64
mem_attn_layers: 2
num_maskmem: 7
sigmoid_scale_for_mem_enc: 20.0
sigmoid_bias_for_mem_enc: -10.0
directly_add_no_mem_embed: true
use_high_res_features_in_sam: true
pred_obj_scores: true (with MLP)
fixed_no_obj_ptr: true
use_mlp_for_obj_ptr_proj: true
multimask_output_in_sam: true
multimask_output_for_tracking: true
use_multimask_token_for_obj_ptr: true
iou_prediction_use_sigmoid: true
add_tpos_enc_to_obj_ptrs: false
```

---

## 4. Binary Weight Format Extension

### Option A (recommended): Reuse SAM2 magic with new backbone_type

The SAM2 file header already has a `backbone_type` field (currently always `1` for Hiera). We extend it:

```
backbone_type = 1  →  Hiera (existing SAM2)
backbone_type = 2  →  RepViT + Perceiver (EdgeTAM)
```

This avoids a new magic number. The loader dispatches on `backbone_type` after reading the SAM2 header.

### Extended header fields (when backbone_type == 2):

After the existing SAM2 header fields, append EdgeTAM-specific hyperparameters:

```
┌─────────────────────────────────────────────┐
│ ... existing SAM2 header (backbone_type=2)  │
│                                              │
│ === EdgeTAM-specific block (new) ===         │
│ [4 bytes]  repvit_num_stages: 4              │
│ [4 bytes]  repvit_stages[0]: 2               │
│ [4 bytes]  repvit_stages[1]: 2               │
│ [4 bytes]  repvit_stages[2]: 14              │
│ [4 bytes]  repvit_stages[3]: 2               │
│ [4 bytes]  repvit_channels[0]: 48            │
│ [4 bytes]  repvit_channels[1]: 96            │
│ [4 bytes]  repvit_channels[2]: 192           │
│ [4 bytes]  repvit_channels[3]: 384           │
│ [4 bytes]  repvit_se_ratio_x100: 25          │
│ [4 bytes]  has_spatial_perceiver: 1           │
│ [4 bytes]  perceiver_depth: 2                │
│ [4 bytes]  perceiver_dim: 64                 │
│ [4 bytes]  perceiver_num_latents_1d: 256     │
│ [4 bytes]  perceiver_num_latents_2d: 256     │
│ [4 bytes]  perceiver_ff_mult: 4              │
│ [4 bytes]  mem_attn_ca_type: 1               │  (0=RoPEv1, 1=RoPEv2)
│ [4 bytes]  mem_attn_ca_q_size: 64            │
│ [4 bytes]  mem_attn_ca_k_size: 16            │
└─────────────────────────────────────────────┘
```

The existing header fields `hiera_embed_dim`, `hiera_stages`, etc. are ignored when `backbone_type == 2` (they can be written as 0 or dummy values).

---

## 5. Python Weight Conversion Script

### New file: `convert_edgetam_to_ggml.py`

This script converts the EdgeTAM PyTorch checkpoint to the ggml binary format. Key responsibilities:

### 5.1 BatchNorm Fusion

RepViT uses BatchNorm everywhere. Since ggml has no BN op, we **fuse BN into the preceding convolution** during conversion.

Additionally, the RepViT token mixer uses **RepVGG-style reparameterization**: three parallel branches (DW 3×3 + 1×1 + identity) are summed and passed through a final BN. At inference time, all three branches + final BN must be fused into a **single DW 3×3 conv + bias**.

#### BN Fusion Helper

```python
def fuse_bn_into_conv(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d."""
    # BN: y = (x - mean) / sqrt(var + eps) * weight + bias
    # Fused: W_fused = W_conv * bn_weight / sqrt(bn_var + eps)
    #         b_fused = (b_conv - bn_mean) * bn_weight / sqrt(bn_var + eps) + bn_bias
    bn_std = (bn_var + eps).sqrt()
    scale = bn_weight / bn_std

    # For depthwise conv (groups == in_channels), scale is per-channel
    if conv_weight.dim() == 4:
        fused_weight = conv_weight * scale.view(-1, 1, 1, 1)
    else:
        fused_weight = conv_weight * scale

    if conv_bias is not None:
        fused_bias = (conv_bias - bn_mean) * scale + bn_bias
    else:
        fused_bias = -bn_mean * scale + bn_bias

    return fused_weight, fused_bias
```

#### RepVGG Reparameterization (Token Mixer)

The token mixer (`RepVggDw`) has the forward pass: `out = final_BN(DWConv3x3(x) + Conv1x1(x) + x)`.
At export time, we fuse all three branches + final BN into a single DW 3×3 conv:

```python
def repvgg_reparameterize(dw_conv_w, dw_bn, conv1_w, conv1_bn, final_bn, ch, eps=1e-5):
    """Fuse RepVGG 3-way parallel branches + optional final BN into single DW 3x3.
    
    PyTorch depthwise conv weight layout: (ch, 1, kH, kW)
    final_bn: tuple (weight, bias, mean, var) or None for legacy mode (nn.Identity)
    """
    # 1. Fuse each branch's internal BN into its conv
    dw_fused_w, dw_fused_b = fuse_bn_into_conv(dw_conv_w, None, *dw_bn, eps=eps)
    c1_fused_w, c1_fused_b = fuse_bn_into_conv(conv1_w, None, *conv1_bn, eps=eps)

    # 2. Pad 1×1 kernel to 3×3 (zeros around center pixel)
    # PyTorch layout: (ch, 1, 1, 1) → F.pad pads last 2 dims → (ch, 1, 3, 3)
    c1_padded = F.pad(c1_fused_w, [1, 1, 1, 1])

    # 3. Identity kernel: center pixel = 1 for each channel (depthwise)
    # PyTorch layout: (ch, 1, 3, 3) — index [:, 0, 1, 1] is center of 3×3
    id_kernel = torch.zeros_like(dw_fused_w)       # (ch, 1, 3, 3)
    id_kernel[:, 0, 1, 1] = 1.0

    # 4. Sum all three 3×3 kernels and biases
    merged_w = dw_fused_w + c1_padded + id_kernel
    merged_b = dw_fused_b + c1_fused_b             # identity branch has zero bias

    # 5. Fuse final BN into the merged conv (skip if legacy mode / nn.Identity)
    if final_bn is not None:
        final_w, final_b = fuse_bn_into_conv(merged_w, merged_b, *final_bn, eps=eps)
        return final_w, final_b
    return merged_w, merged_b  # legacy mode: no final BN, merged result is final
```

#### Legacy mode handling

EdgeTAM uses `repvit_m1.dist_in1k` which sets `legacy=True` in TIMM. In legacy mode:
- `RepVggDw.conv1` is `ConvNorm` (1×1 conv + BN) — has BN parameters
- `RepVggDw.bn` is `nn.Identity()` — **no state_dict entries** (no final BN to fuse)

The conversion script must detect that `token_mixer.bn` has no weight/bias keys (since it's Identity) and skip the final BN fusion step in `repvgg_reparameterize()`. In that case, the merged kernel from summing the three branches is the final output (no step 5).

#### Where each fusion applies:

- **Stem:** conv1 + BN → fuse. conv2 + BN → fuse. (standard BN fusion, no RepVGG)
- **Each block's token_mixer:** conv (DW 3×3) + conv.bn + conv1 (1×1) + conv1.bn → sum branches → **no final BN fusion** (legacy mode, bn is Identity). Output: single DW 3×3 weight + bias
- **Each block's channel_mixer:** conv1 + BN → fuse. conv2 + BN → fuse. (standard BN fusion)
- **SE blocks:** fc1, fc2 — Conv2d with bias, NO BN. Written as-is.
- **Downsample pre_block token_mixer:** Same RepVGG reparameterization as above (legacy mode)
- **Downsample spatial/channel/ffn convs:** Standard BN fusion

### 5.2 Tensor Name Mapping

```python
# ── RepViT backbone ──────────────────────────────────────────────
# After BN fusion + RepVGG reparameterization:
#   - Token mixer: 3 branches + final BN → single DW 3×3 weight + bias
#   - All other conv+BN pairs → single weight + bias

# Stem
image_encoder.trunk.body.stem.conv1.c  + .bn  →  repvit.stem.conv1.weight/bias
image_encoder.trunk.body.stem.conv2.c  + .bn  →  repvit.stem.conv2.weight/bias

# Stage blocks — token mixer (RepVGG reparameterized)
# Input tensors consumed by repvgg_reparameterize():
#   token_mixer.conv.c + token_mixer.conv.bn     (DW 3×3 branch)
#   token_mixer.conv1.c + token_mixer.conv1.bn   (1×1 branch, may lack BN in non-legacy)
#   token_mixer.bn                                (final BN after 3-way add)
# Output: single fused DW 3×3
image_encoder.trunk.body.stages_{s}.blocks.{b}.token_mixer.{conv,conv1,bn}
    → repvit.stages.{s}.blocks.{b}.tm.weight/bias  (fused DW 3×3)

image_encoder.trunk.body.stages_{s}.blocks.{b}.se.fc1
    → repvit.stages.{s}.blocks.{b}.se.fc1.weight/bias    (no BN fusion)
image_encoder.trunk.body.stages_{s}.blocks.{b}.se.fc2
    → repvit.stages.{s}.blocks.{b}.se.fc2.weight/bias    (no BN fusion)
image_encoder.trunk.body.stages_{s}.blocks.{b}.channel_mixer.conv1.c + .bn
    → repvit.stages.{s}.blocks.{b}.cm.conv1.weight/bias  (1×1 expand)
image_encoder.trunk.body.stages_{s}.blocks.{b}.channel_mixer.conv2.c + .bn
    → repvit.stages.{s}.blocks.{b}.cm.conv2.weight/bias  (1×1 project)

# Downsample modules (between stages 0→1, 1→2, 2→3)
image_encoder.trunk.body.stages_{s}.downsample.spatial_downsample.c + .bn
    → repvit.stages.{s}.ds.spatial.weight/bias
image_encoder.trunk.body.stages_{s}.downsample.channel_downsample.c + .bn
    → repvit.stages.{s}.ds.channel.weight/bias
# Downsample pre_block token mixer — also RepVGG reparameterized
image_encoder.trunk.body.stages_{s}.downsample.pre_block.token_mixer.{conv,conv1,bn}
    → repvit.stages.{s}.ds.pre.tm.weight/bias  (fused DW 3×3)
image_encoder.trunk.body.stages_{s}.downsample.pre_block.channel_mixer.conv1.c + .bn
    → repvit.stages.{s}.ds.pre.cm.conv1.weight/bias
image_encoder.trunk.body.stages_{s}.downsample.pre_block.channel_mixer.conv2.c + .bn
    → repvit.stages.{s}.ds.pre.cm.conv2.weight/bias
image_encoder.trunk.body.stages_{s}.downsample.ffn.conv1.c + .bn
    → repvit.stages.{s}.ds.ffn.conv1.weight/bias
image_encoder.trunk.body.stages_{s}.downsample.ffn.conv2.c + .bn
    → repvit.stages.{s}.ds.ffn.conv2.weight/bias

# ── FPN Neck (same as SAM2) ──────────────────────────────────────
image_encoder.neck.convs.{i}.conv  → fpn.convs.{i}.weight/bias

# ── Spatial Perceiver (NEW) ──────────────────────────────────────
spatial_perceiver.latents                → perceiver.latents
spatial_perceiver.latents_2d             → perceiver.latents_2d
spatial_perceiver.norm.weight/bias       → perceiver.norm.weight/bias
spatial_perceiver.layers.{i}.attn.norm_latents.weight/bias
    → perceiver.layers.{i}.ca.norm_latents.weight/bias
spatial_perceiver.layers.{i}.attn.norm_x.weight/bias
    → perceiver.layers.{i}.ca.norm_x.weight/bias
spatial_perceiver.layers.{i}.attn.to_q.weight
    → perceiver.layers.{i}.ca.q.weight      (no bias)
spatial_perceiver.layers.{i}.attn.to_kv.weight
    → perceiver.layers.{i}.ca.kv.weight     (no bias, shape [128, 64])
spatial_perceiver.layers.{i}.attn.to_out.weight
    → perceiver.layers.{i}.ca.out.weight    (no bias)
spatial_perceiver.layers.{i}.ff.0.weight/bias  → perceiver.layers.{i}.ff.norm.weight/bias  (LayerNorm)
spatial_perceiver.layers.{i}.ff.1.weight       → perceiver.layers.{i}.ff.fc1.weight        (no bias)
spatial_perceiver.layers.{i}.ff.3.weight       → perceiver.layers.{i}.ff.fc2.weight        (no bias)
spatial_perceiver.layers.{i}.self_attn.norm.weight/bias
    → perceiver.layers.{i}.sa.norm.weight/bias
spatial_perceiver.layers.{i}.self_attn.to_q.weight
    → perceiver.layers.{i}.sa.q.weight
spatial_perceiver.layers.{i}.self_attn.to_kv.weight
    → perceiver.layers.{i}.sa.kv.weight
spatial_perceiver.layers.{i}.self_attn.to_out.weight
    → perceiver.layers.{i}.sa.out.weight
spatial_perceiver.layers.{i}.self_ff.0.weight/bias  → perceiver.layers.{i}.sa_ff.norm.weight/bias
spatial_perceiver.layers.{i}.self_ff.1.weight       → perceiver.layers.{i}.sa_ff.fc1.weight
spatial_perceiver.layers.{i}.self_ff.3.weight       → perceiver.layers.{i}.sa_ff.fc2.weight

# ── Memory Attention, Memory Encoder, SAM Prompt/Mask Decoder ────
# Same renaming rules as convert_sam2_to_ggml.py (see that file)

# ── Top-level parameters ─────────────────────────────────────────
maskmem_tpos_enc                → mem_enc.tpos_enc
mask_downsample.weight/bias     → trk_mask_ds.weight/bias
no_mem_embed                    → no_mem_embed
no_mem_pos_enc                  → no_mem_pos_enc
no_obj_ptr                      → no_obj_ptr
obj_ptr_proj.layers.{i}        → obj_ptr_proj.layers.{i}
```

### 5.3 Tensors to Skip

```python
skip_patterns = [
    "num_batches_tracked",  # BN running stats (consumed by fusion/reparam)
    "running_mean",         # BN running stats (consumed by fusion/reparam)
    "running_var",          # BN running stats (consumed by fusion/reparam)
    # BN weight/bias are consumed by fuse_bn_into_conv / repvgg_reparameterize
    # token_mixer.conv.bn, token_mixer.conv1.bn, token_mixer.bn → all consumed by RepVGG reparam
]
```

### 5.4 Conversion Command

```bash
uv run python convert_edgetam_to_ggml.py \
    --model ~/Documents/EdgeTAM/checkpoints/edgetam.pt \
    --output models/edgetam_f16.ggml \
    --ftype 1
```

### 5.5 Expected tensor count

After BN fusion, RepVGG reparameterization, and removal of running stats:
- RepViT backbone: ~120 tensors (token mixer is now 1 DW 3×3 w+b instead of 2 conv w+b each)
- FPN neck: 8 tensors
- Perceiver: 42 tensors
- Memory attention (2 layers): 54 tensors
- Memory encoder: 20 tensors
- SAM prompt encoder: 17 tensors
- SAM mask decoder: ~130 tensors
- Top-level: ~10 tensors
- **Total: ~400 tensors**

---

## 6. C++ Weight Structs

### 6.1 RepViT Block

```cpp
// A single RepViT block (token mixer + optional SE + channel mixer)
struct edgetam_repvit_block {
    // Token mixer: single fused DW 3×3 conv (after RepVGG reparameterization)
    // Original: BN(DW3x3(x) + Conv1x1(x) + x) → fused into single DW 3×3
    ggml_tensor * tm_w;          // [3, 3, 1, ch_in]        (RepVGG-fused depthwise)
    ggml_tensor * tm_b;          // [ch_in]

    // Squeeze-and-excitation (only on even-indexed blocks)
    bool has_se;
    ggml_tensor * se_fc1_w;      // [1, 1, ch_in, ch_rd]    (Conv2d 1×1, ch_rd ≈ ch_in/4)
    ggml_tensor * se_fc1_b;      // [ch_rd]
    ggml_tensor * se_fc2_w;      // [1, 1, ch_rd, ch_in]
    ggml_tensor * se_fc2_b;      // [ch_in]

    // Channel mixer: 1×1 expand → 1×1 project
    ggml_tensor * cm_conv1_w;    // [1, 1, ch_in, ch_in*2]  (fused with BN)
    ggml_tensor * cm_conv1_b;    // [ch_in*2]
    ggml_tensor * cm_conv2_w;    // [1, 1, ch_in*2, ch_in]  (fused with BN)
    ggml_tensor * cm_conv2_b;    // [ch_in]
};
```

### 6.2 RepViT Downsample

```cpp
// Downsample module between stages
struct edgetam_repvit_downsample {
    // Pre-block (RepViT block at prev-stage channels, uses RepVGG-fused token mixer)
    // SE is always disabled in pre_block (use_se=False in TIMM RepVitDownsample)
    edgetam_repvit_block pre_block;  // pre_block.has_se = false always

    // Spatial downsample: DW Conv 3×3, stride=2
    ggml_tensor * spatial_w;     // [3, 3, 1, ch_in]  (depthwise, fused with BN)
    ggml_tensor * spatial_b;     // [ch_in]

    // Channel expand: 1×1 Conv
    ggml_tensor * channel_w;     // [1, 1, ch_in, ch_out]  (fused with BN)
    ggml_tensor * channel_b;     // [ch_out]

    // FFN: 1×1 expand → 1×1 project
    ggml_tensor * ffn_conv1_w;   // [1, 1, ch_out, ch_out*2]  (fused with BN)
    ggml_tensor * ffn_conv1_b;   // [ch_out*2]
    ggml_tensor * ffn_conv2_w;   // [1, 1, ch_out*2, ch_out]  (fused with BN)
    ggml_tensor * ffn_conv2_b;   // [ch_out]
};
```

### 6.3 RepViT Stage

```cpp
struct edgetam_repvit_stage {
    std::vector<edgetam_repvit_block> blocks;
    // Downsample exists for stages 1, 2, 3 (not stage 0)
    bool has_downsample;
    edgetam_repvit_downsample downsample;
};
```

### 6.4 Full RepViT Backbone

```cpp
struct edgetam_repvit {
    // Stem: 2 conv layers (3→24→48, each stride 2)
    ggml_tensor * stem_conv1_w;  // [3, 3, 3, 24]  (fused with BN)
    ggml_tensor * stem_conv1_b;  // [24]
    ggml_tensor * stem_conv2_w;  // [3, 3, 24, 48]  (fused with BN)
    ggml_tensor * stem_conv2_b;  // [48]

    // 4 stages
    edgetam_repvit_stage stages[4];
    // stages[0]: 2 blocks (ch=48),  no downsample before it (stem does it)
    // stages[1]: 2 blocks (ch=96),  downsample before it at start of stage
    // stages[2]: 14 blocks (ch=192), downsample before it
    // stages[3]: 2 blocks (ch=384), downsample before it
};
```

### 6.5 Perceiver Resampler

```cpp
struct edgetam_perceiver_layer {
    // Cross-attention (latents attend to memory features)
    ggml_tensor * ca_norm_latents_w, * ca_norm_latents_b;  // [64]
    ggml_tensor * ca_norm_x_w, * ca_norm_x_b;              // [64]
    ggml_tensor * ca_q_w;                                    // [64, 64]  (no bias)
    ggml_tensor * ca_kv_w;                                   // [128, 64] (no bias, projects to k and v)
    ggml_tensor * ca_out_w;                                  // [64, 64]  (no bias)

    // FFN after cross-attention
    ggml_tensor * ff_norm_w, * ff_norm_b;  // [64] LayerNorm
    ggml_tensor * ff_fc1_w;                 // [256, 64]  (no bias)
    ggml_tensor * ff_fc2_w;                 // [64, 256]  (no bias)

    // Self-attention on latents
    ggml_tensor * sa_norm_w, * sa_norm_b;  // [64]
    ggml_tensor * sa_q_w;                   // [64, 64]  (no bias)
    ggml_tensor * sa_kv_w;                  // [128, 64] (no bias)
    ggml_tensor * sa_out_w;                 // [64, 64]  (no bias)

    // FFN after self-attention
    ggml_tensor * sa_ff_norm_w, * sa_ff_norm_b;  // [64]
    ggml_tensor * sa_ff_fc1_w;                     // [256, 64]
    ggml_tensor * sa_ff_fc2_w;                     // [64, 256]
};

struct edgetam_perceiver {
    ggml_tensor * latents_1d;    // [256, 64]  learnable 1D latent tokens
    ggml_tensor * latents_2d;    // [256, 64]  learnable 2D latent tokens
    ggml_tensor * norm_w, * norm_b;  // [64]  final LayerNorm

    std::vector<edgetam_perceiver_layer> layers;  // [perceiver_depth = 2]
};
```

### 6.6 Model Struct Extension

```cpp
struct sam3_model {
    sam3_hparams hparams;

    // Existing backbones
    sam3_vit vit;                    // SAM3 ViT
    sam2_hiera hiera;                // SAM2 Hiera

    // NEW: EdgeTAM backbone + perceiver
    edgetam_repvit repvit;
    edgetam_perceiver perceiver;

    // Shared components (same as before)
    sam2_fpn_neck fpn;
    sam3_sam_prompt_enc sam_pe;
    sam3_sam_mask_dec sam_dec;
    sam3_mem_enc mem_enc;
    sam3_mem_attn mem_attn;

    // ... existing SAM3-only components (neck_det, neck_trk, text, etc.)
};
```

---

## 7. RepViT Backbone Forward Pass

### 7.1 Stem

```
Input: (B, 3, 1024, 1024)
  → Conv2d(3, 24, k=3, s=2, p=1) + fused_BN → GELU    → (B, 24, 512, 512)
  → Conv2d(24, 48, k=3, s=2, p=1) + fused_BN           → (B, 48, 256, 256)
```

The stem uses stride-2 convolutions, reducing spatial resolution by 4× and setting channel count to 48. **Note:** Only the first stem conv has GELU activation. The second stem conv has NO activation (matches TIMM `RepVitStem`).

**ggml ops:**
```cpp
x = ggml_conv_2d(ctx, stem_conv1_w, input, 2, 2, 1, 1, 1, 1);
x = ggml_add(ctx, x, stem_conv1_b);
x = ggml_gelu_erf(ctx, x);               // GELU after conv1 only
x = ggml_conv_2d(ctx, stem_conv2_w, x, 2, 2, 1, 1, 1, 1);
x = ggml_add(ctx, x, stem_conv2_b);       // NO GELU after conv2
```

### 7.2 RepViT Block Forward

Each block has three components: **token mixer** → **SE** (optional) → **channel mixer**, with a single residual connection spanning only the channel mixer:

```
RepViT Block Forward:
  x = DWConv3×3_fused(x)                            (reparameterized token mixer)
  x = SE(x) if has_se else x                        (squeeze-and-excitation)
  identity = x                                       (save AFTER SE for residual)
  x = ChannelMixer(x)                               (1×1 expand → GELU → 1×1 project)
  output = identity + x                              (residual skips channel mixer only)
```

**Key architectural detail:** The identity is captured **after SE**, not before it. The residual connection only spans the channel mixer. This matches TIMM's `RepViTBlock.forward()`:
```python
x = self.token_mixer(x)       # RepVggDw (fused to single DW 3×3 at export)
x = self.se(x)                # SE or Identity
identity = x                   # save AFTER SE
x = self.channel_mixer(x)
return identity + x            # residual only skips channel mixer
```

**Token Mixer (after RepVGG reparameterization):**
```
x_tm = DWConv3×3(x, stride=1, pad=1)  + bias      (single fused op)
```
**ggml ops:** `ggml_conv_2d_dw(ctx, tm_w, x, 1, 1, 1, 1, 1, 1)` → `ggml_add(ctx, ..., tm_b)`.

**Squeeze-and-Excitation detail (activation is ReLU, not GELU):**
```
SE(x):
  s = global_avg_pool(x)          → (B, ch, 1, 1)
  s = Conv1×1(ch, ch_rd)(s)      → (B, ch_rd, 1, 1)  + bias   [ch_rd = make_divisible(ch*0.25, 8)]
  s = ReLU(s)
  s = Conv1×1(ch_rd, ch)(s)      → (B, ch, 1, 1)     + bias
  s = Sigmoid(s)
  return x * s                    (channel-wise scaling)
```

**Channel Mixer:**
```
x_cm = Conv1×1(ch, ch*2)(x) + bias → GELU            (expand)
x_cm = Conv1×1(ch*2, ch)(x_cm) + bias                (project)
```

**ggml ops:**
- Token mixer DW conv: `ggml_conv_2d_dw_direct(ctx, tm_w_f32, x, 1, 1, 1, 1, 1, 1)` — **must cast kernel to F32** if quantized (see `sam3_conv_transpose_weight()` pattern). Use `_direct` variant, not `ggml_conv_2d_dw`.
- SE global avg pool: `ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, H, W, H, W, 0, 0)`
- SE 1×1 convs: `ggml_conv_2d_sk_p0` or `ggml_mul_mat` after reshape
- SE activation: `ggml_relu` (NOT gelu), `ggml_sigmoid`, `ggml_mul`
- Channel mixer 1×1 convs: `ggml_conv_2d_sk_p0`
- Channel mixer activation: `ggml_gelu_erf` (matches existing sam3.cpp code)

### 7.3 Downsample Module Forward

Between stages, spatial resolution halves and channel count doubles:

```
Downsample(x):  (B, ch_in, H, W) → (B, ch_out, H/2, W/2)
  x = pre_block(x)                          (RepViT block at ch_in, uses RepVGG token mixer)
  x = DWConv3×3(ch_in, ch_in, s=2) + BN     (spatial downsample — use ggml_conv_2d_dw_direct with stride=2)
  x = Conv1×1(ch_in, ch_out) + BN           (channel expansion)
  x_in = x
  x = Conv1×1(ch_out, ch_out*2) + BN → GELU (FFN expand)
  x = Conv1×1(ch_out*2, ch_out) + BN        (FFN project)
  x = x + x_in                               (residual)
```

**ggml ops:** Spatial downsample uses `ggml_conv_2d_dw_direct(ctx, spatial_w_f32, x, 2, 2, 1, 1, 1, 1)` (stride=2 depthwise, F32 kernel required). Channel expansion and FFN use `ggml_conv_2d_sk_p0` (1×1 convs).

### 7.4 Full Forward Pass

```
x = stem(image)                                    → (B, 48, 256, 256)
x = stage_0.blocks[0..1](x)                        → (B, 48, 256, 256)
xs[0] = x                                           (FPN input[0], 256×256 — discarded by scalp)
x = stage_1.downsample(x)                          → (B, 96, 128, 128)
x = stage_1.blocks[0..1](x)                        → (B, 96, 128, 128)
xs[1] = x                                           (FPN input[1], 128×128 — high-res for SAM)
x = stage_2.downsample(x)                          → (B, 192, 64, 64)
x = stage_2.blocks[0..13](x)                       → (B, 192, 64, 64)
xs[2] = x                                           (FPN input[2], 64×64 — primary feature)
x = stage_3.downsample(x)                          → (B, 384, 32, 32)
x = stage_3.blocks[0..1](x)                        → (B, 384, 32, 32)
xs[3] = x                                           (FPN input[3], 32×32 — coarsest)

outputs: [xs[0], xs[1], xs[2], xs[3]]  (ordered fine→coarse)
         channels: [48, 96, 192, 384]
         spatial:  [256², 128², 64², 32²]
After FPN + scalp=1: features = [256×256, 128×128, 64×64]  (32×32 discarded)
```

**SE block presence pattern** (derived from checkpoint):
```
Stage 0: blocks with SE = [0]      (block 0 has SE, block 1 doesn't)
Stage 1: blocks with SE = [0]
Stage 2: blocks with SE = [0, 2, 4, 6, 8, 10, 12]  (every even block)
Stage 3: blocks with SE = [0]
```

This pattern can be hard-coded or derived: **SE on the first block of each pair** (every even-indexed block).

### 7.5 Graph Isolation

Following CLAUDE.md rules, the RepViT + FPN must be a single self-contained sub-graph:

```cpp
static bool edgetam_encode_image(
    sam3_state & state, const sam3_model & model, const sam3_image & img)
{
    // 1. Create ggml_context + ggml_gallocr
    // 2. Build RepViT graph: stem → stages → collect 4 feature maps
    // 3. Build FPN graph: 4 lateral 1×1 convs → top-down → scalp
    // 4. ggml_gallocr_alloc → set input (preprocessed image) → compute
    // 5. Copy 3 FPN outputs to state.neck_trk[0..2] via CPU
    // 6. Compute sinusoidal PE for each level
    // 7. Free context + gallocr
    return true;
}
```

---

## 8. Spatial Perceiver Forward Pass

### 8.1 When it runs

The perceiver runs **after the memory encoder** and **before the memory bank stores the features**. It compresses (B, 64, H, W) memory features into (B, 512, 64) latent tokens.

### 8.2 Forward: 1D Path

```
Input: maskmem_features  (B, 64, H, W)  where H=W=64 (for 1024 image)

1D path:
  x = flatten(maskmem_features)           → (B, H*W, 64) = (B, 4096, 64)
  pos = flatten(maskmem_pos_enc)          → (B, H*W, 64) = (B, 4096, 64)  [always passed when not None]
  latents = expand(perceiver.latents_1d)  → (B, 256, 64)

  For each perceiver layer:
    // Cross-attention: latents attend to flattened features (1 head, scale = 1/sqrt(64) = 0.125)
    latents_norm = LayerNorm(latents)
    x_norm = LayerNorm(x)
    q = Linear(latents_norm)              → (B, 256, 64)
    k, v = chunk(Linear(x_norm), 2)      → each (B, 4096, 64)
    k = k + pos, v = v + pos             (pos added to BOTH K and V, AFTER projection)
    attn = scaled_dot_product(q, k, v, scale=0.125)  → (B, 256, 64)
    out = Linear(attn)                    → (B, 256, 64)
    latents = latents + out               (residual)

    // FFN
    latents = latents + FFN(LayerNorm(latents))
      where FFN = Linear(64→256) → GELU → Linear(256→64)

    // Self-attention on latents
    latents_norm = LayerNorm(latents)
    q = Linear(latents_norm)
    k, v = chunk(Linear(latents_norm), 2)
    attn = scaled_dot_product(q, k, v)
    latents = latents + Linear(attn)

    // FFN after self-attention
    latents = latents + FFN(LayerNorm(latents))

  latents_1d = LayerNorm(latents)          → (B, 256, 64)
  pos_1d = zeros_like(latents_1d)          → (B, 256, 64)  (zeroed out for 1D path;
                                              only set when input pos was not None)
```

**Perceiver ggml notes:**
- Attention scale: `1/sqrt(64) = 0.125` — pass to `ggml_flash_attn_ext(ctx, q, k, v, nullptr, 0.125f, 0.0f, 0.0f)`
- FFN activation: **GELU** (not ReLU). This differs from memory attention FFN which uses ReLU.
- All perceiver Linear layers have **no bias** — use `ggml_mul_mat` only (no `ggml_add` for bias)
- Pos encoding: added to **both K and V** in perceiver cross-attention, **AFTER projection** (differs from memory attention cross-attention which adds pos to K only, BEFORE projection)
- **`pos_enc_at_key_value` flag is dead code in the Python source** (lines 279-284 of perceiver.py: the second `if pos is not None` unconditionally overwrites the flag's effect). The actual behavior: pos is always passed to attention layers when not None. The C++ should NOT implement a `pos_enc_at_key_value` gate — just always pass pos when available.

### 8.3 Forward: 2D Path

```
Input: maskmem_features  (B, 64, H, W)

2D path:
  x = permute(maskmem_features)            → (B, H, W, 64)
  num_window = sqrt(256) = 16
  window_size = H / 16 = 4                 (for H=64)
  x = window_partition(x, window_size=4)   → (B*16*16, 4, 4, 64) = (B*256, 4, 4, 64)
  x = flatten(x, dims=1..2)               → (B*256, 16, 64)

  latents = expand(perceiver.latents_2d)   → (B*256, 1, 64)  [1 latent per window]

  For each perceiver layer:
    // Cross-attention: 1 latent per window attends to 16 window tokens
    (same ops as 1D path but without positional encoding)

    // Self-attention + FFN (same as 1D path)

  latents_2d = view(latents_2d, B, 16, 16, 64)  → (B, 16, 16, 64)
  latents_2d = permute(latents_2d)                → (B, 64, 16, 16)
  pos_2d = PositionEmbeddingSine(latents_2d)      → (B, 64, 16, 16)

  latents_2d = flatten(permute(latents_2d))       → (B, 256, 64)
  pos_2d = flatten(permute(pos_2d))               → (B, 256, 64)

  latents_2d = LayerNorm(latents_2d)
```

### 8.4 Combined Output

```
latents = concat(latents_1d, latents_2d, dim=1)  → (B, 512, 64)
pos = concat(pos_1d, pos_2d, dim=1)              → (B, 512, 64)
```

**Asymmetric position encoding structure (important for downstream RoPE):**
When input pos is not None (normal case during memory encoding):
- `pos_1d` = **zeros** (B, 256, 64) — 1D latents get no spatial position
- `pos_2d` = **sinusoidal PE** (B, 256, 64) — from PositionEmbeddingSine applied to the 16×16 grid
- Combined: `pos = [zeros | sinusoidal]` (B, 512, 64)

This asymmetric pos feeds into the memory bank. In memory attention cross-attention, `memory + pos` adds sinusoidal PE to only the 2D latent tokens, while 1D latents get no additive PE. Combined with the RoPE token splitting (Section 9.3 — 1D tokens excluded from RoPE), the 1D latents are effectively position-agnostic.

**Token order `[1D, 2D]` must be preserved.** Both the additive pos and the RoPE token splitting in memory attention rely on 1D latents being first and 2D latents being second within each memory frame.

### 8.5 Shared Layer Weights

**Important:** The 1D and 2D paths share the **same** `self.layers` weights. From the Python code:
```python
for layer in self.layers:
    latents = layer(latents, x, _pos)    # 1D path uses these layers
for layer in self.layers:
    latents_2d = layer(latents_2d, x)    # 2D path reuses SAME layers
```

This means both paths must be built into the same sub-graph (model weight tensors are safe to reference from any graph per CLAUDE.md rules). The 1D and 2D paths run sequentially within one sub-graph, sharing the perceiver layer weights.

Also note: `pos_1d` is set to zeros only when the input `pos` is not None (which is the normal case during memory encoding, since pos comes from the memory encoder's sinusoidal PE). If pos were None, pos_1d would also be None.

### 8.6 Graph Isolation

The perceiver runs as its own sub-graph:
```cpp
static void edgetam_perceiver_forward(
    sam3_state & state, const sam3_model & model,
    const std::vector<float> & mem_features,  // (64 * H * W) from memory encoder
    const std::vector<float> & mem_pos,       // (64 * H * W) position encoding
    int H, int W,
    std::vector<float> & out_latents,         // (512 * 64) output
    std::vector<float> & out_pos)             // (512 * 64) output
{
    // Own ggml_context + gallocr
    // Build graph: 1D path then 2D path (sequential, shared layer weights)
    // Compute, extract results, free
}
```

---

## 9. Memory Attention with RoPEv2

### 9.1 Key Difference

SAM2 uses `RoPEAttention` for cross-attention (same spatial resolution for q and k). EdgeTAM uses `RoPEAttentionv2` with **asymmetric** q/k resolutions:
- Query spatial size: 64×64 (from current frame features at 64×64)
- Key spatial size: 16×16 (from perceiver's 2D latents at 16×16)

### 9.2 RoPEv2 Forward

```
RoPEAttentionv2.forward(q, k, v, num_k_exclude_rope, rope_k_repeat):
  q = q_proj(q)                     → (B, N_q, 256)
  k = k_proj(k)                     → (B, N_k, 256)  (kv_in_dim=64 → 256)
  v = v_proj(v)                     → (B, N_k, 256)

  q = separate_heads(q, 1 head)     → (B, 1, N_q, 256)
  k = separate_heads(k, 1 head)     → (B, 1, N_k, 256)
  v = separate_heads(v, 1 head)     → (B, 1, N_k, 256)

  // Apply RoPE with separate frequencies for q and k
  freqs_cis_q = compute_axial_cis(dim=256, end_x=64, end_y=64)  → [4096, 128] complex
  freqs_cis_k = compute_axial_cis(dim=256, end_x=16, end_y=16)  → [256, 128] complex

  q = apply_rotary_enc_v2(q, freqs_cis_q, repeat_freqs=1)

  num_k_rope = N_k - num_k_exclude_rope   (exclude object pointers from RoPE)
  k[:, :, :num_k_rope] = apply_rotary_enc_v2(k[:, :, :num_k_rope], freqs_cis_k, repeat_freqs=rope_k_repeat)

  attn = scaled_dot_product_attention(q, k, v)  → (B, 1, N_q, 256)
  out = recombine_heads(attn)                     → (B, N_q, 256)
  out = out_proj(out)                              → (B, N_q, 256)
```

### 9.3 apply_rotary_enc_v2

The v2 variant handles the case where `repeat_freqs > 1` (multiple memory frames stacked):
1. If `N_tokens == freqs_cis.shape[0] * repeat_freqs`: apply RoPE to all tokens
2. Otherwise: split tokens into "rope" and "no-rope" portions per frame, apply RoPE only to "rope" tokens, re-concatenate

**Key distinction from v1:** v2 applies RoPE to q and k independently (v1 applies jointly). This allows different spatial resolutions.

**Critical: 1D/2D token splitting in key RoPE.** The perceiver outputs 512 tokens per frame = 256 1D latents + 256 2D latents (concatenated in that order). Only the 2D latents have spatial grid structure (16×16), so only they should receive RoPE. The splitting happens automatically in `apply_rotary_enc_v2`:

```
Per-frame key tokens:  [256 × 1D latents (no spatial grid) | 256 × 2D latents (16×16 grid)]
freqs_cis_k has 256 entries (16×16)
repeat_freqs = num_spatial_mem (number of memory frames)

N_tokens = num_spatial_mem * 512
freqs_cis.shape[0] * repeat_freqs = 256 * num_spatial_mem  ≠  N_tokens

Token split (per frame, repeated num_spatial_mem times):
  no_rope_tokens = N_tokens // repeat_freqs - freqs_cis.shape[0] = 512 - 256 = 256
  x_no_rope = first 256 tokens per frame  → 1D latents, kept as-is
  x_rope    = last  256 tokens per frame  → 2D latents, RoPE applied

Output: [1D_no_rope | 2D_with_rope] per frame  (order preserved)
```

**The C++ must preserve the `[1D, 2D]` token order per memory frame.** If 1D and 2D latents are swapped, the wrong tokens will receive RoPE and attention will produce silently wrong results.

### 9.4 Memory attention position encoding flags

From the EdgeTAM config, the memory attention layer uses these position encoding flags:

```yaml
pos_enc_at_input: true        # MemoryAttention: output += 0.1 * curr_pos at input
pos_enc_at_attn: false        # Self-attention: q = k = norm(tgt), NO pos added
pos_enc_at_cross_attn_keys: true   # Cross-attention: k = memory + pos (BEFORE k_proj)
pos_enc_at_cross_attn_queries: false  # Cross-attention: q = norm(tgt), NO pos added
```

**This means:**
- **Input:** `curr_features += 0.1 * curr_pos` (position encoding scaled by 0.1)
- **Self-attention:** Q and K are just `norm(tgt)` — no additive position encoding (RoPE handles position)
- **Cross-attention queries:** just `norm(tgt)` — no additive position (RoPE on q handles it)
- **Cross-attention keys:** `memory + pos` at **64-dim BEFORE k_proj** — the 64-dim position encoding from the perceiver is added to the 64-dim memory features, then projected to 256-dim by k_proj
- **Cross-attention values:** just `memory` (no position encoding added to v)

**Important for C++ implementation:** The pos addition to cross-attention keys happens BEFORE the linear projection, not after. The existing `_forward_ca` code:
```python
k = memory + pos if self.pos_enc_at_cross_attn_keys else memory  # 64-dim addition
# Then inside RoPEAttentionv2: k = self.k_proj(k)  # 64→256 projection
```

### 9.5 Existing memory attention code changes

The existing `sam3_build_mem_attn_graph()` currently handles 4 layers with RoPEv1. For EdgeTAM:
- Use 2 layers
- Self-attention: RoPEv1 with **dynamically computed feat_sizes=[64,64]** (not the config's [32,32] — the Python `RoPEAttention.forward()` recomputes `freqs_cis` when `q.shape[-2] != cached`, and with 64×64=4096 tokens, it becomes [64,64])
- Cross-attention: use precomputed freqs_cis with q_sizes=[64,64] and k_sizes=[16,16]
- Cross-attention k/v projection input dim is 64 (not 256)
- FFN hidden dimension: 2048 (same as SAM2, uses ReLU activation)

**Important:** The C++ code must precompute self-attention RoPE at the actual feature size (64×64 for EdgeTAM at 1024 resolution), not the config value of [32,32]. The existing `sam3_ensure_tracker_pe_caches` already uses `sam3_eff_feat_size()` which resolves to the correct size — but `hparams.feat_size()` must be extended for EdgeTAM (see Section 11.3).

The simplest approach: **add an `if (is_edgetam)` branch** in the memory attention graph builder that:
1. Uses the correct layer count from `hparams.mem_attn_layers` (already a param)
2. Uses separate q/k RoPE frequency tables
3. Projects k/v from 64-dim (memory) instead of 256-dim
4. Adds pos to memory features BEFORE k_proj (not after)

---

## 10. Reusable Components (No Changes Needed)

These components in sam3.cpp work as-is for EdgeTAM. The only change is the dispatch condition.

### 10.1 FPN Neck
Already implemented for SAM2 in `sam2_build_fpn_neck_graph()`. EdgeTAM uses the same architecture:
- 4 lateral 1×1 convs to 256-dim
- Top-down fusion at levels [2, 3]
- scalp=1

The input channel list differs ([384, 192, 96, 48] vs [1152, 576, 288, 144]) but the FPN code is channel-agnostic — it reads from `fpn.convs[i].weight`.

### 10.2 SAM Prompt Encoder
Identical. Same `sam_pe.*` tensors.

### 10.3 SAM Mask Decoder
Identical. Same `sam_dec.*` tensors. Uses `use_high_res_features=true`, `pred_obj_scores=true`.

### 10.4 Memory Encoder
Identical architecture. **One difference:** the mask downsampler uses kernel_size=3, stride=2, padding=1 (not k=4, s=4 like SAM2). This means **4 downsampling layers** (2^4 = 16x) instead of SAM2's 2 layers (4^2 = 16x), plus 1 final 1×1 conv = **5 conv layers inside MaskDownSampler** (vs 3 for SAM2). The existing `sam3_mem_enc` struct has `ds_conv_w[5]` (5 slots) which accommodates this, but the loading/registration code must handle the different layer count.

**Channel progression (two separate components):**
- **MaskDownSampler** (5 convs in `ds_conv_w[0..4]`): 1→4→16→64→256 (4 strided convs, each k=3 s=2 p=1, with LayerNorm2d + GELU) → 256→256 (final 1×1 conv, embed_dim=256 default)
- **MemoryEncoder.out_proj** (separate `out_proj_w`): 256→64 (1×1 conv, since config `out_dim=64 ≠ in_dim=256`)

Note: The MaskDownSampler's `embed_dim` defaults to 256 (not overridden in edgetam.yaml). The 256→64 reduction is NOT inside MaskDownSampler — it's `MemoryEncoder.out_proj`, which is already stored separately in `sam3_mem_enc.out_proj_w/b`.

### 10.5 Video Tracking Pipeline
The `sam3_propagate_single()` / `sam3_track_frame()` functions orchestrate:
1. Image encoding
2. Memory attention conditioning
3. SAM head forward
4. Memory encoding
5. Memory bank update

For EdgeTAM, only steps 1 (backbone) and 2 (memory attention) change. Steps 3-5 are identical. The perceiver runs as a sub-step of step 4 (after memory encoder, before storing in bank).

---

## 11. Model Loading & Dispatch

### 11.1 Detection Logic

In `sam3_load_model()`:

```cpp
if (magic == SAM2_MAGIC) {
    sam2_load_hparams(fin, hp);
    if (hp.backbone_type == 1) {
        model_type = SAM3_MODEL_SAM2;       // Hiera backbone
    } else if (hp.backbone_type == 2) {
        model_type = SAM3_MODEL_EDGETAM;    // RepViT + Perceiver
        edgetam_load_extra_hparams(fin, hp);  // read EdgeTAM-specific fields
    }
}
```

### 11.2 Tensor Registration

```cpp
if (hp.model_type == SAM3_MODEL_EDGETAM) {
    edgetam_register_repvit_tensors(model);   // backbone
    edgetam_register_perceiver_tensors(model); // perceiver
    sam2_register_fpn_tensors(model);          // reuse
    sam2_register_sam_pe_tensors(model);       // reuse
    sam2_register_sam_dec_tensors(model);      // reuse
    sam2_register_mem_enc_tensors(model);      // reuse
    edgetam_register_mem_attn_tensors(model);  // modified (2 layers, RoPEv2 dims)
    sam2_register_toplevel_tensors(model);     // reuse
}
```

### 11.3 hparams Extension

Add to `sam3_hparams`:

```cpp
// EdgeTAM backbone
int32_t backbone_type = 1;          // 1=hiera, 2=repvit
int32_t repvit_stages[4] = {};      // [2, 2, 14, 2]
int32_t repvit_channels[4] = {};    // [48, 96, 192, 384]

// Perceiver
int32_t has_perceiver = 0;
int32_t perceiver_depth = 0;
int32_t perceiver_dim = 0;
int32_t perceiver_n_latents_1d = 0;
int32_t perceiver_n_latents_2d = 0;
int32_t perceiver_ff_mult = 0;

// Memory attention cross-attn type
int32_t mem_attn_ca_type = 0;       // 0=RoPEv1, 1=RoPEv2
int32_t mem_attn_ca_q_size = 32;
int32_t mem_attn_ca_k_size = 32;

bool is_edgetam() const { return backbone_type == 2; }

// Feature size for the active backbone (extends existing feat_size())
int32_t feat_size() const {
    if (is_edgetam()) return img_size / 16;  // 1024/16 = 64
    return is_sam2() ? hiera_feat_size() : n_img_embd();
}
```

---

## 12. Pipeline Integration

### 12.1 sam3_encode_image Dispatch

```cpp
bool sam3_encode_image(sam3_state & state, const sam3_model & model, const sam3_image & img) {
    if (model.hparams.is_edgetam()) {
        return edgetam_encode_image(state, model, img);
    } else if (model.hparams.is_sam2()) {
        return sam2_encode_image_hiera(state, model, img);
    } else {
        // SAM3 path
        return sam3_encode_image_vit(state, model, img);
    }
}
```

### 12.2 Memory Encoding with Perceiver

In the memory encoding step (called from `sam3_propagate_single` or equivalent):

```cpp
// After memory encoder produces (B, 64, H, W) features:
if (model.hparams.has_perceiver) {
    // Run perceiver sub-graph
    edgetam_perceiver_forward(state, model, mem_features, mem_pos, H, W,
                              compressed_features, compressed_pos);
    // Store compressed (512, 64) latents in memory bank instead of (H*W, 64)
} else {
    // SAM2 path: store full (H*W, 64) features in memory bank
}
```

### 12.3 Memory Attention Input

When cross-attending to memories, EdgeTAM cross-attention receives:
- Query: current frame features (64*64 = 4096 tokens, 256-dim)
- Key/Value: perceiver latents from memory bank (512 tokens per frame, 64-dim)

The cross-attention key/value projection accepts 64-dim input (kv_in_dim=64), which is why the k_proj and v_proj weights have shape [256, 64] instead of [256, 256].

### 12.4 Memory Layout in Memory Bank

For EdgeTAM, each memory slot stores:
- `maskmem_features`: (512, 64) from perceiver [was (H*W, 64) for SAM2]
- `maskmem_pos_enc`: (512, 64) from perceiver [was (H*W, 64) for SAM2]
- Object pointer: (1, 256) [same as SAM2]

**Token order within each memory slot MUST be `[1D latents (256), 2D latents (256)]`.** This is critical for two reasons:
1. **Additive pos:** `maskmem_pos_enc` has zeros for the first 256 tokens (1D) and sinusoidal PE for the last 256 tokens (2D). In memory attention, `k = memory + pos` adds PE only to 2D latents.
2. **RoPE splitting:** `apply_rotary_enc_v2` with `freqs_cis_k` of shape [256, 128] (16×16 grid) splits per-frame tokens into `no_rope_tokens=256` (first, 1D) and `rope_tokens=256` (last, 2D). Only 2D latents receive rotary position encoding.

The RoPE encoding for memory keys uses k_sizes=[16,16] because the 2D latents form a 16×16 grid (256 = 16*16). The 1D latents are position-agnostic (no additive PE, no RoPE).

---

## 13. Tensor Name Mapping Reference

### Checkpoint → GGML name mapping (complete)

See Section 5.2 for the full mapping. Key prefixes:

| Checkpoint prefix | GGML prefix | Tensor count |
|-------------------|-------------|-------------|
| `image_encoder.trunk.body.*` | `repvit.*` | ~120 (after BN fusion + RepVGG reparam) |
| `image_encoder.neck.*` | `fpn.*` | 8 |
| `spatial_perceiver.*` | `perceiver.*` | 42 |
| `memory_attention.*` | `mem_attn.*` | 54 |
| `memory_encoder.*` | `mem_enc.*` | 20 |
| `sam_prompt_encoder.*` | `sam_pe.*` | 17 |
| `sam_mask_decoder.*` | `sam_dec.*` | ~130 |
| Top-level | Various | ~10 |

---

## 14. Implementation Order (Phased)

### Phase 1: Weight Conversion Script (~1 day)

**Goal:** Convert `edgetam.pt` → `edgetam_f16.ggml` with correct BN fusion and RepVGG reparameterization.

**Steps:**
1. Create `convert_edgetam_to_ggml.py` based on `convert_sam2_to_ggml.py`
2. Implement `fuse_bn_into_conv()` helper function
3. Implement `repvgg_reparameterize()` — fuse 3 token mixer branches + final BN into single DW 3×3
4. Map all RepViT backbone tensor names (with BN fusion + RepVGG reparam)
5. Map all Perceiver tensor names
6. Reuse existing SAM2 mappings for shared components
7. Write extended header with `backbone_type=2` + EdgeTAM-specific hparams
8. Write all tensors in ggml format

**Verify:**
- Compare fused/reparameterized conv weight shapes against expected
- Count output tensors (~400)
- File size should be ~20 MB for f16
- Verify RepVGG reparameterization correctness: run Python model with original weights and with reparameterized weights on same input, compare outputs (should be numerically identical up to FP precision)

### Phase 2: Model Loading + RepViT Backbone (~2 days)

**Goal:** Load EdgeTAM weights and run RepViT image encoding.

**Files to modify:** `sam3.cpp`, `sam3.h`

**Steps:**
1. Add `SAM3_MODEL_EDGETAM` to the model type enum
2. Extend `sam3_hparams` with EdgeTAM fields
3. Add `edgetam_repvit_block`, `edgetam_repvit_stage`, `edgetam_repvit`, `edgetam_perceiver` structs
4. Implement `edgetam_load_extra_hparams()`
5. Implement `edgetam_register_repvit_tensors()` — create all backbone tensor nodes
6. Implement `edgetam_register_perceiver_tensors()` — create all perceiver tensor nodes
7. Add EdgeTAM branch in `sam3_load_model()`
8. Implement `edgetam_repvit_block_forward()` — block: fused DW 3×3 → SE → channel mixer + skip (use `ggml_conv_2d_dw` for DW conv)
9. Implement `edgetam_repvit_downsample_forward()` — downsample graph builder (use `ggml_conv_2d_dw` with stride=2 for spatial downsample)
10. Implement `edgetam_build_repvit_graph()` — full backbone: stem (GELU after conv1 only, no GELU after conv2) → 4 stages → 4 feature maps
11. Implement `edgetam_encode_image()` — RepViT + FPN + copy to state

**Verify:**
- Dump RepViT output tensor (the 4 stage features) and compare against Python reference
- Use `tests/dump_tensors.py` pattern with EdgeTAM model
- Check shapes: stage outputs should be [48,256,256], [96,128,128], [192,64,64], [384,32,32]

### Phase 3: FPN + Image Segmentation (~0.5 days)

**Goal:** End-to-end image segmentation with EdgeTAM (PVS only).

**Steps:**
1. Verify FPN neck reuse — the existing `sam2_build_fpn_neck_graph()` should work with EdgeTAM's FPN weights (different input channels but same 1×1 conv structure)
2. If needed, minor adjustments to handle different backbone channel counts
3. Wire `edgetam_encode_image()` → FPN → state caching
4. Test PVS (point/box segmentation) on a single image — this uses the same prompt encoder + mask decoder

**Verify:**
- Load EdgeTAM model
- Encode test image
- Run PVS with a point prompt
- Compare mask output against Python reference

### Phase 4: Spatial Perceiver (~1 day)

**Goal:** Implement the perceiver resampler sub-graph.

**Steps:**
1. Implement `edgetam_perceiver_attention_forward()` — cross-attention between latents and features
2. Implement `edgetam_perceiver_self_attention_forward()` — self-attention on latents
3. Implement `edgetam_perceiver_forward_1d()` — flatten + cross-attend + FFN + self-attend + FFN
4. Implement `edgetam_perceiver_forward_2d()` — window partition + cross-attend + reshape + PE
5. Implement `edgetam_perceiver_forward()` — combine 1D + 2D paths, run as sub-graph

**Key ggml patterns:**
- Window partitioning: reshape (B, H, W, C) → (B, H/ws, ws, W/ws, ws, C) → permute → reshape
- Cross-attention: Q from latents, K/V from features (with optional pos enc addition)
- No bias in Linear layers — just `ggml_mul_mat`

**Verify:**
- Dump perceiver output (512, 64) latents and compare against Python reference
- Verify both 1D and 2D paths produce expected shapes

### Phase 5: Memory Attention with RoPEv2 (~1 day)

**Goal:** Modify memory attention for EdgeTAM's asymmetric cross-attention.

**Steps:**
1. Implement `compute_axial_cis()` helper for precomputing RoPE frequencies at arbitrary resolutions
2. Implement `apply_rotary_enc_v2()` in ggml — handles `repeat_freqs` parameter
3. Modify or fork `sam3_build_mem_attn_graph()` to support:
   - 2 layers (driven by `hparams.mem_attn_layers`)
   - Separate q/k RoPE frequency tables (q: 64×64, k: 16×16)
   - Cross-attention k/v input dim = 64
   - `rope_k_repeat` parameter for multi-frame memory stacking
4. Integrate with existing memory conditioning pipeline

**Verify:**
- Build a memory attention sub-graph with dummy inputs
- Compare output against Python reference with same inputs
- Test with multiple memory frames to verify rope_k_repeat

### Phase 6: Video Tracking Integration (~1 day)

**Goal:** Full video tracking with EdgeTAM.

**Steps:**
1. Wire perceiver into the memory encoding step of `sam3_propagate_single()` (or equivalent tracker function)
2. Update memory bank to store perceiver-compressed features (512 latents, 64-dim) instead of full spatial features
3. Update memory retrieval to pass perceiver features to cross-attention
4. Handle the `num_spatial_mem` parameter for RoPEv2 `rope_k_repeat`
5. Test multi-frame tracking

**Verify:**
- Track an object across 10 frames of test video
- Compare per-frame masks against Python reference
- Run `sam3_benchmark` with EdgeTAM model

### Phase 7: Quantization + Polish (~0.5 days)

**Goal:** Q4_0/Q8_0 quantization support, benchmarks, example updates.

**Steps:**
1. Verify `sam3_quantize` works on EdgeTAM ggml files (it should — quantization is architecture-agnostic)
2. Run benchmarks: EdgeTAM f16, q8_0, q4_0 on Metal + CPU
3. Update `main_image.cpp` and `main_video.cpp` if needed (should work automatically via dispatch)
4. Update README with EdgeTAM benchmarks and model zoo entry

**Verify:**
- `sam3_benchmark --filter edgetam` works
- Q4_0 model tracks correctly
- Interactive examples work with EdgeTAM model

---

## 15. Verification Strategy

### 15.1 Tensor Dump Comparison

For each phase, follow the existing `tests/dump_tensors.py` pattern:

1. **Python side:** Run EdgeTAM with hooks to dump intermediate tensors as raw binary files
2. **C++ side:** Run sam3.cpp EdgeTAM with dumps at the same points
3. **Compare:** `tests/compare_tensors.py` with tolerance 1e-4 (f32) or 1e-2 (f16)

### 15.2 Key Checkpoints

| Phase | Tensor to compare | Expected shape |
|-------|-------------------|---------------|
| P2 | RepViT stage 0 output | (1, 48, 256, 256) |
| P2 | RepViT stage 3 output | (1, 384, 32, 32) |
| P3 | FPN level 2 output | (1, 256, 64, 64) |
| P3 | PVS mask output | (1, 3, 256, 256) |
| P4 | Perceiver 1D latents | (1, 256, 64) |
| P4 | Perceiver 2D latents | (1, 256, 64) |
| P4 | Perceiver combined output | (1, 512, 64) |
| P5 | Memory attention output | (1, 256, 64, 64) |
| P6 | Per-frame tracking mask | (1, 1, 256, 256) |

### 15.3 End-to-End Test

```bash
# Image segmentation
./build/examples/sam3_image --model models/edgetam_f16.ggml --image data/test_image.jpg

# Video tracking
./build/examples/sam3_video --model models/edgetam_f16.ggml --video data/test_video.mp4

# Benchmark
./build/examples/sam3_benchmark --filter edgetam --n-frames 5
```

---

## 16. Appendix: RepViT Block Architecture

### RepViT-M1 Configuration

```
Stem:
  Conv(3→24, k=3, s=2, p=1) + BN + GELU  →  (B, 24, 512, 512)
  Conv(24→48, k=3, s=2, p=1) + BN         →  (B, 48, 256, 256)   [NO activation after conv2]

Stage 0: ch=48, 2 blocks
  Block 0: DW3x3_fused(48) → SE(48, rd=16) → CM(48→96→48) + skip
  Block 1: DW3x3_fused(48) → CM(48→96→48) + skip

Downsample 0→1:
  pre_block(48, no SE) → DW_s2(48) → expand(48→96) → FFN(96→192→96) + skip

Stage 1: ch=96, 2 blocks
  Block 0: DW3x3_fused(96) → SE(96, rd=24) → CM(96→192→96) + skip
  Block 1: DW3x3_fused(96) → CM(96→192→96) + skip

Downsample 1→2:
  pre_block(96, no SE) → DW_s2(96) → expand(96→192) → FFN(192→384→192) + skip

Stage 2: ch=192, 14 blocks
  Block 0:  DW3x3_fused(192) → SE(192, rd=48) → CM(192→384→192) + skip
  Block 1:  DW3x3_fused(192) → CM(192→384→192) + skip
  Block 2:  DW3x3_fused(192) → SE → CM(192→384→192) + skip
  ...pattern: SE on even blocks [0,2,4,6,8,10,12]
  Block 13: DW3x3_fused(192) → CM(192→384→192) + skip

Downsample 2→3:
  pre_block(192, no SE) → DW_s2(192) → expand(192→384) → FFN(384→768→384) + skip

Stage 3: ch=384, 2 blocks
  Block 0: DW3x3_fused(384) → SE(384, rd=96) → CM(384→768→384) + skip
  Block 1: DW3x3_fused(384) → CM(384→768→384) + skip

Notation:
  DW3x3_fused = RepVGG-reparameterized depthwise 3×3 conv (single op)
  CM = channel mixer (Conv1×1 expand → GELU → Conv1×1 project)
  SE = squeeze-excite (GAP → FC → ReLU → FC → Sigmoid → scale), rd = reduced channels
  + skip = residual from AFTER SE to after CM output (bypasses CM only, NOT SE)
```

### SE Ratio

The SE ratio is `rd_ratio=0.25`, but the actual reduced channels use `make_divisible(ch*0.25, 8, round_limit=0.0)` which rounds UP to the nearest multiple of 8:

| Stage | ch_in | ch*0.25 | make_divisible(_, 8) → ch_rd | Effective ratio |
|-------|-------|---------|------------------------------|-----------------|
| 0 | 48 | 12.0 | 16 | 16/48 ≈ 1/3 |
| 1 | 96 | 24.0 | 24 | 24/96 = 1/4 |
| 2 | 192 | 48.0 | 48 | 48/192 = 1/4 |
| 3 | 384 | 96.0 | 96 | 96/384 = 1/4 |

Reduced channels are determined by the weight tensor shapes — no need to encode in hparams. SE activation is **ReLU** (not GELU). SE has **no BatchNorm** (norm_layer=None in TIMM SqueezeExcite).

---

## 17. Appendix: Perceiver Weight Shapes

All perceiver weights (from checkpoint, 42 tensors for 2 layers + globals):

```
perceiver.latents:         (256, 64)      1D latent tokens
perceiver.latents_2d:      (256, 64)      2D latent tokens
perceiver.norm.weight:     (64,)          final LayerNorm
perceiver.norm.bias:       (64,)

Per-layer (×2):
  perceiver.layers.{i}.ca.norm_latents.weight:  (64,)
  perceiver.layers.{i}.ca.norm_latents.bias:    (64,)
  perceiver.layers.{i}.ca.norm_x.weight:        (64,)
  perceiver.layers.{i}.ca.norm_x.bias:          (64,)
  perceiver.layers.{i}.ca.q.weight:             (64, 64)     no bias
  perceiver.layers.{i}.ca.kv.weight:            (128, 64)    no bias, chunk into k+v
  perceiver.layers.{i}.ca.out.weight:           (64, 64)     no bias

  perceiver.layers.{i}.ff.norm.weight:          (64,)        LayerNorm
  perceiver.layers.{i}.ff.norm.bias:            (64,)
  perceiver.layers.{i}.ff.fc1.weight:           (256, 64)    no bias
  perceiver.layers.{i}.ff.fc2.weight:           (64, 256)    no bias

  perceiver.layers.{i}.sa.norm.weight:          (64,)
  perceiver.layers.{i}.sa.norm.bias:            (64,)
  perceiver.layers.{i}.sa.q.weight:             (64, 64)     no bias
  perceiver.layers.{i}.sa.kv.weight:            (128, 64)    no bias
  perceiver.layers.{i}.sa.out.weight:           (64, 64)     no bias

  perceiver.layers.{i}.sa_ff.norm.weight:       (64,)
  perceiver.layers.{i}.sa_ff.norm.bias:         (64,)
  perceiver.layers.{i}.sa_ff.fc1.weight:        (256, 64)    no bias
  perceiver.layers.{i}.sa_ff.fc2.weight:        (64, 256)    no bias
```

---

## 18. Appendix: Full Checkpoint Tensor Inventory

### Original checkpoint: 982 tensors

| Category | Count | Notes |
|----------|-------|-------|
| RepViT backbone | 676 | Includes BN running stats (removed during conversion) |
| RepViT backbone (inference only) | 358 | After removing running_mean/var/num_batches_tracked |
| FPN neck | 8 | 4 convs × (weight + bias) |
| Spatial perceiver | 42 | Latents + 2 layers × 20 weights |
| Memory attention | 54 | 2 layers × 26 weights + 2 (norm) |
| Memory encoder | 20 | Mask downsampler + fuser + projections |
| SAM prompt encoder | 17 | PE + embeddings + mask downscaling |
| SAM mask decoder | ~130 | Transformer + hyper MLPs + upscaling |
| Top-level | ~10 | Mem/obj embeddings + projections |

### After BN fusion + RepVGG reparameterization: ~400 tensors

Two fusion steps reduce the backbone tensor count:
1. **BN fusion** collapses (conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked) into (fused_conv.weight, fused_conv.bias) — a 3:1 reduction.
2. **RepVGG reparameterization** further collapses each token mixer's 3 branches + final BN (conv.c+bn, conv1.c+bn, bn) into a single (dw_weight, dw_bias) — eliminating the separate 1×1 conv tensors.

### Expected GGML file sizes

| Precision | Estimated Size |
|-----------|---------------|
| f32 | ~40 MB |
| f16 | ~20 MB |
| q8_0 | ~12 MB |
| q4_0 | ~7 MB |

EdgeTAM is dramatically smaller than SAM2 models (SAM2 Tiny f16 = 75 MB). This makes it ideal for on-device inference.
