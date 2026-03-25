# SAM3.cpp — Complete Implementation Plan

> Port of Meta's SAM 3 (Segment Anything Model 3) to C++ using ggml.
> Single source file (`sam3.cpp`) + single header (`sam3.h`), minimal dependencies.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Dependencies](#3-dependencies)
4. [Build System](#4-build-system)
5. [Binary Weight Format](#5-binary-weight-format)
6. [Python Weight Conversion Script](#6-python-weight-conversion-script)
7. [Header File API (`sam3.h`)](#7-header-file-api-sam3h)
8. [Internal Structs (`sam3.cpp`)](#8-internal-structs-sam3cpp)
9. [Model Architecture — Full Forward Pass](#9-model-architecture--full-forward-pass)
10. [Function Inventory](#10-function-inventory)
11. [Video Tracking & Memory Bank](#11-video-tracking--memory-bank)
12. [Debugging & Verification Strategy](#12-debugging--verification-strategy)
13. [Example Executables](#13-example-executables)
14. [Implementation Order (Step-by-Step)](#14-implementation-order-step-by-step)
15. [Appendix: Tensor Shape Reference](#15-appendix-tensor-shape-reference)
16. [Appendix: Activation Functions Reference](#16-appendix-activation-functions-reference)

---

## 1. Project Overview

### What is SAM 3?

SAM 3 is a ~850M parameter model that detects, segments, and tracks objects in images and videos using **concept prompts** (short noun phrases like "yellow school bus", image exemplars, or both). It extends SAM 2 with:

- **Promptable Concept Segmentation (PCS)**: text/exemplar → all matching instance masks
- **Promptable Visual Segmentation (PVS)**: points/boxes/masks → single instance mask (SAM 1/2 style)
- **Video tracking**: memory bank + spatio-temporal masklet propagation

### Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SAM 3 Architecture                            │
│                                                                      │
│  ┌─────────────┐   ┌──────────┐   ┌────────────┐   ┌────────────┐  │
│  │ ViT Backbone │──▶│  Neck    │──▶│  Fusion    │──▶│   DETR     │  │
│  │ (PE, 32 blk) │   │(SimpleFPN)│  │  Encoder   │   │  Decoder   │  │
│  │ 1024-dim     │   │ 256-dim  │   │ 6 layers   │   │ 6 layers   │  │
│  └──────┬───────┘   └────┬─────┘   └─────┬──────┘   └─────┬──────┘  │
│         │                │               │                 │         │
│         │           ┌────┴─────┐    ┌────┴──────┐    ┌────┴──────┐  │
│         │           │SAM2 Neck │    │Text Encoder│    │Seg Head   │  │
│         │           │(for trk) │    │(24 layers) │    │(MaskFrmr) │  │
│         │           └────┬─────┘    └───────────┘    └───────────┘  │
│         │                │                                           │
│  ┌──────┴───────────────┴──────────────────────────────────────┐    │
│  │                    TRACKER PATH (video)                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │    │
│  │  │ Memory   │  │ Memory   │  │  SAM     │  │  Memory    │  │    │
│  │  │ Attention│  │  Bank    │  │  Decoder │  │  Encoder   │  │    │
│  │  │ (4 lyr)  │  │ (7 slots)│  │(2WayTrfm)│  │ (fuser)   │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

The model has two main paths sharing a single backbone:
1. **Detector** (DETR-based): text prompt → fusion encoder → decoder → boxes + masks
2. **Tracker** (SAM 2-based): memory bank → memory attention → SAM mask decoder → propagated masks

### Design Principles for This Port

- **One source file** (`sam3.cpp`) + **one header** (`sam3.h`)
- **Structs only**, no classes
- **C++14**: `std::unique_ptr`, `std::shared_ptr`, `std::make_unique`, etc.
- **ggml** for all tensor operations and the entire forward pass
- **stb_image** / **stb_image_write** for image I/O
- **No other dependencies** beyond the C++ standard library
- **Forward pass only** (inference), no backward/training
- **Metal backend** for GPU acceleration on macOS

---

## 2. Directory Structure

```
sam3.cpp/
├── CMakeLists.txt                  # Root build: library + ggml submodule
├── PLAN.md                         # This file
├── sam3.pdf                        # Paper
├── sam3.h                          # Public API header
├── sam3.cpp                        # Full implementation (~8000-12000 lines)
├── convert_sam3_to_ggml.py         # Weight conversion script (PyTorch → ggml binary)
├── ggml/                           # ggml submodule (latest)
├── stb/                            # stb_image.h + stb_image_write.h
│   ├── stb_image.h
│   └── stb_image_write.h
├── examples/
│   ├── CMakeLists.txt              # Builds example executables
│   ├── main_image.cpp              # Image segmentation example (interactive window)
│   ├── main_video.cpp              # Video segmentation example (interactive window)
│   └── third-party/                # SDL2 + ImGui for GUI (optional)
│       ├── imgui/
│       └── ...
├── tests/
│   ├── CMakeLists.txt
│   ├── dump_tensors.py             # Python: run official SAM3, dump intermediate tensors
│   ├── dump_tensors.cpp            # C++: run sam3.cpp, dump intermediate tensors
│   ├── compare_tensors.py          # Compare binary tensor dumps (Python vs C++)
│   ├── test_image_encoder.py       # Per-block verification for image encoder
│   ├── test_prompt_encoder.py      # Prompt encoder verification
│   ├── test_mask_decoder.py        # Mask decoder verification
│   ├── test_text_encoder.py        # Text encoder verification
│   ├── test_fusion_encoder.py      # Fusion encoder verification
│   ├── test_detr_decoder.py        # DETR decoder verification
│   ├── test_memory_encoder.py      # Memory encoder verification
│   ├── test_memory_attention.py    # Memory attention verification
│   └── test_end_to_end.py          # Full pipeline verification
└── scripts/
    ├── download_model.sh           # Download SAM3 checkpoint from HuggingFace
    └── setup_test_env.sh           # Clone official repo + install deps for testing
```

---

## 3. Dependencies

| Dependency | Purpose | Source |
|---|---|---|
| **ggml** | Tensor operations, Metal backend, graph computation | Git submodule (latest `master`) |
| **stb_image.h** | Load PNG/JPEG images | Single header, vendored |
| **stb_image_write.h** | Write PNG/JPEG output masks | Single header, vendored |
| **C++14 std library** | Smart pointers, containers, algorithms | System |
| **SDL2** (examples only) | Window creation, input handling | System package |
| **Dear ImGui** (examples only) | GUI widgets | Vendored in `third-party/` |
| **ffmpeg CLI** (video example) | Decode video frames to images | System, called via `popen()` |

The **library itself** (`sam3.cpp` + `sam3.h`) depends only on ggml, stb, and std. SDL2/ImGui are only for the example executables.

---

## 4. Build System

### Root `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.14)
project(sam3.cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml options — enable Metal on Apple
option(SAM3_METAL "Enable Metal backend" ON)
if(APPLE AND SAM3_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

# Add ggml
add_subdirectory(ggml)

# sam3 static library
add_library(sam3 STATIC sam3.cpp sam3.h)
target_include_directories(sam3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/stb)
target_link_libraries(sam3 PUBLIC ggml)
target_compile_features(sam3 PUBLIC cxx_std_14)

# Examples (only when top-level)
option(SAM3_BUILD_EXAMPLES "Build example executables" ON)
if(SAM3_BUILD_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()

# Tests
option(SAM3_BUILD_TESTS "Build test executables" OFF)
if(SAM3_BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

### `examples/CMakeLists.txt`

```cmake
find_package(SDL2 REQUIRED)
add_subdirectory(third-party)

# Image example
add_executable(sam3_image main_image.cpp)
target_link_libraries(sam3_image PRIVATE sam3 imgui-sdl2 SDL2::SDL2)

# Video example
add_executable(sam3_video main_video.cpp)
target_link_libraries(sam3_video PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
```

---

## 5. Binary Weight Format

### File Layout

The `.ggml` file uses a custom binary format, similar to sam.cpp but extended for SAM 3's architecture.

```
┌─────────────────────────────────────────────┐
│                  FILE HEADER                 │
├─────────────────────────────────────────────┤
│ [4 bytes]  magic: 0x73616D33 ("sam3")       │
│ [4 bytes]  version: 1                        │
│ [4 bytes]  ftype: 0=f32, 1=f16              │
│ [4 bytes]  n_tensors: total tensor count     │
│                                              │
│ === Hyperparameters block ===                │
│ [4 bytes]  img_size: 1008                    │
│ [4 bytes]  patch_size: 14                    │
│ [4 bytes]  vit_embed_dim: 1024               │
│ [4 bytes]  vit_depth: 32                     │
│ [4 bytes]  vit_num_heads: 16                 │
│ [4 bytes]  vit_mlp_ratio_x1000: 4625         │
│ [4 bytes]  vit_window_size: 24               │
│ [4 bytes]  n_global_attn_blocks: 4           │
│ [4 bytes]  global_attn_indices[0]: 7         │
│ [4 bytes]  global_attn_indices[1]: 15        │
│ [4 bytes]  global_attn_indices[2]: 23        │
│ [4 bytes]  global_attn_indices[3]: 31        │
│                                              │
│ [4 bytes]  text_width: 1024                  │
│ [4 bytes]  text_heads: 16                    │
│ [4 bytes]  text_layers: 24                   │
│ [4 bytes]  text_context_length: 32           │
│ [4 bytes]  text_vocab_size: 49408            │
│ [4 bytes]  text_output_dim: 256              │
│                                              │
│ [4 bytes]  neck_out_dim: 256                 │
│ [4 bytes]  fusion_encoder_layers: 6          │
│ [4 bytes]  fusion_encoder_heads: 8           │
│ [4 bytes]  detr_decoder_layers: 6            │
│ [4 bytes]  detr_decoder_heads: 8             │
│ [4 bytes]  detr_num_queries: 200             │
│ [4 bytes]  detr_ffn_dim: 2048                │
│                                              │
│ [4 bytes]  geom_encoder_layers: 3            │
│ [4 bytes]  n_presence_tokens: 1              │
│ [4 bytes]  n_geom_queries: 4                 │
│                                              │
│ [4 bytes]  sam_embed_dim: 256                │
│ [4 bytes]  sam_mask_decoder_depth: 2         │
│ [4 bytes]  sam_num_multimask: 3              │
│ [4 bytes]  sam_iou_head_depth: 3             │
│                                              │
│ [4 bytes]  mem_encoder_out_dim: 64           │
│ [4 bytes]  mem_attn_layers: 4               │
│ [4 bytes]  num_maskmem: 7                    │
│ [4 bytes]  max_obj_ptrs: 16                  │
│                                              │
│ [4 bytes]  n_amb_experts: 2                  │
│                                              │
├─────────────────────────────────────────────┤
│              TENSOR RECORDS                  │
│  (repeated n_tensors times)                  │
├─────────────────────────────────────────────┤
│ [4 bytes]  n_dims                            │
│ [4 bytes]  name_length                       │
│ [4 bytes]  dtype: 0=f32, 1=f16               │
│ [n_dims × 4 bytes]  shape (reversed order)   │
│ [name_length bytes]  tensor name (UTF-8)     │
│ [PADDING to 32-byte alignment]               │
│ [data_size bytes]  raw tensor data           │
└─────────────────────────────────────────────┘
```

**Key design choices:**
- 32-byte alignment for tensor data (enables mmap + SIMD)
- Shape dimensions stored in reversed (column-major) order for ggml
- Version field for future format changes
- All hyperparameters in the header so the C++ code doesn't need a separate config file

---

## 6. Python Weight Conversion Script

### `convert_sam3_to_ggml.py`

This script loads the official SAM 3 PyTorch checkpoint and writes the ggml binary format.

```python
#!/usr/bin/env python3
"""Convert SAM 3 PyTorch checkpoint to ggml binary format."""

import argparse
import struct
import numpy as np
import torch
import json
import os
from pathlib import Path

# --- Constants ---
MAGIC = 0x73616D33  # "sam3"
VERSION = 1
FTYPE_F32 = 0
FTYPE_F16 = 1

# Hyperparameter defaults (from model_builder.py)
HPARAMS = {
    "img_size": 1008,
    "patch_size": 14,
    "vit_embed_dim": 1024,
    "vit_depth": 32,
    "vit_num_heads": 16,
    "vit_mlp_ratio_x1000": 4625,
    "vit_window_size": 24,
    "n_global_attn_blocks": 4,
    "global_attn_indices": [7, 15, 23, 31],
    "text_width": 1024,
    "text_heads": 16,
    "text_layers": 24,
    "text_context_length": 32,
    "text_vocab_size": 49408,
    "text_output_dim": 256,
    "neck_out_dim": 256,
    "fusion_encoder_layers": 6,
    "fusion_encoder_heads": 8,
    "detr_decoder_layers": 6,
    "detr_decoder_heads": 8,
    "detr_num_queries": 200,
    "detr_ffn_dim": 2048,
    "geom_encoder_layers": 3,
    "n_presence_tokens": 1,
    "n_geom_queries": 4,
    "sam_embed_dim": 256,
    "sam_mask_decoder_depth": 2,
    "sam_num_multimask": 3,
    "sam_iou_head_depth": 3,
    "mem_encoder_out_dim": 64,
    "mem_attn_layers": 4,
    "num_maskmem": 7,
    "max_obj_ptrs": 16,
    "n_amb_experts": 2,
}
```

**Conversion logic overview:**

1. Load `sam3.pt` from HuggingFace (or local path)
2. Split keys into `detector.*` and `tracker.*` prefixes
3. Rename keys to a flat namespace (e.g., `detector.backbone.visual.trunk.blocks.0.norm1.weight` → `vit.blocks.0.norm1.weight`)
4. Write file header with magic + version + ftype + hparams
5. For each tensor:
   - Skip training-only tensors (e.g., `mask_downscaler` unused intermediate, DAC-specific, loss heads)
   - Determine dtype: 1D tensors, embeddings, and positional encodings → always f32; everything else → specified ftype
   - Reshape bias tensors as needed (e.g., patch_embed bias from `[C]` → `[1, C, 1, 1]`)
   - Write tensor record: n_dims, name_length, dtype, shape (reversed), name, padding, data
6. Print summary: total tensors, file size

**Key tensor name mappings** (PyTorch → ggml):

```
# ViT backbone
detector.backbone.visual.trunk.patch_embed.proj.weight  → vit.patch_embed.weight
detector.backbone.visual.trunk.blocks.{i}.norm1.weight  → vit.blocks.{i}.norm1.weight
detector.backbone.visual.trunk.blocks.{i}.norm1.bias    → vit.blocks.{i}.norm1.bias
detector.backbone.visual.trunk.blocks.{i}.attn.qkv.weight → vit.blocks.{i}.attn.qkv.weight
detector.backbone.visual.trunk.blocks.{i}.attn.qkv.bias   → vit.blocks.{i}.attn.qkv.bias
detector.backbone.visual.trunk.blocks.{i}.attn.proj.weight → vit.blocks.{i}.attn.proj.weight
detector.backbone.visual.trunk.blocks.{i}.attn.proj.bias   → vit.blocks.{i}.attn.proj.bias
detector.backbone.visual.trunk.blocks.{i}.norm2.weight  → vit.blocks.{i}.norm2.weight
detector.backbone.visual.trunk.blocks.{i}.norm2.bias    → vit.blocks.{i}.norm2.bias
detector.backbone.visual.trunk.blocks.{i}.mlp.lin1.weight → vit.blocks.{i}.mlp.lin1.weight
detector.backbone.visual.trunk.blocks.{i}.mlp.lin1.bias   → vit.blocks.{i}.mlp.lin1.bias
detector.backbone.visual.trunk.blocks.{i}.mlp.lin2.weight → vit.blocks.{i}.mlp.lin2.weight
detector.backbone.visual.trunk.blocks.{i}.mlp.lin2.bias   → vit.blocks.{i}.mlp.lin2.bias
detector.backbone.visual.trunk.pos_embed               → vit.pos_embed

# ViT absolute positional embedding (tiled from 336 pretrain to 1008)
# Shape: [1, 72, 72, 1024] (after interpolation in the conversion script)

# Neck (detector path)
detector.backbone.visual.neck.convs.{i}.0.weight  → neck.det.convs.{i}.conv.weight
detector.backbone.visual.neck.convs.{i}.0.bias    → neck.det.convs.{i}.conv.bias
# Deconv layers for upsampling scales
detector.backbone.visual.neck.backbone_norms.{i}.weight → neck.det.norms.{i}.weight
...

# Neck (tracker/SAM2 path) — separate weights
# Same structure as detector neck but under tracker prefix

# Text encoder
detector.backbone.text.transformer.resblocks.{i}.attn.in_proj_weight → text.blocks.{i}.attn.in_proj.weight
detector.backbone.text.transformer.resblocks.{i}.attn.in_proj_bias   → text.blocks.{i}.attn.in_proj.bias
detector.backbone.text.transformer.resblocks.{i}.attn.out_proj.weight → text.blocks.{i}.attn.out_proj.weight
detector.backbone.text.transformer.resblocks.{i}.attn.out_proj.bias   → text.blocks.{i}.attn.out_proj.bias
detector.backbone.text.transformer.resblocks.{i}.mlp.c_fc.weight     → text.blocks.{i}.mlp.fc1.weight
detector.backbone.text.transformer.resblocks.{i}.mlp.c_fc.bias       → text.blocks.{i}.mlp.fc1.bias
detector.backbone.text.transformer.resblocks.{i}.mlp.c_proj.weight   → text.blocks.{i}.mlp.fc2.weight
detector.backbone.text.transformer.resblocks.{i}.mlp.c_proj.bias     → text.blocks.{i}.mlp.fc2.bias
detector.backbone.text.transformer.resblocks.{i}.ln_1.weight         → text.blocks.{i}.ln1.weight
detector.backbone.text.transformer.resblocks.{i}.ln_1.bias           → text.blocks.{i}.ln1.bias
detector.backbone.text.transformer.resblocks.{i}.ln_2.weight         → text.blocks.{i}.ln2.weight
detector.backbone.text.transformer.resblocks.{i}.ln_2.bias           → text.blocks.{i}.ln2.bias
detector.backbone.text.token_embedding.weight    → text.token_embed.weight
detector.backbone.text.positional_embedding      → text.pos_embed
detector.backbone.text.ln_final.weight           → text.ln_final.weight
detector.backbone.text.ln_final.bias             → text.ln_final.bias
detector.backbone.text.resizer.weight            → text.resizer.weight
detector.backbone.text.resizer.bias              → text.resizer.bias

# Fusion encoder (6 layers)
detector.encoder.layers.{i}.self_attn.in_proj_weight → fenc.layers.{i}.sa.in_proj.weight
detector.encoder.layers.{i}.self_attn.out_proj.weight → fenc.layers.{i}.sa.out_proj.weight
detector.encoder.layers.{i}.cross_attn.in_proj_weight → fenc.layers.{i}.ca.in_proj.weight
detector.encoder.layers.{i}.cross_attn.out_proj.weight → fenc.layers.{i}.ca.out_proj.weight
detector.encoder.layers.{i}.linear1.weight → fenc.layers.{i}.ffn.fc1.weight
detector.encoder.layers.{i}.linear2.weight → fenc.layers.{i}.ffn.fc2.weight
detector.encoder.layers.{i}.norm1.weight → fenc.layers.{i}.norm1.weight
detector.encoder.layers.{i}.norm2.weight → fenc.layers.{i}.norm2.weight
detector.encoder.layers.{i}.norm3.weight → fenc.layers.{i}.norm3.weight

# DETR decoder (6 layers)
detector.decoder.layers.{i}.self_attn.in_proj_weight → ddec.layers.{i}.sa.in_proj.weight
detector.decoder.layers.{i}.cross_attn.in_proj_weight → ddec.layers.{i}.ca.in_proj.weight
detector.decoder.layers.{i}.cross_attn_text.in_proj_weight → ddec.layers.{i}.ca_text.in_proj.weight
detector.decoder.layers.{i}.linear1.weight → ddec.layers.{i}.ffn.fc1.weight
detector.decoder.layers.{i}.linear2.weight → ddec.layers.{i}.ffn.fc2.weight
detector.decoder.layers.{i}.bbox_embed.layers.{j}.weight → ddec.layers.{i}.bbox_mlp.{j}.weight
detector.decoder.query_embed.weight → ddec.query_embed.weight
detector.decoder.presence_token → ddec.presence_token

# Geometry encoder (3 layers)
detector.geometry_encoder.layers.{i}.* → geom.layers.{i}.*
detector.geometry_encoder.point_proj.weight → geom.point_proj.weight
detector.geometry_encoder.box_proj.weight → geom.box_proj.weight
detector.geometry_encoder.type_embed.weight → geom.type_embed.weight

# Segmentation head (MaskFormer)
detector.segmentation_head.pixel_decoder.* → seg.pixel_dec.*
detector.segmentation_head.mask_embed.weight → seg.mask_embed.weight

# SAM prompt encoder (tracker)
tracker.sam_prompt_encoder.point_embeddings.{i}.weight → sam_pe.point_embed.{i}
tracker.sam_prompt_encoder.not_a_point_embed.weight → sam_pe.not_a_point
tracker.sam_prompt_encoder.no_mask_embed.weight → sam_pe.no_mask
tracker.sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix → sam_pe.pe_gaussian

# SAM mask decoder (tracker)
tracker.sam_mask_decoder.transformer.layers.{i}.* → sam_dec.transformer.{i}.*
tracker.sam_mask_decoder.iou_token.weight → sam_dec.iou_token
tracker.sam_mask_decoder.mask_tokens.weight → sam_dec.mask_tokens
tracker.sam_mask_decoder.output_upscaling.* → sam_dec.upscale.*
tracker.sam_mask_decoder.output_hypernetworks_mlps.{i}.* → sam_dec.hyper.{i}.*
tracker.sam_mask_decoder.iou_prediction_head.* → sam_dec.iou_head.*
tracker.sam_mask_decoder.obj_score_token.weight → sam_dec.obj_score_token
tracker.sam_mask_decoder.pred_obj_score_head.* → sam_dec.obj_score_head.*

# Memory encoder
tracker.maskmem_backbone.mask_downsampler.* → mem_enc.downsample.*
tracker.maskmem_backbone.fuser.* → mem_enc.fuser.*
tracker.maskmem_backbone.pix_feat_proj.weight → mem_enc.pix_proj.weight
tracker.maskmem_backbone.output_proj.weight → mem_enc.out_proj.weight
tracker.maskmem_tpos_enc.{i} → mem_enc.tpos.{i}

# Memory attention (tracker transformer)
tracker.memory_attention.layers.{i}.self_attn.* → mem_attn.layers.{i}.sa.*
tracker.memory_attention.layers.{i}.cross_attn.* → mem_attn.layers.{i}.ca.*
tracker.memory_attention.layers.{i}.linear1.* → mem_attn.layers.{i}.ffn.fc1.*
tracker.memory_attention.layers.{i}.linear2.* → mem_attn.layers.{i}.ffn.fc2.*

# Object pointer projection
tracker.obj_ptr_proj.* → obj_ptr_proj.*
tracker.no_obj_ptr → no_obj_ptr
tracker.obj_ptr_tpos_proj.* → obj_ptr_tpos_proj.*
```

**Tensors to SKIP** (training-only, not needed for inference):
- Any tensor containing `loss`, `criterion`, `_dn_`, `label_enc`
- `dac_` prefixed tensors (DAC dual supervision heads)
- `semantic_seg_head` (semantic segmentation, separate from instance)
- Layer scale parameters if value is 1.0 (identity)
- Anything under `data_preprocessor`

---

## 7. Header File API (`sam3.h`)

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ─── Forward declarations (opaque, defined in sam3.cpp) ───
struct sam3_model;
struct sam3_state;

// ─── Public data types ───

struct sam3_point {
    float x;
    float y;
};

struct sam3_box {
    float x0;  // top-left x
    float y0;  // top-left y
    float x1;  // bottom-right x
    float y1;  // bottom-right y
};

struct sam3_image {
    int     width;
    int     height;
    int     channels;  // 3 for RGB
    std::vector<uint8_t> data;
};

struct sam3_mask {
    int     width;
    int     height;
    float   iou_score;      // predicted IoU
    float   obj_score;      // object presence score
    int     instance_id;    // unique instance ID (for video tracking)
    std::vector<uint8_t> data;  // binary mask (0 or 255)
};

struct sam3_detection {
    sam3_box  box;
    float     score;         // confidence score (presence * per-query)
    float     iou_score;     // predicted IoU
    int       instance_id;
    sam3_mask  mask;
};

// Result of a single-frame segmentation
struct sam3_result {
    std::vector<sam3_detection> detections;
};

// Parameters for model loading
struct sam3_params {
    std::string model_path;    // path to .ggml weights file
    int         n_threads = 4; // CPU threads for ggml
    bool        use_gpu   = true;  // use Metal backend if available
    int         seed      = 42;
};

// Parameters for image PCS (text-prompted concept segmentation)
struct sam3_pcs_params {
    std::string text_prompt;                  // noun phrase (e.g., "red car")
    std::vector<sam3_box>  pos_exemplars;     // positive image exemplar boxes
    std::vector<sam3_box>  neg_exemplars;     // negative image exemplar boxes
    float       score_threshold = 0.5f;       // detection confidence threshold
    float       nms_threshold   = 0.1f;       // NMS IoU threshold
};

// Parameters for image PVS (visual-prompted single instance segmentation)
struct sam3_pvs_params {
    std::vector<sam3_point> pos_points;   // positive click points
    std::vector<sam3_point> neg_points;   // negative click points
    sam3_box                box;          // bounding box prompt (set x0=x1=0 if unused)
    bool                    use_box = false;
    bool                    multimask = false; // return multiple masks
};

// Video tracker state (opaque)
struct sam3_tracker;

// Parameters for video tracking
struct sam3_video_params {
    std::string text_prompt;
    float       score_threshold     = 0.5f;
    float       nms_threshold       = 0.1f;
    float       assoc_iou_threshold = 0.1f;
    int         hotstart_delay      = 15;
    int         max_keep_alive      = 30;
    int         recondition_every   = 16;
    int         fill_hole_area      = 16;
};

// ─── Public API ───

// Load model from ggml binary file. Returns nullptr on failure.
std::shared_ptr<sam3_model> sam3_load_model(const sam3_params & params);

// Allocate inference state (computation buffers). One state per concurrent inference.
std::unique_ptr<sam3_state> sam3_create_state(const sam3_model & model, const sam3_params & params);

// Precompute image backbone features (called once per image).
// All subsequent PCS/PVS calls reuse these features.
bool sam3_encode_image(
    sam3_state       & state,
    const sam3_model & model,
    const sam3_image & image
);

// Promptable Concept Segmentation on an already-encoded image.
// Returns all detected instances matching the text prompt.
sam3_result sam3_segment_pcs(
    sam3_state             & state,
    const sam3_model       & model,
    const sam3_pcs_params  & params
);

// Promptable Visual Segmentation on an already-encoded image.
// Returns mask(s) for a single instance indicated by points/box.
sam3_result sam3_segment_pvs(
    sam3_state             & state,
    const sam3_model       & model,
    const sam3_pvs_params  & params
);

// ─── Video API ───

// Create a video tracker. Call once per video.
std::unique_ptr<sam3_tracker> sam3_create_tracker(
    const sam3_model       & model,
    const sam3_video_params & params
);

// Process one video frame. Runs detection + tracking + memory update.
// Returns all tracked masklets for this frame.
sam3_result sam3_track_frame(
    sam3_tracker           & tracker,
    sam3_state             & state,
    const sam3_model       & model,
    const sam3_image       & frame
);

// Add interactive refinement to a specific tracked instance.
// Points are positive/negative clicks on the current frame.
bool sam3_refine_instance(
    sam3_tracker              & tracker,
    sam3_state                & state,
    const sam3_model          & model,
    int                         instance_id,
    const std::vector<sam3_point> & pos_points,
    const std::vector<sam3_point> & neg_points
);

// Get the current frame index of the tracker.
int sam3_tracker_frame_index(const sam3_tracker & tracker);

// Reset tracker state (start fresh on new video).
void sam3_tracker_reset(sam3_tracker & tracker);

// Free model resources (also handled by shared_ptr destructor).
void sam3_free_model(sam3_model & model);

// Free state resources.
void sam3_free_state(sam3_state & state);

// ─── Utility ───

// Load image from file using stb_image.
sam3_image sam3_load_image(const std::string & path);

// Save mask to file as PNG.
bool sam3_save_mask(const sam3_mask & mask, const std::string & path);

// Decode a single video frame at a given index using ffmpeg.
sam3_image sam3_decode_video_frame(const std::string & video_path, int frame_index);

// Get video metadata (frame count, fps, width, height).
struct sam3_video_info {
    int width;
    int height;
    int n_frames;
    float fps;
};
sam3_video_info sam3_get_video_info(const std::string & video_path);
```

---

## 8. Internal Structs (`sam3.cpp`)

All structs below are defined **only in `sam3.cpp`** (hidden from consumers).

### 8.1 Hyperparameters

```cpp
struct sam3_hparams {
    int32_t img_size            = 1008;
    int32_t patch_size          = 14;
    int32_t vit_embed_dim       = 1024;
    int32_t vit_depth           = 32;
    int32_t vit_num_heads       = 16;
    int32_t vit_mlp_dim         = 4736;  // 1024 * 4.625
    int32_t vit_window_size     = 24;
    int32_t global_attn_indices[4] = {7, 15, 23, 31};

    int32_t text_width          = 1024;
    int32_t text_heads          = 16;
    int32_t text_layers         = 24;
    int32_t text_context_length = 32;
    int32_t text_vocab_size     = 49408;
    int32_t text_output_dim     = 256;

    int32_t neck_dim            = 256;

    int32_t fenc_layers         = 6;
    int32_t fenc_heads          = 8;
    int32_t fenc_ffn_dim        = 2048;

    int32_t ddec_layers         = 6;
    int32_t ddec_heads          = 8;
    int32_t ddec_ffn_dim        = 2048;
    int32_t ddec_num_queries    = 200;

    int32_t geom_layers         = 3;
    int32_t n_presence_tokens   = 1;
    int32_t n_geom_queries      = 4;

    int32_t sam_embed_dim       = 256;
    int32_t sam_dec_depth       = 2;
    int32_t sam_n_multimask     = 3;
    int32_t sam_iou_head_depth  = 3;

    int32_t mem_out_dim         = 64;
    int32_t mem_attn_layers     = 4;
    int32_t num_maskmem         = 7;
    int32_t max_obj_ptrs        = 16;

    int32_t n_amb_experts       = 2;

    // Derived
    int32_t n_img_tokens() const { return (img_size / patch_size) * (img_size / patch_size); } // 72*72=5184
    int32_t n_img_embd()   const { return img_size / patch_size; }  // 72
    int32_t vit_head_dim() const { return vit_embed_dim / vit_num_heads; }  // 64
    bool    is_global_attn(int layer) const;
};
```

### 8.2 ViT Backbone Layers

```cpp
struct sam3_vit_block {
    // norm1
    struct ggml_tensor * norm1_w;  // [embed_dim]
    struct ggml_tensor * norm1_b;  // [embed_dim]

    // attention: fused QKV
    struct ggml_tensor * attn_qkv_w;  // [3*embed_dim, embed_dim]
    struct ggml_tensor * attn_qkv_b;  // [3*embed_dim]
    struct ggml_tensor * attn_proj_w; // [embed_dim, embed_dim]
    struct ggml_tensor * attn_proj_b; // [embed_dim]

    // norm2
    struct ggml_tensor * norm2_w;  // [embed_dim]
    struct ggml_tensor * norm2_b;  // [embed_dim]

    // MLP
    struct ggml_tensor * mlp_lin1_w; // [mlp_dim, embed_dim]
    struct ggml_tensor * mlp_lin1_b; // [mlp_dim]
    struct ggml_tensor * mlp_lin2_w; // [embed_dim, mlp_dim]
    struct ggml_tensor * mlp_lin2_b; // [embed_dim]
};

struct sam3_vit {
    struct ggml_tensor * patch_embed_w;  // [embed_dim, 3, patch_size, patch_size]
    struct ggml_tensor * pos_embed;      // [1, n_img_embd, n_img_embd, embed_dim]
    struct ggml_tensor * ln_pre_w;       // [embed_dim]  (ln_pre = True)
    struct ggml_tensor * ln_pre_b;       // [embed_dim]

    std::vector<sam3_vit_block> blocks;  // [vit_depth] = 32 blocks
};
```

### 8.3 Neck (SimpleFPN)

```cpp
struct sam3_neck_scale {
    // Each scale has deconv/conv + conv1x1 + conv3x3
    // The exact structure varies per scale (see Section 9.2)
    struct ggml_tensor * deconv_w;     // ConvTranspose2d weight (if upsampling)
    struct ggml_tensor * deconv_b;
    struct ggml_tensor * deconv2_w;    // Second deconv (only for 4x scale)
    struct ggml_tensor * deconv2_b;
    struct ggml_tensor * conv1x1_w;    // Conv1x1 projection
    struct ggml_tensor * conv1x1_b;
    struct ggml_tensor * conv3x3_w;    // Conv3x3 refinement
    struct ggml_tensor * conv3x3_b;
    struct ggml_tensor * pool_w;       // MaxPool (only for 0.5x scale, no weight — just op)
};

struct sam3_neck {
    sam3_neck_scale scales[4];         // 4x, 2x, 1x, 0.5x upsampling
    // Backbone layer norms for multi-level extraction (if used)
    struct ggml_tensor * norms_w[4];
    struct ggml_tensor * norms_b[4];
};
```

### 8.4 Text Encoder

```cpp
struct sam3_text_block {
    // Self-attention (causal)
    struct ggml_tensor * attn_in_proj_w;  // [3*text_width, text_width]
    struct ggml_tensor * attn_in_proj_b;  // [3*text_width]
    struct ggml_tensor * attn_out_proj_w; // [text_width, text_width]
    struct ggml_tensor * attn_out_proj_b; // [text_width]
    // LayerNorms
    struct ggml_tensor * ln1_w;  // [text_width]
    struct ggml_tensor * ln1_b;  // [text_width]
    struct ggml_tensor * ln2_w;  // [text_width]
    struct ggml_tensor * ln2_b;  // [text_width]
    // MLP
    struct ggml_tensor * mlp_fc1_w;  // [text_width*4, text_width]
    struct ggml_tensor * mlp_fc1_b;  // [text_width*4]
    struct ggml_tensor * mlp_fc2_w;  // [text_width, text_width*4]
    struct ggml_tensor * mlp_fc2_b;  // [text_width]
    // LayerScale (if present)
    struct ggml_tensor * ls1;  // [text_width]
    struct ggml_tensor * ls2;  // [text_width]
};

struct sam3_text_encoder {
    struct ggml_tensor * token_embed_w;   // [text_vocab_size, text_width]
    struct ggml_tensor * pos_embed;       // [text_context_length, text_width]
    struct ggml_tensor * ln_final_w;      // [text_width]
    struct ggml_tensor * ln_final_b;      // [text_width]
    struct ggml_tensor * resizer_w;       // [text_output_dim, text_width]
    struct ggml_tensor * resizer_b;       // [text_output_dim]

    std::vector<sam3_text_block> blocks;  // [text_layers] = 24 blocks
};
```

### 8.5 Fusion Encoder

```cpp
struct sam3_fenc_layer {
    // Self-attention (pre-norm)
    struct ggml_tensor * sa_in_proj_w;   // [3*256, 256]
    struct ggml_tensor * sa_in_proj_b;
    struct ggml_tensor * sa_out_proj_w;  // [256, 256]
    struct ggml_tensor * sa_out_proj_b;
    struct ggml_tensor * norm1_w;        // [256]
    struct ggml_tensor * norm1_b;

    // Cross-attention to text/exemplar tokens
    struct ggml_tensor * ca_q_w;   // [256, 256]
    struct ggml_tensor * ca_q_b;
    struct ggml_tensor * ca_kv_w;  // [2*256, 256]
    struct ggml_tensor * ca_kv_b;
    struct ggml_tensor * ca_out_w; // [256, 256]
    struct ggml_tensor * ca_out_b;
    struct ggml_tensor * norm2_w;  // [256]
    struct ggml_tensor * norm2_b;

    // FFN
    struct ggml_tensor * ffn_fc1_w; // [2048, 256]
    struct ggml_tensor * ffn_fc1_b; // [2048]
    struct ggml_tensor * ffn_fc2_w; // [256, 2048]
    struct ggml_tensor * ffn_fc2_b; // [256]
    struct ggml_tensor * norm3_w;   // [256]
    struct ggml_tensor * norm3_b;
};

struct sam3_fusion_encoder {
    std::vector<sam3_fenc_layer> layers; // [6]
};
```

### 8.6 DETR Decoder

```cpp
struct sam3_ddec_layer {
    // Self-attention among queries
    struct ggml_tensor * sa_in_proj_w;
    struct ggml_tensor * sa_in_proj_b;
    struct ggml_tensor * sa_out_proj_w;
    struct ggml_tensor * sa_out_proj_b;
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    // Cross-attention to conditioned image features
    struct ggml_tensor * ca_q_w;
    struct ggml_tensor * ca_q_b;
    struct ggml_tensor * ca_kv_w;
    struct ggml_tensor * ca_kv_b;
    struct ggml_tensor * ca_out_w;
    struct ggml_tensor * ca_out_b;
    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    // Cross-attention to text tokens
    struct ggml_tensor * ca_text_q_w;
    struct ggml_tensor * ca_text_q_b;
    struct ggml_tensor * ca_text_kv_w;
    struct ggml_tensor * ca_text_kv_b;
    struct ggml_tensor * ca_text_out_w;
    struct ggml_tensor * ca_text_out_b;
    struct ggml_tensor * norm3_w;
    struct ggml_tensor * norm3_b;

    // FFN
    struct ggml_tensor * ffn_fc1_w;
    struct ggml_tensor * ffn_fc1_b;
    struct ggml_tensor * ffn_fc2_w;
    struct ggml_tensor * ffn_fc2_b;
    struct ggml_tensor * norm4_w;
    struct ggml_tensor * norm4_b;

    // Box refinement MLP (3 layers: 256 → 256 → 256 → 4)
    struct ggml_tensor * bbox_mlp_w[3];
    struct ggml_tensor * bbox_mlp_b[3];
};

struct sam3_detr_decoder {
    struct ggml_tensor * query_embed;     // [num_queries, 256*2] (content + positional)
    struct ggml_tensor * presence_token;  // [1, 256]

    // DotProductScoring MLP for classification
    struct ggml_tensor * score_mlp_w[2];  // 256→2048→256
    struct ggml_tensor * score_mlp_b[2];
    struct ggml_tensor * score_ln_w;      // [256]
    struct ggml_tensor * score_ln_b;

    // Presence head MLP
    struct ggml_tensor * presence_head_w[2];
    struct ggml_tensor * presence_head_b[2];

    std::vector<sam3_ddec_layer> layers;  // [6]
};
```

### 8.7 Geometry / Exemplar Encoder

```cpp
struct sam3_geom_layer {
    // Self-attention
    struct ggml_tensor * sa_in_proj_w;
    struct ggml_tensor * sa_in_proj_b;
    struct ggml_tensor * sa_out_proj_w;
    struct ggml_tensor * sa_out_proj_b;
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    // Cross-attention to backbone features
    struct ggml_tensor * ca_q_w;
    struct ggml_tensor * ca_q_b;
    struct ggml_tensor * ca_kv_w;
    struct ggml_tensor * ca_kv_b;
    struct ggml_tensor * ca_out_w;
    struct ggml_tensor * ca_out_b;
    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    // FFN
    struct ggml_tensor * ffn_fc1_w;
    struct ggml_tensor * ffn_fc1_b;
    struct ggml_tensor * ffn_fc2_w;
    struct ggml_tensor * ffn_fc2_b;
    struct ggml_tensor * norm3_w;
    struct ggml_tensor * norm3_b;
};

struct sam3_geom_encoder {
    struct ggml_tensor * point_proj_w;   // [256, 2]
    struct ggml_tensor * point_proj_b;
    struct ggml_tensor * box_proj_w;     // [256, 4]
    struct ggml_tensor * box_proj_b;
    struct ggml_tensor * type_embed;     // [num_types, 256]
    struct ggml_tensor * cls_token;      // [1, 256]
    struct ggml_tensor * post_proj_w;    // [256, 256]
    struct ggml_tensor * post_proj_b;

    std::vector<sam3_geom_layer> layers; // [3]
};
```

### 8.8 Segmentation Head (MaskFormer)

```cpp
struct sam3_seg_head {
    // Pixel decoder (3 upsampling stages)
    struct ggml_tensor * up_conv_w[3];   // Conv2d for each upsample stage
    struct ggml_tensor * up_conv_b[3];
    struct ggml_tensor * up_norm_w[3];   // LayerNorm2d for each stage
    struct ggml_tensor * up_norm_b[3];

    // Cross-attention to prompt (for semantic seg)
    struct ggml_tensor * ca_prompt_q_w;
    struct ggml_tensor * ca_prompt_q_b;
    struct ggml_tensor * ca_prompt_kv_w;
    struct ggml_tensor * ca_prompt_kv_b;
    struct ggml_tensor * ca_prompt_out_w;
    struct ggml_tensor * ca_prompt_out_b;

    // Mask embedding projection
    struct ggml_tensor * mask_embed_w;   // [256, 256]
    struct ggml_tensor * mask_embed_b;
};
```

### 8.9 SAM Prompt Encoder (Tracker path)

```cpp
struct sam3_sam_prompt_encoder {
    struct ggml_tensor * pe_gaussian;       // [2, 128] random Gaussian matrix
    struct ggml_tensor * point_embed[4];    // [256] each: neg, pos, box_tl, box_br
    struct ggml_tensor * not_a_point_embed; // [256]
    struct ggml_tensor * no_mask_embed;     // [256]

    // Mask downscaling (Conv2d stack)
    struct ggml_tensor * mask_ds_conv_w[3]; // [4,1,2,2], [16,4,2,2], [256,16,1,1]
    struct ggml_tensor * mask_ds_conv_b[3];
    struct ggml_tensor * mask_ds_norm_w[2]; // LayerNorm2d for first two convs
    struct ggml_tensor * mask_ds_norm_b[2];
};
```

### 8.10 SAM Mask Decoder (Tracker path)

```cpp
struct sam3_sam_attn {
    struct ggml_tensor * q_w;    // [internal_dim, embedding_dim]
    struct ggml_tensor * q_b;
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;
    struct ggml_tensor * out_w;  // [embedding_dim, internal_dim]
    struct ggml_tensor * out_b;
};

struct sam3_twoway_block {
    sam3_sam_attn self_attn;          // 256-dim, 8 heads, downsample=1
    sam3_sam_attn cross_attn_tok2img; // 256-dim, 8 heads, downsample=2 → internal=128
    sam3_sam_attn cross_attn_img2tok; // 256-dim, 8 heads, downsample=2 → internal=128
    struct ggml_tensor * norm1_w;    // [256]
    struct ggml_tensor * norm1_b;
    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;
    struct ggml_tensor * norm3_w;
    struct ggml_tensor * norm3_b;
    struct ggml_tensor * norm4_w;
    struct ggml_tensor * norm4_b;
    struct ggml_tensor * mlp_fc1_w;  // [2048, 256]
    struct ggml_tensor * mlp_fc1_b;
    struct ggml_tensor * mlp_fc2_w;  // [256, 2048]
    struct ggml_tensor * mlp_fc2_b;
};

struct sam3_sam_mask_decoder {
    struct ggml_tensor * iou_token;        // [1, 256]
    struct ggml_tensor * mask_tokens;      // [4, 256]  (1 single + 3 multimask)
    struct ggml_tensor * obj_score_token;  // [1, 256]

    std::vector<sam3_twoway_block> transformer_layers;  // [2]

    // Final cross-attention (tokens → image)
    sam3_sam_attn final_attn_tok2img;
    struct ggml_tensor * final_norm_w;
    struct ggml_tensor * final_norm_b;

    // Upscaling (ConvTranspose2d)
    struct ggml_tensor * upscale1_w;  // ConvTranspose2d(256, 64, k=2, s=2)
    struct ggml_tensor * upscale1_b;
    struct ggml_tensor * upscale1_norm_w;  // LayerNorm2d(64)
    struct ggml_tensor * upscale1_norm_b;
    struct ggml_tensor * upscale2_w;  // ConvTranspose2d(64, 32, k=2, s=2)
    struct ggml_tensor * upscale2_b;

    // High-res feature convolutions (use_high_res_features=true)
    struct ggml_tensor * conv_s0_w;  // Conv2d for stage 0 high-res features
    struct ggml_tensor * conv_s0_b;
    struct ggml_tensor * conv_s1_w;  // Conv2d for stage 1 high-res features
    struct ggml_tensor * conv_s1_b;

    // Hypernetwork MLPs: 4 × MLP(256→256→256→32)
    struct ggml_tensor * hyper_w[4][3];  // [4 masks][3 layers]
    struct ggml_tensor * hyper_b[4][3];

    // IoU prediction: MLP(256→256→256→4)
    struct ggml_tensor * iou_head_w[3];
    struct ggml_tensor * iou_head_b[3];

    // Object score prediction: MLP(256→256→256→1)
    struct ggml_tensor * obj_score_head_w[3];
    struct ggml_tensor * obj_score_head_b[3];
};
```

### 8.11 Memory Encoder

```cpp
struct sam3_mem_encoder {
    // Mask downsampler (4 Conv2d stages + final 1x1)
    struct ggml_tensor * ds_conv_w[5];  // stages 0-3 (k=3,s=2) + final (k=1)
    struct ggml_tensor * ds_conv_b[5];
    struct ggml_tensor * ds_norm_w[4];  // LayerNorm2d for stages 0-3
    struct ggml_tensor * ds_norm_b[4];

    // Pixel feature projection
    struct ggml_tensor * pix_proj_w;    // Conv1x1(256, 256)
    struct ggml_tensor * pix_proj_b;

    // Fuser (2 CXBlock layers)
    struct ggml_tensor * fuser_dw_conv_w[2]; // Depthwise Conv(256, k=7, p=3)
    struct ggml_tensor * fuser_dw_conv_b[2];
    struct ggml_tensor * fuser_norm_w[2];    // LayerNorm2d(256)
    struct ggml_tensor * fuser_norm_b[2];
    struct ggml_tensor * fuser_fc1_w[2];     // Linear(256, 1024)
    struct ggml_tensor * fuser_fc1_b[2];
    struct ggml_tensor * fuser_fc2_w[2];     // Linear(1024, 256)
    struct ggml_tensor * fuser_fc2_b[2];
    struct ggml_tensor * fuser_gamma[2];     // LayerScale [256]

    // Output projection
    struct ggml_tensor * out_proj_w;    // Conv1x1(256, 64)
    struct ggml_tensor * out_proj_b;

    // Temporal positional encodings
    struct ggml_tensor * tpos[7];       // [1, 1, 1, 64] for each memory slot
};
```

### 8.12 Memory Attention (Tracker Transformer)

```cpp
struct sam3_mem_attn_layer {
    // Self-attention with RoPE (1 head, 256 dim)
    struct ggml_tensor * sa_q_w;
    struct ggml_tensor * sa_q_b;
    struct ggml_tensor * sa_k_w;
    struct ggml_tensor * sa_k_b;
    struct ggml_tensor * sa_v_w;
    struct ggml_tensor * sa_v_b;
    struct ggml_tensor * sa_out_w;
    struct ggml_tensor * sa_out_b;
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    // Cross-attention with RoPE (1 head, kv_dim=64)
    struct ggml_tensor * ca_q_w;     // [256, 256]
    struct ggml_tensor * ca_q_b;
    struct ggml_tensor * ca_k_w;     // [256, 64]  (kv_in_dim=64)
    struct ggml_tensor * ca_k_b;
    struct ggml_tensor * ca_v_w;     // [256, 64]
    struct ggml_tensor * ca_v_b;
    struct ggml_tensor * ca_out_w;   // [256, 256]
    struct ggml_tensor * ca_out_b;
    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    // FFN
    struct ggml_tensor * ffn_fc1_w;  // [2048, 256]
    struct ggml_tensor * ffn_fc1_b;
    struct ggml_tensor * ffn_fc2_w;  // [256, 2048]
    struct ggml_tensor * ffn_fc2_b;
    struct ggml_tensor * norm3_w;
    struct ggml_tensor * norm3_b;
};

struct sam3_memory_attention {
    std::vector<sam3_mem_attn_layer> layers;  // [4]
};
```

### 8.13 Top-Level Model

```cpp
struct sam3_model {
    sam3_hparams             hparams;

    sam3_vit                 vit;
    sam3_neck                neck_det;       // Detector path neck
    sam3_neck                neck_trk;       // Tracker path neck (separate weights)
    sam3_text_encoder        text_enc;
    sam3_fusion_encoder      fenc;
    sam3_detr_decoder        ddec;
    sam3_geom_encoder        geom_enc;
    sam3_seg_head            seg_head;

    sam3_sam_prompt_encoder   sam_pe;
    sam3_sam_mask_decoder     sam_dec;
    sam3_mem_encoder          mem_enc;
    sam3_memory_attention     mem_attn;

    // Object pointer projection
    struct ggml_tensor *     obj_ptr_proj_w;  // MLP(256→256→256)
    struct ggml_tensor *     obj_ptr_proj_b[3];
    struct ggml_tensor *     no_obj_ptr;      // [1, 256] learned
    struct ggml_tensor *     obj_ptr_tpos_proj_w; // Linear for temporal pos enc
    struct ggml_tensor *     obj_ptr_tpos_proj_b;

    // RoPE precomputed frequencies (shared)
    struct ggml_tensor *     rope_freqs_cis;  // [5184, 32] complex (stored as [5184, 64] float pairs)

    // ggml context + backend
    struct ggml_context *    ctx;
    ggml_backend_t           backend;
    ggml_backend_buffer_t    buffer;

    // Tensor lookup map
    std::map<std::string, struct ggml_tensor *> tensors;
};
```

### 8.14 Inference State

```cpp
struct sam3_state {
    // Cached backbone outputs (computed once per image/frame)
    struct ggml_tensor * vit_output;        // [B, 1024, 72, 72]
    struct ggml_tensor * neck_det_feats[3]; // FPN levels for detector
    struct ggml_tensor * neck_trk_feats[3]; // FPN levels for tracker
    struct ggml_tensor * neck_det_pe[3];    // Positional encodings for detector
    struct ggml_tensor * neck_trk_pe[3];    // Positional encodings for tracker

    // Input image dimensions (for postprocessing)
    int orig_width;
    int orig_height;

    // ggml computation resources
    struct ggml_context * ctx_compute;
    ggml_backend_t        backend;
    ggml_backend_buffer_t buf_compute;

    // Reusable graph allocator
    struct ggml_gallocr * galloc;
};
```

### 8.15 Video Tracker State

```cpp
struct sam3_masklet {
    int   instance_id;
    int   first_frame;           // Frame where this masklet first appeared
    int   last_seen_frame;       // Last frame where it was confidently seen
    float last_score;            // Last detection/tracking score
    bool  confirmed;             // Past hotstart delay

    // Per-frame mask logits (for memory encoding)
    struct ggml_tensor * last_mask_logits;  // [1, 1, 288, 288]

    // Object pointer (for memory bank)
    struct ggml_tensor * obj_ptr;           // [1, 256]

    // Masklet detection score (MDS)
    int   mds_sum;               // Running sum of ∆(τ)
    int   mds_match_count;       // Frames matched to detection
    int   mds_unmatch_count;     // Frames not matched

    // For duplicate detection
    std::vector<int> dup_overlap_frames;   // Frames overlapping with another masklet
};

struct sam3_memory_slot {
    struct ggml_tensor * spatial_feats;   // [64, 72, 72]
    struct ggml_tensor * spatial_pe;      // [64, 72, 72]
    int   frame_index;
    bool  is_conditioning_frame;
};

struct sam3_tracker {
    sam3_video_params params;
    int frame_index;

    // Active masklets
    std::vector<sam3_masklet> masklets;
    int next_instance_id;

    // Memory bank (per tracked object)
    // Maps instance_id → vector of memory slots
    std::map<int, std::vector<sam3_memory_slot>> memory_banks;

    // Object pointer bank (per tracked object)
    // Maps instance_id → vector of (frame_idx, obj_ptr)
    std::map<int, std::vector<std::pair<int, struct ggml_tensor *>>> obj_ptr_banks;

    // Hotstart queues (newly detected, pending confirmation)
    std::vector<sam3_masklet> pending_masklets;

    // ggml context for tracker-owned tensors
    struct ggml_context * ctx_tracker;
    ggml_backend_buffer_t buf_tracker;
};
```

### 8.16 BPE Tokenizer

```cpp
struct sam3_bpe_tokenizer {
    std::unordered_map<std::string, int>               encoder;   // token → id
    std::unordered_map<int, std::string>               decoder;   // id → token
    std::vector<std::pair<std::string, std::string>>   bpe_merges;
    std::unordered_map<std::string, std::string>       bpe_cache;

    int sot_token;  // 49406
    int eot_token;  // 49407
};
```

---

## 9. Model Architecture — Full Forward Pass

### 9.1 Image Preprocessing

```
Input:  sam3_image (H_orig × W_orig × 3, uint8)
Output: ggml tensor [1, 3, 1008, 1008], float32

Steps:
1. Resize to 1008×1008 (bilinear interpolation)
2. Convert to float32, scale to [0, 1]
3. Normalize: (pixel - 0.5) / 0.5  → range [-1, 1]
4. Store as CHW layout in ggml tensor
```

### 9.2 ViT Backbone (32 blocks)

```
Input:  [1, 3, 1008, 1008]
Output: [1, 1024, 72, 72]

1. Patch Embedding:
   Conv2d(3 → 1024, kernel=14, stride=14, no bias)
   [1, 3, 1008, 1008] → [1, 1024, 72, 72] → permute → [1, 72, 72, 1024]

2. Add positional embedding: [1, 72, 72, 1024] (tiled from 24×24 pretrain)

3. LayerNorm (ln_pre)

4. For each block i in 0..31:
   a. x_norm = LayerNorm(x)                              // norm1

   b. If is_global_attn(i):  (i ∈ {7, 15, 23, 31})
        q, k, v = split(Linear(x_norm, qkv), 3)          // [1, 5184, 1024] each
        Apply 2D axial RoPE to q, k
        attn = softmax(q @ k^T / sqrt(64)) @ v            // scaled dot-product, 16 heads
      Else (window attention):
        x_win = window_partition(x_norm, 24)               // [9, 24, 24, 1024]
        q, k, v = split(Linear(x_win, qkv), 3)
        Apply 2D axial RoPE to q, k
        attn = softmax(q @ k^T / sqrt(64)) @ v
        x_attn = window_unpartition(attn, 24, (72, 72))

   c. x = x + proj(attn)                                  // residual

   d. x = x + MLP(LayerNorm(x))                           // norm2 + MLP
      MLP: Linear(1024, 4736) → GELU → Linear(4736, 1024)

5. Permute to [1, 1024, 72, 72]
```

**RoPE Implementation (2D Axial):**

```
head_dim = 64
half = 32
theta = 10000.0

For each spatial position (y, x) in 72×72 grid:
  freqs_x = x / (theta^(2k/64)) for k in 0..15   (16 frequencies)
  freqs_y = y / (theta^(2k/64)) for k in 0..15

  rope_x = [cos(freqs_x), -sin(freqs_x), sin(freqs_x), cos(freqs_x)]  // complex multiply
  rope_y = [cos(freqs_y), -sin(freqs_y), sin(freqs_y), cos(freqs_y)]

Apply to first half of head_dim with x-freqs, second half with y-freqs.
```

**In ggml:** Precompute `rope_freqs_cis` as a [5184, 64] tensor (32 complex pairs = 64 floats). Apply via element-wise multiply and add pattern.

### 9.3 Neck (SimpleFPN)

```
Input:  [1, 1024, 72, 72] from ViT
Output: 3 feature maps (after scalp=1 drops the lowest level):
  - [1, 256, 288, 288]  (4× upsampled)
  - [1, 256, 144, 144]  (2× upsampled)
  - [1, 256, 72, 72]    (same resolution)

Scale 4× (288×288):
  ConvTranspose2d(1024→512, k=2, s=2) → GELU      → [1, 512, 144, 144]
  ConvTranspose2d(512→256, k=2, s=2)  → GELU      → [1, 256, 288, 288]
  Conv1x1(256→256) → Conv3x3(256→256, pad=1)       → [1, 256, 288, 288]

Scale 2× (144×144):
  ConvTranspose2d(1024→512, k=2, s=2) → GELU      → [1, 512, 144, 144]
  Conv1x1(512→256) → Conv3x3(256→256, pad=1)       → [1, 256, 144, 144]

Scale 1× (72×72):
  Conv1x1(1024→256) → Conv3x3(256→256, pad=1)      → [1, 256, 72, 72]

Scale 0.5× (36×36):  [DISCARDED due to scalp=1]
  MaxPool2d(k=2, s=2) → Conv1x1(1024→256) → Conv3x3 → [1, 256, 36, 36]

Duplicate neck for tracker (sam2_neck) with separate weights.
Also compute sinusoidal positional encodings for each scale.
```

### 9.4 Text Encoder (24 blocks)

```
Input:  text string
Output: text_features [seq_len, 1, 256], text_mask [1, seq_len]

1. BPE Tokenize: text → token_ids [1, L]  (L ≤ 32, pad with 0s)
2. Token embed: [1, L] → [L, 1, 1024]
3. Add positional embedding: pos_embed[:L]
4. For each block i in 0..23:
   a. x = x + LayerScale(MHA(LayerNorm(x), causal_mask=True))
      MHA: 16 heads, head_dim=64
   b. x = x + LayerScale(MLP(LayerNorm(x)))
      MLP: Linear(1024, 4096) → GELU → Linear(4096, 1024)
5. Final LayerNorm
6. Project: Linear(1024, 256)  → text_features [L, 1, 256]
7. Build attention mask from token padding
```

**Causal mask:** Lower-triangular mask (each position can only attend to previous positions).

### 9.5 Geometry / Exemplar Encoder

```
Input:  exemplar boxes [N, 4] + labels [N] (pos/neg) + backbone features
Output: exemplar_tokens [N+1, 1, 256]  (includes CLS token)

Per exemplar:
1. Coordinate embedding: Linear(4, 256)  → [1, 256]
2. Label embedding: embed_table[label]   → [1, 256]
3. ROI-pool visual features from backbone at box location → [1, 256]
4. Positional encoding: sinusoidal PE at box center → [1, 256]
5. Sum all: coord + label + roi + pos    → [1, 256]

Concatenate all exemplar tokens + CLS token → [N+1, 1, 256]

3 transformer layers:
  Self-attention among exemplar tokens
  Cross-attention to backbone features
  FFN
  All with LayerNorm + residual

Final projection: Linear(256, 256)
```

### 9.6 Fusion Encoder (6 layers)

```
Input:
  - image_features: flatten FPN features → [HW, 1, 256]
    (use highest-resolution level: 288×288 = 82944 tokens, or
     multi-scale flatten depending on implementation)
  - prompt_tokens: concat(text_features, exemplar_tokens) → [T, 1, 256]
  - positional_encodings for image features

Output: conditioned_features [HW, 1, 256]

For each layer i in 0..5:
  Pre-norm architecture:
  1. Self-attention on image features:
     q = k = LayerNorm(image_feats + pos_enc)
     v = LayerNorm(image_feats)
     image_feats = image_feats + dropout(MHA(q, k, v))  // 8 heads
     LayerNorm

  2. Cross-attention (image → text):
     q = LayerNorm(image_feats)
     k, v = prompt_tokens
     image_feats = image_feats + dropout(MHA(q, k, v))  // 8 heads
     LayerNorm

  3. FFN:
     image_feats = image_feats + dropout(
       Linear(2048→256, ReLU(Linear(256→2048, LayerNorm(image_feats))))
     )
     LayerNorm

Note: The actual implementation uses a single feature level (72×72=5184 tokens)
from the backbone, not the full multi-scale FPN. The FPN is used by the
segmentation head. The fusion encoder operates on the 72×72 features.
```

### 9.7 DETR Decoder (6 layers)

```
Input:
  - conditioned_features from fusion encoder: [5184, 1, 256]
  - text_features: [L, 1, 256]
  - query_embed: [200, 512] → split into content [200, 256] + pos [200, 256]
  - presence_token: [1, 256]

Output:
  - class_scores: [1, 200]
  - pred_boxes: [1, 200, 4]  (cx, cy, w, h in [0,1])
  - presence_score: [1, 1]
  - query_outputs: [1, 200, 256]

Queries = concat(presence_token, content_queries) → [201, 256]
Reference_boxes = sigmoid(Linear(pos_queries)) → [200, 4]  (initial anchor boxes)

For each layer i in 0..5:
  1. Self-attention among queries (with positional encoding):
     q = k = LayerNorm(queries + query_pos)
     v = LayerNorm(queries)
     queries = queries + MHA(q, k, v, 8 heads)

  2. Cross-attention to conditioned image features:
     q = LayerNorm(queries) + query_pos
     k = conditioned_features + pos_enc  [with box-relative position bias]
     v = conditioned_features
     queries = queries + MHA(q, k, v, 8 heads)

  3. Cross-attention to text tokens:
     q = LayerNorm(queries) + query_pos
     k, v = text_features
     queries = queries + MHA(q, k, v, 8 heads)

  4. FFN:
     queries = queries + Linear(2048→256, ReLU(Linear(256→2048, LayerNorm(queries))))

  5. Box refinement:
     box_delta = MLP(queries[:200])  // 3-layer: 256→256→256→4
     reference_boxes = sigmoid(inverse_sigmoid(reference_boxes) + box_delta)

Classification:
  text_proj = MLP(text_features)                    // 256→2048→256, ReLU, residual
  class_scores = queries[:200] @ text_proj.T         // dot product
  presence_score = MLP(queries[presence_idx])         // 256→256→1, sigmoid

Final: score_i = class_score_i × presence_score
```

### 9.8 Segmentation Head (MaskFormer)

```
Input:
  - FPN features: [1, 256, 288, 288], [1, 256, 144, 144], [1, 256, 72, 72]
  - Object queries from DETR decoder: [N_selected, 256]
  - Text features: [L, 256]

Output:
  - Instance masks: [N_selected, H_orig, W_orig]

1. Pixel Decoder (upsample FPN features):
   Start from lowest resolution, progressively upsample + merge:
   feat = FPN[2]                                    // [1, 256, 72, 72]
   feat = upsample(feat, 2×) + Conv(FPN[1])         // [1, 256, 144, 144]
   feat = upsample(feat, 2×) + Conv(FPN[0])         // [1, 256, 288, 288]
   Apply LayerNorm + conv at each stage

2. Cross-attention to prompt (optional, for semantic path):
   pixel_feats = MHA(pixel_feats, text_features)

3. Mask prediction:
   mask_embed = Linear(query_outputs)               // [N, 256]
   masks = einsum('nc,nchw->nhw', mask_embed, pixel_feats)  // [N, 288, 288]

4. Bilinear interpolate masks to original image resolution.

5. Post-process:
   - Apply sigmoid
   - Filter by confidence threshold
   - Non-maximum suppression (NMS) on boxes
   - Threshold masks at 0.0
```

### 9.9 SAM Prompt Encoder (Tracker Path — PVS)

```
Input:
  - Points: [N_pts, 2] coordinates + labels
  - Box: [4] coordinates (optional)
  - Mask: [1, 1, 288, 288] (optional)

Output:
  - sparse_embed: [1, N_tokens, 256]
  - dense_embed: [1, 256, 72, 72]

Per point:
  1. Normalize coords: coord / img_size  → [0, 1]
  2. Shift: coord + 0.5 (center of pixel)
  3. Map to [-1, 1]: coord * 2 - 1
  4. PE: matmul(coord, pe_gaussian) * 2π → sin/cos concat → [256]
  5. Add type embedding: point_embed[label]

Box handling:
  Encode as 2 points: top-left (label=2) + bottom-right (label=3)

Dense embedding:
  If mask provided: mask_downscale(mask) → [1, 256, 72, 72]
  Else: broadcast(no_mask_embed) → [1, 256, 72, 72]
```

### 9.10 SAM Mask Decoder (Tracker Path)

```
Input:
  - image_embed: [1, 256, 72, 72]  (memory-conditioned for video, or raw for image PVS)
  - sparse_embed: [1, N_prompt, 256]
  - dense_embed: [1, 256, 72, 72]
  - high_res_feats: from neck (288×288 and 144×144 levels)

Output:
  - masks: [1, 4, 288, 288]  (4 candidate masks)
  - iou_pred: [1, 4]
  - obj_score: [1, 1]

1. Build tokens:
   tokens = concat(obj_score_token[1,256], iou_token[1,256], mask_tokens[4,256], sparse_embed)
   → [1, 6+N_prompt, 256]

2. src = image_embed + dense_embed → flatten → [1, 5184, 256]
   pos_src = positional_encoding → [1, 5184, 256]

3. TwoWayTransformer (2 layers):
   For each layer:
     a. Self-attention on tokens (skip q+pe on first layer)
     b. Cross-attention: tokens → src (tokens query image)
        Q: tokens (256), K: src (256), V: src (256)
        downsample_rate=2 → internal_dim=128, head_dim=16
     c. MLP on tokens: Linear(256→2048) → ReLU → Linear(2048→256)
     d. Cross-attention: src → tokens (image queries tokens)
        downsample_rate=2 → internal_dim=128
   All with pre-LayerNorm + residual

   Final: one more cross-attention tokens→src, then LayerNorm on tokens

4. Extract special tokens:
   obj_token = tokens[:, 0, :]         // [1, 256]
   iou_token = tokens[:, 1, :]         // [1, 256]
   mask_tokens = tokens[:, 2:6, :]     // [1, 4, 256]

5. Upscale src → image space:
   src_2d = reshape(src) → [1, 256, 72, 72]

   // With high-res features:
   up1 = ConvTranspose2d(256→64, k=2, s=2)(src_2d) + conv_s1(high_res_144)
   up1 = GELU(LayerNorm2d(up1))                    → [1, 64, 144, 144]

   up2 = ConvTranspose2d(64→32, k=2, s=2)(up1) + conv_s0(high_res_288)
   up2 = GELU(up2)                                  → [1, 32, 288, 288]

6. Hypernetwork MLPs:
   For i in 0..3:
     hyper_out[i] = MLP(mask_tokens[:, i, :])       // 256→256→256→32
   hyper = stack(hyper_out)                          // [1, 4, 32]

7. Masks:
   masks = hyper @ up2.reshape(1, 32, 288*288)       // [1, 4, 288*288]
   masks = reshape(masks, [1, 4, 288, 288])

8. IoU prediction:
   iou_pred = MLP(iou_token)                         // 256→256→256→4, sigmoid

9. Object score:
   obj_score = MLP(obj_token)                         // 256→256→256→1, sigmoid

10. Dynamic multimask selection:
    If outputting single mask:
      stability = IoU(mask > +0.05, mask > -0.05)
      If stability < 0.98:
        Use mask with highest predicted IoU from masks[1:4]
      Else:
        Use masks[0]
```

### 9.11 Memory Encoder (Video)

```
Input:
  - backbone_pixel_features: [1, 256, 72, 72]  (from tracker neck)
  - mask_logits: [1, 1, H, W]                   (predicted mask, any resolution)

Output:
  - memory_features: [1, 64, 72, 72]
  - memory_pos_enc: [1, 64, 72, 72]

1. Project pixel features:
   pix = Conv1x1(256→256)(backbone_pixel_features)  → [1, 256, 72, 72]

2. Process mask:
   mask = sigmoid(mask_logits) × 20 - 10             // Scale to [-10, 10]
   mask = interpolate(mask, (1152, 1152))             // Upsample
   mask_feat = MaskDownSampler(mask):
     Conv2d(1→4, k=3, s=2, p=1) → LayerNorm2d → GELU   → [1, 4, 576, 576]
     Conv2d(4→16, k=3, s=2, p=1) → LayerNorm2d → GELU  → [1, 16, 288, 288]
     Conv2d(16→64, k=3, s=2, p=1) → LayerNorm2d → GELU → [1, 64, 144, 144]
     Conv2d(64→256, k=3, s=2, p=1) → LayerNorm2d → GELU → [1, 256, 72, 72]
     Conv1x1(256→256)                                     → [1, 256, 72, 72]

3. Fuse:
   fused = pix + mask_feat                            → [1, 256, 72, 72]
   fused = CXBlock(fused) × 2:                        // 2 CXBlock layers
     residual = fused
     fused = DepthwiseConv(256, k=7, p=3)(fused)
     fused = LayerNorm2d(fused)
     fused = Linear(256→1024)(fused) → GELU → Linear(1024→256)(fused)
     fused = residual + LayerScale(fused)

4. Project to memory dim:
   memory = Conv1x1(256→64)(fused)                    → [1, 64, 72, 72]

5. Positional encoding:
   memory_pe = PositionEmbeddingSine(64)(memory)       → [1, 64, 72, 72]
```

### 9.12 Memory Attention (Video)

```
Input:
  - current_features: [5184, 1, 256]            (from tracker neck, current frame)
  - current_pos_enc: [5184, 1, 256]
  - spatial_memories: list of [5184, 1, 64]      (from past frames)
  - spatial_memory_pe: list of [5184, 1, 64]     (positional encodings)
  - obj_pointers: list of [4, 1, 64]             (256-dim split into 4×64)
  - obj_pointer_pe: list of [4, 1, 64]           (temporal positional encodings)

Output:
  - conditioned_features: [1, 256, 72, 72]

1. Concatenate all memory tokens:
   For each spatial memory m:
     Add temporal positional encoding: m += tpos_enc[slot_index]
   spatial_prompt = concat(all spatial memories)     → [N_mem × 5184, 1, 64]
   obj_prompt = concat(all obj pointers)              → [N_ptr × 4, 1, 64]
   prompt = concat(spatial_prompt, obj_prompt)         → [total_mem_tokens, 1, 64]
   prompt_pe = same structure for positional encodings

2. 4-layer transformer with RoPE:
   For each layer:
     a. Self-attention (RoPE, 1 head, 256 dim):
        q = Linear(256, 256)(LayerNorm(current_features))
        k = Linear(256, 256)(LayerNorm(current_features))
        v = Linear(256, 256)(LayerNorm(current_features))
        Apply 2D axial RoPE to q, k (72×72 grid)
        sa_out = softmax(q @ k^T / sqrt(256)) @ v
        current_features += sa_out

     b. Cross-attention (RoPE, 1 head, kv_dim=64):
        q = Linear(256, 256)(LayerNorm(current_features))
        k = Linear(64, 256)(prompt)                 // project 64→256
        v = Linear(64, 256)(prompt)
        Apply RoPE to q; for k, repeat q's RoPE freqs (rope_k_repeat=True)
        ca_out = softmax(q @ k^T / sqrt(256)) @ v
        current_features += ca_out

     c. FFN:
        current_features += Linear(2048→256, ReLU(Linear(256→2048,
                            LayerNorm(current_features))))

3. Reshape: [5184, 1, 256] → [1, 256, 72, 72]
```

---

## 10. Function Inventory

### 10.1 Public API Functions (declared in `sam3.h`)

| Function | Description |
|---|---|
| `sam3_load_model()` | Load ggml binary, allocate backend buffer, populate model tensors |
| `sam3_create_state()` | Allocate computation state (graph allocator, intermediate buffers) |
| `sam3_encode_image()` | Run ViT backbone + neck on an image (caches features in state) |
| `sam3_segment_pcs()` | Run text encoder + fusion encoder + DETR decoder + seg head |
| `sam3_segment_pvs()` | Run SAM prompt encoder + mask decoder for single instance |
| `sam3_create_tracker()` | Initialize video tracker state |
| `sam3_track_frame()` | Process one video frame (detect + track + memory update) |
| `sam3_refine_instance()` | Add click refinement to a tracked instance |
| `sam3_tracker_frame_index()` | Get current frame index |
| `sam3_tracker_reset()` | Reset tracker for new video |
| `sam3_free_model()` | Free model resources |
| `sam3_free_state()` | Free state resources |
| `sam3_load_image()` | Load image from file (stb_image) |
| `sam3_save_mask()` | Save mask to PNG (stb_image_write) |
| `sam3_decode_video_frame()` | Decode single video frame via ffmpeg |
| `sam3_get_video_info()` | Get video metadata via ffmpeg |

### 10.2 Internal Functions (defined in `sam3.cpp`)

#### Model Loading

| Function | Description |
|---|---|
| `sam3_load_hparams(fin, hparams)` | Read hyperparameters from file header |
| `sam3_load_tensors(fin, model)` | Read all tensor records, allocate via ggml_allocr |
| `sam3_init_backend(model, params)` | Initialize ggml backend (Metal or CPU) |
| `sam3_precompute_rope(model)` | Precompute 2D axial RoPE frequency table |
| `sam3_load_bpe_vocab(tokenizer)` | Load BPE vocabulary (embedded or from file) |

#### Image Preprocessing

| Function | Description |
|---|---|
| `sam3_preprocess_image(img, ctx)` | Resize + normalize → ggml tensor [1,3,1008,1008] |
| `sam3_preprocess_to_f32(img)` | Convert sam3_image to internal float32 representation |

#### Backbone

| Function | Description |
|---|---|
| `sam3_build_vit_graph(ctx, model, state, input)` | Build ggml graph for full ViT forward |
| `sam3_vit_block_forward(ctx, model, block, x, layer_idx)` | Single ViT block forward |
| `sam3_vit_attention(ctx, model, block, x, is_global)` | Multi-head attention with RoPE |
| `sam3_window_partition(ctx, x, window_size)` | Partition tensor into windows |
| `sam3_window_unpartition(ctx, x, window_size, hw)` | Reverse window partition |
| `sam3_apply_rope_2d(ctx, q, k, rope_freqs, n_heads, hw)` | Apply 2D axial RoPE |
| `sam3_build_neck_graph(ctx, model, vit_out, det_or_trk)` | Build SimpleFPN graph |

#### Text Encoder

| Function | Description |
|---|---|
| `sam3_tokenize(tokenizer, text)` | BPE tokenize a string → vector<int32_t> |
| `sam3_bpe_encode(tokenizer, word)` | BPE merge loop for single word |
| `sam3_build_text_encoder_graph(ctx, model, token_ids)` | Build text encoder ggml graph |
| `sam3_build_causal_mask(ctx, seq_len)` | Build lower-triangular attention mask |

#### Geometry / Exemplar Encoder

| Function | Description |
|---|---|
| `sam3_build_geom_encoder_graph(ctx, model, boxes, labels, backbone_feats)` | Encode exemplars |
| `sam3_roi_pool(ctx, features, box, output_size)` | ROI-align from feature map |
| `sam3_sinusoidal_pe_2d(ctx, coords, num_feats)` | 2D sinusoidal positional encoding |

#### Fusion Encoder

| Function | Description |
|---|---|
| `sam3_build_fenc_graph(ctx, model, image_feats, prompt_tokens, pos_enc)` | Fusion encoder |
| `sam3_fenc_layer_forward(ctx, model, layer, x, prompt, pos)` | Single fusion layer |

#### DETR Decoder

| Function | Description |
|---|---|
| `sam3_build_ddec_graph(ctx, model, enc_feats, text_feats, pos_enc)` | Full DETR decoder |
| `sam3_ddec_layer_forward(ctx, model, layer, queries, enc_feats, text_feats, ref_boxes)` | Single decoder layer |
| `sam3_box_refine(ctx, model, layer, queries, ref_boxes)` | Iterative box refinement |
| `sam3_inverse_sigmoid(ctx, x)` | Inverse sigmoid: log(x / (1-x)) |
| `sam3_compute_box_rpb(ctx, ref_boxes, feat_hw)` | Box-relative positional bias |
| `sam3_dot_product_scoring(ctx, model, queries, text_feats)` | Classification via dot product |

#### Segmentation Head

| Function | Description |
|---|---|
| `sam3_build_seg_head_graph(ctx, model, fpn_feats, query_outputs)` | Mask prediction |
| `sam3_pixel_decoder(ctx, model, fpn_feats)` | Upsample FPN features |
| `sam3_mask_predict(ctx, query_embeds, pixel_feats)` | Einsum for per-instance masks |

#### SAM Prompt Encoder (Tracker)

| Function | Description |
|---|---|
| `sam3_build_sam_pe_graph(ctx, model, points, labels, box, mask)` | SAM prompt encoding |
| `sam3_pe_random_encode(ctx, model, coords)` | Random Fourier feature PE |

#### SAM Mask Decoder (Tracker)

| Function | Description |
|---|---|
| `sam3_build_sam_dec_graph(ctx, model, img_embed, sparse, dense, high_res)` | Full mask decoder |
| `sam3_twoway_block_forward(ctx, model, block, tokens, src, token_pe, src_pe, first_layer)` | Single 2-way block |
| `sam3_sam_attention(ctx, attn, q, k, v)` | Attention with optional downsample |
| `sam3_upscale_masks(ctx, model, src, high_res_feats)` | Transpose conv upscaling |
| `sam3_hypernetwork_mlp(ctx, model, mask_idx, token)` | Per-mask hypernetwork |
| `sam3_select_best_mask(masks, iou_pred, stability_delta, stability_thresh)` | Dynamic mask selection |

#### Memory Encoder

| Function | Description |
|---|---|
| `sam3_build_mem_enc_graph(ctx, model, pixel_feats, mask_logits)` | Memory encoding |
| `sam3_mask_downsample(ctx, model, mask)` | Conv stack to downsample mask |
| `sam3_cxblock(ctx, model, x, block_idx)` | CXBlock (depthwise conv + FFN + LayerScale) |

#### Memory Attention

| Function | Description |
|---|---|
| `sam3_build_mem_attn_graph(ctx, model, curr_feats, memories, obj_ptrs)` | Memory cross-attention |
| `sam3_mem_attn_layer_forward(ctx, model, layer, x, prompt, pos)` | Single memory attention layer |
| `sam3_rope_attention(ctx, q, k, v, rope_freqs, feat_hw, is_cross)` | RoPE attention (1 head) |

#### Video Tracking Logic

| Function | Description |
|---|---|
| `sam3_detect_frame(state, model, tracker)` | Run detector on current frame |
| `sam3_propagate_masklets(state, model, tracker)` | Propagate all tracked masklets |
| `sam3_propagate_single(state, model, tracker, masklet)` | Propagate one masklet |
| `sam3_match_detections(tracker, detections, propagated)` | IoU-based matching |
| `sam3_update_tracker(tracker, matched, unmatched_det, unmatched_trk)` | Update tracker state |
| `sam3_update_memory(state, model, tracker, masklet)` | Encode + store memory for one masklet |
| `sam3_extract_obj_ptr(model, sam_output_token, obj_score)` | Extract object pointer |
| `sam3_apply_hotstart(tracker)` | Apply hotstart delay logic |
| `sam3_check_confirmation(tracker)` | Check confirmation window criteria |
| `sam3_remove_duplicates(tracker)` | Remove duplicate masklets |
| `sam3_suppress_masklets(tracker)` | Suppress unconfirmed masklets (MDS < 0) |
| `sam3_recondition_masklets(state, model, tracker)` | Re-prompt from high-confidence detections |
| `sam3_detection_guided_reprompt(state, model, tracker)` | Re-prompt from drifted predictions |
| `sam3_apply_non_overlapping(masks)` | Enforce non-overlapping multi-object masks |
| `sam3_select_memory_frames(tracker, masklet)` | Select best memory frames for attention |

#### Post-Processing

| Function | Description |
|---|---|
| `sam3_nms(detections, iou_threshold)` | Non-maximum suppression on boxes |
| `sam3_fill_holes(mask, area_threshold)` | Fill small holes in binary mask |
| `sam3_remove_sprinkles(mask, area_threshold)` | Remove small disconnected regions |
| `sam3_connected_components(mask)` | Connected components labeling |
| `sam3_compute_iou(mask_a, mask_b)` | IoU between two masks |
| `sam3_compute_iom(mask_a, mask_b)` | Intersection-over-Minimum |
| `sam3_stability_score(mask_logits, delta)` | Mask stability score |
| `sam3_bilinear_interpolate(mask, target_h, target_w)` | Bilinear upsampling |

#### Utility Functions

| Function | Description |
|---|---|
| `sam3_layer_norm(ctx, x, w, b)` | LayerNorm operation |
| `sam3_layer_norm_2d(ctx, x, w, b)` | LayerNorm2d for [B,C,H,W] tensors |
| `sam3_mlp(ctx, x, w1, b1, w2, b2, act)` | 2-layer MLP with activation |
| `sam3_mlp_3layer(ctx, x, w, b, act)` | 3-layer MLP (arrays of 3 weights/biases) |
| `sam3_multihead_attention(ctx, q, k, v, n_heads, mask)` | Multi-head attention |
| `sam3_sigmoid(ctx, x)` | Element-wise sigmoid |
| `sam3_conv2d(ctx, x, w, b, stride, padding)` | 2D convolution wrapper |
| `sam3_conv_transpose2d(ctx, x, w, b, stride)` | Transposed 2D convolution |
| `sam3_maxpool2d(ctx, x, kernel, stride)` | Max pooling |

#### Graph Execution Helpers

| Function | Description |
|---|---|
| `sam3_graph_compute(backend, graph, n_threads)` | Execute ggml graph |
| `sam3_alloc_graph(galloc, graph)` | Allocate graph with gallocr |
| `sam3_measure_graph(ctx, build_fn)` | Measure graph memory requirements |

---

## 11. Video Tracking & Memory Bank

### 11.1 Per-Frame Pipeline

```
sam3_track_frame(tracker, state, model, frame):

1. ── BACKBONE ──
   sam3_encode_image(state, model, frame)
   // Caches vit_output, neck features in state

2. ── DETECTION ──
   detections = sam3_detect_frame(state, model, tracker)
   // Text encode → fusion encode → DETR decode → seg head
   // Filter by score_threshold, apply NMS

3. ── PROPAGATION ──
   For each active masklet m in tracker.masklets:
     propagated[m.id] = sam3_propagate_single(state, model, tracker, m)
     // Prepare memory-conditioned features:
     //   - Select closest memory frames (up to 4 spatial, 16 obj ptrs)
     //   - Run memory attention (4-layer transformer)
     //   - Run SAM mask decoder
     //   - Extract new obj_ptr
     //   - Return predicted mask + obj_score

4. ── MATCHING ──
   (matched, unmatched_det, unmatched_trk) =
     sam3_match_detections(tracker, detections, propagated)
   // IoU matching with threshold 0.1

5. ── UPDATE ──
   sam3_update_tracker(tracker, matched, unmatched_det, unmatched_trk)
   // Matched: update masklet with detection refinement
   // Unmatched detections: create new pending masklets
   // Unmatched tracks: maintain with propagation

6. ── HOTSTART / DISAMBIGUATION ──
   sam3_apply_hotstart(tracker)
   // Check confirmation window (15 frames)
   sam3_remove_duplicates(tracker)
   sam3_suppress_masklets(tracker)

7. ── PERIODIC RE-PROMPTING ──
   if frame_index % recondition_every == 0:
     sam3_recondition_masklets(state, model, tracker)
   sam3_detection_guided_reprompt(state, model, tracker)

8. ── MEMORY UPDATE ──
   For each confirmed masklet m:
     sam3_update_memory(state, model, tracker, m)
     // Encode mask + features via memory encoder
     // Store in memory bank (sliding window of 7)
     // Store obj_ptr in pointer bank (up to 16)

9. ── POST-PROCESS ──
   masks = collect all confirmed masklet masks
   sam3_apply_non_overlapping(masks)
   For each mask:
     sam3_fill_holes(mask, fill_hole_area)

10. ── OUTPUT ──
    tracker.frame_index++
    return sam3_result { detections with masks }
```

### 11.2 Memory Bank Management

```
Per-masklet memory bank (sliding window):
- Stores up to num_maskmem=7 spatial memory frames
- Slot 0: conditioning frame (first detection or user prompt)
- Slots 1-6: recent frames with confident predictions
- Each slot: { spatial_feats[64,72,72], spatial_pe[64,72,72], frame_idx }

Memory selection policy:
- Always include conditioning frame(s) (max 4)
- Select temporally closest frames for remaining slots
- When use_memory_selection=True:
  - Compute eff_iou_score = sigmoid(obj_score) * iou_score
  - Filter frames where eff_iou_score > 0.01
  - Always keep temporally closest frame as fallback

Object pointer bank:
- Stores up to max_obj_ptrs=16 pointers per masklet
- Each pointer: [256] vector extracted from SAM output token
- Projected through obj_ptr_proj MLP
- If object not visible (obj_score ≤ 0): use no_obj_ptr
- For cross-attention: split [256] into 4 tokens of [64] each
- Temporal PE: sine embedding of frame distance, projected

Memory attention input assembly:
  spatial_memories = []
  for each selected memory frame m:
    feat = m.spatial_feats + tpos_enc[slot_index]  // Add temporal PE
    spatial_memories.append(feat.flatten())          // [5184, 64]

  obj_ptrs = []
  for each selected pointer p:
    ptr = p.obj_ptr.reshape(4, 64)                  // Split into 4 tokens
    ptr += temporal_pe(frame_distance)
    obj_ptrs.append(ptr)                             // [4, 64]

  all_memory = concat(spatial_memories + obj_ptrs)   // [total, 64]
```

### 11.3 Disambiguation Strategies

#### Track Confirmation Delay (hotstart_delay = 15)

```
When a new masklet is created:
  masklet.confirmed = false
  masklet.first_frame = current_frame

After each frame (during delay window [t, t+T]):
  For each unconfirmed masklet:
    Compute MDS: S(first_frame, current_frame) = Σ Δ(τ)
    where Δ(τ) = +1 if matched to detection, -1 otherwise

    if S < threshold (V=0):
      Mark for removal  (unconfirmed — more misses than hits)

    Check for duplicates:
      If two masklets consistently overlap same detection for ≥ T/2 frames:
        Remove the later-started one
```

#### Masklet Suppression

```
For confirmed masklets:
  if MDS over entire lifetime < 0:
    Zero out mask output (but keep in tracker state)
    Object may recover if matched again later
```

#### Periodic Re-Prompting (every 16 frames)

```
For each detection d on current frame:
  Find best matching tracked masklet m
  if IoU(d, m.mask) ≥ 0.8 and d.score ≥ 0.8 and m.score ≥ 0.8:
    Re-initialize m's memory with detection mask
    This refreshes the memory bank with reliable detection output
```

#### Detection-Guided Re-Prompting

```
For each tracked masklet m:
  Find best matching detection d
  if IoU_bbox(d, m.box) < 0.85:
    Recondition m using detector's mask output
    This catches tracker drift
```

---

## 12. Debugging & Verification Strategy

### 12.1 Overview

To ensure numerical correctness, we implement a rigorous layer-by-layer verification pipeline that compares C++ outputs against the official PyTorch implementation.

### 12.2 Setup Scripts

**`scripts/setup_test_env.sh`:**
```bash
#!/bin/bash
# Clone official SAM3 repo
git clone https://github.com/facebookresearch/sam3.git /tmp/sam3_official
cd /tmp/sam3_official
pip install -e .

# Download checkpoint
python -c "from sam3.model_builder import download_ckpt_from_hf; download_ckpt_from_hf()"
```

**`scripts/download_model.sh`:**
```bash
#!/bin/bash
# Download SAM3 checkpoint from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/sam3', 'sam3.pt', local_dir='./models/')
"
# Convert to ggml format
python convert_sam3_to_ggml.py --model models/sam3.pt --output models/sam3.ggml --ftype 1
```

### 12.3 Tensor Dump Format

Both Python and C++ dump intermediate tensors to a simple binary format:

```
Per tensor file (.bin):
  [4 bytes] n_dims
  [4 bytes × n_dims] shape
  [4 bytes] dtype (0=f32)
  [n_elements × 4 bytes] data (always float32 for comparison)
```

File naming convention: `{module}_{layer}_{op}_{step}.bin`

Examples:
- `vit_block_00_norm1_output.bin`
- `vit_block_00_attn_qkv.bin`
- `vit_block_00_attn_rope_q.bin`
- `vit_block_00_attn_scores.bin`
- `vit_block_00_attn_output.bin`
- `vit_block_00_mlp_output.bin`
- `text_block_05_attn_output.bin`
- `fenc_layer_02_sa_output.bin`
- `ddec_layer_03_ca_output.bin`
- `sam_dec_twoway_0_self_attn.bin`
- `mem_attn_layer_1_ca_output.bin`

### 12.4 Python Dump Script (`tests/dump_tensors.py`)

```python
"""
Run official SAM3 model and dump intermediate tensors for verification.

Usage:
  python tests/dump_tensors.py \
    --model-path /path/to/sam3.pt \
    --image test_image.jpg \
    --output-dir /tmp/sam3_tensors_py/ \
    --text-prompt "red car" \
    --dump-level all  # or: backbone, text, fusion, decoder, sam
"""

# Hook into every module to capture inputs/outputs
# Use torch.register_forward_hook() on each layer
# Save tensors as .bin files in the output directory
# Fixed random seed for reproducibility

# For each block in ViT:
#   Hook norm1 output
#   Hook QKV output
#   Hook RoPE-applied Q, K
#   Hook attention scores (pre-softmax)
#   Hook attention output
#   Hook MLP output
#   Hook block output (after residual)

# Same granularity for:
#   Text encoder blocks
#   Fusion encoder layers
#   DETR decoder layers
#   SAM mask decoder two-way blocks
#   Memory attention layers
#   Memory encoder stages
```

### 12.5 C++ Dump Executable (`tests/dump_tensors.cpp`)

```cpp
// Mirrors the Python dump script but using sam3.cpp
// Uses the same image input and text prompt
// Hooks are implemented by adding dump points in the graph build functions
// Each dump point copies a tensor to CPU and writes it as .bin

// Key: The graph build functions accept an optional
// std::map<std::string, ggml_tensor*> * dump_map
// which, when non-null, captures intermediate tensors

// After graph execution, iterate dump_map and write each tensor to disk
```

### 12.6 Comparison Script (`tests/compare_tensors.py`)

```python
"""
Compare tensor dumps from Python and C++ implementations.

Usage:
  python tests/compare_tensors.py \
    --py-dir /tmp/sam3_tensors_py/ \
    --cpp-dir /tmp/sam3_tensors_cpp/ \
    --tolerance 1e-5 \
    --report report.txt

Output:
  For each tensor:
    - Name
    - Shape match (True/False)
    - Max absolute difference
    - Mean absolute difference
    - Max relative difference
    - PASS/FAIL based on tolerance
    - If FAIL: indices of worst mismatches

  Summary:
    - Total tensors compared
    - Passed / Failed
    - First failure point (helps identify where divergence starts)
    - Cumulative error growth (to detect if errors are compounding)
"""

import numpy as np
import os
import struct

def load_tensor(path):
    with open(path, 'rb') as f:
        n_dims = struct.unpack('i', f.read(4))[0]
        shape = [struct.unpack('i', f.read(4))[0] for _ in range(n_dims)]
        dtype = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)
    return data

def compare(py_dir, cpp_dir, tol=1e-5):
    py_files = sorted(os.listdir(py_dir))
    results = []
    for fname in py_files:
        if not fname.endswith('.bin'):
            continue
        cpp_path = os.path.join(cpp_dir, fname)
        if not os.path.exists(cpp_path):
            results.append((fname, 'MISSING', None, None))
            continue
        py_tensor = load_tensor(os.path.join(py_dir, fname))
        cpp_tensor = load_tensor(cpp_path)
        if py_tensor.shape != cpp_tensor.shape:
            results.append((fname, 'SHAPE_MISMATCH', py_tensor.shape, cpp_tensor.shape))
            continue
        max_abs = np.max(np.abs(py_tensor - cpp_tensor))
        mean_abs = np.mean(np.abs(py_tensor - cpp_tensor))
        status = 'PASS' if max_abs < tol else 'FAIL'
        results.append((fname, status, max_abs, mean_abs))
    return results
```

### 12.7 Per-Module Test Scripts

Each test script focuses on a single module, running it in isolation:

**`tests/test_image_encoder.py`:**
- Load SAM3 model
- Feed a fixed random input `[1, 3, 1008, 1008]`
- Run only the ViT backbone
- Dump output + every block's intermediate
- Verify against C++ output for the same input

**`tests/test_text_encoder.py`:**
- Tokenize a fixed string "a red car on the road"
- Run text encoder, dump all intermediate tensors
- Verify causal mask, attention patterns, final features

**`tests/test_prompt_encoder.py`:**
- Feed fixed points [(100, 200, label=1), (300, 400, label=0)]
- Run SAM prompt encoder
- Dump sparse + dense embeddings

**`tests/test_mask_decoder.py`:**
- Feed fixed image embedding (random or from backbone)
- Feed fixed sparse/dense prompt embeddings
- Run SAM mask decoder
- Dump: all two-way block intermediates, upscaled features, hypernetwork outputs, final masks

**`tests/test_fusion_encoder.py`:**
- Feed fixed backbone features + text features
- Run 6-layer fusion encoder
- Dump each layer's SA, CA, FFN outputs

**`tests/test_detr_decoder.py`:**
- Feed fixed encoded features + text features
- Run 6-layer DETR decoder with queries
- Dump each layer + box refinement + classification scores

**`tests/test_memory_encoder.py`:**
- Feed fixed pixel features + mask logits
- Run memory encoder
- Dump: mask downsample stages, fuser outputs, final 64-dim features

**`tests/test_memory_attention.py`:**
- Feed fixed current features + synthetic memory tensors
- Run 4-layer memory attention
- Dump: each layer's SA (with RoPE), CA, FFN outputs

**`tests/test_end_to_end.py`:**
- Full pipeline: image → text prompt → detected masks
- Compare final detection boxes, scores, masks
- For video: process 5 frames, compare all masklet outputs per frame

### 12.8 Tolerances

| Module | Expected max abs error | Notes |
|---|---|---|
| Patch embed | < 1e-6 | Simple conv, should be exact |
| ViT blocks 0-7 | < 1e-5 | Errors may accumulate slightly |
| ViT blocks 8-31 | < 1e-4 | Accumulated through 32 layers |
| Text encoder | < 1e-5 | Independent path |
| Fusion encoder | < 1e-4 | Cross-attention may introduce variance |
| DETR decoder | < 1e-4 | Box refinement is iterative sigmoid |
| SAM mask decoder | < 1e-4 | 2 layers, relatively shallow |
| Memory attention | < 1e-4 | RoPE + cross-attention |
| Final masks (logits) | < 1e-3 | Accumulated from all modules |
| Final masks (binary) | Exact match | After thresholding |

### 12.9 Numerical Debugging Checklist

- [ ] Patch embedding weights loaded correctly (verify first 10 values)
- [ ] Positional embedding shape and values match (check tiling from 24×24 to 72×72)
- [ ] RoPE frequencies match (check freqs_cis against PyTorch)
- [ ] Window partition/unpartition is identity when composed
- [ ] Causal mask in text encoder is correct (lower triangular)
- [ ] BPE tokenization matches official tokenizer output for 10 test strings
- [ ] Inverse sigmoid is numerically stable (clamp input to [1e-5, 1-1e-5])
- [ ] Box refinement: sigmoid(inv_sigmoid(prev) + delta) matches PyTorch
- [ ] Memory temporal PE added correctly (right slot index)
- [ ] Object pointer split from [256] to [4, 64] matches
- [ ] RoPE rope_k_repeat in cross-attention behaves correctly
- [ ] Mask downsampler interpolation to 1152×1152 before downsampling
- [ ] Non-overlapping mask constraint produces same result
- [ ] Fill holes / connected components matches (test with known patterns)

---

## 13. Example Executables

### 13.1 Image Example (`examples/main_image.cpp`)

An interactive SDL2 + ImGui window that lets the user:

1. **Open an image** (file dialog or drag-and-drop)
2. **Type a text prompt** (text input box)
3. **Click to add positive/negative exemplar boxes** (click + drag)
4. **See all detected instances** highlighted with colored masks
5. **Click on a specific instance** to refine with PVS points
6. **Export masks** as PNG files

```
Main loop:
  1. Load model (sam3_load_model)
  2. Create state (sam3_create_state)
  3. On image load:
     - sam3_encode_image(state, model, image)
  4. On text prompt change:
     - sam3_segment_pcs(state, model, pcs_params) → result
     - Render masks as colored overlays
  5. On click (PVS refinement):
     - sam3_segment_pvs(state, model, pvs_params) → refined mask
  6. Render: image + mask overlays + UI controls
```

**UI Layout:**
```
┌────────────────────────────────────────────────┐
│ [Text prompt: ___________]  [Segment]  [Clear] │
│                                                  │
│ ┌──────────────────────────────────────────────┐│
│ │                                              ││
│ │              Image Canvas                    ││
│ │         (with mask overlays)                 ││
│ │                                              ││
│ │     Click: add positive point (green)        ││
│ │     Right-click: add negative point (red)    ││
│ │     Drag: draw exemplar box                  ││
│ │                                              ││
│ └──────────────────────────────────────────────┘│
│                                                  │
│ Detections: 5 instances found                    │
│ Score threshold: [====|====] 0.50                │
│ [x] Show masks  [ ] Multi-mask  [Export masks]   │
└────────────────────────────────────────────────┘
```

### 13.2 Video Example (`examples/main_video.cpp`)

An interactive SDL2 + ImGui window for video segmentation:

1. **Open a video** (file dialog or command line)
2. **Type a text prompt**
3. **Play/pause** video with tracking
4. **Click to refine** specific instances on any frame
5. **See tracked masklets** with persistent instance colors
6. **Export** masklet video or individual frame masks

```
Main loop:
  1. Load model
  2. Create state + tracker
  3. Decode video frames (via ffmpeg)
  4. For each frame:
     a. sam3_track_frame(tracker, state, model, frame)
     b. Render frame + tracked masklet overlays
     c. Display instance IDs and scores
  5. On click: sam3_refine_instance(tracker, state, model, id, points)
  6. Playback controls: play/pause/step/speed
```

**Video decoding:** Use `popen("ffmpeg -i video.mp4 -f rawvideo -pix_fmt rgb24 -", "r")` to pipe decoded frames. This avoids adding ffmpeg as a library dependency.

**UI Layout:**
```
┌────────────────────────────────────────────────┐
│ [Text: ________]  [▶ Play] [⏸ Pause] [⏭ Step] │
│ Frame: 42/300   FPS: 24.0   Objects: 7         │
│                                                  │
│ ┌──────────────────────────────────────────────┐│
│ │                                              ││
│ │              Video Canvas                    ││
│ │      (with colored masklet overlays)         ││
│ │      Instance IDs shown as labels            ││
│ │                                              ││
│ │     Click: add positive refinement point     ││
│ │     Right-click: add negative point          ││
│ │                                              ││
│ └──────────────────────────────────────────────┘│
│                                                  │
│ Tracked instances:                               │
│   #1 (red)    score: 0.92  ■ confirmed           │
│   #2 (blue)   score: 0.88  ■ confirmed           │
│   #3 (green)  score: 0.45  ○ pending (hotstart)  │
│ [Export video]  [Export frames]                   │
└────────────────────────────────────────────────┘
```

---

## 14. Implementation Order (Step-by-Step)

This section defines the exact order in which to implement, from zero to a working demo. Each step is a self-contained milestone.

### Phase 0: Project Setup

**Step 0.1: Repository initialization**
- Create directory structure as defined in Section 2
- `git init`
- Add ggml as git submodule: `git submodule add https://github.com/ggerganov/ggml.git`
- Download `stb_image.h` and `stb_image_write.h` into `stb/`
- Write root `CMakeLists.txt`
- Verify `cmake .. && make` builds ggml successfully

**Step 0.2: Skeleton files**
- Create `sam3.h` with all public struct definitions and function declarations (Section 7)
- Create `sam3.cpp` with all internal struct definitions (Section 8) and empty function stubs
- Verify compilation

### Phase 1: Weight Conversion

**Step 1.1: Download checkpoint**
- Write `scripts/download_model.sh`
- Download `sam3.pt` from HuggingFace
- Inspect checkpoint keys: `python -c "import torch; ckpt = torch.load('sam3.pt'); [print(k, v.shape) for k, v in ckpt.items()]"`

**Step 1.2: Write conversion script**
- Implement `convert_sam3_to_ggml.py` (Section 6)
- Map all PyTorch tensor names to ggml names
- Write binary file in format defined in Section 5
- Verify: print summary of tensors written, total file size
- Expected output: `sam3.ggml` (~1.6 GB for f16, ~3.2 GB for f32)

**Step 1.3: Implement model loading in C++**
- Implement `sam3_load_hparams()`
- Implement `sam3_load_tensors()`
- Implement `sam3_init_backend()` (Metal on macOS, CPU fallback)
- Implement `sam3_load_model()` — tie it all together
- **Verify:** Load model, print all tensor names + shapes, compare against Python checkpoint

### Phase 2: BPE Tokenizer

**Step 2.1: Embed vocabulary**
- Extract BPE merges from SAM3's `bpe_simple_vocab_16e6.txt.gz`
- Either embed directly in `sam3.cpp` as a compressed byte array, or load from file
- Implement `sam3_load_bpe_vocab()`
- Implement `sam3_bpe_encode()` and `sam3_tokenize()`

**Step 2.2: Verify tokenizer**
- Test with 20+ strings, compare against Python's `SimpleTokenizer`
- Ensure SOT/EOT tokens, padding, truncation all match

### Phase 3: Image Encoder (ViT Backbone)

**Step 3.1: Preprocessing**
- Implement `sam3_preprocess_image()`: resize to 1008×1008, normalize
- Implement `sam3_load_image()` using stb_image
- **Verify:** Dump preprocessed tensor, compare against PyTorch preprocessing

**Step 3.2: Patch embedding**
- Implement patch embed as `ggml_conv_2d` with kernel=stride=14
- **Verify:** Compare output against PyTorch patch_embed

**Step 3.3: Positional embedding**
- Handle tiled positional embedding (from 24×24 pretrained to 72×72)
- The conversion script should handle interpolation; C++ just loads and adds
- **Verify:** Check pos_embed values match

**Step 3.4: RoPE precomputation**
- Implement `sam3_precompute_rope()`: compute 2D axial frequency table [5184, 64]
- **Verify:** Compare against PyTorch's `compute_axial_cis()`

**Step 3.5: Window partition/unpartition**
- Implement using `ggml_view` + `ggml_permute` (or `ggml_win_part`/`ggml_win_unpart` if available)
- **Verify:** Round-trip test (partition → unpartition = identity)

**Step 3.6: Single ViT block**
- Implement `sam3_vit_block_forward()`: norm → attn (with RoPE) → residual → norm → MLP → residual
- Handle both window attention (24×24 windows) and global attention
- **Verify:** Compare block 0 output against PyTorch

**Step 3.7: Full ViT forward**
- Implement `sam3_build_vit_graph()`: ln_pre → 32 blocks → final permute
- **Verify:** Compare full ViT output [1, 1024, 72, 72] against PyTorch

**Step 3.8: Neck (SimpleFPN)**
- Implement `sam3_build_neck_graph()` for both detector and tracker paths
- ConvTranspose2d for upsampling, Conv1x1 + Conv3x3 for refinement
- **Verify:** Compare each scale level output against PyTorch

**Step 3.9: Positional encoding (sinusoidal)**
- Implement `sam3_sinusoidal_pe_2d()` for neck feature maps
- **Verify:** Compare against PyTorch PositionEmbeddingSine

### Phase 4: Text Encoder

**Step 4.1: Text encoder forward**
- Implement `sam3_build_text_encoder_graph()`
- 24 blocks: causal MHA + LayerScale + MLP + LayerScale
- Build causal mask (lower triangular)
- Final LayerNorm + Linear projection (1024→256)
- **Verify:** Compare text features for 5 test prompts

### Phase 5: Detector (Image PCS)

**Step 5.1: Fusion encoder**
- Implement `sam3_build_fenc_graph()`: 6 layers of SA + CA + FFN
- Pre-norm architecture with cross-attention to text tokens
- **Verify:** Compare each layer output

**Step 5.2: DETR decoder**
- Implement `sam3_build_ddec_graph()`: query init, 6 layers, box refinement
- Box refinement: inverse_sigmoid + delta + sigmoid
- DotProductScoring for classification
- Presence token scoring
- **Verify:** Compare predicted boxes, scores, presence

**Step 5.3: Segmentation head**
- Implement `sam3_build_seg_head_graph()`: pixel decoder + mask prediction
- Einsum of query embeddings × pixel features → per-instance masks
- **Verify:** Compare mask logits

**Step 5.4: Post-processing**
- Implement NMS, score thresholding, mask binarization
- Implement bilinear interpolation to original size
- **Verify:** Compare final detections (boxes, masks, scores) end-to-end

**Step 5.5: Geometry / Exemplar encoder (optional at this stage)**
- Implement for image exemplar support
- ROI pooling, coordinate embedding, type embedding, 3-layer transformer
- **Verify:** Compare exemplar-conditioned detection results

**Step 5.6: Wire up `sam3_segment_pcs()`**
- Connect: text encode → fusion encode → DETR decode → seg head → post-process
- **Milestone:** Can segment images with text prompts!

### Phase 6: SAM-style Visual Prompting (Image PVS)

**Step 6.1: SAM prompt encoder**
- Implement `sam3_build_sam_pe_graph()`: point/box encoding with random Fourier PE
- **Verify:** Compare sparse + dense embeddings

**Step 6.2: SAM mask decoder**
- Implement `sam3_build_sam_dec_graph()`: TwoWayTransformer + upscaling + hypernetwork
- **Verify:** Compare masks, IoU predictions, object scores

**Step 6.3: Wire up `sam3_segment_pvs()`**
- Connect: SAM prompt encode → (backbone features from state) → mask decode → post-process
- **Milestone:** Can segment with clicks/boxes on images!

### Phase 7: Video Tracking

**Step 7.1: Memory encoder**
- Implement `sam3_build_mem_enc_graph()`: mask downsample + fuser + projection
- **Verify:** Compare memory features

**Step 7.2: Memory attention**
- Implement `sam3_build_mem_attn_graph()`: 4-layer transformer with RoPE cross-attention
- Implement `sam3_rope_attention()` with kv_in_dim=64 and rope_k_repeat
- **Verify:** Compare conditioned features

**Step 7.3: Object pointer extraction**
- Implement `sam3_extract_obj_ptr()`: MLP projection of SAM output token
- Handle occlusion case (use no_obj_ptr)
- **Verify:** Compare object pointers

**Step 7.4: Tracker infrastructure**
- Implement `sam3_create_tracker()`
- Implement memory bank (sliding window of 7)
- Implement object pointer bank (up to 16)
- Implement `sam3_select_memory_frames()`

**Step 7.5: Single-frame propagation**
- Implement `sam3_propagate_single()`: memory attention → SAM mask decoder
- **Verify:** Compare propagated mask for frame N+1 given frame N's memory

**Step 7.6: Detection + matching**
- Implement `sam3_detect_frame()` (reuse PCS pipeline)
- Implement `sam3_match_detections()` (IoU-based matching)
- Implement `sam3_update_tracker()`

**Step 7.7: Disambiguation strategies**
- Implement hotstart delay (confirmation window)
- Implement duplicate removal
- Implement masklet suppression (MDS < 0)
- Implement periodic re-prompting
- Implement detection-guided re-prompting
- **Verify:** Compare tracker output for a 30-frame test video

**Step 7.8: Post-processing**
- Implement non-overlapping constraints
- Implement hole filling (connected components)
- Implement `sam3_fill_holes()` and `sam3_remove_sprinkles()`

**Step 7.9: Wire up `sam3_track_frame()`**
- Full per-frame pipeline: detect → propagate → match → update → memory
- Implement `sam3_refine_instance()`
- **Milestone:** Can track objects through video!

### Phase 8: Interactive Examples

**Step 8.1: Image example**
- Set up SDL2 + ImGui in `examples/third-party/`
- Implement `main_image.cpp`: image loading, text prompt, mask rendering
- Test with real images

**Step 8.2: Video example**
- Implement video frame decoding via ffmpeg pipe
- Implement `main_video.cpp`: video playback, tracking, mask overlay
- Test with real videos

**Step 8.3: Performance optimization**
- Profile with Instruments on macOS
- Ensure Metal backend is used for all heavy operations
- Optimize memory allocation (reuse graph allocators)
- Target: <100ms per image, <50ms per video frame (on M1+)

### Phase 9: Polish

**Step 9.1: End-to-end testing**
- Run full test suite (all per-module tests + end-to-end)
- Fix any numerical discrepancies
- Test edge cases: empty prompts, very large images, long videos

**Step 9.2: Memory optimization**
- Implement f16 inference for reduced memory
- Quantization support (q4_0, q8_0) for backbone if ggml supports it
- Memory mapping for model loading

**Step 9.3: Documentation**
- README with build instructions, usage examples
- Inline code comments for non-obvious operations

---

## 15. Appendix: Tensor Shape Reference

### Image Path

| Stage | Tensor | Shape |
|---|---|---|
| Input | Preprocessed image | `[1, 3, 1008, 1008]` |
| Patch embed | After conv | `[1, 1024, 72, 72]` |
| ViT | After permute + pos_embed | `[1, 72, 72, 1024]` |
| ViT window | Partitioned (local attn) | `[9, 24, 24, 1024]` |
| ViT window | Q, K, V per head | `[9, 16, 576, 64]` |
| ViT global | Full attention | `[1, 16, 5184, 64]` |
| ViT | Output (after final permute) | `[1, 1024, 72, 72]` |
| Neck 4× | FPN level 0 | `[1, 256, 288, 288]` |
| Neck 2× | FPN level 1 | `[1, 256, 144, 144]` |
| Neck 1× | FPN level 2 | `[1, 256, 72, 72]` |
| Text | Token IDs | `[1, 32]` |
| Text | After embedding | `[32, 1, 1024]` |
| Text | After encoder | `[32, 1, 1024]` |
| Text | After resizer | `[32, 1, 256]` |
| Fusion enc | Image features (flat) | `[5184, 1, 256]` |
| Fusion enc | Prompt tokens | `[L, 1, 256]` |
| DETR dec | Query embed | `[200, 512]` |
| DETR dec | Object queries | `[201, 1, 256]` (incl. presence token) |
| DETR dec | Reference boxes | `[1, 200, 4]` |
| DETR dec | Class scores | `[1, 200]` |
| DETR dec | Predicted boxes | `[1, 200, 4]` |
| Seg head | Pixel features (upsampled) | `[1, 256, 288, 288]` |
| Seg head | Instance masks | `[1, N, 288, 288]` |

### Tracker Path

| Stage | Tensor | Shape |
|---|---|---|
| SAM PE | Sparse embeddings | `[1, N_pts, 256]` |
| SAM PE | Dense embeddings | `[1, 256, 72, 72]` |
| SAM dec | Tokens (initial) | `[1, 6+N_pts, 256]` |
| SAM dec | Source (flat) | `[1, 5184, 256]` |
| SAM dec | Up1 (after deconv) | `[1, 64, 144, 144]` |
| SAM dec | Up2 (after deconv) | `[1, 32, 288, 288]` |
| SAM dec | Masks (low-res) | `[1, 4, 288, 288]` |
| SAM dec | IoU prediction | `[1, 4]` |
| SAM dec | Object score | `[1, 1]` |
| Mem enc | Mask downsampled | `[1, 256, 72, 72]` |
| Mem enc | Fused features | `[1, 256, 72, 72]` |
| Mem enc | Memory features | `[1, 64, 72, 72]` |
| Mem enc | Temporal pos enc | `[7, 1, 1, 64]` |
| Mem attn | Current features | `[5184, 1, 256]` |
| Mem attn | Spatial memory (per frame) | `[5184, 1, 64]` |
| Mem attn | Object pointer (per obj) | `[4, 1, 64]` |
| Mem attn | Conditioned output | `[5184, 1, 256]` → `[1, 256, 72, 72]` |
| Obj ptr | Extracted pointer | `[1, 256]` |
| Obj ptr | Split for cross-attn | `[4, 1, 64]` |

---

## 16. Appendix: Activation Functions Reference

| Activation | Formula | Where used |
|---|---|---|
| GELU | `x × Φ(x)` where Φ is the standard normal CDF | ViT MLP, Neck deconv, Text MLP, SAM decoder upscale, Memory encoder (CXBlock, MaskDownSampler) |
| ReLU | `max(0, x)` | Fusion encoder FFN, DETR decoder FFN, Geometry encoder FFN, SAM TwoWayAttentionBlock MLP, DotProductScoring MLP |
| Sigmoid | `1 / (1 + exp(-x))` | Box prediction, mask prediction, object score, presence score, IoU prediction |
| Softmax | `exp(x_i) / Σ exp(x_j)` | All attention layers |
| LayerScale | `x × γ` (learned scalar) | ViT blocks (if init_values set), Text encoder blocks |

---

## Summary

This plan describes the complete port of SAM 3 (~850M params) to C++ using ggml. The implementation covers:

- **~50 internal structs** holding model weights and state
- **~80 functions** for the forward pass, graph construction, and post-processing
- **16 public API functions** in the header
- **1 Python conversion script** for weight conversion
- **10+ test scripts** for layer-by-layer numerical verification
- **2 interactive example executables** (image + video)

The total C++ implementation is estimated at **8,000-12,000 lines** in `sam3.cpp`, with the header at **~200 lines**.

Key technical challenges:
1. **2D Axial RoPE** in ggml — requires custom implementation or use of ggml_rope_ext
2. **Window attention** — ggml_win_part/ggml_win_unpart or manual view/permute
3. **Memory bank management** — non-trivial state management across frames
4. **ConvTranspose2d** — ensure ggml's implementation matches PyTorch stride/padding
5. **BPE tokenizer** — must exactly match CLIP's tokenizer for correctness
6. **Box-relative positional bias** — DETR-specific positional encoding in decoder cross-attention
7. **Connected components** — needed for hole filling, must implement in pure C++
