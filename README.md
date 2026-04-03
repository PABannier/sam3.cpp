# sam3.cpp

State-of-the-art image and video segmentation in portable C/C++

![SAM3 Image Segmentation Demo](https://i.imgur.com/xmnxNP1.gif)
![SAM3 Video Segmentation Demo](https://i.imgur.com/5y9QcrY.gif)

---

## Why sam3.cpp?

Running Meta's Segment Anything models typically requires Python, PyTorch, and a CUDA GPU. **sam3.cpp** eliminates all of that. It's a single C++ library that runs SAM 2, SAM 2.1, SAM 3, and EdgeTAM inference on CPU and Apple Metal. No Python runtime, no GPU drivers, no heavyweight dependencies. Just compile and segment.

- **4 model families in one library**: SAM 2, SAM 2.1 (Hiera backbone), SAM 3 (ViT backbone + text detection), and EdgeTAM (RepViT backbone — 22x faster than SAM 2 on mobile)
- **4-bit quantization**: EdgeTAM fits in **15 MB** with Q4_0; SAM 2.1 Tiny in **22 MB** tracking at **~1 fps** on Metal; SAM 3 drops from 3.4 GB to 673 MB
- **Apple Metal GPU acceleration**: the full ViT/Hiera/RepViT backbone and transformer decoder run on the GPU via ggml's Metal backend
- **Text-prompted detection** (SAM 3): type `"cat"` and get every cat in the image, no clicks needed. This is the PCS (Promptable Concept Segmentation) path, unique to SAM 3
- **Point/box segmentation** (all models): click a point or draw a box, get a mask. The classic SAM workflow, available for SAM 2, 2.1, 3, and EdgeTAM
- **Video object tracking**: memory bank with 7 temporal slots propagates masks across frames. Add instances mid-video, refine with clicks, export per-frame PNGs
- **On-device tracking** (EdgeTAM): RepViT-M1 backbone with Perceiver memory compression achieves real-time tracking on Apple Silicon — 27 MB model, **~440 ms/frame** on Metal
- **Single-file library**: `sam3.cpp` + `sam3.h`. Structs and free functions only, C++14, no exceptions, no inheritance
- **Zero dependencies** beyond [ggml](https://github.com/ggerganov/ggml) (tensor computation, bundled as submodule) and [stb](https://github.com/nothings/stb) (image I/O, vendored)

## Quick Start

```bash
# Clone
git clone --recursive https://github.com/PABannier/sam3.cpp
cd sam3.cpp

# Build (Metal GPU enabled automatically on macOS)
mkdir build && cd build
cmake ..
make -j

# Download a model (SAM 2.1 Tiny, 75 MB)
# See "Model Zoo" below for all available models and download links
curl -L -o ../models/sam2.1_hiera_tiny_f16.ggml \
  https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam2.1_hiera_tiny_f16.ggml

# Segment an image interactively (requires SDL2)
./examples/sam3_image --model ../models/sam2.1_hiera_tiny_f16.ggml --image ../data/test_image.jpg

# Track objects in a video interactively (requires SDL2)
./examples/sam3_video --model ../models/sam2.1_hiera_tiny_f16.ggml --video ../data/test_video.mp4
```

The interactive apps use SDL2 + ImGui. If SDL2 isn't found, only the benchmark and quantize tools are built.

## Benchmarks

Video object tracking latency on **Apple M4 Pro (24 GB)**, 5 frames at 1008x1008 resolution, 4 threads. Each run is isolated in a forked subprocess.

### SAM 3 (Full — text detection + visual tracking)

| Model | Size | Track/frame Metal (s) | Track/frame CPU (s) | Total Metal (s) | Total CPU (s) |
|-------|------|-----------------------|---------------------|-----------------|---------------|
| sam3-f32 | 3.2 GB | - | 40.5 | - | 200.4 |
| sam3-f16 | 1.7 GB | **7.7** | 23.8 | 38.1 | 117.5 |
| sam3-q8_0 | 1.0 GB | **7.8** | 23.3 | 38.7 | 115.2 |
| sam3-q4_1 | 756 MB | - | 24.5 | - | 120.9 |
| sam3-q4_0 | 673 MB | **7.8** | 23.9 | 38.7 | 117.7 |

### SAM 3 Visual-Only (no text encoder, tracking only)

| Model | Size | Track/frame Metal (s) | Track/frame CPU (s) | Total Metal (s) | Total CPU (s) |
|-------|------|-----------------------|---------------------|-----------------|---------------|
| sam3-visual-f16 | 901 MB | **6.6** | 22.6 | 32.7 | 111.2 |
| sam3-visual-q8_0 | 493 MB | **6.7** | 22.0 | 33.0 | 108.4 |
| sam3-visual-q4_1 | 318 MB | - | 23.1 | - | 113.9 |
| sam3-visual-q4_0 | 275 MB | **6.7** | 22.3 | 33.0 | 110.0 |

### EdgeTAM (RepViT backbone + Perceiver)

| Model | Size | Track/frame Metal (s) | Track/frame CPU (s) | Total Metal (s) | Total CPU (s) |
|-------|------|-----------------------|---------------------|-----------------|---------------|
| edgetam_f16 | 27 MB | **0.4** | 1.1 | 2.2 | 5.2 |
| edgetam_q8_0 | 19 MB | **0.4** | 1.1 | 2.1 | 5.1 |
| edgetam_q4_0 | 15 MB | **0.4** | 1.1 | 2.1 | 5.1 |

### SAM 2 / SAM 2.1 (Hiera backbone)

| Model | Size | Track/frame Metal (s) | Track/frame CPU (s) | Total Metal (s) | Total CPU (s) |
|-------|------|-----------------------|---------------------|-----------------|---------------|
| sam2_hiera_tiny_f16 | 75 MB | **0.9** | 2.7 | 4.0 | 12.6 |
| sam2_hiera_tiny_q8_0 | 40 MB | **0.9** | 2.5 | 4.0 | 11.7 |
| sam2_hiera_tiny_q4_0 | **22 MB** | **0.9** | 2.5 | 4.0 | 11.7 |
| sam2_hiera_small_f16 | 89 MB | **0.9** | 2.9 | 4.1 | 13.7 |
| sam2_hiera_small_q8_0 | 47 MB | **0.9** | 2.7 | 4.1 | 12.5 |
| sam2_hiera_small_q4_0 | 26 MB | **0.9** | 2.7 | 4.1 | 12.7 |
| sam2_hiera_base_plus_f16 | 155 MB | **1.0** | 4.2 | 4.7 | 20.2 |
| sam2_hiera_base_plus_q8_0 | 83 MB | - | 3.9 | - | 18.9 |
| sam2_hiera_large_f16 | 429 MB | - | 8.4 | - | 40.9 |
| sam2_hiera_large_q8_0 | 230 MB | - | 7.6 | - | 37.1 |
| | | | | | |
| sam2.1_hiera_tiny_f16 | 75 MB | **0.8** | 2.6 | 4.0 | 12.3 |
| sam2.1_hiera_tiny_q8_0 | 40 MB | **0.9** | 2.4 | 4.0 | 11.4 |
| sam2.1_hiera_tiny_q4_0 | **22 MB** | **0.9** | 2.5 | 4.0 | 11.5 |
| sam2.1_hiera_small_f16 | 89 MB | **0.9** | 2.9 | 4.1 | 13.5 |
| sam2.1_hiera_small_q8_0 | 47 MB | **0.9** | 2.7 | 4.1 | 12.5 |
| sam2.1_hiera_small_q4_0 | 26 MB | **0.9** | 2.7 | 4.1 | 12.6 |
| sam2.1_hiera_base_plus_f16 | 155 MB | **1.0** | 4.2 | 4.7 | 20.1 |
| sam2.1_hiera_base_plus_q8_0 | 83 MB | - | 3.9 | - | 18.6 |
| sam2.1_hiera_large_f16 | 430 MB | - | 8.5 | - | 41.4 |
| sam2.1_hiera_large_q8_0 | 230 MB | - | 7.7 | - | 37.7 |


<details>
<summary><b>Reproduce these benchmarks</b></summary>

```bash
# Full benchmark (all models, both backends)
./build/examples/sam3_benchmark

# GPU only, all models
./build/examples/sam3_benchmark --gpu-only

# Quick iteration (tiny models, 3 frames)
./build/examples/sam3_benchmark --filter tiny --n-frames 3 --gpu-only

# CPU only, specific model
./build/examples/sam3_benchmark --cpu-only --filter sam2.1_hiera_small
```

Options: `--models-dir <path>`, `--video <path>`, `--n-frames <n>`, `--n-threads <n>`, `--filter <substr>`, `--cpu-only`, `--gpu-only`

</details>

## Model Zoo

All models are available in GGML format on Hugging Face:

**[PABannier/sam3.cpp](https://huggingface.co/PABannier/sam3.cpp)** — 52 model files covering 4 architectures x multiple sizes x up to 5 precisions.

### SAM 3 (850M params — ViT-32 backbone + text encoder + DETR decoder)

| Variant | Precision | Size | Features |
|---------|-----------|------|----------|
| sam3 | f32 | 3.4 GB | Text detection (PCS) + point/box segmentation (PVS) + video tracking |
| sam3 | f16 | 1.7 GB | Same |
| sam3 | q8_0 | 1.0 GB | Same |
| sam3 | q4_1 | 756 MB | Same |
| sam3 | q4_0 | 707 MB | Same |
| sam3-visual | f16 | 946 MB | Point/box segmentation (PVS) + video tracking (no text) |
| sam3-visual | q8_0 | 517 MB | Same |
| sam3-visual | q4_1 | 318 MB | Same |
| sam3-visual | q4_0 | 289 MB | Same |

### SAM 2 / SAM 2.1 (Hiera backbone, 4 sizes)

| Family | Size | Params | f32 | f16 | q8_0 | q4_1 | q4_0 |
|--------|------|--------|-----|-----|------|------|------|
| SAM 2 | Tiny | 39M | 156 MB | 79 MB | 43 MB | 26 MB | 24 MB |
| SAM 2 | Small | 46M | 184 MB | 94 MB | 50 MB | 30 MB | 28 MB |
| SAM 2 | Base+ | 81M | 323 MB | 163 MB | 88 MB | 53 MB | 48 MB |
| SAM 2 | Large | 224M | 898 MB | 451 MB | 241 MB | 144 MB | 130 MB |
| SAM 2.1 | Tiny | 39M | 156 MB | 79 MB | 43 MB | 26 MB | 24 MB |
| SAM 2.1 | Small | 46M | 184 MB | 94 MB | 50 MB | 30 MB | 28 MB |
| SAM 2.1 | Base+ | 81M | 323 MB | 163 MB | 88 MB | 53 MB | 48 MB |
| SAM 2.1 | Large | 224M | 898 MB | 451 MB | 241 MB | 144 MB | 130 MB |

### EdgeTAM (RepViT-M1 backbone + Perceiver memory compressor)

| Variant | Precision | Size | Features |
|---------|-----------|------|----------|
| edgetam | f16 | 27 MB | Point/box segmentation (PVS) + video tracking |
| edgetam | q8_0 | 19 MB | Same |
| edgetam | q4_0 | 15 MB | Same |

### Feature Matrix

| Capability | SAM 3 | SAM 3 Visual | SAM 2 / 2.1 | EdgeTAM |
|-----------|-------|-------------|-------------|---------|
| Text-prompted detection (PCS) | Yes | - | - | - |
| Point/box segmentation (PVS) | Yes | Yes | Yes | Yes |
| Multi-mask output | Yes | Yes | Yes | Yes |
| Video tracking (memory bank) | Yes | Yes | Yes | Yes |
| Interactive refinement | Yes | Yes | Yes | Yes |
| Quantization (Q4/Q8) | Yes | Yes | Yes | Yes |
| Metal GPU | Yes | Yes | Yes | Yes |

## Building from Source

### Prerequisites

- C++14 compiler (Clang, GCC, MSVC)
- CMake 3.14+
- (Optional) SDL2 — for the interactive image/video examples
- (Optional) ffmpeg — for video frame decoding

### Build

```bash
git clone --recursive https://github.com/PABannier/sam3.cpp
cd sam3.cpp
mkdir build && cd build
cmake ..
make -j
```

Metal is enabled automatically on macOS. To disable it:

```bash
cmake .. -DSAM3_METAL=OFF
```

To build tests:

```bash
cmake .. -DSAM3_BUILD_TESTS=ON
make -j
```

## Usage

### Image Segmentation (Interactive GUI)

```bash
# Point/box segmentation with any model
./sam3_image --model models/sam2.1_hiera_tiny_f16.ggml --image photo.jpg

# Text-prompted detection (SAM 3 only)
./sam3_image --model models/sam3-f16.ggml --image photo.jpg
# → Type "cat" in the text field, click [Segment]
```

**Controls:**
- **Left-click** — add positive point
- **Right-click** — add negative point
- **Drag** — draw bounding box
- **Text field + Segment** — detect all instances matching the text prompt (SAM 3 only)
- **Export** — save masks as PNG

### Video Tracking (Interactive GUI)

```bash
# Visual tracking (SAM 2/2.1/3/EdgeTAM)
./sam3_video --model models/sam2.1_hiera_small_f16.ggml --video input.mp4

# Text-prompted tracking (SAM 3 only)
./sam3_video --model models/sam3-f16.ggml --video input.mp4
```

**Controls:**
- Click a point or draw a box on a paused frame to add an instance
- Click on an existing tracked mask to refine it
- Play/Pause/Step for playback
- Export per-frame mask PNGs

### C++ API

```cpp
#include "sam3.h"

// Load model
sam3_params params;
params.model_path = "models/sam2.1_hiera_tiny_f16.ggml";
params.use_gpu    = true;
params.n_threads  = 4;

auto model = sam3_load_model(params);
auto state = sam3_create_state(*model, params);

// Encode image (call once, reuse for multiple prompts)
auto image = sam3_load_image("photo.jpg");
sam3_encode_image(*state, *model, image);

// Segment with a point click
sam3_pvs_params pvs;
pvs.pos_points.push_back({315.0f, 250.0f});
sam3_result result = sam3_segment_pvs(*state, *model, pvs);

for (auto& det : result.detections) {
    sam3_save_mask(det.mask, "mask.png");
}
```

```cpp
// Text-prompted detection (SAM 3 full model only)
sam3_pcs_params pcs;
pcs.text_prompt     = "yellow school bus";
pcs.score_threshold = 0.5f;
sam3_result result = sam3_segment_pcs(*state, *model, pcs);
// → result.detections contains every matching instance
```

```cpp
// Video tracking
auto tracker = sam3_create_visual_tracker(*model, {});

// Frame 0: encode + add instance with a click
sam3_encode_image(*state, *model, frame0);
sam3_pvs_params pvs;
pvs.pos_points.push_back({315.0f, 250.0f});
sam3_tracker_add_instance(*tracker, *state, *model, pvs);

// Subsequent frames: propagate masks
for (int f = 1; f < n_frames; f++) {
    sam3_result result = sam3_propagate_frame(*tracker, *state, *model, frames[f]);
    // result.detections[i].mask — tracked mask for each instance
}
```

### Quantization

Convert F32/F16 weights to smaller quantized formats:

```bash
./sam3_quantize models/sam3-f16.ggml models/sam3-q4_0.ggml q4_0
# Supported types: q4_0, q4_1, q8_0
```

## Converting Weights

Convert official PyTorch checkpoints to GGML format:

```bash
# SAM 3
uv run python convert_sam3_to_ggml.py \
    --model sam3.pt \
    --output models/sam3-f16.ggml \
    --ftype 1 \
    --tokenizer /path/to/tokenizer

# SAM 3 visual-only (no text encoder, smaller file)
uv run python convert_sam3_to_ggml.py \
    --model sam3.pt \
    --output models/sam3-visual-f16.ggml \
    --ftype 1 \
    --visual-only

# SAM 2 / SAM 2.1
uv run python convert_sam2_to_ggml.py \
    --model sam2.1_hiera_large.pt \
    --config sam2.1_hiera_l.yaml \
    --output models/sam2.1_hiera_large_f16.ggml \
    --ftype 1

# EdgeTAM
uv run python convert_edgetam_to_ggml.py \
    --model edgetam.pt \
    --output models/edgetam_f16.ggml \
    --ftype 1
```

`--ftype 0` = float32, `--ftype 1` = float16 (recommended). Then quantize with `sam3_quantize`.

## Acknowledgments

- [Meta AI Research](https://github.com/facebookresearch) — SAM, SAM 2, SAM 3, and EdgeTAM
- [ggml](https://github.com/ggerganov/ggml) — the tensor computation library that makes this possible
- [sam.cpp](https://github.com/YavorGIvanov/sam.cpp) — the original SAM 1 C++ port that inspired this project's architecture

## License

MIT
