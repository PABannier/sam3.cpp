# CLAUDE.md

## Project

sam3.cpp — a C++14 port of Meta's SAM 3 (Segment Anything Model 3) using ggml for inference on CPU, Metal, and CUDA.

## Architecture

- **One library**: `sam3.cpp` (implementation) + `sam3.h` (public API).
- **Structs and free functions only**. No classes, no inheritance, no virtual dispatch, no polymorphism.
- **C++14 idioms**: `std::unique_ptr`, `std::shared_ptr`, `std::make_unique`, move semantics, lambdas, `auto`. Use them.
- **Speed is a first-class citizen**. Avoid unnecessary copies, prefer in-place ggml ops (`_inplace` variants), minimize allocations in hot paths. Always use the fastest available ggml kernels: prefer `ggml_flash_attn_ext` over manual Q·K^T→softmax→V when the backend supports it, use fused ops where ggml provides them, and check `ggml/examples/` for the most up-to-date patterns. Profile before over-engineering.

## ggml graph isolation (CRITICAL)

**Each logical pipeline stage MUST run in its own ggml sub-graph** with its own `ggml_context`, `ggml_cgraph`, and `ggml_gallocr`. Data flows between stages as CPU-side `std::vector<float>` buffers.

**Why:** The ggml graph allocator (`ggml_gallocr`) reuses intermediate tensor buffers once their consumers have executed. In a single large graph spanning multiple transformer stages, the allocator overwrites buffers that downstream stages still need. This produces silently wrong numerical results — not crashes, just garbage outputs that are extremely hard to debug.

**Concrete rules:**

1. **One sub-graph per transformer block/stage.** The text encoder, geometry encoder, fusion encoder, DETR decoder, segmentation head, memory encoder, and memory attention each get their own `ggml_context` + `ggml_gallocr`. Build → allocate → set inputs → compute → read outputs → free.

2. **NEVER use state tensors as graph operands.** Tensors from `state.neck_trk[*]`, `state.neck_det[*]`, or any previous graph's output MUST NOT appear as arguments to `ggml_add`, `ggml_reshape`, `ggml_permute`, or any graph builder function. `ggml_build_forward_expand` traces the entire dependency tree — using a state tensor pulls in ALL its ancestors (the full ViT + neck recomputation: 2500+ nodes, ~40 seconds). Instead, create a fresh input tensor and copy data via CPU:
   ```cpp
   // WRONG — pulls in entire ViT recomputation:
   auto* x = ggml_reshape_3d(ctx, state.neck_trk[2], D, N, 1);

   // CORRECT — isolated input, no dependency chain:
   auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N, 1);
   ggml_set_name(x, "input"); ggml_set_input(x);
   // after ggml_gallocr_alloc_graph:
   std::vector<float> buf(D * N);
   ggml_backend_tensor_get(state.neck_trk[2], buf.data(), 0, buf.size() * sizeof(float));
   ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * sizeof(float));
   ```

3. **Model weight tensors are safe.** The model's weight tensors (in `model.xxx.weight`) live in a separate persistent buffer and are never managed by the graph allocator. They can be referenced directly in graph ops (e.g., `ggml_mul_mat(ctx, model.layer.weight, x)`).

**Functions that follow this pattern:** `sam3_segment_pcs` (5 sub-graphs), `sam3_segment_pvs`, `sam3_propagate_single`, `sam3_encode_memory`.

## Implementation plan

All work follows the phased plan in `PLAN.md`. Read it before starting any phase. Each phase has concrete steps, verification criteria, and the exact structs/functions to implement.

## Reference implementations

When lost on how to structure the ggml forward pass, how to build graphs, or how to load weights:

1. **sam.cpp** (https://github.com/YavorGIvanov/sam.cpp) — the original SAM 1 port to C++/ggml. Study `sam.cpp` and `sam.h` for patterns: graph construction, two-pass measure+compute, `ggml_backend_tensor_set`, window partition, attention with relative position, mask decoder upscaling. Our code follows the same conventions.

2. **ggml examples** (`ggml/examples/` in the submodule) — canonical, up-to-date examples of how to use ggml APIs. Check these for: backend init, graph allocation (`ggml_gallocr`), tensor creation, `ggml_backend_graph_compute`, Metal usage. The ggml API evolves; the submodule examples are always correct for our pinned version.

3. **SAM 3 official repo** (https://github.com/facebookresearch/sam3) — the ground truth for the forward pass. When in doubt about tensor shapes, operation order, activation functions, or any architectural detail, read the Python source. The paper is in `sam3.pdf`.

## Code style

- Prefix all internal (static) functions with `sam3_`.
- ggml graph-building functions take `ggml_context *` as first arg and return `ggml_tensor *`.
- Weight structs hold raw `ggml_tensor *` pointers (owned by the model's ggml context).
- Use `fprintf(stderr, ...)` for diagnostics, not `std::cerr`.
- No exceptions. Check return values. Functions that can fail return `bool` or `nullptr`.

## Dependencies

Only: ggml (submodule), stb_image/stb_image_write (vendored in `stb/`), C++14 standard library. Nothing else in the library. SDL2/ImGui are example-only.

## Python

`uv` is the package manager. Use `uv run python` for all Python execution (scripts, tests, weight conversion). Never use bare `python` or `pip` — always `uv run python` and `uv pip install`.

## Build

```bash
cd build && cmake .. && make -j$(sysctl -n hw.ncpu)
```

Tests: `cmake .. -DSAM3_BUILD_TESTS=ON`

## Benchmarking

`sam3_benchmark` tracks an object across video frames and reports latency for every model × backend combination. Each run is forked into a subprocess so a crash does not kill the suite.

```bash
# Full benchmark (all 49 models × Metal + CPU):
./build/examples/sam3_benchmark

# Quick iteration (e.g. testing an optimization) — 4 runs, ~30 s:
./build/examples/sam3_benchmark --filter tiny --n-frames 3 --filter-prec f16,q4_0

# Metal only:
./build/examples/sam3_benchmark --gpu-only

# CPU only:
./build/examples/sam3_benchmark --cpu-only
```

**Quick-iteration recipe:** when profiling or testing optimizations, `--filter tiny --n-frames 3` limits to the SAM2/2.1 tiny models on both Metal and CPU in f16 and q4_0 — just 4 runs total, enough to see whether a change helps without waiting for the full suite.

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir <path>` | `models/` | Directory containing `.ggml` files |
| `--video <path>` | `data/test_video.mp4` | Video file |
| `--point-x <f>` | `315.0` | X coordinate of the tracking point |
| `--point-y <f>` | `250.0` | Y coordinate of the tracking point |
| `--n-frames <n>` | `10` | Number of frames to track |
| `--n-threads <n>` | `4` | CPU thread count |
| `--cpu-only` | | Skip Metal runs |
| `--gpu-only` | | Skip CPU runs |
| `--filter <substr>` | | Only run models whose filename contains `<substr>` |

Output columns: model name, file size, backend, load time, init time (frame 0 encode + add instance), average per-frame tracking time, total pipeline time, detection count, status. Diagnostics go to stderr; the final table goes to stdout (pipe-friendly: `./build/examples/sam3_benchmark 2>/dev/null > results.txt`).

## Weights

PyTorch checkpoint → `convert_sam3_to_ggml.py` → `.ggml` binary. The conversion stores every tensor (1465 total). The C++ loader registers all 1465 and reads them via `ggml_backend_tensor_set`.
