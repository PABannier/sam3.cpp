# CLAUDE.md

## Project

sam3.cpp — a C++14 port of Meta's SAM 3 (Segment Anything Model 3) using ggml for inference on CPU and Metal.

## Architecture

- **One library**: `sam3.cpp` (implementation) + `sam3.h` (public API).
- **Structs and free functions only**. No classes, no inheritance, no virtual dispatch, no polymorphism.
- **C++14 idioms**: `std::unique_ptr`, `std::shared_ptr`, `std::make_unique`, move semantics, lambdas, `auto`. Use them.
- **Speed is a first-class citizen**. Avoid unnecessary copies, prefer in-place ggml ops (`_inplace` variants), reuse graph allocators, minimize allocations in hot paths. Always use the fastest available ggml kernels: prefer `ggml_flash_attn_ext` over manual Q·K^T→softmax→V when the backend supports it, use fused ops where ggml provides them, and check `ggml/examples/` for the most up-to-date patterns. Profile before over-engineering.

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

## Build

```bash
cd build && cmake .. && make -j$(sysctl -n hw.ncpu)
```

Tests: `cmake .. -DSAM3_BUILD_TESTS=ON`

## Weights

PyTorch checkpoint → `convert_sam3_to_ggml.py` → `.ggml` binary. The conversion stores every tensor (1465 total). The C++ loader registers all 1465 and reads them via `ggml_backend_tensor_set`.
