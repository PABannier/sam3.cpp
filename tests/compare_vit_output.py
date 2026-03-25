#!/usr/bin/env python3
"""Compare C++ and Python ViT outputs in detail to find outliers.

The C++ dump stores data in ggml column-major order with shape ne[0], ne[1], ne[2].
For a tensor with ne=[E, W, H], element (e, w, h) is at flat index e + w*E + h*E*W.
This is the REVERSE of numpy's row-major convention.
"""
import numpy as np
import sys
import os

def load_ggml_tensor(path):
    """Load a tensor dumped by sam3_dump_state_tensor.
    Returns data in ggml flat order and the ggml shape (ne[0], ne[1], ne[2], ...).
    """
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",")]
    with open(path + ".bin", "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
    return data, shape

def ggml_to_pytorch_bchw(data, ggml_shape):
    """Convert ggml [E, W, H] flat data to PyTorch [1, C, H, W] array.
    ggml element (e, w, h) at flat index e + w*E + h*E*W
    maps to PyTorch [0, e, h, w].
    """
    E, W, H = ggml_shape[:3]
    result = np.zeros((1, E, H, W), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            for e in range(E):
                result[0, e, h, w] = data[e + w*E + h*E*W]
    return result

def load_pytorch_tensor(path):
    """Load a tensor saved by the Python dumper (standard row-major)."""
    with open(path + ".shape") as f:
        shape = [int(x) for x in f.read().strip().split(",")]
    with open(path + ".bin", "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
    return data.reshape(shape)

ref_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/ref_phase3"
cpp_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/ref_phase3/cpp_out_phase3"

# Load Python reference (BCHW)
py_vit = load_pytorch_tensor(f"{ref_dir}/vit_output_bchw")  # [1, 1024, 72, 72]
print(f"Python ViT output: shape={py_vit.shape}, range=[{py_vit.min():.4f}, {py_vit.max():.4f}]")

# Load C++ output (ggml: ne=[E=1024, W=72, H=72])
cpp_data, cpp_shape = load_ggml_tensor(f"{cpp_dir}/vit_output")
print(f"C++ ViT output:    ggml_shape={cpp_shape}, range=[{cpp_data.min():.4f}, {cpp_data.max():.4f}]")

# Transpose properly
E, W, H = cpp_shape[:3]
print(f"  Transposing ggml [E={E}, W={W}, H={H}] to PyTorch [1, {E}, {H}, {W}]...")
cpp_transposed = ggml_to_pytorch_bchw(cpp_data, cpp_shape)

diff = np.abs(cpp_transposed - py_vit)
print(f"\nDifference stats:")
print(f"  max_diff:  {diff.max():.6f}")
print(f"  mean_diff: {diff.mean():.6f}")
print(f"  median:    {np.median(diff):.6f}")
print(f"  99.9th %:  {np.percentile(diff, 99.9):.6f}")
print(f"  99.99th %: {np.percentile(diff, 99.99):.6f}")

# Find the worst elements
flat_diff = diff.flatten()
worst_indices = np.argsort(flat_diff)[-20:][::-1]
print(f"\nTop 20 worst mismatches:")
for idx in worst_indices:
    b, c, h, w = np.unravel_index(idx, diff.shape)
    py_val = py_vit[b, c, h, w]
    cpp_val = cpp_transposed[b, c, h, w]
    d = flat_diff[idx]
    rel = d / max(abs(py_val), 1e-8)
    print(f"  [{b},{c:4d},{h:2d},{w:2d}] py={py_val:10.4f} cpp={cpp_val:10.4f} diff={d:.4f} rel={rel:.4f}")

# Per-channel analysis
chan_max = diff[0].max(axis=(1, 2))  # [1024]
print(f"\nPer-channel max diff distribution:")
for threshold in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    n = (chan_max > threshold).sum()
    print(f"  channels with max_diff > {threshold:5.1f}: {n}/{E}")

# Overall element analysis
for threshold in [0.01, 0.1, 0.5, 1.0, 5.0]:
    n = (diff > threshold).sum()
    print(f"Elements with diff > {threshold}: {n}/{diff.size} ({100*n/diff.size:.2f}%)")

# Cosine similarity
dot = np.sum(cpp_transposed * py_vit)
norm_a = np.sqrt(np.sum(cpp_transposed**2))
norm_b = np.sqrt(np.sum(py_vit**2))
cosine = dot / (norm_a * norm_b)
print(f"\nCosine similarity: {cosine:.8f}")
