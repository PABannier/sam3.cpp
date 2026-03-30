#!/usr/bin/env python3
"""Compare SAM2 backbone tensors between Python and C++."""
import numpy as np
import os

def compare(name, ref, cpp, atol=1e-3):
    if ref.size != cpp.size:
        print(f"  FAIL {name:40s}  SIZE MISMATCH ref={ref.size} cpp={cpp.size}")
        return False
    ref_f = ref.flatten()
    cpp_f = cpp.flatten()
    diff = np.abs(ref_f - cpp_f)
    eps = 1e-8
    max_d = diff.max()
    mean_d = diff.mean()
    cos = np.dot(ref_f, cpp_f) / (np.linalg.norm(ref_f) * np.linalg.norm(cpp_f) + eps)
    p95 = np.percentile(diff, 95)
    p99 = np.percentile(diff, 99)
    status = "PASS" if cos > 0.99 else "FAIL"
    print(f"  {status} {name:40s}")
    print(f"         mae={mean_d:.6e}  max={max_d:.6e}  cos={cos:.8f}  p95={p95:.6e}  p99={p99:.6e}")
    if status == "FAIL":
        worst = np.argmax(diff)
        print(f"         worst idx={worst}  ref={ref_f[worst]:.6e}  cpp={cpp_f[worst]:.6e}")
    return status == "PASS"

py_dir = "/tmp/debug_sam2_pipeline"
cpp_dir = "/tmp/debug_sam2_cpp"

pairs = [
    ("imgenc_backbone_fpn_0", "neck_trk_0", (1, 256, 256, 256)),
    ("imgenc_backbone_fpn_1", "neck_trk_1", (1, 256, 128, 128)),
    ("imgenc_backbone_fpn_2", "neck_trk_2", (1, 256, 64, 64)),
]

print("=== SAM2 Backbone + FPN Comparison ===\n")
n_pass = 0
for py_name, cpp_name, py_shape in pairs:
    py_path = os.path.join(py_dir, py_name + ".bin")
    cpp_path = os.path.join(cpp_dir, cpp_name + ".bin")

    py_data = np.fromfile(py_path, dtype=np.float32)
    cpp_data = np.fromfile(cpp_path, dtype=np.float32)

    print(f"  {py_name}: {py_data.size} floats, {cpp_name}: {cpp_data.size} floats")

    # Just compare flat — if cosine is high, the data matches regardless of layout
    if compare(f"{py_name} vs {cpp_name}", py_data, cpp_data):
        n_pass += 1

print(f"\n{n_pass}/{len(pairs)} passed")
