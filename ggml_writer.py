"""Tensor record writer shared by the convert_*_to_ggml.py scripts."""

import struct
import numpy as np

FTYPE_F32 = 0
FTYPE_F16 = 1

# Names containing any of these (embeddings, positions, tokens, ...) stay f32.
KEEP_F32 = ("embed", "tpos", "pe_gaussian", "token", "no_obj", "no_mem", "gamma")


def write_tensor(fout, name, data, ftype, keep_f32=KEEP_F32):
    """Write one tensor record with 32-byte aligned data. 1D tensors always stay f32."""
    use_f16 = ftype == FTYPE_F16 and data.ndim >= 2 and not any(s in name for s in keep_f32)
    name_bytes = name.encode("utf-8")
    fout.write(struct.pack("<3i", data.ndim, len(name_bytes), FTYPE_F16 if use_f16 else FTYPE_F32))
    fout.write(struct.pack(f"<{data.ndim}i", *reversed(data.shape)))  # ggml dims are reversed
    fout.write(name_bytes)
    fout.write(b"\x00" * (-fout.tell() % 32))
    fout.write(data.astype(np.float16 if use_f16 else np.float32).tobytes())
