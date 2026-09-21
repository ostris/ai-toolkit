import json
import struct
from typing import Dict, Optional

import torch

# safetensors dtype tags
_DTYPE_TAGS = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

_CHUNK_BYTES = 256 * 1024 * 1024


def save_file_streamed(
    tensors: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, str]] = None,
):
    """Write a safetensors file with sequential writes instead of ftruncate+mmap.
    Some filesystems (kernel ntfs3) reject one large ftruncate; incremental writes succeed."""
    header = {}
    offset = 0
    flat = {}
    for name, t in tensors.items():
        if t.dtype not in _DTYPE_TAGS:
            raise ValueError(f"Unsupported dtype {t.dtype} for tensor {name}")
        t = t.detach().to("cpu").contiguous()
        nbytes = t.numel() * t.element_size()
        header[name] = {
            "dtype": _DTYPE_TAGS[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        flat[name] = t
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # pad header to 8-byte alignment like the reference writer
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name, t in flat.items():
            if t.numel() == 0:
                continue
            buf = memoryview(t.reshape(-1).view(torch.uint8).numpy())
            for start in range(0, len(buf), _CHUNK_BYTES):
                f.write(buf[start:start + _CHUNK_BYTES])
