"""Binary stream framing for engine responses.

One generation request is answered with a chunked HTTP body made of frames:

    u32 header_len (little endian) | json header (utf-8) |
    u64 payload_len (little endian) | payload bytes

Every frame header has "type" and "request_id". Text-only frames carry an
empty payload. The javascript reader (ui/src/lib/engineStream.ts) mirrors
this exactly. Frame types:

    start     resolved model/sample config, output kind, latent preview info
    status    a holder stage line (loading / quantizing / encoding ...)
    progress  {step, total, elapsed}
    latent    raw tensor payload; header carries shape, dtype, layout
    result    {path, url, kind, ext, ...} for each produced file
    error     {message, cancelled}
    end       terminal frame
"""

import json
import struct
from typing import Iterator, Optional, Tuple

import numpy as np

_HDR = struct.Struct("<I")
_LEN = struct.Struct("<Q")


def encode_frame(header: dict, payload: bytes = b"") -> bytes:
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HDR.pack(len(hdr)) + hdr + _LEN.pack(len(payload)) + payload


def tensor_payload(tensor, dtype="float16") -> Tuple[dict, bytes]:
    """Serialize a tensor to raw bytes + the header fields that describe it."""
    import torch

    t = tensor.detach()
    if t.is_cuda or t.device.type != "cpu":
        t = t.to("cpu")
    if dtype == "float16":
        t = t.to(torch.float16)
    elif dtype == "float32":
        t = t.to(torch.float32)
    arr = t.contiguous().numpy()
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}, arr.tobytes()


def payload_to_array(header: dict, payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=np.dtype(header["dtype"])).reshape(header["shape"])


class FrameReader:
    """Decode frames from an iterator of byte chunks (tests / python clients)."""

    def __init__(self, chunks: Iterator[bytes]):
        self._chunks = chunks
        self._buf = bytearray()
        self._eof = False

    def _fill(self, n: int) -> bool:
        while len(self._buf) < n and not self._eof:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                self._eof = True
                break
            if chunk:
                self._buf.extend(chunk)
        return len(self._buf) >= n

    def _take(self, n: int) -> bytes:
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def read(self) -> Optional[Tuple[dict, bytes]]:
        if not self._fill(_HDR.size):
            return None
        (hlen,) = _HDR.unpack(self._take(_HDR.size))
        if not self._fill(hlen + _LEN.size):
            raise EOFError("truncated frame header")
        header = json.loads(self._take(hlen).decode("utf-8"))
        (plen,) = _LEN.unpack(self._take(_LEN.size))
        if not self._fill(plen):
            raise EOFError("truncated frame payload")
        return header, self._take(plen)

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
