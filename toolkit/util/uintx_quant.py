"""
Bit-exact reimplementation of torchao 0.10.0's UIntXWeightOnlyConfig weight
quantization as an OstrisQuantizer backend, so the uint2..uint7 qtypes keep
producing byte-identical weights after torchao drops uintx support. Existing
accuracy recovery adapters were trained against these exact quantized bases,
so every op below mirrors torchao's sequence (same order, same dtypes):

  choose_qparams_affine (ASYMMETRIC, block_size (1, 64), preserve_zero=True,
  eps=float32 eps, INT zero-point domain, scale in the weight dtype):
    min/max per 64-wide group along in_features, extended to include 0
    scale      = (max_val_pos - min_val_neg) / (qmax - qmin), clamped to eps
    zero_point = clamp(qmin - round(min_val_neg / scale), qmin, qmax) as int
  quantize_affine:
    q = clamp(round(w * (1.0 / scale)) + zero_point, qmin, qmax)
  dequantize_affine:
    w = ((q as int) - zero_point) cast to the weight dtype, times scale

The arithmetic is done in the original weight dtype (usually bfloat16) because
that is what torchao did; doing it in float32 would round differently.

uint8 resolves to a backend too, but can_quantize always refuses it: torchao
0.10.0's uint8 path raised inside UintxTensor.from_uint8 (packing only supports
1..7 bits) before the module was touched, so "uint8" layers have always been
silently left unquantized. Refusing keeps that exact behavior — flip the check
in can_quantize if real uint8 quantization is ever wanted for new models.

Buffers on the module (registered by quantize_):
  uintx_packed      quantized codes packed into power-of-2 bit shards (like
                    torchao's UintxTensor: e.g. uint3 = a 2-bit + a 1-bit
                    shard), concatenated into one flat uint8 buffer. Shards
                    unpack with a couple of elementwise shift/mask kernels,
                    which is much cheaper per forward than a generic bitstream.
  uintx_scale       per-group scale, stored as a uint8 byte view of the weight
                    dtype so module.to(dtype=...) can't cast it
  uintx_zero_point  per-group zero point, uint8 (values are in [0, qmax])
"""

import torch

from toolkit.util.ostris_quant import OstrisLinear, OstrisQuantizer

UINTX_QTYPES = {f"uint{bits}": bits for bits in range(2, 9)}

_EPS = torch.finfo(torch.float32).eps


def _pack_shard(vals: torch.Tensor, k: int) -> torch.Tensor:
    """Pack flat uint8 values (< 2**k) into bytes, 8 // k values per byte.
    Values are laid out in 8//k contiguous chunks (chunk j holds bits
    [j*k, j*k+k) of every byte) so pack and unpack touch memory coalesced."""
    vpb = 8 // k
    if vpb == 1:
        return vals.clone()
    pad = (-vals.numel()) % vpb
    if pad:
        vals = torch.cat([vals, vals.new_zeros(pad)])
    chunks = vals.view(vpb, -1)
    out = chunks[0].clone()
    for j in range(1, vpb):
        out |= chunks[j] << (j * k)
    return out


def _unpack_shard(packed: torch.Tensor, k: int, numel: int) -> torch.Tensor:
    vpb = 8 // k
    if vpb == 1:
        return packed[:numel]
    out = torch.empty(vpb, packed.numel(), dtype=torch.uint8, device=packed.device)
    for j in range(vpb):
        torch.bitwise_right_shift(packed, j * k, out=out[j])
    out.bitwise_and_((1 << k) - 1)
    return out.view(-1)[:numel]


def pack_uintx(codes: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack integer codes (values < 2**nbits) into concatenated power-of-2 bit
    shards; bits [offset, offset+k) of each code land in the k-bit shard."""
    flat = codes.flatten().to(torch.uint8)
    shards = []
    offset = 0
    for k in (8, 4, 2, 1):
        if nbits & k:
            if k == nbits:  # single shard, values are already < 2**k
                shards.append(_pack_shard(flat, k))
            else:
                shards.append(_pack_shard((flat >> offset) & ((1 << k) - 1), k))
            offset += k
    return torch.cat(shards) if len(shards) > 1 else shards[0]


def unpack_uintx(packed: torch.Tensor, nbits: int, numel: int) -> torch.Tensor:
    """Inverse of pack_uintx. Returns a flat uint8 tensor of length numel."""
    out = None
    offset = 0
    pos = 0
    for k in (8, 4, 2, 1):
        if nbits & k:
            vpb = 8 // k
            nbytes = -(-numel // vpb)
            vals = _unpack_shard(packed[pos : pos + nbytes], k, numel)
            if out is None:
                # a multi-shard first value is always a fresh tensor from
                # _unpack_shard, safe to mutate; a lone 8-bit shard is a view
                # of the buffer but is never combined, so read-only is fine
                out = vals
            else:
                out |= vals.bitwise_left_shift_(offset)
            offset += k
            pos += nbytes
    return out


class UIntXQuantizer(OstrisQuantizer):
    # quantization runs in the weight's own dtype (that is what torchao did);
    # skipping the float32 copy keeps peak memory at torchao levels
    wants_fp32_weight = False

    def __init__(self, nbits: int, group_size: int = 64):
        self.nbits = nbits
        self.group_size = group_size
        self.qmin = 0
        self.qmax = (1 << nbits) - 1

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        if self.nbits == 8:
            # see module docstring: uint8 has always been a silent no-op
            return False
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            return False
        # torchao asserted on non-divisible in_features, which the quantize loop
        # caught, leaving the layer unquantized; refuse for the same end state
        return weight.shape[1] % self.group_size == 0

    @torch.no_grad()
    def quantize_(self, module: torch.nn.Linear, weight: torch.Tensor) -> None:
        # weight arrives in its original dtype (wants_fp32_weight is False)
        self._quantize_impl(module, weight)

    @torch.no_grad()
    def requantize_(self, module: "OstrisLinear", fp_weight: torch.Tensor) -> None:
        # the torchao path cast the merged weight to the original dtype and
        # re-quantized in that dtype
        self._quantize_impl(module, fp_weight.to(module.ostris_orig_dtype))

    def _quantize_impl(self, module: torch.nn.Module, w: torch.Tensor) -> None:
        out_f, in_f = w.shape
        groups = in_f // self.group_size
        wv = w.contiguous().view(out_f, groups, self.group_size)

        min_val = torch.amin(wv, dim=2)
        max_val = torch.amax(wv, dim=2)
        # preserve_zero: the qparams must be able to represent 0.0 exactly
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(self.qmax - self.qmin)
        scale = torch.clamp(scale, min=_EPS)
        zero_point = self.qmin - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax).to(torch.int32)

        q = torch.clamp(
            torch.round(wv * (1.0 / scale.view(out_f, groups, 1)))
            + zero_point.view(out_f, groups, 1),
            self.qmin,
            self.qmax,
        )
        q = q.view(out_f, in_f).to(torch.uint8)

        module.register_buffer(
            "uintx_packed", pack_uintx(q, self.nbits), persistent=False
        )
        module.register_buffer(
            "uintx_scale", scale.contiguous().view(torch.uint8), persistent=False
        )
        module.register_buffer(
            "uintx_zero_point", zero_point.to(torch.uint8), persistent=False
        )

    def _dequantize_native(self, module: "OstrisLinear") -> torch.Tensor:
        """Reconstruct the weight in its original dtype. torchao subtracted the
        zero point in int32 then cast; doing it directly in the weight dtype is
        bit-identical (codes and zero points are integers <= 255, and every
        integer in [-255, 255] is exact in bf16/fp16/fp32) with half the
        intermediate memory traffic."""
        out_f, in_f = module.out_features, module.in_features
        groups = in_f // self.group_size
        scale = module.uintx_scale.view(module.ostris_orig_dtype)
        q = unpack_uintx(module.uintx_packed, self.nbits, out_f * in_f)
        dq = q.view(out_f, groups, self.group_size).to(scale.dtype)
        dq -= module.uintx_zero_point.to(scale.dtype).view(out_f, groups, 1)
        dq *= scale.view(out_f, groups, 1)
        return dq.view(out_f, in_f)

    def dequantize(self, module: "OstrisLinear") -> torch.Tensor:
        return self._dequantize_native(module).to(torch.float32)

    def forward(self, module: "OstrisLinear", x: torch.Tensor) -> torch.Tensor:
        # skip the float32 round-trip of the base implementation; the weight is
        # frozen, so build it outside autograd
        with torch.no_grad():
            w = self._dequantize_native(module)
            if w.dtype != x.dtype:
                w = w.to(x.dtype)
        return torch.nn.functional.linear(x, w, module.bias)
