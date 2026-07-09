"""
OrbitQuant weight-only quantization backend (orbit2 / orbit3 / orbit4 qtypes).

Implements the weight half of "OrbitQuant: Data-Agnostic Quantization for Image and
Video Diffusion Transformers" (arXiv:2607.02461) as an OstrisQuantizer backend (see
toolkit/util/ostris_quant.py). Each linear weight is rotated with a randomized
permuted block-Hadamard (RPBH) rotation shared per input dimension, split into
per-row norms and unit directions, and the directions are quantized with a Lloyd-Max
codebook fit to the fixed post-rotation coordinate marginal N(0, 1/d).
No calibration data is needed.

At runtime the forward rotation is applied to the activations instead of un-rotating
the weight, so the two cancel in the matmul (rotate = multiply by P):

    W' = W P^T,   y = dequant(W') (P x) ~= W x

Activations are not quantized. This is meant for holding a frozen base model at low
bit-width while training adapters on top; activation quantization would only add
noise without fused low-bit kernels.

Quantized state attached to each module:
  orbit_packed     packed codebook indices (uint8 bitstream)
  orbit_row_norms  per output row l2 norms of the rotated weight (original dtype)
  orbit_codebook   Lloyd-Max centroids for N(0, 1/in_features) (float32)
  orbit_perm / orbit_inv_perm / orbit_signs   shared RPBH rotation for in_features
  orbit_bits / orbit_block                    bit-width and Hadamard block size
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from toolkit.print import print_acc
from toolkit.util.ostris_quant import OstrisQuantizer

# qtype name -> bits per weight coordinate
ORBIT_QTYPES = {"orbit2": 2, "orbit3": 3, "orbit4": 4}

# below this Hadamard block size the rotated coordinates are not gaussian enough for
# the shared N(0, 1/d) codebook to be valid, so conversion is skipped for the layer
MIN_HADAMARD_BLOCK = 32

_normal_codebook_cache: Dict[int, torch.Tensor] = {}
_rotation_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
_skip_warned = set()


def gaussian_lloyd_max(bits: int, iters: int = 200) -> torch.Tensor:
    """MSE-optimal (Lloyd-Max) centroids for the standard normal, 2**bits levels,
    returned in ascending order as float32. Cached per bit-width."""
    if bits in _normal_codebook_cache:
        return _normal_codebook_cache[bits]
    levels = 2 ** bits
    # init at the gaussian quantile midpoints
    q = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
    c = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
    inf = torch.tensor([math.inf], dtype=torch.float64)
    for _ in range(iters):
        edges = (c[:-1] + c[1:]) / 2.0
        lo = torch.cat([-inf, edges])
        hi = torch.cat([edges, inf])
        # centroid update: E[X | lo < X < hi] = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
        phi_lo = torch.exp(-lo * lo / 2.0) / math.sqrt(2.0 * math.pi)
        phi_hi = torch.exp(-hi * hi / 2.0) / math.sqrt(2.0 * math.pi)
        cdf_lo = 0.5 * (1.0 + torch.erf(lo / math.sqrt(2.0)))
        cdf_hi = 0.5 * (1.0 + torch.erf(hi / math.sqrt(2.0)))
        c = (phi_lo - phi_hi) / (cdf_hi - cdf_lo)
    c = c.to(torch.float32)
    _normal_codebook_cache[bits] = c
    return c


def hadamard_block_size(d: int) -> int:
    # largest power of two dividing d
    return d & (-d)


def rpbh_params(d: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permutation (int64) and Rademacher signs (int8) of the RPBH rotation for input
    dimension d, on cpu. Sampled once per dimension with a seed derived from d so the
    rotation is identical across layers, runs, and resumes."""
    if d in _rotation_cache:
        return _rotation_cache[d]
    g = torch.Generator().manual_seed(0x0EB17 + d)
    perm = torch.randperm(d, generator=g)
    signs = torch.randint(0, 2, (d,), generator=g, dtype=torch.int8) * 2 - 1
    _rotation_cache[d] = (perm, signs)
    return perm, signs


def _fwht(x: torch.Tensor, h: int) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform applied to each contiguous block of h
    coordinates along the last dimension (h must be a power of two dividing the dim)."""
    shape = x.shape
    x = x.reshape(-1, h)
    m = x.shape[0]
    step = 1
    while step < h:
        y = x.view(m, h // (2 * step), 2, step)
        x = torch.stack((y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]), dim=2).view(m, h)
        step *= 2
    return (x * h ** -0.5).view(shape)


def rpbh_forward(x: torch.Tensor, perm: torch.Tensor, signs: torch.Tensor, h: int) -> torch.Tensor:
    """y = blkdiag(H_h D) P x applied to the last dimension of x."""
    y = torch.index_select(x, -1, perm) * signs.to(x.dtype)
    return _fwht(y, h)


def rpbh_inverse(y: torch.Tensor, inv_perm: torch.Tensor, signs: torch.Tensor, h: int) -> torch.Tensor:
    """Inverse of rpbh_forward (the rotation is orthogonal, H is self-inverse)."""
    z = _fwht(y, h) * signs.to(y.dtype)
    return torch.index_select(z, -1, inv_perm)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes (values < 2**bits, any shape) into a flat uint8 bitstream."""
    flat = codes.flatten().to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    shifts = torch.arange(bits - 1, -1, -1, device=flat.device, dtype=torch.uint8)
    bit_mat = (flat.unsqueeze(-1) >> shifts) & 1  # (n, bits)
    byte_mat = bit_mat.view(-1, 8)
    weights = torch.tensor([1 << i for i in range(7, -1, -1)], device=flat.device, dtype=torch.uint8)
    return (byte_mat * weights).sum(-1, dtype=torch.uint8)


def unpack_codes(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Inverse of pack_codes. Returns a flat uint8 tensor of length numel."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bit_mat = ((packed.unsqueeze(-1) >> shifts) & 1).view(-1, bits)
    weights = torch.tensor([1 << i for i in range(bits - 1, -1, -1)], device=packed.device, dtype=torch.uint8)
    codes = (bit_mat * weights).sum(-1, dtype=torch.uint8)
    return codes[:numel]


@torch.no_grad()
def _quantize_rows(
    w_fp32: torch.Tensor,
    perm: torch.Tensor,
    signs: torch.Tensor,
    h: int,
    codebook: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate weight rows into the RPBH basis and quantize their unit directions
    against the codebook. Returns (packed codes, float32 row norms)."""
    w_rot = rpbh_forward(w_fp32, perm, signs, h)
    row_norms = w_rot.norm(dim=1)
    unit = w_rot / (row_norms + 1e-10).unsqueeze(1)
    edges = (codebook[:-1] + codebook[1:]) / 2
    codes = torch.bucketize(unit, edges, out_int32=True).to(torch.uint8)
    return pack_codes(codes, bits), row_norms


class OrbitQuantizer(OstrisQuantizer):
    """OrbitQuant backend. One instance per bit-width, shareable across modules."""

    def __init__(self, bits: int):
        self.bits = bits

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        d = module.in_features
        h = hadamard_block_size(d)
        if h < MIN_HADAMARD_BLOCK:
            if d not in _skip_warned:
                _skip_warned.add(d)
                print_acc(
                    f"OrbitQuant: skipping linears with in_features={d} "
                    f"(power-of-two block {h} is too small for the rotation)"
                )
            return False
        return True

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        d = module.in_features
        h = hadamard_block_size(d)
        device = weight_fp32.device
        perm_cpu, signs_cpu = rpbh_params(d)
        perm = perm_cpu.to(device=device, dtype=torch.int32)
        inv_perm = torch.argsort(perm_cpu).to(device=device, dtype=torch.int32)
        signs = signs_cpu.to(device)
        codebook = (gaussian_lloyd_max(self.bits) * d ** -0.5).to(device)
        packed, row_norms = _quantize_rows(weight_fp32, perm, signs, h, codebook, self.bits)
        module.register_buffer("orbit_packed", packed, persistent=False)
        module.register_buffer("orbit_row_norms", row_norms.to(module.weight.dtype), persistent=False)
        module.register_buffer("orbit_codebook", codebook, persistent=False)
        module.register_buffer("orbit_perm", perm, persistent=False)
        module.register_buffer("orbit_inv_perm", inv_perm, persistent=False)
        module.register_buffer("orbit_signs", signs, persistent=False)
        module.orbit_bits = self.bits
        module.orbit_block = h

    def _dequantize_rotated(self, module, dtype: torch.dtype) -> torch.Tensor:
        """Materialize the rotated-basis weight W' = W P^T in the given dtype."""
        numel = module.out_features * module.in_features
        codes = unpack_codes(module.orbit_packed, module.orbit_bits, numel)
        w = torch.index_select(module.orbit_codebook.to(dtype), 0, codes.to(torch.int32))
        w = w.view(module.out_features, module.in_features)
        return w * module.orbit_row_norms.to(dtype).unsqueeze(1)

    def dequantize(self, module) -> torch.Tensor:
        w = self._dequantize_rotated(module, torch.float32)
        return rpbh_inverse(w, module.orbit_inv_perm, module.orbit_signs, module.orbit_block)

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.orbit_packed.device, dtype=torch.float32)
        packed, row_norms = _quantize_rows(
            w, module.orbit_perm, module.orbit_signs, module.orbit_block,
            module.orbit_codebook, module.orbit_bits,
        )
        module.orbit_packed = packed
        module.orbit_row_norms = row_norms.to(module.ostris_orig_dtype)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        # rotate the activation instead of un-rotating the weight; the rotations
        # cancel in the matmul. the weight is frozen, so build it outside autograd;
        # gradients still flow to x through the rotation and the matmul
        with torch.no_grad():
            w = self._dequantize_rotated(module, x.dtype)
        x_rot = rpbh_forward(x, module.orbit_perm, module.orbit_signs, module.orbit_block)
        return F.linear(x_rot, w, module.bias)
