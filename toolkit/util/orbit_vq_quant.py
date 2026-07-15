"""
OrbitVQ weight-only quantization backend (orbitvq2 / orbitvq3 / orbitvq4 qtypes).

Extends the OrbitQuant recipe (toolkit/util/orbit_quant.py) with three accuracy
upgrades aimed at low bit-widths, at the cost of no longer being the paper's pure
scalar method:

  1. Lattice vector codebooks. Groups of coordinates are quantized jointly against a
     codebook built from the densest lattice for the group dimension (E8 for 8-dim
     groups at 2 bits, D4 for 4-dim groups at 3/4 bits) instead of one coordinate at
     a time. A scalar codebook wastes indices on corner combinations that iid
     gaussian coordinates never produce; a lattice codebook spends all its codewords
     on the spherical shell where rotated weights actually live.
  2. Per-group scales. One scale per GROUP_SIZE rotated coordinates (instead of one
     norm per row) absorbs the finite-sample energy fluctuation between groups.
  3. Least-squares scale refit. After codes are chosen the scale is refit to the
     MSE-optimal value <w, c> / <c, c>.

Everything stays data-free and deterministic: the same seeded RPBH rotation as
OrbitQuant, codebooks that are fixed mathematical objects (lattice points sorted by
norm), and precomputed distortion-optimal lattice scales for the N(0,1) source.

Encoding uses the closed-form nearest-lattice-point algorithm plus a hash lookup
into the truncated codebook; the rare vectors whose nearest lattice point falls
outside the codebook fall back to an exact brute-force search.

Bits per param: index bits exactly (2/3/4) + 16-bit group scales / GROUP_SIZE
(+0.125 at the default 128).

Quantized state attached to each module:
  ovq_packed     packed codeword indices (uint8 bitstream, index_bits per vector)
  ovq_scales     per (row, group) least-squares scales (original dtype)
  ovq_perm / ovq_inv_perm / ovq_signs   shared RPBH rotation for in_features
  ovq_block / ovq_group                 Hadamard block size and group size
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from toolkit.print import print_acc
from toolkit.util.ostris_quant import OstrisQuantizer
from toolkit.util.orbit_quant import (
    MIN_HADAMARD_BLOCK,
    hadamard_block_size,
    rpbh_forward,
    rpbh_inverse,
    rpbh_params,
)

# qtype name -> quantizer constructor kwargs
ORBIT_VQ_QTYPES = {
    "orbitvq2": {"bits": 2, "vec_dim": 8, "lattice": "E8", "codebook_size": 2 ** 16},
    "orbitvq3": {"bits": 3, "vec_dim": 4, "lattice": "D4", "codebook_size": 2 ** 12},
    "orbitvq4": {"bits": 4, "vec_dim": 4, "lattice": "D4", "codebook_size": 2 ** 16},
}

# coordinates per stored scale (the group-norm granularity)
GROUP_SIZE = 128

# encode -> least-squares scale refit rounds. re-encoding with the refit scale
# (rounds=2) measured only 0.0005 lower relative error at 2-bit while doubling the
# encode cost, so one round is the default
LS_REFIT_ROUNDS = 1

# distortion-optimal lattice scale for a unit-variance gaussian source, per
# (lattice, codebook_size), found by an exact nearest-codeword sweep over a seeded
# 60k gaussian sample. per-coordinate MSE at these scales (vs scalar Lloyd-Max):
#   E8-65536  2-bit: 0.0916 (0.1175)   D4-4096  3-bit: 0.0301 (0.0345)
#   D4-65536  4-bit: 0.0089 (0.0095)
# fixed constants so codebooks are bit-identical everywhere.
BETA = {
    ("E8", 2 ** 16): 0.9800,
    ("D4", 2 ** 12): 0.4722,
    ("D4", 2 ** 16): 0.2617,
}

# key packing for the hash lookup: doubled coordinates + _KEY_OFFSET must fit in
# _KEY_BITS bits per dimension (covers every codebook point of the sizes above)
_KEY_BITS = 6
_KEY_OFFSET = 32

_skip_warned = set()
_master_tables: Dict[Tuple[str, int], "_VQTables"] = {}
_device_tables: Dict[Tuple[str, int, str], "_VQTables"] = {}


def enumerate_lattice_codebook(lattice: str, size: int) -> torch.Tensor:
    """The `size` lattice points closest to the origin (ties broken lexicographically,
    so the result is fully deterministic), as float32 (size, dim), in lattice units.

    D4 = {x in Z^4 : sum(x) even} (densest 4-dim packing).
    E8 = D8 union (D8 + 1/2) (densest 8-dim packing). In doubled coordinates both
    become: uniform-parity integer vectors with sum divisible by 4.
    """
    if lattice == "D4":
        dim, reach = 4, 27  # doubled coords in [-26, 26], covers 65536 points
        vals = torch.arange(-(reach - 1), reach, 2, dtype=torch.int32)  # even only
        parities = [vals]
    elif lattice == "E8":
        dim = 8
        # doubled coords: even in [-6, 6] (integer points, norm^2 <= 12) and odd in
        # [-5, 5] (half points); both ranges cover far more than 65536 points
        parities = [
            torch.arange(-6, 7, 2, dtype=torch.int32),
            torch.arange(-5, 6, 2, dtype=torch.int32),
        ]
    else:
        raise ValueError(f"unknown lattice {lattice}")

    kept = []
    for vals in parities:
        # chunk over the first coordinate to keep the cartesian product small
        for v0 in vals.tolist():
            rest = torch.cartesian_prod(*([vals] * (dim - 1)))
            pts = torch.cat(
                [torch.full((rest.shape[0], 1), v0, dtype=torch.int32), rest], dim=1
            )
            pts = pts[pts.sum(dim=1).remainder(4) == 0]
            norm2 = (pts.to(torch.int64) ** 2).sum(dim=1)
            # generous radius cut just to bound memory before the exact sort below
            keep = norm2 <= (48 if lattice == "E8" else 26 ** 2 + 1)
            kept.append(pts[keep])
    pts = torch.cat(kept)

    # sort by (norm, lexicographic key) and take the closest `size` points
    norm2 = (pts.to(torch.int64) ** 2).sum(dim=1)
    key = _point_keys(pts)
    order = torch.argsort(norm2 * (1 << (_KEY_BITS * dim)) + key)
    pts = pts[order[:size]]
    if pts.shape[0] < size:
        raise RuntimeError(f"lattice enumeration too small for {lattice}/{size}")
    return pts.to(torch.float32) / 2.0  # back to lattice units


def _point_keys(doubled_pts: torch.Tensor) -> torch.Tensor:
    """int64 hash key per point from doubled integer coordinates."""
    digits = doubled_pts.to(torch.int64) + _KEY_OFFSET
    key = torch.zeros(doubled_pts.shape[0], dtype=torch.int64, device=doubled_pts.device)
    for i in range(doubled_pts.shape[1]):
        key = key | (digits[:, i] << (_KEY_BITS * i))
    return key


def _round_Dn(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of D_n (integer vectors with even coordinate sum) to each row."""
    f = x.round()
    odd = f.to(torch.int64).sum(dim=-1).remainder(2) != 0
    err = x - f
    idx = err.abs().argmax(dim=-1, keepdim=True)
    step = torch.where(err.gather(-1, idx) >= 0, 1.0, -1.0).to(x.dtype)
    adjusted = f.scatter(-1, idx, f.gather(-1, idx) + step)
    return torch.where(odd.unsqueeze(-1), adjusted, f)


def _round_lattice(x: torch.Tensor, lattice: str) -> torch.Tensor:
    """Closed-form nearest lattice point (Conway & Sloane) to each row of x."""
    a = _round_Dn(x)
    if lattice == "D4":
        return a
    # E8 = D8 union (D8 + 1/2): take the closer of the two cosets
    b = _round_Dn(x - 0.5) + 0.5
    da = (x - a).square().sum(dim=-1)
    db = (x - b).square().sum(dim=-1)
    return torch.where((da <= db).unsqueeze(-1), a, b)


class _VQTables:
    """Codebook + hash tables for one (lattice, codebook_size), on one device."""

    def __init__(self, lattice: str, size: int):
        self.lattice = lattice
        self.size = size
        self.beta = BETA[(lattice, size)]
        points = enumerate_lattice_codebook(lattice, size)
        self.codebook = points * self.beta  # (size, dim) float32, source units
        keys = _point_keys((points * 2).to(torch.int32))
        self.sorted_keys, order = torch.sort(keys)
        self.key_to_index = order.to(torch.int32)
        # for the brute-force fallback: argmax(z.c - |c|^2/2) == nearest codeword
        self.half_sq_norms = self.codebook.square().sum(dim=1) / 2
        self.codebook_t = self.codebook.T.contiguous()

    def to(self, device: torch.device) -> "_VQTables":
        out = object.__new__(_VQTables)
        out.lattice, out.size, out.beta = self.lattice, self.size, self.beta
        out.codebook = self.codebook.to(device)
        out.sorted_keys = self.sorted_keys.to(device)
        out.key_to_index = self.key_to_index.to(device)
        out.half_sq_norms = self.half_sq_norms.to(device)
        out.codebook_t = self.codebook_t.to(device)
        return out


def get_vq_tables(lattice: str, size: int, device) -> _VQTables:
    mkey = (lattice, size)
    if mkey not in _master_tables:
        _master_tables[mkey] = _VQTables(lattice, size)
    dkey = (lattice, size, str(device))
    if dkey not in _device_tables:
        _device_tables[dkey] = _master_tables[mkey].to(torch.device(device))
    return _device_tables[dkey]


def encode_vectors(z: torch.Tensor, tables: _VQTables) -> torch.Tensor:
    """Exact nearest-codeword indices (int32) for rows of z (float32, source units).

    Closed-form lattice rounding + hash lookup; rows whose nearest lattice point is
    outside the truncated codebook fall back to a brute-force search.
    """
    dim = z.shape[-1]
    p = _round_lattice(z / tables.beta, tables.lattice)
    digits = (p * 2).round().to(torch.int64) + _KEY_OFFSET
    in_range = ((digits >= 0) & (digits < (1 << _KEY_BITS))).all(dim=-1)
    key = torch.zeros(z.shape[0], dtype=torch.int64, device=z.device)
    for i in range(dim):
        key = key | (digits[:, i].clamp(0, (1 << _KEY_BITS) - 1) << (_KEY_BITS * i))
    pos = torch.searchsorted(tables.sorted_keys, key).clamp(max=tables.size - 1)
    hit = in_range & (tables.sorted_keys.gather(0, pos) == key)
    idx = tables.key_to_index.gather(0, pos.to(torch.int64))

    miss = ~hit
    n_miss = int(miss.sum())
    if n_miss > 0:
        # on cuda run the search in fp16 (tensor cores + half the score-matrix
        # traffic); the score gap between competing codewords for these overload
        # vectors is far above fp16 noise. cpu stays fp32.
        dt = torch.float16 if z.device.type == "cuda" else torch.float32
        z_miss = z[miss].to(dt)
        cb_t = tables.codebook_t.to(dt)
        half_norms = tables.half_sq_norms.to(dt)
        found = torch.empty(n_miss, dtype=torch.int32, device=z.device)
        # chunk rows so the score matrix stays ~256MB while each matmul stays large
        # enough to saturate the gpu (overload vectors are ~15-20% of the total)
        chunk = max(256, (2 ** 26) // tables.size)
        for s in range(0, n_miss, chunk):
            scores = z_miss[s:s + chunk] @ cb_t - half_norms
            found[s:s + chunk] = scores.argmax(dim=1).to(torch.int32)
        idx[miss] = found
    return idx


def pack_indices(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer indices (< 2**bits, bits <= 16) into a flat uint8 bitstream."""
    flat = idx.flatten().to(torch.int32)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    shifts = torch.arange(bits - 1, -1, -1, device=flat.device, dtype=torch.int32)
    bit_mat = ((flat.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)  # (n, bits)
    byte_mat = bit_mat.view(-1, 8)
    weights = torch.tensor([1 << i for i in range(7, -1, -1)], device=flat.device, dtype=torch.uint8)
    return (byte_mat * weights).sum(-1, dtype=torch.uint8)


def unpack_indices(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Inverse of pack_indices. Returns a flat int32 tensor of length numel."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bit_mat = ((packed.unsqueeze(-1) >> shifts) & 1).view(-1, bits)  # uint8
    idx = None
    # accumulate in <=8-bit chunks so intermediates stay uint8
    for c0 in range(0, bits, 8):
        cw = min(8, bits - c0)
        w = torch.tensor([1 << i for i in range(cw - 1, -1, -1)], device=packed.device, dtype=torch.uint8)
        part = (bit_mat[:, c0:c0 + cw] * w).sum(-1, dtype=torch.uint8).to(torch.int32)
        part = part << (bits - c0 - cw)
        idx = part if idx is None else idx | part
    return idx[:numel]


class OrbitVQQuantizer(OstrisQuantizer):
    """RPBH rotation + lattice vector codebook + per-group least-squares scales.
    One instance per qtype, shareable across modules."""

    def __init__(self, bits: int, vec_dim: int, lattice: str, codebook_size: int,
                 group_size: int = GROUP_SIZE):
        self.bits = bits
        self.vec_dim = vec_dim
        self.lattice = lattice
        self.codebook_size = codebook_size
        self.group_size = group_size
        self.index_bits = bits * vec_dim

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        d = module.in_features
        h = hadamard_block_size(d)
        if h < MIN_HADAMARD_BLOCK:
            if d not in _skip_warned:
                _skip_warned.add(d)
                print_acc(
                    f"OrbitVQ: skipping linears with in_features={d} "
                    f"(power-of-two block {h} is too small for the rotation)"
                )
            return False
        return True

    def _encode_rotated(self, w_rot: torch.Tensor, g: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a rotated weight (m, d) in float32. Returns (packed indices,
        float32 (m, d//g) group scales)."""
        tables = get_vq_tables(self.lattice, self.codebook_size, w_rot.device)
        m, d = w_rot.shape
        u = w_rot.view(m, d // g, g)
        # initial scale: per-group rms, so standardized coords are ~N(0, 1)
        scale = u.norm(dim=-1, keepdim=True) / g ** 0.5 + 1e-12
        idx = None
        for _ in range(LS_REFIT_ROUNDS):
            z = (u / scale).reshape(-1, self.vec_dim)
            idx = encode_vectors(z, tables)
            c = tables.codebook.index_select(0, idx).view(m, d // g, g)
            # least-squares optimal scale given the chosen codewords
            num = (u * c).sum(dim=-1, keepdim=True)
            den = c.square().sum(dim=-1, keepdim=True) + 1e-12
            scale = num / den
        return pack_indices(idx, self.index_bits), scale.view(m, d // g)

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        d = module.in_features
        h = hadamard_block_size(d)
        g = min(self.group_size, h)
        device = weight_fp32.device
        perm_cpu, signs_cpu = rpbh_params(d)
        perm = perm_cpu.to(device=device, dtype=torch.int32)
        inv_perm = torch.argsort(perm_cpu).to(device=device, dtype=torch.int32)
        signs = signs_cpu.to(device)
        w_rot = rpbh_forward(weight_fp32, perm, signs, h)
        packed, scales = self._encode_rotated(w_rot, g)
        module.register_buffer("ovq_packed", packed, persistent=False)
        module.register_buffer("ovq_scales", scales.to(module.weight.dtype), persistent=False)
        module.register_buffer("ovq_perm", perm, persistent=False)
        module.register_buffer("ovq_inv_perm", inv_perm, persistent=False)
        module.register_buffer("ovq_signs", signs, persistent=False)
        module.ovq_block = h
        module.ovq_group = g

    def _dequantize_rotated(self, module, dtype: torch.dtype) -> torch.Tensor:
        """Materialize the rotated-basis weight W' = W P^T in the given dtype."""
        tables = get_vq_tables(self.lattice, self.codebook_size, module.ovq_packed.device)
        m, d = module.out_features, module.in_features
        g = module.ovq_group
        idx = unpack_indices(module.ovq_packed, self.index_bits, m * d // self.vec_dim)
        w = tables.codebook.to(dtype).index_select(0, idx).view(m, d // g, g)
        w = w * module.ovq_scales.to(dtype).unsqueeze(-1)
        return w.view(m, d)

    def dequantize(self, module) -> torch.Tensor:
        w = self._dequantize_rotated(module, torch.float32)
        return rpbh_inverse(w, module.ovq_inv_perm, module.ovq_signs, module.ovq_block)

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.ovq_packed.device, dtype=torch.float32)
        w_rot = rpbh_forward(w, module.ovq_perm, module.ovq_signs, module.ovq_block)
        packed, scales = self._encode_rotated(w_rot, module.ovq_group)
        module.ovq_packed = packed
        module.ovq_scales = scales.to(module.ostris_orig_dtype)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        # rotate the activation instead of un-rotating the weight; the rotations
        # cancel in the matmul. the weight is frozen, so build it outside autograd;
        # gradients still flow to x through the rotation and the matmul
        with torch.no_grad():
            w = self._dequantize_rotated(module, x.dtype)
        x_rot = rpbh_forward(x, module.ovq_perm, module.ovq_signs, module.ovq_block)
        return F.linear(x_rot, w, module.bias)
