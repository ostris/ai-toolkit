"""NVFP4 (ModelOpt/ComfyUI-style) OstrisQuantizer backend — qtype "nvfp4".

Plain block-16 nvfp4 weight storage without ConvRot's rotation: fp4 e2m1
codes packed two per byte, one fp8 e4m3 scale per 16 elements, one fp32
per-tensor scale, plus an optional AWQ ``pre_quant_scale`` applied
elementwise to the input activation before the matmul (ModelOpt convention,
matching ComfyUI's quantized ops).

This is the layout ComfyUI checkpoints tagged ``{"format": "nvfp4"}`` carry
(e.g. the Comfy-Org MiniMax-H3 text encoder). Those exports set
``full_precision_matrix_mult`` — the activations are NOT fp4-quantized — so
the forward here is always the dequantized matmul in the activation's dtype:
weights stay at ~4.25 bits in memory and the math runs on any GPU (or CPU),
no Blackwell fp4 tensor cores required. The triton dequant kernel in
convrot_quant is used when available; pure torch otherwise.

Quantized state attached to each module (uint8 byte views, like the convrot
backends, so nn.Module._apply dtype casts can't corrupt them):
  nv4_qdata      packed e2m1 codes (uint8, out x in/2; low nibble = even column)
  nv4_scales     e4m3 block scales (out x in/16)
  nv4_pts        fp32 per-tensor scale (1 element)
  nv4_pre_scale  optional fp32 AWQ input scale (in,)
"""

from typing import Optional

import torch
import torch.nn.functional as F

from toolkit.util.convrot_quant import BLOCK, dequantize_nvfp4, quantize_nvfp4
from toolkit.util.ostris_quant import OstrisQuantizer
from toolkit.print import print_acc

NVFP4_QTYPES = ("nvfp4",)

_skip_warned = set()


def unswizzle_nvfp4_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo the cuBLAS 128x4-tile block-scale layout (comfy_kitchen's
    ``to_blocked``) back to a row-major (rows, cols) matrix. ComfyUI nvfp4
    checkpoints store ``weight_scale`` swizzled; when the dims are already
    tile-aligned the shape is unchanged and only the element order differs."""
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    x = scales.reshape(-1, 32, 16)
    x = x.reshape(-1, 32, 4, 4).transpose(1, 2)
    x = x.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    x = x.reshape(n_row_blocks, n_col_blocks, 128, 4)
    x = x.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return x[:rows, :cols].contiguous()


def swap_nvfp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """ComfyUI packs fp4 pairs high-nibble-first; the toolkit's decode is
    low-nibble-first. Swapping nibbles converts between the two."""
    return ((packed << 4) | (packed >> 4)).contiguous()


class Nvfp4Quantizer(OstrisQuantizer):
    """Block-16 nvfp4 weights, full-precision activations. One instance is
    shareable across modules."""

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        if module.in_features % BLOCK != 0:
            if module.in_features not in _skip_warned:
                _skip_warned.add(module.in_features)
                print_acc(
                    f"nvfp4: skipping linears with in_features={module.in_features} "
                    f"(needs in divisible by {BLOCK})"
                )
            return False
        return True

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        packed, scales, pts = quantize_nvfp4(weight_fp32, optimize_scales=True)
        self.attach_(module, packed, scales, pts, pre_scale=None)

    @staticmethod
    def attach_(
        module: torch.nn.Module,
        packed: torch.Tensor,  # uint8 (out, in/2)
        scales: torch.Tensor,  # float8_e4m3fn (out, in/16)
        pts: torch.Tensor,  # fp32 scalar per-tensor scale
        pre_scale: Optional[torch.Tensor] = None,  # (in,) AWQ input scale
    ) -> None:
        """Register the quantized representation on the module. Used both by
        quantize_ and by importers of pre-quantized checkpoints."""
        module.register_buffer("nv4_qdata", packed.contiguous(), persistent=False)
        module.register_buffer(
            "nv4_scales", scales.contiguous().view(torch.uint8), persistent=False
        )
        module.register_buffer(
            "nv4_pts",
            pts.detach().float().clone().reshape(1).view(torch.uint8),
            persistent=False,
        )
        if pre_scale is not None:
            module.register_buffer(
                "nv4_pre_scale",
                pre_scale.detach().float().clone().contiguous().view(torch.uint8),
                persistent=False,
            )

    @staticmethod
    def _pts(module) -> torch.Tensor:
        return module.nv4_pts.view(torch.float32).reshape(())

    @staticmethod
    def _pre_scale(module) -> Optional[torch.Tensor]:
        buf = getattr(module, "nv4_pre_scale", None)
        return None if buf is None else buf.view(torch.float32)

    def _dequantize_weight(self, module, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_nvfp4(
            module.nv4_qdata,
            module.nv4_scales.view(torch.float8_e4m3fn),
            self._pts(module),
            module.out_features,
            module.in_features,
            dtype,
        )

    def dequantize(self, module) -> torch.Tensor:
        """The weight as stored. NOTE: with an AWQ pre_quant_scale present the
        stored weight expects pre-scaled activations; folding the scale back
        (w * pre_scale per column) would reconstruct the original-basis weight
        but is deliberately not done here — forward() owns that contract."""
        return self._dequantize_weight(module, torch.float32)

    def dequantize_folded(self, module) -> torch.Tensor:
        """Weight for raw (un-pre-scaled) activations: the AWQ pre_quant_scale
        multiplies the input elementwise, which folds into the weight columns."""
        w = self._dequantize_weight(module, torch.float32)
        pre_scale = self._pre_scale(module)
        if pre_scale is not None:
            w = w * pre_scale.unsqueeze(0)
        return w

    def requantize_(self, module, fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(device=module.nv4_qdata.device, dtype=torch.float32)
        packed, scales, pts = quantize_nvfp4(w, optimize_scales=True)
        module.nv4_qdata = packed
        module.nv4_scales = scales.view(torch.uint8)
        module.nv4_pts = pts.detach().clone().reshape(1).view(torch.uint8)

    def forward(self, module, x: torch.Tensor) -> torch.Tensor:
        pre_scale = self._pre_scale(module)
        if pre_scale is not None:
            x = x * pre_scale.to(dtype=x.dtype)
        with torch.no_grad():
            w = self._dequantize_weight(module, x.dtype)
        return F.linear(x, w, module.bias)
