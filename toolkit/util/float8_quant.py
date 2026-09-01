"""ComfyUI-style float8 weight storage as an Ostris backend.

Matches the comfy_quant ``{"format": "float8_e4m3fn",
"full_precision_matrix_mult": true}`` layout: the weight stored as
torch.float8_e4m3fn plus one fp32 per-tensor scale, matmuls running on the
dequantized weight (W8A16 numerics). Used both to import comfy fp8/fp8-mixed
checkpoints and to quantize/export in that format.
"""

from typing import Optional

import torch

from toolkit.util.ostris_quant import OstrisLinear, OstrisQuantizer

FLOAT8_QTYPES = ["float8_e4m3fn"]

F8_MAX = torch.finfo(torch.float8_e4m3fn).max


class Float8Quantizer(OstrisQuantizer):
    """fp8_e4m3 weight + fp32 per-tensor scale, dequantized matmul."""

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        scale = (weight_fp32.abs().max() / F8_MAX).clamp(min=1e-12)
        q = (weight_fp32 / scale).clamp(-F8_MAX, F8_MAX).to(torch.float8_e4m3fn)
        self.attach_(module, q, scale)

    @staticmethod
    def attach_(
        module: torch.nn.Module,
        qweight: torch.Tensor,  # float8_e4m3fn (out, in)
        scale: torch.Tensor,  # fp32 scalar
    ) -> None:
        """Register the quantized representation on the module. Used both by
        quantize_ and by importers of pre-quantized checkpoints."""
        module.register_buffer("f8_qdata", qweight.contiguous(), persistent=False)
        module.register_buffer(
            "f8_scale",
            scale.detach().float().clone().reshape(1).view(torch.uint8),
            persistent=False,
        )

    def dequantize(self, module: "OstrisLinear") -> torch.Tensor:
        scale = module.f8_scale.view(torch.float32)[0]
        return module.f8_qdata.to(torch.float32) * scale

    @torch.no_grad()
    def requantize_(self, module: "OstrisLinear", fp_weight: torch.Tensor) -> None:
        w = fp_weight.to(torch.float32)
        scale = (w.abs().max() / F8_MAX).clamp(min=1e-12)
        module.f8_qdata.copy_((w / scale).clamp(-F8_MAX, F8_MAX).to(torch.float8_e4m3fn))
        module.f8_scale.copy_(scale.reshape(1).view(torch.uint8))
