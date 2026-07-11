"""
Quantization-agnostic custom quantized linear.

OstrisLinear is a drop-in nn.Linear replacement whose weight is held by a pluggable
quantizer backend (OstrisQuantizer). Backends own the quantized representation
(buffers + per-module attributes) and how the forward pass computes W x from it; the
module and the rest of the toolkit stay backend agnostic. The first backend is
OrbitQuant (toolkit/util/orbit_quant.py) via the orbit2/orbit3/orbit4 qtypes; add new
backends by implementing OstrisQuantizer and resolving them in get_ostris_quantizer.

Modules are converted in place by convert_linear_to_ostris via class swap, so the
original module object (and any references to it, e.g. LoRA org_module or parent
containers) stays valid.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class OstrisQuantizer:
    """Base class for weight quantization backends used by OstrisLinear.

    Backends are stateless with respect to tensors: everything tensor-shaped must be
    registered as a buffer on the module inside quantize_ (so device moves and dtype
    casts through nn.Module._apply keep working), and read back off the module in the
    other methods. One backend instance may be shared by many modules.
    """

    # the qtype string this instance was resolved from (stamped by
    # get_ostris_quantizer); pre-quantized saves need it to restore the backend
    qtype: Optional[str] = None

    def can_quantize(self, module: torch.nn.Linear) -> bool:
        """Whether this backend can quantize the given linear (e.g. shape constraints)."""
        return True

    def quantize_(self, module: torch.nn.Linear, weight_fp32: torch.Tensor) -> None:
        """Build the quantized representation of weight_fp32 and attach it to the
        module (register_buffer for tensors, plain attributes for scalars). Called
        while the module is still an nn.Linear, before the weight param is removed."""
        raise NotImplementedError

    def dequantize(self, module: "OstrisLinear") -> torch.Tensor:
        """Reconstruct the full weight in the original basis, in float32."""
        raise NotImplementedError

    def requantize_(self, module: "OstrisLinear", fp_weight: torch.Tensor) -> None:
        """Re-quantize in place from a full precision weight in the original basis
        (used by the continuous merge/reset method)."""
        raise NotImplementedError

    def forward(self, module: "OstrisLinear", x: torch.Tensor) -> torch.Tensor:
        # default: dequantize per forward and run a plain linear. backends can
        # override with a cheaper formulation. the weight is frozen, so build it
        # outside autograd; gradients still flow to x through the matmul
        with torch.no_grad():
            w = self.dequantize(module).to(x.dtype)
        return F.linear(x, w, module.bias)


class OstrisLinear(torch.nn.Linear):
    """A linear layer whose weight is quantized by an OstrisQuantizer backend.

    Never instantiate directly: created in place by convert_linear_to_ostris. The
    weight parameter is removed; the quantized representation lives in backend-owned
    buffers, plus:
      ostris_quantizer   the backend instance
      ostris_orig_dtype  dtype of the original weight (used for dequantized views)
    """

    is_ostris_quantized = True

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        """Reconstruct the weight in the original basis and dtype."""
        return self.ostris_quantizer.dequantize(self).to(self.ostris_orig_dtype)

    @property
    def weight(self):
        # materializes the full dequantized weight. kept for code that inspects the
        # weight (shape/dtype/device) and for the network merge paths, which detect
        # the marker via toolkit.util.quantize.is_quantized_tensor
        w = self.dequantize_weight()
        w._is_ostris_weight = True
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ostris_quantizer.forward(self, x)

    @torch.no_grad()
    def requantize_(self, fp_weight: torch.Tensor) -> None:
        self.ostris_quantizer.requantize_(self, fp_weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # emit a plain full precision weight so full-model saves need no special casing
        destination[prefix + "weight"] = self.dequantize_weight()
        if self.bias is not None:
            destination[prefix + "bias"] = (
                self.bias if keep_vars else self.bias.detach()
            )


def get_ostris_quantizer(qtype: str) -> Optional[OstrisQuantizer]:
    """Resolve a qtype string to a quantizer backend instance, or None if the qtype
    does not belong to a custom backend. Add new backends here."""
    from toolkit.util.orbit_quant import ORBIT_QTYPES, OrbitQuantizer
    from toolkit.util.orbit_vq_quant import ORBIT_VQ_QTYPES, OrbitVQQuantizer
    from toolkit.util.convrot_quant import CONVROT_QTYPES, get_convrot_quantizer

    quantizer = None
    if qtype in ORBIT_QTYPES:
        quantizer = OrbitQuantizer(ORBIT_QTYPES[qtype])
    elif qtype in ORBIT_VQ_QTYPES:
        quantizer = OrbitVQQuantizer(**ORBIT_VQ_QTYPES[qtype])
    elif qtype in CONVROT_QTYPES:
        quantizer = get_convrot_quantizer(qtype)
    if quantizer is not None:
        quantizer.qtype = qtype
    return quantizer


@torch.no_grad()
def convert_linear_to_ostris(
    module: torch.nn.Linear, quantizer: OstrisQuantizer
) -> bool:
    """Quantize an nn.Linear in place (class swap). Returns True if the module was
    converted (or already was), False if it is not a candidate."""
    if isinstance(module, OstrisLinear):
        return True
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or not weight.dtype.is_floating_point:
        return False
    if type(weight.data) is not torch.Tensor:
        # already holds a quantized tensor subclass (e.g. torchao)
        return False
    if not quantizer.can_quantize(module):
        return False
    quantizer.quantize_(module, weight.data.to(torch.float32))
    module.ostris_quantizer = quantizer
    module.ostris_orig_dtype = weight.dtype
    del module._parameters["weight"]
    if module.bias is not None:
        module.bias.requires_grad_(False)
    module.__class__ = OstrisLinear
    return True
