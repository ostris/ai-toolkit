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
    # get_ostris_quantizer); quantized saves need it to restore the backend
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


_wrong_device_warned = False


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
        if x.is_cuda and not hasattr(self, "_layer_memory_manager"):
            # a module left behind on the wrong device (usually cpu after a
            # low_vram load) would run its dequant/matmul on cpu threads and
            # silently hammer the cpu. move it to the input's device once.
            # memory-managed modules are excluded: they keep cpu buffers by
            # design and stage them to the gpu per forward.
            buf = next((b for b in self._buffers.values() if b is not None), None)
            if buf is not None and buf.device != x.device:
                global _wrong_device_warned
                if not _wrong_device_warned:
                    _wrong_device_warned = True
                    from toolkit.print import print_acc
                    print_acc(
                        f"OstrisLinear: quantized weights found on {buf.device} while the "
                        f"input is on {x.device}; moving them to {x.device}. This usually "
                        f"means something left the model behind after a low_vram load."
                    )
                self.to(x.device)
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
        # quantized saves read this back to restore the backend on load
        quantizer.qtype = qtype
    return quantizer


# metadata key shared with toolkit/models/classes/_mixin.py save_quantized
QUANT_LAYERS_METADATA_KEY = "aitk_quantization"


@torch.no_grad()
def save_quantized_layers(
    modules: Dict[str, "OstrisLinear"],
    file_path: str,
    metadata: Optional[Dict[str, str]] = None,
    extra_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Save a set of quantized linears (keyed by their submodule path in the
    target model) as a single safetensors file: backend buffers + bias per
    module, with the qtype/attrs needed to restore them recorded in the file
    metadata. Same layout as OstrisModelMixin.save_quantized, but partial —
    apply with load_quantized_layers."""
    import json
    from safetensors.torch import save_file

    quant_map = {}
    state_dict = {}
    for name, module in modules.items():
        if module.ostris_quantizer.qtype is None:
            raise ValueError(
                f"Cannot save quantized module '{name}': its quantizer has no "
                "qtype recorded (was it created through get_ostris_quantizer?)."
            )
        entry = {
            "qtype": module.ostris_quantizer.qtype,
            "dtype": str(module.ostris_orig_dtype).replace("torch.", ""),
            "buffers": [],
            "attrs": {},
        }
        if module.bias is not None:
            state_dict[f"{name}.bias"] = module.bias
        for buf_name, buf in module._buffers.items():
            if buf is None:
                continue
            state_dict[f"{name}.{buf_name}"] = buf
            entry["buffers"].append(buf_name)
        for attr, value in vars(module).items():
            if attr.startswith("_") or attr in (
                "training", "in_features", "out_features",
                "ostris_quantizer", "ostris_orig_dtype",
            ):
                continue
            if isinstance(value, (bool, int, float, str)):
                entry["attrs"][attr] = value
        quant_map[name] = entry

    if extra_tensors:
        # training-state extras (e.g. QAT master weights); load_quantized_layers
        # ignores them, so deployment loads are unaffected
        state_dict.update(extra_tensors)
    state_dict = {
        k: v.detach().to("cpu", copy=True).contiguous() for k, v in state_dict.items()
    }
    meta = dict(metadata or {})
    meta[QUANT_LAYERS_METADATA_KEY] = json.dumps(
        {"modules": quant_map, "layers_only": True}
    )
    save_file(state_dict, file_path, metadata=meta)


@torch.no_grad()
def load_quantized_layers(root: torch.nn.Module, file_path: str) -> int:
    """Apply a file written by save_quantized_layers onto a model: restores the
    backend buffers (and bias) of each recorded module. Target modules may
    already be OstrisLinear (model quantized on load — buffers are replaced) or
    still plain nn.Linear (converted in place, no quantization math needed).
    Returns the number of modules restored."""
    import json
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(file_path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    if QUANT_LAYERS_METADATA_KEY not in meta:
        raise ValueError(f"{file_path} has no quantized-layer metadata")
    quant_map = json.loads(meta[QUANT_LAYERS_METADATA_KEY])["modules"]
    state_dict = load_file(file_path)

    for name, entry in quant_map.items():
        module = root.get_submodule(name)
        quantizer = get_ostris_quantizer(entry["qtype"])
        if quantizer is None:
            raise ValueError(f"Unknown qtype '{entry['qtype']}' in {file_path}")
        # figure out the device the module currently lives on
        ref = next(
            (t for t in module._parameters.values() if t is not None),
            next((t for t in module._buffers.values() if t is not None), None),
        )
        device = ref.device if ref is not None else torch.device("cpu")
        if isinstance(module, OstrisLinear):
            # replacing an existing quantized state (possibly another backend):
            # its buffers are exclusively backend state, drop them all
            module._buffers.clear()
        else:
            if "weight" in module._parameters:
                del module._parameters["weight"]
            module.ostris_orig_dtype = getattr(torch, entry["dtype"])
            module.__class__ = OstrisLinear
        for buf_name in entry["buffers"]:
            module.register_buffer(
                buf_name,
                state_dict.pop(f"{name}.{buf_name}").to(device),
                persistent=False,
            )
        for attr, value in entry.get("attrs", {}).items():
            setattr(module, attr, value)
        module.ostris_quantizer = quantizer
        bias_key = f"{name}.bias"
        if bias_key in state_dict and module.bias is not None:
            module.bias.data.copy_(
                state_dict.pop(bias_key).to(device, module.bias.dtype)
            )
        if module.bias is not None:
            module.bias.requires_grad_(False)
    return len(quant_map)


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
