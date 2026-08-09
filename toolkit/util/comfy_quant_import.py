"""Import ComfyUI pre-quantized checkpoints onto toolkit modules.

ComfyUI quantized checkpoints mark each quantized submodule with a
``<prefix>.comfy_quant`` uint8 tensor holding a JSON config, alongside the
quantized ``weight`` and its scale tensors. This module walks those markers
and converts the matching submodules in place:

  - ``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": G}``
    per-output-row symmetric int8 on regular-Hadamard-rotated weights — the
    exact storage of the toolkit's convrot8 backend
    (toolkit/util/convrot_quant.py:ConvRotInt8Quantizer), so the tensors are
    attached to its buffers directly (no requantization). Without the
    ``convrot`` flag the rotation block is 1, i.e. plain per-row int8, which
    the same backend also decodes (rotate is the identity at rot_size 1).
  - ``{"format": "nvfp4"}`` block-16 fp4 with e4m3 block scales, an fp32
    per-tensor scale and an optional AWQ ``pre_quant_scale`` — attached to
    the nvfp4 backend (toolkit/util/nvfp4_quant.py).
  - an int8 marker on an ``nn.Embedding`` swaps in :class:`Int8Embedding`
    (per-row scales, dequantized per lookup).

Linears become OstrisLinear (class swap in place, like
convert_linear_to_ostris), so LoRA attachment, memory management and the
quantized save paths all work unchanged.
"""

import json
from typing import Dict, Tuple

import torch

from toolkit.util.nvfp4_quant import (
    Nvfp4Quantizer,
    swap_nvfp4_nibbles,
    unswizzle_nvfp4_scales,
)
from toolkit.util.ostris_quant import OstrisLinear, get_ostris_quantizer


def parse_comfy_quant_blob(blob: torch.Tensor) -> dict:
    return json.loads(bytes(blob.cpu().tolist()).decode("utf-8"))


class Int8Embedding(torch.nn.Module):
    """An embedding table stored as per-row symmetric int8. Rows are
    dequantized per lookup, so the full-precision table never materializes."""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype):
        super().__init__()
        self.num_embeddings, self.embedding_dim = qweight.shape
        self.output_dtype = dtype
        self.register_buffer("qweight", qweight.contiguous(), persistent=False)
        self.register_buffer(
            "scales",
            scales.detach().float().reshape(-1).contiguous().view(torch.uint8),
            persistent=False,
        )

    @property
    def weight(self):
        # full dequantized table, for code that inspects it
        scales = self.scales.view(torch.float32)
        return (self.qweight.float() * scales.unsqueeze(1)).to(self.output_dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # the table may stay CPU-resident under text-encoder offloading: run
        # the (tiny) lookup on the table's device, return on the caller's
        flat = input_ids.reshape(-1).to(self.qweight.device)
        rows = self.qweight.index_select(0, flat).float()
        scales = self.scales.view(torch.float32).index_select(0, flat)
        out = (rows * scales.unsqueeze(1)).to(self.output_dtype)
        return out.to(input_ids.device).reshape(*input_ids.shape, self.embedding_dim)


def _to_ostris(module: torch.nn.Linear, quantizer, orig_dtype: torch.dtype) -> OstrisLinear:
    if "weight" in module._parameters:
        del module._parameters["weight"]
    module.ostris_quantizer = quantizer
    module.ostris_orig_dtype = orig_dtype
    if module.bias is not None:
        module.bias.requires_grad_(False)
    module.__class__ = OstrisLinear
    return module


@torch.no_grad()
def import_comfy_quantized_layers(
    root: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    orig_dtype: torch.dtype = torch.bfloat16,
    key_map=None,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Convert every module a ``comfy_quant`` marker points at and attach its
    quantized tensors. Consumes the quantized entries from ``state_dict`` and
    returns ``(remaining_state_dict, num_converted)`` — load the remainder
    with the regular load_state_dict.

    ``key_map`` optionally maps a checkpoint prefix to the module path in
    ``root`` (e.g. comfy text encoder keys onto transformers module paths).
    """
    state_dict = dict(state_dict)
    converted = 0

    marker_keys = [k for k in state_dict.keys() if k.endswith(".comfy_quant")]
    for marker_key in marker_keys:
        prefix = marker_key[: -len(".comfy_quant")]
        conf = parse_comfy_quant_blob(state_dict.pop(marker_key))
        fmt = conf.get("format")
        module_path = key_map(prefix) if key_map is not None else prefix
        module = root.get_submodule(module_path)

        weight = state_dict.pop(f"{prefix}.weight")
        weight_scale = state_dict.pop(f"{prefix}.weight_scale", None)

        if isinstance(module, torch.nn.Embedding):
            if fmt != "int8_tensorwise":
                raise ValueError(
                    f"Unsupported comfy quant format {fmt!r} on embedding {prefix}"
                )
            parent_path, _, attr = module_path.rpartition(".")
            parent = root.get_submodule(parent_path) if parent_path else root
            setattr(parent, attr, Int8Embedding(weight, weight_scale, orig_dtype))
            converted += 1
            continue

        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"comfy_quant marker {prefix} points at {type(module).__name__}, "
                "expected nn.Linear or nn.Embedding"
            )

        if fmt == "int8_tensorwise":
            rot = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1
            quantizer = get_ostris_quantizer("convrot8")
            module.register_buffer("cr8_qdata", weight.contiguous(), persistent=False)
            module.register_buffer(
                "cr8_scales",
                weight_scale.detach().float().reshape(-1).contiguous().view(torch.uint8),
                persistent=False,
            )
            module.cr8_rot_size = rot
        elif fmt == "nvfp4":
            quantizer = get_ostris_quantizer("nvfp4")
            # normalize comfy_kitchen's storage to the toolkit's conventions:
            # fp4 pairs are packed high-nibble-first and the e4m3 block scales
            # are stored in the swizzled cuBLAS 128x4 tile layout
            scales = unswizzle_nvfp4_scales(
                weight_scale.view(torch.float8_e4m3fn),
                module.out_features,
                module.in_features // 16,
            )
            Nvfp4Quantizer.attach_(
                module,
                packed=swap_nvfp4_nibbles(weight),
                scales=scales,
                pts=state_dict.pop(f"{prefix}.weight_scale_2"),
                pre_scale=state_dict.pop(f"{prefix}.pre_quant_scale", None),
            )
        else:
            raise ValueError(
                f"Unsupported comfy quant format {fmt!r} on {prefix} "
                "(supported: int8_tensorwise, nvfp4)"
            )

        # drop unused calibration extras if present
        state_dict.pop(f"{prefix}.input_scale", None)

        _to_ostris(module, quantizer, orig_dtype)
        bias = state_dict.pop(f"{prefix}.bias", None)
        if bias is not None and module.bias is not None:
            # bias may still be a meta parameter when the model was built under
            # a meta device context
            module._parameters["bias"] = torch.nn.Parameter(
                bias.detach().clone(), requires_grad=False
            )
        converted += 1

    return state_dict, converted
