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


@torch.no_grad()
def split_fused_quantized_keys(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    dst_prefixes,
) -> Dict[str, torch.Tensor]:
    """Split one fused quantized comfy entry (``<prefix>.weight`` /
    ``.weight_scale`` / ``.comfy_quant`` / ...) into equal row ranges under
    ``dst_prefixes`` (out-dim concat order). Exact for every supported format:
    int8 rows and their per-row scales slice; fp8's per-tensor scale and
    nvfp4's weight_scale_2 / pre_quant_scale are shared by every split; nvfp4
    block scales are unswizzled, row-split, and re-swizzled. Mutates and
    returns state_dict. Used by classes whose module layout splits a fused
    checkpoint projection (e.g. qkv -> to_q/to_k/to_v)."""
    from toolkit.util.nvfp4_quant import swizzle_nvfp4_scales

    marker = state_dict.pop(f"{prefix}.comfy_quant")
    conf = parse_comfy_quant_blob(marker)
    fmt = conf.get("format")

    weight = state_dict.pop(f"{prefix}.weight")
    scale = state_dict.pop(f"{prefix}.weight_scale", None)
    pts = state_dict.pop(f"{prefix}.weight_scale_2", None)
    pre = state_dict.pop(f"{prefix}.pre_quant_scale", None)
    bias = state_dict.pop(f"{prefix}.bias", None)
    state_dict.pop(f"{prefix}.input_scale", None)

    n = len(dst_prefixes)
    if weight.shape[0] % n != 0:
        raise ValueError(
            f"{prefix}: fused out dim {weight.shape[0]} does not split into {n}"
        )
    rows = weight.shape[0] // n

    scale_parts = None
    if scale is not None:
        if fmt == "nvfp4":
            in_features = weight.shape[1] * 2  # packed fp4 pairs
            full = unswizzle_nvfp4_scales(
                scale.view(torch.float8_e4m3fn), weight.shape[0], in_features // 16
            )
            scale_parts = [
                swizzle_nvfp4_scales(p).view(torch.float8_e4m3fn)
                for p in full.split(rows, dim=0)
            ]
        elif scale.ndim == 0 or scale.numel() == 1:
            scale_parts = [scale.clone() for _ in range(n)]
        else:
            scale_parts = list(scale.reshape(weight.shape[0], -1).split(rows, dim=0))

    for i, dst in enumerate(dst_prefixes):
        state_dict[f"{dst}.comfy_quant"] = marker.clone()
        state_dict[f"{dst}.weight"] = weight[i * rows : (i + 1) * rows].contiguous()
        if scale_parts is not None:
            state_dict[f"{dst}.weight_scale"] = scale_parts[i].contiguous()
        if pts is not None:
            state_dict[f"{dst}.weight_scale_2"] = pts.clone()
        if pre is not None:
            state_dict[f"{dst}.pre_quant_scale"] = pre.clone()
        if bias is not None:
            state_dict[f"{dst}.bias"] = bias[i * rows : (i + 1) * rows].contiguous()
    return state_dict


@torch.no_grad()
def fuse_split_quantized_keys(
    state_dict: Dict[str, torch.Tensor],
    src_prefixes,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Inverse of split_fused_quantized_keys: concatenate N split quantized
    comfy entries back into one fused entry (out-dim concat in src order).
    All parts must share the same format config; fp8 parts must share the same
    per-tensor scale (true for entries produced by the splitter). Mutates and
    returns state_dict."""
    from toolkit.util.nvfp4_quant import swizzle_nvfp4_scales

    markers = [state_dict.pop(f"{p}.comfy_quant") for p in src_prefixes]
    confs = [parse_comfy_quant_blob(m) for m in markers]
    if any(c != confs[0] for c in confs[1:]):
        raise ValueError(f"{prefix}: split parts carry different quant configs")
    fmt = confs[0].get("format")

    weights = [state_dict.pop(f"{p}.weight") for p in src_prefixes]
    scales = [state_dict.pop(f"{p}.weight_scale", None) for p in src_prefixes]
    ptss = [state_dict.pop(f"{p}.weight_scale_2", None) for p in src_prefixes]
    pres = [state_dict.pop(f"{p}.pre_quant_scale", None) for p in src_prefixes]
    biases = [state_dict.pop(f"{p}.bias", None) for p in src_prefixes]

    state_dict[f"{prefix}.comfy_quant"] = markers[0]
    weight = torch.cat(weights, dim=0).contiguous()
    state_dict[f"{prefix}.weight"] = weight
    if scales[0] is not None:
        if fmt == "nvfp4":
            in_features = weight.shape[1] * 2
            rows = [w.shape[0] for w in weights]
            full = torch.cat(
                [
                    unswizzle_nvfp4_scales(
                        s.view(torch.float8_e4m3fn), r, in_features // 16
                    )
                    for s, r in zip(scales, rows)
                ],
                dim=0,
            )
            state_dict[f"{prefix}.weight_scale"] = swizzle_nvfp4_scales(full).view(
                torch.float8_e4m3fn
            )
        elif scales[0].ndim == 0 or scales[0].numel() == 1:
            if any(
                not torch.equal(s.reshape(-1), scales[0].reshape(-1)) for s in scales[1:]
            ):
                raise ValueError(
                    f"{prefix}: per-tensor scales differ across split parts"
                )
            state_dict[f"{prefix}.weight_scale"] = scales[0]
        else:
            state_dict[f"{prefix}.weight_scale"] = torch.cat(
                [s.reshape(w.shape[0], -1) for s, w in zip(scales, weights)], dim=0
            ).contiguous()
    if ptss[0] is not None:
        state_dict[f"{prefix}.weight_scale_2"] = ptss[0]
    if pres[0] is not None:
        state_dict[f"{prefix}.pre_quant_scale"] = pres[0]
    if biases[0] is not None:
        state_dict[f"{prefix}.bias"] = torch.cat(biases, dim=0).contiguous()
    return state_dict


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

        if fmt == "float8_e4m3fn":
            # fp8_e4m3 weight + fp32 per-tensor scale, dequantized matmul
            from toolkit.util.float8_quant import Float8Quantizer

            quantizer = get_ostris_quantizer("float8_e4m3fn")
            Float8Quantizer.attach_(
                module,
                weight.view(torch.float8_e4m3fn)
                if weight.dtype != torch.float8_e4m3fn
                else weight,
                weight_scale,
            )
        elif fmt == "int8_tensorwise":
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
                "(supported: int8_tensorwise, nvfp4, float8_e4m3fn)"
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

    # legacy ComfyUI scaled-fp8 checkpoints (e.g. the wan *_fp8_scaled files):
    # a top-level ``scaled_fp8`` marker tensor plus per-layer fp8 ``weight``
    # and scalar fp32 ``scale_weight`` — the float8 backend's exact storage.
    # ``scale_input`` (activation quant) is dropped; matmuls run dequantized.
    if "scaled_fp8" in state_dict:
        from toolkit.util.float8_quant import Float8Quantizer

        state_dict.pop("scaled_fp8")
        for scale_key in [k for k in state_dict if k.endswith(".scale_weight")]:
            prefix = scale_key[: -len(".scale_weight")]
            module_path = key_map(prefix) if key_map is not None else prefix
            module = root.get_submodule(module_path)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(
                    f"scaled_fp8 entry {prefix} points at {type(module).__name__}, "
                    "expected nn.Linear"
                )
            weight = state_dict.pop(f"{prefix}.weight")
            scale = state_dict.pop(scale_key)
            state_dict.pop(f"{prefix}.scale_input", None)
            quantizer = get_ostris_quantizer("float8_e4m3fn")
            Float8Quantizer.attach_(
                module,
                weight
                if weight.dtype == torch.float8_e4m3fn
                else weight.view(torch.float8_e4m3fn),
                scale,
            )
            _to_ostris(module, quantizer, orig_dtype)
            bias = state_dict.pop(f"{prefix}.bias", None)
            if bias is not None and module.bias is not None:
                module._parameters["bias"] = torch.nn.Parameter(
                    bias.detach().clone(), requires_grad=False
                )
            converted += 1

    return state_dict, converted
