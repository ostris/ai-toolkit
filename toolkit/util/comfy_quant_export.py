"""Export toolkit-quantized modules into ComfyUI ``comfy_quant`` checkpoints —
the inverse of toolkit/util/comfy_quant_import.py.

Every quantized OstrisLinear whose backend has a comfy storage format emits
its quantized tensors plus the ``<prefix>.comfy_quant`` uint8 JSON marker:

  - convrot8 (int8_tensorwise + convrot): weight int8 [out, in], fp32
    weight_scale [out, 1] (comfy_kitchen's per-channel convention)
  - nvfp4: high-nibble-first packed fp4 pairs, e4m3 block scales re-swizzled
    to the cuBLAS 128x4 tile layout, fp32 weight_scale_2 per-tensor scale and
    optional AWQ pre_quant_scale
  - float8_e4m3fn: fp8_e4m3 weight + one fp32 per-tensor weight_scale
  - convrotcomfyw4a4: via convrot_quant.export_comfy_convrot_w4a4

Biases are NOT emitted here — they are ordinary parameters and flow through
the regular state_dict path.
"""

import json
from typing import Dict, List, Tuple

import torch

from toolkit.util.ostris_quant import OstrisLinear


def comfy_quant_marker(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


@torch.no_grad()
def export_comfy_quantized_layers(
    root: torch.nn.Module,
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """Comfy-format state-dict entries for every quantized OstrisLinear in
    ``root``. Returns ``(entries, exported_names, unexportable_names)`` —
    entries are keyed by module path in root's layout (run
    convert_state_dict_on_save afterwards for the checkpoint layout);
    unexportable_names lists quantized modules whose backend has no comfy
    storage format (the caller decides whether to dequantize instead)."""
    from toolkit.util.convrot_quant import (
        ConvRotComfyW4A4Quantizer,
        export_comfy_convrot_w4a4,
    )
    from toolkit.util.nvfp4_quant import swap_nvfp4_nibbles, swizzle_nvfp4_scales

    entries: Dict[str, torch.Tensor] = {}
    exported: List[str] = []
    unexportable: List[str] = []

    for name, module in root.named_modules():
        if not isinstance(module, OstrisLinear):
            continue

        if hasattr(module, "cr8_qdata"):
            rot = int(getattr(module, "cr8_rot_size", 1) or 1)
            conf = {"format": "int8_tensorwise"}
            if rot > 1:
                conf.update({"convrot": True, "convrot_groupsize": rot})
            entries[f"{name}.weight"] = module.cr8_qdata.detach().cpu().contiguous()
            entries[f"{name}.weight_scale"] = (
                module.cr8_scales.view(torch.float32)
                .detach()
                .cpu()
                .reshape(module.out_features, 1)
                .contiguous()
            )
            entries[f"{name}.comfy_quant"] = comfy_quant_marker(conf)
        elif hasattr(module, "nv4_qdata"):
            entries[f"{name}.weight"] = swap_nvfp4_nibbles(
                module.nv4_qdata.detach().cpu()
            )
            scales = module.nv4_scales.view(torch.float8_e4m3fn).detach().cpu()
            entries[f"{name}.weight_scale"] = swizzle_nvfp4_scales(
                scales.reshape(module.out_features, module.in_features // 16)
            ).view(torch.float8_e4m3fn)
            entries[f"{name}.weight_scale_2"] = (
                module.nv4_pts.view(torch.float32).detach().cpu().reshape(())
            )
            if hasattr(module, "nv4_pre_scale"):
                entries[f"{name}.pre_quant_scale"] = (
                    module.nv4_pre_scale.view(torch.float32).detach().cpu().contiguous()
                )
            entries[f"{name}.comfy_quant"] = comfy_quant_marker({"format": "nvfp4"})
        elif hasattr(module, "f8_qdata"):
            entries[f"{name}.weight"] = module.f8_qdata.detach().cpu().contiguous()
            entries[f"{name}.weight_scale"] = (
                module.f8_scale.view(torch.float32).detach().cpu().reshape(())
            )
            entries[f"{name}.comfy_quant"] = comfy_quant_marker(
                {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}
            )
        elif isinstance(module.ostris_quantizer, ConvRotComfyW4A4Quantizer):
            layer_entries = export_comfy_convrot_w4a4(module, f"{name}.")
            layer_entries.pop(f"{name}.bias", None)
            entries.update(
                {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in layer_entries.items()}
            )
        else:
            unexportable.append(name)
            continue
        exported.append(name)

    return entries, exported, unexportable
