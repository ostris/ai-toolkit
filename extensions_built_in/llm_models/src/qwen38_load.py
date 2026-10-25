"""Build a Qwen38ForConditionalGeneration from a ComfyUI single file (quantized layers
attached to the toolkit backends as-is) or an HF layout (bf16 shards)."""

import json
import os
from typing import Callable, Optional, Tuple

import torch
from safetensors import safe_open

from toolkit.print import print_acc

from .qwen38_model import Qwen38ForConditionalGeneration


def _list_source_files(source: str):
    """(files, is_single_file) for a .safetensors path, a local HF dir or a hub repo id."""
    if os.path.isfile(source):
        return [source], True
    if os.path.isdir(source):
        d = source
    else:
        import huggingface_hub

        d = huggingface_hub.snapshot_download(source, allow_patterns=["*.safetensors", "*.json"])
    index = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(index):
        files = sorted(set(json.load(open(index))["weight_map"].values()))
        return [os.path.join(d, f) for f in files], False
    single = os.path.join(d, "model.safetensors")
    if os.path.isfile(single):
        return [single], False
    raise FileNotFoundError(f"No safetensors weights found under {d}")


def _read_keys(files):
    keys = set()
    for f in files:
        with safe_open(f, "pt") as sf:
            keys.update(sf.keys())
    return keys


def _load_into_device(files, device) -> dict:
    sd = {}
    dev = str(device)
    for f in files:
        with safe_open(f, "pt", device=dev) as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)
    return sd


@torch.no_grad()
def load_qwen38(
    source: str,
    config,
    dtype: torch.dtype,
    device,
    status_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Qwen38ForConditionalGeneration, bool]:
    """Returns (model on `device`, is_prequantized). Quantized ComfyUI layers keep
    their stored format (convrotw4a8 / convrot8 OstrisLinear, int8 embeddings)."""
    from accelerate import init_empty_weights

    from toolkit.util.comfy_quant_import import import_comfy_quantized_layers

    files, _ = _list_source_files(source)
    keys = _read_keys(files)
    with_mtp = any(k.startswith("mtp.") for k in keys)
    config.vision_config._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=False):
        model = Qwen38ForConditionalGeneration(config, with_mtp=with_mtp)
    model.eval()

    if status_fn is not None:
        status_fn(f"Reading {len(files)} weight file(s) onto {device}")
    sd = _load_into_device(files, device)
    prequantized = any(k.endswith(".comfy_quant") for k in sd)
    if prequantized:
        sd, n = import_comfy_quantized_layers(model, sd, orig_dtype=dtype, rotated_embedding_rows=True)
        print_acc(f"Attached {n} pre-quantized ComfyUI layers")
    sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    result = model.load_state_dict(sd, assign=True, strict=False)
    allowed = set()
    for name, module in model.named_modules():
        if hasattr(module, "ostris_quantizer"):
            allowed.add(f"{name}.weight")
            allowed.add(f"{name}.bias")
    missing = [k for k in result.missing_keys if k not in allowed]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Qwen3.8 checkpoint mismatch. missing: {missing[:8]} unexpected: {result.unexpected_keys[:8]}")
    leftover = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if leftover:
        raise RuntimeError(f"Params never loaded: {leftover[:8]}")
    # buffers built in __init__ (rope tables) still live on the cpu
    for module in model.modules():
        if hasattr(module, "ostris_quantizer"):
            continue
        for pname, prm in list(module.named_parameters(recurse=False)):
            if prm.is_floating_point() and prm.dtype != dtype:
                prm.data = prm.data.to(dtype)
    model.to(device)
    return model, prequantized


@torch.no_grad()
def dequantize_vision_tower(model: Qwen38ForConditionalGeneration, dtype: torch.dtype) -> int:
    """Replace the vision tower's quantized linears with plain Linears holding the
    dequantized weights. The stored int8 weights are kept exactly; only the per-token
    int8 activation quantization goes away, which measured a 6% drift of the image
    embeddings against the bf16 reference. Runs once per image, so the cost is memory
    only (~0.4 GB for the 27B tower). Returns the number of layers converted."""
    from toolkit.util.ostris_quant import OstrisLinear

    visual = model.model.visual
    n = 0
    for name, mod in list(visual.named_modules()):
        if not isinstance(mod, OstrisLinear):
            continue
        lin = torch.nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None, device="meta", dtype=dtype)
        lin.weight = torch.nn.Parameter(mod.ostris_quantizer.dequantize(mod).to(dtype), requires_grad=False)
        if mod.bias is not None:
            lin.bias = torch.nn.Parameter(mod.bias.detach().to(dtype), requires_grad=False)
        parent_name, _, attr = name.rpartition(".")
        parent = visual.get_submodule(parent_name) if parent_name else visual
        setattr(parent, attr, lin)
        n += 1
    return n
