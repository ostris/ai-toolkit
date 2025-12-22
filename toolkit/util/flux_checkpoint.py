import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file
# Mapping adapted from kohya-ss/sd-scripts (Apache-2.0).
_BFL_TO_DIFFUSERS_MAP: Dict[str, List[str]] = {
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias": ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias": ["time_text_embed.timestep_embedder.linear_2.bias"],
    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias": ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias": ["time_text_embed.text_embedder.linear_2.bias"],
    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias": ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias": ["time_text_embed.guidance_embedder.linear_2.bias"],
    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias": ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias": ["x_embedder.bias"],
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias": ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias": ["norm1_context.linear.bias"],
    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale": ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale": ["attn.norm_added_k.weight"],
    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias": ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias": ["ff.net.2.bias"],
    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias": ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias": ["ff_context.net.2.bias"],
    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias": ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias": ["attn.to_add_out.bias"],
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias": ["norm.linear.bias"],
    "single_blocks.().linear1.weight": [
        "attn.to_q.weight",
        "attn.to_k.weight",
        "attn.to_v.weight",
        "proj_mlp.weight",
    ],
    "single_blocks.().linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().linear2.bias": ["proj_out.bias"],
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale": ["attn.norm_k.weight"],
    "final_layer.linear.weight": ["proj_out.weight"],
    "final_layer.linear.bias": ["proj_out.bias"],
    "final_layer.adaLN_modulation.1.weight": ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias": ["norm_out.linear.bias"],
}

def _expand_sharded_paths(path: str) -> List[str]:
    match = re.match(r"(.*)-0+1-of-0+([0-9]+)\.safetensors$", path)
    if not match:
        return [path]
    base, total = match.groups()
    total_int = int(total)
    width = max(len(total), 5)
    return [
        f"{base}-{str(i).zfill(width)}-of-{total.zfill(width)}.safetensors"
        for i in range(1, total_int + 1)
    ]

def _load_safetensors(paths: Iterable[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    state_dict: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, str] = {}
    for file_path in paths:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata.update(f.metadata())
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict, metadata

def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> None:
    if not any(key.startswith(prefix) for key in state_dict):
        return
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            state_dict[key[len(prefix):]] = state_dict.pop(key)

def _swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)

def detect_flux_checkpoint_format(keys: Iterable[str]) -> str:
    for key in keys:
        if key.startswith("transformer_blocks.") or key.startswith("single_transformer_blocks."):
            return "diffusers"
        if key.startswith("double_blocks.") or key.startswith("single_blocks."):
            return "bfl"
    return "unknown"

def infer_flux_variant(
    keys: Iterable[str],
    metadata: Optional[Dict[str, str]] = None,
    fallback_path: Optional[str] = None,
) -> str:
    metadata = metadata or {}
    lower_meta = {k.lower(): v.lower() for k, v in metadata.items()}
    for value in lower_meta.values():
        if "schnell" in value:
            return "schnell"
        if "dev" in value:
            return "dev"

    if any("schnell" in key.lower() for key in keys):
        return "schnell"

    if fallback_path:
        lowered = fallback_path.lower()
        if "schnell" in lowered:
            return "schnell"
        if "dev" in lowered:
            return "dev"

    has_guidance = any(key.startswith("guidance_in.") for key in keys)
    return "dev" if has_guidance else "schnell"

def load_flux_transformer_state(
    path: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str], str, str]:
    expanded_paths = _expand_sharded_paths(path)
    state_dict, metadata = _load_safetensors(expanded_paths)
    _strip_prefix(state_dict, "model.diffusion_model.")
    _strip_prefix(state_dict, "transformer.")
    fmt = detect_flux_checkpoint_format(state_dict.keys())
    variant = infer_flux_variant(state_dict.keys(), metadata, path)
    return state_dict, metadata, fmt if fmt != "unknown" else "bfl", variant

def _extract_block_counts(keys: Iterable[str]) -> Tuple[int, int]:
    double_indices = {
        int(match.group(1))
        for key in keys
        if (match := re.match(r"double_blocks\.([0-9]+)\.", key))
    }
    single_indices = {
        int(match.group(1))
        for key in keys
        if (match := re.match(r"single_blocks\.([0-9]+)\.", key))
    }
    num_double = max(double_indices) + 1 if double_indices else 0
    num_single = max(single_indices) + 1 if single_indices else 0
    return num_double, num_single

def _split_tensor_to_match_shapes(
    tensor: torch.Tensor,
    target_shapes: List[torch.Size],
) -> List[torch.Tensor]:
    if len(target_shapes) == 1:
        return [tensor]

    ndim = tensor.ndim
    for axis in range(ndim):
        axis_sizes = [shape[axis] if axis < len(shape) else 1 for shape in target_shapes]
        if tensor.shape[axis] != sum(axis_sizes):
            continue

        compatible = True
        for shape in target_shapes:
            for dim in range(ndim):
                if dim == axis:
                    continue
                expected = shape[dim] if dim < len(shape) else 1
                if tensor.shape[dim] != expected:
                    compatible = False
                    break
            if not compatible:
                break
        if not compatible:
            continue

        splits = torch.split(tensor, axis_sizes, dim=axis)
        return [chunk.contiguous() for chunk in splits]

    raise ValueError(
        f"Unable to split tensor of shape {tuple(tensor.shape)} "
        f"to match target shapes {[tuple(shape) for shape in target_shapes]}"
    )

def convert_bfl_to_diffusers_state(
    bfl_state: Dict[str, torch.Tensor],
    reference_shapes: Dict[str, torch.Size],
    *,
    num_double_blocks: Optional[int] = None,
    num_single_blocks: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if num_double_blocks is None or num_single_blocks is None:
        detected_double, detected_single = _extract_block_counts(bfl_state.keys())
        num_double_blocks = num_double_blocks or detected_double
        num_single_blocks = num_single_blocks or detected_single

    diffusers_state: Dict[str, torch.Tensor] = {}

    for bfl_key, diff_keys in _BFL_TO_DIFFUSERS_MAP.items():
        if "()" in bfl_key:
            if bfl_key.startswith("double_blocks."):
                for idx in range(num_double_blocks or 0):
                    resolved_key = bfl_key.replace("()", str(idx))
                    if resolved_key not in bfl_state:
                        continue
                    tensor = bfl_state[resolved_key]
                    prefixes = [
                        f"transformer_blocks.{idx}.{suffix}" for suffix in diff_keys
                    ]
                    target_shapes = [reference_shapes[prefix] for prefix in prefixes]
                    chunks = _split_tensor_to_match_shapes(tensor, target_shapes)
                    for name, chunk in zip(prefixes, chunks):
                        diffusers_state[name] = chunk
            elif bfl_key.startswith("single_blocks."):
                for idx in range(num_single_blocks or 0):
                    resolved_key = bfl_key.replace("()", str(idx))
                    if resolved_key not in bfl_state:
                        continue
                    tensor = bfl_state[resolved_key]
                    prefixes = [
                        f"single_transformer_blocks.{idx}.{suffix}" for suffix in diff_keys
                    ]
                    target_shapes = [reference_shapes[prefix] for prefix in prefixes]
                    chunks = _split_tensor_to_match_shapes(tensor, target_shapes)
                    for name, chunk in zip(prefixes, chunks):
                        diffusers_state[name] = chunk
            continue

        if bfl_key not in bfl_state:
            continue

        tensor = bfl_state[bfl_key]
        if bfl_key in {
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias",
        }:
            tensor = _swap_scale_shift(tensor)

        target_names = diff_keys
        target_shapes = [reference_shapes[name] for name in target_names]
        chunks = _split_tensor_to_match_shapes(tensor, target_shapes)
        for name, chunk in zip(target_names, chunks):
            diffusers_state[name] = chunk

    return diffusers_state

def load_diffusers_state_from_file(path: str) -> Dict[str, torch.Tensor]:
    state_dict = load_file(path, device="cpu")
    _strip_prefix(state_dict, "transformer.")
    return state_dict

def determine_default_flux_base_repo(variant: str) -> str:
    if variant == "schnell":
        return "black-forest-labs/FLUX.1-schnell"
    return "black-forest-labs/FLUX.1-dev"
