"""FSDP v2 utilities for multi-GPU parameter sharding.

When multi-GPU + LoRA training is detected, FSDP v2 is used to shard the frozen
base model parameters across GPUs, reducing per-GPU memory. Only the transformer
is FSDP-wrapped; VAE and text encoders are excluded.
"""

import torch.nn as nn
from typing import List

# Common transformer block attribute names across diffusion model architectures.
# Used as a fallback when the model doesn't implement get_transformer_block_names().
# Follows the same pattern as HuggingFace finetrainers.
KNOWN_BLOCK_ATTR_NAMES = [
    "transformer_blocks",
    "single_transformer_blocks",
    "double_stream_blocks",
    "single_stream_blocks",
    "double_blocks",
    "single_blocks",
    "temporal_transformer_blocks",
    "blocks",
    "layers",
]


def get_block_class_names(
    transformer: nn.Module,
    model=None,
) -> List[str]:
    """Introspect the transformer to find its block class names for FSDP wrapping.

    First tries model.get_transformer_block_names() if available, then falls back
    to scanning known attribute names on the transformer module.

    Args:
        transformer: The transformer/unet module to introspect.
        model: The parent model object (e.g. StableDiffusion) that may have
               get_transformer_block_names().

    Returns:
        List of unique class name strings for FSDP transformer-based wrapping.
    """
    block_attr_names = None

    # Try the model's declared block names first
    if model is not None and hasattr(model, "get_transformer_block_names"):
        block_attr_names = model.get_transformer_block_names()

    # Fallback: scan known attribute names on the transformer
    if not block_attr_names:
        block_attr_names = []
        for attr_name in KNOWN_BLOCK_ATTR_NAMES:
            blocks = getattr(transformer, attr_name, None)
            if blocks is not None and isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
                block_attr_names.append(attr_name)

    # Extract unique class names from the discovered block lists
    class_names = set()
    for attr_name in block_attr_names:
        blocks = getattr(transformer, attr_name, None)
        if blocks is None:
            continue
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            class_names.add(type(blocks[0]).__name__)
        elif isinstance(blocks, nn.Module):
            # Some models have sub-modules that contain blocks rather than
            # being ModuleLists directly. Check for nested ModuleLists.
            for child_name, child in blocks.named_children():
                if isinstance(child, nn.ModuleList) and len(child) > 0:
                    class_names.add(type(child[0]).__name__)
                    break

    return list(class_names)


def create_fsdp_plugin(transformer_block_class_names: List[str]):
    """Create an Accelerate FSDP v2 plugin for parameter sharding.

    Args:
        transformer_block_class_names: Class names of transformer blocks to wrap
            as individual FSDP units (e.g. ["FluxTransformerBlock", "FluxSingleTransformerBlock"]).

    Returns:
        FullyShardedDataParallelPlugin configured for FSDP v2 with FULL_SHARD.
    """
    from accelerate import FullyShardedDataParallelPlugin

    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=transformer_block_class_names,
        reshard_after_forward=True,  # FULL_SHARD: shard params after forward for max memory savings
        activation_checkpointing=False,  # toolkit handles this separately
        cpu_ram_efficient_loading=True,
        state_dict_type="FULL_STATE_DICT",  # needed for LoRA weight extraction
    )
    return plugin
