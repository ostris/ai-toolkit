"""FSDP2 (torch.distributed.fsdp.fully_shard) wiring for quantized training.

Shards whole transformer blocks (frozen mxfp8 base weights AND LoRA params
alike) across ranks: FSDP2 all-gathers our compressed fp8 bytes via the
MXFp8StaticFrozenWeight collective dispatch, and reduce-scatters LoRA grads
automatically (LoRA params become DTensor).

Call `apply_fsdp2` after model load + quantization, before optimizer creation.
"""

import torch
import torch.nn as nn

from toolkit.distributed import is_distributed, get_rank, get_world_size

# attribute names that hold the transformer block lists across model families
_BLOCK_LIST_ATTRS = (
    "transformer_blocks", "single_transformer_blocks",  # flux / flux2 / qwen-image
    "double_blocks", "single_blocks",                   # sd3 / hidream style
    "layers", "blocks",
)


def _find_block_lists(model: nn.Module):
    """Find top-level child block-lists (e.g. transformer_blocks), skipping
    lists nested inside another list's blocks."""
    found = []
    for mod in model.modules():
        for attr in _BLOCK_LIST_ATTRS:
            lst = getattr(mod, attr, None)
            if isinstance(lst, (nn.ModuleList, list)) and len(lst) > 0 and isinstance(lst[0], nn.Module):
                found.append((mod, attr, lst))
    keep = []
    for parent, attr, lst in found:
        inside_other = False
        for p2, a2, l2 in found:
            if l2 is lst:
                continue
            if any(parent is b or _is_descendant(parent, b) for b in l2):
                inside_other = True
                break
        if not inside_other:
            keep.append((parent, attr, lst))
    return keep


def _is_descendant(mod: nn.Module, ancestor: nn.Module) -> bool:
    return any(m is mod for m in ancestor.modules())


def apply_fsdp2(model: nn.Module, mesh=None) -> int:
    """fully_shard every transformer block, then the root. Returns #blocks sharded."""
    if not is_distributed():
        return 0
    from torch.distributed.fsdp import fully_shard

    n = 0
    for parent, attr, lst in _find_block_lists(model):
        for block in lst:
            fully_shard(block, mesh=mesh)
            n += 1
    fully_shard(model, mesh=mesh)
    rank = get_rank()
    if rank == 0:
        print(f"[fsdp2] fully_shard applied to {n} transformer blocks across "
              f"{get_world_size()} ranks")
    return n


def gather_state_dict_for_save(state_dict: dict) -> dict:
    """Convert DTensor params to full tensors (all-gather) so every rank has the
    complete state dict; caller decides which rank writes."""
    from torch.distributed._tensor import DTensor
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, DTensor):
            out[k] = v.full_tensor()
        else:
            out[k] = v
    return out
