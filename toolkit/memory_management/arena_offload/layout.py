"""Static host layout and canonical-arena packing primitives.

This module owns no CUDA stream, execution hook, trace, queue, or transfer
lifetime. It describes host storage, wrapper reconstruction, and typed views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from toolkit.quantization.storage import module_storage_binding
from toolkit.memory_management import pin_manager

LEAF_ALIGN = 256


@dataclass(frozen=True)
class LeafSpec:
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    role: str


@dataclass(frozen=True)
class LinearSpec:
    name: str
    tensors: tuple[LeafSpec, ...]
    execution_key: tuple
    weight_leaf_count: int
    weight_template: torch.Tensor
    weight_requires_grad: bool
    bias_requires_grad: bool
    substitutions: tuple = ()

    def tensor(self, name: str) -> LeafSpec | None:
        return next((item for item in self.tensors if item.role == name), None)


@dataclass
class BlockPack:
    block_key: str
    host_flat: torch.Tensor
    linears: tuple[LinearSpec, ...]
    required_pin_bytes: int
    pinned: bool
    pin_handle: object | None = None


@dataclass(frozen=True)
class LayerStorageView:
    """Opaque ordered tensors for one logical layer.

    Storage movement deliberately does not attach an execution operation or
    interpret tensor roles. The architecture adapter binds execution after the
    immutable storage ABI has been finalized.
    """

    spec: LinearSpec
    tensors: tuple[torch.Tensor, ...]


def release_pack(pack: "BlockPack | None") -> None:
    """Release a canonical block pack's pin grant."""
    if pack is None:
        return
    pin_manager.release(pack.pin_handle)
    pack.pin_handle = None
    pack.pinned = False


@dataclass(frozen=True)
class LeafDescriptor:
    role: str
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True)
class LinearLayout:
    name: str
    leaf_descriptors: tuple[LeafDescriptor, ...]
    weight_leaf_count: int
    weight_requires_grad: bool
    bias_requires_grad: bool
    execution_key: tuple
    weight_template: torch.Tensor | None
    substitutions: tuple = ()

    def leaf(self, role: str) -> LeafDescriptor | None:
        return next((leaf for leaf in self.leaf_descriptors if leaf.role == role), None)


@dataclass(frozen=True)
class BlockLayout:
    block_key: str
    linears: tuple[LinearLayout, ...]
    nbytes: int


def flatten_leaves(value: torch.Tensor) -> list[torch.Tensor]:
    try:
        names, _ = value.__tensor_flatten__()
    except Exception:
        return [value]
    leaves = []
    for name in names:
        child = getattr(value, name, None)
        if child is not None:
            leaves.extend(flatten_leaves(child))
    return leaves


def rebuild_from_leaves(template: torch.Tensor, leaves: Iterable[torch.Tensor]):
    iterator = iter(leaves)

    def rebuild(value):
        try:
            names, context = value.__tensor_flatten__()
        except Exception:
            return next(iterator)
        children = {}
        for name in names:
            child = getattr(value, name, None)
            children[name] = None if child is None else rebuild(child)
        return type(value).__tensor_unflatten__(children, context, value.size(), value.stride())

    return rebuild(template)


def inspect_block(block_key: str, entries) -> BlockLayout:
    cursor = 0
    linears = []
    for name, module in entries:
        weight = module._parameters.get("weight")
        bias = module._parameters.get("bias")
        binding = module_storage_binding(module)
        leaves = [item.tensor for item in binding.tensors]
        roles = [item.name for item in binding.tensors]
        descriptors = []
        for role, leaf in zip(roles, leaves):
            cursor = (cursor + LEAF_ALIGN - 1) // LEAF_ALIGN * LEAF_ALIGN
            nbytes = leaf.numel() * leaf.element_size()
            descriptors.append(LeafDescriptor(role, cursor, nbytes, leaf.dtype, tuple(leaf.shape)))
            cursor += nbytes
        linears.append(LinearLayout(
            name=name,
            leaf_descriptors=tuple(descriptors),
            weight_leaf_count=binding.weight_leaf_count,
            weight_requires_grad=bool(weight.requires_grad) if weight is not None else False,
            bias_requires_grad=bool(bias.requires_grad) if bias is not None else False,
            execution_key=binding.execution_key,
            weight_template=binding.weight_template,
            substitutions=binding.substitutions,
        ))
    return BlockLayout(block_key, tuple(linears), cursor)


def typed_view(flat: torch.Tensor, leaf: LeafDescriptor) -> torch.Tensor:
    return flat[leaf.offset:leaf.offset + leaf.nbytes].view(leaf.dtype).reshape(leaf.shape)


def linear_views(flat: torch.Tensor, layout: LinearLayout):
    views = tuple(typed_view(flat, leaf) for leaf in layout.leaf_descriptors)
    weight_views = views[:layout.weight_leaf_count]
    weight = (
        weight_views[0]
        if layout.weight_leaf_count == 1
        else rebuild_from_leaves(layout.weight_template, weight_views)
    )
    bias = views[layout.weight_leaf_count] if len(views) > layout.weight_leaf_count else None
    return weight, bias


def substitution_views(flat: torch.Tensor, layout: LinearLayout) -> dict[str, torch.Tensor]:
    """Reconstruct every declared module-state target from one flat."""
    views = tuple(typed_view(flat, leaf) for leaf in layout.leaf_descriptors)
    return {
        substitution.name: substitution.reconstruct(
            tuple(views[index] for index in substitution.tensor_indices)
        )
        for substitution in layout.substitutions
    }
