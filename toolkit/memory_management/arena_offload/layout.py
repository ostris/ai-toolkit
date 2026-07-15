"""Static host layout and canonical-arena packing primitives.

This module owns no CUDA stream, execution hook, trace, queue, or transfer
lifetime. It describes host storage, wrapper reconstruction, and typed views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from toolkit.quantization.storage import linear_storage_binding, module_storage_binding
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
    view_maker: object | None = None
    # Ownership of ``host_flat``'s pin grant. ``pin_handle`` is the PinHandle
    # returned by pin_manager.pin_alloc for this pack's OWN flat allocation
    # (None when the pack didn't allocate -- e.g. it borrows another owner's
    # storage). ``owns_flat`` gates release_pack: a borrowed pack (arena-backed,
    # borrowed_from_arena=True) must never release someone else's handle.
    pin_handle: object | None = None
    owns_flat: bool = True
    borrowed_from_arena: bool = False


@dataclass(frozen=True)
class LayerStorageView:
    """Opaque ordered tensors for one logical layer.

    Storage movement deliberately does not attach an execution operation or
    interpret tensor roles. The architecture adapter binds execution after the
    immutable storage ABI has been finalized.
    """

    spec: LinearSpec
    tensors: tuple[torch.Tensor, ...]


def layer_storage_views(pack: BlockPack) -> tuple[LayerStorageView, ...]:
    """Expose a block's immutable execution declarations in leaf order."""
    return tuple(
        LayerStorageView(
            spec=spec,
            tensors=tuple(typed_view(pack.host_flat, leaf) for leaf in spec.tensors),
        )
        for spec in pack.linears
    )


def _rebuild_from_leaves(src, leaves_iter):
    try:
        names, ctx = src.__tensor_flatten__()
    except Exception:
        return next(leaves_iter)
    moved = {}
    for name in names:
        inner = getattr(src, name, None)
        moved[name] = None if inner is None else _rebuild_from_leaves(inner, leaves_iter)
    return type(src).__tensor_unflatten__(moved, ctx, src.size(), src.stride())


def _aligned_offsets(leaves: Iterable[torch.Tensor], align: int = LEAF_ALIGN):
    offsets = []
    total = 0
    for leaf in leaves:
        total = (total + align - 1) // align * align
        offsets.append(total)
        total += leaf.numel() * leaf.element_size()
    return offsets, total


def _empty_host_flat(
    nbytes: int,
    *,
    pin: bool = True,
    kind: str = "ingraph_pack",
    pin_mechanism: str = "alloc",
) -> tuple[torch.Tensor, bool, object | None]:
    if not pin:
        return torch.empty(nbytes, dtype=torch.uint8), False, None
    if pin_mechanism == "register":
        # I1: prepare the page-aligned buffer WITHOUT registering it yet --
        # pack_block_host copies the leaves into it (ordinary pageable
        # memcpy) before the caller commits the cudaHostRegister pin. See
        # pin_register_prepare's docstring for why population-before-pin is
        # faster than registering a virgin buffer.
        candidate, padded = pin_manager.pin_register_prepare(nbytes)
        return candidate, False, ("register_pending", padded, kind)
    else:
        handle = pin_manager.pin_alloc(
            nbytes,
            kind,
            required=False,
            mode="sampling",
        )
    return handle.tensor, bool(handle.pinned), handle


def release_pack(pack: "BlockPack | None") -> None:
    """Release a pack's own pin grant.

    A borrowed pack (``owns_flat=False``, e.g. arena-backed) must never
    release someone else's handle -- the owner (the arena) is responsible for
    its own flat's lifetime."""
    if pack is None or not pack.owns_flat:
        return
    pin_manager.release(pack.pin_handle)
    pack.pin_handle = None
    pack.pinned = False


def pack_block_host(
    block_key: str,
    linears,
    *,
    repoint: bool = True,
    pin: bool = True,
    kind: str = "ingraph_pack",
    pin_mechanism: str = "alloc",
) -> BlockPack:
    """Pack declared Linear storage tuples into one aligned host buffer."""
    normalized = []
    leaves = []
    for entry in linears:
        if len(entry) == 2:
            name, module = entry
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            name, weight, bias = entry
            module = None
        binding = linear_storage_binding(weight, bias)
        tensors = tuple(item.tensor for item in binding.tensors)
        normalized.append((name, module, weight, bias, binding, tensors))
        leaves.extend(tensors)

    offsets, total = _aligned_offsets(leaves)
    host, pinned, pin_handle = _empty_host_flat(
        total, pin=pin, kind=kind, pin_mechanism=pin_mechanism
    )
    register_pending = isinstance(pin_handle, tuple) and pin_handle[:1] == (
        "register_pending",
    )
    try:
        for leaf, offset in zip(leaves, offsets):
            nbytes = leaf.numel() * leaf.element_size()
            host[offset:offset + nbytes].view(leaf.dtype).reshape(leaf.shape).copy_(leaf)

        if register_pending:
            _, _padded, register_kind = pin_handle
            pin_handle = pin_manager.pin_register_commit(
                host, total, register_kind, required=False
            )
            host = pin_handle.tensor
            pinned = bool(pin_handle.pinned)

        cursor = 0
        specs = []
        for name, module, weight, bias, binding, tensors in normalized:
            tensor_specs = []
            views = []
            for declared, leaf in zip(binding.tensors, tensors):
                offset = offsets[cursor]
                nbytes = leaf.numel() * leaf.element_size()
                tensor_specs.append(
                    LeafSpec(
                        offset=offset,
                        nbytes=nbytes,
                        dtype=leaf.dtype,
                        shape=tuple(leaf.shape),
                        role=declared.name,
                    )
                )
                views.append(
                    host[offset:offset + nbytes].view(leaf.dtype).reshape(leaf.shape)
                )
                cursor += 1

            if repoint and module is not None:
                weight_views = views[:binding.weight_leaf_count]
                weight_view = (
                    weight_views[0]
                    if binding.weight_leaf_count == 1
                    else _rebuild_from_leaves(binding.weight_template, iter(weight_views))
                )
                module.weight = torch.nn.Parameter(
                    weight_view,
                    requires_grad=getattr(weight, "requires_grad", False),
                )
                if bias is not None:
                    module.bias = torch.nn.Parameter(
                        views[binding.weight_leaf_count],
                        requires_grad=getattr(bias, "requires_grad", False),
                    )

            specs.append(
                LinearSpec(
                    name=name,
                    tensors=tuple(tensor_specs),
                    execution_key=binding.execution_key,
                    weight_leaf_count=binding.weight_leaf_count,
                    weight_template=binding.weight_template,
                    weight_requires_grad=getattr(weight, "requires_grad", False),
                    bias_requires_grad=(
                        getattr(bias, "requires_grad", False)
                        if bias is not None
                        else False
                    ),
                    substitutions=binding.substitutions,
                )
            )
    except Exception:
        pin_manager.release(pin_handle)
        raise

    pack = BlockPack(
        block_key=block_key,
        host_flat=host,
        linears=tuple(specs),
        required_pin_bytes=int(total),
        pinned=bool(pinned),
        pin_handle=pin_handle,
        owns_flat=True,
        borrowed_from_arena=False,
    )
    pack.view_maker = make_block_view_maker(pack)
    return pack


class ArenaBorrowError(ValueError):
    """A block's live params don't actually live in the flat they were
    expected to borrow from -- caller must fall back to an owned pack."""


def pack_block_host_from_flat(block_key: str, linears, flat: torch.Tensor) -> "BlockPack":
    """Describe declared storage tuples already resident in an owned flat."""
    flat_storage = flat.untyped_storage()
    flat_ptr = flat.data_ptr()
    flat_end = flat_ptr + flat.numel() * flat.element_size()

    def offset_of(leaf: torch.Tensor) -> int:
        if leaf.untyped_storage().data_ptr() != flat_storage.data_ptr():
            raise ArenaBorrowError(f"arena_layout_mismatch:{block_key}:not_in_flat")
        offset = leaf.data_ptr() - flat_ptr
        nbytes = leaf.numel() * leaf.element_size()
        if offset < 0 or offset + nbytes > flat_end - flat_ptr:
            raise ArenaBorrowError(f"arena_layout_mismatch:{block_key}:out_of_range")
        return offset

    specs = []
    for entry in linears:
        if len(entry) == 2:
            name, module = entry
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            name, weight, bias = entry
        binding = linear_storage_binding(weight, bias)
        tensor_specs = []
        for declared in binding.tensors:
            leaf = declared.tensor
            tensor_specs.append(
                LeafSpec(
                    offset=offset_of(leaf),
                    nbytes=leaf.numel() * leaf.element_size(),
                    dtype=leaf.dtype,
                    shape=tuple(leaf.shape),
                    role=declared.name,
                )
            )
        specs.append(
            LinearSpec(
                name=name,
                tensors=tuple(tensor_specs),
                execution_key=binding.execution_key,
                weight_leaf_count=binding.weight_leaf_count,
                weight_template=binding.weight_template,
                weight_requires_grad=getattr(weight, "requires_grad", False),
                bias_requires_grad=(
                    getattr(bias, "requires_grad", False)
                    if bias is not None
                    else False
                ),
                substitutions=binding.substitutions,
            )
        )

    pack = BlockPack(
        block_key=block_key,
        host_flat=flat,
        linears=tuple(specs),
        required_pin_bytes=int(flat.numel() * flat.element_size()),
        pinned=bool(pin_manager.is_host_pinned(flat)),
        pin_handle=None,
        owns_flat=False,
        borrowed_from_arena=True,
    )
    pack.view_maker = make_block_view_maker(pack)
    return pack


class IngraphPackError(RuntimeError):
    """A block pack could not be built or borrowed. ``reasons`` carries the
    stable fail-closed tokens callers surface as ``_ingraph_unavailable_reasons``
    (``non_pinned_pack``, ``unsupported_quant_wrapper``, ``wrapper_pack_missing``,
    ``arena_borrow_required``)."""

    def __init__(self, reasons, message: str = ""):
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(message or ",".join(self.reasons))


@dataclass
class PackBuildResult:
    # Keyed by the model's STABLE block_key string, never a positional index --
    # the caller maps its own indices back locally.
    packs: "dict[str, BlockPack]"
    borrowed: int
    owned: int
    pageable: int  # always 0 on success (a pageable pack raises non_pinned_pack)
    reasons: tuple = ()


def build_or_borrow_block_packs(
    arena,
    entries_by_block: dict,
    *,
    repoint: bool = False,
    pin_mechanism: str = "register",
    allow_owned_fallback: bool = True,
) -> PackBuildResult:
    """Borrow each block's pack from the pinned arena, else build an owned one.

    The single place the in-graph pack policy lives, shared by every model's
    ``enable_ingraph_sampling`` / ``enable_ingraph_training`` glue (see the
    "in-graph arena protocol" in ``pinned_arena``). Nothing here knows about any
    particular model: ``entries_by_block`` maps a stable ``block_key`` to that
    block's ``(name, module)`` linear entries, and ``arena`` is duck-typed (any
    object exposing ``try_borrow_pack(block_key, entries)``), so this module
    never imports ``pinned_arena`` -- which imports it.

    Policy, centralized so callers cannot re-implement it inconsistently:

    * Borrow when the arena holds a current, pinned flat for the block: zero
      alloc, zero copy, no second pin of the same bytes.
    * Otherwise build an owned pack, but only if ``allow_owned_fallback``.
      Under strict pinned-arena validation the caller passes False so a silent
      fall back to owned packs cannot make a run "pass" without proving a
      single borrow.
    * Every streamed pack must be pinned; a pageable one fails the whole set
      closed (``non_pinned_pack``) -- strict in-graph is all-or-nothing.
    * On any failure, release ONLY packs we own. ``release_pack`` no-ops on a
      borrowed pack (``owns_flat=False``), so the arena's flats are never freed
      out from under it.
    """
    packs: "dict[str, BlockPack]" = {}
    borrowed = 0
    owned = 0
    try:
        for block_key, raw_entries in entries_by_block.items():
            entries = list(raw_entries)
            pack = arena.try_borrow_pack(block_key, entries) if arena is not None else None
            if pack is not None:
                borrowed += 1
            else:
                if not allow_owned_fallback:
                    raise IngraphPackError(
                        ("arena_borrow_required",),
                        f"arena_borrow_required: block {block_key!r} is not "
                        "borrowable from the pinned arena",
                    )
                try:
                    pack = pack_block_host(
                        block_key,
                        entries,
                        repoint=repoint,
                        pin_mechanism=pin_mechanism,
                    )
                except ValueError as error:
                    message = str(error)
                    reason = (
                        "wrapper_pack_missing"
                        if "wrapper packing" in message
                        else "unsupported_quant_wrapper"
                    )
                    raise IngraphPackError(
                        (reason,), f"{reason} ({message})"
                    ) from error
                owned += 1
            packs[block_key] = pack
        if any(not pack.pinned for pack in packs.values()):
            raise IngraphPackError(("non_pinned_pack",))
    except BaseException:
        for pack in packs.values():
            release_pack(pack)
        raise
    return PackBuildResult(packs=packs, borrowed=borrowed, owned=owned, pageable=0)


def is_streamed_module(module) -> bool:
    """The memory manager's marker for "this Linear's weights live on the host
    and are fetched per call".

    Read it BEFORE stripping compile contaminants -- the strip deletes the
    attribute, after which every leaf looks resident.
    """
    return hasattr(module, "_layer_memory_manager")


def resident_linear_tensors(module) -> tuple:
    """Return a Linear's declared opaque storage tuple in stable order."""
    binding = linear_storage_binding(module.weight, getattr(module, "bias", None))
    return tuple(item.tensor for item in binding.tensors)


@dataclass(frozen=True)
class BlockLeafPlan:
    """Where each of a block's Linear leaves gets its weights this phase.

    The pack is a transfer-coalescing device (one H2D for N leaves), NOT a
    residency decision. The memory planner splits residency per-Linear, so a
    block is routinely part streamed / part resident. ``sources`` records, in
    the caller's canonical leaf order, whether each leaf reads from the fetched
    flat (``(True, i)`` -> ``streamed_views[i]``) or straight off its resident
    Parameter (``(False, i)`` -> ``resident_args[i]``). Both are trace-time
    constants, so the compiled block specializes on its residency pattern.

    ``pack is None`` means every leaf is resident: no flat, no fetch, no token.
    """

    block_key: str
    pack: "BlockPack | None"
    sources: tuple
    resident_args: tuple
    borrowed_from_arena: bool = False

    @property
    def streams(self) -> bool:
        return self.pack is not None


def assemble_leaf_args(plan: BlockLeafPlan, streamed_views: tuple = ()) -> tuple:
    """Interleave fetched views and resident Parameters back into the block's
    canonical leaf order. Pure Python over trace-time constants."""
    return tuple(
        streamed_views[index] if from_pack else plan.resident_args[index]
        for from_pack, index in plan.sources
    )


@dataclass
class BlockPlanResult:
    plans: "dict[str, BlockLeafPlan]"
    borrowed: int
    owned: int
    fully_resident: int
    streamed_leaves: int
    resident_leaves: int
    reasons: tuple = ()


def build_block_leaf_plans(
    arena,
    entries_by_block: dict,
    *,
    is_streamed=is_streamed_module,
    repoint: bool = False,
    pin_mechanism: str = "register",
    allow_owned_fallback: bool = True,
) -> BlockPlanResult:
    """Plan every block's leaves, packing only the ones the manager streams.

    ``entries_by_block`` maps a stable ``block_key`` to that block's FULL
    ``(name, module)`` leaf list in canonical order. This splits each block by
    ``is_streamed``, builds/borrows a pack over the streamed subset only, and
    reads the resident leaves straight off their Parameters.

    Packing only the streamed subset is what lets the trunk coexist with the
    planner's per-Linear residency: a partially-resident block yields a smaller
    flat (so a smaller prefetch ring) and skips the fetch entirely for leaves
    already on the device. Asking the arena for leaves it never offloaded is
    what produced ``borrow refused: stale_modules=3/8``.
    """
    streamed_by_block: dict = {}
    for block_key, raw_entries in entries_by_block.items():
        streamed = [(name, module) for name, module in raw_entries if is_streamed(module)]
        if streamed:
            streamed_by_block[block_key] = streamed

    result = build_or_borrow_block_packs(
        arena,
        streamed_by_block,
        repoint=repoint,
        pin_mechanism=pin_mechanism,
        allow_owned_fallback=allow_owned_fallback,
    )
    try:
        plans: "dict[str, BlockLeafPlan]" = {}
        streamed_leaves = 0
        resident_leaves = 0
        for block_key, raw_entries in entries_by_block.items():
            pack = result.packs.get(block_key)
            stream_index = {
                name: index
                for index, (name, _) in enumerate(streamed_by_block.get(block_key, ()))
            }
            sources = []
            resident_args = []
            for name, module in raw_entries:
                index = stream_index.get(name)
                if index is not None:
                    sources.append((True, index))
                    streamed_leaves += 1
                    continue
                try:
                    tensors = resident_linear_tensors(module)
                except ValueError as error:
                    raise IngraphPackError(
                        ("unsupported_quant_wrapper",),
                        f"unsupported_quant_wrapper ({block_key}.{name}: {error})",
                    ) from error
                sources.append((False, len(resident_args)))
                resident_args.append(tensors)
                resident_leaves += 1
            plans[block_key] = BlockLeafPlan(
                block_key=block_key,
                pack=pack,
                sources=tuple(sources),
                resident_args=tuple(resident_args),
                borrowed_from_arena=bool(pack is not None and pack.borrowed_from_arena),
            )
    except BaseException:
        for pack in result.packs.values():
            release_pack(pack)
        raise
    return BlockPlanResult(
        plans=plans,
        borrowed=result.borrowed,
        owned=result.owned,
        fully_resident=sum(1 for plan in plans.values() if not plan.streams),
        streamed_leaves=streamed_leaves,
        resident_leaves=resident_leaves,
    )


def _flat_view(
    flat: torch.Tensor,
    offset: int,
    nbytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return flat[offset:offset + nbytes].view(dtype).reshape(shape)


def _flat_clone_view(
    flat: torch.Tensor,
    offset: int,
    nbytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return flat[offset:offset + nbytes].clone().view(dtype).reshape(shape)

def leaf_view(flat: torch.Tensor, spec: LeafSpec) -> torch.Tensor:
    return _flat_view(flat, spec.offset, spec.nbytes, spec.dtype, spec.shape)


def block_storage_views(
    flat: torch.Tensor,
    pack: BlockPack,
) -> dict[str, LayerStorageView]:
    """Return opaque ordered tensor views without binding execution."""
    out = {}
    for spec in pack.linears:
        tensors = tuple(leaf_view(flat, item) for item in spec.tensors)
        out[spec.name] = LayerStorageView(
            spec=spec,
            tensors=tensors,
        )
    return out


def make_block_view_maker(pack: BlockPack):
    """Return a flat-buffer view maker that yields only tensor tuples."""
    entries = []
    for spec in pack.linears:
        entries.append(tuple(
            (item.offset, item.nbytes, item.dtype, item.shape)
            for item in spec.tensors
        ))
    entries = tuple(entries)

    def view_maker(flat: torch.Tensor, _entries=entries):
        out = []
        for tensor_entries in _entries:
            tensors = []
            for index, item in enumerate(tensor_entries):
                view = _flat_view(flat, item[0], item[1], item[2], item[3])
                tensors.append(view if index == 0 else view.clone())
            out.append(tuple(tensors))
        return tuple(out)

    return view_maker


def block_tensor_views(flat: torch.Tensor, pack: BlockPack) -> tuple:
    maker = pack.view_maker
    if maker is None:
        maker = make_block_view_maker(pack)
        pack.view_maker = maker
    return maker(flat)




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


# Compatibility for legacy manager/residency imports during extraction.
_flatten_leaves = flatten_leaves


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
