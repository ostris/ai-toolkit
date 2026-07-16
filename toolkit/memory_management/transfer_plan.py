"""Static multi-range transfer plans.

A ``BlockTransferPlan`` says which byte ranges of a canonical block's host
flat (``canonical_arena.BlockRecord``) need to move to the device for one
residency phase, coalesced and packed into a compact destination layout.
Pure data model + coalescing algorithm live here (no CUDA needed to build
or inspect a plan); the Arena transfer runtime submits the copies through
``mm::fetch_start_multi`` while preserving single-ticket ring/backpressure
semantics.

Plans are immutable and fingerprintable: two plans built from the same block
and the same set of streamed leaf names always produce identical ranges and
the same fingerprint, independent of the tensor or storage identity behind the
block's host flat at build time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType

import torch

from toolkit.memory_management.canonical_arena import BlockRecord
from toolkit.memory_management.arena_offload.layout import LEAF_ALIGN



class TransferPlanError(ValueError):
    """A transfer plan could not be built from a block's canonical layout."""


@dataclass(frozen=True)
class LeafRange:
    """One coalesced source(host) -> destination(compact device) copy span."""

    src_offset: int
    dst_offset: int
    nbytes: int


@dataclass(frozen=True)
class CompactLeafSpec:
    """A streamed leaf's location within the plan's compact destination
    buffer -- the offset is
    in DESTINATION (compact device buffer) coordinates, not the arena
    flat's."""

    dst_offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    role: str


@dataclass(frozen=True)
class BlockTransferPlan:
    block_key: str
    ranges: tuple[LeafRange, ...]
    compact_nbytes: int
    # leaf_name -> {declared tensor name: CompactLeafSpec}
    leaf_specs: dict
    streamed_leaf_names: tuple[str, ...]
    fully_streamed: bool
    fingerprint: str

    @property
    def num_ranges(self) -> int:
        return len(self.ranges)

    def compact_leaf_view(self, device_flat: torch.Tensor, leaf_name: str, role: str) -> torch.Tensor:
        spec = self.leaf_specs[leaf_name][role]
        return (
            device_flat[spec.dst_offset:spec.dst_offset + spec.nbytes]
            .view(spec.dtype)
            .reshape(spec.shape)
        )

    def ranges_tensor(self) -> torch.Tensor:
        """(N, 3) int64 CPU tensor of [src_offset, dst_offset, nbytes] rows,
        matching the ``mm::fetch_start_multi`` custom op's argument shape."""
        if not self.ranges:
            return torch.empty((0, 3), dtype=torch.int64)
        return torch.tensor(
            [[r.src_offset, r.dst_offset, r.nbytes] for r in self.ranges],
            dtype=torch.int64,
        )


def _leaf_items(block: BlockRecord, streamed_leaf_names: frozenset):
    """(src_offset, nbytes, leaf_name, role, LeafSpec) for every streamed
    leaf's declared storage tensors, sorted by source offset."""
    items = []
    for spec in block.pack.linears:
        if spec.name not in streamed_leaf_names:
            continue
        for leaf_spec in spec.tensors:
            role = leaf_spec.role
            items.append((leaf_spec.offset, leaf_spec.nbytes, spec.name, role, leaf_spec))
    items.sort(key=lambda item: item[0])
    for (off, n, *_), (noff, *_rest) in pairwise(items):
        if noff < off + n:
            raise TransferPlanError(
                f"transfer_plan_overlap:{block.block_key}: leaf ranges overlap "
                "-- canonical arena layout invariant violated"
            )
    return items


def build_transfer_plan(
    block: BlockRecord,
    streamed_leaf_names,
    *,
    slack_bytes: int = LEAF_ALIGN - 1,
) -> BlockTransferPlan:
    """Build a coalesced multi-range transfer plan for the STREAMED subset
    of ``block``'s leaves. ``streamed_leaf_names`` is the residency
    decision owned by the planner and controllers; this function only turns
    "which leaves stream this phase" into "which byte ranges to copy and
    where."

    Two source items coalesce into one range when the gap between them is
    at most ``slack_bytes`` (default: one leaf's alignment padding,
    ``LEAF_ALIGN - 1``) -- adjacent streamed leaves separated only by
    padding merge into a single copy; a resident (non-streamed) leaf in
    between breaks the run. Coalescing never changes a leaf's own
    destination offset formula (range-relative), only how many discrete
    copies the runtime submits.
    """
    streamed = frozenset(streamed_leaf_names)
    known = frozenset(spec.name for spec in block.pack.linears)
    unknown = streamed - known
    if unknown:
        raise TransferPlanError(
            f"transfer_plan_unknown_leaf:{block.block_key}:{sorted(unknown)[0]}"
        )
    if slack_bytes < 0:
        raise TransferPlanError("transfer_plan_slack_bytes must be non-negative")
    items = _leaf_items(block, streamed)
    if not items:
        raise TransferPlanError(f"transfer_plan_empty:{block.block_key}: no streamed leaves")

    ranges: list[LeafRange] = []
    leaf_specs: dict[str, dict[str, CompactLeafSpec]] = {}
    compact_total = 0

    def flush(src_start: int, src_end: int, dst_start: int, pending: list) -> int:
        length = src_end - src_start
        ranges.append(LeafRange(src_offset=src_start, dst_offset=dst_start, nbytes=length))
        for off, n, name, role, leaf_spec in pending:
            leaf_specs.setdefault(name, {})[role] = CompactLeafSpec(
                dst_offset=dst_start + (off - src_start),
                nbytes=n,
                dtype=leaf_spec.dtype,
                shape=leaf_spec.shape,
                role=leaf_spec.role,
            )
        return length

    src_start, first_len = items[0][0], items[0][1]
    src_end = src_start + first_len
    dst_start = 0
    pending = [items[0]]
    for item in items[1:]:
        off, n = item[0], item[1]
        gap = off - src_end
        if gap <= slack_bytes:
            src_end = off + n
            pending.append(item)
        else:
            compact_total += flush(src_start, src_end, dst_start, pending)
            src_start, src_end, dst_start, pending = off, off + n, compact_total, [item]
    compact_total += flush(src_start, src_end, dst_start, pending)

    all_leaf_names = tuple(spec.name for spec in block.pack.linears)
    fully_streamed = streamed == frozenset(all_leaf_names)
    frozen_leaf_specs = MappingProxyType(
        {
            name: MappingProxyType(dict(roles))
            for name, roles in leaf_specs.items()
        }
    )

    fp_source = repr(
        (
            block.block_key,
            tuple(sorted(streamed)),
            tuple((r.src_offset, r.dst_offset, r.nbytes) for r in ranges),
            tuple(
                (
                    name,
                    tuple(
                        (role, spec.dst_offset, spec.nbytes, str(spec.dtype), spec.shape)
                        for role, spec in sorted(roles.items())
                    ),
                )
                for name, roles in sorted(frozen_leaf_specs.items())
            ),
        )
    )
    fingerprint = hashlib.sha1(fp_source.encode("utf-8")).hexdigest()[:16]

    return BlockTransferPlan(
        block_key=block.block_key,
        ranges=tuple(ranges),
        compact_nbytes=compact_total,
        leaf_specs=frozen_leaf_specs,
        streamed_leaf_names=tuple(sorted(streamed)),
        fully_streamed=fully_streamed,
        fingerprint=fingerprint,
    )
