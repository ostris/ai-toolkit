"""Manager-owned device residency sidecars for the immutable host arena.

Residency changes never mutate a Parameter or the canonical host allocation.
They publish optional device tensors keyed by stable ``(block, leaf)`` keys;
execution adapters choose a sidecar or a compact fetched view from a static
transfer plan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.arena_offload.layout import (
    flatten_leaves,
    rebuild_from_leaves,
    typed_view,
)

LeafKey = tuple[str, str]
DEFAULT_DEMOTION_PIN_RESERVE_BLOCKS = 2

def _interleave_priority(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    bits = (count - 1).bit_length()
    value = index
    reversed_bits = 0
    for _ in range(bits):
        reversed_bits = (reversed_bits << 1) | (value & 1)
        value >>= 1
    return reversed_bits / float(1 << bits)
class ResidencyError(RuntimeError):
    """A residency plan or transition violated an immutable-arena invariant."""


@dataclass(frozen=True)
class ResidencyPlan:
    phase: str
    resident_leaf_keys: frozenset[LeafKey]
    fingerprint: str
    @classmethod
    def fit_whole_blocks(
        cls,
        arena: CanonicalArena,
        resident_budget_bytes: int,
        *,
        phase: str,
        prefer_resident_keys=(),
    ) -> ResidencyPlan:
        """Select complete canonical blocks within a sidecar-only byte budget.

        Existing fully resident blocks are preferred to avoid needless sidecar
        churn when a sampling plan grows or shrinks between images.
        """
        budget = max(0, int(resident_budget_bytes))
        preferred = frozenset(
            (str(block), str(leaf))
            for block, leaf in prefer_resident_keys
        )

        blocks = []
        block_keys = arena.block_keys()

        for order, block_key in enumerate(block_keys):
            record = arena.block_record(block_key)
            leaf_keys = tuple(
                (block_key, leaf_name)
                for leaf_name in record.leaf_names
            )
            already_full = all(key in preferred for key in leaf_keys)

            blocks.append(
                {
                    "order": order,
                    "block_key": block_key,
                    # Slightly conservative because this includes page rounding.
                    "nbytes": int(record.committed_bytes),
                    "leaf_keys": leaf_keys,
                    "already_full": already_full,
                }
            )

        blocks.sort(
            key=lambda item: (
                not item["already_full"],
                item["nbytes"],
                _interleave_priority(item["order"], len(blocks)),
            )
        )

        selected = []
        used = 0

        for item in blocks:
            nbytes = item["nbytes"]
            if used + nbytes > budget:
                continue
            selected.extend(item["leaf_keys"])
            used += nbytes

        return cls.build(phase, selected)
    @classmethod
    def build(cls, phase: str, resident_leaf_keys) -> ResidencyPlan:
        keys = frozenset((str(block), str(leaf)) for block, leaf in resident_leaf_keys)
        source = repr((str(phase), tuple(sorted(keys))))
        fingerprint = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
        return cls(str(phase), keys, fingerprint)

    @classmethod
    def from_smart_plan(
        cls, arena: CanonicalArena, smart_plan: dict, *, phase: str
    ) -> ResidencyPlan:
        """Adapt the existing planner's ``offload_ids`` decision to sidecars.

        This is the Slice 3 planner seam: priority and capacity remain owned by
        ``MemoryManager.smart_training_plan``; only the mutation target changes.
        """
        offload_ids = set(smart_plan.get("offload_ids", ()))
        resident = []
        for block_key in arena.block_keys():
            block = arena.block_record(block_key)
            block_is_fully_resident = all(
                id(module) not in offload_ids
                for module in block.modules
            )
            if not block_is_fully_resident:
                continue
            resident.extend(
                (block_key, leaf_name)
                for leaf_name in block.leaf_names
            )
        return cls.build(phase, resident)

    def resident_in_block(self, block_key: str) -> frozenset[str]:
        return frozenset(leaf for block, leaf in self.resident_leaf_keys if block == block_key)


@dataclass(frozen=True)
class ResidentLeaf:
    key: LeafKey
    tensors: tuple[torch.Tensor, ...]
    weight_leaf_count: int
    weight_template: torch.Tensor
    ready_event: torch.cuda.Event | None
    nbytes: int

    @property
    def weight(self):
        weight_tensors = self.tensors[:self.weight_leaf_count]
        if self.weight_leaf_count == 1:
            return weight_tensors[0]
        return rebuild_from_leaves(self.weight_template, weight_tensors)

    @property
    def bias(self):
        if len(self.tensors) == self.weight_leaf_count:
            return None
        return self.tensors[self.weight_leaf_count]


@dataclass(frozen=True)
class ResidencyDelta:
    promoted: tuple[LeafKey, ...]
    demoted: tuple[LeafKey, ...]
    resident_bytes: int


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return sum(leaf.numel() * leaf.element_size() for leaf in flatten_leaves(tensor))


def _record_stream(tensor: torch.Tensor | None, stream) -> None:
    if tensor is None:
        return
    for leaf in flatten_leaves(tensor):
        leaf.record_stream(stream)


class ResidencyState:
    """Atomic per-Linear sidecar state over one immutable canonical arena."""

    def __init__(self, arena: CanonicalArena, device) -> None:
        if not arena.canonicalized:
            raise ResidencyError("residency_requires_canonicalized_arena")
        self.arena = arena
        self.device = torch.device(device)
        self._sidecars: dict[LeafKey, ResidentLeaf] = {}
        self._plan = ResidencyPlan.build("empty", ())
        self._copy_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    @property
    def plan(self) -> ResidencyPlan:
        return self._plan

    def _all_keys(self) -> frozenset[LeafKey]:
        return frozenset(
            (block_key, leaf_name)
            for block_key in self.arena.block_keys()
            for leaf_name in self.arena.block_record(block_key).leaf_names
        )

    def _canonical_leaf(self, key: LeafKey):
        block_key, leaf_name = key
        block = self.arena.block_record(block_key)
        if block is None:
            raise ResidencyError(f"unknown_residency_block:{block_key}")
        try:
            spec = block.leaf_spec(leaf_name)
            module = block.module_for_leaf(leaf_name)
        except KeyError as error:
            raise ResidencyError(f"unknown_residency_leaf:{block_key}.{leaf_name}") from error
        return block, spec, module

    def _build_sidecar(self, key: LeafKey) -> ResidentLeaf:
        block, spec, _module = self._canonical_leaf(key)
        non_blocking = (
            self.device.type == "cuda"
            and pin_manager.is_host_pinned(block.host_flat)
        )
        stream_context = (
            torch.cuda.stream(self._copy_stream)
            if self._copy_stream is not None
            else torch.no_grad()
        )
        with torch.no_grad(), stream_context:
            tensors = tuple(
                typed_view(block.host_flat, item).to(
                    self.device, non_blocking=non_blocking
                )
                for item in spec.tensors
            )
            event = None
            if self._copy_stream is not None:
                event = torch.cuda.Event()
                event.record(self._copy_stream)
        return ResidentLeaf(
            key=key,
            tensors=tensors,
            weight_leaf_count=spec.weight_leaf_count,
            weight_template=spec.weight_template,
            ready_event=event,
            nbytes=sum(_tensor_bytes(tensor) for tensor in tensors),
        )

    def reconcile(self, plan: ResidencyPlan) -> ResidencyDelta:
        desired = plan.resident_leaf_keys
        unknown = desired - self._all_keys()
        if unknown:
            block, leaf = sorted(unknown)[0]
            raise ResidencyError(f"unknown_residency_leaf:{block}.{leaf}")

        before_arena = self.arena.immutable_signature()
        current = set(self._sidecars)
        additions = tuple(sorted(desired - current))
        removals = tuple(sorted(current - desired))
        pending: dict[LeafKey, ResidentLeaf] = {}
        try:
            for key in additions:
                pending[key] = self._build_sidecar(key)
        except Exception as error:
            # Copies may already be queued on the private stream. Drain them
            # before pending tensors are released, while leaving published
            # sidecars and the active plan exactly unchanged.
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            pending.clear()
            if self.arena.immutable_signature() != before_arena:
                raise ResidencyError(
                    "pin_ledger_changed_during_failed_promotion"
                ) from error
            raise

        next_sidecars = {
            key: value for key, value in self._sidecars.items() if key not in removals
        }
        next_sidecars.update(pending)
        self._sidecars = next_sidecars
        self._plan = plan
        if self.arena.immutable_signature() != before_arena:
            raise ResidencyError("pin_ledger_changed_during_residency_transition")
        return ResidencyDelta(additions, removals, self.resident_bytes())

    def promote(self, key: LeafKey, *, phase: str | None = None) -> bool:
        normalized = (str(key[0]), str(key[1]))
        if normalized in self._sidecars:
            return False
        plan = ResidencyPlan.build(
            phase or self._plan.phase, self._sidecars.keys() | {normalized}
        )
        self.reconcile(plan)
        return True

    def demote(self, key: LeafKey, *, phase: str | None = None) -> bool:
        normalized = (str(key[0]), str(key[1]))
        if normalized not in self._sidecars:
            return False
        plan = ResidencyPlan.build(
            phase or self._plan.phase, set(self._sidecars) - {normalized}
        )
        self.reconcile(plan)
        return True

    def resident_leaf(self, key: LeafKey) -> ResidentLeaf | None:
        sidecar = self._sidecars.get((str(key[0]), str(key[1])))
        if sidecar is None:
            return None
        if sidecar.ready_event is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(sidecar.ready_event)
            for tensor in sidecar.tensors:
                _record_stream(tensor, current)
        return sidecar

    def resident_tensor(self, key: LeafKey) -> torch.Tensor | None:
        sidecar = self.resident_leaf(key)
        return None if sidecar is None else sidecar.weight

    def streamed_leaf_names(self, block_key: str) -> tuple[str, ...]:
        block = self.arena.block_record(block_key)
        if block is None:
            raise ResidencyError(f"unknown_residency_block:{block_key}")
        resident = self._plan.resident_in_block(block_key)
        return tuple(name for name in block.leaf_names if name not in resident)

    def resident_leaf_bytes(self, key: LeafKey) -> int:
        """Return published sidecar bytes without synchronizing its copy event."""
        sidecar = self._sidecars.get((str(key[0]), str(key[1])))
        return 0 if sidecar is None else int(sidecar.nbytes)

    def resident_bytes(self) -> int:
        return sum(sidecar.nbytes for sidecar in self._sidecars.values())

    def planned_addition_bytes(self, plan: ResidencyPlan) -> int:
        """Exact sidecar payload allocated before an atomic plan is published."""
        additions = plan.resident_leaf_keys - set(self._sidecars)
        total = 0
        for key in additions:
            _block, spec, _module = self._canonical_leaf(key)
            total += sum(int(item.nbytes) for item in spec.tensors)
        return total

    def synchronize_copies(self) -> None:
        """Settle queued promotions before their host sources are unpinned."""
        if self._copy_stream is not None:
            self._copy_stream.synchronize()

    def clear(self, *, phase: str = "clear") -> ResidencyDelta:
        return self.reconcile(ResidencyPlan.build(phase, ()))


def ordered_demotion_block_keys(
    arena: CanonicalArena,
    residency: ResidencyState,
    plan: ResidencyPlan,
    *,
    protected_leaf_keys=(),
) -> tuple[str, ...]:
    """Fully resident blocks in the controller's deterministic demotion order."""
    protected = frozenset(
        (str(block), str(leaf)) for block, leaf in protected_leaf_keys
    )
    candidates = []
    for order, block_key in enumerate(arena.block_keys()):
        record = arena.block_record(block_key)
        leaf_keys = tuple((block_key, name) for name in record.leaf_names)
        if any(key in protected for key in leaf_keys) or not all(
            key in plan.resident_leaf_keys for key in leaf_keys
        ):
            continue
        payload_bytes = sum(
            item.nbytes
            for leaf_name in record.leaf_names
            for item in record.leaf_spec(leaf_name).tensors
        )
        candidates.append(
            (-payload_bytes, order, str(block_key))
        )
    return tuple(block_key for _bytes, _order, block_key in sorted(candidates))


def pin_requirements_for_plan(
    arena: CanonicalArena,
    residency: ResidencyState,
    plan: ResidencyPlan,
    *,
    protected_leaf_keys=(),
    reserve_blocks: int = DEFAULT_DEMOTION_PIN_RESERVE_BLOCKS,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return required streamed pins and optional known demotion reserves."""
    streamed = set()
    for block_key in arena.block_keys():
        record = arena.block_record(block_key)
        if any(
            (block_key, leaf_name) not in plan.resident_leaf_keys
            for leaf_name in record.leaf_names
        ):
            streamed.add(str(block_key))
    ordered = ordered_demotion_block_keys(
        arena,
        residency,
        plan,
        protected_leaf_keys=protected_leaf_keys,
    )
    reserve = ordered[:max(0, int(reserve_blocks))]
    return frozenset(streamed), reserve
