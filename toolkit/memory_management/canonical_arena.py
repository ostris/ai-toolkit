"""Canonical host arena for immutable frozen base weights.

One page-exclusive pinned host flat per block, immutable leaf metadata, and
a ONE-TIME repoint of frozen base Parameters into views over those flats
(the canonical storage invariant). This is deliberately NOT the legacy
``pinned_arena.py``:
no generation counter, no ``is_current``/staleness oracle over live module
storage, no invalidate/restore/rebuild path, no borrowed-vs-owned pack
taxonomy. Promotion, demotion, and sampling
transitions must never repoint a Parameter again once ``canonicalize()``
has run -- that is the job of the residency sidecars, not this
module.

Construction is destination-first and transactional: preparation allocates
the final host flats without mutating Parameters, population fills those flats,
and commit publishes all Parameter views atomically or restores the originals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.arena_offload.layout import BlockPack, LinearSpec, release_pack


ARENA_KIND = "weights"


class CanonicalArenaError(ValueError):
    """A build, canonicalize, or guard operation violated an arena invariant."""


def _entry_module_and_leaves(entry):
    if len(entry) != 2 or entry[1] is None:
        raise CanonicalArenaError(
            "canonical_arena_requires_module_entries: blocks must be built "
            "from (name, module) entries so Parameters can be repointed"
        )
    name, module = entry
    weight = module.weight
    bias = getattr(module, "bias", None)
    return name, module, weight, bias


def _assert_entries_frozen(entries) -> None:
    """Require frozen base weights before any Parameter is repointed.

    LoRA and adapter parameters remain ordinary trainable state outside the
    arena. Checking the full batch first prevents partial canonicalization when
    any managed leaf is trainable.
    """
    for entry in entries:
        name, _module, weight, bias = _entry_module_and_leaves(entry)
        if getattr(weight, "requires_grad", False):
            raise CanonicalArenaError(f"canonical_arena_trainable_leaf:{name}:weight")
        if bias is not None and getattr(bias, "requires_grad", False):
            raise CanonicalArenaError(f"canonical_arena_trainable_leaf:{name}:bias")


@dataclass(frozen=True)
class BlockRecord:
    """One block's immutable canonical host representation.

    No residency state, no generation counter, no live-module currentness
    test -- packs are views over arena records.
    """

    block_key: str
    pack: BlockPack
    leaf_names: tuple[str, ...]
    modules: tuple[torch.nn.Module, ...]

    def module_for_leaf(self, leaf_name: str) -> torch.nn.Module:
        try:
            return self.modules[self.leaf_names.index(leaf_name)]
        except ValueError as error:
            raise KeyError(
                f"no leaf named {leaf_name!r} in block {self.block_key!r}"
            ) from error

    @property
    def host_flat(self) -> torch.Tensor:
        return self.pack.host_flat

    @property
    def committed_bytes(self) -> int:
        return self.pack.required_pin_bytes

    def leaf_spec(self, leaf_name: str) -> LinearSpec:
        for spec in self.pack.linears:
            if spec.name == leaf_name:
                return spec
        raise KeyError(f"no leaf named {leaf_name!r} in block {self.block_key!r}")


@dataclass
class CanonicalArenaStats:
    blocks: int = 0
    pinned_bytes: int = 0


class CanonicalArena:
    """Owns one process's canonical, page-exclusive pinned host blocks.

    Not thread-safe; ``canonicalize()``/``release()`` are expected to run on
    the same thread that drives model attach/detach, exactly once each per
    instance -- construct a new ``CanonicalArena`` rather than rebuilding.
    """

    def __init__(self) -> None:
        self._blocks: dict[str, BlockRecord] = {}
        self._canonicalized = False

    @property
    def canonicalized(self) -> bool:
        return self._canonicalized

    # -- canonicalize -------------------------------------------------------

    def canonicalize(
        self, entries_by_block: dict, *, kind: str = ARENA_KIND
    ) -> CanonicalArenaStats:
        """Build every block's canonical flat and repoint its Parameters,
        exactly once. ``entries_by_block`` maps ``block_key -> iterable of
        (name, module)`` entries, all frozen (``requires_grad=False``).

        Caller sequencing responsibility (this method cannot see it): run
        AFTER load/quantize/freeze and BEFORE LoRA attach, optimizer
        construction, or compile -- once Parameters are
        repointed here, nothing may replace them again for the life of the
        arena.

        Fails loud and releases any blocks already built in this call if
        any block cannot be admitted (trainable leaf, unsupported quant
        wrapper, or pin budget exceeded) -- there is no silent pageable
        fallback in this arena (that is an admission-policy decision for
        the caller, not a mechanism this class provides).
        """
        if self._canonicalized:
            raise CanonicalArenaError(
                "canonical_arena_double_canonicalize: canonicalize() may "
                "only run once per arena instance -- runtime "
                "promotion/demotion/sampling transitions must never "
                "repoint a Parameter again"
            )
        # Validate every block's leaves are frozen BEFORE building any of
        # them, so a trainable leaf anywhere fails closed with zero
        # Parameters repointed (see _assert_entries_frozen).
        normalized: dict[str, list] = {}
        for block_key, raw_entries in entries_by_block.items():
            entries = list(raw_entries)
            _assert_entries_frozen(entries)
            normalized[block_key] = entries

        from toolkit.memory_management.arena_offload.construction import PreparedCanonicalBuild

        build = PreparedCanonicalBuild(self, normalized, kind=kind)
        build.populate_from_model()
        return build.commit()

    def prepare(
        self,
        entries_by_block: dict,
        *,
        model=None,
        kind: str = ARENA_KIND,
        pin_on_finish: bool = True,
    ):
        """Prepare final destinations without mutating model Parameters."""
        from toolkit.memory_management.arena_offload.construction import PreparedCanonicalBuild

        normalized = {key: list(entries) for key, entries in entries_by_block.items()}
        for entries in normalized.values():
            _assert_entries_frozen(entries)
        return PreparedCanonicalBuild(
            self,
            normalized,
            model=model,
            kind=kind,
            pin_on_finish=pin_on_finish,
        )

    # -- whole-model .to() interception ------------------------------------

    @staticmethod
    def guard_whole_model_to(model: torch.nn.Module) -> None:
        """Route whole-model movement through the published arena runtime."""
        if getattr(model, "_mm_canonical_to_guarded", False):
            return

        originals = {
            name: getattr(model, name) for name in ("to", "cuda", "cpu")
        }

        def runtime_authority():
            runtime = getattr(model, "_arena_offload_runtime", None)
            if runtime is None:
                raise CanonicalArenaError(
                    "canonical_arena_whole_model_move_before_runtime"
                )
            return runtime

        def _guarded_to(*args, **kwargs):
            device, dtype, _non_blocking, memory_format = (
                torch._C._nn._parse_to(*args, **kwargs)
            )
            return runtime_authority().handle_whole_model_move(
                device, dtype=dtype, memory_format=memory_format
            )

        def _guarded_cuda(device=None):
            runtime = runtime_authority()
            return runtime.handle_whole_model_move(
                runtime.device if device is None else device
            )

        def _guarded_cpu():
            return runtime_authority().handle_whole_model_move("cpu")

        model.to = _guarded_to
        model.cuda = _guarded_cuda
        model.cpu = _guarded_cpu
        model._mm_canonical_to_guarded = True
        model._mm_canonical_movement_originals = originals

    @staticmethod
    def unguard_whole_model_to(model: torch.nn.Module) -> None:
        originals = getattr(model, "_mm_canonical_movement_originals", None)
        if originals is not None:
            for name, original in originals.items():
                setattr(model, name, original)
            del model._mm_canonical_movement_originals
        if hasattr(model, "_mm_canonical_to_guarded"):
            del model._mm_canonical_to_guarded

    # -- introspection ------------------------------------------------------

    def block_record(self, block_key: str) -> BlockRecord | None:
        return self._blocks.get(block_key)

    def block_pack(self, block_key: str) -> BlockPack | None:
        record = self._blocks.get(block_key)
        return None if record is None else record.pack

    def block_keys(self) -> tuple[str, ...]:
        return tuple(self._blocks.keys())

    def committed_pinned_bytes(self) -> int:
        return sum(
            record.committed_bytes
            for record in self._blocks.values()
            if record.pack.pinned
        )

    def pinned_block_keys(self) -> frozenset[str]:
        return frozenset(
            block_key
            for block_key, record in self._blocks.items()
            if record.pack.pinned
        )

    def pin_block(self, block_key: str, *, required: bool = True, device=None) -> bool:
        """Register one populated canonical flat without replacing its storage."""
        record = self._blocks.get(str(block_key))
        if record is None:
            raise CanonicalArenaError(f"unknown_canonical_block:{block_key}")
        if record.pack.pinned:
            return False
        handle = pin_manager.pin_register_commit(
            record.host_flat,
            record.committed_bytes,
            ARENA_KIND,
            device=device,
            required=required,
        )
        if not handle.pinned:
            return False
        record.pack.pin_handle = handle
        record.pack.pinned = True
        return True

    def unpin_block(self, block_key: str) -> bool:
        """Unregister one canonical flat while retaining its populated bytes."""
        record = self._blocks.get(str(block_key))
        if record is None:
            raise CanonicalArenaError(f"unknown_canonical_block:{block_key}")
        if not record.pack.pinned:
            return False
        pin_manager.release(record.pack.pin_handle)
        record.pack.pin_handle = None
        record.pack.pinned = False
        return True

    def stats(self) -> CanonicalArenaStats:
        return CanonicalArenaStats(
            blocks=len(self._blocks), pinned_bytes=self.committed_pinned_bytes()
        )

    def immutable_signature(self) -> tuple:
        """Return this arena's immutable host-storage commitment.

        Process-wide pin-ledger state is deliberately excluded: unrelated
        consumers and canonical registration policy may grow or shrink while
        residency sidecars change without mutating canonical host storage.
        """
        return (
            self._canonicalized,
            tuple(
                (
                    block_key,
                    record.host_flat.data_ptr(),
                    record.committed_bytes,
                    pin_manager.is_arena_backed(record.host_flat),
                )
                for block_key, record in self._blocks.items()
            ),
        )

    # -- explicit unload ------------------------------------------------

    def release(self) -> None:
        """Release every block pin and every committed byte explicitly.

        ``pin_manager`` is the sole pin authority. Safe to call on a partially
        built or already released arena.
        """
        first_error = None
        for block_key, record in tuple(self._blocks.items()):
            try:
                release_pack(record.pack)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                continue
            pin_manager.unregister_arena_storage(record.pack.host_flat)
            self._blocks.pop(block_key, None)
        self._canonicalized = bool(self._blocks)
        if first_error is not None:
            raise first_error
