"""Destination-first transactional construction of canonical arena storage."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from toolkit.memory_management import pin_manager
from toolkit.quantization.storage import module_storage_binding, named_tensor_storage
from .layout import (
    BlockPack,
    LeafSpec,
    LinearSpec,
    inspect_block,
    LayerStorageView,
    linear_views,
    make_block_view_maker,
    substitution_views,
    typed_view,
)


class CanonicalBuildError(RuntimeError):
    pass


class CanonicalStateConsumedError(CanonicalBuildError):
    """A direct-load source mapping was mutated before construction failed."""


class CanonicalStateInferenceError(CanonicalBuildError):
    pass


@dataclass(frozen=True)
class _StateDestination:
    key: tuple[str, str, str]
    shape: tuple[int, ...]
    dtype: torch.dtype


def _module_paths(model) -> dict[int, str]:
    if model is None:
        raise CanonicalStateInferenceError("canonical_state_inference_requires_model")
    paths = {}
    duplicates = set()
    try:
        modules = model.named_modules(remove_duplicate=False)
    except TypeError:
        modules = model.named_modules()
    for path, module in modules:
        previous = paths.get(id(module))
        if previous is not None and previous != path:
            duplicates.add(id(module))
        else:
            paths[id(module)] = str(path)
    for identity in duplicates:
        paths.pop(identity, None)
    return paths


def _same_storage_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Whether two tensor objects name the same physical/meta storage view."""
    try:
        same_storage = (
            left.untyped_storage()._cdata == right.untyped_storage()._cdata
        )
    except (AttributeError, RuntimeError):
        return False
    return bool(
        same_storage
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and left.dtype == right.dtype
    )


def infer_state_dict_schema(
    model, entries_by_block, *, allow_unserialized_storage: bool = False
) -> dict:
    """Infer serialized source entries for managed physical destinations.

    Direct model-source construction may include declared execution buffers
    that are intentionally non-persistent. State-dict consumers keep the
    strict default because those destinations cannot be populated from an
    absent serialized source.
    """
    module_paths = _module_paths(model)
    schema = {}
    for block_key, raw_entries in entries_by_block.items():
        block_schema = {}
        for entry_name, module in tuple(raw_entries):
            module_path = module_paths.get(id(module))
            if module_path is None:
                raise CanonicalStateInferenceError(
                    f"canonical_state_module_path_ambiguous:{block_key}:{entry_name}"
                )
            binding = module_storage_binding(module)
            serialized = []
            groups = []
            for relative_key, value in module.state_dict(keep_vars=True).items():
                leaves = named_tensor_storage(value)
                start = len(serialized)
                serialized.extend(leaves)
                groups.append((str(relative_key), start, len(serialized)))
            matched = []
            claimed = set()
            for declared in binding.tensors:
                candidates = [
                    index
                    for index, serialized_leaf in enumerate(serialized)
                    if _same_storage_view(
                        serialized_leaf.tensor, declared.tensor
                    )
                ]
                if not candidates:
                    if allow_unserialized_storage:
                        continue
                    raise CanonicalStateInferenceError(
                        "canonical_state_storage_not_serialized:"
                        f"{block_key}:{entry_name}:{declared.name}"
                    )
                if len(candidates) != 1 or candidates[0] in claimed:
                    raise CanonicalStateInferenceError(
                        "canonical_state_storage_match_ambiguous:"
                        f"{block_key}:{entry_name}:{declared.name}"
                    )
                match = candidates[0]
                matched.append((match, declared))
                claimed.add(match)
            matched_by_index = {index: declared for index, declared in matched}
            for relative_key, start, end in groups:
                selected = [
                    (index, matched_by_index[index])
                    for index in range(start, end)
                    if index in matched_by_index
                ]
                if not selected:
                    continue
                if len(selected) != end - start:
                    raise CanonicalStateInferenceError(
                        f"canonical_state_partial_serialized_entry:{module_path}.{relative_key}"
                    )
                source_key = (
                    f"{module_path}.{relative_key}"
                    if module_path
                    else relative_key
                )
                if source_key in block_schema:
                    raise CanonicalStateInferenceError(
                        f"canonical_state_duplicate_source:{source_key}"
                    )
                destinations = []
                for index, declared in selected:
                    serialized_leaf = serialized[index]
                    tensor = serialized_leaf.tensor
                    if (
                        tuple(tensor.shape) != tuple(declared.tensor.shape)
                        or tensor.dtype != declared.tensor.dtype
                    ):
                        raise CanonicalStateInferenceError(
                            "canonical_state_target_schema_mismatch:"
                            f"{source_key}:{declared.name}"
                        )
                    destinations.append(
                        _StateDestination(
                            (str(block_key), str(entry_name), declared.name),
                            tuple(tensor.shape),
                            tensor.dtype,
                        )
                    )
                block_schema[source_key] = tuple(destinations)
        schema[str(block_key)] = block_schema
    return schema


def validate_state_dict_schema(state_dict, schema) -> None:
    """Prove inference for every managed source before destructive copying."""
    for block_schema in schema.values():
        for source_key, destinations in block_schema.items():
            if source_key not in state_dict:
                raise CanonicalStateInferenceError(
                    f"canonical_state_missing_source:{source_key}"
                )
            leaves = named_tensor_storage(state_dict[source_key])
            if len(leaves) != len(destinations):
                raise CanonicalStateInferenceError(
                    "canonical_state_source_count_mismatch:"
                    f"{source_key}:source={len(leaves)}:expected={len(destinations)}"
                )
            for leaf, destination in zip(leaves, destinations, strict=True):
                if (
                    tuple(leaf.tensor.shape) != destination.shape
                    or leaf.tensor.dtype != destination.dtype
                ):
                    raise CanonicalStateInferenceError(
                        f"canonical_state_source_schema_mismatch:{source_key}"
                    )


@dataclass
class _PreparedBlock:
    key: str
    entries: tuple
    layout: object
    flat: torch.Tensor
    pending: object
    handle: object | None = None
    pack: BlockPack | None = None


class PreparedCanonicalBuild:
    """A prepared arena build whose model publication is atomic."""

    def __init__(
        self,
        arena,
        entries_by_block,
        *,
        model=None,
        kind="weights",
        pin_on_finish: bool = True,
    ):
        self.arena = arena
        self.model = model
        self.kind = kind
        self.pin_on_finish = bool(pin_on_finish)
        self.blocks = []
        self.destinations = {}
        self.entries_by_block = {}
        self.state_schema = {}
        self._originals = []
        self._populated = False
        self._committed = False
        if arena.canonicalized:
            raise CanonicalBuildError("canonical_arena_double_canonicalize")
        try:
            for key, raw_entries in entries_by_block.items():
                self.add_block(key, raw_entries)
        except Exception:
            self.rollback()
            raise

    def add_block(self, key, raw_entries) -> None:
        """Prepare one final block layout for a bounded direct loader."""
        if self._populated or self._committed:
            raise CanonicalBuildError("canonical_build_already_populated")
        if key in self.entries_by_block:
            raise CanonicalBuildError(f"canonical_build_duplicate_block:{key}")
        entries = tuple(raw_entries)
        block_schema = (
            infer_state_dict_schema(
                self.model,
                {str(key): entries},
                allow_unserialized_storage=True,
            )[str(key)]
            if self.model is not None
            else {}
        )
        layout = inspect_block(key, entries)
        for name, module in entries:
            binding = module_storage_binding(module)
            for substitution in binding.substitutions:
                target = substitution.name
                if target in module._parameters:
                    value = module._parameters[target]
                    if value is not None and value.requires_grad:
                        raise CanonicalBuildError(
                            f"canonical_arena_trainable_leaf:{key}:{name}.{target}"
                        )
                    self._originals.append((module, target, "parameter", value))
                elif target in module._buffers:
                    self._originals.append(
                        (module, target, "buffer", module._buffers[target])
                    )
                else:
                    raise CanonicalBuildError(
                        f"canonical_arena_missing_target:{key}:{name}.{target}"
                    )
        flat, padded = pin_manager.pin_register_prepare(layout.nbytes)
        block = _PreparedBlock(key, entries, layout, flat, padded)
        self.blocks.append(block)
        self.entries_by_block[key] = entries
        self.state_schema[str(key)] = block_schema
        for linear in layout.linears:
            for leaf in linear.leaf_descriptors:
                self.destinations[(key, linear.name, leaf.role)] = typed_view(flat, leaf)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self._committed:
            self.rollback()
        return False

    def populate(self, source) -> None:
        try:
            source(self.destinations)
            self._finish_population()
        except Exception:
            self.rollback()
            raise

    def populate_from_model(self) -> None:
        try:
            for destination_key, source in self.model_source_leaves():
                self.destinations[destination_key].copy_(source)
            self._finish_population()
        except Exception:
            self.rollback()
            raise

    def populate_block_from_model(self, block_key: str) -> None:
        """Copy one bounded loaded block without finalizing the whole build."""
        for destination_key, source in self.model_source_leaves(block_key=block_key):
            self.destinations[destination_key].copy_(source)

    def populate_from_state_dict_consuming(
        self,
        state_dict,
        *,
        blocks=None,
    ) -> tuple[str, ...]:
        """Populate final flats one block at a time and consume their sources.

        Source keys and physical leaves are inferred from each managed
        module's own ``state_dict`` surface and storage declaration. When
        ``blocks`` is supplied it yields ``(block_key, entries)`` pairs and
        each block is allocated only when its turn begins. Otherwise the
        already-prepared blocks are populated in their existing order.

        Every fully copied source entry is removed from ``state_dict`` before
        the next block is requested. The mapping is therefore intentionally
        destructive even if a later block fails; callers must discard it on
        failure, while the canonical build still rolls its own resources back.
        """
        if self._populated or self._committed:
            raise CanonicalBuildError("canonical_build_already_populated")
        if blocks is not None and self.blocks:
            raise CanonicalBuildError("canonical_build_blocks_already_prepared")

        consumed = []
        try:
            if blocks is None:
                block_items = ((block.key, None) for block in tuple(self.blocks))
            else:
                block_items = iter(blocks)

            for block_key, entries in block_items:
                if entries is not None:
                    self.add_block(block_key, entries)
                block_schema = self.state_schema.get(str(block_key), {})
                if not block_schema:
                    raise CanonicalBuildError(
                        f"canonical_build_unknown_block:{block_key}"
                    )
                source_keys = tuple(block_schema)
                for source_key in source_keys:
                    self.copy_state_entry(source_key, state_dict[source_key])
                for source_key in source_keys:
                    state_dict.pop(source_key)
                consumed.extend(source_keys)

            self._finish_population()
            return tuple(consumed)
        except Exception as error:
            cleanup_error = None
            try:
                self.rollback()
            except BaseException as rollback_error:
                cleanup_error = rollback_error
            if consumed:
                consumed_error = CanonicalStateConsumedError(
                    "canonical_state_consumed_before_build_failure"
                )
                if cleanup_error is not None:
                    consumed_error.add_note(
                        "canonical rollback also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise consumed_error from error
            if cleanup_error is not None:
                raise cleanup_error from error
            raise

    def copy_state_entry(self, source_key: str, value) -> bool:
        """Copy one serialized state entry when it belongs to this build."""
        destinations = None
        for block_schema in self.state_schema.values():
            destinations = block_schema.get(str(source_key))
            if destinations is not None:
                break
        if destinations is None:
            return False
        leaves = named_tensor_storage(value)
        if len(leaves) != len(destinations):
            raise CanonicalBuildError(
                f"canonical_state_source_count_mismatch:{source_key}"
            )
        for leaf, destination in zip(leaves, destinations, strict=True):
            source = leaf.tensor
            if (
                tuple(source.shape) != destination.shape
                or source.dtype != destination.dtype
            ):
                raise CanonicalBuildError(
                    f"canonical_state_source_schema_mismatch:{source_key}"
                )
            self.destinations[destination.key].copy_(source)
        del source
        return True

    def release_block_sources_to_meta(self, block_key: str) -> None:
        """Drop a direct loader's bounded source after its final flat is populated."""
        block = next(item for item in self.blocks if item.key == block_key)
        meta_flat = torch.empty(block.layout.nbytes, dtype=torch.uint8, device="meta")
        replacements = {}
        modules = dict(block.entries)
        meta_linears = []
        for linear in block.layout.linears:
            module = modules[linear.name]
            for target, value in substitution_views(meta_flat, linear).items():
                requires_grad = (
                    linear.weight_requires_grad
                    if target == "weight"
                    else linear.bias_requires_grad if target == "bias" else False
                )
                replacements[(module, target)] = self._publish_state(
                    module, target, value, requires_grad=requires_grad
                )
            meta_weight = module._parameters.get("weight")
            meta_linears.append(
                replace(
                    linear,
                    weight_template=(
                        meta_weight.data if meta_weight is not None else None
                    ),
                )
            )
        block.layout = replace(block.layout, linears=tuple(meta_linears))
        self._originals = [
            (module, target, kind, replacements.get((module, target), value))
            for module, target, kind, value in self._originals
        ]

    def finish_population(self) -> None:
        try:
            self._finish_population()
        except Exception:
            self.rollback()
            raise

    def storage_views(self, block_key: str) -> tuple[LayerStorageView, ...]:
        """Return populated immutable storage declarations before commit."""
        if not self._populated:
            raise CanonicalBuildError("canonical_build_not_populated")
        try:
            block = next(item for item in self.blocks if item.key == block_key)
        except StopIteration as error:
            raise CanonicalBuildError(
                f"canonical_build_unknown_block:{block_key}"
            ) from error
        return tuple(
            LayerStorageView(
                spec=LinearSpec(
                    name=linear.name,
                    tensors=tuple(
                        LeafSpec(**leaf.__dict__)
                        for leaf in linear.leaf_descriptors
                    ),
                    execution_key=linear.execution_key,
                    weight_leaf_count=linear.weight_leaf_count,
                    weight_template=linear.weight_template,
                    weight_requires_grad=linear.weight_requires_grad,
                    bias_requires_grad=linear.bias_requires_grad,
                    substitutions=linear.substitutions,
                ),
                tensors=tuple(
                    typed_view(block.flat, leaf)
                    for leaf in linear.leaf_descriptors
                ),
            )
            for linear in block.layout.linears
        )

    def model_source_leaves(self, *, block_key: str | None = None):
        """Yield loaded model leaves keyed exactly like final destinations."""
        for block in self.blocks:
            if block_key is not None and block.key != block_key:
                continue
            by_name = dict(block.entries)
            for linear in block.layout.linears:
                module = by_name[linear.name]
                binding = module_storage_binding(module)
                for descriptor, declared in zip(
                    linear.leaf_descriptors, binding.tensors, strict=True
                ):
                    yield (
                        block.key,
                        linear.name,
                        descriptor.role,
                    ), declared.tensor

    def _finish_population(self) -> None:
        for block in self.blocks:
            if not self.pin_on_finish:
                continue
            handle = pin_manager.pin_register_commit(
                block.flat, block.layout.nbytes, self.kind, required=False
            )
            if not handle.pinned:
                pin_manager.release(handle)
                raise CanonicalBuildError(f"canonical_arena_pin_budget_exceeded:{block.key}")
            block.handle = handle
            block.flat = handle.tensor
            # Validate every supported reconstruction before publication.
            for linear in block.layout.linears:
                if linear.weight_leaf_count:
                    # Preserve the established wrapper-validation seam while
                    # executable substitutions become the source of truth.
                    linear_views(block.flat, linear)
                substitution_views(block.flat, linear)
        self._populated = True

    @staticmethod
    def _publish_state(module, target, value, *, requires_grad=False):
        if target in module._parameters:
            published = torch.nn.Parameter(value, requires_grad=requires_grad)
            setattr(module, target, published)
            return published
        if target in module._buffers:
            setattr(module, target, value)
            return value
        raise CanonicalBuildError(f"canonical_arena_missing_target:{target}")

    def commit(self):
        if not self._populated:
            self.rollback()
            raise CanonicalBuildError("canonical_build_not_populated")
        from toolkit.memory_management.canonical_arena import BlockRecord, CanonicalArenaStats
        published = []
        try:
            for block in self.blocks:
                specs = []
                for linear in block.layout.linears:
                    module = dict(block.entries)[linear.name]
                    for target, value in substitution_views(
                        block.flat, linear
                    ).items():
                        requires_grad = (
                            linear.weight_requires_grad
                            if target == "weight"
                            else linear.bias_requires_grad if target == "bias" else False
                        )
                        self._publish_state(
                            module,
                            target,
                            value,
                            requires_grad=requires_grad,
                        )
                    published.append(module)
                    specs.append(LinearSpec(
                        name=linear.name,
                        tensors=tuple(
                            LeafSpec(**leaf.__dict__)
                            for leaf in linear.leaf_descriptors
                        ),
                        execution_key=linear.execution_key,
                        weight_leaf_count=linear.weight_leaf_count,
                        weight_template=linear.weight_template,
                        weight_requires_grad=linear.weight_requires_grad,
                        bias_requires_grad=linear.bias_requires_grad,
                        substitutions=linear.substitutions,
                    ))
                pack = BlockPack(
                    block.key,
                    block.flat,
                    tuple(specs),
                    block.layout.nbytes,
                    bool(block.handle and block.handle.pinned),
                    pin_handle=block.handle,
                )
                pack.view_maker = make_block_view_maker(pack)
                block.pack = pack
                names = tuple(name for name, _ in block.entries)
                modules = tuple(module for _, module in block.entries)
                self.arena._blocks[block.key] = BlockRecord(block.key, pack, names, modules)
                pin_manager.register_arena_storage(block.flat)
            self.arena._canonicalized = True
            if self.model is not None:
                self.arena.guard_whole_model_to(self.model)
            self._committed = True
            return CanonicalArenaStats(
                len(self.blocks),
                sum(
                    b.layout.nbytes
                    for b in self.blocks
                    if b.handle is not None and b.handle.pinned
                ),
            )
        except Exception:
            self.rollback()
            raise

    def rollback(self) -> None:
        for module, target, kind, value in reversed(self._originals):
            # Bypass user/module publication hooks: rollback must remain valid
            # even when the injected/real failure was an attribute assignment.
            if kind == "parameter":
                module._parameters[target] = value
            else:
                module._buffers[target] = value
        if self.model is not None:
            self.arena.unguard_whole_model_to(self.model)
        first_error = None
        for block in self.blocks:
            try:
                pin_manager.release(block.handle)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                continue
            if block.pack is not None:
                pin_manager.unregister_arena_storage(block.flat)
                self.arena._blocks.pop(block.key, None)
            block.handle = None
        self.arena._canonicalized = bool(self.arena._blocks)
        self._populated = False
        resources = getattr(self, "_arena_resources", None)
        if (
            first_error is None
            and resources is not None
            and not resources.canonical_committed
        ):
            resources.release()
        if first_error is not None:
            raise first_error
