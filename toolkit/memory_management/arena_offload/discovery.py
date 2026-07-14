"""Generic repeated-block discovery and pre-commit state accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from toolkit.quantization.storage import module_storage_binding


class BlockDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class BlockStateAccounting:
    managed_entries: int
    managed_bytes: int
    trainable_entries: int
    resident_entries: int
    resident_bytes: int


@dataclass(frozen=True)
class BlockSelection:
    container_paths: tuple[str, ...]
    blocks: tuple[torch.nn.Module, ...]
    block_keys: tuple[str, ...]
    entries_by_block: dict[str, tuple[tuple[str, torch.nn.Module], ...]]
    accounting: BlockStateAccounting


def _resolve_path(model, path: str):
    value = model
    for component in str(path).split("."):
        value = getattr(value, component, None)
        if value is None:
            raise BlockDiscoveryError(f"block_container_not_found:{path}")
    return value


def managed_entries(block: torch.nn.Module) -> tuple[tuple[str, torch.nn.Module], ...]:
    entries = []
    try:
        modules = block.named_modules(remove_duplicate=False)
    except TypeError:
        modules = block.named_modules()
    for path, module in modules:
        if not path or not isinstance(module, torch.nn.Linear):
            continue
        try:
            module_storage_binding(module)
        except Exception as error:
            raise BlockDiscoveryError(
                f"unsupported_managed_module:{path}:{type(module).__qualname__}"
            ) from error
        entries.append((path, module))
    return tuple(entries)


def _container_candidate(path, container):
    if not isinstance(container, (torch.nn.ModuleList, torch.nn.Sequential)):
        return None
    blocks = tuple(container)
    if len(blocks) < 2 or any(not isinstance(block, torch.nn.Module) for block in blocks):
        return None
    entries = tuple(managed_entries(block) for block in blocks)
    populated = sum(bool(items) for items in entries)
    if populated < 2:
        return None
    payload = 0
    for items in entries:
        for _name, module in items:
            binding = module_storage_binding(module)
            payload += sum(
                int(item.tensor.numel() * item.tensor.element_size())
                for item in binding.tensors
            )
    return str(path), blocks, entries, payload


def _select_containers(model, container_paths):
    if container_paths:
        selected = []
        for path in tuple(container_paths):
            candidate = _container_candidate(path, _resolve_path(model, path))
            if candidate is None:
                raise BlockDiscoveryError(f"invalid_block_container:{path}")
            selected.append(candidate)
        return tuple(selected)

    candidates = []
    for path, module in model.named_modules():
        if not path:
            continue
        candidate = _container_candidate(path, module)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise BlockDiscoveryError("no_repeated_block_container")
    candidates.sort(key=lambda item: (-item[3], item[0]))
    if len(candidates) > 1 and candidates[0][3] == candidates[1][3]:
        paths = ",".join(item[0] for item in candidates if item[3] == candidates[0][3])
        raise BlockDiscoveryError(f"ambiguous_block_container:{paths}")
    return (candidates[0],)


def _named_buffers(block):
    for module_path, module in block.named_modules():
        for name, value in module._buffers.items():
            if value is None:
                continue
            path = f"{module_path}.{name}" if module_path else name
            yield path, value


def _tensor_identity(tensor):
    if tensor.device.type == "meta":
        return ("meta", id(tensor))
    if tensor.numel() == 0:
        return ("empty", tensor.device.type, id(tensor))
    try:
        return (
            tensor.device.type,
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.numel(),
        )
    except Exception:
        return ("object", id(tensor))


def _state_bytes(value) -> int:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        return 0
    return int(value.numel() * value.element_size())


def _audit_state(blocks, block_keys, entries_by_block) -> BlockStateAccounting:
    managed_objects = {}
    managed_storage = {}
    managed_entries_count = 0
    managed_bytes = 0
    trainable_entries = 0
    resident_entries = 0
    resident_bytes = 0
    resident_seen = set()

    for block, block_key in zip(blocks, block_keys, strict=True):
        declared = {}
        for module_path, module in entries_by_block[block_key]:
            binding = module_storage_binding(module)
            for substitution in binding.substitutions:
                target = substitution.name
                state_path = f"{module_path}.{target}"
                if state_path in declared:
                    raise BlockDiscoveryError(
                        f"conflicting_replacement_declaration:{block_key}.{state_path}"
                    )
                if torch.nn.utils.parametrize.is_parametrized(module, target):
                    raise BlockDiscoveryError(
                        f"parametrized_managed_state:{block_key}.{state_path}"
                    )
                if target in module._parameters:
                    value = module._parameters[target]
                elif target in module._buffers:
                    value = module._buffers[target]
                else:
                    raise BlockDiscoveryError(
                        f"missing_replacement_target:{block_key}.{state_path}"
                    )
                if value is None:
                    raise BlockDiscoveryError(
                        f"missing_replacement_target:{block_key}.{state_path}"
                    )
                declared[state_path] = value
                previous = managed_objects.get(id(value))
                if previous is not None:
                    raise BlockDiscoveryError(
                        f"shared_managed_state:{previous}:{block_key}.{state_path}"
                    )
                managed_objects[id(value)] = f"{block_key}.{state_path}"
                managed_entries_count += 1
            for tensor in binding.tensors:
                identity = _tensor_identity(tensor.tensor)
                previous = managed_storage.get(identity)
                if previous is not None and previous[0] != block_key:
                    raise BlockDiscoveryError(
                        f"shared_managed_storage:{previous[1]}:{block_key}.{module_path}"
                    )
                if previous is None:
                    managed_bytes += _state_bytes(tensor.tensor)
                    managed_storage[identity] = (
                        block_key,
                        f"{block_key}.{module_path}.{tensor.name}",
                        tensor.tensor,
                    )

        parameters = tuple(block.named_parameters(recurse=True, remove_duplicate=False))
        # Non-persistent buffers still participate in execution and must be
        # present in the complete state audit. Persistence only controls the
        # default state_dict serialization surface.
        buffers = tuple(_named_buffers(block))
        enumerated = {name for name, _value in parameters + buffers}
        missing = set(declared) - enumerated
        if missing:
            raise BlockDiscoveryError(
                f"managed_state_not_enumerated:{block_key}.{sorted(missing)[0]}"
            )
        for name, value in parameters + buffers:
            if name in declared:
                continue
            if isinstance(value, torch.nn.Parameter) and value.requires_grad:
                trainable_entries += 1
                continue
            resident_entries += 1
            identity = _tensor_identity(value)
            if identity not in resident_seen:
                resident_seen.add(identity)
                resident_bytes += _state_bytes(value)

    return BlockStateAccounting(
        managed_entries=managed_entries_count,
        managed_bytes=managed_bytes,
        trainable_entries=trainable_entries,
        resident_entries=resident_entries,
        resident_bytes=resident_bytes,
    )


def discover_blocks(model, *, container_paths=()) -> BlockSelection:
    """Select repeated blocks and audit all state before canonical commit."""
    containers = _select_containers(model, tuple(container_paths or ()))
    paths = tuple(item[0] for item in containers)
    blocks = tuple(block for item in containers for block in item[1])
    if len({id(block) for block in blocks}) != len(blocks):
        raise BlockDiscoveryError("duplicate_selected_block")

    block_keys = tuple(
        f"{path}.{index}"
        for path, _blocks, _entries, _payload in containers
        for index in range(len(_blocks))
    )
    entry_groups = tuple(
        entries
        for _path, _blocks, groups, _payload in containers
        for entries in groups
    )
    entries_by_block = dict(zip(block_keys, entry_groups, strict=True))
    managed_modules = [
        module
        for entries in entry_groups
        for _name, module in entries
    ]
    if len({id(module) for module in managed_modules}) != len(managed_modules):
        raise BlockDiscoveryError("overlapping_managed_module")
    accounting = _audit_state(blocks, block_keys, entries_by_block)
    return BlockSelection(
        container_paths=paths,
        blocks=blocks,
        block_keys=block_keys,
        entries_by_block=entries_by_block,
        accounting=accounting,
    )
