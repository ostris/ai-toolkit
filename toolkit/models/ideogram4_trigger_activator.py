"""Isolated text activator components for Ideogram 4 trigger binding.

This module deliberately has no dependency on the Ideogram pipeline, model, or
trainer. Runtime trigger masks can be passed explicitly or obtained lazily from
``toolkit.trigger_binding`` when that integration module is available.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import inspect
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_TAP_LAYERS: Tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
_STATE_PREFIX = "ideogram4_text_activator."
_RUNTIME_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "ideogram4_trigger_activator_runtime", default=None
)


def _runtime_module() -> Any:
    try:
        return importlib.import_module("toolkit.trigger_binding")
    except ImportError:
        return None


def _runtime_value(name: str, default: Any = None) -> Any:
    runtime = _RUNTIME_CONTEXT.get()
    if runtime is not None:
        if isinstance(runtime, Mapping) and name in runtime:
            return runtime[name]
        value = getattr(runtime, name, None)
        if value is not None:
            return value() if callable(value) else value
    module = _runtime_module()
    if module is None:
        return default
    for accessor in (f"get_current_{name}", f"get_{name}"):
        value = getattr(module, accessor, None)
        if callable(value):
            return value()
    value = getattr(module, name, default)
    return value() if callable(value) else value


@contextlib.contextmanager
def trigger_runtime(runtime: Any) -> Iterator[Any]:
    """Provide a local runtime compatible with the future trigger binding API."""

    token = _RUNTIME_CONTEXT.set(runtime)
    try:
        yield runtime
    finally:
        _RUNTIME_CONTEXT.reset(token)


def _extract_tensor(output: Any) -> Tuple[Optional[Tensor], Any]:
    if torch.is_tensor(output):
        return output, lambda value: value
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0], lambda value: (value,) + output[1:]
    if isinstance(output, list) and output and torch.is_tensor(output[0]):
        return output[0], lambda value: [value] + output[1:]
    if isinstance(output, Mapping):
        for key in ("hidden_states", "last_hidden_state"):
            if key in output and torch.is_tensor(output[key]):
                def rebuild(value: Tensor, key: str = key) -> Any:
                    copied = output.copy()
                    copied[key] = value
                    return copied
                return output[key], rebuild
    return None, lambda value: output


def _normalize_mask(mask: Optional[Tensor], reference: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=reference.device)
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(-1)
    while mask.ndim < reference.ndim:
        mask = mask.unsqueeze(-1)
    if mask.ndim != reference.ndim:
        raise ValueError(f"mask rank {mask.ndim} is incompatible with tensor rank {reference.ndim}")
    try:
        torch.broadcast_shapes(mask.shape, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot broadcast to {tuple(reference.shape)}"
        ) from exc
    return mask.to(dtype=reference.dtype)


class AtomicLearnedEmbedding(nn.Module):
    """A standalone embedding parameter that never mutates Qwen's token table.

    ``initializer`` is copied into a frozen buffer. Learned mode uses only the
    independent parameter; frozen mode is useful for probes, and bypass mode
    leaves the original hidden states unchanged.
    """

    MODES = {"learned", "frozen", "bypass"}

    def __init__(
        self,
        embedding_dim: int,
        tokens: int = 1,
        initializer: Optional[Tensor] = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or tokens <= 0:
            raise ValueError("embedding_dim and tokens must be positive")
        shape = (int(tokens), int(embedding_dim))
        if initializer is None:
            frozen = torch.empty(shape).normal_(mean=0.0, std=float(init_std))
        else:
            frozen = torch.as_tensor(initializer).detach().clone()
            if frozen.ndim == 1:
                frozen = frozen.unsqueeze(0)
            if frozen.shape == (1, embedding_dim) and tokens > 1:
                frozen = frozen.expand(tokens, -1).clone()
            if tuple(frozen.shape) != shape:
                raise ValueError(f"initializer shape must be {shape}, got {tuple(frozen.shape)}")
        self.weight = nn.Parameter(frozen.clone())
        self.register_buffer("frozen_initializer", frozen, persistent=True)
        self.mode = "learned"
        self.active = True

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported embedding mode: {mode}")
        self.mode = mode

    def vectors(self, mode: Optional[str] = None) -> Optional[Tensor]:
        mode = mode or self.mode
        if mode == "learned":
            return self.weight
        if mode == "frozen":
            return self.frozen_initializer.detach()
        if mode == "bypass":
            return None
        raise ValueError(f"unsupported embedding mode: {mode}")

    def forward(
        self,
        hidden_states: Tensor,
        token_mask: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
        mode: Optional[str] = None,
    ) -> Tensor:
        vectors = self.vectors(mode)
        if not self.active or vectors is None:
            return hidden_states
        mask = _normalize_mask(token_mask if token_mask is not None else _runtime_value("token_mask"), hidden_states)
        if mask is None:
            return hidden_states
        if vectors.shape[0] == 1:
            replacement = vectors[0]
        else:
            if token_indices is None:
                token_indices = _runtime_value("token_indices")
            if token_indices is None:
                raise ValueError("token_indices are required for a multi-token atomic embedding")
            indices = torch.as_tensor(token_indices, device=hidden_states.device, dtype=torch.long)
            indices = indices.clamp(min=0, max=vectors.shape[0] - 1)
            replacement = F.embedding(indices, vectors)
        replacement = replacement.to(device=hidden_states.device, dtype=hidden_states.dtype)
        while replacement.ndim < hidden_states.ndim:
            replacement = replacement.unsqueeze(0)
        return torch.lerp(hidden_states, replacement, mask)


class MaskedLowRankAdapter(nn.Module):
    """Low-rank residual adapter whose update is restricted to a token mask."""

    def __init__(
        self,
        hidden_size: int,
        rank: int = 1,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        learnable_scale: bool = False,
        scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale_init)), persistent=True)
        self.active = True
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        if not self.active:
            return hidden_states
        mask = _normalize_mask(token_mask if token_mask is not None else _runtime_value("token_mask"), hidden_states)
        if mask is None or not bool(torch.any(mask != 0)):
            return hidden_states
        update = self.up(self.down(self.dropout(hidden_states)))
        scale = self.scale.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + update * mask * scale * (self.alpha / self.rank)


class MaskedTapAdapter(MaskedLowRankAdapter):
    def __init__(self, tap_layer: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tap_layer = int(tap_layer)


@dataclass
class ProbeDiagnostics:
    active: bool
    trainable_parameters: int
    total_parameters: int
    embedding_delta_norm: float
    adapter_update_norms: Dict[str, float]
    tap_layers: Tuple[int, ...]
    hook_count: int


class _AdapterWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module, adapter: MaskedLowRankAdapter) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.adapter = adapter

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.wrapped(*args, **kwargs)
        hidden, rebuild = _extract_tensor(output)
        return rebuild(self.adapter(hidden)) if hidden is not None else output


class TextActivator(nn.Module):
    """Unified atomic embedding, Qwen adapter, and 13-tap aggregator."""

    COMPONENTS = ("embedding", "te_adapter", "tap_adapters")

    def __init__(
        self,
        embedding_dim: int,
        hidden_size: Optional[int] = None,
        embedding_tokens: int = 1,
        initializer: Optional[Tensor] = None,
        te_adapter: Optional[MaskedLowRankAdapter] = None,
        tap_layers: Sequence[int] = DEFAULT_TAP_LAYERS,
        tap_rank: int = 1,
        tap_alpha: Optional[float] = None,
        tap_dropout: float = 0.0,
        tap_learnable_scale: bool = False,
        tap_scale_init: float = 1.0,
        per_tap: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size or embedding_dim)
        layers = tuple(int(layer) for layer in tap_layers)
        if len(layers) != 13 or len(set(layers)) != 13:
            raise ValueError("Ideogram 4 requires exactly 13 unique tap layer keys")
        self.embedding = AtomicLearnedEmbedding(embedding_dim, embedding_tokens, initializer)
        self.te_adapter = te_adapter
        per_tap = per_tap or {}
        adapters: Dict[str, MaskedTapAdapter] = {}
        for layer in layers:
            overrides = dict(per_tap.get(layer, per_tap.get(str(layer), {})))
            adapters[str(layer)] = MaskedTapAdapter(
                tap_layer=layer,
                hidden_size=hidden_size,
                rank=int(overrides.pop("rank", tap_rank)),
                alpha=overrides.pop("alpha", tap_alpha),
                dropout=float(overrides.pop("dropout", tap_dropout)),
                learnable_scale=bool(overrides.pop("learnable_scale", tap_learnable_scale)),
                scale_init=float(overrides.pop("scale_init", tap_scale_init)),
            )
            if overrides:
                raise ValueError(f"unknown per_tap options for layer {layer}: {sorted(overrides)}")
        self.tap_adapters = nn.ModuleDict(adapters)
        self.component_active = {
            "embedding": True,
            "te_adapter": te_adapter is not None,
            "tap_adapters": True,
        }
        self._hooks: List[Any] = []
        self._wrapped: List[Tuple[nn.Module, str, nn.Module]] = []
        self._probe_inputs: Dict[str, Tensor] = {}
        self._probe_updates: Dict[str, float] = {}

    @property
    def tap_layers(self) -> Tuple[int, ...]:
        return tuple(int(key) for key in self.tap_adapters.keys())

    def set_runtime_mode(self, mode: Optional[str]) -> None:
        if mode is None:
            return
        try:
            runtime = importlib.import_module("toolkit.trigger_binding")
            state = runtime.get_activator_runtime_state(mode)
        except (ImportError, AttributeError):
            enabled = mode not in {"activator_bypass", "stock_literal"}
            state = type("RuntimeState", (), {
                "embedding_enabled": enabled,
                "internal_enabled": enabled,
                "tap_enabled": enabled,
            })()
        self.set_component_mode("embedding", active=state.embedding_enabled)
        self.set_component_mode("te_adapter", active=state.internal_enabled)
        self.set_component_mode("tap_adapters", active=state.tap_enabled)
        self.embedding.set_mode("frozen" if not state.embedding_enabled else "learned")

    def has_trainable_parameters(self) -> bool:
        return any(parameter.requires_grad for parameter in self.parameters())

    def set_component_mode(
        self, component: str, *, active: Optional[bool] = None, trainable: Optional[bool] = None
    ) -> None:
        if component not in self.COMPONENTS:
            raise KeyError(f"unknown component: {component}")
        module = getattr(self, component)
        if active is not None:
            self.component_active[component] = bool(active)
            if module is not None and hasattr(module, "active"):
                module.active = bool(active)
            if component == "tap_adapters":
                for adapter in self.tap_adapters.values():
                    adapter.active = bool(active)
        if trainable is not None and module is not None:
            module.requires_grad_(bool(trainable))

    def apply_embedding(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        if not self.component_active["embedding"]:
            return hidden_states
        return self.embedding(hidden_states, **kwargs)

    def apply_te_adapter(self, hidden_states: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        if not self.component_active["te_adapter"] or self.te_adapter is None:
            return hidden_states
        return self.te_adapter(hidden_states, token_mask)

    def apply_tap(self, tap_layer: int, hidden_states: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        key = str(int(tap_layer))
        if key not in self.tap_adapters:
            raise KeyError(f"unconfigured tap layer: {tap_layer}")
        if not self.component_active["tap_adapters"]:
            return hidden_states
        before = hidden_states
        output = self.tap_adapters[key](hidden_states, token_mask)
        if torch.is_grad_enabled():
            self._probe_updates[key] = float((output.detach() - before.detach()).float().norm().item())
        return output

    def parameter_groups(self, learning_rates: Optional[Mapping[str, float]] = None) -> List[Dict[str, Any]]:
        learning_rates = learning_rates or {}
        groups: List[Dict[str, Any]] = []
        for name in self.COMPONENTS:
            module = getattr(self, name)
            if module is None:
                continue
            params = [parameter for parameter in module.parameters() if parameter.requires_grad]
            if params:
                group: Dict[str, Any] = {"params": params, "name": f"text_activator.{name}"}
                if name in learning_rates:
                    group["lr"] = float(learning_rates[name])
                groups.append(group)
        return groups

    def install_qwen_hooks(
        self,
        qwen: nn.Module,
        te_module_names: Iterable[str] = (),
        tap_module_names: Optional[Mapping[int, str]] = None,
        use_wrappers: bool = False,
    ) -> None:
        self.remove_qwen_hooks()
        named = dict(qwen.named_modules())
        tap_module_names = tap_module_names or {}
        for name in te_module_names:
            if name not in named:
                raise KeyError(f"Qwen module not found: {name}")
            self._attach(named[name], name, self.apply_te_adapter, use_wrappers, qwen)
        for layer, name in tap_module_names.items():
            if str(int(layer)) not in self.tap_adapters:
                raise KeyError(f"unconfigured tap layer: {layer}")
            if name not in named:
                raise KeyError(f"Qwen tap module not found: {name}")
            callback = lambda hidden, layer=int(layer): self.apply_tap(layer, hidden)
            self._attach(named[name], name, callback, use_wrappers, qwen)

    def _attach(self, module: nn.Module, name: str, callback: Any, wrapper: bool, root: nn.Module) -> None:
        if wrapper:
            parent, attr = self._resolve_parent(root, name)
            wrapped = _AdapterWrapper(module, _CallbackAdapter(callback))
            setattr(parent, attr, wrapped)
            self._wrapped.append((parent, attr, module))
            return

        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
            hidden, rebuild = _extract_tensor(output)
            return rebuild(callback(hidden)) if hidden is not None else output

        self._hooks.append(module.register_forward_hook(hook))

    @staticmethod
    def _resolve_parent(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
        if not name:
            raise ValueError("the root Qwen module cannot be replaced by a wrapper")
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
        return parent, parts[-1]

    def remove_qwen_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        for parent, attr, original in reversed(self._wrapped):
            setattr(parent, attr, original)
        self._wrapped.clear()

    def probe_diagnostics(self) -> ProbeDiagnostics:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        embedding_delta = (self.embedding.weight.detach() - self.embedding.frozen_initializer).float().norm()
        return ProbeDiagnostics(
            active=any(self.component_active.values()),
            trainable_parameters=trainable,
            total_parameters=total,
            embedding_delta_norm=float(embedding_delta.item()),
            adapter_update_norms=dict(self._probe_updates),
            tap_layers=self.tap_layers,
            hook_count=len(self._hooks) + len(self._wrapped),
        )

    def activator_state_dict(self) -> "OrderedDict[str, Tensor]":
        return OrderedDict((_STATE_PREFIX + key, value) for key, value in super().state_dict().items())

    def load_activator_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        foreign = sorted(key for key in state_dict if not key.startswith(_STATE_PREFIX))
        if foreign and strict:
            raise RuntimeError(f"foreign state dict keys: {foreign}")
        stripped = OrderedDict(
            (key[len(_STATE_PREFIX):], value)
            for key, value in state_dict.items()
            if key.startswith(_STATE_PREFIX)
        )
        expected = super().state_dict()
        if strict:
            missing = sorted(set(expected) - set(stripped))
            unexpected = sorted(set(stripped) - set(expected))
            shape_errors = sorted(
                key for key in set(expected) & set(stripped)
                if tuple(expected[key].shape) != tuple(stripped[key].shape)
            )
            if missing or unexpected or shape_errors:
                raise RuntimeError(
                    f"invalid activator state dict; missing={missing}, unexpected={unexpected}, "
                    f"shape_mismatch={shape_errors}"
                )
        return super().load_state_dict(stripped, strict=strict)

    def state_dict(self, *args: Any, **kwargs: Any) -> "OrderedDict[str, Tensor]":
        destination = kwargs.get("destination")
        if destination is not None or args:
            return super().state_dict(*args, **kwargs)
        return self.activator_state_dict()

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True, assign: bool = False
    ) -> torch.nn.modules.module._IncompatibleKeys:
        if assign and "assign" not in inspect.signature(super().load_state_dict).parameters:
            raise TypeError("this torch version does not support assign=True")
        if assign:
            stripped = OrderedDict(
                (key[len(_STATE_PREFIX):], value)
                for key, value in state_dict.items()
                if key.startswith(_STATE_PREFIX)
            )
            return super().load_state_dict(stripped, strict=strict, assign=True)
        return self.load_activator_state_dict(state_dict, strict=strict)


class _CallbackAdapter(nn.Module):
    def __init__(self, callback: Any) -> None:
        super().__init__()
        self.callback = callback

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.callback(hidden_states)


__all__ = [
    "AtomicLearnedEmbedding",
    "DEFAULT_TAP_LAYERS",
    "MaskedLowRankAdapter",
    "MaskedTapAdapter",
    "ProbeDiagnostics",
    "TextActivator",
    "trigger_runtime",
]
