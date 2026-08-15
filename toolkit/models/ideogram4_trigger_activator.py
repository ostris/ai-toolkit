"""Trigger-selective text activator components for Ideogram 4.

The implementation is intentionally independent from the Ideogram model classes.
It supports the v7 one-token/shared-post-layer path as well as the v8 virtual-token,
module-LoRA and bounded-gamma architecture.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import inspect
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_TAP_LAYERS: Tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
SUPPORTED_VIRTUAL_TOKEN_COUNTS = (1, 2, 4)
SUPPORTED_TE_ADAPTER_MODES = ("shared_post_layer", "module_lora")
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


class BoundedGamma(nn.Module):
    """A scalar constrained to a closed interval via a sigmoid parameterization."""

    def __init__(
        self,
        initial: float = 1.0,
        minimum: float = 0.0,
        maximum: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if not minimum < maximum:
            raise ValueError("gamma minimum must be smaller than maximum")
        if not minimum <= initial <= maximum:
            raise ValueError("gamma initial value must lie inside its bounds")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        ratio = (float(initial) - self.minimum) / (self.maximum - self.minimum)
        ratio = min(max(ratio, 1.0e-6), 1.0 - 1.0e-6)
        raw = torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        self.trainable = bool(trainable)
        self.register_buffer("initial_value", torch.tensor(float(initial), dtype=torch.float32), persistent=True)
        if trainable:
            self.raw = nn.Parameter(raw)
        else:
            self.register_buffer("raw", raw, persistent=True)

    def forward(self, reference: Optional[Tensor] = None) -> Tensor:
        if self.trainable:
            value = self.minimum + (self.maximum - self.minimum) * torch.sigmoid(self.raw)
        else:
            value = self.initial_value
        if reference is not None:
            value = value.to(device=reference.device, dtype=reference.dtype)
        return value

    def value(self) -> float:
        return float(self().detach().cpu().item())


class AtomicLearnedEmbedding(nn.Module):
    """Independent one-, two- or four-vector trigger embedding."""

    MODES = {"learned", "frozen", "bypass"}

    def __init__(
        self,
        embedding_dim: int,
        tokens: int = 1,
        initializer: Optional[Tensor] = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or tokens not in SUPPORTED_VIRTUAL_TOKEN_COUNTS:
            raise ValueError("embedding_dim must be positive and tokens must be one of 1, 2 or 4")
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

    @property
    def tokens(self) -> int:
        return int(self.weight.shape[0])

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
        gamma: Optional[Tensor] = None,
    ) -> Tensor:
        vectors = self.vectors(mode)
        if not self.active or vectors is None:
            return hidden_states
        mask = _normalize_mask(
            token_mask if token_mask is not None else _runtime_value("token_mask"), hidden_states
        )
        if mask is None or not bool(torch.any(mask != 0)):
            return hidden_states
        if self.tokens == 1:
            replacement = vectors[0]
        else:
            token_indices = token_indices if token_indices is not None else _runtime_value("token_indices")
            if token_indices is None:
                raise ValueError("token_indices are required for a multi-token atomic embedding")
            indices = torch.as_tensor(token_indices, device=hidden_states.device, dtype=torch.long)
            if indices.shape != hidden_states.shape[:-1]:
                raise ValueError(
                    f"token_indices shape {tuple(indices.shape)} must match hidden states prefix "
                    f"{tuple(hidden_states.shape[:-1])}"
                )
            indices = indices.clamp(min=0, max=self.tokens - 1)
            replacement = F.embedding(indices, vectors)
        replacement = replacement.to(device=hidden_states.device, dtype=hidden_states.dtype)
        while replacement.ndim < hidden_states.ndim:
            replacement = replacement.unsqueeze(0)
        delta = (replacement - hidden_states) * mask
        if gamma is not None:
            delta = delta * gamma.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + delta


class MaskedLowRankAdapter(nn.Module):
    """Low-rank residual adapter applied to a hidden-state tensor."""

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

    def update(self, hidden_states: Tensor) -> Tensor:
        return self.up(self.down(self.dropout(hidden_states))) * self.scale.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        ) * (self.alpha / self.rank)

    def forward(
        self,
        hidden_states: Tensor,
        token_mask: Optional[Tensor] = None,
        gamma: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.active:
            return hidden_states
        mask = _normalize_mask(
            token_mask if token_mask is not None else _runtime_value("token_mask"), hidden_states
        )
        if mask is None or not bool(torch.any(mask != 0)):
            return hidden_states
        update = self.update(hidden_states) * mask
        if gamma is not None:
            update = update * gamma.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + update


class MaskedModuleLoRA(nn.Module):
    """LoRA update for a concrete Linear module with independent parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 1,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("module LoRA dimensions and rank must be positive")
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.active = True
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        module_input: Tensor,
        module_output: Tensor,
        token_mask: Optional[Tensor] = None,
        gamma: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.active:
            return module_output
        mask = _normalize_mask(
            token_mask if token_mask is not None else _runtime_value("token_mask"), module_output
        )
        if mask is None or not bool(torch.any(mask != 0)):
            return module_output
        update = self.up(self.down(self.dropout(module_input))) * (self.alpha / self.rank)
        update = update.to(module_output.dtype) * mask
        if gamma is not None:
            update = update * gamma.to(device=module_output.device, dtype=module_output.dtype)
        return module_output + update


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
    gamma_values: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _AdapterWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.adapter = adapter

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.wrapped(*args, **kwargs)
        hidden, rebuild = _extract_tensor(output)
        return rebuild(self.adapter(hidden)) if hidden is not None else output


class TextActivator(nn.Module):
    """Unified virtual embedding, internal adapter and 13-tap activator."""

    COMPONENTS = ("embedding", "te_adapter", "tap_adapters")

    def __init__(
        self,
        embedding_dim: int,
        hidden_size: Optional[int] = None,
        embedding_tokens: int = 1,
        initializer: Optional[Tensor] = None,
        te_adapter: Optional[MaskedLowRankAdapter] = None,
        te_adapter_mode: str = "shared_post_layer",
        te_rank: int = 1,
        te_alpha: Optional[float] = None,
        te_dropout: float = 0.0,
        te_target_modules: Sequence[str] = ("down_proj",),
        te_layers: Any = "all",
        tap_layers: Sequence[int] = DEFAULT_TAP_LAYERS,
        tap_rank: int = 1,
        tap_alpha: Optional[float] = None,
        tap_dropout: float = 0.0,
        tap_learnable_scale: bool = False,
        tap_scale_init: float = 1.0,
        per_tap: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        gamma_init: float = 1.0,
        gamma_min: float = 0.0,
        gamma_max: float = 1.0,
        gamma_trainable: bool = False,
        component_gammas: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size or embedding_dim)
        layers = tuple(int(layer) for layer in tap_layers)
        if len(layers) != 13 or len(set(layers)) != 13:
            raise ValueError("Ideogram 4 requires exactly 13 unique tap layer keys")
        if te_adapter_mode not in SUPPORTED_TE_ADAPTER_MODES:
            raise ValueError(f"unsupported te adapter mode: {te_adapter_mode}")
        if embedding_tokens not in SUPPORTED_VIRTUAL_TOKEN_COUNTS:
            raise ValueError("embedding_tokens must be one of 1, 2 or 4")
        self.embedding = AtomicLearnedEmbedding(embedding_dim, embedding_tokens, initializer)
        self.te_adapter_mode = te_adapter_mode
        self.te_adapter = te_adapter
        if self.te_adapter_mode == "shared_post_layer" and self.te_adapter is None and te_rank > 0:
            self.te_adapter = MaskedLowRankAdapter(
                hidden_size, rank=te_rank, alpha=te_alpha, dropout=te_dropout
            )
        self.te_rank = int(te_rank)
        self.te_alpha = float(te_rank if te_alpha is None else te_alpha)
        self.te_dropout = float(te_dropout)
        self.te_target_modules = tuple(str(name) for name in te_target_modules)
        self.te_layers = te_layers
        self.module_lora_adapters = nn.ModuleDict()
        self._module_lora_installed_on: Optional[int] = None
        self._module_lora_handles: List[Any] = []

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
            "te_adapter": self.te_adapter is not None or self.te_adapter_mode == "module_lora",
            "tap_adapters": True,
        }
        self.gamma = BoundedGamma(gamma_init, gamma_min, gamma_max, gamma_trainable)
        gamma_modules: Dict[str, BoundedGamma] = {}
        for component, options in (component_gammas or {}).items():
            if component not in self.COMPONENTS:
                raise ValueError(f"unknown component gamma: {component}")
            opts = dict(options)
            gamma_modules[component] = BoundedGamma(
                initial=float(opts.pop("initial", gamma_init)),
                minimum=float(opts.pop("minimum", gamma_min)),
                maximum=float(opts.pop("maximum", gamma_max)),
                trainable=bool(opts.pop("trainable", gamma_trainable)),
            )
            if opts:
                raise ValueError(f"unknown gamma options for {component}: {sorted(opts)}")
        self.component_gammas = nn.ModuleDict(gamma_modules)
        self._hooks: List[Any] = []
        self._wrapped: List[Tuple[nn.Module, str, nn.Module]] = []
        self._probe_updates: Dict[str, float] = {}

    @property
    def tap_layers(self) -> Tuple[int, ...]:
        return tuple(int(key) for key in self.tap_adapters.keys())

    @property
    def virtual_tokens(self) -> int:
        return self.embedding.tokens

    def _gamma_for(self, component: str, reference: Tensor) -> Tensor:
        value = self.gamma(reference)
        if component in self.component_gammas:
            value = value * self.component_gammas[component](reference)
        return value

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
        same_shape_frozen_bypass = mode == "activator_bypass"
        self.set_component_mode(
            "embedding", active=state.embedding_enabled or same_shape_frozen_bypass
        )
        self.set_component_mode("te_adapter", active=state.internal_enabled)
        self.set_component_mode("tap_adapters", active=state.tap_enabled)
        if state.embedding_enabled:
            self.embedding.set_mode("learned")
        elif same_shape_frozen_bypass:
            self.embedding.set_mode("frozen")
        else:
            self.embedding.set_mode("bypass")

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
            if component == "te_adapter":
                for adapter in self.module_lora_adapters.values():
                    adapter.active = bool(active)
            if component == "tap_adapters":
                for adapter in self.tap_adapters.values():
                    adapter.active = bool(active)
        if trainable is not None:
            if module is not None:
                module.requires_grad_(bool(trainable))
            if component == "te_adapter":
                self.module_lora_adapters.requires_grad_(bool(trainable))

    def apply_embedding(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        if not self.component_active["embedding"]:
            return hidden_states
        runtime_mode = kwargs.pop("runtime_mode", None)
        frozen_bypass = runtime_mode == "activator_bypass"
        mode = "frozen" if frozen_bypass else None
        return self.embedding(
            hidden_states,
            mode=mode,
            gamma=None if frozen_bypass else self._gamma_for("embedding", hidden_states),
            **kwargs,
        )

    def apply_te_adapter(
        self,
        hidden_states: Tensor,
        token_mask: Optional[Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Tensor:
        if (
            not self.component_active["te_adapter"]
            or self.te_adapter_mode != "shared_post_layer"
            or self.te_adapter is None
        ):
            return hidden_states
        return self.te_adapter(
            hidden_states, token_mask, gamma=self._gamma_for("te_adapter", hidden_states)
        )

    def apply_tap(
        self, tap_layer: int, hidden_states: Tensor, token_mask: Optional[Tensor] = None
    ) -> Tensor:
        key = str(int(tap_layer))
        if key not in self.tap_adapters:
            raise KeyError(f"unconfigured tap layer: {tap_layer}")
        if not self.component_active["tap_adapters"]:
            return hidden_states
        before = hidden_states
        output = self.tap_adapters[key](
            hidden_states, token_mask, gamma=self._gamma_for("tap_adapters", hidden_states)
        )
        if torch.is_grad_enabled():
            self._probe_updates[key] = float((output.detach() - before.detach()).float().norm().item())
        return output

    def _selected_te_layers(self, layer_count: int) -> Tuple[int, ...]:
        if self.te_layers == "all":
            return tuple(range(layer_count))
        layers = tuple(int(layer) for layer in self.te_layers)
        invalid = [layer for layer in layers if layer < 0 or layer >= layer_count]
        if invalid:
            raise ValueError(f"module LoRA layer indices out of range: {invalid}")
        return layers

    def install_module_lora(self, language_model: nn.Module) -> None:
        if self.te_adapter_mode != "module_lora":
            return
        identity = id(language_model)
        if self._module_lora_installed_on == identity:
            return
        self.remove_module_lora()
        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise ValueError("module_lora requires language_model.layers")
        for layer_idx in self._selected_te_layers(len(layers)):
            layer = layers[layer_idx]
            matched = 0
            for module_name, module in layer.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                leaf = module_name.rsplit(".", 1)[-1]
                if leaf not in self.te_target_modules:
                    continue
                key = f"layer_{layer_idx}__{module_name.replace('.', '__')}"
                adapter = MaskedModuleLoRA(
                    module.in_features,
                    module.out_features,
                    rank=self.te_rank,
                    alpha=self.te_alpha,
                    dropout=self.te_dropout,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                adapter.active = self.component_active["te_adapter"]
                self.module_lora_adapters[key] = adapter

                def hook(_module: nn.Module, inputs: Tuple[Any, ...], output: Any, key: str = key) -> Any:
                    hidden, rebuild = _extract_tensor(output)
                    if hidden is None or not inputs or not torch.is_tensor(inputs[0]):
                        return output
                    adapted = self.module_lora_adapters[key](
                        inputs[0], hidden, gamma=self._gamma_for("te_adapter", hidden)
                    )
                    return rebuild(adapted)

                self._module_lora_handles.append(module.register_forward_hook(hook))
                matched += 1
            if matched == 0:
                raise ValueError(
                    f"no target modules {self.te_target_modules} found in Qwen layer {layer_idx}"
                )
        self._module_lora_installed_on = identity

    def remove_module_lora(self) -> None:
        for handle in self._module_lora_handles:
            handle.remove()
        self._module_lora_handles.clear()
        self._module_lora_installed_on = None

    def parameter_groups(self, learning_rates: Optional[Mapping[str, float]] = None) -> List[Dict[str, Any]]:
        learning_rates = learning_rates or {}
        groups: List[Dict[str, Any]] = []
        component_modules = {
            "embedding": [self.embedding],
            "te_adapter": [module for module in (self.te_adapter, self.module_lora_adapters) if module is not None],
            "tap_adapters": [self.tap_adapters],
        }
        gamma_parameters = list(self.gamma.parameters())
        for name in self.COMPONENTS:
            params: List[nn.Parameter] = []
            for module in component_modules[name]:
                params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
            if name in self.component_gammas:
                params.extend(
                    parameter for parameter in self.component_gammas[name].parameters() if parameter.requires_grad
                )
            if name == "embedding":
                params.extend(parameter for parameter in gamma_parameters if parameter.requires_grad)
            seen: set[int] = set()
            params = [parameter for parameter in params if not (id(parameter) in seen or seen.add(id(parameter)))]
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

    def gamma_values(self) -> Dict[str, float]:
        values = {"global": self.gamma.value()}
        values.update({name: gamma.value() for name, gamma in self.component_gammas.items()})
        return values

    def runtime_metadata(self) -> Dict[str, Any]:
        return {
            "architecture_version": 8,
            "virtual_tokens": self.virtual_tokens,
            "te_adapter_mode": self.te_adapter_mode,
            "te_target_modules": list(self.te_target_modules),
            "te_layers": self.te_layers if self.te_layers == "all" else list(self.te_layers),
            "te_rank": self.te_rank,
            "tap_layers": list(self.tap_layers),
            "tap_ranks": {key: adapter.rank for key, adapter in self.tap_adapters.items()},
            "gamma": self.gamma_values(),
            "component_active": dict(self.component_active),
            "module_lora_keys": list(self.module_lora_adapters.keys()),
        }

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
            hook_count=len(self._hooks) + len(self._wrapped) + len(self._module_lora_handles),
            gamma_values=self.gamma_values(),
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
    "BoundedGamma",
    "DEFAULT_TAP_LAYERS",
    "MaskedLowRankAdapter",
    "MaskedModuleLoRA",
    "MaskedTapAdapter",
    "ProbeDiagnostics",
    "SUPPORTED_TE_ADAPTER_MODES",
    "SUPPORTED_VIRTUAL_TOKEN_COUNTS",
    "TextActivator",
    "trigger_runtime",
]
