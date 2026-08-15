from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


_PHASES = frozenset({"a1", "a2", "b"})


class ReachabilityCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParameterDiagnostic:
    name: str
    requires_grad: bool
    in_optimizer: bool
    grad_state: str = "not_checked"
    grad_norm: Optional[float] = None


@dataclass
class ReachabilityDiagnostics:
    phase: str
    stage: str
    passed: bool
    complete: bool
    target: str
    frozen: str
    checks: Dict[str, bool] = field(default_factory=dict)
    parameters: Dict[str, Tuple[ParameterDiagnostic, ...]] = field(default_factory=dict)
    output_difference: Optional[float] = None
    messages: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.passed

    @property
    def reachable(self) -> bool:
        return self.passed

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CombinedReachabilityDiagnostics:
    phase: str
    passed: bool
    complete: bool
    static: ReachabilityDiagnostics
    gradient: Optional[ReachabilityDiagnostics] = None

    @property
    def ok(self) -> bool:
        return self.passed

    @property
    def reachable(self) -> bool:
        return self.passed

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_phase(phase: str) -> str:
    normalized = str(phase).lower()
    if normalized not in _PHASES:
        raise ReachabilityCheckError(f"unsupported phase {phase!r}; expected a1, a2, or b")
    return normalized


def _phase_modules(activator: Any, network: Any, phase: str) -> Tuple[str, Any, str, Any]:
    normalized = _normalize_phase(phase)
    if normalized == "b":
        return "network", network, "activator", activator
    return "activator", activator, "network", network


def _named_parameters(module: Any, prefix: str) -> List[Tuple[str, torch.nn.Parameter]]:
    if module is None:
        return []
    named = getattr(module, "named_parameters", None)
    if not callable(named):
        raise ReachabilityCheckError(f"{prefix} must expose named_parameters()")
    return [(f"{prefix}.{name}" if name else prefix, parameter) for name, parameter in named()]


def _flatten_parameter_source(source: Any) -> Optional[List[torch.nn.Parameter]]:
    if source is None:
        return None
    if hasattr(source, "param_groups"):
        source = source.param_groups
    if isinstance(source, torch.nn.Parameter):
        return [source]
    if isinstance(source, Mapping):
        source = source.get("params", [])
    if not isinstance(source, Iterable):
        raise ReachabilityCheckError("optimizer/params must be an optimizer or iterable of parameters/groups")

    flattened: List[torch.nn.Parameter] = []
    for item in source:
        if isinstance(item, torch.nn.Parameter):
            flattened.append(item)
        elif isinstance(item, Mapping):
            nested = item.get("params", [])
            flattened.extend(parameter for parameter in nested if isinstance(parameter, torch.nn.Parameter))
        else:
            raise ReachabilityCheckError("parameter source contains an unsupported entry")
    return flattened


def _resolve_component_parameters(
    component_name: str,
    selector: Any,
    activator: Any,
    network: Any,
    all_named: Mapping[str, torch.nn.Parameter],
) -> List[Tuple[str, torch.nn.Parameter]]:
    if isinstance(selector, torch.nn.Parameter):
        matches = [(name, parameter) for name, parameter in all_named.items() if parameter is selector]
    elif isinstance(selector, str):
        if any(character in selector for character in "*?["):
            patterns = (selector, f"activator.{selector}", f"network.{selector}")
            matches = [
                (name, parameter)
                for name, parameter in all_named.items()
                if any(fnmatchcase(name, pattern) for pattern in patterns)
            ]
        else:
            prefixes = (selector, f"activator.{selector}", f"network.{selector}")
            matches = [
                (name, parameter)
                for name, parameter in all_named.items()
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
            ]
    elif hasattr(selector, "named_parameters"):
        matches = [
            (name, parameter)
            for name, parameter in all_named.items()
            if any(parameter is selected for selected in selector.parameters())
        ]
    elif isinstance(selector, Iterable):
        matches = []
        for item in selector:
            matches.extend(_resolve_component_parameters(component_name, item, activator, network, all_named))
    else:
        raise ReachabilityCheckError(f"named component {component_name!r} has an unsupported selector")
    unique = {id(parameter): (name, parameter) for name, parameter in matches}
    return sorted(unique.values(), key=lambda item: item[0])


def _partition_named_components(
    activator: Any,
    network: Any,
    phase: str,
    named_components: Optional[Mapping[str, Any]],
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]], Dict[str, List[Tuple[str, torch.nn.Parameter]]], List[str]]:
    target_name, target_module, frozen_name, frozen_module = _phase_modules(activator, network, phase)
    target_module_parameters = _named_parameters(target_module, target_name)
    phase_frozen_parameters = _named_parameters(frozen_module, frozen_name)
    if named_components is None:
        target_parameters = [(name, parameter) for name, parameter in target_module_parameters if parameter.requires_grad]
        inactive = [(name, parameter) for name, parameter in target_module_parameters if not parameter.requires_grad]
        return target_parameters, inactive + phase_frozen_parameters, {}, []
    if not isinstance(named_components, Mapping) or not named_components:
        raise ReachabilityCheckError("named_components must be a non-empty mapping")
    all_named = dict(target_module_parameters + phase_frozen_parameters)
    component_parameters: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}
    missing = []
    expected_ids: set[int] = set()
    for component_name, selector in named_components.items():
        if not isinstance(component_name, str) or not component_name:
            raise ReachabilityCheckError("named component names must be non-empty strings")
        matches = _resolve_component_parameters(component_name, selector, activator, network, all_named)
        if not matches:
            missing.append(component_name)
        component_parameters[component_name] = matches
        expected_ids.update(id(parameter) for _, parameter in matches)
    target_ids = {id(parameter) for _, parameter in target_module_parameters}
    foreign = [name for name, entries in component_parameters.items() if any(id(parameter) not in target_ids for _, parameter in entries)]
    if foreign:
        raise ReachabilityCheckError(f"named components select phase-frozen parameters: {sorted(foreign)}")
    target_parameters = [(name, parameter) for name, parameter in target_module_parameters if id(parameter) in expected_ids]
    frozen_parameters = [
        (name, parameter)
        for name, parameter in target_module_parameters + phase_frozen_parameters
        if id(parameter) not in expected_ids
    ]
    return target_parameters, frozen_parameters, component_parameters, missing


def _parameter_diagnostics(
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    optimizer_ids: set[int],
    gradients: Optional[Mapping[int, Optional[torch.Tensor]]] = None,
) -> Tuple[ParameterDiagnostic, ...]:
    result = []
    for name, parameter in named_parameters:
        gradient = None if gradients is None else gradients.get(id(parameter))
        if gradients is None:
            grad_state = "not_checked"
            grad_norm = None
        elif gradient is None:
            grad_state = "none"
            grad_norm = None
        elif not torch.isfinite(gradient).all().item():
            grad_state = "nonfinite"
            grad_norm = float("nan")
        else:
            grad_norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
            grad_state = "nonzero" if grad_norm > 0.0 else "zero"
        result.append(
            ParameterDiagnostic(
                name=name,
                requires_grad=bool(parameter.requires_grad),
                in_optimizer=id(parameter) in optimizer_ids,
                grad_state=grad_state,
                grad_norm=grad_norm,
            )
        )
    return tuple(result)


def check_optimizer_isolation(
    activator: Any,
    network: Any,
    phase: str,
    *,
    optimizer: Any = None,
    params: Any = None,
    named_components: Optional[Mapping[str, Any]] = None,
    raise_on_error: bool = False,
) -> ReachabilityDiagnostics:
    """Validate phase trainability and optimizer membership without needing a batch."""
    normalized = _normalize_phase(phase)
    target_name, _, frozen_name, _ = _phase_modules(activator, network, normalized)
    target_parameters, frozen_parameters, component_parameters, missing_components = _partition_named_components(
        activator, network, normalized, named_components
    )
    parameter_source = optimizer if optimizer is not None else params
    optimizer_parameters = _flatten_parameter_source(parameter_source)
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters or []}

    target_trainable = bool(target_parameters)
    frozen_frozen = all(not parameter.requires_grad for _, parameter in frozen_parameters)
    optimizer_available = optimizer_parameters is not None
    target_in_optimizer = optimizer_available and bool(target_parameters) and all(
        id(parameter) in optimizer_ids for _, parameter in target_parameters
    )
    frozen_out_optimizer = optimizer_available and all(
        id(parameter) not in optimizer_ids for _, parameter in frozen_parameters
    )
    components_present = not missing_components
    components_require_grad = all(
        entries and all(parameter.requires_grad for _, parameter in entries)
        for entries in component_parameters.values()
    ) if named_components is not None else True
    checks = {
        "target_parameters_present": bool(target_parameters),
        "target_requires_grad": target_trainable,
        "named_components_present": components_present,
        "named_components_require_grad": components_require_grad,
        "frozen_requires_grad_false": frozen_frozen,
        "optimizer_available": optimizer_available,
        "target_in_optimizer": target_in_optimizer,
        "frozen_out_of_optimizer": frozen_out_optimizer,
    }
    messages = []
    if missing_components:
        messages.append(f"named components have no matching parameters: {sorted(missing_components)}")
    if not components_require_grad:
        messages.append("some named component parameters do not require gradients")
    if not target_parameters:
        messages.append(f"{target_name} has no parameters")
    if not target_trainable:
        messages.append(f"not all {target_name} parameters require gradients")
    if not frozen_frozen:
        messages.append(f"some {frozen_name} parameters still require gradients")
    if not optimizer_available:
        messages.append("optimizer/params unavailable; optimizer isolation is pending")
    elif not target_in_optimizer:
        messages.append(f"some {target_name} parameters are missing from optimizer")
    if optimizer_available and not frozen_out_optimizer:
        messages.append(f"some frozen {frozen_name} parameters are present in optimizer")

    diagnostics = ReachabilityDiagnostics(
        phase=normalized,
        stage="static_optimizer_isolation",
        passed=all(checks.values()),
        complete=optimizer_available,
        target=target_name,
        frozen=frozen_name,
        checks=checks,
        parameters={
            "target": _parameter_diagnostics(target_parameters, optimizer_ids),
            "frozen": _parameter_diagnostics(frozen_parameters, optimizer_ids),
            **{
                f"component:{name}": _parameter_diagnostics(entries, optimizer_ids)
                for name, entries in component_parameters.items()
            },
        },
        messages=messages,
    )
    if raise_on_error and not diagnostics.passed:
        raise ReachabilityCheckError(f"static optimizer isolation failed: {diagnostics.as_dict()}")
    return diagnostics


def _unpack_forward_result(result: Any) -> Tuple[Any, Optional[torch.Tensor]]:
    if isinstance(result, Mapping):
        output = result.get("output", result.get("prediction"))
        loss = result.get("loss")
        return output, loss
    if isinstance(result, tuple) and len(result) == 2:
        output, loss = result
        return output, loss
    return result, result if isinstance(result, torch.Tensor) and result.numel() == 1 else None


def _tensor_leaves(value: Any) -> List[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        leaves: List[torch.Tensor] = []
        for key in sorted(value):
            leaves.extend(_tensor_leaves(value[key]))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_tensor_leaves(item))
        return leaves
    return []


def _output_difference(active: Any, bypass: Any) -> Optional[float]:
    active_tensors = _tensor_leaves(active)
    bypass_tensors = _tensor_leaves(bypass)
    if not active_tensors or len(active_tensors) != len(bypass_tensors):
        return None
    differences = []
    for active_tensor, bypass_tensor in zip(active_tensors, bypass_tensors):
        if active_tensor.shape != bypass_tensor.shape:
            return float("inf")
        difference = (active_tensor.detach().float() - bypass_tensor.detach().float()).abs()
        differences.append(float(difference.max().item()) if difference.numel() else 0.0)
    return max(differences, default=0.0)


def check_gradient_reachability(
    activator: Any,
    network: Any,
    phase: str,
    *,
    loss: Optional[torch.Tensor] = None,
    forward_callback: Optional[Callable[[str], Any]] = None,
    active_output: Any = None,
    bypass_output: Any = None,
    optimizer: Any = None,
    params: Any = None,
    named_components: Optional[Mapping[str, Any]] = None,
    difference_atol: float = 0.0,
    require_output_difference: bool = True,
    raise_on_error: bool = False,
) -> ReachabilityDiagnostics:
    """Probe the first real graph without mutating ``parameter.grad`` or consuming it."""
    normalized = _normalize_phase(phase)
    target_name, _, frozen_name, _ = _phase_modules(activator, network, normalized)
    target_parameters, frozen_parameters, component_parameters, missing_components = _partition_named_components(
        activator, network, normalized, named_components
    )

    if forward_callback is not None:
        active_result = forward_callback("active")
        bypass_result = forward_callback("bypass")
        callback_output, callback_loss = _unpack_forward_result(active_result)
        active_output = callback_output if active_output is None else active_output
        bypass_output = _unpack_forward_result(bypass_result)[0] if bypass_output is None else bypass_output
        loss = callback_loss if loss is None else loss
    if loss is None:
        raise ReachabilityCheckError("gradient gate requires a scalar loss or forward_callback returning loss")
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ReachabilityCheckError("loss must be a scalar torch.Tensor")

    probed_parameters = []
    seen_parameter_ids: set[int] = set()
    for _, parameter in target_parameters + frozen_parameters:
        if parameter.requires_grad and id(parameter) not in seen_parameter_ids:
            probed_parameters.append(parameter)
            seen_parameter_ids.add(id(parameter))
    probe_gradients = torch.autograd.grad(
        loss,
        probed_parameters,
        retain_graph=True,
        allow_unused=True,
    ) if probed_parameters and loss.requires_grad else tuple(None for _ in probed_parameters)
    gradients: Dict[int, Optional[torch.Tensor]] = {
        id(parameter): gradient for parameter, gradient in zip(probed_parameters, probe_gradients)
    }
    for _, parameter in frozen_parameters:
        if not parameter.requires_grad:
            gradients[id(parameter)] = parameter.grad

    optimizer_parameters = _flatten_parameter_source(optimizer if optimizer is not None else params)
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters or []}
    target_details = _parameter_diagnostics(target_parameters, optimizer_ids, gradients)
    frozen_details = _parameter_diagnostics(frozen_parameters, optimizer_ids, gradients)
    component_details = {
        name: _parameter_diagnostics(entries, optimizer_ids, gradients)
        for name, entries in component_parameters.items()
    }
    target_gradients_reachable = bool(target_details) and all(
        item.grad_state in {"zero", "nonzero"} for item in target_details
    )
    target_has_nonzero_gradient = any(item.grad_state == "nonzero" for item in target_details)
    components_present = not missing_components
    components_have_gradients = all(
        details and all(item.grad_state in {"zero", "nonzero"} for item in details)
        for details in component_details.values()
    ) if named_components is not None else True
    components_have_nonzero_gradients = all(
        any(item.grad_state == "nonzero" for item in details)
        for details in component_details.values()
    ) if named_components is not None else True
    frozen_gradients_absent = all(item.grad_state == "none" for item in frozen_details)
    difference = _output_difference(active_output, bypass_output)
    outputs_available = difference is not None
    outputs_differ = outputs_available and difference > float(difference_atol)
    checks = {
        "target_gradients_reachable_and_finite": target_gradients_reachable,
        "target_has_nonzero_gradient": target_has_nonzero_gradient,
        "named_components_present": components_present,
        "named_components_gradients_reachable_and_finite": components_have_gradients,
        "named_components_have_nonzero_gradient": components_have_nonzero_gradients,
        "frozen_grad_absent": frozen_gradients_absent,
        "active_bypass_outputs_available": outputs_available,
        "active_bypass_outputs_differ": outputs_differ or not require_output_difference,
    }
    messages = []
    if missing_components:
        messages.append(f"named components have no matching parameters: {sorted(missing_components)}")
    if not components_have_gradients:
        messages.append("some named components have missing or non-finite gradients")
    if not components_have_nonzero_gradients:
        messages.append("some named components have zero gradients")
    if not target_gradients_reachable:
        messages.append(f"some {target_name} parameters have missing or non-finite gradients")
    if not target_has_nonzero_gradient:
        messages.append(f"all {target_name} gradients are zero")
    if not frozen_gradients_absent:
        messages.append(f"some frozen {frozen_name} parameters have .grad populated")
    if not outputs_available:
        messages.append("active/bypass outputs unavailable or structurally incompatible")
    elif require_output_difference and not outputs_differ:
        messages.append(f"active/bypass maximum absolute difference {difference} did not exceed {difference_atol}")

    diagnostics = ReachabilityDiagnostics(
        phase=normalized,
        stage="first_real_loss_gradient",
        passed=all(checks.values()),
        complete=outputs_available,
        target=target_name,
        frozen=frozen_name,
        checks=checks,
        parameters={
            "target": target_details,
            "frozen": frozen_details,
            **{f"component:{name}": details for name, details in component_details.items()},
        },
        output_difference=difference,
        messages=messages,
    )
    if raise_on_error and not diagnostics.passed:
        raise ReachabilityCheckError(f"gradient reachability failed: {diagnostics.as_dict()}")
    return diagnostics


def validate_reachability_and_isolation(
    activator: Any,
    network: Any,
    phase: str,
    *,
    optimizer: Any = None,
    params: Any = None,
    named_components: Optional[Mapping[str, Any]] = None,
    loss: Optional[torch.Tensor] = None,
    forward_callback: Optional[Callable[[str], Any]] = None,
    active_output: Any = None,
    bypass_output: Any = None,
    difference_atol: float = 0.0,
    require_output_difference: bool = True,
    raise_on_error: bool = False,
) -> CombinedReachabilityDiagnostics:
    """Run the static gate and, when a real graph is supplied, the gradient gate."""
    static = check_optimizer_isolation(
        activator,
        network,
        phase,
        optimizer=optimizer,
        params=params,
        named_components=named_components,
        raise_on_error=False,
    )
    gradient = None
    if loss is not None or forward_callback is not None:
        gradient = check_gradient_reachability(
            activator,
            network,
            phase,
            loss=loss,
            forward_callback=forward_callback,
            active_output=active_output,
            bypass_output=bypass_output,
            optimizer=optimizer,
            params=params,
            named_components=named_components,
            difference_atol=difference_atol,
            require_output_difference=require_output_difference,
            raise_on_error=False,
        )
    complete = static.complete and gradient is not None and gradient.complete
    passed = static.passed and gradient is not None and gradient.passed
    diagnostics = CombinedReachabilityDiagnostics(
        phase=_normalize_phase(phase),
        passed=passed,
        complete=complete,
        static=static,
        gradient=gradient,
    )
    if raise_on_error and not diagnostics.passed:
        raise ReachabilityCheckError(f"reachability/isolation validation failed: {diagnostics.as_dict()}")
    return diagnostics
