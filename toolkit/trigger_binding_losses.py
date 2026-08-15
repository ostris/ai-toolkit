from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


TensorOrFloat = Union[torch.Tensor, float]
TapTensors = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass(frozen=True)
class ContextConsistencyResult:
    per_item: torch.Tensor
    cosine_per_item: torch.Tensor
    magnitude_per_item: torch.Tensor
    valid_taps_per_item: torch.Tensor
    warmup_scale: float

    @property
    def loss(self) -> torch.Tensor:
        return self.per_item.mean()


@dataclass(frozen=True)
class A1LossResult:
    loss: torch.Tensor
    per_item: torch.Tensor
    diffusion_per_item: torch.Tensor
    bypass_diffusion_per_item: torch.Tensor
    activator_gain_per_item: torch.Tensor
    gain_floor_per_item: torch.Tensor
    context_per_item: torch.Tensor
    source_per_item: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


@dataclass(frozen=True)
class A2LossResult:
    loss: torch.Tensor
    per_item: torch.Tensor
    diffusion_per_item: torch.Tensor
    bypass_diffusion_per_item: torch.Tensor
    activator_gain_per_item: torch.Tensor
    gain_floor_per_item: torch.Tensor
    context_per_item: torch.Tensor
    source_per_item: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


def _validate_per_item(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional per-item tensor')
    return tensor


def _scalar_like(value: TensorOrFloat, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def per_item_diffusion_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes must match')
    if prediction.ndim < 2:
        raise ValueError('prediction and target must include batch and feature dimensions')
    return F.mse_loss(prediction.float(), target.float(), reduction='none').flatten(1).mean(1)


def normalized_activator_gain(
    activator_loss: torch.Tensor,
    bypass_loss: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    _validate_per_item(activator_loss, 'activator_loss')
    _validate_per_item(bypass_loss, 'bypass_loss')
    if activator_loss.shape != bypass_loss.shape:
        raise ValueError('activator_loss and bypass_loss shapes must match')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    return 1.0 - activator_loss / (bypass_loss.detach() + epsilon)


def interpolate_schedule(
    keyframes: Sequence[Mapping[str, float]],
    step: int,
    interpolation: str = 'smoothstep',
) -> float:
    if not keyframes:
        raise ValueError('schedule must contain at least one keyframe')
    if interpolation not in {'linear', 'smoothstep'}:
        raise ValueError("interpolation must be 'linear' or 'smoothstep'")

    normalized = []
    previous_step = None
    for keyframe in keyframes:
        if 'step' not in keyframe or 'value' not in keyframe:
            raise ValueError('each keyframe must contain step and value')
        frame_step = int(keyframe['step'])
        value = float(keyframe['value'])
        if previous_step is not None and frame_step <= previous_step:
            raise ValueError('keyframe steps must be strictly increasing')
        previous_step = frame_step
        normalized.append((frame_step, value))

    if step <= normalized[0][0]:
        return normalized[0][1]
    if step >= normalized[-1][0]:
        return normalized[-1][1]

    for (left_step, left_value), (right_step, right_value) in zip(normalized, normalized[1:]):
        if left_step <= step <= right_step:
            fraction = (step - left_step) / (right_step - left_step)
            if interpolation == 'smoothstep':
                fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            return left_value + fraction * (right_value - left_value)
    raise RuntimeError('could not interpolate schedule')


def scheduled_gain_floor(
    step: int,
    keyframes: Sequence[Mapping[str, float]],
    interpolation: str = 'smoothstep',
) -> float:
    return interpolate_schedule(keyframes, step, interpolation)


def activator_gain_floor_hinge(
    activator_gain: torch.Tensor,
    floor: TensorOrFloat,
) -> torch.Tensor:
    _validate_per_item(activator_gain, 'activator_gain')
    return torch.relu(_scalar_like(floor, activator_gain) - activator_gain)


def _stack_taps(taps: TapTensors, name: str) -> torch.Tensor:
    if isinstance(taps, torch.Tensor):
        if taps.ndim < 4:
            raise ValueError(f'{name} tensor must have shape [batch, taps, tokens, ...]')
        return taps
    taps = tuple(taps)
    if not taps:
        raise ValueError(f'{name} must contain at least one tap')
    first_shape = taps[0].shape
    if len(first_shape) < 3:
        raise ValueError(f'{name} tap tensors must have shape [batch, tokens, ...]')
    if any(tap.shape != first_shape for tap in taps):
        raise ValueError(f'all {name} tap tensors must have matching shapes')
    return torch.stack(taps, dim=1)


def _prepare_context_mask(
    token_mask: Optional[torch.Tensor],
    trigger_mask: Optional[torch.Tensor],
    mask_mode: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    batch, tap_count, token_count = reference.shape[:3]
    if mask_mode not in {'all', 'trigger', 'nontrigger'}:
        raise ValueError("mask_mode must be 'all', 'trigger', or 'nontrigger'")
    selected = token_mask
    if mask_mode in {'trigger', 'nontrigger'}:
        if trigger_mask is None:
            raise ValueError(f'trigger_mask is required for mask_mode={mask_mode}')
        branch_mask = trigger_mask if mask_mode == 'trigger' else ~trigger_mask.bool()
        selected = branch_mask if selected is None else selected.bool() & branch_mask.bool()
    if selected is None:
        selected = torch.ones((batch, token_count), device=reference.device, dtype=torch.bool)
    selected = selected.to(device=reference.device, dtype=torch.bool)
    if selected.ndim == 2:
        if selected.shape != (batch, token_count):
            raise ValueError('token masks must have shape [batch, tokens]')
        selected = selected[:, None, :].expand(batch, tap_count, token_count)
    elif selected.ndim == 3:
        if selected.shape != (batch, tap_count, token_count):
            raise ValueError('per-tap token masks must have shape [batch, taps, tokens]')
    else:
        raise ValueError('token masks must have two or three dimensions')
    return selected


def _prepare_source_trigger_mask(
    trigger_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    selected = _prepare_context_mask(valid_mask, trigger_mask, 'trigger', reference)
    counts = selected.sum(dim=2)
    if torch.any(counts == 0):
        empty = torch.nonzero(counts == 0, as_tuple=False)[0].tolist()
        raise ValueError(
            f'{name} trigger mask selects no valid tokens for batch {empty[0]}, tap {empty[1]}'
        )
    return selected


def _masked_pool_residual(
    active_taps: TapTensors,
    bypass_taps: TapTensors,
    trigger_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    name: str,
    pooling: str,
    expected_taps: int,
) -> torch.Tensor:
    active = _stack_taps(active_taps, f'{name}_active_taps').float()
    bypass = _stack_taps(bypass_taps, f'{name}_bypass_taps').float()
    if active.shape != bypass.shape:
        raise ValueError(f'{name} active and bypass tap collections must have matching shapes')
    if active.shape[1] != expected_taps:
        raise ValueError(f'expected {expected_taps} context taps, got {active.shape[1]} for {name}')
    if pooling not in {'mean', 'sum'}:
        raise ValueError("pooling must be 'mean' or 'sum'")

    residual = (active - bypass.detach()).flatten(start_dim=3)
    selected = _prepare_source_trigger_mask(trigger_mask, valid_mask, residual, name)
    selected_float = selected.to(residual.dtype).unsqueeze(-1)
    pooled = (residual * selected_float).sum(dim=2)
    if pooling == 'mean':
        pooled = pooled / selected_float.sum(dim=2)
    return pooled


def pooled_trigger_residual_consistency(
    source_active_taps: TapTensors,
    source_bypass_taps: TapTensors,
    reference_active_taps: TapTensors,
    reference_bypass_taps: TapTensors,
    *,
    source_trigger_mask: torch.Tensor,
    reference_trigger_mask: torch.Tensor,
    source_valid_mask: Optional[torch.Tensor] = None,
    reference_valid_mask: Optional[torch.Tensor] = None,
    pooling: str = 'mean',
    detach_reference: bool = False,
    cosine_weight: float = 1.0,
    magnitude_weight: float = 0.0,
    min_delta_norm: float = 1.0e-6,
    step: int = 0,
    warmup_steps: int = 0,
    expected_taps: int = 13,
    epsilon: float = 1.0e-8,
) -> ContextConsistencyResult:
    if cosine_weight < 0 or magnitude_weight < 0:
        raise ValueError('context consistency weights must be non-negative')
    if not isinstance(detach_reference, bool):
        raise ValueError('detach_reference must be boolean')
    if min_delta_norm < 0 or epsilon <= 0:
        raise ValueError('norm thresholds must be non-negative and epsilon must be positive')
    if warmup_steps < 0:
        raise ValueError('warmup_steps must be non-negative')

    source_delta = _masked_pool_residual(
        source_active_taps,
        source_bypass_taps,
        source_trigger_mask,
        source_valid_mask,
        name='source',
        pooling=pooling,
        expected_taps=expected_taps,
    )
    reference_delta = _masked_pool_residual(
        reference_active_taps,
        reference_bypass_taps,
        reference_trigger_mask,
        reference_valid_mask,
        name='reference',
        pooling=pooling,
        expected_taps=expected_taps,
    )
    if source_delta.shape[:2] != reference_delta.shape[:2]:
        raise ValueError('source and reference must have matching batch and tap dimensions')
    if source_delta.shape[2:] != reference_delta.shape[2:]:
        raise ValueError('source and reference tap feature dimensions must match after pooling')
    if detach_reference:
        reference_delta = reference_delta.detach()

    source_norm = torch.linalg.vector_norm(source_delta, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference_delta, dim=-1)
    cosine_loss = 1.0 - F.cosine_similarity(source_delta, reference_delta, dim=-1, eps=epsilon)
    magnitude_scale = 0.5 * (source_norm.detach() + reference_norm.detach())
    magnitude_loss = torch.abs(source_norm - reference_norm) / (magnitude_scale + epsilon)
    valid = (source_norm.detach() >= min_delta_norm) & (reference_norm.detach() >= min_delta_norm)
    valid_float = valid.to(source_delta.dtype)
    counts = valid_float.sum(dim=1)
    denominator = counts.clamp_min(1.0)
    cosine_per_item = (cosine_loss * valid_float).sum(dim=1) / denominator
    magnitude_per_item = (magnitude_loss * valid_float).sum(dim=1) / denominator

    warmup_scale = 1.0 if warmup_steps == 0 else min(max(float(step) / warmup_steps, 0.0), 1.0)
    per_item = warmup_scale * (
        float(cosine_weight) * cosine_per_item + float(magnitude_weight) * magnitude_per_item
    )
    return ContextConsistencyResult(
        per_item=per_item,
        cosine_per_item=cosine_per_item,
        magnitude_per_item=magnitude_per_item,
        valid_taps_per_item=counts,
        warmup_scale=warmup_scale,
    )


def delta_context_consistency(
    on_taps: TapTensors,
    bypass_taps: TapTensors,
    reference_on_taps: TapTensors,
    reference_bypass_taps: TapTensors,
    *,
    token_mask: Optional[torch.Tensor] = None,
    trigger_mask: Optional[torch.Tensor] = None,
    mask_mode: str = 'nontrigger',
    cosine_weight: float = 1.0,
    magnitude_weight: float = 0.0,
    min_delta_norm: float = 1.0e-6,
    step: int = 0,
    warmup_steps: int = 0,
    expected_taps: int = 13,
    epsilon: float = 1.0e-8,
) -> ContextConsistencyResult:
    on = _stack_taps(on_taps, 'on_taps').float()
    bypass = _stack_taps(bypass_taps, 'bypass_taps').float()
    reference_on = _stack_taps(reference_on_taps, 'reference_on_taps').float().detach()
    reference_bypass = _stack_taps(reference_bypass_taps, 'reference_bypass_taps').float().detach()
    if on.shape != bypass.shape or on.shape != reference_on.shape or on.shape != reference_bypass.shape:
        raise ValueError('all context tap collections must have matching shapes')
    if on.shape[1] != expected_taps:
        raise ValueError(f'expected {expected_taps} context taps, got {on.shape[1]}')
    if cosine_weight < 0 or magnitude_weight < 0:
        raise ValueError('context consistency weights must be non-negative')
    if min_delta_norm < 0 or epsilon <= 0:
        raise ValueError('norm thresholds must be non-negative and epsilon must be positive')

    on_delta = (on - bypass.detach()).flatten(start_dim=3)
    reference_delta = (reference_on - reference_bypass).flatten(start_dim=3)
    on_norm = torch.linalg.vector_norm(on_delta, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference_delta, dim=-1)
    cosine = F.cosine_similarity(on_delta, reference_delta, dim=-1, eps=epsilon)
    cosine_loss = 1.0 - cosine
    magnitude_loss = torch.abs(on_norm - reference_norm) / (reference_norm.detach() + epsilon)

    selected = _prepare_context_mask(token_mask, trigger_mask, mask_mode, on_delta)
    norm_gate = (on_norm.detach() >= min_delta_norm) & (reference_norm.detach() >= min_delta_norm)
    valid = selected & norm_gate
    valid_float = valid.to(on_delta.dtype)
    counts = valid_float.sum(dim=(1, 2))
    denominator = counts.clamp_min(1.0)
    cosine_per_item = (cosine_loss * valid_float).sum(dim=(1, 2)) / denominator
    magnitude_per_item = (magnitude_loss * valid_float).sum(dim=(1, 2)) / denominator

    if warmup_steps < 0:
        raise ValueError('warmup_steps must be non-negative')
    warmup_scale = 1.0 if warmup_steps == 0 else min(max(float(step) / warmup_steps, 0.0), 1.0)
    per_item = warmup_scale * (
        float(cosine_weight) * cosine_per_item + float(magnitude_weight) * magnitude_per_item
    )
    return ContextConsistencyResult(
        per_item=per_item,
        cosine_per_item=cosine_per_item,
        magnitude_per_item=magnitude_per_item,
        valid_taps_per_item=counts,
        warmup_scale=warmup_scale,
    )


def aggregate_paired_source_losses(
    losses_by_source: Mapping[str, torch.Tensor],
    source_weights: Optional[Mapping[str, float]] = None,
    *,
    normalize_weights: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    if not losses_by_source:
        raise ValueError('losses_by_source must not be empty')
    names = tuple(losses_by_source.keys())
    first = _validate_per_item(losses_by_source[names[0]], f'losses_by_source[{names[0]}]')
    for name in names[1:]:
        current = _validate_per_item(losses_by_source[name], f'losses_by_source[{name}]')
        if current.shape != first.shape:
            raise ValueError('all paired source losses must have matching per-item shapes')

    weights = {name: float(source_weights[name]) if source_weights is not None else 1.0 for name in names}
    if source_weights is not None and set(source_weights) != set(names):
        raise ValueError('source_weights keys must exactly match losses_by_source keys')
    if any(weight < 0 for weight in weights.values()):
        raise ValueError('source weights must be non-negative')
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError('source weights must not all be zero')
    if normalize_weights:
        weights = {name: weight / total_weight for name, weight in weights.items()}

    weighted = {name: losses_by_source[name] * weights[name] for name in names}
    aggregate = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
    return aggregate, weighted, weights


def _mean_metric(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().mean().item())


def _source_metrics(prefix: str, source_per_item: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    return {f'{prefix}/source/{name}': _mean_metric(value) for name, value in source_per_item.items()}


def compute_a1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    bypass_prediction: Optional[torch.Tensor] = None,
    gain_floor: TensorOrFloat = 0.0,
    gain_epsilon: float = 1.0e-6,
    diffusion_weight: float = 1.0,
    gain_floor_weight: float = 0.0,
    context: Optional[ContextConsistencyResult] = None,
    context_weight: float = 1.0,
    paired_source_losses: Optional[Mapping[str, torch.Tensor]] = None,
    source_weights: Optional[Mapping[str, float]] = None,
) -> A1LossResult:
    diffusion = per_item_diffusion_mse(prediction, target)
    if bypass_prediction is None:
        bypass_diffusion = diffusion.detach()
        gain = torch.zeros_like(diffusion)
        floor_per_item = torch.zeros_like(diffusion)
    else:
        bypass_diffusion = per_item_diffusion_mse(bypass_prediction, target)
        gain = normalized_activator_gain(diffusion, bypass_diffusion, gain_epsilon)
        floor_per_item = activator_gain_floor_hinge(gain, gain_floor)
    context_per_item = torch.zeros_like(diffusion) if context is None else context.per_item
    if context_per_item.shape != diffusion.shape:
        raise ValueError('context loss must match diffusion batch shape')
    base = (
        float(diffusion_weight) * diffusion
        + float(gain_floor_weight) * floor_per_item
        + float(context_weight) * context_per_item
    )
    source_inputs = {'primary': base} if paired_source_losses is None else dict(paired_source_losses)
    per_item, source_per_item, effective_weights = aggregate_paired_source_losses(
        source_inputs, source_weights
    )
    metrics = {
        'a1/loss': _mean_metric(per_item),
        'a1/diffusion_mse': _mean_metric(diffusion),
        'a1/bypass_diffusion_mse': _mean_metric(bypass_diffusion),
        'a1/activator_gain': _mean_metric(gain),
        'a1/gain_floor_loss': _mean_metric(floor_per_item),
        'a1/context': _mean_metric(context_per_item),
        'a1/context_cosine': 0.0 if context is None else _mean_metric(context.cosine_per_item),
        'a1/context_magnitude': 0.0 if context is None else _mean_metric(context.magnitude_per_item),
        'a1/context_valid_taps': 0.0 if context is None else _mean_metric(context.valid_taps_per_item),
        'a1/context_warmup_scale': 0.0 if context is None else context.warmup_scale,
    }
    metrics.update({f'a1/source_weight/{name}': weight for name, weight in effective_weights.items()})
    metrics.update(_source_metrics('a1', source_per_item))
    return A1LossResult(
        per_item.mean(), per_item, diffusion, bypass_diffusion, gain,
        floor_per_item, context_per_item, source_per_item, metrics
    )


def compute_a2_loss(
    activator_prediction: torch.Tensor,
    bypass_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gain_floor: TensorOrFloat,
    gain_epsilon: float = 1.0e-6,
    diffusion_weight: float = 1.0,
    gain_floor_weight: float = 1.0,
    context: Optional[ContextConsistencyResult] = None,
    context_weight: float = 1.0,
    paired_source_losses: Optional[Mapping[str, torch.Tensor]] = None,
    source_weights: Optional[Mapping[str, float]] = None,
) -> A2LossResult:
    diffusion = per_item_diffusion_mse(activator_prediction, target)
    bypass_diffusion = per_item_diffusion_mse(bypass_prediction, target)
    gain = normalized_activator_gain(diffusion, bypass_diffusion, gain_epsilon)
    floor_per_item = activator_gain_floor_hinge(gain, gain_floor)
    context_per_item = torch.zeros_like(diffusion) if context is None else context.per_item
    if context_per_item.shape != diffusion.shape:
        raise ValueError('context loss must match diffusion batch shape')
    base = (
        float(diffusion_weight) * diffusion
        + float(gain_floor_weight) * floor_per_item
        + float(context_weight) * context_per_item
    )
    source_inputs = {'primary': base} if paired_source_losses is None else dict(paired_source_losses)
    per_item, source_per_item, effective_weights = aggregate_paired_source_losses(
        source_inputs, source_weights
    )
    floor_tensor = _scalar_like(gain_floor, gain)
    metrics = {
        'a2/loss': _mean_metric(per_item),
        'a2/diffusion_mse': _mean_metric(diffusion),
        'a2/bypass_diffusion_mse': _mean_metric(bypass_diffusion),
        'a2/activator_gain': _mean_metric(gain),
        'a2/gain_floor': float(floor_tensor.detach().float().mean().item()),
        'a2/gain_floor_loss': _mean_metric(floor_per_item),
        'a2/gain_floor_satisfied': _mean_metric((gain.detach() >= floor_tensor).float()),
        'a2/context': _mean_metric(context_per_item),
        'a2/context_cosine': 0.0 if context is None else _mean_metric(context.cosine_per_item),
        'a2/context_magnitude': 0.0 if context is None else _mean_metric(context.magnitude_per_item),
        'a2/context_valid_taps': 0.0 if context is None else _mean_metric(context.valid_taps_per_item),
        'a2/context_warmup_scale': 0.0 if context is None else context.warmup_scale,
    }
    metrics.update({f'a2/source_weight/{name}': weight for name, weight in effective_weights.items()})
    metrics.update(_source_metrics('a2', source_per_item))
    return A2LossResult(
        per_item.mean(),
        per_item,
        diffusion,
        bypass_diffusion,
        gain,
        floor_per_item,
        context_per_item,
        source_per_item,
        metrics,
    )


@dataclass(frozen=True)
class CausalResponseDiagnostics:
    alpha: torch.Tensor
    beta: torch.Tensor
    omega: torch.Tensor
    old_gain: torch.Tensor
    reconstructed_gain: torch.Tensor
    reconstruction_error: torch.Tensor
    reference_energy: torch.Tensor
    residual_energy: torch.Tensor
    response_mse: torch.Tensor
    base_mse: torch.Tensor
    omega_tolerance: float
    uses_shared_reference: bool

    @property
    def omega_within_tolerance(self) -> torch.Tensor:
        return self.omega >= -self.omega_tolerance

    @property
    def all_omega_within_tolerance(self) -> bool:
        return bool(torch.all(self.omega_within_tolerance).item())


@dataclass(frozen=True)
class ResponseHierarchyResult:
    loss: torch.Tensor
    class_means: Dict[str, torch.Tensor]
    adjacent_deficits: Dict[str, torch.Tensor]
    adjacent_losses: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class EffectConsistencyResult:
    loss: torch.Tensor
    per_item: torch.Tensor
    structured_mean: torch.Tensor
    natural_mean: torch.Tensor
    mean_gap: torch.Tensor


def _per_item_parameter(
    value: TensorOrFloat,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = _scalar_like(value, reference)
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1 and tensor.shape[0] == reference.shape[0]:
        return tensor.reshape((reference.shape[0],) + (1,) * (reference.ndim - 1))
    try:
        return torch.broadcast_to(tensor, reference.shape)
    except RuntimeError as error:
        raise ValueError(f'{name} must be scalar, per-item, or broadcastable to the prediction shape') from error


def condition_local_response_target(
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    rho: TensorOrFloat,
    *,
    detach_base: bool = True,
    detach_target: bool = True,
) -> torch.Tensor:
    if base_prediction.shape != target.shape:
        raise ValueError('base_prediction and target shapes must match')
    if base_prediction.ndim < 2:
        raise ValueError('base_prediction and target must include batch and feature dimensions')
    base = base_prediction.detach() if detach_base else base_prediction
    endpoint = target.detach() if detach_target else target
    rho_tensor = _per_item_parameter(rho, base, 'rho')
    if not bool(torch.all(torch.isfinite(rho_tensor)).item()):
        raise ValueError('rho must contain only finite values')
    if bool(torch.any((rho_tensor < 0) | (rho_tensor > 1)).item()):
        raise ValueError('rho must be in the closed interval [0, 1]')
    return base + rho_tensor * (endpoint - base)


def per_item_response_mse(
    response_prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    rho: TensorOrFloat,
    *,
    detach_base: bool = True,
    detach_target: bool = True,
) -> torch.Tensor:
    response_target = condition_local_response_target(
        base_prediction,
        target,
        rho,
        detach_base=detach_base,
        detach_target=detach_target,
    )
    return per_item_diffusion_mse(response_prediction, response_target)


def _broadcast_reference_direction(
    v_ref: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    direction = v_ref.to(device=reference.device, dtype=reference.dtype)
    if direction.shape == reference.shape:
        return direction
    if direction.shape == reference.shape[1:]:
        direction = direction.unsqueeze(0)
    try:
        return torch.broadcast_to(direction, reference.shape)
    except RuntimeError as error:
        raise ValueError('v_ref must match or broadcast to the prediction shape') from error


def causal_response_decomposition(
    response_prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    v_ref: Optional[torch.Tensor] = None,
    epsilon: float = 1.0e-6,
    omega_tolerance: float = 1.0e-6,
    detach_base: bool = True,
    detach_target: bool = True,
    detach_v_ref: bool = True,
) -> CausalResponseDiagnostics:
    if response_prediction.shape != base_prediction.shape or response_prediction.shape != target.shape:
        raise ValueError('response_prediction, base_prediction, and target shapes must match')
    if response_prediction.ndim < 2:
        raise ValueError('predictions must include batch and feature dimensions')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    if omega_tolerance < 0:
        raise ValueError('omega_tolerance must be non-negative')

    response = response_prediction.float()
    base = base_prediction.float()
    endpoint = target.float()
    if detach_base:
        base = base.detach()
    if detach_target:
        endpoint = endpoint.detach()
    residual = response - base
    if v_ref is None:
        direction = endpoint - base
        uses_shared_reference = False
    else:
        direction = _broadcast_reference_direction(v_ref, response)
        if detach_v_ref:
            direction = direction.detach()
        uses_shared_reference = True

    residual_flat = residual.flatten(1)
    direction_flat = direction.flatten(1)
    reference_energy = direction_flat.square().mean(1)
    residual_energy = residual_flat.square().mean(1)
    cross_energy = (residual_flat * direction_flat).mean(1)
    denominator = reference_energy.detach() + float(epsilon)
    alpha = cross_energy / denominator
    beta = residual_energy / denominator
    omega = beta - alpha.square()

    reference_endpoint = base + direction
    response_mse = per_item_diffusion_mse(response, reference_endpoint)
    base_mse = per_item_diffusion_mse(base, reference_endpoint)
    old_gain = 1.0 - response_mse / (base_mse.detach() + float(epsilon))
    reconstructed_gain = 2.0 * alpha - beta
    reconstruction_error = old_gain - reconstructed_gain
    return CausalResponseDiagnostics(
        alpha=alpha,
        beta=beta,
        omega=omega,
        old_gain=old_gain,
        reconstructed_gain=reconstructed_gain,
        reconstruction_error=reconstruction_error,
        reference_energy=reference_energy,
        residual_energy=residual_energy,
        response_mse=response_mse,
        base_mse=base_mse,
        omega_tolerance=float(omega_tolerance),
        uses_shared_reference=uses_shared_reference,
    )


def _positive_floor_deficit(response: torch.Tensor, floor: TensorOrFloat) -> torch.Tensor:
    _validate_per_item(response, 'response')
    return torch.relu(_scalar_like(floor, response) - response)


def soft_response_floor(
    response: torch.Tensor,
    floor: TensorOrFloat,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    _validate_per_item(response, 'response')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    deficit = _scalar_like(floor, response) - response
    return float(temperature) * F.softplus(deficit / float(temperature))


def huber_response_floor(
    response: torch.Tensor,
    floor: TensorOrFloat,
    *,
    delta: float = 0.1,
) -> torch.Tensor:
    if delta <= 0:
        raise ValueError('delta must be positive')
    deficit = _positive_floor_deficit(response, floor)
    delta_tensor = _scalar_like(delta, response)
    return torch.where(
        deficit <= delta_tensor,
        0.5 * deficit.square() / delta_tensor,
        deficit - 0.5 * delta_tensor,
    )


def clamped_response_floor(
    response: torch.Tensor,
    floor: TensorOrFloat,
    *,
    max_deficit: float = 1.0,
    squared: bool = True,
) -> torch.Tensor:
    if max_deficit <= 0:
        raise ValueError('max_deficit must be positive')
    deficit = _positive_floor_deficit(response, floor).clamp_max(float(max_deficit))
    return deficit.square() if squared else deficit


def response_floor_penalty(
    response: torch.Tensor,
    floor: TensorOrFloat,
    *,
    mode: str = 'soft',
    temperature: float = 0.05,
    huber_delta: float = 0.1,
    max_deficit: float = 1.0,
    squared: bool = True,
) -> torch.Tensor:
    if mode == 'soft':
        return soft_response_floor(response, floor, temperature=temperature)
    if mode == 'huber':
        return huber_response_floor(response, floor, delta=huber_delta)
    if mode == 'clamped':
        return clamped_response_floor(response, floor, max_deficit=max_deficit, squared=squared)
    raise ValueError("mode must be 'soft', 'huber', or 'clamped'")


def off_direction_penalty(
    omega: torch.Tensor,
    *,
    tolerance: float = 1.0e-6,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    _validate_per_item(omega, 'omega')
    if tolerance < 0:
        raise ValueError('tolerance must be non-negative')
    if max_value is not None and max_value <= 0:
        raise ValueError('max_value must be positive when provided')
    penalty = torch.relu(omega - float(tolerance))
    return penalty if max_value is None else penalty.clamp_max(float(max_value))


def _normalized_adjacent_margins(
    order: Sequence[str],
    margins: Union[TensorOrFloat, Sequence[float], Mapping[str, float]],
) -> Dict[str, float]:
    pair_names = [f'{lower}->{higher}' for lower, higher in zip(order, order[1:])]
    if isinstance(margins, Mapping):
        if set(margins) != set(pair_names):
            raise ValueError('margin mapping keys must exactly match adjacent condition pairs')
        result = {name: float(margins[name]) for name in pair_names}
    elif isinstance(margins, Sequence) and not isinstance(margins, (str, bytes, torch.Tensor)):
        if len(margins) != len(pair_names):
            raise ValueError('margin sequence must have one value per adjacent condition pair')
        result = {name: float(value) for name, value in zip(pair_names, margins)}
    else:
        margin = float(torch.as_tensor(margins).item())
        result = {name: margin for name in pair_names}
    if any(value < 0 for value in result.values()):
        raise ValueError('hierarchy margins must be non-negative')
    return result


def adjacent_response_hierarchy_loss(
    responses_by_class: Mapping[str, torch.Tensor],
    *,
    order: Sequence[str] = ('far', 'neutral', 'hard', 'trigger'),
    margins: Union[TensorOrFloat, Sequence[float], Mapping[str, float]] = 0.0,
    mode: str = 'soft',
    temperature: float = 0.05,
    huber_delta: float = 0.1,
    max_deficit: float = 1.0,
) -> ResponseHierarchyResult:
    order = tuple(order)
    if len(order) < 2 or len(set(order)) != len(order):
        raise ValueError('order must contain at least two unique class names')
    if set(responses_by_class) != set(order):
        raise ValueError('responses_by_class keys must exactly match order')
    class_means: Dict[str, torch.Tensor] = {}
    for name in order:
        values = _validate_per_item(responses_by_class[name], f'responses_by_class[{name}]')
        if values.numel() == 0:
            raise ValueError(f'responses_by_class[{name}] must not be empty')
        class_means[name] = values.mean()

    normalized_margins = _normalized_adjacent_margins(order, margins)
    adjacent_deficits: Dict[str, torch.Tensor] = {}
    adjacent_losses: Dict[str, torch.Tensor] = {}
    for lower, higher in zip(order, order[1:]):
        pair_name = f'{lower}->{higher}'
        gap = class_means[higher] - class_means[lower]
        deficit = _scalar_like(normalized_margins[pair_name], gap) - gap
        adjacent_deficits[pair_name] = deficit
        scalar_gap = gap.reshape(1)
        pair_loss = response_floor_penalty(
            scalar_gap,
            normalized_margins[pair_name],
            mode=mode,
            temperature=temperature,
            huber_delta=huber_delta,
            max_deficit=max_deficit,
        ).squeeze(0)
        adjacent_losses[pair_name] = pair_loss
    loss = torch.stack(tuple(adjacent_losses.values())).mean()
    return ResponseHierarchyResult(loss, class_means, adjacent_deficits, adjacent_losses)


def structured_natural_effect_consistency(
    structured_effect: torch.Tensor,
    natural_effect: torch.Tensor,
    *,
    reduction: str = 'paired',
    huber_delta: Optional[float] = None,
) -> EffectConsistencyResult:
    structured = _validate_per_item(structured_effect, 'structured_effect')
    natural = _validate_per_item(natural_effect, 'natural_effect')
    if reduction not in {'paired', 'mean'}:
        raise ValueError("reduction must be 'paired' or 'mean'")
    if reduction == 'paired':
        if structured.shape != natural.shape:
            raise ValueError('paired structured and natural effects must have matching shapes')
        difference = structured - natural
    else:
        difference = (structured.mean() - natural.mean()).reshape(1)
    if huber_delta is None:
        per_item = difference.square()
    else:
        if huber_delta <= 0:
            raise ValueError('huber_delta must be positive')
        absolute = difference.abs()
        delta = _scalar_like(huber_delta, difference)
        per_item = torch.where(
            absolute <= delta,
            0.5 * difference.square() / delta,
            absolute - 0.5 * delta,
        )
    structured_mean = structured.mean()
    natural_mean = natural.mean()
    return EffectConsistencyResult(
        loss=per_item.mean(),
        per_item=per_item,
        structured_mean=structured_mean,
        natural_mean=natural_mean,
        mean_gap=structured_mean - natural_mean,
    )


conditional_response_target = condition_local_response_target
causal_residual_decomposition = causal_response_decomposition
batch_mean_response_hierarchy_loss = adjacent_response_hierarchy_loss
