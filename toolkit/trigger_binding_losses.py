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
