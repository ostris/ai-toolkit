import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from toolkit.config_modules import TriggerSelectiveTrainingConfig
from toolkit.prompt_utils import inject_trigger_into_prompt


_ALLOWED_INTERPOLATIONS = {'linear', 'smoothstep'}


@dataclass(frozen=True)
class NegativeStyleSample:
    category: str
    phrase: str


def _validate_keyframes(keyframes: Sequence[Dict], value_keys: Sequence[str], name: str):
    if not keyframes:
        raise ValueError(f'{name} must contain at least one keyframe')
    previous_step = None
    for index, keyframe in enumerate(keyframes):
        if 'step' not in keyframe:
            raise ValueError(f'{name} keyframe {index} is missing step')
        step = int(keyframe['step'])
        if step < 0:
            raise ValueError(f'{name} keyframe steps must be non-negative')
        if previous_step is not None and step <= previous_step:
            raise ValueError(f'{name} keyframe steps must be strictly increasing')
        previous_step = step
        for key in value_keys:
            if key not in keyframe:
                raise ValueError(f'{name} keyframe {index} is missing {key}')
            value = float(keyframe[key])
            if not math.isfinite(value):
                raise ValueError(f'{name} keyframe {index} has a non-finite {key}')
            if value < 0:
                raise ValueError(f'{name} keyframe {index} has a negative {key}')


def validate_trigger_selective_config(
    config: TriggerSelectiveTrainingConfig,
    trigger_word: Optional[str],
):
    if not config.enabled:
        return
    if not trigger_word or not trigger_word.strip():
        raise ValueError('trigger_selective_training requires a non-empty trigger_word')

    negative_styles = config.negative_styles
    if negative_styles.sample_scope != 'per_item':
        raise ValueError("trigger_selective_training negative_styles.sample_scope must be 'per_item' in v1")
    if not negative_styles.categories:
        raise ValueError('trigger_selective_training requires at least one negative style category')
    if (
        negative_styles.expected_category_count is not None
        and negative_styles.expected_category_count != len(negative_styles.categories)
    ):
        raise ValueError(
            'trigger_selective_training negative_styles.expected_category_count does not match categories'
        )

    names = set()
    probability_sum = 0.0
    for category in negative_styles.categories:
        if not category.name or not category.name.strip():
            raise ValueError('trigger_selective_training category names must be non-empty')
        if category.name in names:
            raise ValueError(f'duplicate trigger_selective_training category name: {category.name}')
        names.add(category.name)
        if not math.isfinite(category.probability) or category.probability < 0:
            raise ValueError(f'invalid probability for trigger_selective_training category {category.name}')
        probability_sum += category.probability
        if not category.phrases:
            raise ValueError(f'trigger_selective_training category {category.name} has no phrases')
        for phrase in category.phrases:
            if not isinstance(phrase, str):
                raise ValueError(f'trigger_selective_training phrases in {category.name} must be strings')
            if '[trigger]' in phrase or '[name]' in phrase:
                raise ValueError(f'trigger_selective_training phrase in {category.name} contains a placeholder')
            if trigger_word in phrase:
                raise ValueError(f'trigger_selective_training phrase in {category.name} contains trigger_word')
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError('trigger_selective_training category probabilities must sum to 1.0')

    if config.path3.loss_type != 'hinge':
        raise ValueError("trigger_selective_training path3.loss_type must be 'hinge'")
    if config.path3.decoy_gain_mode not in {'detached', 'positive_clamped'}:
        raise ValueError(
            "trigger_selective_training path3.decoy_gain_mode must be 'detached' or 'positive_clamped'"
        )
    if not math.isfinite(config.path3.gain_epsilon) or config.path3.gain_epsilon <= 0:
        raise ValueError('trigger_selective_training path3.gain_epsilon must be positive')

    for interpolation, name in (
        (config.path3.margin_schedule.interpolation, 'path3.margin_schedule'),
        (config.loss_schedule.interpolation, 'loss_schedule'),
    ):
        if interpolation not in _ALLOWED_INTERPOLATIONS:
            raise ValueError(f'{name}.interpolation must be linear or smoothstep')

    _validate_keyframes(
        config.path3.margin_schedule.keyframes,
        ('value',),
        'trigger_selective_training path3.margin_schedule',
    )
    _validate_keyframes(
        config.loss_schedule.keyframes,
        ('path1', 'path2', 'path3'),
        'trigger_selective_training loss_schedule',
    )
    for keyframe in config.loss_schedule.keyframes:
        if sum(float(keyframe[key]) for key in ('path1', 'path2', 'path3')) <= 0:
            raise ValueError('trigger_selective_training loss schedule weights cannot all be zero')

    if config.logging.log_every <= 0:
        raise ValueError('trigger_selective_training logging.log_every must be positive')
    if not config.logging.metrics_filename or os.path.basename(config.logging.metrics_filename) != config.logging.metrics_filename:
        raise ValueError('trigger_selective_training logging.metrics_filename must be a filename')
    if any(step < 0 for step in config.logging.gradient_diagnostic_steps):
        raise ValueError('trigger_selective_training gradient diagnostic steps must be non-negative')


def _interpolation_fraction(fraction: float, interpolation: str) -> float:
    fraction = min(max(float(fraction), 0.0), 1.0)
    if interpolation == 'linear':
        return fraction
    if interpolation == 'smoothstep':
        return fraction * fraction * (3.0 - 2.0 * fraction)
    raise ValueError(f'unsupported interpolation: {interpolation}')


def interpolate_keyframes(
    keyframes: Sequence[Dict],
    step: int,
    value_keys: Sequence[str],
    interpolation: str,
) -> Dict[str, float]:
    if step <= int(keyframes[0]['step']):
        return {key: float(keyframes[0][key]) for key in value_keys}
    if step >= int(keyframes[-1]['step']):
        return {key: float(keyframes[-1][key]) for key in value_keys}

    for left, right in zip(keyframes, keyframes[1:]):
        left_step = int(left['step'])
        right_step = int(right['step'])
        if left_step <= step <= right_step:
            fraction = (step - left_step) / (right_step - left_step)
            fraction = _interpolation_fraction(fraction, interpolation)
            return {
                key: float(left[key]) + fraction * (float(right[key]) - float(left[key]))
                for key in value_keys
            }
    raise RuntimeError('could not interpolate keyframes')


def get_scheduled_margin(config: TriggerSelectiveTrainingConfig, step: int) -> float:
    return interpolate_keyframes(
        config.path3.margin_schedule.keyframes,
        step,
        ('value',),
        config.path3.margin_schedule.interpolation,
    )['value']


def get_scheduled_loss_weights(config: TriggerSelectiveTrainingConfig, step: int) -> Dict[str, float]:
    weights = interpolate_keyframes(
        config.loss_schedule.keyframes,
        step,
        ('path1', 'path2', 'path3'),
        config.loss_schedule.interpolation,
    )
    if config.loss_schedule.normalize_weights:
        total = sum(weights.values())
        if total <= 0:
            raise ValueError('trigger_selective_training interpolated loss weights sum to zero')
        weights = {key: value / total for key, value in weights.items()}
    return weights


def sample_negative_styles(
    config: TriggerSelectiveTrainingConfig,
    count: int,
    rng: Optional[random.Random] = None,
) -> List[NegativeStyleSample]:
    rng = rng or random
    categories = config.negative_styles.categories
    selected = rng.choices(categories, weights=[category.probability for category in categories], k=count)
    return [NegativeStyleSample(category=category.name, phrase=rng.choice(category.phrases)) for category in selected]


def resolve_trigger_placeholder(raw_prompt: str, replacement: str, require_placeholder: bool = True) -> str:
    if require_placeholder and '[trigger]' not in raw_prompt:
        raise ValueError('TST strict placeholder validation failed: caption does not contain [trigger]')
    return inject_trigger_into_prompt(raw_prompt, replacement, add_if_not_present=False)


def resolve_prompt_variants(
    raw_prompts: Sequence[str],
    trigger_word: str,
    negative_samples: Sequence[NegativeStyleSample],
    require_placeholder: bool = True,
) -> Tuple[List[str], List[str]]:
    if len(raw_prompts) != len(negative_samples):
        raise ValueError('raw prompt and negative sample counts must match')
    trigger_prompts = []
    decoy_prompts = []
    for raw_prompt, sample in zip(raw_prompts, negative_samples):
        trigger_prompts.append(resolve_trigger_placeholder(raw_prompt, trigger_word, require_placeholder))
        decoy_prompts.append(resolve_trigger_placeholder(raw_prompt, sample.phrase, require_placeholder))
    return trigger_prompts, decoy_prompts


def per_item_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError('prediction and target shapes must match')
    return F.mse_loss(prediction.float(), target.float(), reduction='none').flatten(1).mean(1)


def normalized_gain(student_loss: torch.Tensor, base_loss: torch.Tensor, epsilon: float) -> torch.Tensor:
    return 1.0 - student_loss / (base_loss.detach() + epsilon)


def trigger_advantage_hinge(
    trigger_gain: torch.Tensor,
    decoy_gain: torch.Tensor,
    margin: float,
    decoy_gain_mode: str = 'detached',
) -> torch.Tensor:
    if decoy_gain_mode == 'detached':
        decoy_component = decoy_gain.detach()
    elif decoy_gain_mode == 'positive_clamped':
        decoy_component = torch.relu(decoy_gain)
    else:
        raise ValueError(f'unsupported decoy gain mode: {decoy_gain_mode}')
    margin_tensor = torch.as_tensor(margin, device=trigger_gain.device, dtype=trigger_gain.dtype)
    return torch.relu(margin_tensor - trigger_gain + decoy_component)


def shared_loss_target(
    trainer,
    noise: torch.Tensor,
    batch,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if hasattr(trainer.sd, 'get_loss_target'):
        target = trainer.sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps)
    elif trainer.sd.is_flow_matching:
        target = noise - batch.latents
    else:
        target = noise
    return target.detach()


def apply_differential_guidance_target(
    trainer,
    target: torch.Tensor,
    reference_prediction: torch.Tensor,
) -> torch.Tensor:
    if not (
        trainer.train_config.do_guidance_loss
        and trainer.train_config.do_differential_guidance
    ):
        return target
    scale = trainer.train_config.differential_guidance_scale
    return (reference_prediction.detach() + scale * (target - reference_prediction.detach())).detach()


@contextmanager
def network_disabled(network):
    if network is None:
        yield
        return
    previous = network.is_active
    network.is_active = False
    try:
        yield
    finally:
        network.is_active = previous


class TSTMetricsWriter:
    def __init__(self, output_dir: str, filename: str = 'tst_metrics.jsonl'):
        self.path = os.path.join(output_dir, filename)

    def write(self, record: Dict):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
