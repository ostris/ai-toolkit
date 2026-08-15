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

    schedules = [
        (config.path3.margin_schedule, 'path3.margin_schedule'),
        (config.loss_schedule, 'loss_schedule'),
    ]
    if config.path3.gain_floor.enabled:
        schedules.append((config.path3.gain_floor.schedule, 'path3.gain_floor.schedule'))
    if config.caption_sources.enabled:
        schedules.append((config.caption_sources.schedule, 'caption_sources.schedule'))
    for schedule, name in schedules:
        if schedule.interpolation not in _ALLOWED_INTERPOLATIONS:
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

    if config.path3.gain_floor.enabled:
        if not math.isfinite(config.path3.gain_floor.weight) or config.path3.gain_floor.weight < 0:
            raise ValueError('trigger_selective_training path3.gain_floor.weight must be non-negative')
        _validate_keyframes(
            config.path3.gain_floor.schedule.keyframes,
            ('value',),
            'trigger_selective_training path3.gain_floor.schedule',
        )

    if config.caption_sources.enabled:
        if not config.caption_sources.sources:
            raise ValueError('trigger_selective_training caption_sources requires at least one source')
        source_names = [source.name for source in config.caption_sources.sources]
        if any(not name or not name.strip() for name in source_names):
            raise ValueError('trigger_selective_training caption source names must be non-empty')
        if len(set(source_names)) != len(source_names):
            raise ValueError('trigger_selective_training caption source names must be unique')
        main_sources = [source for source in config.caption_sources.sources if source.use_main_dataset]
        if len(main_sources) != 1:
            raise ValueError('trigger_selective_training caption_sources requires exactly one use_main_dataset source')
        if len(config.caption_sources.sources) == 1:
            source = config.caption_sources.sources[0]
            if not source.use_main_dataset or source.caption_ext.lower() != '.json':
                raise ValueError(
                    'trigger_selective_training single caption source must be a JSON use_main_dataset source'
                )
        for source in config.caption_sources.sources:
            if source.format not in {'text', 'json'}:
                raise ValueError(f'unsupported caption source format for {source.name}: {source.format}')
            if not source.caption_ext:
                raise ValueError(f'caption source {source.name} requires caption_ext')
            if not source.use_main_dataset and not source.path:
                raise ValueError(f'caption source {source.name} requires path')
            if source.format == 'json' and not source.caption_field:
                raise ValueError(f'JSON caption source {source.name} requires caption_field')
        _validate_keyframes(
            config.caption_sources.schedule.keyframes,
            tuple(source_names),
            'trigger_selective_training caption_sources.schedule',
        )
        for keyframe in config.caption_sources.schedule.keyframes:
            if sum(float(keyframe[name]) for name in source_names) <= 0:
                raise ValueError('trigger_selective_training caption source weights cannot all be zero')

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


def _normalize_scheduled_weights(weights: Dict[str, float], enabled: bool, name: str) -> Dict[str, float]:
    if not enabled:
        return weights
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f'{name} interpolated weights sum to zero')
    return {key: value / total for key, value in weights.items()}


def get_scheduled_loss_weights(config: TriggerSelectiveTrainingConfig, step: int) -> Dict[str, float]:
    weights = interpolate_keyframes(
        config.loss_schedule.keyframes,
        step,
        ('path1', 'path2', 'path3'),
        config.loss_schedule.interpolation,
    )
    return _normalize_scheduled_weights(
        weights,
        config.loss_schedule.normalize_weights,
        'trigger_selective_training loss schedule',
    )


def get_scheduled_caption_source_weights(
    config: TriggerSelectiveTrainingConfig,
    step: int,
) -> Dict[str, float]:
    if not config.caption_sources.enabled:
        return {}
    source_names = tuple(source.name for source in config.caption_sources.sources)
    weights = interpolate_keyframes(
        config.caption_sources.schedule.keyframes,
        step,
        source_names,
        config.caption_sources.schedule.interpolation,
    )
    return _normalize_scheduled_weights(
        weights,
        config.caption_sources.schedule.normalize_weights,
        'trigger_selective_training caption source schedule',
    )


def sample_caption_sources(
    config: TriggerSelectiveTrainingConfig,
    step: int,
    count: int,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], Dict[str, float]]:
    weights = get_scheduled_caption_source_weights(config, step)
    if not weights:
        return [], weights
    rng = rng or random
    names = list(weights.keys())
    return rng.choices(names, weights=[weights[name] for name in names], k=count), weights


def get_scheduled_gain_floor(config: TriggerSelectiveTrainingConfig, step: int) -> float:
    if not config.path3.gain_floor.enabled:
        return 0.0
    return interpolate_keyframes(
        config.path3.gain_floor.schedule.keyframes,
        step,
        ('value',),
        config.path3.gain_floor.schedule.interpolation,
    )['value']


def trigger_gain_floor_hinge(trigger_gain: torch.Tensor, gain_floor: float) -> torch.Tensor:
    floor_tensor = torch.as_tensor(
        gain_floor,
        device=trigger_gain.device,
        dtype=trigger_gain.dtype,
    )
    return torch.relu(floor_tensor - trigger_gain)


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
    trigger_word,
    negative_samples: Sequence[NegativeStyleSample],
    require_placeholder: bool = True,
) -> Tuple[List[str], List[str]]:
    if len(raw_prompts) != len(negative_samples):
        raise ValueError('raw prompt and negative sample counts must match')
    trigger_words = (
        list(trigger_word)
        if isinstance(trigger_word, (list, tuple))
        else [trigger_word] * len(raw_prompts)
    )
    if len(trigger_words) != len(raw_prompts):
        raise ValueError('trigger word and raw prompt counts must match')
    trigger_prompts = []
    decoy_prompts = []
    for raw_prompt, effective_trigger, sample in zip(raw_prompts, trigger_words, negative_samples):
        trigger_prompts.append(resolve_trigger_placeholder(raw_prompt, effective_trigger, require_placeholder))
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
