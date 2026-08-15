import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.config_modules import TriggerDataSplitConfig, TriggerValidationConfig


PredictionOrLoss = Union[torch.Tensor, float, int]
EvaluationCallable = Callable[[], PredictionOrLoss]


@dataclass(frozen=True)
class TriggerValidationResult:
    trigger_gain: float
    decoy_gain: float
    raw_gap: float
    effective_gap: float
    base_trigger_loss: float
    student_trigger_loss: float
    base_decoy_loss: float
    student_decoy_loss: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _validate_filename(value: str, field_name: str):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty filename')
    if os.path.basename(value) != value or value in {'.', '..'}:
        raise ValueError(f'{field_name} must be a filename, not a path')


def _validate_manifest_path(value: Optional[str], field_name: str, require_exists: bool):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty path when validation is enabled')
    normalized = os.path.abspath(os.path.expanduser(value))
    if require_exists and not os.path.isfile(normalized):
        raise ValueError(f'{field_name} does not exist or is not a file: {value}')


def validate_trigger_data_split_config(
    config: 'TriggerDataSplitConfig',
    *,
    require_manifest_file: bool = False,
):
    if not isinstance(config.enabled, bool):
        raise ValueError('three_phase_trigger_training.data_split.enabled must be boolean')
    if not config.enabled:
        return
    if config.seed < 0:
        raise ValueError('three_phase_trigger_training.data_split.seed must be non-negative')
    if not math.isfinite(config.heldout_fraction) or not 0.0 < config.heldout_fraction < 1.0:
        raise ValueError(
            'three_phase_trigger_training.data_split.heldout_fraction must be strictly between 0 and 1'
        )
    if not isinstance(config.reuse_existing, bool):
        raise ValueError('three_phase_trigger_training.data_split.reuse_existing must be boolean')
    _validate_manifest_path(
        config.manifest_path,
        'three_phase_trigger_training.data_split.manifest_path',
        require_manifest_file,
    )


def validate_trigger_validation_config(
    config: 'TriggerValidationConfig',
    *,
    require_manifest_files: bool = True,
    data_split_config: Optional['TriggerDataSplitConfig'] = None,
):
    if not isinstance(config.enabled, bool):
        raise ValueError('three_phase_trigger_training.validation.enabled must be boolean')
    if not config.enabled:
        return
    if config.every <= 0:
        raise ValueError('three_phase_trigger_training.validation.every must be positive')
    if config.seed < 0:
        raise ValueError('three_phase_trigger_training.validation.seed must be non-negative')
    if bool(config.fixed_timesteps) == bool(config.fixed_sigmas):
        raise ValueError('validation must configure exactly one of fixed_timesteps or fixed_sigmas')
    if any(value < 0 for value in config.fixed_timesteps):
        raise ValueError('validation.fixed_timesteps must be non-negative')
    if any(not math.isfinite(value) or value < 0 for value in config.fixed_sigmas):
        raise ValueError('validation.fixed_sigmas must be finite and non-negative')
    if len(set(config.fixed_timesteps)) != len(config.fixed_timesteps):
        raise ValueError('validation.fixed_timesteps must be unique')
    if len(set(config.fixed_sigmas)) != len(config.fixed_sigmas):
        raise ValueError('validation.fixed_sigmas must be unique')

    split_manifest = getattr(config, 'data_split_manifest', None)
    managed_split_enabled = data_split_config is not None and data_split_config.enabled
    if managed_split_enabled:
        configured_manifest = split_manifest or data_split_config.manifest_path
        _validate_manifest_path(
            configured_manifest,
            'three_phase_trigger_training.data_split.manifest_path',
            False,
        )
        if config.train_probe_manifest is not None or config.heldout_manifest is not None:
            raise ValueError(
                'managed data_split cannot be combined with legacy train-probe/held-out manifests'
            )
    elif split_manifest is not None:
        _validate_manifest_path(
            split_manifest,
            'validation.data_split_manifest',
            require_manifest_files,
        )
        if config.train_probe_manifest is not None or config.heldout_manifest is not None:
            raise ValueError(
                'validation.data_split_manifest cannot be combined with legacy train-probe/held-out manifests'
            )
    else:
        _validate_manifest_path(
            config.train_probe_manifest,
            'validation.train_probe_manifest',
            require_manifest_files,
        )
        _validate_manifest_path(
            config.heldout_manifest,
            'validation.heldout_manifest',
            require_manifest_files,
        )
        if os.path.abspath(os.path.expanduser(config.train_probe_manifest)) == os.path.abspath(
            os.path.expanduser(config.heldout_manifest)
        ):
            raise ValueError('validation train-probe and held-out manifests must be different files')

    if not config.caption_sources or any(
        not isinstance(source, str) or not source.strip() for source in config.caption_sources
    ):
        raise ValueError('validation.caption_sources must contain non-empty strings')
    if len(set(config.caption_sources)) != len(config.caption_sources):
        raise ValueError('validation.caption_sources must be unique')
    if not config.negative_phrases or any(
        not isinstance(phrase, str) for phrase in config.negative_phrases
    ):
        raise ValueError('validation.negative_phrases must contain strings')
    if not math.isfinite(config.gain_epsilon) or config.gain_epsilon <= 0:
        raise ValueError('validation.gain_epsilon must be positive')

    filenames = (
        config.train_probe_output_filename,
        config.heldout_output_filename,
        config.aggregate_output_filename,
    )
    for field_name, filename in zip(
        (
            'validation.train_probe_output_filename',
            'validation.heldout_output_filename',
            'validation.aggregate_output_filename',
        ),
        filenames,
    ):
        _validate_filename(filename, field_name)
    if len(set(filenames)) != len(filenames):
        raise ValueError('validation output filenames must be unique')


@contextmanager
def isolated_rng(seed: Optional[int] = None):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def make_python_rng(seed: int) -> random.Random:
    return random.Random(seed)


def make_torch_generator(seed: int, device: Union[str, torch.device] = 'cpu') -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _to_loss(
    value: PredictionOrLoss,
    target: Optional[torch.Tensor],
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], PredictionOrLoss]],
    name: str,
) -> float:
    if target is not None:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'{name} prediction must be a tensor when target is provided')
        if value.shape != target.shape:
            raise ValueError(f'{name} prediction and target shapes must match')
        computed = loss_fn(value, target) if loss_fn is not None else F.mse_loss(
            value.float(), target.float(), reduction='mean'
        )
    else:
        computed = value
    tensor = torch.as_tensor(computed).detach().float().cpu()
    if tensor.numel() != 1:
        raise ValueError(f'{name} must produce a scalar loss for one item')
    result = float(tensor.item())
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{name} loss must be finite and non-negative')
    return result


def evaluate_gain(
    base_trigger: EvaluationCallable,
    student_trigger: EvaluationCallable,
    base_decoy: EvaluationCallable,
    student_decoy: EvaluationCallable,
    *,
    target: Optional[torch.Tensor] = None,
    trigger_target: Optional[torch.Tensor] = None,
    decoy_target: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], PredictionOrLoss]] = None,
    epsilon: float = 1.0e-6,
) -> TriggerValidationResult:
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError('epsilon must be positive')
    if target is not None and (trigger_target is not None or decoy_target is not None):
        raise ValueError('target cannot be combined with branch-specific targets')
    effective_trigger_target = target if trigger_target is None else trigger_target
    effective_decoy_target = target if decoy_target is None else decoy_target

    with torch.no_grad():
        base_trigger_loss = _to_loss(base_trigger(), effective_trigger_target, loss_fn, 'base_trigger')
        student_trigger_loss = _to_loss(student_trigger(), effective_trigger_target, loss_fn, 'student_trigger')
        base_decoy_loss = _to_loss(base_decoy(), effective_decoy_target, loss_fn, 'base_decoy')
        student_decoy_loss = _to_loss(student_decoy(), effective_decoy_target, loss_fn, 'student_decoy')

    trigger_gain = 1.0 - student_trigger_loss / (base_trigger_loss + epsilon)
    decoy_gain = 1.0 - student_decoy_loss / (base_decoy_loss + epsilon)
    return TriggerValidationResult(
        trigger_gain=trigger_gain,
        decoy_gain=decoy_gain,
        raw_gap=trigger_gain - decoy_gain,
        effective_gap=trigger_gain - max(decoy_gain, 0.0),
        base_trigger_loss=base_trigger_loss,
        student_trigger_loss=student_trigger_loss,
        base_decoy_loss=base_decoy_loss,
        student_decoy_loss=student_decoy_loss,
    )


def aggregate_results(records: Iterable[Union[TriggerValidationResult, Mapping[str, Any]]]) -> Dict[str, Any]:
    normalized = [record.as_dict() if isinstance(record, TriggerValidationResult) else dict(record) for record in records]
    if not normalized:
        raise ValueError('cannot aggregate an empty validation result collection')
    metrics = (
        'trigger_gain',
        'decoy_gain',
        'raw_gap',
        'effective_gap',
        'base_trigger_loss',
        'student_trigger_loss',
        'base_decoy_loss',
        'student_decoy_loss',
    )
    aggregate: Dict[str, Any] = {'count': len(normalized)}
    for metric in metrics:
        values = [float(record[metric]) for record in normalized]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f'cannot aggregate non-finite {metric}')
        aggregate[metric] = sum(values) / len(values)
    aggregate['trigger_gain_positive_rate'] = sum(
        float(record['trigger_gain']) > 0.0 for record in normalized
    ) / len(normalized)
    aggregate['effective_gap_positive_rate'] = sum(
        float(record['effective_gap']) > 0.0 for record in normalized
    ) / len(normalized)
    return aggregate


class JSONLWriter:
    def __init__(self, output_dir: str, filename: str):
        _validate_filename(filename, 'filename')
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError('output_dir must be a non-empty path')
        self.path = os.path.join(output_dir, filename)

    def write(self, record: Union[TriggerValidationResult, Mapping[str, Any]]):
        payload = record.as_dict() if isinstance(record, TriggerValidationResult) else dict(record)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
