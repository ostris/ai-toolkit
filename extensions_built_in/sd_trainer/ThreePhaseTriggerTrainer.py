import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

import yaml

from jobs.process import BaseExtensionProcess
from toolkit.config_modules import (
    ThreePhaseTriggerTrainingConfig,
    TriggerBindingPhaseSourceConfig,
    validate_three_phase_trigger_training_config,
)
from toolkit.paths import TOOLKIT_ROOT, get_path


class ThreePhaseTriggerTrainer(BaseExtensionProcess):
    PHASE_NAMES = ('a1', 'b', 'a2')

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.three_phase_config = ThreePhaseTriggerTrainingConfig(
            **self.get_conf('three_phase_trigger_training', {})
        )
        if self.three_phase_config.literal is None:
            self.three_phase_config.literal = self.get_conf('trigger_word', None)
        validate_three_phase_trigger_training_config(
            self.three_phase_config,
            self.get_conf('trigger_word', None),
        )
        self.phase_snapshots: Dict[str, str] = {}
        self.phase_contracts: Dict[str, str] = {}
        self.active_phase: Optional[str] = None
        if not self.three_phase_config.enabled:
            return

        self.training_folder = get_path(self.get_conf('training_folder', required=True))
        configured_output_root = self.three_phase_config.artifacts.output_root
        if configured_output_root is None:
            configured_output_root = os.path.join(self.training_folder, self.name)
        self.run_root = get_path(configured_output_root)
        self.snapshot_root = os.path.join(self.run_root, 'phase_configs')
        self.contract_root = os.path.join(self.run_root, 'contracts')
        os.makedirs(self.snapshot_root, exist_ok=True)
        os.makedirs(self.contract_root, exist_ok=True)

    def _phase_root(self, phase_name: str) -> str:
        return os.path.join(self.run_root, f'phase_{phase_name}')

    def _phase_artifacts(self, phase_name: str):
        return getattr(self.three_phase_config.artifacts, f'phase_{phase_name}')

    def _source_path(
        self,
        source: TriggerBindingPhaseSourceConfig,
        artifact_field: str,
    ) -> Optional[str]:
        if source.path is not None:
            return get_path(source.path)
        if source.phase is None:
            return None

        source_artifacts = self._phase_artifacts(source.phase)
        artifact_name = getattr(source_artifacts, artifact_field)
        source_root = self._phase_root(source.phase)
        if source.step == 'final':
            return os.path.join(source_root, source_artifacts.final_dir, artifact_name)
        return os.path.join(
            source_root,
            source_artifacts.checkpoint_dir,
            str(source.step),
            artifact_name,
        )

    def _resolve_phase_sources(self, phase_name: str) -> Dict[str, Optional[str]]:
        phase = self.three_phase_config.get_phase(phase_name)
        return {
            'embedding': self._source_path(
                phase.text_activator_source,
                'embedding_filename',
            ),
            'te_adapter': self._source_path(
                phase.text_activator_source,
                'te_adapter_filename',
            ),
            'tap_adapters': self._source_path(
                phase.text_activator_source,
                'tap_adapter_filename',
            ),
            'diffusion_lora': self._source_path(
                phase.diffusion_lora_source,
                'diffusion_lora_filename',
            ),
        }

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _source_records(self, phase_name: str) -> Dict[str, Dict[str, Optional[str]]]:
        records = {}
        for name, path in self._resolve_phase_sources(phase_name).items():
            records[name] = {
                'path': path,
                'sha256': self._sha256_file(path) if path is not None and os.path.isfile(path) else None,
            }
        return records

    def _required_phase_sources(self, phase_name: str) -> Dict[str, str]:
        phase = self.three_phase_config.get_phase(phase_name)
        sources = self._resolve_phase_sources(phase_name)
        required = {}
        if phase.text_activator_source.phase is not None or phase.text_activator_source.path is not None:
            enabled_components = {
                'embedding': self.three_phase_config.text_activator.embedding.enabled,
                'te_adapter': self.three_phase_config.text_activator.te_adapter.enabled,
                'tap_adapters': self.three_phase_config.text_activator.tap_adapters.enabled,
            }
            required.update({name: sources[name] for name, enabled in enabled_components.items() if enabled})
        if phase.diffusion_lora_source.phase is not None or phase.diffusion_lora_source.path is not None:
            required['diffusion_lora'] = sources['diffusion_lora']
        return required

    def _verify_phase_inputs(self, phase_name: str):
        missing = [
            f'{name}: {path}'
            for name, path in self._required_phase_sources(phase_name).items()
            if path is None or not os.path.isfile(path)
        ]
        if missing:
            raise FileNotFoundError(
                f'Three-phase trigger phase {phase_name} is missing required input artifact(s): '
                + '; '.join(missing)
            )

    @staticmethod
    def _phase_caption_sources(phase, parent_caption_sources: Dict) -> Dict:
        caption_sources = copy.deepcopy(phase.caption_sources)
        if not caption_sources:
            return caption_sources
        paired = caption_sources.pop('paired', None)
        phase_sources = caption_sources.get('sources', [])
        if isinstance(paired, bool):
            if paired and phase_sources:
                paired = [source.get('name') for source in phase_sources]
            else:
                paired = (
                    [source.get('name') for source in parent_caption_sources.get('sources', [])]
                    if paired else None
                )
        if isinstance(paired, list):
            available_sources = phase_sources or parent_caption_sources.get('sources', [])
            paired_names = set(paired)
            caption_sources.setdefault('enabled', True)
            caption_sources['sources'] = [
                copy.deepcopy(source)
                for source in available_sources
                if source.get('name') in paired_names
            ]
            missing = paired_names - {source.get('name') for source in caption_sources['sources']}
            if missing:
                raise ValueError(
                    f'phase {phase.phase_name} caption_sources.paired references unknown source(s): '
                    + ', '.join(sorted(missing))
                )
        weights = caption_sources.pop('weights', None)
        sources = caption_sources.get('sources', [])
        if sources and not caption_sources.get('schedule'):
            if weights is None:
                weights = {source['name']: 1.0 for source in sources}
            caption_sources['schedule'] = {
                'interpolation': 'smoothstep',
                'normalize_weights': True,
                'keyframes': [
                    {'step': 0, **{source['name']: float(weights.get(source['name'], 0.0)) for source in sources}}
                ],
            }
        return caption_sources

    def _build_child_process_config(self, phase_name: str) -> OrderedDict:
        phase = self.three_phase_config.get_phase(phase_name)
        child = copy.deepcopy(OrderedDict(self.raw_process_config))
        child['type'] = 'sd_trainer'
        child['name'] = f'phase_{phase_name}'
        child['training_folder'] = self.run_root
        child['train'] = copy.deepcopy(child.get('train', {}))
        child['train']['steps'] = phase.steps
        child['train']['optimizer'] = phase.optimizer
        child['train']['optimizer_params'] = copy.deepcopy(phase.optimizer_params)
        child['save'] = copy.deepcopy(child.get('save', {}))
        if phase.save_steps:
            child['save']['save_every'] = min(phase.save_steps)

        learning_rate_aliases = {
            'lr': 'lr',
            'diffusion': 'lr',
            'diffusion_lora': 'lr',
            'unet': 'unet_lr',
            'text_encoder': 'text_encoder_lr',
            'embedding': 'embedding_lr',
            'adapter': 'adapter_lr',
        }
        for key, value in phase.learning_rates.items():
            child['train'][learning_rate_aliases.get(key, key)] = value

        child['train'].update(copy.deepcopy(phase.train))
        child['three_phase_trigger_training'] = copy.deepcopy(
            child.get('three_phase_trigger_training', {})
        )
        runtime = child['three_phase_trigger_training'].setdefault('runtime', {})
        runtime.update({
            'active_phase': phase_name,
            'orchestrated': True,
            'run_root': self.run_root,
            'config_snapshot': self.phase_snapshots.get(
                phase_name,
                os.path.join(self.snapshot_root, f'phase_{phase_name}.yaml'),
            ),
            'completion_contract': self.phase_contracts.get(
                phase_name,
                os.path.join(self.contract_root, f'phase_{phase_name}.json'),
            ),
        })
        sources = self._resolve_phase_sources(phase_name)
        parent_tst = child.get('trigger_selective_training', {})
        caption_sources = self._phase_caption_sources(
            phase,
            parent_tst.get('caption_sources', {}),
        )
        if not caption_sources and phase_name in ('a1', 'a2'):
            caption_sources = copy.deepcopy(parent_tst.get('caption_sources', {}))
        child['three_phase_trigger_training']['phase_runtime'] = {
            'caption_sources': caption_sources,
            'losses': copy.deepcopy(phase.losses),
            'save_steps': list(phase.save_steps),
            'resume': {
                'enabled': phase.resume.enabled,
                'checkpoint': phase.resume.checkpoint,
            },
            'sources': sources,
        }
        child_phase = child['three_phase_trigger_training'][f'phase_{phase_name}']
        text_source_paths = [
            sources[name]
            for name in ('embedding', 'te_adapter', 'tap_adapters')
            if sources[name] is not None
        ]
        child_phase['text_activator_source'] = {
            'path': text_source_paths[0] if len(set(text_source_paths)) == 1 else None,
            'step': 'final',
        }
        child_phase['diffusion_lora_source'] = {
            'path': sources['diffusion_lora'],
            'step': 'final',
        }
        child['trigger_selective_training'] = copy.deepcopy(child.get('trigger_selective_training', {}))
        child['trigger_selective_training']['caption_sources'] = copy.deepcopy(caption_sources)
        if phase_name == 'a2' and sources['diffusion_lora'] is not None:
            child['network'] = copy.deepcopy(child.get('network', {}))
            child['network']['pretrained_lora_path'] = sources['diffusion_lora']
        if phase_name == 'b':
            child['trigger_selective_training'] = copy.deepcopy(child.get('trigger_selective_training', {}))
            child['trigger_selective_training']['phase_local_step'] = True
            child['trigger_selective_training']['source_artifacts'] = self._source_records(phase_name)
        return child

    def build_child_job_config(self, phase_name: str) -> OrderedDict:
        child_process = self._build_child_process_config(phase_name)
        child_job = OrderedDict({
            'job': 'extension',
            'config': OrderedDict({
                'name': child_process['name'],
                'process': [child_process],
            }),
            'meta': copy.deepcopy(self.job.meta),
        })
        return child_job

    def _write_yaml_atomic(self, path: str, data: OrderedDict):
        temp_path = path + '.tmp'
        serializable = json.loads(json.dumps(data))
        with open(temp_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
        os.replace(temp_path, path)

    def _write_json_atomic(self, path: str, data: Dict):
        temp_path = path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(temp_path, path)

    def write_phase_snapshot(self, phase_name: str) -> str:
        snapshot_path = os.path.join(self.snapshot_root, f'phase_{phase_name}.yaml')
        self.phase_snapshots[phase_name] = snapshot_path
        self._write_yaml_atomic(snapshot_path, self.build_child_job_config(phase_name))
        return snapshot_path

    def completion_contract(self, phase_name: str, status: str, return_code: Optional[int] = None) -> Dict:
        phase = self.three_phase_config.get_phase(phase_name)
        phase_artifacts = self._phase_artifacts(phase_name)
        phase_root = self._phase_root(phase_name)
        return {
            'schema_version': 1,
            'phase': phase_name,
            'status': status,
            'return_code': return_code,
            'completed_at': datetime.now(timezone.utc).isoformat() if status in ('completed', 'failed') else None,
            'config_snapshot': self.phase_snapshots[phase_name],
            'phase_root': phase_root,
            'steps': phase.steps,
            'inputs': self._source_records(phase_name),
            'artifacts': {
                'metrics_file': os.path.join(phase_root, phase_artifacts.metrics_file),
                'console_log': os.path.join(phase_root, phase_artifacts.console_log),
                'checkpoint_dir': os.path.join(phase_root, phase_artifacts.checkpoint_dir),
                'final_dir': os.path.join(phase_root, phase_artifacts.final_dir),
                'embedding': os.path.join(phase_root, phase_artifacts.final_dir, phase_artifacts.embedding_filename),
                'te_adapter': os.path.join(phase_root, phase_artifacts.final_dir, phase_artifacts.te_adapter_filename),
                'tap_adapters': os.path.join(phase_root, phase_artifacts.final_dir, phase_artifacts.tap_adapter_filename),
                'diffusion_lora': os.path.join(phase_root, phase_artifacts.final_dir, phase_artifacts.diffusion_lora_filename),
            },
        }

    def write_completion_contract(
        self,
        phase_name: str,
        status: str,
        return_code: Optional[int] = None,
    ) -> str:
        contract_path = os.path.join(self.contract_root, f'phase_{phase_name}.json')
        self.phase_contracts[phase_name] = contract_path
        self._write_json_atomic(
            contract_path,
            self.completion_contract(phase_name, status, return_code),
        )
        return contract_path

    def run_phase(self, phase_name: str):
        self._verify_phase_inputs(phase_name)
        self.active_phase = phase_name
        snapshot_path = self.write_phase_snapshot(phase_name)
        self.write_completion_contract(phase_name, 'running')
        command = [sys.executable, os.path.join(TOOLKIT_ROOT, 'run.py'), snapshot_path]
        result = subprocess.run(command, cwd=TOOLKIT_ROOT, check=False)
        status = 'completed' if result.returncode == 0 else 'failed'
        self.write_completion_contract(phase_name, status, result.returncode)
        if result.returncode != 0:
            raise RuntimeError(
                f'Three-phase trigger child phase {phase_name} failed with exit code {result.returncode}'
            )

    def _contract_is_verified(self, phase_name: str) -> bool:
        path = os.path.join(self.contract_root, f'phase_{phase_name}.json')
        if not os.path.isfile(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                contract = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        if not (
            contract.get('phase') == phase_name
            and contract.get('status') == 'completed'
            and contract.get('return_code') == 0
        ):
            return False
        for record in contract.get('inputs', {}).values():
            if not isinstance(record, dict):
                return False
            source_path = record.get('path')
            expected_hash = record.get('sha256')
            if source_path is None:
                continue
            if not os.path.isfile(source_path):
                return False
            if expected_hash is not None and self._sha256_file(source_path) != expected_hash:
                return False
        return True

    def run(self):
        super().run()
        if not self.three_phase_config.enabled:
            return
        for phase_name in self.PHASE_NAMES:
            if self._contract_is_verified(phase_name):
                continue
            self.run_phase(phase_name)
        self.active_phase = None

    def on_error(self, error: Exception):
        if self.active_phase is not None and self.active_phase in self.phase_snapshots:
            self.write_completion_contract(self.active_phase, 'failed')
