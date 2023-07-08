from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from .BaseJob import BaseJob
from collections import OrderedDict
from typing import List

from jobs.process import BaseExtractProcess


class ExtractJob(BaseJob):
    process: List[BaseExtractProcess]

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.base_model_path = self.get_conf('base_model', required=True)
        self.base_model = None
        self.extract_model_path = self.get_conf('extract_model', required=True)
        self.extract_model = None
        self.output_folder = self.get_conf('output_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.device = self.get_conf('device', 'cpu')

        if 'process' not in self.config:
            raise ValueError('config file is invalid. Missing "config.process" key')
        if len(self.config['process']) == 0:
            raise ValueError('config file is invalid. "config.process" must be a list of processes')

        # add the processes
        self.process = []
        for i, process in enumerate(self.config['process']):
            if 'type' not in process:
                raise ValueError(f'config file is invalid. Missing "config.process[{i}].type" key')
            if process['type'] == 'locon':
                from jobs.process import LoconExtractProcess
                self.process.append(LoconExtractProcess(i, self, process))
            else:
                raise ValueError(f'config file is invalid. Unknown process type: {process["type"]}')

    def run(self):
        super().run()
        # load models
        print(f"Loading models for extraction")
        print(f" - Loading base model: {self.base_model_path}")
        self.base_model = load_models_from_stable_diffusion_checkpoint(self.is_v2, self.base_model_path)

        print(f" - Loading extract model: {self.extract_model_path}")
        self.extract_model = load_models_from_stable_diffusion_checkpoint(self.is_v2, self.extract_model_path)

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()

