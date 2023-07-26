import json
import os

from jobs import BaseJob
from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from collections import OrderedDict
from typing import List
from jobs.process import BaseExtractProcess, TrainFineTuneProcess
from datetime import datetime
import yaml
from toolkit.paths import REPOS_ROOT

import sys

sys.path.append(REPOS_ROOT)

process_dict = {
    'vae': 'TrainVAEProcess',
    'slider': 'TrainSliderProcess',
    'lora_hack': 'TrainLoRAHack',
    'rescale_sd': 'TrainSDRescaleProcess',
}


class TrainJob(BaseJob):
    process: List[BaseExtractProcess]

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.training_folder = self.get_conf('training_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.device = self.get_conf('device', 'cpu')
        # self.gradient_accumulation_steps = self.get_conf('gradient_accumulation_steps', 1)
        # self.mixed_precision = self.get_conf('mixed_precision', False)  # fp16
        self.log_dir = self.get_conf('log_dir', None)

        self.writer = None
        self.setup_tensorboard()

        # loads the processes from the config
        self.load_processes(process_dict)

    def save_training_config(self):
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        os.makedirs(self.training_folder, exist_ok=True)
        save_dif = os.path.join(self.training_folder, f'run_config_{timestamp}.yaml')
        with open(save_dif, 'w') as f:
            yaml.dump(self.raw_config, f)

    def run(self):
        super().run()
        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()

    def setup_tensorboard(self):
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            now = datetime.now()
            time_str = now.strftime('%Y%m%d-%H%M%S')
            summary_name = f"{self.name}_{time_str}"
            summary_dir = os.path.join(self.log_dir, summary_name)
            self.writer = SummaryWriter(summary_dir)
