from jobs import BaseJob
from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from collections import OrderedDict
from typing import List
from jobs.process import BaseExtractProcess, TrainFineTuneProcess

from toolkit.paths import REPOS_ROOT

import sys

sys.path.append(REPOS_ROOT)

process_dict = {
    'vae': 'TrainVAEProcess',
    'finetune': 'TrainFineTuneProcess'
}


class TrainJob(BaseJob):
    process: List[BaseExtractProcess]

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.training_folder = self.get_conf('training_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.device = self.get_conf('device', 'cpu')
        self.gradient_accumulation_steps = self.get_conf('gradient_accumulation_steps', 1)
        self.mixed_precision = self.get_conf('mixed_precision', False)  # fp16
        self.logging_dir = self.get_conf('logging_dir', None)

        self.writer = None
        self.setup_tensorboard()

        # loads the processes from the config
        self.load_processes(process_dict)

    def run(self):
        super().run()
        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()

    def setup_tensorboard(self):
        if self.logging_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=self.logging_dir,
                filename_suffix=f"_{self.name}"
            )
