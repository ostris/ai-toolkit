import random
from datetime import datetime
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Union

import torch
import yaml

from jobs.process.BaseProcess import BaseProcess

if TYPE_CHECKING:
    from jobs import TrainJob, BaseJob, ExtensionJob
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm


class BaseTrainProcess(BaseProcess):

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.process_id: int
        self.config: OrderedDict
        self.writer: 'SummaryWriter'
        self.job: Union['TrainJob', 'BaseJob', 'ExtensionJob']
        self.progress_bar: 'tqdm' = None

        self.training_seed = self.get_conf('training_seed', self.job.training_seed if hasattr(self.job, 'training_seed') else None)
        # if training seed is set, use it
        if self.training_seed is not None:
            torch.manual_seed(self.training_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.training_seed)
            random.seed(self.training_seed)

        self.progress_bar = None
        self.writer = None
        self.training_folder = self.get_conf('training_folder',
                                             self.job.training_folder if hasattr(self.job, 'training_folder') else None)
        self.save_root = os.path.join(self.training_folder, self.name)
        self.step = 0
        self.first_step = 0
        self.log_dir = self.get_conf('log_dir', self.job.log_dir if hasattr(self.job, 'log_dir') else None)
        self.setup_tensorboard()
        self.save_training_config()

    def run(self):
        super().run()
        # implement in child class
        # be sure to call super().run() first
        pass

    # def print(self, message, **kwargs):
    def print(self, *args):
        if self.progress_bar is not None:
            self.progress_bar.write(' '.join(map(str, args)))
            self.progress_bar.update()
        else:
            print(*args)

    def setup_tensorboard(self):
        if self.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            now = datetime.now()
            time_str = now.strftime('%Y%m%d-%H%M%S')
            summary_name = f"{self.name}_{time_str}"
            summary_dir = os.path.join(self.log_dir, summary_name)
            self.writer = SummaryWriter(summary_dir)

    def save_training_config(self):
        os.makedirs(self.save_root, exist_ok=True)
        save_dif = os.path.join(self.save_root, f'config.yaml')
        with open(save_dif, 'w', encoding='utf-8') as f:
            yaml.dump(self.job.raw_config, f, allow_unicode=True)

