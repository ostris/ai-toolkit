import os
from collections import OrderedDict
from typing import ForwardRef

from jobs.process.BaseProcess import BaseProcess


class BaseTrainProcess(BaseProcess):
    process_id: int
    config: OrderedDict
    progress_bar: ForwardRef('tqdm') = None

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.progress_bar = None
        self.writer = self.job.writer
        self.training_folder = self.get_conf('training_folder', self.job.training_folder)
        self.save_root = os.path.join(self.training_folder, self.job.name)
        self.step = 0
        self.first_step = 0

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
