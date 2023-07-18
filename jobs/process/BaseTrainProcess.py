from collections import OrderedDict
from jobs.process.BaseProcess import BaseProcess


class BaseTrainProcess(BaseProcess):
    process_id: int
    config: OrderedDict

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)

    def run(self):
        # implement in child class
        # be sure to call super().run() first
        pass
