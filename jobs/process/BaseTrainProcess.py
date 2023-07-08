from collections import OrderedDict
from jobs import TrainJob
from jobs.process.BaseProcess import BaseProcess


class BaseTrainProcess(BaseProcess):
    job: TrainJob
    process_id: int
    config: OrderedDict

    def __init__(
            self,
            process_id: int,
            job: TrainJob,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.process_id = process_id
        self.job = job
        self.config = config

    def run(self):
        # implement in child class
        # be sure to call super().run() first
        pass
