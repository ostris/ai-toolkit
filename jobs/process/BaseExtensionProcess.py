from collections import OrderedDict
from typing import ForwardRef
from jobs.process.BaseProcess import BaseProcess


class BaseExtensionProcess(BaseProcess):
    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.process_id: int
        self.config: OrderedDict
        self.progress_bar: ForwardRef('tqdm') = None

    def run(self):
        super().run()
