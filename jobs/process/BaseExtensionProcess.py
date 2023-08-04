from collections import OrderedDict
from typing import ForwardRef
from jobs.process.BaseProcess import BaseProcess


class BaseExtensionProcess(BaseProcess):
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

    def run(self):
        super().run()
