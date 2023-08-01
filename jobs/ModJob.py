import os
from collections import OrderedDict
from jobs import BaseJob
from toolkit.metadata import get_meta_for_safetensors
from toolkit.train_tools import get_torch_dtype

process_dict = {
    'rescale_lora': 'ModRescaleLoraProcess',
}


class ModJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device = self.get_conf('device', 'cpu')

        # loads the processes from the config
        self.load_processes(process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
