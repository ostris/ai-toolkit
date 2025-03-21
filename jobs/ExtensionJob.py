import os
from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension_registry import ExtensionRegistry

class ExtensionJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device = self.get_conf("device", "cpu")
        self.process_dict = ExtensionRegistry().get_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
