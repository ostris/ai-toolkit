from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from collections import OrderedDict
from jobs import BaseJob
from toolkit.train_tools import get_torch_dtype

process_dict = {
}


class MergeJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.dtype = self.get_conf('dtype', 'fp16')
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.is_v2 = self.get_conf('is_v2', False)
        self.device = self.get_conf('device', 'cpu')

        # loads the processes from the config
        self.load_processes(process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
