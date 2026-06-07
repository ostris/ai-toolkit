import os
from collections import OrderedDict

from safetensors.torch import save_file

from jobs.process.BaseProcess import BaseProcess
from toolkit.metadata import get_meta_for_safetensors
from toolkit.train_tools import get_torch_dtype


class BaseMergeProcess(BaseProcess):

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.process_id: int
        self.config: OrderedDict
        self.output_path = self.get_conf('output_path', required=True)
        self.dtype = self.get_conf('dtype', self.job.dtype)
        self.torch_dtype = get_torch_dtype(self.dtype)

    def run(self):
        # implement in child class
        # be sure to call super().run() first
        pass

    def save(self, state_dict):
        # prepare meta
        save_meta = get_meta_for_safetensors(self.meta, self.job.name)

        # save
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(self.torch_dtype)
            state_dict[key] = v

        # having issues with meta
        save_file(state_dict, self.output_path, save_meta)

        print(f"Saved to {self.output_path}")
