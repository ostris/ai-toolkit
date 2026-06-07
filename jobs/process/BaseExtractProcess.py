import os
from collections import OrderedDict

from safetensors.torch import save_file

from jobs.process.BaseProcess import BaseProcess
from toolkit.metadata import get_meta_for_safetensors

from typing import ForwardRef

from toolkit.train_tools import get_torch_dtype


class BaseExtractProcess(BaseProcess):

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.config: OrderedDict
        self.output_folder: str
        self.output_filename: str
        self.output_path: str
        self.process_id = process_id
        self.job = job
        self.config = config
        self.dtype = self.get_conf('dtype', self.job.dtype)
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.extract_unet = self.get_conf('extract_unet', self.job.extract_unet)
        self.extract_text_encoder = self.get_conf('extract_text_encoder', self.job.extract_text_encoder)

    def run(self):
        # here instead of init because child init needs to go first
        self.output_path = self.get_output_path()
        # implement in child class
        # be sure to call super().run() first
        pass

    # you can override this in the child class if you want
    # call super().get_output_path(prefix="your_prefix_", suffix="_your_suffix") to extend this
    def get_output_path(self, prefix=None, suffix=None):
        config_output_path = self.get_conf('output_path', None)
        config_filename = self.get_conf('filename', None)
        # replace [name] with name

        if config_output_path is not None:
            config_output_path = config_output_path.replace('[name]', self.job.name)
            return config_output_path

        if config_output_path is None and config_filename is not None:
            # build the output path from the output folder and filename
            return os.path.join(self.job.output_folder, config_filename)

        # build our own

        if suffix is None:
            # we will just add process it to the end of the filename if there is more than one process
            # and no other suffix was given
            suffix = f"_{self.process_id}" if len(self.config['process']) > 1 else ''

        if prefix is None:
            prefix = ''

        output_filename = f"{prefix}{self.output_filename}{suffix}"

        return os.path.join(self.job.output_folder, output_filename)

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
