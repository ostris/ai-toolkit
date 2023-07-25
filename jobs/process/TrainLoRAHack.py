# ref:
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
import time
from collections import OrderedDict
import os

from toolkit.config_modules import SliderConfig
from toolkit.paths import REPOS_ROOT
import sys

sys.path.append(REPOS_ROOT)
sys.path.append(os.path.join(REPOS_ROOT, 'leco'))
from toolkit.train_tools import get_torch_dtype, apply_noise_offset
import gc

import torch
from leco import train_util, model_util
from leco.prompt_util import PromptEmbedsCache
from .BaseSDTrainProcess import BaseSDTrainProcess, StableDiffusion


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class LoRAHack:
    def __init__(self, **kwargs):
        self.type = kwargs.get('type', 'suppression')


class TrainLoRAHack(BaseSDTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.hack_config = LoRAHack(**self.get_conf('hack', {}))

    def hook_before_train_loop(self):
        # we don't need text encoder so move it to cpu
        self.sd.text_encoder.to("cpu")
        flush()
        # end hook_before_train_loop

        if self.hack_config.type == 'suppression':
            # set all params to self.current_suppression
            params = self.network.parameters()
            for param in params:
                # get random noise for each param
                noise = torch.randn_like(param) - 0.5
                # apply noise to param
                param.data = noise * 0.001


    def supress_loop(self):
        dtype = get_torch_dtype(self.train_config.dtype)


        loss_dict = OrderedDict(
            {'sup': 0.0}
        )
        # increase noise
        for param in self.network.parameters():
            # get random noise for each param
            noise = torch.randn_like(param) - 0.5
            # apply noise to param
            param.data = param.data + noise * 0.001



        return loss_dict

    def hook_train_loop(self):
        if self.hack_config.type == 'suppression':
            return self.supress_loop()
        else:
            raise NotImplementedError(f'unknown hack type: {self.hack_config.type}')
        # end hook_train_loop
