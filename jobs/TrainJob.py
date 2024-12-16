import json
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed

from jobs import BaseJob
from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from collections import OrderedDict
from typing import List
from jobs.process import BaseExtractProcess, TrainFineTuneProcess
from datetime import datetime
import yaml
from toolkit.paths import REPOS_ROOT

import sys

sys.path.append(REPOS_ROOT)

process_dict = {
    'vae': 'TrainVAEProcess',
    'slider': 'TrainSliderProcess',
    'slider_old': 'TrainSliderProcessOld',
    'lora_hack': 'TrainLoRAHack',
    'rescale_sd': 'TrainSDRescaleProcess',
    'esrgan': 'TrainESRGANProcess',
    'reference': 'TrainReferenceProcess',
}


class TrainJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.training_folder = self.get_conf('training_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.device_config = self.get_conf('device', 'cpu')
        
        if isinstance(self.device_config, str) and self.device_config.startswith('cuda'):
            devices = self.device_config.split(',')
            if len(devices) > 1:
                # 多 GPU 情况
                self.distributed = True
                self.local_rank = int(os.getenv('LOCAL_RANK', 0))
                self.device = torch.device(f'cuda:{self.local_rank}')
                self.world_size = len(devices)
                dist.init_process_group(backend='nccl', rank=self.local_rank, world_size=self.world_size)
            else:
                # 单 GPU 情况
                self.distributed = False
                self.device = torch.device(self.device_config)
        else:
            # 默认 CPU 情况
            self.distributed = False
            self.device = torch.device('cpu')
        # self.gradient_accumulation_steps = self.get_conf('gradient_accumulation_steps', 1)
        # self.mixed_precision = self.get_conf('mixed_precision', False)  # fp16
        self.log_dir = self.get_conf('log_dir', None)
        
        self.model = self.get_flux_model()
        
        self.use_deepspeed = self.get_conf('deepspeed', False)
        if self.use_deepspeed:
            deepspeed_config = self.get_conf('deepspeed_config', None)
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                config=deepspeed_config,
                model=self.model,
                model_parameters=self.model.parameters()
            )
        elif self.distributed:
            # 使用 DDP 包装模型
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # 将模型移动到相应设备
        self.model = self.model.to(self.device)  
        

        # loads the processes from the config
        self.load_processes(process_dict)


    def run(self):
        super().run()
        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
            
    def cleanup(self):
        if self.distributed:
            dist.destroy_process_group()
