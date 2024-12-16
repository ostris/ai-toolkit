import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension import get_all_extensions_process_dict
from toolkit.paths import CONFIG_ROOT

class ExtensionJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device_config = self.get_conf('device', 'cpu')
        
        # 判断是否使用多个 GPU
        if isinstance(self.device_config, str) and self.device_config.startswith('cuda'):
            devices = self.device_config.split(',')
            if len(devices) > 1:
                # 多 GPU 情况
                self.distributed = True
                self.local_rank = int(os.getenv('LOCAL_RANK', 0))
                if self.local_rank < len(devices):
                    self.device = torch.device(f'cuda:{devices[self.local_rank].strip()}')  # 确保设备字符串正确
                else:
                    raise RuntimeError(f"Invalid LOCAL_RANK {self.local_rank}, exceeding available devices {devices}")
                self.world_size = len(devices)
                dist.init_process_group(backend='nccl', rank=self.local_rank, world_size=self.world_size)
            else:
                # 单 GPU 情况
                self.distributed = False
                self.device = torch.device(self.device_config.strip())  # 单 GPU 情况，去除空格
        else:
            # 默认 CPU 情况
            self.distributed = False
            self.device = torch.device('cpu')

        # 加载扩展的进程
        self.process_dict = get_all_extensions_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()

    def cleanup(self):
        if self.distributed:
            dist.destroy_process_group()
